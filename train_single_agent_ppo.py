# train_single_agent_ppo.py
#
# Primary entry: independent multi-agent PPO (IPPO) on MultiGrid — the take-home task.
# Run:
#   cd C:\Users\hp\TISL\multigrid
#   python train_single_agent_ppo.py --env_id MultiGrid-Cluttered-Fixed-15x15 --total_timesteps 500000
#   python train_single_agent_ppo.py --total_episodes 100000  # episode budget (overrides --total_timesteps)
#
# Still exports ``make_wrapped_env`` / ``MultigridTinyCNNExtractor`` for eval_single_agent_ppo.py.

from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Gym has been unmaintained.*",
)

import gym
import numpy as np
import torch as th
from gymnasium import spaces
from gym.spaces import Box as GymBox
from gym.spaces import Dict as GymDict
from gym.spaces import Discrete as GymDiscrete
from shimmy import GymV21CompatibilityV0
from stable_baselines3 import PPO
from stable_baselines3.common import utils as sb3_utils
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from torch import nn

# Side-effect imports: register MultiGrid env ids.
from envs import gym_multigrid  # noqa: F401
from envs.gym_multigrid import multigrid_envs  # noqa: F401


# ---------------------------------------------------------------------------
# CNN (shared by single-agent eval and multi-agent policies)
# ---------------------------------------------------------------------------


class MultigridTinyCNNExtractor(BaseFeaturesExtractor):
  """Dict(image=CHW uint8, direction=(1,) uint8) -> feature vector."""

  def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
    super().__init__(observation_space, features_dim)
    if not isinstance(observation_space, spaces.Dict):
      raise TypeError("Expected Dict observation space")
    img_sp = observation_space.spaces["image"]
    if len(img_sp.shape) != 3:
      raise ValueError(f"Expected CHW image space, got shape {img_sp.shape}")
    c, h, w = map(int, img_sp.shape)

    self.cnn = nn.Sequential(
        nn.Conv2d(c, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
    )
    with th.no_grad():
      n_flat = int(self.cnn(th.zeros(1, c, h, w)).shape[1])

    self.image_fc = nn.Sequential(nn.Linear(n_flat, 64), nn.ReLU())
    self.dir_embed = nn.Embedding(4, 16)
    self.tail = nn.Sequential(nn.Linear(64 + 16, features_dim), nn.ReLU())

  def forward(self, observations: dict) -> th.Tensor:
    img = observations["image"]
    if img.dtype != th.float32:
      img = img.float()
    if img.max() > 1.0:
      img = img / 255.0
    x = self.image_fc(self.cnn(img))
    d = observations["direction"].long().squeeze(-1).clamp(0, 3)
    d_emb = self.dir_embed(d)
    return self.tail(th.cat([x, d_emb], dim=1))


# ---------------------------------------------------------------------------
# Single-agent Gym wrapper + Shimmy (for eval_single_agent_ppo.py)
# ---------------------------------------------------------------------------


class MultigridGymWrapper(gym.Wrapper):
  """Classic Gym API: single-agent view for SB3 + Shimmy."""

  metadata = {"render_modes": []}

  def __init__(self, gym_env: gym.Env):
    super().__init__(gym_env)
    aspace = gym_env.action_space
    if isinstance(aspace, GymDiscrete):
      n_actions = int(aspace.n)
    elif isinstance(aspace, GymBox):
      n_actions = int(aspace.high[0] + 1)
    else:
      raise TypeError(f"Unsupported action space type: {type(aspace)}")
    self.action_space = GymDiscrete(n_actions)

    raw_obs = gym_env.observation_space
    if not isinstance(raw_obs, GymDict):
      raise TypeError(f"Expected gym.spaces.Dict obs space, got {type(raw_obs)}")
    gimg = raw_obs.spaces["image"]
    gdir = raw_obs.spaces["direction"]
    if len(gimg.shape) == 4 and gimg.shape[0] == 1:
      img_shape = tuple(gimg.shape[1:])
    else:
      img_shape = tuple(gimg.shape)
    dir_shape = tuple(gdir.shape)
    d_low = int(np.asarray(gdir.low).reshape(-1)[0])
    d_high = int(np.asarray(gdir.high).reshape(-1)[0])

    self.observation_space = GymDict({
        "image": GymBox(low=0, high=255, shape=img_shape, dtype=np.uint8),
        "direction": GymBox(low=d_low, high=d_high, shape=dir_shape, dtype=np.uint8),
    })

  def _process_obs(self, obs: dict) -> dict:
    img = obs["image"]
    if isinstance(img, (list, tuple)):
      img = np.asarray(img[0])
    else:
      img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:
      img = img[0]
    d = obs["direction"]
    if isinstance(d, (list, tuple)):
      d = np.asarray(d, dtype=np.uint8).reshape(-1)
    else:
      d = np.asarray(d, dtype=np.uint8).reshape(-1)
    return {"image": img.astype(np.uint8, copy=False), "direction": d}

  def reset(self):
    obs = self.env.reset()
    return self._process_obs(obs)

  def step(self, action):
    a = int(np.asarray(action).reshape(()))
    obs, reward, done, info = self.env.step([a])
    obs = self._process_obs(obs)
    reward_scalar = np.asarray(reward).reshape(-1)[0]
    done_b = bool(np.asarray(done).reshape(-1)[0])
    return obs, float(reward_scalar), done_b, info


def make_wrapped_env(env_id: str, seed: int | None = None) -> GymV21CompatibilityV0:
  e = gym.make(env_id)
  if seed is not None:
    e.seed(seed)
  return GymV21CompatibilityV0(env=MultigridGymWrapper(e))


def build_dummy_vec_for_policy(env_id: str) -> VecTransposeImage:
  """Same per-agent obs layout as multi-agent (tiny CNN); used only to build SB3 policies."""

  def _thunk():
    return make_wrapped_env(env_id, seed=None)

  return VecTransposeImage(DummyVecEnv([_thunk]))


# ---------------------------------------------------------------------------
# Multi-agent: joint obs -> per-agent dict for policy (CHW, batch 1)
# ---------------------------------------------------------------------------


def joint_obs_to_agent_policy_obs(
    joint_obs: dict,
    agent_i: int,
    n_agents: int,
) -> dict[str, np.ndarray]:
  """One agent slice from MultiGrid joint observation; numpy CHW + direction (1,1)."""
  imgs = joint_obs["image"]
  dirs = joint_obs["direction"]
  img = np.asarray(imgs[agent_i])
  if img.ndim == 4 and img.shape[0] == 1:
    img = img[0]
  di = dirs[agent_i]
  if isinstance(di, (list, tuple, np.ndarray)) and np.asarray(di).size != 1:
    d = np.asarray(di, dtype=np.uint8).reshape(-1)
  else:
    d = np.asarray([int(di)], dtype=np.uint8).reshape(-1)
  img = img.astype(np.uint8, copy=False)
  # HWC -> CHW
  img_chw = np.transpose(img, (2, 0, 1))
  return {
      "image": np.expand_dims(img_chw, 0),
      "direction": np.expand_dims(d, 0),
  }


def build_ppo_for_agent(
    template_env_id: str,
    seed: int,
    tensorboard_log: str | None,
    n_steps: int,
) -> PPO:
  # Each PPO needs its own VecEnv instance (do not share one VecEnv across agents).
  vec = build_dummy_vec_for_policy(template_env_id)
  return PPO(
      "MultiInputPolicy",
      vec,
      seed=seed,
      verbose=0,
      tensorboard_log=tensorboard_log,
      n_steps=n_steps,
      batch_size=min(256, max(2, n_steps)),
      policy_kwargs={
          "features_extractor_class": MultigridTinyCNNExtractor,
          "features_extractor_kwargs": {"features_dim": 128},
      },
  )


def collect_joint_rollout(
    joint_env: gym.Env,
    ppos: list[PPO],
    n_agents: int,
    n_steps: int,
) -> dict[str, np.ndarray | float]:
  """Fill each PPO rollout buffer and return episode stats seen in this rollout."""
  # start a fresh on-policy buffer for each agent before collecting this rollout chunk.
  for p in ppos:
    p.rollout_buffer.reset()
    p.policy.set_training_mode(False)

  obs_joint = joint_env.reset()
  episode_starts = np.ones((1,), dtype=np.float32)
  ep_len = 0
  ep_agent_returns = np.zeros(n_agents, dtype=np.float64)
  rollout_episode_lengths: list[int] = []
  rollout_collective_returns: list[float] = []
  rollout_agent_returns: list[np.ndarray] = []

  # collect n_steps of joint interaction, while storing each agent's own transition.
  for _ in range(n_steps):
    actions_joint: list[int] = []
    values_list: list[th.Tensor] = []
    log_probs_list: list[th.Tensor] = []

    for i in range(n_agents):
      obs_i = joint_obs_to_agent_policy_obs(obs_joint, i, n_agents)
      obs_tensor = obs_as_tensor(obs_i, ppos[i].device)
      with th.no_grad():
        actions, values, log_probs = ppos[i].policy(obs_tensor)
      act = int(actions.cpu().numpy().reshape(()))
      actions_joint.append(act)
      values_list.append(values)
      log_probs_list.append(log_probs)

    next_obs, rewards, done, _info = joint_env.step(actions_joint)
    done = bool(np.asarray(done).reshape(()))
    rew_arr = rewards
    if not isinstance(rew_arr, (list, tuple)):
      rew_arr = [float(rew_arr)] * n_agents
    else:
      rew_arr = [float(np.asarray(r).reshape(())) for r in rew_arr]
    ep_len += 1
    ep_agent_returns += np.asarray(rew_arr, dtype=np.float64)

    for i in range(n_agents):
      obs_i = joint_obs_to_agent_policy_obs(obs_joint, i, n_agents)
      ppos[i].rollout_buffer.add(
          obs_i,
          np.array([[actions_joint[i]]], dtype=np.float32),
          np.array([rew_arr[i]], dtype=np.float32),
          episode_starts,
          values_list[i],
          log_probs_list[i],
      )

    obs_joint = next_obs
    episode_starts = np.array([float(done)], dtype=np.float32)
    if done:
      # snapshot per-episode returns so we can log one point per completed episode.
      rollout_episode_lengths.append(ep_len)
      rollout_agent_returns.append(ep_agent_returns.copy())
      rollout_collective_returns.append(float(np.sum(ep_agent_returns)))
      ep_len = 0
      ep_agent_returns[:] = 0.0
      obs_joint = joint_env.reset()
      episode_starts = np.ones((1,), dtype=np.float32)

  dones_last = np.array([float(done)], dtype=np.float32)
  last_vals: list[th.Tensor] = []
  with th.no_grad():
    for i in range(n_agents):
      oi = joint_obs_to_agent_policy_obs(obs_joint, i, n_agents)
      last_vals.append(
          ppos[i].policy.predict_values(obs_as_tensor(oi, ppos[i].device))
      )

  # compute advantages/returns for each agent buffer with the same terminal flag.
  for i in range(n_agents):
    ppos[i].rollout_buffer.compute_returns_and_advantage(
        last_values=last_vals[i],
        dones=dones_last,
    )
  if len(rollout_episode_lengths) == 0:
    return {
        "episodes_in_rollout": 0,
        "episode_collective_returns": np.array([], dtype=np.float64),
        "episode_agent_returns": np.zeros((0, n_agents), dtype=np.float64),
    }
  return {
      "episodes_in_rollout": len(rollout_episode_lengths),
      "episode_collective_returns": np.asarray(rollout_collective_returns, dtype=np.float64),
      "episode_agent_returns": np.vstack(rollout_agent_returns),
  }


def train_multi_agent_ippo(args: argparse.Namespace) -> None:
  # seed numpy/torch so runs are repeatable for a fixed seed.
  np.random.seed(args.seed)
  th.manual_seed(args.seed)

  # create the shared multi-agent environment and detect number of agents.
  joint_env = gym.make(args.env_id)
  joint_env.seed(args.seed)
  n_agents = int(joint_env.n_agents)

  ppos: list[PPO] = []
  for i in range(n_agents):
    # build one independent ppo policy per agent (ippo).
    ppos.append(
        build_ppo_for_agent(
            args.template_env_id,
            args.seed + i,
            args.tensorboard_log,
            args.n_steps,
        )
    )

  for i, p in enumerate(ppos):
    # ``learn()`` normally configures this; we call ``train()`` manually.
    p._logger = sb3_utils.configure_logger(
        verbose=0,
        tensorboard_log=p.tensorboard_log,
        tb_log_name=f"PPO_agent_{i}",
        reset_num_timesteps=True,
    )

  wandb = None
  if args.wandb_mode != "disabled":
    try:
      import wandb as _wandb

      _wandb.init(project=args.wandb_project, config=vars(args), mode=args.wandb_mode)
      # Handout: collective (and per-agent) return vs episode index — one W&B step per episode.
      _wandb.define_metric("episode/x_axis")
      _wandb.define_metric("episode/*", step_metric="episode/x_axis")
      wandb = _wandb
    except Exception:
      pass

  total_env_steps = 0
  total_episodes = 0
  cumulative_collective_return = 0.0
  cumulative_agent_returns = np.zeros(n_agents, dtype=np.float64)
  stop_by_episodes = args.total_episodes is not None

  while True:
    # collect one rollout chunk across the joint env.
    rollout_stats = collect_joint_rollout(joint_env, ppos, n_agents, args.n_steps)

    progress_num = total_episodes if stop_by_episodes else total_env_steps
    progress_den = (
        args.total_episodes if stop_by_episodes else args.total_timesteps
    )
    for p in ppos:
      # update each policy on its own buffer, then advance its internal timestep counter.
      p._update_current_progress_remaining(progress_num, max(progress_den, 1))
      p.train()
      p.num_timesteps += args.n_steps
    total_env_steps += args.n_steps

    ep_returns = rollout_stats["episode_collective_returns"]
    ep_agent_r = rollout_stats["episode_agent_returns"]
    episodes_in_rollout = int(rollout_stats["episodes_in_rollout"])

    if episodes_in_rollout > 0:
      for j in range(episodes_in_rollout):
        total_episodes += 1
        cumulative_collective_return += float(ep_returns[j])
        cumulative_agent_returns += ep_agent_r[j].astype(np.float64, copy=False)
        if wandb is not None:
          # log one wandb row per completed episode so x-axis is true episode count.
          row = {
              "episode/x_axis": total_episodes,
              "episode/collective_return": float(ep_returns[j]),
              "episode/cumulative_collective_return": cumulative_collective_return,
          }
          for i in range(n_agents):
            row[f"episode/agent_{i}_return"] = float(ep_agent_r[j, i])
            row[f"episode/cumulative_agent_{i}_return"] = float(
                cumulative_agent_returns[i]
            )
          wandb.log(row)

    if args.verbose:
      if stop_by_episodes:
        print(
            f"completed episodes: {total_episodes} / {args.total_episodes}  "
            f"(joint env steps: {total_env_steps})"
        )
      else:
        print(f"joint env steps: {total_env_steps} / {args.total_timesteps}")

    if stop_by_episodes:
      # stop by completed episodes when requested; otherwise fall back to joint env steps.
      if total_episodes >= args.total_episodes:
        break
    elif total_env_steps >= args.total_timesteps:
      break

  joint_env.close()

  if wandb is not None:
    try:
      wandb.finish()
    except Exception:
      pass

  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
  for i, p in enumerate(ppos):
    path = os.path.join(args.save_dir, f"{args.save_prefix}_agent_{i}.zip")
    p.save(path)
    print(f"Saved {path}")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
      description="IPPO: one PPO policy per agent on a multi-agent MultiGrid env.",
  )
  p.add_argument(
      "--env_id",
      type=str,
      default="MultiGrid-Cluttered-Fixed-15x15",
      help="Registered MultiGrid multi-agent env id.",
  )
  p.add_argument(
      "--template_env_id",
      type=str,
      default="MultiGrid-Cluttered-Fixed-Single-6x6-v0",
      help="Single-agent env used only to match observation_space for SB3 (same view size).",
  )
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
      "--total_timesteps",
      type=int,
      default=50_000,
      help="Stop after this many joint env steps (ignored if --total_episodes is set).",
  )
  p.add_argument(
      "--total_episodes",
      type=int,
      default=None,
      help=(
          "If set, stop after this many completed multi-agent episodes (same counter as W&B "
          "episode/x_axis). Overrides --total_timesteps for the stopping condition."
      ),
  )
  p.add_argument(
      "--n_steps",
      type=int,
      default=2048,
      help="Rollout length per iteration (joint env steps).",
  )
  p.add_argument("--tensorboard_log", type=str, default=None)
  p.add_argument("--save_dir", type=str, default="models")
  p.add_argument("--save_prefix", type=str, default="ippo_multigrid")
  p.add_argument("--wandb_project", type=str, default="multigrid-ippo")
  p.add_argument("--wandb_mode", type=str, default="disabled", help="e.g. online or disabled")
  p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
  return p.parse_args()


def main():
  args = parse_args()
  train_multi_agent_ippo(args)


if __name__ == "__main__":
  main()
