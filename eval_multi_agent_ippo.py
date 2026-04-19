# eval_multi_agent_ippo.py
#
# Check that trained IPPO policies act in the joint env and compare return to random actions.
#
#   cd C:\Users\hp\TISL\multigrid
#   python eval_multi_agent_ippo.py --save_dir models --save_prefix ippo_multigrid
#
# Handout-style trajectory (full map + partial obs + cumulative reward plots):
#   python eval_multi_agent_ippo.py --handout_video --save_prefix ippo_multigrid
#
# After training with defaults, models are: models\ippo_multigrid_agent_0.zip, ...

from __future__ import annotations

import argparse
import glob
import os
import shutil
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Gym has been unmaintained.*",
)

import gym
import numpy as np
from moviepy.editor import ImageSequenceClip
from stable_baselines3 import PPO

from envs import gym_multigrid  # noqa: F401
from envs.gym_multigrid import multigrid_envs  # noqa: F401

import train_single_agent_ppo as tsp  # noqa: F401 — MultigridTinyCNNExtractor for load


def _rewards_to_list(rewards, n_agents: int) -> list[float]:
  if isinstance(rewards, (list, tuple)):
    return [float(np.asarray(r).reshape(())) for r in rewards]
  return [float(np.asarray(rewards).reshape(()))] * n_agents


def run_episodes(
    env: gym.Env,
    n_agents: int,
    n_episodes: int,
    seed: int,
    models: list[PPO] | None,
    random_policy: bool,
    deterministic: bool,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns (collective_returns [n_ep], per_agent_returns [n_ep, n_agents])."""
  rng = np.random.default_rng(seed)
  collective = np.zeros(n_episodes, dtype=np.float64)
  per_agent = np.zeros((n_episodes, n_agents), dtype=np.float64)

  n_actions = int(env.action_space.high[0] + 1)

  for ep in range(n_episodes):
    env.seed(seed + ep)
    obs = env.reset()
    done = False
    col = 0.0
    pa = np.zeros(n_agents, dtype=np.float64)

    while not done:
      if random_policy:
        actions_joint = [int(rng.integers(0, n_actions)) for _ in range(n_agents)]
      else:
        assert models is not None
        actions_joint = []
        for i in range(n_agents):
          obs_i = tsp.joint_obs_to_agent_policy_obs(obs, i, n_agents)
          act, _ = models[i].predict(obs_i, deterministic=deterministic)
          actions_joint.append(int(np.asarray(act).reshape(())))

      obs, rewards, done, _info = env.step(actions_joint)
      done = bool(np.asarray(done).reshape(()))
      rlist = _rewards_to_list(rewards, n_agents)
      col += sum(rlist)
      pa += np.asarray(rlist, dtype=np.float64)

    collective[ep] = col
    per_agent[ep, :] = pa

  return collective, per_agent


def make_rollout_video(
    env: gym.Env,
    n_agents: int,
    models: list[PPO],
    out_path: str,
    deterministic: bool,
    seed: int,
    max_steps: int = 200,
    fps: int = 10,
) -> None:
  """Run one episode and save an RGB video of the full environment only (no reward plots)."""
  frames: list[np.ndarray] = []
  env.seed(seed)
  obs = env.reset()
  done = False
  t = 0
  while not done and t < max_steps:
    frame = env.render("rgb_array")
    frames.append(np.asarray(frame))

    actions_joint: list[int] = []
    for i in range(n_agents):
      obs_i = tsp.joint_obs_to_agent_policy_obs(obs, i, n_agents)
      act, _ = models[i].predict(obs_i, deterministic=deterministic)
      actions_joint.append(int(np.asarray(act).reshape(())))

    obs, _rewards, done, _info = env.step(actions_joint)
    done = bool(np.asarray(done).reshape(()))
    t += 1

  frames.append(np.asarray(env.render("rgb_array")))
  out_dir = os.path.dirname(os.path.abspath(out_path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  clip = ImageSequenceClip(frames, fps=fps)
  clip.write_videofile(out_path, codec="libx264", audio=False, verbose=False, logger=None)


def make_handout_trajectory_video(
    env: gym.Env,
    n_agents: int,
    models: list[PPO],
    *,
    frame_dir: str,
    video_out: str,
    expt_name: str,
    deterministic: bool,
    seed: int,
    max_steps: int = 200,
    fps: int = 10,
) -> None:
  """Handout-style video via utils.plot_single_frame + utils.make_video."""
  from utils import make_video, plot_single_frame

  raw = env.unwrapped
  action_dict = {int(a.value): a.name for a in raw.actions}

  os.makedirs(frame_dir, exist_ok=True)
  for old in glob.glob(os.path.join(frame_dir, f"{expt_name}_*.png")):
    os.remove(old)

  env.seed(seed)
  obs = env.reset()
  full_images: list[np.ndarray] = []
  agents_partial_images: list[list[np.ndarray]] = []
  actions_taken: list[list[int]] = []
  reward_rows: list[list[float]] = []

  done = False
  t = 0
  while not done and t < max_steps:
    full_images.append(np.asarray(env.render("rgb_array")))
    partials = []
    for i in range(n_agents):
      img_i = np.asarray(obs["image"][i])
      if img_i.ndim == 4 and img_i.shape[0] == 1:
        img_i = img_i[0]
      partials.append(raw.get_obs_render(img_i))
    agents_partial_images.append(partials)

    actions_joint: list[int] = []
    for i in range(n_agents):
      obs_i = tsp.joint_obs_to_agent_policy_obs(obs, i, n_agents)
      act, _ = models[i].predict(obs_i, deterministic=deterministic)
      actions_joint.append(int(np.asarray(act).reshape(())))
    actions_taken.append(actions_joint)

    obs, rewards, done, _info = env.step(actions_joint)
    done = bool(np.asarray(done).reshape(()))
    reward_rows.append(_rewards_to_list(rewards, n_agents))
    t += 1

  if not reward_rows:
    return

  rewards_arr = np.asarray(reward_rows, dtype=np.float64)
  for frame_id in range(len(actions_taken)):
    plot_single_frame(
        frame_id,
        full_images[frame_id],
        agents_partial_images[frame_id],
        actions_taken[frame_id],
        rewards_arr,
        action_dict,
        frame_dir,
        expt_name,
        predicted_actions=None,
        all_actions=actions_taken,
    )

  tmp_video_stem = "_handout_concat"
  make_video(frame_dir, video_name=tmp_video_stem, frame_rate=fps)
  produced = os.path.join(frame_dir, f"{tmp_video_stem}.mp4")
  out_dir = os.path.dirname(os.path.abspath(video_out))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  shutil.move(produced, video_out)


def main() -> None:
  p = argparse.ArgumentParser(description="Evaluate IPPO checkpoints on a multi-agent MultiGrid env.")
  p.add_argument("--env_id", type=str, default="MultiGrid-Cluttered-Fixed-15x15")
  p.add_argument("--template_env_id", type=str, default="MultiGrid-Cluttered-Fixed-Single-6x6-v0")
  p.add_argument("--save_dir", type=str, default="models")
  p.add_argument("--save_prefix", type=str, default="ippo_multigrid")
  p.add_argument("--n_episodes", type=int, default=30)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument(
      "--deterministic",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Greedy actions for trained policies (default: True).",
  )
  p.add_argument(
      "--skip_random",
      action="store_true",
      help="Do not run random baseline (faster).",
  )
  p.add_argument(
      "--make_video",
      action="store_true",
      help="Save a simple full-frame RGB rollout MP4 (not the handout dashboard layout).",
  )
  p.add_argument(
      "--handout_video",
      action="store_true",
      help="Save handout-style trajectory video (utils.plot_single_frame: map, partial obs, returns).",
  )
  p.add_argument("--video_out", type=str, default="videos/ippo_eval_rollout.mp4")
  p.add_argument(
      "--handout_video_out",
      type=str,
      default="videos/ippo_handout_trajectory.mp4",
  )
  p.add_argument("--video_max_steps", type=int, default=200)
  p.add_argument("--video_fps", type=int, default=10)
  args = p.parse_args()

  env = gym.make(args.env_id)
  n_agents = int(env.n_agents)

  if not args.skip_random:
    print(f"Random policy baseline ({args.n_episodes} episodes)...")
    col_r, pa_r = run_episodes(
        env, n_agents, args.n_episodes, args.seed, None, True, False,
    )
    print(
        f"  Collective return: mean={col_r.mean():.3f}  std={col_r.std():.3f}  "
        f"per-agent means={pa_r.mean(axis=0)}"
    )

  models: list[PPO] = []
  for i in range(n_agents):
    path = os.path.join(args.save_dir, f"{args.save_prefix}_agent_{i}.zip")
    vec = tsp.build_dummy_vec_for_policy(args.template_env_id)
    print(f"Loading {path} ...")
    try:
      models.append(PPO.load(path, env=vec))
    except ModuleNotFoundError as e:
      if "numpy" in str(e).lower():
        raise RuntimeError(
            "Failed to load checkpoint (often NumPy major version mismatch). "
            "Train and eval in the same venv with the same `numpy` as in requirements.txt "
            "(e.g. numpy<2 with pandas 2.0.x). Re-save models after fixing the env."
        ) from e
      raise

  print(f"Trained policies ({args.n_episodes} episodes, deterministic={args.deterministic})...")
  col_t, pa_t = run_episodes(
      env, n_agents, args.n_episodes, args.seed + 10_000, models, False, args.deterministic,
  )
  print(
      f"  Collective return: mean={col_t.mean():.3f}  std={col_t.std():.3f}  "
      f"per-agent means={pa_t.mean(axis=0)}"
  )

  if args.make_video:
    print(f"Saving simple RGB rollout video to {args.video_out} ...")
    make_rollout_video(
        env=env,
        n_agents=n_agents,
        models=models,
        out_path=args.video_out,
        deterministic=args.deterministic,
        seed=args.seed + 20_000,
        max_steps=args.video_max_steps,
        fps=args.video_fps,
    )
    print(f"Saved video: {args.video_out}")

  if args.handout_video:
    stem = os.path.splitext(os.path.basename(args.handout_video_out))[0]
    frame_dir = os.path.join(
        os.path.dirname(os.path.abspath(args.handout_video_out)) or ".",
        f"{stem}_frames",
    )
    expt_name = f"{args.save_prefix}_handout"
    print(f"Rendering handout-style frames under {frame_dir} ...")
    make_handout_trajectory_video(
        env=env,
        n_agents=n_agents,
        models=models,
        frame_dir=frame_dir,
        video_out=args.handout_video_out,
        expt_name=expt_name,
        deterministic=args.deterministic,
        seed=args.seed + 25_000,
        max_steps=args.video_max_steps,
        fps=args.video_fps,
    )
    print(f"Saved handout trajectory video: {args.handout_video_out}")

  if not args.skip_random:
    if col_t.mean() > col_r.mean():
      print("Trained policies beat random baseline on mean collective return (good sign).")
    else:
      print(
          "Trained mean is not above random yet — may need more training or different seed/hparams."
      )

  env.close()


if __name__ == "__main__":
  main()
