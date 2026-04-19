# eval_single_agent_ppo.py
#
# Run from repo root:
#   python eval_single_agent_ppo.py --model ppo_single_multigrid.zip
#
# Uses the same env stack and policy_kwargs as train_single_agent_ppo.py.

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Gym has been unmaintained.*",
)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from envs import gym_multigrid  # noqa: F401
from envs.gym_multigrid import multigrid_envs  # noqa: F401

# Import the training module so pickle/unpickle can resolve
# ``train_single_agent_ppo.MultigridTinyCNNExtractor`` from the saved zip.
import train_single_agent_ppo  # noqa: F401

from train_single_agent_ppo import make_wrapped_env


def build_vec_env(env_id: str) -> VecTransposeImage:
  def _thunk():
    return make_wrapped_env(env_id, seed=None)

  vec = DummyVecEnv([_thunk])
  return VecTransposeImage(vec)


def parse_args():
  p = argparse.ArgumentParser(description="Evaluate a saved single-agent PPO zip.")
  p.add_argument(
      "--model",
      type=str,
      default="ppo_single_multigrid.zip",
      help="Path to .zip from model.save()",
  )
  p.add_argument(
      "--env_id",
      type=str,
      default="MultiGrid-Cluttered-Fixed-Single-6x6-v0",
      help="Must match the env used in training.",
  )
  p.add_argument(
      "--n_eval_episodes",
      type=int,
      default=20,
      help="Number of full episodes to average.",
  )
  p.add_argument(
      "--deterministic",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Use deterministic policy (greedy); --no-deterministic for sampling.",
  )
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args()


def main():
  args = parse_args()
  np.random.seed(args.seed)

  vec = build_vec_env(args.env_id)

  print(f"Loading {args.model} ...")
  # ``import train_single_agent_ppo`` above lets SB3 restore the custom CNN class.
  model = PPO.load(args.model, env=vec)

  print(
      f"Evaluating on {args.env_id} "
      f"({args.n_eval_episodes} episodes, deterministic={args.deterministic})..."
  )
  mean_r, std_r = evaluate_policy(
      model,
      vec,
      n_eval_episodes=args.n_eval_episodes,
      deterministic=args.deterministic,
      warn=True,
  )
  print(f"Mean episode reward: {mean_r:.4f} +/- {std_r:.4f}")


if __name__ == "__main__":
  main()
