# SPDX-License-Identifier: MIT
"""Train a PPO policy to balance the triple inverted pendulum, using rsl-rl.

Checkpoints and TensorBoard logs land in ``logs/rsl_rl/triple_cartpole/<timestamp>/``.

Usage::

    .\\run.ps1 scripts/train.py --headless
    .\\run.ps1 scripts/train.py --headless --num_envs 8192 --max_iterations 6000
    .\\run.ps1 scripts/train.py --headless --resume            # continue the newest run

Watch progress with::

    tensorboard --logdir logs/rsl_rl/triple_cartpole
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=None, help="Override the number of environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the number of PPO iterations.")
parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
parser.add_argument("--resume", action="store_true", help="Resume from the newest checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from a specific checkpoint (.pt).")
parser.add_argument("--push_force", type=float, default=None, help="Random sideways kick on the top link [N].")
parser.add_argument("--init_scale", type=float, default=None, help="Multiply the reset randomisation ranges.")
parser.add_argument("--init_tilt", type=float, default=None, help="Reset spread of the link angles alone [rad].")
parser.add_argument("--max_tilt", type=float, default=None, help="Termination threshold on absolute link tilt [rad].")
parser.add_argument("--video", action="store_true", help="Record rollout videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of each recorded video [steps].")
parser.add_argument("--video_interval", type=int, default=5000, help="Steps between video recordings.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "triple_cartpole"

def resolve_resume_path() -> str | None:
    if args_cli.checkpoint:
        path = Path(args_cli.checkpoint)
        found = next((candidate for candidate in (path, REPO_ROOT / path) if candidate.is_file()), None)
        if found is None:
            raise SystemExit(f"Checkpoint not found: {args_cli.checkpoint}")
        return str(found)
    if not args_cli.resume:
        return None
    checkpoints = sorted(LOG_ROOT.glob("*/model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise SystemExit(f"--resume given but no checkpoint found under {LOG_ROOT}")
    return str(checkpoints[-1])


RESUME_PATH = resolve_resume_path()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

import triple_cartpole  # noqa: F401
from triple_cartpole import TASK_ID

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    env_cfg = triple_cartpole.TripleCartpoleEnvCfg()
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.push_force is not None:
        env_cfg.push_force = args_cli.push_force
    if args_cli.max_tilt is not None:
        env_cfg.max_tilt = args_cli.max_tilt
    if args_cli.init_scale is not None:
        for name in ("init_cart_pos_range", "init_cart_vel_range", "init_pole_angle_range", "init_pole_vel_range"):
            setattr(env_cfg, name, getattr(env_cfg, name) * args_cli.init_scale)
    if args_cli.init_tilt is not None:
        env_cfg.init_pole_angle_range = args_cli.init_tilt
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    resume_path = RESUME_PATH
    log_dir = LOG_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)
    print(f"[train] logging to {log_dir}")

    env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": str(log_dir / "videos" / "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[train] recording videos during training")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if resume_path:
        print(f"[train] resuming from {resume_path}")
        runner.load(resume_path)

    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)

    start = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[train] finished in {round(time.time() - start, 1)} s")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
