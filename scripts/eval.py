# SPDX-License-Identifier: MIT
"""Score a trained checkpoint: does it actually balance, or does it just outlast the episode?

Training reward only tells you a policy survived its 8 s episode; a checkpoint can clear
that most of the time and still fall over at 15 s. This replays the deterministic policy
over a horizon of your choosing and reports how many environments never terminated.

Usage::

    .\\run.ps1 scripts/eval.py --headless                          # newest checkpoint, 8 s
    .\\run.ps1 scripts/eval.py --headless --episode_s 60           # the convergence test
    .\\run.ps1 scripts/eval.py --headless --init_scale 5           # harder starts
    .\\run.ps1 scripts/eval.py --headless --checkpoint logs/.../model_2700.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to average over.")
parser.add_argument("--checkpoint", nargs="*", default=None, help="Checkpoints to score. Default: the newest one.")
parser.add_argument("--episode_s", type=float, default=None, help="Override the episode length [s].")
parser.add_argument("--init_scale", type=float, default=1.0, help="Multiply the reset randomisation ranges.")
parser.add_argument("--init_tilt", type=float, default=None, help="Reset spread of the link angles alone [rad].")
parser.add_argument("--push_force", type=float, default=None, help="Random sideways kick on the top link [N].")
parser.add_argument("--max_tilt", type=float, default=None, help="Termination threshold on absolute link tilt [rad].")
parser.add_argument("--seed", type=int, default=0, help="Every checkpoint replays from the same initial states.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "triple_cartpole"


def default_checkpoint() -> Path | None:
    bundled = REPO_ROOT / "checkpoints" / "model_robust.pt"
    if bundled.is_file():
        return bundled
    checkpoints = sorted(LOG_ROOT.glob("*/model_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        return None
    newest = checkpoints[-1]
    pinned = newest.parent / "model_best.pt"
    return pinned if pinned.is_file() else newest

def resolve_checkpoints() -> list[Path]:
    if args_cli.checkpoint:
        paths, missing = [], []
        for c in args_cli.checkpoint:
            path = Path(c)
            found = next((candidate for candidate in (path, REPO_ROOT / path) if candidate.is_file()), None)
            (paths if found else missing).append(found or c)
        if missing:
            raise SystemExit("Checkpoint(s) not found:\n  " + "\n  ".join(str(m) for m in missing))
        return paths
    found = default_checkpoint()
    if found is None:
        raise SystemExit(f"No checkpoint found under {LOG_ROOT}. Train first.")
    return [found]


CHECKPOINTS = resolve_checkpoints()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

import triple_cartpole  # noqa: F401
from triple_cartpole import TASK_ID


def main() -> None:
    checkpoints = CHECKPOINTS

    env_cfg = triple_cartpole.TripleCartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.episode_s is not None:
        env_cfg.episode_length_s = args_cli.episode_s
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.push_force is not None:
        env_cfg.push_force = args_cli.push_force
    if args_cli.max_tilt is not None:
        env_cfg.max_tilt = args_cli.max_tilt
    for name in ("init_cart_pos_range", "init_cart_vel_range", "init_pole_angle_range", "init_pole_vel_range"):
        setattr(env_cfg, name, getattr(env_cfg, name) * args_cli.init_scale)
    if args_cli.init_tilt is not None:
        env_cfg.init_pole_angle_range = args_cli.init_tilt
    env_cfg.seed = args_cli.seed

    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args_cli.seed
    env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    unwrapped = env.unwrapped
    horizon = int(unwrapped.max_episode_length) - 1
    print(
        f"\n[eval] {unwrapped.num_envs} envs, {horizon} steps ({horizon * unwrapped.step_dt:.0f} s),"
        f" init_tilt +/-{env_cfg.init_pole_angle_range:g} rad, cart +/-{env_cfg.init_cart_pos_range:g} m,"
        f" push {env_cfg.push_force:g} N, max_tilt {env_cfg.max_tilt:g} rad\n"
    )
    print(
        f"{'checkpoint':<18}{'balanced':>10}{'mean_steps':>12}{'peak_tilt':>11}"
        f"{'end_tilt':>10}{'peak_cart':>11}{'mean_force':>12}"
    )

    for path in checkpoints:
        runner.load(str(path))
        policy = runner.get_inference_policy(device=unwrapped.device)
        policy_nn = runner.alg.policy
        torch.manual_seed(args_cli.seed)
        with torch.inference_mode():
            obs, _ = env.reset()
        policy_nn.reset()

        alive = torch.ones(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
        steps = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
        peak_tilt = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
        peak_cart = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
        force_sum = torch.zeros((), device=unwrapped.device)

        with torch.inference_mode():
            for _ in range(horizon):
                obs, _, dones, _ = env.step(policy(obs))
                policy_nn.reset(dones)
                live = alive.float()
                steps += live
                tilt = unwrapped._link_angle.abs().amax(dim=-1)
                peak_tilt = torch.where(alive, torch.maximum(peak_tilt, tilt), peak_tilt)
                peak_cart = torch.where(alive, torch.maximum(peak_cart, unwrapped._cart_pos.abs()), peak_cart)
                force_sum += (unwrapped._effort.abs().squeeze(-1) * live).sum()
                alive &= ~unwrapped.reset_terminated.reshape(-1).bool()

        held = bool(alive.any())
        tilt_col = f"{peak_tilt[alive].max().item():.3f}" if held else "-"
        end_col = f"{unwrapped._link_angle.abs().amax(dim=-1)[alive].mean().item():.3f}" if held else "-"
        cart_col = f"{peak_cart[alive].max().item():.2f}" if held else "-"
        print(
            f"{path.name:<18}{100.0 * alive.float().mean().item():>9.1f}%{steps.mean().item():>12.1f}"
            f"{tilt_col:>11}{end_col:>10}{cart_col:>11}{(force_sum / steps.sum().clamp(min=1)).item():>11.1f}N"
        )

    print(
        f"\n[eval] 'balanced' = never terminated: every link within {env_cfg.max_tilt} rad"
        f" and the cart within +/-{env_cfg.max_cart_pos} m for the whole horizon."
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
