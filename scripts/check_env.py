# SPDX-License-Identifier: MIT
"""Integration check for the generated asset and the RL environment.

Verifies the things that silently go wrong when a modelled USD is wired into Isaac Lab:
the articulation parses with the expected degrees of freedom, the base really is fixed,
gravity acts on the poles, and the single actuated joint actually moves the cart.

Usage::

    .\\run.ps1 scripts/check_env.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import triple_cartpole  # noqa: F401
from triple_cartpole import TASK_ID
from triple_cartpole.robot import CART_JOINT, POLE_JOINTS

RESULTS: list[tuple[bool, str]] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    RESULTS.append((bool(condition), label if not detail else f"{label} -- {detail}"))


def main() -> int:
    env_cfg = triple_cartpole.TripleCartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env_cfg.enable_termination = False
    env_cfg.episode_length_s = 60.0
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(TASK_ID, cfg=env_cfg).unwrapped
    robot = env.robot
    obs, _ = env.reset()

    print("\n=== articulation ===")
    print(f"  bodies ({robot.num_bodies}) : {robot.body_names}")
    print(f"  joints ({robot.num_joints}) : {robot.joint_names}")
    print(f"  masses [kg]         : {robot.data.default_mass[0].tolist()}")
    print(f"  fixed base          : {robot.is_fixed_base}")
    print(f"  obs shape           : {tuple(obs['policy'].shape)}")

    check(robot.num_joints == 4, "4 degrees of freedom", f"got {robot.num_joints}")
    check(robot.is_fixed_base, "articulation is fixed base")
    check(
        set(robot.joint_names) == {CART_JOINT, *POLE_JOINTS},
        "joint names match the config",
        f"got {robot.joint_names}",
    )
    check(obs["policy"].shape == (env.num_envs, env_cfg.observation_space), "observation width")
    check(
        torch.isfinite(obs["policy"]).all().item(),
        "observations are finite",
    )
    pole_idx, _ = robot.find_joints(list(POLE_JOINTS), preserve_order=True)
    check(len(set(pole_idx)) == 3, "three distinct pole joints resolved", f"{pole_idx}")

    root_pos_before = robot.data.root_pos_w.clone()

    print("\n=== falling under gravity (zero force, 180 steps) ===")
    zeros = torch.zeros(env.num_envs, 1, device=env.device)
    for _ in range(180):
        env.step(zeros)
    tilt = env._link_angle.abs().max().item()
    root_drift = (robot.data.root_pos_w - root_pos_before).abs().max().item()
    print(f"  max |tilt|          : {tilt:.3f} rad")
    print(f"  max root drift      : {root_drift:.2e} m")
    check(tilt > 0.2, "poles topple under gravity", f"max tilt {tilt:.3f} rad")
    check(root_drift < 1e-3, "base stays fixed while the poles swing", f"drift {root_drift:.2e} m")

    print("\n=== pushing the cart (full force, 60 steps) ===")
    env.reset()
    push = torch.ones(env.num_envs, 1, device=env.device)
    for _ in range(60):
        env.step(push)
    cart_pos = env._cart_pos.mean().item()
    print(f"  mean cart position  : {cart_pos:+.3f} m")
    check(cart_pos > 0.1, "actuated joint moves the cart", f"reached {cart_pos:+.3f} m")

    print("\n=== joint limits ===")
    limits = robot.data.joint_pos_limits[0]
    for name, lim in zip(robot.joint_names, limits.tolist()):
        print(f"  {name:<18}: [{lim[0]:+.3f}, {lim[1]:+.3f}]")

    print("\n=== summary ===")
    failures = 0
    for ok, label in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1
    print(f"\n{len(RESULTS) - failures}/{len(RESULTS)} checks passed")

    env.close()
    return 1 if failures else 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
