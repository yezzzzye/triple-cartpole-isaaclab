# SPDX-License-Identifier: MIT
"""Render one continuous policy rollout, with optional external impulses and no resets."""
from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher
from _common import ROOT, make_runner, resolve_checkpoint

sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("swingup", "balance"), default="swingup")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--initial_pose", choices=("downward", "random", "upright"), default=None)
parser.add_argument("--steps", type=int, default=0, help="Stop after N steps; 0 runs until the window closes.")
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--disturb", type=float, default=None, help="External force on pole3 [N]; 0 disables it.")
parser.add_argument("--disturb_every", type=float, default=25.0)
parser.add_argument("--disturb_ms", type=float, default=200.0)
parser.add_argument("--real_time", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 0 or args.disturb_every <= 0 or args.disturb_ms <= 0:
    parser.error("Invalid environment count, horizon, or impulse timing.")
args.initial_pose = args.initial_pose or ("downward" if args.task == "swingup" else "upright")
args.disturb = args.disturb if args.disturb is not None else (12.0 if args.task == "swingup" else 0.0)
if args.disturb < 0:
    parser.error("Disturbance force must be nonnegative.")
if args.task == "balance" and args.initial_pose != "upright":
    parser.error("Use the swingup task for downward or large random initial states.")
checkpoint = resolve_checkpoint(args.checkpoint, task=args.task)
app = AppLauncher(args).app

import gymnasium as gym
import torch

from triple_cartpole import SWINGUP_TASK_ID, TASK_ID, TripleCartpoleEnvCfg, TripleCartpoleSwingupEnvCfg
from triple_cartpole.env_cfg import set_swingup_initial_state

torch.set_num_threads(1)


def main() -> None:
    cfg = TripleCartpoleSwingupEnvCfg() if args.task == "swingup" else TripleCartpoleEnvCfg()
    task_id = SWINGUP_TASK_ID if args.task == "swingup" else TASK_ID
    cfg.scene.num_envs = args.num_envs
    if args.num_envs == 1:
        cfg.scene.clone_in_fabric = False
    cfg.seed = args.seed
    cfg.push_force = 0.0
    cfg.enable_termination = False
    cfg.episode_length_s = 3600.0
    if args.device:
        cfg.sim.device = args.device
    if args.task == "swingup":
        initial = "down_exact" if args.initial_pose == "downward" else args.initial_pose
        set_swingup_initial_state(cfg, initial)
        cfg.viewer.eye, cfg.viewer.lookat = (12.0, -32.0, 5.0), (0.0, 0.0, 0.0)
    env, runner = make_runner(gym.make(task_id, cfg=cfg), task_id, device=cfg.sim.device)
    runner.load(str(checkpoint))
    raw = env.unwrapped
    policy = runner.get_inference_policy(device=raw.device)
    print(f"[demo] policy={checkpoint}; initial={args.initial_pose}; no mid-rollout resets", flush=True)
    if args.disturb:
        body_ids, _ = raw.robot.find_bodies("pole3")
        wrench = raw.robot.permanent_wrench_composer
        forces = torch.zeros(raw.num_envs, 1, 3, device=raw.device)
        period = max(1, round(args.disturb_every / raw.step_dt))
        duration = max(1, round(args.disturb_ms / 1000 / raw.step_dt))
        kick_until, direction = -1, -1.0
    step = 0
    with torch.inference_mode():
        obs, _ = env.reset()
        runner.alg.policy.reset()
        while app.is_running() and (not args.steps or step < args.steps):
            if (raw.episode_length_buf >= raw.max_episode_length - 2).any():
                break  # finish before the simulator's timeout can reset the robot
            started = time.perf_counter()
            if args.disturb:
                if step > 0 and step % period == 0:
                    direction = -direction
                    kick_until = step + duration
                    print(f"[demo] impulse {direction * args.disturb:+g} N for {duration * raw.step_dt:g} s")
                forces[:, 0, 0] = direction * args.disturb if step < kick_until else 0.0
                wrench.set_forces_and_torques(forces=forces, body_ids=body_ids, is_global=True)
            obs, _, dones, _ = env.step(policy(obs))
            if dones.any():
                raise RuntimeError("Unexpected reset during the continuous demonstration.")
            step += 1
            if args.real_time:
                time.sleep(max(0.0, raw.step_dt - (time.perf_counter() - started)))
    tilt = torch.atan2(torch.sin(raw._link_angle), torch.cos(raw._link_angle)).abs().max().item()
    cart = raw._cart_pos.abs().max().item()
    status = "STABLE" if tilt <= cfg.max_tilt and cart < cfg.max_cart_pos else "FALLEN"
    print(f"[demo] {step * raw.step_dt:g} s: {status}; final tilt {tilt:.3f} rad; cart {cart:.3f} m")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
