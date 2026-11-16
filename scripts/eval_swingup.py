# SPDX-License-Identifier: MIT
"""Continuous swing-up and recovery: one actor, no resets or state writes after release.

Examples:
    .\run.ps1 scripts/eval_swingup.py --headless --initial down --checkpoint path/to/model.pt
    .\run.ps1 scripts/eval_swingup.py --headless --initial random --push_force 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
from _common import ROOT, make_runner, resolve_checkpoint

sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--initial", choices=("down", "down_exact", "random", "upright"), default="down")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episode_s", type=float, default=40.0)
parser.add_argument("--hold_s", type=float, default=10.0)
parser.add_argument("--tilt_limit", type=float, default=0.35, help="Max absolute angle of each link [rad].")
parser.add_argument("--push_force", type=float, default=4.0)
parser.add_argument("--init_tilt", type=float, default=None, help="Override near-upright reset angle range [rad].")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--recovery_force", type=float, default=0.0, help="Global X force after the first 10-second hold [N].")
parser.add_argument("--recovery_ms", type=float, default=200.0, help="Duration of the recovery impulse [ms].")
parser.add_argument("--output", type=Path, default=None, help="Save aggregate and per-trial results as JSON.")
parser.add_argument("--video", action="store_true", help="Record the complete uncut rollout.")
parser.add_argument("--camera", choices=("three_quarter", "front", "side"), default="three_quarter")
parser.add_argument("--video_dir", type=Path, default=ROOT / "reports" / "videos")
parser.add_argument("--video_name", type=str, default=None)
parser.add_argument("--trace", type=Path, default=None, help="Save every step of environment 0 as CSV.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.hold_s < 10 or args.episode_s <= args.hold_s:
    parser.error("Use --hold_s >= 10 and an episode longer than the hold.")
if args.recovery_force < 0 or args.recovery_ms <= 0:
    parser.error("Recovery force must be nonnegative and its duration positive.")
if args.recovery_force and args.push_force:
    parser.error("Use --push_force 0 for the isolated recovery test.")
if args.video:
    args.enable_cameras = True

app = AppLauncher(args).app

import math
import time
import hashlib
import json
import importlib.metadata
import csv

import gymnasium as gym
import torch

import triple_cartpole
from triple_cartpole import SWINGUP_TASK_ID
from triple_cartpole.env_cfg import set_swingup_initial_state

torch.set_num_threads(1)


def main() -> None:
    path = resolve_checkpoint(args.checkpoint)
    cfg = triple_cartpole.TripleCartpoleSwingupEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.num_envs == 1:
        cfg.scene.clone_in_fabric = False
    cfg.episode_length_s = args.episode_s + 2.0
    cfg.enable_termination = False  # no mid-evaluation reset, even after track failure
    cfg.push_force = args.push_force
    cfg.push_in_global_frame = True
    if args.init_tilt is not None:
        cfg.init_pole_angle_range = args.init_tilt
    cfg.seed = args.seed
    set_swingup_initial_state(cfg, args.initial)
    if args.device:
        cfg.sim.device = args.device

    if args.video:
        camera_eyes = {
            "three_quarter": (17.0, -47.0, 7.0),
            "front": (0.0, -50.0, 7.0),
            "side": (45.0, -25.0, 10.0),
        }
        cfg.viewer.eye = camera_eyes[args.camera]
        cfg.viewer.lookat = (0.0, 0.0, 1.0)
    env = gym.make(SWINGUP_TASK_ID, cfg=cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        video_dir = args.video_dir if args.video_dir.is_absolute() else ROOT / args.video_dir
        video_name = args.video_name or f"swingup_{args.initial}_{args.camera}_seed{args.seed}"
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(video_dir), step_trigger=lambda step: step == 0,
            video_length=math.ceil(args.episode_s * 60), name_prefix=video_name,
            disable_logger=True,
        )
    env, runner = make_runner(env, SWINGUP_TASK_ID, device=cfg.sim.device)
    runner.load(str(path))
    print(f"[eval] loaded {path}", flush=True)
    policy = runner.get_inference_policy(device=cfg.sim.device)
    policy_nn = runner.alg.policy

    raw = env.unwrapped
    n = raw.num_envs
    horizon = round(args.episode_s / raw.step_dt)
    hold_steps = round(args.hold_s / raw.step_dt)
    alive = torch.ones(n, dtype=torch.bool, device=raw.device)
    streak = torch.zeros(n, dtype=torch.long, device=raw.device)
    longest_streak = torch.zeros(n, dtype=torch.long, device=raw.device)
    achieved = torch.zeros(n, dtype=torch.bool, device=raw.device)
    first_success = torch.full((n,), float("nan"), device=raw.device)
    peak_cart = torch.zeros(n, device=raw.device)
    best_height = torch.full((n,), -1.0, device=raw.device)
    time_alive = torch.zeros(n, device=raw.device)
    kick_start = torch.full((n,), -1, dtype=torch.long, device=raw.device)
    kick_steps = max(1, math.ceil(args.recovery_ms / 1000 / raw.step_dt))
    recovered = torch.zeros_like(achieved)
    recovery_streak = torch.zeros_like(streak)
    recovery_time = torch.full_like(first_success, float("nan"))
    post_kick_peak_tilt = torch.zeros(n, device=raw.device)
    left_upright = torch.zeros_like(achieved)
    max_force = torch.zeros(n, device=raw.device)
    if args.recovery_force:
        kick_body_ids, _ = raw.robot.find_bodies("pole3")
        kick_wrench = raw.robot.permanent_wrench_composer
        kick_force = torch.zeros(n, len(kick_body_ids), 3, device=raw.device)
        kick_sign = torch.where(torch.arange(n, device=raw.device) % 2 == 0, 1.0, -1.0)

    torch.manual_seed(args.seed)
    started = time.perf_counter()
    trace_rows = []
    with torch.inference_mode():
        obs, _ = env.reset()
        initial_observations = obs["policy"].clone()
        if args.video:
            # Warm up the renderer without advancing physics or rewriting robot states.
            for _ in range(10):
                raw.render()  # creates and warms the RGB annotator as well as the renderer
        print(f"[eval] reset {n} environments", flush=True)
        policy_nn.reset()
        for step in range(horizon):
            if args.recovery_force:
                start_now = achieved & alive & (kick_start < 0)
                kick_start[start_now] = step
                kicking = (kick_start >= 0) & (step < kick_start + kick_steps)
                kick_force[:, 0, 0] = args.recovery_force * kick_sign * kicking.float()
                kick_wrench.set_forces_and_torques(forces=kick_force, body_ids=kick_body_ids, is_global=True)
            action = policy(obs)
            if not torch.isfinite(action).all():
                raise RuntimeError("Policy produced a non-finite action.")
            obs, _, dones, _ = env.step(action)
            policy_nn.reset(dones)
            if dones.any():
                raise RuntimeError("Unexpected reset during continuous evaluation.")
            # The simulator keeps every trajectory running; failures are masked from
            # scoring and never trigger a reset or a controller change.
            policy_obs = obs["policy"]
            cos_links = policy_obs[:, 5:8]
            cart = policy_obs[:, 0].abs() * cfg.cart_pos_norm
            alive &= ~dones.bool().reshape(-1) & (cart < cfg.max_cart_pos)
            time_alive += alive.float() * raw.step_dt
            upright = (cos_links > math.cos(args.tilt_limit)).all(dim=-1) & (cart < cfg.max_cart_pos)
            streak = torch.where(alive & upright, streak + 1, 0)
            longest_streak = torch.maximum(longest_streak, streak)
            new_success = alive & ~achieved & (streak >= hold_steps)
            first_success[new_success] = (step + 1) * raw.step_dt
            achieved |= new_success
            max_force = torch.maximum(max_force, raw._effort[:, 0].abs())
            if args.recovery_force:
                after_kick = (kick_start >= 0) & (step >= kick_start + kick_steps)
                tilt = torch.acos(cos_links.clamp(-1.0, 1.0)).amax(dim=-1)
                during_or_after_kick = kick_start >= 0
                post_kick_peak_tilt = torch.where(
                    during_or_after_kick, torch.maximum(post_kick_peak_tilt, tilt), post_kick_peak_tilt
                )
                left_upright |= during_or_after_kick & ~upright
                recovery_streak = torch.where(after_kick & alive & upright, recovery_streak + 1, 0)
                newly_recovered = ~recovered & (recovery_streak >= hold_steps)
                recovery_time[newly_recovered] = (
                    step + 1 - kick_start[newly_recovered] - kick_steps
                ) * raw.step_dt
                recovered |= newly_recovered
            if args.trace:
                theta = torch.atan2(policy_obs[0, 2:5], cos_links[0])
                trace_rows.append([
                    (step + 1) * raw.step_dt, policy_obs[0, 0].item() * cfg.cart_pos_norm,
                    *theta.cpu().tolist(), *policy_obs[0, 8:11].mul(cfg.link_rate_norm).cpu().tolist(),
                    raw._effort[0, 0].item(), kick_force[0, 0, 0].item() if args.recovery_force else 0.0,
                    streak[0].item() * raw.step_dt, recovery_streak[0].item() * raw.step_dt,
                ])
            peak_cart = torch.maximum(peak_cart, cart)
            best_height = torch.where(alive, torch.maximum(best_height, cos_links.mean(dim=-1)), best_height)
            if (step + 1) % max(10, min(600, horizon // 4)) == 0:
                print(f"[eval] {step + 1}/{horizon} steps in {time.perf_counter() - started:.1f} s", flush=True)

    print(f"checkpoint: {path}")
    print(f"initial: {args.initial}; {n} trials; horizon: {args.episode_s:g} s; random push: +/-{args.push_force:g} N every {cfg.push_interval_s:g} s")
    print(f"criterion: all three links within +/-{args.tilt_limit:g} rad for {args.hold_s:g} continuous seconds, cart within +/-{cfg.max_cart_pos:g} m")
    print(f"success: {achieved.sum().item()}/{n} ({achieved.float().mean().item() * 100:.1f}%)")
    print(f"ever upright: {(longest_streak > 0).sum().item()}/{n}; mean longest hold: {longest_streak.float().mean().item() * raw.step_dt:.2f} s")
    if achieved.any():
        print(f"mean time to complete hold: {first_success[achieved].mean().item():.2f} s")
    print(f"mean valid time before track limit or horizon: {time_alive.mean().item():.2f} s")
    print(f"mean best link-height score: {best_height.mean().item():.3f}; mean peak cart travel: {peak_cart.mean().item():.2f} m")
    if args.recovery_force:
        full_success = achieved & recovered & alive
        print(f"recovery impulse: +/-{args.recovery_force:g} N on pole3 for {kick_steps * raw.step_dt * 1000:g} ms after the first hold")
        print(f"complete swing-up/hold/impulse/hold: {full_success.sum().item()}/{n} ({100 * full_success.float().mean().item():.1f}%)")
        print(f"left upright tolerance under impulse: {left_upright.sum().item()}/{n}; recovered after leaving: {(full_success & left_upright).sum().item()}/{n}")
    if args.output:
        def values(tensor):
            return [None if isinstance(v, float) and not math.isfinite(v) else v for v in tensor.cpu().tolist()]

        report = {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": args.seed, "initial": args.initial, "num_trials": n,
            "horizon_s": args.episode_s, "hold_s": args.hold_s, "tilt_limit_rad": args.tilt_limit,
            "cart_limit_m": cfg.max_cart_pos, "control_dt_s": raw.step_dt,
            "push_force_N": args.push_force, "push_interval_s": cfg.push_interval_s,
            "push_duration_ms": cfg.push_ms, "recovery_force_N": args.recovery_force,
            "disturbance_frame": "global X",
            "recovery_duration_ms": kick_steps * raw.step_dt * 1000,
            "no_mid_trial_resets": True, "single_policy": True, "actor_class": type(policy_nn).__name__,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "rsl-rl-lib", "tensordict", "h5py")},
            "swingup_hold_successes": achieved.sum().item(),
            "recovery_sequence_successes": (achieved & recovered & alive).sum().item() if args.recovery_force else None,
            "recovered_after_tilt_excursion": (achieved & recovered & alive & left_upright).sum().item() if args.recovery_force else None,
            "trials": {
                "initial_observations": initial_observations.cpu().tolist(),
                "swingup_hold_success": values(achieved), "first_hold_completed_s": values(first_success),
                "longest_hold_s": values(longest_streak.float() * raw.step_dt),
                "valid_at_end": values(alive), "kick_started_s": values(torch.where(kick_start >= 0, kick_start * raw.step_dt, float("nan"))),
                "post_kick_hold_success": values(recovered), "post_kick_hold_completed_after_impulse_s": values(recovery_time),
                "post_kick_peak_tilt_rad": values(post_kick_peak_tilt), "left_upright_after_kick": values(left_upright),
                "max_cart_force_N": values(max_force),
                "peak_cart_travel_m": values(peak_cart),
                "final_absolute_tilt_rad": torch.acos(cos_links.clamp(-1.0, 1.0)).cpu().tolist(),
            },
        }
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[eval] saved {output}")
    if args.trace:
        trace_path = args.trace if args.trace.is_absolute() else ROOT / args.trace
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", newline="", encoding="utf-8") as trace_file:
            writer = csv.writer(trace_file)
            writer.writerow(["time_s", "cart_m", "pole1_rad", "pole2_rad", "pole3_rad",
                             "pole1_rate_rad_s", "pole2_rate_rad_s", "pole3_rate_rad_s",
                             "cart_force_N", "recovery_force_N", "upright_streak_s", "recovery_streak_s"])
            writer.writerows(trace_rows)
        print(f"[eval] saved {trace_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
