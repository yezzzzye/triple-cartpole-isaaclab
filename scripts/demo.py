# SPDX-License-Identifier: MIT
"""Render the triple inverted pendulum, with or without a trained policy.

Modes:

``auto``    play back a trained checkpoint if one exists, otherwise fall back to ``free``.
``policy``  run a trained policy.
``free``    no force on the cart. Termination is switched off, so the pendulum topples
            and swings chaotically -- the quickest way to sanity-check the asset.
``sine``    sweep the cart back and forth with a sinusoidal force.

``--disturb`` layers on top of any mode: every few seconds an external sideways force is
applied to a link for a short burst, alternating direction, and the peak tilt reached
before the next kick is printed. This is the disturbance-rejection test -- the balancing
is done entirely by the policy, which never sees the kick coming.

``--showcase`` is the ready-to-record preset: start visibly tilted, let the policy recover,
then kick the top link every few seconds. Automatic termination is disabled so a failed
recovery remains visible instead of silently teleporting back to the reset pose.

By default the scene is reset every 10 seconds and the same demonstration starts again.
Use ``--repeat_s 0`` to disable looping.

Usage::

    .\\run.ps1 policy
    .\\run.ps1 disturb
    .\\run.ps1 demo --disturb 3 --disturb_every 2 --disturb_body pole1
    .\\run.ps1 demo --showcase --repeat_s 0
    .\\run.ps1 demo --showcase --manual_force --disturb 0 --repeat_s 0
    .\\run.ps1 demo --initial_pose fallen --initial_hold_s 1
    .\\run.ps1 demo --mode sine --num_envs 1
    .\\run.ps1 demo --mode free --initial_tilt_deg 5 --initial_hold_s 1 --repeat_s 10
    .\\run.ps1 demo --checkpoint logs/rsl_rl/triple_cartpole/<run>/model_2700.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=1, help="Number of pendulums to render.")
parser.add_argument("--mode", choices=["auto", "policy", "free", "sine"], default="auto")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to an rsl-rl checkpoint (.pt).")
parser.add_argument("--steps", type=int, default=0, help="Stop after N steps. 0 runs until the window is closed.")
parser.add_argument("--repeat_s", type=float, default=10.0, help="Reset and replay every N seconds. 0 disables it.")
parser.add_argument("--sine_hz", type=float, default=0.2, help="Frequency of the sine sweep [Hz].")
parser.add_argument("--showcase", action="store_true", help="Ready-to-record tilted recovery and disturbance demo.")
parser.add_argument(
    "--initial_pose",
    choices=["upright", "tilted", "fallen"],
    default="upright",
    help="Initial pose. 'fallen' places all links horizontally; the balance-only policy may not recover.",
)
parser.add_argument(
    "--initial_tilt_deg",
    type=float,
    default=None,
    help="Absolute tilt of all links in degrees; overrides --initial_pose.",
)
parser.add_argument(
    "--initial_hold_s", type=float, default=0.0, help="Freeze the requested initial pose for this many seconds."
)
parser.add_argument(
    "--disturb", type=float, nargs="?", const=1.0, default=0.0, help="Sideways kick force [N]. 0 disables it."
)
parser.add_argument("--disturb_every", type=float, default=3.0, help="Seconds between kicks.")
parser.add_argument("--disturb_ms", type=float, default=100.0, help="Duration of each kick [ms].")
parser.add_argument("--disturb_body", type=str, default="pole3", help="Link(s) the kick is applied to.")
parser.add_argument(
    "--disturb_marker_s",
    type=float,
    default=0.75,
    help="Keep a red marker on the disturbed body for this many seconds.",
)
parser.add_argument("--manual_force", action="store_true", help="Show an Isaac Sim panel for manual force control.")
parser.add_argument("--manual_force_value", type=float, default=2.0, help="Initial force value in the UI [N].")
parser.add_argument(
    "--real_time", action=argparse.BooleanOptionalAction, default=True, help="Throttle playback to wall-clock time."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.showcase:
    args_cli.mode = "policy"
    if args_cli.initial_tilt_deg is None and args_cli.initial_pose == "upright":
        args_cli.initial_pose = "tilted"
    if args_cli.initial_hold_s == 0.0:
        args_cli.initial_hold_s = 1.0
    if args_cli.disturb == 0.0 and not args_cli.manual_force:
        args_cli.disturb = 1.0

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


def resolve_mode() -> tuple[str, Path | None]:
    mode = args_cli.mode
    checkpoint = None
    if args_cli.checkpoint:
        path = Path(args_cli.checkpoint)
        checkpoint = next((candidate for candidate in (path, REPO_ROOT / path) if candidate.is_file()), None)
        if checkpoint is None:
            raise SystemExit(f"Checkpoint not found: {args_cli.checkpoint}")
    if mode in ("auto", "policy") and checkpoint is None:
        checkpoint = default_checkpoint()
    if mode == "auto":
        mode = "policy" if checkpoint else "free"
    if mode == "policy" and checkpoint is None:
        raise SystemExit(f"No checkpoint found under {LOG_ROOT}. Train first, or use --mode free.")
    return mode, checkpoint


MODE, CHECKPOINT = resolve_mode()


def requested_initial_tilt_deg() -> float:
    if args_cli.initial_tilt_deg is not None:
        return args_cli.initial_tilt_deg
    return {"upright": 0.0, "tilted": 2.0, "fallen": 90.0}[args_cli.initial_pose]


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from pxr import Gf

import triple_cartpole  # noqa: F401
from triple_cartpole import TASK_ID


def main() -> None:
    mode, checkpoint = MODE, CHECKPOINT
    initial_tilt_deg = requested_initial_tilt_deg()

    env_cfg = triple_cartpole.TripleCartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if initial_tilt_deg:
        env_cfg.init_cart_pos_range = 0.0
        env_cfg.init_cart_vel_range = 0.0
        env_cfg.init_pole_angle_range = 0.0
        env_cfg.init_pole_vel_range = 0.0
        env_cfg.robot_cfg.init_state.joint_pos["cart_to_pole1"] = math.radians(initial_tilt_deg)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if mode != "policy":
        env_cfg.enable_termination = False
        env_cfg.episode_length_s = 20.0
    if args_cli.disturb or args_cli.manual_force or initial_tilt_deg or args_cli.initial_hold_s:
        # Keep failures visible. Otherwise DirectRLEnv immediately resets a policy as soon as
        # any link exceeds max_tilt, which looks deceptively like a successful recovery.
        env_cfg.enable_termination = False
        env_cfg.episode_length_s = 3600.0

    print(f"[demo] mode={mode}" + (f" checkpoint={checkpoint}" if mode == "policy" else ""))
    env = gym.make(TASK_ID, cfg=env_cfg)

    policy = None
    policy_nn = None
    if mode == "policy":
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry

        agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = runner.alg.policy
        obs = env.get_observations()
    else:
        obs, _ = env.reset()

    unwrapped = env.unwrapped
    robot = unwrapped.robot
    dt = unwrapped.step_dt
    zeros = torch.zeros(unwrapped.num_envs, 1, device=unwrapped.device)
    step = 0

    print(f"[demo] {unwrapped.num_envs} env(s), {robot.num_joints} DOF: {robot.joint_names}")

    initial_joint_pos = robot.data.default_joint_pos.clone()
    initial_joint_vel = torch.zeros_like(robot.data.default_joint_vel)

    if initial_tilt_deg:
        print(f"[demo] initial aligned-link tilt {initial_tilt_deg:+g} deg")

    hold_steps = max(0, round(args_cli.initial_hold_s / dt))
    if hold_steps:
        print(f"[demo] holding the initial pose for {hold_steps * dt:.1f} s")

    repeat_steps = max(0, round(args_cli.repeat_s / dt))
    if repeat_steps:
        print(f"[demo] repeating every {repeat_steps * dt:g} s")

    kick_ids: list[int] = []
    kick_until, kick_sign = -1, -1.0
    peak_tilt = abs(math.radians(initial_tilt_deg))
    interval_peak_tilt = peak_tilt
    if args_cli.disturb or args_cli.manual_force:
        kick_ids, kick_names = robot.find_bodies(args_cli.disturb_body)
        if not kick_ids:
            raise SystemExit(f"No link matches '{args_cli.disturb_body}'. Links: {robot.body_names}")
        wrench = robot.permanent_wrench_composer
        kick = torch.zeros(unwrapped.num_envs, len(kick_ids), 3, device=unwrapped.device)
        kick_period = max(1, round(args_cli.disturb_every / dt))
        kick_hold = max(1, round(0.001 * args_cli.disturb_ms / dt))
        if args_cli.disturb:
            print(
                f"[demo] disturbance {args_cli.disturb:g} N on {', '.join(kick_names)}"
                f" for {args_cli.disturb_ms:g} ms every {args_cli.disturb_every:g} s"
            )
        if args_cli.manual_force:
            print(f"[demo] manual force UI enabled for {', '.join(kick_names)}")

        marker_hold = max(kick_hold, round(args_cli.disturb_marker_s / dt))
        marker_until = -1

        impact_prims = []
        for marker_index in range(unwrapped.num_envs * len(kick_ids)):
            marker_cfg = sim_utils.SphereCfg(
                radius=0.45,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                    emissive_color=(1.0, 0.0, 0.0),
                    opacity=0.78,
                ),
            )
            marker_prim = marker_cfg.func(
                f"/World/Visuals/DisturbanceImpact_{marker_index}",
                marker_cfg,
                translation=(0.0, 0.0, -1000.0),
            )
            sim_utils.set_prim_visibility(marker_prim, False)
            impact_prims.append(marker_prim)

        def show_disturbance_annotation(positions: torch.Tensor) -> None:
            points = positions.detach().cpu().tolist()
            for marker_prim, (x, y, z) in zip(impact_prims, points, strict=True):
                marker_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))
                sim_utils.set_prim_visibility(marker_prim, True)

        def hide_disturbance_annotation() -> None:
            for marker_prim in impact_prims:
                sim_utils.set_prim_visibility(marker_prim, False)

    ui_state = {"pending_force": 0.0, "reset": False, "auto": bool(args_cli.disturb)}
    control_window = None
    status_label = None
    if args_cli.manual_force:
        if args_cli.headless:
            print("[demo] manual force UI is hidden in headless mode")
        else:
            import omni.ui as ui

            force_model = ui.SimpleFloatModel(abs(args_cli.manual_force_value))
            control_window = ui.Window("Triple Cartpole - Manual Force", width=360, height=245)
            with control_window.frame:
                with ui.VStack(spacing=8, style={"margin": 10}):
                    ui.Label("Stable policy is running", height=24)
                    ui.Label(f"Target body: {', '.join(kick_names)}", height=22)
                    with ui.HStack(height=28):
                        ui.Label("Force (N)", width=80)
                        ui.FloatField(model=force_model, width=65)
                        ui.FloatSlider(model=force_model, min=0.1, max=10.0, step=0.1)

                    def request_manual_force(sign: float) -> None:
                        ui_state["pending_force"] = sign * max(0.1, abs(force_model.get_value_as_float()))

                    with ui.HStack(height=42, spacing=8):
                        ui.Button("<-  Push -X", clicked_fn=lambda: request_manual_force(-1.0))
                        ui.Button("Push +X  ->", clicked_fn=lambda: request_manual_force(1.0))

                    with ui.HStack(height=36, spacing=8):
                        if args_cli.disturb:
                            auto_button = ui.Button(
                                "Auto disturbance: ON" if ui_state["auto"] else "Auto disturbance: OFF"
                            )

                            def toggle_auto() -> None:
                                ui_state["auto"] = not ui_state["auto"]
                                auto_button.text = (
                                    "Auto disturbance: ON" if ui_state["auto"] else "Auto disturbance: OFF"
                                )

                            auto_button.set_clicked_fn(toggle_auto)
                        else:
                            ui.Label("Automatic disturbance: disabled")
                        ui.Button("Reset now", clicked_fn=lambda: ui_state.__setitem__("reset", True))
                    status_label = ui.Label("Ready - use the buttons to apply force", height=28)

    cycle = 1
    cycle_step = 0

    def start_kick(signed_force: float, source: str) -> None:
        nonlocal kick_until, marker_until, interval_peak_tilt
        kick[..., 0] = signed_force
        kick_until = step + kick_hold
        marker_until = step + marker_hold
        interval_peak_tilt = 0.0
        marker_positions = robot.data.body_pos_w[:, kick_ids, :].reshape(-1, 3)
        show_disturbance_annotation(marker_positions)
        message = f"{source}: {signed_force:+.1f} N -> {', '.join(kick_names)}"
        print(f"[demo] {message} [RED MARKER]")
        if status_label is not None:
            status_label.text = message

    def report_cycle(label: str) -> None:
        final_tilt = unwrapped._link_angle.abs().max().item()
        final_cart = unwrapped._cart_pos.abs().max().item()
        status = (
            "STABLE"
            if final_tilt <= unwrapped.cfg.max_tilt and final_cart <= unwrapped.cfg.max_cart_pos
            else "FALLEN"
        )
        print(
            f"[demo] {label}: {status}; peak tilt {peak_tilt:.3f} rad,"
            f" final tilt {final_tilt:.3f} rad, final cart {final_cart:.3f} m"
        )

    while simulation_app.is_running():
        loop_start = time.time()

        if (repeat_steps and cycle_step >= repeat_steps) or ui_state["reset"]:
            report_cycle(f"cycle {cycle} finished")
            ui_state["reset"] = False
            cycle += 1
            cycle_step = 0
            if kick_ids:
                wrench.reset()
                hide_disturbance_annotation()
            kick_until, kick_sign = -1, -1.0
            marker_until = -1
            peak_tilt = abs(math.radians(initial_tilt_deg))
            interval_peak_tilt = peak_tilt
            # Trigger the environment's regular timeout reset inside env.step(). This keeps
            # PhysX state writes in IsaacLab's safe simulation-step lifecycle.
            unwrapped.episode_length_buf.fill_(unwrapped.max_episode_length - 1)
            with torch.inference_mode():
                if mode == "policy":
                    obs, _, dones, _ = env.step(zeros)
                    policy_nn.reset(dones)
                else:
                    obs, _, _, _, _ = env.step(zeros)
            print(f"[demo] cycle {cycle} started")

        if kick_ids:
            demo_step = cycle_step - hold_steps
            if args_cli.disturb and ui_state["auto"] and demo_step > 0 and demo_step % kick_period == 0:
                if kick_until > 0:
                    print(f"[demo] peak tilt since the last kick {interval_peak_tilt:.3f} rad")
                kick_sign = -kick_sign
                start_kick(kick_sign * args_cli.disturb, "auto kick")
            if ui_state["pending_force"]:
                start_kick(ui_state["pending_force"], "manual kick")
                ui_state["pending_force"] = 0.0
            if step < kick_until:
                wrench.reset()
                wrench.set_forces_and_torques(forces=kick, body_ids=kick_ids, is_global=True)
            elif step == kick_until:
                wrench.reset()
            if step == marker_until:
                hide_disturbance_annotation()

        with torch.inference_mode():
            if cycle_step < hold_steps:
                actions = zeros
            elif mode == "policy":
                actions = policy(obs)
            else:
                if mode == "sine":
                    actions = zeros + math.sin(2.0 * math.pi * args_cli.sine_hz * cycle_step * dt)
                else:
                    actions = zeros

            if mode == "policy":
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)
            else:
                obs, _, _, _, _ = env.step(actions)

            if cycle_step < hold_steps:
                robot.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel)
                obs = env.get_observations() if mode == "policy" else unwrapped._get_observations()

        current_tilt = unwrapped._link_angle.abs().max().item()
        peak_tilt = max(peak_tilt, current_tilt)
        if kick_ids:
            interval_peak_tilt = max(interval_peak_tilt, current_tilt)

        step += 1
        cycle_step += 1
        if args_cli.steps and step >= args_cli.steps:
            break
        if args_cli.real_time:
            remaining = dt - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    report_cycle(f"cycle {cycle} stopped")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
