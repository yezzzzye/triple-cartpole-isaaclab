# SPDX-License-Identifier: MIT
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .robot import TRIPLE_CARTPOLE_CFG, TRIPLE_CARTPOLE_SWINGUP_CFG


@configclass
class TripleCartpoleEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 8.0
    action_space = 1
    observation_space = 11
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    robot_cfg: ArticulationCfg = TRIPLE_CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=12.0, replicate_physics=True, clone_in_fabric=True
    )
    viewer: ViewerCfg = ViewerCfg(eye=(0.0, -26.0, 9.0), lookat=(0.0, 0.0, 5.0), origin_type="env", env_index=0)

    ground_size = 400.0
    ground_top = -0.05

    action_scale = 50.0

    max_cart_pos = 2.8
    cart_pos_norm = 2.8
    max_tilt = 0.35
    enable_termination = True
    terminate_on_tilt = True
    reset_mode = "balance"
    upright_reset_fraction = 0.35
    downward_reset_fraction = 0.35
    downward_noise = 0.08
    curriculum_steps = 0
    start_upright_fraction = 0.35
    start_downward_fraction = 0.35
    start_upright_angle_range = 0.02

    init_cart_pos_range = 0.10
    init_cart_vel_range = 0.05
    init_pole_angle_range = 0.02
    init_pole_angle_bias = 0.0
    init_pole_vel_range = 0.02

    cart_vel_norm = 5.0
    link_rate_norm = 5.0

    push_force = 0.0
    push_interval_s = 2.0
    push_ms = 100.0
    push_body = "pole3"
    push_in_global_frame = False

    rew_scale_alive = 1.0
    rew_scale_upright = 2.0
    rew_scale_terminated = -20.0
    rew_scale_cart_pos = -0.5
    rew_scale_cart_vel = -0.02
    rew_scale_link_rate = -0.01
    rew_scale_action = -0.05
    rew_scale_action_rate = -0.05
    rew_scale_upright_bonus = 0.0
    rew_scale_stable_bonus = 0.0
    rew_scale_edge = 0.0
    rew_scale_energy = 0.0


@configclass
class TripleCartpoleSwingupEnvCfg(TripleCartpoleEnvCfg):
    """One force policy for swing-up, balance, and recovery from random pushes."""

    robot_cfg: ArticulationCfg = TRIPLE_CARTPOLE_SWINGUP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=30.0, replicate_physics=True, clone_in_fabric=True
    )
    episode_length_s = 40.0
    ground_size = 4000.0
    ground_top = -10.0  # all three 3 m links can hang freely below the cart
    action_scale = 50.0
    max_cart_pos = 11.8
    max_tilt = 0.35  # success threshold, not an episode termination
    terminate_on_tilt = False
    reset_mode = "swingup"
    init_cart_pos_range = 0.10
    init_cart_vel_range = 0.05
    init_pole_angle_range = 0.4
    init_pole_vel_range = 0.02
    cart_vel_norm = 5.0
    link_rate_norm = 5.0
    push_force = 4.0
    push_interval_s = 3.0

    curriculum_steps = 32000  # 1000 PPO iterations at 32 steps per iteration
    start_upright_fraction = 0.65
    start_downward_fraction = 0.20
    start_upright_angle_range = 0.08

    rew_scale_alive = 0.5
    rew_scale_upright = 3.0
    rew_scale_terminated = -50.0
    rew_scale_cart_pos = -0.5
    rew_scale_cart_vel = -0.01
    rew_scale_link_rate = 0.0
    rew_scale_action = -0.002
    rew_scale_action_rate = -0.005
    rew_scale_upright_bonus = 6.0
    rew_scale_stable_bonus = 2.0
    rew_scale_edge = -6.0
    rew_scale_energy = 3.0


def set_swingup_initial_state(cfg: TripleCartpoleSwingupEnvCfg, initial: str) -> None:
    """Select a release distribution through reset configuration, without changing live states."""
    if initial in ("down", "down_exact"):
        cfg.upright_reset_fraction, cfg.downward_reset_fraction = 0.0, 1.0
        if initial == "down_exact":
            cfg.downward_noise = 0.0
            cfg.init_cart_pos_range = cfg.init_cart_vel_range = cfg.init_pole_vel_range = 0.0
    elif initial == "random":
        cfg.upright_reset_fraction, cfg.downward_reset_fraction = 0.0, 0.0
    elif initial == "upright":
        cfg.upright_reset_fraction, cfg.downward_reset_fraction = 1.0, 0.0
    else:
        raise ValueError(f"Unknown initial distribution: {initial}")
    cfg.curriculum_steps = 0
    cfg.start_upright_fraction = cfg.upright_reset_fraction
    cfg.start_downward_fraction = cfg.downward_reset_fraction
