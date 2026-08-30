# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .robot import CART_JOINT, POLE_JOINTS, TRIPLE_CARTPOLE_CFG


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
    max_tilt = 0.35
    enable_termination = True

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

    rew_scale_alive = 1.0
    rew_scale_upright = 2.0
    rew_scale_terminated = -20.0
    rew_scale_cart_pos = -0.5
    rew_scale_cart_vel = -0.02
    rew_scale_link_rate = -0.01
    rew_scale_action = -0.05
    rew_scale_action_rate = -0.05


class TripleCartpoleEnv(DirectRLEnv):
    cfg: TripleCartpoleEnvCfg

    def __init__(self, cfg: TripleCartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(CART_JOINT)
        self._pole_dof_idx, _ = self.robot.find_joints(list(POLE_JOINTS), preserve_order=True)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._previous_actions = torch.zeros_like(self.actions)
        self._effort = torch.zeros_like(self.actions)

        if self.cfg.push_force > 0.0:
            self._push_body_ids, _ = self.robot.find_bodies(self.cfg.push_body)
            self._push_wrench = self.robot.permanent_wrench_composer
            self._push_wrench_b = torch.zeros(self.num_envs, len(self._push_body_ids), 3, device=self.device)
            self._push_period = max(1, round(self.cfg.push_interval_s / self.step_dt))
            self._push_hold = max(1, round(0.001 * self.cfg.push_ms / self.step_dt))
            self._push_timer = torch.randint(1, self._push_period + 1, (self.num_envs,), device=self.device)
            self._push_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._compute_intermediate_values()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        thickness = 0.5
        ground_cfg = sim_utils.CuboidCfg(
            size=(self.cfg.ground_size, self.cfg.ground_size, thickness),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.21, 0.23, 0.27), roughness=0.85),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        )
        ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, self.cfg.ground_top - 0.5 * thickness))

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.87, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self) -> None:
        self._cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
        self._cart_vel = self.joint_vel[:, self._cart_dof_idx[0]]
        self._link_angle = torch.cumsum(self.joint_pos[:, self._pole_dof_idx], dim=-1)
        self._link_rate = torch.cumsum(self.joint_vel[:, self._pole_dof_idx], dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)
        self._effort = self.cfg.action_scale * self.actions
        if self.cfg.push_force > 0.0:
            self._apply_push()

    def _apply_push(self) -> None:
        self._push_timer -= 1
        fire = self._push_timer <= 0
        self._push_timer = torch.where(fire, torch.full_like(self._push_timer, self._push_period), self._push_timer)
        self._push_left = torch.where(fire, torch.full_like(self._push_left, self._push_hold), self._push_left)
        sampled = sample_uniform(-self.cfg.push_force, self.cfg.push_force, (self.num_envs,), self.device)
        held = self._push_wrench_b[:, 0, 0]
        self._push_wrench_b[:, 0, 0] = torch.where(fire, sampled, held) * (self._push_left > 0).float()
        self._push_left = (self._push_left - 1).clamp(min=0)
        self._push_wrench.set_forces_and_torques(forces=self._push_wrench_b, body_ids=self._push_body_ids)

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self._effort, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        obs = torch.cat(
            (
                (self._cart_pos / self.cfg.max_cart_pos).unsqueeze(-1),
                (self._cart_vel / self.cfg.cart_vel_norm).unsqueeze(-1),
                torch.sin(self._link_angle),
                torch.cos(self._link_angle),
                self._link_rate / self.cfg.link_rate_norm,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        terminated = self.reset_terminated.float()
        upright = torch.cos(self._link_angle).mean(dim=-1)

        reward = (
            cfg.rew_scale_alive * (1.0 - terminated)
            + cfg.rew_scale_upright * upright
            + cfg.rew_scale_terminated * terminated
            + cfg.rew_scale_cart_pos * (self._cart_pos / cfg.max_cart_pos).square()
            + cfg.rew_scale_cart_vel * self._cart_vel.square()
            + cfg.rew_scale_link_rate * self._link_rate.square().sum(dim=-1)
            + cfg.rew_scale_action * self.actions.square().sum(dim=-1)
            + cfg.rew_scale_action_rate * (self.actions - self._previous_actions).square().sum(dim=-1)
        )
        self._previous_actions = self.actions.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if not self.cfg.enable_termination:
            return torch.zeros_like(time_out), time_out
        out_of_bounds = self._cart_pos.abs() > self.cfg.max_cart_pos
        toppled = (self._link_angle.abs() > self.cfg.max_tilt).any(dim=-1)
        return out_of_bounds | toppled, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        cfg = self.cfg
        num_reset = len(env_ids)
        num_poles = len(self._pole_dof_idx)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        joint_pos[:, self._cart_dof_idx] += sample_uniform(
            -cfg.init_cart_pos_range, cfg.init_cart_pos_range, (num_reset, 1), self.device
        )
        joint_vel[:, self._cart_dof_idx] += sample_uniform(
            -cfg.init_cart_vel_range, cfg.init_cart_vel_range, (num_reset, 1), self.device
        )
        joint_pos[:, self._pole_dof_idx] += cfg.init_pole_angle_bias + sample_uniform(
            -cfg.init_pole_angle_range, cfg.init_pole_angle_range, (num_reset, num_poles), self.device
        )
        joint_vel[:, self._pole_dof_idx] += sample_uniform(
            -cfg.init_pole_vel_range, cfg.init_pole_vel_range, (num_reset, num_poles), self.device
        )

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        if self.cfg.push_force > 0.0:
            self._push_left[env_ids] = 0
            self._push_timer[env_ids] = torch.randint(1, self._push_period + 1, (num_reset,), device=self.device)
