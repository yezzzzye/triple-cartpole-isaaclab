# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
TRIPLE_CARTPOLE_USD = ASSETS_DIR / "triple_cartpole_physics.usd"

CART_JOINT = "slider_to_cart"
POLE_JOINTS = ("cart_to_pole1", "pole1_to_pole2", "pole2_to_pole3")


TRIPLE_CARTPOLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(TRIPLE_CARTPOLE_USD),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={CART_JOINT: 0.0, **{name: 0.0 for name in POLE_JOINTS}},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=[CART_JOINT],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=list(POLE_JOINTS),
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
