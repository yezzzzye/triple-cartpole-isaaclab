# SPDX-License-Identifier: MIT
import gymnasium as gym

from . import agents
from .env import TripleCartpoleEnv, TripleCartpoleEnvCfg
from .robot import TRIPLE_CARTPOLE_CFG, TRIPLE_CARTPOLE_USD

TASK_ID = "Triple-Cartpole-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point=f"{__name__}.env:TripleCartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:TripleCartpoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TripleCartpolePPORunnerCfg",
    },
)

__all__ = [
    "TASK_ID",
    "TRIPLE_CARTPOLE_CFG",
    "TRIPLE_CARTPOLE_USD",
    "TripleCartpoleEnv",
    "TripleCartpoleEnvCfg",
]
