# SPDX-License-Identifier: MIT
import gymnasium as gym

from . import agents
from .env import TripleCartpoleEnv
from .env_cfg import TripleCartpoleEnvCfg, TripleCartpoleSwingupEnvCfg
from .robot import TRIPLE_CARTPOLE_CFG, TRIPLE_CARTPOLE_USD

TASK_ID = "Triple-Cartpole-Direct-v0"
SWINGUP_TASK_ID = "Triple-Cartpole-Swingup-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point=f"{__name__}.env:TripleCartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TripleCartpoleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TripleCartpolePPORunnerCfg",
    },
)

gym.register(
    id=SWINGUP_TASK_ID,
    entry_point=f"{__name__}.env:TripleCartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TripleCartpoleSwingupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TripleCartpoleSwingupPPORunnerCfg",
    },
)

__all__ = [
    "TASK_ID",
    "SWINGUP_TASK_ID",
    "TRIPLE_CARTPOLE_CFG",
    "TRIPLE_CARTPOLE_USD",
    "TripleCartpoleEnv",
    "TripleCartpoleEnvCfg",
    "TripleCartpoleSwingupEnvCfg",
]
