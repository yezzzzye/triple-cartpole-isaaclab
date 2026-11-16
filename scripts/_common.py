# SPDX-License-Identifier: MIT
"""Shared paths and policy loading for CLI entry points.

This module is safe to import before AppLauncher; simulator imports stay inside functions.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log_root(task: str) -> Path:
    name = "triple_cartpole_swingup" if task == "swingup" else "triple_cartpole"
    return ROOT / "logs" / "rsl_rl" / name


def resolve_checkpoint(path: str | Path | None = None, *, task: str = "swingup", latest: bool = False) -> Path:
    """Use an explicit file, the delivered policy, or the newest training checkpoint."""
    if path is not None:
        candidate = Path(path).expanduser()
        for found in (candidate, ROOT / candidate):
            if found.is_file():
                return found.resolve()
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    pinned = ROOT / "checkpoints" / f"{task}_policy.pt"
    if not latest and pinned.is_file():
        return pinned
    checkpoints = sorted(log_root(task).glob("*/model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        if pinned.is_file():
            return pinned
        raise FileNotFoundError(f"No {task} checkpoint found under {log_root(task)}")
    best = checkpoints[-1].parent / "model_best.pt"
    return best if not latest and best.is_file() else checkpoints[-1]


def make_runner(env, task_id: str, *, device: str, log_dir: Path | None = None, agent_cfg=None):
    """Wrap an existing environment and construct the registered RSL-RL runner."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    cfg = agent_cfg or load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    cfg.device = device
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, cfg.to_dict(), log_dir=str(log_dir) if log_dir else None, device=device)
    return wrapped, runner
