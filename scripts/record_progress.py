#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record a whole training run as one video: every checkpoint, in order, with a HUD.

A reward curve tells you *that* PPO learned something. This shows *what* it learned, by putting
every saved checkpoint through the same trial: released from the same lean, then kicked on the
top link with a force that escalates stage by stage. Early checkpoints cannot even recover the
lean; later ones recover it and then survive a shove that topples their predecessors.

The trial is the point. Replaying every checkpoint under nominal conditions makes for a video
that stops changing a few hundred iterations in, because nominal balance is solved by then --
what keeps improving afterwards only shows up when something is actively trying to knock the
pendulum over.

One simulator session covers the whole sweep -- Isaac Sim takes far longer to start than a
checkpoint takes to replay -- and frames are encoded as they are rendered, so memory does not
grow with the length of the video.

Usage::

    .\\run.ps1 scripts\\record_progress.py --headless
    .\\run.ps1 scripts\\record_progress.py --headless --lang zh
    .\\run.ps1 scripts\\record_progress.py --headless --every 200 --kicks 2 4 6 8
    .\\run.ps1 scripts\\record_progress.py --headless --lean_deg 0 --kicks    # plain replay
    .\\run.ps1 scripts\\record_progress.py --headless --run logs/.../<run> logs/.../<resumed>

Several runs are replayed as one sweep, which is what a resumed run is: the iteration counter
carries across, so the continuation belongs on the same curve.

Rendering is the point here, so ``--enable_cameras`` is switched on for you: ``--headless``
alone is enough.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--run",
    type=str,
    nargs="+",
    default=None,
    help="Training run directories, replayed as one sweep. Default: the newest one.",
)
parser.add_argument("--out", type=str, default=None, help="Output .mp4. Default: <run>/videos/training_progress.mp4")
parser.add_argument("--every", type=int, default=100, help="Only use checkpoints whose iteration divides this.")
parser.add_argument("--from_iter", type=int, default=None, help="Skip checkpoints before this iteration.")
parser.add_argument("--to_iter", type=int, default=None, help="Skip checkpoints after this iteration.")
parser.add_argument(
    "--epilogue",
    action="store_true",
    help="Also append the last checkpoint when it comes after the best one, to show a late collapse.",
)
parser.add_argument("--lean_s", type=float, default=5.0, help="Seconds of the opening recovery stage.")
parser.add_argument("--lean_deg", type=float, default=2.0, help="Lean every joint starts the clip at [deg].")
parser.add_argument("--kicks", type=float, nargs="*", default=[1.0, 2.0, 3.0, 4.0], help="Kick force per stage [N].")
parser.add_argument("--stage_s", type=float, default=4.0, help="Seconds per kick stage.")
parser.add_argument("--kick_every", type=float, default=2.0, help="Seconds between kicks inside a stage.")
parser.add_argument("--kick_ms", type=float, default=100.0, help="Duration of each kick [ms].")
parser.add_argument("--kick_body", type=str, default="pole3", help="Link the kick is applied to.")
parser.add_argument("--trial_max_tilt", type=float, default=1.2, help="Tilt at which a clip counts as a fall [rad].")
parser.add_argument("--seed", type=int, default=0, help="Every checkpoint faces the same initial states.")
parser.add_argument("--fps", type=int, default=30, help="Frame rate of the output video.")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed. 1.0 is real time.")
parser.add_argument("--num_envs", type=int, default=9, help="Environments to run. Environment 0 is the one on camera.")
parser.add_argument(
    "--env_spacing",
    type=float,
    default=60.0,
    help="Grid spacing between environments [m]. Wider than training, to keep neighbours out of shot.",
)
parser.add_argument("--width", type=int, default=1280, help="Frame width [px]; rounded down to a multiple of 16.")
parser.add_argument("--height", type=int, default=720, help="Frame height [px]; rounded down to a multiple of 16.")
parser.add_argument("--lang", choices=["en", "zh"], default="en", help="Language of the HUD labels.")
parser.add_argument("--font", type=str, default=None, help="TrueType font for the HUD. Default: picked per --lang.")
parser.add_argument("--eye", type=float, nargs=3, default=None, help="Camera position relative to environment 0 [m].")
parser.add_argument("--lookat", type=float, nargs=3, default=None, help="Camera target relative to environment 0 [m].")
parser.add_argument(
    "--markers",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Draw posts at the cart's travel limit, so its position is readable against the empty ground.",
)
parser.add_argument(
    "--cards",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Bookend the sweep with a title card and a summary card.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Rendering in headless mode needs the cameras extension. Same handling as train.py --video.
args_cli.enable_cameras = True

LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "triple_cartpole"


@dataclass
class Segment:
    """One checkpoint's slot in the video, plus what it did while being recorded."""

    path: Path
    iteration: int
    label: str = ""
    """``best`` / ``late``: keys into the text table, drawn as a badge."""

    falls: int = 0
    """Terminations of the environment on camera during this segment."""
    held: int = 0
    of: int = 0
    """``held`` of ``of`` environments never terminated during this segment."""
    mean_force: float = 0.0
    peak_tilt: float = 0.0
    peak_cart: float = 0.0
    rail_falls: int = 0
    """Of ``falls``, how many were the cart reaching its travel limit rather than a topple."""

    stage_held: list[int] = field(default_factory=list)
    """Per stage: how many environments had still never fallen when it ended."""

    @property
    def reached(self) -> int:
        """How many stages in a row left at least one environment standing.

        Scored over the whole population rather than over the environment on camera. A single
        stumble anywhere in a stage would condemn the camera's whole stage, which throws away
        the difference between a policy that recovers and one that never had a chance.
        """
        count = 0
        for held in self.stage_held:
            if held <= 0:
                break
            count += 1
        return count


@dataclass
class Curve:
    """Per-iteration training scalars, read back from the run's tensorboard file."""

    iters: list[int] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    """Wall-clock seconds since the first iteration."""

    def nearest(self, iteration: int) -> int | None:
        """Index of the logged iteration closest to ``iteration``.

        Checkpoints and log entries share an iteration counter, but only every hundredth
        iteration is saved, so an exact match cannot be assumed for a hand-picked checkpoint.
        """
        if not self.iters:
            return None
        return min(range(len(self.iters)), key=lambda i: abs(self.iters[i] - iteration))


@dataclass(frozen=True)
class Limits:
    """Full-scale values for the HUD meters, taken from the environment config."""

    tilt: float
    cart: float
    force: float


@dataclass(frozen=True)
class Trial:
    """The exam every checkpoint sits: recover from a lean, then take escalating kicks.

    Replaying every checkpoint under the same easy conditions is what makes a progress video
    go flat: nominal balance is solved within a few hundred iterations and looks identical
    from there on. An escalating trial keeps a question on screen that the current checkpoint
    cannot yet answer, so the growth reads as how far into the trial it gets.
    """

    lean_steps: int
    stage_steps: int
    kick_period: int
    kick_hold: int
    kicks: tuple[float, ...]

    @property
    def stages(self) -> int:
        return 1 + len(self.kicks)

    @property
    def total_steps(self) -> int:
        return self.lean_steps + self.stage_steps * len(self.kicks)

    def stage_at(self, step: int) -> int:
        if step < self.lean_steps:
            return 0
        return min(1 + (step - self.lean_steps) // self.stage_steps, len(self.kicks))


def resolve_runs() -> list[Path]:
    """Locate the training runs to replay, in the order they were trained.

    More than one is the normal case once a run has been resumed: the iteration counter carries
    across, so a resumed run is the same training curve continued in a new directory.
    """
    if args_cli.run:
        # Accept paths relative to either the shell's directory or the repo, since the README
        # quotes repo-relative log paths.
        runs = []
        for spec in args_cli.run:
            found = next((p for p in (Path(spec), REPO_ROOT / spec) if p.is_dir()), None)
            if found is None:
                raise SystemExit(f"Run directory not found: {spec}")
            runs.append(found)
        return runs
    if not LOG_ROOT.is_dir():
        raise SystemExit(f"No runs under {LOG_ROOT}. Train first.")
    runs = sorted((p for p in LOG_ROOT.iterdir() if p.is_dir() and any(p.glob("model_*.pt"))), key=lambda p: p.name)
    if not runs:
        raise SystemExit(f"No run under {LOG_ROOT} contains a checkpoint. Train first.")
    return [runs[-1]]


def resolve_segments(runs: list[Path]) -> list[Segment]:
    """Pick the checkpoints to replay, in training order.

    The sweep ends at whichever iteration ``model_best.pt`` is a copy of rather than at the last
    one, because PPO's final checkpoint is not reliably its best -- the reference run happens to
    end mid-collapse. ``--epilogue`` puts that collapse back on the end deliberately.
    """
    numbered: dict[int, Path] = {}
    for run in runs:
        # A later run resumed from an earlier one repeats the iteration it resumed at; the
        # continuation is the one worth replaying.
        for path in run.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)", path.stem)
            if match:
                numbered[int(match.group(1))] = path
    if not numbered:
        raise SystemExit(f"No model_<iteration>.pt checkpoints in {', '.join(str(r) for r in runs)}")

    best_iter = None
    for run in reversed(runs):
        pinned = run / "model_best.pt"
        if not pinned.is_file():
            continue
        # model_best.pt is a copy of one of the numbered checkpoints, and which one is only
        # recoverable from the bytes.
        digest = hashlib.sha1(pinned.read_bytes()).hexdigest()
        match_iter = next(
            (i for i, p in sorted(numbered.items()) if hashlib.sha1(p.read_bytes()).hexdigest() == digest), None
        )
        if match_iter is not None:
            best_iter = match_iter
            break

    last_iter = max(numbered)
    end = args_cli.to_iter if args_cli.to_iter is not None else (last_iter if best_iter is None else best_iter)
    start = args_cli.from_iter if args_cli.from_iter is not None else -1
    every = max(args_cli.every, 1)

    chosen = sorted(i for i in numbered if start <= i <= end and i % every == 0)
    if end in numbered and start <= end and end not in chosen:
        chosen.append(end)
    if not chosen:
        raise SystemExit("No checkpoint matched the --every / --from_iter / --to_iter filters.")

    segments = [Segment(path=numbered[i], iteration=i) for i in chosen]
    if args_cli.epilogue and last_iter > chosen[-1]:
        segments.append(Segment(path=numbered[last_iter], iteration=last_iter, label="late"))
    for seg in segments:
        if seg.iteration == best_iter:
            seg.label = "best"
    return segments


def load_curve(runs: list[Path]) -> Curve:
    """Read the reward curve out of the runs' tensorboard event files, end to end.

    Optional: without it the HUD just loses the curve strip and the elapsed-time readout.
    """
    curve = Curve()
    merged: dict[int, tuple[float, float]] = {}
    offset = 0.0
    for run in runs:
        if not any(run.glob("events.out.tfevents.*")):
            print(f"[record] no tensorboard file in {run}; its stretch of the curve will be missing")
            continue
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
            acc.Reload()
            if "Train/mean_reward" not in acc.Tags()["scalars"]:
                raise KeyError("Train/mean_reward")
            events = acc.Scalars("Train/mean_reward")
            # Wall-clock accumulates across runs; the iterations a resumed run repeats are taken
            # from the continuation, so the curve stays single-valued.
            for event in events:
                merged[event.step] = (event.value, offset + event.wall_time - events[0].wall_time)
            offset += events[-1].wall_time - events[0].wall_time
        except Exception as exc:  # noqa: BLE001 -- a missing curve must not cost a whole render
            print(f"[record] could not read the reward curve of {run.name} ({type(exc).__name__}: {exc})")
    for iteration in sorted(merged):
        value, elapsed = merged[iteration]
        curve.iters.append(iteration)
        curve.reward.append(value)
        curve.elapsed.append(elapsed)
    if curve.iters:
        print(f"[record] reward curve: {len(curve.iters)} iterations, {curve.elapsed[-1] / 60.0:.1f} min of training")
    return curve


def train_env_count(run: Path) -> int | None:
    """How many environments the run itself was trained with, for the closing card."""
    cfg = run / "params" / "env.yaml"
    if not cfg.is_file():
        return None
    match = re.search(r"num_envs:\s*(\d+)", cfg.read_text(encoding="utf-8", errors="ignore"))
    return int(match.group(1)) if match else None


RUNS = resolve_runs()
SEGMENTS = resolve_segments(RUNS)
CURVE = load_curve(RUNS)
TRAIN_ENVS = train_env_count(RUNS[-1])
OUT_PATH = Path(args_cli.out) if args_cli.out else RUNS[-1] / "videos" / "training_progress.mp4"
# Everything above is resolved before the simulator starts. Isaac Sim's shutdown path swallows
# exceptions and exits 0, so a bad path raised after this point would fail silently -- and only
# after a slow app launch.

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import triple_cartpole  # noqa: E402, F401  -- registers the gym task
from triple_cartpole import TASK_ID  # noqa: E402

TEXT = {
    "en": {
        "title": "TRIPLE INVERTED PENDULUM  \u00b7  PPO",
        "subtitle": "one horizontal force on the cart, four degrees of freedom to stabilise",
        "iteration": "ITERATION",
        "of": "of {total}",
        "trained": "{time} into training",
        "tilt": "max link tilt",
        "cart": "cart position",
        "force": "cart force",
        "upright": "never fell",
        "reward": "mean episode reward",
        "balancing": "BALANCING  {t:.1f}s",
        "fell": "FELL  x{n}",
        "best": "BEST CHECKPOINT",
        "late": "LATE COLLAPSE",
        "stage_lean": "RECOVER FROM {deg:.0f}° LEAN",
        "stage_kick": "KICK  {n:.0f} N",
        "trial": "trial",
        "card_title": "Learning to balance a triple inverted pendulum",
        "card_lines": [
            "One horizontal force on the cart has to stabilise four degrees of freedom: the cart,",
            "plus three free hinges stacked into a 10 m tower that falls whichever way you lean it.",
            "",
            "Every checkpoint below sits the same exam, released from the same starting states:",
            "recover from a {deg:.0f}° lean, then take kicks of {kicks} N on the top link.",
            "",
            "{n} checkpoints from iteration {first} to {last}, {sec:.0f}s each, at real-time speed.",
        ],
        "end_title": "From flailing to standing up to a shove",
        "end_lines": [
            "iteration {first}: cleared {first_reached} of {stages} stages, fell {first_falls}x on camera",
            "iteration {last}: cleared {last_reached} of {stages}, {last_upright} of {last_of} never fell",
            "",
            "{iters} iterations over {envs} parallel environments, {time} of wall-clock training.",
            "Mean episode reward went from {r0:.0f} to {r1:.0f}, and saturated long before the end.",
            "What kept improving after that is the part the reward curve cannot show: the exam.",
            "",
            "Note what finally stops it: {last_rail} of its last {last_falls} falls were the cart",
            "reaching the end of its rail, not the pendulum toppling. Recovery costs metres.",
        ],
    },
    "zh": {
        "title": "三级倒立摆  ·  PPO 强化学习",
        "subtitle": "小车上只有一个水平推力，却要同时稳住四个自由度",
        "iteration": "训练迭代",
        "of": "共 {total} 次",
        "trained": "已训练 {time}",
        "tilt": "最大杆倾角",
        "cart": "小车位置",
        "force": "推力",
        "upright": "全程未倒",
        "reward": "平均回合奖励",
        "balancing": "稳定平衡  {t:.1f}s",
        "fell": "已倒下 {n} 次",
        "best": "最佳策略",
        "late": "后期崩溃",
        "stage_lean": "从 {deg:.0f}° 倾斜中恢复",
        "stage_kick": "外力扰动  {n:.0f} N",
        "trial": "考核",
        "card_title": "强化学习是怎么学会稳住三级倒立摆的",
        "card_lines": [
            "小车上只有一个水平推力，要同时稳住四个自由度：",
            "小车本身，加上三个完全自由的铰链，一座 10 米高、朝哪边都会倒的塔。",
            "",
            "下面每个 checkpoint 都从完全相同的初始状态出发，接受同一场考核：",
            "先从 {deg:.0f}° 的倾斜中恢复，再依次承受 {kicks} N 的顶端冲击。",
            "",
            "从第 {first} 次迭代到第 {last} 次，共 {n} 个，每个 {sec:.0f} 秒，实时速度。",
        ],
        "end_title": "从手忙脚乱，到推不倒",
        "end_lines": [
            "第 {first} 次迭代：{stages} 关里过了 {first_reached} 关，镜头里倒下 {first_falls} 次",
            "第 {last} 次迭代：过了 {last_reached} 关，{last_of} 个摆里 {last_upright} 个全程未倒",
            "",
            "共 {iters} 次迭代、{envs} 个并行环境，实际训练用时 {time}。",
            "平均回合奖励从 {r0:.0f} 涨到 {r1:.0f}，而且很早就不再上涨了。",
            "此后仍在增长的那部分，正是奖励曲线看不见、只有这场考核才问得出来的能力。",
            "",
            "最后拦住它的是什么：最后 {last_falls} 次倒下里有 {last_rail} 次是小车走到了轨道尽头，",
            "而不是摆真的倒了。扶正一根 3 米长的摆，需要用米来计的行程。",
        ],
    },
}

FONT_CANDIDATES = {
    "en": [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "zh": [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ],
}
BOLD_CANDIDATES = {
    "en": ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"],
    "zh": ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf"],
}

INK = (236, 240, 246, 255)
DIM = (236, 240, 246, 155)
FAINT = (255, 255, 255, 85)
PANEL = (10, 13, 20, 155)
GOOD = (104, 214, 148, 255)
WARN = (240, 196, 92, 255)
BAD = (235, 106, 96, 255)
ACCENT = (120, 186, 255, 255)


def font_path(bold: bool) -> str:
    """Resolve a TrueType face for the chosen language, or explain why we cannot."""
    if args_cli.font:
        if not Path(args_cli.font).is_file():
            raise SystemExit(f"Font not found: {args_cli.font}")
        return args_cli.font
    for candidate in (BOLD_CANDIDATES if bold else FONT_CANDIDATES)[args_cli.lang]:
        if Path(candidate).is_file():
            return candidate
    # Falling back to PIL's bitmap font would silently drop every CJK glyph and break the
    # anchored layout, so ask for a face rather than drawing tofu.
    kind = "bold" if bold else "regular"
    raise SystemExit(f"No {kind} TrueType font found for --lang {args_cli.lang}. Pass --font <path to .ttf/.ttc>.")


class Hud:
    """Draws the overlay: which iteration is on screen, and what it is doing."""

    def __init__(self, size: tuple[int, int], curve: Curve, total_iters: int, limits: Limits, trial: Trial):
        self.w, self.h = size
        self.curve = curve
        self.total_iters = total_iters
        self.limits = limits
        self.trial = trial
        # The joints lean by the same angle each, and they stack: what the eye sees at the top
        # of the tower is the sum.
        self.lean_top_deg = args_cli.lean_deg * 3
        self.txt = TEXT[args_cli.lang]
        self.s = self.h / 720.0
        s = self.s

        regular, bold = font_path(bold=False), font_path(bold=True)
        self.f_title = ImageFont.truetype(bold, int(22 * s))
        self.f_sub = ImageFont.truetype(regular, int(14 * s))
        self.f_huge = ImageFont.truetype(bold, int(50 * s))
        self.f_label = ImageFont.truetype(regular, int(14 * s))
        self.f_value = ImageFont.truetype(bold, int(19 * s))
        self.f_badge = ImageFont.truetype(bold, int(17 * s))
        self.f_card = ImageFont.truetype(regular, int(20 * s))
        self.f_card_title = ImageFont.truetype(bold, int(33 * s))

        self.pad = int(26 * s)
        self.strip_h = int(94 * s)
        self.strip_top = self.h - self.strip_h
        self.reward_lo = self.reward_hi = 0.0
        self.baseline = self.h - int(26 * s)
        """y the curve's area fill drops to."""
        # One point per pixel column, so the polyline stays cheap however long the run was.
        self.curve_pts = self._bucket_curve() if curve.iters else []

    def _bucket_curve(self) -> list[tuple[float, float]]:
        s = self.s
        x0, x1 = self.pad, self.w - self.pad
        y_top = self.strip_top + int(34 * s)
        y_bottom = self.h - int(12 * s)
        self.baseline = y_bottom

        lo, hi = min(self.curve.reward), max(self.curve.reward)
        span = max(hi - lo, 1e-6)
        self.reward_lo, self.reward_hi = lo - 0.06 * span, hi + 0.06 * span

        columns = max(int(x1 - x0), 2)
        totals = [0.0] * columns
        counts = [0] * columns
        last = max(self.total_iters, 1)
        for iteration, reward in zip(self.curve.iters, self.curve.reward):
            col = min(int(columns * iteration / last), columns - 1)
            totals[col] += reward
            counts[col] += 1

        points, previous = [], self.curve.reward[0]
        for col in range(columns):
            # Columns with no iteration in them hold the previous value, which keeps the line
            # continuous when the run is shorter than the frame is wide.
            value = totals[col] / counts[col] if counts[col] else previous
            previous = value
            frac = (value - self.reward_lo) / (self.reward_hi - self.reward_lo)
            points.append((x0 + col, y_bottom - frac * (y_bottom - y_top)))
        return points

    def _panel(self, d: ImageDraw.ImageDraw, box: tuple[float, float, float, float], radius: float = 10.0) -> None:
        d.rounded_rectangle(box, radius=radius * self.s, fill=PANEL)

    def _bar(self, d: ImageDraw.ImageDraw, x, y, w, h, frac: float, colour, centered: bool = False) -> None:
        """A thin meter. ``centered`` grows a signed value out from the middle."""
        d.rounded_rectangle((x, y, x + w, y + h), radius=h / 2, fill=(255, 255, 255, 38))
        frac = max(-1.0, min(1.0, frac))
        if centered:
            mid = x + w / 2
            span = abs(frac) * w / 2
            if span > 1:
                left, right = (mid, mid + span) if frac >= 0 else (mid - span, mid)
                d.rounded_rectangle((left, y, right, y + h), radius=h / 2, fill=colour)
            d.line((mid, y - 1, mid, y + h + 1), fill=(255, 255, 255, 120), width=max(1, int(self.s)))
        elif frac > 0:
            d.rounded_rectangle((x, y, x + max(frac * w, h), y + h), radius=h / 2, fill=colour)

    def _meter(self, d, x_right, y, label, value, frac, colour, centered=False) -> float:
        """One label / value / bar row, right-aligned at ``x_right``. Returns the next row's y."""
        s = self.s
        bar_w = int(166 * s)
        d.text((x_right - bar_w, y), label, font=self.f_label, fill=DIM)
        d.text((x_right, y - 3 * s), value, font=self.f_value, fill=INK, anchor="ra")
        self._bar(d, x_right - bar_w, y + int(22 * s), bar_w, int(6 * s), frac, colour, centered)
        return y + int(43 * s)

    def _badge(self, d: ImageDraw.ImageDraw, x: float, y: float, text: str, colour) -> float:
        """A pill of coloured text. Returns the x of its right edge, to butt the next one up."""
        s = self.s
        w = d.textlength(text, font=self.f_badge) + 26 * s
        d.rounded_rectangle((x, y, x + w, y + 34 * s), radius=8 * s, fill=(*colour[:3], 48), outline=colour, width=1)
        d.text((x + 13 * s, y + 17 * s), text, font=self.f_badge, fill=colour, anchor="lm")
        return x + w

    def _curve(self, d: ImageDraw.ImageDraw, iteration: int) -> None:
        s = self.s
        self._panel(d, (0, self.strip_top, self.w, self.h), radius=0)
        caption = self.txt["reward"]
        if self.curve.reward:
            caption += f"    {min(self.curve.reward):.0f} – {max(self.curve.reward):.0f}"
        d.text((self.pad, self.strip_top + int(9 * s)), caption, font=self.f_label, fill=DIM)
        if not self.curve_pts:
            return

        # Filled, because a 3000-iteration curve squeezed into one strip reads as a shape long
        # before it reads as a line.
        cut = min(int(len(self.curve_pts) * iteration / max(self.total_iters, 1)), len(self.curve_pts) - 1)
        d.polygon(
            [(self.curve_pts[0][0], self.baseline), *self.curve_pts, (self.curve_pts[-1][0], self.baseline)],
            fill=(255, 255, 255, 26),
        )
        if cut > 1:
            done = self.curve_pts[: cut + 1]
            d.polygon([(done[0][0], self.baseline), *done, (done[-1][0], self.baseline)], fill=(*ACCENT[:3], 60))
            d.line(done, fill=ACCENT, width=max(2, int(2.5 * s)))
        d.line(self.curve_pts, fill=(255, 255, 255, 70), width=max(1, int(2 * s)))

        px, py = self.curve_pts[cut]
        d.line((px, self.strip_top + int(32 * s), px, self.baseline), fill=FAINT)
        r = 4.5 * s
        d.ellipse((px - r, py - r, px + r, py + r), fill=(255, 255, 255, 255), outline=ACCENT, width=max(1, int(s)))

        # No axis ticks: the iteration span is already spelled out top left, and the strip is
        # too shallow to carry labels over the area fill legibly.
        reward = f"{self.curve.reward[self.curve.nearest(iteration)]:.0f}"
        d.text((self.w - self.pad, self.strip_top + int(9 * s)), reward, font=self.f_label, fill=INK, anchor="ra")

    def _trial(self, d: ImageDraw.ImageDraw, seg: Segment, stage: int, kick: float) -> None:
        """Which question the checkpoint is being asked, and which ones it has already answered."""
        s = self.s
        x_right, y = self.w - self.pad, self.strip_top - 58 * s
        # The block sits over open sky, which is the brightest part of the frame.
        self._panel(d, (x_right - 218 * s, y - 10 * s, x_right + 14 * s, y + 60 * s))
        label = (
            self.txt["stage_lean"].format(deg=self.lean_top_deg)
            if stage == 0
            else self.txt["stage_kick"].format(n=self.trial.kicks[stage - 1])
        )
        d.text((x_right, y), self.txt["trial"], font=self.f_label, fill=DIM, anchor="ra")
        d.text((x_right, y + 17 * s), label, font=self.f_badge, fill=ACCENT if kick else INK, anchor="ra")

        pip_w, pip_h, gap = 24 * s, 6 * s, 5 * s
        total = self.trial.stages
        x = x_right - total * pip_w - (total - 1) * gap
        for i in range(total):
            if i < len(seg.stage_held):
                # How much of the population was still standing when the stage ended, so a
                # stage the policy half-survived reads as half-survived.
                share = seg.stage_held[i] / max(seg.of, 1)
                fill = GOOD if share >= 0.5 else WARN if share > 0.0 else BAD
            elif i == stage:
                fill = (*ACCENT[:3], 170)
            else:
                fill = (255, 255, 255, 40)
            d.rounded_rectangle((x, y + 44 * s, x + pip_w, y + 44 * s + pip_h), radius=pip_h / 2, fill=fill)
            x += pip_w + gap

    def draw(
        self, frame: np.ndarray, seg: Segment, state: tuple[float, float, float, float], stage: int, kick: float
    ) -> np.ndarray:
        """Composite the HUD onto one rendered frame. ``state`` is (tilt, cart, force, held_s)."""
        tilt, cart, force, held_s = state
        base = Image.fromarray(frame).convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        s, pad = self.s, self.pad

        # -- top left: what this is, and where we are in training
        self._panel(d, (pad - 14 * s, pad - 12 * s, pad + 436 * s, pad + 148 * s))
        d.text((pad, pad), self.txt["title"], font=self.f_title, fill=INK)
        d.text((pad, pad + 29 * s), self.txt["subtitle"], font=self.f_sub, fill=DIM)
        d.text((pad, pad + 58 * s), self.txt["iteration"], font=self.f_label, fill=DIM)
        d.text((pad, pad + 74 * s), str(seg.iteration), font=self.f_huge, fill=INK)
        x_after = pad + d.textlength(str(seg.iteration), font=self.f_huge) + 12 * s
        d.text((x_after, pad + 102 * s), self.txt["of"].format(total=self.total_iters), font=self.f_label, fill=DIM)
        nearest = self.curve.nearest(seg.iteration)
        if nearest is not None:
            mins, secs = divmod(int(self.curve.elapsed[nearest]), 60)
            trained = self.txt["trained"].format(time=f"{mins:d}:{secs:02d}")
            d.text((x_after, pad + 120 * s), trained, font=self.f_label, fill=DIM)

        # -- top right: the live state of the pendulum on camera
        x_right, y = self.w - pad, pad
        self._panel(d, (x_right - 194 * s, pad - 12 * s, x_right + 14 * s, pad + 158 * s))
        tilt_frac = abs(tilt) / max(self.limits.tilt, 1e-6)
        y = self._meter(
            d,
            x_right,
            y,
            self.txt["tilt"],
            f"{np.degrees(tilt):+.1f}\u00b0",
            tilt_frac,
            BAD if tilt_frac > 0.75 else WARN if tilt_frac > 0.4 else GOOD,
        )
        y = self._meter(d, x_right, y, self.txt["cart"], f"{cart:+.2f} m", cart / self.limits.cart, ACCENT, True)
        y = self._meter(d, x_right, y, self.txt["force"], f"{force:+.1f} N", force / self.limits.force, ACCENT, True)
        summary = f"{self.txt['upright']}   {seg.held}/{seg.of}"
        d.text((x_right, y - 8 * s), summary, font=self.f_label, fill=DIM, anchor="ra")

        # -- badge: is the pendulum on camera up, and has it fallen during this segment
        badge = self.txt["fell"].format(n=seg.falls) if seg.falls else self.txt["balancing"].format(t=held_s)
        row = self.strip_top - 58 * s
        right = self._badge(d, pad - 14 * s, row, badge, BAD if seg.falls else GOOD)
        if seg.label:
            self._badge(d, right + 10 * s, row, self.txt[seg.label], ACCENT)

        self._trial(d, seg, stage, kick)
        self._curve(d, seg.iteration)
        return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))

    def card(self, title: str, lines: list[str]) -> np.ndarray:
        """A full-frame text card, for the bookends."""
        s = self.s
        img = Image.new("RGB", (self.w, self.h), (9, 11, 16))
        d = ImageDraw.Draw(img)
        y = self.h / 2 - (len(lines) + 3) * 18 * s
        d.text((self.w / 2, y), title, font=self.f_card_title, fill=INK[:3], anchor="ma")
        y += 54 * s
        d.line((self.w / 2 - 110 * s, y, self.w / 2 + 110 * s, y), fill=ACCENT[:3], width=max(1, int(2 * s)))
        y += 34 * s
        for line in lines:
            d.text((self.w / 2, y), line, font=self.f_card, fill=DIM[:3], anchor="ma")
            y += 33 * s
        return np.asarray(img)


def write_card(writer, hud: Hud, title: str, lines: list[str], seconds: float, fade: float = 0.5) -> None:
    """Hold a card, fading in and out of black so it does not cut straight into the action."""
    frame = hud.card(title, lines).astype(np.float32)
    n_fade = max(int(fade * args_cli.fps), 1)
    n_hold = max(int(seconds * args_cli.fps) - 2 * n_fade, 1)
    for i in range(n_fade):
        writer.append_data((frame * (i + 1) / n_fade).astype(np.uint8))
    held = frame.astype(np.uint8)
    for _ in range(n_hold):
        writer.append_data(held)
    for i in range(n_fade):
        writer.append_data((frame * (1.0 - (i + 1) / n_fade)).astype(np.uint8))


def spawn_limit_markers(origin: np.ndarray, limit: float) -> None:
    """Posts at the cart's travel limit. Visual only -- no collider, no rigid body.

    The ground is a plain 400 m slab, which leaves the eye nothing to judge cart travel
    against; two posts turn "the cart is moving" into "the cart is nearly out of room".
    """
    post = sim_utils.CuboidCfg(
        size=(0.10, 0.10, 1.1),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.33, 0.26), roughness=0.6),
    )
    for i, x in enumerate((-limit, limit)):
        post.func(f"/World/Markers/limit_{i}", post, translation=(float(origin[0]) + x, float(origin[1]), 0.5))


def trial_seconds() -> float:
    return args_cli.lean_s + args_cli.stage_s * len(args_cli.kicks)


def build_env(width: int, height: int):
    """Create the task, wrapped for rsl-rl, with the camera framed on environment 0."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry

    env_cfg = triple_cartpole.TripleCartpoleEnvCfg()
    env_cfg.scene.num_envs = max(args_cli.num_envs, 1)
    # The extra environments are here for the statistic, not for the shot. At the training
    # spacing they crowd into frame around the one on camera; pushed apart they fall outside
    # it, which costs nothing since inter-environment collisions are filtered anyway.
    env_cfg.scene.env_spacing = max(args_cli.env_spacing, 1.0)
    env_cfg.viewer.resolution = (width, height)
    # Swung a little off the motion axis, unlike the dead-on demo view: the distant
    # environments then sit beside the pendulum on camera instead of hovering behind it, and
    # the cart reads as travelling through a scene rather than sliding along a flat backdrop.
    env_cfg.viewer.eye = (6.0, -26.0, 8.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 5.2)
    if args_cli.eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.eye)
    if args_cli.lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.lookat)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    # A time-limit reset in the middle of a segment would read as an unexplained jump, so push
    # the horizon past the segment: then every reset on screen is a real failure.
    env_cfg.episode_length_s = trial_seconds() + 2.0
    # Every clip opens leaning, by the same amount for every checkpoint, so the first question
    # the trial asks is one an early policy cannot answer.
    env_cfg.init_pole_angle_bias = float(np.radians(args_cli.lean_deg))
    env_cfg.max_tilt = args_cli.trial_max_tilt

    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    limits = Limits(tilt=env_cfg.max_tilt, cart=env_cfg.max_cart_pos, force=env_cfg.action_scale)
    return env, agent_cfg, limits


def record_segment(env, runner, writer, hud: Hud, seg: Segment, stride: int, trial: Trial, kick_ids: list[int]) -> None:
    """Put one checkpoint through the trial, collecting the numbers the HUD shows."""
    unwrapped = env.unwrapped
    runner.load(str(seg.path))
    policy = runner.get_inference_policy(device=unwrapped.device)
    policy_nn = runner.alg.policy

    seg.of = unwrapped.num_envs
    seg.held = unwrapped.num_envs
    fell_once = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
    force_sum, force_steps, held_s = 0.0, 0, 0.0

    wrench = unwrapped.robot.permanent_wrench_composer
    kick = torch.zeros(unwrapped.num_envs, max(len(kick_ids), 1), 3, device=unwrapped.device)
    kick_until, kick_sign, stage, last_cart = -1, -1.0, 0, 0.0

    with torch.inference_mode():
        # The reset belongs inside inference mode. Stepping the previous segment turned the
        # articulation's buffers into inference tensors, and a reset writes to them -- from
        # normal mode torch refuses that with "inplace update to inference tensor".
        # Seeding it first is what makes this a trial rather than a demo: every checkpoint is
        # released from the identical set of states, so the clips are comparable.
        torch.manual_seed(args_cli.seed)
        obs, _ = env.reset()
        policy_nn.reset()
        step = 0
        for _ in range(max(round(trial.total_steps / stride), 1)):
            for _ in range(stride):
                now = trial.stage_at(step)
                if now != stage:
                    seg.stage_held.append(int((~fell_once).sum()))
                    stage = now
                if kick_ids and stage >= 1 and (step - trial.lean_steps) % trial.kick_period == 0:
                    kick_sign = -kick_sign
                    kick[..., 0] = kick_sign * trial.kicks[stage - 1]
                    kick_until = step + trial.kick_hold
                if step < kick_until:
                    # is_global needs the composer's cached link poses refreshed, which reset()
                    # is what forces; see the same pattern in demo.py.
                    wrench.reset()
                    wrench.set_forces_and_torques(forces=kick, body_ids=kick_ids, is_global=True)
                elif step == kick_until:
                    wrench.reset()

                obs, _, dones, _ = env.step(policy(obs))
                policy_nn.reset(dones)
                terminated = unwrapped.reset_terminated.reshape(-1).bool()
                fell_once |= terminated
                if bool(terminated[0]):
                    seg.falls += 1
                    # env.step() has already reset the environment by the time this runs, so
                    # the cart is back at the origin and cannot say what ended the episode.
                    # The reading from the step before the fall still can.
                    if last_cart > 0.95 * unwrapped.cfg.max_cart_pos:
                        seg.rail_falls += 1
                    held_s = 0.0
                else:
                    held_s += unwrapped.step_dt
                force_sum += float(unwrapped._effort.abs().mean())
                force_steps += 1
                last_cart = abs(float(unwrapped._cart_pos[0]))
                seg.peak_tilt = max(seg.peak_tilt, float(unwrapped._link_angle[0].abs().max()))
                seg.peak_cart = max(seg.peak_cart, last_cart)
                step += 1

            seg.held = int((~fell_once).sum())
            tilt = unwrapped._link_angle[0]
            state = (
                float(tilt[int(tilt.abs().argmax())]),
                float(unwrapped._cart_pos[0]),
                float(unwrapped._effort[0, 0]),
                held_s,
            )
            live_kick = float(kick[0, 0, 0]) if step <= kick_until else 0.0
            writer.append_data(hud.draw(unwrapped.render(), seg, state, stage, live_kick))

        seg.stage_held.append(int((~fell_once).sum()))
        wrench.reset()

    seg.mean_force = force_sum / max(force_steps, 1)
    print(
        f"[record] iter {seg.iteration:>5}: cleared {seg.reached}/{trial.stages} stages,"
        f" {seg.held}/{seg.of} never fell, {seg.falls} fall(s) on camera"
        f" ({seg.rail_falls} out of rail), peak tilt {np.degrees(seg.peak_tilt):5.1f} deg,"
        f" peak cart {seg.peak_cart:4.2f} m, mean force {seg.mean_force:5.2f} N,"
        f" per stage {'/'.join(str(h) for h in seg.stage_held)}"
    )


def main() -> None:
    width = max(args_cli.width // 16 * 16, 256)
    height = max(args_cli.height // 16 * 16, 144)
    print(
        f"[record] run{'s' if len(RUNS) > 1 else ''} {', '.join(r.name for r in RUNS)},"
        f" {len(SEGMENTS)} checkpoints: {[s.iteration for s in SEGMENTS]}"
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    from rsl_rl.runners import OnPolicyRunner

    env, agent_cfg, limits = build_env(width, height)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    unwrapped = env.unwrapped

    if args_cli.markers:
        try:
            spawn_limit_markers(unwrapped.scene.env_origins[0].cpu().numpy(), limits.cart)
        except Exception as exc:  # noqa: BLE001 -- decoration, not a reason to lose the render
            print(f"[record] could not spawn the limit markers ({type(exc).__name__}: {exc})")

    control_hz = 1.0 / unwrapped.step_dt
    stride = max(int(round(control_hz * args_cli.speed / args_cli.fps)), 1)
    total_iters = max((CURVE.iters[-1] + 1) if CURVE.iters else 0, SEGMENTS[-1].iteration, 1)

    trial = Trial(
        lean_steps=max(1, round(args_cli.lean_s * control_hz)),
        stage_steps=max(1, round(args_cli.stage_s * control_hz)),
        kick_period=max(1, round(args_cli.kick_every * control_hz)),
        kick_hold=max(1, round(0.001 * args_cli.kick_ms * control_hz)),
        kicks=tuple(args_cli.kicks),
    )
    kick_ids: list[int] = []
    if trial.kicks:
        kick_ids, kick_names = unwrapped.robot.find_bodies(args_cli.kick_body)
        if not kick_ids:
            raise SystemExit(f"No link matches '{args_cli.kick_body}'. Links: {unwrapped.robot.body_names}")
        print(
            f"[record] trial: {args_cli.lean_s:g}s recovering from {3 * args_cli.lean_deg:g} deg,"
            f" then {' / '.join(f'{k:g}' for k in trial.kicks)} N on {', '.join(kick_names)}"
            f" every {args_cli.kick_every:g}s, {args_cli.stage_s:g}s per stage"
        )

    hud = Hud((width, height), CURVE, total_iters, limits, trial)
    print(
        f"[record] {width}x{height} @ {args_cli.fps} fps, {stride} control step(s) per frame"
        f" ({stride * args_cli.fps / control_hz:.2f}x real time) -> {OUT_PATH}"
    )

    writer = imageio.get_writer(str(OUT_PATH), fps=args_cli.fps, codec="libx264", quality=7, pixelformat="yuv420p")
    try:
        # The replicator annotator hands back an empty buffer until the renderer has warmed up,
        # which would otherwise open the video on a handful of black frames.
        with torch.inference_mode():
            env.reset()
        for _ in range(12):
            unwrapped.render()

        if args_cli.cards:
            fields = {
                "n": len(SEGMENTS),
                "first": SEGMENTS[0].iteration,
                "last": SEGMENTS[-1].iteration,
                "sec": trial_seconds(),
                "deg": hud.lean_top_deg,
                "kicks": " / ".join(f"{k:g}" for k in trial.kicks),
            }
            lines = [line.format(**fields) for line in hud.txt["card_lines"]]
            write_card(writer, hud, hud.txt["card_title"], lines, seconds=7.0)

        for seg in SEGMENTS:
            record_segment(env, runner, writer, hud, seg, stride, trial, kick_ids)

        if args_cli.cards:
            first, last = SEGMENTS[0], SEGMENTS[-1]
            mins, secs = divmod(int(CURVE.elapsed[-1]) if CURVE.elapsed else 0, 60)
            fields = {
                "first": first.iteration,
                "last": last.iteration,
                "first_falls": first.falls,
                "first_force": first.mean_force,
                "first_reached": first.reached,
                "last_reached": last.reached,
                "stages": trial.stages,
                "last_upright": last.held,
                "last_of": last.of,
                "last_force": last.mean_force,
                "last_falls": last.falls,
                "last_rail": last.rail_falls,
                "iters": total_iters,
                "envs": TRAIN_ENVS or unwrapped.num_envs,
                "time": f"{mins:d}:{secs:02d}",
                # The reward the video ends on, which is not the run's last one: the sweep
                # stops at the best checkpoint.
                "r0": CURVE.reward[0] if CURVE.reward else float("nan"),
                "r1": CURVE.reward[CURVE.nearest(last.iteration)] if CURVE.reward else float("nan"),
            }
            lines = [line.format(**fields) for line in hud.txt["end_lines"]]
            write_card(writer, hud, hud.txt["end_title"].format(**fields), lines, seconds=8.0)
    finally:
        writer.close()
        env.close()

    size_mb = OUT_PATH.stat().st_size / 1e6 if OUT_PATH.is_file() else 0.0
    print(f"\n[record] wrote {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Isaac Sim's close() tears the process down before the interpreter gets to print an
        # unhandled exception, which turns any failure mid-sweep into a silent exit 0.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
