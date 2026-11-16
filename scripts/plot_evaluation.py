# SPDX-License-Identifier: MIT
"""Plot the recorded physical trajectory, including both uninterrupted holds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trace", type=Path, default=ROOT / "reports" / "demo_trace.csv")
parser.add_argument("--report", type=Path, default=ROOT / "reports" / "demo_recovery.json")
parser.add_argument("--output", type=Path, default=ROOT / "reports" / "swingup_recovery.png")
args = parser.parse_args()


def main() -> None:
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    report = json.loads(args.report.read_text(encoding="utf-8"))
    with args.trace.open(encoding="utf-8") as source:
        data = list(csv.DictReader(source))
    column = lambda name: np.array([float(row[name]) for row in data])
    time = column("time_s")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    fig.suptitle("同一策略：下垂起摆、持续直立、受冲击后恢复", fontsize=18, y=0.985)
    colors = ["#007a99", "#80a627", "#d97333"]
    for index, color in enumerate(colors, 1):
        axes[0].plot(time, column(f"pole{index}_rad"), label=f"摆杆 {index}", color=color, linewidth=1.4)
    tilt_limit = report["tilt_limit_rad"]
    axes[0].axhspan(-tilt_limit, tilt_limit, color="#27a778", alpha=0.15, label="直立容差 ±0.35 rad")
    axes[0].set_ylabel("绝对倾角 (rad)")
    axes[0].set_ylim(-3.4, 3.4)
    axes[0].legend(loc="upper right", ncol=4, fontsize=9)
    axes[1].plot(time, column("cart_m"), color="#34465e")
    axes[1].set_ylabel("小车位置 (m)")
    axes[2].plot(time, column("cart_force_N"), color="#44546e", label="策略输出的小车驱动力")
    axes[2].plot(time, column("recovery_force_N"), color="#ce3232", label="顶杆外部冲击")
    axes[2].set_ylabel("力 (N)")
    axes[2].set_xlabel("连续仿真时间 (s)")
    axes[2].legend(loc="upper right", fontsize=9)

    trials = report["trials"]
    first = trials["first_hold_completed_s"][0]
    kick = trials["kick_started_s"][0]
    recovered_after = trials["post_kick_hold_completed_after_impulse_s"][0]
    completed = kick + report["recovery_duration_ms"] / 1000 + recovered_after
    for axis in axes:
        for end in (first, completed):
            axis.axvspan(end - report["hold_s"], end, color="#008866", alpha=0.10)
        axis.axvline(kick, color="#ce3232", linewidth=1, linestyle="--")
        axis.grid(alpha=0.2)
        axis.set_xlim(0, report["horizon_s"])
    axes[0].text(first - 5, 2.8, "首次连续 10 秒", ha="center", fontsize=10, color="#00734e")
    axes[0].text(completed - 5, 2.8, "恢复后连续 10 秒", ha="center", fontsize=10, color="#00734e")
    fig.text(0.5, 0.01, "60 Hz 控制；仅小车受控；冲击 12 N × 200 ms；全程无重置、无状态写回、无控制器切换", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
