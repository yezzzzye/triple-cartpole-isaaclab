# 小车三级倒立摆

Isaac Lab 的 Direct RL 任务：一个水平移动小车连接三根串联摆杆。唯一控制输入为小车水平驱动力（±50 N），三个转动关节均无驱动。
同一个 PPO 策略完成下垂 / 大角度初态起摆、连续直立至少 10 秒及扰动恢复。正式评测不切换控制器、不改写运行中的机器人状态、不中途重置。

[English](README.md) · [45 秒连续实录](reports/videos/swingup_down_exact_seed2-step-0.mp4) · [评测协议与结果](reports/README.md)

## 快速开始

在 PowerShell 中进入项目目录后执行。课程资料包的项目位于 `code` 子目录；直接从 GitHub 克隆时，克隆目录本身就是项目根目录。

```powershell
.\demo.ps1                       # 固定策略：精确下垂起步，每 25 秒受一次冲击
.\demo.ps1 --initial_pose random # 大角度随机起步
.\demo.ps1 --disturb 0           # 关闭演示中的外部冲击
.\evaluate.ps1                   # 三组严格连续评测，结果写入 reports
```

演示始终连续运行；关闭窗口结束，也可用 `--steps N` 限定步数。默认读取 `checkpoints/swingup_policy.pt`，可用 `--checkpoint PATH` 指定其他权重。

## 环境

| 组件 | 已验证版本 |
| --- | --- |
| Isaac Sim | 5.1.0 standalone，Python 3.11.13 |
| Isaac Lab | 2.3.2 源码 |
| PyTorch / rsl-rl-lib | 2.7.0+cu128 / 3.0.1 |
| NumPy / tensordict / h5py | 1.26.0 / 0.11.0 / 3.11.0 |

`run.ps1` 配置项目及 Isaac Lab 的导入路径，调用 Isaac Sim 自带 Python。本机默认安装路径已写入启动器；其他机器可设置：

```powershell
$env:ISAACSIM_PATH = 'D:\path\to\isaac-sim'
$env:ISAACLAB_PATH = 'D:\path\to\IsaacLab'
```

## 训练与检查

```powershell
# 默认任务为 swingup；PPO 从头训练
.\run.ps1 scripts/train.py --task swingup --headless --num_envs 2048 --max_iterations 4000 --seed 2

# 优先续训最新训练权重；日志中没有权重时使用交付策略
.\run.ps1 scripts/train.py --headless --resume
.\run.ps1 scripts/train.py --headless --checkpoint checkpoints/swingup_policy.pt

# 模型物理检查；观察学习过程
.\run.ps1 scripts/check_env.py --task swingup --headless
tensorboard --logdir logs/rsl_rl/triple_cartpole_swingup
```

训练目录保存权重、配置和 TensorBoard 数据。最终迭代不一定最佳，应使用 `scripts/eval_swingup.py --checkpoint PATH` 的连续评测选择策略。

## 模型与验收

原始 USD 的几何和质量保持不变：小车 1 kg，各杆 0.5 kg，相邻铰点间距 3 m。
起摆资产使用 ±12 m 移动关节行程，地面下移至 z = −10 m，确保三杆自然下垂。评测小车边界为 ±11.8 m。
11 维观测为小车位置、速度，三杆绝对角度的正弦、余弦及角速度；绝对角度由相对关节角累加得到。

严格评测要求：三杆均处于直立 ±0.35 rad，持续 10 秒；随后顶杆受到世界 X 方向 12 N、200 ms 冲击，再保持 10 秒。使用同一个交付策略的结果：

| 初态 | 完整流程成功 |
| --- | ---: |
| 精确静止下垂 | 128/128 |
| 下垂附近随机 | 128/128 |
| 三杆大角度随机 | 226/256（88.3%） |

随机初态仍可能失败，主要是起摆时轨道越界。详细采样、时限、逐次结果与限制见[评测报告](reports/README.md)。

## 模块与文件

| 路径 | 职责 |
| --- | --- |
| `triple_cartpole/env_cfg.py` | 基础稳定 / 起摆配置、评测初态分布 |
| `triple_cartpole/env.py` | 仿真动作、观测、奖励、终止和训练重置 |
| `triple_cartpole/robot.py` | 资产与关节、执行器配置 |
| `triple_cartpole/agents/` | PPO 超参数 |
| `scripts/_common.py` | 路径、权重选择和策略运行器 |
| `scripts/train.py` / `demo.py` | 训练 / 连续实时回放 |
| `scripts/eval_swingup.py` / `eval.py` | 起摆验收 / 基础稳定评测 |
| `scripts/check_env.py` / `prepare_asset.py` | 物理检查 / 资产转换 |
| `scripts/plot_evaluation.py` | 评测轨迹绘图 |
| `assets/` | 原始模型和两种任务资产 |
| `checkpoints/` | 交付权重与训练配置、来源记录 |
| `reports/` | 正式评测 JSON、轨迹 CSV、图和完整视频 |
| `logs/` | 本地训练历史；不进入版本控制 |

原始 `assets/triple_cartpole.usd` 为场景文件。转换脚本移除场景对象、重新设置 articulation root 与固定底座、修正关节引用，仅对小车关节添加力驱动。重新生成起摆资产：

```powershell
.\run.ps1 scripts/prepare_asset.py --output assets/triple_cartpole_swingup.usd --slider-limit-m 12
```

基础稳定任务保留 ±3 m 资产及独立策略，可用 `.\demo.ps1 --task balance`、`scripts/train.py --task balance` 和 `scripts/eval.py` 运行。

## 许可

MIT，见 [LICENSE](LICENSE)。
