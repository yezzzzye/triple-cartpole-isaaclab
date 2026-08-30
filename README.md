# Triple Cartpole - Isaac Lab

基于 Isaac Lab Direct Workflow 与 rsl-rl PPO 实现的小车三级倒立摆强化学习项目。

系统包含一个水平移动的小车和三根串联摆杆，共 4 个自由度，但只有小车移动关节能够直接施加驱动力。
任务目标是在有限轨道内稳定三根摆杆，并评估策略对初始偏差和外部扰动的恢复能力。

```text
                 pole3
                   |
                 pole2        4 DOF / 1 actuator
                   |          - slider_to_cart: actuated prismatic joint
                 pole1        - three passive revolute joints
              +----+----+
           ==== cart ======    horizontal force only
```

## 项目特点

- Isaac Lab `DirectRLEnv` 连续控制任务
- 4096 个并行环境的 GPU 采样配置
- 11 维观测、1 维连续动作
- PPO 训练、checkpoint 恢复与批量评估
- 随机外力训练与交互式扰动演示
- USD 资产转换和物理集成检查
- 单一 PowerShell 入口 `run.ps1`

## 系统模型

| 刚体 | 质量 | 说明 |
| --- | ---: | --- |
| `rail` | 1.0 kg | 固定基座，承载移动关节 |
| `cart` | 1.0 kg | 沿 X 轴移动，关节范围约 +/-3.0 m |
| `pole1` | 0.5 kg | 第一级被动摆杆 |
| `pole2` | 0.5 kg | 第二级被动摆杆 |
| `pole3` | 0.5 kg | 第三级被动摆杆 |

关节结构：

| 关节 | 类型 | 驱动 |
| --- | --- | --- |
| `slider_to_cart` | Prismatic | 水平力驱动 |
| `cart_to_pole1` | Revolute | 被动 |
| `pole1_to_pole2` | Revolute | 被动 |
| `pole2_to_pole3` | Revolute | 被动 |

原始几何和质量数据位于 `assets/triple_cartpole.usd`。`scripts/prepare_asset.py` 将原始场景转换为
Isaac Lab 可加载的固定基座 articulation，并生成 `assets/triple_cartpole_physics.usd`。

## 环境要求

| 组件 | 版本 |
| --- | --- |
| Isaac Sim | 5.1.0 standalone |
| Isaac Lab | 2.3.2 source checkout |
| Python | 3.11（Isaac Sim 自带） |
| PyTorch | 2.7.0 + CUDA 12.8 |
| rsl-rl-lib | 3.0.1 |
| 操作系统 | Windows / PowerShell |

## 配置

```powershell
git clone https://github.com/yezzzzye/triple-cartpole-isaaclab.git
cd triple-cartpole-isaaclab

$env:ISAACSIM_PATH = 'D:\path\to\isaac-sim-standalone-5.1.0-windows-x86_64'
$env:ISAACLAB_PATH = 'D:\path\to\IsaacLab'
```

项目不需要安装到 Isaac Sim Python 环境。`run.ps1` 会把仓库和 Isaac Lab 源码目录加入 `PYTHONPATH`，
再调用 Isaac Sim 自带的 `python.bat`。

## 统一入口

```powershell
.\run.ps1 help
```

| 命令 | 功能 |
| --- | --- |
| `.\run.ps1` | 使用附带 checkpoint 运行策略演示 |
| `.\run.ps1 check` | 检查自由度、固定基座、重力和驱动 |
| `.\run.ps1 physics` | 运行无控制自由摆动 |
| `.\run.ps1 policy` | 运行策略恢复演示 |
| `.\run.ps1 disturb` | 运行自动和手动外力扰动演示 |
| `.\run.ps1 eval` | 进行 30 s 无扰动批量评估 |
| `.\run.ps1 robust` | 进行 30 s、2 N 随机扰动评估 |
| `.\run.ps1 train` | 启动 PPO 训练 |
| `.\run.ps1 demo ...` | 将参数直接传给 `scripts/demo.py` |

高级用法仍可直接指定脚本：

```powershell
.\run.ps1 scripts\eval.py --headless --num_envs 512 --episode_s 60
```

## 快速开始

首先验证物理资产：

```powershell
.\run.ps1 check
```

检查内容包括：

- 5 个刚体和 4 个关节是否正确解析
- articulation 是否为固定基座
- 三根摆杆是否在重力作用下倾倒
- 小车是否响应水平驱动力
- 观测维度和数值是否有效

验证通过后运行附带的鲁棒 checkpoint：

```powershell
.\run.ps1
```

仓库只保留 `checkpoints/model_robust.pt` 作为可复现演示模型。训练日志、中间 checkpoint 和视频不纳入版本控制。

## MDP 定义

### 动作

动作空间为 1 维连续量：

```text
a in [-1, 1]
F = 50 a N
```

动作经裁剪后映射为施加在 `slider_to_cart` 上的水平力。

### 观测

策略观测共 11 维：

| 分量 | 维数 | 处理方式 |
| --- | ---: | --- |
| 小车位置 | 1 | 除以最大允许位置 2.8 m |
| 小车速度 | 1 | 除以 5 m/s |
| 三根摆杆绝对倾角 | 6 | 每根使用 `sin(theta)` 和 `cos(theta)` |
| 三根摆杆绝对角速度 | 3 | 除以 5 rad/s |

三级摆杆使用相对关节角的累加值计算绝对倾角，避免链式结构出现局部角度较小但整体已经明显倾斜的情况。

### 时间尺度

- 物理仿真频率：120 Hz
- 控制频率：60 Hz
- `decimation = 2`
- 默认回合时长：8 s，约 480 个控制步

### 奖励

单步奖励由以下部分组成：

```text
+ 1.00 * alive
+ 2.00 * mean(cos(absolute_tilt))
- 0.50 * normalized_cart_position^2
- 0.02 * cart_velocity^2
- 0.01 * sum(link_angular_velocity^2)
- 0.05 * action^2
- 0.05 * action_delta^2
- 20.0  when terminated
```

### 终止条件

- 任一摆杆绝对倾角超过 0.35 rad
- 小车位置超出 +/-2.8 m
- 达到回合时间上限时产生 timeout

## PPO 训练

默认训练配置：

- 4096 个并行环境
- 每个环境每次收集 32 步
- Actor/Critic 隐藏层：256、128、64
- ELU 激活函数
- `gamma = 0.995`
- 自适应学习率，初始值 `1e-3`
- 默认 3000 次迭代，每 100 次保存 checkpoint

启动训练：

```powershell
.\run.ps1 train
```

覆盖并行环境数和训练迭代数：

```powershell
.\run.ps1 train --num_envs 8192 --max_iterations 6000
```

从最新日志恢复：

```powershell
.\run.ps1 train --resume
```

加入 reset 随机化和训练扰动：

```powershell
.\run.ps1 train --push_force 4 --init_scale 4 --max_tilt 0.6
```

训练输出位于：

```text
logs/rsl_rl/triple_cartpole/<timestamp>/
```

## 评估

无扰动评估：

```powershell
.\run.ps1 eval
```

2 N 随机扰动评估：

```powershell
.\run.ps1 robust
```

自定义评估：

```powershell
.\run.ps1 eval --num_envs 512 --episode_s 60 --init_scale 2
.\run.ps1 eval --checkpoint checkpoints/model_robust.pt
```

评估输出包含：

- `balanced`：整个评估区间内未终止的环境比例
- `mean_steps`：平均有效步数
- `peak_tilt`：存活环境的最大绝对倾角
- `end_tilt`：结束时平均绝对倾角
- `peak_cart`：最大绝对小车位置
- `mean_force`：平均控制力

评估会固定随机种子，使多个 checkpoint 面对相同的初始状态。

## 扰动演示

```powershell
.\run.ps1 disturb
```

默认演示从小倾角开始，随后对顶端摆杆施加短时侧向外力。渲染窗口中的红色标记表示当前受力刚体，
手动控制面板可以选择正向或反向推力。

自定义扰动参数：

```powershell
.\run.ps1 demo --showcase --disturb 3 --disturb_every 2 --disturb_ms 100
.\run.ps1 demo --showcase --disturb_body pole1 --manual_force
```

## 训练进度视频

`scripts/record_progress.py` 可以按照迭代顺序加载一组 checkpoint，并生成带状态信息的训练进度视频：

```powershell
.\run.ps1 scripts\record_progress.py --headless --lang zh
```

视频文件输出到对应训练日志目录，不纳入版本控制。

## 目录结构

```text
assets/
  triple_cartpole.usd
  triple_cartpole_physics.usd
checkpoints/
  model_robust.pt
triple_cartpole/
  robot.py
  env.py
  agents/rsl_rl_ppo_cfg.py
scripts/
  prepare_asset.py
  check_env.py
  train.py
  eval.py
  demo.py
  record_progress.py
run.ps1
```

## 许可证

项目代码和随仓库提供的资产采用 [MIT License](LICENSE)。
