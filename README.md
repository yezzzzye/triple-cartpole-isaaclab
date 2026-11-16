# Triple cartpole

Isaac Lab Direct RL task: one horizontally moving cart connected to three serial pendulum
links. The only control action is cart force (±50 N); all pendulum joints are passive.
One PPO policy performs swing-up, a continuous upright hold of at least 10 seconds, and
disturbance recovery. Acceptance trials never switch controllers, rewrite live robot states,
or reset during a trajectory.

[中文说明](README.zh-CN.md) · [45-second uncut rollout](reports/videos/swingup_down_exact_seed2-step-0.mp4) · [Evaluation protocol and results](reports/README.md)

## Run

Run commands from the repository root. In the local course bundle, this repository lives in `code/`; videos and other course deliverables live beside it.

```powershell
.\demo.ps1                       # Exact downward release, impulse every 25 seconds
.\demo.ps1 --initial_pose random # Large random initial angles
.\demo.ps1 --disturb 0           # Disable playback impulses
.\evaluate.ps1                   # Strict acceptance trials, saved under reports
```

Playback is continuous. Close the window to stop, or use `--steps N`. The default policy is
`checkpoints/swingup_policy.pt`; use `--checkpoint PATH` to load another.

## Runtime

Verified with Isaac Sim 5.1.0 standalone / Python 3.11.13, Isaac Lab 2.3.2 source,
PyTorch 2.7.0+cu128, rsl-rl-lib 3.0.1, NumPy 1.26.0, tensordict 0.11.0, and h5py 3.11.0.
`run.ps1` configures import paths and uses the simulator's bundled Python. Override local
installation paths when needed:

```powershell
$env:ISAACSIM_PATH = 'D:\path\to\isaac-sim'
$env:ISAACLAB_PATH = 'D:\path\to\IsaacLab'
```

## Train and inspect

```powershell
# swingup is the default task
.\run.ps1 scripts/train.py --task swingup --headless --num_envs 2048 --max_iterations 4000 --seed 2
.\run.ps1 scripts/train.py --headless --resume
.\run.ps1 scripts/train.py --headless --checkpoint checkpoints/swingup_policy.pt
.\run.ps1 scripts/check_env.py --task swingup --headless
tensorboard --logdir logs/rsl_rl/triple_cartpole_swingup
```

Resume selects the newest training checkpoint, falling back to the delivered policy when
training logs contain no weights. Each training run saves weights, configuration, and
TensorBoard data. Select policies with `scripts/eval_swingup.py --checkpoint PATH` rather
than assuming the final iteration is best.

## Model and acceptance

The authored geometry and masses are retained: cart 1 kg, each link 0.5 kg, consecutive
pivots 3 m apart. Swing-up uses ±12 m cart travel and ground at z = −10 m so all links
can hang freely. The scoring boundary is ±11.8 m. Eleven observations describe cart
position and velocity, sine/cosine of absolute link angles, and absolute angular rates.

Acceptance requires all links within ±0.35 rad of upright for 10 seconds, then a global X
12 N / 200 ms impulse on pole3, followed by another 10-second hold. Measured complete
sequences: exact downward **128/128**, perturbed downward **128/128**, large random angles
**226/256 (88.3%)**. Random starts can still fail, mainly by crossing the track boundary
during swing-up. See the [report](reports/README.md) for sampling, horizons, individual
trials, and limitations.

## Modules

| Path | Responsibility |
| --- | --- |
| `triple_cartpole/env_cfg.py` | Balance / swing-up settings and evaluation initial distributions |
| `triple_cartpole/env.py` | Physics actions, observations, rewards, terminations, training resets |
| `triple_cartpole/robot.py` | Assets, joints, actuator settings |
| `triple_cartpole/agents/` | PPO configuration |
| `scripts/_common.py` | Paths, checkpoint selection, policy runner |
| `scripts/train.py` / `demo.py` | Training / continuous playback |
| `scripts/eval_swingup.py` / `eval.py` | Swing-up acceptance / baseline balance scoring |
| `scripts/check_env.py` / `prepare_asset.py` | Physics checks / asset conversion |
| `scripts/plot_evaluation.py` | Trajectory plotting |
| `assets/` | Authored model and two task assets |
| `checkpoints/` | Delivered policies, training configuration, provenance |
| `reports/` | Acceptance JSON, CSV, plot, complete video |
| `logs/` | Local training history, excluded from version control |

The authored USD is a scene. Asset conversion removes scene objects, sets the articulation
root and fixed base, repairs joint references, and adds a force drive only to the cart.
Regenerate the swing-up asset with:

```powershell
.\run.ps1 scripts/prepare_asset.py --output assets/triple_cartpole_swingup.usd --slider-limit-m 12
```

The original ±3 m balance task and a separate policy remain available through
`.\demo.ps1 --task balance`, `scripts/train.py --task balance`, and `scripts/eval.py`.

## License

MIT; see [LICENSE](LICENSE).
