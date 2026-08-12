# G1-D Bimanual Manipulation (and Locomotion)

Training, evaluation, and deployment code for a **bimanual pick-up policy** on a Unitree **G1-D**
humanoid (fixed trunk, dual arms + dexterous hands), learned from teleoperation demonstrations with a
**flow-matching Vision-Language-Action policy** (NVIDIA GR00T N1.5). The task: with both arms, gather a
**cylinder** off a table, secure it, and lift it to the chest.

This repo is a **methods reference** — the pipeline, the training recipe, and (the part I think is most
reusable) a **multimodal-aware offline evaluation suite** and a **spatial success-heatmap** protocol. It
does not ship trained weights or datasets; the object is a plain cylinder and the only assets are the
public G1 humanoid and a generic table.

## Pipeline

```
teleop demos ──▶ data/convert_to_lerobot.py ──▶ LeRobot v2.0 dataset
                                                     │
                     train/run_sweep.py (GR00T N1.5 flow-matching finetune)
                                                     │
        eval/  ── offline multimodal battery  +  spatial success heatmap
                                                     │
              deploy/policy_runner.py  (live camera preview, compliant PD, delta decode)
```

### 1. Data (`data/`)
- `convert_to_lerobot.py` — turn per-frame teleop recordings (RGB + joint states/actions) into a
  GR00T-ready **LeRobot v2.0** dataset. 3 cameras (head + two wrists), 28-dim state/action
  (2×7 arm + 2×7 hand). Supports **absolute** or **delta** (residual-to-current) action encoding.
- `make_split.py` — reproducible train / held-out split, stratified by a **difficulty score**
  (`z(episode_length) + z(arm_travel)`), so the held-out set spans easy→hard uniformly.

### 2. Training (`train/`)
GR00T N1.5's action head is a **flow-matching** transformer, so the training objective is already a
generative one (not MSE regression). `run_sweep.py` runs a small **controlled comparison** of design
choices — absolute vs. delta actions, and mixed vs. curriculum data ordering — each evaluated on the
same held-out split. See `train/README.md` for the **8-bit paged-AdamW recipe** that makes a full
action-head finetune fit a single 24 GB GPU (the naive fp32 optimizer does not).

### 3. Evaluation (`eval/`) — the interesting part
Plain action-MSE is a poor metric for a **multimodal** task (many valid ways to gather/lift), because it
rewards mode-averaging. The suite samples the flow head N times per observation and reports best-of-N
error, sample diversity, timing-robust DTW, an open-loop horizon-error curve, and task-specific
"did the arms actually spread-and-commit / did the hand close" signals. See `eval/EVAL_METRICS.md` for
the full list with citations. `eval/spatial_heatmap.py` builds a **success-rate heatmap** over object
positions (with confidence intervals, training-distribution overlay, and failure-mode breakdown) — the
protocol for reporting real-robot generalization.

### 4. Deployment (`deploy/`)
`policy_runner.py` — a light control loop with a **live camera preview** (confirm the streams before
engaging), **compliant PD** with a joint-velocity cap, a Cartesian **governance box**, and support for
both absolute and **delta** policies. Records a review video of every run.

### 5. Simulation (`sim/`)
`cylinder_scene.py` — a minimal Isaac scene (G1 + table + cylinder) used to render the demonstration
videos in `docs/videos/`.

## Requirements
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (flow-matching VLA + finetuning)
- A LeRobot-format dataset produced by `data/convert_to_lerobot.py`
- Single 24 GB GPU is enough for finetuning with the 8-bit recipe in `train/README.md`

## Notes
Offline metrics rank checkpoints and catch mode-collapse; they cannot capture closed-loop compounding
error — the real gate is on-robot success (`eval/spatial_heatmap.py`). Trained weights and demonstration
datasets are intentionally not included.

## License
MIT — see `LICENSE`.
