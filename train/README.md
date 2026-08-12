# Training

Finetune GR00T N1.5's flow-matching action head on the LeRobot dataset produced by `data/`. Two practical
notes that took real debugging:

## Fitting a full action-head finetune on a single 24 GB GPU

The naive full finetune (fp32 AdamW, tuning the vision tower) does **not** fit 24 GB — the optimizer states
alone (~15 GB) overflow at the first optimizer step, independent of batch size. Two changes make it fit:

1. **8-bit paged AdamW.** Use `bitsandbytes` and set `optim="paged_adamw_8bit"` in the `TrainingArguments`.
   This cuts the optimizer moments ~4× and pages them to CPU on spikes. This is the single change that makes
   a real full finetune trainable on a 24 GB card.
2. **Freeze the vision tower** (`--no-tune-visual`). Beyond the memory savings, a frozen pretrained vision
   backbone is *less* prone to overfitting a small dataset's exact appearance, which tends to help
   generalization to new backgrounds/tables. Keep it consistent across runs you want to compare.

Also enable `--gradient-checkpointing`, and split the effective batch with `--gradient-accumulation-steps`
(e.g. micro-batch 4 × accum 8 = effective 32).

## The comparison sweep (`run_sweep.py`)

Rather than guess, `run_sweep.py` trains three variants and evaluates each on the **same** held-out split:

| Run | Joint space | Data ordering |
|---|---|---|
| `absolute` (control) | absolute qpos | all data mixed |
| `delta` | residual-to-current (`action = target − measured`) | all data mixed |
| `curriculum` | absolute qpos | easy subset first, then full (warm-started) |

Each changes exactly one thing vs. the control, so `absolute` vs. `delta` isolates the joint-space encoding
and `absolute` vs. `curriculum` isolates data ordering. **Iterations are a within-run axis**: checkpoints
are saved periodically and evaluated individually, giving the MSE-vs-steps curve for free — extend a run
only if its held-out curve is still descending (not if only the training loss is falling).

## Reading the results

`offline_eval.py` gives a fast scalar for checkpoint ranking; **`action_diag.py` is the one to trust** for
this task (see `../eval/EVAL_METRICS.md`) because plain MSE rewards mode-averaging. A delta policy in
particular scores deceptively low on MSE (predicting a small residual ≈ "stay near current"), so confirm any
winner with the multimodal battery and then on the robot (`../eval/spatial_heatmap.py`).

## Dependencies
`bitsandbytes`, and the [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) finetune script (pointed to via
`--finetune-script`). Register a data-config for this embodiment (3 cameras, 28-dim state/action) named to
match `--data-config`.
