# Offline evaluation of a bimanual flow-matching policy

Why not plain action-MSE: GR00T N1.5's action head is a **flow-matching** generator and the task is
**multimodal** (gather the cylinder from either side → push together → lift). Single-sample MSE conflates
"wrong action" with "right action, different valid mode", and is minimized by collapsing to the conditional
mean — the classic mode-averaging failure. The flagship policy papers (Diffusion Policy, ACT, RT-1/2, Octo,
π0) report **task success**, not offline MSE, for this reason; offline metrics are only a cheap signal for
checkpoint ranking / regression catching. `action_diag.py` implements the multimodal-aware battery.

## The battery (`action_diag.py`) — sample the flow head N times per observation

| Metric | What it catches | Read |
|---|---|---|
| `mean_L2` vs `bestN_L2`, `gap_ratio`, `bestof{1,2,4,8}_L2` | mode-averaging (best-of-N is *the* multimodal metric, cf. minADE/minFDE) | big mean→bestN drop (small `gap_ratio`) = commits to distinct valid modes; ~1 = averaging |
| `diversity` (std across N) | mode collapse | ≈0 everywhere = collapsed flow head |
| `chunk_dtw`, `traj_dtw` | trajectory shape, timing-robust | lower = better; DTW aligns then measures |
| `horizon_err_{first,last,slope}` | compounding error within the plan (open-loop proxy) | steep positive slope = degrades over the horizon |
| `jerk` | jitter / non-committal oscillation | quality tripwire, NOT accuracy |
| `arm_*` vs `hand_*`, per-phase (`_early/_mid/_late/_gather`) | which limb / which phase fails | the gather phase is the multimodal-critical one |
| `spread_corr` | reproduces arms-spread-then-converge coordination | →1 good; ~0 = stays flat/middle (averaging) |
| `grasp_corr`, `grasp_pred_peak` vs `grasp_demo_peak` | does the hand *commit* to closing | peak far below demo = under-committing |

**Healthy multimodal policy** = low `bestN_L2` + healthy `diversity` + large mean→best gap + high
`spread_corr`/`grasp_corr` + flat `horizon_slope`. **A collapsed policy that flatters legacy MSE** = low
`diversity`, `gap_ratio`≈1, low `spread_corr` — this battery exposes it.

## Deliberately omitted (and why)
- **MMD / Wasserstein / precision-recall** (Gretton 2012; Kynkäänniemi 2019) need *multiple demo samples per
  state* to form a target distribution. A held-out set with one ground-truth per frame can't compute them
  cleanly; `diversity` + best-of-N + `gap_ratio` serve as the collapse detector instead.
- GR00T ships `gr00t/eval/open_loop_eval.py` but it is **single-sample MSE/MAE** — the weak metric; this
  suite supersedes it with N-sampling + multimodal metrics.

## Hard caveat
All of this is **open-loop** and cannot capture closed-loop compounding error (Ross & Bagnell 2011;
arXiv 2503.09722). Use it to rank checkpoints and catch collapse; confirm the winner on the **robot**
(`deploy/policy_runner.py`, plus `eval/spatial_heatmap.py` for a success map over object positions).

## Sources
Diffusion Policy https://arxiv.org/abs/2303.04137 · minADE/minFDE (Argoverse/nuScenes) · MMD Gretton JMLR
2012 · Improved Precision/Recall https://arxiv.org/abs/1904.06991 · Soft-DTW https://arxiv.org/abs/1703.01541 ·
compounding error https://arxiv.org/abs/2503.09722, DAgger https://arxiv.org/abs/1011.0686 ·
eval best-practice https://arxiv.org/abs/2409.09491 · GR00T N1.5
https://research.nvidia.com/labs/gear/gr00t-n1_5/ · Isaac-GR00T https://github.com/NVIDIA/Isaac-GR00T
