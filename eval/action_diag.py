"""Multimodal-aware OFFLINE diagnostics for a GR00T flow-matching bimanual policy.

Plain action-MSE is a poor yardstick for this task: it is multimodal (gather the cylinder from either
side -> push together -> lift to chest) and a flow policy models the whole distribution, so a single-sample
MSE rewards mode-averaging and under-movement. This runs the battery the field uses to judge multimodal /
diffusion / flow policies offline (no robot needed), sampling the flow head N times per observation:

  METRICS (per checkpoint, aggregated over the held-out episodes, split by phase):
   - mean_L2 / bestN_L2  : mean-of-samples vs best-of-N sample error. A LARGE mean-vs-best GAP = the policy
                           commits to distinct valid modes (good); a SMALL gap = mode-averaging.
   - diversity           : std across the N samples. ~0 everywhere = mode collapse.
   - chunk_dtw / traj_dtw: Dynamic-Time-Warping distance to the demo -> timing-robust trajectory match.
   - horizon_err_*       : open-loop error vs steps-ahead within the predicted plan (compounding proxy).
   - jerk                : mean 2nd-difference within a predicted chunk (jitter / non-committal proxy).
   - grasp_commit        : does the predicted hand actually close during gather/lift (profile correlation)?
   - spread_corr         : does the policy reproduce the arms-spread-then-converge coordination?

  Phases: overall + early/mid/late thirds (reach / gather / lift) + a data-driven GATHER window
  (frames where |spread| > 60% of its episode max). Also saves demo-vs-prediction trajectory plots.

  Requires Isaac-GR00T on PYTHONPATH and a LeRobot-format held-out split manifest (see data/make_split.py).
  python eval/action_diag.py --ckpt <dir> --split split.json [--delta] \
      --tag absolute --n-samples 8 --stride 20 --out eval/diag_results.csv --plot-dir eval/diag_plots
"""
import argparse, csv, json, os
import cv2, numpy as np

INSTRUCTION = "Pick up the cylinder from the table and lift it to the chest."
DATA_CONFIG = "unitree_g1_bimanual"
SL = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "left_hand": slice(14, 21), "right_hand": slice(21, 28)}
H, W = 480, 640
HORIZON = 16
# arm joint order: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
L_SROLL, R_SROLL, L_SPITCH, R_SPITCH = 1, 8, 0, 7

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--split", required=True, help="split manifest json (keys: base, heldout=[episode dirs])")
p.add_argument("--delta", action="store_true")
p.add_argument("--tag", default="")
p.add_argument("--n-samples", type=int, default=8)
p.add_argument("--stride", type=int, default=20)
p.add_argument("--out", default="")
p.add_argument("--plot-dir", default="")
p.add_argument("--plot-episodes", type=int, default=2)
A = p.parse_args()


def vec(dp, kind):
    d = dp[kind]
    return np.array(d["left_arm"]["qpos"] + d["right_arm"]["qpos"] +
                    d["left_ee"]["qpos"] + d["right_ee"]["qpos"], np.float32)


def load_policy(ckpt):
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    dc = DATA_CONFIG_MAP[DATA_CONFIG]
    return Gr00tPolicy(model_path=ckpt, modality_config=dc.modality_config(),
                       modality_transform=dc.transform(), embodiment_tag="new_embodiment")


def imgs(ep, k):
    out = []
    for c in range(3):
        b = cv2.imread(f"{ep}/colors/{k:06d}_color_{c}.jpg")
        b = cv2.resize(b, (W, H)) if b.shape[:2] != (H, W) else b
        out.append(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
    return out


def obs_at(ims, state):
    return {"video.head": ims[0][None], "video.left_wrist": ims[1][None], "video.right_wrist": ims[2][None],
            "state.left_arm": state[SL["left_arm"]][None], "state.right_arm": state[SL["right_arm"]][None],
            "state.left_hand": state[SL["left_hand"]][None], "state.right_hand": state[SL["right_hand"]][None],
            "annotation.human.task_description": [INSTRUCTION]}


def pred_chunk(policy, obs):
    a = policy.get_action(obs)
    def g(k):
        v = np.asarray(a[k]); return v[0] if v.ndim == 3 else v
    return np.concatenate([g("action.left_arm"), g("action.right_arm"),
                           g("action.left_hand"), g("action.right_hand")], axis=-1).astype(np.float32)


def dtw(a, b):
    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf); D[0, 0] = 0.0
    C = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[na, nb] / (na + nb))


def closedness(a28):
    return (np.linalg.norm(a28[..., SL["left_hand"]], axis=-1) + np.linalg.norm(a28[..., SL["right_hand"]], axis=-1)) / 2


def spread(a28):
    return a28[..., L_SROLL] - a28[..., R_SROLL]


def phase_masks(n, spread_prof):
    third = max(1, n // 3)
    masks = {"early": np.zeros(n, bool), "mid": np.zeros(n, bool), "late": np.zeros(n, bool)}
    masks["early"][:third] = True; masks["mid"][third:2 * third] = True; masks["late"][2 * third:] = True
    sp = np.abs(spread_prof - spread_prof[0])
    thr = 0.6 * sp.max() if sp.max() > 1e-6 else np.inf
    gather = sp >= thr
    masks["gather"] = gather if gather.any() else masks["mid"]
    return masks


def main():
    man = json.load(open(A.split)); base = man["base"]
    policy = load_policy(A.ckpt)
    agg = {}
    def add(k, v):
        agg.setdefault(k, []).append(float(v))
    plotted = 0

    for eid in man["heldout"]:
        ep = f"{base}/{eid}"
        frames = json.load(open(f"{ep}/data.json"))["data"]
        n = len(frames)
        states = np.stack([vec(f, "states") for f in frames])
        acts = np.stack([vec(f, "actions") for f in frames])
        demo_spread = spread(acts)
        masks = phase_masks(n, demo_spread)
        idxs = list(range(0, n - 1, A.stride))
        first_pred_mean, first_pred_all = np.full((n, 28), np.nan), {}

        for t in idxs:
            gt_chunk = acts[t:min(t + HORIZON, n)]
            samples = []
            for _ in range(A.n_samples):
                ch = pred_chunk(policy, obs_at(imgs(ep, t), states[t]))
                if A.delta:
                    ch = ch + states[t]
                samples.append(ch)
            samples = np.stack(samples)
            first = samples[:, 0, :]
            gt0 = gt_chunk[0]
            l2 = np.linalg.norm(first - gt0, axis=-1)
            arm_l2 = np.linalg.norm(first[:, :14] - gt0[:14], axis=-1)
            hand_l2 = np.linalg.norm(first[:, 14:] - gt0[14:], axis=-1)
            add("mean_L2", l2.mean()); add("bestN_L2", l2.min())
            add("arm_mean_L2", arm_l2.mean()); add("arm_bestN_L2", arm_l2.min())
            add("hand_mean_L2", hand_l2.mean()); add("hand_bestN_L2", hand_l2.min())
            add("diversity", first.std(axis=0).mean())
            for k in (1, 2, 4, 8):
                if k <= A.n_samples:
                    add(f"bestof{k}_L2", l2[:k].min())
            herr = [np.linalg.norm(samples[:, j, :] - gt_chunk[j], axis=-1).min() for j in range(len(gt_chunk))]
            add("horizon_err_first", herr[0]); add("horizon_err_last", herr[-1])
            add("horizon_slope", (herr[-1] - herr[0]) / max(1, len(gt_chunk) - 1))
            hh = len(gt_chunk)
            add("chunk_dtw", min(dtw(s[:hh], gt_chunk) for s in samples))
            add("jerk", np.linalg.norm(np.diff(samples.mean(0), 2, axis=0), axis=-1).mean())
            for ph, mk in masks.items():
                if mk[t]:
                    add(f"bestN_L2_{ph}", l2.min()); add(f"mean_L2_{ph}", l2.mean())
            first_pred_mean[t] = first.mean(0); first_pred_all[t] = first

        q = np.array(idxs)
        pred_traj, demo_traj = first_pred_mean[q], acts[q]
        add("traj_dtw", dtw(pred_traj, demo_traj))
        ps, ds = spread(pred_traj), spread(demo_traj)
        if ps.std() > 1e-6 and ds.std() > 1e-6:
            add("spread_corr", np.corrcoef(ps, ds)[0, 1])
        pc, dc = closedness(pred_traj), closedness(demo_traj)
        add("grasp_pred_peak", pc.max()); add("grasp_demo_peak", dc.max())
        if pc.std() > 1e-6 and dc.std() > 1e-6:
            add("grasp_corr", np.corrcoef(pc, dc)[0, 1])
        if A.plot_dir and plotted < A.plot_episodes:
            plot_episode(ep, eid, q, acts, first_pred_all); plotted += 1

    row = {k: float(np.nanmean(v)) for k, v in agg.items()}
    row.update(tag=A.tag, ckpt=A.ckpt, delta=int(A.delta))
    row["gap_ratio"] = row.get("bestN_L2", np.nan) / row.get("mean_L2", np.nan)
    print(f"\n===== ACTION DIAGNOSTICS: {A.tag or A.ckpt} =====")
    for k in ["mean_L2", "bestN_L2", "gap_ratio", "diversity", "bestof1_L2", "bestof2_L2", "bestof4_L2",
              "bestof8_L2", "arm_mean_L2", "arm_bestN_L2", "hand_mean_L2", "hand_bestN_L2", "chunk_dtw",
              "traj_dtw", "jerk", "horizon_err_first", "horizon_err_last", "horizon_slope",
              "spread_corr", "grasp_corr", "grasp_pred_peak", "grasp_demo_peak",
              "bestN_L2_gather", "bestN_L2_early", "bestN_L2_mid", "bestN_L2_late"]:
        if k in row:
            print(f"  {k:20s} {row[k]:.5f}")
    if A.out:
        new = not os.path.exists(A.out)
        keys = ["tag", "ckpt", "delta", "mean_L2", "bestN_L2", "gap_ratio", "diversity", "bestof1_L2",
                "bestof2_L2", "bestof4_L2", "bestof8_L2", "arm_mean_L2", "arm_bestN_L2", "hand_mean_L2",
                "hand_bestN_L2", "chunk_dtw", "traj_dtw", "jerk", "horizon_err_first", "horizon_err_last",
                "horizon_slope", "spread_corr", "grasp_corr", "grasp_pred_peak", "grasp_demo_peak",
                "bestN_L2_gather", "bestN_L2_early", "bestN_L2_mid", "bestN_L2_late"]
        with open(A.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[diag] appended -> {A.out}")


def plot_episode(ep, eid, q, acts, first_pred_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(A.plot_dir, exist_ok=True)
    N = A.n_samples
    names = ["L shoulder-roll (spread L)", "R shoulder-roll (spread R)", "grasp closedness", "shoulder-pitch (lift)"]
    idxs = [L_SROLL, R_SROLL, "grasp", "lift"]
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 9), sharex=True)
    for ax, name, idx in zip(axes, names, idxs):
        if idx == "grasp":
            demo = closedness(acts); pf = lambda a: closedness(a)
        elif idx == "lift":
            demo = -(acts[:, L_SPITCH] + acts[:, R_SPITCH]) / 2; pf = lambda a: -(a[L_SPITCH] + a[R_SPITCH]) / 2
        else:
            demo = acts[:, idx]; pf = lambda a, i=idx: a[i]
        ax.plot(np.arange(len(demo)), demo, "k-", lw=2, label="demo")
        for t in q:
            if t in first_pred_all:
                ax.scatter([t] * N, [pf(s) for s in first_pred_all[t]], s=8, c="C0", alpha=0.4)
        ax.plot(q, [pf(first_pred_all[t].mean(0)) if t in first_pred_all else np.nan for t in q],
                "C1o-", lw=1, ms=3, label="pred mean")
        ax.set_ylabel(name, fontsize=8); ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"{A.tag}  {eid}  (demo vs {N}-sample predictions)")
    fig.tight_layout()
    out = f"{A.plot_dir}/{A.tag}_{eid.replace('/', '_')}.png"
    fig.savefig(out, dpi=90); plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
