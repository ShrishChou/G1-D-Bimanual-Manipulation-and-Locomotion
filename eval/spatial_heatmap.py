"""Spatial success-rate heatmap for real-robot generalization.

Reads a trials CSV logged during evaluation and renders a professional generalization figure:
  (1) success-rate heatmap over object (x, y) positions, with per-cell trial counts and Wilson confidence
      intervals, the robot base marked, and the TRAINING object positions overlaid (the "succeeds where data
      was dense" story);
  (2) success-vs-distance-from-base falloff curve with CIs;
  (3) failure-mode breakdown.

Protocol (report this alongside the figure): fixed grid, N trials per cell, randomized object yaw within a
cell, a crisp pre-registered success criterion, and every trial logged (none dropped). Success is a rate,
not a bit -- that is why we bin and show CIs.

trials.csv columns:  x, y, success            (x,y in metres in the base frame; success in {0,1})
  optional:          yaw, failure_mode, trial, video
training CSV (--train-positions): columns x, y   (object positions seen during data collection)

  python eval/spatial_heatmap.py --trials trials.csv [--train-positions train_xy.csv] \
      --cell 0.05 --out docs/heatmap.png
"""
import argparse, csv, math
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--trials", required=True)
p.add_argument("--train-positions", default="")
p.add_argument("--cell", type=float, default=0.05, help="grid cell size (m)")
p.add_argument("--out", default="heatmap.png")
A = p.parse_args()


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial success rate (robust at small n)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def read_trials(path):
    rows = list(csv.DictReader(open(path)))
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    s = np.array([int(float(r["success"])) for r in rows])
    fm = [r.get("failure_mode", "") for r in rows]
    return x, y, s, fm


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    x, y, s, fm = read_trials(A.trials)
    c = A.cell
    xi = np.floor(x / c).astype(int); yi = np.floor(y / c).astype(int)
    cells = {}
    for a, b, ok in zip(xi, yi, s):
        cells.setdefault((a, b), []).append(ok)

    xs = sorted({a for a, _ in cells}); ys = sorted({b for _, b in cells})
    rate = np.full((len(ys), len(xs)), np.nan)
    ntxt = {}
    for (a, b), oks in cells.items():
        r, lo, hi = wilson(sum(oks), len(oks))
        j, i = ys.index(b), xs.index(a)
        rate[j, i] = r
        ntxt[(j, i)] = f"{sum(oks)}/{len(oks)}"

    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(2, 2, width_ratios=[1.4, 1], height_ratios=[1, 1], figure=fig)

    # (1) heatmap
    ax = fig.add_subplot(gs[:, 0])
    extent = [min(xs) * c, (max(xs) + 1) * c, min(ys) * c, (max(ys) + 1) * c]
    im = ax.imshow(rate, origin="lower", extent=extent, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for (j, i), t in ntxt.items():
        ax.text((xs[i] + 0.5) * c, (ys[j] + 0.5) * c, t, ha="center", va="center", fontsize=6, color="black")
    if A.train_positions:
        tr = list(csv.DictReader(open(A.train_positions)))
        tx = [float(r["x"]) for r in tr]; ty = [float(r["y"]) for r in tr]
        ax.scatter(tx, ty, s=10, marker="x", c="black", alpha=0.5, label="training positions")
    ax.scatter([0], [0], marker="^", s=140, c="blue", edgecolor="white", zorder=5, label="robot base")
    ax.set_xlabel("x (m, forward)"); ax.set_ylabel("y (m, lateral)")
    ax.set_title(f"Success rate over object position  (N={len(s)} trials, cell={c} m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="success rate")

    # (2) distance falloff
    axd = fig.add_subplot(gs[0, 1])
    d = np.sqrt(x ** 2 + y ** 2)
    edges = np.linspace(d.min(), d.max(), 7)
    mids, r_, lo_, hi_ = [], [], [], []
    for k in range(len(edges) - 1):
        m = (d >= edges[k]) & (d < edges[k + 1] if k < len(edges) - 2 else d <= edges[k + 1])
        if m.sum():
            r, lo, hi = wilson(int(s[m].sum()), int(m.sum()))
            mids.append((edges[k] + edges[k + 1]) / 2); r_.append(r); lo_.append(lo); hi_.append(hi)
    axd.errorbar(mids, r_, yerr=[np.array(r_) - lo_, np.array(hi_) - r_], marker="o", capsize=3)
    axd.set_ylim(-0.05, 1.05); axd.set_xlabel("distance from base (m)"); axd.set_ylabel("success rate")
    axd.set_title("Success vs. reach distance (Wilson CI)")

    # (3) failure modes
    axf = fig.add_subplot(gs[1, 1])
    fails = [m for m, ok in zip(fm, s) if ok == 0 and m]
    if fails:
        labels, counts = np.unique(fails, return_counts=True)
        axf.barh(labels, counts, color="C3"); axf.set_xlabel("count")
        axf.set_title("Failure modes")
    else:
        axf.text(0.5, 0.5, "no failure_mode labels", ha="center"); axf.axis("off")

    overall, olo, ohi = wilson(int(s.sum()), len(s))
    fig.suptitle(f"Overall success {overall:.0%}  (95% CI {olo:.0%}-{ohi:.0%}, N={len(s)})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(A.out, dpi=140)
    fig.savefig(A.out.rsplit(".", 1)[0] + ".svg")
    print(f"[heatmap] overall {overall:.1%} (CI {olo:.1%}-{ohi:.1%}, N={len(s)}) -> {A.out}")


if __name__ == "__main__":
    main()
