"""Controlled comparison of design choices for the bimanual flow-matching policy.

Runs three finetunes and evaluates each on the SAME held-out split, so differences are attributable:
  absolute    absolute-qpos actions, all data mixed              (control)
  delta       residual-to-current actions, all data mixed        (isolates joint-space encoding)
  curriculum  absolute, warm up on the easy subset then full     (isolates data ordering)

Iterations are swept WITHIN a run via the per-checkpoint held-out curve (eval every checkpoint), so we
don't spend separate runs on "train longer". Between runs it prunes optimizer state / non-best checkpoints
to stay within disk. Uses your Isaac-GR00T finetune script (see train/README.md for the 8-bit recipe that
fits a 24 GB GPU); this orchestrator just sequences train -> eval -> prune -> summarize.

  python train/run_sweep.py --finetune-script /path/to/gr00t_finetune.py \
      --data-root /path/to/lerobot_datasets --split split.json --out-root ./checkpoints
"""
import argparse, csv, glob, json, os, shutil, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--finetune-script", required=True, help="Isaac-GR00T finetune script (8-bit recipe)")
ap.add_argument("--data-root", required=True, help="dir containing the abs_full / delta_full / abs_easy datasets")
ap.add_argument("--split", required=True)
ap.add_argument("--out-root", default="./checkpoints")
ap.add_argument("--data-config", default="unitree_g1_bimanual")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--save-steps", type=int, default=1000)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--accum", type=int, default=8, help="grad-accum (effective batch = batch*accum)")
ap.add_argument("--eval", default=os.path.join(os.path.dirname(__file__), "..", "eval", "offline_eval.py"))
A = ap.parse_args()
RESULTS = os.path.join(A.out_root, "sweep_results.csv")
SUMMARY = os.path.join(A.out_root, "sweep_summary.md")
BASE_MODEL = "nvidia/GR00T-N1.5-3B"


def sh(cmd):
    print("\n$ " + " ".join(cmd), flush=True); subprocess.run(cmd, check=True)


def train(name, dataset, max_steps, base=BASE_MODEL):
    outdir = os.path.join(A.out_root, name)
    for d in glob.glob(f"{outdir}_*"):
        shutil.rmtree(d, ignore_errors=True)
    sh([sys.executable, A.finetune_script, "--dataset-path", dataset, "--data-config", A.data_config,
        "--embodiment-tag", "new_embodiment", "--video-backend", "torchvision_av", "--num-gpus", "1",
        "--batch-size", str(A.batch), "--gradient-accumulation-steps", str(A.accum),
        "--gradient-checkpointing", "--report-to", "tensorboard", "--max-steps", str(max_steps),
        "--save-steps", str(A.save_steps), "--output-dir", outdir, "--base-model-path", base])
    dirs = sorted(glob.glob(f"{outdir}_*"), key=os.path.getmtime)
    return dirs[-1] if dirs else outdir


def ckpts(rundir):
    return sorted(glob.glob(f"{rundir}/checkpoint-*"), key=lambda d: int(d.split("-")[-1]))


def evaluate(rundir, tag, delta=False):
    for c in ckpts(rundir):
        cmd = [sys.executable, A.eval, "--ckpt", c, "--split", A.split, "--out", RESULTS,
               "--tag", f"{tag}@{c.split('-')[-1]}"]
        if delta:
            cmd.append("--delta")
        try:
            sh(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[eval] FAILED {c}: {e}", flush=True)


def best_step(tag):
    if not os.path.exists(RESULTS):
        return None
    rows = [r for r in csv.DictReader(open(RESULTS)) if r["tag"].startswith(tag + "@")]
    return min(rows, key=lambda r: float(r["mse_total"]))["tag"].split("@")[-1] if rows else None


def prune(rundir, tag):
    cs = ckpts(rundir)
    if not cs:
        return
    keep = {cs[-1].split("-")[-1]} | ({best_step(tag)} if best_step(tag) else set())
    for c in cs:
        if c.split("-")[-1] in keep:
            for junk in ("optimizer.pt", "rng_state.pth", "scheduler.pt", "trainer_state.json"):
                fp = os.path.join(c, junk)
                if os.path.exists(fp):
                    os.remove(fp)
        else:
            shutil.rmtree(c, ignore_errors=True)
    print(f"[prune] {tag}: kept {sorted(keep)} -- {shutil.disk_usage(A.out_root).free/1e9:.0f} GB free", flush=True)


def summarize():
    if not os.path.exists(RESULTS):
        return
    runs = {}
    for r in csv.DictReader(open(RESULTS)):
        runs.setdefault(r["tag"].split("@")[0], []).append(r)
    with open(SUMMARY, "w") as f:
        f.write("# Sweep results\n\nHeld-out action MSE (lower=better). Same held-out for all runs.\n\n")
        f.write("| Run | Best step | MSE total | curve |\n|---|---|---|---|\n")
        for run, rs in runs.items():
            rs = sorted(rs, key=lambda r: int(r["tag"].split("@")[-1]))
            b = min(rs, key=lambda r: float(r["mse_total"]))
            curve = " ".join(f"{r['tag'].split('@')[-1]}:{float(r['mse_total']):.4f}" for r in rs)
            f.write(f"| {run} | {b['tag'].split('@')[-1]} | {float(b['mse_total']):.4f} | {curve} |\n")
    print(open(SUMMARY).read(), flush=True)


def main():
    os.makedirs(A.out_root, exist_ok=True)
    abs_full = os.path.join(A.data_root, "abs_full")
    delta_full = os.path.join(A.data_root, "delta_full")
    abs_easy = os.path.join(A.data_root, "abs_easy")

    r = train("absolute", abs_full, A.steps); evaluate(r, "absolute"); prune(r, "absolute"); summarize()
    r = train("delta", delta_full, A.steps); evaluate(r, "delta", delta=True); prune(r, "delta"); summarize()
    r1 = train("curriculum_p1", abs_easy, A.steps // 2)
    r2 = train("curriculum_p2", abs_full, A.steps, base=ckpts(r1)[-1])
    evaluate(r2, "curriculum"); prune(r2, "curriculum"); shutil.rmtree(r1, ignore_errors=True); summarize()
    print("\n=== SWEEP DONE ===", flush=True)


if __name__ == "__main__":
    main()
