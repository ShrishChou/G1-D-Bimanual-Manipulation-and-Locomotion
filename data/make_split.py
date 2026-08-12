"""Build a reproducible train / held-out split, stratified by difficulty.

Difficulty = z(episode_length) + z(arm_travel) (long + high-reach/repositioning episodes), computed across
all source datasets so the held-out set spans easy->hard uniformly and is not biased to one collection.

Writes split.json with dataset-qualified episode ids ("dataset_a/episode_0007"):
  heldout     : stratified sample (hard / mid / easy-grasp / gripper-free), excluded from ALL training
  train_full  : everything else
  train_easy  : train_full below the difficulty p70 (for a curriculum warm-up phase)

  python data/make_split.py --base /path/to/raw --datasets dataset_a dataset_b --out split.json
"""
import argparse, glob, json, os, random
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="root holding the dataset folders")
ap.add_argument("--datasets", nargs="+", required=True, help="dataset folder names under --base")
ap.add_argument("--out", default="split.json")
ap.add_argument("--heldout", type=int, default=8)
ap.add_argument("--seed", type=int, default=42)
A = ap.parse_args()
rng = random.Random(A.seed)


def stats(ep):
    d = json.load(open(f"{ep}/data.json"))["data"]
    aL = np.array([f["actions"]["left_arm"]["qpos"] for f in d])
    aR = np.array([f["actions"]["right_arm"]["qpos"] for f in d])
    eeL = np.array([f["actions"]["left_ee"]["qpos"] for f in d])
    eeR = np.array([f["actions"]["right_ee"]["qpos"] for f in d])
    arm_tr = float(np.abs(aL.max(0) - aL.min(0)).sum() + np.abs(aR.max(0) - aR.min(0)).sum())
    hand = float(max(np.linalg.norm(eeL, axis=1).max(), np.linalg.norm(eeR, axis=1).max()))
    return len(d), arm_tr, hand


ids, length, travel, grasp = [], [], [], []
for ds in A.datasets:
    for ep in sorted(glob.glob(f"{A.base}/{ds}/episode_*")):
        if not os.path.exists(f"{ep}/data.json"):
            continue
        n, at, hd = stats(ep)
        ids.append(f"{ds}/{os.path.basename(ep)}"); length.append(n); travel.append(at); grasp.append(hd > 0.1)

length = np.array(length, float); travel = np.array(travel, float); grasp = np.array(grasp)
z = lambda x: (x - x.mean()) / x.std()
diff = z(length) + z(travel)
p70 = float(np.quantile(diff, 0.70))
order = np.argsort(-diff)

hard_pool = [ids[i] for i in order[:max(4, len(ids) // 10)]]
grasp_ids = [ids[i] for i in np.where(grasp)[0]]
free_ids = [ids[i] for i in np.where(~grasp)[0]]
mid_pool = [ids[i] for i in np.argsort(np.abs(diff - np.median(diff)))[:20]]
heldout = []
def take(pool, k):
    cand = [x for x in pool if x not in heldout]; rng.shuffle(cand); heldout.extend(cand[:k])
per = max(1, A.heldout // 4)
take(hard_pool, per)
take(mid_pool, per)
take([x for x in grasp_ids if diff[ids.index(x)] < p70], per)
take(free_ids, A.heldout - len(heldout))
heldout = sorted(set(heldout))
train_full = sorted(x for x in ids if x not in heldout)
train_easy = sorted(x for x in train_full if diff[ids.index(x)] <= p70)

json.dump({"base": A.base, "difficulty": "z(length)+z(arm_travel)", "p70": p70,
           "heldout": heldout, "train_full": train_full, "train_easy": train_easy,
           "counts": {"total": len(ids), "heldout": len(heldout),
                      "train_full": len(train_full), "train_easy": len(train_easy)}},
          open(A.out, "w"), indent=2)
print(f"[split] total={len(ids)} heldout={len(heldout)} train_full={len(train_full)} train_easy={len(train_easy)} -> {A.out}")
