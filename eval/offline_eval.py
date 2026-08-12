"""Single-sample offline held-out action error (absolute-joint MSE), delta-aware.

A quick checkpoint-ranking metric: run the policy on the held-out episodes, compare the first predicted
action to the recorded target. For --delta checkpoints the prediction is reconstructed to absolute
(query_state + delta) so numbers are comparable across absolute and delta models. Prefer action_diag.py
for the multimodal picture; this is the cheap scalar.

Requires Isaac-GR00T on PYTHONPATH.
  python eval/offline_eval.py --ckpt <dir> --split split.json [--delta] --out eval/offline.csv --tag absolute
"""
import argparse, csv, json, os
import cv2, numpy as np

INSTRUCTION = "Pick up the cylinder from the table and lift it to the chest."
DATA_CONFIG = "unitree_g1_bimanual"
SL = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "left_hand": slice(14, 21), "right_hand": slice(21, 28)}
H, W = 480, 640

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--split", required=True)
p.add_argument("--delta", action="store_true")
p.add_argument("--stride", type=int, default=15)
p.add_argument("--out", default="")
p.add_argument("--tag", default="")
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


def obs_at(ep, k, state):
    def img(c):
        b = cv2.imread(f"{ep}/colors/{k:06d}_color_{c}.jpg")
        b = cv2.resize(b, (W, H)) if b.shape[:2] != (H, W) else b
        return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    return {"video.head": img(0)[None], "video.left_wrist": img(1)[None], "video.right_wrist": img(2)[None],
            "state.left_arm": state[SL["left_arm"]][None], "state.right_arm": state[SL["right_arm"]][None],
            "state.left_hand": state[SL["left_hand"]][None], "state.right_hand": state[SL["right_hand"]][None],
            "annotation.human.task_description": [INSTRUCTION]}


def pred28(policy, obs):
    a = policy.get_action(obs)
    def g(k):
        v = np.asarray(a[k]); v = v[0] if v.ndim == 3 else v
        return v[0]
    return np.concatenate([g("action.left_arm"), g("action.right_arm"),
                           g("action.left_hand"), g("action.right_hand")]).astype(np.float32)


def main():
    man = json.load(open(A.split)); base = man["base"]
    policy = load_policy(A.ckpt)
    arm_se, hand_se, n = 0.0, 0.0, 0
    for eid in man["heldout"]:
        ep = f"{base}/{eid}"
        frames = json.load(open(f"{ep}/data.json"))["data"]
        for k in range(0, len(frames), A.stride):
            s = vec(frames[k], "states"); gt = vec(frames[k], "actions")
            pr = pred28(policy, obs_at(ep, k, s))
            if A.delta:
                pr = pr + s
            arm_se += float(np.mean((pr[:14] - gt[:14]) ** 2))
            hand_se += float(np.mean((pr[14:] - gt[14:]) ** 2))
            n += 1
    arm, hand = arm_se / n, hand_se / n
    tot = (arm * 14 + hand * 14) / 28
    print(f"[eval] {A.tag or A.ckpt}  frames={n}  MSE total={tot:.5f}  arm={arm:.5f}  hand={hand:.5f}")
    if A.out:
        new = not os.path.exists(A.out)
        with open(A.out, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tag", "ckpt", "delta", "frames", "mse_total", "mse_arm", "mse_hand"])
            w.writerow([A.tag, A.ckpt, int(A.delta), n, f"{tot:.6f}", f"{arm:.6f}", f"{hand:.6f}"])


if __name__ == "__main__":
    main()
