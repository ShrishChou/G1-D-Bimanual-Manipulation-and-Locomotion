"""Proprioceptive pick-and-hold reward for the cylinder task -- auto-labels rollouts WITHOUT vision or object
tracking. This is the shared signal for reward-only RL, the RL-Token value head, and auto-labeling DAgger
corrections. All terms come from the recorded G1 state (Dex3 qpos/torque + arm joints -> FK palm height).

  reward(frame) = w_grip * grip_closure + w_load * loaded_grip + w_lift * lift_height   (each in [0,1])
  success       = (grip & loaded & lifted) SUSTAINED over the final HOLD_FRAMES

  grip_closure : Dex3 finger qpos magnitude (fingers curled)
  loaded_grip  : Dex3 finger |torque| -- a hand holding the object pulls more current/torque than empty air
  lift_height  : palm z above the table, via pinocchio FK from the arm joints (torso frame)

  conda activate gr00t
  python eval/pick_reward.py --episodes 5      # sanity: reward should rise to ~1 by the end of each demo
"""
import argparse
import glob
import json

import numpy as np
import pinocchio as pin

URDF = "$DATA_ROOT/robots/g1/urdf/g1_body29_hand14.urdf"
ARMJ = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="$DATA_ROOT/teleop_raw/cylinder_pick_clean")
p.add_argument("--episodes", type=int, default=5)
p.add_argument("--hold-frames", type=int, default=15, help="sustained window at the end for success")
p.add_argument("--w-grip", type=float, default=0.3)
p.add_argument("--w-load", type=float, default=0.3)
p.add_argument("--w-lift", type=float, default=0.4)
# success thresholds (fractions of each term's per-dataset scale; tune once on real data)
p.add_argument("--grip-thr", type=float, default=0.5)
p.add_argument("--load-thr", type=float, default=0.4)
p.add_argument("--lift-thr", type=float, default=0.05, help="palm rise above episode-start height (m)")
A = p.parse_args()

_m = pin.buildModelFromUrdf(URDF); _d = _m.createData()
_lqi = [_m.idx_qs[_m.getJointId(f"left_{j}_joint")] for j in ARMJ]
_rqi = [_m.idx_qs[_m.getJointId(f"right_{j}_joint")] for j in ARMJ]
_torso = _m.getFrameId("torso_link")
_palmL, _palmR = _m.getFrameId("left_hand_palm_link"), _m.getFrameId("right_hand_palm_link")


def palm_heights(la, ra):
    """avg palm z (L,R) in torso frame for one frame's arm joints."""
    q = pin.neutral(_m)
    for i, qi in enumerate(_lqi): q[qi] = la[i]
    for i, qi in enumerate(_rqi): q[qi] = ra[i]
    pin.forwardKinematics(_m, _d, q); pin.updateFramePlacements(_m, _d)
    tinv = _d.oMf[_torso].inverse()
    return float((tinv * _d.oMf[_palmL]).translation[2]), float((tinv * _d.oMf[_palmR]).translation[2])


def episode_terms(path):
    fr = json.load(open(path))["data"]
    grip, load, ztop = [], [], []
    for f in fr:
        s = f["states"]
        grip.append((np.linalg.norm(s["left_ee"]["qpos"]) + np.linalg.norm(s["right_ee"]["qpos"])) / 2)
        load.append((np.linalg.norm(s["left_ee"]["torque"]) + np.linalg.norm(s["right_ee"]["torque"])) / 2)
        zl, zr = palm_heights(s["left_arm"]["qpos"], s["right_arm"]["qpos"])
        ztop.append((zl + zr) / 2)
    return np.array(grip), np.array(load), np.array(ztop)


def normalize(x):  # -> [0,1] within the episode (min-max); replace with dataset-global scale for RL
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def main():
    eps = sorted(glob.glob(f"{A.dataset}/episode_*"))[: A.episodes]
    print(f"{'episode':22s} {'reward: start->end':>22s}  {'lift(m)':>8s}  success")
    for e in eps:
        grip, load, ztop = episode_terms(f"{e}/data.json")
        lift = ztop - ztop[0]                              # rise above episode start
        r = A.w_grip * normalize(grip) + A.w_load * normalize(load) + A.w_lift * normalize(np.maximum(lift, 0))
        hold = r[-A.hold_frames:]
        # success = grip closed + loaded + lifted, sustained at the end
        succ = (normalize(grip)[-A.hold_frames:].mean() > A.grip_thr and
                normalize(load)[-A.hold_frames:].mean() > A.load_thr and
                lift[-A.hold_frames:].mean() > A.lift_thr)
        print(f"{e.split('/')[-1]:22s} {r[0]:.2f} -> {r[-1]:.2f} (hold {hold.mean():.2f})  "
              f"{lift[-A.hold_frames:].mean():+.3f}  {'YES' if succ else 'no'}")


if __name__ == "__main__":
    main()
