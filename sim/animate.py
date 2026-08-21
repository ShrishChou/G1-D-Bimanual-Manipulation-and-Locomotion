"""Kinematic pick-and-place animation of the G1 in the graduated scene (no physics, no policy -- poses are
scripted and interpolated, the cylinder is attached to the hand by forward-kinematics). Randomizes the Amazon
package placement each run, plans an A* path around them, and renders the robot picking the cylinder at the front
table, carrying it along the path, and placing it in the container hole. Writes one MP4 per run.

    MUJOCO_GL=glfw python animate.py --n 5 --outdir /tmp/anims
"""
import argparse
import os
import sys

import numpy as np
import mujoco
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S

# resolved per-model by name -- see sim/grasp_ik.py (this model interleaves arm, hand, arm, hand)
_KIN = None


def _kin(model):
    global _KIN
    if _KIN is None or _KIN.m is not model:
        from grasp_ik import Kin
        _KIN = Kin(model)
    return _KIN
HAND_BODY = "right_wrist_yaw_link"
ARM_HOME = np.zeros(7)
ARM_REACH_R = np.array([-0.55, -0.20, 0.0, 1.05, 0, 0, 0])   # shoulder fwd + elbow bend -> hand in front
ARM_REACH_L = np.array([-0.55, 0.20, 0.0, 1.05, 0, 0, 0])


def yaw_quat(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def lerp(a, b, t):
    return np.asarray(a) + (np.asarray(b) - np.asarray(a)) * t


def smooth(t):
    return t * t * (3 - 2 * t)


def set_robot(d, base_xy, yaw, armL, armR):
    d.qpos[0:3] = [base_xy[0], base_xy[1], 0.793]
    d.qpos[3:7] = yaw_quat(yaw)
    k = _kin(m)
    d.qpos[k.q_armL] = armL
    d.qpos[k.q_armR] = armR


def set_cylinder(d, gadr, pos, quat=S.CYL_STAND_QUAT):
    d.qpos[gadr:gadr + 3] = pos
    d.qpos[gadr + 3:gadr + 7] = quat


def hand_pos(m, d):
    return np.array(d.body(HAND_BODY).xpos)


def carry_from_hand(m, d):
    """cylinder pose that looks held: centered on the hand, upright, mid at hand height."""
    h = hand_pos(m, d)
    return [h[0], h[1], h[2] - S.CYL_H / 2]


def resample_path(pts, n):
    """Evenly resample a polyline (list of (x,y)) into n points; returns points + per-point yaw (tangent)."""
    pts = np.array(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n)
    xs = np.interp(s, cum, pts[:, 0]); ys = np.interp(s, cum, pts[:, 1])
    out = np.stack([xs, ys], 1)
    yaws = []
    for i in range(n):
        j = min(i + 1, n - 1); k = max(i - 1, 0)
        yaws.append(np.arctan2(out[j, 1] - out[k, 1], out[j, 0] - out[k, 0]))
    return out, yaws


def make_frames(m, d, gadr, path_world):
    """Yield (base_xy, yaw, armL, armR, cylinder_pose|'hand') across the pick->carry->place phases."""
    cylinder_table = [*S.CYL_XY, S.TOP]
    cylinder_hole = [S.CONT_XY[0], S.CONT_XY[1], S.TOP + S.STAND_H]
    frames = []
    # 1) reach toward the cylinder (facing +x), cylinder on the table
    for i in range(26):
        t = smooth(i / 25)
        frames.append(((0, 0), 0.0, lerp(ARM_HOME, ARM_REACH_L, t), lerp(ARM_HOME, ARM_REACH_R, t), cylinder_table))
    # 2) grasp: cylinder rises from table to the hand
    set_robot(d, (0, 0), 0.0, ARM_REACH_L, ARM_REACH_R); mujoco.mj_forward(m, d)
    hp = carry_from_hand(m, d)
    for i in range(16):
        t = smooth(i / 15)
        frames.append(((0, 0), 0.0, ARM_REACH_L, ARM_REACH_R, lerp(cylinder_table, hp, t)))
    # 3) turn in place to face the path start
    y0 = resample_path(path_world, 2)[1][0]
    for i in range(18):
        t = smooth(i / 17)
        frames.append(((0, 0), t * y0, ARM_REACH_L, ARM_REACH_R, "hand"))
    # 4) traverse the path, facing the tangent, carrying the cylinder
    pts, yaws = resample_path(path_world, 55)
    for (x, y), yw in zip(pts, yaws):
        frames.append(((x, y), yw, ARM_REACH_L, ARM_REACH_R, "hand"))
    # 5) place: face the container (+y) and lower the cylinder into the hole
    for i in range(22):
        t = smooth(i / 21)
        yw = lerp(yaws[-1], np.pi / 2, t)
        frames.append((S.GOAL, float(yw), ARM_REACH_L, ARM_REACH_R, "PLACE:" + str(t)))
    # 6) retract the arms; cylinder rests in the hole
    for i in range(16):
        t = smooth(i / 15)
        frames.append((S.GOAL, np.pi / 2, lerp(ARM_REACH_L, ARM_HOME, t), lerp(ARM_REACH_R, ARM_HOME, t), cylinder_hole))
    return frames, hp, cylinder_hole


def render_run(seed, outdir, cam):
    rng = np.random.default_rng(seed)
    packages = S.random_packages(rng)
    path = S.plan_path(packages)
    grid = S.occupancy(packages)
    path_world = [grid.c2w(r, c) for r, c in path]
    m, info = S.build(packages, path)
    d = mujoco.MjData(m)
    gadr = info["cyl_qadr"]
    r = mujoco.Renderer(m, height=720, width=1280)
    frames, hp, cylinder_hole = make_frames(m, d, gadr, path_world)

    out = os.path.join(outdir, f"pickplace_{seed}.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for base_xy, yaw, armL, armR, gp in frames:
        set_robot(d, base_xy, yaw, armL, armR)
        mujoco.mj_forward(m, d)
        if isinstance(gp, str) and gp == "hand":
            set_cylinder(d, gadr, carry_from_hand(m, d))
        elif isinstance(gp, str) and gp.startswith("PLACE:"):
            t = float(gp.split(":")[1])
            set_cylinder(d, gadr, lerp(carry_from_hand(m, d), cylinder_hole, t))
        else:
            set_cylinder(d, gadr, gp)
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam)
        vw.write(r.render()[:, :, ::-1])
    vw.release()
    print(f"[anim] seed {seed}: {len(packages)} packages, {len(path)}-wp path -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--outdir", default="/tmp/anims")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 1.0, 0.3]; cam.distance = 5.8; cam.azimuth = 215; cam.elevation = -35
    for s in range(args.n):
        render_run(s, args.outdir, cam)
    print("[anim] done")
