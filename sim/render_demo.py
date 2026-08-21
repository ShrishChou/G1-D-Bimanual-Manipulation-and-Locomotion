"""Render the README media for the mobile pick-place pipeline, entirely offline.

Runs the SAME `nav_fsm.run_fsm` supervised-autonomy loop that ships to the robot, against the MuJoCo
digital twin (`SimIO`), and writes an mp4 plus a small inline GIF for each scenario:

    nominal    pick -> plan (A*) -> accept -> non-holonomic move -> place
    failsafe   identical, but an unplanned object sits in the corridor; the LiDAR SafetyMonitor
               must E-STOP the base before reaching it (the run deliberately does NOT place)

Nothing here needs a GPU, a robot, a policy checkpoint or a dataset -- only the public G1 MJCF.

    python sim/render_demo.py --out docs/videos
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nav"))
sys.path.insert(0, HERE)

import nav_fsm as F          # noqa: E402
import nav_planner as N      # noqa: E402
import scene as S            # noqa: E402


def demo_camera():
    """Closer three-quarter view than the FSM's default debug camera, framed on the corridor."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.15, 0.85, 0.55]
    cam.distance = 3.9
    cam.azimuth = 208
    cam.elevation = -22
    return cam


def to_gif(mp4, gif, width=480, fps=12, speed=1.7, max_seconds=None):
    """Small looping GIF for inline README display.

    GitHub will not play an mp4 that is committed to the repo, so the READMEs embed a GIF and link the
    mp4 for full quality. `speed` drops source frames to compress the timeline, which is what actually
    keeps the file small enough to sit inline.
    """
    import imageio.v2 as imageio

    cap = cv2.VideoCapture(mp4)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps * speed / fps)))
    limit = int(max_seconds * src_fps) if max_seconds else None
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok or (limit and i > limit):
            break
        if i % stride == 0:
            h = int(fr.shape[0] * width / fr.shape[1])
            frames.append(cv2.cvtColor(cv2.resize(fr, (width, h)), cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    # quantize to a small palette -- keeps the GIF inline-friendly on a README
    imageio.mimsave(gif, frames, duration=1.0 / fps, loop=0, palettesize=64, subrectangles=True)
    return len(frames), os.path.getsize(gif)


def run(scenario, out_dir, seed=0):
    intruder = (0.0, 0.95) if scenario == "failsafe" else None
    F.ESTOP["tripped"], F.ESTOP["reason"] = False, ""
    rng = np.random.default_rng(seed)
    packages = S.random_packages(rng)
    mp4 = os.path.join(out_dir, f"twin_{scenario}.mp4")
    io = F.SimIO(packages, demo_camera(), video_path=mp4, intruder=intruder)
    try:
        ok = F.run_fsm(io, S.GOAL, auto=True)
    finally:
        io.close()
    tripped = F.ESTOP["tripped"]
    gif = os.path.join(out_dir, f"twin_{scenario}.gif")
    nf, size = to_gif(mp4, gif)
    print(f"[{scenario}] completed_place={ok} estop={tripped or '-'} "
          f"mp4={os.path.getsize(mp4)/1e6:.1f}MB gif={size/1e6:.1f}MB ({nf} frames)")
    return ok, tripped


def plan_png(out_dir, seed=0):
    """The top-down view the operator actually reviews at the ACCEPT gate."""
    rng = np.random.default_rng(seed)
    packages = S.random_packages(rng)
    grid = S.occupancy(packages)
    N.ROBOT_RADIUS = S.ROBOT_RADIUS
    path = N.plan(grid, S.START, S.GOAL)
    img = N.render_topdown(grid, S.START, 0.0, S.GOAL, path)
    # crop to the occupied region (the 8 m grid is mostly empty) + keep the prompt banner
    ys, xs = np.where(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 55)
    if len(xs):
        pad = 40
        x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
        img = img[y0:y1, x0:x1]
    p = os.path.join(out_dir, "plan_topdown.png")
    cv2.imwrite(p, img)
    print(f"[plan] {p}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "videos"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", default="nominal,failsafe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.scenarios.split(","):
        run(s.strip(), a.out, a.seed)
    plan_png(a.out, a.seed)
