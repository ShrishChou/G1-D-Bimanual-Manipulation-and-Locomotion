#!/usr/bin/env python3
"""Single-script LIVE LiDAR viewer. Opens a top-down window showing what the 2D LiDAR sees in real time
(robot at the center, forward = up). The front distance is shown in the title so you can drive toward the
table and watch it. Close the window or Ctrl-C to quit.

  python nav/lidar_live.py
  python nav/lidar_live.py --range 4      # axis half-extent (m)

Needs a display (run it on the machine's desktop, not headless ssh). Uses unitree_sdk2py in the `tv` env.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy"))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from laserscan_idl import LaserScan_

FRONT_DEG = 0.0     # LiDAR bearing pointing robot-forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", type=float, default=4.0, help="plot half-extent (m)")
    ap.add_argument("--cone", type=float, default=15.0, help="front cone half-width for the distance readout (deg)")
    a = ap.parse_args()
    ChannelFactoryInitialize(0, "enp2s0")
    sub = ChannelSubscriber("rt/slamware_ros_sdk_server_node/scan", LaserScan_); sub.Init()

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    scat = ax.scatter([], [], s=6, c="red")
    ax.plot(0, 0, "o", color="blue", ms=10)                     # robot
    ax.annotate("", xy=(0, 0.5), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="blue", lw=2))  # forward
    R = a.range
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("left  <-  lateral (m)  ->  right"); ax.set_ylabel("forward (m) up")
    for rr in (0.5, 1.0, 2.0):                                  # range rings
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, color="grey", ls=":", alpha=0.5))
    title = ax.set_title("LiDAR live")

    try:
        while plt.fignum_exists(fig.number):
            m = None; t0 = time.time()
            while m is None and time.time() - t0 < 1:
                m = sub.Read(); time.sleep(0.02)
            if m is None:
                title.set_text("no scan..."); plt.pause(0.05); continue
            r = np.array(m.ranges, float)
            ang = m.angle_min + np.arange(len(r)) * m.angle_increment
            ok = np.isfinite(r) & (r > 0) & (r < m.range_max)
            rr, aa = r[ok], ang[ok]
            # top-down robot frame: forward = up (+y), left = left (-x)
            xs = -rr * np.sin(aa - math.radians(FRONT_DEG))
            ys = rr * np.cos(aa - math.radians(FRONT_DEG))
            scat.set_offsets(np.c_[xs, ys] if len(xs) else np.empty((0, 2)))
            off = np.abs((aa - math.radians(FRONT_DEG) + math.pi) % (2 * math.pi) - math.pi)
            cone = off <= math.radians(a.cone)
            front = float(np.min(rr[cone])) if cone.any() else float("nan")
            title.set_text(f"LiDAR live  |  FRONT nearest = {front:.3f} m  |  {ok.sum()} pts  |  min return {rr.min():.2f}m"
                           if len(rr) else "LiDAR live | no returns")
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    print("[lidar_live] done")


if __name__ == "__main__":
    main()
