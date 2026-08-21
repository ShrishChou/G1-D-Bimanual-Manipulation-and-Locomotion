#!/usr/bin/env python3
"""Live front-distance stream from the 2D LiDAR. Drive the robot toward the table and watch the distance
drop -- wherever it stops reporting (jumps to '--' / no returns) is the true minimum range. Ctrl-C to quit.

  python nav/front_dist.py            # +/-15 deg front cone
  python nav/front_dist.py --cone 8   # narrower cone (more 'straight ahead')

Shows: nearest & median in the front cone, the overall nearest return, and the declared range_min.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy"))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from laserscan_idl import LaserScan_

FRONT_DEG = 0.0     # LiDAR bearing pointing robot-forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cone", type=float, default=15.0, help="half-width of the front cone (deg)")
    a = ap.parse_args()
    ChannelFactoryInitialize(0, "enp2s0")
    sub = ChannelSubscriber("rt/slamware_ros_sdk_server_node/scan", LaserScan_); sub.Init()
    print(f"[front_dist] LIVE  (front cone +/-{a.cone:.0f} deg). Drive toward the table; watch the nearest value. Ctrl-C to quit.",
          flush=True)
    try:
        while True:
            m = None; t0 = time.time()
            while m is None and time.time() - t0 < 2:
                m = sub.Read(); time.sleep(0.02)
            if m is None:
                print("\r[front_dist] no scan ...            ", end="", flush=True); continue
            r = np.array(m.ranges, float)
            ang = m.angle_min + np.arange(len(r)) * m.angle_increment
            off = np.abs((ang - math.radians(FRONT_DEG) + math.pi) % (2 * math.pi) - math.pi)
            valid = np.isfinite(r) & (r > 0) & (r < m.range_max)
            overall = float(np.min(r[valid])) if valid.any() else float("nan")
            cone = valid & (off <= math.radians(a.cone))
            if cone.any():
                near = float(np.min(r[cone])); med = float(np.median(r[cone])); n = int(cone.sum())
                print(f"\r[front_dist] FRONT nearest={near:.3f}m  median={med:.3f}m  ({n} pts)   "
                      f"overall-nearest={overall:.3f}m  range_min={m.range_min:.3f}m     ", end="", flush=True)
            else:
                print(f"\r[front_dist] FRONT: -- no returns in cone --   overall-nearest={overall:.3f}m  "
                      f"range_min={m.range_min:.3f}m     ", end="", flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\n[front_dist] done", flush=True)


if __name__ == "__main__":
    main()
