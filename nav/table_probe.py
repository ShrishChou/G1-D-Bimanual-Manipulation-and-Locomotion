#!/usr/bin/env python3
"""Live LiDAR table probe. Drive the robot toward the table and watch whether the front LiDAR can SEE the
table face and at what distance. Use this to (a) confirm the table is visible (>~0.78m) at your approach
waypoint, and (b) pick a --table-stop value. Ctrl-C to quit.

  python nav/table_probe.py

Prints, per update: number of face points, median forward distance to the face, tilt (0 = squared up),
and the nearest return. If it says "table not visible", the table is inside the ~0.77m blind zone or there's
no flat face in the front sector -- back the robot up until it appears.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy"))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from laserscan_idl import LaserScan_

FRONT_DEG = 0.0      # LiDAR bearing that points robot-forward
SECTOR = 40.0        # half-width of the front sector (deg)
MAXR = 3.0


def main():
    ChannelFactoryInitialize(0, "enp2s0")
    sub = ChannelSubscriber("rt/slamware_ros_sdk_server_node/scan", LaserScan_); sub.Init()
    print("[table_probe] LIVE -- drive toward the table, watch the distance. Ctrl-C to quit.", flush=True)
    try:
        while True:
            m = None; t0 = time.time()
            while m is None and time.time() - t0 < 2:
                m = sub.Read(); time.sleep(0.02)
            if m is None:
                print("\r[table_probe] no scan ...", end="", flush=True); continue
            r = np.array(m.ranges, float)
            ang = m.angle_min + np.arange(len(r)) * m.angle_increment
            d = np.abs((ang - math.radians(FRONT_DEG) + math.pi) % (2 * math.pi) - math.pi)
            valid = np.isfinite(r) & (r > 0.1) & (r < m.range_max)   # fixed floor, not range_min (tracks nearest)
            nearest = float(np.min(r[valid])) if valid.any() else float("nan")
            sel = (d <= math.radians(SECTOR)) & valid & (r < MAXR)
            if sel.sum() < 8:
                print(f"\r[table_probe] table NOT visible in front sector (nearest return {nearest:.2f}m, "
                      f"range_min {m.range_min:.2f}m)   ", end="", flush=True)
            else:
                a = ang[sel] - math.radians(FRONT_DEG)
                rr = r[sel]
                x = rr * np.cos(a); y = rr * np.sin(a)
                dist = float(np.median(x))
                yc, xc = y - y.mean(), x - x.mean()
                slope = float((yc * xc).sum() / ((yc * yc).sum() + 1e-9))
                tilt = math.degrees(math.atan(slope))
                print(f"\r[table_probe] face pts={int(sel.sum()):3d}  dist={dist:.3f}m  tilt={tilt:+5.1f}deg  "
                      f"nearest={nearest:.2f}m   ", end="", flush=True)
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n[table_probe] done", flush=True)


if __name__ == "__main__":
    main()
