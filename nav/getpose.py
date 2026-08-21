#!/usr/bin/env python3
"""Live pose collector. Run it once, drive the robot around, and hit SPACE at each waypoint to capture
the robot's absolute SLAM-map pose. Each capture is printed (copy it into chat) AND appended to
nav/collected_poses.txt. Press q or Ctrl-C to quit.

  python nav/getpose.py

The base is a ground robot on a flat floor (2D SLAM), so the full pose is x, y, yaw (z is always 0).
"""
import math
import os
import select
import sys
import termios
import threading
import time
import tty

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_

OUTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collected_poses.txt")
_pose = {"v": None}


def yaw_of(o):
    return math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))


def main():
    ChannelFactoryInitialize(0, "enp2s0")
    sub = ChannelSubscriber("rt/robot_pose", PoseStamped_); sub.Init()

    def reader():
        while True:
            m = sub.Read()
            if m is not None:
                p = m.pose.position
                _pose["v"] = (float(p.x), float(p.y), float(yaw_of(m.pose.orientation)))
            time.sleep(0.02)
    threading.Thread(target=reader, daemon=True).start()

    t0 = time.time()
    while _pose["v"] is None and time.time() - t0 < 5:
        time.sleep(0.05)
    if _pose["v"] is None:
        print("[getpose] no rt/robot_pose -- is the robot connected / in data-acquisition mode?"); return

    print("[getpose] LIVE. Drive the robot, hit SPACE to capture a waypoint, q to quit.", flush=True)
    print(f"[getpose] captures also appended to {OUTFILE}", flush=True)
    n = 0
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        last = 0.0
        while True:
            x, y, yaw = _pose["v"]
            if time.time() - last > 0.1:
                last = time.time()
                print(f"\r[getpose] live  x={x:+.4f}  y={y:+.4f}  yaw={math.degrees(yaw):+7.2f} deg   (SPACE=capture, q=quit)   ",
                      end="", flush=True)
            k = sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None
            if k == " ":
                n += 1
                line = f"wp{n}: x={x:.4f}  y={y:.4f}  yaw={math.degrees(yaw):.2f}"
                print(f"\n  >>> CAPTURED {line}", flush=True)
                with open(OUTFILE, "a") as f:
                    f.write(line + "\n")
            elif k in ("q", "\x1b"):
                break
            time.sleep(0.02)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print(f"\n[getpose] {n} pose(s) saved to {OUTFILE}", flush=True)


if __name__ == "__main__":
    main()
