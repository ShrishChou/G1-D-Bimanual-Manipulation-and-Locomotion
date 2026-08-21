#!/usr/bin/env python3
"""Capture the robot's CURRENT pose as the charger DOCK, and generate the 3 homing waypoints (global
frame, absolute) into nav/waypoints.json. Run this WHILE the robot is sitting docked at the charger.

Docking geometry (dock = current pose, heading H = dock yaw):
  home_approach : --front m IN FRONT of the dock (+H direction), facing H+180 ("the other way")
  home_rotated  : same position, facing H            (rotate 180 in place)
  home_dock     : the dock pose itself               (reverse --front m into the charger)

Homing sequence (each drift-free via nav_move):
  1) nav_move --name home_approach --go              # drive to the approach point
  2) nav_move --name home_rotated  --go              # rotate 180 in place
  3) nav_move --name home_rotated  --go --back-after <front>   # reverse into the dock

  python dock_waypoints.py            # capture + generate (default 10cm)
  python dock_waypoints.py --front 0.10
"""
import argparse
import json
import math
import os
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_

WP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoints.json")


def yaw_of(o):
    return math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp2s0")
    ap.add_argument("--front", type=float, default=0.10, help="approach/back-up distance in front of the dock (m)")
    a = ap.parse_args()

    ChannelFactoryInitialize(0, a.iface)
    sub = ChannelSubscriber("rt/robot_pose", PoseStamped_); sub.Init()
    S = None; t0 = time.time()
    while S is None and time.time() - t0 < 5:
        m = sub.Read()
        if m is not None:
            S = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))
        time.sleep(0.02)
    if S is None:
        print("[dock_waypoints] no rt/robot_pose -- is the robot connected?"); return
    dx, dy, dyaw = S
    fx, fy = math.cos(dyaw), math.sin(dyaw)                      # dock's forward (into the room)

    h180 = math.atan2(math.sin(dyaw + math.pi), math.cos(dyaw + math.pi))              # H+180 wrapped
    approach = {"x": dx + a.front * fx, "y": dy + a.front * fy, "yaw": h180}            # in front, facing away
    rotated = {"x": dx + a.front * fx, "y": dy + a.front * fy, "yaw": dyaw}             # same pos, facing H
    dock = {"x": dx, "y": dy, "yaw": dyaw}

    wps = json.load(open(WP_FILE)) if os.path.exists(WP_FILE) else {}
    wps.update({"home_approach": approach, "home_rotated": rotated, "home_dock": dock})
    json.dump(wps, open(WP_FILE, "w"), indent=2)

    print(f"[dock_waypoints] DOCK captured (current pose): ({dx:.3f}, {dy:.3f}, {math.degrees(dyaw):.1f} deg)")
    for k in ["home_approach", "home_rotated", "home_dock"]:
        w = wps[k]
        print(f"    {k:14s} x={w['x']:7.3f} y={w['y']:7.3f} yaw={math.degrees(w['yaw']):7.1f} deg")
    print("\nHoming sequence (drift-free):")
    print("  1) nav_move.py --name home_approach --go")
    print("  2) nav_move.py --name home_rotated  --go")
    print(f"  3) nav_move.py --name home_rotated  --go --back-after {a.front:.2f}")


if __name__ == "__main__":
    main()
