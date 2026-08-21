#!/usr/bin/env python3
"""Teach-and-repeat waypoint capture for the robot's slamware SLAM.

Drive the robot (with the controller) to a spot, then snapshot its current /robot_pose
into a named list. nav_goto.py --name <name> later drives back to it. Read-only (no motion).

Env first (same as nav_goto.py):
  source /opt/ros/humble/setup.bash
  source $SLAM_WS/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
  export CYCLONEDDS_URI=file://$REPO/deploy/cyclonedds.xml

  python3 waypoints.py save pick_table   # snapshot current pose as "pick_table"
  python3 waypoints.py list              # show saved waypoints
  python3 waypoints.py del pick_table    # remove one
  python3 waypoints.py live              # stream pose (drive around, watch coords)
"""
import argparse
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

WP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoints.json")


def load():
    if os.path.exists(WP_FILE):
        return json.load(open(WP_FILE))
    return {}


def save(wps):
    json.dump(wps, open(WP_FILE, "w"), indent=2)


class PoseReader(Node):
    def __init__(self):
        super().__init__("waypoints")
        self.pose = None
        self.create_subscription(PoseStamped, "/robot_pose", self._cb, 10)

    def _cb(self, m):
        yaw = 2 * math.atan2(m.pose.orientation.z, m.pose.orientation.w)
        self.pose = (m.pose.position.x, m.pose.position.y, yaw)

    def wait(self, secs=5.0):
        t0 = time.time()
        while self.pose is None and time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.pose


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save"); s.add_argument("name")
    sub.add_parser("list")
    d = sub.add_parser("del"); d.add_argument("name")
    sub.add_parser("live")
    a = ap.parse_args()

    if a.cmd == "list":
        wps = load()
        if not wps:
            print("[waypoints] none saved yet ->", WP_FILE); return
        print(f"[waypoints] {len(wps)} saved in {WP_FILE}:")
        for name, w in wps.items():
            print(f"  {name:20s} x={w['x']:7.3f} y={w['y']:7.3f} yaw={math.degrees(w['yaw']):6.1f} deg")
        return
    if a.cmd == "del":
        wps = load()
        if a.name in wps:
            del wps[a.name]; save(wps); print(f"[waypoints] deleted '{a.name}'")
        else:
            print(f"[waypoints] '{a.name}' not found")
        return

    rclpy.init()
    n = PoseReader()
    p = n.wait()
    if p is None:
        print("[waypoints] no /robot_pose (check env / robot connection)"); rclpy.shutdown(); return

    if a.cmd == "live":
        print("[waypoints] streaming pose (Ctrl-C to stop) -- drive with the controller:")
        try:
            while True:
                rclpy.spin_once(n, timeout_sec=0.2)
                x, y, yaw = n.pose
                print(f"\r  x={x:7.3f}  y={y:7.3f}  yaw={math.degrees(yaw):6.1f} deg   ", end="", flush=True)
        except KeyboardInterrupt:
            print()
        finally:
            rclpy.shutdown()
        return

    # save
    x, y, yaw = n.pose
    wps = load()
    overwrite = a.name in wps
    wps[a.name] = {"x": x, "y": y, "yaw": yaw}
    save(wps)
    print(f"[waypoints] {'overwrote' if overwrite else 'saved'} '{a.name}': "
          f"x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f} deg -> {WP_FILE}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
