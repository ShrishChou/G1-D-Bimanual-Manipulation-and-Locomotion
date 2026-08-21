#!/usr/bin/env python3
"""Send a goto goal to the robot's built-in slamware SLAM planner (it plans a path + drives itself).
Goal is in the MAP frame (slamware_map), same frame as /robot_pose.

Env first:
  source /opt/ros/humble/setup.bash
  source $SLAM_WS/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
  export CYCLONEDDS_URI=file://$REPO/deploy/cyclonedds.xml

  python3 nav_goto.py                        # DRY-RUN: print robot pose + whether the robot is listening (no motion)
  python3 nav_goto.py --x 1.0 --y 2.0 --go   # ACTUALLY drive to raw map coords
  python3 nav_goto.py --name pick_table --go # drive to a saved waypoint (see waypoints.py)
Ctrl-C cancels the action (stops the robot).
"""
import argparse
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from slamware_ros_sdk.msg import MoveToRequest, CancelActionRequest

MOVE_TO = "/slamware_ros_sdk_server_node/move_to"
CANCEL = "/slamware_ros_sdk_server_node/cancel_action"
WP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoints.json")


class NavGoto(Node):
    def __init__(self):
        super().__init__("nav_goto")
        self.pub = self.create_publisher(MoveToRequest, MOVE_TO, 10)
        self.cancel_pub = self.create_publisher(CancelActionRequest, CANCEL, 10)
        self.pose = None
        self.create_subscription(PoseStamped, "/robot_pose", self._pose_cb, 10)

    def _pose_cb(self, m):
        self.pose = (m.pose.position.x, m.pose.position.y)

    def wait_pose(self, secs=5.0):
        t0 = time.time()
        while self.pose is None and time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.pose

    def send_goal(self, x, y, yaw):
        g = MoveToRequest()
        g.location = Point(x=float(x), y=float(y), z=0.0)
        g.yaw = float(yaw)
        self.pub.publish(g)

    def cancel(self):
        self.cancel_pub.publish(CancelActionRequest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float)
    ap.add_argument("--y", type=float)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--name", help="saved waypoint name (from waypoints.py); overrides --x/--y/--yaw")
    ap.add_argument("--tol", type=float, default=0.20, help="arrival tolerance (m)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--go", action="store_true", help="ACTUALLY send the goal (robot MOVES). Omit = dry-run.")
    a = ap.parse_args()

    if a.name:
        wps = json.load(open(WP_FILE)) if os.path.exists(WP_FILE) else {}
        if a.name not in wps:
            print(f"[nav_goto] waypoint '{a.name}' not found in {WP_FILE} (have: {list(wps)})")
            return
        a.x, a.y, a.yaw = wps[a.name]["x"], wps[a.name]["y"], wps[a.name]["yaw"]
        print(f"[nav_goto] waypoint '{a.name}' -> x={a.x:.3f} y={a.y:.3f} yaw={math.degrees(a.yaw):.1f} deg")

    rclpy.init()
    n = NavGoto()
    p = n.wait_pose()
    subs = n.pub.get_subscription_count()
    print(f"[nav_goto] robot pose (map frame): {('%.3f, %.3f' % p) if p else 'UNKNOWN'}", flush=True)
    print(f"[nav_goto] move_to subscribers (robot listening / type-compatible): {subs}", flush=True)

    if not a.go or a.x is None or a.y is None:
        print("[nav_goto] DRY-RUN -- no goal sent. Add --x X --y Y --go to drive.", flush=True)
        rclpy.shutdown(); return
    if subs == 0:
        print("[nav_goto] ABORT: nobody subscribed to move_to (robot not listening or msg type mismatch).", flush=True)
        rclpy.shutdown(); return

    print(f"[nav_goto] sending goal x={a.x} y={a.y} yaw={a.yaw} (robot will drive) ...", flush=True)
    try:
        n.send_goal(a.x, a.y, a.yaw)
        t0 = time.time()
        arrived = False
        while time.time() - t0 < a.timeout:
            rclpy.spin_once(n, timeout_sec=0.1)
            if n.pose:
                d = math.hypot(n.pose[0] - a.x, n.pose[1] - a.y)
                print(f"\r[nav_goto] pose {n.pose[0]:6.2f},{n.pose[1]:6.2f}  dist {d:5.2f} m   ", end="", flush=True)
                if d <= a.tol:
                    arrived = True; break
        print("\n[nav_goto] ARRIVED" if arrived else "\n[nav_goto] TIMEOUT", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        n.cancel()
        print("[nav_goto] sent cancel_action (stop)", flush=True)
        rclpy.shutdown()


if __name__ == "__main__":
    main()
