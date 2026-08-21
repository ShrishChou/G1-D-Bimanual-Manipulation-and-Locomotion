#!/usr/bin/env python3
"""Command Unitree's built-in 3D nav (Nav2 under the hood) — the full obstacle-avoiding
planner that uses the Livox/Hesai 3D LiDAR. ONLY works when the robot is in normal/sport
mode with the nav stack launched (NOT in ai/manipulation mode).

Goal path A (native): publish custom_interface/NavigationToPose -> "nav_to_pose",
  watch custom_interface/NavState on "nav_state" (nav_true/nav_false/nav_interrupt).
Goal path B (fallback): publish geometry_msgs/PoseStamped -> "/goal_pose" (Nav2 default).

Env first:
  source /opt/ros/humble/setup.bash
  source $SLAM_WS/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
  export CYCLONEDDS_URI=file://$REPO/deploy/cyclonedds.xml

  python3 unitree_nav.py                          # DRY-RUN: report whether the nav stack is up (no goal)
  python3 unitree_nav.py --x 2.0 --y 1.0 --go     # drive there (map frame) via nav_to_pose
"""
import argparse
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from custom_interface.msg import NavigationToPose, NavState

NAV_TO_POSE = "nav_to_pose"
NAV_STATE = "nav_state"
GOAL_POSE = "/goal_pose"


class UnitreeNav(Node):
    def __init__(self):
        super().__init__("unitree_nav")
        self.pub = self.create_publisher(NavigationToPose, NAV_TO_POSE, 10)
        self.goal_pub = self.create_publisher(PoseStamped, GOAL_POSE, 10)
        self.state = None
        self.create_subscription(NavState, NAV_STATE, self._state_cb, 10)

    def _state_cb(self, m):
        self.state = (m.id, m.state)

    def spin(self, secs):
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.1)

    def send(self, x, y, yaw, use_goal_pose):
        if use_goal_pose:
            g = PoseStamped()
            g.header.frame_id = "map"
            g.header.stamp = self.get_clock().now().to_msg()
            g.pose.position.x, g.pose.position.y = float(x), float(y)
            g.pose.orientation.z = math.sin(yaw / 2.0)
            g.pose.orientation.w = math.cos(yaw / 2.0)
            self.goal_pub.publish(g)
        else:
            m = NavigationToPose()
            m.id = "g1d_%d" % int(time.time())
            m.pose_x, m.pose_y, m.pose_z = float(x), float(y), 0.0
            m.quat_x = m.quat_y = 0.0
            m.quat_z = math.sin(yaw / 2.0)
            m.quat_w = math.cos(yaw / 2.0)
            self.pub.publish(m)
            return m.id
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float)
    ap.add_argument("--y", type=float)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--goal-pose", action="store_true", help="use Nav2 /goal_pose instead of nav_to_pose")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--go", action="store_true", help="ACTUALLY send the goal (robot MOVES). Omit = dry-run.")
    a = ap.parse_args()

    rclpy.init()
    n = UnitreeNav()
    n.spin(2.0)
    nav_subs = n.pub.get_subscription_count()
    goal_subs = n.goal_pub.get_subscription_count()
    print(f"[unitree_nav] nav_to_pose subscribers: {nav_subs}   /goal_pose subscribers: {goal_subs}", flush=True)
    up = nav_subs > 0 or goal_subs > 0
    print(f"[unitree_nav] nav stack appears {'UP' if up else 'DOWN (not in sport mode / not launched)'}", flush=True)

    if not a.go or a.x is None or a.y is None:
        print("[unitree_nav] DRY-RUN -- no goal sent. Add --x X --y Y --go to drive.", flush=True)
        rclpy.shutdown(); return
    if not up:
        print("[unitree_nav] ABORT: nav stack not listening.", flush=True)
        rclpy.shutdown(); return

    gid = n.send(a.x, a.y, a.yaw, a.goal_pose)
    print(f"[unitree_nav] goal sent (id={gid}) x={a.x} y={a.y} yaw={a.yaw}", flush=True)
    t0 = time.time()
    try:
        while time.time() - t0 < a.timeout:
            rclpy.spin_once(n, timeout_sec=0.2)
            if n.state and (gid is None or n.state[0] == gid):
                print(f"\n[unitree_nav] nav_state = {n.state[1]}", flush=True)
                if n.state[1] in ("nav_true", "nav_false", "nav_interrupt"):
                    break
            print(f"\r[unitree_nav] navigating... {time.time()-t0:4.0f}s", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[unitree_nav] interrupted", flush=True)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
