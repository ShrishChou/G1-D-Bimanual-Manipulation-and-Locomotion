#!/usr/bin/env python3
"""Carry-phase navigation for SPORT mode: drive to a goal using Unitree's OWN nav stack
(Livox MID-360 3D LiDAR -> LIO-SAM SLAM -> nav2 costmap + A* global + local planner), which does
the SLAM, global planning, and dynamic obstacle/person avoidance internally. This wrapper:
  - captures/replays goals in the nav 'map' frame (via TF map->base_link),
  - sends the goal (custom_interface/NavigationToPose -> nav_to_pose),
  - monitors nav_state + progress, and implements the WAIT-IF-BLOCKED policy:
      Unitree's local planner already tries to plan around obstacles; if it ultimately fails
      (nav_false = path blocked, e.g. a person standing in the only corridor), we WAIT and RETRY
      until it clears or we time out.

ROS2 side -- source ROS2 + slamware_ws (for custom_interface), set the cyclonedds env, then:
  python3 carry_nav.py capture place_table       # save current pose (nav map frame) as a waypoint
  python3 carry_nav.py list
  python3 carry_nav.py goto place_table          # DRY-RUN: check nav stack + show plan, no motion
  python3 carry_nav.py goto place_table --go     # drive there (Unitree nav avoids obstacles/people)
Ctrl-C cancels (stops the robot).
"""
import argparse
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
import tf2_ros
from custom_interface.msg import NavigationToPose, NavState

WP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unitree_waypoints.json")
MAP_FRAME = "map"
BASE_FRAME = "base_link"


def load():
    return json.load(open(WP_FILE)) if os.path.exists(WP_FILE) else {}


def save(w):
    json.dump(w, open(WP_FILE, "w"), indent=2)


class CarryNav(Node):
    def __init__(self):
        super().__init__("carry_nav")
        self.pub = self.create_publisher(NavigationToPose, "nav_to_pose", 10)
        self.state = None
        self.create_subscription(NavState, "nav_state", self._state_cb, 10)
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)

    def _state_cb(self, m):
        self.state = (m.id, m.state)

    def pose_in_map(self, secs=5.0):
        """robot pose in the nav 'map' frame via TF (works whenever the Unitree nav stack is up)."""
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                tf = self.tf_buf.lookup_transform(MAP_FRAME, BASE_FRAME, rclpy.time.Time())
                t = tf.transform.translation
                q = tf.transform.rotation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                return (t.x, t.y, yaw)
            except Exception:
                continue
        return None

    def send_goal(self, x, y, yaw, gid):
        m = NavigationToPose()
        m.id = gid
        m.pose_x, m.pose_y, m.pose_z = float(x), float(y), 0.0
        m.quat_x = m.quat_y = 0.0
        m.quat_z = math.sin(yaw / 2.0)
        m.quat_w = math.cos(yaw / 2.0)
        self.pub.publish(m)


def cmd_capture(n, name):
    p = n.pose_in_map()
    if p is None:
        print("[carry_nav] no TF map->base_link (is the Unitree nav stack up in sport mode?)"); return
    w = load(); w[name] = {"x": p[0], "y": p[1], "yaw": p[2]}; save(w)
    print(f"[carry_nav] saved '{name}': x={p[0]:.3f} y={p[1]:.3f} yaw={math.degrees(p[2]):.1f}deg")


def cmd_list():
    w = load()
    if not w:
        print("[carry_nav] no waypoints yet ->", WP_FILE); return
    for k, v in w.items():
        print(f"  {k:20s} x={v['x']:7.3f} y={v['y']:7.3f} yaw={math.degrees(v['yaw']):6.1f}deg")


def cmd_goto(n, a):
    if a.name:
        w = load()
        if a.name not in w:
            print(f"[carry_nav] '{a.name}' not found (have: {list(w)})"); return
        tx, ty, tyaw = w[a.name]["x"], w[a.name]["y"], w[a.name]["yaw"]
    else:
        tx, ty, tyaw = a.x, a.y, math.radians(a.yaw)

    subs = n.pub.get_subscription_count()
    p = n.pose_in_map()
    print(f"[carry_nav] nav_to_pose subscribers (Unitree nav up?): {subs}")
    print(f"[carry_nav] current map pose: {('%.2f,%.2f' % (p[0], p[1])) if p else 'UNKNOWN (no TF)'}")
    print(f"[carry_nav] goal: x={tx:.3f} y={ty:.3f} yaw={math.degrees(tyaw):.1f}deg")
    if not a.go:
        print("[carry_nav] DRY-RUN -- no goal sent. Add --go to drive."); return
    if subs == 0:
        print("[carry_nav] ABORT: nav_to_pose has no subscriber (Unitree nav stack not running)."); return

    attempt = 0
    t_start = time.time()
    while time.time() - t_start < a.overall_timeout:
        attempt += 1
        gid = f"carry_{int(time.time())}_{attempt}"
        n.state = None
        n.send_goal(tx, ty, tyaw, gid)
        print(f"[carry_nav] attempt {attempt}: goal sent (id={gid})", flush=True)
        t0 = time.time()
        result = None
        while time.time() - t0 < a.attempt_timeout:
            rclpy.spin_once(n, timeout_sec=0.2)
            p = n.pose_in_map(secs=0.1)
            d = math.hypot(p[0] - tx, p[1] - ty) if p else float("nan")
            if n.state and n.state[0] == gid:
                result = n.state[1]; break
            print(f"\r[carry_nav]   navigating... {time.time()-t0:4.0f}s  dist={d:5.2f}m", end="", flush=True)
        print()
        if result == "nav_true":
            print("[carry_nav] ARRIVED."); return
        if result == "nav_interrupt":
            print("[carry_nav] interrupted; stopping."); return
        # nav_false or timeout -> path blocked (e.g. person). Unitree already tried to plan around;
        # wait for the scene to clear, then retry.
        print(f"[carry_nav] blocked/failed (state={result}); waiting {a.retry_wait:.0f}s for path to clear, then retry...")
        time.sleep(a.retry_wait)
    print("[carry_nav] gave up (overall timeout).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("name")
    sub.add_parser("list")
    g = sub.add_parser("goto")
    g.add_argument("name", nargs="?")
    g.add_argument("--x", type=float); g.add_argument("--y", type=float); g.add_argument("--yaw", type=float, default=0.0)
    g.add_argument("--go", action="store_true", help="ACTUALLY drive (robot moves). Omit = dry-run.")
    g.add_argument("--attempt-timeout", type=float, default=90.0, help="per-goal timeout before treating as blocked")
    g.add_argument("--retry-wait", type=float, default=5.0, help="wait between retries when blocked (person clears)")
    g.add_argument("--overall-timeout", type=float, default=600.0)
    a = ap.parse_args()

    rclpy.init()
    n = CarryNav()
    try:
        if a.cmd == "capture":
            cmd_capture(n, a.name)
        elif a.cmd == "list":
            cmd_list()
        elif a.cmd == "goto":
            if not a.name and (a.x is None or a.y is None):
                print("[carry_nav] give a waypoint name or --x --y"); return
            cmd_goto(n, a)
    except KeyboardInterrupt:
        print("\n[carry_nav] interrupted")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
