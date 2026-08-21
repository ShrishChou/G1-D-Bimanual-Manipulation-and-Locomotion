#!/usr/bin/env python3
"""Visualize what the SLAM sees: the occupancy map + the robot pose + the live LiDAR scan, rendered to a
PNG. One-shot by default, or --live to refresh continuously (open the PNG in an auto-refreshing viewer).

ROS2 side -- source ROS2 + slamware_ws + the cyclonedds env, then:
  python3 nav/slam_view.py                     # one snapshot -> nav/slam_view.png
  python3 nav/slam_view.py --live 1.0          # refresh every 1.0s
  python3 nav/slam_view.py --out /tmp/slam.png

Black = walls/occupied, white = free/explored, grey = unknown, blue = robot (arrow = heading),
red dots = live LiDAR returns.
"""
import argparse
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slam_view.png")


class Viewer(Node):
    def __init__(self):
        super().__init__("slam_view")
        q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
        self.grid = None; self.pose = None; self.scan = None
        self.create_subscription(OccupancyGrid, "/slamware_ros_sdk_server_node/map", self._g, q)
        self.create_subscription(PoseStamped, "/robot_pose", self._p, 10)
        self.create_subscription(LaserScan, "/slamware_ros_sdk_server_node/scan", self._s, 10)

    def _g(self, m): self.grid = m
    def _p(self, m): self.pose = m
    def _s(self, m): self.scan = m

    def wait(self, secs=8.0):
        t0 = time.time()
        while (self.grid is None or self.pose is None) and time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.1)


def render(v, out):
    g = v.grid
    a = np.array(g.data, dtype=np.int16).reshape(g.info.height, g.info.width)
    res, ox, oy = g.info.resolution, g.info.origin.position.x, g.info.origin.position.y
    img = np.full((g.info.height, g.info.width, 3), 0.6)   # unknown grey
    img[(a >= 0) & (a <= 50)] = [1, 1, 1]                   # free
    img[a > 50] = [0, 0, 0]                                 # occupied
    extent = [ox, ox + g.info.width * res, oy, oy + g.info.height * res]

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.imshow(img, origin="lower", extent=extent)
    p = v.pose.pose; o = p.orientation
    yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
    px, py = p.position.x, p.position.y
    if v.scan is not None:
        s = v.scan; ang = s.angle_min
        xs, ys = [], []
        for r in s.ranges:
            if s.range_min < r < s.range_max and not math.isinf(r) and not math.isnan(r):
                xs.append(px + r * math.cos(yaw + ang)); ys.append(py + r * math.sin(yaw + ang))
            ang += s.angle_increment
        ax.plot(xs, ys, ".", color="red", ms=1.5)
    ax.plot(px, py, "o", color="blue", ms=9)
    ax.arrow(px, py, 0.5 * math.cos(yaw), 0.5 * math.sin(yaw), head_width=0.2, color="blue")
    ax.set_title(f"SLAM  robot ({px:.2f},{py:.2f}) yaw {math.degrees(yaw):.0f}deg\n"
                 f"map {g.info.width}x{g.info.height} @ {res:.2f}m   red=live scan")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--live", type=float, default=0.0, help="refresh period (s); 0 = one shot")
    a = ap.parse_args()
    rclpy.init()
    v = Viewer()
    try:
        while True:
            v.wait()
            if v.grid is None or v.pose is None:
                print("[slam_view] no map/pose yet (check env / robot)"); break
            render(v, a.out)
            print(f"[slam_view] wrote {a.out}  (robot {v.pose.pose.position.x:.2f},{v.pose.pose.position.y:.2f})", flush=True)
            if a.live <= 0:
                break
            time.sleep(a.live)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
