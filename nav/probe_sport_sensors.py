#!/usr/bin/env python3
"""Read-only probe of the sport/normal-mode front sensors: 3D LiDAR cloud + obstacle topics
+ nav-stack liveness. Front CAMERA is grabbed separately by grab_front.py (uses unitree_sdk2py
in the `tv` env, not rclpy). Run this AFTER the robot is in normal mode with nav launched.

Env: source ROS2 + slamware_ws, RMW=rmw_cyclonedds_cpp, ROS_LOCALHOST_ONLY=0, CYCLONEDDS_URI=...
  python3 probe_sport_sensors.py
"""
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

CLOUDS = ["/unitree/slam_mapping/points", "/unitree/slam_relocation/points"]


class Probe(Node):
    def __init__(self):
        super().__init__("probe_sport_sensors")
        self.counts = {t: 0 for t in CLOUDS}
        self.last = {t: None for t in CLOUDS}
        for t in CLOUDS:
            self.create_subscription(PointCloud2, t, lambda m, t=t: self._cb(t, m), 10)

    def _cb(self, t, m):
        self.counts[t] += 1
        self.last[t] = (m.width * m.height, m.header.frame_id)


def main():
    rclpy.init()
    n = Probe()
    t0 = time.time()
    while time.time() - t0 < 8:
        rclpy.spin_once(n, timeout_sec=0.1)
    for t in CLOUDS:
        c = n.counts[t]
        info = f"{n.last[t][0]} pts, frame '{n.last[t][1]}'" if n.last[t] else "no data"
        print(f"[probe] {t}: {c} msgs in 8s  ({info})")
    print("[probe] (front camera: run grab_front.py in the tv env)")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
