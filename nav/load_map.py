#!/usr/bin/env python3
"""Restore a saved slamware map (.stcm) AND localize the robot in it, so the map frame (and all absolute
waypoints) come back after a restart. RUN ON THE ROBOT (Foxy). You must pass the robot's CURRENT pose in
that map -- typically the robot restarts parked at the charger, so pass the saved 'home' pose.

On robot (deactivate conda first -- Foxy needs system python3.8, not the tv env):
  # use /usr/bin/python3 explicitly (avoids the conda tv/base env Python clash)
  source /opt/ros/foxy/setup.bash
  source /unitree/module/slamware_service_pc4/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  /usr/bin/python3 load_map.py /home/unitree/slam_maps/current.stcm <x> <y> <yaw_deg>
"""
import math
import sys

import time

import rclpy
from geometry_msgs.msg import Pose
from slamware_ros_sdk.srv import SyncSetStcm
from slamware_ros_sdk.msg import SetMapUpdateRequest, SetMapLocalizationRequest, MapKind

if len(sys.argv) < 5:
    print("usage: load_map.py <map.stcm> <x> <y> <yaw_deg>"); sys.exit(1)
path = sys.argv[1]
x, y, yaw = float(sys.argv[2]), float(sys.argv[3]), math.radians(float(sys.argv[4]))
data = open(path, "rb").read()

rclpy.init()
n = rclpy.create_node("load_map")
cli = n.create_client(SyncSetStcm, "/sync_set_stcm")
if not cli.wait_for_service(timeout_sec=8.0):
    print("sync_set_stcm not available"); sys.exit(1)
req = SyncSetStcm.Request()
req.raw_stcm = list(data)
p = Pose()
p.position.x = x; p.position.y = y
p.orientation.z = math.sin(yaw / 2.0); p.orientation.w = math.cos(yaw / 2.0)
req.robot_pose = p
fut = cli.call_async(req)
rclpy.spin_until_future_complete(n, fut, timeout_sec=30.0)
if fut.result() is None:
    print("set_stcm failed/timeout"); sys.exit(1)
print(f"restored map from {path} ({len(data)} bytes), localized at ({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg)")

# LOCK the map: localization ON + map-building OFF, so live obstacles (boxes) are NOT written into the map
# and the frame/waypoints stay fixed.
up = n.create_publisher(SetMapUpdateRequest, "/slamware_ros_sdk_server_node/set_map_update", 10)
loc = n.create_publisher(SetMapLocalizationRequest, "/slamware_ros_sdk_server_node/set_map_localization", 10)
time.sleep(0.5)
mu = SetMapUpdateRequest(); mu.enabled = False; mu.kind = MapKind(kind=MapKind.SLAMMAP)
ml = SetMapLocalizationRequest(); ml.enabled = True
for _ in range(5):
    up.publish(mu); loc.publish(ml); time.sleep(0.1)
print("map locked: building OFF, localization ON (boxes will be transient, not added to the map)")
rclpy.shutdown()
