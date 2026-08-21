#!/usr/bin/env python3
"""Save the current slamware map to a .stcm file. RUN ON THE ROBOT (Foxy) -- the slamware srv types live
there; the Foxy<->Humble type hash differs so this can't reliably run from the host.

On robot (deactivate conda first -- Foxy needs system python3.8, not the tv env):
  # use /usr/bin/python3 explicitly (avoids the conda tv/base env Python clash)
  source /opt/ros/foxy/setup.bash
  source /unitree/module/slamware_service_pc4/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  /usr/bin/python3 save_map.py /home/unitree/slam_maps/current.stcm
"""
import os
import sys

import rclpy
from slamware_ros_sdk.srv import SyncGetStcm

out = sys.argv[1] if len(sys.argv) > 1 else "/home/unitree/slam_maps/current.stcm"
os.makedirs(os.path.dirname(out), exist_ok=True)
rclpy.init()
n = rclpy.create_node("save_map")
cli = n.create_client(SyncGetStcm, "/sync_get_stcm")
if not cli.wait_for_service(timeout_sec=8.0):
    print("sync_get_stcm not available"); sys.exit(1)
fut = cli.call_async(SyncGetStcm.Request())
rclpy.spin_until_future_complete(n, fut, timeout_sec=30.0)
if fut.result() is None:
    print("get_stcm failed/timeout"); sys.exit(1)
data = bytes(fut.result().raw_stcm)
open(out, "wb").write(data)
print(f"saved {len(data)} bytes -> {out}")
rclpy.shutdown()
