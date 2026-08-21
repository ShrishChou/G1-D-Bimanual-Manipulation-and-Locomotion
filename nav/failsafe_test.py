"""Standalone failsafe test -- NO ROBOT MOTION. Subscribes to the LiDAR (and optionally the head camera) and
prints, live, whether the failsafe WOULD trip: any object inside the front danger sector (LiDAR) or the lens
suddenly occluded (camera). Wave a hand in front to confirm both detectors fire BEFORE you trust them during
movement. Ctrl-C to quit.

    conda activate <env>
    python failsafe_test.py --iface enp2s0            # LiDAR only
    python failsafe_test.py --iface enp2s0 --camera   # + camera occlusion check (needs teleimager_bridge)
"""
import argparse
import math
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

LIDAR_TOPIC = "rt/utlidar/cloud"   # TODO(robot): verify the actual G1-D LiDAR topic
DANGER_R = 0.45                    # m
FRONT_DEG = 60                    # +/- half-angle of the front sector
Z_MIN, Z_MAX = 0.10, 1.6

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="enp2s0")
ap.add_argument("--camera", action="store_true", help="also test the camera-occlusion failsafe")
a = ap.parse_args()

ChannelFactoryInitialize(0, a.iface)
cloud = {"pts": np.zeros((0, 3))}
sub = ChannelSubscriber(LIDAR_TOPIC, PointCloud2_); sub.Init()


def _lidar():
    while True:
        m = sub.Read()
        if m is not None:
            try:
                buf = np.frombuffer(bytes(m.data), np.uint8).reshape(-1, m.point_step)
                xyz = np.stack([buf[:, o:o + 4].copy().view(np.float32).ravel() for o in (0, 4, 8)], 1)
                cloud["pts"] = xyz[np.isfinite(xyz).all(1)]
            except Exception:
                pass
        time.sleep(0.02)


threading.Thread(target=_lidar, daemon=True).start()

cam_state = {"base": None, "occluded": False}
if a.camera:
    from multiprocessing import shared_memory
    from teleop.image_server.image_client import ImageClient   # noqa: E402
    HEAD = (480, 640, 3)
    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(HEAD)))
    img = np.ndarray(HEAD, np.uint8, buffer=shm.buf)
    ic = ImageClient(tv_img_shape=HEAD, tv_img_shm_name=shm.name, server_address="127.0.0.1")
    threading.Thread(target=ic.receive_process, daemon=True).start()

    def _cam():
        while True:
            b = float(img.mean())
            if cam_state["base"] is None and b > 5:
                cam_state["base"] = b
            # lens occluded (object right in front) -> brightness collapses vs the running baseline
            cam_state["occluded"] = cam_state["base"] is not None and b < 0.35 * cam_state["base"]
            time.sleep(0.1)
    threading.Thread(target=_cam, daemon=True).start()


def lidar_trip():
    p = cloud["pts"]
    if not len(p):
        return False, 99.0
    p = p[(p[:, 2] > Z_MIN) & (p[:, 2] < Z_MAX)]
    if not len(p):
        return False, 99.0
    d = np.hypot(p[:, 0], p[:, 1])
    ang = np.abs(np.arctan2(p[:, 1], p[:, 0]))          # front = +x in the LiDAR frame
    front = (ang < math.radians(FRONT_DEG))
    near = d[front].min() if front.any() else 99.0
    return near < DANGER_R, float(near)


print(f"[failsafe_test] LiDAR topic {LIDAR_TOPIC}; danger < {DANGER_R} m in +/-{FRONT_DEG} deg. Ctrl-C to quit.", flush=True)
try:
    while True:
        trip, near = lidar_trip()
        cam = " CAM-OCCLUDED" if cam_state["occluded"] else ""
        estop = trip or cam_state["occluded"]
        print(f"\rLiDAR nearest-front {near:5.2f} m | {'>>> WOULD E-STOP <<<' if estop else 'safe            '}"
              f"{cam}   ", end="", flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n[failsafe_test] done", flush=True)
