"""G1-D AprilTag navigation: back up -> rotate right 90 deg -> find the tag -> drive straight to it.

Runs AFTER the pick skill (deploy/run_skill.py). Uses the robot's built-in walking controller via LocoClient
(velocity commands; the robot keeps its own balance), odometry from SportModeState_ (position + yaw) for an
accurate backup distance and 90 deg turn, and the head ZED camera (ZMQ from Teleimager) + cv2.aruco for
marker detection. The final approach is visual servoing: center the tag (yaw) and drive forward until it
is close -- a straight line as long as the tag stays centered.

FIRST, verify the camera + tag detection WITHOUT moving:
    conda activate <env>
    python deploy/nav_apriltag.py --test-cam --tag-id 0
Then the full routine (robot standing, loco service active):
    python deploy/nav_apriltag.py --tag-id 0 --back-dist 0.5 --turn-deg -90

NEEDS ON-ROBOT VERIFICATION (marked TODO): the SportModeState topic name (--state-topic), LocoClient Move
sign/units, and the stop-distance proxy (tag pixel size). Tune with --test-cam first.
"""
import argparse
import math
import sys
import threading
import time

import cv2
import numpy as np
import zmq

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="enp2s0")
ap.add_argument("--robot-host", default="<WORKSTATION_IP>", help="Teleimager host (head cam ZMQ)")
ap.add_argument("--head-port", type=int, default=55555)
ap.add_argument("--tag-id", type=int, default=0, help="marker id to navigate to")
ap.add_argument("--dict", default="DICT_4X4_50",
                help="cv2.aruco dictionary, e.g. DICT_4X4_50 (ArUco) or DICT_APRILTAG_36h11 (AprilTag)")
ap.add_argument("--state-topic", default="rt/sportmodestate", help="TODO verify on robot (maybe rt/lf/sportmodestate)")
ap.add_argument("--back-dist", type=float, default=0.5, help="meters to back up")
ap.add_argument("--turn-deg", type=float, default=-90.0, help="degrees to rotate (negative = right)")
ap.add_argument("--approach-vx", type=float, default=0.15, help="forward speed while servoing (m/s)")
ap.add_argument("--back-vx", type=float, default=0.15, help="backup speed (m/s)")
ap.add_argument("--turn-w", type=float, default=0.4, help="turn rate (rad/s)")
ap.add_argument("--servo-kw", type=float, default=1.2, help="yaw gain on normalized tag bearing")
ap.add_argument("--stop-size", type=float, default=140.0, help="tag pixel size to stop at (bigger = closer)")
ap.add_argument("--search-w", type=float, default=0.3, help="rotate rate while searching for the tag (rad/s)")
ap.add_argument("--test-cam", action="store_true", help="detect + print tag bearing/size, MOVE NOTHING (safe)")
ap.add_argument("--rate", type=float, default=20.0, help="control loop Hz")
args = ap.parse_args()

ChannelFactoryInitialize(0, args.iface)
DT = 1.0 / args.rate
if not hasattr(cv2.aruco, args.dict):
    sys.exit(f"[nav] unknown --dict {args.dict}")
ARUCO = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict)),
                                cv2.aruco.DetectorParameters())


class HeadCam:
    """Latest head-camera frame (left ZED eye) over ZMQ from Teleimager. Non-blocking; holds last frame."""
    def __init__(self, host, port):
        ctx = zmq.Context()
        self.s = ctx.socket(zmq.SUB)
        self.s.setsockopt(zmq.CONFLATE, 1)
        self.s.setsockopt_string(zmq.SUBSCRIBE, "")
        self.s.setsockopt(zmq.RCVTIMEO, 100)
        self.s.connect(f"tcp://{host}:{port}")
        self.frame = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                buf = self.s.recv()
            except Exception:
                continue
            arr = np.frombuffer(buf, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                self.frame = img[:, : img.shape[1] // 2]   # ZED side-by-side -> left eye

    def detect(self, tag_id):
        """(found, bearing[-1..1 left..right], size_px) for tag_id in the latest frame."""
        f = self.frame
        if f is None:
            return False, 0.0, 0.0
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = ARUCO.detectMarkers(gray)
        if ids is None:
            return False, 0.0, 0.0
        for c, i in zip(corners, ids.flatten()):
            if int(i) == tag_id:
                pts = c.reshape(4, 2)
                bearing = (pts[:, 0].mean() - f.shape[1] / 2.0) / (f.shape[1] / 2.0)   # -1 left .. +1 right
                size = float(np.mean([np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]))
                return True, float(bearing), size
        return False, 0.0, 0.0


class Odom:
    """Latest yaw (rad) + planar position from SportModeState_."""
    def __init__(self, topic):
        self.yaw = None
        self.pos = None
        self.sub = ChannelSubscriber(topic, SportModeState_)
        self.sub.Init()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            time.sleep(0.005)
            m = self.sub.Read()
            if m is not None:
                self.yaw = float(m.imu_state.rpy[2])
                self.pos = np.array([m.position[0], m.position[1]])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


cam = HeadCam(args.robot_host, args.head_port)

# ---- SAFE: camera + detection test, no motion ----
if args.test_cam:
    print(f"[nav] --test-cam: looking for marker {args.tag_id} ({args.dict}); Ctrl-C to stop. NO motion.", flush=True)
    try:
        while True:
            found, bearing, size = cam.detect(args.tag_id)
            print(f"\rtag {args.tag_id}: {'FOUND' if found else 'none '}  bearing {bearing:+.2f}  size {size:6.1f}px"
                  f"  (stop at {args.stop_size})   ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[nav] test-cam done", flush=True)
    sys.exit(0)

# ---- full routine (moves the robot) ----
odom = Odom(args.state_topic)
loco = LocoClient()
loco.SetTimeout(0.5)
loco.Init()
time.sleep(0.5)
if odom.yaw is None:
    print(f"[nav] WARNING: no odometry on {args.state_topic} -- turn/backup will fall back to TIMED (less accurate). "
          f"Verify the topic name.", flush=True)


def move_for(vx, vy, vyaw, secs):
    n = max(1, int(secs / DT))
    for _ in range(n):
        loco.Move(vx, vy, vyaw)
        time.sleep(DT)


try:
    # 1) BACK UP -----------------------------------------------------------------
    print(f"[nav] backing up {args.back_dist} m", flush=True)
    if odom.pos is not None:
        p0 = odom.pos.copy()
        while np.linalg.norm(odom.pos - p0) < args.back_dist:
            loco.Move(-args.back_vx, 0.0, 0.0); time.sleep(DT)
    else:
        move_for(-args.back_vx, 0.0, 0.0, args.back_dist / max(args.back_vx, 1e-3))
    loco.StopMove(); time.sleep(0.3)

    # 2) ROTATE (right 90 by default) --------------------------------------------
    print(f"[nav] rotating {args.turn_deg} deg", flush=True)
    tgt = math.radians(args.turn_deg)
    w = -abs(args.turn_w) if args.turn_deg < 0 else abs(args.turn_w)
    if odom.yaw is not None:
        y0 = odom.yaw
        while abs(_wrap(odom.yaw - y0)) < abs(tgt) - math.radians(3):
            loco.Move(0.0, 0.0, w); time.sleep(DT)
    else:
        move_for(0.0, 0.0, w, abs(tgt) / max(abs(args.turn_w), 1e-3))
    loco.StopMove(); time.sleep(0.3)

    # 3) SEARCH for the tag (rotate slowly until found) --------------------------
    print(f"[nav] searching for tag {args.tag_id}", flush=True)
    t0 = time.time()
    while not cam.detect(args.tag_id)[0]:
        loco.Move(0.0, 0.0, args.search_w); time.sleep(DT)
        if time.time() - t0 > 20:
            loco.StopMove(); print("[nav] tag not found in 20s -- stopping", flush=True); sys.exit(1)
    loco.StopMove(); time.sleep(0.2)

    # 4) SERVO straight to the tag (center bearing, drive forward until close) ----
    print("[nav] approaching tag", flush=True)
    lost = 0
    while True:
        found, bearing, size = cam.detect(args.tag_id)
        if not found:
            lost += 1
            loco.Move(0.0, 0.0, 0.0)
            if lost > int(1.5 * args.rate):
                print("[nav] lost the tag -- stopping", flush=True); break
            time.sleep(DT); continue
        lost = 0
        if size >= args.stop_size:
            print(f"[nav] reached tag (size {size:.0f} >= {args.stop_size})", flush=True); break
        vyaw = float(np.clip(-args.servo_kw * bearing, -args.turn_w, args.turn_w))
        vx = args.approach_vx * (1.0 - min(1.0, abs(bearing)))   # slow down when off-center
        loco.Move(vx, 0.0, vyaw); time.sleep(DT)
    loco.StopMove()
    print("[nav] done", flush=True)
finally:
    try:
        loco.StopMove()
    except Exception:
        pass
