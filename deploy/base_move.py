"""Base-locomotion sequence for the wheeled AGV base: back up, turn right 90 deg, drive forward.

The base is a wheeled AGV driven by a standard ROS cmd_vel Twist on rt/cmd_vel_no_limit (linear.x = fwd/back,
angular.z = yaw; +z is CCW/left, so right turn = negative). Odometry is on rt/odommodestate (SportModeState).
This is a separate AGV controller -- independent of the ai/normal humanoid mode -- so no LocoClient, no mode
switch, and it works alongside arm control.

    conda activate <env>
    python $REPO/deploy/base_move.py

Each SPACE runs the next leg:  #1 back --back m   #2 turn RIGHT --turn-deg   #3 forward --fwd m
    q / ESC / Ctrl-C = stop (zero velocity) + quit.
"""
import argparse
import math
import os
import select
import signal
import sys
import termios
import threading
import time
import tty

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Twist_ as mkTwist
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laserscan_idl import LaserScan_

p = argparse.ArgumentParser()
p.add_argument("--iface", default="enp2s0")
p.add_argument("--back", type=float, default=0.16, help="metres to reverse")
p.add_argument("--fwd", type=float, default=1.4, help="metres to drive forward (fixed, odom closed-loop) in the sequence")
p.add_argument("--turn-deg", type=float, default=90.0, help="degrees to turn RIGHT (clockwise)")
p.add_argument("--vel", type=float, default=0.05, help="linear speed (m/s)")
p.add_argument("--wvel", type=float, default=0.3, help="turn speed (rad/s)")
p.add_argument("--stop-margin", type=float, default=0.01, help="stop this far before target to cancel coast (m)")
p.add_argument("--stop-dist", type=float, default=0.50,
               help="forward leg: stop when the front LiDAR reads this range (m). NOTE: LiDAR range_min ~0.46 m, "
                    "so it CANNOT sense closer than that -- 1-2 cm is physically impossible on this sensor.")
p.add_argument("--front-deg", type=float, default=0.0, help="LiDAR bearing that points robot-forward (calibrate with --scan-only)")
p.add_argument("--front-width", type=float, default=15.0, help="half-width (deg) of the front sector to watch")
p.add_argument("--fwd-max", type=float, default=1.5, help="safety: max metres to drive forward before giving up")
p.add_argument("--scan-only", action="store_true", help="just print the front LiDAR distance + nearest bearing (calibration)")
p.add_argument("--forward-only", action="store_true", help="run ONLY the LiDAR forward-to-object leg (skip back/turn)")
p.add_argument("--cmd-topic", default="rt/cmd_vel_no_limit")
p.add_argument("--odom-topic", default="rt/agv/odom", help="AGV odometry (nav_msgs/Odometry)")
p.add_argument("--scan-topic", default="rt/slamware_ros_sdk_server_node/scan")
args = p.parse_args()

ChannelFactoryInitialize(0, args.iface)
pub = ChannelPublisher(args.cmd_topic, Twist_); pub.Init()

_pose = {"v": None}
_sub = ChannelSubscriber(args.odom_topic, Odometry_); _sub.Init()


def _yaw(o):
    return math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))


def _odom():
    while True:
        m = _sub.Read()
        if m is not None:
            p = m.pose.pose.position; o = m.pose.pose.orientation
            _pose["v"] = (float(p.x), float(p.y), float(_yaw(o)))
        time.sleep(0.005)


threading.Thread(target=_odom, daemon=True).start()

_scan = {"m": None}
_scan_sub = ChannelSubscriber(args.scan_topic, LaserScan_); _scan_sub.Init()


def _scan_reader():
    while True:
        m = _scan_sub.Read()
        if m is not None:
            _scan["m"] = m
        time.sleep(0.02)


threading.Thread(target=_scan_reader, daemon=True).start()


def _angles(m):
    return m.angle_min + np.arange(len(m.ranges)) * m.angle_increment


def front_range():
    """Min valid LiDAR range in the front sector (m), or inf if nothing seen."""
    m = _scan["m"]
    if m is None:
        return float("inf")
    r = np.array(m.ranges, float)
    d = np.abs((_angles(m) - math.radians(args.front_deg) + math.pi) % (2 * math.pi) - math.pi)
    sel = (d <= math.radians(args.front_width)) & np.isfinite(r) & (r > m.range_min)
    return float(r[sel].min()) if sel.any() else float("inf")


def nearest_bearing():
    m = _scan["m"]
    if m is None:
        return None, float("inf")
    r = np.array(m.ranges, float)
    ok = np.isfinite(r) & (r > m.range_min)
    if not ok.any():
        return None, float("inf")
    i = int(np.where(ok)[0][np.argmin(r[ok])])
    return math.degrees(_angles(m)[i]), float(r[i])


def send_vel(vx, wz):
    t = mkTwist(); t.linear.x = float(vx); t.angular.z = float(wz); pub.Write(t)


def stop_base():
    for _ in range(5):
        send_vel(0.0, 0.0); time.sleep(0.02)


def _kill(*_):
    stop_base()
    print("\n[base_move] Ctrl-C -> stop", flush=True)
    os._exit(0)


signal.signal(signal.SIGINT, _kill)
signal.signal(signal.SIGTERM, _kill)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _have_odom():
    for _ in range(200):
        if _pose["v"] is not None:
            return True
        time.sleep(0.01)
    return False


def drive(dist, poller):
    """+dist forward, -dist backward. Closed-loop on odom if available, else timed."""
    sgn = 1.0 if dist >= 0 else -1.0
    if _have_odom():
        x0, y0, _ = _pose["v"]
        print(f"[base_move] drive {dist*100:+.0f} cm at {args.vel:.2f} m/s (ODOM, start x={x0:.3f} y={y0:.3f})...", flush=True)
        while True:
            d = math.hypot(_pose["v"][0] - x0, _pose["v"][1] - y0)
            if d >= abs(dist) - args.stop_margin:
                break
            send_vel(sgn * args.vel, 0.0)
            if poller.poll() in ("q", "\x1b"):
                break
            print(f"\r[base_move]   odom moved {d*100:5.1f} / {abs(dist)*100:.0f} cm  ", end="", flush=True)
            time.sleep(0.03)
        stop_base()
        time.sleep(0.3)                                       # let it settle, then report final (incl. coast)
        print(f"\n[base_move]   FINAL odom moved {math.hypot(_pose['v'][0]-x0, _pose['v'][1]-y0)*100:.1f} cm "
              f"(target {abs(dist)*100:.0f})", flush=True)
    else:
        dur = abs(dist) / max(args.vel, 1e-3)
        print(f"[base_move] no odom -> TIMED drive {dist*100:+.0f} cm ({dur:.1f}s)...", flush=True)
        t0 = time.time()
        while time.time() - t0 < dur:
            send_vel(sgn * args.vel, 0.0)
            if poller.poll() in ("q", "\x1b"):
                break
            time.sleep(0.05)
        stop_base()
        print("[base_move]   done (open-loop)", flush=True)


def turn_right(deg, poller):
    """Turn clockwise (right) by deg. +angular.z is CCW (left), so right = negative."""
    if _have_odom():
        _, _, y0 = _pose["v"]; target = math.radians(abs(deg))
        print(f"[base_move] turn RIGHT {deg:.0f} deg at {args.wvel:.2f} rad/s (odom)...", flush=True)
        while abs(_wrap(_pose["v"][2] - y0)) < target:
            send_vel(0.0, -abs(args.wvel))
            if poller.poll() in ("q", "\x1b"):
                break
            time.sleep(0.05)
        stop_base()
        print(f"[base_move]   turned {math.degrees(abs(_wrap(_pose['v'][2]-y0))):.1f} deg", flush=True)
    else:
        dur = math.radians(abs(deg)) / max(args.wvel, 1e-3)
        print(f"[base_move] no odom -> TIMED turn RIGHT {deg:.0f} deg ({dur:.1f}s)...", flush=True)
        t0 = time.time()
        while time.time() - t0 < dur:
            send_vel(0.0, -abs(args.wvel))
            if poller.poll() in ("q", "\x1b"):
                break
            time.sleep(0.05)
        stop_base()
        print("[base_move]   done (open-loop)", flush=True)


def drive_to_object(poller):
    """Drive forward until the front LiDAR sees something within --stop-dist (or --fwd-max travelled)."""
    for _ in range(200):
        if _scan["m"] is not None:
            break
        time.sleep(0.01)
    if _scan["m"] is None:
        print("[base_move] NO LiDAR scan on {} -- aborting forward".format(args.scan_topic), flush=True); return
    x0, y0 = (_pose["v"][0], _pose["v"][1]) if _pose["v"] else (0.0, 0.0)
    print(f"[base_move] forward until front LiDAR <= {args.stop_dist:.2f} m "
          f"(front sector {args.front_deg:+.0f}+/-{args.front_width:.0f} deg)...", flush=True)
    while True:
        fr = front_range()
        trav = math.hypot(_pose["v"][0] - x0, _pose["v"][1] - y0) if _pose["v"] else 0.0
        print(f"\r[base_move]   front {fr:5.2f} m   traveled {trav*100:5.1f} cm   ", end="", flush=True)
        if fr <= args.stop_dist:
            break
        if trav >= args.fwd_max:
            print("\n[base_move]   --fwd-max reached, stopping (no object within range)", flush=True); break
        send_vel(args.vel, 0.0)
        if poller.poll() in ("q", "\x1b"):
            break
        time.sleep(0.05)
    stop_base()
    print(f"\n[base_move]   stopped: front LiDAR {front_range():.2f} m", flush=True)


if args.forward_only:
    LEGS = [("forward to object (LiDAR)", lambda pl: drive_to_object(pl))]
else:
    LEGS = [("back", lambda pl: drive(-args.back, pl)),
            ("turn right", lambda pl: turn_right(args.turn_deg, pl)),
            ("forward", lambda pl: drive(args.fwd, pl))]   # fixed distance (odom), not LiDAR


class KeyPoller:
    def __enter__(self):
        self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); tty.setcbreak(self.fd); return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


if args.scan_only:
    print("[base_move] SCAN MONITOR (q to quit). Put an object directly in front; the bearing whose range "
          "drops to it is your robot-forward -> pass it as --front-deg.", flush=True)
    try:
        with KeyPoller() as poller:
            while True:
                b, br = nearest_bearing()
                bs = "none" if b is None else f"{round(b):+d}"
                print(f"\r[scan] front({args.front_deg:+.0f}+/-{args.front_width:.0f}deg)={front_range():5.2f} m   "
                      f"nearest bearing={bs} deg @ {br:5.2f} m   ", end="", flush=True)
                if poller.poll() in ("q", "\x1b"):
                    break
                time.sleep(0.1)
    finally:
        print("\n[base_move] scan monitor done", flush=True)
    sys.exit(0)

print(f"[base_move] cmd={args.cmd_topic}  odom={args.odom_topic}  scan={args.scan_topic}", flush=True)
_seq = "forward to object (LiDAR)" if args.forward_only else f"forward {args.fwd*100:.0f}cm"
print(f"[base_move] READY: SPACE runs each leg -> back {args.back*100:.0f}cm, turn right {args.turn_deg:.0f}deg, "
      f"{_seq}.  q/Ctrl-C = stop.", flush=True)
i = 0
try:
    with KeyPoller() as poller:
        while True:
            k = poller.poll()
            if k == " ":
                if i < len(LEGS):
                    name, fn = LEGS[i]
                    print(f"\n[base_move] leg {i+1}/{len(LEGS)}: {name}", flush=True)
                    fn(poller)
                    i += 1
                    print("[base_move] " + ("all legs done (q to quit)" if i >= len(LEGS)
                                            else f"SPACE -> next: {LEGS[i][0]}"), flush=True)
                else:
                    print("[base_move] sequence complete (q to quit)", flush=True)
            elif k in ("q", "\x1b"):
                break
            time.sleep(0.02)
finally:
    stop_base()
    print("\n[base_move] done", flush=True)
