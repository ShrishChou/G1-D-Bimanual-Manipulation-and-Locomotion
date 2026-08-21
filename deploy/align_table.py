"""Standalone LiDAR square-up: rotate the base until it is square (perpendicular) to the flat table/surface in
front. Fits the front LiDAR sector to a line and drives its tilt to zero with a proportional controller.

    conda activate <env>
    python align_table.py --monitor          # just print the measured tilt (no motion) to sanity-check
    python align_table.py                     # SPACE to start squaring up; q/Ctrl-C stops
    python align_table.py --flip              # if it turns the WRONG way

tilt = 0 means square. Positive tilt = table's left side is farther. --gain<1 avoids overshoot.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Twist_ as mkTwist
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from laserscan_idl import LaserScan_

p = argparse.ArgumentParser()
p.add_argument("--iface", default="enp2s0")
p.add_argument("--monitor", action="store_true", help="just print tilt, no motion")
p.add_argument("--front-deg", type=float, default=0.0, help="LiDAR bearing pointing robot-forward")
p.add_argument("--sector", type=float, default=20.0, help="half-width (deg) of the front sector to fit (narrower = cleaner central patch)")
p.add_argument("--max-range", type=float, default=3.0, help="ignore points farther than this (m)")
p.add_argument("--face-depth", type=float, default=0.20, help="keep only points within this depth of the nearest surface (isolates the table face from the background)")
p.add_argument("--max-resid", type=float, default=0.025, help="reject tilt readings whose line-fit residual exceeds this (m) -> only trust clean flat fits")
p.add_argument("--tol", type=float, default=2.0, help="squared when |tilt| below this (deg)")
p.add_argument("--gain", type=float, default=0.5, help="rotate this fraction of the measured tilt each step (<1 = no overshoot)")
p.add_argument("--max-step", type=float, default=10.0, help="cap rotation per step (deg)")
p.add_argument("--wvel", type=float, default=0.25, help="turn speed (rad/s)")
p.add_argument("--flip", action="store_true", help="invert rotation direction")
args = p.parse_args()

ChannelFactoryInitialize(0, args.iface)
pub = ChannelPublisher("rt/cmd_vel_no_limit", Twist_); pub.Init()
odom_sub = ChannelSubscriber("rt/agv/odom", Odometry_); odom_sub.Init()
scan_sub = ChannelSubscriber("rt/slamware_ros_sdk_server_node/scan", LaserScan_); scan_sub.Init()

_pose = {"v": None}
_scan = {"m": None}


def _yaw(o):
    return math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))


def _odom():
    while True:
        m = odom_sub.Read()
        if m is not None:
            q = m.pose.pose.position
            _pose["v"] = (float(q.x), float(q.y), float(_yaw(m.pose.pose.orientation)))
        time.sleep(0.005)


def _scanr():
    while True:
        m = scan_sub.Read()
        if m is not None:
            _scan["m"] = m
        time.sleep(0.02)


threading.Thread(target=_odom, daemon=True).start()
threading.Thread(target=_scanr, daemon=True).start()


def send_vel(vx, wz):
    m = mkTwist(); m.linear.x = float(vx); m.angular.z = float(wz); pub.Write(m)


def stop_base():
    for _ in range(5):
        send_vel(0.0, 0.0); time.sleep(0.02)


def _kill(*_):
    stop_base(); print("\n[align] stop", flush=True); os._exit(0)


signal.signal(signal.SIGINT, _kill)
signal.signal(signal.SIGTERM, _kill)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def table_tilt():
    """Robust line fit of the front face. Returns (tilt_rad, dist_m, npts, resid_m, yspread_m) or Nones."""
    m = _scan["m"]
    if m is None:
        return None, None, 0, None, None
    r = np.array(m.ranges, float)
    ang = m.angle_min + np.arange(len(r)) * m.angle_increment
    d = np.abs((ang - math.radians(args.front_deg) + math.pi) % (2 * math.pi) - math.pi)
    sel = (d <= math.radians(args.sector)) & np.isfinite(r) & (r > m.range_min) & (r < args.max_range)
    if sel.sum() < 8:
        return None, None, int(sel.sum()), None, None
    rs = r[sel]
    face = rs <= rs.min() + args.face_depth              # keep only the FRONT-most surface (the table), drop background
    if face.sum() < 8:
        return None, None, int(face.sum()), None, None
    a = (ang[sel] - math.radians(args.front_deg))[face]
    rr = rs[face]
    x = rr * np.cos(a); y = rr * np.sin(a)               # x = forward dist, y = lateral

    def fit(xx, yy):
        yc = yy - yy.mean()
        s = float((yc * (xx - xx.mean())).sum() / ((yc * yc).sum() + 1e-9))   # x = s*y + c
        c = xx.mean() - s * yy.mean()
        res = xx - (s * yy + c)
        return s, res

    slope, res = fit(x, y)
    keep = np.abs(res) < 2.5 * (np.std(res) + 1e-6)      # one robust pass: drop points off the line
    if keep.sum() >= 8:
        slope, res = fit(x[keep], y[keep])
        x, y = x[keep], y[keep]
    resid = float(np.sqrt((res ** 2).mean()))
    return math.atan(slope), float(x.mean()), int(len(x)), resid, float(y.max() - y.min())


def turn_by(rad, poller):
    if abs(rad) < math.radians(0.3) or _pose["v"] is None:
        return
    y0 = _pose["v"][2]; target = abs(rad); wz = abs(args.wvel) * (1 if rad > 0 else -1)
    while abs(_wrap(_pose["v"][2] - y0)) < target:
        send_vel(0.0, wz)
        if poller.poll() in ("q", "\x1b"):
            break
        time.sleep(0.05)
    stop_base()


class KeyPoller:
    def __enter__(self):
        self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); tty.setcbreak(self.fd); return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


for _ in range(300):                                     # wait for first scan
    if _scan["m"] is not None:
        break
    time.sleep(0.01)

def median_tilt(n=15):
    """Median tilt over n fresh scans, keeping ONLY low-residual (clean flat) fits -> rejects the bad scans."""
    vals = []
    for _ in range(n):
        t, dist, npts, resid, ysp = table_tilt()
        if t is not None and resid is not None and resid <= args.max_resid:
            vals.append((t, dist, npts, resid, ysp))
        time.sleep(0.03)
    if not vals:
        return None, None, 0, None, None
    ts = sorted(v[0] for v in vals)
    med = ts[len(ts) // 2]
    _, dist, npts, resid, ysp = min(vals, key=lambda v: abs(v[0] - med))
    return med, dist, npts, resid, ysp


if args.monitor:
    from collections import deque
    print("[align] MONITOR (no motion). raw + smoothed tilt, fit residual, lateral spread. Ctrl-C to quit.\n"
          "  low residual (<~2cm) + healthy spread = clean flat face; high residual = clutter/legs (unreliable).",
          flush=True)
    buf = deque(maxlen=15)
    while True:
        tilt, dist, n, resid, ysp = table_tilt()
        if tilt is None:
            print(f"\r[align] no face (pts={n})                                   ", end="", flush=True)
        else:
            buf.append(tilt); sm = sorted(buf)[len(buf) // 2]
            print(f"\r[align] raw {math.degrees(tilt):+5.1f}  smooth {math.degrees(sm):+5.1f} deg   "
                  f"dist {dist:.2f}m  resid {resid*100:4.1f}cm  spread {ysp:.2f}m  pts={n}    ",
                  end="", flush=True)
        time.sleep(0.1)

flip = -1.0 if args.flip else 1.0
print("[align] SPACE to square up to the table; q/Ctrl-C stops.", flush=True)
with KeyPoller() as poller:
    while True:
        k = poller.poll()
        if k == " ":
            break
        if k in ("q", "\x1b"):
            _kill()
        time.sleep(0.02)

    stable = 0
    for it in range(20):
        tilt, dist, n, resid, ysp = median_tilt()       # median over several scans -> stable
        if tilt is None:
            print(f"\n[align] no table face (pts={n}) -- widen --sector or move closer", flush=True); break
        td = math.degrees(tilt)
        print(f"[align] iter {it}: tilt {td:+.1f} deg  face {dist:.2f}m  resid {resid*100:.1f}cm  pts={n}", flush=True)
        if resid is not None and resid > 0.05:
            print("[align]   WARNING: high fit residual -> not a clean flat face; alignment unreliable", flush=True)
        if abs(td) <= args.tol:
            stable += 1
            if stable >= 2:
                break
            time.sleep(0.2); continue
        stable = 0
        step = flip * args.gain * tilt
        step = max(-math.radians(args.max_step), min(math.radians(args.max_step), step))
        turn_by(step, poller)
        time.sleep(0.5)                                  # settle + fresh scan
    stop_base()
    tf = median_tilt()[0]
    print(f"\n[align] done -- final tilt {math.degrees(tf or 0):+.1f} deg", flush=True)
