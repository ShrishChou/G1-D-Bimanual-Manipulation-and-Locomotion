#!/usr/bin/env python3
"""Standalone table alignment (base only, no arm) -- test the 'drive forward while squaring up to the table'
motion in isolation. Drives forward toward the table while continuously correcting heading to the table
face, stopping flush at --stop (LiDAR distance). Table-referenced, immune to SLAM drift.

`tv` env:
  python nav/table_align.py            # DRY-RUN (prints tilt/dist, no motion)
  python nav/table_align.py --go       # drive+align (flush ~0.30, default stop 0.32)
  python nav/table_align.py --go --stop 0.30 --flip

Measured on this robot: 0.30m LiDAR = front flush against the table; LiDAR minimum ~0.296m.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy"))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Twist_ as mkTwist
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_, PoseStamped_
from laserscan_idl import LaserScan_

FRONT_DEG = 0.0
FACE_DEPTH = 0.20       # keep only returns within this of the nearest surface (isolates the flat face)
_scan = {"m": None}
_pose = {"v": None}     # (x, y, yaw) absolute, for the nudge/turn/reverse maneuvers


def yaw_of(o):
    return math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))


def ang_norm(x):
    return math.atan2(math.sin(x), math.cos(x))


def table_fit(sector):
    """Return (tilt_rad, forward_distance_m) of the table face in the front sector, or (None, None)."""
    m = _scan["m"]
    if m is None:
        return None, None
    r = np.array(m.ranges, float)
    ang = m.angle_min + np.arange(len(r)) * m.angle_increment
    off = np.abs((ang - math.radians(FRONT_DEG) + math.pi) % (2 * math.pi) - math.pi)
    # NOTE: use a fixed floor, NOT m.range_min -- range_min dynamically tracks the nearest return, so it
    # would filter out the table face itself as you get close (~0.30m) and the fit would fail short.
    sel = (off <= math.radians(sector)) & np.isfinite(r) & (r > 0.1) & (r < 3.0)
    if sel.sum() < 8:
        return None, None
    rr = r[sel]; a = ang[sel] - math.radians(FRONT_DEG)
    # isolate the flat face NEAREST us -- drop returns beyond it (background / wall edges / corners), else
    # the line fit skews and reports a huge bogus tilt as the robot moves.
    keep = rr <= (rr.min() + FACE_DEPTH)
    if keep.sum() < 8:
        return None, None
    rr = rr[keep]; a = a[keep]
    x = rr * np.cos(a); y = rr * np.sin(a)
    slope = float(((y - y.mean()) * (x - x.mean())).sum() / (((y - y.mean()) ** 2).sum() + 1e-9))
    return math.atan(slope), float(np.median(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp2s0")
    ap.add_argument("--stop", type=float, default=0.30, help="LiDAR distance to stop flush at (m); 0.30 = touching")
    ap.add_argument("--sector", type=float, default=40.0, help="front sector half-width for the face fit (deg)")
    ap.add_argument("--face-depth", type=float, default=0.20, help="keep returns within this of the nearest surface (m) -- isolates the flat face")
    ap.add_argument("--vmax", type=float, default=0.10, help="max forward speed (m/s)")
    ap.add_argument("--wmax", type=float, default=0.20, help="max yaw rate (rad/s)")
    ap.add_argument("--kp", type=float, default=0.6, help="P gain on tilt (rad/s per rad) -- low enough to not saturate")
    ap.add_argument("--kd", type=float, default=0.15, help="D gain (damps overshoot/oscillation)")
    ap.add_argument("--deadband", type=float, default=2.0, help="stop correcting heading when |tilt| < this (deg)")
    ap.add_argument("--freeze-dist", type=float, default=0.50, help="below this table distance, stop rotating and creep straight (m)")
    ap.add_argument("--nudge", type=float, default=0.0, help="after the LiDAR stop, drive forward this far open-loop (m) "
                    "to go past where the LiDAR can see (e.g. front under the table). Referenced from the aligned stop.")
    ap.add_argument("--turn-after", type=float, default=0.0, help="after arriving, rotate in place this many deg (+CCW)")
    ap.add_argument("--back-after", type=float, default=0.0, help="after the turn, reverse this many m (e.g. back into a charger)")
    ap.add_argument("--flip", action="store_true", help="invert the rotation direction")
    ap.add_argument("--go", action="store_true", help="ACTUALLY move (robot MOVES). Omit = dry-run.")
    a = ap.parse_args()
    flip = -1.0 if a.flip else 1.0
    global FACE_DEPTH
    FACE_DEPTH = a.face_depth

    ChannelFactoryInitialize(0, a.iface)
    pub = ChannelPublisher("rt/cmd_vel_no_limit", Twist_); pub.Init()
    sub = ChannelSubscriber("rt/slamware_ros_sdk_server_node/scan", LaserScan_); sub.Init()
    psub = ChannelSubscriber("rt/robot_pose", PoseStamped_); psub.Init()
    import threading

    def reader():
        while True:
            m = sub.Read()
            if m is not None:
                _scan["m"] = m
            p = psub.Read()
            if p is not None:
                _pose["v"] = (float(p.pose.position.x), float(p.pose.position.y), float(yaw_of(p.pose.orientation)))
            time.sleep(0.02)
    threading.Thread(target=reader, daemon=True).start()

    def send(vx, wz):
        t = mkTwist(); t.linear.x = float(vx); t.angular.z = float(wz); pub.Write(t)

    t0 = time.time()
    while _scan["m"] is None and time.time() - t0 < 5:
        time.sleep(0.05)
    tilt, d = table_fit(a.sector)
    if tilt is None:
        print("[table_align] table not visible in the front sector -- aim the robot at the table and retry.")
        return
    print(f"[table_align] start: table dist={d:.3f}m tilt={math.degrees(tilt):+.1f}deg  "
          f"target flush={a.stop:.2f}m  {'DRIVING' if a.go else 'DRY-RUN (no motion)'}", flush=True)
    if not a.go:
        print("[table_align] DRY-RUN -- add --go to drive. It would move forward while squaring up.")
        return

    dead = math.radians(a.deadband)
    # PRE-SQUARE: if we start badly angled (e.g. SLAM left us facing 47deg off the wall), rotate IN PLACE
    # first to face the surface squarely -- no forward, no diverge-abort -- so the drive-in starts aligned.
    if tilt is not None and abs(tilt) > math.radians(8.0):
        print(f"[table_align] pre-square in place from {math.degrees(tilt):+.0f}deg ...", flush=True)
        t_ps = time.time()
        try:
            while time.time() - t_ps < 25:
                tilt, d = table_fit(a.sector)
                if tilt is None:
                    send(0.0, 0.0); time.sleep(0.05); continue
                if abs(tilt) <= dead:
                    break
                send(0.0, max(-a.wmax, min(a.wmax, -flip * a.kp * tilt)))
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        for _ in range(5):
            send(0.0, 0.0); time.sleep(0.02)
        tilt, d = table_fit(a.sector)
        print(f"[table_align] pre-squared to {(math.degrees(tilt) if tilt is not None else 0):+.1f}deg", flush=True)
    prev_tilt = tilt; prev_t = time.time()
    t0 = time.time(); last = 0.0; lost = 0; arrived = False
    try:
        while time.time() - t0 < 60:
            tilt, d = table_fit(a.sector)
            if d is None:
                lost += 1
                if lost > 6:
                    send(0.0, 0.0); print("\n[table_align] lost the table face -- stopping"); break
                send(0.0, 0.0); time.sleep(0.05); continue
            lost = 0
            if d - a.stop <= 0.005:                    # arrival check FIRST -> creep all the way to --stop
                arrived = True; break
            frozen = d < a.freeze_dist
            # abort on a diverging tilt ONLY while we actually steer by it (far from the table). Close in,
            # the wide-sector fit is noisy and the tilt is unreliable -- but we're creeping STRAIGHT there
            # anyway (frozen), so ignore it instead of false-stopping.
            if not frozen and abs(tilt) > math.radians(35):
                send(0.0, 0.0)
                print(f"\n[table_align] tilt {math.degrees(tilt):+.0f}deg too large while steering (bad fit?) -- "
                      f"STOPPED. Try --flip, or start with the table more squarely ahead.", flush=True)
                break
            now = time.time()
            dtl = (tilt - prev_tilt) / max(1e-3, now - prev_t)
            prev_tilt, prev_t = tilt, now
            v = max(0.03, min(a.vmax, 0.4 * (d - a.stop)))
            if frozen or abs(tilt) < dead:             # creep straight when close (tilt unreliable) or squared
                wz = 0.0
            else:
                wz = max(-a.wmax, min(a.wmax, -flip * (a.kp * tilt + a.kd * dtl)))
            send(v, wz)
            if time.time() - last > 0.2:
                last = time.time()
                tag = "freeze" if d < a.freeze_dist else ("square" if abs(tilt) < dead else "align")
                print(f"\r[table_align] dist={d:.3f}m tilt={math.degrees(tilt):+5.1f}deg  v={v:.2f} w={wz:+.2f} [{tag}]   ",
                      end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            send(0.0, 0.0); time.sleep(0.02)

    # open-loop nudge forward past the LiDAR's minimum (e.g. front under the table), from the aligned stop
    if arrived and a.nudge > 0 and _pose["v"] is not None:
        print(f"\n[table_align] nudge {a.nudge*100:.0f}cm forward (open-loop, past LiDAR range) ...", flush=True)
        sx, sy = _pose["v"][0], _pose["v"][1]
        t1 = time.time()
        try:
            while time.time() - t1 < 20:
                if math.hypot(_pose["v"][0] - sx, _pose["v"][1] - sy) >= a.nudge:
                    break
                send(0.06, 0.0); time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            for _ in range(5):
                send(0.0, 0.0); time.sleep(0.02)
        print(f"[table_align] nudged {math.hypot(_pose['v'][0]-sx, _pose['v'][1]-sy)*100:.1f}cm", flush=True)

    # rotate in place (e.g. 180), then reverse (e.g. back into a charger)
    if arrived and abs(a.turn_after) > 0.1 and _pose["v"] is not None:
        # Turn to an EXACT target heading (SLAM/LiDAR yaw), not a blind accumulate: phase 1 forces the
        # commanded direction (sign of --turn-after) coarsely (avoids the 180 ambiguity), phase 2 closed-loops
        # onto the exact absolute target heading so we end facing the right way.
        direction = 1.0 if a.turn_after > 0 else -1.0
        start_yaw = _pose["v"][2]
        target_yaw = ang_norm(start_yaw + math.radians(a.turn_after))
        target_abs = abs(math.radians(a.turn_after))
        print(f"\n[table_align] rotate {a.turn_after:+.0f}deg ({'CCW/left' if direction > 0 else 'CW/right'}) "
              f"to heading {math.degrees(target_yaw):+.1f}deg ...", flush=True)
        acc = 0.0; prev = start_yaw; t1 = time.time()
        try:
            while time.time() - t1 < 40 and acc < target_abs - math.radians(20):   # phase 1: coarse, direction-forced
                cur = _pose["v"][2]; acc += abs(ang_norm(cur - prev)); prev = cur
                send(0.0, direction * a.wmax); time.sleep(0.05)
            while time.time() - t1 < 40:                                            # phase 2: fine, onto exact heading
                err = ang_norm(target_yaw - _pose["v"][2])
                if abs(err) <= math.radians(1.5):
                    break
                send(0.0, max(-a.wmax, min(a.wmax, 1.2 * err))); time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            for _ in range(5):
                send(0.0, 0.0); time.sleep(0.02)
        print(f"[table_align] heading now {math.degrees(_pose['v'][2]):+.1f}deg (target {math.degrees(target_yaw):+.1f})", flush=True)
    if arrived and a.back_after > 0.001 and _pose["v"] is not None:
        # reverse while HOLDING heading (correct any yaw drift) so it backs up straight in the right direction
        hold_yaw = _pose["v"][2]
        print(f"[table_align] reverse {a.back_after*100:.0f}cm holding heading {math.degrees(hold_yaw):+.1f}deg "
              f"(rear-blind, slow) ...", flush=True)
        sx, sy = _pose["v"][0], _pose["v"][1]
        t1 = time.time()
        try:
            while time.time() - t1 < 20:
                if math.hypot(_pose["v"][0] - sx, _pose["v"][1] - sy) >= a.back_after:
                    break
                err = ang_norm(hold_yaw - _pose["v"][2])
                send(-0.05, max(-0.15, min(0.15, 0.8 * err))); time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            for _ in range(5):
                send(0.0, 0.0); time.sleep(0.02)
        print(f"[table_align] reversed {math.hypot(_pose['v'][0]-sx, _pose['v'][1]-sy)*100:.1f}cm "
              f"(heading {math.degrees(_pose['v'][2]):+.1f}deg)", flush=True)

    tilt, d = table_fit(a.sector)
    print(f"\n[table_align] DONE: dist={(d*100 if d is not None else float('nan')):.1f}cm "
          f"tilt={(math.degrees(tilt) if tilt is not None else 0):+.1f}deg", flush=True)


if __name__ == "__main__":
    main()
