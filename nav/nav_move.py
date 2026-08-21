#!/usr/bin/env python3
"""Closed-loop base move to a map-frame waypoint using slamware's ABSOLUTE pose (rt/robot_pose),
with a 2D-LiDAR safety layer. Drift-free: it drives until the absolute pose reaches the target,
so wheel slip / dead-reckoning error does not accumulate (unlike open-loop cmd_vel + odom).

Runs in the robot's ai mode (unitree_sdk2py / bare DDS), so use the `tv` conda env:
  python nav_move.py --name pick_table         # DRY-RUN (no motion)
  python nav_move.py --name pick_table --go     # drive there
  python nav_move.py --x 1.2 --y -0.4 --yaw 90 --go

Obstacle handling (2D LiDAR forward sector, base-mounted -> not occluded by a held object):
  clear            -> drive to goal
  obstacle ahead   -> steer toward the freer side if there is lateral clearance ("plan around")
  fully blocked    -> stop and WAIT until it clears, then resume
Ctrl-C stops the base.
"""
import argparse
import json
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy"))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Twist_ as mkTwist
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_, PoseStamped_
from laserscan_idl import LaserScan_

WP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoints.json")


def yaw_of(o):
    return math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))


def ang_norm(a):
    """wrap radians to [-pi, pi]"""
    return math.atan2(math.sin(a), math.cos(a))


class Base:
    def __init__(self, iface):
        ChannelFactoryInitialize(0, iface)
        self.pub = ChannelPublisher("rt/cmd_vel_no_limit", Twist_); self.pub.Init()
        self.pose_sub = ChannelSubscriber("rt/robot_pose", PoseStamped_); self.pose_sub.Init()
        self.scan_sub = ChannelSubscriber("rt/slamware_ros_sdk_server_node/scan", LaserScan_); self.scan_sub.Init()
        self.pose = None          # (x, y, yaw_rad), map frame, ABSOLUTE
        self.pose_t = 0.0
        self.beams = None         # list of (bearing_rad, range_m); bearing 0 = robot forward, +CCW/left
        self._run = True
        threading.Thread(target=self._pose_loop, daemon=True).start()
        threading.Thread(target=self._scan_loop, daemon=True).start()

    def _pose_loop(self):
        while self._run:
            m = self.pose_sub.Read()
            if m is not None:
                self.pose = (float(m.pose.position.x), float(m.pose.position.y), float(yaw_of(m.pose.orientation)))
                self.pose_t = time.time()
            time.sleep(0.01)

    def _scan_loop(self):
        while self._run:
            s = self.scan_sub.Read()
            if s is not None:
                b = []
                a = s.angle_min
                for r in s.ranges:
                    if s.range_min < r < s.range_max and not math.isinf(r) and not math.isnan(r):
                        b.append((ang_norm(a), float(r)))
                    a += s.angle_increment
                self.beams = b
            time.sleep(0.02)

    def min_range(self, center_deg, half_deg):
        """nearest return within +/- half_deg of center_deg (0=forward, +left). None if no beams."""
        if not self.beams:
            return None
        c = math.radians(center_deg); h = math.radians(half_deg)
        vals = [r for (bg, r) in self.beams if abs(ang_norm(bg - c)) <= h]
        return min(vals) if vals else None

    def send(self, vx, wz):
        t = mkTwist()
        t.linear.x = float(vx); t.angular.z = float(wz)
        self.pub.Write(t)

    def stop(self):
        self.send(0.0, 0.0)

    def close(self):
        self._run = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp2s0")
    ap.add_argument("--name", help="waypoint name from waypoints.json")
    ap.add_argument("--x", type=float); ap.add_argument("--y", type=float)
    ap.add_argument("--yaw", type=float, help="final heading, DEGREES (map frame)")
    ap.add_argument("--capture", metavar="NAME", help="save the robot's CURRENT absolute pose as this waypoint (global frame), then exit")
    ap.add_argument("--turn-after", type=float, default=0.0, help="after arriving, rotate in place this many deg (+CCW)")
    ap.add_argument("--back-after", type=float, default=0.0, help="after the turn, reverse this many m (closed-loop)")
    ap.add_argument("--go", action="store_true", help="ACTUALLY drive (robot MOVES). Omit = dry-run.")
    # tolerances / gains
    ap.add_argument("--pos-tol", type=float, default=0.05, help="position tolerance (m)")
    ap.add_argument("--yaw-tol", type=float, default=3.0, help="final heading tolerance (deg)")
    ap.add_argument("--vmax", type=float, default=0.22, help="max linear speed (m/s)")
    ap.add_argument("--wmax", type=float, default=0.5, help="max angular speed (rad/s)")
    ap.add_argument("--vmin", type=float, default=0.04, help="min linear speed to beat stiction (m/s)")
    ap.add_argument("--k-lin", type=float, default=0.7)
    ap.add_argument("--k-ang", type=float, default=1.3)
    ap.add_argument("--face-gate", type=float, default=25.0, help="only drive forward when heading err < this (deg)")
    ap.add_argument("--accel", type=float, default=0.35, help="max linear accel (m/s^2) for slew limiting")
    # obstacle layer
    ap.add_argument("--fwd-half", type=float, default=30.0, help="forward safety sector half-width (deg)")
    ap.add_argument("--stop-dist", type=float, default=0.60, help="stop if forward obstacle nearer than this (m)")
    ap.add_argument("--caution-dist", type=float, default=1.5, help="slow down below this forward clearance (m)")
    ap.add_argument("--side-center", type=float, default=45.0, help="left/right avoidance sector center (deg)")
    ap.add_argument("--side-half", type=float, default=25.0)
    ap.add_argument("--side-clear", type=float, default=1.3, help="a side counts as open if clearance > this (m)")
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    # capture the current ABSOLUTE pose as a waypoint (global frame), then exit -- no motion
    if a.capture:
        b = Base(a.iface)
        p = None; t0 = time.time()
        while p is None and time.time() - t0 < 5:
            p = b.pose; time.sleep(0.05)
        b.close()
        if p is None:
            print("[nav_move] no rt/robot_pose -- cannot capture"); return
        wps = json.load(open(WP_FILE)) if os.path.exists(WP_FILE) else {}
        wps[a.capture] = {"x": p[0], "y": p[1], "yaw": p[2]}
        json.dump(wps, open(WP_FILE, "w"), indent=2)
        print(f"[nav_move] captured '{a.capture}': x={p[0]:.3f} y={p[1]:.3f} yaw={math.degrees(p[2]):.1f}deg -> {WP_FILE}")
        return

    # resolve target
    if a.name:
        wps = json.load(open(WP_FILE)) if os.path.exists(WP_FILE) else {}
        if a.name not in wps:
            print(f"[nav_move] waypoint '{a.name}' not found (have: {list(wps)})"); return
        tx, ty, tyaw = wps[a.name]["x"], wps[a.name]["y"], wps[a.name]["yaw"]
    elif a.x is not None and a.y is not None:
        tx, ty = a.x, a.y
        tyaw = math.radians(a.yaw) if a.yaw is not None else None
    else:
        print("[nav_move] give --name or --x --y [--yaw]"); return

    b = Base(a.iface)
    t0 = time.time()
    while b.pose is None and time.time() - t0 < 5:
        time.sleep(0.05)
    if b.pose is None:
        print("[nav_move] no rt/robot_pose (check env/robot)"); b.close(); return

    print(f"[nav_move] start pose ({b.pose[0]:.3f},{b.pose[1]:.3f},{math.degrees(b.pose[2]):.1f}deg) "
          f"-> target ({tx:.3f},{ty:.3f}," + (f"{math.degrees(tyaw):.1f}deg)" if tyaw is not None else "any yaw)"))
    print(f"[nav_move] {'DRIVING' if a.go else 'DRY-RUN (no motion)'}  pos_tol={a.pos_tol}m yaw_tol={a.yaw_tol}deg", flush=True)

    dt = 0.05
    v_prev = 0.0
    phase = "align"  # align (face target) -> drive -> yaw (final heading) -> done
    last = 0.0
    arrived = False
    try:
        while time.time() - t0 < a.timeout:
            loop = time.time()
            if time.time() - b.pose_t > 0.5:      # stale pose -> safe stop
                if a.go:
                    b.stop()
                v_prev = 0.0
                time.sleep(dt); continue

            px, py, pyaw = b.pose
            dx, dy = tx - px, ty - py
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)
            head_err = ang_norm(bearing - pyaw)

            # ---- obstacle sensing (forward sector) ----
            fwd = b.min_range(0.0, a.fwd_half)
            left = b.min_range(a.side_center, a.side_half)
            right = b.min_range(-a.side_center, a.side_half)
            fwd_v = fwd if fwd is not None else 99.0

            vx = wz = 0.0
            action = ""

            # ---- goal reached? ----
            if dist <= a.pos_tol:
                if tyaw is None or abs(ang_norm(tyaw - pyaw)) <= math.radians(a.yaw_tol):
                    if a.go:
                        b.stop()
                    print(f"\n[nav_move] ARRIVED at ({px:.3f},{py:.3f},{math.degrees(pyaw):.1f}deg)", flush=True)
                    arrived = True
                    break
                phase = "yaw"

            if phase == "yaw":
                yaw_err = ang_norm(tyaw - pyaw)
                wz = max(-a.wmax, min(a.wmax, a.k_ang * yaw_err))
                action = f"final-yaw err={math.degrees(yaw_err):+.1f}"
            else:
                # decide ALIGN (rotate in place) vs DRIVE first. Rotation is safe -- never block it on a
                # forward obstacle. Obstacles only matter when driving, and only if they're genuinely
                # BETWEEN us and the target (closer than the remaining distance + footprint margin);
                # something beyond the target is not in the way.
                margin = 0.20
                if abs(head_err) > math.radians(a.face_gate) and dist > a.pos_tol:
                    phase = "align"
                    wz = max(-a.wmax, min(a.wmax, a.k_ang * head_err)); vx = 0.0
                    action = f"align head_err={math.degrees(head_err):+.1f}"
                else:
                    phase = "drive"
                    blocked = fwd_v < a.stop_dist and fwd_v < dist + margin
                    if blocked:
                        l = left if left is not None else 99.0
                        r = right if right is not None else 99.0
                        if max(l, r) > a.side_clear:
                            side = 1.0 if l >= r else -1.0
                            wz = side * a.wmax * 0.6
                            vx = max(0.0, min(a.vmin * 1.5, 0.06))
                            action = f"AVOID -> {'left' if side > 0 else 'right'} (fwd {fwd_v:.2f} L{l:.2f} R{r:.2f})"
                        else:
                            vx = wz = 0.0
                            action = f"BLOCKED, WAITING (fwd {fwd_v:.2f})"
                    else:
                        vx = min(a.vmax, a.k_lin * dist)
                        if fwd_v < a.caution_dist and fwd_v < dist + margin:   # slow only for in-path obstacles
                            vx *= max(0.2, (fwd_v - a.stop_dist) / (a.caution_dist - a.stop_dist))
                        vx = max(a.vmin, vx) if dist > a.pos_tol else 0.0
                        wz = max(-a.wmax, min(a.wmax, a.k_ang * head_err))
                        action = f"drive d={dist:.3f} head={math.degrees(head_err):+.1f} fwd={fwd_v:.2f}"

            # slew-limit linear
            dv = vx - v_prev
            max_dv = a.accel * dt
            vx = v_prev + max(-max_dv, min(max_dv, dv))
            v_prev = vx

            if a.go:
                b.send(vx, wz)

            if time.time() - last > 0.2:
                last = time.time()
                print(f"\r[nav_move] {phase:5s} pose({px:6.2f},{py:6.2f},{math.degrees(pyaw):6.1f}) "
                      f"d={dist:5.3f} v={vx:+.2f} w={wz:+.2f} | {action}      ", end="", flush=True)

            sleep = dt - (time.time() - loop)
            if sleep > 0:
                time.sleep(sleep)
        else:
            if a.go:
                b.stop()
            print("\n[nav_move] TIMEOUT", flush=True)

        # ---- post-arrival maneuver: rotate in place, then reverse (e.g. return-home dock) ----
        if arrived and a.go and (abs(a.turn_after) > 0.1 or a.back_after > 0.001):
            if abs(a.turn_after) > 0.1:
                tgt = ang_norm(b.pose[2] + math.radians(a.turn_after))
                print(f"[nav_move] rotate-after {a.turn_after:+.0f} deg ...", flush=True)
                t1 = time.time()
                while time.time() - t1 < 30:
                    err = ang_norm(tgt - b.pose[2])
                    if abs(err) <= math.radians(a.yaw_tol):
                        break
                    b.send(0.0, max(-a.wmax, min(a.wmax, 1.3 * err)))
                    time.sleep(dt)
                b.stop()
                print(f"[nav_move]   heading now {math.degrees(b.pose[2]):.1f} deg", flush=True)
            if a.back_after > 0.001:
                # NOTE: the 2D LiDAR is rear-blind -- keep this short/slow and supervise
                print(f"[nav_move] reverse {a.back_after*100:.0f} cm (rear-blind, slow) ...", flush=True)
                sx, sy = b.pose[0], b.pose[1]
                t1 = time.time()
                while time.time() - t1 < 30:
                    if math.hypot(b.pose[0] - sx, b.pose[1] - sy) >= a.back_after:
                        break
                    b.send(-min(0.08, a.vmax * 0.5), 0.0)
                    time.sleep(dt)
                b.stop()
                print(f"[nav_move]   reversed {math.hypot(b.pose[0]-sx, b.pose[1]-sy)*100:.1f} cm", flush=True)
    except KeyboardInterrupt:
        print("\n[nav_move] interrupted", flush=True)
    finally:
        if a.go:
            b.stop()
        b.close()


if __name__ == "__main__":
    main()
