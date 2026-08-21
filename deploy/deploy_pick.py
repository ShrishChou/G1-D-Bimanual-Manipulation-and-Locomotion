"""Full approach + pick orchestrator (one process, one mode).

Sequence (SPACE advances each step; q/Ctrl-C aborts safely):
  1) base forward  --fwd0 (30 cm)         cmd_vel + odom
  2) base turn LEFT --turn-deg (90 deg)   cmd_vel + odom
  3) arm -> inference START pose          G1Interface (episode start), then HELD by the arm controller
  4) trunk up --trunk-up (3 in)           cmd_hispeed.z + hispeed_state.y
  5) base forward --fwd1 (5 cm)           cmd_vel + odom (arm keeps holding)
  6) run the pick INFERENCE               PolicyClient loop (governance box), --lift-stop / --steps / q to stop

Base (AGV cmd_vel), trunk (cmd_hispeed), and arm/hands (G1Interface) are independent channels, so they coexist
in the robot's normal 'ai' mode -- no mode switching. Requires the model server + image server up (like run_policy).

    conda activate <env>
    python $REPO/deploy/deploy_pick.py --episode 46
"""
import argparse
import json
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/deploy")
from policy_client import PolicyClient, Governance  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Twist_ as mkTwist  # noqa: E402
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Point32_ as mkPoint32  # noqa: E402
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_, Point32_, PoseStamped_  # noqa: E402
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_  # noqa: E402
from laserscan_idl import LaserScan_  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--iface", default="enp2s0")
p.add_argument("--episode", type=int, default=46, help="episode whose START pose the arm inits to")
p.add_argument("--port", type=int, default=8000)
# --- Path A: drift-free absolute SLAM waypoints (rt/robot_pose closed-loop) instead of open-loop odom moves ---
p.add_argument("--waypoints", action="store_true",
               help="drive to absolute map-frame poses from --wp-file (drift-free) instead of open-loop base_drive/turn")
p.add_argument("--wp-file", default=f"{ROOT}/nav/waypoints.json", help="waypoint plan (from nav/ai_waypoints.py)")
p.add_argument("--wp-pos-tol", type=float, default=0.05, help="waypoint arrival tolerance (m)")
p.add_argument("--wp-yaw-tol", type=float, default=3.0, help="waypoint final-heading tolerance (deg)")
p.add_argument("--wp-vmax", type=float, default=0.22, help="max linear speed for waypoint drive (m/s)")
p.add_argument("--wp-wmax", type=float, default=0.5, help="max angular speed for waypoint drive (rad/s)")
p.add_argument("--wp-stop-dist", type=float, default=0.70, help="stop the waypoint drive if a forward obstacle is nearer (m)")
p.add_argument("--no-pick-align", action="store_true",
               help="disable the LiDAR square-up + distance correction at P2 (waypoint mode)")
p.add_argument("--pick-table-dist", type=float, default=0.0,
               help="target forward distance to the table face at P2 (m); 0 = heading square-up only. "
                    "Must be > LiDAR range_min (~0.77m) to be measurable.")
# --- table-referenced final approach (reliable, immune to SLAM drift): after P1, square up + closed-loop
#     to --table-stop from the table face (LiDAR), then a FIXED open-loop --table-nudge to seat at the pick ---
p.add_argument("--table-approach", action="store_true",
               help="waypoint mode: replace the SLAM drive to P2 with a table-referenced approach (square up + "
                    "LiDAR closed-loop to --table-stop + fixed --table-nudge). Reliable regardless of SLAM drift.")
p.add_argument("--table-stop", type=float, default=0.30,
               help="LiDAR distance to the table face to stop at (m). Measured: 0.30 = front flush against the "
                    "table, LiDAR min ~0.296. Default 0.30 = flush; raise for a gap.")
p.add_argument("--table-nudge", type=float, default=0.0,
               help="OPTIONAL fixed open-loop forward after the LiDAR stop (m); usually 0 since the LiDAR "
                    "closed-loop reaches the pick distance directly.")
# approach geometry
p.add_argument("--fwd0", type=float, default=0.60, help="initial forward (m)")
p.add_argument("--turn-deg", type=float, default=90.0, help="turn LEFT this many degrees (validated via IMU yaw)")
p.add_argument("--trunk-up", type=float, default=0.1016, help="raise trunk (m); 0.1016 = 4 in (lowered 1 in from the original 5 in)")
p.add_argument("--fwd1", type=float, default=0.559, help="final forward after arm init (m); 10 in + 1 ft")
p.add_argument("--base-vel", type=float, default=0.05, help="base linear speed (m/s)")
p.add_argument("--final-speed-scale", type=float, default=0.8, help="speed scale for the final forward only (slower/controlled)")
p.add_argument("--wvel", type=float, default=0.3, help="base turn speed (rad/s)")
p.add_argument("--trunk-speed", type=float, default=1.0, help="trunk velocity cmd (0..1)")
p.add_argument("--align", action="store_true", help="do the LiDAR square-up after the turn (off for now)")
# LiDAR square-up to the table (after the turn)
p.add_argument("--front-deg", type=float, default=0.0, help="LiDAR bearing that points robot-forward")
p.add_argument("--align-sector", type=float, default=40.0, help="half-width (deg) of the front sector used to fit the table face")
p.add_argument("--align-tol", type=float, default=1.5, help="stop squaring up when |tilt| below this (deg)")
p.add_argument("--align-max-step", type=float, default=15.0, help="max rotation per align iteration (deg)")
p.add_argument("--align-flip", action="store_true", help="invert the square-up rotation direction if it turns the wrong way")
p.add_argument("--scan-topic", default="rt/slamware_ros_sdk_server_node/scan")
# arm / policy (mirror run_policy)
p.add_argument("--hz", type=float, default=15.0)
p.add_argument("--requery", type=int, default=8)
p.add_argument("--max-joint-step", type=float, default=0.01)
p.add_argument("--hand-step", type=float, default=0.02)
p.add_argument("--down-limit", type=float, default=0.18)
p.add_argument("--box-margin", type=float, default=0.30)
p.add_argument("--motion", action="store_true", help="rt/arm_sdk (AI mode). Omit for rt/lowcmd.")
p.add_argument("--no-hand", action="store_true")
p.add_argument("--gravity", action="store_true")
p.add_argument("--steps", type=int, default=100000)
p.add_argument("--lift-stop", type=float, default=0.0, help="stop the policy once hands rise this many m; then hold grasp")
# --- pick (ported run_heatmap loop: async + temporal ensembling) ---
p.add_argument("--start-file", default="$ASSETS/avg_start_new.npy",
               help="avg start-pose .npy (14-dim arm) the arm homes to")
p.add_argument("--slew", type=float, default=1.0, help="policy arm march per step (rad)")
p.add_argument("--hand-step-pick", type=float, default=0.75, help="grip march per step")
p.add_argument("--ens-coeff", type=float, default=0.4, help="temporal-ensembling weight (exp(-coeff*age))")
p.add_argument("--pick-hz", type=float, default=50.0, help="pick control-loop rate")
p.add_argument("--max-run-steps", type=int, default=500, help="pick loop length before the post-pick sequence")
# --- fw2: gentle align-to-table nudge before the pick ---
p.add_argument("--fw2", type=float, default=0.01, help="slow forward nudge to align to the table (m)")
p.add_argument("--fw2-vel", type=float, default=0.02, help="very slow speed for the fw2 nudge (m/s)")
# --- post-pick sequence (arm holds the grasp throughout) ---
p.add_argument("--post-trunk-up", type=float, default=0.0762, help="post-pick trunk up (m); 3 in")
p.add_argument("--post-back1", type=float, default=0.30, help="post-pick slow back-off (m)")
p.add_argument("--post-reverse", type=float, default=0.45, help="waypoint mode: straight reverse after the pick (m), NO rotation")
p.add_argument("--post-trunk-down", type=float, default=0.1016, help="post-pick trunk down (m); 0.1016 = 4 in")
p.add_argument("--post-back2", type=float, default=0.30, help="post-pick back (m)")
p.add_argument("--post-fwd", type=float, default=0.981, help="post-pick forward after rotating to start heading (m)")
p.add_argument("--post-trunk-up2", type=float, default=0.2921, help="post-pick trunk up (m); 11.5 in")
p.add_argument("--post-fwd2", type=float, default=0.4318, help="post-pick very slow final forward (m); 17 in")
p.add_argument("--yaw-tol", type=float, default=1.5, help="deg tolerance when rotating back to the start heading")
# --- drop (release the object) ---
p.add_argument("--drop-trunk-down", type=float, default=0.1016, help="lower trunk at the drop (m); 4 in")
p.add_argument("--drop-spread", type=float, default=0.15, help="(legacy) lateral palm spread at the drop (m); unused by the rotate-outward release")
p.add_argument("--drop-rot1", type=float, default=0.20, help="drop: first OUTWARD rotation of the shoulder_roll joint (rad)")
p.add_argument("--drop-rot2", type=float, default=0.30, help="drop: further OUTWARD shoulder_roll rotation after the trunk dip (rad)")
p.add_argument("--drop-yaw", type=float, default=0.30, help="drop: OUTWARD rotation of the shoulder_yaw joint (next under shoulder_roll) (rad)")
p.add_argument("--drop-dip", type=float, default=0.03, help="drop: small trunk dip (m) between the two outward rotations")
# --- post-drop: back off, lower trunk fully, return to home (waypoints path) ---
p.add_argument("--post-drop-reverse", type=float, default=0.40, help="after the drop + arms-out: reverse this far (m) before homing")
p.add_argument("--hands-lower", type=float, default=0.20, help="after homing prep: lower the hands this far (m) to a rest position")
p.add_argument("--post-drop-trunk-down", type=float, default=0.25, help="post-drop: lower trunk this far (m) before homing")
p.add_argument("--home-back", type=float, default=0.12, help="homing: reverse this far (m) into the charger after the 180 turn")
p.add_argument("--home-far-gap", type=float, default=0.45, help="homing: obstacle-avoided move stops this far (m) short of home_stage (clear of the charger)")
args = p.parse_args()

dt = 1.0 / args.hz
STOP = threading.Event()
g1 = None
_OLD = None


def _hard_stop(*_):
    signal.alarm(2)
    try:
        stop_base()
    except Exception:
        pass
    try:
        if _OLD is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _OLD)
    except Exception:
        pass
    try:
        if g1 is not None:
            g1.stop()
    except Exception:
        pass
    print("\n[deploy] stopped", flush=True)
    os._exit(0)


# ---- base + trunk channels (created after G1Interface brings up the DDS factory) ----
base_pub = trunk_pub = None
_pose = {"v": None}       # (x, y, yaw) from AGV odom (drifts -- used by open-loop base_drive/turn)
_slam = {"v": None}       # (x, y, yaw) from rt/robot_pose (ABSOLUTE slamware SLAM -- drift-free waypoints)
_trunk = {"y": None}      # trunk height (m)
_trunk_plan = {"h": None} # ABSOLUTE trunk target plan (m), anchored to the start height; never < 0
_scan = {"m": None}       # latest LaserScan
_imu = {"yaw": None}      # IMU yaw (rad), for exact turns
WP = {}                   # loaded waypoint plan (map frame) when --waypoints


def _yaw(o):
    return math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))


def send_vel(vx, wz):
    m = mkTwist(); m.linear.x = float(vx); m.angular.z = float(wz); base_pub.Write(m)


def stop_base():
    if base_pub is not None:
        for _ in range(5):
            send_vel(0.0, 0.0); time.sleep(0.02)


def send_trunk(z):
    m = mkPoint32(); m.x = 0.0; m.y = 0.0; m.z = float(z); trunk_pub.Write(m)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class KeyPoller:
    def __enter__(self):
        global _OLD
        self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); _OLD = self.old
        tty.setcbreak(self.fd); return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


def wait_space(poller, label):
    print(f"\n[deploy] SPACE -> {label}   (q aborts)", flush=True)
    while True:
        k = poller.poll()
        if k == " ":
            return True
        if k in ("q", "\x1b"):
            return False
        time.sleep(0.02)


def base_drive(dist, poller, vel=None):
    vel = args.base_vel if vel is None else vel
    for _ in range(200):
        if _pose["v"] is not None:
            break
        time.sleep(0.01)
    if _pose["v"] is None:
        print("[deploy] no odom -- skipping drive"); return
    x0, y0, _ = _pose["v"]; sgn = 1.0 if dist >= 0 else -1.0
    print(f"[deploy] base drive {dist*100:+.0f} cm at {vel:.3f} m/s ...", flush=True)
    while math.hypot(_pose["v"][0] - x0, _pose["v"][1] - y0) < abs(dist):
        send_vel(sgn * vel, 0.0)
        if poller.poll() in ("q", "\x1b"):
            break
        time.sleep(0.05)
    stop_base()
    print(f"[deploy]   moved {math.hypot(_pose['v'][0]-x0, _pose['v'][1]-y0)*100:.1f} cm", flush=True)


def turn_by(rad, poller):
    """Rotate by `rad` (>0 = left/CCW) closed-loop on odom yaw."""
    if abs(rad) < math.radians(0.5) or _pose["v"] is None:
        return
    y0 = _pose["v"][2]; target = abs(rad); wz = abs(args.wvel) * (1 if rad > 0 else -1)
    while abs(_wrap(_pose["v"][2] - y0)) < target:
        send_vel(0.0, wz)
        if poller.poll() in ("q", "\x1b"):
            break
        time.sleep(0.05)
    stop_base()


def turn_to_yaw(target_yaw, poller):
    """Rotate (shortest way) until IMU yaw == target_yaw. Returns to the exact start heading."""
    src = (lambda: _imu["yaw"]) if _imu["yaw"] is not None else (lambda: _pose["v"][2] if _pose["v"] else None)
    if src() is None:
        print("[deploy] no yaw source -- skipping rotate-to-start"); return
    tol = math.radians(args.yaw_tol)
    print(f"[deploy] rotate to start heading ({math.degrees(target_yaw):+.1f} deg) ...", flush=True)
    while True:
        err = _wrap(src() - target_yaw)
        if abs(err) < tol:
            break
        send_vel(0.0, -math.copysign(abs(args.wvel), err))    # err>0 -> rotate right (neg yaw) to reduce
        if poller.poll() in ("q", "\x1b"):
            break
        time.sleep(0.05)
    stop_base()
    print(f"[deploy]   heading now {math.degrees(src()):+.1f} deg (target {math.degrees(target_yaw):+.1f})", flush=True)


def turn_relative(deg, poller):
    """Rotate in place by a RELATIVE angle, FORCING the direction (sign of deg: +CCW/left, -CW/right).
    Accumulates the actual yaw change so a 180 turn goes the intended way -- shortest-path is ambiguous
    at 180 and picks a direction arbitrarily."""
    def src():
        return _imu["yaw"] if _imu["yaw"] is not None else (_pose["v"][2] if _pose["v"] else None)
    if src() is None:
        print("[deploy] no yaw source -- skipping relative turn"); return
    target = math.radians(abs(deg))
    wz = math.copysign(abs(args.wvel), deg)          # + = CCW/left, - = CW/right
    prev = src(); acc = 0.0
    print(f"[deploy] turn {'LEFT' if deg > 0 else 'RIGHT'} {abs(deg):.0f} deg (forced direction) ...", flush=True)
    t0 = time.time()
    while acc < target and time.time() - t0 < 30:
        cur = src()
        acc += abs(_wrap(cur - prev)); prev = cur
        send_vel(0.0, wz)
        if poller.poll() in ("q", "\x1b"):
            break
        time.sleep(0.05)
    stop_base()
    print(f"[deploy]   turned ~{math.degrees(acc):.0f} deg", flush=True)


def _fwd_clearance(half_deg=30.0):
    """nearest LiDAR return within +/- half_deg of robot-forward (0 deg); None if no scan."""
    m = _scan["m"]
    if m is None:
        return None
    h = math.radians(half_deg)
    best = None
    a = m.angle_min
    for r in m.ranges:
        if m.range_min < r < m.range_max and not math.isinf(r) and not math.isnan(r):
            if abs(_wrap(a)) <= h:
                best = r if best is None else min(best, r)
        a += m.angle_increment
    return best


def goto_waypoint(name, poller, avoid=False, timeout=90.0):
    """Drive the base to an ABSOLUTE map-frame pose (WP[name]) closed-loop on rt/robot_pose (drift-free).
      avoid=False (default): OBJECT-AGNOSTIC -- ignore the LiDAR entirely (for moves where tables/walls are
        close and would falsely block).
      avoid=True: obstacle-aware -- stop/WAIT for returns genuinely between us and the target.
    Logs clearly if BLOCKED (waiting on an obstacle), STUCK (no progress), or TIMEOUT."""
    if name not in WP:
        print(f"[deploy][nav] waypoint '{name}' missing from {args.wp_file}"); return False
    tx, ty, tyaw = WP[name]["x"], WP[name]["y"], WP[name]["yaw"]
    for _ in range(300):
        if _slam["v"] is not None:
            break
        time.sleep(0.01)
    if _slam["v"] is None:
        print("[deploy][nav] no rt/robot_pose -- cannot run waypoint drive"); return False

    print(f"[deploy][nav] goto '{name}' -> ({tx:.3f},{ty:.3f},{math.degrees(tyaw):.1f}deg) "
          f"[{'OBSTACLE-AWARE' if avoid else 'object-agnostic'}]", flush=True)
    pos_tol = args.wp_pos_tol; yaw_tol = math.radians(args.wp_yaw_tol)
    vmax, wmax, vmin = args.wp_vmax, args.wp_wmax, 0.04
    face_gate = math.radians(25.0); v_prev = 0.0; dt = 0.05
    waiting = False
    t_start = time.time()
    best_dist = float("inf"); t_progress = time.time(); stuck_logged = False
    while True:
        px, py, pyaw = _slam["v"]
        dx, dy = tx - px, ty - py
        dist = math.hypot(dx, dy)

        if time.time() - t_start > timeout:
            stop_base()
            print(f"\n[deploy][nav] TIMEOUT on '{name}' after {timeout:.0f}s (still {dist:.2f}m away). Aborting move.", flush=True)
            return False
        # progress / stuck watchdog
        if dist < best_dist - 0.02:
            best_dist = dist; t_progress = time.time(); stuck_logged = False
        elif time.time() - t_progress > 6.0 and not stuck_logged and dist > pos_tol:
            print(f"\n[deploy][nav] STUCK on '{name}': no progress for 6s at ({px:.2f},{py:.2f}), "
                  f"{dist:.2f}m still to go (blocked / wheels slipping / bad localization?)", flush=True)
            stuck_logged = True

        # arrived?
        if dist <= pos_tol:
            yaw_err = _wrap(tyaw - pyaw)
            if abs(yaw_err) <= yaw_tol:
                stop_base(); print(f"[deploy][nav]   arrived '{name}' ({px:.3f},{py:.3f},{math.degrees(pyaw):.1f}deg)", flush=True)
                return True
            send_vel(0.0, max(-wmax, min(wmax, 1.3 * yaw_err)))
            if poller.poll() in ("q", "\x1b"):
                stop_base(); print("[deploy][nav] aborted by key", flush=True); return False
            time.sleep(dt); continue
        bearing = math.atan2(dy, dx); head_err = _wrap(bearing - pyaw)

        # obstacle safety ONLY when avoid=True: obstacle must be between us and the target, and we must be
        # roughly facing the goal (rotation is never blocked).
        fwd = _fwd_clearance() if avoid else None
        if avoid and fwd is not None and fwd < args.wp_stop_dist and fwd < dist + 0.20 and abs(head_err) < face_gate:
            if not waiting:
                print(f"\n[deploy][nav] BLOCKED on '{name}': obstacle {fwd:.2f}m ahead -- WAITING to clear", flush=True)
                waiting = True
            stop_base(); v_prev = 0.0
            if poller.poll() in ("q", "\x1b"):
                stop_base(); print("[deploy][nav] aborted by key", flush=True); return False
            time.sleep(dt); continue
        if waiting:
            print(f"[deploy][nav]   '{name}' path cleared -- resuming", flush=True); waiting = False

        if abs(head_err) > face_gate:                       # rotate to face target first
            vx, wz = 0.0, max(-wmax, min(wmax, 1.3 * head_err))
        else:                                               # drive with heading correction
            vx = max(vmin, min(vmax, 0.7 * dist))
            if avoid and fwd is not None and fwd < args.wp_stop_dist + 0.6:   # ease off near obstacles
                vx *= max(0.2, (fwd - args.wp_stop_dist) / 0.6)
            wz = max(-wmax, min(wmax, 1.3 * head_err))
        vx = v_prev + max(-0.35 * dt, min(0.35 * dt, vx - v_prev)); v_prev = vx   # slew
        send_vel(vx, wz)
        if poller.poll() in ("q", "\x1b"):
            stop_base(); print("[deploy][nav] aborted by key", flush=True); return False
        time.sleep(dt)


def table_tilt():
    """Fit the table face in the front LiDAR sector; return its tilt (rad, 0 = square to the robot), or None."""
    m = _scan["m"]
    if m is None:
        return None
    r = np.array(m.ranges, float)
    ang = m.angle_min + np.arange(len(r)) * m.angle_increment
    d = np.abs((ang - math.radians(args.front_deg) + math.pi) % (2 * math.pi) - math.pi)
    sel = (d <= math.radians(args.align_sector)) & np.isfinite(r) & (r > 0.1) & (r < 3.0)   # fixed floor, not range_min (tracks nearest)
    if sel.sum() < 8:
        return None
    a = ang[sel] - math.radians(args.front_deg)          # relative to forward
    rr = r[sel]
    x = rr * np.cos(a); y = rr * np.sin(a)               # x = forward distance, y = lateral
    yc, xc = y - y.mean(), x - x.mean()
    slope = float((yc * xc).sum() / ((yc * yc).sum() + 1e-9))   # x = slope*y + c ; slope 0 = square
    return math.atan(slope)


def align_to_table(poller):
    flip = -1.0 if args.align_flip else 1.0
    for _ in range(200):
        if _scan["m"] is not None:
            break
        time.sleep(0.01)
    for _ in range(8):
        tilt = table_tilt()
        if tilt is None:
            print("\n[deploy] no table face in the front sector -- skipping square-up", flush=True); return
        td = math.degrees(tilt)
        print(f"[deploy] table tilt {td:+.1f} deg", flush=True)
        if abs(td) <= args.align_tol:
            break
        step = max(-math.radians(args.align_max_step), min(math.radians(args.align_max_step), flip * tilt))
        turn_by(step, poller)
        time.sleep(0.4)                                   # settle + fresh scan
    print(f"[deploy] squared up (tilt {math.degrees(table_tilt() or 0):+.1f} deg)", flush=True)


def table_distance():
    """Median forward distance (m) to the table face in the front LiDAR sector, or None if not visible
    (e.g. the table is inside the ~0.77m LiDAR blind zone, or no flat face is found)."""
    m = _scan["m"]
    if m is None:
        return None
    r = np.array(m.ranges, float)
    ang = m.angle_min + np.arange(len(r)) * m.angle_increment
    d = np.abs((ang - math.radians(args.front_deg) + math.pi) % (2 * math.pi) - math.pi)
    sel = (d <= math.radians(args.align_sector)) & np.isfinite(r) & (r > 0.1) & (r < 3.0)   # fixed floor, not range_min (tracks nearest)
    if sel.sum() < 8:
        return None
    a = ang[sel] - math.radians(args.front_deg)
    return float(np.median(r[sel] * np.cos(a)))            # forward component


def pick_align(poller):
    """Local LiDAR correction at the pick pose: square the heading to the table face (fixes the rotational
    part of the SLAM drift) and, if --pick-table-dist is set, nudge fore/aft to that table distance.
    Degrades gracefully: if the table is in the LiDAR blind zone (<~0.77m) it can't be seen -> we skip and
    rely on the visuomotor policy (it grasps from the cameras) for the fine correction."""
    print("[deploy][align] LiDAR square-up at pick pose ...", flush=True)
    if table_tilt() is None:
        print("[deploy][align] table not visible in the LiDAR (blind zone <0.77m or no flat face) -- "
              "skipping local align; the visuomotor policy will correct from the cameras.", flush=True)
        return
    align_to_table(poller)                                  # heading
    d0 = table_distance()
    print(f"[deploy][align] table distance now = {d0*100:.1f} cm" if d0 is not None else
          "[deploy][align] table distance now = (not measurable)", flush=True)
    if args.pick_table_dist > 0 and d0 is not None:
        target = args.pick_table_dist
        print(f"[deploy][align] correcting fore/aft to {target*100:.0f} cm ...", flush=True)
        t0 = time.time()
        while time.time() - t0 < 20:
            d = table_distance()
            if d is None:
                print("[deploy][align] lost the table face -- stopping fore/aft correction", flush=True); break
            err = d - target                               # >0 too far -> forward; <0 too close -> back
            if abs(err) < 0.01:
                break
            send_vel(math.copysign(min(0.05, 0.5 * abs(err) + 0.02), err), 0.0)
            if poller.poll() in ("q", "\x1b"):
                break
            time.sleep(0.05)
        stop_base()
        df = table_distance()
        print(f"[deploy][align] table distance = {df*100:.1f} cm (target {target*100:.0f})" if df else
              "[deploy][align] fore/aft done", flush=True)


def approach_table(poller):
    """Drive FORWARD toward the table while CONTINUOUSLY squaring up to its face, stopping flush at
    --table-stop (LiDAR distance to the table face). 'Move forward while aligning' -> arrives straight and
    flush. Table-referenced -> immune to the ~8cm SLAM drift. Needs the table ahead in the front LiDAR
    sector at the start. q/Ctrl-C aborts; returns True on arrival."""
    for _ in range(200):
        if _scan["m"] is not None:
            break
        time.sleep(0.01)
    if table_tilt() is None:
        print("[deploy][table] table not visible in the front LiDAR sector -- aim the robot roughly at the "
              "table (face ahead) and retry.", flush=True)
        return False
    flip = -1.0 if args.align_flip else 1.0
    stop_d = args.table_stop
    print(f"[deploy][table] approaching table to {stop_d*100:.0f}cm while squaring up ...", flush=True)
    t0 = time.time(); last = 0.0; lost = 0
    while time.time() - t0 < 60:
        d = table_distance()
        if d is None:                                   # face slipped out of the sector / too close
            lost += 1
            if lost > 6:
                stop_base(); print("\n[deploy][table] lost the table face -- stopping", flush=True); break
            send_vel(0.0, 0.0); time.sleep(0.05); continue
        lost = 0
        if d - stop_d <= 0.005:                         # reached the flush distance
            break
        tilt = table_tilt()
        v = max(0.03, min(args.wp_vmax * 0.5, 0.4 * (d - stop_d)))     # ease off as it nears the table
        # low-gain P (won't saturate) + 2deg deadband + freeze rotation when close (noisy fits near the table)
        if d < 0.50 or tilt is None or abs(tilt) < math.radians(2.0):
            wz = 0.0
        else:
            wz = max(-0.20, min(0.20, -flip * 0.6 * tilt))
        send_vel(v, wz)
        if time.time() - last > 0.3:
            last = time.time()
            td = math.degrees(tilt) if tilt is not None else float("nan")
            print(f"\r[deploy][table] dist={d:.3f}m tilt={td:+.1f}deg  v={v:.2f} w={wz:+.2f}   ", end="", flush=True)
        if poller.poll() in ("q", "\x1b"):
            stop_base(); print("\n[deploy][table] aborted", flush=True); return False
        time.sleep(0.05)
    stop_base()
    df = table_distance(); tf = table_tilt()
    print(f"\n[deploy][table] arrived flush: dist={df*100:.1f}cm  tilt={(math.degrees(tf) if tf is not None else 0):+.1f}deg"
          if df is not None else "\n[deploy][table] stopped (table not measurable)", flush=True)
    if args.table_nudge > 0:
        print(f"[deploy][table] extra open-loop nudge {args.table_nudge*100:.0f}cm ...", flush=True)
        base_drive(args.table_nudge, poller, vel=args.fw2_vel)
    return True


def base_turn_left(deg, poller):
    # prefer IMU yaw for an exact turn; fall back to odom yaw
    use_imu = _imu["yaw"] is not None
    src = (lambda: _imu["yaw"]) if use_imu else (lambda: _pose["v"][2] if _pose["v"] else None)
    for _ in range(200):
        if src() is not None:
            break
        time.sleep(0.01)
    if src() is None:
        print("[deploy] no yaw source -- skipping turn"); return
    y0 = src(); yo0 = _pose["v"][2] if _pose["v"] else None
    target = math.radians(abs(deg))
    print(f"[deploy] base turn LEFT {deg:.0f} deg (via {'IMU' if use_imu else 'odom'}) ...", flush=True)
    while abs(_wrap(src() - y0)) < target:
        send_vel(0.0, abs(args.wvel))                 # +yaw = CCW = left
        if poller.poll() in ("q", "\x1b"):
            break
        time.sleep(0.05)
    stop_base()
    imu_done = math.degrees(abs(_wrap(src() - y0)))
    odo_done = math.degrees(abs(_wrap(_pose["v"][2] - yo0))) if yo0 is not None else float("nan")
    print(f"[deploy]   turned {imu_done:.1f} deg (IMU)  |  odom {odo_done:.1f} deg", flush=True)


def trunk_up(delta, poller):
    """Move the trunk by `delta` m, but tracked as an ABSOLUTE target anchored to the start height and
    capped at 0 -- the trunk is never commanded to a negative height, and targets don't drift with re-reads."""
    for _ in range(200):
        if _trunk["y"] is not None:
            break
        time.sleep(0.01)
    if _trunk["y"] is None:
        print("[deploy] no trunk state -- skipping trunk"); return
    if _trunk_plan["h"] is None:                        # anchor the absolute plan to the start position
        _trunk_plan["h"] = _trunk["y"]
    target = max(0.0, _trunk_plan["h"] + delta)         # ABSOLUTE target, capped at 0 (never negative)
    _trunk_plan["h"] = target
    cur = _trunk["y"]
    if abs(target - cur) < 0.005:
        print(f"[deploy] trunk already at {cur*100:.1f} cm (abs target {target*100:.1f}) -- no move", flush=True)
        return
    up = target > cur
    z = args.trunk_speed if up else -args.trunk_speed
    print(f"[deploy] trunk {cur*100:.1f} -> {target*100:.1f} cm (abs) ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 30:
        send_trunk(z)
        h = _trunk["y"]
        if (up and h >= target - 0.004) or (not up and h <= target + 0.004):
            break
        if poller.poll() in ("q", "\x1b"):
            break
        time.sleep(0.01)
    for _ in range(10):
        send_trunk(0.0); time.sleep(0.01)
    print(f"[deploy]   trunk at {_trunk['y']*100:.1f} cm", flush=True)


def main():
    global g1, base_pub, trunk_pub
    from live_source import G1Interface, CLOSED_L, CLOSED_R

    # refuse to start if another deploy_pick / run_heatmap (or its Dex3 spawn child) is alive -> avoids
    # multiple arm/hand controllers fighting over DDS (the jitter/no-lift failures).
    me = os.getpid()
    others = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cl = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "ignore")
        except Exception:
            continue
        if ("deploy_pick.py" in cl or "run_heatmap.py" in cl):
            others.append(pid)
    if others:
        print(f"[deploy] REFUSING: {len(others)} other deploy_pick/run_heatmap process(es) alive (pids {others}). "
              f"Kill them first:  kill -9 {' '.join(others)}", flush=True)
        return

    print("[deploy] constructing G1 interface (arm + Dex3 + cameras) ...", flush=True)
    g1 = G1Interface(motion=args.motion, hands=not args.no_hand)   # one process owns everything (pick + hold grasp)
    ik = g1.arm_ik
    pc = PolicyClient(port=args.port)

    # base + trunk channels (factory is up now)
    base_pub = ChannelPublisher("rt/cmd_vel_no_limit", Twist_); base_pub.Init()
    trunk_pub = ChannelPublisher("rt/cmd_hispeed", Point32_); trunk_pub.Init()
    odom_sub = ChannelSubscriber("rt/agv/odom", Odometry_); odom_sub.Init()
    trunk_sub = ChannelSubscriber("rt/hispeed_state", Point32_); trunk_sub.Init()
    scan_sub = ChannelSubscriber(args.scan_topic, LaserScan_); scan_sub.Init()
    imu_sub = ChannelSubscriber("rt/secondary_imu", IMUState_); imu_sub.Init()
    slam_sub = ChannelSubscriber("rt/robot_pose", PoseStamped_); slam_sub.Init()   # absolute SLAM pose (waypoints)

    if args.waypoints:
        global WP
        if not os.path.exists(args.wp_file):
            print(f"[deploy] --waypoints but {args.wp_file} missing -- run nav/ai_waypoints.py first"); return
        WP = json.load(open(args.wp_file))
        # p3 is skipped now (replaced by a straight reverse); p2 only used when NOT table-approach
        need = ["p1", "p4", "p5", "home_stage"] + ([] if args.table_approach else ["p2"])
        miss = [w for w in need if w not in WP]
        if miss:
            print(f"[deploy] waypoint file missing {miss}"); return
        # homing intermediate: a point --home-far-gap BEHIND home_stage (away from the charger) so the long
        # obstacle-avoided move stops clear of the charger; the final close-in to home_stage runs LiDAR-free.
        hs = WP["home_stage"]
        WP["home_far"] = {"x": hs["x"] - args.home_far_gap * math.cos(hs["yaw"]),
                          "y": hs["y"] - args.home_far_gap * math.sin(hs["yaw"]),
                          "yaw": hs["yaw"]}
        print(f"[deploy] WAYPOINT MODE (drift-free): loaded {list(WP)} from {args.wp_file}", flush=True)

    def _odom():
        while True:
            m = odom_sub.Read()
            if m is not None:
                q = m.pose.pose.position
                _pose["v"] = (float(q.x), float(q.y), float(_yaw(m.pose.pose.orientation)))
            time.sleep(0.005)

    def _trunk_state():
        while True:
            m = trunk_sub.Read()
            if m is not None:
                _trunk["y"] = float(m.y)
            time.sleep(0.01)

    def _scan_reader():
        while True:
            m = scan_sub.Read()
            if m is not None:
                _scan["m"] = m
            time.sleep(0.02)

    def _imu_reader():
        while True:
            m = imu_sub.Read()
            if m is not None:
                try:
                    _imu["yaw"] = float(m.rpy[2])
                except Exception:
                    pass
            time.sleep(0.005)

    def _slam_reader():
        while True:
            m = slam_sub.Read()
            if m is not None:
                _slam["v"] = (float(m.pose.position.x), float(m.pose.position.y), float(_yaw(m.pose.orientation)))
            time.sleep(0.01)

    threading.Thread(target=_odom, daemon=True).start()
    threading.Thread(target=_trunk_state, daemon=True).start()
    threading.Thread(target=_scan_reader, daemon=True).start()
    threading.Thread(target=_imu_reader, daemon=True).start()
    threading.Thread(target=_slam_reader, daemon=True).start()

    signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
    signal.signal(signal.SIGINT, _hard_stop)
    signal.signal(signal.SIGTERM, _hard_stop)

    gtau = (lambda q: g1.gravity_tau(q)) if args.gravity else (lambda q: None)
    start_arm = np.load(args.start_file).astype(np.float64)[:14]   # avg start pose

    # record the START heading now (to rotate back to it post-pick)
    for _ in range(300):
        if _imu["yaw"] is not None:
            break
        time.sleep(0.01)
    start_yaw = _imu["yaw"] if _imu["yaw"] is not None else (_pose["v"][2] if _pose["v"] else 0.0)
    print(f"[deploy] start heading = {math.degrees(start_yaw):+.1f} deg", flush=True)

    # anchor the trunk plan to its START position -- from here every trunk move is an ABSOLUTE target
    # (start + cumulative offsets), floored at 0, so heights don't drift and are never asked to go negative
    for _ in range(200):
        if _trunk["y"] is not None:
            break
        time.sleep(0.01)
    if _trunk["y"] is not None:
        _trunk_plan["h"] = _trunk["y"]
        print(f"[deploy] trunk start position = {_trunk['y']*100:.1f} cm (absolute plan; targets floored at 0)", flush=True)

    def run_pick(poller, home_first=True):
        """Ported run_heatmap pick: async worker + temporal ensembling. Returns (arm_cmd, gl, gr) held, or None."""
        HCHUNK = 16
        pick_dt = 1.0 / args.pick_hz
        shared = {"obs": None, "chunk": None, "state_q": None, "seq": 0}
        slock = threading.Lock(); stop_worker = threading.Event()

        def worker():
            while not stop_worker.is_set():
                with slock:
                    obs = shared["obs"]
                if obs is None:
                    time.sleep(0.005); continue
                h, l, r, ss = obs
                try:
                    ch = pc.query(h, l, r, ss)
                except Exception as e:
                    print(f"[infer] {type(e).__name__}: {e}", flush=True); time.sleep(0.05); continue
                with slock:
                    shared["chunk"] = ch; shared["state_q"] = np.asarray(ss, np.float64); shared["seq"] += 1

        threading.Thread(target=worker, daemon=True).start()

        if home_first:
            # home the arm to the avg start pose NOW (just before the policy) so the arm stays tucked during the
            # trunk lift + forward approach, and the policy still starts from its training start distribution.
            _, _, _, sh = g1.read(); q0 = np.asarray(sh[:14], np.float64); cmd_h = q0.copy()
            print(f"[pick] homing arm to start pose (max err {np.abs(start_arm-q0).max():.2f} rad) ...", flush=True)
            for _ in range(int(8 * args.pick_hz)):
                cmd_h += np.clip(start_arm - cmd_h, -args.max_joint_step, args.max_joint_step)
                g1.command_arm(cmd_h, tauff=gtau(cmd_h))
                _, _, _, sc = g1.read()
                if float(np.abs(start_arm - np.asarray(sc[:14], np.float64)).max()) < 0.05:
                    break
                if poller.poll() in ("q", "\x1b"):
                    stop_worker.set(); return None
                time.sleep(1.0 / args.pick_hz)
            _, _, _, sh = g1.read()
            if float(np.abs(np.asarray(sh[:14], np.float64) - q0).max()) < 0.02:
                print("[pick] *** arm did not move (limp?). Try --motion (rt/arm_sdk) vs no --motion (rt/lowcmd).",
                      flush=True)
                stop_worker.set(); return None
        else:
            # arm was already homed earlier (waypoints path homes right after the trunk raise) -- start inference now
            print("[pick] arm already homed -- starting inference directly.", flush=True)

        print(f">>> PICK running ({args.max_run_steps} steps @ {args.pick_hz:.0f}Hz). q to stop early.", flush=True)
        _, _, _, s0 = g1.read(); arm_cmd = np.asarray(s0[:14], np.float64); gl = gr = 0.0
        ens_buf = []; last_seq = -1; aborted = False; step = 0
        for step in range(args.max_run_steps):
            if poller.poll() in ("q", "\x1b"):
                aborted = True; break
            head, lw, rw, st = g1.read()
            with slock:
                shared["obs"] = (head, lw, rw, st)
                ch, sq, seq = shared["chunk"], shared["state_q"], shared["seq"]
            if ch is not None and seq != last_seq:
                last_seq = seq
                absk = np.stack([PolicyClient.chunk_to_target(ch, kk) for kk in range(HCHUNK)]).astype(np.float64)
                ens_buf.append([step, absk])
            ens_buf = [c for c in ens_buf if c[0] + HCHUNK > step]
            num = None; wsum = 0.0
            for i, c in enumerate(reversed(ens_buf)):
                idx = step - c[0]
                if 0 <= idx < HCHUNK:
                    w = np.exp(-max(0.0, args.ens_coeff) * i)
                    num = c[1][idx] * w if num is None else num + c[1][idx] * w
                    wsum += w
            if wsum > 0:
                tgt = num / wsum
                arm_q, hlt, hrt = tgt[:14], tgt[14:21], tgt[21:28]
                arm_cmd += np.clip(arm_q - arm_cmd, -args.slew, args.slew)
                gl += float(np.clip(g1._grasp_scalar(hlt, CLOSED_L) - gl, -args.hand_step_pick, args.hand_step_pick))
                gr += float(np.clip(g1._grasp_scalar(hrt, CLOSED_R) - gr, -args.hand_step_pick, args.hand_step_pick))
                g1.command_arm(arm_cmd, tauff=gtau(arm_cmd))
                g1.command_hand_scalar(gl, gr)
            if step % 50 == 0:
                print(f"[pick] step {step}/{args.max_run_steps}  grasp L={gl:.2f} R={gr:.2f}", flush=True)
            time.sleep(pick_dt)
        stop_worker.set()
        print(f"[pick] {'ABORTED' if aborted else 'complete'} ({step} steps).", flush=True)
        return None if aborted else (arm_cmd, gl, gr)

    with KeyPoller() as poller:
        print("[deploy] READY. Robot in normal 'ai' mode; model + image servers up. Clear space around the base.",
              flush=True)

        def raise_arm():
            print("[deploy] raising arms to start pose ...", flush=True)
            for _ in range(int(4.0 / dt)):
                g1.command_arm(start_arm, tauff=gtau(start_arm))
                cur = np.asarray(g1.arm.get_current_dual_arm_q())
                if float(np.abs(start_arm - cur).max()) < 0.03:
                    break
                time.sleep(dt)

        # start->pick runs UNATTENDED in the waypoints path (no SPACE); q still aborts each move.
        # post-pick / drop / homing stay SPACE-gated (they call wait_space directly).
        def gate(label):
            if args.waypoints:
                print(f"\n[deploy] AUTO -> {label}", flush=True)
                return poller.poll() not in ("q", "\x1b")
            return wait_space(poller, label)

        if args.waypoints:
            # SLAM path (drift-free absolute): P1 -> trunk up -> home arms -> P2 -> PICK -> P3 -> trunk down
            #                                  -> P4 -> trunk up -> P5 -> drop
            if not gate("drive to P1 (approach, drift-free SLAM)"):
                g1.stop(); return
            if not goto_waypoint("p1", poller, avoid=True):
                g1.stop(); return
            if not gate(f"trunk up {args.trunk_up*100:.1f} cm"):
                g1.stop(); return
            trunk_up(args.trunk_up, poller)
            # home the arms to the inference start pose now (clear of the table) so they don't strike it
            # during the forward approach; run_pick then starts inference without re-homing.
            if not gate("home arms to start pose (clear of the table)"):
                g1.stop(); return
            raise_arm()
            if args.table_approach:
                # reliable table-referenced final approach (immune to SLAM drift)
                if not gate("TABLE approach (square up + LiDAR closed-loop + fixed nudge)"):
                    g1.stop(); return
                if not approach_table(poller):
                    g1.stop(); return
            else:
                if not gate("drive to P2 (pick pose, drift-free SLAM)"):
                    g1.stop(); return
                if not goto_waypoint("p2", poller):
                    g1.stop(); return
                if not args.no_pick_align:
                    if not gate("LiDAR square-up correction at pick pose"):
                        g1.stop(); return
                    pick_align(poller)
        else:
            # --- 1) forward 30 cm ---
            if not wait_space(poller, f"base forward {args.fwd0*100:.0f} cm"):
                g1.stop(); return
            base_drive(args.fwd0, poller)

            # --- 2) turn left 90 ---
            if not wait_space(poller, f"turn LEFT {args.turn_deg:.0f} deg"):
                g1.stop(); return
            base_turn_left(args.turn_deg, poller)

            # --- 2b) square up to the table via LiDAR (optional; off unless --align) ---
            if args.align:
                if not wait_space(poller, "square up to the table (LiDAR)"):
                    g1.stop(); return
                align_to_table(poller)

            # (arm is NOT homed here anymore -- it stays tucked; run_pick homes it right before the pick)

            # --- 4) trunk up 3 in ---
            if not wait_space(poller, f"trunk up {args.trunk_up*100:.1f} cm"):
                g1.stop(); return
            trunk_up(args.trunk_up, poller)

            # --- 5) final forward (slow/controlled) ---
            if not wait_space(poller, f"base forward {args.fwd1*100:.0f} cm (slow)"):
                g1.stop(); return
            base_drive(args.fwd1, poller, vel=args.base_vel * args.final_speed_scale)

            # --- 5b) fw2: very slow align-to-table nudge (low speed) ---
            if not wait_space(poller, f"slow align nudge {args.fw2*100:.0f} cm"):
                g1.stop(); return
            base_drive(args.fw2, poller, vel=args.fw2_vel)

        # --- 6) PICK (ported run_heatmap loop) ---
        if not gate("RUN PICK"):
            g1.stop(); return
        held = run_pick(poller, home_first=not args.waypoints)
        if held is None:
            g1.stop(); return
        arm_cmd, gl, gr = held

        def hold_grasp():
            g1.command_arm(arm_cmd, tauff=gtau(arm_cmd)); g1.command_hand_scalar(gl, gr)
        hold_grasp()

        # fully autonomous: continue straight into the post-pick sequence (q/Ctrl-C still aborts)
        if not gate("PICK done -- continue"):
            g1.stop(); return

        # --- 7) POST-PICK (arm + grip hold the object throughout) ---
        if args.waypoints:
            # after the pick: reverse STRAIGHT back (no rotation) to clear the table, then skip P3 and go
            # straight to P4 -> trunk moves -> P5 (drop pose)
            steps = [
                (f"reverse {args.post_reverse*100:.0f}cm STRAIGHT back (no rotation)",
                 lambda: base_drive(-args.post_reverse, poller, vel=args.base_vel * args.final_speed_scale)),
                (f"trunk DOWN {args.post_trunk_down*100:.0f}cm", lambda: trunk_up(-args.post_trunk_down, poller)),
                ("drive to P4 (transit, OBSTACLE-CHECKED)", lambda: goto_waypoint("p4", poller, avoid=True)),
                (f"trunk UP {args.post_trunk_up2*100:.0f}cm", lambda: trunk_up(args.post_trunk_up2, poller)),
                ("drive to P5 (drop pose, object-agnostic)", lambda: goto_waypoint("p5", poller)),
            ]
        else:
            steps = [
                (f"trunk UP {args.post_trunk_up*100:.0f}cm", lambda: trunk_up(args.post_trunk_up, poller)),
                (f"back {args.post_back1*100:.0f}cm (slow)", lambda: base_drive(-args.post_back1, poller, vel=args.base_vel*args.final_speed_scale)),
                (f"trunk DOWN {args.post_trunk_down*100:.0f}cm", lambda: trunk_up(-args.post_trunk_down, poller)),
                (f"back {args.post_back2*100:.0f}cm", lambda: base_drive(-args.post_back2, poller)),
                ("rotate to START heading", lambda: turn_to_yaw(start_yaw, poller)),
                (f"forward {args.post_fwd*100:.0f}cm", lambda: base_drive(args.post_fwd, poller)),
                (f"trunk UP {args.post_trunk_up2*100:.0f}cm", lambda: trunk_up(args.post_trunk_up2, poller)),
                (f"slow forward {args.post_fwd2*100:.0f}cm", lambda: base_drive(args.post_fwd2, poller, vel=args.fw2_vel)),
            ]
        for label, fn in steps:
            if not gate(f"POST-PICK: {label}"):
                g1.stop(); return
            fn(); hold_grasp()

        # --- 8) DROP: lower trunk, open the claws, rotate arms outward (fully autonomous) ---
        if not gate(f"DROP (trunk -{args.drop_trunk_down*100:.0f}cm, open claws, arms out)"):
            g1.stop(); return
        trunk_up(-args.drop_trunk_down, poller); hold_grasp()
        print("[deploy] opening claws ...", flush=True)
        for _ in range(int(1.0 / dt)):                      # open both hands, keep arm held
            g1.command_arm(arm_cmd, tauff=gtau(arm_cmd)); g1.command_hand_scalar(0.0, 0.0)
            time.sleep(dt)
        print("[deploy] releasing: rotate shoulders OUTWARD (not up) with a small trunk dip ...", flush=True)
        # shoulder_roll = idx 1 (L) / 8 (R); +left / -right abducts the arms OUTWARD. Rotating this joint
        # swings the arms out sideways (object rolls outward) instead of the IK spread that lifted them up.
        oc = arm_cmd.copy()

        def slew_to(q_target, secs=4.0):
            nonlocal oc
            for _ in range(int(secs / dt)):
                oc += np.clip(q_target - oc, -args.max_joint_step, args.max_joint_step)
                g1.command_arm(oc, tauff=gtau(oc)); g1.command_hand_scalar(0.0, 0.0)
                if float(np.abs(q_target - oc).max()) < 0.02:
                    break
                if poller.poll() in ("q", "\x1b"):
                    break
                time.sleep(dt)

        q1 = arm_cmd.copy(); q1[1] += args.drop_rot1; q1[8] -= args.drop_rot1   # shoulder_roll outward (phase 1)
        slew_to(q1)
        trunk_up(-args.drop_dip, poller)                                        # drop the trunk a tiny bit
        q2 = q1.copy(); q2[1] += args.drop_rot2; q2[8] -= args.drop_rot2        # shoulder_roll outward more (phase 2)
        slew_to(q2)
        # then the next joint down (shoulder_yaw, idx 2 L / 9 R) rotates each arm further outward
        q3 = q2.copy(); q3[2] += args.drop_yaw; q3[9] -= args.drop_yaw
        slew_to(q3)

        # --- 9) POST-DROP (waypoints path): back off, lower trunk all the way, lower hands, then return to home ---
        if args.waypoints:
            def hold_open():
                g1.command_arm(oc, tauff=gtau(oc)); g1.command_hand_scalar(0.0, 0.0)

            def lower_hands():
                """Lower the (open) hands to a rest position via IK before driving home."""
                nonlocal oc
                Tl, Tr = ik.fk(oc)
                Tl[2, 3] -= args.hands_lower; Tr[2, 3] -= args.hands_lower   # z down
                try:
                    q_low, _ = ik.solve_ik(Tl, Tr, np.asarray(g1.arm.get_current_dual_arm_q()),
                                           np.asarray(g1.arm.get_current_dual_arm_dq()))
                    q_low = np.asarray(q_low, np.float64)
                except Exception as e:
                    print(f"[deploy] hands-lower IK failed ({e}) -- skipping", flush=True); return
                print(f"[deploy] lowering hands {args.hands_lower*100:.0f}cm ...", flush=True)
                for _ in range(int(5.0 / dt)):
                    oc += np.clip(q_low - oc, -args.max_joint_step, args.max_joint_step)
                    g1.command_arm(oc, tauff=gtau(oc)); g1.command_hand_scalar(0.0, 0.0)
                    if float(np.abs(q_low - oc).max()) < 0.02:
                        break
                    if poller.poll() in ("q", "\x1b"):
                        break
                    time.sleep(dt)

            post_drop = [
                (f"reverse {args.post_drop_reverse*100:.0f}cm STRAIGHT back",
                 lambda: base_drive(-args.post_drop_reverse, poller, vel=args.base_vel * args.final_speed_scale)),
                (f"trunk DOWN {args.post_drop_trunk_down*100:.0f}cm",
                 lambda: trunk_up(-args.post_drop_trunk_down, poller)),
                ("hands to a lowered rest position", lower_hands),
                ("HOME: drive to home_far (OBSTACLE-CHECKED, stops clear of the charger)",
                 lambda: goto_waypoint("home_far", poller, avoid=True)),
                ("HOME: close in to home_stage (LiDAR-free, object-agnostic)",
                 lambda: goto_waypoint("home_stage", poller, avoid=False)),
                ("HOME: rotate 180 in place",
                 lambda: turn_relative(-180.0, poller)),
                (f"HOME: back {args.home_back*100:.0f}cm into the charger",
                 lambda: base_drive(-args.home_back, poller)),
            ]
            for label, fn in post_drop:
                if not gate(f"POST-DROP: {label}"):
                    g1.stop(); return
                fn(); hold_open()

        print("[deploy] SEQUENCE COMPLETE (dropped). q/Ctrl-C to finish.", flush=True)
        while True:
            g1.command_arm(oc, tauff=gtau(oc)); g1.command_hand_scalar(0.0, 0.0)
            if poller.poll() in ("q", "\x1b"):
                break
            time.sleep(dt)
    g1.stop()
    print("[deploy] done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
