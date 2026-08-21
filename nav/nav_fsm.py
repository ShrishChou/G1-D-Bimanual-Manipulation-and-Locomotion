"""Supervised-autonomy FSM for the G1-D mobile pick-place, with a real-time digital twin.

    PICK (deterministic arm) -> PLAN (A*, show + WAIT FOR ACCEPT) -> MOVE (real-time, non-holonomic,
    LiDAR-avoiding, slow) -> PLACE (deterministic arm)

The SAME FSM runs on a RobotIO backend: SimIO (MuJoCo -- this is the digital twin, fully testable offline)
or RealIO (LocoClient + DDS odometry/joints + LiDAR -- for the robot, commission carefully). The base is
NON-HOLONOMIC (forward/back + turn only), so each path leg is: turn-in-place to face the waypoint, then drive
straight. A dedicated SafetyMonitor watches the LiDAR and E-STOPS on any unexpected object in the danger zone.

Validate in the twin (writes a video of the whole run):
    MUJOCO_GL=glfw python nav_fsm.py --auto --out /tmp/fsm.mp4
"""
import argparse
import os
import signal
import sys
import threading
import time

import numpy as np
import mujoco
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import nav_planner as N
import scene as S

# ---- slow, controlled motion limits (shared sim + real) ----
V_FWD = 0.15          # m/s straight-drive cap
W_TURN = 0.4          # rad/s turn cap
POS_TOL = 0.06        # m waypoint arrival tolerance
YAW_TOL = 0.05        # rad heading tolerance before driving
DANGER_R = 0.45       # m: an obstacle point closer than this in the front sector -> E-STOP
FRONT_DEG = 60        # +/- half-angle of the front danger sector
ARM_HOME = np.zeros(7)
ARM_REACH_R = np.array([-0.55, -0.20, 0.0, 1.05, 0, 0, 0])
ARM_REACH_L = np.array([-0.55, 0.20, 0.0, 1.05, 0, 0, 0])
ESTOP = {"tripped": False, "reason": ""}
ACTIVE_IO = None      # set to the live RobotIO so the Ctrl-C handler can stop the base from anywhere


def install_killswitch(io):
    """Ctrl-C at ANY phase -> immediately stop the base and abort (keyboard e-stop). On the real robot io.stop()
    is LocoClient.StopMove(). Physical e-stop remains the ultimate safety."""
    global ACTIVE_IO
    ACTIVE_IO = io

    def _kill(*_):
        ESTOP["tripped"] = True; ESTOP["reason"] = "Ctrl-C (keyboard kill)"
        try:
            if ACTIVE_IO is not None:
                ACTIVE_IO.stop()
        except Exception:
            pass
        print("\n[KILL] Ctrl-C -> StopMove + abort", flush=True)
        os._exit(130)
    signal.signal(signal.SIGINT, _kill)
    signal.signal(signal.SIGTERM, _kill)


def yaw_quat(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SimIO:
    """Digital-twin backend: kinematic non-holonomic base + FK arm + LiDAR simulated from the scene packages."""
    def __init__(self, packages, cam, video_path=None, intruder=None):
        self.packages = packages
        self.intruder = intruder
        self.model, info = S.build(packages, intruder=intruder)
        self.d = mujoco.MjData(self.model)
        self.gadr = info["cyl_qadr"]
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.armL, self.armR = ARM_HOME.copy(), ARM_HOME.copy()
        self.cylinder = [*S.CYL_XY, S.TOP, *S.CYL_STAND_QUAT]
        self.carry = False
        self.r = mujoco.Renderer(self.model, height=720, width=1280)
        self.cam = cam
        self.vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720)) if video_path else None
        self.twin = None      # optional live Rerun twin (set by main when --twin)
        # pre-sample LiDAR-visible points on each package footprint (world), for avoidance + safety
        self._pkg_pts = self._sample_packages()
        # unexpected-object points (the intruder) -- what the failsafe watches; the planner never saw these
        self._intr_pts = np.array([[intruder[0] + dx, intruder[1] + dy]
                                   for dx in (-0.1, 0, 0.1) for dy in (-0.1, 0, 0.1)]) if intruder else np.zeros((0, 2))

    def _sample_packages(self):
        pts = []
        for px, py, yaw in self.packages:
            c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
            for u in np.linspace(-S.PKG[0], S.PKG[0], 6):
                for v in np.linspace(-S.PKG[1], S.PKG[1], 4):
                    pts.append([px + u * c - v * s, py + u * s + v * c])
        return np.array(pts)

    def get_pose(self):
        return self.x, self.y, self.yaw

    def lidar_points(self):
        """All points within ~3 m of the base (what the LiDAR sees: known packages + any intruder)."""
        p = np.vstack([self._pkg_pts, self._intr_pts]) if len(self._intr_pts) else self._pkg_pts
        return p[np.linalg.norm(p - [self.x, self.y], axis=1) < 3.0] if len(p) else p

    def unexpected_points(self):
        """Points NOT in the planned map (intruders) -- the failsafe trips only on these, like map-vs-scan
        differencing on the real robot (the planner already keeps clearance from known obstacles)."""
        p = self._intr_pts
        return p[np.linalg.norm(p - [self.x, self.y], axis=1) < 3.0] if len(p) else p

    def drive(self, vx, vyaw, dt):
        vx = float(np.clip(vx, -V_FWD, V_FWD)); vyaw = float(np.clip(vyaw, -W_TURN, W_TURN))
        self.yaw = wrap(self.yaw + vyaw * dt)
        self.x += vx * np.cos(self.yaw) * dt
        self.y += vx * np.sin(self.yaw) * dt

    def stop(self):
        pass

    def _hand_pos(self):
        return np.array(self.d.body("right_wrist_yaw_link").xpos)

    def render(self):
        self.d.qpos[0:3] = [self.x, self.y, 0.793]
        self.d.qpos[3:7] = yaw_quat(self.yaw)
        self.d.qpos[22:29] = self.armL; self.d.qpos[29:36] = self.armR
        mujoco.mj_forward(self.model, self.d)
        if self.carry:
            h = self._hand_pos()
            self.d.qpos[self.gadr:self.gadr + 3] = [h[0], h[1], h[2] - S.CYL_H / 2]
            self.d.qpos[self.gadr + 3:self.gadr + 7] = S.CYL_STAND_QUAT
        else:
            self.d.qpos[self.gadr:self.gadr + 7] = self.cylinder
        mujoco.mj_forward(self.model, self.d)
        if self.twin is not None:
            self.twin.update(self.lidar_points())
        self.r.update_scene(self.d, self.cam)
        img = self.r.render()[:, :, ::-1]
        if self.vw:
            self.vw.write(img)
        return img

    def close(self):
        if self.vw:
            self.vw.release()

    def occupancy(self):
        return S.occupancy(self.packages)

    def do_pick(self):
        for i in range(26):
            t = i / 25.0
            self.armL = ARM_HOME + (ARM_REACH_L - ARM_HOME) * t
            self.armR = ARM_HOME + (ARM_REACH_R - ARM_HOME) * t
            self.render()
        for _ in range(16):
            self.render()
        self.carry = True

    def do_place(self):
        self.cylinder = [S.CONT_XY[0], S.CONT_XY[1], S.TOP + S.STAND_H, *S.CYL_STAND_QUAT]
        for _ in range(18):
            self.render()
        self.carry = False
        for i in range(16):
            t = i / 15.0
            self.armL = ARM_REACH_L + (ARM_HOME - ARM_REACH_L) * t
            self.armR = ARM_REACH_R + (ARM_HOME - ARM_REACH_R) * t
            self.render()


def safety_check(io):
    """Return True if an UNEXPECTED object is inside the front danger sector (E-STOP condition)."""
    x, y, yaw = io.get_pose()
    for px, py in io.unexpected_points():
        d = np.hypot(px - x, py - y)
        if d < DANGER_R and abs(wrap(np.arctan2(py - y, px - x) - yaw)) < np.radians(FRONT_DEG):
            return True
    return False


def safety_monitor(io, hz=20):
    """Independent thread: any unexpected object in the danger zone -> E-STOP + stop the base immediately."""
    while not ESTOP["tripped"]:
        if safety_check(io):
            ESTOP["tripped"] = True; ESTOP["reason"] = "LiDAR: object in danger zone"
            io.stop()
            print(f"\n[SAFETY] E-STOP: {ESTOP['reason']}", flush=True)
            return
        time.sleep(1.0 / hz)


def execute_leg(io, tx, ty, dt=1 / 30, render=True):
    """Non-holonomic leg: turn-in-place to face (tx,ty), then drive straight to it. Aborts on E-STOP."""
    # 1) turn to face the target
    while not ESTOP["tripped"]:
        x, y, yaw = io.get_pose()
        err = wrap(np.arctan2(ty - y, tx - x) - yaw)
        if abs(err) < YAW_TOL:
            break
        io.drive(0.0, np.clip(2.0 * err, -W_TURN, W_TURN), dt)
        if render: io.render()
    # 2) drive straight, correcting heading, until arrived (front-sector obstacle check each tick)
    while not ESTOP["tripped"]:
        x, y, yaw = io.get_pose()
        dist = np.hypot(tx - x, ty - y)
        if dist < POS_TOL:
            break
        err = wrap(np.arctan2(ty - y, tx - x) - yaw)
        io.drive(V_FWD * max(0.2, 1 - abs(err)), np.clip(1.5 * err, -W_TURN, W_TURN), dt)
        if render: io.render()
    io.drive(0, 0, dt)


def run_fsm(io, place_target, auto=True):
    print("[FSM] PICK (deterministic)", flush=True)
    io.do_pick()

    print("[FSM] PLAN", flush=True)
    grid = io.occupancy()
    N.ROBOT_RADIUS = S.ROBOT_RADIUS
    path = N.plan(grid, io.get_pose()[:2], place_target)
    if path is None:
        print("[FSM] no path to target -- abort"); return False
    path_world = [grid.c2w(r, c) for r, c in path]
    if getattr(io, "twin", None):
        io.twin.log_path(path_world)
    print(f"[FSM] planned {len(path_world)}-waypoint path to {place_target}. Showing plan...", flush=True)
    # (twin/top-down shown to the user here) -- HUMAN ACCEPT gate:
    if auto:
        print("[FSM] AUTO-ACCEPT (interactive mode would wait for the user to accept)", flush=True)
    else:
        if input("[FSM] Accept plan and MOVE? [y/N] ").strip().lower() != "y":
            print("[FSM] rejected"); return False

    print("[FSM] MOVE (real-time, non-holonomic, LiDAR-guarded)", flush=True)
    threading.Thread(target=safety_monitor, args=(io,), daemon=True).start()
    for (tx, ty) in path_world[1:]:
        if ESTOP["tripped"]:
            break
        execute_leg(io, tx, ty)
    # face the container for the place
    while not ESTOP["tripped"]:
        x, y, yaw = io.get_pose()
        err = wrap(np.pi / 2 - yaw)
        if abs(err) < YAW_TOL:
            break
        io.drive(0, np.clip(2 * err, -W_TURN, W_TURN), 1 / 30); io.render()

    if ESTOP["tripped"]:
        print("[FSM] halted by safety E-STOP -- not placing"); return False
    print("[FSM] PLACE (deterministic)", flush=True)
    io.do_place()
    print("[FSM] done", flush=True)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="auto-accept the plan (for the offline twin video)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--intruder", action="store_true", help="drop an unexpected object in the corridor -> demo the failsafe")
    ap.add_argument("--out", default="/tmp/fsm.mp4")
    ap.add_argument("--real", action="store_true", help="run on the REAL G1-D (RealIO); default is the sim twin")
    ap.add_argument("--iface", default="enp2s0")
    ap.add_argument("--pick-skill", help="deploy/run_skill.py JSON for the deterministic pick (real)")
    ap.add_argument("--place-skill", help="deploy/run_skill.py JSON for the deterministic place (real)")
    ap.add_argument("--place-x", type=float, help="place target x (specify where to go; no AprilTag)")
    ap.add_argument("--place-y", type=float, help="place target y")
    ap.add_argument("--twin", help="write a live Rerun digital-twin recording to this .rrd path")
    args = ap.parse_args()

    target = (args.place_x, args.place_y) if args.place_x is not None and args.place_y is not None else S.GOAL

    if args.real:
        from real_io import RealIO
        io = RealIO(iface=args.iface, pick_skill=args.pick_skill, place_skill=args.place_skill)
        install_killswitch(io)                             # Ctrl-C = StopMove + abort, any phase
        try:
            run_fsm(io, target, auto=args.auto)
        finally:
            io.close()
    else:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.2, 1.0, 0.3]; cam.distance = 5.8; cam.azimuth = 215; cam.elevation = -35
        rng = np.random.default_rng(args.seed)
        packages = S.random_packages(rng)
        intruder = (0.0, 0.95) if args.intruder else None  # unplanned object mid-corridor
        io = SimIO(packages, cam, video_path=args.out, intruder=intruder)
        if args.twin:
            from twin import Twin
            io.twin = Twin(io.model, io.d, save_path=args.twin)
        install_killswitch(io)                             # Ctrl-C = immediate stop + abort, any phase
        try:
            run_fsm(io, target, auto=args.auto)
        finally:
            io.close()
        print("[FSM] video ->", args.out)
