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
from grasp_ik import OPEN, Kin

# ---- slow, controlled motion limits (shared sim + real) ----
V_FWD = 0.15          # m/s straight-drive cap
W_TURN = 0.4          # rad/s turn cap
POS_TOL = 0.06        # m waypoint arrival tolerance
YAW_TOL = 0.05        # rad heading tolerance before driving
DANGER_R = 0.45       # m: an obstacle point closer than this in the front sector -> E-STOP
FRONT_DEG = 60        # +/- half-angle of the front danger sector
ARM_HOME = np.zeros(7)
# Bimanual grasp geometry. The arm can only reach ~0.35 m forward at table height (measured on the
# model, not assumed), so the base drives to a standoff first instead of over-reaching.
PICK_STANDOFF = 0.28       # m: nominal base-to-object distance (clamped so the base clears the table)
PLACE_CLEAR = 0.06         # m: how far above the container rim the object is released
PREGRASP_BACK = 0.13       # m: hands start this far behind the object, and this much wider
GRASP_HALF_W = 0.085       # m: palm offset either side of a 45 mm-radius cylinder
LIFT_H = 0.12              # m: how far the object is lifted off the table
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
        self.kin = Kin(self.model)
        self.base_z = info["base_z"]   # measured per robot: wheels or feet on the floor
        self.lift = 0.0                # Z-lift extension (wheeled bases only)
        self.odo = 0.0                 # distance travelled, for rolling the wheels
        self.armL, self.armR = ARM_HOME.copy(), ARM_HOME.copy()
        self.handL, self.handR = OPEN.copy(), OPEN.copy()
        self.cyl_geoms = [g for g in range(self.model.ngeom)
                          if self.model.geom_bodyid[g] == self.model.body("cylinder").id]
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
        self.odo += abs(vx) * dt

    def stop(self):
        pass

    def _grasp_center(self):
        """Where a two-handed object actually sits: the midpoint between the two palms."""
        return self.kin.grasp_center(self.d)

    def _solve_grasp(self, center, half_width, back=0.0):
        """IK both arms onto a symmetric grasp about `center`, in the base frame."""
        c = np.array(center, float)
        fwd = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        left = np.array([-np.sin(self.yaw), np.cos(self.yaw), 0.0])
        c = c - back * fwd
        qL, qR, err = self.kin.ik_bimanual(self.d, c, half_width=half_width, lateral=left)
        return qL, qR, err

    def render(self):
        self.d.qpos[0:3] = [self.x, self.y, self.base_z]
        self.d.qpos[3:7] = yaw_quat(self.yaw)
        self.kin.apply(self.d, self.armL, self.armR, self.handL, self.handR)
        self.kin.set_lift(self.d, self.lift)
        self.kin.spin_wheels(self.d, self.odo)
        mujoco.mj_forward(self.model, self.d)
        if self.carry:
            # the object is held BETWEEN the hands, so it rides the palm midpoint
            self.d.qpos[self.gadr:self.gadr + 3] = self._grasp_center()
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
        """Approach, reach, close two hands around the object, and lift it.

        On the robot this whole phase is a taught deterministic skill (deploy/run_skill.py) or the VLA
        policy (deploy/deploy_pick.py); here it is solved kinematically so the twin shows the same
        two-handed geometry rather than a hand snapping to the object.
        """
        obj = np.array([*S.CYL_XY, S.TOP + S.CYL_H / 2])

        # 1) close to a standoff where the grasp is inside the arm workspace, but never closer than
        # the table allows. The base is a disc of radius S.ROBOT_RADIUS and the table face is solid, so
        # driving to a fixed object-relative standoff put the chassis inside the table -- which is
        # exactly what "the base phases through the table" looked like.
        table_limit = float(np.hypot(obj[0] - (S.FRONT_NEAR_EDGE - S.DOCK_HALF - 0.02),
                                     obj[1] - self.y))
        standoff = max(PICK_STANDOFF, table_limit)
        while not ESTOP["tripped"]:
            d_obj = float(np.hypot(obj[0] - self.x, obj[1] - self.y))
            if d_obj <= standoff + 0.01:
                break
            err = wrap(np.arctan2(obj[1] - self.y, obj[0] - self.x) - self.yaw)
            self.drive(V_FWD * max(0.25, 1 - abs(err)), np.clip(1.5 * err, -W_TURN, W_TURN), 1 / 30)
            self.render()
        self.drive(0, 0, 1 / 30)

        # 2) pre-grasp: hands open, apart and behind the object
        qL0, qR0, _ = self._solve_grasp(obj, GRASP_HALF_W + PREGRASP_BACK, back=PREGRASP_BACK)
        for i in range(24):
            a = i / 23.0
            self.armL, self.armR = ARM_HOME + (qL0 - ARM_HOME) * a, ARM_HOME + (qR0 - ARM_HOME) * a
            self.render()

        # 3) converge: bring both palms in to the object's sides
        qL1, qR1, err = self._solve_grasp(obj, GRASP_HALF_W)
        for i in range(20):
            a = i / 19.0
            self.armL, self.armR = qL0 + (qL1 - qL0) * a, qR0 + (qR1 - qR0) * a
            self.render()

        # 4) close the fingers until they actually touch the cylinder. The stopping point is found by
        # bisecting against MuJoCo's contact detection, so the fingers land on the surface of whatever
        # radius the object has instead of driving through it at a hand-tuned angle.
        def _pose_hands(frac):
            self.handL = self.kin.closed_pose("L", frac)
            self.handR = self.kin.closed_pose("R", frac)
            self.kin.apply(self.d, self.armL, self.armR, self.handL, self.handR)
            self.d.qpos[self.gadr:self.gadr + 3] = self.kin.grasp_center(self.d)
            mujoco.mj_forward(self.model, self.d)

        grip = self.kin.close_on_object(self.d, self.cyl_geoms, _pose_hands)
        for i in range(12):
            a = (i / 11.0) * grip
            self.handL = self.kin.closed_pose("L", a)
            self.handR = self.kin.closed_pose("R", a)
            self.render()
        self.grip = grip

        # 5) lift -- the object is now held between the palms
        self.carry = True
        qL2, qR2, _ = self._solve_grasp(obj + np.array([0, 0, LIFT_H]), GRASP_HALF_W)
        for i in range(18):
            a = i / 17.0
            self.armL, self.armR = qL1 + (qL2 - qL1) * a, qR1 + (qR2 - qR1) * a
            self.render()
        print(f"[PICK] two-handed grasp: palm IK residual {err*1000:.0f} mm, "
              f"object at {np.round(self._grasp_center(), 3)}", flush=True)

    def do_place(self):
        """Carry the object over the container's hole, release it, and let it drop through.

        Deliberately NOT a downward move into the plate: the object is centred above the rim, the
        hands open, and only then does it fall. Setting it down through the container was the other
        half of the "everything phases through everything" look.
        """
        rim_z = S.TOP + S.STAND_H + 0.5 * S.IN          # top face of the container plate
        above = np.array([S.CONT_XY[0], S.CONT_XY[1], rim_z + S.CYL_H / 2 + PLACE_CLEAR])
        rest = np.array([S.CONT_XY[0], S.CONT_XY[1], S.TOP + S.CYL_H / 2])   # stands on the table,
        #                                                                      protruding through the hole
        qL_hold, qR_hold = self.armL.copy(), self.armR.copy()

        qL_up, qR_up, err = self._solve_grasp(above, GRASP_HALF_W)
        for i in range(22):                      # traverse to directly over the hole
            a = i / 21.0
            self.armL, self.armR = qL_hold + (qL_up - qL_hold) * a, qR_hold + (qR_up - qR_hold) * a
            self.render()
        for _ in range(8):
            self.render()

        grip = getattr(self, "grip", 1.0)
        for i in range(10):                      # open the hands -- the object is still held by them
            a = 1.0 - i / 9.0
            self.handL = self.kin.closed_pose("L", grip * a)
            self.handR = self.kin.closed_pose("R", grip * a)
            self.render()

        # released: fall from where it was let go down into the hole
        self.carry = False
        drop_from = self.kin.grasp_center(self.d).copy()
        for i in range(14):
            a = (i + 1) / 14.0
            z = drop_from[2] + (rest[2] - drop_from[2]) * (a * a)   # quadratic: gravity, not a lerp
            self.cylinder = [rest[0], rest[1], float(z), *S.CYL_STAND_QUAT]
            self.render()
        self.cylinder = [rest[0], rest[1], float(rest[2]), *S.CYL_STAND_QUAT]

        for i in range(18):                      # withdraw to home
            a = i / 17.0
            self.armL, self.armR = qL_up + (ARM_HOME - qL_up) * a, qR_up + (ARM_HOME - qR_up) * a
            self.render()
        print(f"[PLACE] hold-above IK residual {err * 1000:.0f} mm; released "
              f"{PLACE_CLEAR * 100:.0f} cm above the rim; object settled at {np.round(rest, 3)}",
              flush=True)


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


def dock_to_table(io, near_edge_y, dt=1 / 30):
    """Creep straight forward until the base is one DOCK_HALF off the table face.

    A* plans to a pose outside the table inflated by the TURNING radius, which parks the base ~10 cm
    further out than it can actually stand -- far enough to put the container beyond arm reach. The
    base does not rotate during this last stretch, so it is closed with the shallower half-depth,
    the same split nav/table_align.py makes on the robot.
    """
    target = near_edge_y - S.DOCK_HALF - 0.02
    while not ESTOP["tripped"]:
        if io.y >= target - 0.01:
            break
        err = wrap(np.pi / 2 - io.yaw)
        io.drive(V_FWD * 0.5, np.clip(1.5 * err, -W_TURN, W_TURN), dt)
        io.render()
    io.drive(0, 0, dt)


def back_off(io, distance=0.30, dt=1 / 30):
    """Reverse straight by `distance` before planning.

    Docking deliberately parks the base inside the region A* treats as blocked -- the planner inflates
    by the turning radius, the dock uses the shallower half-depth. Planning from there fails on the
    START cell, so the robot backs out of the table's swept zone first, which is what it would have to
    do on hardware before turning anyway.
    """
    x0, y0 = io.x, io.y
    while not ESTOP["tripped"]:
        if float(np.hypot(io.x - x0, io.y - y0)) >= distance:
            break
        io.drive(-V_FWD * 0.6, 0.0, dt)
        io.render()
    io.drive(0, 0, dt)


def run_fsm(io, place_target, auto=True):
    print("[FSM] PICK (deterministic)", flush=True)
    io.do_pick()

    back_off(io)
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
    dock_to_table(io, S.PLACE_NEAR_EDGE)
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
