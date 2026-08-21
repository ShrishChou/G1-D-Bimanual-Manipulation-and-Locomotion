"""RealIO -- the real-robot backend for nav_fsm.py. Same interface as SimIO, so the identical FSM
(pick -> plan -> accept -> move -> place) runs on the G1-D.

  get_pose()            odometry (x,y,yaw) relative to the FSM start, from SportModeState_
  drive(vx, vyaw)       LocoClient.Move(vx, 0.0, vyaw)  -- NON-HOLONOMIC: vy pinned to 0 (fwd/back + turn only)
  stop()                LocoClient.StopMove()
  occupancy()           2D grid built from the live LiDAR PointCloud2 (this becomes the planned map)
  unexpected_points()   live LiDAR points NOT in the planned map (map-vs-scan diff) -> the failsafe watches these
  do_pick()/do_place()  deterministic arm skills via deploy/run_skill.py
  render()              push state to the Rerun twin (wired in the twin step)

COMMISSION CAREFULLY -- items to verify on the robot are marked TODO(robot): the SportModeState topic name,
the LiDAR topic + PointCloud2 field layout, and the LiDAR->base mount transform. Keep a physical e-stop in hand;
the FSM's Ctrl-C killswitch and the LiDAR failsafe both call stop() (StopMove).
"""
import math
import subprocess
import sys
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import nav_planner as N

STATE_TOPIC = "rt/sportmodestate"      # TODO(robot): verify (may be rt/lf/sportmodestate)
LIDAR_TOPIC = "rt/utlidar/cloud"       # TODO(robot): verify the actual G1-D LiDAR PointCloud2 topic
LIDAR_MOUNT = (0.0, 0.0, 0.4)          # TODO(robot): LiDAR position in the base frame (x,y,z)
Z_MIN, Z_MAX = 0.10, 1.6               # obstacle height band (drop floor + overhead)
RUN_SKILL = "$REPO/deploy/run_skill.py"


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class RealIO:
    def __init__(self, iface="enp2s0", pick_skill=None, place_skill=None, motion=True):
        ChannelFactoryInitialize(0, iface)
        self.loco = LocoClient(); self.loco.SetTimeout(0.5); self.loco.Init()
        self.pick_skill, self.place_skill, self.motion = pick_skill, place_skill, motion
        self._pose = None          # (x,y,yaw) raw odometry
        self._start = None         # odometry at FSM start -> everything reported relative to this
        self._cloud = np.zeros((0, 3))
        self.planned_grid = None   # set in occupancy(); used for map-vs-scan differencing
        self.twin = None           # optional Rerun twin hook (set by nav_fsm when --twin)
        self._alive = True
        self._sub_state = ChannelSubscriber(STATE_TOPIC, SportModeState_); self._sub_state.Init()
        self._sub_lidar = ChannelSubscriber(LIDAR_TOPIC, PointCloud2_); self._sub_lidar.Init()
        threading.Thread(target=self._state_loop, daemon=True).start()
        threading.Thread(target=self._lidar_loop, daemon=True).start()
        for _ in range(200):       # wait for the first odometry sample, then latch the start frame
            if self._pose is not None:
                break
            time.sleep(0.01)
        self._start = self._pose if self._pose is not None else (0.0, 0.0, 0.0)

    # ---- sensor threads ----
    def _state_loop(self):
        while self._alive:
            m = self._sub_state.Read()
            if m is not None:
                self._pose = (float(m.position[0]), float(m.position[1]), float(m.imu_state.rpy[2]))
            time.sleep(0.005)

    def _lidar_loop(self):
        while self._alive:
            m = self._sub_lidar.Read()
            if m is not None:
                self._cloud = self._parse_cloud(m)
            time.sleep(0.02)

    def _parse_cloud(self, msg):
        """PointCloud2 -> Nx3 float32 xyz (assumes x,y,z float32 at field offsets 0/4/8; verify on robot)."""
        try:
            step = msg.point_step
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, step)
            xyz = np.stack([buf[:, o:o + 4].copy().view(np.float32).ravel() for o in (0, 4, 8)], axis=1)
            return xyz[np.isfinite(xyz).all(1)]
        except Exception:
            return np.zeros((0, 3))

    # ---- RobotIO interface ----
    def get_pose(self):
        p, s = self._pose or (0, 0, 0), self._start
        return (p[0] - s[0], p[1] - s[1], _wrap(p[2] - s[2]))

    def _cloud_world_xy(self):
        """LiDAR points -> world (start-frame) xy, height-filtered to the obstacle band."""
        c = self._cloud
        if not len(c):
            return np.zeros((0, 2))
        c = c[(c[:, 2] > Z_MIN) & (c[:, 2] < Z_MAX)]
        x, y, yaw = self.get_pose()
        lx = c[:, 0] + LIDAR_MOUNT[0]; ly = c[:, 1] + LIDAR_MOUNT[1]
        wx = x + lx * math.cos(yaw) - ly * math.sin(yaw)
        wy = y + lx * math.sin(yaw) + ly * math.cos(yaw)
        return np.stack([wx, wy], 1)

    def occupancy(self):
        g = N.OccupancyGrid(size_m=8.0, res=0.05)
        g.add_points(self._cloud_world_xy())
        self.planned_grid = g            # snapshot as the planned map for later differencing
        return g

    def unexpected_points(self):
        """Live LiDAR points whose cell is FREE in the planned map -> a new/unexpected object."""
        if self.planned_grid is None:
            return np.zeros((0, 2))
        pts = self._cloud_world_xy()
        out = []
        for x, y in pts:
            r, c = self.planned_grid.w2c(x, y)
            if self.planned_grid.in_bounds(r, c) and not self.planned_grid.occ[r, c]:
                out.append((x, y))
        return np.array(out) if out else np.zeros((0, 2))

    def drive(self, vx, vyaw, dt=None):
        self.loco.Move(float(vx), 0.0, float(vyaw))     # NON-HOLONOMIC: no lateral velocity

    def stop(self):
        try:
            self.loco.StopMove()
        except Exception:
            pass

    def _run_skill(self, skill):
        if not skill:
            print("[RealIO] no skill provided -- skipping (teach one with teach_skill.py). TODO(robot).", flush=True)
            return
        cmd = ["python", RUN_SKILL, "--skill", skill] + (["--motion"] if self.motion else [])
        print(f"[RealIO] running deterministic skill: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=False)                # TODO(robot): add --auto to run_skill so it self-starts

    def do_pick(self):
        self._run_skill(self.pick_skill)

    def do_place(self):
        self._run_skill(self.place_skill)

    def render(self):
        if self.twin is not None:
            self.twin.update(self)                       # Rerun twin (wired in the twin step)

    def close(self):
        self._alive = False
        self.stop()
