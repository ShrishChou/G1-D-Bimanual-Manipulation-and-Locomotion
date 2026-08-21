"""Live G1-D hardware I/O for policy deployment, built on the teleop stack (the reference for how we
read cameras + robot state and command the arms/Dex3).

Provides:
  G1Interface.read()               -> (head, left_wrist, right_wrist, state28)   [perception + proprioception]
  G1Interface.command_arm(q14)     -> compliant PD move of both arms to joint targets
  G1Interface.command_hand(hl7,hr7)-> drive the Dex3 via the grasp path (scalar from the policy hand target)
  G1Interface.go_home() / stop()

Camera layout mirrors teleop: teleimager_bridge republishes head|Lwrist|Rwrist on :5555; the ImageClient
fills tv_img_array (head, 480x640) and wrist_img_array (2 wrists, 480x1280 -> left [:, :640], right [:, 640:]).

NOTE: hardware-dependent -- validate on the real G1-D. Run in the g1-teleop pixi env (has televuer /
unitree_sdk2py / teleop deps).  Set TELEOP_DIR if the teleop repo lives elsewhere.
"""
import os, sys, threading, time
import numpy as np
from multiprocessing import shared_memory, Array, Lock

TELEOP_DIR = os.environ.get("TELEOP_DIR", "$TELEOP_DIR")
HEAD_H, HEAD_W = 480, 640

# Dex3 closed poses (hardware order thumb0,thumb1,thumb2,middle0,middle1,index0,index1) -- must match the
# teleop Dex3_1_Controller grasp defaults so the grasp scalar maps consistently with the training data.
CLOSED_L = np.array([0.0, -0.9, -1.7, 1.4, 1.6, 1.4, 1.6])
CLOSED_R = np.array([0.0, 0.9, 1.7, -1.4, -1.6, -1.4, -1.6])


class G1Interface:
    def __init__(self, iface="enp2s0", arm_kp=None, hand_kp=3.0, sim=False, motion=False, hands=True, cameras=True):
        sys.path.insert(0, TELEOP_DIR)
        from teleop.robot_control.robot_arm import G1_29_ArmController
        from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        # ---- cameras (head + 2 wrists). SKIP for demo-following: the ImageClient's decode thread + big
        # array copies starve the arm's 240Hz publish thread -> watchdog drops. Demo replay needs no cameras.
        tv_shape = (HEAD_H, HEAD_W, 3)
        wr_shape = (HEAD_H, HEAD_W * 2, 3)
        self._tv_shm = self._wr_shm = None
        if cameras:
            from teleop.image_server.image_client import ImageClient
            self._tv_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(tv_shape)))
            self._wr_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(wr_shape)))
            self.tv_img = np.ndarray(tv_shape, dtype=np.uint8, buffer=self._tv_shm.buf)
            self.wr_img = np.ndarray(wr_shape, dtype=np.uint8, buffer=self._wr_shm.buf)
            img_client = ImageClient(tv_img_shape=tv_shape, tv_img_shm_name=self._tv_shm.name,
                                     wrist_img_shape=wr_shape, wrist_img_shm_name=self._wr_shm.name,
                                     server_address="127.0.0.1")
            threading.Thread(target=img_client.receive_process, daemon=True).start()
        else:
            self.tv_img = np.zeros(tv_shape, np.uint8)
            self.wr_img = np.zeros(wr_shape, np.uint8)

        # The teleop controllers load assets via relative paths ('../assets/...'), so build them with
        # the working directory set to the teleop dir; restore cwd afterwards.
        _cwd = os.getcwd()
        os.chdir(os.path.join(TELEOP_DIR, "teleop"))
        try:
            # ---- DDS pinned to the isolated wired link BEFORE controllers (as in teleop) ----
            ChannelFactoryInitialize(1) if sim else ChannelFactoryInitialize(0, iface)

            # ---- arms ----
            # motion=True -> rt/arm_sdk (weight kNotUsedJoint0=1 overlays arm control while the robot
            #   balances; usual G1-D mode). motion=False -> rt/lowcmd (robot must be in low-level/debug).
            # kp is written ONCE at controller init from kp_low (shoulder/elbow) & kp_wrist; the loop never
            # re-applies it, so to change kp we must overwrite the command msg (see _set_arm_kp). kp_low too
            # low => the arm can't hold itself against gravity (sags/looks limp), so default to teleop's 80.
            self._arm_ik = None                  # lazy: only the compliant-test governance needs it
            self.arm = G1_29_ArmController(motion_mode=motion, simulation_mode=sim)
            if arm_kp is not None:
                self._set_arm_kp(float(arm_kp))

            # ---- Dex3 hands via the grasp path (index-0 scalar), same wiring as teleop ----
            # NOTE: the Dex3 controller forks a child process that uses DDS; if that disrupts the arm's DDS
            # publisher (arm goes limp), construct with hands=False to isolate arm control.
            self._lh_in = Array('d', 75, lock=True)
            self._rh_in = Array('d', 75, lock=True)
            self._hand_state = Array('d', 14, lock=False)
            self._hand_action = Array('d', 14, lock=False)
            self.hand = None
            if hands:
                self.hand = Dex3_1_Controller(self._lh_in, self._rh_in, Lock(), self._hand_state, self._hand_action,
                                              simulation_mode=sim, swap_hands=True, hand_kp=hand_kp, grasp_mode=True)
        finally:
            os.chdir(_cwd)

    def _set_arm_kp(self, kp):
        """Overwrite the arm joints' kp in the command msg (controller only sets kp once, at init).
        WARNING: too low (<~50) and the arm can't hold itself against gravity -> it sags / looks limp."""
        from teleop.robot_control.robot_arm import G1_29_JointArmIndex
        for jid in G1_29_JointArmIndex:
            self.arm.msg.motor_cmd[jid].kp = float(kp)
        self.arm.kp_low = float(kp)
        if kp < 50:
            print(f"[G1Interface] WARNING: arm kp={kp} is very low; the arm may sag under gravity.", flush=True)

    @property
    def arm_ik(self):
        """Lazily build the arm IK (used only by the compliant-test governance). G1_29_ArmIK loads its
        URDF via a relative path ('../assets/g1/...'), so build it with cwd = the teleop dir."""
        if self._arm_ik is None:
            from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
            cwd = os.getcwd()
            try:
                os.chdir(os.path.join(TELEOP_DIR, "teleop"))
                self._arm_ik = G1_29_ArmIK()
            finally:
                os.chdir(cwd)
        return self._arm_ik

    def read(self):
        head = self.tv_img.copy()
        left = self.wr_img[:, :HEAD_W].copy()
        right = self.wr_img[:, HEAD_W:].copy()
        arm_q = np.asarray(self.arm.get_current_dual_arm_q(), np.float32)      # 14
        hand_q = np.asarray(self._hand_state[:], np.float32)                   # 14 (L7 + R7)
        return head, left, right, np.concatenate([arm_q, hand_q]).astype(np.float32)

    def command_arm(self, q14, tauff=None):
        tau = np.zeros(14) if tauff is None else np.asarray(tauff, np.float64)
        self.arm.ctrl_dual_arm(np.asarray(q14, np.float64), tau)

    def gravity_tau(self, q14):
        """Gravity-compensation feed-forward torque for the 14 arm joints (so kp doesn't have to hold the
        arm's weight -> no droop into the table). Uses the arm IK's pinocchio model."""
        import pinocchio as pin
        m = self.arm_ik.reduced_robot.model
        d = m.createData()
        return np.asarray(pin.computeGeneralizedGravity(m, d, np.asarray(q14, float).flatten()), np.float64)

    @staticmethod
    def _grasp_scalar(target7, closed7):
        d = float(np.dot(closed7, closed7))
        return float(np.clip(np.dot(np.asarray(target7), closed7) / d, 0.0, 1.0)) if d > 0 else 0.0

    def command_hand(self, hl7, hr7):
        """Drive the Dex3 via the grasp scalar derived from the policy's 7-joint hand target."""
        self.command_hand_scalar(self._grasp_scalar(hl7, CLOSED_L), self._grasp_scalar(hr7, CLOSED_R))

    def command_hand_scalar(self, gl, gr):
        """Drive the Dex3 open<->closed with explicit grasp scalars in [0,1] (used with slew limiting)."""
        if self.hand is None:
            return
        with self._lh_in.get_lock():
            self._lh_in[0] = float(np.clip(gl, 0.0, 1.0))
        with self._rh_in.get_lock():
            self._rh_in[0] = float(np.clip(gr, 0.0, 1.0))

    def go_home(self):
        self.arm.ctrl_dual_arm_go_home(shutoff=True)

    def stop(self):
        try:
            self.hand.teleop.value = False
        except Exception:
            pass
        self.go_home()
        for shm in (getattr(self, "_tv_shm", None), getattr(self, "_wr_shm", None)):
            if shm is not None:
                try:
                    shm.close(); shm.unlink()
                except Exception:
                    pass


# ---- shared-memory contract between g1_bridge.py (g1-teleop env) and the Isaac monitor ----
SHM_STATE, SHM_HEAD, SHM_LW, SHM_RW = "cylinder_state", "cylinder_head", "cylinder_lw", "cylinder_rw"


class LiveG1Source:
    """Isaac-side reader. Attaches to the shared-memory buffers published by g1_bridge.py (which runs in
    the g1-teleop env and does the DDS/camera reads). This keeps the teleop/unitree_sdk2py deps OUT of the
    Isaac python -- the two processes only share /dev/shm.  Start g1_bridge.py first."""

    def __init__(self, **kw):
        from multiprocessing import shared_memory
        try:
            self._s = shared_memory.SharedMemory(name=SHM_STATE)
            self._h = shared_memory.SharedMemory(name=SHM_HEAD)
            self._l = shared_memory.SharedMemory(name=SHM_LW)
            self._r = shared_memory.SharedMemory(name=SHM_RW)
        except FileNotFoundError:
            raise SystemExit("[LiveG1Source] shared memory not found -- start the bridge first:\n"
                             "  (g1-teleop env) python deploy/g1_bridge.py            # real robot\n"
                             "  (gr00t env)     python deploy/g1_bridge.py --fake --episode 46   # offline test")
        self.state = np.ndarray((28,), np.float64, self._s.buf)
        self.head = np.ndarray((HEAD_H, HEAD_W, 3), np.uint8, self._h.buf)
        self.lw = np.ndarray((HEAD_H, HEAD_W, 3), np.uint8, self._l.buf)
        self.rw = np.ndarray((HEAD_H, HEAD_W, 3), np.uint8, self._r.buf)

    def read(self):
        return self.head.copy(), self.lw.copy(), self.rw.copy(), self.state.copy().astype(np.float32)
