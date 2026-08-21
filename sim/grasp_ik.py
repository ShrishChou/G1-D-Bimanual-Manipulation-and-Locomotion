"""Bimanual grasp kinematics for the G1 + Dex3 digital twin.

Two jobs:

  * resolve every joint / body index **by name** rather than by a hardcoded qpos slice. The G1-D model
    interleaves each hand after its own arm (left arm, left hand, right arm, right hand), so a slice that
    is correct for the hand-less model silently writes finger angles into the wrong place.
  * solve a **symmetric two-handed grasp**: place both palms on opposite sides of an upright cylinder so
    the object ends up centred between the hands, which is how the taught skill and the policy both do it.

The solver is damped least squares on each arm's 7 joints, with a nullspace pull toward a natural
carry posture. Position-only: for a cylinder the palm normal is fully determined by which side it is on,
so a fixed inward wrist bias in the posture target is enough and avoids an orientation objective that
fights the reach at the edge of the workspace.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM_L = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
         "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
ARM_R = [n.replace("left", "right") for n in ARM_L]
HAND_L = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
          "left_hand_middle_0_joint", "left_hand_middle_1_joint",
          "left_hand_index_0_joint", "left_hand_index_1_joint"]
HAND_R = [n.replace("left", "right") for n in HAND_L]

PALM_L, PALM_R = "left_wrist_yaw_link", "right_wrist_yaw_link"
PALM_OFFSET = np.array([0.06, 0.0, 0.0])     # wrist frame -> palm centre

# Natural bimanual carry posture (shoulders forward, elbows bent, wrists rolled inward).
POSTURE_L = np.array([-0.45, 0.28, 0.05, 0.95, 0.0, 0.0, -0.35])
POSTURE_R = np.array([-0.45, -0.28, -0.05, 0.95, 0.0, 0.0, 0.35])
OPEN = np.zeros(7)
CLOSED = np.array([0.5, 0.9, 0.7, 0.9, 0.8, 0.9, 0.8])   # thumb wraps, index+middle curl


class Kin:
    """Name-resolved index tables + FK/IK helpers for one G1 model."""

    def __init__(self, model):
        self.m = model
        self.q_armL, self.v_armL = self._adr(ARM_L)
        self.q_armR, self.v_armR = self._adr(ARM_R)
        self.q_handL, _ = self._adr(HAND_L)
        self.q_handR, _ = self._adr(HAND_R)
        self.lim_armL = np.array([model.jnt_range[self._jid(n)] for n in ARM_L])
        self.lim_armR = np.array([model.jnt_range[self._jid(n)] for n in ARM_R])

    def _jid(self, name):
        i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise KeyError(f"joint {name!r} not in model -- wrong G1 variant?")
        return i

    def _adr(self, names):
        ids = [self._jid(n) for n in names]
        return (np.array([self.m.jnt_qposadr[i] for i in ids]),
                np.array([self.m.jnt_dofadr[i] for i in ids]))

    # ---- forward kinematics -------------------------------------------------
    def palm(self, data, side):
        b = data.body(PALM_L if side == "L" else PALM_R)
        return np.array(b.xpos) + np.array(b.xmat).reshape(3, 3) @ PALM_OFFSET

    def grasp_center(self, data):
        """Midpoint of the two palms -- where a two-handed object actually sits."""
        return 0.5 * (self.palm(data, "L") + self.palm(data, "R"))

    # ---- inverse kinematics -------------------------------------------------
    def ik_arm(self, data, side, target, q0=None, iters=200, damping=0.02, posture_gain=0.015, step=0.9):
        """Damped-least-squares IK for one 7-DoF arm. Returns joint angles; does not mutate `data`."""
        qadr = self.q_armL if side == "L" else self.q_armR
        vadr = self.v_armL if side == "L" else self.v_armR
        lim = self.lim_armL if side == "L" else self.lim_armR
        posture = POSTURE_L if side == "L" else POSTURE_R
        body = PALM_L if side == "L" else PALM_R
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body)

        q = np.array(data.qpos[qadr] if q0 is None else q0, float)
        saved = np.array(data.qpos)
        jacp = np.zeros((3, self.m.nv))
        best, best_err = q.copy(), np.inf
        for _ in range(iters):
            data.qpos[qadr] = q
            mujoco.mj_kinematics(self.m, data)
            mujoco.mj_comPos(self.m, data)
            cur = self.palm(data, side)
            err = target - cur
            e = float(np.linalg.norm(err))
            if e < best_err:
                best, best_err = q.copy(), e
            if e < 2e-3:
                break
            # Jacobian AT THE PALM POINT, not the body origin -- jacBody would linearise the
            # wrong point and stall the solve ~40 mm short.
            mujoco.mj_jac(self.m, data, jacp, None, cur, bid)
            J = jacp[:, vadr]
            JT = J.T
            dq = JT @ np.linalg.solve(J @ JT + damping ** 2 * np.eye(3), err)
            null = (np.eye(7) - np.linalg.pinv(J) @ J) @ (posture - q)
            q = np.clip(q + step * dq + posture_gain * null, lim[:, 0], lim[:, 1])
        data.qpos[:] = saved
        mujoco.mj_kinematics(self.m, data)
        return best, best_err

    def ik_bimanual(self, data, center, half_width=0.085, **kw):
        """Both palms on opposite sides of `center` (world frame, cylinder assumed upright).

        Returns (qL, qR, worst_position_error_m).
        """
        tL = np.array(center) + np.array([0.0, half_width, 0.0])
        tR = np.array(center) + np.array([0.0, -half_width, 0.0])
        qL, eL = self.ik_arm(data, "L", tL, **kw)
        qR, eR = self.ik_arm(data, "R", tR, **kw)
        return qL, qR, max(eL, eR)

    # ---- state application --------------------------------------------------
    def apply(self, data, qL=None, qR=None, handL=None, handR=None):
        if qL is not None:
            data.qpos[self.q_armL] = qL
        if qR is not None:
            data.qpos[self.q_armR] = qR
        if handL is not None:
            data.qpos[self.q_handL] = handL
        if handR is not None:
            data.qpos[self.q_handR] = handR
