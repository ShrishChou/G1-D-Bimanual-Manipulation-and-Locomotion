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

# Wheeled-base extras. Present on the G1-D (AGV base + two-stage Z-lift), absent on the legged G1,
# so every lookup is optional and the legged model simply reports None.
WHEELS = ["Left_Wheel_Joint", "Right_Wheel_Joint"]
LIFT = ["LZ_mt_Joint", "LZ_it_Joint"]
PALM_OFFSET = np.array([0.06, 0.0, 0.0])     # wrist frame -> palm centre

# Natural bimanual carry posture (shoulders forward, elbows bent, wrists rolled inward).
POSTURE_L = np.array([-0.45, 0.28, 0.05, 0.95, 0.0, 0.0, -0.35])
POSTURE_R = np.array([-0.45, -0.28, -0.05, 0.95, 0.0, 0.0, 0.35])
OPEN = np.zeros(7)
THUMB_FRACTION = 0.45      # how far the two straddling thumb joints swing at full closure


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
        self.lim_handL = np.array([model.jnt_range[self._jid(n)] for n in HAND_L])
        self.lim_handR = np.array([model.jnt_range[self._jid(n)] for n in HAND_R])
        self.q_wheels = self._adr_opt(WHEELS)
        self.q_lift = self._adr_opt(LIFT)
        self.lift_max = (float(model.jnt_range[self._jid(LIFT[0])][1])
                         + float(model.jnt_range[self._jid(LIFT[1])][1])) if self.q_lift is not None else 0.0

    def _adr_opt(self, names):
        """qpos addresses if EVERY name exists, else None (feature simply absent on this robot)."""
        try:
            return np.array([self.m.jnt_qposadr[self._jid(n)] for n in names])
        except KeyError:
            return None

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
    def ik_arm(self, data, side, target, q0=None, **kw):
        """Damped-least-squares IK for one 7-DoF arm, tried from several seeds.

        Seeding matters more than it looks. DLS is a local method, and seeding from "wherever the arm
        happens to be" stalls in a local minimum whenever the caller has already posed the arm -- the
        staged deployment sequence extends the arms to a pre-grasp pose before the grasp is solved, and
        that seed alone produced a 311 mm residual where a neutral seed converges to 1 mm. So try the
        current pose, the carry posture and zeros, and keep the best. Returns (q, error); `data` is
        left as it was found.
        """
        posture = POSTURE_L if side == "L" else POSTURE_R
        seeds = []
        if q0 is not None:
            seeds.append(np.asarray(q0, float))
        qadr = self.q_armL if side == "L" else self.q_armR
        seeds += [np.array(data.qpos[qadr], float), posture.copy(), np.zeros(7)]
        best, best_err = None, np.inf
        for s in seeds:
            q, e = self._ik_arm_once(data, side, target, s, **kw)
            if e < best_err:
                best, best_err = q, e
            if best_err < 2e-3:
                break
        return best, best_err

    def _ik_arm_once(self, data, side, target, q0, iters=200, damping=0.02, posture_gain=0.015, step=0.9):
        """One damped-least-squares solve from a single seed."""
        qadr = self.q_armL if side == "L" else self.q_armR
        vadr = self.v_armL if side == "L" else self.v_armR
        lim = self.lim_armL if side == "L" else self.lim_armR
        posture = POSTURE_L if side == "L" else POSTURE_R
        body = PALM_L if side == "L" else PALM_R
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body)

        q = np.array(q0, float)
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

    def ik_bimanual(self, data, center, half_width=0.085, lateral=(0.0, 1.0, 0.0), **kw):
        """Both palms on opposite sides of `center`, offset along `lateral` (the robot's LEFT axis).

        `lateral` matters: the offset is the robot's left/right direction, not world +y. Hardcoding
        world +y works only while the robot happens to face +x -- once it turns to face the place
        table the same call puts one palm in front of the object and the other behind it, and the
        solve blows up to a metre-scale residual instead of failing loudly.

        Returns (qL, qR, worst_position_error_m).
        """
        lat = np.asarray(lateral, float)
        lat = lat / (np.linalg.norm(lat) + 1e-12)
        tL = np.array(center, float) + lat * half_width
        tR = np.array(center, float) - lat * half_width
        qL, eL = self.ik_arm(data, "L", tL, **kw)
        qR, eR = self.ik_arm(data, "R", tR, **kw)
        return qL, qR, max(eL, eR)

    # ---- grasping -----------------------------------------------------------
    def closed_pose(self, side, fraction=1.0):
        """Finger angles at `fraction` of full closure, derived from the joint LIMITS.

        The two Dex3 hands are mirrored: left middle/index close negative and right positive, left
        thumb_2 closes positive and right negative. A single hand-authored "closed" vector applied to
        both therefore closes one hand and is silently clamped to zero on the other -- which is exactly
        what made the object look like it was passing through an open left hand. Reading the direction
        off each joint's own range makes the pose correct on either hand by construction.
        """
        lim = self.lim_handL if side == "L" else self.lim_handR
        out = np.zeros(7)
        for i, (lo, hi) in enumerate(lim):
            if abs(lo) < 1e-9:            # one-sided, closes positive
                out[i] = hi * fraction
            elif abs(hi) < 1e-9:          # one-sided, closes negative
                out[i] = lo * fraction
            else:                         # straddles zero (thumb base): mirror by side
                out[i] = (hi if side == "L" else lo) * THUMB_FRACTION * fraction
        return out

    def close_on_object(self, data, obj_geoms, apply_fn, lo=0.0, hi=1.0, steps=14, back_off=0.06):
        """Close both hands until the fingers first touch `obj_geoms`, then ease off slightly.

        Bisects the closure fraction against MuJoCo's own contact detection rather than trusting a
        hand-tuned angle, so the fingers land ON the surface for whatever object radius is in the
        scene. `apply_fn(fraction)` must pose the hands and refresh kinematics.
        """
        obj = set(obj_geoms)
        hand = self._hand_geoms(data)

        def touching(f):
            apply_fn(f)
            for c in data.contact[: data.ncon]:
                if (c.geom1 in obj and c.geom2 in hand) or (c.geom2 in obj and c.geom1 in hand):
                    return True
            return False

        if not touching(hi):
            return hi                       # never reaches the object; stay fully closed
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            if touching(mid):
                hi = mid
            else:
                lo = mid
        return max(0.0, hi * (1.0 - back_off))

    def _hand_geoms(self, data):
        if getattr(self, "_hg", None) is None:
            bodies = {i for i in range(self.m.nbody)
                      if "hand" in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or "")}
            self._hg = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in bodies}
        return self._hg

    _hg = None

    # ---- wheeled-base extras (no-ops on the legged model) -------------------
    def set_lift(self, data, height):
        """Extend the two-stage Z-lift by `height` metres, split across the stages."""
        if self.q_lift is None:
            return
        h = float(np.clip(height, 0.0, self.lift_max))
        data.qpos[self.q_lift] = h / 2.0

    def spin_wheels(self, data, distance, radius=None):
        """Roll the drive wheels to match `distance` travelled -- pure cosmetics, but a mobile base
        sliding with frozen wheels is the single most obvious tell in a rendered clip."""
        if self.q_wheels is None:
            return
        r = radius if radius else self.wheel_radius(data)
        if r > 1e-6:
            data.qpos[self.q_wheels] = distance / r

    def wheel_radius(self, data):
        """Wheel radius inferred from the hub height when the base is standing on the floor."""
        try:
            return float(abs(data.body("Left_Wheel_Link").xpos[2] - self._floor_z))
        except Exception:  # noqa: BLE001
            return 0.0

    _floor_z = 0.0

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
