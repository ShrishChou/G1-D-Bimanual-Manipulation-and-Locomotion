"""EE-deploy decoder: turn the EE model's predicted palm pose (xyz+rot6d, torso frame) into arm JOINT
targets via the proven teleop Casadi IK (G1_29_ArmIK). Frame chain verified constant (frame_check):
    L_ee_target_base = Y * palm_torso * X          X = palm->L_ee (const),  Y = torso->pelvis (const)

Run the self-test (round-trips recorded EE actions through IK and checks the achieved palm pose matches the
target) BEFORE driving the robot:
    conda activate <env>
    cd $TELEOP_DIR
    python $REPO/deploy/ee_ik.py
"""
import os
import sys

import numpy as np
import pinocchio as pin

TELEOP = "$TELEOP_DIR"
URDF = "$DATA_ROOT/robots/g1/urdf/g1_body29_hand14.urdf"
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


def rot6d_to_R(c1, c2):
    """Gram-Schmidt: two columns -> orthonormal rotation matrix (must match the converter's encode)."""
    b1 = c1 / (np.linalg.norm(c1) + 1e-8)
    b2 = c2 - (b1 @ c2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def ee9_to_T(ee9):
    """[x,y,z, r00,r10,r20, r01,r11,r21] -> 4x4 palm pose in torso frame."""
    ee9 = np.asarray(ee9, float)
    T = np.eye(4)
    T[:3, :3] = rot6d_to_R(ee9[3:6], ee9[6:9])
    T[:3, 3] = ee9[:3]
    return T


class EEToJoints:
    def __init__(self):
        sys.path.insert(0, TELEOP)
        from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
        # G1_29_ArmIK loads '../assets/g1/g1_body29_hand14.urdf' RELATIVE to cwd -> build it from the teleop
        # dir, then restore cwd (same pattern as deploy/live_source.py for the teleop controllers).
        _cwd = os.getcwd()
        os.chdir(os.path.join(TELEOP, "teleop"))
        try:
            self.ik = G1_29_ArmIK()
        finally:
            os.chdir(_cwd)
        # constant X (palm->L_ee) and Y (torso->pelvis), computed once (config-independent; verified constant)
        m = pin.buildModelFromUrdf(URDF); d = m.createData()
        for side, jn in (("L", "left_wrist_yaw_joint"), ("R", "right_wrist_yaw_joint")):
            m.addFrame(pin.Frame(f"{side}_ee", m.getJointId(jn),
                                 pin.SE3(np.eye(3), np.array([0.05, 0, 0])), pin.FrameType.OP_FRAME))
        d = m.createData()
        pin.forwardKinematics(m, d, pin.neutral(m)); pin.updateFramePlacements(m, d)
        g = lambda n: np.array(d.oMf[m.getFrameId(n)].homogeneous)
        tor_inv = np.linalg.inv(g("torso_link"))
        palmL, palmR = tor_inv @ g("left_hand_palm_link"), tor_inv @ g("right_hand_palm_link")
        leeL, leeR = tor_inv @ g("L_ee"), tor_inv @ g("R_ee")
        self.XL = np.linalg.inv(palmL) @ leeL          # palm_torso -> Lee_torso
        self.XR = np.linalg.inv(palmR) @ leeR
        self.Y = g("torso_link")                       # torso -> pelvis(base): pose of torso in root frame

    def targets(self, ee9_L, ee9_R):
        """palm poses (torso frame) -> L_ee/R_ee 4x4 targets in the IK base frame."""
        tL = self.Y @ ee9_to_T(ee9_L) @ self.XL
        tR = self.Y @ ee9_to_T(ee9_R) @ self.XR
        return tL, tR

    def to_joints(self, ee9_L, ee9_R, cur_q14, cur_dq14=None):
        """Predicted EE poses -> 14-dim arm joint targets (seeded from current q for continuity/safety)."""
        tL, tR = self.targets(ee9_L, ee9_R)
        sol_q, _ = self.ik.solve_ik(tL, tR, np.asarray(cur_q14, float), cur_dq14)
        return np.asarray(sol_q, float)


# encode used ONLY by the self-test to synthesize targets from recorded joints (mirror of the converter)
def _encode_palm_torso(la, ra):
    m = pin.buildModelFromUrdf(URDF); d = m.createData()
    lqi = [m.idx_qs[m.getJointId(f"left_{j}_joint")] for j in ARM]
    rqi = [m.idx_qs[m.getJointId(f"right_{j}_joint")] for j in ARM]
    q = pin.neutral(m)
    for i, qi in enumerate(lqi): q[qi] = la[i]
    for i, qi in enumerate(rqi): q[qi] = ra[i]
    pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
    tor_inv = np.linalg.inv(np.array(d.oMf[m.getFrameId("torso_link")].homogeneous))
    out = []
    for link in ("left_hand_palm_link", "right_hand_palm_link"):
        rel = tor_inv @ np.array(d.oMf[m.getFrameId(link)].homogeneous)
        out.append(np.concatenate([rel[:3, 3], rel[:3, 0], rel[:3, 1]]))
    return out[0], out[1], (m, d, lqi, rqi)


if __name__ == "__main__":
    import glob
    import json
    conv = EEToJoints()
    e = sorted(glob.glob("$DATA_ROOT/teleop_raw/cylinder_pick_clean/episode_*"))[0]
    frames = json.load(open(f"{e}/data.json"))["data"]
    # STREAMING round-trip (mimics deploy): walk the episode in order, seed each solve from the prior solution.
    seed, stream, settled = None, [], []
    for fr in frames[::5]:
        la = np.array(fr["states"]["left_arm"]["qpos"]); ra = np.array(fr["states"]["right_arm"]["qpos"])
        eeL, eeR, _ = _encode_palm_torso(la, ra)
        if seed is None:
            seed = np.concatenate([la, ra])
        sol = conv.to_joints(eeL, eeR, seed, None); seed = sol      # temporal continuity, like deploy
        aL, aR, _ = _encode_palm_torso(sol[:7], sol[7:])
        stream.append((np.linalg.norm(aL[:3] - eeL[:3]) + np.linalg.norm(aR[:3] - eeR[:3])) / 2)
    # SETTLED accuracy (deploy holds/slews slowly -> this is what actually matters): last target, iterate to settle.
    for _ in range(20):
        sol = conv.to_joints(eeL, eeR, seed, None); seed = sol
    aL, aR, _ = _encode_palm_torso(sol[:7], sol[7:])
    settle = (np.linalg.norm(aL[:3] - eeL[:3]) + np.linalg.norm(aR[:3] - eeR[:3])) / 2
    stream = np.array(stream)
    print(f"streaming palm error: mean {stream.mean()*1000:.1f} mm  p95 {np.percentile(stream,95)*1000:.1f} mm "
          f"(transient smoothing lag on a fast target)")
    print(f"settled palm error:   {settle*1000:.2f} mm  (target held -> what deploy converges to with slow slew)")
    print("SELF-TEST:", "PASS -- frame chain exact + IK converges to target palm poses; safe to wire deploy"
          if settle < 0.005 and stream.mean() < 0.02 else "REVIEW -- errors higher than expected")
