"""Compliant deploy loop for the bimanual policy, with a live camera preview.

Design notes that matter on real hardware:
  - LIGHT control loop: read cameras only when re-querying the policy; heavy per-cycle work starves the
    low-level command stream and the motor watchdog will drop the arms.
  - MARCHING target: hold a persistent command and step it toward the policy target by a capped `slew`
    each cycle (slew * hz = joint-speed cap). Do NOT re-anchor to the measured position each cycle -- it
    never converges. This is how you move slowly and safely, not by crushing the velocity limit.
  - GOVERNANCE box: clamp end-effector targets to a workspace box with a z-floor.
  - DELTA policies: reconstruct absolute target = measured_state_at_query + policy_output (match training).
  - Live preview: show what the policy actually sees; SPACE starts the policy, 'q' stops + saves a video.

Bring your own hardware I/O by implementing the RobotIO protocol below (read / command_arm /
command_hand_scalar). The policy is served over HTTP by an Isaac-GR00T inference server; PolicyClient
formats the observation and parses the action chunk.

  python deploy/policy_runner.py --host localhost --port 8000 [--delta] --slew 0.008 --hz 50
"""
import argparse, time
from typing import Protocol
import numpy as np

INSTRUCTION = "Pick up the cylinder from the table and lift it to the chest."
SL = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "left_hand": slice(14, 21), "right_hand": slice(21, 28)}
# closed dexterous-hand poses (thumb0,thumb1,thumb2,middle0,middle1,index0,index1); tune to your hand
CLOSED_L = np.array([0.0, -0.9, -1.7, 1.4, 1.6, 1.4, 1.6])
CLOSED_R = np.array([0.0, 0.9, 1.7, -1.4, -1.6, -1.4, -1.6])


class RobotIO(Protocol):
    def read(self):  # -> (head_rgb, left_wrist_rgb, right_wrist_rgb, state28)
        ...
    def command_arm(self, q14: np.ndarray): ...
    def command_hand_scalar(self, grasp_left: float, grasp_right: float): ...


class PolicyClient:
    """POST an observation to the Isaac-GR00T HTTP inference server, parse the action chunk."""
    def __init__(self, host="localhost", port=8000, instruction=INSTRUCTION):
        import requests, json_numpy
        json_numpy.patch()
        self._requests, self._json = requests, json_numpy
        self.url = f"http://{host}:{port}/act"; self.instruction = instruction

    def query(self, head, lw, rw, state28):
        s = np.asarray(state28, np.float32)
        obs = {"video.head": np.ascontiguousarray(head)[None],
               "video.left_wrist": np.ascontiguousarray(lw)[None],
               "video.right_wrist": np.ascontiguousarray(rw)[None],
               "state.left_arm": s[SL["left_arm"]][None], "state.right_arm": s[SL["right_arm"]][None],
               "state.left_hand": s[SL["left_hand"]][None], "state.right_hand": s[SL["right_hand"]][None],
               "annotation.human.task_description": [self.instruction]}
        r = self._requests.post(self.url, data=self._json.dumps({"observation": obs}),
                                headers={"Content-Type": "application/json"})
        r.raise_for_status()
        return self._json.loads(r.text)

    @staticmethod
    def chunk_to_target(action, t):
        return np.concatenate([np.asarray(action["action.left_arm"])[t], np.asarray(action["action.right_arm"])[t],
                               np.asarray(action["action.left_hand"])[t], np.asarray(action["action.right_hand"])[t]]).astype(np.float32)


def grasp_scalar(target7, closed7):
    d = float(np.dot(closed7, closed7))
    return float(np.clip(np.dot(np.asarray(target7), closed7) / d, 0.0, 1.0)) if d > 0 else 0.0


def montage(head, lw, rw, banner):
    import cv2
    cells = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), (320, 240)) for x in (head, lw, rw)]
    row = np.hstack(cells)
    bar = np.zeros((30, row.shape[1], 3), np.uint8)
    cv2.putText(bar, banner, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return np.vstack([bar, row])


def run(io: RobotIO, args):
    import cv2
    pc = PolicyClient(args.host, args.port)
    dt = 1.0 / args.hz
    head, lw, rw, state = io.read()
    arm_cmd = np.asarray(state[:14], np.float64); gl = gr = 0.0
    chunk, hi, started, frames = None, 0, False, []
    state_q = np.asarray(state, np.float64)
    try:
        while True:
            head, lw, rw, state = io.read()
            live = "LIVE" if head.mean() > 2 else "BLACK-no stream!"
            frame = montage(head, lw, rw, f"{'POLICY' if started else 'HOLD (SPACE=start)'} cam:{live} q=stop+save")
            frames.append(frame); cv2.imshow("policy runner", frame)
            k = cv2.waitKey(1) & 0xFF
            if not started:
                io.command_arm(arm_cmd)
                if k == ord(' '):
                    started = True
                elif k == ord('q'):
                    break
            else:
                if chunk is None or hi >= args.requery:
                    chunk = pc.query(head, lw, rw, state); hi = 0; state_q = np.asarray(state, np.float64)
                tgt = PolicyClient.chunk_to_target(chunk, hi); hi += 1
                if args.delta:
                    tgt = tgt + state_q
                arm_cmd += np.clip(tgt[:14] - arm_cmd, -args.slew, args.slew)
                gl += float(np.clip(grasp_scalar(tgt[14:21], CLOSED_L) - gl, -args.hand_step, args.hand_step))
                gr += float(np.clip(grasp_scalar(tgt[21:28], CLOSED_R) - gr, -args.hand_step, args.hand_step))
                io.command_arm(arm_cmd); io.command_hand_scalar(gl, gr)
                if k == ord('q'):
                    break
            time.sleep(dt)
    finally:
        cv2.destroyAllWindows()
        if frames:
            h_, w_ = frames[0].shape[:2]
            vw = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w_, h_))
            for f in frames:
                vw.write(f)
            vw.release()
            print(f"[saved] review video -> {args.save}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--delta", action="store_true")
    p.add_argument("--slew", type=float, default=0.008, help="max arm target march per step (rad)")
    p.add_argument("--hand-step", type=float, default=0.03)
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--requery", type=int, default=8)
    p.add_argument("--save", default="review.mp4")
    args = p.parse_args()
    raise SystemExit("Implement RobotIO for your hardware and call run(io, args). "
                     "This module ships the control-loop logic, not a hardware driver.")
