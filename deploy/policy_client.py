"""Shared deploy core: GR00T /act client (3-camera cylinder model) + workspace governance.

No Isaac / no hardware deps -> importable from the sim runner (Isaac env) and the compliant
hardware runner alike.  Camera + state reading is done by the caller (see sources); this module
just formats the observation, queries the policy, and enforces the safety box.
"""
import numpy as np

try:
    import requests
    import json_numpy
    json_numpy.patch()
except Exception:  # allow import for governance-only use / tests
    requests = None

INSTRUCTION = "Pick up the cylinder from the table and hold it close to your chest."

# state / action layout (matches the trained unitree_g1_cylinder config, 28-dim)
SL = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "left_hand": slice(14, 21), "right_hand": slice(21, 28)}


class PolicyClient:
    """POST an observation to the GR00T HTTP inference server and parse the 16-step action chunk."""

    def __init__(self, host="localhost", port=8000, instruction=INSTRUCTION):
        self.url = f"http://{host}:{port}/act"
        self.instruction = instruction

    def query(self, head, left_wrist, right_wrist, state28):
        """head/left_wrist/right_wrist: (H,W,3) uint8.  state28: (28,) float. -> dict of (16,7) arrays."""
        if requests is None:
            raise RuntimeError("policy_client needs 'requests' + 'json_numpy' in this env -- "
                               "run: pip install requests json-numpy")
        s = np.asarray(state28, np.float32)
        obs = {
            "video.head": np.ascontiguousarray(head)[None],
            "video.left_wrist": np.ascontiguousarray(left_wrist)[None],
            "video.right_wrist": np.ascontiguousarray(right_wrist)[None],
            "state.left_arm": s[SL["left_arm"]][None],
            "state.right_arm": s[SL["right_arm"]][None],
            "state.left_hand": s[SL["left_hand"]][None],
            "state.right_hand": s[SL["right_hand"]][None],
            "annotation.human.task_description": [self.instruction],
        }
        r = requests.post(self.url, data=json_numpy.dumps({"observation": obs}),
                          headers={"Content-Type": "application/json"})
        r.raise_for_status()
        return json_numpy.loads(r.text)

    @staticmethod
    def chunk_to_target(action, t):
        """Assemble the 28-dim joint target at horizon step t (arm14 + hand14), model state order."""
        return np.concatenate([
            np.asarray(action["action.left_arm"])[t], np.asarray(action["action.right_arm"])[t],
            np.asarray(action["action.left_hand"])[t], np.asarray(action["action.right_hand"])[t],
        ]).astype(np.float32)


class Governance:
    """Axis-aligned end-effector safety box with a z-floor. The policy cannot drive a hand outside it."""

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, np.float64)
        self.hi = np.asarray(hi, np.float64)

    @classmethod
    def from_hands(cls, lw, rw, down_limit=0.05, margin_x=0.22, margin_y_back=0.12,
                   reach_forward=0.50, lift=0.22):
        """Build the box from the two start-pose hand positions + governance margins."""
        hx, hy, hz = np.array([lw[0], rw[0]]), np.array([lw[1], rw[1]]), np.array([lw[2], rw[2]])
        lo = np.array([hx.min() - margin_x, hy.min() - margin_y_back, hz.min() - down_limit])
        hi = np.array([hx.max() + margin_x, reach_forward, hz.max() + lift])
        return cls(lo, hi)

    def clamp(self, p):
        return np.minimum(np.maximum(np.asarray(p, np.float64), self.lo), self.hi)

    def violations(self, ee_left, ee_right):
        """Return a list of human-readable violations for the two end-effector points (empty = safe)."""
        out = []
        for name, p in (("left", ee_left), ("right", ee_right)):
            p = np.asarray(p, np.float64)
            if p[2] < self.lo[2]:
                out.append(f"{name} hand below z-floor ({p[2]:.3f} < {self.lo[2]:.3f})")
            for ax, i in (("x", 0), ("y", 1), ("z", 2)):
                if p[i] < self.lo[i] - 1e-6 or p[i] > self.hi[i] + 1e-6:
                    out.append(f"{name} hand out of box on {ax} ({p[i]:.3f} not in "
                               f"[{self.lo[i]:.3f},{self.hi[i]:.3f}])")
        return out
