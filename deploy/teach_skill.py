"""Kinesthetic teach tool -- author a deterministic pick/carry skill by hand (runs entirely in the terminal).

The arm and the hands each toggle between HOLD (frozen target + firm kp -> holds firmly, cannot fall) and
FREE (target tracks your hand at the same firm kp, so the kp term cancels and it's light to move). HOLD is the
default, so nothing drops unless you deliberately unlock it. A raw terminal has no key-release event, so we
use SPACE / h as toggles rather than hold-to-move.

Each waypoint stores: the 14-dim arm pose, the palm pose (FK, torso frame, for straight-line CART segments),
and an explicit 7-dim target for EACH hand plus the grip gain to reach it. run_skill.py replays it.

  conda activate <env>
  cd $TELEOP_DIR
  python $REPO/deploy/teach_skill.py --out cylinder_pick.json --motion

Keys (type in this terminal, no Enter):
  SPACE  toggle ARM free/hold        h  toggle HANDS free/hold
  c  capture POSED waypoint          g  capture GRASP (fingers close at --grasp-kp)
  o  capture OPEN waypoint           t  toggle last waypoint JOINT<->CART
  u  undo last     s  save     q or ESC  quit
"""
import argparse
import json
import os
import select
import signal
import sys
import termios
import time
import tty

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dex3_direct import Dex3Direct
from ee_ik import _encode_palm_torso              # FK: (armL7, armR7) -> (palmL9, palmR9) torso frame
from live_source import CLOSED_L, CLOSED_R, G1Interface

p = argparse.ArgumentParser()
p.add_argument("--out", required=True, help="skill JSON to write (under this deploy dir unless absolute)")
p.add_argument("--iface", default="enp2s0")
p.add_argument("--motion", action="store_true", help="rt/arm_sdk (AI mode); omit for rt/lowcmd (debug)")
p.add_argument("--kp", type=float, default=80.0,
               help="HOLD gain: position PD holds the frozen pose (verified rock-solid at 80). Raise if it droops.")
p.add_argument("--free-kp", type=float, default=0.0,
               help="FREE-drive gain: 0 = fully limp/back-drivable (you support the arm's weight while moving "
                    "it). Higher adds resistance/stickiness -- keep at 0 for the lightest drag.")
p.add_argument("--pose-kp", type=float, default=2.0, help="finger gain to hold a posed configuration")
p.add_argument("--grasp-kp", type=float, default=1.2, help="finger gain for a 'g' torque-close (stalls on contact)")
p.add_argument("--dur", type=float, default=2.5, help="default arm-move seconds stored on each waypoint")
p.add_argument("--settle", type=float, default=0.3, help="default dwell seconds stored on each waypoint")
args = p.parse_args()

out_path = args.out if os.path.isabs(args.out) else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
OPEN = np.zeros(7)

# Guard: two instances fight over the same DDS domain and both hang on subscribe. Refuse to start if another
# teach_skill is already alive (this is what caused the pile-up + "Waiting to subscribe dds..." forever).
# Scan /proc for PYTHON processes only (so the wrapping shell/pgrep/grep can't self-match).
def _other_pythons(tag):
    me = os.getpid()
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cl = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if cl and cl[0] and os.path.basename(cl[0].decode("utf-8", "ignore")).startswith("python") \
                and tag in b" ".join(cl).decode("utf-8", "ignore"):
            out.append(d)
    return out


_others = _other_pythons("teach_skill.py")
if _others:
    sys.exit(f"[teach] another teach_skill is already running (PID {' '.join(_others)}). Stop it first:\n"
             f"    kill {' '.join(_others)}\nthen wait ~5s (DDS lease) and retry.")

# Robust shutdown: register BEFORE constructing G1Interface so Ctrl-C works even during the DDS-subscribe wait.
# Hard-exit with a 2s SIGALRM backstop so a blocking DDS stop() can never wedge the process (which piles up
# duplicate DDS participants and breaks the next run's discovery).
g1 = None
dex = None
_OLD_TERM = None


def _hard_exit(*_):
    signal.alarm(2)
    try:
        if _OLD_TERM is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _OLD_TERM)
    except Exception:
        pass
    for obj in (dex, g1):
        try:
            if obj is not None:
                obj.stop()
        except Exception:
            pass
    print("\n[teach] stopped", flush=True)
    os._exit(0)


signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
signal.signal(signal.SIGINT, _hard_exit)
signal.signal(signal.SIGTERM, _hard_exit)

print("[teach] connecting to robot DDS (Ctrl-C to abort)...", flush=True)
g1 = G1Interface(motion=args.motion, hands=False, cameras=False, iface=args.iface)   # arm only
dex = Dex3Direct(kp=args.pose_kp, iface=args.iface)                                   # hands direct
aq = lambda: np.asarray(g1.arm.get_current_dual_arm_q(), np.float64)
g1._set_arm_kp(args.kp)


class KeyPoller:
    """Read single keystrokes from the terminal without blocking or needing Enter."""
    def __enter__(self):
        global _OLD_TERM
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        _OLD_TERM = self.old
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


def save():
    json.dump({"name": os.path.splitext(os.path.basename(out_path))[0], "waypoints": wps}, open(out_path, "w"), indent=2)
    print(f"\n[teach] saved {len(wps)} waypoints -> {out_path}", flush=True)


def capture(fingerL, fingerR, grip_kp, kind):
    q = aq()
    palmL, palmR, _ = _encode_palm_torso(q[:7], q[7:])
    wps.append({"name": f"wp{len(wps)}", "q": q.tolist(), "palmL": palmL.tolist(), "palmR": palmR.tolist(),
                "fingerL": np.asarray(fingerL).tolist(), "fingerR": np.asarray(fingerR).tolist(),
                "grip_kp": round(grip_kp, 3), "cart": False, "dur": args.dur, "settle": args.settle})
    print(f"\n[teach] captured wp{len(wps)-1} [{kind}]  grip_kp {grip_kp}", flush=True)


wps = []
arm_free = False                      # False = HOLD frozen pose (kp high); True = free-drive (kp low)
held_q = aq()                         # the pose HOLD locks onto
hands_free = False
time.sleep(0.5)                       # let the Dex3 reader threads catch a first state sample
held_fL, held_fR = dex.read()
if not dex.hands_online():
    print("[teach] WARNING: Dex3 hands are NOT publishing state (fingers read as 0). Power/enable BOTH hands "
          "to pose/capture fingers. If one hand is dead, reset it (deploy/reset_dex3.py). Arm still works.",
          flush=True)

print("[teach] SPACE = let go (free-drive) <-> hold.  Default HOLDS.  h=hands free/hold  c/g/o=capture  "
      "s=save  q=quit", flush=True)
last = time.time()
tick = 0
try:
    with KeyPoller() as kp:
        while True:
            now = time.time(); dt = now - last; last = now
            if arm_free:
                g1.command_arm(aq(), np.zeros(14))            # limp: no position pull, no gravity ff -> back-drive
            else:
                g1.command_arm(held_q, np.zeros(14))          # HOLD: pure position PD at kp -> rock-solid (verified)

            qL, qR = dex.read()
            if hands_free:
                held_fL, held_fR = qL, qR
                dex.set_free("L"); dex.set_free("R")
            else:
                dex.set_pose("L", held_fL, kp=args.pose_kp); dex.set_pose("R", held_fR, kp=args.pose_kp)
            dex.publish()

            tick += 1
            if tick % 25 == 0:
                hl = "on" if dex.seen["L"] else "OFF"
                hr = "on" if dex.seen["R"] else "OFF"
                print(f"\rARM {'FREE' if arm_free else 'HOLD'} | HANDS {'FREE' if hands_free else 'HOLD'} "
                      f"(L:{hl} R:{hr}) | waypoints {len(wps)}   ", end="", flush=True)

            k = kp.poll()
            if k is None:
                time.sleep(0.008); continue
            if k == " ":
                arm_free = not arm_free
                if arm_free:
                    g1._set_arm_kp(args.free_kp)              # let go
                    print("\n[teach] ARM FREE -- support & move it", flush=True)
                else:
                    held_q = aq()                             # lock onto wherever it is now
                    g1._set_arm_kp(args.kp)
                    print("\n[teach] ARM HOLD -- locked here", flush=True)
            elif k == "h":
                hands_free = not hands_free
                if not hands_free:
                    held_fL, held_fR = dex.read()
                print(f"\n[teach] HANDS {'FREE -- pose them' if hands_free else 'HOLD -- frozen'}", flush=True)
            elif k == "c":
                capture(qL, qR, args.pose_kp, "posed")
            elif k == "g":
                capture(CLOSED_L, CLOSED_R, args.grasp_kp, "grasp")
            elif k == "o":
                capture(OPEN, OPEN, args.pose_kp, "open")
            elif k == "t" and wps:
                wps[-1]["cart"] = not wps[-1]["cart"]
                print(f"\n[teach] wp{len(wps)-1} -> {'CART' if wps[-1]['cart'] else 'joint'}", flush=True)
            elif k == "u" and wps:
                wps.pop(); print(f"\n[teach] undo -> {len(wps)} waypoints", flush=True)
            elif k == "s":
                save()
            elif k in ("q", "\x1b"):
                if wps and not os.path.exists(out_path):
                    save()
                break
            time.sleep(0.008)
finally:
    dex.stop()
    g1.stop()
    print("\n[teach] done", flush=True)
