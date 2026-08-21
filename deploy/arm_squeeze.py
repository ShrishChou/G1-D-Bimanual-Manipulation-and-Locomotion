"""Standalone BIMANUAL arm-squeeze tester -- hold an object by compressing it between the two palms.

This is NOT a finger grip. Both arms push toward each other, aiming at a point INSIDE the object: each palm's
target is shifted inward (toward the other palm) by --depth. The object blocks the palms, so the joint
position error becomes a constant inward push (force ~= kp * error). When they settle against the object they
hold that relative pose while still pressing -- exactly "keep pushing in, then hold position + force". Gravity
is carried by the kp=80 position loop, so no gravity model is needed.

    conda activate <env>
    cd $TELEOP_DIR
    python $REPO/deploy/arm_squeeze.py --motion --depth 0.03

Keys (type in this terminal, no Enter):
    SPACE = engage/disengage squeeze      [ / ] = less / more squeeze depth
    o = release to current pose (stop pushing)   q / ESC / Ctrl-C = stop

Workflow: pose both palms on either side of the object (freedrive/teleop), run this, tap SPACE. It ramps the
palms inward by --depth and holds -- the object stops them and they compress it. Tune --depth for grip force.
SAFETY: with NO object between the hands the palms will move --depth toward each other; keep it small (2-4 cm)
and start with a hand ready near the e-stop.
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
from ee_ik import EEToJoints, _encode_palm_torso
from live_source import G1Interface

p = argparse.ArgumentParser()
p.add_argument("--iface", default="enp2s0")
p.add_argument("--motion", action="store_true", help="rt/arm_sdk (AI mode); omit for rt/lowcmd (debug)")
p.add_argument("--kp", type=float, default=80.0, help="arm position gain (squeeze force ~= kp * joint error at contact)")
p.add_argument("--depth", type=float, default=0.03, help="m: how far each palm aims PAST contact, toward the other palm (grip force)")
p.add_argument("--rate", type=float, default=120.0, help="command rate (Hz)")
p.add_argument("--ramp", type=float, default=1.5, help="seconds to ease into the squeeze target")
p.add_argument("--goto", help="skill JSON (from waypoint_teleop): move to its saved arm pose on the first SPACE")
p.add_argument("--approach", type=float, default=3.0, help="seconds to ease into the --goto start pose")
p.add_argument("--max-step", type=float, default=0.03, help="per-tick per-joint clamp (rad) while ramping")
p.add_argument("--step", type=float, default=0.01, help="m: depth change per [ / ] press")
args = p.parse_args()

dt = 1.0 / args.rate
g1 = None
_OLD = None


def _hard_stop(*_):
    signal.alarm(2)
    try:
        if _OLD is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _OLD)
    except Exception:
        pass
    try:
        if g1 is not None:
            g1.stop()
    except Exception:
        pass
    print("\n[arm_squeeze] stopped", flush=True)
    os._exit(0)


signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
signal.signal(signal.SIGINT, _hard_stop)
signal.signal(signal.SIGTERM, _hard_stop)

print("[arm_squeeze] connecting to robot DDS (Ctrl-C to abort)...", flush=True)
g1 = G1Interface(motion=args.motion, hands=False, cameras=False, iface=args.iface)
g1._set_arm_kp(args.kp)
ee = EEToJoints()
aq = lambda: np.asarray(g1.arm.get_current_dual_arm_q(), np.float64)


def squeeze_target(q14, depth):
    """From the current pose, shift each palm toward the other by `depth` (aim inside the object); IK -> q14."""
    palmL, palmR, _ = _encode_palm_torso(q14[:7], q14[7:])
    posL, posR = palmL[:3].copy(), palmR[:3].copy()
    d = posR - posL
    n = np.linalg.norm(d)
    if n < 1e-6:
        return q14.copy()
    u = d / n                                   # unit vector L -> R (the inward direction for the left palm)
    tgtL = np.concatenate([posL + depth * u, palmL[3:]])   # move L toward R
    tgtR = np.concatenate([posR - depth * u, palmR[3:]])   # move R toward L
    return np.asarray(ee.to_joints(tgtL, tgtR, q14), float)


class KeyPoller:
    def __enter__(self):
        global _OLD
        self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); _OLD = self.old
        tty.setcbreak(self.fd); return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


def ramp_to(target, poller, dur=None):
    """Ease the arm command from the current pose to `target`, per-tick clamped. Returns aborted?"""
    last = aq()
    n = max(2, int((dur if dur else args.ramp) / dt))
    for i in range(1, n + 1):
        s = i / n
        want = last + np.clip(target - last, -args.max_step, args.max_step)
        g1.command_arm(want, np.zeros(14))
        last = want
        k = poller.poll()
        if k in ("q", "\x1b"):
            return True
        time.sleep(dt)
    return False


goto_wps = []
if args.goto:
    _d = json.load(open(args.goto))
    goto_wps = [np.asarray(w["q"], float) for w in _d["waypoints"]]      # ordered approach poses
    print(f"[arm_squeeze] loaded {len(goto_wps)} start pose(s) from {args.goto}", flush=True)

depth = args.depth
engaged = False
wp_idx = 0                     # how many saved poses we've moved through
cmd = aq().copy()             # what we're currently commanding (hold pose)
if not goto_wps:
    print(f"[arm_squeeze] HOLDING. Tap SPACE to squeeze (depth={depth*100:.1f} cm, kp={args.kp:.0f}).  "
          "[ / ]=depth  o=release  q=quit.", flush=True)
else:
    print(f"[arm_squeeze] HOLDING. Tap SPACE to step through {len(goto_wps)} saved pose(s) in order, "
          "then SPACE again to squeeze. o=release  q=quit.", flush=True)
tick = 0
try:
    with KeyPoller() as poller:
        while True:
            g1.command_arm(cmd, np.zeros(14))     # hold the current commanded target (squeeze or hold pose)

            tick += 1
            if tick % 30 == 0:
                err = float(np.abs(cmd - aq()).max())     # blocked-by-object joint error ~ squeeze effort
                st = f"SQUEEZE depth={depth*100:.1f}cm" if engaged else "HOLD          "
                print(f"\r[arm_squeeze] {st}  kp={args.kp:.0f}  joint-err~{err:.3f} rad   ", end="", flush=True)

            k = poller.poll()
            if k == " ":
                if wp_idx < len(goto_wps):
                    print(f"\n[arm_squeeze] moving to saved pose {wp_idx+1}/{len(goto_wps)} ({args.approach:.0f}s)...", flush=True)
                    if ramp_to(goto_wps[wp_idx], poller, dur=args.approach):
                        break
                    cmd = goto_wps[wp_idx].copy(); wp_idx += 1
                    print(("[arm_squeeze] at final pose. Tap SPACE to squeeze." if wp_idx >= len(goto_wps)
                           else f"[arm_squeeze] at pose {wp_idx}/{len(goto_wps)}. SPACE for the next."), flush=True)
                elif not engaged:
                    tgt = squeeze_target(aq(), depth)
                    print(f"\n[arm_squeeze] engaging squeeze (depth={depth*100:.1f} cm)...", flush=True)
                    if ramp_to(tgt, poller):
                        break
                    cmd = tgt; engaged = True
                    print("[arm_squeeze] SQUEEZING -- holding relative pose + pushing in", flush=True)
                else:
                    cmd = aq().copy(); engaged = False        # freeze at where it is now
                    print("\n[arm_squeeze] released to current pose (HOLD)", flush=True)
            elif k == "o":
                cmd = aq().copy(); engaged = False
                print("\n[arm_squeeze] released to current pose (HOLD)", flush=True)
            elif k in ("]", "["):
                depth = max(0.0, depth + (args.step if k == "]" else -args.step))
                print(f"\n[arm_squeeze] depth={depth*100:.1f} cm", flush=True)
                if engaged:                                    # re-aim from the current pose at the new depth
                    tgt = squeeze_target(aq(), depth)
                    if ramp_to(tgt, poller):
                        break
                    cmd = tgt
            elif k in ("q", "\x1b"):
                break
            time.sleep(dt)
finally:
    g1.stop()
    print("\n[arm_squeeze] done", flush=True)
