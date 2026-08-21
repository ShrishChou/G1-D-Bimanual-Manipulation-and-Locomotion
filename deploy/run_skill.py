"""Deterministic skill executor -- replay a hand-authored pick/carry skill (from teach_skill.py).

No policy, no perception: the object sits at a known pose (jig), so the whole motion is an open-loop sequence
of waypoints. Arm segments interpolate with a smoothstep profile (JOINT space by default; straight-line CART
via the teleop IK for flagged segments) with gravity feed-forward carrying the weight. Each hand's 3 fingers
interpolate to that waypoint's explicit 7-dim target at the waypoint's grip gain -- a GRASP waypoint targets
the fully-closed pose at a LOW gain, so the fingers press to contact and stall at a gentle, bounded force
("close as far as it can with k=1.2") instead of over-currenting.

  conda activate <env>
  cd $TELEOP_DIR
  python $REPO/deploy/run_skill.py --skill cylinder_pick.json --motion

Type in this terminal (no Enter):  SPACE = start,  q = abort (hold where it is),  ESC = quit.
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
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dex3_direct import Dex3Direct
from ee_ik import EEToJoints, ee9_to_T
from live_source import G1Interface

p = argparse.ArgumentParser()
p.add_argument("--skill", required=True, help="skill JSON from teach_skill.py")
p.add_argument("--iface", default="enp2s0")
p.add_argument("--motion", action="store_true", help="rt/arm_sdk (AI mode); omit for rt/lowcmd (debug)")
p.add_argument("--rate", type=float, default=120.0, help="command rate (Hz)")
p.add_argument("--approach-dur", type=float, default=3.0, help="seconds to ease from current pose into waypoint 0")
p.add_argument("--max-step", type=float, default=0.04, help="per-tick per-joint safety clamp (rad)")
p.add_argument("--speed", type=float, default=1.0, help="global speed scale (<1 = slower; e.g. 0.3 for a cautious first replay)")
args = p.parse_args()

skill_path = args.skill if os.path.isabs(args.skill) else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.skill)
skill = json.load(open(skill_path))
wps = skill["waypoints"]
assert wps, "skill has no waypoints"
need_cart = any(w.get("cart") for w in wps)

# Guard: two instances fight over DDS and both hang on subscribe. Refuse to start if another is alive.
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


_others = _other_pythons("run_skill.py")
if _others:
    sys.exit(f"[skill] another run_skill is already running (PID {' '.join(_others)}). Stop it first:\n"
             f"    kill {' '.join(_others)}\nthen wait ~5s (DDS lease) and retry.")

# Robust shutdown (see teach_skill.py): Ctrl-C works during the DDS wait; hard-exit with a 2s backstop.
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
    print("\n[skill] stopped", flush=True)
    os._exit(0)


signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
signal.signal(signal.SIGINT, _hard_exit)
signal.signal(signal.SIGTERM, _hard_exit)

print("[skill] connecting to robot DDS (Ctrl-C to abort)...", flush=True)
g1 = G1Interface(motion=args.motion, hands=False, cameras=False, iface=args.iface)   # arm only
dex = Dex3Direct(iface=args.iface)                                                    # hands direct
aq = lambda: np.asarray(g1.arm.get_current_dual_arm_q(), np.float64)
ee_conv = EEToJoints() if need_cart else None
if need_cart:
    print("[skill] CART segments present -> straight-line palm moves via teleop IK", flush=True)

dt = 1.0 / args.rate
smooth = lambda s: s * s * (3.0 - 2.0 * s)
LAST = {"q": aq(), "fL": np.zeros(7), "fR": np.zeros(7), "kp": 1.5}


class KeyPoller:
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


def send(q, fL, fR, kp):
    q = np.asarray(q, np.float64)
    g1.command_arm(q, np.zeros(14))   # pure position PD at kp=80 holds/tracks rock-solid; no gravity ff
    dex.set_pose("L", fL, kp=kp)
    dex.set_pose("R", fR, kp=kp)
    dex.publish()
    LAST.update(q=q, fL=np.asarray(fL), fR=np.asarray(fR), kp=kp)


def aborted(kp):
    k = kp.poll()
    return k in ("q", "\x1b")


def hold(q, fL, fR, kp, secs, poller):
    for _ in range(max(1, int(secs / dt))):
        send(q, fL, fR, kp)
        if aborted(poller):
            return True
        time.sleep(dt)
    return False


def _cart_ee9(palm_a, palm_b, s):
    """Straight-line palm interpolation: linear position lerp + geodesic (shortest-arc) rotation."""
    Ta, Tb = ee9_to_T(palm_a), ee9_to_T(palm_b)
    Ra, Rb = Ta[:3, :3], Tb[:3, :3]
    pos = (1.0 - s) * Ta[:3, 3] + s * Tb[:3, 3]
    R = Ra @ pin.exp3(s * pin.log3(Ra.T @ Rb))
    return np.concatenate([pos, R[:, 0], R[:, 1]])


def move(prev, tgt, dur, cart, poller):
    """Interpolate arm prev->tgt and each hand prev->tgt over dur at tgt's grip gain. Returns (last_q, aborted)."""
    pq, tq = np.asarray(prev["q"]), np.asarray(tgt["q"])
    pfL, tfL = np.asarray(prev["fingerL"]), np.asarray(tgt["fingerL"])
    pfR, tfR = np.asarray(prev["fingerR"]), np.asarray(tgt["fingerR"])
    kp = tgt.get("grip_kp", 1.5)
    n = max(2, int(dur / dt))
    last = pq.copy()
    for i in range(1, n + 1):
        s = smooth(i / n)
        if cart:
            q = ee_conv.to_joints(_cart_ee9(prev["palmL"], tgt["palmL"], s),
                                  _cart_ee9(prev["palmR"], tgt["palmR"], s), last)
        else:
            q = pq + (tq - pq) * s
        q = last + np.clip(q - last, -args.max_step, args.max_step)   # safety: bound per-tick motion
        send(q, pfL + (tfL - pfL) * s, pfR + (tfR - pfR) * s, kp)
        last = q
        if aborted(poller):
            return last, True
        time.sleep(dt)
    return last, False


try:
    with KeyPoller() as poller:
        print(f"[skill] {skill.get('name','?')} ({len(wps)} waypoints). SPACE=start  q=abort/hold  ESC=quit", flush=True)
        started, killed = False, False
        while not started:
            g1.command_arm(aq(), np.zeros(14))           # hold current arm; leave hands free until start
            dex.set_free("L"); dex.set_free("R"); dex.publish()
            k = poller.poll()
            if k == " ":
                started = True
            elif k in ("q", "\x1b"):
                killed = True
                break
            time.sleep(dt)

        if not killed:
            cur_fL, cur_fR = dex.read()
            cur = {"q": aq().tolist(), "palmL": wps[0]["palmL"], "palmR": wps[0]["palmR"],
                   "fingerL": cur_fL, "fingerR": cur_fR}
            print(f"[skill] approach -> wp0 ({args.approach_dur / args.speed:.1f}s)", flush=True)
            last, ab = move(cur, wps[0], args.approach_dur / args.speed, False, poller)
            if not ab:
                ab = hold(last, wps[0]["fingerL"], wps[0]["fingerR"], wps[0].get("grip_kp", 1.5), wps[0].get("settle", 0.3), poller)
            for i in range(1, len(wps)):
                if ab:
                    break
                w = wps[i]
                print(f"[skill] wp{i-1}->wp{i}  {'CART' if w.get('cart') else 'joint'}  dur {w.get('dur',2.5) / args.speed:.1f}s  "
                      f"grip_kp {w.get('grip_kp',1.5)}", flush=True)
                last, ab = move(wps[i - 1], w, w.get("dur", 2.5) / args.speed, w.get("cart", False), poller)
                if not ab:
                    ab = hold(last, w["fingerL"], w["fingerR"], w.get("grip_kp", 1.5), w.get("settle", 0.3), poller)
            print("[skill] done -- holding final pose (q/ESC to release)" if not ab else "[skill] ABORTED -- holding", flush=True)
            while True:
                send(last, LAST["fL"], LAST["fR"], LAST["kp"])
                if aborted(poller):
                    break
                time.sleep(dt)
finally:
    dex.stop()
    g1.stop()
    print("\n[skill] exit", flush=True)
