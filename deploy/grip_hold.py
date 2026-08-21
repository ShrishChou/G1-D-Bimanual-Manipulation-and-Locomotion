"""Standalone constant-force grip holder for the two Dex3 hands -- test that a steady inward squeeze alone
keeps the object in the air, independent of the arms/base.

Force-closure grip: command the FULLY-CLOSED pose (CLOSED_L/R) at a LOW kp. The object physically blocks the
fingers, so the position error stays constant -> each motor applies a constant inward torque = kp * error. The
squeeze self-limits (if the object shifts, the fingers follow to the new surface at the same force) and it is
re-published every tick, so it holds continuously -- through arm moves, base turns, whatever runs alongside it.

    conda activate <env>
    python $REPO/deploy/grip_hold.py --force 1.0

Keys (type in this terminal, no Enter):
    SPACE = toggle SQUEEZE on/off      o = open (release, back-drivable)
    [ / ] = decrease / increase force  q / ESC / Ctrl-C = release + quit

Workflow: run it (starts OPEN), place the object between the hands, tap SPACE to squeeze, check it holds.
Tune --force (kp) so it holds without straining the motors. Run this with the teleop / run_skill STOPPED
(two publishers on rt/dex3/*/cmd fight).
"""
import argparse
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
from live_source import CLOSED_L, CLOSED_R

p = argparse.ArgumentParser()
p.add_argument("--iface", default="enp2s0")
p.add_argument("--force", type=float, default=1.0, help="grip gain kp (higher = harder squeeze); force ~= kp * (closed - contact)")
p.add_argument("--kd", type=float, default=0.1, help="finger damping")
p.add_argument("--rate", type=float, default=100.0, help="command rate (Hz)")
p.add_argument("--step", type=float, default=0.2, help="force change per [ / ] press")
args = p.parse_args()

dt = 1.0 / args.rate
dex = None
_OLD = None


def _release_and_exit(*_):
    signal.alarm(2)
    try:
        if _OLD is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _OLD)
    except Exception:
        pass
    try:
        if dex is not None:
            dex.stop()                 # set both hands free (back-drivable) + publish
    except Exception:
        pass
    print("\n[grip_hold] released", flush=True)
    os._exit(0)


signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
signal.signal(signal.SIGINT, _release_and_exit)
signal.signal(signal.SIGTERM, _release_and_exit)

print("[grip_hold] connecting to Dex3 hands (Ctrl-C to abort)...", flush=True)
dex = Dex3Direct(kp=args.force, kd=args.kd, iface=args.iface, init_dds=True)
for _ in range(300):                   # wait until both hands publish state
    if dex.hands_online():
        break
    time.sleep(0.01)
if not dex.hands_online():
    onl = {s: dex.seen[s] for s in ("L", "R")}
    print(f"[grip_hold] WARNING: hand state not seen for both hands {onl} -- check the hand is powered/publishing.", flush=True)


class KeyPoller:
    def __enter__(self):
        global _OLD
        self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); _OLD = self.old
        tty.setcbreak(self.fd); return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


def force_proxy(force):
    """Approximate current inward effort per hand: kp * |closed - measured| (Nm-ish), a stand-in for grip force."""
    qL, qR = dex.read()
    return force * float(np.abs(CLOSED_L - qL).max()), force * float(np.abs(CLOSED_R - qR).max())


force = args.force
squeeze = False
print(f"[grip_hold] OPEN. Place the object, tap SPACE to squeeze. force(kp)={force:.2f}  "
      "[ / ]=adjust  o=open  q=quit.", flush=True)
tick = 0
try:
    with KeyPoller() as poller:
        while True:
            if squeeze:
                dex.set_pose("L", CLOSED_L, kp=force)
                dex.set_pose("R", CLOSED_R, kp=force)
            else:
                dex.set_free("L"); dex.set_free("R")
            dex.publish()

            tick += 1
            if tick % 20 == 0:
                if squeeze:
                    fL, fR = force_proxy(force)
                    print(f"\r[grip_hold] SQUEEZE force(kp)={force:.2f}  effort L~{fL:.2f} R~{fR:.2f}   ", end="", flush=True)
                else:
                    print(f"\r[grip_hold] OPEN     force(kp)={force:.2f}                          ", end="", flush=True)

            k = poller.poll()
            if k == " ":
                squeeze = not squeeze
                print(f"\n[grip_hold] {'SQUEEZE' if squeeze else 'OPEN'}", flush=True)
            elif k == "o":
                squeeze = False
                print("\n[grip_hold] OPEN", flush=True)
            elif k == "]":
                force += args.step
                print(f"\n[grip_hold] force(kp)={force:.2f}", flush=True)
            elif k == "[":
                force = max(0.0, force - args.step)
                print(f"\n[grip_hold] force(kp)={force:.2f}", flush=True)
            elif k in ("q", "\x1b"):
                break
            time.sleep(dt)
finally:
    dex.stop()
    print("\n[grip_hold] done", flush=True)
