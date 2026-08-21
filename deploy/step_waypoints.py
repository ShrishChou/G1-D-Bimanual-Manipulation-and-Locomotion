"""Step through taught waypoints ONE AT A TIME to verify the arm reaches each reliably.

Run it, then each SPACE eases the arm to the next saved waypoint (and commands that waypoint's Dex3 finger
pose). No auto-sequencing -- you advance manually so you can inspect each pose.

    conda activate <env>
    cd $TELEOP_DIR
    python $REPO/deploy/step_waypoints.py \
        --skill $REPO/deploy/cylinder_wp.json

Keys (type in this terminal, no Enter):
    SPACE = go to the next waypoint      r = restart at waypoint 0
    q / ESC / Ctrl-C = stop

No --motion (routes to rt/lowcmd, the mode your robot listens to). Eases into each pose over --approach with a
per-tick clamp; start slow and keep a hand near the e-stop.
"""
import argparse
import json
import math
import os
import select
import signal
import sys
import termios
import threading
import time
import tty

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dex3_direct import Dex3Direct
from live_source import G1Interface
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

p = argparse.ArgumentParser()
p.add_argument("--skill", required=True, help="waypoint JSON from waypoint_teleop.py")
p.add_argument("--iface", default="enp2s0")
p.add_argument("--motion", action="store_true", help="rt/arm_sdk (AI mode); omit for rt/lowcmd (default)")
p.add_argument("--kp", type=float, default=150.0, help="arm hold gain (too high + fast moves trips the motor overload -> limp/snap; too low sags)")
p.add_argument("--gravity", action="store_true", help="add model gravity feed-forward so the arm holds firmly without sag (kp handles less)")
p.add_argument("--grip-kp", type=float, default=1.5, help="Dex3 finger gain when a waypoint sets the hands")
p.add_argument("--rate", type=float, default=120.0, help="command rate (Hz)")
p.add_argument("--approach", type=float, default=2.0, help="seconds each move takes (larger = slower/gentler)")
p.add_argument("--max-step", type=float, default=0.04, help="per-tick per-joint clamp (rad) -- safety cap only")
p.add_argument("--base-back", type=float, default=0.03, help="metres to drive the base BACKWARD on the final SPACE")
p.add_argument("--base-vel", type=float, default=0.03, help="base reverse speed (m/s, deliberately slow)")
args = p.parse_args()

dt = 1.0 / args.rate
skill = json.load(open(args.skill))
wps = skill.get("waypoints", [])
if not wps:
    print("[step] no waypoints in skill"); sys.exit(1)

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
    print("\n[step] stopped", flush=True)
    os._exit(0)


signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
signal.signal(signal.SIGINT, _hard_stop)
signal.signal(signal.SIGTERM, _hard_stop)

print("[step] connecting to robot DDS (Ctrl-C to abort)...", flush=True)
g1 = G1Interface(motion=args.motion, hands=False, cameras=False, iface=args.iface)
dex = Dex3Direct(kp=args.grip_kp, iface=args.iface, init_dds=False)
g1._set_arm_kp(args.kp)
aq = lambda: np.asarray(g1.arm.get_current_dual_arm_q(), np.float64)


def tauff(q):
    return g1.gravity_tau(q) if args.gravity else np.zeros(14)


# base locomotion (LocoClient) for the final slow back-off; odometry from SportModeState for closed-loop distance
loco = None
try:
    loco = LocoClient(); loco.SetTimeout(0.5); loco.Init()
except Exception as e:
    print(f"[step] LocoClient init failed ({e}) -- final base move will be skipped", flush=True)
_pose = {"v": None}
# odometry topic varies by firmware/mode -> subscribe to both and use whichever publishes
_state_subs = [ChannelSubscriber(t, SportModeState_) for t in ("rt/sportmodestate", "rt/lf/sportmodestate")]
for _s in _state_subs:
    _s.Init()


def _odom():
    while True:
        for _s in _state_subs:
            m = _s.Read()
            if m is not None:
                _pose["v"] = (float(m.position[0]), float(m.position[1]), float(m.imu_state.rpy[2]))
        time.sleep(0.005)


threading.Thread(target=_odom, daemon=True).start()


def base_back(dist, vel, poller):
    """Drive the base straight BACKWARD `dist` m at `vel` m/s. Closed-loop on odometry if available, else a
    timed open-loop move (distance = vel * time). Non-holonomic (reverse only). q aborts."""
    if loco is None:
        print("\n[step] no LocoClient -- cannot move the base", flush=True); return
    for _ in range(200):                                    # give odometry up to ~2 s to appear
        if _pose["v"] is not None:
            break
        time.sleep(0.01)
    if _pose["v"] is not None:
        x0, y0, _ = _pose["v"]
        print(f"\n[step] base BACK {dist*100:.0f} cm at {vel:.2f} m/s (closed-loop)...", flush=True)
        while math.hypot(_pose["v"][0] - x0, _pose["v"][1] - y0) < dist:
            loco.Move(-abs(vel), 0.0, 0.0)
            if poller.poll() in ("q", "\x1b"):
                break
            time.sleep(0.05)
        loco.StopMove()
        moved = math.hypot(_pose["v"][0] - x0, _pose["v"][1] - y0)
        print(f"[step] base moved back {moved*100:.1f} cm (odom)", flush=True)
    else:
        # No SportModeState -> the robot is in low-level mode; the high-level loco service isn't running, so
        # LocoClient.Move just errors (ClientStub). Skip cleanly rather than spam. Move the base in a separate
        # session with the robot in normal/sport mode.
        print("\n[step] base SKIPPED -- no odometry / loco service (robot is in low-level arm mode; "
              "LocoClient needs high-level mode). Move the base separately.", flush=True)


class KeyPoller:
    def __enter__(self):
        global _OLD
        self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); _OLD = self.old
        tty.setcbreak(self.fd); return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


def _smooth(s):
    return s * s * (3.0 - 2.0 * s)          # smoothstep: ease in and out (zero speed at both ends)


def go_to(wp, poller):
    """Move the arm to wp['q'] over --approach seconds on a smooth ease-in/out profile (constant duration,
    speed set by --approach), holding that waypoint's fingers throughout. Aborted?"""
    fL, fR = wp["fingerL"], wp["fingerR"]
    start = aq(); tgt = np.asarray(wp["q"], float)
    n = max(2, int(args.approach / dt))
    last = start
    for i in range(1, n + 1):
        des = start + (tgt - start) * _smooth(i / n)                       # time-parameterized trajectory
        want = last + np.clip(des - last, -args.max_step, args.max_step)   # per-tick clamp is a safety cap
        g1.command_arm(want, tauff(want))
        dex.set_pose("L", fL, kp=args.grip_kp); dex.set_pose("R", fR, kp=args.grip_kp); dex.publish()
        last = want
        if poller.poll() in ("q", "\x1b"):
            return True
        time.sleep(dt)
    return False


arm_done = False                # whether the arm sequence (all waypoints) has run
base_done = False               # whether the final base back-off has run
cmd = aq().copy()               # current arm hold target
grip = None                     # (fL, fR) to hold, or None -> hands free
print(f"[step] {len(wps)} waypoints as ONE move, then base back {args.base_back*100:.0f} cm. "
      "SPACE=run  r=restart  q=quit.", flush=True)
tick = 0
try:
    with KeyPoller() as poller:
        while True:
            g1.command_arm(cmd, tauff(cmd))
            if grip is None:
                dex.set_free("L"); dex.set_free("R")
            else:
                dex.set_pose("L", grip[0], kp=args.grip_kp); dex.set_pose("R", grip[1], kp=args.grip_kp)
            dex.publish()

            tick += 1
            if tick % 30 == 0:
                nxt = ("run arm sequence" if not arm_done
                       else (f"base back {args.base_back*100:.0f}cm" if not base_done else "done"))
                print(f"\r[step] SPACE -> {nxt}   ", end="", flush=True)

            k = poller.poll()
            if k == " ":
                if not arm_done:
                    abort = False
                    for j, w in enumerate(wps):                      # ONE continuous move through every waypoint
                        print(f"\n[step] -> waypoint {j} ({args.approach:.0f}s)...", flush=True)
                        if go_to(w, poller):
                            abort = True; break
                        cmd = np.asarray(w["q"], float)
                        grip = (w["fingerL"], w["fingerR"])
                    if abort:
                        break
                    arm_done = True
                    print("[step] arm sequence complete -- SPACE = base back", flush=True)
                elif not base_done:
                    base_back(args.base_back, args.base_vel, poller)   # final slow locomotion back-off
                    base_done = True
                    print("[step] sequence complete (r to restart)", flush=True)
                else:
                    print("\n[step] all done (r to restart)", flush=True)
            elif k == "r":
                arm_done = False; base_done = False
                print("\n[step] restart -- SPACE runs the arm sequence", flush=True)
            elif k in ("q", "\x1b"):
                break
            time.sleep(dt)
finally:
    g1.stop()
    print("\n[step] done", flush=True)
