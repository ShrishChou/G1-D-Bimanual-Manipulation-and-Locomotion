"""Non-VR kinesthetic teach for the G1 arms, SPACE-toggled -- the simple, reliable version.

  LOCKED (default): firm position hold (kp high, pure PD) -- rock-solid, holds where it is, even with the
                    heavy Dex3 hands on. While locked it measures the holding torque (= gravity at this pose).
  LIGHT (tap SPACE): low kp + that measured gravity fed forward -> the arm is light, move it by hand.
  tap SPACE again -> LOCKS the current pose and holds it.

So: tap SPACE, move the arm where you want, tap SPACE to lock it, press w to capture the waypoint. Repeat.
(Terminal can't detect key-release, so SPACE is a toggle rather than press-and-hold -- same result.)

    conda activate <env>
    cd $TELEOP_DIR
    python $REPO/deploy/freedrive.py --out cylinder_pick.json

SAFETY: it starts LOCKED (holding). Only tap SPACE once you have a hand on the arm. Ctrl-C stops immediately.
If it droops while LOCKED, raise --lock-kp. If it's too heavy in LIGHT, lower --move-kp.
"""
import argparse
import json
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
from ee_ik import _encode_palm_torso
from live_source import G1Interface

_HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--iface", default="enp2s0")
p.add_argument("--motion", action="store_true", help="rt/arm_sdk (AI mode)")
p.add_argument("--comp", choices=["space", "hold"], default="space",
               help="space = SPACE toggles light(move)/locked(hold). hold = always stiff PD with deadband give-way.")
p.add_argument("--lock-kp", type=float, default=80.0, help="LOCKED hold gain (raise if it droops under the hands)")
p.add_argument("--move-kp", type=float, default=6.0, help="LIGHT gain while you move it (lower = lighter)")
p.add_argument("--grav-gain", type=float, default=1.0, help="measured-gravity feed-forward scale in LIGHT (droops -> >1, floats -> <1)")
p.add_argument("--lpf", type=float, default=0.05, help="how fast the measured gravity is tracked while LOCKED")
p.add_argument("--tau-clamp", type=float, default=20.0, help="Nm: hard clamp on the feed-forward torque (safety)")
p.add_argument("--deadband", type=float, default=0.05, help="hold-mode sticky-setpoint give-way threshold (rad)")
p.add_argument("--diag", action="store_true", help="print measured tau_est vs model gravity per arm joint")
p.add_argument("--out", help="skill JSON to write if you capture waypoints")
args = p.parse_args()

out_path = (args.out if not args.out or os.path.isabs(args.out)
            else os.path.join(_HERE, args.out))

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
    print("\n[freedrive] stopped", flush=True)
    os._exit(0)


signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
signal.signal(signal.SIGINT, _hard_stop)
signal.signal(signal.SIGTERM, _hard_stop)

print("[freedrive] connecting to robot DDS (Ctrl-C to abort)...", flush=True)
g1 = G1Interface(motion=args.motion, hands=False, cameras=False, iface=args.iface)
aq = lambda: np.asarray(g1.arm.get_current_dual_arm_q(), np.float64)

# The arm controller's lowstate buffer strips tau_est, so read the RAW lowstate ourselves (read-only
# subscriber -> no DDS-publisher conflict, no factory re-init) for the measured joint torques.
from unitree_sdk2py.core.channel import ChannelSubscriber           # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_        # noqa: E402
from teleop.robot_control.robot_arm import G1_29_JointArmIndex      # noqa: E402
_ARM_IDS = [int(j) for j in G1_29_JointArmIndex]
_ls_sub = ChannelSubscriber("rt/lowstate", LowState_); _ls_sub.Init()
_tau = {"v": np.zeros(14)}


def _ls_reader():
    while True:
        m = _ls_sub.Read()
        if m is not None:
            try:
                _tau["v"] = np.array([m.motor_state[i].tau_est for i in _ARM_IDS], float)
            except Exception:
                pass
        time.sleep(0.005)


threading.Thread(target=_ls_reader, daemon=True).start()


def arm_tau_est():
    return _tau["v"].copy()


# bootstrap: hold firmly ~1.5 s so tau_est settles to the true holding torque (= gravity + hands at this pose)
g1._set_arm_kp(args.lock_kp)
q0 = aq()
for _ in range(75):
    g1.command_arm(q0, np.zeros(14))
    time.sleep(0.02)
grav = arm_tau_est().copy()
setpoint = aq().copy()
light = False


class KeyPoller:
    def __enter__(self):
        global _OLD
        self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); _OLD = self.old
        tty.setcbreak(self.fd); return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


wps = []
if args.comp == "space":
    print(f"[freedrive] LOCKED (holding). Tap SPACE -> LIGHT (move by hand); tap SPACE -> lock the pose. "
          f"w=add waypoint  s=save  q=quit (Ctrl-C stops).", flush=True)
else:
    print(f"[freedrive] HOLD kp={args.lock_kp}: push a joint past the deadband and it gives way, then holds. "
          "w=add waypoint  s=save  q=quit (Ctrl-C stops).", flush=True)

tick = 0
try:
    with KeyPoller() as kp:
        while True:
            q = aq(); tau = arm_tau_est()
            if args.comp == "space":
                if light:
                    setpoint = q.copy()                                  # follow your hand
                    tau_ff = args.grav_gain * np.clip(grav, -args.tau_clamp, args.tau_clamp)
                else:
                    grav = (1 - args.lpf) * grav + args.lpf * tau        # keep measuring gravity while locked
                    tau_ff = np.zeros(14)                                # pure PD hold -> rock-solid
            else:
                err = q - setpoint
                setpoint = np.where(np.abs(err) > args.deadband, q, setpoint)
                tau_ff = np.zeros(14)
            g1.command_arm(setpoint, tau_ff)

            tick += 1
            if tick % 30 == 0:
                st = (("LIGHT (moving)  " if light else "LOCKED (holding)") if args.comp == "space" else "hold")
                print(f"\rfree-drive [{st}] | waypoints {len(wps)}   ", end="", flush=True)
            if args.diag and tick % 50 == 0:
                print(f"\n[diag] tau_est {np.round(tau, 2)}")
                print(f"[diag] grav_ff {np.round(args.grav_gain * grav, 2)}", flush=True)

            k = kp.poll()
            if k == " " and args.comp == "space":
                light = not light
                if light:
                    g1._set_arm_kp(args.move_kp)
                    print(f"\n[freedrive] LIGHT -- move the arm (kp={args.move_kp})", flush=True)
                else:
                    setpoint = aq().copy()                               # lock the current pose FIRST
                    g1._set_arm_kp(args.lock_kp)                         # then stiffen -> holds it
                    print("\n[freedrive] LOCKED", flush=True)
            elif k == "w":
                palmL, palmR, _ = _encode_palm_torso(q[:7], q[7:])
                wps.append({"name": f"wp{len(wps)}", "q": q.tolist(), "palmL": palmL.tolist(), "palmR": palmR.tolist(),
                            "fingerL": [0] * 7, "fingerR": [0] * 7, "grip_kp": 1.5, "cart": False, "dur": 2.5, "settle": 0.3})
                print(f"\n[freedrive] captured wp{len(wps)-1}", flush=True)
            elif k == "s" and out_path:
                json.dump({"name": os.path.splitext(os.path.basename(out_path))[0], "waypoints": wps},
                          open(out_path, "w"), indent=2)
                print(f"\n[freedrive] saved {len(wps)} waypoints -> {out_path}", flush=True)
            elif k in ("q", "\x1b"):
                break
            time.sleep(0.01)
finally:
    g1.stop()
    print("\n[freedrive] done", flush=True)
