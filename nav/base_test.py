"""Standalone G1-D base-motion test: move SLOWLY ~1 inch forward/back and turn a few degrees, closed-loop on
odometry, so you can confirm the base responds correctly and the direction signs are right BEFORE running the
full FSM. Non-holonomic (forward/back + turn only). Ctrl-C = StopMove + exit at any moment.

    conda activate <env>
    python base_test.py --iface enp2s0
Keys (type in this terminal):  f=fwd 1in   b=back 1in   l=turn left 5deg   r=turn right 5deg   q=quit
"""
import argparse
import math
import os
import select
import signal
import sys
import termios
import threading
import time
import tty

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

IN = 0.0254
V = 0.05          # m/s  (deliberately slow)
W = 0.20          # rad/s
STATE_TOPIC = "rt/sportmodestate"      # TODO(robot): verify

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="enp2s0")
ap.add_argument("--step-in", type=float, default=1.0, help="straight step size (inches)")
ap.add_argument("--turn-deg", type=float, default=5.0, help="turn step size (degrees)")
a = ap.parse_args()

ChannelFactoryInitialize(0, a.iface)
loco = LocoClient(); loco.SetTimeout(0.5); loco.Init()
pose = {"v": None}
sub = ChannelSubscriber(STATE_TOPIC, SportModeState_); sub.Init()


def _odom():
    while True:
        m = sub.Read()
        if m is not None:
            pose["v"] = (float(m.position[0]), float(m.position[1]), float(m.imu_state.rpy[2]))
        time.sleep(0.005)


threading.Thread(target=_odom, daemon=True).start()


def _kill(*_):
    try:
        loco.StopMove()
    except Exception:
        pass
    print("\n[base_test] Ctrl-C -> StopMove", flush=True)
    os._exit(0)


signal.signal(signal.SIGINT, _kill)
signal.signal(signal.SIGTERM, _kill)


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def wait_odom():
    for _ in range(300):
        if pose["v"] is not None:
            return True
        time.sleep(0.01)
    return False


def drive_straight(inches):
    if not wait_odom():
        print("[base_test] NO ODOMETRY -- aborting move"); return
    x0, y0, _ = pose["v"]; target = abs(inches) * IN; sgn = 1 if inches > 0 else -1
    while math.hypot(pose["v"][0] - x0, pose["v"][1] - y0) < target:
        loco.Move(sgn * V, 0.0, 0.0); time.sleep(0.05)
    loco.StopMove()
    print(f"[base_test] moved {inches:+.1f} in  (odom {math.hypot(pose['v'][0]-x0, pose['v'][1]-y0)/IN:.2f} in)", flush=True)


def turn(deg):
    if not wait_odom():
        print("[base_test] NO ODOMETRY -- aborting turn"); return
    _, _, y0 = pose["v"]; target = math.radians(abs(deg)); sgn = 1 if deg > 0 else -1
    while abs(wrap(pose["v"][2] - y0)) < target:
        loco.Move(0.0, 0.0, sgn * W); time.sleep(0.05)
    loco.StopMove()
    print(f"[base_test] turned {deg:+.1f} deg  (odom {math.degrees(abs(wrap(pose['v'][2]-y0))):.1f})", flush=True)


print("[base_test] READY. f=fwd b=back l=left r=right q=quit  (Ctrl-C stops immediately)", flush=True)
fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
try:
    tty.setcbreak(fd)
    while True:
        if select.select([sys.stdin], [], [], 0.1)[0]:
            k = sys.stdin.read(1)
            if k == "f": drive_straight(a.step_in)
            elif k == "b": drive_straight(-a.step_in)
            elif k == "l": turn(a.turn_deg)
            elif k == "r": turn(-a.turn_deg)
            elif k in ("q", "\x1b"): break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    loco.StopMove()
    print("\n[base_test] done", flush=True)
