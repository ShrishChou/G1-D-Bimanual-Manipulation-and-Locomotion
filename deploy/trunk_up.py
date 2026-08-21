"""Raise/lower the G1-D Deluxe trunk in software.

The trunk is VELOCITY-controlled on rt/cmd_hispeed.z (the same channel the joystick drives: z = up/down speed,
-1..+1; z>0 = up). Height feedback is rt/hispeed_state.y (metres). We publish z at --speed and close the loop on
the height, stopping (z=0) at current + --up.

    conda activate <env>
    python $REPO/deploy/trunk_up.py --up 0.03    # TEST small
    python $REPO/deploy/trunk_up.py --up 0.1524  # 6 inches

Ctrl-C stops (z=0). --up negative lowers. --max-height/--min-height clamp for safety.
"""
import argparse
import os
import signal
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Point32_ as mkPoint32
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_

p = argparse.ArgumentParser()
p.add_argument("--iface", default="enp2s0")
p.add_argument("--up", type=float, default=0.1524, help="metres to raise (negative = lower). 0.1524 = 6 inches")
p.add_argument("--speed", type=float, default=1.0, help="trunk velocity command magnitude 0..1 (smaller = slower)")
p.add_argument("--max-height", type=float, default=0.45, help="safety clamp on absolute trunk height (m)")
p.add_argument("--min-height", type=float, default=0.05, help="safety clamp (m)")
p.add_argument("--tol", type=float, default=0.004, help="arrival tolerance (m)")
p.add_argument("--rate", type=float, default=100.0)
p.add_argument("--timeout", type=float, default=30.0)
args = p.parse_args()

dt = 1.0 / args.rate
ChannelFactoryInitialize(0, args.iface)
pub = ChannelPublisher("rt/cmd_hispeed", Point32_); pub.Init()
sub = ChannelSubscriber("rt/hispeed_state", Point32_); sub.Init()


def height():
    m = sub.Read()
    return None if m is None else float(m.y)


def _vel(z):
    m = mkPoint32(); m.x = 0.0; m.y = 0.0; m.z = float(z); pub.Write(m)


def _stop_and_exit(*_):
    for _ in range(10):
        _vel(0.0); time.sleep(0.01)
    print("\n[trunk] stopped (z=0)", flush=True)
    os._exit(0)


signal.signal(signal.SIGINT, _stop_and_exit)
signal.signal(signal.SIGTERM, _stop_and_exit)

cur = None
for _ in range(300):
    cur = height()
    if cur is not None:
        break
    time.sleep(0.01)
if cur is None:
    print("[trunk] no hispeed_state -- aborting"); raise SystemExit(1)

target = min(args.max_height, max(args.min_height, cur + args.up))
up = target >= cur
z = args.speed if up else -args.speed
print(f"[trunk] {cur*100:.1f} cm -> {target*100:.1f} cm ({'UP' if up else 'DOWN'}, z={z:+.2f})", flush=True)

t0 = time.time()
while time.time() - t0 < args.timeout:
    _vel(z)
    h = height()
    if h is not None:
        print(f"\r[trunk] height {h*100:5.1f} / {target*100:.1f} cm  ", end="", flush=True)
        if (up and h >= target - args.tol) or (not up and h <= target + args.tol):
            break
    time.sleep(dt)

for _ in range(10):
    _vel(0.0); time.sleep(0.01)                # stop
print(f"\n[trunk] done at {(height() or target)*100:.1f} cm", flush=True)
