#!/usr/bin/env python3
"""Sport-mode base control via the loco service (reachable only in sport/advanced mode).
In ai mode the base took rt/cmd_vel_no_limit; in SPORT mode the base is driven by loco.SetVelocity /
loco.Move. Use this for fine positioning at the pick/place site; use carry_nav.py for long transit.

`tv` env:
  python base_sport.py --check                    # loco reachable?
  python base_sport.py --vx 0.15 --secs 2 --go    # creep forward 0.15 m/s for 2 s
  python base_sport.py --vyaw 0.3 --secs 1 --go   # rotate
Dry-run (no --go) just reports loco reachability + what it WOULD send. Ctrl-C stops.
"""
import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp2s0")
    ap.add_argument("--vx", type=float, default=0.0, help="forward vel (m/s)")
    ap.add_argument("--vy", type=float, default=0.0, help="lateral vel (m/s)")
    ap.add_argument("--vyaw", type=float, default=0.0, help="yaw rate (rad/s, +CCW)")
    ap.add_argument("--secs", type=float, default=1.0, help="duration to hold the velocity")
    ap.add_argument("--check", action="store_true", help="just report loco reachability")
    ap.add_argument("--go", action="store_true", help="ACTUALLY move (robot moves). Omit = dry-run.")
    a = ap.parse_args()

    ChannelFactoryInitialize(0, a.iface)
    loco = LocoClient(); loco.SetTimeout(3.0); loco.Init()
    code = loco.GetServerApiVersion()
    reachable = isinstance(code, tuple) and code[0] == 0
    print(f"[base_sport] loco GetServerApiVersion={code} -> {'REACHABLE (sport mode)' if reachable else 'NOT reachable (are we in sport mode?)'}")
    if a.check or not a.go:
        print(f"[base_sport] DRY-RUN: would Move(vx={a.vx}, vy={a.vy}, vyaw={a.vyaw}) for {a.secs}s")
        return
    if not reachable:
        print("[base_sport] ABORT: loco not reachable."); return

    print(f"[base_sport] Move(vx={a.vx}, vy={a.vy}, vyaw={a.vyaw}) for {a.secs}s ...", flush=True)
    t0 = time.time()
    try:
        while time.time() - t0 < a.secs:
            loco.Move(a.vx, a.vy, a.vyaw, True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        loco.StopMove()
        print("[base_sport] StopMove.")


if __name__ == "__main__":
    main()
