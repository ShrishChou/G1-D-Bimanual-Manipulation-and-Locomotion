#!/usr/bin/env python3
"""Lean place/drop for SPORT mode (arm_sdk): hold the current grasped pose, open the Dex3 hands,
and spread the arms outward to release the object. Arm-only (no base) -- pair with carry_nav.py for
the transit. Mirrors deploy_pick.py's drop step but standalone and mode-agnostic.

`tv` env:
  python place_drop.py --motion            # DRY-RUN (prints plan, no motion)
  python place_drop.py --motion --go       # release: open hands + spread arms
--motion => rt/arm_sdk (sport mode). Omit --motion only for ai/lowcmd debug.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iface", default="enp2s0")
    p.add_argument("--motion", action="store_true", help="rt/arm_sdk (sport mode); omit for rt/lowcmd (ai/debug)")
    p.add_argument("--spread", type=float, default=0.15, help="move each palm outward laterally (m)")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--go", action="store_true", help="ACTUALLY move (robot moves). Omit = dry-run.")
    args = p.parse_args()
    dt = 1.0 / args.hz

    from live_source import G1Interface

    print(f"[place_drop] {'DRIVING (arm_sdk)' if args.motion else 'lowcmd'}  spread={args.spread}m  "
          f"{'GO' if args.go else 'DRY-RUN'}", flush=True)
    g1 = G1Interface(motion=args.motion, hands=True, cameras=False, iface=args.iface)
    ik = g1.arm_ik

    def gtau(q):
        try:
            return g1.gravity_tau(q)
        except Exception:
            return np.zeros(14)

    arm_cmd = np.asarray(g1.arm.get_current_dual_arm_q(), float)
    print("[place_drop] current arm q captured (holding).", flush=True)

    if not args.go:
        Tl, Tr = ik.fk(arm_cmd)
        print(f"[place_drop] DRY-RUN: would open hands, then spread palms by {args.spread}m "
              f"(L +y, R -y) from current pose. No motion sent.")
        return

    # 1) open both hands, keep arm held
    print("[place_drop] opening claws ...", flush=True)
    for _ in range(int(1.0 / dt)):
        g1.command_arm(arm_cmd, tauff=gtau(arm_cmd)); g1.command_hand_scalar(0.0, 0.0)
        time.sleep(dt)

    # 2) spread arms outward (release)
    print("[place_drop] arms outward ...", flush=True)
    Tl, Tr = ik.fk(arm_cmd)
    Tl[1, 3] += args.spread; Tr[1, 3] -= args.spread
    try:
        q_out, _ = ik.solve_ik(Tl, Tr, np.asarray(g1.arm.get_current_dual_arm_q()),
                               np.asarray(g1.arm.get_current_dual_arm_dq()))
    except Exception as e:
        print(f"[place_drop] arms-out IK failed ({e}); skipping spread", flush=True); q_out = arm_cmd.copy()
    oc = arm_cmd.copy()
    t0 = time.time()
    while time.time() - t0 < 4.0:
        step = np.clip(q_out - oc, -0.02, 0.02)
        oc = oc + step
        g1.command_arm(oc, tauff=gtau(oc)); g1.command_hand_scalar(0.0, 0.0)
        if float(np.abs(q_out - oc).max()) < 0.02:
            break
        time.sleep(dt)
    print("[place_drop] released. holding open pose. Ctrl-C to finish.", flush=True)
    try:
        while True:
            g1.command_arm(oc, tauff=gtau(oc)); g1.command_hand_scalar(0.0, 0.0)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
