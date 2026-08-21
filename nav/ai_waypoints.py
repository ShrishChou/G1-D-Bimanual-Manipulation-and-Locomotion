#!/usr/bin/env python3
"""Generate the ai-mode pick/place waypoints (absolute map-frame poses) from the CURRENT start pose
plus the offsets we built into deploy_pick, with 30 cm staging for the arm raise / final approach.
Saves to nav/waypoints.json so nav_move.py --name <wp> can drive each one drift-free (closed-loop on
rt/robot_pose). Run FROM the starting place:

  python ai_waypoints.py            # capture start + print plan (saves waypoints)
  python ai_waypoints.py --end-fwd 2.0 --end-right 0.0

Geometry (all relative to the captured start S = current pose):
  approach:  forward --fwd0, turn LEFT --turn deg, forward --fwd-infer   -> INFERENCE pose
  infer_stage = 30 cm (--stage-back) BEHIND inference along its heading   (raise arms here)
  place (from the deploy post-pick offsets): from inference, back --post-back along the inference heading,
        rotate RIGHT to the start heading, forward --post-fwd -> END_PRETRUNK (raise trunk here),
        then forward --post-fwd2 -> END_FINAL (drop). Both face the start heading.
Sequence to run (drift-free via nav_move):
  start -> infer_stage -> (raise arms + trunk up) -> infer -> PICK -> infer_stage -> (trunk down)
        -> end_pretrunk -> (raise trunk) -> end_final -> (drop)
"""
import argparse
import json
import math
import os
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_

WP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoints.json")


def yaw_of(o):
    return math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))


def d(yaw):
    return (math.cos(yaw), math.sin(yaw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp2s0")
    ap.add_argument("--fwd0", type=float, default=0.60, help="forward before the turn (m)")
    ap.add_argument("--turn", type=float, default=90.0, help="turn LEFT toward the table (deg, +CCW)")
    ap.add_argument("--fwd-infer", type=float, default=0.569, help="forward after the turn to reach inference (m)")
    ap.add_argument("--stage-back", type=float, default=0.30, help="staging offset behind inference (m)")
    ap.add_argument("--post-back", type=float, default=0.60, help="back off along inference heading after pick (m)")
    ap.add_argument("--post-fwd", type=float, default=0.981, help="forward (start heading) to the pre-trunk-raise end pose (m)")
    ap.add_argument("--post-fwd2", type=float, default=0.4318, help="forward to the final/drop pose after raising trunk (m)")
    ap.add_argument("--start-x", type=float, help="override START x (map, m) instead of capturing live pose")
    ap.add_argument("--start-y", type=float, help="override START y (map, m)")
    ap.add_argument("--start-yaw", type=float, help="override START yaw (map, DEGREES)")
    a = ap.parse_args()

    if a.start_x is not None and a.start_y is not None and a.start_yaw is not None:
        S = (a.start_x, a.start_y, math.radians(a.start_yaw))
        print(f"[ai_waypoints] using OVERRIDE start (not live pose): {S[0]:.3f},{S[1]:.3f},{a.start_yaw:.1f}deg")
    else:
        ChannelFactoryInitialize(0, a.iface)
        sub = ChannelSubscriber("rt/robot_pose", PoseStamped_); sub.Init()
        S = None; t0 = time.time()
        while S is None and time.time() - t0 < 5:
            m = sub.Read()
            if m is not None:
                S = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))
            time.sleep(0.02)
        if S is None:
            print("[ai_waypoints] no rt/robot_pose -- is the robot connected?"); return
    sx, sy, sθ = S

    # --- approach / inference ---
    ax, ay = sx + a.fwd0 * d(sθ)[0], sy + a.fwd0 * d(sθ)[1]      # after fwd0, heading sθ
    th1 = sθ + math.radians(a.turn)                              # turn left
    ix = ax + a.fwd_infer * d(th1)[0]; iy = ay + a.fwd_infer * d(th1)[1]   # inference pose
    isx = ix - a.stage_back * d(th1)[0]; isy = iy - a.stage_back * d(th1)[1]  # 30cm behind inference

    # --- end / place (deploy post-pick geometry: back along inference heading, rotate right to start
    #     heading, forward to pre-trunk pose, forward again to final/drop) ---
    bx = ix - a.post_back * d(th1)[0]; by = iy - a.post_back * d(th1)[1]        # backed off from inference
    epx = bx + a.post_fwd * d(sθ)[0]; epy = by + a.post_fwd * d(sθ)[1]          # pre-trunk-raise end pose
    efx = epx + a.post_fwd2 * d(sθ)[0]; efy = epy + a.post_fwd2 * d(sθ)[1]      # final / drop pose

    wp = {}
    if os.path.exists(WP_FILE):
        wp = json.load(open(WP_FILE))
    wp.update({
        "start":        {"x": sx,  "y": sy,  "yaw": sθ},
        "infer_stage":  {"x": isx, "y": isy, "yaw": th1},
        "infer":        {"x": ix,  "y": iy,  "yaw": th1},
        "end_pretrunk": {"x": epx, "y": epy, "yaw": sθ},
        "end_final":    {"x": efx, "y": efy, "yaw": sθ},
    })
    json.dump(wp, open(WP_FILE, "w"), indent=2)

    print(f"[ai_waypoints] START captured: ({sx:.3f}, {sy:.3f}, {math.degrees(sθ):.1f} deg)")
    print("[ai_waypoints] plan (map frame), saved to", WP_FILE)
    for k in ["start", "infer_stage", "infer", "end_pretrunk", "end_final"]:
        w = wp[k]
        print(f"    {k:12s} x={w['x']:7.3f} y={w['y']:7.3f} yaw={math.degrees(w['yaw']):7.1f} deg")
    print("\nRun order (each drift-free via nav_move --name <wp> --go):")
    print("  1 infer_stage  (drive)    2 raise arms + trunk up    3 infer (fwd 30cm)    4 PICK")
    print("  5 infer_stage  (back 30)  6 trunk down               7 end_pretrunk (drive)")
    print("  8 raise trunk             9 end_final (drive)       10 drop")


if __name__ == "__main__":
    main()
