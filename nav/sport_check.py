#!/usr/bin/env python3
"""RPC half of the sport-mode readiness gate (mode + loco + front camera). Runs over unitree_sdk2py
(`tv` env). The topic-liveness half (3D LiDAR cloud, frontvideostream, robot_pose, nav_to_pose) is done
via ROS2 in sport_check.sh -- reading the big PointCloud2/video types over unitree_sdk2py can block, so
we keep those on the ROS2 side.

  python sport_check.py
"""
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.go2.video.video_client import VideoClient

IFACE = "enp2s0"
OUT = "/tmp/sport_front_cam.jpg"


def line(tag, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag:5s}  {detail}", flush=True)
    return ok


def main():
    ChannelFactoryInitialize(0, IFACE)
    print("----- RPC checks (mode / loco / front cam) -----", flush=True)
    allok = True

    ms = MotionSwitcherClient(); ms.SetTimeout(2.0); ms.Init()
    code, mode = ms.CheckMode()
    name = (mode or {}).get("name", "?")
    allok &= line("mode", name != "ai" and code == 0, f"CheckMode={mode}  (need != 'ai')")

    loco = LocoClient(); loco.SetTimeout(1.5); loco.Init()
    lc = loco.GetServerApiVersion()
    allok &= line("loco", isinstance(lc, tuple) and lc[0] == 0, f"GetServerApiVersion={lc}  (base driving)")

    vc = VideoClient(); vc.SetTimeout(1.5); vc.Init()
    t0 = time.time()
    c, d = vc.GetImageSample()
    ok = c == 0 and bool(d)
    if ok:
        open(OUT, "wb").write(bytes(d))
    allok &= line("fcam", ok, f"code={c} bytes={len(d) if d else 0} ({time.time()-t0:.1f}s)"
                  + (f" -> {OUT}" if ok else "  (3102=off in ai)"))

    print("RPC:", "ALL PASS" if allok else "some FAIL (expected in ai mode)", flush=True)


if __name__ == "__main__":
    main()
