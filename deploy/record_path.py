"""Kinesthetic path recorder for the G1 arms -- move the arms by hand at low kp, record the whole trajectory,
replay it later.

RECORD (default): the arms sit at a LOW kp (compliant -> easy to push by hand; hands set free/back-drivable).
  SPACE  = start recording (samples arm q, waist yaw, and both Dex3 finger poses every tick)
  SPACE  = stop + SAVE the trajectory to --out
  q/ESC/Ctrl-C = quit

    conda activate <env>
    cd $TELEOP_DIR
    python $REPO/deploy/record_path.py --motion \
        --out $REPO/deploy/path1.json

REPLAY: play a saved trajectory back at a firm kp (position control), optionally slower.
    python .../record_path.py --motion --replay .../path1.json --speed 0.5
  SPACE = start, q/ESC = abort/hold.

SAFETY: in RECORD the arm is soft and will sag if you let go -- keep a hand on it. In REPLAY it eases into the
first frame over --approach seconds; start with --speed 0.5 and a hand near the e-stop. Ctrl-C stops immediately.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dex3_direct import Dex3Direct
from live_source import G1Interface

p = argparse.ArgumentParser()
p.add_argument("--iface", default="enp2s0")
p.add_argument("--motion", action="store_true", help="rt/arm_sdk (AI mode); omit for rt/lowcmd (debug)")
p.add_argument("--replay", help="trajectory JSON to replay (omit to RECORD)")
p.add_argument("--out", help="trajectory JSON to write when recording")
p.add_argument("--kp", type=float, default=15.0, help="RECORD: low arm gain so it's easy to move by hand")
p.add_argument("--replay-kp", type=float, default=80.0, help="REPLAY: firm arm gain for position playback")
p.add_argument("--grip-kp", type=float, default=1.5, help="REPLAY: Dex3 finger gain")
p.add_argument("--rate", type=float, default=100.0, help="sample/command rate (Hz)")
p.add_argument("--speed", type=float, default=1.0, help="REPLAY speed scale (<1 = slower)")
p.add_argument("--approach", type=float, default=3.0, help="REPLAY: seconds to ease into the first frame")
p.add_argument("--max-step", type=float, default=0.04, help="REPLAY: per-tick per-joint clamp (rad)")
p.add_argument("--stride", type=int, default=0, help="REPLAY: play every Nth frame (0 = auto to ~100 fps; raise for oversampled files)")
args = p.parse_args()

dt = 1.0 / args.rate
g1 = None
dex = None
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
    print("\n[record_path] stopped", flush=True)
    os._exit(0)


signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
signal.signal(signal.SIGINT, _hard_stop)
signal.signal(signal.SIGTERM, _hard_stop)

print("[record_path] connecting to robot DDS (Ctrl-C to abort)...", flush=True)
g1 = G1Interface(motion=args.motion, hands=False, cameras=False, iface=args.iface)
dex = Dex3Direct(kp=args.grip_kp, iface=args.iface, init_dds=False)
aq = lambda: np.asarray(g1.arm.get_current_dual_arm_q(), np.float64)


def get_waist():
    try:
        return float(g1.arm.get_current_waist_yaw())
    except Exception:
        return 0.0


def set_waist(w):
    try:
        g1.arm.set_waist_yaw(float(w))
    except Exception:
        pass


class KeyPoller:
    def __enter__(self):
        global _OLD
        self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); _OLD = self.old
        tty.setcbreak(self.fd); return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


def do_record():
    out_path = (args.out if os.path.isabs(args.out)
                else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out))
    g1._set_arm_kp(args.kp)                       # soft -> easy to move by hand
    frames = []
    recording = False
    print(f"[record_path] READY (kp={args.kp:.0f}, arm is soft). Move it by hand. SPACE=start/stop+save  q=quit.",
          flush=True)
    tick = 0
    with KeyPoller() as poller:
        while True:
            q = aq()
            g1.command_arm(q, np.zeros(14))       # compliant follow (target=current -> minimal resistance)
            dex.set_free("L"); dex.set_free("R"); dex.publish()   # hands back-drivable + read their state
            fL, fR = dex.read()
            if recording:
                frames.append({"q": q.tolist(), "waist": get_waist(),
                               "fingerL": fL.tolist(), "fingerR": fR.tolist()})
            tick += 1
            if tick % 25 == 0:
                print(f"\r[record_path] {'RECORDING' if recording else 'idle     '} frames={len(frames)}   ",
                      end="", flush=True)
            k = poller.poll()
            if k == " ":
                recording = not recording
                if recording:
                    frames = []
                    print("\n[record_path] RECORDING -- move the arms; SPACE again to stop+save", flush=True)
                else:
                    json.dump({"name": os.path.splitext(os.path.basename(out_path))[0], "rate": args.rate,
                               "frames": frames}, open(out_path, "w"))
                    print(f"\n[record_path] saved {len(frames)} frames ({len(frames)*dt:.1f}s) -> {out_path}",
                          flush=True)
            elif k in ("q", "\x1b"):
                break
            time.sleep(dt)                        # sample at --rate (was missing -> massive oversampling)


def do_replay():
    d = json.load(open(args.replay))
    frames = d["frames"]
    if not frames:
        print("[record_path] empty trajectory"); return
    g1._set_arm_kp(args.replay_kp)
    print(f"[record_path] REPLAY {len(frames)} frames @ speed {args.speed:.2f}. SPACE=start  q=abort.", flush=True)

    def send(fr, kp):
        g1.command_arm(np.asarray(fr["q"], float), np.zeros(14))
        dex.set_pose("L", fr["fingerL"], kp=kp); dex.set_pose("R", fr["fingerR"], kp=kp); dex.publish()
        set_waist(fr.get("waist", 0.0))

    with KeyPoller() as poller:
        while True:                                  # wait for SPACE; hold current, hands free
            g1.command_arm(aq(), np.zeros(14)); dex.set_free("L"); dex.set_free("R"); dex.publish()
            k = poller.poll()
            if k == " ":
                break
            if k in ("q", "\x1b"):
                return
            time.sleep(dt)

        # ease from current pose into the first recorded frame
        last = aq(); tgt = np.asarray(frames[0]["q"], float)
        n = max(2, int(args.approach / dt))
        print(f"[record_path] approach -> frame 0 ({args.approach:.1f}s)", flush=True)
        for i in range(1, n + 1):
            want = last + np.clip((tgt - last) * (i / n), -args.max_step, args.max_step)
            g1.command_arm(want, np.zeros(14)); dex.set_pose("L", frames[0]["fingerL"], kp=args.grip_kp)
            dex.set_pose("R", frames[0]["fingerR"], kp=args.grip_kp); dex.publish()
            last = want
            if poller.poll() in ("q", "\x1b"):
                print("\n[record_path] aborted during approach", flush=True); return
            time.sleep(dt)

        # Decimate to ~100 fps playback (auto from the file's rate), then --speed sets the wall-clock dwell.
        rate = float(d.get("rate", 100.0))
        stride = args.stride if args.stride > 0 else max(1, int(round(rate / 100.0)))
        dwell = (stride / rate) / max(args.speed, 1e-3)
        print(f"[record_path] playing {len(range(0, len(frames), stride))} frames (stride {stride}, "
              f"~{dwell*1000:.0f} ms/frame)", flush=True)
        prev = tgt
        for i in range(0, len(frames), stride):
            q = np.asarray(frames[i]["q"], float)
            q = prev + np.clip(q - prev, -args.max_step, args.max_step)
            send(dict(frames[i], q=q.tolist()), args.grip_kp)
            prev = q
            if poller.poll() in ("q", "\x1b"):
                print("\n[record_path] ABORTED -- holding", flush=True); break
            time.sleep(dwell)
        print("\n[record_path] done -- holding final pose (q/ESC to release)", flush=True)
        while True:
            g1.command_arm(prev, np.zeros(14)); dex.publish()
            if poller.poll() in ("q", "\x1b"):
                break
            time.sleep(dt)


try:
    if args.replay:
        do_replay()
    else:
        if not args.out:
            print("[record_path] --out is required when recording"); sys.exit(2)
        do_record()
finally:
    g1.stop()
    print("\n[record_path] done", flush=True)
