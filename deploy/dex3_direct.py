"""Minimal direct Dex3-1 driver -- explicit per-finger control the teleop controller doesn't expose.

The teleop Dex3_1_Controller only drives an open<->closed grasp SCALAR or retargets a glove skeleton, and
its only motor-disable is the fault-reset. For kinesthetic teaching we need three things it doesn't give:
  * DISABLE the finger motors so the 7 joints of each hand are back-drivable (pose them by hand),
  * READ all 7 joint angles per hand,
  * COMMAND explicit 7-dim targets with a chosen kp (low kp -> torque-limited close that stalls on contact).

This talks to rt/dex3/{left,right}/{cmd,state} directly. Run it with the teleop / run_heatmap Dex3 controller
STOPPED -- two publishers on the same topic fight. Construct AFTER a G1Interface (its arm controller brings
up the DDS factory); we (re)initialize defensively in case it isn't up yet.

Joint order (Dex3 hardware): [thumb0, thumb1, thumb2, middle0, middle1, index0, index1].
"""
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_

NUM = 7
TOPIC = {"L": ("rt/dex3/left/cmd", "rt/dex3/left/state"),
         "R": ("rt/dex3/right/cmd", "rt/dex3/right/state")}


def _ris_mode(motor_id, status=0x01, timeout=0):
    """RIS motor mode byte (matches teleop robot_hand_unitree._RIS_Mode). status: 0x01 enable, 0x00 disable."""
    m = 0
    m |= (motor_id & 0x0F)
    m |= (status & 0x07) << 4
    m |= (timeout & 0x01) << 7
    return m


class Dex3Direct:
    def __init__(self, kp=1.5, kd=0.2, iface="enp2s0", init_dds=False):
        # DO NOT re-init the DDS factory by default: the caller (G1Interface / the teleop) already inited it,
        # and a SECOND ChannelFactoryInitialize invalidates publishers created before it -> the arm's arm_sdk
        # publisher silently dies and the arm stops moving. Only init here for truly standalone use.
        if init_dds:
            try:
                ChannelFactoryInitialize(0, iface)
            except Exception:
                pass
        self.kp, self.kd = kp, kd
        self.pub, self.sub, self.msg = {}, {}, {}
        self.last = {"L": np.zeros(NUM), "R": np.zeros(NUM)}
        self.seen = {"L": False, "R": False}   # has this hand ever published state?
        for side, (ct, st) in TOPIC.items():
            p = ChannelPublisher(ct, HandCmd_); p.Init(); self.pub[side] = p
            s = ChannelSubscriber(st, HandState_); s.Init(); self.sub[side] = s
            self.msg[side] = unitree_hg_msg_dds__HandCmd_()
        # start disabled (back-drivable) until the caller commands a pose
        self.set_free("L"); self.set_free("R")
        # State Read() BLOCKS until a publisher exists; a silent/unpowered hand would wedge the caller. Read in
        # per-side daemon threads (one blocked side can't starve the other) and hand the main loop last-known.
        self._alive = True
        for side in ("L", "R"):
            threading.Thread(target=self._reader, args=(side,), daemon=True).start()

    def _reader(self, side):
        # Sleep FIRST every iteration so this never busy-spins on a high-rate topic -- a tight Read() loop
        # starves the GIL and stalls the co-hosted vuer websocket / teleop loop. ~200 Hz sampling is plenty.
        while self._alive:
            time.sleep(0.005)
            m = self.sub[side].Read()
            if m is not None:
                self.last[side] = np.array([m.motor_state[i].q for i in range(NUM)])
                self.seen[side] = True

    def read(self):
        """(qL[7], qR[7]) latest joint angles (zeros until the hand first publishes); never blocks."""
        return self.last["L"].copy(), self.last["R"].copy()

    def hands_online(self):
        return self.seen["L"] and self.seen["R"]

    def set_pose(self, side, q7, kp=None):
        """Enable + drive the 7 motors of one hand to q7 (kp None -> self.kp)."""
        kp = self.kp if kp is None else kp
        msg = self.msg[side]
        for i in range(NUM):
            msg.motor_cmd[i].mode = _ris_mode(i, status=0x01)
            msg.motor_cmd[i].q = float(q7[i])
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].kp = float(kp)
            msg.motor_cmd[i].kd = self.kd

    def set_free(self, side):
        """Disable the 7 motors of one hand -> back-drivable (pose by hand)."""
        msg = self.msg[side]
        for i in range(NUM):
            msg.motor_cmd[i].mode = _ris_mode(i, status=0x00)
            msg.motor_cmd[i].q = 0.0
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].kp = 0.0
            msg.motor_cmd[i].kd = 0.0

    def publish(self):
        """Send both hands' current command msg -- call every control tick to keep the stream alive."""
        for side in ("L", "R"):
            self.pub[side].Write(self.msg[side])

    def stop(self):
        self._alive = False
        self.set_free("L"); self.set_free("R"); self.publish()
