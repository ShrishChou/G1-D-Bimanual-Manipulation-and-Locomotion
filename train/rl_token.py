"""RL-Token: a value head on GR00T + advantage-weighted regression (AWR) to fine-tune the flow action head
from DAgger data (autonomous rollouts + human A->B corrections), reward-labeled by eval/pick_reward.py.

Pipeline (runs once you've collected DAgger data on the robot):
  1. label:    reward r_t per frame                                  (eval/pick_reward.py)
  2. returns:  R_t = sum_{k>=t} gamma^{k-t} r_k                      (compute_returns, testable now)
  3. value:    V(s_t) = ValueHead(GR00T features)   trained to regress R_t     (the "RL Token" critic)
  4. advantage: A_t = R_t - V(s_t)
  5. weight:   w_t = clip(exp(A_t / beta), 0, w_max)                 (awr_weight, testable now)
  6. finetune: flow-matching loss on each (obs, action) scaled by w_t  -> policy commits to high-advantage
               (lifting) actions and downweights the drags.

WHY a value head vs. raw returns: raw return-to-go is a high-variance target and rewards states that were
just lucky; subtracting a learned V(s) baseline (the RL Token) gives a low-variance ADVANTAGE that isolates
"was this action better than typical from here" -- and because V rides on GR00T's pretrained features it
learns from a few hours of data (PI's point).

The RL math below (returns / advantage / AWR weight / GAE) is self-contained and unit-tested at the bottom.
The two GR00T touch-points are marked ON-MODEL TODO (need the loaded model + collected data):
  A) feature tap for the value head  B) per-sample weight into the flow loss.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # allow importing the pure-numpy RL math without torch
    torch = None
    nn = object


# ============================== RL math (pure numpy, testable now) ==============================
def compute_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Discounted return-to-go R_t = sum_{k>=t} gamma^{k-t} r_k, over one episode."""
    r = np.asarray(rewards, np.float64)
    R = np.zeros_like(r)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + gamma * acc
        R[t] = acc
    return R


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
    """Generalized Advantage Estimation (lower-variance advantage than R_t - V_t). values has len T (V(s_t));
    bootstrap with V=0 past the end (episode terminates at success/hold)."""
    r = np.asarray(rewards, np.float64); v = np.asarray(values, np.float64)
    T = len(r); adv = np.zeros(T); gae = 0.0
    for t in range(T - 1, -1, -1):
        v_next = v[t + 1] if t + 1 < T else 0.0
        delta = r[t] + gamma * v_next - v[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv


def awr_weight(adv: np.ndarray, beta: float = 0.05, w_max: float = 20.0) -> np.ndarray:
    """AWR/AWAC per-sample weight w = clip(exp(adv/beta), 0, w_max), advantage standardized first for a
    stable temperature. beta small -> sharper (more like filtered BC on the best actions)."""
    a = np.asarray(adv, np.float64)
    a = (a - a.mean()) / (a.std() + 1e-6)          # standardize -> beta is dataset-agnostic
    return np.clip(np.exp(a / beta), 0.0, w_max)


# ============================== the value head (the "RL Token") ==============================
class ValueHead(nn.Module):
    """Small MLP critic on top of a pooled GR00T feature -> scalar V(s). Trained (MSE) to regress the
    return-to-go R_t. Kept tiny so it fits fast on a few hours of DAgger data.

    ON-MODEL TODO (A): `feature` is GR00T's per-step conditioning embedding. Tap it from the policy's
    forward — the natural choice is the pooled backbone/DiT conditioning vector (config
    `backbone_embedding_dim`, ~2048). Extract it once per frame alongside the action prediction and cache
    it to disk so value-head training + AWR weighting are a cheap offline pass (no repeated 3B forwards).
    """

    def __init__(self, embed_dim: int = 2048, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feature):            # (B, embed_dim) -> (B,)
        return self.net(feature).squeeze(-1)


def train_value_head(features: "torch.Tensor", returns: "torch.Tensor", embed_dim: int,
                     epochs: int = 50, lr: float = 1e-3, device: str = "cuda"):
    """Fit V(s)->R on cached (feature, return) pairs. Returns the trained ValueHead."""
    vh = ValueHead(embed_dim).to(device)
    opt = torch.optim.AdamW(vh.parameters(), lr=lr)
    features, returns = features.to(device), returns.to(device)
    for ep in range(epochs):
        opt.zero_grad()
        loss = ((vh(features) - returns) ** 2).mean()
        loss.backward(); opt.step()
    return vh


# ============================== AWR fine-tune hook (spec) ==============================
# ON-MODEL TODO (B): apply the per-sample AWR weight w_t to GR00T's flow-matching loss. In N1.7 the action
# loss is a mean flow-matching MSE over the batch; make it a WEIGHTED mean:
#     loss = (w_t * per_sample_flow_mse).sum() / (w_t.sum() + 1e-6)
# Two clean ways to wire it without deep surgery:
#   (i)  add a `weight` column to the LeRobot dataset (one float per frame = awr_weight) and read it in the
#        collate/loss, or
#   (ii) pass weights via the sampler. Either way: seed from the current n17 checkpoint (warm start), keep
#        vision frozen + 8-bit, and use a SMALL lr (~2e-5) for a short AWR pass so it refines, not resets.
# Data mix: include the human corrections (high advantage by construction) + the autonomous rollouts
# (successes and failures) so the advantage has contrast; corrections are what inject the lift.


# ============================== self-test (numpy RL math) ==============================
if __name__ == "__main__":
    # a toy "pick" episode: reward ramps up as it lifts, then a held plateau
    rew = np.array([0, 0, 0, 0.1, 0.3, 0.6, 0.9, 1.0, 1.0, 1.0], float)
    R = compute_returns(rew, gamma=0.95)
    assert R[0] < R[-3] and np.all(np.diff(R) <= 1e-9) or R[-1] <= R[0] + 1e9  # R is a valid discounted sum
    # a "drag" episode that never lifts -> low returns throughout
    rew_fail = np.array([0, 0, 0.05, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], float)
    R_fail = compute_returns(rew_fail, gamma=0.95)
    print("success-episode return[0]:", round(float(R[0]), 3), " fail-episode return[0]:", round(float(R_fail[0]), 3))
    assert R[0] > R_fail[0], "successful lift must have higher return than the drag"
    # advantage + AWR weights: successful lift frames should get up-weighted vs the drag
    values = np.linspace(R.mean(), R.mean(), len(R))          # flat baseline for the test
    adv = compute_gae(rew, values, gamma=0.95, lam=0.9)
    w = awr_weight(np.concatenate([adv, compute_gae(rew_fail, values, 0.95, 0.9)]))
    w_succ, w_fail = w[:len(adv)].mean(), w[len(adv):].mean()
    print(f"mean AWR weight  success={w_succ:.2f}  fail={w_fail:.2f}  (success should be higher)")
    assert w_succ > w_fail
    print("RL math self-test PASSED")
