#!/usr/bin/env python
"""Empirical bias-variance frontier for M* candidates (boxed objective):

    F_n(M) = (E|W_m - W|)^2 + (1/n) E[W_m^2],   W = prod_t k_t per trajectory

evaluated on the measured trajectories. Four families = {const, scaled}
threshold x {clip, zero}:
    TIS    : clip, const     log M = min(D, +c)            (one-sided clip)
    IcePop : zero, const     W_m = 0 if any |D| > c
    KPop   : zero, scaled    W_m = 0 if any |D| > c_z * sig(p_t)
    OURS   : clip, scaled    log M = clip(D, -c_z sig, +c_z sig)
sig(p_t) = c1 (1 - p_t) floored at 1e-6 (bf16 floor), c1 from binned LS.

Usage: python analysis/mstar_frontier.py {moe|dense}
"""
import sys

import numpy as np
import pandas as pd

LOCAL = "/home/kzhao2/gap_measurement/results"
N_EFF = 1024  # batch size in the variance term; ordering shifts only via 1/n

arch = sys.argv[1]
t = pd.read_parquet(f"{LOCAL}/tokens_{arch}.parquet",
                    columns=["traj_id", "logp_train", "D"])
o = np.argsort(t["traj_id"].to_numpy(), kind="stable")
D = t["D"].to_numpy(np.float64)[o]
lp = t["logp_train"].to_numpy(np.float64)[o]
tr = t["traj_id"].to_numpy()[o]
b = np.flatnonzero(np.r_[True, tr[1:] != tr[:-1]])
ends = np.r_[b[1:], len(tr)] - 1  # not needed for reduceat sums
ntraj = len(b)

# scale law fit sigma(p) = c1 (1-p), floored
p = np.exp(lp)
s_ax = np.clip(-lp, 1e-7, None)
edges = np.unique(np.quantile(s_ax, np.linspace(0, 1, 37)))
idx = np.clip(np.searchsorted(edges, s_ax, side="right") - 1, 0, len(edges) - 2)
logs, logg = [], []
for k in range(len(edges) - 1):
    m = idx == k
    med = np.median(D[m])
    sg = 1.4826 * np.median(np.abs(D[m] - med))
    g = 1 - np.exp(-np.median(s_ax[m]))
    if sg > 0 and g > 0:
        logs.append(np.log(sg)), logg.append(np.log(g))
c1 = float(np.exp(np.mean(np.array(logs) - np.array(logg))))
sig = np.maximum(c1 * (1 - p), 1e-6)
z = D / sig

S = np.add.reduceat(D, b)                       # log W per traj
W = np.exp(S)
print(f"[{arch}] ntraj={ntraj}  c1={c1:.4g}")
print(f"  log W: mean={S.mean():.3f} sd={S.std():.3f} "
      f"q[1,50,99]={np.quantile(S,[.01,.5,.99]).round(2)}")
print(f"  E[W]={W.mean():.4f} (identity says 1; tail under-sampled)  "
      f"ESS/n={(W.sum()**2/(W**2).sum())/ntraj:.4f}")
base_var = (W ** 2).mean()
print(f"  no-mask: E|W_m-W|=0, E[W^2]={base_var:.3f}, "
      f"F_n={base_var/N_EFF:.6f}")


def report(name, logWm, zero_mask_frac=None):
    Wm = np.exp(logWm)
    bias = np.abs(Wm - W).mean()
    var = (Wm ** 2).mean()
    F = bias ** 2 + var / N_EFF
    extra = f" maskedTraj={zero_mask_frac:.3f}" if zero_mask_frac is not None else ""
    print(f"    {name:28s} E|Wm-W|={bias:.4f}  E[Wm^2]={var:.4f}  "
          f"F_1024={F:.6f}{extra}")
    return F


print("  -- TIS (clip, const c on D) --")
for c in [0.05, 0.1, 0.2, 0.5, 1.0]:
    report(f"c={c}", np.add.reduceat(np.minimum(D, c), b))

print("  -- IcePop (zero, const |D|<=c) --")
for c in [0.05, 0.1, 0.2, 0.5, 1.0]:
    bad = np.add.reduceat((np.abs(D) > c).astype(np.float64), b) > 0
    logWm = np.where(bad, -np.inf, S)
    report(f"c={c}", logWm, bad.mean())

print("  -- KPop (zero, scaled |z|<=cz) --")
for cz in [3, 5, 8, 12, 20]:
    bad = np.add.reduceat((np.abs(z) > cz).astype(np.float64), b) > 0
    logWm = np.where(bad, -np.inf, S)
    report(f"cz={cz}", logWm, bad.mean())

print("  -- OURS (clip, scaled z in [-cz,cz]) --")
for cz in [3, 5, 8, 12, 20]:
    report(f"cz={cz}", np.add.reduceat(np.clip(D, -cz * sig, cz * sig), b))

print("  -- OURS-asym (clip, z in [-az,bz]) --")
for az, bz in [(5, 3), (8, 5), (12, 8), (20, 12)]:
    report(f"az={az},bz={bz}",
           np.add.reduceat(np.clip(D, -az * sig, bz * sig), b))
