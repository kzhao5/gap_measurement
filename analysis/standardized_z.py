#!/usr/bin/env python
"""STANDARDIZED_LOGKT_EXPERIMENT.md -- executed verbatim.

Z_t = logK / (c_hat*(1-p_t)), c_hat fit on 70% trajs (20 p-quantile bins,
weighted linear LS), diagnostics on 30% test trajs.
Usage: python analysis/standardized_z.py {moe|dense}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCH = sys.argv[1] if len(sys.argv) > 1 else "moe"
LOCAL = "/home/kzhao2/gap_measurement/results"
FIGS = os.path.join(LOCAL, "figs")
BLUE, INK, GRAY, RED = "#2a78d6", "#0b0b0b", "#8a8a8a", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d9d9d9", "axes.grid": True,
    "grid.color": "#ececec", "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11.5, "figure.dpi": 150,
})

t = pd.read_parquet(f"{LOCAL}/tokens_{ARCH}.parquet",
                    columns=["traj_id", "pos", "D", "logp_train"])
logK = t["D"].to_numpy(np.float64)
p = np.exp(t["logp_train"].to_numpy(np.float64))
traj = t["traj_id"].to_numpy()

# -- 2. split by trajectory 70/30 --------------------------------------
uniq = np.unique(traj)
rng = np.random.default_rng(20260810)
rng.shuffle(uniq)
tr_ids = uniq[: int(0.7 * len(uniq))]
is_tr = np.isin(traj, tr_ids)
print(f"[{ARCH}] tokens={len(logK):,} train={is_tr.sum():,} test={(~is_tr).sum():,}")

# -- 3. estimate c on train (20 quantile bins, weighted linear LS) -----
ptr, ktr = p[is_tr], logK[is_tr]
edges = np.unique(np.quantile(ptr, np.linspace(0, 1, 21)))
bi = np.clip(np.searchsorted(edges, ptr, side="right") - 1, 0, len(edges) - 2)
pb, sb, nb = [], [], []
for b in range(len(edges) - 1):
    m = bi == b
    if m.sum() < 100:
        continue
    x = ktr[m]
    pb.append(x.size and float(ptr[m].mean()))
    sb.append(1.4826 * float(np.median(np.abs(x - np.median(x)))))
    nb.append(int(m.sum()))
pb, sb, nb = map(np.asarray, (pb, sb, nb))
g = 1.0 - pb
c_hat = float((nb * sb * g).sum() / (nb * g * g).sum())
sw = (nb * sb).sum() / nb.sum()
r2 = 1.0 - float((nb * (sb - c_hat * g) ** 2).sum()) / float(
    (nb * (sb - sw) ** 2).sum())
print(f"c_hat={c_hat:.4f}  R^2={r2:.4f}")

# -- 4. Z on test ------------------------------------------------------
pte, kte = p[~is_tr], logK[~is_tr]
Z = kte / (c_hat * np.maximum(1.0 - pte, 1e-6))

# -- 5A. ECDF collapse (10 p bins) -------------------------------------
e10 = np.unique(np.quantile(pte, np.linspace(0, 1, 11)))
b10 = np.clip(np.searchsorted(e10, pte, side="right") - 1, 0, len(e10) - 2)
n_bins = len(e10) - 1
ramp = plt.cm.Blues(np.linspace(0.35, 0.95, n_bins))
fig, ax = plt.subplots(figsize=(9, 6.4))
zg = np.linspace(-8, 8, 801)
rows_csv = []
QN = [0.001, 0.01, 0.10, 0.50, 0.90, 0.99, 0.999]
qmat = np.empty((n_bins, len(QN)))
for b in range(n_bins):
    m = b10 == b
    zb = np.sort(Z[m])
    ec = np.searchsorted(zb, zg, side="right") / max(len(zb), 1)
    ax.plot(zg, ec, color=ramp[b], lw=1.5,
            label=f"p̄={pte[m].mean():.4g} (n={m.sum():,})")
    qmat[b] = np.quantile(Z[m], QN)
    rows_csv.append([b, int(m.sum()), float(pte[m].mean()),
                     1.4826 * float(np.median(np.abs(kte[m] - np.median(kte[m]))))]
                    + [float(q) for q in qmat[b]])
ax.set_xlabel("z"); ax.set_ylabel("P(Z ≤ z | p bin)")
ax.set_title(f"ECDF of Z by p-quantile bin ({ARCH}, test set) — "
             "collapse = shared F_Z")
ax.legend(fontsize=7.5, ncol=2, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIGS}/z_ecdf_by_p_{ARCH}.png"); plt.close(fig)

# -- 5B. standardized quantiles vs p ------------------------------------
fig, ax = plt.subplots(figsize=(9, 6.4))
qcol = plt.cm.coolwarm(np.linspace(0.08, 0.92, len(QN)))
x = 1.0 - np.array([r[2] for r in rows_csv])
for j, qn in enumerate(QN):
    ax.plot(x, qmat[:, j], "-o", ms=4.5, lw=1.6, color=qcol[j],
            label=f"q{qn:g}")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=5)
ax.invert_xaxis()
ax.set_xlabel("1 − p̄  (log; left = uncertain, right = confident)  "
              "[dev: spec says x=p̄, log(1−p̄) used for readability]")
ax.set_ylabel("Z quantile (symlog)")
ax.set_title(f"Standardized quantiles vs p bin ({ARCH}) — flat = pass")
ax.legend(fontsize=8.5, ncol=2)
fig.tight_layout(); fig.savefig(f"{FIGS}/z_quantiles_by_p_{ARCH}.png"); plt.close(fig)

# -- 5C. tail survival --------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6.2))
zt = np.linspace(2, 60, 400)
Zs = np.sort(Z)
sf_pos = 1.0 - np.searchsorted(Zs, zt, side="right") / len(Zs)
sf_neg = np.searchsorted(Zs, -zt, side="left") / len(Zs)
ax.semilogy(zt, np.maximum(sf_pos, 1e-9), color=BLUE, lw=2,
            label="P(Z > z)  (right tail)")
ax.semilogy(zt, np.maximum(sf_neg, 1e-9), color=RED, lw=2,
            label="P(Z < −z)  (left tail)")
ax.set_xlabel("z"); ax.set_ylabel("survival probability (log)")
ax.set_title(f"Tail shape of Z ({ARCH}, all test tokens)")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIGS}/z_tail_survival_{ARCH}.png"); plt.close(fig)

# -- 6. CSV -------------------------------------------------------------
cols = ["p_bin", "n", "p_mean", "sigma_logK", "z_q001", "z_q01", "z_q10",
        "z_q50", "z_q90", "z_q99", "z_q999"]
pd.DataFrame(rows_csv, columns=cols).to_csv(
    f"{LOCAL}/standardized_logkt_summary_{ARCH}.csv", index=False)

# quick machine verdict aids
core = slice(0, n_bins - 1)
print("q10 range over bins:", qmat[:, 2].min(), qmat[:, 2].max())
print("q90 range over bins:", qmat[:, 4].min(), qmat[:, 4].max())
print("q999 by bin:", np.round(qmat[:, 6], 1).tolist())
print(f"wrote CSV + 3 figs for {ARCH}")
