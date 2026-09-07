"""Cross-confidence-layer gradient coherence matrix (gradient_coherence_guide).

Coh(p,p') = <u_bar(p), u_bar(p')>, u_bar(p) = normalized E[A*(e_y-pi)^T R | p]
(L1), or the flattened outer product with h^T R_h (L2, 128x32 = 4096-dim).

Noise floor: per-bucket half-split self-consistency (attenuation correction
Coh/sqrt(self*self')) + permutation null (shuffled pbin labels).
Primary readout: mass-weighted median of |Coh_corr| over far pairs
(|b-b'| >= 4). Pre-registered verdict: <=0.2 diversity (D) -> MSE mainline
revives (decreasing band); >=0.5 coherent -> increasing cap, negative
closure; between -> report matrix, RL adjudicates.

Only tokens with A != 0 carry update direction; zeros are excluded
(they scale means but cannot rotate them).

Usage: python grad_coherence.py --shards 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import CODE_ROOT, DATA_ROOT
from token_fixedpoint import compute_A

NB = 16
FAR = 4
N_PERM_L1 = 200
N_PERM_L2 = 50
N_HALF = 10
N_BOOT = 200
OUT = os.path.join(CODE_ROOT, "results", "coherence")
FIG = os.path.join(CODE_ROOT, "results", "figs", "coherence")


def bucket_means(feat, pbin, nb=NB):
    """[N,d] -> [nb,d] bucket means (excluding nothing; caller filters)."""
    d = feat.shape[1]
    U = np.zeros((nb, d))
    n = np.bincount(pbin, minlength=nb)
    for j in range(d):
        U[:, j] = np.bincount(pbin, weights=feat[:, j], minlength=nb)
    return U / np.maximum(n, 1)[:, None], n


def outer_bucket_means(a, b, pbin, nb=NB):
    """bucket means of flatten(outer(a_t, b_t)) without materializing:
    [N,da],[N,db] -> [nb, da*db]."""
    da, db = a.shape[1], b.shape[1]
    U = np.zeros((nb, da * db))
    n = np.bincount(pbin, minlength=nb)
    for k in range(nb):
        m = pbin == k
        if m.any():
            U[k] = (a[m].T @ b[m]).ravel() / m.sum()
    return U, n


def coh_matrix(U):
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    Un = U / np.maximum(norms, 1e-300)
    return Un @ Un.T


def far_readout(C, n, corr=None):
    """mass-weighted median of |C| over pairs |b-b'| >= FAR."""
    vals, wts = [], []
    for i in range(NB):
        for j in range(NB):
            if abs(i - j) >= FAR:
                v = C[i, j]
                if corr is not None:
                    s = corr[i] * corr[j]
                    if s < 0.05:
                        continue
                    v = v / np.sqrt(s)
                vals.append(abs(v))
                wts.append(n[i] * n[j])
    vals, wts = np.array(vals), np.array(wts, float)
    o = np.argsort(vals)
    cw = np.cumsum(wts[o])
    return float(vals[o][np.searchsorted(cw, cw[-1] / 2)])


def half_self(feat, pbin, rng, outer_b=None):
    """average over N_HALF random half-splits of per-bucket cos(U1,U2)."""
    acc = np.zeros(NB)
    for _ in range(N_HALF):
        h = rng.random(len(pbin)) < 0.5
        if outer_b is None:
            U1, _ = bucket_means(feat[h], pbin[h])
            U2, _ = bucket_means(feat[~h], pbin[~h])
        else:
            U1, _ = outer_bucket_means(feat[h], outer_b[h], pbin[h])
            U2, _ = outer_bucket_means(feat[~h], outer_b[~h], pbin[~h])
        n1 = np.linalg.norm(U1, axis=1) * np.linalg.norm(U2, axis=1)
        acc += (U1 * U2).sum(1) / np.maximum(n1, 1e-300)
    return acc / N_HALF


def analyze_level(name, feat, pbin, n_tok, rng, outer_b=None, n_perm=100):
    if outer_b is None:
        U, n = bucket_means(feat, pbin)
    else:
        U, n = outer_bucket_means(feat, outer_b, pbin)
    C = coh_matrix(U)
    selfd = half_self(feat, pbin, rng, outer_b)
    raw = far_readout(C, n)
    corrected = far_readout(C, n, corr=selfd)
    # permutation null on the raw far readout
    null = []
    pb = pbin.copy()
    for _ in range(n_perm):
        rng.shuffle(pb)
        if outer_b is None:
            Up, np_ = bucket_means(feat, pb)
        else:
            Up, np_ = outer_bucket_means(feat, outer_b, pb)
        null.append(far_readout(coh_matrix(Up), np_))
    null = np.array(null)
    decay = [float(np.median([abs(C[i, j]) for i in range(NB)
                              for j in range(NB) if abs(i - j) == lag]))
             for lag in range(1, NB)]
    return dict(name=name, far_raw=raw, far_corrected=corrected,
                self_diag=[float(x) for x in selfd],
                self_min=float(selfd.min()),
                null_median=float(np.median(null)),
                null_q95=float(np.percentile(null, 95)),
                above_null=bool(raw > np.percentile(null, 95)),
                decay=decay), C, n, selfd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    shards = [int(s) for s in args.shards.split(",")]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    parts = []
    for s in shards:
        h = pq.read_table(
            os.path.join(DATA_ROOT, "harvest", "moe",
                         f"harv_shard{s:03d}.parquet")).to_pandas()
        rec = pq.read_table(
            os.path.join(DATA_ROOT, "recompute", "moe",
                         f"recomp_shard{s:03d}_bf16.parquet"),
            columns=["traj_id", "pos", "logp_train"]).to_pandas()
        parts.append(h.merge(rec, on=["traj_id", "pos"]))
    import pandas as pd
    df = pd.concat(parts, ignore_index=True)
    Adf, acc = compute_A(shards)
    df = df.merge(Adf, on="traj_id", how="left")
    df["prompt_id"] = df["traj_id"] // 1000
    A = df["A"].values
    live = np.abs(A) > 1e-9
    df = df[live].reset_index(drop=True)
    A = A[live]
    p = np.exp(df["logp_train"].astype(np.float64).values)
    gp = np.stack(df["g_proj"].values).astype(np.float32)
    hp = np.stack(df["h_proj"].values).astype(np.float32)
    print(f"[coh] {len(df)} live tokens (A!=0), reward acc {acc:.3f}")

    PB = np.quantile(p, np.linspace(0, 1, NB + 1))
    pbin = np.clip(np.digitize(p, PB[1:-1]), 0, NB - 1)
    u1 = A[:, None].astype(np.float32) * gp
    rng = np.random.default_rng(0)

    report = {"n_live_tokens": int(len(df)), "reward_acc": acc,
              "p_bin_edges": [float(x) for x in PB]}

    r1, C1, n1, s1 = analyze_level("L1", u1, pbin, len(df), rng,
                                   n_perm=N_PERM_L1)
    r2, C2, n2, s2 = analyze_level("L2", u1, pbin, len(df), rng,
                                   outer_b=hp, n_perm=N_PERM_L2)
    report["L1"], report["L2"] = r1, r2

    # sensitivities on L1: |A| weighting and positive-A-only
    uabs = np.abs(A)[:, None].astype(np.float32) * gp
    ra, _, _, _ = analyze_level("L1_absA", uabs, pbin, len(df), rng, n_perm=30)
    pos = A > 0
    rp, _, _, _ = analyze_level("L1_posA", u1[pos], pbin[pos], int(pos.sum()),
                                rng, n_perm=30)
    report["sens_absA_far_corr"] = ra["far_corrected"]
    report["sens_posA_far_corr"] = rp["far_corrected"]

    # prompt-cluster bootstrap of the L1 corrected far readout
    prompts = df["prompt_id"].values
    upr = np.unique(prompts)
    sums = np.zeros((len(upr), NB, u1.shape[1]), dtype=np.float64)
    cnts = np.zeros((len(upr), NB))
    pidx = {q: i for i, q in enumerate(upr)}
    pi = np.array([pidx[q] for q in prompts])
    for j in range(u1.shape[1]):
        np.add.at(sums[:, :, j], (pi, pbin), u1[:, j])
    np.add.at(cnts, (pi, pbin), 1)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(upr), len(upr))
        U = sums[pick].sum(0) / np.maximum(cnts[pick].sum(0), 1)[:, None]
        boots.append(far_readout(coh_matrix(U), cnts[pick].sum(0)))
    report["L1_boot_far_raw_ci"] = [float(np.percentile(boots, 2.5)),
                                    float(np.percentile(boots, 97.5))]

    # rho(p): within-bucket direction coherence (L1)
    rho = []
    for b in range(NB):
        m = pbin == b
        ub = u1[m].mean(0)
        ub /= max(np.linalg.norm(ub), 1e-300)
        rho.append(float((u1[m] @ ub).mean()
                         / max(np.linalg.norm(u1[m], axis=1).mean(), 1e-300)))
    report["rho_p"] = rho

    # verdict (pre-registered, L2 is the arbiter; L1 must agree on branch)
    key = r2["far_corrected"] if r2["self_min"] > 0.3 else r1["far_corrected"]
    src = "L2" if r2["self_min"] > 0.3 else "L1(L2 underpowered)"
    if key <= 0.2:
        verdict = (f"DIVERSITY (D) holds ({src} far|Coh|={key:.3f} <= 0.2): "
                   "token-MSE derives the DECREASING band "
                   "lambda_+ phi^(1/(1-xi+)); sigma-TIS is its light-tail "
                   "limit -> MSE mainline REVIVES")
    elif key >= 0.5:
        verdict = (f"COHERENT ({src} far|Coh|={key:.3f} >= 0.5): token-MSE "
                   "gives increasing cap, opposite to RL -> negative closure "
                   "confirmed")
    else:
        verdict = (f"INTERMEDIATE ({src} far|Coh|={key:.3f}): report matrix, "
                   "both theory lines, RL adjudicates")
    report["verdict"] = verdict

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for a, (C, s, nm, r) in zip(
            ax[:2], [(C1, s1, "L1", r1), (C2, s2, "L2", r2)]):
        im = a.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        plt.colorbar(im, ax=a, fraction=0.046)
        a.set_title(f"{nm}: far|Coh| raw {r['far_raw']:.3f} / corr "
                    f"{r['far_corrected']:.3f}\nnull95 {r['null_q95']:.3f}, "
                    f"self_min {r['self_min']:.2f}")
        a.set_xlabel("p bucket (low->high conf)")
    ax[2].plot(range(1, NB), r1["decay"], "o-", ms=4, label="L1")
    ax[2].plot(range(1, NB), r2["decay"], "s-", ms=4, label="L2")
    ax[2].axhline(r1["null_q95"], ls="--", c="gray", label="L1 null q95")
    ax[2].set_xlabel("|b-b'|"); ax[2].set_ylabel("median |Coh|")
    ax[2].legend(fontsize=8); ax[2].set_title("distance decay")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "coherence_matrix.png"), dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(NB), s1, "o-", label="L1")
    ax[0].plot(range(NB), s2, "s-", label="L2")
    ax[0].axhline(0.5, ls="--", c="gray")
    ax[0].set_title("half-split self-consistency"); ax[0].legend(fontsize=8)
    ax[1].plot(range(NB), rho, "o-")
    ax[1].set_title("rho(p): within-bucket direction coherence")
    ax[1].set_xlabel("p bucket")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "coherence_diag.png"), dpi=150)

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("L1", "L2")}, indent=2))
    for nm in ("L1", "L2"):
        r = report[nm]
        print(nm, {k: (round(v, 3) if isinstance(v, float) else v)
                   for k, v in r.items() if k not in ("self_diag", "decay")})
    print(f"[coherence] verdict: {verdict}")


if __name__ == "__main__":
    main()
