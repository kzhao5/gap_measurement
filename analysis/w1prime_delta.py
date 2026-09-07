"""(W1') within-trajectory coupling: measure Delta(phi) = log(L1/L2).

Implements 理论问题实验.md sections 2-8:
  log_k = logp_train - logp_infer   (logp_old = training-side forward)
  p     = exp(logp_train)
  phi   = max(1-p, eps0);  sigma = c*phi;  Z = (log_k + sigma^2/2)/sigma
  log_C = sigma*z_plus - sigma^2/2;  log_m = min(log_k, log_C)
  S_-t  = (traj sum of log_m) - log_m          [leave-one-out, mandatory]
  Delta(bucket) = logL1 - logL2 via offset logsumexp
Controls: T-stratification, z_plus sensitivity {0.5,1,2}x, cluster bootstrap
over trajectories, mu/var channel decomposition, omega^2, phi/Z autocorr.

Calibration: c = 0.156 (linear-LS sigma(logk|p) fit, this same dataset);
eps0 = 5e-3 (production floor); z_plus_ref = 2.3/0.156 = 14.74 (production
band lambda_+ = 2.3 = c*z_ref, so the target slope for setting B is 2.3).

Usage:
  python w1prime_delta.py --dataset full  --boot 500   # main campaign data
  python w1prime_delta.py --dataset fresh --boot 500   # 4-GPU w1p replication
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import logsumexp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import DATA_ROOT, CODE_ROOT

C1 = 0.156
EPS0 = 5e-3
LAM_PLUS_PROD = 2.3
Z_REF = LAM_PLUS_PROD / C1
MIN_BUCKET_N = 100

DATASETS = {
    # (shard list, recompute filename pattern)
    "full": (list(range(8)), "recomp_shard{s:03d}_bf16.parquet"),
    "fresh": (list(range(900, 904)), "recomp_shard{s:03d}_bf16_w1p.parquet"),
    # J1 batch: disjoint prompts (skip 250) + offset RNG streams
    "fresh2": (list(range(910, 914)), "recomp_shard{s:03d}_bf16_w1p.parquet"),
}


def load(dataset):
    shards, rec_pat = DATASETS[dataset]
    parts = []
    for s in shards:
        tokp = os.path.join(DATA_ROOT, "gen", "moe",
                            f"tokens_shard{s:03d}.parquet")
        recp = os.path.join(DATA_ROOT, "recompute", "moe", rec_pat.format(s=s))
        tok = pq.read_table(
            tokp, columns=["traj_id", "pos", "logp_infer"]).to_pandas()
        rec = pq.read_table(
            recp, columns=["traj_id", "pos", "logp_train"]).to_pandas()
        df = tok.merge(rec, on=["traj_id", "pos"], how="inner")
        if len(df) != len(tok):
            print(f"[load] shard {s}: {len(tok)} gen tokens, "
                  f"{len(df)} joined (recompute subsampled or missing rows)")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["traj_id", "pos"], ignore_index=True)
    # same-dtype comparison: both fp32 -> float64 difference
    df["log_k"] = df["logp_train"].astype(np.float64) - \
        df["logp_infer"].astype(np.float64)
    p = np.exp(df["logp_train"].astype(np.float64))
    df["phi"] = np.maximum(1.0 - p, EPS0)
    df["sigma"] = C1 * df["phi"]
    df["Z"] = (df["log_k"] + df["sigma"] ** 2 / 2) / df["sigma"]
    df["T"] = df.groupby("traj_id")["pos"].transform("size")
    print(f"[load] {dataset}: {len(df)} tokens, "
          f"{df['traj_id'].nunique()} trajs, "
          f"median T {df['T'].median():.0f}")
    return df


def with_operator(df, z_plus):
    """log_m under cap z_plus and the leave-one-out sum S_-t."""
    log_C = df["sigma"] * z_plus - df["sigma"] ** 2 / 2
    log_m = np.minimum(df["log_k"], log_C)
    tot = log_m.groupby(df["traj_id"]).transform("sum")
    return log_m.values, (tot - log_m).values


def bucket_delta(phi, S, T, bins, min_n=MIN_BUCKET_N):
    idx = np.digitize(phi, bins)
    rows = []
    for b in np.unique(idx):
        m = idx == b
        N = int(m.sum())
        if N < min_n:
            continue
        Sb = S[m]
        o = Sb.mean()
        logL1 = logsumexp(Sb - o) - np.log(N)          # = log L1 - o
        logL2 = logsumexp(2 * Sb - 2 * o) - np.log(N)  # = log L2 - 2o
        # add offsets back per the protocol: Delta = (logL1+o) - (logL2+2o).
        # (Omitting them adds +o = +mu(bucket) to Delta and contaminates the
        # phi-slope with the mu-slope -- the bug behind the first W1' report.)
        delta = (logL1 + o) - (logL2 + 2 * o)
        # ESS = (sum w)^2 / sum w^2, offset-free
        ess = float(np.exp(2 * (logL1 + np.log(N)) - (logL2 + np.log(N))))
        rows.append(dict(
            bin=int(b), phi=float(np.median(phi[m])), n=N,
            Delta=float(delta), mu=float(o),
            var=float(Sb.var(ddof=1)),
            T_med=float(np.median(T[m])), ess=ess,
        ))
    res = pd.DataFrame(rows)
    res["gauss_pred"] = -res["mu"] - 1.5 * res["var"]
    return res


def wls(x, y, w):
    """weighted least squares y = a + b*x -> (a, b)"""
    sw = np.sqrt(np.asarray(w, dtype=np.float64))
    X = np.column_stack([np.ones_like(x), x]) * sw[:, None]
    beta, *_ = np.linalg.lstsq(X, np.asarray(y) * sw, rcond=None)
    return float(beta[0]), float(beta[1])


def slopes_from_buckets(res):
    a_A, b_A = wls(np.log(res["phi"].values), res["Delta"].values,
                   res["n"].values)
    a_B, b_B = wls(res["phi"].values, res["Delta"].values, res["n"].values)
    return dict(slope_A=b_A, icept_A=a_A, slope_B=b_B, icept_B=a_B)


def traj_slices(df):
    """(traj start offsets) for fast cluster-bootstrap gathers; df is
    sorted by traj_id."""
    tid = df["traj_id"].values
    starts = np.r_[0, np.nonzero(tid[1:] != tid[:-1])[0] + 1, len(tid)]
    return starts


def bootstrap(df, S, bins, n_boot, seed, max_trajs=None):
    """Cluster bootstrap over trajectories: resample whole trajs, redo
    bucketing + both regressions. S_-t is within-traj so it is invariant
    under resampling."""
    starts = traj_slices(df)
    n_traj = len(starts) - 1
    phi = df["phi"].values
    T = df["T"].values
    rng = np.random.RandomState(seed)
    pool = np.arange(n_traj)
    if max_trajs is not None and n_traj > max_trajs:
        pool = rng.choice(n_traj, size=max_trajs, replace=False)
    sA, sB, curves = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(pool, size=len(pool), replace=True)
        idx = np.concatenate([np.arange(starts[i], starts[i + 1])
                              for i in pick])
        res = bucket_delta(phi[idx], S[idx], T[idx], bins)
        if len(res) < 3:
            continue
        sl = slopes_from_buckets(res)
        sA.append(sl["slope_A"])
        sB.append(sl["slope_B"])
        curves.append(res.set_index("bin")["Delta"])
    band = pd.concat(curves, axis=1)
    return (np.array(sA), np.array(sB),
            band.quantile(0.025, axis=1), band.quantile(0.975, axis=1))


def autocorr(df, col, max_lag=20, max_trajs=5000, seed=0):
    """Pooled within-trajectory autocorrelation (per-traj demeaned)."""
    starts = traj_slices(df)
    n_traj = len(starts) - 1
    rng = np.random.RandomState(seed)
    keep = (np.arange(n_traj) if n_traj <= max_trajs
            else np.sort(rng.choice(n_traj, size=max_trajs, replace=False)))
    x = df[col].values.astype(np.float64)
    num = np.zeros(max_lag + 1)
    dl = np.zeros(max_lag + 1)
    dr = np.zeros(max_lag + 1)
    for i in keep:
        xi = x[starts[i]:starts[i + 1]]
        xi = xi - xi.mean()
        L = len(xi)
        for lag in range(1, min(max_lag, L - 1) + 1):
            a, b = xi[:-lag], xi[lag:]
            num[lag] += a @ b
            dl[lag] += a @ a
            dr[lag] += b @ b
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / np.sqrt(dl * dr)
    return r[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--shards", default=None,
                    help="comma list overriding the shard set (smoke)")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--boot-max-trajs", type=int, default=4000,
                    help="cap bootstrap cluster pool (speed on 24.7M rows)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    outdir = args.out or os.path.join(CODE_ROOT, "results", "w1prime")
    os.makedirs(outdir, exist_ok=True)
    tag = args.dataset

    if args.shards:
        DATASETS[args.dataset] = ([int(s) for s in args.shards.split(",")],
                                  DATASETS[args.dataset][1])
        tag += "_sub"
    df = load(args.dataset)
    bins = np.geomspace(EPS0, 1.0, 21)
    report = {"dataset": tag, "n_tokens": int(len(df)),
              "n_traj": int(df["traj_id"].nunique()),
              "c1": C1, "eps0": EPS0, "z_ref": Z_REF,
              "lambda_plus_target": LAM_PLUS_PROD}

    # ---- main pass at z_ref ------------------------------------------------
    log_m, S = with_operator(df, Z_REF)
    res = bucket_delta(df["phi"].values, S, df["T"].values, bins)
    res.to_csv(os.path.join(outdir, f"buckets_{tag}.csv"), index=False)
    sl = slopes_from_buckets(res)
    report["main"] = sl
    report["gauss_check_max_reldev"] = float(
        (np.abs(res["Delta"] - res["gauss_pred"]) /
         np.maximum(np.abs(res["Delta"]), 1e-3)).max())
    report["min_ess"] = float(res["ess"].min())

    sA, sB, lo, hi = bootstrap(df, S, bins, args.boot, seed=7,
                               max_trajs=args.boot_max_trajs)
    report["boot"] = {
        "n_eff": int(len(sB)),
        "slope_A_ci": [float(np.percentile(sA, 2.5)),
                       float(np.percentile(sA, 97.5))],
        "slope_B_ci": [float(np.percentile(sB, 2.5)),
                       float(np.percentile(sB, 97.5))],
        "slope_A_se": float(sA.std(ddof=1)),
        "slope_B_se": float(sB.std(ddof=1)),
    }

    # ---- channel decomposition (mu, var vs phi) ----------------------------
    _, mu_b = wls(res["phi"].values, res["mu"].values, res["n"].values)
    _, v_b = wls(res["phi"].values, res["var"].values, res["n"].values)
    report["channels"] = {"mu_slope": mu_b, "var_slope": v_b,
                          "net_pred_slope": -mu_b - 1.5 * v_b}

    # ---- T stratification --------------------------------------------------
    tq = df.groupby("traj_id")["T"].first()
    edges = tq.quantile([0, .25, .5, .75, 1.0]).values
    strata = {}
    for k in range(4):
        ids = tq[(tq >= edges[k]) & (tq <= edges[k + 1])].index
        m = df["traj_id"].isin(ids).values
        r = bucket_delta(df["phi"].values[m], S[m], df["T"].values[m], bins)
        if len(r) >= 3:
            strata[f"T[{edges[k]:.0f},{edges[k+1]:.0f}]"] = r
    report["strata_slopes"] = {k: slopes_from_buckets(r)["slope_B"]
                               for k, r in strata.items()}

    # ---- z_plus sensitivity ------------------------------------------------
    sens = {}
    for mult in (0.5, 1.0, 2.0):
        z = Z_REF * mult
        _, Sz = with_operator(df, z)
        rz = bucket_delta(df["phi"].values, Sz, df["T"].values, bins)
        sens[mult] = rz
        report.setdefault("sensitivity", {})[str(mult)] = \
            slopes_from_buckets(rz)

    # ---- omega^2 (trajectory-level hidden scale) ---------------------------
    per = pd.DataFrame({"traj": df["traj_id"].values, "log_m": log_m,
                        "q": df["sigma"].values ** 2})
    pt = per.groupby("traj").agg(L=("log_m", "sum"), Q=("q", "sum"))
    om = float(pt["L"].var(ddof=1) - pt["Q"].mean())
    rng = np.random.RandomState(11)
    oms = [pt.iloc[rng.choice(len(pt), len(pt))].agg(
        {"L": "var", "Q": "mean"}).pipe(lambda s: s["L"] - s["Q"])
        for _ in range(300)]
    report["omega2"] = {"hat": om,
                        "ci": [float(np.percentile(oms, 2.5)),
                               float(np.percentile(oms, 97.5))],
                        "var_L": float(pt["L"].var(ddof=1)),
                        "mean_Q": float(pt["Q"].mean())}

    # ---- autocorrelation ---------------------------------------------------
    ac_phi = autocorr(df, "phi")
    ac_Z = autocorr(df, "Z")
    report["autocorr"] = {"phi_lag1": float(ac_phi[0]),
                          "Z_lag1": float(ac_Z[0]),
                          "phi_lag5": float(ac_phi[4]),
                          "Z_lag5": float(ac_Z[4])}

    # ---- verdict (section 7) ----------------------------------------------
    b = sl["slope_B"]
    ci = report["boot"]["slope_B_ci"]
    signif = not (ci[0] <= 0 <= ci[1])
    if abs(b) < 0.1 or not signif:
        verdict = ("W1-OK: b insignificant or |b|<0.1 -> (W1) holds to first "
                   "order; t-channel killed by FOC; constant-cap Delta")
    elif b < 0:
        verdict = ("VARIANCE-CHANNEL: b<0 significant -> MSE wants TIGHTER "
                   "band at low confidence; constant cap is loose at the "
                   "sequence level (report as finding)")
    else:
        verdict = ("MEAN-CHANNEL: b>0 -> phi_t band derivable from MSE; "
                   "rewrite Step 4-5 with non-independent FOC"
                   if abs(b) > 0.5 * LAM_PLUS_PROD else
                   "MEAN-CHANNEL (weak): b>0 significant but << lambda_+")
    report["verdict"] = verdict

    with open(os.path.join(outdir, f"report_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- figures -----------------------------------------------------------
    figdir = os.path.join(CODE_ROOT, "results", "figs")
    os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    r0 = res.set_index("bin")
    common = r0.index.intersection(lo.index)
    ax[0].fill_between(r0.loc[common, "phi"], lo[common], hi[common],
                       alpha=.25, label="95% traj bootstrap")
    ax[0].plot(res["phi"], res["Delta"], "o-", ms=4, label=r"$\Delta(\varphi)$")
    xs = np.geomspace(EPS0, 1, 100)
    ax[0].plot(xs, sl["icept_B"] + LAM_PLUS_PROD * xs, "--", c="gray",
               label=r"target slope $\lambda_+=2.3$")
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\varphi$")
    ax[0].set_ylabel(r"$\Delta$")
    ax[0].set_title(f"{tag}: b={b:.3f} "
                    f"[{ci[0]:.3f},{ci[1]:.3f}]  (A: {sl['slope_A']:.4f})")
    ax[0].legend(fontsize=8)
    for name, r in strata.items():
        ax[1].plot(r["phi"], r["Delta"], "o-", ms=3, label=name)
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\varphi$")
    ax[1].set_title("T-stratified")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"W1P_delta_{tag}.png"), dpi=150)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(res["phi"], -res["mu"], "o-", ms=4, label=r"$-\mu(\varphi)$")
    ax[0].plot(res["phi"], -1.5 * res["var"], "s-", ms=4,
               label=r"$-1.5\,v(\varphi)$")
    ax[0].plot(res["phi"], res["Delta"], "k.-", label=r"$\Delta$ (direct)")
    ax[0].plot(res["phi"], res["gauss_pred"], "x--", c="gray",
               label=r"$-\mu-1.5v$")
    ax[0].set_xscale("log"); ax[0].legend(fontsize=8)
    ax[0].set_title("channel decomposition + gauss check")
    for mult, rz in sens.items():
        ax[1].plot(rz["phi"], rz["Delta"], "o-", ms=3,
                   label=f"z+ = {mult}x ref")
    ax[1].set_xscale("log"); ax[1].legend(fontsize=8)
    ax[1].set_title(r"$z_+$ sensitivity")
    lags = np.arange(1, 21)
    ax[2].plot(lags, ac_phi, "o-", ms=3, label=r"$\varphi$ channel")
    ax[2].plot(lags, ac_Z, "s-", ms=3, label=r"$Z$ channel")
    ax[2].axhline(0, c="gray", lw=.5)
    ax[2].set_xlabel("lag"); ax[2].set_title("within-traj autocorrelation")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"W1P_diag_{tag}.png"), dpi=150)

    print(json.dumps(report, indent=2))
    print(f"[w1prime] verdict: {verdict}")


if __name__ == "__main__":
    main()
