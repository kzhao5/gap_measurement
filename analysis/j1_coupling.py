"""(J1) trajectory coupling law: beta1, beta2, lambda_eff.

Implements 理论问题实验.md (J1 revision) sections 3-10:
  E[S_-t | phi]   = alpha1 - beta1*sigma   (mean channel, widens band)
  Var[S_-t | phi] = alpha2 + beta2*sigma   (variance channel, narrows band)
  lambda_eff = c*(beta1 - 1.5*beta2)       -> compare against ablation lambda_+

Gate (section 6, FIRST): cumulant truncation acceptance per bucket,
|Delta_cum - Delta_direct| < 10%|Delta_direct| and ESS > 50. If it fails for
most buckets, the closed form is abandoned and lambda_eff is replaced by the
semi-parametric slope of Delta_direct vs phi (which is exactly the W1' b).

ESS note: the doc's literal formula exp(2logL1 - logL2 - logN) is ESS/N^2
(cannot exceed 1/N, so the ">50" gate would be unusable); the intended
quantity is the standard effective sample count (sum w)^2 / sum w^2 =
exp(2*LSE(S) - LSE(2S)) in [1, N], which is what we compute.

Usage:
  python j1_coupling.py --dataset full   --boot 500
  python j1_coupling.py --dataset fresh2 --boot 500
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import CODE_ROOT
from w1prime_delta import (
    C1, EPS0, LAM_PLUS_PROD, Z_REF, DATASETS,
    load, with_operator, wls, traj_slices,
)

MIN_BUCKET_N = 100


def wls_r2(x, y, w, icept, slope):
    w = np.asarray(w, dtype=np.float64)
    yhat = icept + slope * np.asarray(x)
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def buckets_j1(phi, sigma, S, T, bins, min_n=MIN_BUCKET_N):
    idx = np.digitize(phi, bins)
    rows = []
    for b in np.unique(idx):
        m = idx == b
        N = int(m.sum())
        if N < min_n:
            continue
        Sb = S[m]
        mu = Sb.mean()
        v = Sb.var(ddof=1)
        o = mu
        logL1 = logsumexp(Sb - o) - np.log(N) + o
        logL2 = logsumexp(2 * Sb - 2 * o) - np.log(N) + 2 * o
        rows.append(dict(
            bin=int(b), n=N,
            phi=float(np.median(phi[m])), sigma=float(np.median(sigma[m])),
            T_med=float(np.median(T[m])),
            mu=float(mu), v=float(v),
            Delta_direct=float(logL1 - logL2),
            Delta_cum=float(-mu - 1.5 * v),
            ess=float(np.exp(2 * logL1 - logL2 + np.log(N))),
        ))
    res = pd.DataFrame(rows)
    if len(res):
        res["cum_err"] = (res["Delta_cum"] - res["Delta_direct"]).abs()
        res["ok"] = ((res["cum_err"] < 0.1 * res["Delta_direct"].abs())
                     & (res["ess"] > 50))
    return res


def fit_j1(res):
    """Returns (beta1, beta2, lam_eff, lam_semi, fits) or None if too thin."""
    usable = res[res["ess"] > 50]
    if len(usable) < 3:
        return None
    ok = res[res["ok"]] if int(res["ok"].sum()) >= 3 else usable
    a1, s1 = wls(ok["sigma"].values, ok["mu"].values, ok["n"].values)
    a2, s2 = wls(ok["sigma"].values, ok["v"].values, ok["n"].values)
    beta1, beta2 = -s1, s2
    lam_eff = C1 * (beta1 - 1.5 * beta2)
    _, lam_semi = wls(usable["phi"].values, usable["Delta_direct"].values,
                      usable["n"].values)
    fits = dict(a1=a1, s1=s1, a2=a2, s2=s2,
                r2_mu=wls_r2(ok["sigma"], ok["mu"], ok["n"], a1, s1),
                r2_v=wls_r2(ok["sigma"], ok["v"], ok["n"], a2, s2),
                n_ok=int(res["ok"].sum()), n_buckets=len(res))
    return beta1, beta2, lam_eff, lam_semi, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--boot-max-trajs", type=int, default=4000)
    args = ap.parse_args()

    tag = args.dataset
    outdir = os.path.join(CODE_ROOT, "results", "j1")
    os.makedirs(outdir, exist_ok=True)

    df = load(args.dataset)
    bins = np.geomspace(EPS0, 1.0, 21)
    phi = df["phi"].values
    sig = df["sigma"].values
    T = df["T"].values

    report = {"dataset": tag, "n_tokens": int(len(df)),
              "n_traj": int(df["traj_id"].nunique()),
              "c1": C1, "eps0": EPS0, "z_ref": Z_REF,
              "lambda_plus_ablation": LAM_PLUS_PROD}

    # ---- main pass at z_ref -----------------------------------------------
    _, S = with_operator(df, Z_REF)
    res = buckets_j1(phi, sig, S, T, bins)
    res.to_csv(os.path.join(outdir, f"buckets_{tag}.csv"), index=False)
    fit = fit_j1(res)
    beta1, beta2, lam_eff, lam_semi, fits = fit
    report["main"] = dict(beta1=beta1, beta2=beta2, lam_eff=lam_eff,
                          lam_semi=lam_semi, **fits)
    report["cum_pass_frac"] = float(res["ok"].mean())

    # ---- trajectory bootstrap ---------------------------------------------
    starts = traj_slices(df)
    n_traj = len(starts) - 1
    rng = np.random.RandomState(3)
    pool = np.arange(n_traj)
    if n_traj > args.boot_max_trajs:
        pool = rng.choice(n_traj, size=args.boot_max_trajs, replace=False)
    b1s, b2s, les, lss = [], [], [], []
    for _ in range(args.boot):
        pick = rng.choice(pool, size=len(pool), replace=True)
        idx = np.concatenate([np.arange(starts[i], starts[i + 1])
                              for i in pick])
        f = fit_j1(buckets_j1(phi[idx], sig[idx], S[idx], T[idx], bins))
        if f is None:
            continue
        b1s.append(f[0]); b2s.append(f[1]); les.append(f[2]); lss.append(f[3])
    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    report["boot"] = dict(n_eff=len(les),
                          beta1_ci=ci(b1s), beta2_ci=ci(b2s),
                          lam_eff_ci=ci(les), lam_semi_ci=ci(lss),
                          lam_eff_se=float(np.std(les, ddof=1)))

    # ---- T stratification --------------------------------------------------
    tq = df.groupby("traj_id")["T"].first()
    edges = tq.quantile([0, .25, .5, .75, 1.0]).values
    strata = {}
    for k in range(4):
        ids = tq[(tq >= edges[k]) & (tq <= edges[k + 1])].index
        m = df["traj_id"].isin(ids).values
        f = fit_j1(buckets_j1(phi[m], sig[m], S[m], T[m], bins))
        if f is not None:
            strata[f"T[{edges[k]:.0f},{edges[k+1]:.0f}]"] = dict(
                beta1=f[0], beta2=f[1], lam_eff=f[2])
    report["strata"] = strata

    # ---- z_plus sensitivity ------------------------------------------------
    sens = {}
    for mult in (0.5, 1.0, 2.0):
        _, Sz = with_operator(df, Z_REF * mult)
        f = fit_j1(buckets_j1(phi, sig, Sz, T, bins))
        if f is not None:
            sens[str(mult)] = dict(beta1=f[0], beta2=f[1], lam_eff=f[2])
    report["sensitivity"] = sens

    # ---- verdict (section 9) ----------------------------------------------
    b1lo, b1hi = report["boot"]["beta1_ci"]
    b2lo, b2hi = report["boot"]["beta2_ci"]
    lelo, lehi = report["boot"]["lam_eff_ci"]
    b1_sig = not (b1lo <= 0 <= b1hi)
    b2_sig = not (b2lo <= 0 <= b2hi)
    le_sig = not (lelo <= 0 <= lehi)
    if not b1_sig and not b2_sig:
        verdict = "W1-FIRST-ORDER: beta1, beta2 both insignificant"
    elif lam_eff > 0 and le_sig and lam_eff > 0.1 * LAM_PLUS_PROD:
        verdict = ("MEAN-CHANNEL: lam_eff > 0 same order as lambda_+ -> "
                   "rewrite Step 4-5 via J1; (W1) becomes beta_j=0 special case")
    elif lam_eff > 0 and le_sig:
        verdict = ("PARTIAL: lam_eff > 0 but an order below lambda_+ -> "
                   "coupling real, band basis stays with E2 measurement")
    elif le_sig:
        verdict = ("VARIANCE-CHANNEL: lam_eff < 0 -> theory wants TIGHTER "
                   "band at low confidence; constant cap loose at sequence "
                   "level (finding, not failure)")
    else:
        verdict = "INCONCLUSIVE: lam_eff CI covers 0 with significant betas"
    if report["cum_pass_frac"] < 0.5:
        verdict += (" | CUMULANT GATE FAILED for most buckets -> closed form "
                    "abandoned, use lam_semi (semi-parametric Delta slope)")
    report["verdict"] = verdict

    with open(os.path.join(outdir, f"report_{tag}.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- figures -----------------------------------------------------------
    figdir = os.path.join(CODE_ROOT, "results", "figs")
    okm = res["ok"].values
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.linspace(0, res["sigma"].max() * 1.05, 50)
    ax[0].scatter(res["sigma"], res["mu"], s=np.sqrt(res["n"]) / 3,
                  c=np.where(okm, "C0", "lightgray"))
    ax[0].plot(xs, fits["a1"] + fits["s1"] * xs, "C0--",
               label=f"beta1={beta1:.2f}, R2={fits['r2_mu']:.3f}")
    ax[0].set_xlabel(r"$\sigma$"); ax[0].set_ylabel(r"$\mu(S_{-t})$")
    ax[0].legend(fontsize=8); ax[0].set_title(f"{tag}: mean channel")
    ax[1].scatter(res["sigma"], res["v"], s=np.sqrt(res["n"]) / 3,
                  c=np.where(okm, "C1", "lightgray"))
    ax[1].plot(xs, fits["a2"] + fits["s2"] * xs, "C1--",
               label=f"beta2={beta2:.2f}, R2={fits['r2_v']:.3f}")
    ax[1].set_xlabel(r"$\sigma$"); ax[1].set_ylabel(r"$v(S_{-t})$")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"variance channel | lam_eff={lam_eff:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"J1_fits_{tag}.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(res["phi"], res["Delta_direct"], "ko-", ms=4, label="Delta direct")
    ax.plot(res["phi"], res["Delta_cum"], "rx--", ms=5, label="Delta cumulant")
    for _, r in res.iterrows():
        ax.annotate(f"{r['ess']:.0f}", (r["phi"], r["Delta_direct"]),
                    fontsize=6, xytext=(0, 6), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel(r"$\varphi$")
    ax.set_title(f"{tag}: cumulant acceptance "
                 f"(pass {report['cum_pass_frac']:.0%}, ESS annotated)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"J1_accept_{tag}.png"), dpi=150)

    print(json.dumps(report, indent=2))
    print(f"[j1] verdict: {verdict}")


if __name__ == "__main__":
    main()
