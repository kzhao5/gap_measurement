"""Verify log k = c(1-p)Z + delta*B (VERIFY_Z_B_CONTAMINATION_MODEL.md).

Data sources:
  campaign  : 12.3M-token fresh MoE measurement (clean scale law A, Z pivot B)
  paired    : lg0-e3 checkpoint pairs (B^emp = pre_v86 - pre_v57/v28; C,D,E)
  ktdump    : real-RL per-token dumps with version field (eps F, clip
              enrichment H, group-level additivity G at true lags 1-2)
Closure I solves the robust-MSE stationarity for tau -> lambda_theory,
compared against the RL-optimal lambda_+ = 2.3. J maps eta(p) separability.

Usage: python contamination_verify.py [--dump recipe_sigmatis]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import CODE_ROOT, DATA_ROOT
from w1prime_delta import load as load_campaign, traj_slices

RLROOT = "/home/kzhao2/nobackup/autodelete/areal_rl"
PAIRED = os.path.join(DATA_ROOT, "paired")
OUT = os.path.join(CODE_ROOT, "results", "contamination")
FIG = os.path.join(CODE_ROOT, "results", "figs", "contamination")
N_BATCH = 1024
LAM_RL = 2.3
EPS0 = 5e-3


def wfit_origin(x, y, w):
    """y = c*x weighted; returns c, weighted R^2."""
    c = np.sum(w * x * y) / np.sum(w * x * x)
    r = y - c * x
    r2 = 1 - np.sum(w * r**2) / np.sum(w * (y - np.average(y, weights=w))**2)
    return float(c), float(r2)


def mad_scale(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


# ---------------------------------------------------------------- A + B
def exp_AB(df, tag, rep, nbins=30):
    lk = df["log_k"].values
    p = np.exp(df["logp_train"].astype(np.float64).values)
    u = 1.0 - p
    qb = np.quantile(p, np.linspace(0, 1, nbins + 1))
    bi = np.clip(np.digitize(p, qb[1:-1]), 0, nbins - 1)
    rows = []
    for b in range(nbins):
        m = bi == b
        if m.sum() < 100:
            continue
        rows.append(dict(bin=b, n=int(m.sum()), u_med=float(np.median(u[m])),
                         mad=mad_scale(lk[m]), std=float(lk[m].std()),
                         iqr=float(np.subtract(*np.quantile(lk[m],
                                                            [.75, .25]))
                                   / 1.349)))
    sc = pd.DataFrame(rows)
    core = sc[sc["u_med"] >= 1e-3]          # exclude finite-precision floor
    w = core["n"].values.astype(float)
    c_hat, r2 = wfit_origin(core["u_med"].values, core["mad"].values, w)
    # affine and power alternatives
    X = np.column_stack([np.ones(len(core)), core["u_med"]])
    ab, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None],
                             core["mad"].values * np.sqrt(w), rcond=None)
    lg, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(len(core)), np.log(core["u_med"])])
        * np.sqrt(w)[:, None],
        np.log(core["mad"].values) * np.sqrt(w), rcond=None)
    pred = c_hat * core["u_med"].values
    maxrel = float(np.max(np.abs(core["mad"].values - pred) / pred))
    rep[f"A_{tag}"] = dict(
        c=c_hat, weighted_R2=r2, max_rel_bin_error=maxrel,
        affine=dict(a=float(ab[0]), c=float(ab[1])),
        power=dict(logc=float(lg[0]), gamma=float(lg[1])),
        n_core_bins=len(core), n_floor_bins=int((sc["u_med"] < 1e-3).sum()))
    sc.to_csv(os.path.join(OUT, f"clean_scale_{tag}.csv"), index=False)

    # ---- B: Z pivotality on 10 bins
    Z = lk / (c_hat * np.maximum(u, 1e-12))
    qb10 = np.quantile(p, np.linspace(0, 1, 11))
    b10 = np.clip(np.digitize(p, qb10[1:-1]), 0, 9)
    gcore = Z[(u >= 1e-3)]
    gq = np.quantile(gcore, [.01, .99])
    gref = gcore[(gcore >= gq[0]) & (gcore <= gq[1])]
    zrows = []
    for b in range(10):
        zb = Z[b10 == b]
        if len(zb) < 1000:
            continue
        qs = np.quantile(zb, [.001, .01, .1, .5, .9, .99, .999])
        zc = zb[(zb >= gq[0]) & (zb <= gq[1])]
        sub = np.random.default_rng(b).choice(
            zc, size=min(len(zc), 200000), replace=False)
        ks = float(stats.ks_2samp(sub, np.random.default_rng(99).choice(
            gref, size=min(len(gref), 200000), replace=False)).statistic)
        w1 = float(stats.wasserstein_distance(sub[:50000], gref[:50000]))
        zrows.append(dict(bin=b, u_med=float(np.median(1 - p[b10 == b])),
                          **{f"q{q}": float(v) for q, v in zip(
                              ["001", "01", "10", "50", "90", "99", "999"],
                              qs)}, ks_core=ks, w1_core=w1))
    zdf = pd.DataFrame(zrows)
    zdf.to_csv(os.path.join(OUT, f"Z_pivotality_{tag}.csv"), index=False)
    nonfloor = zdf[zdf["u_med"] >= 1e-3]
    rep[f"B_{tag}"] = dict(
        core_q10_spread=float(nonfloor["q10"].max() - nonfloor["q10"].min()),
        core_q90_spread=float(nonfloor["q90"].max() - nonfloor["q90"].min()),
        max_ks_core=float(nonfloor["ks_core"].max()),
        note="floor bins (u<1e-3) = finite-precision regime, excluded")
    return c_hat, Z, u, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="recipe_sigmatis")
    ap.add_argument("--shards", default="0,1,2,3")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rep = {}

    # ================= campaign clean data: A + B =========================
    dfc = load_campaign("full")
    c_hat, Zc, uc, pc = exp_AB(dfc, "campaign", rep)
    print(f"[A] campaign c={c_hat:.4f} R2={rep['A_campaign']['weighted_R2']:.3f}")

    # ================= paired data: A(fresh), C, D, E =====================
    parts = []
    for s in [int(x) for x in args.shards.split(",")]:
        base = pq.read_table(os.path.join(
            PAIRED, f"trajs_s{s}.parquet")).to_pylist()
        flat = {"traj_id": [], "pos": [], "logp_stale_dec": []}
        for t in base:
            P = len(t["prompt_token_ids"])
            for k, lp in enumerate(t["logp_stale_dec"]):
                flat["traj_id"].append(t["traj_id"] + s * 1000000)
                flat["pos"].append(P + k)
                flat["logp_stale_dec"].append(lp)
        d = pd.DataFrame(flat)
        for name in ("pre_v57", "pre_v86", "pre_v28", "hf_v86"):
            e = pq.read_table(os.path.join(
                PAIRED, f"{name}_s{s}.parquet")).to_pandas()
            e["traj_id"] = e["traj_id"] + s * 1000000
            e = e.rename(columns={"logp": name})
            d = d.merge(e, on=["traj_id", "pos"])
        parts.append(d)
    dp = pd.concat(parts, ignore_index=True).sort_values(
        ["traj_id", "pos"], ignore_index=True)
    dp["log_k"] = dp["hf_v86"].astype(np.float64) - \
        dp["pre_v86"].astype(np.float64)
    dp["logp_train"] = dp["hf_v86"]
    rep["paired_n_tokens"] = int(len(dp))
    c_pair, _, up, pp = exp_AB(dp, "paired_fresh", rep, nbins=20)

    B29 = (dp["pre_v86"].astype(np.float64)
           - dp["pre_v57"].astype(np.float64)).values
    B58 = (dp["pre_v86"].astype(np.float64)
           - dp["pre_v28"].astype(np.float64)).values
    ctrl = (dp["pre_v57"].astype(np.float64)
            - dp["logp_stale_dec"].astype(np.float64)).values
    rep["prefill_vs_decode_ctrl"] = dict(
        median=float(np.median(ctrl)), mad=mad_scale(ctrl))
    lk0 = dp["log_k"].values

    # NOTE on sign: B here = log pi_fresh - log pi_stale on stale-generated
    # tokens. The threat model's contamination enters log k^(l) = old -
    # stale = logk0 + B, so positive B inflates log k (right tail).
    for nm, Bv in (("lag29", B29), ("lag58", B58)):
        X = np.column_stack([np.ones(len(Bv)), up, lk0])
        beta, *_ = np.linalg.lstsq(X, Bv, rcond=None)
        sd = dict(a=float(beta[0]), b1_u=float(beta[1]),
                  b2_logk0=float(beta[2]),
                  b1_effect=float(abs(beta[1]) * up.std() / Bv.std()),
                  b2_effect=float(abs(beta[2]) * lk0.std() / Bv.std()))
        qb = np.quantile(pp, np.linspace(0, 1, 11))
        bi = np.clip(np.digitize(pp, qb[1:-1]), 0, 9)
        brows = []
        for b in range(10):
            m = bi == b
            if m.sum() < 100:
                continue
            qs = np.quantile(Bv[m], [.1, .5, .9, .95, .99])
            brows.append(dict(
                p_bin=b, u_med=float(np.median(up[m])), n=int(m.sum()),
                mean_B=float(Bv[m].mean()), median_B=float(qs[1]),
                q10=float(qs[0]), q90=float(qs[2]), q95=float(qs[3]),
                q99=float(qs[4]),
                positive_rate=float((Bv[m] > 0).mean()),
                med_std=float(np.median(Bv[m] / np.maximum(
                    c_hat * up[m], 1e-9)))))
        bdf = pd.DataFrame(brows)
        bdf.to_csv(os.path.join(OUT, f"B_paired_{nm}.csv"), index=False)
        med_by_bin = bdf["median_B"].values
        rep[f"C_{nm}"] = dict(
            **sd, median_B_range=[float(med_by_bin.min()),
                                  float(med_by_bin.max())],
            median_B_global=float(np.median(Bv)),
            standardized_blowup_ratio=float(
                bdf["med_std"].iloc[-1] / max(abs(bdf["med_std"].iloc[0]),
                                              1e-9)))
        rep[f"D_{nm}"] = dict(
            rho_plus=float((Bv > 0).mean()),
            p_gt=[float((Bv > x).mean()) for x in (.01, .05, .1)],
            p_lt=[float((Bv < -x).mean()) for x in (.01, .05)])
        rep[f"E_{nm}"] = dict(
            q90=float(np.quantile(Bv, .90)), q95=float(np.quantile(Bv, .95)),
            q99=float(np.quantile(Bv, .99)),
            q999=float(np.quantile(Bv, .999)), max=float(Bv.max()),
            q99_half=float(np.quantile(Bv[:len(Bv) // 2], .99)))

    # ================= ktdump: F, H, G ====================================
    files = sorted(glob.glob(os.path.join(RLROOT, "ktdump", args.dump,
                                          "*.parquet")))
    dts = []
    for f in files:
        t = pq.read_table(f).to_pandas()
        t["lag"] = t["version"].max() - t["version"]
        dts.append(t)
    dd = pd.concat(dts, ignore_index=True)
    dd["log_k"] = dd["prox_logp"].astype(np.float64) - \
        dd["old_logp"].astype(np.float64)
    pd_ = np.exp(dd["prox_logp"].astype(np.float64).values)
    ud = 1.0 - pd_
    lkd = dd["log_k"].values
    lag = dd["lag"].values
    rep["ktdump"] = dict(dump=args.dump, n_tokens=int(len(dd)),
                         n_files=len(files))
    eps1, eps2 = float((lag >= 1).mean()), float((lag >= 2).mean())
    rep["F_epsilon"] = dict(
        lag_ge1=eps1, lag_ge2=eps2,
        B_gt={"0.01": float((B29 > .01).mean()),
              "0.05": float((B29 > .05).mean()),
              "0.10": float((B29 > .10).mean())},
        clustering="not measurable: dumps carry no traj/pos structure")

    # H: clip enrichment, delta = lag>=1
    hrows = []
    delta = lag >= 1
    for lam in (1.0, 1.3, 1.6, 2.0, 2.3, 2.6, 3.1, 4.0):
        A = lkd > lam * np.maximum(ud, EPS0)
        cr = float(A.mean())
        prec = float(delta[A].mean()) if A.any() else np.nan
        rec = float(A[delta].mean())
        fpr = float(A[~delta].mean())
        hrows.append(dict(arch="moe", lam=lam, clip_rate=cr,
                          contam_precision=prec, contam_recall=rec,
                          contam_enrichment=prec / max(eps1, 1e-12),
                          clean_false_positive_rate=fpr))
    hdf = pd.DataFrame(hrows)
    hdf.to_csv(os.path.join(OUT, "clip_enrichment.csv"), index=False)
    rep["H_at_2.3"] = hdf[hdf.lam == 2.3].iloc[0].to_dict()

    # G: group-level additivity at true lags: quantile shifts lag1 vs lag0
    qb = np.quantile(pd_, np.linspace(0, 1, 9))
    bi = np.clip(np.digitize(pd_, qb[1:-1]), 0, 7)
    grows = []
    for b in range(8):
        m0 = (bi == b) & (lag == 0)
        m1 = (bi == b) & (lag >= 1)
        if m0.sum() < 500 or m1.sum() < 500:
            continue
        for q in (.5, .9, .99):
            shift = float(np.quantile(lkd[m1], q) - np.quantile(lkd[m0], q))
            grows.append(dict(p_bin=b, u_med=float(np.median(ud[bi == b])),
                              q=q, shift=shift,
                              shift_over_u=shift / max(
                                  float(np.median(ud[bi == b])), 1e-9)))
    gdf = pd.DataFrame(grows)
    gdf.to_csv(os.path.join(OUT, "G_lag_quantile_shifts.csv"), index=False)
    g99 = gdf[gdf.q == .99]
    if len(g99) >= 3:
        cv_add = float(g99["shift"].std() / max(abs(g99["shift"].mean()),
                                                1e-9))
        cv_scale = float(g99["shift_over_u"].std()
                         / max(abs(g99["shift_over_u"].mean()), 1e-9))
        rep["G_additive_vs_scaling"] = dict(
            cv_shift_additive_model=cv_add, cv_shift_scaling_model=cv_scale,
            additive_wins=bool(cv_add < cv_scale))

    # ================= I: theory closure ==================================
    zcore = Zc[(uc >= 1e-3)]
    zs = np.random.default_rng(1).choice(zcore, size=min(len(zcore),
                                                         2_000_000),
                                         replace=False)
    mz = float(zs.mean())
    for eps_name, eps in (("lag_ge1", eps1), ("lag_ge2", eps2)):
        lo, hi = mz, float(np.quantile(zs, 0.999999)) * 2
        for _ in range(80):
            tau = 0.5 * (lo + hi)
            gl = (1 - eps) * np.mean(np.maximum(zs - tau, 0)) \
                - (eps + 1 / (N_BATCH - 1)) * (tau - mz)
            if gl > 0:
                lo = tau
            else:
                hi = tau
        tau_th = 0.5 * (lo + hi)
        rep[f"I_closure_{eps_name}"] = dict(
            eps=eps, tau_theory=float(tau_th),
            lambda_theory=float(c_hat * tau_th), lambda_RL=LAM_RL,
            ratio=float(c_hat * tau_th / LAM_RL))

    # ================= J: separability eta(p) =============================
    tau_use = rep["I_closure_lag_ge1"]["tau_theory"]
    for bname, bmax in (("q95", rep["E_lag29"]["q95"]),
                        ("q99", rep["E_lag29"]["q99"])):
        etas = []
        for u0 in np.geomspace(1e-3, 1, 30):
            thr = tau_use - bmax / (c_hat * u0)
            etas.append(dict(u=float(u0),
                             eta=float((zs <= thr).mean())))
        ed = pd.DataFrame(etas)
        ok01 = ed[ed.eta < 0.01]["u"]
        rep[f"J_eta_{bname}"] = dict(
            Bmax=float(bmax),
            u_where_eta_lt_01=float(ok01.min()) if len(ok01) else None)

    # ================= acceptance checklist ===============================
    A_ok = rep["A_campaign"]["weighted_R2"] > 0.95
    B_ok = rep["B_campaign"]["max_ks_core"] < 0.15
    C_ok = (rep["C_lag29"]["b1_effect"] < 0.1
            and rep["C_lag29"]["b2_effect"] < 0.1)
    D_ok = rep["D_lag29"]["rho_plus"] >= 0.6
    onesided = rep["D_lag29"]["rho_plus"] >= 0.9
    E_ok = np.isfinite(rep["E_lag29"]["q99"])
    H_ok = rep["H_at_2.3"]["contam_enrichment"] > 1.5
    I_ok = 0.1 < rep["I_closure_lag_ge1"]["ratio"] < 10
    G_ok = rep.get("G_additive_vs_scaling", {}).get("additive_wins", False)
    rep["checklist"] = dict(
        A_scale_law=A_ok, B_pivotal_core=B_ok, C_B_nonscaling=C_ok,
        D_positive_dominant=D_ok, D_strongly_onesided=onesided,
        E_high_prob_bounded=E_ok, F_eps_estimable=True,
        G_additive_beats_scaling=G_ok, H_clip_enriched=H_ok,
        I_order_of_magnitude_closure=I_ok)
    level = ("A" if all([A_ok, B_ok, C_ok, D_ok, E_ok]) else
             "B" if (C_ok and D_ok) else "C")
    rep["evidence_level"] = level
    rep["verdict"] = (
        f"Level {level}: paired displacement is "
        f"{'non-scaling, positive-dominant' if (C_ok and D_ok) else 'PROBLEMATIC'}"
        f"; rho+={rep['D_lag29']['rho_plus']:.2f}; "
        f"clip enrichment@2.3={rep['H_at_2.3']['contam_enrichment']:.1f}x; "
        f"lambda_theory/lambda_RL={rep['I_closure_lag_ge1']['ratio']:.2f}")

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)

    # ================= key figures ========================================
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    sc = pd.read_csv(os.path.join(OUT, "clean_scale_campaign.csv"))
    ax[0, 0].loglog(sc["u_med"], sc["mad"], "o", ms=4)
    xs = np.geomspace(sc["u_med"].min(), 1, 40)
    ax[0, 0].loglog(xs, c_hat * xs, "--",
                    label=f"c(1-p), c={c_hat:.3f}")
    ax[0, 0].legend(fontsize=8); ax[0, 0].set_title("01 clean scale law")
    zdf = pd.read_csv(os.path.join(OUT, "Z_pivotality_campaign.csv"))
    for col in ("q10", "q50", "q90"):
        ax[0, 1].plot(zdf["bin"], zdf[col], "o-", ms=3, label=col)
    ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("03 Z quantile flatness")
    b29 = pd.read_csv(os.path.join(OUT, "B_paired_lag29.csv"))
    ax[0, 2].plot(b29["u_med"], b29["median_B"], "o-", label="median B lag29")
    ax[0, 2].plot(b29["u_med"], b29["q90"], "s--", label="q90")
    ax[0, 2].set_xscale("log"); ax[0, 2].legend(fontsize=8)
    ax[0, 2].set_title("04 B vs 1-p (flat = additive)")
    ax[1, 0].hist(B29, bins=200, range=(-1, 1), log=True)
    ax[1, 0].axvline(0, c="k", lw=.5)
    ax[1, 0].set_title(f"05 signed B lag29 (rho+="
                       f"{rep['D_lag29']['rho_plus']:.2f})")
    ax[1, 1].plot(hdf["lam"], hdf["contam_enrichment"], "o-")
    ax[1, 1].axvline(2.3, ls="--", c="gray")
    ax[1, 1].set_title("09 clip contamination enrichment")
    ax[1, 1].set_xlabel("lambda")
    for bname, c in (("q95", "C0"), ("q99", "C1")):
        bmax = rep[f"J_eta_{bname}"]["Bmax"]
        us = np.geomspace(1e-3, 1, 30)
        ax[1, 2].semilogx(us, [(zs <= tau_use - bmax / (c_hat * u)).mean()
                               for u in us], c=c, label=f"Bmax={bname}")
    ax[1, 2].axhline(0.01, ls="--", c="gray")
    ax[1, 2].legend(fontsize=8); ax[1, 2].set_title("11 eta(p) separability")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "contamination_summary.png"), dpi=150)

    print(json.dumps({k: v for k, v in rep.items()
                      if not isinstance(v, dict) or len(str(v)) < 400},
                     indent=2, default=str))
    print(f"[contamination] {rep['verdict']}")


if __name__ == "__main__":
    main()
