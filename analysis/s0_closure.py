"""S0 raw law + best-response family closure + fixed-point iteration.

Implements S0_AND_CLOSURE_EXPERIMENT_GUIDE.md end to end:
  A: S0_-t = alpha0 - lambda0*(1-p) + beta0*D + eps  (raw leave-one-out law)
     acceptance: E_D = |beta0|*SD(D)/SD(S0) < 0.05, nonlinearity dR2 < 0.01,
     residual ECDF collapse across p/D bins, exp-moment stability (LSE+ESS)
  B: closure regressions of S^{C,lambda}_-t over the pre-registered grid
  C: direct Delta closure on a (p, D) cell grid,
     Delta = log E[e^{2S}] - log E[e^{S}] (sign per THIS guide),
     fit Delta = A - lambda'*(1-p) + gamma*D, E_Delta < 0.10
  D: update map (C,lambda)->(C',lambda'): lambda' from C-fit; C' via the
     1-D self-consistency kappa' = n[1 - E prod min{k, kappa' e^{-A+l'u}}]
  E: fixed-point iteration from 5 initializations, <=20 iters, operator
     convergence on a held-out (k,p) grid
  Risk: R_n(M) = (E|W_M - W|)^2 + E[W_M^2]/n on held-out trajectories for
     {nocorr, TIS, IcePop, KPop, iterates, M*}
  Robustness: per-arch (moe/dense), T-strata for the raw law, trajectory
     permutation null (break within-traj association, keep marginals)

n = 1024 (batch size, matches the M* frontier computation in ANALYSIS.md).
Split: 70/30 by trajectory, seed 42. All heavy expectations via logsumexp.

Usage: python s0_closure.py --arch moe [--boot 300] [--shards 0,1]
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

N_BATCH = 1024
GRID_C = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
GRID_L = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
INITS_EXTRA = [(1.0, 0.0), (1.0, 1.0), (0.75, 2.0), (1.5, 0.5)]
ESS_MIN_CELL = 50

OUT = os.path.join(CODE_ROOT, "results", "s0closure")
FIG = os.path.join(CODE_ROOT, "results", "figs", "s0closure")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load(arch, shards):
    parts = []
    for s in shards:
        tok = pq.read_table(
            os.path.join(DATA_ROOT, "gen", arch, f"tokens_shard{s:03d}.parquet"),
            columns=["traj_id", "pos", "logp_infer"]).to_pandas()
        rec = pq.read_table(
            os.path.join(DATA_ROOT, "recompute", arch,
                         f"recomp_shard{s:03d}_bf16.parquet"),
            columns=["traj_id", "pos", "logp_train"]).to_pandas()
        parts.append(tok.merge(rec, on=["traj_id", "pos"], how="inner"))
    df = pd.concat(parts, ignore_index=True).sort_values(
        ["traj_id", "pos"], ignore_index=True)
    df["D"] = df["logp_train"].astype(np.float64) - \
        df["logp_infer"].astype(np.float64)
    df["u"] = 1.0 - np.exp(df["logp_train"].astype(np.float64))
    print(f"[load] {arch}: {len(df)} tokens, {df.traj_id.nunique()} trajs")
    return df


def traj_starts(tid):
    return np.r_[0, np.nonzero(tid[1:] != tid[:-1])[0] + 1, len(tid)]


def loo_sums(x, starts):
    """leave-one-out sums per token given trajectory start offsets."""
    sums = np.add.reduceat(x, starts[:-1])
    tot = np.repeat(sums, np.diff(starts))
    return tot - x


# --------------------------------------------------------------------------
# regression helpers
# --------------------------------------------------------------------------

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    return beta, r2, resid


def fit_law(S, u, D, with_D=True):
    """S = a + b_u*u + b_D*D. Returns dict with guide's parameterization
    (lambda = -b_u, beta = b_D)."""
    cols = [np.ones_like(u), u] + ([D] if with_D else [])
    beta, r2, resid = ols(np.column_stack(cols), S)
    out = dict(alpha=float(beta[0]), lam=float(-beta[1]), r2=float(r2))
    out["beta_D"] = float(beta[2]) if with_D else 0.0
    out["E_D"] = float(abs(out["beta_D"]) * D.std() / max(S.std(), 1e-12))
    return out, resid


def expanded_r2(S, u, D, tr, te):
    """held-out R2 gain from quadratic terms."""
    Xl = np.column_stack([np.ones_like(u), u, D])
    Xe = np.column_stack([np.ones_like(u), u, D, u ** 2, D ** 2])
    r2 = {}
    for name, X in (("linear", Xl), ("expanded", Xe)):
        beta, *_ = np.linalg.lstsq(X[tr], S[tr], rcond=None)
        pr = S[te] - X[te] @ beta
        r2[name] = 1 - pr.var() / S[te].var()
    return float(r2["expanded"] - r2["linear"]), float(r2["linear"])


def log_moments(eps):
    """log E[e^eps], log E[e^2eps], ESS for both, top-0.1% share of L1."""
    N = len(eps)
    l1 = logsumexp(eps) - np.log(N)
    l2 = logsumexp(2 * eps) - np.log(N)
    ess1 = float(np.exp(2 * logsumexp(eps) - logsumexp(2 * eps)))
    ess2 = float(np.exp(2 * logsumexp(2 * eps) - logsumexp(4 * eps)))
    k = max(1, int(0.001 * N))
    top = np.partition(eps, -k)[-k:]
    share = float(np.exp(logsumexp(top) - logsumexp(eps)))
    w = np.clip(eps, None, np.quantile(eps, 0.999))
    l1w = logsumexp(w) - np.log(N)
    return dict(logE1=float(l1), logE2=float(l2), ess1=ess1, ess2=ess2,
                top01_share=share, logE1_winsor=float(l1w))


# --------------------------------------------------------------------------
# operator machinery
# --------------------------------------------------------------------------

def log_m_op(D, u, C, lam):
    return np.minimum(D, np.log(C) + lam * u)


def delta_surface(S, u, D, ubins, dbins):
    """Delta = log E[e^2S|cell] - log E[e^S|cell] on a (u, D) cell grid."""
    ui = np.digitize(u, ubins)
    di = np.digitize(D, dbins)
    cell = ui * 100 + di
    rows = []
    for c in np.unique(cell):
        m = cell == c
        N = int(m.sum())
        if N < ESS_MIN_CELL:
            continue
        Sb = S[m]
        l1 = logsumexp(Sb) - np.log(N)
        l2 = logsumexp(2 * Sb) - np.log(N)
        ess2 = float(np.exp(2 * logsumexp(Sb) - logsumexp(2 * Sb)))
        if ess2 < ESS_MIN_CELL:
            continue
        rows.append(dict(u=float(np.median(u[m])), D=float(np.median(D[m])),
                         n=N, Delta=float(l2 - l1), ess2=ess2))
    return pd.DataFrame(rows)


def fit_delta(surf):
    """Delta = A - lambda'*u + gamma*D, n-weighted; plus closure error
    E_Delta = RMSE[Delta - (A - lambda'*u)] / SD(Delta).
    Returns NaNs when too few stable cells survive the ESS filter (extreme
    grid corners where the cap silences nearly everything)."""
    if len(surf) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")
    w = np.sqrt(surf["n"].values.astype(float))
    X = np.column_stack([np.ones(len(surf)), surf["u"], surf["D"]])
    beta, *_ = np.linalg.lstsq(X * w[:, None], surf["Delta"].values * w,
                               rcond=None)
    A, lam_p, gamma = float(beta[0]), float(-beta[1]), float(beta[2])
    pred2 = A - lam_p * surf["u"].values
    e = float(np.sqrt(np.average((surf["Delta"] - pred2) ** 2,
                                 weights=surf["n"]))
              / max(surf["Delta"].std(), 1e-12))
    return A, lam_p, gamma, e


def kappa_bisect(D, u, starts, A, lam_p, n=N_BATCH):
    """Solve kappa = n[1 - E prod_t min{k_t, kappa e^{-A + lam'*u_t}}]."""
    def EW(kap):
        lm = np.minimum(D, np.log(kap) - A + lam_p * u)
        s = np.add.reduceat(lm, starts[:-1])
        return float(np.exp(logsumexp(s) - np.log(len(s))))

    lo, hi = 1e-9, float(n)
    g = lambda k: n * (1.0 - EW(k)) - k
    if g(hi) > 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def br_step(df_tr, starts, C, lam, ubins, dbins):
    """one best-response step (C,lambda) -> (C',lambda', diagnostics)."""
    D, u = df_tr["D"].values, df_tr["u"].values
    lm = log_m_op(D, u, C, lam)
    S = loo_sums(lm, starts)
    surf = delta_surface(S, u, D, ubins, dbins)
    A, lam_p, gamma, e = fit_delta(surf)
    if not np.isfinite(A):
        return C, lam, dict(A=A, gamma=gamma, E_Delta=e, kappa=float("nan"))
    kap = kappa_bisect(D, u, starts, A, lam_p)
    Cp = float(kap * np.exp(-A))
    return Cp, lam_p, dict(A=A, gamma=gamma, E_Delta=e, kappa=kap)


# --------------------------------------------------------------------------
# risk
# --------------------------------------------------------------------------

def risk(logWm, logW, n=N_BATCH):
    """R = (E|W_m - W|)^2 + E[W_m^2]/n, log-space per trajectory."""
    hi = np.maximum(logWm, logW)
    d = np.abs(logWm - logW)
    logabs = hi + np.log1p(-np.exp(-np.maximum(d, 1e-300)))
    logabs[d < 1e-12] = -np.inf
    N = len(logWm)
    t1 = np.exp(logsumexp(logabs) - np.log(N)) ** 2
    t2 = np.exp(logsumexp(2 * logWm) - np.log(N)) / n
    return float(t1 + t2), float(t2)


def method_logM(name, D, u, ptr, pin, C=None, lam=None):
    if name == "nocorr":
        return np.zeros_like(D)
    if name == "TIS":
        return np.clip(D, np.log(1e-6), np.log(2.0))
    if name == "IcePop":  # mask [0.5, 5]: out-of-range token excluded (m=1)
        return np.where((D >= np.log(0.5)) & (D <= np.log(5.0)), D, 0.0)
    if name == "KPop":  # mask on binary KL > 2.0
        p, q = np.clip(ptr, 1e-12, 1 - 1e-12), np.clip(pin, 1e-12, 1 - 1e-12)
        bkl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
        return np.where(bkl <= 2.0, D, 0.0)
    return log_m_op(D, u, C, lam)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["moe", "dense"], required=True)
    ap.add_argument("--shards", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--boot-max-trajs", type=int, default=5000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    arch = args.arch

    df = load(arch, [int(s) for s in args.shards.split(",")])
    tid = df["traj_id"].values
    starts_all = traj_starts(tid)
    utraj = df["traj_id"].unique()
    rng = np.random.RandomState(42)
    test_ids = set(rng.choice(utraj, size=int(0.3 * len(utraj)),
                              replace=False))
    is_te = df["traj_id"].isin(test_ids).values
    df_tr, df_te = df[~is_te].reset_index(drop=True), \
        df[is_te].reset_index(drop=True)
    st_tr = traj_starts(df_tr["traj_id"].values)
    st_te = traj_starts(df_te["traj_id"].values)
    report = {"arch": arch, "n_batch": N_BATCH,
              "n_tokens": int(len(df)), "n_traj": int(len(utraj))}

    D, u = df_tr["D"].values, df_tr["u"].values
    Dte, ute = df_te["D"].values, df_te["u"].values

    # ---------------- Experiment A: raw law -------------------------------
    S0 = loo_sums(D, st_tr)
    S0te = loo_sums(Dte, st_te)
    lawA, _ = fit_law(S0, u, D, with_D=True)
    # trajectory bootstrap
    n_tr = len(st_tr) - 1
    pool = np.arange(n_tr)
    if n_tr > args.boot_max_trajs:
        pool = rng.choice(n_tr, size=args.boot_max_trajs, replace=False)
    bl, bb = [], []
    for _ in range(args.boot):
        pick = rng.choice(pool, size=len(pool), replace=True)
        idx = np.concatenate([np.arange(st_tr[i], st_tr[i + 1])
                              for i in pick])
        f, _ = fit_law(S0[idx], u[idx], D[idx])
        bl.append(f["lam"]); bb.append(f["beta_D"])
    lawA["lambda0_ci"] = [float(np.percentile(bl, 2.5)),
                          float(np.percentile(bl, 97.5))]
    lawA["beta0_ci"] = [float(np.percentile(bb, 2.5)),
                        float(np.percentile(bb, 97.5))]
    # nonlinearity on a train/test split of tokens (held-out trajs)
    Sall = np.concatenate([S0, S0te])
    uall = np.concatenate([u, ute]); Dall = np.concatenate([D, Dte])
    tr_m = np.r_[np.ones(len(S0), bool), np.zeros(len(S0te), bool)]
    dr2, r2lin = expanded_r2(Sall, uall, Dall, tr_m, ~tr_m)
    lawA["delta_R2"] = dr2
    lawA["accept_ED"] = lawA["E_D"] < 0.05
    lawA["accept_dR2"] = dr2 < 0.01
    # residual collapse on held-out
    eps = S0te - lawA["alpha"] + lawA["lam"] * ute
    qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    spread = {}
    mad = float(np.median(np.abs(eps - np.median(eps))))
    for var, name in ((ute, "p"), (Dte, "D")):
        bins = np.quantile(var, np.linspace(0, 1, 11))
        bi = np.clip(np.digitize(var, bins[1:-1]), 0, 9)
        qt = np.array([[np.quantile(eps[bi == b], q) for q in qs]
                       for b in range(10)])
        spread[name] = [float(x) for x in (qt.max(0) - qt.min(0)) / mad]
    lawA["Q_spread_over_MAD"] = spread
    lawA["moments"] = log_moments(eps)
    # permutation null (break within-traj association)
    Dp = D.copy(); rng.shuffle(Dp)
    S0p = loo_sums(Dp, st_tr)
    lawNull, _ = fit_law(S0p, u, D)
    lawA["lambda0_permutation_null"] = lawNull["lam"]
    # T-strata
    T = np.diff(st_tr); Ttok = np.repeat(T, T)
    edges = np.quantile(T, [0, 1 / 3, 2 / 3, 1.0])
    lawA["lambda0_T_strata"] = {}
    for k in range(3):
        m = (Ttok >= edges[k]) & (Ttok <= edges[k + 1])
        f, _ = fit_law(S0[m], u[m], D[m])
        lawA["lambda0_T_strata"][f"T[{edges[k]:.0f},{edges[k+1]:.0f}]"] = \
            f["lam"]
    report["A_raw_law"] = lawA
    lam0, alpha0 = lawA["lam"], lawA["alpha"]
    pd.DataFrame([dict(arch=arch, alpha_0=alpha0, lambda_0=lam0,
                       beta_0=lawA["beta_D"], beta_effect=lawA["E_D"],
                       R2=lawA["r2"], delta_R2=dr2, **lawA["moments"])]) \
        .to_csv(os.path.join(OUT, f"s0_raw_law_{arch}.csv"), index=False)

    # first response: C1 from the kappa self-consistency at (A,lam') built
    # from the raw law: Delta ~ alpha0 - lam0*u in the guide's first step
    ubins = np.quantile(u, np.linspace(0, 1, 11))
    dbins = np.quantile(D, np.linspace(0, 1, 11))
    kap1 = kappa_bisect(D, u, st_tr, alpha0, lam0)
    C1 = float(kap1 * np.exp(-alpha0))
    C1 = min(max(C1, 1e-3), 100.0)
    report["first_response"] = dict(C1=C1, lambda1=lam0, kappa1=kap1)

    # ---------------- Experiment B + C: closure grid ----------------------
    grid_rows = []
    grid = [(C, L) for C in GRID_C for L in GRID_L] + [(C1, lam0)]
    for (C, L) in grid:
        lm = log_m_op(D, u, C, L)
        S = loo_sums(lm, st_tr)
        fB, _ = fit_law(S, u, D)
        surf = delta_surface(S, u, D, ubins, dbins)
        A, lam_p, gamma, eD = fit_delta(surf)
        grid_rows.append(dict(
            arch=arch, C=C, lam=L, alpha=fB["alpha"],
            lambda_prime_B=fB["lam"], beta=fB["beta_D"],
            beta_effect=fB["E_D"], R2=fB["r2"],
            A_delta=A, lambda_prime_delta=lam_p, gamma_delta=gamma,
            delta_closure_error=eD, n_cells=len(surf),
        ))
    griddf = pd.DataFrame(grid_rows)
    griddf.to_csv(os.path.join(OUT, f"closure_grid_{arch}.csv"), index=False)
    val = griddf[np.isfinite(griddf["delta_closure_error"])]
    report["B_closure"] = dict(
        frac_ED_ok=float((griddf["beta_effect"] < 0.05).mean()),
        frac_EDelta_ok=float((val["delta_closure_error"] < 0.10).mean()),
        n_grid=len(griddf), n_delta_valid=len(val),
        max_ED=float(griddf["beta_effect"].max()),
        max_gamma=float(val["gamma_delta"].abs().max()),
    )

    # ---------------- Experiment D/E: fixed-point iteration ---------------
    inits = [(C1, lam0)] + INITS_EXTRA
    iters_rows, paths = [], {}
    for init in inits:
        C, L = init
        path = [(C, L)]
        for r in range(20):
            Cp, Lp, diag = br_step(df_tr, st_tr, C, L, ubins, dbins)
            Cp = min(max(Cp, 1e-3), 100.0)
            gridu = np.linspace(0, 1, 101)
            op_change = float(np.max(np.abs(
                np.log(Cp) + Lp * gridu - (np.log(C) + L * gridu))))
            iters_rows.append(dict(arch=arch, initialization=str(init),
                                   iteration=r, C=C, lam=L, C_next=Cp,
                                   lambda_next=Lp, operator_change=op_change,
                                   **diag))
            done = (abs(Cp - C) / (C + 1e-12) < 1e-3
                    and abs(Lp - L) / (abs(L) + 1e-12) < 1e-3)
            C, L = Cp, Lp
            path.append((C, L))
            if done:
                break
        paths[str(init)] = path
    itdf = pd.DataFrame(iters_rows)
    itdf.to_csv(os.path.join(OUT, f"fixed_point_iterations_{arch}.csv"),
                index=False)
    finals = {k: v[-1] for k, v in paths.items()}
    fc = np.array([v[0] for v in finals.values()])
    fl = np.array([v[1] for v in finals.values()])
    report["E_fixed_point"] = dict(
        finals={k: [float(a), float(b)] for k, (a, b) in finals.items()},
        C_star=float(np.median(fc)), lambda_star=float(np.median(fl)),
        unique=bool(fc.std() < 0.05 * max(fc.mean(), 1e-9)
                    and fl.std() < 0.05 * max(abs(fl.mean()), 1e-9)),
    )
    Cs, Ls = report["E_fixed_point"]["C_star"], \
        report["E_fixed_point"]["lambda_star"]

    # ---------------- held-out risk ---------------------------------------
    ptr = np.exp(df_te["logp_train"].values)
    pin = np.exp(df_te["logp_infer"].values)
    logW = np.add.reduceat(Dte, st_te[:-1])
    risks = {}
    methods = [("nocorr", None), ("TIS", None), ("IcePop", None),
               ("KPop", None), ("M1", (C1, lam0)), ("Mstar", (Cs, Ls))]
    seen = 1
    for k, v in paths.items():
        if len(v) > 2:
            methods.insert(5, (f"M2_{seen}", v[2])); seen += 1
            break
    for name, par in methods:
        lm = method_logM(name if par is None else "op", Dte, ute, ptr, pin,
                         *(par or (None, None)))
        logWm = np.add.reduceat(lm, st_te[:-1])
        r, t2 = risk(logWm, logW)
        risks[name] = dict(risk=r, var_term=t2)
    report["risk_heldout"] = risks

    # ---------------- acceptance ------------------------------------------
    report["acceptance"] = dict(
        claim1_raw_law=bool(lawA["accept_ED"] and lawA["accept_dR2"]),
        claim2_closure=bool(
            report["B_closure"]["frac_ED_ok"] >= 0.9
            and report["B_closure"]["frac_EDelta_ok"] >= 0.9),
        fixed_point_converged=bool(
            itdf.groupby("initialization")["operator_change"].last()
            .max() < 0.05),
        unique=report["E_fixed_point"]["unique"],
    )

    with open(os.path.join(OUT, f"report_{arch}.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---------------- figures ---------------------------------------------
    # 01 mean vs uncertainty; 02/03 heatmaps; 04/05 ECDFs
    bins = np.quantile(u, np.linspace(0, 1, 21))
    bi = np.clip(np.digitize(u, bins[1:-1]), 0, 19)
    mb = pd.DataFrame({"u": u, "S": S0, "b": bi}).groupby("b").agg(
        u=("u", "median"), S=("S", "mean"))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(mb["u"], mb["S"], "o", ms=4)
    xs = np.linspace(0, mb["u"].max(), 50)
    ax.plot(xs, alpha0 - lam0 * xs, "--",
            label=f"a0-l0*u, l0={lam0:.2f} [{lawA['lambda0_ci'][0]:.2f},"
                  f"{lawA['lambda0_ci'][1]:.2f}]")
    ax.set_xlabel("1-p"); ax.set_ylabel("E[S0_-t]")
    ax.set_title(f"{arch}: raw law (E_D={lawA['E_D']:.4f}, dR2={dr2:.4f})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"01_s0_mean_vs_uncertainty_{arch}.png"),
                dpi=150)

    ui10 = np.clip(np.digitize(u, ubins[1:-1]), 0, 9)
    di10 = np.clip(np.digitize(D, dbins[1:-1]), 0, 9)
    H = np.full((10, 10), np.nan)
    R = np.full((10, 10), np.nan)
    for a in range(10):
        for b in range(10):
            m = (ui10 == a) & (di10 == b)
            if m.sum() >= 30:
                H[b, a] = S0[m].mean()
                R[b, a] = (S0[m] - (alpha0 - lam0 * u[m])).mean()
    for M, nm, ttl in ((H, "02_s0_mean_heatmap_p_logk",
                        "mean S0 | (1-p, logk)"),
                       (R, "03_s0_residual_heatmap", "residual heatmap")):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="RdBu_r")
        fig.colorbar(im); ax.set_xlabel("1-p decile")
        ax.set_ylabel("log k decile"); ax.set_title(f"{arch}: {ttl}")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"{nm}_{arch}.png"), dpi=150)

    for var, nm in ((ute, "04_s0_residual_ecdf_by_p"),
                    (Dte, "05_s0_residual_ecdf_by_logk")):
        qb = np.quantile(var, np.linspace(0, 1, 11))
        vb = np.clip(np.digitize(var, qb[1:-1]), 0, 9)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for b in range(10):
            e = np.sort(eps[vb == b])
            ax.plot(e, np.linspace(0, 1, len(e)), lw=0.8, alpha=0.7)
        ax.set_xlim(np.quantile(eps, [0.005, 0.995]))
        ax.set_title(f"{arch}: residual ECDF collapse ({nm.split('_')[-1]})")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"{nm}_{arch}.png"), dpi=150)

    # 06/07/09 closure maps, 10/11 fixed point, 13 risk
    for col, nm in (("beta_effect", "06_closure_beta_effect_heatmap"),
                    ("lambda_prime_delta", "07_lambda_prime_map"),
                    ("delta_closure_error", "09_delta_closure_error_heatmap")):
        piv = griddf.iloc[:-1].pivot(index="lam", columns="C", values=col)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(piv.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        fig.colorbar(im); ax.set_xlabel("C"); ax.set_ylabel("lambda")
        ax.set_title(f"{arch}: {col}")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"{nm}_{arch}.png"), dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, v in paths.items():
        arr = np.array(v)
        ax[0].plot(arr[:, 0], arr[:, 1], "o-", ms=3, label=k)
        ax[1].plot(arr[:, 1], "o-", ms=3)
    ax[0].set_xlabel("C"); ax[0].set_ylabel("lambda")
    ax[0].set_title(f"{arch}: fixed-point paths -> "
                    f"({Cs:.3f}, {Ls:.3f})")
    ax[0].legend(fontsize=6)
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("lambda_r")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"10_fixed_point_path_C_lambda_{arch}.png"),
                dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    names = list(risks)
    ax.bar(names, [risks[k]["risk"] for k in names])
    ax.set_yscale("log"); ax.set_title(f"{arch}: held-out risk R_n")
    plt.xticks(rotation=30, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"13_risk_by_iteration_{arch}.png"), dpi=150)

    print(json.dumps(report["acceptance"], indent=2))
    print(f"[s0closure] {arch}: lam0={lam0:.3f} E_D={lawA['E_D']:.4f} "
          f"closure ED-ok {report['B_closure']['frac_ED_ok']:.0%} "
          f"EDelta-ok {report['B_closure']['frac_EDelta_ok']:.0%} "
          f"fixed point ({Cs:.3f}, {Ls:.3f})")


if __name__ == "__main__":
    main()
