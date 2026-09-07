#!/usr/bin/env python
"""K_t distribution model-selection protocol -- numeric tables only.

Implements kt_model_selection_protocol.md on analysis/tokens_{arch}.parquet.
Outputs Tables 0-7 as markdown + json per (arch, split).

PRE-REGISTERED OPERATIONAL CHOICES (fixed before the full run; [DEV] marks
points where the protocol text was ambiguous or underspecified):
 P1  zero atom: tokens with D == 0.0 exactly removed before all fits; w0
     reported in the header. E[K] gates include the atom:
     E[K] = w0 + (1-w0)*E_cont[e^D].
 P2  tail axis: tails are power laws in K=e^D (right) / 1/K (left), i.e.
     exponential in D beyond u. Hill a_hat = 1/mean(|D|-u : |D|>u).
     M5 tails fit a GPD in D per side (nests the exponential at xi=0);
     E[K] tail integral diverges for xi>0 or (xi=0, beta>=1) -> inf.
 P3  M5 main threshold psi = 1% per side; Table 5 scans psi.
 P4  M4 u+- profiled over symmetric psi in {0.5,1,2,5}% by train NLL.
 P5  splits: primary "traj" = by-trajectory 80/20, seed 20260809;
     stationarity "time" = fit pos<=P80(pos), eval pos>P80. Time-split
     tables are point values only (CIs are trajectory-bootstrap objects).
 P6  bootstrap: 500 trajectory block-bootstrap reps for empirical CIs;
     GPD xi CIs: 200 reps over exceedances resampled by trajectory at
     fixed threshold u (standard POT practice).
 P7  empirical reference for T1/T2/T3 = full continuous sample (models
     have <=8 params, overfit negligible); T4 strictly on the eval split.
 P8  T3 definitions [DEV]: m*(eps,side) = unconditional side quantile
     (P(D>m*+)=eps; P(D<-m*-)=eps); C_n = E[max |D| over n tokens], model
     via integral of (1-F_|D|^n), empirical via disjoint contiguous
     blocks of n tokens in data order; phase stat = (1-eps)*sigma_L/eps,
     sigma_L = 1.4826*median(|D| : D<0) (model version by 1e6-draw MC).
 P9  T2 Markov column: right side exact bound e^-x (from E[K]=1); left
     side plug-in Ehat[1/K]*e^-x [DEV: no exact left-side bound exists].
 P10 M7 buckets: moe = p x margin_min_infer x pos quartiles (4x4x4);
     dense = p x pos octiles (8x8) [DEV: dense has no margin column].
     mu=0 fixed, sigma_b = 1.4826*median|D|_b. Normality per bucket: AD
     statistic vs case-0 5% critical value 2.492 for the rejection rate;
     the p-value column is the specified-parameter KS p [DEV: scipy has
     no case-0 AD p-value].
 P11 gates: gate1 PASS iff implied E[K] finite and in [0.9, 1.1];
     gate2 PASS iff 0 violations on the Markov k-grid.
 P12 M3 fitted on a 2M subsample; M6 EM on a 4M subsample, k in {2,3} by
     BIC on that subsample.
 P13 M5 body density for NLL: 2048-bin histogram with 0.5 pseudocount.
 P14 T1 cell = ln(Q_model / Q_emp); nonpositive model quantile -> nan.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import optimize, stats

ADIR = "/home/kzhao2/nobackup/autodelete/gap_measurement/analysis"
SEED = 20260809
Q_T1 = [1e-2, 1e-3, 1e-4, 1e-5]
T2_LEVELS = [0.50, 0.90, 0.99, 0.999, 0.9999, 0.99999]
MARKOV_K = [2.0, 5.0, 20.0, np.exp(3), np.exp(7), np.exp(13)]
EPS_GRID = [1e-3, 1e-2, 0.1]
N_GRID = [100, 1000, 10000]
PSI_SCAN = [0.002, 0.005, 0.01, 0.02, 0.05]
PSI_MAIN = 0.01
N_BOOT = 500
N_BOOT_GPD = 200


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- data ----
def load(arch, smoke=False):
    cols = ["traj_id", "pos", "logp_infer", "D"]
    if arch == "moe":
        cols.append("margin_min_infer")
    t = pd.read_parquet(f"{ADIR}/tokens_{arch}.parquet", columns=cols)
    if smoke:
        keep = t.traj_id.isin(t.traj_id.unique()[:600])
        t = t[keep]
    D_all = t["D"].to_numpy(np.float64)
    traj = t["traj_id"].to_numpy()
    pos = t["pos"].to_numpy(np.int64)
    p = np.exp(t["logp_infer"].to_numpy(np.float64))
    margin = t["margin_min_infer"].to_numpy(np.float64) if arch == "moe" else None
    nz = D_all != 0.0
    w0 = 1.0 - nz.mean()
    return dict(D_all=D_all, traj=traj, pos=pos, p=p, margin=margin,
                nz=nz, w0=w0)


def make_split(d, kind):
    if kind == "traj":
        uniq = np.unique(d["traj"])
        rng = np.random.default_rng(SEED)
        rng.shuffle(uniq)
        m = np.isin(d["traj"], uniq[: int(0.8 * len(uniq))])
    else:  # time / stationarity split
        m = d["pos"] <= np.quantile(d["pos"], 0.8)
    return m


# ------------------------------------------------------------- helpers ----
def sq(D, q, side):
    """Unconditional side quantile: P(D > sq+)=q; P(D < -sq-)=q."""
    return float(np.quantile(D, 1 - q)) if side == "+" else float(-np.quantile(D, q))


def hill(D, u, side):
    exc = (D[D > u] - u) if side == "+" else (-D[D < -u] - u)
    return 1.0 / max(exc.mean(), 1e-300), len(exc)


def sigma_L(D):
    neg = D[D < 0]
    return 1.4826 * np.median(np.abs(neg)) if len(neg) else np.nan


def block_max_mean(absD, n):
    m = (len(absD) // n) * n
    if m == 0:
        return np.nan
    return float(absD[:m].reshape(-1, n).max(axis=1).mean())


def ad_stat(x, cdf):
    """Anderson-Darling A^2 against a fully specified cdf."""
    n = len(x)
    z = np.clip(cdf(np.sort(x)), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(z) + np.log1p(-z[::-1]))))


# -------------------------------------------------------------- models ----
class Model:
    name = "?"
    n_params = 0
    # subclasses: logpdf(x), sf(x)->P(X>x), cdf(x), implied_EK_cont()

    def cdf(self, x):
        return 1.0 - self.sf(x)

    def ppf_upper(self, q):
        """x with P(X > x) = q."""
        lo, hi = -1.0, 1.0
        while self.sf(hi) > q:
            hi *= 2
            if hi > 1e6:
                return np.nan
        while self.sf(lo) < q:
            lo *= 2
            if lo < -1e6:
                return np.nan
        return optimize.brentq(lambda v: self.sf(v) - q, lo, hi, xtol=1e-12)

    def quantile_side(self, q, side):
        return self.ppf_upper(q) if side == "+" else -self.ppf_upper(1 - q)

    def sf_abs(self, x):
        return self.sf(x) + self.cdf(-x)

    def emax_abs(self, n, xmax):
        g = np.concatenate([[0.0], np.geomspace(1e-8, max(xmax, 1.0) * 3, 3000)])
        with np.errstate(divide="ignore"):
            lf = np.log1p(-np.clip(self.sf_abs(g), 0, 1 - 1e-16))
        return float(np.trapz(-np.expm1(n * lf), g))

    def sigma_L_model(self, rng=None):
        # exact: median of |D| given D<0 solves cdf(-m) = cdf(0)/2
        c0 = float(self.cdf(0.0))
        x = self.ppf_upper(1 - c0 / 2)
        return 1.4826 * abs(x)


class M1Gauss(Model):
    name, n_params = "M1_gauss", 2

    def fit(self, D, aux):
        self.mu, self.s = float(D.mean()), float(D.std())

    def logpdf(self, x):
        return stats.norm.logpdf(x, self.mu, self.s)

    def sf(self, x):
        return stats.norm.sf(x, self.mu, self.s)

    def ppf_upper(self, q):
        return float(stats.norm.isf(q, self.mu, self.s))

    def cdf(self, x):
        return stats.norm.cdf(x, self.mu, self.s)

    def implied_EK_cont(self):
        return float(np.exp(self.mu + self.s ** 2 / 2))

    def params(self):
        return dict(mu=self.mu, sigma=self.s)


class M2Laplace(Model):
    name, n_params = "M2_laplace", 2

    def fit(self, D, aux):
        self.mu = float(np.median(D))
        self.b = float(np.mean(np.abs(D - self.mu)))

    def logpdf(self, x):
        return stats.laplace.logpdf(x, self.mu, self.b)

    def sf(self, x):
        return stats.laplace.sf(x, self.mu, self.b)

    def ppf_upper(self, q):
        return float(stats.laplace.isf(q, self.mu, self.b))

    def cdf(self, x):
        return stats.laplace.cdf(x, self.mu, self.b)

    def implied_EK_cont(self):
        return float(np.exp(self.mu) / (1 - self.b ** 2)) if self.b < 1 else np.inf

    def params(self):
        return dict(mu=self.mu, b=self.b)


class M3StudentT(Model):
    name, n_params = "M3_student_t", 3

    def fit(self, D, aux):
        rng = np.random.default_rng(SEED)
        sub = rng.choice(D, min(2_000_000, len(D)), replace=False)
        self.df, self.mu, self.s = stats.t.fit(sub)
        # asymptotic SE of df via numeric hessian on the subsample
        def nll(th):
            return -stats.t.logpdf(sub, th[0], th[1], th[2]).sum()
        h = 1e-3 * max(self.df, 1.0)
        f0 = nll([self.df, self.mu, self.s])
        fp = nll([self.df + h, self.mu, self.s])
        fm = nll([self.df - h, self.mu, self.s])
        d2 = max((fp - 2 * f0 + fm) / h ** 2, 1e-12)
        self.df_se = 1.0 / np.sqrt(d2)

    def logpdf(self, x):
        return stats.t.logpdf(x, self.df, self.mu, self.s)

    def sf(self, x):
        return stats.t.sf(x, self.df, self.mu, self.s)

    def ppf_upper(self, q):
        return float(stats.t.isf(q, self.df, self.mu, self.s))

    def cdf(self, x):
        return stats.t.cdf(x, self.df, self.mu, self.s)

    def implied_EK_cont(self):
        return np.inf  # polynomial D-tail => E[e^D] diverges for any finite nu

    def params(self):
        return dict(nu=self.df, nu_se=self.df_se, mu=self.mu, sigma=self.s)


class M4TruncTPareto(Model):
    """Truncated-t body on (-um, up) + exponential-in-D (Pareto-in-K) tails."""
    name, n_params = "M4_t_pareto", 6

    def fit(self, D, aux):
        best = None
        for psi in [0.005, 0.01, 0.02, 0.05]:
            up, um = sq(D, psi, "+"), sq(D, psi, "-")
            if up <= 0 or um <= 0:
                continue
            body = D[(D > -um) & (D < up)]
            ap, _ = hill(D, up, "+")
            am, _ = hill(D, um, "-")
            eb = max(len(body), 1)

            def nll_body(th):
                s, nu = np.exp(th)
                nu = min(max(nu, 0.05), 200.0)  # optimizer guard [DEV]
                z = stats.t.logpdf(body, nu, 0.0, s)
                mass = stats.t.cdf(up / s, nu) - stats.t.cdf(-um / s, nu)
                return -(z.sum() - eb * np.log(max(mass, 1e-300)))

            r = optimize.minimize(nll_body, [np.log(np.std(body) + 1e-9), np.log(3.0)],
                                  method="Nelder-Mead",
                                  options=dict(xatol=1e-4, fatol=1.0, maxiter=300))
            s, nu = np.exp(r.x)
            nu = min(max(nu, 0.05), 200.0)
            wb = 1 - 2 * psi
            nll_tot = (r.fun - eb * np.log(wb)
                       - (D > up).sum() * np.log(psi * ap) + ap * ((D[D > up] - up).sum())
                       - (D < -um).sum() * np.log(psi * am) + am * ((-D[D < -um] - um).sum()))
            if best is None or nll_tot < best[0]:
                best = (nll_tot, psi, up, um, s, nu, ap, am)
        _, self.psi, self.up, self.um, self.s, self.nu, self.ap, self.am = best
        self.wb = 1 - 2 * self.psi
        self.mass = stats.t.cdf(self.up / self.s, self.nu) - stats.t.cdf(-self.um / self.s, self.nu)
        # asymptotic SE of nu (body hessian, numeric)
        self.nu_se = np.nan

    def logpdf(self, x):
        x = np.asarray(x, float)
        out = np.full(x.shape, -np.inf)
        b = (x > -self.um) & (x < self.up)
        out[b] = (np.log(self.wb) + stats.t.logpdf(x[b], self.nu, 0, self.s)
                  - np.log(self.mass))
        r = x >= self.up
        out[r] = np.log(self.psi * self.ap) - self.ap * (x[r] - self.up)
        l = x <= -self.um
        out[l] = np.log(self.psi * self.am) - self.am * (-x[l] - self.um)
        return out

    def sf(self, x):
        x = np.asarray(x, float)
        out = np.empty(x.shape)
        r = x >= self.up
        out[r] = self.psi * np.exp(-self.ap * (x[r] - self.up))
        l = x <= -self.um
        out[l] = 1 - self.psi * np.exp(-self.am * (-x[l] - self.um))
        b = ~r & ~l
        tb = (stats.t.cdf(x[b] / self.s, self.nu) - stats.t.cdf(-self.um / self.s, self.nu))
        out[b] = 1 - self.psi - self.wb * tb / self.mass
        return out if out.shape else float(out)

    def cdf(self, x):
        x = np.asarray(x, float)
        out = np.empty(x.shape)
        l = x <= -self.um
        out[l] = self.psi * np.exp(-self.am * (-x[l] - self.um))
        r = x >= self.up
        out[r] = 1 - self.psi * np.exp(-self.ap * (x[r] - self.up))
        b = ~r & ~l
        tb = (stats.t.cdf(x[b] / self.s, self.nu) - stats.t.cdf(-self.um / self.s, self.nu))
        out[b] = self.psi + self.wb * tb / self.mass
        return out if out.shape else float(out)

    def ppf_upper(self, q):
        if q <= self.psi:
            return self.up + np.log(self.psi / q) / self.ap
        if q >= 1 - self.psi:
            return -(self.um + np.log(self.psi / (1 - q)) / self.am)
        target = 1 - q  # cdf value in body
        f = (target - self.psi) / self.wb * self.mass + stats.t.cdf(-self.um / self.s, self.nu)
        return float(self.s * stats.t.ppf(f, self.nu))

    def implied_EK_cont(self):
        # quantile-space body integral: robust to the near-degenerate spike
        lo = stats.t.cdf(-self.um / self.s, self.nu)
        hi = stats.t.cdf(self.up / self.s, self.nu)
        c = np.linspace(1e-9, 1 - 1e-9, 2_000_001)
        x = self.s * stats.t.ppf(lo + c * (hi - lo), self.nu)
        body = self.wb * np.trapz(np.exp(x), c)
        right = (self.psi * self.ap * np.exp(self.up) / (self.ap - 1)
                 if self.ap > 1 else np.inf)
        left = self.psi * self.am * np.exp(-self.um) / (self.am + 1)
        return float(body + right + left)

    def params(self):
        return dict(psi=self.psi, u_plus=self.up, u_minus=self.um, sigma=self.s,
                    nu=self.nu, a_plus=self.ap, a_minus=self.am)


class M5EmpGPD(Model):
    """Empirical histogram body + two-sided GPD tails in D (POT, psi=1%)."""
    name, n_params = "M5_emp_gpd", 6

    def fit(self, D, aux):
        self.psi = PSI_MAIN
        self.up, self.um = sq(D, self.psi, "+"), sq(D, self.psi, "-")
        excp = D[D > self.up] - self.up
        excm = -D[D < -self.um] - self.um
        self.xip, _, self.bp = stats.genpareto.fit(excp, floc=0)
        self.xim, _, self.bm = stats.genpareto.fit(excm, floc=0)
        body = D[(D >= -self.um) & (D <= self.up)]
        self.wb = 1 - 2 * self.psi
        self.edges = np.linspace(-self.um, self.up, 2049)
        cnt, _ = np.histogram(body, bins=self.edges)
        prob = (cnt + 0.5) / (cnt.sum() + 0.5 * len(cnt))
        self.binp = prob
        self.bincdf = np.concatenate([[0.0], np.cumsum(prob)])
        self.binw = self.edges[1] - self.edges[0]
        self.body_mean_eK = float(np.exp(body).mean())

    def logpdf(self, x):
        x = np.asarray(x, float)
        out = np.full(x.shape, -np.inf)
        r = x > self.up
        out[r] = (np.log(self.psi) +
                  stats.genpareto.logpdf(x[r] - self.up, self.xip, 0, self.bp))
        l = x < -self.um
        out[l] = (np.log(self.psi) +
                  stats.genpareto.logpdf(-x[l] - self.um, self.xim, 0, self.bm))
        b = ~r & ~l
        idx = np.clip(((x[b] - self.edges[0]) / self.binw).astype(int), 0, 2047)
        out[b] = np.log(self.wb * self.binp[idx] / self.binw)
        return out

    def sf(self, x):
        x = np.asarray(x, float)
        out = np.empty(x.shape)
        r = x > self.up
        out[r] = self.psi * stats.genpareto.sf(x[r] - self.up, self.xip, 0, self.bp)
        l = x < -self.um
        out[l] = 1 - self.psi * stats.genpareto.sf(-x[l] - self.um, self.xim, 0, self.bm)
        b = ~r & ~l
        fr = np.clip((x[b] - self.edges[0]) / self.binw, 0, 2048)
        i0 = np.clip(fr.astype(int), 0, 2047)
        cdfb = self.bincdf[i0] + self.binp[i0] * (fr - i0)
        out[b] = 1 - self.psi - self.wb * np.clip(cdfb, 0, 1)
        return out if out.shape else float(out)

    def cdf(self, x):
        x = np.asarray(x, float)
        out = np.empty(x.shape)
        l = x < -self.um
        out[l] = self.psi * stats.genpareto.sf(-x[l] - self.um, self.xim, 0, self.bm)
        r = x > self.up
        out[r] = 1 - self.psi * stats.genpareto.sf(x[r] - self.up, self.xip, 0, self.bp)
        b = ~r & ~l
        fr = np.clip((x[b] - self.edges[0]) / self.binw, 0, 2048)
        i0 = np.clip(fr.astype(int), 0, 2047)
        cdfb = self.bincdf[i0] + self.binp[i0] * (fr - i0)
        out[b] = self.psi + self.wb * np.clip(cdfb, 0, 1)
        return out if out.shape else float(out)

    def ppf_upper(self, q):
        if q <= self.psi:
            return self.up + float(stats.genpareto.isf(min(q / self.psi, 1.0),
                                                       self.xip, 0, self.bp))
        if q >= 1 - self.psi:
            return -(self.um + float(stats.genpareto.isf(min((1 - q) / self.psi, 1.0),
                                                         self.xim, 0, self.bm)))
        c = (1 - q - self.psi) / self.wb
        j = np.searchsorted(self.bincdf, c) - 1
        j = np.clip(j, 0, 2047)
        frac = (c - self.bincdf[j]) / max(self.binp[j], 1e-300)
        return float(self.edges[0] + (j + frac) * self.binw)

    def implied_EK_cont(self):
        if self.xip > 0 or (abs(self.xip) < 1e-9 and self.bp >= 1):
            right = np.inf
        else:
            g = np.geomspace(1e-9, 200 * self.bp, 20000)
            right = self.psi * np.exp(self.up) * np.trapz(
                np.exp(g) * stats.genpareto.pdf(g, self.xip, 0, self.bp), g)
        g = np.geomspace(1e-9, 2000 * self.bm, 20000)
        left = self.psi * np.exp(-self.um) * np.trapz(
            np.exp(-g) * stats.genpareto.pdf(g, self.xim, 0, self.bm), g)
        return float(self.wb / max(self.wb, 1e-12) * self.body_mean_eK * self.wb
                     + right + left)

    def params(self):
        return dict(psi=self.psi, u_plus=self.up, u_minus=self.um,
                    xi_plus=self.xip, beta_plus=self.bp,
                    xi_minus=self.xim, beta_minus=self.bm)


class M6GaussMix(Model):
    name, n_params = "M6_gmm", 8

    def fit(self, D, aux):
        from sklearn.mixture import GaussianMixture
        rng = np.random.default_rng(SEED)
        sub = rng.choice(D, min(4_000_000, len(D)), replace=False).reshape(-1, 1)
        best = None
        for k in (2, 3):
            gm = GaussianMixture(k, covariance_type="full", random_state=SEED,
                                 max_iter=200, n_init=1).fit(sub)
            bic = gm.bic(sub)
            if best is None or bic < best[0]:
                best = (bic, k, gm)
        _, self.k, gm = best
        self.n_params = 3 * self.k - 1
        self.w = gm.weights_.ravel()
        self.mu = gm.means_.ravel()
        self.sd = np.sqrt(gm.covariances_.ravel())

    def logpdf(self, x):
        x = np.asarray(x, float)[..., None]
        lp = stats.norm.logpdf(x, self.mu, self.sd) + np.log(self.w)
        return np.squeeze(np.logaddexp.reduce(lp, axis=-1))

    def sf(self, x):
        x = np.asarray(x, float)[..., None]
        return np.squeeze((stats.norm.sf(x, self.mu, self.sd) * self.w).sum(-1))

    def cdf(self, x):
        x = np.asarray(x, float)[..., None]
        return np.squeeze((stats.norm.cdf(x, self.mu, self.sd) * self.w).sum(-1))

    def implied_EK_cont(self):
        return float((self.w * np.exp(self.mu + self.sd ** 2 / 2)).sum())

    def params(self):
        return dict(k=self.k, w=self.w.tolist(), mu=self.mu.tolist(),
                    sigma=self.sd.tolist())


class M7CondVar(Model):
    """Scale-mixture: D|bucket ~ N(0, sigma_b^2); buckets from (p, margin, pos)."""
    name = "M7_condvar"

    def fit(self, D, aux):
        p, pos, margin = aux["p"], aux["pos"], aux["margin"]
        if margin is not None:
            qs4 = [0.25, 0.5, 0.75]
            self.feat_edges = [np.quantile(p, qs4), np.quantile(margin, qs4),
                               np.quantile(pos, qs4)]
            feats = [p, margin, pos]
            dims = [4, 4, 4]
        else:
            qs8 = np.arange(1, 8) / 8
            self.feat_edges = [np.quantile(p, qs8), np.quantile(pos, qs8)]
            feats = [p, pos]
            dims = [8, 8]
        self.dims = dims
        idx = np.zeros(len(D), int)
        for f, e, m in zip(feats, self.feat_edges, np.cumprod([1] + dims[:-1])):
            idx += np.searchsorted(e, f) * m
        self.nb = int(np.prod(dims))
        self.sig = np.zeros(self.nb)
        self.w = np.zeros(self.nb)
        self.bucket_meanp = np.zeros(self.nb)
        for b in range(self.nb):
            m = idx == b
            self.w[b] = m.mean()
            if m.sum() > 10:
                self.sig[b] = max(1.4826 * np.median(np.abs(D[m])), 1e-8)
                self.bucket_meanp[b] = p[m].mean()
            else:
                self.sig[b] = 1e-8
        self.n_params = self.nb
        self._idx_fit = idx
        self._D_fit = D

    def bucket_of(self, aux):
        feats = ([aux["p"], aux["margin"], aux["pos"]] if aux["margin"] is not None
                 else [aux["p"], aux["pos"]])
        idx = np.zeros(len(feats[0]), int)
        for f, e, m in zip(feats, self.feat_edges,
                           np.cumprod([1] + self.dims[:-1])):
            idx += np.searchsorted(e, f) * m
        return idx

    def logpdf_cond(self, x, idx):
        return stats.norm.logpdf(x, 0.0, self.sig[idx])

    def logpdf(self, x):  # marginal mixture (for tables)
        x = np.asarray(x, float)[..., None]
        lp = stats.norm.logpdf(x, 0.0, self.sig) + np.log(np.maximum(self.w, 1e-300))
        return np.squeeze(np.logaddexp.reduce(lp, axis=-1))

    def sf(self, x):
        x = np.asarray(x, float)[..., None]
        return np.squeeze((stats.norm.sf(x, 0.0, self.sig) * self.w).sum(-1))

    def cdf(self, x):
        x = np.asarray(x, float)[..., None]
        return np.squeeze((stats.norm.cdf(x, 0.0, self.sig) * self.w).sum(-1))

    def implied_EK_cont(self):
        return float((self.w * np.exp(self.sig ** 2 / 2)).sum())

    def params(self):
        return dict(n_buckets=self.nb, dims=self.dims)


# -------------------------------------------------------------- bootstrap --
def traj_bootstrap(d, n_reps, seed=SEED):
    """Yield index arrays resampling whole trajectories with replacement."""
    traj = d["traj"]
    order = np.argsort(traj, kind="stable")
    st = traj[order]
    bounds = np.flatnonzero(np.concatenate([[True], st[1:] != st[:-1], [True]]))
    starts, stops = bounds[:-1], bounds[1:]
    ntr = len(starts)
    rng = np.random.default_rng(seed)
    for _ in range(n_reps):
        pick = rng.integers(0, ntr, ntr)
        yield order[np.concatenate([np.arange(starts[i], stops[i]) for i in pick])]


def empirical_boot_stats(d):
    """One pass of trajectory bootstrap -> dict of stat arrays (n_reps, ...)."""
    D_all, nz = d["D_all"], d["nz"]
    recs = []
    for idx in traj_bootstrap(d, N_BOOT):
        Db_all = D_all[idx]
        Db = Db_all[Db_all != 0.0]
        row = [np.exp(Db_all).mean(), np.exp(-Db_all).mean()]
        for q in Q_T1 + EPS_GRID:
            row += [sq(Db, q, "+"), sq(Db, q, "-")]
        row.append(sigma_L(Db))
        ustar = np.quantile(np.abs(Db), 0.99)
        row += [(Db > ustar).mean(), (Db < -ustar).mean()]
        for psi in PSI_SCAN:
            for side in "+-":
                u = sq(Db, psi, side)
                row.append(hill(Db, u, side)[0])
        recs.append(row)
    return np.array(recs)


def ci(v, lo=2.5, hi=97.5):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [np.nan, np.nan]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


# ---------------------------------------------------------------- tables ---
def fmt_table(rows, cols, title, remark):
    out = [f"### {title}", "", "| " + " | ".join(cols) + " |",
           "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(
            x if isinstance(x, str) else
            ("inf" if np.isinf(x) else ("nan" if not np.isfinite(x) else f"{x:.4g}"))
            for x in r) + " |")
    out += ["", f"> {remark}", ""]
    return "\n".join(out)


def run(arch, split_kind, smoke=False):
    log(f"=== {arch} / split={split_kind} ===")
    d = load(arch, smoke)
    D_all = d["D_all"]
    fit_mask = make_split(d, split_kind)
    nzf = d["nz"] & fit_mask
    nze = d["nz"] & ~fit_mask
    D_fit = D_all[nzf]
    D_eval = D_all[nze]
    D_full = D_all[d["nz"]]
    aux_fit = dict(p=d["p"][nzf], pos=d["pos"][nzf],
                   margin=None if d["margin"] is None else d["margin"][nzf])
    aux_eval = dict(p=d["p"][nze], pos=d["pos"][nze],
                    margin=None if d["margin"] is None else d["margin"][nze])
    w0 = d["w0"]
    log(f"n_all={len(D_all)} w0={w0:.5f} n_fit={len(D_fit)} n_eval={len(D_eval)}")

    models = [M1Gauss(), M2Laplace(), M3StudentT(), M4TruncTPareto(),
              M5EmpGPD(), M6GaussMix(), M7CondVar()]
    for m in models:
        t0 = time.time()
        m.fit(D_fit, aux_fit)
        log(f"fit {m.name}: {time.time()-t0:.1f}s params={m.params()}")

    do_boot = split_kind == "traj"
    if do_boot:
        t0 = time.time()
        B = empirical_boot_stats(d)
        log(f"bootstrap {N_BOOT} reps: {time.time()-t0:.1f}s")
    ncol = 2 + 2 * len(Q_T1 + EPS_GRID) + 1 + 2 + 2 * len(PSI_SCAN)

    md = [f"# K_t model selection -- {arch}, split={split_kind}",
          f"(w0 = {w0:.5f} zero-atom removed; n_cont = {len(D_full)}; "
          f"fit/eval = {len(D_fit)}/{len(D_eval)}; "
          f"bootstrap = {'500 traj reps' if do_boot else 'n/a (time split)'})", ""]
    J = dict(arch=arch, split=split_kind, w0=w0,
             n=dict(all=len(D_all), cont=len(D_full), fit=len(D_fit),
                    eval=len(D_eval)),
             models={m.name: m.params() for m in models})

    # ---- Table 0: qualification audit
    EK_emp = w0 + (1 - w0) * float(np.exp(D_full).mean())
    EK_ci = ci(B[:, 0]) if do_boot else [np.nan, np.nan]
    rows0 = []
    for m in models:
        ek_c = m.implied_EK_cont()
        ek = w0 + (1 - w0) * ek_c if np.isfinite(ek_c) else np.inf
        g1 = "PASS" if (np.isfinite(ek) and 0.9 <= ek <= 1.1) else "FAIL"
        viol = 0
        for k in MARKOV_K:
            pk = (1 - w0) * float(m.sf(np.log(k)))
            if pk > 1.0 / k:
                viol += 1
        g2 = "PASS" if viol == 0 else "FAIL"
        rows0.append([m.name, ek, f"[{EK_ci[0]:.4f},{EK_ci[1]:.4f}]", g1, viol, g2])
        J.setdefault("table0", []).append(dict(model=m.name, implied_EK=ek,
                                               gate1=g1, markov_viol=viol, gate2=g2))
    J["EK_empirical"] = dict(point=EK_emp, ci=EK_ci)
    md.append(fmt_table(rows0,
        ["model", "implied_E[K]", "E[K]_emp_CI", "gate1", "markov_viol", "gate2"],
        f"Table 0 -- qualification audit (empirical E[K]={EK_emp:.4f})",
        "gate1: implied E[K] finite and in [0.9,1.1]; gate2: 0 Markov violations "
        f"on k grid {['%.3g' % k for k in MARKOV_K]}."))

    # ---- Table 1: tail quantile calibration
    rows1 = []
    cells_emp = {}
    for j, q in enumerate(Q_T1):
        for s, side in enumerate("+-"):
            qe = sq(D_full, q, side)
            col = 2 + 2 * j + s
            w = (np.log(ci(B[:, col])[1] / max(ci(B[:, col])[0], 1e-300))
                 if do_boot else np.nan)
            cells_emp[(q, side)] = (qe, w)
    for m in models:
        r = [m.name]
        errs = []
        for q in Q_T1:
            for side in "+-":
                qm = m.quantile_side(q, side)
                qe, _ = cells_emp[(q, side)]
                e = np.log(qm / qe) if (qm and qm > 0 and qe > 0) else np.nan
                errs.append(e)
                r.append(e)
        r.append(np.nanmean(np.abs(errs)))
        rows1.append(r)
        J.setdefault("table1", []).append(dict(model=m.name, err=errs,
                                               mean_abs=float(np.nanmean(np.abs(errs)))))
    cols1 = ["model"] + [f"{s}{q:g}" for q in Q_T1 for s in "+-"] + ["mean|err|"]
    emp_w = " ".join(f"{s}{q:g}:w={cells_emp[(q,s)][1]:.3f}" for q in Q_T1 for s in "+-")
    md.append(fmt_table(rows1, cols1,
        "Table 1 -- tail quantile calibration, cell = ln(Q_model/Q_emp)",
        f"empirical bootstrap log-widths: {emp_w}"))

    # ---- Table 2: exceedance probability comparison
    Dp = D_full[D_full > 0]
    Dm = -D_full[D_full < 0]
    Einv = float(np.exp(-D_full).mean())
    rows2 = []
    for side, arr in (("+", Dp), ("-", Dm)):
        for lev in T2_LEVELS:
            x = float(np.quantile(arr, lev))
            emp = (arr > x).mean() * len(arr) / len(D_full)
            r = [f"{side}{lev:g}", x, np.log10(max(emp, 1e-300))]
            for m in models:
                pm = m.sf(x) if side == "+" else m.cdf(-x)
                r.append(np.log10(max(float(pm), 1e-300)))
            bound = np.exp(-x) if side == "+" else Einv * np.exp(-x)
            r.append(np.log10(bound))
            rows2.append(r)
            J.setdefault("table2", []).append(dict(side=side, level=lev, x=x,
                log10_emp=r[2], log10_models=r[3:-1], log10_markov=r[-1]))
    md.append(fmt_table(rows2,
        ["side_lev", "x", "log10_emp"] + [m.name for m in models] + ["log10_markov"],
        "Table 2 -- CCDF at empirical side-|D| levels (log10)",
        "markov: right side exact e^-x from E[K]=1; left side plug-in "
        f"Ehat[1/K]*e^-x with Ehat[1/K]={Einv:.4f} [P9]."))

    # ---- Table 3: downstream threshold errors
    absD = np.abs(D_full)
    xmax = float(absD.max())
    rng = np.random.default_rng(SEED + 1)
    rows3 = []
    sL_emp = sigma_L(D_full)
    emp_cn = {n: block_max_mean(absD, n) for n in N_GRID}
    for m in models:
        r = [m.name]
        rel = []
        for q in EPS_GRID:
            for side in "+-":
                qe = sq(D_full, q, side)
                qm = m.quantile_side(q, side)
                e = abs(qm - qe) / abs(qe) if qe else np.nan
                rel.append(e)
                r.append(e)
        for n in N_GRID:
            cm = m.emax_abs(n, xmax)
            e = abs(cm - emp_cn[n]) / emp_cn[n]
            rel.append(e)
            r.append(e)
        sL_m = m.sigma_L_model(rng)
        for q in EPS_GRID:
            pe = (1 - q) * sL_emp / q
            pm = (1 - q) * sL_m / q
            e = abs(pm - pe) / pe
            rel.append(e)
            r.append(e)
        r.append(float(np.nanmean(rel)))
        rows3.append(r)
        J.setdefault("table3", []).append(dict(model=m.name, rel_err=rel,
                                               mean=float(np.nanmean(rel))))
    cols3 = (["model"] + [f"m*{s}{q:g}" for q in EPS_GRID for s in "+-"]
             + [f"C_{n}" for n in N_GRID] + [f"ph{q:g}" for q in EPS_GRID]
             + ["mean|rel|"])
    md.append(fmt_table(rows3, cols3,
        "Table 3 -- downstream threshold relative errors (PRIMARY)",
        f"emp: sigma_L={sL_emp:.4g}, C_n={ {n: round(emp_cn[n],4) for n in N_GRID} }; "
        "defs per P8."))

    # ---- Table 4: overall fit
    rows4 = []
    qq_c = np.linspace(0.25, 0.75, 21)
    qq_t = np.linspace(0.99, 0.99999, 21)
    eq_c = np.quantile(D_eval, qq_c)
    eq_t = np.quantile(D_eval, qq_t)
    idx_eval = None
    for m in models:
        if isinstance(m, M7CondVar):
            idx_eval = m.bucket_of(aux_eval)
            lp = m.logpdf_cond(D_eval, idx_eval)
            lp_fit = m.logpdf_cond(D_fit, m._idx_fit)
        else:
            lp = m.logpdf(D_eval)
            lp_fit = m.logpdf(D_fit)
        nll = -float(np.mean(lp))
        bic = m.n_params * np.log(len(D_fit)) - 2 * float(np.sum(lp_fit))
        sub = D_eval[:: max(1, len(D_eval) // 500000)]
        pit = np.clip(m.cdf(np.sort(sub)), 0, 1)
        ks = float(np.max(np.abs(pit - np.linspace(0, 1, len(pit)))))
        mq_c = np.array([m.quantile_side(1 - q, "+") for q in qq_c])
        mq_t = np.array([m.quantile_side(1 - q, "+") for q in qq_t])
        sl_c = float(np.polyfit(eq_c, mq_c, 1)[0])
        sl_t = float(np.polyfit(eq_t, mq_t, 1)[0])
        rows4.append([m.name, nll, bic, ks, sl_c, sl_t])
        J.setdefault("table4", []).append(dict(model=m.name, nll=nll, bic=bic,
                                               pit_ks=ks, qq_central=sl_c,
                                               qq_tail=sl_t))
    md.append(fmt_table(rows4,
        ["model", "eval_meanNLL", "BIC", "PIT_KS", "QQ_slope_c", "QQ_slope_t"],
        "Table 4 -- overall fit (eval split)",
        "M7 NLL is conditional (per-bucket); others marginal. QQ slopes: "
        "model-vs-empirical quantiles, central q in [.25,.75], tail q>.99."))

    # ---- Table 5: Hill / GPD plateau scan (M5 exclusive)
    rows5 = []
    plateau_pairs = 0
    prev_ci = {}
    for pi, psi in enumerate(PSI_SCAN):
        r = [f"{psi:g}"]
        rec = dict(psi=psi)
        for s, side in enumerate("+-"):
            u = sq(D_full, psi, side)
            a_pt, nexc = hill(D_full, u, side)
            col = B[:, ncol - 2 * len(PSI_SCAN) + 2 * pi + s] if do_boot else None
            a_ci = ci(col) if do_boot else [np.nan, np.nan]
            exc = (D_full[D_full > u] - u) if side == "+" else (-D_full[D_full < -u] - u)
            xi, _, beta = stats.genpareto.fit(exc, floc=0)
            xis = []
            if do_boot:
                sub_traj = d["traj"][d["nz"]][(D_full > u) if side == "+"
                                              else (D_full < -u)]
                o = np.argsort(sub_traj, kind="stable")
                es, ts = exc[o], sub_traj[o]
                b2 = np.flatnonzero(np.concatenate([[True], ts[1:] != ts[:-1],
                                                    [True]]))
                st2, sp2 = b2[:-1], b2[1:]
                rng2 = np.random.default_rng(SEED + 7 + pi * 2 + s)
                for _ in range(N_BOOT_GPD):
                    pick = rng2.integers(0, len(st2), len(st2))
                    idxb = np.concatenate(
                        [np.arange(st2[i], sp2[i]) for i in pick])
                    if len(idxb) > 50:
                        xis.append(stats.genpareto.fit(es[idxb], floc=0)[0])
            xi_ci = ci(np.array(xis)) if xis else [np.nan, np.nan]
            r += [u, a_pt, f"[{a_ci[0]:.3g},{a_ci[1]:.3g}]", xi,
                  f"[{xi_ci[0]:.3g},{xi_ci[1]:.3g}]", nexc]
            rec[side] = dict(u=u, a=a_pt, a_ci=a_ci, xi=xi, xi_ci=xi_ci,
                             beta=beta, n_exc=int(nexc))
            if do_boot and side in prev_ci:
                lo1, hi1 = prev_ci[side]
                if not (a_ci[1] < lo1 or a_ci[0] > hi1):
                    plateau_pairs += 1
            prev_ci[side] = a_ci
        rows5.append(r)
        J.setdefault("table5", []).append(rec)
    n_adj = 2 * (len(PSI_SCAN) - 1)
    plateau = "PLATEAU" if plateau_pairs == n_adj else \
        f"PARTIAL({plateau_pairs}/{n_adj})"
    md.append(fmt_table(rows5,
        ["psi", "u+", "a+", "a+_CI", "xi+", "xi+_CI", "n+",
         "u-", "a-", "a-_CI", "xi-", "xi-_CI", "n-"],
        "Table 5 -- Hill/GPD threshold scan",
        f"plateau verdict (adjacent a CIs overlap, both sides): {plateau}."))
    J["table5_verdict"] = plateau

    # ---- Table 6: M7 ellipse verdict
    m7 = models[6]
    idx_full = m7.bucket_of(dict(p=d["p"][d["nz"]], pos=d["pos"][d["nz"]],
                                 margin=None if d["margin"] is None
                                 else d["margin"][d["nz"]]))
    rng3 = np.random.default_rng(SEED + 2)
    ad_rej, ks_ps, worst = 0, [], []
    used = 0
    for b in range(m7.nb):
        xb = D_full[idx_full == b]
        if len(xb) < 200:
            continue
        used += 1
        if len(xb) > 20000:
            xb = rng3.choice(xb, 20000, replace=False)
        sb = m7.sig[b]
        a2 = ad_stat(xb, lambda v: stats.norm.cdf(v, 0, sb))
        ksp = stats.kstest(xb, lambda v: stats.norm.cdf(v, 0, sb)).pvalue
        ks_ps.append(ksp)
        if a2 > 2.492:
            ad_rej += 1
        worst.append((a2, b))
    worst.sort(reverse=True)
    rej_rate = ad_rej / max(used, 1)
    rows6a = [[f"{rej_rate:.3f}", used, f"{np.median(ks_ps):.3g}",
               " ".join(f"b{b}:A2={a:.1f}" for a, b in worst[:5])]]
    md.append(fmt_table(rows6a, ["AD_rej@5%", "n_buckets_used", "median_KS_p",
                                 "worst5"],
        "Table 6a -- bucket normality",
        "AD case-0 critical 2.492 for rejection; p column is KS (P10)."))
    J["table6a"] = dict(ad_rej_rate=rej_rate, n_used=used,
                        median_ks_p=float(np.median(ks_ps)))

    ok = m7.w > 0
    mp = np.clip(m7.bucket_meanp[ok], 1e-6, 1 - 1e-6)
    sg = m7.sig[ok]
    wt = m7.w[ok]
    rows6b = []
    for nm, g in (("const", np.ones_like(mp)), ("c(1-p)", 1 - mp),
                  ("c*sqrt((1-p)/p)", np.sqrt((1 - mp) / mp))):
        c = float((wt * sg * g).sum() / (wt * g * g).sum())
        ssr = float((wt * (sg - c * g) ** 2).sum())
        sst = float((wt * (sg - (wt * sg).sum() / wt.sum()) ** 2).sum())
        r2 = 1 - ssr / max(sst, 1e-300)
        rows6b.append([nm, c, r2])
        J.setdefault("table6b", []).append(dict(shape=nm, c=c, r2=r2))
    md.append(fmt_table(rows6b, ["shape", "c_hat", "R2_weighted"],
        "Table 6b -- sigma(bucket) shape fits (P10 numeric)",
        f"sigma(bucket) range: [{sg.min():.2g}, {sg.max():.2g}], "
        f"64-bucket table in json."))
    J["table6b_sigma"] = m7.sig.tolist()

    def invgamma_nu(s2, w):
        def nll(th):
            a, b = np.exp(th)
            return -float((w * stats.invgamma.logpdf(s2, a, scale=b)).sum())
        r = optimize.minimize(nll, [np.log(2), np.log(np.median(s2))],
                              method="Nelder-Mead")
        return 2 * np.exp(r.x[0])
    s2 = sg ** 2
    nu_eq = invgamma_nu(s2, wt)
    nus = []
    rng4 = np.random.default_rng(SEED + 3)
    for _ in range(200):
        pick = rng4.integers(0, len(s2), len(s2))
        try:
            nus.append(invgamma_nu(s2[pick], wt[pick]))
        except Exception:
            pass
    nu_ci = ci(np.array(nus))
    m4nu = models[3].nu
    m5a = J["table5"][PSI_SCAN.index(PSI_MAIN)]
    compat_m4 = "compatible" if nu_ci[0] <= m4nu <= nu_ci[1] else "incompatible"
    rows6c = [[nu_eq, f"[{nu_ci[0]:.3g},{nu_ci[1]:.3g}]", m4nu, compat_m4,
               m5a["+"]["a"], m5a["-"]["a"]]]
    md.append(fmt_table(rows6c, ["nu_eq(invGamma)", "nu_eq_CI", "M4_nu",
                                 "M4_compat", "M5_a+", "M5_a-"],
        "Table 6c -- implied tail index cross-check",
        "nu_eq: inv-gamma MLE on bucket sigma^2 (scale-mixture => t_nu); "
        "M5 a on the K axis is a different tail family -- report side by side."))
    J["table6c"] = dict(nu_eq=nu_eq, nu_ci=nu_ci, m4_nu=m4nu, compat=compat_m4)

    # ---- Table 7: asymmetry dossier
    ustar = float(np.quantile(absD, 0.99))
    em_p = float((D_full > ustar).mean())
    em_m = float((D_full < -ustar).mean())
    ep_ci = ci(B[:, 2 + 2 * len(Q_T1 + EPS_GRID) + 1]) if do_boot else [np.nan] * 2
    em_ci = ci(B[:, 2 + 2 * len(Q_T1 + EPS_GRID) + 2]) if do_boot else [np.nan] * 2
    ratio = em_m / max(em_p, 1e-300)
    if do_boot:
        rcol = (B[:, 2 + 2 * len(Q_T1 + EPS_GRID) + 2]
                / np.maximum(B[:, 2 + 2 * len(Q_T1 + EPS_GRID) + 1], 1e-300))
        ratio_ci = ci(rcol)
    else:
        ratio_ci = [np.nan, np.nan]
    rows7 = []
    for side in "+-":
        rec = m5a[side]
        eps_side = em_p if side == "+" else em_m
        eci = ep_ci if side == "+" else em_ci
        rows7.append([side, eps_side, f"[{eci[0]:.3g},{eci[1]:.3g}]",
                      rec["a"], f"[{rec['a_ci'][0]:.3g},{rec['a_ci'][1]:.3g}]",
                      rec["xi"], f"[{rec['xi_ci'][0]:.3g},{rec['xi_ci'][1]:.3g}]",
                      rec["u"]])
    rows7.append(["-/+", ratio, f"[{ratio_ci[0]:.3g},{ratio_ci[1]:.3g}]",
                  "", "", "", "", ""])
    md.append(fmt_table(rows7,
        ["side", "eps_hat(u*)", "eps_CI", "a_hat", "a_CI", "xi_hat", "xi_CI", "u(psi=1%)"],
        f"Table 7 -- two-sided asymmetry dossier (u* = |D| q99 = {ustar:.4g})",
        "eps at common threshold u*; a/xi at per-side psi=1% thresholds."))
    J["table7"] = dict(u_star=ustar, eps_plus=em_p, eps_minus=em_m,
                       eps_plus_ci=ep_ci, eps_minus_ci=em_ci,
                       ratio=ratio, ratio_ci=ratio_ci)

    # ---- verdicts
    t3rank = sorted(J["table3"], key=lambda r: r["mean"])
    t1rank = sorted(J["table1"], key=lambda r: r["mean_abs"])
    passers = [r["model"] for r in J["table0"]
               if r["gate1"] == "PASS" and r["gate2"] == "PASS"]
    t4rank = sorted([r for r in J["table4"] if r["model"] in passers],
                    key=lambda r: r["nll"])
    J["verdict"] = dict(
        algorithm_champion=t3rank[0]["model"],
        algorithm_runner_up=t3rank[1]["model"],
        t1_best=t1rank[0]["model"],
        narrative_champion=t4rank[0]["model"] if t4rank else "NONE_PASSED",
        gate_passers=passers)
    md.append(f"### Verdict\n\n- algorithm champion (T3 primary): "
              f"**{J['verdict']['algorithm_champion']}** "
              f"(runner-up {J['verdict']['algorithm_runner_up']}; T1 best "
              f"{J['verdict']['t1_best']})\n- narrative champion (T4 NLL among "
              f"gate passers {passers}): **{J['verdict']['narrative_champion']}**\n"
              f"- ellipse verdict: AD_rej={rej_rate:.3f} "
              f"({'<' if rej_rate < 0.10 else '>='}0.10), 6c {compat_m4} => "
              f"{'ESTABLISHED' if rej_rate < 0.10 and compat_m4 == 'compatible' else 'NOT established'}\n")

    tag = "_smoke" if smoke else ""
    with open(f"{ADIR}/kt_{arch}_{split_kind}{tag}.md", "w") as f:
        f.write("\n".join(md))
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and not np.isfinite(o):
            return str(o)
        return o
    with open(f"{ADIR}/kt_{arch}_{split_kind}{tag}.json", "w") as f:
        json.dump(_clean(J), f, indent=1, default=str)
    log(f"wrote kt_{arch}_{split_kind}{tag}.md/.json  verdict={J['verdict']}")


if __name__ == "__main__":
    arch = sys.argv[1]
    smoke = "--smoke" in sys.argv
    if smoke:
        N_BOOT, N_BOOT_GPD = 30, 20
    splits = ["traj", "time"] if "--split-both" in sys.argv or not smoke else ["traj"]
    for sk in splits:
        run(arch, sk, smoke)
