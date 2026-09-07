#!/usr/bin/env python
"""A-tier audit E1-E6 (问题.md): martingale mean identity, standardized
refit, sqrt(pos) deconfounding, per-position identity + autocorrelation,
sequence-level numbers, folk-band audit. All CPU, existing token tables.

Pre-registered operational choices [DEV = where 问题.md was ambiguous]:
  - p = exp(logp_train) (pi_old side, KPop's variable); pos = generation
    step t_rel = pos - min(pos|traj) + 1.
  - sigma_b (bucket scale) = 1.4826 * MAD; sigma_b^2 its square.
  - CI on mu / r_b: plain normal SE = sd/sqrt(n) [DEV: no cluster corr].
  - E1b clean stratum: moe = buckets in bottom tercile of hard-flip rate;
    dense = all buckets. Weighted (n_b) OLS of mu_b on sigma_b^2.
  - E2 sigma_hat: c1*(1-p), with sqrt(t/t0) factor ONLY if E3 median
    within-decile slope >= 0.25 (protocol verdict rule); t0 = median t.
  - E2a tail freq at |x| > 4*sigma_MAD(x) for both coordinates [DEV].
    dBIC = BIC(k=1) - BIC(k=2), GMM on 2M subsample.
  - E2b/E6 s-band threshold z_psi = 4.
  - E4c pairs at lag tau via index shift within trajectories (positions
    are consecutive); stratum = whether the EARLIER token hard-flipped.
  - E6 derived band: mask iff k outside [0.72, 1.29]; the "+cap 2" note
    is reported but the audit treats the band as the mask set [DEV].
    Drift mass = sum(e^D - 1 - D) (nonneg compensator integrand);
    leak rate = unmasked share of that mass.

Usage: python analysis/martingale_audit.py {moe|dense}
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

LOCAL = "/home/kzhao2/gap_measurement/results"
OUT = "/home/kzhao2/nobackup/autodelete/gap_measurement/analysis"
Z_PSI = 4.0


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-300


def fmt(rows, cols, title, remark):
    o = [f"### {title}", "", "| " + " | ".join(cols) + " |",
         "|" + "---|" * len(cols)]
    for r in rows:
        o.append("| " + " | ".join(
            x if isinstance(x, str) else
            ("nan" if not np.isfinite(x) else f"{x:.4g}") for x in r) + " |")
    o += ["", f"> {remark}", ""]
    return "\n".join(o)


arch = sys.argv[1]
cols = ["traj_id", "pos", "D", "logp_train"]
if arch == "moe":
    cols.append("hard_flip")
t = pd.read_parquet(f"{LOCAL}/tokens_{arch}.parquet", columns=cols)
t = t.sort_values(["traj_id", "pos"], kind="stable").reset_index(drop=True)
D = t["D"].to_numpy(np.float64)
p = np.exp(t["logp_train"].to_numpy(np.float64))
traj = t["traj_id"].to_numpy()
trel = (t["pos"] - t.groupby("traj_id")["pos"].transform("min")).to_numpy() + 1
flip = (t["hard_flip"].to_numpy() if arch == "moe"
        else np.zeros(len(t), bool))
n = len(D)
log(f"{arch}: {n} tokens")
md = [f"# A-tier martingale audit -- {arch} ({n:,} tokens)\n"]
J = {"arch": arch, "n": n}

# ---------------- bucket structure: p deciles x pos quartiles ----------
pdec_e = np.quantile(p, np.linspace(0, 1, 11))[1:-1]
pseg_e = np.quantile(trel, [0.25, 0.5, 0.75])
pdec = np.searchsorted(pdec_e, p)
pseg = np.searchsorted(pseg_e, trel)
bucket = pdec * 4 + pseg

# ================= E1: mean identity + drift map =======================
rows, recs = [], []
for b in range(40):
    m = bucket == b
    nb = int(m.sum())
    if nb < 1000:
        continue
    mu = float(D[m].mean())
    s2 = float(mad_sigma(D[m]) ** 2)
    se = float(D[m].std() / np.sqrt(nb))
    r = mu + s2 / 2
    fr = float(flip[m].mean())
    recs.append(dict(b=b, pdec=b // 4, pseg=b % 4, n=nb, mu=mu, s2=s2,
                     r=r, lo=r - 1.96 * se, hi=r + 1.96 * se, fliprate=fr))
recs.sort(key=lambda x: x["r"])
neg_sig = [x for x in recs if x["hi"] < 0]
pos_sig = [x for x in recs if x["lo"] > 0]
for x in recs[:8] + recs[-3:]:
    rows.append([f"p{x['pdec']}xt{x['pseg']}", x["n"], x["mu"], x["s2"],
                 x["r"], f"[{x['lo']:.2e},{x['hi']:.2e}]", x["fliprate"]])
md.append(fmt(rows, ["bucket", "n", "mu_hat", "sigma2_hat", "r_b",
                     "r_b 95%CI", "flip_rate"],
    "E1a -- mean-identity residuals r_b = mu + sigma^2/2 "
    "(8 most negative + 3 most positive of 40)",
    f"sig r<0: {len(neg_sig)}/40, sig r>0: {len(pos_sig)}/40. "
    "Martingale prediction: all r<=0; drift map = sig-negative buckets."))
if neg_sig:
    fr_neg = np.mean([x["fliprate"] for x in neg_sig])
    fr_all = np.mean([x["fliprate"] for x in recs])
    md.append(f"drift-map overlap: mean flip-rate in sig-negative buckets "
              f"= {fr_neg:.3f} vs all-bucket mean {fr_all:.3f}\n")
J["E1a"] = recs

# E1b: clean-stratum regression mu ~ sigma^2
if arch == "moe":
    thr = np.quantile([x["fliprate"] for x in recs], 1 / 3)
    clean = [x for x in recs if x["fliprate"] <= thr]
else:
    clean = recs
x2 = np.array([c["s2"] for c in clean])
y = np.array([c["mu"] for c in clean])
w = np.array([c["n"] for c in clean], float)
X = np.vstack([np.ones_like(x2), x2]).T
W = np.diag(w / w.sum())
beta, res_, *_ = np.linalg.lstsq(np.sqrt(W) @ X, np.sqrt(W) @ y, rcond=None)
resid = y - X @ beta
s2b = float((w * resid ** 2).sum() / (w.sum() * (len(y) - 2) / len(y)))
covb = s2b * np.linalg.inv(X.T @ np.diag(w / w.sum()) @ X) / len(y)
slope, slo_se = float(beta[1]), float(np.sqrt(covb[1, 1]))
ok_half = slope - 1.96 * slo_se <= -0.5 <= slope + 1.96 * slo_se
md.append(fmt([[len(clean), slope, f"[{slope-1.96*slo_se:.3f},"
                f"{slope+1.96*slo_se:.3f}]",
                "PASS" if ok_half else "FAIL"]],
    ["n_clean_buckets", "slope", "95%CI", "contains -1/2"],
    "E1b -- clean-stratum regression mu_b ~ sigma_b^2",
    "Acceptance: CI contains -1/2 (lognormal-core mean law)."))
J["E1b"] = dict(slope=slope, se=slo_se, ok=bool(ok_half))

# ================= E3 first (its verdict feeds E2) =====================
rows3, slopes = [], []
for d in range(10):
    xs, ys = [], []
    for s in range(4):
        m = (pdec == d) & (pseg == s)
        if m.sum() < 2000:
            continue
        xs.append(np.log(np.median(trel[m])))
        ys.append(np.log(mad_sigma(D[m])))
    if len(xs) < 3:
        continue
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([np.ones_like(xs), xs]).T
    b_, *_ = np.linalg.lstsq(A, ys, rcond=None)
    r_ = ys - A @ b_
    se_ = float(np.sqrt((r_ ** 2).sum() / max(len(xs) - 2, 1)
                        / ((xs - xs.mean()) ** 2).sum()))
    slopes.append(float(b_[1]))
    rows3.append([f"p-decile {d}", float(b_[1]),
                  f"[{b_[1]-1.96*se_:.3f},{b_[1]+1.96*se_:.3f}]"])
med_slope = float(np.median(slopes))
use_pos = med_slope >= 0.25
md.append(fmt(rows3, ["p decile", "dlog(sigma)/dlog(pos)", "95%CI"],
    "E3 -- within-decile position slope (sqrt-law deconfounding)",
    f"median slope = {med_slope:.3f}; verdict: "
    f"{'~0.5 KV-bell REAL, keep sqrt(pos)' if use_pos else 'position term is an artifact of composition -> drop pos from sigma_hat'}."))
J["E3"] = dict(slopes=slopes, median=med_slope, use_pos=bool(use_pos))

# ================= E2: standardized refit ==============================
# c1 fit in p-bins
sax = np.clip(1 - p, 1e-7, 1)
be = np.unique(np.quantile(sax, np.linspace(0, 1, 33)))
bi = np.clip(np.searchsorted(be, sax) - 1, 0, len(be) - 2)
ls_, lg_ = [], []
for k in range(len(be) - 1):
    m = bi == k
    if m.sum() > 5000:
        ls_.append(np.log(mad_sigma(D[m])))
        lg_.append(np.log(np.median(sax[m])))
c1 = float(np.exp(np.mean(np.array(ls_) - np.array(lg_))))
t0 = float(np.median(trel))
# quantization floor: without it, p->1 tokens divide by ~1e-8 and their
# bf16-comb D values explode in s (kurtosis 4e6 observed) [DEV]
FLOOR = 3e-7 if arch == "moe" else 1e-6
sig_hat = np.maximum(c1 * sax, FLOOR)
if use_pos:
    sig_hat = sig_hat * np.sqrt(trel / t0)
s = (D + sig_hat ** 2 / 2) / sig_hat

def shape_stats(x, name):
    sm = mad_sigma(x)
    q999 = float(np.quantile(np.abs(x - np.median(x)), 0.999))
    kur = float(stats.kurtosis(np.random.default_rng(0).choice(
        x, min(4_000_000, len(x)), replace=False)))
    tail4 = float((np.abs(x - np.median(x)) > 4 * sm).mean())
    from sklearn.mixture import GaussianMixture
    sub = np.random.default_rng(1).choice(
        x, min(2_000_000, len(x)), replace=False).reshape(-1, 1)
    b1 = GaussianMixture(1, random_state=0).fit(sub).bic(sub)
    b2 = GaussianMixture(2, random_state=0, n_init=1).fit(sub).bic(sub)
    return [name, q999 / sm, kur, b1 - b2, tail4]

log("E2a shape stats...")
rows2 = [shape_stats(D, "raw D"), shape_stats(s, "standardized s")]
md.append(fmt(rows2, ["coord", "q999/MAD", "kurtosis", "dBIC(1-2)",
                      "P(|x|>4sigma)"],
    "E2a -- shape before/after standardization "
    f"(sigma_hat = c1(1-p){'*sqrt(t/t0)' if use_pos else ''}, c1={c1:.3f})",
    "fake-tail death = big drops in q999/MAD, kurtosis, tail freq."))
J["E2a"] = rows2

tail = np.abs(s - np.median(s)) > Z_PSI
rows2b = []
for side, m in (("+", tail & (s > 0)), ("-", tail & (s < 0))):
    if m.sum():
        rows2b.append([side, int(m.sum()), float(flip[m].mean()), 0.533])
md.append(fmt(rows2b, ["side", "n_tail", "hard_flip_share", "baseline"],
    f"E2b -- residual tail |s|>{Z_PSI:g} composition",
    "share >> 0.533 => residual tail is the true (jump) branch."))
J["E2b"] = rows2b

rows2c = []
for side in ("+", "-"):
    v = s if side == "+" else -s
    u = np.quantile(v, 0.99)
    exc = v[v > u] - u
    xi, _, bet = stats.genpareto.fit(exc, floc=0)
    o = np.argsort(traj[v > u], kind="stable")
    es_ = exc[o]
    ts_ = traj[v > u][o]
    bnd = np.flatnonzero(np.r_[True, ts_[1:] != ts_[:-1], True])
    st_, sp_ = bnd[:-1], bnd[1:]
    rng = np.random.default_rng(11)
    xis = []
    for _ in range(200):
        pick = rng.integers(0, len(st_), len(st_))
        idx = np.concatenate([np.arange(st_[i], sp_[i]) for i in pick])
        if len(idx) > 50:
            xis.append(stats.genpareto.fit(es_[idx], floc=0)[0])
    lo, hi = np.percentile(xis, [2.5, 97.5])
    rows2c.append([side, float(xi), float(bet), f"[{lo:.3f},{hi:.3f}]"])
md.append(fmt(rows2c, ["side", "xi_s", "beta_s", "xi 95%CI"],
    "E2c -- GPD refit in s coordinates (psi=1%)",
    "compare with raw-D xi (moe: +0.146/-0.275)."))
J["E2c"] = rows2c

# ================= E4: per-position identity + autocorr ================
rows4a = []
ge = np.unique(np.geomspace(1, trel.max() + 1, 13).astype(int))
gi = np.clip(np.searchsorted(ge, trel, side="right") - 1, 0, len(ge) - 2)
for k in range(len(ge) - 1):
    m = gi == k
    if m.sum() < 5000:
        continue
    K = np.exp(D[m])
    mu_ = float(K.mean())
    se_ = float(K.std() / np.sqrt(m.sum()))
    rows4a.append([f"t~{int(np.median(trel[m]))}", int(m.sum()), mu_,
                   f"[{mu_-1.96*se_:.4f},{mu_+1.96*se_:.4f}]",
                   "yes" if abs(mu_ - 1) < 1.96 * se_ else "NO"])
md.append(fmt(rows4a, ["pos bin", "n", "E[K|pos]", "95%CI", "contains 1"],
    "E4a -- per-position identity E[K|pos]=1", "valid y-free conditioning."))
J["E4a"] = rows4a

# E4b variance inflation
bnd = np.flatnonzero(np.r_[True, traj[1:] != traj[:-1], True])
st_, sp_ = bnd[:-1], bnd[1:]
S = np.add.reduceat(D, st_)
var_S = float(S.var())
var_pos = np.zeros(len(ge) - 1)
for k in range(len(ge) - 1):
    m = gi == k
    if m.sum() > 100:
        var_pos[k] = D[m].var()
sum_var = float(np.add.reduceat(var_pos[gi], st_).mean())
rho = var_S / max(sum_var, 1e-300)
md.append(fmt([[var_S, sum_var, (var_S - sum_var) / 2, rho]],
    ["Var(S) actual", "E[sum Var(D_t|pos)]", "2*sum Cov / 2", "rho_infl"],
    "E4b -- accumulation variance vs independent sum",
    "rho~1 => iid approximation exempt; rho>>1 => n_eff discount."))
J["E4b"] = dict(var_S=var_S, sum_var=sum_var, rho=rho)

# E4c autocorrelation by lag, stratified by flip at origin
rows4c = []
for tau in (1, 2, 4, 8, 16, 32):
    a, b_ = D[:-tau], D[tau:]
    same = traj[:-tau] == traj[tau:]
    def corr(m):
        if m.sum() < 1000:
            return np.nan
        return float(np.corrcoef(a[m], b_[m])[0, 1])
    c_all = corr(same)
    c_flip = corr(same & flip[:-tau]) if arch == "moe" else np.nan
    c_clean = corr(same & ~flip[:-tau]) if arch == "moe" else np.nan
    rows4c.append([tau, c_all, c_flip, c_clean])
md.append(fmt(rows4c, ["tau", "corr(all)", "corr(flip origin)",
                       "corr(clean origin)"],
    "E4c -- within-trajectory autocorrelation of D",
    "flip-origin elevation = temporal fingerprint of cascades."))
J["E4c"] = rows4c

# ================= E5: sequence-level numbers ==========================
W = np.exp(S)
Tbar = float(np.diff(bnd).mean())
sig2_pool = float(mad_sigma(D) ** 2)
pred = float(np.exp(Tbar * sig2_pool))
rows5 = [[float(W.mean()), float((W ** 2).mean()),
          float((W.sum() ** 2 / (W ** 2).sum()) / len(W)),
          pred, float((W ** 2).mean() / W.mean() ** 2)]]
md.append(fmt(rows5, ["E[W]", "E[W^2]", "ESS/n", "exp(T*sigma_MAD^2) pred",
                      "E[W^2]/E[W]^2 actual"],
    "E5 -- sequence-level numbers",
    f"Tbar={Tbar:.0f}; prediction uses pooled MAD^2 (iid lognormal)."))
J["E5"] = rows5

# ================= E6: folk-band audit =================================
K = np.exp(D)
drift = np.exp(D) - 1 - D
tot_drift = float(drift.sum())
far = np.abs(D) > 3.5
bands = [
    ("folk [0.5,5]", (K < 0.5) | (K > 5.0)),
    ("derived [0.72,1.29] (+cap2 noted)", (K < 0.72) | (K > 1.29)),
    (f"s-coord |s|>{Z_PSI:g}", tail),
]
rows6 = []
for nm, m in bands:
    rate = float(m.mean())
    fshare = float(flip[m].mean()) if m.sum() else np.nan
    cap = float((m & far).sum() / max(far.sum(), 1)) if far.sum() else np.nan
    leak = float(drift[~m].sum() / tot_drift) if tot_drift > 0 else np.nan
    rows6.append([nm, rate, fshare, cap, leak])
md.append(fmt(rows6, ["band", "mask_rate", "flip_share_in_mask",
                      f"far-tail(|D|>3.5) capture", "drift-mass leak"],
    "E6 -- band audit",
    "leak = compensator mass sum(e^D-1-D) left unmasked / total."))
J["E6"] = rows6

with open(f"{OUT}/martingale_audit_{arch}.md", "w") as f:
    f.write("\n".join(md))
def _c(o):
    if isinstance(o, dict):
        return {k: _c(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_c(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o
with open(f"{OUT}/martingale_audit_{arch}.json", "w") as f:
    json.dump(_c(J), f, indent=1, default=str)
log(f"wrote martingale_audit_{arch}.md/json")
