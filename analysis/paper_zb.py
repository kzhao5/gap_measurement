"""Z+B modelling: validation experiments E-A..E-E + paper Figs 1, 3, 4.

Data: results/tokens_{arch}.parquet (campaign, fresh base model, single version).
Conventions: phi = max(1-p, eps0); equal-mass phi buckets; core-out = |Z|>kappa
with Z = D/(c*phi); core-out n<50 -> grey; trajectory bootstrap 500; kappa in
{4,5,6}. All plotting params saved to results/paper/params_{arch}.json.
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCH = sys.argv[1] if len(sys.argv) > 1 else "moe"
EPS0, KAPPA, KAPPAS, NB, NBOOT = 5e-3, 5, (4, 5, 6), 20, 500
N_BATCH = 1024
ROOT = "/home/kzhao2/gap_measurement"
OUT = f"{ROOT}/results/paper"; FIG = f"{ROOT}/results/figs/paper"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, GRAY = "#1f77b4", "#ff7f0e", "#9a9a9a"
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.grid": True,
                     "grid.color": "#ececec", "axes.spines.top": False,
                     "axes.spines.right": False})

cols = ["traj_id", "pos", "D", "logp_train"] + (["hard_flip", "flip", "n_hard_flip_layers"] if ARCH == "moe" else [])
t = pd.read_parquet(f"{ROOT}/results/tokens_{ARCH}.parquet", columns=cols)
t = t.sort_values(["traj_id", "pos"], ignore_index=True)
D = t["D"].to_numpy(np.float64)
u = np.clip(1.0 - np.exp(t["logp_train"].to_numpy(np.float64)), 0, 1)
phi = np.maximum(u, EPS0)
tid = t["traj_id"].to_numpy()
starts = np.r_[0, np.nonzero(tid[1:] != tid[:-1])[0] + 1, len(tid)]
rep = {"arch": ARCH, "n_tokens": int(len(D)), "n_traj": int(len(starts) - 1),
       "eps0": EPS0, "kappa": KAPPA, "nb": NB}
def mad(v): return 1.4826 * np.median(np.abs(v - np.median(v)))

# ---------------- buckets + c (bulk MAD ~ c*phi on core phi >= 1e-3) --------
edges = np.quantile(u, np.linspace(0, 1, NB + 1))
bi = np.clip(np.digitize(u, edges[1:-1]), 0, NB - 1)
rows = []
for b in range(NB):
    m = bi == b
    rows.append(dict(b=b, n=int(m.sum()), phi=float(np.median(phi[m])), u=float(np.median(u[m])),
                     bulk_mad=mad(D[m])))
B = pd.DataFrame(rows)
core = B[B.u >= 1e-3]; w = core.n.values.astype(float)
c = float(np.sum(w * core.u * core.bulk_mad) / np.sum(w * core.u ** 2))
rep["c"] = c
Z = D / (c * phi)
absZ = np.abs(Z)

# ---------------- E-A dual scale: core-out |D| median per bucket ------------
def dual_rows(kap):
    out = []
    for b in range(NB):
        m = bi == b; mo = m & (absZ > kap)
        out.append(dict(b=b, kappa=kap, n=int(m.sum()), n_out=int(mo.sum()), phi=B.phi[b],
                        bulk_mad=B.bulk_mad[b],
                        out_med=float(np.median(np.abs(D[mo]))) if mo.sum() else np.nan,
                        out_q90=float(np.quantile(np.abs(D[mo]), .9)) if mo.sum() else np.nan,
                        eps_hat=float(mo.sum() / m.sum()),
                        pos_frac=float((D[mo] > 0).mean()) if mo.sum() else np.nan,
                        q95_pos=float(np.quantile(D[mo], .95)) if mo.sum() else np.nan,
                        q99_pos=float(np.quantile(D[mo], .99)) if mo.sum() else np.nan))
    return pd.DataFrame(out)
EA = pd.concat([dual_rows(k) for k in KAPPAS], ignore_index=True)
EA.to_csv(f"{OUT}/EA_dualscale_{ARCH}.csv", index=False)
# trajectory bootstrap for kappa=5 (bulk mad + out med per bucket)
rng = np.random.RandomState(0); ntr = len(starts) - 1
boot_bulk = np.full((NBOOT, NB), np.nan); boot_out = np.full((NBOOT, NB), np.nan)
sub = min(ntr, 4000)
pool = rng.choice(ntr, size=sub, replace=False) if ntr > sub else np.arange(ntr)
for r in range(NBOOT):
    pick = rng.choice(pool, size=len(pool), replace=True)
    idx = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in pick])
    bb, dd, zz = bi[idx], D[idx], absZ[idx]
    for b in range(NB):
        m = bb == b
        if m.sum() < 30: continue
        boot_bulk[r, b] = mad(dd[m]); mo = m & (zz > KAPPA)
        if mo.sum() >= 20: boot_out[r, b] = np.median(np.abs(dd[mo]))
ci = lambda a: (np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0))
bulk_lo, bulk_hi = ci(boot_bulk); out_lo, out_hi = ci(boot_out)

# ---------------- E-B sparsity + clustering ------------------------------
EB = {}
for kap in KAPPAS:
    o = absZ > kap
    EB[str(kap)] = {"eps_hat": float(o.mean()),
                    "per_bucket": [float(((bi == b) & o).sum() / (bi == b).sum()) for b in range(NB)]}
o = absZ > KAPPA
# within-trajectory autocorrelation of the core-out indicator (lags 1..20), permutation null
lags = list(range(1, 21)); acf = []; 
for L in lags:
    same = tid[L:] == tid[:-L]
    a, b_ = o[:-L][same].astype(float), o[L:][same].astype(float)
    acf.append(float(np.corrcoef(a, b_)[0, 1]) if a.std() > 0 else np.nan)
op = o.copy(); rng.shuffle(op); nullacf = []
for L in lags:
    same = tid[L:] == tid[:-L]
    a, b_ = op[:-L][same].astype(float), op[L:][same].astype(float)
    nullacf.append(float(np.corrcoef(a, b_)[0, 1]))
lift = []
for L in (1, 2, 4, 8, 16):
    same = tid[L:] == tid[:-L]; a, b_ = o[:-L][same], o[L:][same]
    lift.append((L, float(b_[a].mean() / max(o.mean(), 1e-12))))
EB["clustering"] = {"acf": acf, "null_acf": nullacf, "lift": lift,
                    "delta_vs_phi_corr": float(np.corrcoef(o.astype(float), phi)[0, 1])}
rep["EB"] = EB

# ---------------- E-C sign/envelope for high-confidence buckets ---------------
EC = EA[(EA.kappa == KAPPA)][["b", "phi", "n_out", "pos_frac", "q95_pos", "q99_pos", "out_med"]]
EC.to_csv(f"{OUT}/EC_sign_envelope_{ARCH}.csv", index=False)
hi = EC[EC.phi < 0.05]
rep["EC_highconf"] = {"pos_frac_mean": float(np.nanmean(hi.pos_frac)), "q99_max": float(np.nanmax(hi.q99_pos)),
                      "out_med_mean": float(np.nanmean(hi.out_med))}

# ---------------- E-D mechanism (MoE only): core-out vs hard routing flip ----
if ARCH == "moe":
    hf = t["hard_flip"].to_numpy().astype(bool)
    rep["ED"] = {"P_hardflip": float(hf.mean()), "P_hardflip_given_out": float(hf[o].mean()),
                 "P_out_given_hardflip": float(o[hf].mean()), "P_out": float(o.mean()),
                 "enrichment": float(hf[o].mean() / max(hf.mean(), 1e-12)),
                 "n_hard_layers_out_mean": float(t["n_hard_flip_layers"].to_numpy()[o].mean()),
                 "n_hard_layers_all_mean": float(t["n_hard_flip_layers"].to_numpy().mean())}
    # by confidence: high vs low
    for nm, msk in (("highconf(u<0.05)", u < 0.05), ("lowconf(u>0.5)", u > 0.5)):
        rep["ED"][nm] = {"P_hf": float(hf[msk].mean()), "P_hf_given_out": float(hf[msk & o].mean()) if (msk & o).sum() else None}

# ---------------- E-E pivot collapse: Z quantiles per bucket -----------------
QS = [.001, .01, .1, .5, .9, .99, .999]
EE = pd.DataFrame([dict(b=b, phi=B.phi[b], n=B.n[b], **{f"q{q}": (float(np.quantile(Z[bi == b], q)) if (bi == b).sum() else np.nan) for q in QS})
                   for b in range(NB)])
EE.to_csv(f"{OUT}/EE_pivot_{ARCH}.csv", index=False)
nonfloor = EE[EE.phi > EPS0]
rep["EE"] = {"q10_spread": float(nonfloor["q0.1"].max() - nonfloor["q0.1"].min()),
             "q90_spread": float(nonfloor["q0.9"].max() - nonfloor["q0.9"].min()),
             "q50_spread": float(nonfloor["q0.5"].max() - nonfloor["q0.5"].min())}

# ---------------- Fig 4 Equalizer + tau* -------------------------------------
Zc = Z[u >= 1e-3]                       # standardized, non-floor
zs = rng.choice(Zc, size=min(len(Zc), 3_000_000), replace=False)
eps_hat = float((np.abs(zs) > KAPPA).mean())
taus = np.linspace(0.5, 40, 400)
H = np.array([np.mean(np.maximum(zs - tau, 0)) for tau in taus])
Dl = eps_hat * taus - (1 - eps_hat) * H
H2, D2 = H ** 2, Dl ** 2
env = np.maximum(H2, D2); tau_star = float(taus[np.argmin(env)])
# also closure-equation tau (robust-MSE stationarity) with eps_hat and n=1024
mz = float(zs.mean()); lo, hi_ = mz, 60.0
for _ in range(80):
    mid = 0.5 * (lo + hi_)
    g = (1 - eps_hat) * np.mean(np.maximum(zs - mid, 0)) - (eps_hat + 1 / (N_BATCH - 1)) * (mid - mz)
    lo, hi_ = (mid, hi_) if g > 0 else (lo, mid)
tau_closure = 0.5 * (lo + hi_)
rep["fig4"] = {"eps_hat_kappa": eps_hat, "tau_star_equalizer": tau_star, "tau_closure_eq": float(tau_closure),
               "tau_calibrated_lamRL_over_c": float(2.3 / c), "lambda_from_tau_star": float(c * tau_star)}
# GPD tail on |Z| for Fig 3 right (simple MLE via scipy)
from scipy.stats import genpareto
if len(zs):
    thr = np.quantile(np.abs(zs), 0.99); exc = np.abs(zs)[np.abs(zs) > thr] - thr
    xi, _, beta = genpareto.fit(exc, floc=0)
    rep["gpd_tail"] = {"threshold_q99": float(thr), "xi": float(xi), "beta": float(beta)}
else:
    rep["gpd_tail"] = None

json.dump(rep, open(f"{OUT}/params_{ARCH}.json", "w"), indent=2)

# ================= Fig 1 dual scale =========================================
E5 = EA[EA.kappa == KAPPA].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7, 5))
xs = np.geomspace(EPS0 / 3, 1, 50)
ax.axvspan(EPS0 / 3, EPS0, color="#eeeeee", zorder=0); ax.text(EPS0 / 2.2, 0.3, "quantization\nfloor", fontsize=8, ha="center")
ax.plot(xs, c * xs, "--", color=GRAY, lw=1, label=rf"$c\varphi$, c={c:.3f}")
ax.axhline(0.12, ls=":", color=GRAY, lw=1, label="0.12")
sz = 20 + 60 * (E5.n / E5.n.max())
ax.fill_between(E5.phi, bulk_lo, bulk_hi, color=BLUE, alpha=.15)
ax.plot(E5.phi, E5.bulk_mad, "o-", color=BLUE, ms=4, label=r"bulk $\sigma_{MAD}(\log k\,|\,\varphi)$")
ok = E5.n_out >= 50
ax.fill_between(E5.phi[ok], out_lo[ok], out_hi[ok], color=ORANGE, alpha=.15)
ax.plot(E5.phi[ok], E5.out_med[ok], "s-", color=ORANGE, ms=4, label=rf"core-out ($|Z|>{KAPPA}$) median $|\log k|$")
ax.plot(E5.phi[~ok], E5.out_med[~ok], "s", color="#cccccc", ms=4)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$\varphi=\max(1-p,\varepsilon_0)$"); ax.set_ylabel(r"$|\log k|$ scale")
ax.set_ylim(max(1e-5, np.nanmin(E5.bulk_mad[E5.bulk_mad > 0]) / 2), 3)
ax.text(0.02, 0.05, f"same 0.12 displacement = {0.12/(c*0.5):.1f}σ at p=0.5, {0.12/(c*0.005):.0f}σ at p=0.995",
        transform=ax.transAxes, fontsize=8)
ax.legend(fontsize=8, loc="upper left"); ax.set_title(f"Fig 1 dual scale ({ARCH}): aleatoric ∝φ vs epistemic flat")
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_dualscale_{ARCH}.png"); plt.close(fig)

# ================= Fig 3 pivot + CCDF =======================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for q in QS:
    ax[0].plot(EE.phi, EE[f"q{q}"], "o-", ms=3, label=f"q{q}")
ax[0].axvspan(EPS0 / 3, EPS0, color="#eeeeee"); ax[0].set_xscale("log"); ax[0].set_yscale("symlog", linthresh=1)
ax[0].set_xlabel(r"$\varphi$"); ax[0].set_ylabel(r"$Z=\log k/(c\varphi)$ quantiles"); ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title("Fig 3a pivot collapse")
az = np.sort(np.abs(zs)); cc = 1 - np.arange(1, len(az) + 1) / len(az)
ax[1].loglog(az[::200], cc[::200], ".", ms=2, color=BLUE, label="|Z| CCDF")
zz = np.geomspace(thr, az.max(), 100)
ax[1].loglog(zz, 0.01 * genpareto.sf(zz - thr, xi, loc=0, scale=beta), "-", color=ORANGE, label=f"GPD tail ξ={xi:.2f}")
ax[1].axvline(tau_star, color="k", ls="--", lw=1, label=rf"$\tau^*$={tau_star:.1f} (equalizer)")
ax[1].axvline(2.3 / c, color=GRAY, ls=":", lw=1, label=rf"$\lambda_+/c$={2.3/c:.1f}")
ax[1].text(tau_star * 1.1, 0.5, rf"mass right of $\tau^*$: {(np.abs(zs)>tau_star).mean():.2e}", fontsize=8)
ax[1].set_xlabel("|Z|"); ax[1].set_ylabel("P(|Z|>z)"); ax[1].legend(fontsize=7); ax[1].set_title("Fig 3b tail")
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_pivot_ccdf_{ARCH}.png"); plt.close(fig)

# ================= Fig 4 Equalizer ==========================================
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(taus, H2, color=BLUE, label=r"$H(\tau)^2=[E(Z-\tau)_+]^2$ (over-clip bias)")
ax.plot(taus, D2, color=ORANGE, label=r"$D(\tau)^2=[\epsilon\tau-(1-\epsilon)H]^2$ (leakage)")
ax.plot(taus, env, color="k", lw=2, alpha=.5, label="max envelope")
ax.axvline(tau_star, color="k", ls="--", lw=1); ax.text(tau_star, env.max() * .5, rf"$\tau^*$={tau_star:.1f}", fontsize=9)
ax.set_yscale("log"); ax.set_xlabel(r"$\tau$"); ax.set_title(rf"Fig 4 equalizer ({ARCH}), $\hat\epsilon$={eps_hat:.3g} (κ={KAPPA})")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{FIG}/fig4_equalizer_{ARCH}.png"); plt.close(fig)

print(json.dumps({k: v for k, v in rep.items() if k not in ("EB",)}, indent=1)[:3000])
print("EB eps_hat:", {k: round(v["eps_hat"], 5) for k, v in EB.items() if k != "clustering"},
      "acf lag1/5:", round(acf[0], 4), round(acf[4], 4), "null:", round(nullacf[0], 4), "lift:", lift[:3])
