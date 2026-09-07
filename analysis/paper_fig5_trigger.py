"""Fig 5 / Experiment B: band trigger dynamics on the champion arm (ktdump/lgrid_0).

(a) trigger rate vs training step (version as step proxy) with train-time-scale
    annotations; (b) p-histograms of clipped tokens in 3 phases vs IcePop-masked
    tokens (recipe_icepop dump, mask outside [0.5,5]); (c) |log k| of clipped
    tokens vs their aleatoric scale c_phase*phi (train-time c per phase, lag=0
    per-file) and vs static c=0.156.
Also redoes clip enrichment / staleness recall on lgrid_0 (replacing the
recipe_sigmatis-based numbers).
"""
import glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/home/kzhao2/nobackup/autodelete/areal_rl/ktdump"
ROOT = "/home/kzhao2/gap_measurement"; OUT = f"{ROOT}/results/paper"; FIG = f"{ROOT}/results/figs/paper"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
LAM, EPS0, C_STATIC = 2.3, 5e-3, 0.156
BLUE, ORANGE, GRAY = "#1f77b4", "#ff7f0e", "#9a9a9a"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "axes.grid": True, "grid.color": "#ececec",
                     "axes.spines.top": False, "axes.spines.right": False})
def mad(v): return 1.4826 * np.median(np.abs(v - np.median(v)))
def load(name):
    dfs = []
    for i, f in enumerate(sorted(glob.glob(f"{R}/{name}/*.parquet"))):
        x = pq.read_table(f).to_pandas(); x["lag"] = x.version.max() - x.version; x["step"] = x.version.max(); x["file"] = i; dfs.append(x)
    d = pd.concat(dfs, ignore_index=True)
    d["lk"] = d.prox_logp.astype(float) - d.old_logp.astype(float)
    d["u"] = np.clip(1 - np.exp(d.prox_logp.astype(float)), 0, 1); d["phi"] = np.maximum(d.u, EPS0)
    return d
def cfit(d):
    p = 1 - d.u.values; qb = np.quantile(p, np.linspace(0, 1, 21)); bi = np.clip(np.digitize(p, qb[1:-1]), 0, 19)
    rows = [(np.median(d.u.values[bi == b]), mad(d.lk.values[bi == b]), (bi == b).sum()) for b in range(20) if (bi == b).sum() > 100]
    sc = pd.DataFrame(rows, columns=["u", "mad", "n"]); core = sc[sc.u >= 1e-3]; w = core.n.values.astype(float)
    return float(np.sum(w * core.u * core["mad"]) / np.sum(w * core.u ** 2))

d = load("lgrid_0"); rep = {"n_tokens": int(len(d)), "steps": [int(d.step.min()), int(d.step.max())]}
d["trig"] = d.lk > LAM * d.phi
smax = d.step.max(); d["phase"] = pd.cut(d.step, [-2, smax / 3, 2 * smax / 3, smax + 1], labels=["early", "mid", "late"])
# per-phase train-time c (lag=0)
cph = {ph: cfit(d[(d.phase == ph) & (d.lag == 0)]) for ph in ["early", "mid", "late"]}
rep["c_train_by_phase"] = cph; rep["c_static"] = C_STATIC
# (a) trigger rate per step
per = d.groupby("step").agg(rate=("trig", "mean"), n=("trig", "size"), stale=("lag", lambda s: (s >= 1).mean())).reset_index()
per.to_csv(f"{OUT}/fig5a_trigger_rate_{'lgrid0'}.csv", index=False)
rep["trigger_rate"] = {"overall": float(d.trig.mean()), "early": float(d[d.phase == "early"].trig.mean()),
                       "mid": float(d[d.phase == "mid"].trig.mean()), "late": float(d[d.phase == "late"].trig.mean()),
                       "slope_per_step": float(np.polyfit(per.step, per.rate, 1)[0])}
# (b) clipped p composition per phase + icepop contrast
ice = load("recipe_icepop"); ice["masked"] = (ice.lk < np.log(0.5)) | (ice.lk > np.log(5.0))
rep["icepop_mask_rate"] = float(ice.masked.mean())
# (c) |log k| of clipped vs aleatoric scale
cl = d[d.trig]
cl_stats = {}
for ph in ["early", "mid", "late"]:
    x = cl[cl.phase == ph]
    cl_stats[ph] = dict(n=int(len(x)), median_abs_logk=float(np.median(np.abs(x.lk))),
                        median_cphi_train=float(np.median(cph[ph] * x.phi)), median_cphi_static=float(np.median(C_STATIC * x.phi)),
                        frac_u_lt_01=float((x.u < 0.1).mean()), frac_p_eq_1=float((x.u <= 0).mean()),
                        median_ratio_train=float(np.median(np.abs(x.lk) / (cph[ph] * x.phi))),
                        frac_ratio_gt5_train=float((np.abs(x.lk) / (cph[ph] * x.phi) > 5).mean()))
rep["clipped"] = cl_stats
# redo enrichment/recall on lgrid_0
st = d.lag >= 1; rep["stale_eps"] = float(st.mean())
enr = {}
for lam in (1.0, 1.6, 2.3, 3.1, 4.0, 6.7):
    A = d.lk > lam * d.phi
    enr[str(lam)] = dict(clip_rate=float(A.mean()), precision=float(st[A].mean()) if A.any() else None, recall=float(A[st].mean()),
                         enrichment=float(st[A].mean() / max(st.mean(), 1e-12)) if A.any() else None,
                         share_u_lt_01=float((d.u[A] < 0.1).mean()) if A.any() else None,
                         # core-out (train-time scale, kappa=5) precision: what fraction of clipped are |Z_train|>5
                         coreout_precision=float((np.abs(d.lk[A]) / (np.array([cph[p] for p in d.phase[A]]) * d.phi[A]) > 5).mean()) if A.any() else None)
rep["enrichment_lgrid0"] = enr
json.dump(rep, open(f"{OUT}/fig5_params.json", "w"), indent=2, default=str)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
ax[0].plot(per.step, per.rate, "o-", ms=3, color=BLUE, label="σ-TIS trigger rate")
ax[0].plot(per.step, per.stale, "s--", ms=2, color=GRAY, label="stale share (lag≥1)")
for ph, x0 in (("early", 0), ("mid", smax / 3), ("late", 2 * smax / 3)):
    ax[0].text(x0 + 2, ax[0].get_ylim()[1] * 0.9 if False else per.rate.max() * 1.05, f"c={cph[ph]:.2f}", fontsize=8)
ax[0].set_xlabel("training step (policy version)"); ax[0].set_ylabel("fraction of tokens"); ax[0].set_yscale("log")
ax[0].legend(fontsize=8); ax[0].set_title("(a) trigger rate vs step")
bins = np.linspace(0, 1, 26)
for ph, col in (("early", "#9ecae1"), ("mid", "#4292c6"), ("late", "#08306b")):
    ax[1].hist(1 - cl[cl.phase == ph].u, bins=bins, density=True, histtype="step", lw=1.5, color=col, label=f"σ-TIS clipped, {ph}")
ax[1].hist(1 - ice[ice.masked].u, bins=bins, density=True, histtype="step", lw=1.5, color=ORANGE, ls="--", label="IcePop masked (all)")
ax[1].set_xlabel(r"$p_t$ of affected token"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=7); ax[1].set_title("(b) who gets clipped: σ-TIS vs IcePop")
for ph, col in (("early", "#9ecae1"), ("mid", "#4292c6"), ("late", "#08306b")):
    x = cl[cl.phase == ph]
    ax[2].scatter(cph[ph] * x.phi.values[::20], np.abs(x.lk.values[::20]), s=3, alpha=.3, color=col, label=f"{ph} (c={cph[ph]:.2f})")
xs = np.geomspace(1e-4, 3, 50); ax[2].plot(xs, xs, "--", color=GRAY, lw=1, label="|log k| = cφ")
ax[2].plot(xs, 5 * xs, ":", color=GRAY, lw=1, label="5cφ (core-out)")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel(r"aleatoric scale $c_{phase}\varphi_t$"); ax[2].set_ylabel(r"$|\log k|$ of clipped token")
ax[2].legend(fontsize=7); ax[2].set_title("(c) clipped |log k| vs aleatoric scale")
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_trigger_dynamics.png"); plt.close(fig)
print(json.dumps(rep, indent=1, default=str))
