#!/usr/bin/env python
"""Empirical conditional law of log k_t -- the object TIS / IcePop / KPop
each assume a parametric form for (see handwritten derivation):

  TIS    : log k_t | h_t        ~ N(-sigma^2/2, sigma^2)          (h-independent)
  IcePop : log k_t | h_t        ~ (1-z) N(-s^2/2,s^2)
                                  + z/2 [-mu-Exp(lam)] + z/2 [mu+Exp(lam)]
  KPop   : log k_t | h_t, y_t   ~ N(0, sigma^2 (1-p_t)^2),  p_t = pi_old(y_t|h_t)

Here k_t = pi_old(y_t|h_t) / pi_infer(y_t|h_t), so log k_t = D.

Conditioning axes (projections of (h_t, y_t)):
  --cond ptrain    s_t = -log pi_old(y_t|h_t)   <- KPop's exact variable
  --cond surprisal s_t = -log pi_infer(y_t|h_t)
  --cond entropy   H(pi_old(.|h_t))  y_t-free   <- E[k_t|h_t]=1 identity test
                   (needs recomp_*_bf16_ent.parquet from slurm/entropy.sbatch)

Usage: python analysis/cond_dist.py {moe|dense} [--cond ptrain]
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "/home/kzhao2/nobackup/autodelete/gap_measurement"
LOCAL = "/home/kzhao2/gap_measurement/results"
FIGS = os.path.join(ROOT, "analysis", "figs")

BLUE = "#2a78d6"
INK = "#0b0b0b"
GRAY = "#8a8a8a"
ORANGE = "#b0651a"
RAMP = ["#cfe0f4", "#9cc0e8", "#6da0db", "#3a7bc4", "#1f5089"]  # light->dark
N_BINS = 36
QS = [0.001, 0.01, 0.10, 0.50, 0.90, 0.99, 0.999]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d9d9d9", "axes.grid": True,
    "grid.color": "#ececec", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 11.5, "figure.dpi": 150,
})


def load(arch, cond):
    t = pd.read_parquet(os.path.join(LOCAL, f"tokens_{arch}.parquet"),
                        columns=["traj_id", "pos", "logp_infer",
                                 "logp_train", "D"])
    if cond == "entropy":
        files = sorted(glob.glob(
            f"{ROOT}/recompute/{arch}/recomp_shard*_bf16_ent.parquet"))
        assert files, "no *_ent.parquet yet -- run slurm/entropy.sbatch"
        e = pd.concat([pd.read_parquet(f, columns=["traj_id", "pos", "entropy"])
                       for f in files])
        t = t.merge(e, on=["traj_id", "pos"], how="inner")
        h = np.clip(t["entropy"].to_numpy(np.float64), 1e-7, None)
        label = r"$H(\pi_{old}(\cdot|h_t))$  ($y_t$-free)"
    elif cond == "ptrain":
        h = np.clip(-t["logp_train"].to_numpy(np.float64), 1e-7, None)
        label = r"$s_t=-\log \pi_{old}(y_t|h_t)$   ($p_t$: KPop's variable)"
    else:
        h = np.clip(-t["logp_infer"].to_numpy(np.float64), 1e-7, None)
        label = r"$s_t=-\log \pi_{infer}(y_t|h_t)$"
    return t["D"].to_numpy(np.float64), h, label


def qbins(h, n):
    edges = np.unique(np.quantile(h, np.linspace(0, 1, n + 1)))
    idx = np.clip(np.searchsorted(edges, h, side="right") - 1,
                  0, len(edges) - 2)
    return idx, len(edges) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=["moe", "dense"])
    ap.add_argument("--cond", choices=["ptrain", "surprisal", "entropy"],
                    default="ptrain")
    args = ap.parse_args()

    D, h, xlabel = load(args.arch, args.cond)
    idx, nb = qbins(h, N_BINS)

    hx, sig, meanD, ek, ek_se = (np.empty(nb) for _ in range(5))
    quant = np.empty((nb, len(QS)))
    asym = np.full(nb, np.nan)
    for b in range(nb):
        m = idx == b
        Db = D[m]
        hx[b] = np.median(h[m])
        quant[b] = np.quantile(Db, QS)
        med = quant[b][3]
        sig[b] = 1.4826 * np.median(np.abs(Db - med)) + 1e-12
        meanD[b] = Db.mean()
        K = np.exp(Db)
        ek[b] = K.mean()
        ek_se[b] = K.std() / np.sqrt(len(K))
        thr = 3 * sig[b]
        npos, nneg = (Db > thr).sum(), (Db < -thr).sum()
        if npos > 50:
            asym[b] = nneg / npos

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))
    cname = "H" if args.cond == "entropy" else "s"

    # (a) quantile fan -----------------------------------------------------
    ax = axes[0, 0]
    ax.fill_between(hx, quant[:, 0], quant[:, 6], color=BLUE, alpha=0.14,
                    lw=0, label="q0.1% - q99.9%")
    ax.fill_between(hx, quant[:, 1], quant[:, 5], color=BLUE, alpha=0.30,
                    lw=0, label="q1% - q99%")
    ax.fill_between(hx, quant[:, 2], quant[:, 4], color=BLUE, alpha=0.55,
                    lw=0, label="q10% - q90%")
    ax.plot(hx, quant[:, 3], color=INK, lw=1.8, label="median")
    g999 = np.quantile(np.abs(D), 0.999)
    for sgn in (1, -1):
        ax.axhline(sgn * g999, color=GRAY, ls="--", lw=1.2)
    ax.text(hx[0] * 1.3, g999 * 1.1, "h-independent const threshold "
            "(TIS/IcePop form), q99.9(|D|)", color=GRAY, fontsize=8, va="bottom")
    if args.cond == "surprisal":
        # D = log p_train - log p_infer <= -log p_infer = s_t
        ax.plot(hx, hx, color=ORANGE, ls="-.", lw=1.2)
        ax.text(hx[-1] * 0.5, hx[-1] * 0.75, r"hard bound $D\leq s_t$",
                color=ORANGE, fontsize=8, rotation=14)
    elif args.cond == "ptrain":
        # D = -s_t - log p_infer >= -s_t
        ax.plot(hx, -hx, color=ORANGE, ls="-.", lw=1.2)
        ax.text(hx[-1] * 0.5, -hx[-1] * 0.85, r"hard bound $D\geq -s_t$",
                color=ORANGE, fontsize=8, rotation=-14)
    ax.set_xscale("log")
    yl = np.quantile(np.abs(D), 0.99999) * 1.5
    ax.set_ylim(-yl, yl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$D=\log k_t$")
    ax.set_title(f"(a) conditional quantile fan of $\\log k_t\\,|\\,{cname}$")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # (b) conditional scale law -------------------------------------------
    ax = axes[0, 1]
    ax.plot(hx, sig, "o", ms=5, color=BLUE, label=r"$\sigma_{MAD}(D\,|\,bin)$")
    if args.cond == "entropy":
        g1, g2 = hx, np.sqrt(hx)
        lb1, lb2 = r"$c\,H$", r"$c\,\sqrt{H}$"
    else:
        p = np.exp(-hx)
        g1 = 1 - p
        g2 = np.sqrt((1 - p) / np.clip(p, 1e-12, None))
        lb1 = r"$c\,(1-p_t)$  -- KPop's assumed law"
        lb2 = r"$c\,\sqrt{(1-p)/p}$  (Fisher-info alt.)"
    w = np.isfinite(sig) & (sig > 0)
    c1 = np.exp(np.mean(np.log(sig[w]) - np.log(np.clip(g1[w], 1e-30, None))))
    c2 = np.exp(np.mean(np.log(sig[w]) - np.log(np.clip(g2[w], 1e-30, None))))
    ax.plot(hx, c1 * g1, ls="--", color=INK, lw=1.5, label=lb1)
    ax.plot(hx, c2 * g2, ls=":", color=ORANGE, lw=1.8, label=lb2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\sigma_{MAD}$")
    ax.set_title("(b) scale law: TIS/IcePop assume flat; KPop assumes "
                 r"$\propto(1-p_t)$")
    ax.legend(loc="upper left", fontsize=8.5)

    # (c) standardized shape collapse -------------------------------------
    ax = axes[0, 2]
    picks = np.unique(np.linspace(2, nb - 2, 5).astype(int))
    zg = np.linspace(-12, 12, 121)
    zc = 0.5 * (zg[:-1] + zg[1:])
    for ci, b in enumerate(picks):
        m = idx == b
        z = (D[m] - quant[b][3]) / sig[b]
        cnt, _ = np.histogram(z, bins=zg, density=True)
        ax.stairs(np.maximum(cnt, 1e-8), zg, color=RAMP[ci], lw=1.6,
                  label=f"{cname}≈{hx[b]:.2g}")
    ax.plot(zc, np.exp(-zc ** 2 / 2) / np.sqrt(2 * np.pi), color=GRAY,
            ls="--", lw=1.4, label="N(0,1) core (TIS/KPop)")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 30)
    ax.set_xlabel(r"$z=(D-med)/\sigma_{MAD}(bin)$")
    ax.set_ylabel("density (log)")
    ax.set_title("(c) shape: near scale-family, non-Gaussian tails")
    ax.legend(fontsize=8, title="uncertainty bin", title_fontsize=8)

    # (d) conditional mean / identity -------------------------------------
    ax = axes[1, 0]
    if args.cond == "entropy":
        # y_t-free axis: the per-step identity E[k_t | h_t] = 1 is testable
        ax.errorbar(hx, ek, yerr=2 * ek_se, fmt="o", ms=5, color=BLUE,
                    ecolor="#9cc0e8", capsize=2.5,
                    label=r"$\mathrm{E}[k_t\,|\,bin]\ \pm 2SE$")
        ax.axhline(1.0, color=INK, ls="--", lw=1.4,
                   label=r"identity $\mathrm{E}[k_t|h_t]=1$")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\mathrm{E}[k_t\,|\,bin]$")
        ax.set_title("(d) conditional identity check ($y_t$-free axis)")
        ax.legend(fontsize=8.5, loc="upper left")
    else:
        neg = meanD < 0
        ax.plot(hx[neg], -meanD[neg], "o", ms=5, color=BLUE,
                label=r"$-\,\mathrm{E}[D\,|\,bin]$  (empirical)")
        if (~neg).any():
            ax.plot(hx[~neg], meanD[~neg], "s", ms=5, mfc="white", mec=BLUE,
                    label=r"$+\,\mathrm{E}[D\,|\,bin]$ (sign flip)")
        ax.plot(hx, sig ** 2 / 2, ls="--", color=INK, lw=1.5,
                label=r"$\sigma_b^2/2$  (lognormal mean, TIS: $\mu=-\sigma^2/2$)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|\mathrm{E}[D]|$")
        ax.set_title("(d) mean: KPop assumes 0; TIS assumes $-\\sigma^2/2$")
        ax.legend(fontsize=8.5, loc="upper left")

    # (e) tail asymmetry ---------------------------------------------------
    ax = axes[1, 1]
    ok = np.isfinite(asym)
    ax.plot(hx[ok], asym[ok], "-o", ms=4.5, lw=1.6, color=BLUE)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.2)
    ax.text(hx[0] * 1.3, 1.05, "symmetric (IcePop assumes z/2 - z/2)",
            color=GRAY, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$P(D<-3\sigma_b)\;/\;P(D>+3\sigma_b)$")
    ax.set_title("(e) outlier-branch asymmetry vs IcePop's symmetric mixture")

    # (f) outlier branch family: exp tail in |z|? -------------------------
    ax = axes[1, 2]
    zt = np.linspace(3, 30, 55)
    for ci, b in enumerate(picks):
        m = idx == b
        z = (D[m] - quant[b][3]) / sig[b]
        zneg = -z[z < -3]
        if len(zneg) < 100:
            continue
        ccdf = np.array([(zneg > v).mean() for v in zt])
        ax.plot(zt, np.maximum(ccdf, 1e-7), color=RAMP[ci], lw=1.6,
                label=f"{cname}≈{hx[b]:.2g}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|z|$ (left branch, beyond $3\sigma_b$)")
    ax.set_ylabel(r"$P(|z|>x \,|\, |z|>3)$ (log)")
    ax.set_title("(f) IcePop Exp($\\lambda$) branch: straight iff exponential")
    ax.legend(fontsize=8, title="bin", title_fontsize=8)

    fig.suptitle(
        f"V8 -- empirical law of $\\log k_t$ vs the assumed laws of "
        f"TIS / IcePop / KPop   ({args.arch}, {len(D):,} tokens, "
        f"axis: {args.cond})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(FIGS, exist_ok=True)
    stem = f"V8_cond_{args.cond}_{args.arch}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"))
    print(f"wrote {FIGS}/{stem}.png  (bins={nb})  global E[K]="
          f"{np.exp(D).mean():.6f}")


if __name__ == "__main__":
    main()
