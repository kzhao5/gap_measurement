"""Figures V1-V7 (protocol section 4).

Reads analysis/tokens_{arch}.parquet, trajs_{arch}.parquet, stats_{arch}.json.
Writes PNG+PDF to analysis/figs/.

Style: light surface, recessive grid, one axis per chart (V5/V7 use panels,
never twin axes). Consistent tokens = blue #2a78d6, flipped = red #d03b3b
(pair CVD-validated: worst-case dE 23.8).

Usage: python analysis/figures.py moe   (or dense)
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy import stats as sps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import DATA_ROOT, MODELS

BLUE = "#2a78d6"   # route-consistent tokens
RED = "#d03b3b"    # flipped tokens
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def _save(fig, figdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def v1_histogram(tok, S, is_moe, figdir):
    D = tok["D"].to_numpy()
    lo, hi = np.quantile(D, [1e-6, 1 - 1e-6])
    span = max(abs(lo), abs(hi))
    bins = np.linspace(-span, span, 241)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if is_moe:
        flip = tok["hard_flip"].to_numpy()
        # red filled below, blue as a bold step outline on top: the two
        # bulks overlap almost entirely, a second filled layer would
        # hide the first
        ax.hist(D[flip], bins=bins, log=True, histtype="stepfilled",
                alpha=0.75, color=RED, label="hard flip (margin>1e-3)")
        ax.hist(D[~flip], bins=bins, log=True, histtype="step",
                lw=1.8, color=BLUE, label="no hard flip")
        sig = S["sigma_hat_hard"]
        ax.axvspan(-sig, sig, color=BLUE, alpha=0.12, lw=0)
        ax.annotate(
            f"$\\hat\\sigma$ = {sig:.1e}\n"
            f"$\\hat\\Delta_{{gap}}/\\hat\\sigma$ = "
            f"{S['gap_over_sigma_hard']:.1f}\n"
            f"$\\hat\\varepsilon_{{hard}}$ = {S['eps_hard']:.2e}   "
            f"(raw any-layer $\\hat\\varepsilon$ = {S['eps_hat']:.2f}, "
            f"tie-saturated)",
            xy=(0.02, 0.95), xycoords="axes fraction", va="top",
            fontsize=9, color=INK,
        )
        ax.legend(loc="upper right")
    else:
        ax.hist(D, bins=bins, log=True, histtype="stepfilled", alpha=0.8,
                color=BLUE)
        ax.annotate(f"$\\hat\\sigma$ = {S['sigma_all']:.1e}",
                    xy=(0.02, 0.95), xycoords="axes fraction", va="top",
                    fontsize=9)
    ax.set_xlabel("D = log p_train − log p_infer")
    ax.set_ylabel("token count (log)")
    ax.set_title(f"V1 — distribution of D ({S['arch']})")
    _save(fig, figdir, f"V1_hist_{S['arch']}")


def v2_qq(tok, S, is_moe, figdir):
    D = tok["D"].to_numpy()
    if is_moe:
        D = D[~tok["hard_flip"].to_numpy()]
    n = min(len(D), 200_000)
    idx = np.random.RandomState(0).choice(len(D), n, replace=False)
    d = np.sort(D[idx])
    q = sps.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    mu, sd = np.median(d), (np.median(np.abs(d - np.median(d))) / 0.6745)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(q, d, ".", ms=2, color=BLUE, rasterized=True)
    ax.plot(q, mu + sd * q, "-", color=MUTED, lw=1.2,
            label="normal (median/MAD fit)")
    ax.set_xlabel("normal quantiles")
    ax.set_ylabel("D | no hard flip, quantiles" if is_moe else "D quantiles")
    ax.set_title(f"V2 — QQ of consistent-subgroup D ({S['arch']})")
    ax.legend(loc="upper left")
    _save(fig, figdir, f"V2_qq_{S['arch']}")


def v3_margin(tok, S, figdir):
    mmin = tok["margin_min_infer"].to_numpy()
    absD = np.abs(tok["D"].to_numpy())
    flip = tok["hard_flip"].to_numpy()
    eps_floor = max(absD[absD > 0].min() if (absD > 0).any() else 1e-12, 1e-12)
    y = np.log10(np.maximum(absD, eps_floor))
    x = np.log10(np.maximum(mmin, 1e-6))
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(x, y, gridsize=60, bins="log", cmap="Blues",
                   mincnt=1, linewidths=0.1)
    if flip.any():
        # subsample the overlay: millions of opaque points would blanket
        # the hexbin underneath
        idx = np.nonzero(flip)[0]
        if len(idx) > 30000:
            idx = np.random.RandomState(1).choice(idx, 30000, replace=False)
        ax.plot(x[idx], y[idx], ".", ms=2, color=RED, alpha=0.2,
                label=f"hard-flipped (subsample of {int(flip.sum())})",
                rasterized=True)
        ax.legend(loc="upper right")
    fig.colorbar(hb, ax=ax, label="token count (log)")
    ax.set_xlabel("log10 margin_min (min over layers, infer side)")
    ax.set_ylabel("log10 |D|")
    ax.set_title(f"V3 — routing margin vs |D| ({S['arch']})")
    _save(fig, figdir, f"V3_margin_{S['arch']}")


def v4_ccdf(tok, S, is_moe, figdir):
    D = tok["D"].to_numpy()
    ks = np.logspace(0.001, np.log10(np.exp(min(np.abs(D).max(), 30))), 200)
    logk = np.log(ks)
    n = len(D)
    up = np.array([(D > c).sum() / n for c in logk])
    dn = np.array([(D < -c).sum() / n for c in logk])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(ks, up, "-", color=BLUE, label="P(K > k)  (upper tail)")
    ax.loglog(ks, dn, "-", color=RED, label="P(K < 1/k)  (lower tail)")
    ax.loglog(ks, np.minimum(1.0, 1.0 / ks), "--", color=MUTED, lw=1.2,
              label="Markov bound 1/k")
    if is_moe:
        ax.annotate(
            f"hard: $\\hat\\varepsilon_+$={S['eps_hard_plus']:.1e}  "
            f"$\\hat\\varepsilon_-$={S['eps_hard_minus']:.1e}",
            xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9,
        )
    ax.set_xlabel("k   (K = p_train / p_infer = $e^D$)")
    ax.set_ylabel("tail probability")
    ax.set_title(f"V4 — two-sided CCDF of K ({S['arch']})")
    ax.legend(loc="lower left")
    _save(fig, figdir, f"V4_ccdf_{S['arch']}")


def v5_sequence(trj, S, is_moe, figdir):
    fig, axes = plt.subplots(1, 2 if is_moe else 1,
                             figsize=(11 if is_moe else 6, 4.2))
    axes = np.atleast_1d(axes)
    sumD = trj["sum_D"].to_numpy()
    span = np.quantile(np.abs(sumD), 0.999)
    ax = axes[0]
    ax.hist(np.clip(sumD, -span, span), bins=120, log=True,
            histtype="stepfilled", alpha=0.8, color=BLUE)
    ax.set_xlabel("$\\sum_t D_t$ per trajectory")
    ax.set_ylabel("trajectory count (log)")
    ax.set_title("V5a — sequence-level aggregate drift")
    if is_moe:
        ax = axes[1]
        T = trj["T"].to_numpy()
        anyf = trj["any_hard_flip"].to_numpy().astype(float)
        edges = np.unique(np.quantile(T, np.linspace(0, 1, 25)).astype(int))
        mids, fracs = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (T >= a) & (T < b)
            if m.sum() >= 20:
                mids.append(T[m].mean())
                fracs.append(anyf[m].mean())
        eps = S["eps_hard"]
        tgrid = np.linspace(1, max(T), 200)
        ax.plot(tgrid, 1 - (1 - eps) ** tgrid, "--", color=MUTED, lw=1.4,
                label="$1-(1-\\hat\\varepsilon_{hard})^T$")
        ax.plot(mids, fracs, "o", ms=5, color=RED, label="observed")
        ax.set_xlabel("trajectory length T (generated tokens)")
        ax.set_ylabel("P(trajectory contains a hard flip)")
        ax.set_ylim(0, 1.02)
        ax.set_title("V5b — flip contamination vs length")
        ax.legend(loc="lower right")
    _save(fig, figdir, f"V5_sequence_{S['arch']}")


def v6_phase(S, figdir):
    per = S["per_layer"]
    sig = np.array([p["sigma_L"] for p in per])
    odds = np.array([max(p["odds_L"], 1e-12) for p in per])
    Ls = np.array([p["L"] for p in per])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(sig, odds, "-o", ms=4, color=BLUE, label="layer grid L=1..24")
    for i in [0, len(Ls) // 2, len(Ls) - 1]:
        ax.annotate(f"L={Ls[i]}", (sig[i], odds[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=MUTED)
    lim = np.array([min(sig.min(), odds.min()) * 0.5,
                    max(sig.max(), odds.max()) * 2])
    ax.plot(lim, lim, "--", color=MUTED, lw=1.2,
            label="reference boundary  $\\varepsilon/(1-\\varepsilon)=\\sigma$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\hat\\sigma_L$  (MAD sigma, consistent up to layer L)")
    ax.set_ylabel("$\\hat\\varepsilon_L/(1-\\hat\\varepsilon_L)$")
    ax.set_title(
        f"V6 — phase locus ({S['arch']}); "
        f"$\\hat L$={S['L_hat']}, $\\hat m^*$={S['m_star_hat']:.2e}"
    )
    ax.legend(loc="best")
    _save(fig, figdir, f"V6_phase_{S['arch']}")


def v7_controls(S, figdir):
    conds, sigmas = [], []
    if "sigma_hat" in S:
        conds.append("cross-engine\nBF16 (main)")
        sigmas.append(S["sigma_hat"])
    elif "sigma_all" in S:
        conds.append("cross-engine\nBF16 (main)")
        sigmas.append(S["sigma_all"])
    if "fp32_minus_bf16_recompute" in S:
        conds.append("recompute\nFP32 vs BF16")
        sigmas.append(S["fp32_minus_bf16_recompute"]["sigma_MAD"])
    if "C1_prefill_minus_decode" in S:
        conds.append("same-engine\nprefill vs decode (C1)")
        sigmas.append(S["C1_prefill_minus_decode"]["sigma_MAD"])
    if "C2_determinism_floor" in S:
        conds.append("recompute twice\n(C2 floor)")
        sigmas.append(S["C2_determinism_floor"]["sigma_MAD"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ypos = np.arange(len(conds))
    vals = np.array(sigmas, dtype=float)
    vals_plot = np.maximum(vals, 1e-12)
    ax.barh(ypos, vals_plot, height=0.55, color=BLUE)
    ax.set_yticks(ypos, conds, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("$\\hat\\sigma_{MAD}$ of the log-prob delta")
    for y, v in zip(ypos, vals):
        ax.annotate(f"{v:.1e}", (max(v, 1e-12), y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=INK)
    ax.set_title(f"V7 — gap attribution across controls ({S['arch']})")
    _save(fig, figdir, f"V7_controls_{S['arch']}")


def main():
    arch = sys.argv[1]
    is_moe = MODELS[arch]["is_moe"]
    adir = os.path.join(DATA_ROOT, "analysis")
    figdir = os.path.join(adir, "figs")
    os.makedirs(figdir, exist_ok=True)
    tok = pq.read_table(os.path.join(adir, f"tokens_{arch}.parquet")).to_pandas()
    trj = pq.read_table(os.path.join(adir, f"trajs_{arch}.parquet")).to_pandas()
    S = json.load(open(os.path.join(adir, f"stats_{arch}.json")))

    v1_histogram(tok, S, is_moe, figdir)
    v2_qq(tok, S, is_moe, figdir)
    if is_moe:
        v3_margin(tok, S, figdir)
    v4_ccdf(tok, S, is_moe, figdir)
    v5_sequence(trj, S, is_moe, figdir)
    if is_moe:
        v6_phase(S, figdir)
    v7_controls(S, figdir)
    print(f"figures for {arch} -> {figdir}")


if __name__ == "__main__":
    main()
