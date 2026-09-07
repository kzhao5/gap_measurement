#!/usr/bin/env python
"""V9 -- THE key figure, Chinese-annotated, self-contained.

Top: empirical conditional distribution of log k_t vs token uncertainty
     (quantile fan) with the two threshold geometries overlaid.
Bottom: the scale law sigma = c1*(1-p_t) across six decades (log-log).

Usage: python analysis/key_figure.py [moe|dense]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

font_manager.fontManager.addfont(
    "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf")
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]

ARCH = sys.argv[1] if len(sys.argv) > 1 else "moe"
LOCAL = "/home/kzhao2/gap_measurement/results"
FIGS = "/home/kzhao2/nobackup/autodelete/gap_measurement/analysis/figs"
BLUE, INK, GRAY, ORANGE, RED = ("#2a78d6", "#0b0b0b", "#8a8a8a",
                                "#b0651a", "#d03b3b")

t = pd.read_parquet(f"{LOCAL}/tokens_{ARCH}.parquet",
                    columns=["logp_train", "D"])
D = t["D"].to_numpy(np.float64)
u = np.clip(1.0 - np.exp(t["logp_train"].to_numpy(np.float64)), 1e-7, 1.0)

edges = np.unique(np.quantile(u, np.linspace(0, 1, 37)))
idx = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(edges) - 2)
nb = len(edges) - 1
QS = [0.001, 0.01, 0.10, 0.50, 0.90, 0.99, 0.999]
ux, sig = np.empty(nb), np.empty(nb)
quant = np.empty((nb, len(QS)))
for b in range(nb):
    m = idx == b
    ux[b] = np.median(u[m])
    quant[b] = np.quantile(D[m], QS)
    sig[b] = 1.4826 * np.median(np.abs(D[m] - quant[b][3])) + 1e-15
w = sig > 1e-12
c1 = float(np.exp(np.mean(np.log(sig[w]) - np.log(ux[w]))))
g999 = float(np.quantile(np.abs(D), 0.999))
KZ = 8  # example adaptive threshold in z units (frontier optimum region)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d9d9d9", "axes.grid": True,
    "grid.color": "#ececec", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 12.5, "figure.dpi": 150,
})
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True,
                              height_ratios=[1.15, 1])

# ---- top: fan + threshold geometries ---------------------------------
ax.fill_between(ux, quant[:, 0], quant[:, 6], color=BLUE, alpha=0.13, lw=0)
ax.fill_between(ux, quant[:, 1], quant[:, 5], color=BLUE, alpha=0.30, lw=0)
ax.fill_between(ux, quant[:, 2], quant[:, 4], color=BLUE, alpha=0.55, lw=0)
ax.plot(ux, quant[:, 3], color=INK, lw=2.0)
ax.axhline(g999, color=GRAY, ls="--", lw=1.6)
ax.axhline(-g999, color=GRAY, ls="--", lw=1.6)
ax.plot(ux, KZ * c1 * ux, color=RED, ls="-", lw=2.0)
ax.plot(ux, -KZ * c1 * ux, color=RED, ls="-", lw=2.0)
ax.set_xscale("log")
yl = float(np.quantile(np.abs(D), 0.99999)) * 1.35
ax.set_ylim(-yl, yl)
ax.set_ylabel(r"$\log k_t$", fontsize=14)
ax.set_title("log k_t 的真实条件分布:宽度完全由 token 不确定度决定"
             f"({ARCH}, {len(D):,} tokens)", fontsize=15, pad=12)
# annotations
ax.annotate("蓝带 = 数据真实所在:深→浅为中间 80% / 98% / 99.8% 的 token",
            xy=(ux[-4], quant[-4, 5]), xytext=(2e-5, yl * 0.62),
            fontsize=11.5, color="#1f5089",
            arrowprops=dict(arrowstyle="->", color="#1f5089", lw=1.2))
ax.annotate("TIS / IcePop:常数阈值(与不确定度无关)\n"
            "→ 对 99.9% 的 token 形同虚设,只『看见』最右端",
            xy=(3e-4, g999), xytext=(1.1e-7, g999 * 0.12),
            fontsize=11.5, color="#555555",
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))
ax.annotate("KPop 几何 / 我们的阈值:±κ·c₁(1−p_t),κ=8\n"
            "→ 跟着分布宽度走,每个 token 都被同样力度约束",
            xy=(ux[-7], KZ * c1 * ux[-7]), xytext=(4e-6, -yl * 0.66),
            fontsize=11.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.text(1.1e-7, -yl * 0.97, "黑线 = 条件中位数(≈0:失配无整体漂移,"
        "但见下联:宽度变化 10⁶ 倍)", fontsize=10.5, color=INK)

# ---- bottom: scale law ----------------------------------------------
ax2.plot(ux[w], sig[w], "o", ms=6, color=BLUE, label="实测宽度 σ(每桶)")
ax2.plot(ux, c1 * ux, ls="--", color=INK, lw=1.8,
         label=f"σ = c₁·(1−p_t),c₁={c1:.3f}(对数最小二乘)")
ax2.axhline(g999, color=GRAY, ls="--", lw=1.6, label="常数阈值(TIS/IcePop)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("token 不确定度 1 − p_t,其中 p_t = π_old(y_t | h_t)"
               "(右 = 越不确定)", fontsize=13.5)
ax2.set_ylabel("分布宽度 σ_MAD", fontsize=13.5)
ax2.set_title("尺度律:宽度 ∝ (1 − p_t),横跨六个数量级 —— "
              "KPop 的门限几何被数据证实", fontsize=14, pad=10)
ax2.legend(loc="upper left", fontsize=11.5, framealpha=0.95)
ax2.annotate("常数阈值在这里比分布宽度大 10⁵ 倍\n(永远不触发)",
             xy=(2e-4, g999), xytext=(2e-4, g999 * 2e-3),
             fontsize=10.5, color="#555555", ha="center",
             arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
ax2.annotate("只有最右端\n两者才相遇", xy=(0.75, g999 * 0.8),
             xytext=(0.1, g999 * 0.04), fontsize=10.5, color="#555555",
             ha="center",
             arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGS, f"V9_key_{ARCH}.{ext}"))
print(f"wrote {FIGS}/V9_key_{ARCH}.png  c1={c1:.4f}  q999={g999:.3f}")
