#!/usr/bin/env python
"""V10 -- distribution of log k_t at the same generation position t.

Top: quantile fan of D | t (t = generation step, pooled over 20k trajs).
Bottom: sigma_MAD(D | t) with a sqrt(KV-length) reference line.

Usage: python analysis/pos_dist.py [moe|dense]
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
BLUE, INK, GRAY, ORANGE = "#2a78d6", "#0b0b0b", "#8a8a8a", "#b0651a"

t = pd.read_parquet(f"{LOCAL}/tokens_{ARCH}.parquet",
                    columns=["traj_id", "pos", "D", "logp_train"])
P = t.groupby("traj_id")["pos"].transform("min")
trel = (t["pos"] - P).to_numpy(np.int64) + 1   # generation step, 1-based
Pmean = float(P.mean())
D = t["D"].to_numpy(np.float64)
lp = t["logp_train"].to_numpy(np.float64)
band = (lp > -3.0) & (lp < -0.5)   # fixed-uncertainty band (deconfound)

edges = np.unique(np.geomspace(1, trel.max() + 1, 33).astype(int))
idx = np.clip(np.searchsorted(edges, trel, side="right") - 1,
              0, len(edges) - 2)
nb = len(edges) - 1
QS = [0.001, 0.01, 0.10, 0.50, 0.90, 0.99, 0.999]
tx, sig, sigb, cnt = (np.empty(nb) for _ in range(4))
quant = np.empty((nb, len(QS)))
for b in range(nb):
    m = idx == b
    cnt[b] = m.sum()
    if cnt[b] < 500:
        tx[b] = np.nan
        continue
    tx[b] = np.median(trel[m])
    quant[b] = np.quantile(D[m], QS)
    sig[b] = 1.4826 * np.median(np.abs(D[m] - quant[b][3])) + 1e-15
    mb = m & band
    if mb.sum() > 300:
        medb = np.median(D[mb])
        sigb[b] = 1.4826 * np.median(np.abs(D[mb] - medb))
    else:
        sigb[b] = np.nan
ok = np.isfinite(tx)
tx, sig, sigb, quant, cnt = tx[ok], sig[ok], sigb[ok], quant[ok], cnt[ok]

# sqrt(KV) reference fitted in log space: sigma = c * sqrt(Pmean + t)
c_fit = float(np.exp(np.mean(np.log(sig) - 0.5 * np.log(Pmean + tx))))

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d9d9d9", "axes.grid": True,
    "grid.color": "#ececec", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 12.5, "figure.dpi": 150,
})
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True,
                              height_ratios=[1.15, 1])

ax.fill_between(tx, quant[:, 0], quant[:, 6], color=BLUE, alpha=0.13, lw=0,
                label="中间 99.8%(q0.1–q99.9)")
ax.fill_between(tx, quant[:, 1], quant[:, 5], color=BLUE, alpha=0.30, lw=0,
                label="中间 98%(q1–q99)")
ax.fill_between(tx, quant[:, 2], quant[:, 4], color=BLUE, alpha=0.55, lw=0,
                label="中间 80%(q10–q90)")
ax.plot(tx, quant[:, 3], color=INK, lw=2.0, label="条件中位数(≈0)")
ax.set_xscale("log")
yl = float(np.quantile(np.abs(D), 0.99995)) * 1.3
ax.set_ylim(-yl, yl)
ax.set_ylabel("log k_t", fontsize=14)
ax.set_title(f"同一生成位置 t 上 log k_t 的分布({ARCH}, "
             f"{len(D):,} tokens,2 万条轨迹按 t 汇合)", fontsize=15, pad=12)
ax.legend(loc="lower left", fontsize=10.5, framealpha=0.95)
ax.annotate("尾带(浅色)在早中段最宽:那里高熵 token 占比高\n"
            "—— 是『各位置的 token 构成』效应,不是位置本身(见下联拆解)",
            xy=(tx[len(tx) // 3], quant[len(tx) // 3, 6]),
            xytext=(2.5, yl * 0.6),
            fontsize=11.5, color="#1f5089",
            arrowprops=dict(arrowstyle="->", color="#1f5089", lw=1.2))

ax2.plot(tx, sig, "o", ms=6, color=BLUE,
         label="原始 σ(t):全部 token 混合")
okb = np.isfinite(sigb)
ax2.plot(tx[okb], sigb[okb], "s", ms=5.5, color=ORANGE, mfc="none", mew=1.8,
         label="固定不确定度带内 σ(t)(−3<log p<−0.5):纯位置效应")
cb = float(np.exp(np.nanmean(np.log(sigb[okb])
                             - 0.5 * np.log(Pmean + tx[okb]))))
ax2.plot(tx, cb * np.sqrt(Pmean + tx), ls="--", color=INK, lw=1.8,
         label=f"√KV 参考线 c·√({Pmean:.0f}+t):若噪声累积主导应贴此线 → 不支持")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("生成位置 t(第 t 个生成 token,对数轴)", fontsize=13.5)
ax2.set_ylabel("分布宽度 σ_MAD", fontsize=13.5)
ax2.set_title("拆解:固定不确定度后 σ(t) 近乎平坦 —— 位置不是独立驱动因素",
              fontsize=14, pad=10)
ax2.legend(loc="lower left", fontsize=10, framealpha=0.95)
ax2.annotate("蓝点的浴缸形(两端高)全部来自 token 构成:\n"
             "开头高熵 token 多;长轨迹=难题,构成又不同",
             xy=(tx[1], sig[1]), xytext=(1.2, sig[1] * 0.15),
             fontsize=10.5, color="#555555",
             arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
ax2.annotate("橙色:同一不确定度的 token,σ 随位置变化 <2 倍且非单调\n"
             "(对比 V9:不确定度轴上是 10⁶ 倍)→ σ̂ 只需 c₁(1−p_t),无需位置项",
             xy=(tx[okb][len(tx[okb]) // 2], sigb[okb][len(tx[okb]) // 2]),
             xytext=(1.3, sigb[okb][0] * 3.2),
             fontsize=10.5, color=ORANGE,
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))

fig.tight_layout()
os.makedirs(FIGS, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGS, f"V10_pos_{ARCH}.{ext}"))
print(f"wrote {FIGS}/V10_pos_{ARCH}.png  c={c_fit:.3e}  Pmean={Pmean:.0f}  "
      f"sigma[first/last]={sig[0]:.2e}/{sig[-1]:.2e}")
