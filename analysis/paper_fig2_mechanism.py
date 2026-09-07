"""Fig 2 (paper): mechanism figure = V9 fan + core-out (|Z|>5) scatter overlay
+ three threshold geometries: TIS constant (log 2), IcePop mask [log 0.5, log 5],
and ours +lambda_+ * max(1-p, eps0) (one-sided). English labels."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ARCH = sys.argv[1] if len(sys.argv) > 1 else "moe"
ROOT = "/home/kzhao2/gap_measurement"; FIG = f"{ROOT}/results/figs/paper"; OUT = f"{ROOT}/results/paper"
EPS0, KAPPA, LAM = 5e-3, 5, 2.3
BLUE, INK, GRAY, ORANGE, RED = "#1f77b4", "#0b0b0b", "#8a8a8a", "#ff7f0e", "#d03b3b"
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.grid": True, "grid.color": "#ececec",
                     "axes.spines.top": False, "axes.spines.right": False})
t = pd.read_parquet(f"{ROOT}/results/tokens_{ARCH}.parquet", columns=["logp_train", "D"])
D = t["D"].to_numpy(np.float64); u = np.clip(1 - np.exp(t["logp_train"].to_numpy(np.float64)), 1e-7, 1)
phi = np.maximum(u, EPS0)
c = json.load(open(f"{OUT}/params_{ARCH}.json"))["c"] if os.path.exists(f"{OUT}/params_{ARCH}.json") else 0.156
Z = D / (c * phi); out = np.abs(Z) > KAPPA
edges = np.unique(np.quantile(u, np.linspace(0, 1, 37))); idx = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(edges) - 2)
nb = len(edges) - 1; QS = [.001, .01, .1, .5, .9, .99, .999]
ux = np.array([np.median(u[idx == b]) for b in range(nb)]); quant = np.array([np.quantile(D[idx == b], QS) for b in range(nb)])
fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(ux, quant[:, 0], quant[:, 6], color=BLUE, alpha=.13, lw=0)
ax.fill_between(ux, quant[:, 1], quant[:, 5], color=BLUE, alpha=.30, lw=0)
ax.fill_between(ux, quant[:, 2], quant[:, 4], color=BLUE, alpha=.55, lw=0, label="aleatoric fan (80/98/99.8%)")
ax.plot(ux, quant[:, 3], color=INK, lw=1.5, label="conditional median")
rng = np.random.RandomState(0); oi = np.nonzero(out)[0]; oi = rng.choice(oi, size=min(len(oi), 20000), replace=False)
ax.scatter(u[oi], D[oi], s=3, color=ORANGE, alpha=.35, label=rf"core-out tokens ($|Z|>{KAPPA}$), epistemic layer")
xs = np.geomspace(1e-7, 1, 300)
ax.axhline(np.log(2.0), color=GRAY, ls="--", lw=1.4, label="TIS cap: log 2")
ax.axhline(np.log(5.0), color=GRAY, ls=":", lw=1.4, label="IcePop mask: [log 0.5, log 5]"); ax.axhline(np.log(0.5), color=GRAY, ls=":", lw=1.4)
ax.plot(xs, LAM * np.maximum(xs, EPS0), color=RED, lw=2.2, label=rf"ours: $+\lambda_+\max(1-p,\varepsilon_0)$, $\lambda_+$={LAM}, $\varepsilon_0$={EPS0}")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-3, linscale=0.5); ax.set_ylim(-8, 8); ax.set_xlim(1e-7, 1.2)
ax.set_xlabel(r"token uncertainty $1-p_t$ (right = less confident)"); ax.set_ylabel(r"$\log k_t$")
ax.set_title(f"Fig 2 mechanism ({ARCH}, {len(D):,} tokens): aleatoric fan, epistemic residual layer, three threshold geometries")
ax.legend(fontsize=8, loc="lower left"); fig.tight_layout()
fig.savefig(f"{FIG}/fig2_mechanism_{ARCH}.png"); print("wrote", f"{FIG}/fig2_mechanism_{ARCH}.png", "c=", c, "core-out frac", out.mean())
