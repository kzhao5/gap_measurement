"""Protocol section-3 statistics + section-5 verdict table inputs.

Reads analysis/tokens_{arch}.parquet + trajs_{arch}.parquet, emits
stats_{arch}.json and a human-readable summary.

Statistics (protocol section 3):
  sigma_hat   = MAD(D | flip=0) / 0.6745
  eps_hat     = mean(flip)
  eps_plus / eps_minus  = one-sided flip rates (by sign of D)
  Delta_gap   = q05(|D| | flip=1) - q999(|D| | flip=0)   (spectral gap)
  sigma_l     = per-layer-grid sigma (consistent-up-to-layer-L subgroup)
  eps_l odds  = eps_L / (1 - eps_L), flip within layers <= L
  seq level   = P(traj contains flip) vs 1-(1-eps)^T theory
  controls    = C1 / C2 / fp32 deltas (sigma_MAD each)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import DATA_ROOT, MODELS

Q = lambda x, q: float(np.quantile(x, q)) if len(x) else float("nan")


def mad_sigma(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))) / 0.6745)


def main():
    arch = sys.argv[1]
    spec = MODELS[arch]
    is_moe = spec["is_moe"]
    adir = os.path.join(DATA_ROOT, "analysis")
    tok = pq.read_table(os.path.join(adir, f"tokens_{arch}.parquet")).to_pandas()
    trj = pq.read_table(os.path.join(adir, f"trajs_{arch}.parquet")).to_pandas()

    D = tok["D"].to_numpy()
    s = {
        "arch": arch,
        "n_tokens": int(len(tok)),
        "n_trajs": int(len(trj)),
        "D_mean": float(D.mean()),
        "D_median": float(np.median(D)),
        "sigma_all": mad_sigma(D),
        "absD_q999": Q(np.abs(D), 0.999),
        "absD_max": float(np.abs(D).max()),
    }

    if is_moe:
        flip = tok["flip"].to_numpy()
        cons, flp = D[~flip], D[flip]
        eps = float(flip.mean())

        # --- hard-flip statistics (tie-aware; primary mechanism variable).
        # Raw any-layer flips saturate at eps~0.92 because bf16 gate probs
        # tie exactly at the top-k boundary; hard flips require a flipped
        # layer with margin > 1e-3.
        hflip = tok["hard_flip"].to_numpy()
        hcons, hflp = D[~hflip], D[hflip]
        heps = float(hflip.mean())
        s.update({
            "eps_hard": heps,
            "eps_hard_plus": float((hflip & (D > 0)).mean()),
            "eps_hard_minus": float((hflip & (D < 0)).mean()),
            "sigma_hat_hard": mad_sigma(hcons),
            "n_hard_flip": int(hflip.sum()),
            "absD_hardflip_q05": Q(np.abs(hflp), 0.05),
            "absD_hardflip_med": Q(np.abs(hflp), 0.5),
            "absD_hardcons_med": Q(np.abs(hcons), 0.5),
            "absD_hardcons_q999": Q(np.abs(hcons), 0.999),
            "mean_tie_layers": float(tok["n_tie_layers"].mean()),
        })
        s["Delta_gap_hard"] = (
            s["absD_hardflip_q05"] - s["absD_hardcons_q999"]
        )
        s["gap_over_sigma_hard"] = (
            s["Delta_gap_hard"] / s["sigma_hat_hard"]
            if s["sigma_hat_hard"] else float("nan")
        )
        thr_h = s["absD_hardcons_q999"]
        outlier_h = np.abs(D) > thr_h
        if outlier_h.any():
            s["p_hardflip_given_outlier"] = float(hflip[outlier_h].mean())
        anyhf = trj["any_hard_flip"].to_numpy()
        Th = trj["T"].to_numpy()
        s["frac_traj_with_hard_flip"] = float(anyhf.mean())
        s["frac_traj_with_hard_flip_theory"] = float(
            np.mean(1 - (1 - heps) ** Th)
        )
        s.update({
            "eps_hat": eps,
            "eps_plus": float((flip & (D > 0)).mean()),
            "eps_minus": float((flip & (D < 0)).mean()),
            "sigma_hat": mad_sigma(cons),
            "n_flip": int(flip.sum()),
            # spectral gap: 5% quantile of flipped |D| minus 99.9% of consistent
            "absD_flip_q05": Q(np.abs(flp), 0.05),
            "absD_cons_q999": Q(np.abs(cons), 0.999),
            "mean_flip_layers": float(tok.loc[flip, "n_flip_layers"].mean())
            if flip.any() else 0.0,
            "margin_min_median_flip": float(
                tok.loc[flip, "margin_min_infer"].median()
            ) if flip.any() else float("nan"),
            "margin_min_median_cons": float(
                tok.loc[~flip, "margin_min_infer"].median()
            ),
        })
        s["Delta_gap"] = s["absD_flip_q05"] - s["absD_cons_q999"]
        s["gap_over_sigma"] = (
            s["Delta_gap"] / s["sigma_hat"] if s["sigma_hat"] else float("nan")
        )
        # relative strength rho = eps / ((1-eps) * sigma)
        s["rho_strength"] = eps / max((1 - eps) * s["sigma_hat"], 1e-30)

        # V1 verdict ingredients: P(flip | outlier)
        thr = s["absD_cons_q999"]
        outlier = np.abs(D) > thr
        if outlier.any():
            s["p_flip_given_outlier"] = float(flip[outlier].mean())

        # per-layer grid (V6): HARD flips restricted to layers <= L
        L = spec["num_layers"]
        bits = tok["hard_flip_bits"].to_numpy().astype(np.uint64)
        per_layer = []
        for l in range(L):
            mask_l = np.uint64((1 << (l + 1)) - 1)
            flip_le = (bits & mask_l) != 0
            eps_l = float(flip_le.mean())
            per_layer.append({
                "L": l + 1,
                "eps_L": eps_l,
                "odds_L": eps_l / max(1 - eps_l, 1e-30),
                "sigma_L": mad_sigma(D[~flip_le]),
                "eps_layer_only": float(((bits >> l) & np.uint64(1)).mean()),
            })
        s["per_layer"] = per_layer
        s["L_hat"] = int(
             1 + int(np.argmax([p["eps_layer_only"] for p in per_layer]))
        )
        # m*: characteristic gate-margin scale of genuine flips
        s["m_star_hat"] = (
            float(tok.loc[hflip, "margin_flip_max"].median())
            if hflip.any() else float("nan")
        )

        # sequence level (V5)
        anyf = trj["any_flip"].to_numpy()
        T = trj["T"].to_numpy()
        s["frac_traj_with_flip"] = float(anyf.mean())
        s["frac_traj_with_flip_theory"] = float(
            np.mean(1 - (1 - eps) ** T)
        )

    # Controls
    for col, name in [
        ("logp_c1", "C1_prefill_minus_decode"),
        ("logp_train_fp32", "fp32_minus_bf16_recompute"),
    ]:
        if col in tok.columns:
            sub = tok.dropna(subset=[col])
            if len(sub):
                base = (
                    sub["logp_infer"] if col == "logp_c1" else sub["logp_train"]
                )
                d = (sub[col] - base).to_numpy()
                s[name] = {
                    "n": int(len(sub)),
                    "mean": float(d.mean()),
                    "sigma_MAD": mad_sigma(d),
                    "max_abs": float(np.abs(d).max()),
                }
                if is_moe:
                    f = sub["flip"].to_numpy()
                    s[name]["sigma_MAD_consistent"] = mad_sigma(d[~f])
    if "logp_c2a" in tok.columns:
        sub = tok.dropna(subset=["logp_c2a", "logp_c2b"])
        d = (sub["logp_c2a"] - sub["logp_c2b"]).to_numpy()
        s["C2_determinism_floor"] = {
            "n": int(len(sub)),
            "nonzero_frac": float((d != 0).mean()),
            "sigma_MAD": mad_sigma(d),
            "max_abs": float(np.abs(d).max()) if len(d) else float("nan"),
        }

    out = os.path.join(adir, f"stats_{arch}.json")
    with open(out, "w") as f:
        json.dump(s, f, indent=2)
    print(json.dumps({k: v for k, v in s.items() if k != "per_layer"},
                     indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
