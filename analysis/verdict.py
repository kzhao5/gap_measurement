"""Pre-registered verdict table (protocol section 5) -> report.md.

Rows (as pre-registered):
  R1  V1 mixture: 2-vs-1 component dBIC on D, and gap Delta_hat/sigma_hat > 5
  R2  V1 in-band density: P(flip | outlier) >> eps_hat
  R3  V3 mechanism: flipped tokens sit at low margin & high |D|
      (AUC of -margin_min predicting flip; median |D| ratio)
  R4  V4 asymmetry: eps_plus significantly > eps_minus (binomial)

Majority of rows -> H-flip vs H-var; emits plug-in parameters
(sigma_hat, eps_hat, eps_s, L_hat, m_star_hat) on H-flip.
"""

import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy import stats as sps
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import DATA_ROOT

GATE_GAP_OVER_SIGMA = 5.0     # pre-registered gate (protocol suggestion)
GATE_DBIC = 10.0              # strong evidence threshold
GATE_OUTLIER_RATIO = 10.0     # P(flip|outlier) / eps_hat
GATE_AUC = 0.75


def main():
    adir = os.path.join(DATA_ROOT, "analysis")
    S = json.load(open(os.path.join(adir, "stats_moe.json")))
    tok = pq.read_table(
        os.path.join(adir, "tokens_moe.parquet"),
        columns=["D", "flip", "hard_flip", "margin_min_infer",
                 "margin_flip_max"],
    ).to_pandas()

    D = tok["D"].to_numpy()
    # Mechanism variable: HARD flips (routing decision differs at a
    # clearly positive gate margin). Raw any-layer flips saturate at
    # eps~0.92 due to exact bf16 gate-prob ties and carry no
    # discriminative power; both are reported in stats_moe.json.
    flip = tok["hard_flip"].to_numpy()
    EPS_KEY = "eps_hard"
    rows = []

    # ---- R1: mixture evidence + spectral gap
    rng = np.random.RandomState(0)
    sub = D[rng.choice(len(D), min(len(D), 500_000), replace=False)]
    x = sub.reshape(-1, 1)
    bic1 = GaussianMixture(1, random_state=0).fit(x).bic(x)
    bic2 = GaussianMixture(2, random_state=0).fit(x).bic(x)
    dbic = bic1 - bic2  # >0 favors 2 components
    gap_ratio = S.get("gap_over_sigma_hard", float("nan"))
    r1 = (dbic > GATE_DBIC) and (gap_ratio > GATE_GAP_OVER_SIGMA)
    rows.append(("R1 mixture + spectral gap",
                 f"dBIC(1->2)={dbic:.0f}, gap/sigma(hard)={gap_ratio:.1f}",
                 "H-flip" if r1 else "H-var"))

    # ---- R2: hard-flip concentration among outliers
    eps_h = S[EPS_KEY]
    ratio = S.get("p_hardflip_given_outlier", 0.0) / max(eps_h, 1e-30)
    r2 = ratio > GATE_OUTLIER_RATIO
    rows.append(("R2 P(hard flip|outlier) / eps_hard",
                 f"{S.get('p_hardflip_given_outlier', float('nan')):.3f} / "
                 f"{eps_h:.2e} = {ratio:.0f}x",
                 "H-flip" if r2 else "H-var"))

    # ---- R3: mechanism (margin predicts flip; |D| separation)
    if flip.any():
        m = tok["margin_min_infer"].to_numpy()
        n_neg = min((~flip).sum(), 200_000)
        neg = m[~flip][rng.choice((~flip).sum(), n_neg, replace=False)]
        pos = m[flip]
        auc = sps.mannwhitneyu(-pos, -neg).statistic / (len(pos) * len(neg))
        dmed_ratio = np.median(np.abs(D[flip])) / max(
            np.median(np.abs(D[~flip])), 1e-30
        )
        r3 = (auc > GATE_AUC) and (dmed_ratio > GATE_GAP_OVER_SIGMA)
        rows.append(("R3 margin->flip mechanism",
                     f"AUC={auc:.3f}, median|D| ratio={dmed_ratio:.0f}x",
                     "H-flip" if r3 else "H-var"))
    else:
        rows.append(("R3 margin->flip mechanism", "no flips observed", "H-var"))

    # ---- R4: one-sided asymmetry (P5), hard flips
    npl = int(S["eps_hard_plus"] * S["n_tokens"])
    nmn = int(S["eps_hard_minus"] * S["n_tokens"])
    if npl + nmn > 0:
        p = sps.binomtest(nmn, npl + nmn, 0.5).pvalue
        r4 = (nmn > npl) and p < 1e-3  # protocol: flips deflate p_train side
        rows.append(("R4 tail asymmetry eps- vs eps+",
                     f"n-={nmn}, n+={npl}, binom p={p:.1e}",
                     "H-flip" if r4 else "H-var"))
    else:
        rows.append(("R4 tail asymmetry", "no tail tokens", "H-var"))

    n_flip_votes = sum(1 for _, _, v in rows if v == "H-flip")
    verdict = "H-flip" if n_flip_votes >= (len(rows) + 1) // 2 else "H-var"

    lines = [
        "# Gap measurement verdict (pre-registered table)", "",
        f"- tokens: {S['n_tokens']:,}  trajectories: {S['n_trajs']:,}",
        f"- raw any-layer flip eps = {S['eps_hat']:.3f} (SATURATED by exact "
        f"bf16 gate-prob ties; mean tie layers/token = "
        f"{S.get('mean_tie_layers', float('nan')):.2f})",
        f"- HARD flip (margin>1e-3): eps_hard = {S['eps_hard']:.3e} "
        f"(+{S['eps_hard_plus']:.2e} / -{S['eps_hard_minus']:.2e})",
        f"- sigma_hat_hard = {S['sigma_hat_hard']:.3e}   "
        f"sigma_all = {S['sigma_all']:.3e}",
        f"- Delta_gap_hard = {S['Delta_gap_hard']:.3e}  (gap/sigma = "
        f"{S['gap_over_sigma_hard']:.1f})", "",
        "| check | observed | vote |", "|---|---|---|",
    ]
    for name, obs, vote in rows:
        lines.append(f"| {name} | {obs} | **{vote}** |")
    lines += [
        "",
        f"## VERDICT: **{verdict}**  ({n_flip_votes}/{len(rows)} rows H-flip)",
        "",
    ]
    if verdict == "H-flip":
        lines += [
            "Plug-in first-batch parameters (hard-flip basis):",
            f"- sigma_hat = {S['sigma_hat_hard']:.4e}",
            f"- eps_hat   = {S['eps_hard']:.4e}",
            f"- frac trajs w/ hard flip = {S['frac_traj_with_hard_flip']:.4f} "
            f"(theory {S['frac_traj_with_hard_flip_theory']:.4f})",
            f"- L_hat     = {S['L_hat']}",
            f"- m_star_hat= {S['m_star_hat']:.4e}",
            "-> proceed to the training sub-protocol (correlation gate -> A/B).",
        ]
    else:
        lines += [
            "-> H-var world: shrink adoption domain, variance-layer "
            "(Gated-LCB / Tier C) path starts first.",
        ]
    for key in ("C1_prefill_minus_decode", "C2_determinism_floor",
                "fp32_minus_bf16_recompute"):
        if key in S:
            lines.append(f"- control {key}: {json.dumps(S[key])}")

    out = os.path.join(adir, "report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
