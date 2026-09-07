# Gap measurement verdict (pre-registered table)

- tokens: 12,293,115  trajectories: 20,000
- raw any-layer flip eps = 0.921 (SATURATED by exact bf16 gate-prob ties; mean tie layers/token = 1.91)
- HARD flip (margin>1e-3): eps_hard = 5.334e-01 (+2.50e-01 / -2.71e-01)
- sigma_hat_hard = 7.712e-03   sigma_all = 3.542e-03
- Delta_gap_hard = -6.966e-01  (gap/sigma = -90.3)

| check | observed | vote |
|---|---|---|
| R1 mixture + spectral gap | dBIC(1->2)=1519571, gap/sigma(hard)=-90.3 | **H-var** |
| R2 P(hard flip|outlier) / eps_hard | 0.793 / 5.33e-01 = 1x | **H-var** |
| R3 margin->flip mechanism | AUC=0.544, median|D| ratio=0x | **H-var** |
| R4 tail asymmetry eps- vs eps+ | n-=3336124, n+=3077221, binom p=0.0e+00 | **H-flip** |

## VERDICT: **H-var**  (1/4 rows H-flip)

-> H-var world: shrink adoption domain, variance-layer (Gated-LCB / Tier C) path starts first.
- control C1_prefill_minus_decode: {"n": 634402, "mean": -0.004143836442381144, "sigma_MAD": 0.0033283806405961514, "max_abs": 12.386866569519043, "sigma_MAD_consistent": 0.014659915119409561}
- control C2_determinism_floor: {"n": 622574, "nonzero_frac": 0.0, "sigma_MAD": 0.0, "max_abs": 0.0}
- control fp32_minus_bf16_recompute: {"n": 12293115, "mean": 0.0005386021221056581, "sigma_MAD": 0.004360486287623644, "max_abs": 13.247023582458496, "sigma_MAD_consistent": 0.016695266589522362}
