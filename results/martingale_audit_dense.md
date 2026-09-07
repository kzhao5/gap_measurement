# A-tier martingale audit -- dense (12,389,115 tokens)

### E1a -- mean-identity residuals r_b = mu + sigma^2/2 (8 most negative + 3 most positive of 40)

| bucket | n | mu_hat | sigma2_hat | r_b | r_b 95%CI | flip_rate |
|---|---|---|---|---|---|---|
| p0xt0 | 2.929e+05 | -0.003525 | 0.001306 | -0.002872 | [-3.12e-03,-2.63e-03] | 0 |
| p0xt1 | 3.047e+05 | -0.003017 | 0.001154 | -0.00244 | [-2.67e-03,-2.21e-03] | 0 |
| p0xt3 | 3.257e+05 | -0.002865 | 0.0009531 | -0.002388 | [-2.59e-03,-2.18e-03] | 0 |
| p0xt2 | 3.156e+05 | -0.002845 | 0.001001 | -0.002345 | [-2.56e-03,-2.13e-03] | 0 |
| p1xt1 | 3.079e+05 | -0.0007859 | 0.0003539 | -0.0006089 | [-7.62e-04,-4.56e-04] | 0 |
| p1xt0 | 3.167e+05 | -0.0007444 | 0.0004314 | -0.0005287 | [-6.84e-04,-3.74e-04] | 0 |
| p1xt2 | 3.072e+05 | -0.0006486 | 0.0003038 | -0.0004967 | [-6.46e-04,-3.47e-04] | 0 |
| p1xt3 | 3.071e+05 | -0.0005569 | 0.0002731 | -0.0004203 | [-5.68e-04,-2.73e-04] | 0 |
| p3xt0 | 3.319e+05 | 0.0002654 | 1.544e-05 | 0.0002732 | [2.42e-04,3.05e-04] | 0 |
| p2xt1 | 3.085e+05 | 0.0002942 | 8.993e-05 | 0.0003392 | [2.59e-04,4.19e-04] | 0 |
| p2xt0 | 3.283e+05 | 0.0002897 | 0.0001096 | 0.0003445 | [2.65e-04,4.24e-04] | 0 |

> sig r<0: 19/40, sig r>0: 21/40. Martingale prediction: all r<=0; drift map = sig-negative buckets.

drift-map overlap: mean flip-rate in sig-negative buckets = 0.000 vs all-bucket mean 0.000

### E1b -- clean-stratum regression mu_b ~ sigma_b^2

| n_clean_buckets | slope | 95%CI | contains -1/2 |
|---|---|---|---|
| 40 | -2.792 | [-2.957,-2.627] | FAIL |

> Acceptance: CI contains -1/2 (lognormal-core mean law).

### E3 -- within-decile position slope (sqrt-law deconfounding)

| p decile | dlog(sigma)/dlog(pos) | 95%CI |
|---|---|---|
| p-decile 0 | -0.07646 | [-0.095,-0.058] |
| p-decile 1 | -0.1064 | [-0.119,-0.093] |
| p-decile 2 | -0.114 | [-0.137,-0.091] |
| p-decile 3 | -0.1126 | [-0.137,-0.088] |
| p-decile 4 | -0.09354 | [-0.134,-0.053] |
| p-decile 5 | -0.08623 | [-0.116,-0.056] |
| p-decile 6 | -0.0654 | [-0.077,-0.054] |
| p-decile 7 | -0.0276 | [-0.069,0.014] |
| p-decile 8 | 1.456e-06 | [-0.000,0.000] |
| p-decile 9 | 4.095e-14 | [-0.000,0.000] |

> median slope = -0.081; verdict: position term is an artifact of composition -> drop pos from sigma_hat.

### E2a -- shape before/after standardization (sigma_hat = c1(1-p), c1=0.000)

| coord | q999/MAD | kurtosis | dBIC(1-2) | P(|x|>4sigma) |
|---|---|---|---|---|
| raw D | 2442 | 16.08 | 7.11e+06 | 0.4011 |
| standardized s | 2442 | 16.08 | 1.208e+07 | 0.4011 |

> fake-tail death = big drops in q999/MAD, kurtosis, tail freq.

### E2b -- residual tail |s|>4 composition

| side | n_tail | hard_flip_share | baseline |
|---|---|---|---|
| + | 3.862e+06 | 0 | 0.533 |
| - | 3.982e+06 | 0 | 0.533 |

> share >> 0.533 => residual tail is the true (jump) branch.

### E2c -- GPD refit in s coordinates (psi=1%)

| side | xi_s | beta_s | xi 95%CI |
|---|---|---|---|
| + | 0.1092 | 1.994e+04 | [0.104,0.114] |
| - | 0.175 | 1.809e+04 | [0.171,0.181] |

> compare with raw-D xi (moe: +0.146/-0.275).

### E4a -- per-position identity E[K|pos]=1

| pos bin | n | E[K|pos] | 95%CI | contains 1 |
|---|---|---|---|---|
| t~1 | 4e+04 | 1 | [0.9996,1.0003] | yes |
| t~3 | 4e+04 | 0.9999 | [0.9997,1.0002] | yes |
| t~7 | 1e+05 | 0.9999 | [0.9997,1.0000] | yes |
| t~13 | 1.4e+05 | 1 | [0.9999,1.0002] | yes |
| t~24 | 3e+05 | 1 | [0.9999,1.0001] | yes |
| t~44 | 5e+05 | 1 | [1.0000,1.0001] | yes |
| t~78 | 8.794e+05 | 1 | [0.9999,1.0000] | yes |
| t~140 | 1.591e+06 | 1 | [1.0000,1.0001] | yes |
| t~249 | 2.676e+06 | 1 | [1.0000,1.0001] | yes |
| t~432 | 3.611e+06 | 1 | [1.0000,1.0000] | yes |
| t~722 | 2.511e+06 | 1 | [1.0000,1.0000] | yes |

> valid y-free conditioning.

### E4b -- accumulation variance vs independent sum

| Var(S) actual | E[sum Var(D_t|pos)] | 2*sum Cov / 2 | rho_infl |
|---|---|---|---|
| 0.4007 | 0.4008 | -6.25e-05 | 0.9997 |

> rho~1 => iid approximation exempt; rho>>1 => n_eff discount.

### E4c -- within-trajectory autocorrelation of D

| tau | corr(all) | corr(flip origin) | corr(clean origin) |
|---|---|---|---|
| 1 | 8.063e-05 | nan | nan |
| 2 | -0.0001745 | nan | nan |
| 4 | 0.0002158 | nan | nan |
| 8 | 0.0001244 | nan | nan |
| 16 | 0.0005929 | nan | nan |
| 32 | -0.0001111 | nan | nan |

> flip-origin elevation = temporal fingerprint of cascades.

### E5 -- sequence-level numbers

| E[W] | E[W^2] | ESS/n | exp(T*sigma_MAD^2) pred | E[W^2]/E[W]^2 actual |
|---|---|---|---|---|
| 1 | 1.493 | 0.6699 | 1 | 1.493 |

> Tbar=619; prediction uses pooled MAD^2 (iid lognormal).

### E6 -- band audit

| band | mask_rate | flip_share_in_mask | far-tail(|D|>3.5) capture | drift-mass leak |
|---|---|---|---|---|
| folk [0.5,5] | 8.072e-08 | 0 | nan | 0.9999 |
| derived [0.72,1.29] (+cap2 noted) | 2.421e-05 | 0 | nan | 0.9954 |
| s-coord |s|>4 | 0.6332 | 0 | nan | 6.742e-10 |

> leak = compensator mass sum(e^D-1-D) left unmasked / total.

### E1-ent (addendum) -- y-free acceptance: entropy-decile x pos buckets

sig r<0: 26/40, sig r>0: 1/40 (marginal; worst pos 2.4e-06) => compensator
nonnegativity PASS. Slope -1.155 +/- 0.274 (no flip stratification available;
steepness attributed to comb-jump compensator).
