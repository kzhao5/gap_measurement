# A-tier martingale audit -- moe (12,293,115 tokens)

### E1a -- mean-identity residuals r_b = mu + sigma^2/2 (8 most negative + 3 most positive of 40)

| bucket | n | mu_hat | sigma2_hat | r_b | r_b 95%CI | flip_rate |
|---|---|---|---|---|---|---|
| p0xt0 | 2.295e+05 | -0.06125 | 0.0356 | -0.04345 | [-4.47e-02,-4.22e-02] | 0.4072 |
| p0xt1 | 2.711e+05 | -0.03903 | 0.02497 | -0.02655 | [-2.75e-02,-2.56e-02] | 0.4182 |
| p1xt0 | 2.64e+05 | -0.02643 | 0.0186 | -0.01713 | [-1.79e-02,-1.63e-02] | 0.4263 |
| p0xt2 | 3.283e+05 | -0.02392 | 0.01938 | -0.01423 | [-1.49e-02,-1.36e-02] | 0.393 |
| p1xt1 | 2.894e+05 | -0.01761 | 0.01393 | -0.01065 | [-1.13e-02,-1.00e-02] | 0.4423 |
| p0xt3 | 4.004e+05 | -0.01586 | 0.01353 | -0.009099 | [-9.58e-03,-8.62e-03] | 0.3667 |
| p1xt2 | 3.229e+05 | -0.01075 | 0.0104 | -0.005544 | [-6.02e-03,-5.07e-03] | 0.4211 |
| p1xt3 | 3.53e+05 | -0.007076 | 0.007679 | -0.003237 | [-3.62e-03,-2.86e-03] | 0.3991 |
| p4xt0 | 3.07e+05 | 0.003654 | 0.0002546 | 0.003782 | [3.66e-03,3.91e-03] | 0.488 |
| p3xt1 | 3.033e+05 | 0.003127 | 0.001366 | 0.00381 | [3.59e-03,4.03e-03] | 0.4911 |
| p3xt0 | 3.019e+05 | 0.004396 | 0.001879 | 0.005335 | [5.07e-03,5.60e-03] | 0.465 |

> sig r<0: 8/40, sig r>0: 30/40. Martingale prediction: all r<=0; drift map = sig-negative buckets.

drift-map overlap: mean flip-rate in sig-negative buckets = 0.409 vs all-bucket mean 0.533

### E1b -- clean-stratum regression mu_b ~ sigma_b^2

| n_clean_buckets | slope | 95%CI | contains -1/2 |
|---|---|---|---|
| 14 | -1.798 | [-1.938,-1.658] | FAIL |

> Acceptance: CI contains -1/2 (lognormal-core mean law).

### E3 -- within-decile position slope (sqrt-law deconfounding)

| p decile | dlog(sigma)/dlog(pos) | 95%CI |
|---|---|---|
| p-decile 0 | -0.2107 | [-0.266,-0.155] |
| p-decile 1 | -0.1962 | [-0.253,-0.140] |
| p-decile 2 | -0.1879 | [-0.226,-0.150] |
| p-decile 3 | -0.1907 | [-0.228,-0.153] |
| p-decile 4 | -0.1933 | [-0.240,-0.146] |
| p-decile 5 | -0.1905 | [-0.246,-0.135] |
| p-decile 6 | -0.1923 | [-0.252,-0.133] |
| p-decile 7 | -0.1686 | [-0.228,-0.109] |
| p-decile 8 | -0.1175 | [-0.159,-0.076] |
| p-decile 9 | 0.3165 | [0.275,0.358] |

> median slope = -0.191; verdict: position term is an artifact of composition -> drop pos from sigma_hat.

### E2a -- shape before/after standardization (sigma_hat = c1(1-p), c1=0.194)

| coord | q999/MAD | kurtosis | dBIC(1-2) | P(|x|>4sigma) |
|---|---|---|---|---|
| raw D | 255.8 | 199.9 | 6.035e+06 | 0.3462 |
| standardized s | 19.87 | 3.968e+06 | 9.624e+06 | 0.02786 |

> fake-tail death = big drops in q999/MAD, kurtosis, tail freq.

### E2b -- residual tail |s|>4 composition

| side | n_tail | hard_flip_share | baseline |
|---|---|---|---|
| + | 2.581e+05 | 0.8002 | 0.533 |
| - | 4.288e+04 | 0.8362 | 0.533 |

> share >> 0.533 => residual tail is the true (jump) branch.

### E2c -- GPD refit in s coordinates (psi=1%)

| side | xi_s | beta_s | xi 95%CI |
|---|---|---|---|
| + | 0.5643 | 2.84 | [0.555,0.573] |
| - | 0.2112 | 0.7328 | [0.205,0.219] |

> compare with raw-D xi (moe: +0.146/-0.275).

### E4a -- per-position identity E[K|pos]=1

| pos bin | n | E[K|pos] | 95%CI | contains 1 |
|---|---|---|---|---|
| t~1 | 4e+04 | 0.9999 | [0.9990,1.0008] | yes |
| t~3 | 4e+04 | 0.9999 | [0.9990,1.0008] | yes |
| t~7 | 9.998e+04 | 1 | [0.9998,1.0010] | yes |
| t~13 | 1.399e+05 | 1 | [0.9994,1.0008] | yes |
| t~24 | 2.996e+05 | 1 | [0.9999,1.0007] | yes |
| t~44 | 4.99e+05 | 1 | [0.9997,1.0005] | yes |
| t~78 | 8.76e+05 | 1 | [0.9998,1.0004] | yes |
| t~140 | 1.57e+06 | 1 | [0.9999,1.0002] | yes |
| t~249 | 2.569e+06 | 0.9999 | [0.9998,1.0000] | yes |
| t~433 | 3.453e+06 | 1 | [0.9999,1.0001] | yes |
| t~733 | 2.706e+06 | 1 | [0.9999,1.0001] | yes |

> valid y-free conditioning.

### E4b -- accumulation variance vs independent sum

| Var(S) actual | E[sum Var(D_t|pos)] | 2*sum Cov / 2 | rho_infl |
|---|---|---|---|
| 6.849 | 5.546 | 0.6514 | 1.235 |

> rho~1 => iid approximation exempt; rho>>1 => n_eff discount.

### E4c -- within-trajectory autocorrelation of D

| tau | corr(all) | corr(flip origin) | corr(clean origin) |
|---|---|---|---|
| 1 | 0.003802 | 0.004456 | 0.002708 |
| 2 | 0.002436 | 0.003375 | 0.0009562 |
| 4 | 0.001089 | 0.001209 | 0.0008955 |
| 8 | 0.001517 | 0.001618 | 0.001354 |
| 16 | 0.001012 | 0.001452 | 0.0003252 |
| 32 | 0.0009959 | 0.001071 | 0.00087 |

> flip-origin elevation = temporal fingerprint of cascades.

### E5 -- sequence-level numbers

| E[W] | E[W^2] | ESS/n | exp(T*sigma_MAD^2) pred | E[W^2]/E[W]^2 actual |
|---|---|---|---|---|
| 0.9386 | 56.76 | 0.01552 | 1.008 | 64.42 |

> Tbar=615; prediction uses pooled MAD^2 (iid lognormal).

### E6 -- band audit

| band | mask_rate | flip_share_in_mask | far-tail(|D|>3.5) capture | drift-mass leak |
|---|---|---|---|---|
| folk [0.5,5] | 0.001657 | 0.7943 | 1 | 0.8196 |
| derived [0.72,1.29] (+cap2 noted) | 0.01954 | 0.654 | 1 | 0.3803 |
| s-coord |s|>4 | 0.02448 | 0.8053 | 1 | 0.6757 |

> leak = compensator mass sum(e^D-1-D) left unmasked / total.

### E1-ent (addendum) -- y-free acceptance: entropy-decile x pos buckets

| test | result | verdict |
|---|---|---|
| compensator nonnegativity (all r_b <= 0) | sig r<0: 32/40, sig r>0: **0/40** (worst pos point 1.6e-06) | **PASS** |
| E1b slope, all buckets | -1.194 [-1.429,-0.959] | steeper than -1/2 (jump compensator co-varies with H, as model predicts) |
| E1b slope, clean stratum (low-flip tercile, 14 buckets) | **-0.732 [-1.202,-0.262], contains -1/2** | **PASS** |

> The p-bucket E1 failure is a y-conditioning (winner's-curse) artifact; with
> y-free buckets the martingale mean decomposition is accepted.
