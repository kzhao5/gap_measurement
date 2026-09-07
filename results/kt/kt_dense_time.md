# K_t model selection -- dense, split=time
(w0 = 0.11587 zero-atom removed; n_cont = 10953573; fit/eval = 8716632/2236941; bootstrap = n/a (time split))

### Table 0 -- qualification audit (empirical E[K]=1.0000)

| model | implied_E[K] | E[K]_emp_CI | gate1 | markov_viol | gate2 |
|---|---|---|---|---|---|
| M1_gauss | 1 | [nan,nan] | PASS | 0 | PASS |
| M2_laplace | 1 | [nan,nan] | PASS | 0 | PASS |
| M3_student_t | inf | [nan,nan] | FAIL | 2 | FAIL |
| M4_t_pareto | 1 | [nan,nan] | PASS | 0 | PASS |
| M5_emp_gpd | inf | [nan,nan] | FAIL | 0 | PASS |
| M6_gmm | 1 | [nan,nan] | PASS | 0 | PASS |
| M7_condvar | 1 | [nan,nan] | PASS | 0 | PASS |

> gate1: implied E[K] finite and in [0.9,1.1]; gate2: 0 Markov violations on k grid ['2', '5', '20', '20.1', '1.1e+03', '4.42e+05'].

### Table 1 -- tail quantile calibration, cell = ln(Q_model/Q_emp)

| model | +0.01 | -0.01 | +0.001 | -0.001 | +0.0001 | -0.0001 | +1e-05 | -1e-05 | mean|err| |
|---|---|---|---|---|---|---|---|---|---|
| M1_gauss | -0.4531 | -0.4741 | -0.567 | -0.613 | -0.8357 | -0.8488 | -0.9556 | -1.013 | 0.72 |
| M2_laplace | -0.8637 | -0.8961 | -0.8002 | -0.8547 | -0.9396 | -0.9598 | -0.9576 | -1.021 | 0.9116 |
| M3_student_t | 12.32 | 12.29 | 26.93 | 26.87 | 41.48 | 41.46 | 56.23 | 56.17 | 34.22 |
| M4_t_pareto | -0.01488 | -0.008532 | 0.2217 | 0.188 | 0.1528 | 0.1467 | 0.1731 | 0.1202 | 0.1282 |
| M5_emp_gpd | 0.009296 | 0.005854 | 0.01628 | -0.01518 | -0.02823 | 0.0221 | 0.1271 | 0.2196 | 0.05545 |
| M6_gmm | -0.08236 | -0.0894 | -0.07909 | -0.1166 | -0.3031 | -0.3098 | -0.3996 | -0.4517 | 0.229 |
| M7_condvar | -0.7518 | -0.7842 | -0.6225 | -0.6771 | -0.7992 | -0.8194 | -0.8695 | -0.9332 | 0.7821 |

> empirical bootstrap log-widths: +0.01:w=nan -0.01:w=nan +0.001:w=nan -0.001:w=nan +0.0001:w=nan -0.0001:w=nan +1e-05:w=nan -1e-05:w=nan

### Table 2 -- CCDF at empirical side-|D| levels (log10)

| side_lev | x | log10_emp | M1_gauss | M2_laplace | M3_student_t | M4_t_pareto | M5_emp_gpd | M6_gmm | M7_condvar | log10_markov |
|---|---|---|---|---|---|---|---|---|---|---|
| +0.5 | 0.0002681 | -0.6416 | -0.3091 | -0.3119 | -0.7847 | -0.7196 | -0.6364 | -0.3551 | -0.5568 | -0.0001164 |
| +0.9 | 0.04094 | -1.341 | -1.184 | -1.96 | -1.12 | -1.345 | -1.335 | -1.233 | -1.842 | -0.01778 |
| +0.99 | 0.1154 | -2.341 | -4.938 | -4.974 | -1.189 | -2.199 | -2.343 | -2.485 | -4.459 | -0.0501 |
| +0.999 | 0.1854 | -3.34 | -11.26 | -7.81 | -1.22 | -3.003 | -3.505 | -4.408 | -8.392 | -0.08051 |
| +0.9999 | 0.2449 | -4.34 | -18.83 | -10.22 | -1.239 | -3.686 | -4.182 | -6.643 | -12.85 | -0.1063 |
| +0.99999 | 0.3445 | -5.332 | -36.05 | -14.25 | -1.262 | -4.829 | -5.009 | -11.65 | -22.76 | -0.1496 |
| -0.5 | 7.563e-05 | -0.5658 | -0.2974 | -0.3041 | -0.7002 | -0.6284 | -0.5211 | -0.313 | -0.5061 | 0.0002817 |
| -0.9 | 0.03614 | -1.265 | -1.021 | -1.765 | -1.111 | -1.274 | -1.26 | -1.152 | -1.717 | -0.01538 |
| -0.99 | 0.1148 | -2.265 | -4.846 | -4.95 | -1.188 | -2.148 | -2.278 | -2.42 | -4.432 | -0.04953 |
| -0.999 | 0.1862 | -3.265 | -11.28 | -7.844 | -1.221 | -2.968 | -3.425 | -4.358 | -8.448 | -0.08055 |
| -0.9999 | 0.2457 | -4.264 | -18.85 | -10.26 | -1.239 | -3.651 | -4.017 | -6.58 | -12.93 | -0.1064 |
| -0.99999 | 0.3491 | -5.261 | -36.84 | -14.44 | -1.263 | -4.837 | -4.727 | -11.78 | -23.3 | -0.1513 |

> markov: right side exact e^-x from E[K]=1; left side plug-in Ehat[1/K]*e^-x with Ehat[1/K]=1.0007 [P9].

### Table 3 -- downstream threshold relative errors (PRIMARY)

| model | m*+0.001 | m*-0.001 | m*+0.01 | m*-0.01 | m*+0.1 | m*-0.1 | C_100 | C_1000 | C_10000 | ph0.001 | ph0.01 | ph0.1 | mean|rel| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1_gauss | 0.4328 | 0.4583 | 0.3643 | 0.3776 | 2.454 | 2.141 | 0.412 | 0.5323 | 0.5961 | 244.6 | 244.6 | 244.6 | 61.8 |
| M2_laplace | 0.5507 | 0.5746 | 0.5784 | 0.5918 | 0.7187 | 0.531 | 0.5646 | 0.6004 | 0.6142 | 97.26 | 97.26 | 97.26 | 24.76 |
| M3_student_t | 4.958e+11 | 4.695e+11 | 2.246e+05 | 2.175e+05 | 0.3236 | 0.3975 | 22.49 | 13.94 | 10.03 | 0.8357 | 0.8357 | 0.8357 | 8.044e+10 |
| M4_t_pareto | 0.2482 | 0.2068 | 0.01477 | 0.008496 | 0.4436 | 0.4575 | 0.1607 | 0.171 | 0.1848 | 0.3569 | 0.3569 | 0.3569 | 0.2472 |
| M5_emp_gpd | 0.01642 | 0.01507 | 0.00934 | 0.005871 | 0.02458 | 0.02712 | 0.02312 | 0.0002464 | 0.1623 | 0.02186 | 0.02186 | 0.02186 | 0.02914 |
| M6_gmm | 0.07605 | 0.11 | 0.07906 | 0.08552 | 0.8254 | 0.835 | 0.07753 | 0.2146 | 0.3003 | 17.49 | 17.49 | 17.49 | 4.59 |
| M7_condvar | 0.4634 | 0.4919 | 0.5285 | 0.5435 | 0.08302 | 0.1831 | 0.4919 | 0.5341 | 0.5701 | 6.032 | 6.032 | 6.032 | 1.832 |

> emp: sigma_L=0.0001121, C_n={100: 0.1277, 1000: 0.2008, 10000: 0.272}; defs per P8.

### Table 4 -- overall fit (eval split)

| model | eval_meanNLL | BIC | PIT_KS | QQ_slope_c | QQ_slope_t |
|---|---|---|---|---|---|
| M1_gauss | -2.231 | -3.801e+07 | 0.3197 | 214.9 | 0.2838 |
| M2_laplace | -2.913 | -4.955e+07 | 0.2795 | 85.16 | 0.3946 |
| M3_student_t | -5.517 | -9.307e+07 | 0.06487 | 0.1622 | 3.884e+24 |
| M4_t_pareto | -5.762 | -9.767e+07 | 0.07066 | 0.4367 | 1.393 |
| M5_emp_gpd | -5.091 | -8.626e+07 | 0.2467 | 1.509 | 1.266 |
| M6_gmm | -3.897 | -6.642e+07 | 0.2286 | 15.99 | 0.5946 |
| M7_condvar | -3.943 | -7.05e+07 | 0.06548 | 4.878 | 0.4277 |

> M7 NLL is conditional (per-bucket); others marginal. QQ slopes: model-vs-empirical quantiles, central q in [.25,.75], tail q>.99.

### Table 5 -- Hill/GPD threshold scan

| psi | u+ | a+ | a+_CI | xi+ | xi+_CI | n+ | u- | a- | a-_CI | xi- | xi-_CI | n- |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.002 | 0.1241 | 27.29 | [nan,nan] | 0.04555 | [nan,nan] | 2.191e+04 | 0.1251 | 23.6 | [nan,nan] | -0.04901 | [nan,nan] | 2.191e+04 |
| 0.005 | 0.1139 | 46.12 | [nan,nan] | 0.5509 | [nan,nan] | 5.477e+04 | 0.116 | 43.09 | [nan,nan] | 0.687 | [nan,nan] | 5.477e+04 |
| 0.01 | 0.09948 | 46.28 | [nan,nan] | 0.1617 | [nan,nan] | 1.095e+05 | 0.1028 | 46.65 | [nan,nan] | 0.2352 | [nan,nan] | 1.095e+05 |
| 0.02 | 0.07626 | 35.74 | [nan,nan] | -0.04359 | [nan,nan] | 2.191e+05 | 0.08063 | 36.89 | [nan,nan] | -0.01591 | [nan,nan] | 2.191e+05 |
| 0.05 | 0.03653 | 26.55 | [nan,nan] | -0.07724 | [nan,nan] | 5.477e+05 | 0.04034 | 26.53 | [nan,nan] | -0.04678 | [nan,nan] | 5.477e+05 |

> plateau verdict (adjacent a CIs overlap, both sides): PARTIAL(0/8).

### Table 6a -- bucket normality

| AD_rej@5% | n_buckets_used | median_KS_p | worst5 |
|---|---|---|---|
| 1.000 | 64 | 0 | b58:A2=7074.2 b50:A2=7040.0 b59:A2=7002.4 b51:A2=6653.3 b49:A2=6602.7 |

> AD case-0 critical 2.492 for rejection; p column is KS (P10).

### Table 6b -- sigma(bucket) shape fits (P10 numeric)

| shape | c_hat | R2_weighted |
|---|---|---|
| const | 0.007585 | 0 |
| c(1-p) | 0.03581 | 0.9861 |
| c*sqrt((1-p)/p) | 0.01284 | 0.9749 |

> sigma(bucket) range: [3.5e-07, 0.036], 64-bucket table in json.

### Table 6c -- implied tail index cross-check

| nu_eq(invGamma) | nu_eq_CI | M4_nu | M4_compat | M5_a+ | M5_a- |
|---|---|---|---|---|---|
| 0.1425 | [0.128,0.16] | 0.05 | incompatible | 46.28 | 46.65 |

> nu_eq: inv-gamma MLE on bucket sigma^2 (scale-mixture => t_nu); M5 a on the K axis is a different tail family -- report side by side.

### Table 7 -- two-sided asymmetry dossier (u* = |D| q99 = 0.115)

| side | eps_hat(u*) | eps_CI | a_hat | a_CI | xi_hat | xi_CI | u(psi=1%) |
|---|---|---|---|---|---|---|---|
| + | 0.004676 | [nan,nan] | 46.28 | [nan,nan] | 0.1617 | [nan,nan] | 0.09948 |
| - | 0.005324 | [nan,nan] | 46.65 | [nan,nan] | 0.2352 | [nan,nan] | 0.1028 |
| -/+ | 1.139 | [nan,nan] |  |  |  |  |  |

> eps at common threshold u*; a/xi at per-side psi=1% thresholds.

### Verdict

- algorithm champion (T3 primary): **M5_emp_gpd** (runner-up M4_t_pareto; T1 best M5_emp_gpd)
- narrative champion (T4 NLL among gate passers ['M1_gauss', 'M2_laplace', 'M4_t_pareto', 'M6_gmm', 'M7_condvar']): **M4_t_pareto**
- ellipse verdict: AD_rej=1.000 (>=0.10), 6c incompatible => NOT established
