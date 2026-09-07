# K_t model selection -- moe, split=time
(w0 = 0.01846 zero-atom removed; n_cont = 12066191; fit/eval = 9623760/2442431; bootstrap = n/a (time split))

### Table 0 -- qualification audit (empirical E[K]=1.0000)

| model | implied_E[K] | E[K]_emp_CI | gate1 | markov_viol | gate2 |
|---|---|---|---|---|---|
| M1_gauss | 1 | [nan,nan] | PASS | 0 | PASS |
| M2_laplace | 1.001 | [nan,nan] | PASS | 0 | PASS |
| M3_student_t | inf | [nan,nan] | FAIL | 4 | FAIL |
| M4_t_pareto | 1 | [nan,nan] | PASS | 0 | PASS |
| M5_emp_gpd | inf | [nan,nan] | FAIL | 0 | PASS |
| M6_gmm | 1 | [nan,nan] | PASS | 0 | PASS |
| M7_condvar | 1.002 | [nan,nan] | PASS | 0 | PASS |

> gate1: implied E[K] finite and in [0.9,1.1]; gate2: 0 Markov violations on k grid ['2', '5', '20', '20.1', '1.1e+03', '4.42e+05'].

### Table 1 -- tail quantile calibration, cell = ln(Q_model/Q_emp)

| model | +0.01 | -0.01 | +0.001 | -0.001 | +0.0001 | -0.0001 | +1e-05 | -1e-05 | mean|err| |
|---|---|---|---|---|---|---|---|---|---|
| M1_gauss | -0.1149 | -0.3023 | -0.6862 | -0.9684 | -1.1 | -1.53 | -1.432 | -2.038 | 1.022 |
| M2_laplace | -0.5501 | -0.7782 | -0.9476 | -1.26 | -1.234 | -1.69 | -1.466 | -2.094 | 1.252 |
| M3_student_t | 12.89 | 12.66 | 26.43 | 26.12 | 40.23 | 39.77 | 54.16 | 53.53 | 33.22 |
| M4_t_pareto | 0.08664 | 0.1152 | -0.1278 | -0.1619 | -0.3402 | -0.5097 | -0.5314 | -0.8692 | 0.3427 |
| M5_emp_gpd | 0.05281 | 0.05938 | 0.04434 | 0.05614 | 0.03006 | 0.06943 | 0.04121 | 0.1135 | 0.05836 |
| M6_gmm | 0.2908 | 0.261 | 0.05587 | -0.1459 | -0.2518 | -0.6237 | -0.5318 | -1.09 | 0.4064 |
| M7_condvar | -0.2137 | -0.4418 | -0.6086 | -0.9213 | -0.9331 | -1.389 | -1.211 | -1.838 | 0.9446 |

> empirical bootstrap log-widths: +0.01:w=nan -0.01:w=nan +0.001:w=nan -0.001:w=nan +0.0001:w=nan -0.0001:w=nan +1e-05:w=nan -1e-05:w=nan

### Table 2 -- CCDF at empirical side-|D| levels (log10)

| side_lev | x | log10_emp | M1_gauss | M2_laplace | M3_student_t | M4_t_pareto | M5_emp_gpd | M6_gmm | M7_condvar | log10_markov |
|---|---|---|---|---|---|---|---|---|---|---|
| +0.5 | 0.002612 | -0.62 | -0.3271 | -0.3309 | -0.7861 | -0.7846 | -0.6277 | -0.6203 | -0.6246 | -0.001134 |
| +0.9 | 0.1034 | -1.319 | -0.8486 | -1.482 | -1.042 | -1.304 | -1.305 | -1.358 | -1.334 | -0.04491 |
| +0.99 | 0.3565 | -2.319 | -3.77 | -4.374 | -1.127 | -2.295 | -2.263 | -2.032 | -3.224 | -0.1548 |
| +0.999 | 0.749 | -3.319 | -13.42 | -8.858 | -1.179 | -3.833 | -3.25 | -3.446 | -7.856 | -0.3253 |
| +0.9999 | 1.309 | -4.319 | -38.37 | -15.25 | -1.218 | -6.027 | -4.26 | -6.73 | -19.07 | -0.5684 |
| +0.99999 | 2.085 | -5.318 | -94.99 | -24.12 | -1.25 | -9.068 | -5.263 | -13.86 | -44.35 | -0.9055 |
| -0.5 | 0.002782 | -0.5848 | -0.2943 | -0.3328 | -0.7905 | -0.7775 | -0.5899 | -0.5903 | -0.6264 | 0.02391 |
| -0.9 | 0.1145 | -1.284 | -0.8594 | -1.609 | -1.049 | -1.286 | -1.263 | -1.301 | -1.394 | -0.02461 |
| -0.99 | 0.439 | -2.284 | -5.08 | -5.316 | -1.142 | -2.219 | -2.229 | -2.05 | -4.015 | -0.1655 |
| -0.999 | 1.043 | -3.284 | -24.42 | -12.21 | -1.202 | -3.967 | -3.21 | -4.544 | -13.01 | -0.4277 |
| -0.9999 | 2.11 | -4.284 | -96.34 | -24.4 | -1.251 | -7.057 | -4.162 | -13.3 | -45.34 | -0.8911 |
| -0.99999 | 4.139 | -5.282 | -300 | -47.59 | -1.298 | -12.93 | -5.153 | -45.88 | -166.4 | -1.772 |

> markov: right side exact e^-x from E[K]=1; left side plug-in Ehat[1/K]*e^-x with Ehat[1/K]=1.0596 [P9].

### Table 3 -- downstream threshold relative errors (PRIMARY)

| model | m*+0.001 | m*-0.001 | m*+0.01 | m*-0.01 | m*+0.1 | m*-0.1 | C_100 | C_1000 | C_10000 | ph0.001 | ph0.01 | ph0.1 | mean|rel| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1_gauss | 0.4965 | 0.6203 | 0.1085 | 0.2609 | 1.68 | 1.272 | 0.4536 | 0.6852 | 0.8143 | 24.09 | 24.09 | 24.09 | 6.555 |
| M2_laplace | 0.6123 | 0.7165 | 0.4231 | 0.5408 | 0.3175 | 0.03743 | 0.6114 | 0.7417 | 0.8296 | 8.47 | 8.47 | 8.47 | 2.52 |
| M3_student_t | 3.011e+11 | 2.202e+11 | 3.962e+05 | 3.154e+05 | 0.2241 | 0.03618 | 75.9 | 34.43 | 16.87 | 0.9355 | 0.9355 | 0.9355 | 4.344e+10 |
| M4_t_pareto | 0.1199 | 0.1495 | 0.0905 | 0.1221 | 0.5923 | 0.624 | 0.01939 | 0.2562 | 0.4725 | 0.9038 | 0.9038 | 0.9038 | 0.4298 |
| M5_emp_gpd | 0.04533 | 0.05775 | 0.05423 | 0.06118 | 0.0003693 | 0.01856 | 0.14 | 0.1297 | 0.1303 | 0.2645 | 0.2645 | 0.2645 | 0.1192 |
| M6_gmm | 0.05746 | 0.1357 | 0.3375 | 0.2983 | 0.1271 | 0.05919 | 0.08802 | 0.265 | 0.5373 | 0.003185 | 0.003185 | 0.003185 | 0.1596 |
| M7_condvar | 0.4559 | 0.602 | 0.1924 | 0.3571 | 0.08059 | 0.2761 | 0.4558 | 0.6463 | 0.7762 | 0.4468 | 0.4468 | 0.4468 | 0.4319 |

> emp: sigma_L=0.004125, C_n={100: 0.5075, 1000: 1.1016, 10000: 2.1841}; defs per P8.

### Table 4 -- overall fit (eval split)

| model | eval_meanNLL | BIC | PIT_KS | QQ_slope_c | QQ_slope_t |
|---|---|---|---|---|---|
| M1_gauss | -1.112 | -1.685e+07 | 0.2724 | 22.91 | 0.1693 |
| M2_laplace | -1.743 | -3.035e+07 | 0.212 | 8.687 | 0.2269 |
| M3_student_t | -2.655 | -5.955e+07 | 0.1265 | 0.04734 | 4.79e+23 |
| M4_t_pareto | -2.97 | -6.556e+07 | 0.112 | 0.06563 | 0.6617 |
| M5_emp_gpd | -2.921 | -6.03e+07 | 0.07532 | 0.6478 | 1.353 |
| M6_gmm | -2.517 | -4.976e+07 | 0.1145 | 0.8497 | 0.6064 |
| M7_condvar | 19.11 | 1.22e+13 | 0.07285 | 0.4134 | 0.2794 |

> M7 NLL is conditional (per-bucket); others marginal. QQ slopes: model-vs-empirical quantiles, central q in [.25,.75], tail q>.99.

### Table 5 -- Hill/GPD threshold scan

| psi | u+ | a+ | a+_CI | xi+ | xi+_CI | n+ | u- | a- | a-_CI | xi- | xi-_CI | n- |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.002 | 0.4875 | 5.143 | [nan,nan] | 0.1391 | [nan,nan] | 2.413e+04 | 0.6451 | 3.019 | [nan,nan] | 0.2481 | [nan,nan] | 2.413e+04 |
| 0.005 | 0.3516 | 6.055 | [nan,nan] | 0.1768 | [nan,nan] | 6.033e+04 | 0.4462 | 3.847 | [nan,nan] | 0.2648 | [nan,nan] | 6.033e+04 |
| 0.01 | 0.2578 | 6.646 | [nan,nan] | 0.1455 | [nan,nan] | 1.207e+05 | 0.3238 | 4.623 | [nan,nan] | 0.2753 | [nan,nan] | 1.207e+05 |
| 0.02 | 0.1854 | 7.843 | [nan,nan] | 0.1984 | [nan,nan] | 2.413e+05 | 0.2238 | 5.608 | [nan,nan] | 0.2964 | [nan,nan] | 2.413e+05 |
| 0.05 | 0.1005 | 9.602 | [nan,nan] | 0.2482 | [nan,nan] | 6.033e+05 | 0.1175 | 7.183 | [nan,nan] | 0.2869 | [nan,nan] | 6.033e+05 |

> plateau verdict (adjacent a CIs overlap, both sides): PARTIAL(0/8).

### Table 6a -- bucket normality

| AD_rej@5% | n_buckets_used | median_KS_p | worst5 |
|---|---|---|---|
| 1.000 | 32 | 6.29e-161 | b3:A2=12605.1 b15:A2=10804.2 b19:A2=10213.1 b31:A2=8725.1 b35:A2=8475.2 |

> AD case-0 critical 2.492 for rejection; p column is KS (P10).

### Table 6b -- sigma(bucket) shape fits (P10 numeric)

| shape | c_hat | R2_weighted |
|---|---|---|
| const | 0.03576 | 0 |
| c(1-p) | 0.1554 | 0.9529 |
| c*sqrt((1-p)/p) | 0.06639 | 0.9404 |

> sigma(bucket) range: [4.4e-06, 0.15], 64-bucket table in json.

### Table 6c -- implied tail index cross-check

| nu_eq(invGamma) | nu_eq_CI | M4_nu | M4_compat | M5_a+ | M5_a- |
|---|---|---|---|---|---|
| 0.1631 | [0.139,0.211] | 0.05 | incompatible | 6.646 | 4.623 |

> nu_eq: inv-gamma MLE on bucket sigma^2 (scale-mixture => t_nu); M5 a on the K axis is a different tail family -- report side by side.

### Table 7 -- two-sided asymmetry dossier (u* = |D| q99 = 0.3964)

| side | eps_hat(u*) | eps_CI | a_hat | a_CI | xi_hat | xi_CI | u(psi=1%) |
|---|---|---|---|---|---|---|---|
| + | 0.003586 | [nan,nan] | 6.646 | [nan,nan] | 0.1455 | [nan,nan] | 0.2578 |
| - | 0.006414 | [nan,nan] | 4.623 | [nan,nan] | 0.2753 | [nan,nan] | 0.3238 |
| -/+ | 1.789 | [nan,nan] |  |  |  |  |  |

> eps at common threshold u*; a/xi at per-side psi=1% thresholds.

### Verdict

- algorithm champion (T3 primary): **M5_emp_gpd** (runner-up M6_gmm; T1 best M5_emp_gpd)
- narrative champion (T4 NLL among gate passers ['M1_gauss', 'M2_laplace', 'M4_t_pareto', 'M6_gmm', 'M7_condvar']): **M4_t_pareto**
- ellipse verdict: AD_rej=1.000 (>=0.10), 6c incompatible => NOT established
