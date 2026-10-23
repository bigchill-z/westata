## 领域汇总表

| 领域 | 对比点 | pass | fail | error | skip | A | B | C | D | E | 最大系数相对差 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| discrete | 70 | 15 | 39 | 13 | 3 | 11 | 16 | 14 | 1 | 12 | 2.09e+01 |
| limited_quantile | 25 | 0 | 22 | 3 | 0 | 1 | 13 | 2 | 5 | 1 | 5.43e-02 |
| linear_iv | 65 | 46 | 15 | 0 | 4 | 49 | 0 | 0 | 0 | 1 | 3.58e-02 |
| panel | 42 | 22 | 18 | 1 | 1 | 26 | 5 | 2 | 3 | 2 | 2.12e-01 |
| postest | 77 | 45 | 32 | 0 | 0 | 43 | 10 | 3 | 2 | 6 | 7.36e-01 |
| systems_causal | 56 | 36 | 16 | 2 | 2 | 41 | 4 | 1 | 1 | 5 | 1.42e+01 |
| tseries | 56 | 35 | 21 | 0 | 0 | 22 | 7 | 10 | 8 | 5 | 1.71e-02 |
| **合计** | **391** | **199** | **163** | **19** | **10** | **193** | **55** | **32** | **20** | **32** | — |

可判定条目 362：A 193 (53.3%)、B 55、C 32、D 20、E 32、NA 30；另 error 19、skip 10。

## 按领域：A 档（≤1e-9）条目

**discrete**（10 条）

| 命令 | 系数相对差 | SE 相对差 |
|---|---|---|
| logit (default) | 1.09e-13 | 3.56e-14 |
| logit e() scalars | 3.63e-10 | — |
| logit, or (odds ratios + SE) | 8.53e-14 | 1.21e-13 |
| probit e() scalars | 2.67e-10 | — |
| poisson e() scalars | 8.36e-16 | — |
| logit subset (if x1 > 0) | 1.32e-10 | 3.69e-11 |
| ztp e() scalars | 1.29e-15 | — |
| ztp e(ll_0) convention | 1.68e-16 | — |
| logit predict, pr | 2.34e-13 | — |
| logit predict, xb | 2.88e-12 | — |

**limited_quantile**：无。

**linear_iv**（40 条）

| 命令 | 系数相对差 | SE 相对差 |
|---|---|---|
| 1 regress basic: mpg weight foreign | 5.11e-15 | 6.81e-16 |
| 2a regress i.foreign | 3.70e-14 | 5.11e-16 |
| 2b regress i.rep78（5 个缺失两侧都删） | 1.18e-13 | 2.78e-15 |
| 3 regress vce(robust) [westata HC1] | 5.11e-15 | 4.57e-15 |
| 4a regress cluster(gid) 14 组：系数/SE/常规标量 | 3.54e-16 | 1.10e-15 |
| 4c regress cluster：p 值（两边都用 t(G-1)） | 3.54e-16 | 1.10e-15 |
| 5 regress [fweight=w]：系数/SE/df | 4.87e-16 | 3.04e-16 |
| 5 regress [aweight=w]：系数/SE/df | 1.06e-15 | 5.58e-16 |
| 5 regress [pweight=w]：系数/SE/df | 1.06e-15 | 8.86e-16 |
| 5 regress [iweight=w]：系数/SE/df | 4.87e-16 | 3.04e-16 |
| 5g regress [fweight=w] vce(robust)：复制型 meat | 4.87e-16 | 1.59e-16 |
| 5h regress [aweight=w] vce(robust)：精度型 meat | 1.06e-15 | 8.86e-16 |
| 6a regress if mpg>20 | 2.98e-14 | 4.16e-16 |
| 6b regress if weight<4000 & mpg>18 | 3.03e-15 | 6.18e-16 |
| 7a areg absorb(gid) [分类 FE] vs FEOLS | 1.13e-15 | 1.82e-16 |
| 7b areg absorb(gid) vce(robust) | 1.13e-15 | 8.33e-16 |
| 7c-bis areg cluster(gid) vs reghdfe 同规格（归因用） | 9.38e-16 | 7.94e-16 |
| 7d areg absorb(gid) vce(cluster t) [非嵌套] | 7.81e-16 | 1.27e-15 |
| 7e areg absorb(cont) [数值存储、8 个取值] | 5.91e-16 | 3.33e-16 |
| 8a reghdfe absorb(gid) vs FEOLS | 9.38e-16 | 1.86e-16 |
| 8b reghdfe absorb(gid t) 双向 vs FEOLS 双向 | 1.08e-15 | 2.05e-16 |
| 8c reghdfe absorb(gid t) vce(robust) | 1.08e-15 | 3.36e-16 |
| 8d reghdfe absorb(gid t) vce(cluster gid) | 1.08e-15 | 2.10e-16 |
| 9 FEOLS(gid+t, HC1) vs areg absorb(gid)+i.t | 9.43e-16 | 7.84e-16 |
| 10a ivregress 2sls 恰好识别 (e1=z1) | 2.64e-15 | 2.76e-15 |
| 10b ivregress 2sls 过度识别 (e1=z1 z2) | 2.64e-15 | 3.24e-15 |
| 10c ivregress 2sls vce(robust) vs westata HC0（实测正确映射） | 2.64e-15 | 6.10e-15 |
| 10e ivregress 2sls vce(robust) small vs westata HC1 | 2.64e-15 | 6.04e-15 |
| 10f ivregress 2sls vce(cluster gid) | 2.64e-15 | 6.06e-15 |
| 11a ivregress liml 恰好识别 (e1=z1) | 2.35e-15 | 2.04e-15 |
| 11b ivregress liml 过度识别 (e1=z1 z2) | 1.55e-15 | 7.19e-16 |
| 11c ivregress liml vce(robust) vs westata HC0（实测正确映射） | 1.55e-15 | 7.32e-16 |
| 11f ivregress liml vce(cluster gid) small vs westata cluster | 1.55e-15 | 8.68e-16 |
| 12a ivregress gmm 过度识别（两步 GMM，核 e(V)） | 2.20e-16 | 3.67e-16 |
| 12a-bis gmm：e(V) 全矩阵逐元素（12a 的对角已含于 SE） | 2.20e-16 | 3.67e-16 |
| 12b ivregress gmm vce(robust) vs westata HC0（实测正确映射） | 2.20e-16 | 8.15e-16 |
| 13 eivreg reliab(x1 .9 x2 .95) | 4.50e-16 | 6.39e-16 |
| 14a OLS e(): N/df_r/df_m/r2/r2_a/rmse/F/ll | 5.11e-15 | 6.81e-16 |
| 14b OLS e(rss)/e(mss)：westata 未存，靠 resid 反算 | 5.11e-15 | 6.81e-16 |
| 14c 加权 OLS e(N)/e(df_r)/e(r2)：fweight 的 N=sum(w) | 4.87e-16 | 3.04e-16 |

**panel**（22 条）

| 命令 | 系数相对差 | SE 相对差 |
|---|---|---|
| xtset 面板声明 | 0.00e+00 | — |
| xtreg, fe（平衡） | 1.27e-16 | 1.85e-16 |
| xtreg, fe 的 e(r2)=r2_w 与 e(rmse)=sigma_e 数值一致 | 0.00e+00 | 0.00e+00 |
| xtreg, fe（非平衡） | 5.26e-16 | 0.00e+00 |
| xtreg, fd 的等价口径（regress D.y D.x1 D.x2, noconstant） | 4.18e-16 | 1.71e-16 |
| xtreg, be | 4.01e-15 | 4.07e-16 |
| xtreg, re（平衡，默认 GLS） | 6.00e-16 | 3.15e-16 |
| xtreg, re（非平衡） | 2.61e-16 | 2.90e-16 |
| xtreg, re 的 theta→0 边界（sigma_u=0 面板） | 2.64e-16 | 0.00e+00 |
| 归因 A：westata 的 vce(cluster) == Stata 的 vce(cluster id) dfadj | 2.54e-16 | 2.15e-15 |
| 归因 B：手工 CR1 G/(G-1)*(N-1)/(N-K), K=3 复现 Stata 默认 vce(cluster id) | 1.27e-16 | 8.76e-15 |
| 归因 C：linearmodels 同一条 fit 只把 auto_df 换成 True 就复现 Stata | 1.27e-16 | 9.40e-15 |
| 归因 D：westata 的 vce(robust)=HC1 与 Stata areg, absorb() vce(robust) 同口径 | 0.00e+00 | 0.00e+00 |
| xtgls, panels(heteroskedastic) corr(independent) | 1.53e-15 | 5.68e-16 |
| xtgls, panels(correlated)（6 个体 x 8 期，Sigma 满秩） | 2.81e-15 | 6.70e-15 |
| xtpcse（平衡，casewise） | 1.10e-15 | 2.98e-15 |
| xtivreg, fe | 8.23e-15 | 2.04e-15 |
| xtivreg, re | 4.74e-15 | 1.23e-15 |
| xtunitroot ips（有单位根（随机游走 y_t=y_{t-1}+e_t）） | 6.29e-15 | — |
| xtunitroot llc（有单位根（随机游走 y_t=y_{t-1}+e_t）） | 2.72e-15 | — |
| xtunitroot ips（无单位根（AR(1) ρ=0.4）） | 1.45e-14 | — |
| xtunitroot llc（无单位根（AR(1) ρ=0.4）） | 4.83e-16 | — |

**postest**（43 条）

| 命令 | 系数相对差 | SE 相对差 |
|---|---|---|
| sample: regress mpg weight foreign | 0.00e+00 | — |
| sample: logit foreign weight mpg | 0.00e+00 | — |
| sample: poisson rep78 mpg weight | 0.00e+00 | — |
| sample: tobit mpg weight foreign, ll(17) | 0.00e+00 | — |
| test: single coefficient (regress) | 2.66e-16 | — |
| test: joint (regress) | 1.02e-15 | — |
| test: constraint weight=0.5 (regress) | 9.21e-16 | — |
| test: equality weight=foreign (regress) | 1.34e-14 | — |
| test: joint after vce(robust) (regress) | 4.81e-15 | — |
| test: single coefficient (logit, Wald chi2) | 1.91e-10 | — |
| test: joint (logit, Wald chi2) | 4.62e-11 | — |
| test: constraint weight=0 (logit) | 1.91e-10 | — |
| lincom: single coefficient (regress) | 6.81e-16 | — |
| lincom: 2*weight - 3*foreign (regress) | 1.40e-14 | — |
| lincom: level(90) interval (regress) | 6.24e-14 | — |
| lincom: weight + mpg (logit, z) | 7.81e-10 | — |
| nlcom: exp(_b[weight]) (nonlinear, z) | 5.87e-11 | — |
| nlcom: exp(_b[weight]) after logit | 1.13e-11 | — |
| nlcom: ratio SE, analytic delta method as arbiter | 6.79e-12 | — |
| testnl: parenthesised joint form | 5.86e-11 | — |
| testnl: single nonlinear constraint | 8.13e-11 | — |
| lrtest: nested logit (df=1) | 9.11e-15 | — |
| lrtest: nested regress (df=1) | 7.03e-15 | — |
| hausman: xtreg fe vs re (positive definite V_fe - V_re) | 2.60e-13 | — |
| contrast: r. (reference), 2-level factor | 5.38e-15 | — |
| contrast: g. (grand mean), 2-level factor | 5.38e-15 | — |
| contrast: a. (adjacent), 2-level factor | 5.38e-15 | — |
| contrast: r. (3-level factor) | 5.35e-14 | — |
| contrast: g. (3-level factor) | 3.81e-15 | — |
| contrast: a. (3-level factor) | 5.35e-14 | — |
| pwcompare: 3-level factor (unadjusted) | 5.35e-14 | — |
| margins: predictive margin (regress, t form) | 4.08e-12 | — |
| predict poisson: n link check (Stata e(b) reproduces Stata n) | 2.64e-16 | — |
| predict tobit: xb link check (Stata e(b) reproduces Stata xb) | 1.73e-16 | — |
| predict tobit: stdp formula check (Stata e(V) reproduces Stata stdp) | 1.32e-15 | — |
| predict regress: xb | 1.45e-15 | — |
| predict regress: stdp | 3.95e-15 | — |
| predict regress: stdf | 2.56e-16 | — |
| predict regress: residuals | 5.11e-13 | — |
| predict logit: pr | 4.59e-10 | — |
| predict logit: xb | 1.11e-10 | — |
| predict logit: stdp | 6.80e-11 | — |
| etable vs esttab (printing layer) | 0.00e+00 | — |

**systems_causal**（33 条）

| 命令 | 系数相对差 | SE 相对差 |
|---|---|---|
| sureg 2eq (default = small-sample VCE) | 5.75e-13 | 6.09e-15 |
| mvreg 2 outcomes, shared design | 7.18e-15 | 9.12e-14 |
| manova p=2, q=1 (Wilks lambda / Rao F / df) | 1.96e-14 | — |
| manova p=3, q=2 (s=2) | 4.68e-15 | — |
| manova p=3, q=3 (approximate Rao F, fractional df2) | 7.55e-14 | — |
| nl linear-in-parameters (well conditioned) | 2.38e-16 | 8.20e-11 |
| teffects ra ATE | 3.84e-15 | 1.21e-15 |
| teffects ra ATET | 3.28e-14 | 3.40e-16 |
| teffects ipw ATE | 1.04e-15 | 3.59e-16 |
| teffects ipw ATET | 6.20e-16 | 1.11e-15 |
| teffects aipw ATE | 1.11e-16 | 1.08e-15 |
| teffects aipw ATET: Stata's PO formulas reconstructed | 4.41e-16 | — |
| didregress two-period, covariate, vce(cluster) | 2.08e-15 | 9.93e-16 |
| didregress two-period, no covariate, vce(cluster) | 1.81e-16 | 0.00e+00 |
| didregress two-period, vce(robust) | 2.08e-15 | 1.06e-15 |
| didregress five-period staggered, vce(cluster) | 1.78e-15 | 1.13e-16 |
| xtdidregress five-period staggered, vce(cluster) | 1.78e-15 | 1.26e-16 |
| xtdidregress two-period, covariate, vce(cluster) | 2.08e-15 | 1.06e-15 |
| didregress vs xtdidregress dof convention | 1.81e-16 | — |
| svy: mean (pw + strata) | 3.84e-15 | 0.00e+00 |
| svy: total (pw + strata) | 3.90e-15 | 0.00e+00 |
| svy: proportion 3 levels (pw + strata) | 0.00e+00 | 1.47e-16 |
| svy: proportion binary (pw + strata) | 0.00e+00 | 0.00e+00 |
| svy: regress (pw + strata) | 9.77e-16 | 0.00e+00 |
| svy: regress with i.factor (pw + strata) | 3.46e-15 | 1.86e-16 |
| svy: logit (pw + strata) | 1.85e-14 | 1.51e-15 |
| svy: mean self-weight, single stratum | 1.27e-16 | 1.26e-16 |
| svy: mean no weights + strata (self-weight per PSU) | 1.27e-16 | 0.00e+00 |
| svy: mean with fpc (population PSUs) | 1.28e-16 | 0.00e+00 |
| svy: singleunit(missing) with lone PSUs | 0.00e+00 | — |
| svy: singleunit(centered) with lone PSUs | 0.00e+00 | 0.00e+00 |
| bootstrap replicate-for-replicate (100 reps, seed 20241024) | 1.41e-15 | — |
| permute: observed (unpermuted) statistic | 2.80e-15 | — |

**tseries**（20 条）

| 命令 | 系数相对差 | SE 相对差 |
|---|---|---|
| var: VAR(1) coefficients | 8.45e-14 | — |
| var: VAR(1) standard errors after per-equation ML normalisation | 2.90e-16 | — |
| var: VAR(1) e(Sigma) after per-equation ML normalisation | 3.25e-15 | — |
| var: Granger causality, VAR(1) after ML normalisation | 2.06e-15 | — |
| var: VAR(2) coefficients | 1.03e-14 | — |
| var: VAR(2) standard errors after per-equation ML normalisation | 1.22e-16 | — |
| var: VAR(2) e(Sigma) after per-equation ML normalisation | 1.08e-14 | — |
| var: Granger causality, VAR(2) after ML normalisation | 2.48e-15 | — |
| vec: rank(1) lags(2) trend(constant) [statsmodels 'co'] | 3.05e-10 | — |
| vec: rank(1) lags(2) trend(rconstant) [statsmodels 'ci'] | 1.49e-10 | — |
| vec: rank(1) lags(1) trend(constant) | 4.21e-11 | — |
| dfuller: y_rw lags(2) | 1.18e-12 | — |
| dfuller: y_rw lags(2) trend | 1.22e-13 | — |
| dfuller: y_rw lags(2) noconstant | 1.46e-15 | — |
| dfuller: y_ar lags(4), stationary series | 0.00e+00 | — |
| dfuller: y_rw lags(0), no augmentation | 7.20e-16 | — |
| pperron: y_rw, default Newey-West lag | 2.06e-13 | — |
| pperron: y_rw lags(3) | 1.79e-13 | — |
| pperron: y_rw trend | 5.01e-14 | — |
| pperron: y_ar, stationary series | 1.56e-15 | — |

## 按领域：E 档（>1e-2）与 error 条目

**discrete**

| 命令 | 档位 | 系数相对差 | SE 相对差 | 备注 |
|---|---|---|---|---|
| logit vce(robust) | error | — | — | ValueError: heteroscedasticity robust covariance does not use keywords |
| logit vce(cluster g) e() scalars | E | 3.72e-01 | — | worst common scalar e(chi2) rel=3.723e-01；Stata-only e(): N_cdf, N_cds, N_clust, consonly, converged, ic, k, k |
| probit vce(robust) | error | — | — | ValueError: heteroscedasticity robust covariance does not use keywords |
| poisson vce(robust) e() scalars | E | 8.12e-02 | — | worst common scalar e(chi2) rel=8.117e-02；Stata-only e(): consonly, converged, ic, k, k_autoCns, k_dv, k_eq, k |
| poisson cov_type='HC1' == 'HC0' | error | 0.00e+00 | — |  |
| logit [fweight=w] | error | — | — | TypeError: _DiscreteBase.fit() got an unexpected keyword argument 'weights' |
| nbreg e() scalars | E | 5.00e-01 | — | worst common scalar e(df_m) rel=5.000e-01；Stata-only e(): alpha, chi2_c, consonly, converged, ic, k, k_autoCns |
| nbreg, dispersion(constant) (NB1) | error | — | — | TypeError: _DiscreteBase.fit() got an unexpected keyword argument 'dispersion' |
| nbreg vce(robust) | error | — | — | ValueError: heteroscedasticity robust covariance does not use keywords |
| nbreg vce(robust) e() scalars | E | 5.00e-01 | — | worst common scalar e(df_m) rel=5.000e-01；Stata-only e(): alpha, consonly, converged, ic, k, k_autoCns, k_aux, |
| zip e() scalars | E | 1.06e-02 | — | worst common scalar e(p) rel=1.065e-02；Stata-only e(): N_zero, consonly, converged, df_c, ic, k, k_autoCns, k_ |
| zip e(p) degrees-of-freedom convention | error | 1.15e+02 | — |  |
| zinb e() scalars | E | 9.91e-01 | — | worst common scalar e(p) rel=9.911e-01；Stata-only e(): N_zero, consonly, converged, df_c, ic, k, k_autoCns, k_ |
| ologit e() scalars | E | 1.00e+00 | — | worst common scalar e(df_m) rel=1.000e+00；Stata-only e(): N_cd, consonly, converged, ic, k, k_autoCns, k_aux,  |
| ologit full e(V) | E | 2.09e+01 | — | includes slope x cut and cut x cut cross terms, where a delta-method Jacobian error shows up first |
| oprobit e() scalars | E | 1.00e+00 | — | worst common scalar e(df_m) rel=1.000e+00；Stata-only e(): N_cd, consonly, converged, ic, k, k_autoCns, k_aux,  |
| oprobit full e(V) | E | 1.05e+01 | — | includes slope x cut and cut x cut cross terms, where a delta-method Jacobian error shows up first |
| ologit vce(robust) | error | — | — | ValueError: ologit: only cov_type='nonrobust' is implemented; the ordered-model robust sandwich is not yet wir |
| oprobit vce(robust) | error | — | — | ValueError: oprobit: only cov_type='nonrobust' is implemented; the ordered-model robust sandwich is not yet wi |
| ztp predict, n (Stata semantics) | E | 8.68e-01 | — | westata TruncatedPoisson.fittedvalues returns the *truncated* mean mu/Pr(y>0), but Stata's `predict, n` after  |
| ztp fittedvalues (model-level, truncated mean vs Stata n) | E | 1.71e+00 | — | this is the estimator's own fittedvalues: a deliberately different quantity from Stata's predict,n, recorded s |
| zip predict, n (StataResults.predict) | error | — | — | ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m? |
| ologit predict, pr | error | — | — | ConsistencyError: predict: the design matrix was not retained on this result; pass exog=... explicitly |
| e(chi2) LR -> Wald switch under vce(robust)/vce(cluster) | error | 3.72e-01 | — |  |
| e(df_m) must exclude auxiliary parameters | error | 1.00e+00 | — |  |

**limited_quantile**

| 命令 | 档位 | 系数相对差 | SE 相对差 | 备注 |
|---|---|---|---|---|
| tobit /sigma SE root cause (diagnostic) | E | 5.43e-02 | 5.43e-02 | westata's reported se(/sigma) is the standard error of ln(sigma), not of sigma: against an independent central |
| heckman ml | error | — | — | UnsupportedError: heckman: only method='twostep' is implemented; 'ml' is not. The maximum likelihood variant r |
| etregress ml | error | — | — | UnsupportedError: etregress: only method='control' is implemented; 'ml' is not. The maximum likelihood variant |
| frontier exponential | error | — | — | UnsupportedError: frontier: distribution='exponential' is not implemented; only 'halfnormal' is. The truncated |

**linear_iv**

| 命令 | 档位 | 系数相对差 | SE 相对差 | 备注 |
|---|---|---|---|---|
| 12d ivregress gmm vce(cluster gid) | E | 3.58e-02 | 1.21e-01 | Stata 的 ivregress gmm 在 vce(cluster) 下把 W 也换成聚类权重矩阵，点估计因此与 hetero-GMM 不同；westata 的 IVGMM 无论 cov_type 都用 hetero |

**panel**

| 命令 | 档位 | 系数相对差 | SE 相对差 | 备注 |
|---|---|---|---|---|
| xtgls, panels(correlated)（30 个体 x 6 期，Sigma 秩亏） | E | 2.12e-01 | 3.39e+00 | 30 个面板只有 6 期 → Sigma 30x30 秩 <= 6；Stata 明确打印 “results are based on a generalized inverse of a singular matrix” |
| xtgls, corr(ar1) | error | — | — | UnsupportedError: xtgls: corr(ar1) is not implemented. Autocorrelation turns Omega from block-diagonal into a  |
| xtabond lags(2)（动态面板；按位置对齐，绕开 westata 的命名错位） | E | 3.49e-02 | 9.40e-01 | 工具数（23）与样本（300）仍对齐；按位置对齐后 L1.y 0.553997 vs 0.541664（2.3e-2）、L2.y 0.071296 vs 0.068892（3.5e-2）、x1 0.840295 vs 0 |

**postest**

| 命令 | 档位 | 系数相对差 | SE 相对差 | 备注 |
|---|---|---|---|---|
| margins: logit dydx(weight) at(mpg=20) | E | 5.38e-02 | — | conditional AME; the same numeric-gradient issue is amplified because the effect itself is smaller |
| margins: logit AME at(mpg=20), analytic delta method as arbiter | E | 5.38e-02 | — | analytic point estimate -0.000351977277986, SE 1.77996692283e-06；Stata -0.000351977277986 / 1.77996692283e-06  |
| tobit: SE of /sigma is the log-scale SE | E | 7.36e-01 | — | Stata var(e.mpg) = 14.2978442 with SE 2.717956239; sigma = 3.781249027 and its SE (delta method) is 0.35939926 |
| tobit: residual degrees of freedom e(df_r) | E | 1.39e-02 | — | Stata: df_r = N - df_m = 74 - 2 = 72, because the scale parameter does not consume a degree of freedom; westat |
| predict tobit: stdf | E | 7.71e-02 | — |  |
| predict tobit: stdf adds rss/N instead of sigma^2 | E | 1.52e-01 | — | Stata: stdf^2 - stdp^2 = sigma^2 = 14.2978442 for every observation；westata: stdf^2 - stdp^2 = rss/N = 12.1274 |

**systems_causal**

| 命令 | 档位 | 系数相对差 | SE 相对差 | 备注 |
|---|---|---|---|---|
| sureg vce(robust) (westata unsupported) | error | — | — | RuntimeError: UnsupportedError: sureg: cov_type='robust' is not implemented; only the unadjusted VCE (cov_type |
| nl default start values (convergence gap) | E | 1.42e+01 | — | westata starts every parameter at 1.0 and Levenberg-Marquardt stalls there (RSS 3.41743e+09 vs Stata's 4.53839 |
| teffects aipw ATET | E | 5.23e-01 | 2.93e-02 | Stata's e(b) also carries the nuisance models (x1, x2, _cons, x1, x2, _cons, x1, x2, _cons); those rows have n |
| teffects nnmatch biasadj(x1 x2) (westata unsupported) | error | — | — | TypeError: TypeError: NearestNeighbourMatching.fit() got an unexpected keyword argument 'bias_adjust' |
| jackknife: residual degrees of freedom used for t tests | E | 1.37e-02 | — | Stata stores e(df_r) = 73.0 (N - 1: one degree of freedom per replicate) and tests with t(df_r); westata repor |
| bootstrap: normal (z) vs t reference distribution | E | 1.71e-02 | — | A discrete difference, so the recorded number is the relative gap between the critical values westata uses and |
| permute: Monte Carlo p-value across 6 seeds (weak effect, 400 reps) | E | 1.21e-02 | — | Neither engine reproduces the other's permutations, so the only meaningful comparison is the distribution of t |

**tseries**

| 命令 | 档位 | 系数相对差 | SE 相对差 | 备注 |
|---|---|---|---|---|
| var: VAR(1) e(Sigma) (as reported) | E | 1.01e-02 | — | Same DOF story as the SEs: Stata's e(Sigma) is the ML estimate, statsmodels' sigma_u the unbiased one, so the  |
| var: Granger causality, VAR(1) (Stata vargranger) | E | 1.00e-02 | — | westata HAS granger causality (VARResults.granger_causality), but it returns statsmodels' kind='f' result -- a |
| var: VAR(2) e(Sigma) (as reported) | E | 1.71e-02 | — | Same DOF story as the SEs: Stata's e(Sigma) is the ML estimate, statsmodels' sigma_u the unbiased one, so the  |
| var: Granger causality, VAR(2) (Stata vargranger) | E | 1.68e-02 | — | westata HAS granger causality (VARResults.granger_causality), but it returns statsmodels' kind='f' result -- a |
| arch: GARCH(1,1) | E | 1.07e-02 | 2.78e-01 | KNOWN DIVERGENCE (the one the old register flagged), reproduced from scratch and now attributed to TWO indepen |

