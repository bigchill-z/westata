# westata —— 完整 API 参考

_westata 版本: 0.3.10 · 生成脚本 `docs/render_api_reference.py`_

`westata` 的全部公开名称，逐项给出对应的 Stata 指令、功能说明，以及一段可照抄运行、并附**真实返回值**的示例。

签名与示例输出均直接取自代码，因此与你手上这一版一致。

示例作用于一个 200 行的数据框 `base`，其列为 `y x1 x2 x3 yb ycnt ycens lo hi y1 y2 firm year ypanel treat ytreat sel yselect`，作用域内另有 `ws`、`np`、`pd`。

## 目录

- **稳健标准误** — [Stata `robust` 与 `HC0`/`HC1` 的对应关系](#稳健标准误stata-的-robust-与-hc0hc1-的对应关系)
- **线性模型与工具变量** — [`OLS`](#ols), [`FEOLS`](#feols), [`PanelOLS`](#panelols), [`RandomEffects`](#randomeffects), [`BetweenOLS`](#betweenols), [`FirstDifferenceOLS`](#firstdifferenceols), [`IV2SLS`](#iv2sls), [`IVLIML`](#ivliml), [`IVGMM`](#ivgmm), [`XtIvreg`](#xtivreg), [`XtGls`](#xtgls), [`XtPcse`](#xtpcse), [`XtAbond`](#xtabond), [`Eivreg`](#eivreg), [`Sureg`](#sureg), [`Mvreg`](#mvreg), [`RobustRegression`](#robustregression), [`QuantileRegression`](#quantileregression), [`Bsqreg`](#bsqreg), [`Sqreg`](#sqreg), [`NeweyWestResult`](#neweywestresult), [`Nl`](#nl)
- **离散选择、计数与受限因变量** — [`Logit`](#logit), [`Probit`](#probit), [`OrderedLogit`](#orderedlogit), [`OrderedProbit`](#orderedprobit), [`Poisson`](#poisson), [`NegativeBinomial`](#negativebinomial), [`NegativeBinomialNB1`](#negativebinomialnb1), [`ZeroInflatedPoisson`](#zeroinflatedpoisson), [`ZeroInflatedNegativeBinomial`](#zeroinflatednegativebinomial), [`TruncatedPoisson`](#truncatedpoisson), [`TruncatedNegativeBinomial`](#truncatednegativebinomial), [`Tobit`](#tobit), [`Truncreg`](#truncreg), [`Intreg`](#intreg), [`Heckman`](#heckman), [`Etregress`](#etregress), [`Frontier`](#frontier), [`XtLogit`](#xtlogit), [`XtProbit`](#xtprobit), [`XtPoisson`](#xtpoisson), [`XtNbreg`](#xtnbreg), [`XtTobit`](#xttobit)
- **时间序列、多元与面板单位根** — [`ARIMA`](#arima), [`GARCH`](#garch), [`VAR`](#var), [`VECM`](#vecm), [`XtUnitroot`](#xtunitroot), [`IpsResult`](#ipsresult), [`LlcResult`](#llcresult), [`UnitRootResult`](#unitrootresult), [`Manova`](#manova), [`ManovaResult`](#manovaresult)
- **处理效应与调查数据** — [`Didregress`](#didregress), [`Xtdidregress`](#xtdidregress), [`TreatmentEffectResult`](#treatmenteffectresult), [`IPW`](#ipw), [`AIPW`](#aipw), [`NearestNeighbourMatching`](#nearestneighbourmatching), [`RegressionAdjustment`](#regressionadjustment), [`SurveyDesign`](#surveydesign), [`SvyMean`](#svymean), [`SvyTotal`](#svytotal), [`SvyProportion`](#svyproportion), [`SvyRegress`](#svyregress), [`SvyLogit`](#svylogit)
- **重抽样** — [`Bootstrap`](#bootstrap), [`Jackknife`](#jackknife), [`Permute`](#permute)
- **后估计函数** — [`testnl`](#testnl), [`lrtest`](#lrtest), [`hausman`](#hausman), [`contrast`](#contrast), [`pwcompare`](#pwcompare), [`nlcom`](#nlcom), [`margins`](#margins), [`estat_classification`](#estat_classification), [`estat_endogenous`](#estat_endogenous), [`estat_firststage`](#estat_firststage), [`estat_gof`](#estat_gof), [`estat_hettest`](#estat_hettest), [`estat_ic`](#estat_ic), [`estat_imtest`](#estat_imtest), [`estat_overid`](#estat_overid), [`estat_ovtest`](#estat_ovtest), [`estat_summarize`](#estat_summarize), [`estat_vce`](#estat_vce), [`estat_vif`](#estat_vif), [`estat_auc`](#estat_auc), [`estat_esize`](#estat_esize), [`estat_moran`](#estat_moran), [`estat_mundlak`](#estat_mundlak), [`estat_szroeter`](#estat_szroeter), [`estat_weakrobust`](#estat_weakrobust), [`linktest`](#linktest), [`suest`](#suest), [`newey`](#newey), [`unitroot`](#unitroot), [`levin_lin_chu`](#levin_lin_chu), [`im_pesaran_shin`](#im_pesaran_shin), [`check_time_spacing`](#check_time_spacing), [`parse_exp_list`](#parse_exp_list), [`normalize_weights`](#normalize_weights)
- **已拟合结果对象** — [`StataResults`](#stataresults), [`params`](#params), [`bse`](#bse), [`pvalues`](#pvalues), [`tvalues`](#tvalues), [`zvalues`](#zvalues), [`fvalue`](#fvalue), [`llf`](#llf), [`aic`](#aic), [`bic`](#bic), [`nobs`](#nobs), [`df_model`](#df_model), [`df_resid`](#df_resid), [`dist`](#dist), [`model_name`](#model_name), [`cov_type`](#cov_type), [`cov`](#cov), [`e`](#e), [`fittedvalues`](#fittedvalues), [`resid`](#resid), [`rsquared`](#rsquared), [`rsquared_adj`](#rsquared_adj), [`model_data`](#model_data), [`converged`](#converged), [`iterations`](#iterations), [`grad_norm`](#grad_norm), [`param_names`](#param_names), [`test`](#test), [`lincom`](#lincom), [`predict`](#predict), [`summary`](#summary), [`wald_test`](#wald_test), [`cov_params`](#cov_params), [`conf_int`](#conf_int)
- **错误与支撑类型** — [`WestataError`](#westataerror), [`ConsistencyError`](#consistencyerror), [`ConvergenceError`](#convergenceerror), [`FormulaError`](#formulaerror), [`UnsupportedError`](#unsupportederror), [`WeightError`](#weighterror), [`ExpListError`](#explisterror), [`PackageNotFoundError`](#packagenotfounderror), [`StoredResults`](#storedresults), [`WeightSpec`](#weightspec), [`WEIGHT_TYPES`](#weight_types), [`SSC`](#ssc), [`VceMatrix`](#vcematrix), [`ParsedExpressionList`](#parsedexpressionlist), [`WaldTest`](#waldtest), [`LincomResult`](#lincomresult), [`IcResult`](#icresult), [`VifResult`](#vifresult), [`MarginsResult`](#marginsresult), [`GofResult`](#gofresult), [`ClassificationResult`](#classificationresult), [`HetTestResult`](#hettestresult), [`ImTestResult`](#imtestresult), [`OvTestResult`](#ovtestresult), [`EndogenousResult`](#endogenousresult), [`FirstStageResult`](#firststageresult), [`OverIdResult`](#overidresult), [`LRTestResult`](#lrtestresult), [`HausmanResult`](#hausmanresult)
- **其他名称** — [`Areg`](#areg), [`MarginsTable`](#marginstable), [`MultinomialLogit`](#multinomiallogit), [`PraisResult`](#praisresult), [`RandomEffectsMLE`](#randomeffectsmle), [`SummarizeResult`](#summarizeresult), [`Synth2Result`](#synth2result), [`SynthResult`](#synthresult), [`dfgls`](#dfgls), [`dfuller`](#dfuller), [`ereturn_list`](#ereturn_list), [`estat_bgodfrey`](#estat_bgodfrey), [`estat_corrgram`](#estat_corrgram), [`estat_dwatson`](#estat_dwatson), [`etable_as_stata`](#etable_as_stata), [`f_test`](#f_test), [`first_stage`](#first_stage), [`fitted_series`](#fitted_series), [`format_ereturn_list`](#format_ereturn_list), [`get_influence`](#get_influence), [`kpss`](#kpss), [`level`](#level), [`model`](#model), [`odds_ratios`](#odds_ratios), [`pperron`](#pperron), [`prais`](#prais), [`rcm`](#rcm), [`rdrobust`](#rdrobust), [`render_estimates_table`](#render_estimates_table), [`render_esttab`](#render_esttab), [`resid_series`](#resid_series), [`response_name`](#response_name), [`stata_cov_params`](#stata_cov_params), [`stata_factor_terms`](#stata_factor_terms), [`stata_param_names`](#stata_param_names), [`stata_params`](#stata_params), [`summarize`](#summarize), [`synth`](#synth), [`synth2`](#synth2), [`t_test`](#t_test), [`tebalance_summarize`](#tebalance_summarize), [`teffects_nnmatch`](#teffects_nnmatch), [`teffects_psmatch`](#teffects_psmatch), [`teffects_ra`](#teffects_ra), [`vargranger`](#vargranger), [`varlmar`](#varlmar), [`varsoc`](#varsoc), [`varstable`](#varstable), [`varwle`](#varwle), [`veclmar`](#veclmar), [`vecrank`](#vecrank), [`vecstable`](#vecstable), [`wntestq`](#wntestq), [`xttest0`](#xttest0)

## 稳健标准误：Stata 的 `robust` 与 `HC0`/`HC1` 的对应关系

> **把稳健标准误与 Stata 对比之前，先读这一节。**
> 同一个 Stata 选项名，在不同命令里对应**不同的小样本因子**；
> 弄混会得到看起来合理、但在第三位就错了的标准误。

### 对应表

| Stata | 对应 westata | 原因 |
|---|---|---|
| `regress ..., vce(robust)` | **`cov_type="HC1"`** | Stata 的 `regress` 乘了 `n/(n-k)` |
| `regress ..., vce(robust) small` | Stata 不接受 | `small` 是 ivregress 专属选项 |
| `ivregress ..., vce(robust)` | **`cov_type="HC0"`** | Stata 的 `ivregress` 默认**不**乘 `n/(n-k)` |
| `ivregress ..., vce(robust) small` | `cov_type="HC1"` | 加 `small` 后 Stata 才乘该因子 |
| `sureg ..., cov_type="robust"` | 字面量 `"robust"` | 其 sandwich 来自 `_reg3robust`，是独立实现，不是 HC1 |

### 面板估计量：`vce(robust)` 与 `vce(cluster id)` 是同一个数

第三种情形，也是最容易让人诧异的：对 `xtreg ..., fe`，
Stata 的 `vce(robust)` 与 `vce(cluster id)` 产生**完全相同**的标准误——
因为固定效应已经被组内去均值，稳健 VCE 天然就是按组聚类的。
40×10 平衡面板实测：

```
xtreg y x1, fe                      0.05440432377516
xtreg y x1, fe vce(robust)          0.05229223550829
xtreg y x1, fe vce(cluster sid)     0.05229223550829   <- 与 robust 完全相同
```

`PanelOLS` 三者全部命中，并且对 `robust` 与 `HC1` 都报 `cov_type="robust"`——
因为在这个族里它们**确实是同一个估计量**，HC0/HC1 并不指向另一种修正。
不要把面板结果上的 `cov_type="robust"` 读作「HC1」，
而应读作「面板稳健 VCE，对 `fe` 而言等于按个体聚类」。

### 实测数据

`grilic.dta` 上 `s` 的标准误：

```
reg lnw s expr, vce(robust)                     0.00638056877997
    westata .fit(cov_type="HC1")               0.00638056877997   <- 命中
    westata .fit(cov_type="HC0")               0.00636792980662   <- 不命中

ivregress 2sls lnw expr (s = rns mrt), vce(robust)   0.04061066342746
    westata .fit(cov_type="HC0")                     0.04061066342742   <- 命中
    westata .fit(cov_type="HC1")                     0.04069126687443   <- 不命中
```

**注意两个命令族对同一个拼写取了相反的形态**：
`regress` 是 HC1，`ivregress` 是 HC0。

### 直接写 `cov_type="robust"`

也可以直接用 Stata 的写法，包内按命令族解析，省去记表：

```python
ws.OLS.from_formula("lnw ~ s + expr", df).fit(cov_type="robust").cov_type      # 'HC1'
ws.IV2SLS.from_formula("lnw ~ expr + [s ~ rns + mrt]", df).fit(
    cov_type="robust"
).cov_type                                                                    # 'HC0'
```

结果对象存的是**规范名**，不是你输入的别名——
回显 `"robust"` 只会重新制造这一节要消除的混淆。

### 被拒绝的选项及原因

`cov_type="HC2"`/`"HC3"` 对 IV 估计量会报错：Stata 的 `ivregress`
没有杠杆校正的稳健 VCE，把 HC0 的数字贴成 HC2 标签是**伪造**
而非近似。`e(df_r)` 存 `N` 是同类原因——Stata 在未给 `small`
时留空，而 statsmodels 的接口无法表示「存的缺失值」。

## 线性模型与工具变量

### `OLS`

**Stata 指令:** `regress y x1 x2`

普通最小二乘，即 Stata 的 `regress`。打印系数、标准误、t 值、p 值与置信区间，以及方差分析与拟合区块（`N`、`F`、`R²`、调整 `R²`、Root MSE）。它是整个线性族的参照实现：小样本修正、`df_r` 约定与平方和区块遵循 `[R] regress`，封装后端的各估计量沿用这些约定，而不另行推导。

**签名**

```python
ws.OLS(endog: 'Any', exog: 'Any', *, weights: 'Any' = None, weight_type: 'str | None' = None, param_names: 'Sequence[str] | None' = None) -> 'None'
```

**`OLS.from_formula`**

```python
OLS.from_formula(formula: 'str', data: 'pd.DataFrame', *, weights: 'Any' = None, weight_type: 'str | None' = None, subset: 'Any' = None) -> "'OLS'"
```

**`OLS.fit`**

```python
OLS.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'SSC | None' = None, level: 'float | None' = None) -> 'StataResults'
```

**示例**

```python
ws.OLS.from_formula('y ~ x1 + x2', base).fit().summary(style='stata')
```

**输出**

```text

      Source |       SS           df       MS      Number of obs   =       200
-------------+----------------------------------   F(2, 197)       =     91.00
       Model |  200.183699         2  100.091849   Prob > F        =    0.0000
    Residual |  216.683448       197  1.09991598   R-squared       =    0.4802
-------------+----------------------------------   Adj R-squared   =    0.4749
       Total |  416.867146       199  2.09480978   Root MSE        =    1.0488

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8406445   .0724711    11.60   0.000     .6977258    .9835633
          x2 |   -.512087   .0748223    -6.84   0.000    -.6596425   -.3645315
       _cons |   1.005003   .0744285    13.50   0.000     .8582236    1.151781
------------------------------------------------------------------------------
```

**注意:** `regress` 不存总平方和，因此没有 `e(tss)`：`SS_total` 行由 `mss + rss` 重构。在含常数项的模型中该分解可加，所以这一步是精确的。

### `FEOLS`

**Stata 指令:** `reghdfe y x1 x2, absorb(firm year)`

高维固定效应 OLS，即 Stata 的 `reghdfe`。固定效应区块写在公式中 `|` 之后（`'y ~ x1 + x2 | firm + year'`），吸收由 PyFixest 的交替投影完成。`cov_type` 只接受 `nonrobust`、`HC0`、`HC1` 与 `cluster`：`HC2`/`HC3` 需要完整虚拟变量设计的杠杆值，而后端对吸收模型不提供该量，因此直接拒绝，不做近似。 它可以预测：`xb`（默认）是线性预测值**加上**吸收的固定效应，`d` 单独给出吸收效应，`residuals`/`dresiduals` 由二者推出——依据 `areg_p.ado:147-158`（同一估计器，只吸收一个维度）。

**签名**

```python
ws.FEOLS(formula: 'str', data: 'pd.DataFrame', *, weights: 'Any' = None, weight_type: 'str | None' = None) -> 'None'
```

**`FEOLS.from_formula`**

```python
FEOLS.from_formula(formula: 'str', data: 'pd.DataFrame', *, weights: 'Any' = None, weight_type: 'str | None' = None, subset: 'Any' = None) -> "'FEOLS'"
```

**`FEOLS.fit`**

```python
FEOLS.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'SSC | None' = None, level: 'float | None' = None) -> 'StataResults'
```

**示例**

```python
ws.FEOLS.from_formula('y ~ x1 + x2 | firm + year', base).fit().summary(style='stata')
```

**输出**

```text
(MWFE estimator converged in 1 iterations)

HDFE Linear regression                            Number of obs     =        200
Absorbing 2 HDFE groups                           F(   2,     154)  =      64.05
                                                  Prob > F          =     0.0000
                                                  R-squared         =     0.6125
                                                  Adj R-squared     =     0.4992
                                                  Within R-sq.      =     0.4541
                                                  Root MSE          =     1.0242

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .7872055   .0809428     9.73   0.000     .6273039    .9471072
          x2 |  -.4812685   .0857135    -5.61   0.000    -.6505945   -.3119426

Absorbed degrees of freedom:
-----------------------------------------------------+
 Absorbed FE | Categories  - Redundant  = Num. Coefs |
-------------+---------------------------------------|
        firm |         45           1         44     |
        year |         45           1         44     |
-----------------------------------------------------+
```

**注意:** `e(rank)` 计入每一个被吸收的固定效应参数，因此大于 `len(results.params)`；`e(df_m)` 只计报告的斜率。当给出 `weight_type='pweight'` 而未显式指定 `cov_type` 时，`fit` 选择 HC1——Stata 的 `pweight` 意味着 `vce(robust)`。`predict` 没有为 `reghdfe` 注册任何类型，因此无法经由它做预测。 吸收**多于一个维度**时预测会被拒绝：多路效应只在相差一个归一化常数的意义下可识别，因此拒绝信息会说明这一点，而不是让读者以为该估计器根本不能预测。

### `PanelOLS`

**Stata 指令:** `xtreg ypanel x1 x2, fe`

固定效应面板回归，即 Stata 的 `xtreg, fe`。`fit(entity_effects=True)` 是默认值。公式与 `OLS` 完全一样地隐含常数项，因此尽管组内变换会吸收它，`e(b)` 仍含 `Intercept`；要对应 Stata 的 `noconstant`，请写 `'ypanel ~ x1 + x2 - 1'`。含缺失值的行在估计之前被整行剔除，所以 `e(N)` 可能小于 `len(data)`；汇总还会打印 `sigma_u`/`sigma_e`/`rho` 区块与「所有 `u_i` 为零」的 F 检验。

**签名**

```python
ws.PanelOLS(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', weights: 'Any' = None, weight_type: 'str | None' = None) -> 'None'
```

**`PanelOLS.from_formula`**

```python
PanelOLS.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', subset: 'Any' = None, weights: 'Any' = None, weight_type: 'str | None' = None) -> "'PanelOLS'"
```

**`PanelOLS.fit`**

```python
PanelOLS.fit(self, entity_effects: 'bool' = True, time_effects: 'bool' = False, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'SSC | None' = None) -> 'StataResults'
```

**示例**

```python
ws.PanelOLS.from_formula('ypanel ~ x1 + x2', base, entity='firm', time='year').fit().summary(style='stata')
```

**输出**

```text

Fixed-effects (within) regression               Number of obs     =        200
Group variable: firm                            Number of groups  =         40

R-squared:                                      Obs per group:
     Within  = 0.3670                                         min =          5
     Between = 0.2358                                         avg =        5.0
     Overall = 0.2956                                         max =          5

                                                F(2, 158)         =      45.79
corr(u_i, Xb) = 0.0844                          Prob > F          =     0.0000

------------------------------------------------------------------------------
      ypanel | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .7088762   .0749043     9.46   0.000     .5609333    .8568191
          x2 |  -.1092899   .0816583    -1.34   0.183    -.2705726    .0519929
       _cons |   .8444225   .0703358    12.01   0.000     .7055029    .9833422
-------------+----------------------------------------------------------------
     sigma_u |  .90040934
     sigma_e |  .99008668
         rho |  .45267062   (fraction of variance due to u_i)
------------------------------------------------------------------------------
F test that all u_i=0: F(39, 158) = 4.10                     Prob > F = 0.0000
```

**注意:** 类文档中记录了一处刻意偏离：在 `cov_type='cluster'` 下，westata 把 `e(df_m)` 存为斜率个数、把 `e(rank)` 存为 `len(e(b))`，而聚类下的 Stata 两个值都小一（类文档字符串载有 StataNow 19.5 的实测数据）。`e(F)`、`e(df_b)` 与 `e(df_r)` 是一致的——只有这两个标签不同；若照抄 Stata，打印出的 F 标签会与其旁的 p 值自相矛盾。

### `RandomEffects`

**Stata 指令:** `xtreg ypanel x1 x2, re`

随机效应 GLS，即 Stata 的 `xtreg, re`。方差成分来自后端默认的 Swamy-Arora 估计量；可选的小样本修正被刻意关闭，使估计量严格等于该文档化的默认式。汇总给出 `sigma_u`、`sigma_e`、`rho` 以及 `u_i` 与线性指数的相关系数，表头是 Wald 卡方检验而非 `xtreg, fe` 所打印的 F；`e(theta)` 保存拟合所用的准去均值系数。

**签名**

```python
ws.RandomEffects(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', weights: 'Any' = None, weight_type: 'str | None' = None) -> 'None'
```

**`RandomEffects.from_formula`**

```python
RandomEffects.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', subset: 'Any' = None, weights: 'Any' = None, weight_type: 'str | None' = None) -> "'PanelOLS'"
```

**`RandomEffects.fit`**

```python
RandomEffects.fit(self, entity_effects: 'bool' = False, time_effects: 'bool' = False, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'SSC | None' = None) -> 'StataResults'
```

**示例**

```python
ws.RandomEffects.from_formula('ypanel ~ x1 + x2', base, entity='firm', time='year').fit().summary(style='stata')
```

**输出**

```text

Random-effects GLS regression                   Number of obs     =        200
Group variable: firm                            Number of groups  =         40

R-squared:                                      Obs per group:
     Within  = 0.3669                                         min =          5
     Between = 0.2363                                         avg =        5.0
     Overall = 0.2957                                         max =          5

                                                Wald chi2(2)      =     101.64
corr(u_i, X) = 0 (assumed)                      Prob > chi2       =     0.0000

------------------------------------------------------------------------------
      ypanel | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .7277599    .073114     9.95   0.000     .5844592    .8710606
          x2 |   -.119745   .0784768    -1.53   0.127    -.2735566    .0340667
       _cons |   .8440041   .1428722     5.91   0.000     .5639798    1.124028
-------------+----------------------------------------------------------------
     sigma_u |  .78751004
     sigma_e |  .99008668
         rho |  .38750008   (fraction of variance due to u_i)
------------------------------------------------------------------------------
```

### `BetweenOLS`

**Stata 指令:** `xtreg ypanel x1 x2, be`

组间估计量，即 Stata 的 `xtreg, be`。在实体均值上做一次 OLS，因此 40 个实体、200 行的面板实际是在 40 个观测上估计：`e(N)` 是真正被消费的组均值个数，`e(df_r) = n_entities - K`。`e()` 存有三个面板 R²（`r2_b`、`r2_o`、`r2_w`），但没有 `sigma_u`/`sigma_e`/`rho` 区块：对均值做回归没有组内变异可供分解。

**签名**

```python
ws.BetweenOLS(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', weights: 'Any' = None, weight_type: 'str | None' = None) -> 'None'
```

**`BetweenOLS.from_formula`**

```python
BetweenOLS.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', subset: 'Any' = None, weights: 'Any' = None, weight_type: 'str | None' = None) -> "'PanelOLS'"
```

**`BetweenOLS.fit`**

```python
BetweenOLS.fit(self, entity_effects: 'bool' = False, time_effects: 'bool' = False, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'SSC | None' = None) -> 'StataResults'
```

**示例**

```python
ws.BetweenOLS.from_formula('ypanel ~ x1 + x2', base, entity='firm', time='year').fit().summary(style='stata')
```

**输出**

```text

Between regression (regression on group means)  Number of obs     =        200
Group variable: firm                            Number of groups  =         40

R-squared:                                      Obs per group:
     Within  = 0.3654                                         min =          5
     Between = 0.2375                                         avg =        5.0
     Overall = 0.2954                                         max =          5

                                                F(2,37)           =       5.76
sd(u_i + avg(e_i.)) = .9034525                  Prob > F          =     0.0066

------------------------------------------------------------------------------
      ypanel | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   1.128085   .3429968     3.29   0.002     .4331071    1.823062
          x2 |  -.2568295   .2873123    -0.89   0.377    -.8389796    .3253206
       _cons |   .8297597   .1453072     5.71   0.000     .5353393     1.12418
------------------------------------------------------------------------------
```

### `FirstDifferenceOLS`

**Stata 指令:** `xtreg ypanel x1 x2, fd`

一阶差分估计量，即 Stata 的 `xtreg, fd`。每期与紧邻的前一期作差（`t` 对 `t - 1`，因此缺失期不会被跨接），模型不含常数项；patsy 隐含的截距在进入后端之前被移除，因为后端拒绝含常数项的一阶差分回归。在 40 个实体、5 期的面板上，可用观测为 160 个，这正是 `e(N)` 报告的数值。

**签名**

```python
ws.FirstDifferenceOLS(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', weights: 'Any' = None, weight_type: 'str | None' = None) -> 'None'
```

**`FirstDifferenceOLS.from_formula`**

```python
FirstDifferenceOLS.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', subset: 'Any' = None, weights: 'Any' = None, weight_type: 'str | None' = None) -> "'PanelOLS'"
```

**`FirstDifferenceOLS.fit`**

```python
FirstDifferenceOLS.fit(self, entity_effects: 'bool' = False, time_effects: 'bool' = False, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'SSC | None' = None) -> 'StataResults'
```

**示例**

```python
ws.FirstDifferenceOLS.from_formula('ypanel ~ x1 + x2', base, entity='firm', time='year').fit().summary(style='stata')
```

**输出**

```text

Group variable: .                                 Number of groups =      40
                                                                Obs per group:
                                                                    min =       5
                                                                    avg =       5
                                                                    max =       5
Log likelihood = -274.0773697                        Prob > chi2      =  0.0000

Number of obs         =        160
F(2, 158)             =      40.91
Prob > F              =     0.0000
R-squared             =     0.3412
Adj R-squared         =     0.3370
Root MSE              =     1.3503

------------------------------------------------------------------------------
      ypanel | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .6218778    .069845     8.90   0.000     .4839276     .759828
          x2 |  -.1015749   .0754635    -1.35   0.180    -.2506223    .0474726
```

**注意:** 公式与所有 patsy 公式一样隐含一个常数项，而该常数项会被静默丢弃：没有保留它的选项，因此 `e(b)` 只含斜率。

### `IV2SLS`

**Stata 指令:** `ivregress 2sls ytreat x1 (treat = x3)`

两阶段最小二乘，即 Stata 的 `ivregress 2sls`。公式采用 linearmodels 的语法 `'ytreat ~ x1 + [treat ~ x3]'`；Stata 的 `(endog = instrument)` 写法会被拒绝，因为该语法交由后端解析，而非重新实现。外生回归元、内生回归元与被排除工具变量分别命名，且 `e(b)` 按 Stata 的顺序报告——内生在前、包含的外生在后、`_cons` 最后——尽管设计矩阵是按相反顺序构建的。

**签名**

```python
ws.IV2SLS(endog: 'Any', exog: 'Any', endog_instr: 'Any', instruments: 'Any', *, param_names: 'Sequence[str] | None' = None, weights: 'Any' = None, weight_type: 'str | None' = None, **kwargs: 'Any') -> 'None'
```

**`IV2SLS.from_formula`**

```python
IV2SLS.from_formula(formula: 'str', data: 'pd.DataFrame', **kwargs: 'Any') -> "'IV2SLS'"
```

**`IV2SLS.fit`**

```python
IV2SLS.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'SSC | None' = None, level: 'float | None' = None) -> 'StataResults'
```

**示例**

```python
ws.IV2SLS.from_formula('ytreat ~ x1 + [treat ~ x3]', base).fit().summary(style='stata')
```

**输出**

```text

Instrumental-variables 2SLS regression            Number of obs   =        200
                                                  Wald chi2(2)    =     135.14
                                                  Prob > chi2     =     0.0000
                                                  R-squared       =     0.5247
                                                  Root MSE        =     .98083

------------------------------------------------------------------------------
      ytreat | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
       treat |   1.444141   .2839408     5.09   0.000     .8876271    2.000654
          x1 |   .7721013   .0689206    11.20   0.000     .6370194    .9071831
       _cons |   .9463986   .1952162     4.85   0.000     .5637819    1.329015
------------------------------------------------------------------------------
Endogenous: treat
Exogenous:  x1 x3
```

**注意:** `HC0` 对应 Stata 的 `vce(robust)`，`HC1` 对应 `vce(robust) small`——与 `regress` 相反，那里 `vce(robust)` 就是 HC1。`HC2`/`HC3` 被拒绝，因为 `ivregress` 没有可供复现的杠杆修正稳健 VCE。`e(df_r)` 存的是 `N`：除非给出 `small`，Stata 将其留作缺失，而缺失值无法被存下来。

### `IVLIML`

**Stata 指令:** `ivregress liml ytreat x1 (treat = x3)`

有限信息最大似然，即 Stata 的 `ivregress liml`。以 LIML 的 `kappa` 求解 k 类问题，`kappa` 由 `e(kappa)` 报告。恰好识别时该 `kappa` 为 1，估计值与 2SLS 相同；模型过度识别时 `kappa` 偏离 1，估计值随之不同。两者都在 fixture 数据上实测：恰好识别时 `e(kappa)` 打印 1.0，系数与 `IV2SLS` 一致；加入第二个工具变量后 `kappa` 约为 1.0183，系数明显不同。

**签名**

```python
ws.IVLIML(endog: 'Any', exog: 'Any', endog_instr: 'Any', instruments: 'Any', *, param_names: 'Sequence[str] | None' = None, weights: 'Any' = None, weight_type: 'str | None' = None, **kwargs: 'Any') -> 'None'
```

**`IVLIML.from_formula`**

```python
IVLIML.from_formula(formula: 'str', data: 'pd.DataFrame', **kwargs: 'Any') -> "'IV2SLS'"
```

**`IVLIML.fit`**

```python
IVLIML.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'SSC | None' = None, level: 'float | None' = None) -> 'StataResults'
```

**示例**

```python
ws.IVLIML.from_formula('ytreat ~ x1 + [treat ~ x3 + x2]', base).fit().summary(style='stata')
```

**输出**

```text

Instrumental-variables LIML regression            Number of obs   =        200
                                                  Wald chi2(2)    =     128.73
                                                  Prob > chi2     =     0.0000
                                                  R-squared       =     0.5189
                                                  Root MSE        =     .98683

------------------------------------------------------------------------------
      ytreat | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
       treat |   1.290524   .2830631     4.56   0.000     .7357303    1.845317
          x1 |   .7653347    .069321    11.04   0.000      .629468    .9012013
       _cons |   1.045101   .1948412     5.36   0.000     .6632198    1.426983
------------------------------------------------------------------------------
Endogenous: treat
Exogenous:  x1 x3 x2
```

**注意:** `cov_type='HC2'`/`'HC3'` 因与 `IV2SLS` 相同的原因被拒绝：`ivregress` 没有可供复现的杠杆修正稳健 VCE。

### `IVGMM`

**Stata 指令:** `ivregress gmm ytreat x1 (treat = x3)`

广义矩估计，即 Stata 的 `ivregress gmm`。`ivregress gmm` 把*两*件事都系于 `vce()`，而不是一件：`e(b)` 的协方差与 GMM 权重矩阵 `W`，后者来自同一个矩协方差。因此 westata 也依据 `cov_type` 选择后端的权重矩阵。模型恰好识别时，这只会改变标准误；过度识别时*点估计*也会变动，这正是 Stata 的行为，也是那里 GMM 与 2SLS 不同的原因。在 fixture 数据上用两个工具变量实测：恰好识别时 `cov_type='cluster'` 不改变系数；加入第二个工具变量后，聚类使系数变动约 1.1e-2。

**签名**

```python
ws.IVGMM(endog: 'Any', exog: 'Any', endog_instr: 'Any', instruments: 'Any', *, param_names: 'Sequence[str] | None' = None, weights: 'Any' = None, weight_type: 'str | None' = None, **kwargs: 'Any') -> 'None'
```

**`IVGMM.from_formula`**

```python
IVGMM.from_formula(formula: 'str', data: 'pd.DataFrame', **kwargs: 'Any') -> "'IV2SLS'"
```

**`IVGMM.fit`**

```python
IVGMM.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Any' = None, ssc: 'Any' = None, **kwargs: 'Any') -> 'StataResults'
```

**示例**

```python
ws.IVGMM.from_formula('ytreat ~ x1 + [treat ~ x3]', base).fit().summary(style='stata')
```

**输出**

```text

Instrumental-variables GMM regression             Number of obs   =        200
                                                  Wald chi2(2)    =     176.20
                                                  Prob > chi2     =     0.0000
                                                  R-squared       =     0.5247
                                                  Root MSE        =     .98083
GMM weight matrix: Robust

------------------------------------------------------------------------------
      ytreat | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
       treat |   1.444141   .2801603     5.15   0.000     .8950365    1.993245
          x1 |   .7721013   .0603991    12.78   0.000     .6537212    .8904813
       _cons |   .9463986   .1922365     4.92   0.000     .5696219    1.323175
------------------------------------------------------------------------------
Endogenous: treat
Exogenous:  x1 x3
```

**注意:** 协方差是后端的 GMM 权重矩阵三明治，无法经由面向 OLS 的 `vce.compute_cov` 路径表达，因此 `cov_kwds` 原样转发给后端，而不是在此处解释。

### `XtIvreg`

**Stata 指令:** `xtivreg ypanel x1 (x2 = x3), fe`

面板工具变量，即 Stata 的 `xtivreg`，`method='fe'`（组内 2SLS）或 `method='re'`（G2SLS）。公式采用 Stata 写法 `'ypanel ~ x1 + (x2 = x3)'`，括号内的工具变量区块被文本拆出，使各部分内部的因子展开与交互项照常工作。对 `fe`，组内变换施于因变量、外生回归元、内生回归元*以及工具变量*——未去均值的工具变量仍携带实体效应，投影会把它本应移除的固定效应重新吸收一部分。常数项随后由实体均值反解得到。

**签名**

```python
ws.XtIvreg(endog: 'Any', exog: 'Any', endog_regressors: 'Any', instruments: 'Any', *, exog_names: 'Sequence[str]', endog_names: 'Sequence[str]', instrument_names: 'Sequence[str]', codes: 'np.ndarray', sizes: 'np.ndarray', method: 'str' = 'fe', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`XtIvreg.from_formula`**

```python
XtIvreg.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str | None' = None, method: 'str' = 'fe', subset: 'Any' = None) -> "'XtIvreg'"
```

**`XtIvreg.fit`**

```python
XtIvreg.fit(self, cov_type: 'str' = 'nonrobust', cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 100, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.XtIvreg.from_formula('ypanel ~ x1 + (x2 = x3)', base, entity='firm', time='year').fit().summary(style='stata')
```

**输出**

```text

Fixed-effects (within) IV regression            Number of obs     =        200
Group variable: firm                            Number of groups  =         40

R-squared:                                      Obs per group:
     Within  =      .                                         min =          5
     Between = 0.0029                                         avg =        5.0
     Overall = 0.0000                                         max =          5

                                                Wald chi2(2)      =       4.33
corr(u_i, Xb) = -0.0059                         Prob > chi2       =     0.1149

------------------------------------------------------------------------------
      ypanel | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .7464423   .4321449     1.73   0.084    -.1005461    1.593431
          x2 |   4.463368   32.15818     0.14   0.890     -58.5655    67.49223
       _cons |   .5517225   2.083237     0.26   0.791    -3.531347    4.634792
------------------------------------------------------------------------------
Endogenous: x2
Exogenous:  x1 x3
-------------+----------------------------------------------------------------
     sigma_u |  1.4971249
     sigma_e |  4.4921699
         rho |  .09996814   (fraction of variance due to u_i)
------------------------------------------------------------------------------
 F test that all u_i=0: F(39,158)  =     0.20                Prob > F = 1.0000
------------------------------------------------------------------------------
```

**注意:** 只实现了 `fe` 与 `re`。Stata 的 `be`、`fd` 以及 Baltagi 的 `ec2sls` 会抛 `UnsupportedError`，而不是退回到某个相近的估计量，因为差别在变换本身，不在显示。

### `XtGls`

**Stata 指令:** `xtgls y x1 x2, panels(heteroskedastic)`

面板 GLS，即 Stata 的 `xtgls`。`panels` 接受 `iid`、`heteroskedastic`（默认）与 `correlated`；`corr` 接受 `independent`（默认）与 `ar1`，后者以各面板共用的一个自相关系数做 Prais-Winsten 变换，再做同样的 GLS 步。这是两步可行 GLS，不是混合 OLS：各面板自身的方差为各自面板加权，因此当这些方差与回归元相关时系数会变动。在 fixture 数据上实测，`panels(heteroskedastic)` 给出 `x1 = .8398055`，而同一数据的 `OLS` 给出 `.8406445`。

**签名**

```python
ws.XtGls(endog: 'Any', exog: 'Any', codes: 'np.ndarray', sizes: 'np.ndarray', *, param_names: 'Sequence[str]', panels: 'str' = 'heteroskedastic', corr: 'str' = 'independent', nmk: 'bool' = False, model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`XtGls.from_formula`**

```python
XtGls.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str | None' = None, panels: 'str' = 'heteroskedastic', corr: 'str' = 'independent', nmk: 'bool' = False, subset: 'Any' = None) -> "'XtGls'"
```

**`XtGls.fit`**

```python
XtGls.fit(self, cov_type: 'str' = 'nonrobust', cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 100, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.XtGls.from_formula('y ~ x1 + x2', base, entity='firm', time='year').fit().summary(style='stata')
```

**输出**

```text
Cross-sectional time-series FGLS regression

Coefficients:  generalized least squares
Panels:        heteroskedastic
Correlation:   no autocorrelation

Estimated covariances      =        40          Number of obs     =        200
Estimated autocorrelations =         0          Number of groups  =         40
Estimated coefficients     =         3          Time periods      =          5
                                                Wald chi2(2)      =     309.67
                                                Prob > chi2       =     0.0000

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8398055   .0564532    14.88   0.000     .7291592    .9504518
          x2 |  -.5251067   .0582604    -9.01   0.000    -.6392949   -.4109186
       _cons |   .9897201   .0582355    17.00   0.000     .8755807     1.10386
```

**注意:** `panels(iid) corr(independent) nmk` 是能复现 `regress` 的组合，因为那里唯一缺少的就是 `nmk` 补上的 `N/(N-k)` 因子。`corr(psar1)`（每个面板单独一个系数）以及 robust/cluster 协方差会直接抛错，而不是把共用 rho 或常规数字挂在错误的标签下一并返回。模块文档字符串提到 `igls`，但 `fit` 并没有该参数，所以传入它会以 `TypeError` 失败，而不会抛 westata 的错误。

### `XtPcse`

**Stata 指令:** `xtpcse y x1 x2, corr(independent)`

面板校正标准误，即 Stata 的 `xtpcse`（Beck-Katz）。点估计为 OLS，协方差为面板校正 VCE 且不施加小样本因子；`selection` 为 `casewise`（Stata 默认，用所有面板共有的期估计扰动协方差）或 `pairwise`（用每一对面板共有的期）。汇总打印面板设计区块——panels、估计的协方差数、自相关数与系数数——并报告 Wald 卡方而非 F。

**签名**

```python
ws.XtPcse(endog: 'Any', exog: 'Any', codes: 'np.ndarray', sizes: 'np.ndarray', *, param_names: 'Sequence[str]', correlation: 'str' = 'independent', selection: 'str' = 'casewise', periods: 'np.ndarray | None' = None, model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`XtPcse.from_formula`**

```python
XtPcse.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str | None' = None, correlation: 'str' = 'independent', selection: 'str' = 'casewise', subset: 'Any' = None) -> "'XtPcse'"
```

**`XtPcse.fit`**

```python
XtPcse.fit(self, cov_type: 'str' = 'pcse', cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 100, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.XtPcse.from_formula('y ~ x1 + x2', base, entity='firm', time='year').fit().summary(style='stata')
```

**输出**

```text
Linear regression, correlated panels corrected standard errors (PCSEs)

Group variable:   firm                          Number of obs     =        200
Time variable:    year                          Number of groups  =         40
Panels:           correlated (balanced)         Obs per group:
Autocorrelation:  no autocorrelation                          min =          5
                                                              avg =          5
                                                              max =          5
Estimated covariances      =       820          R-squared         =     0.4802
Estimated autocorrelations =         0          Wald chi2(2)      =     195.35
Estimated coefficients     =         3          Prob > chi2       =     0.0000

------------------------------------------------------------------------------
             |           Panel-corrected
           y | Coefficient  std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8406445   .0649097    12.95   0.000     .7134238    .9678653
          x2 |   -.512087   .0873288    -5.86   0.000    -.6832483   -.3409257
       _cons |   1.005003    .107612     9.34   0.000     .7940868    1.215918
```

**注意:** 在平衡面板上 `time=` 可以省略，此时期数按行序位置读取。在非平衡面板上则必须给出：否则抛 `ConsistencyError` 并点明该参数，因为按位置配对行所算出的量并不是协方差。`correlation='ar1'`/`'psar1'` 会用 Prais-Winsten 取代 OLS，一律抛 `UnsupportedError`。

### `XtAbond`

**Stata 指令:** `xtabond ypanel x1, lags(1)`

Arellano-Bond / Blundell-Bond 动态面板 GMM，即 Stata 的 `xtabond` 家族。在一阶差分方程上做一步 GMM；`lags` 决定因变量的多少个滞后进入回归，`cov_type` 为 `'gmm'`（默认）或 `'robust'`。只支持平衡面板。工具变量个数以两种方式报告：`e(instruments)`/`e(zrank)` 是 Stata 的「Number of instruments」（比差分侧列数多一，因为 Stata 还声明了差分估计量用不到的水平方程矩），而 `e(instruments_difference)` 是实际使用的列数。

**签名**

```python
ws.XtAbond(endog: 'Any', exog: 'Any', groups: 'Any', periods: 'Any', *, lags: 'int' = 1, param_names: 'Sequence[str]' = ('L1.y',), exog_names: 'Sequence[str]' = ('x1',), system: 'bool' = False, model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`XtAbond.from_formula`**

```python
XtAbond.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', lags: 'int' = 1, system: 'bool' = False, subset: 'Any' = None) -> "'XtAbond'"
```

**`XtAbond.fit`**

```python
XtAbond.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 100, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.XtAbond.from_formula('ypanel ~ x1', base, entity='firm', time='year').fit().summary(style='stata')
```

**输出**

```text
Arellano–Bond dynamic panel-data estimation     Number of obs    =     120
Group variable: firm                            Number of groups =      40
Time variable: year
                                                   Obs per group:
                                                           min =       3
                                                           avg =     3.0
                                                           max =       3
Number of instruments =      8                  Wald chi2(2)     =   30.43
                                                Prob > chi2      =  0.0000
One-step results


------------------------------------------------------------------------------
             | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
        L1.y |   .2296989   .1226223     1.87   0.061    -.0106364    .4700341
          x1 |   .6069829   .1107299     5.48   0.000     .3899563    .8240095
```

**注意:** Sargan、Hansen、AR(1) 与 AR(2) 干脆不报告，而不是近似报告：手册并未完整给出它们的精确构造，而一个错误的过度识别统计量比它的缺席更像权威结论。`system=True` 只是把结果标记为系统 GMM——水平方程的矩并未估计，系数仍是差分估计量的。

### `Eivreg`

**Stata 指令:** `eivreg y x1 x2, reliab(x1 .9)`

含测量误差的回归，即 Stata 的 `eivreg`。`reliab` 是必填项，把回归元名字映射到假定的信度；缺少它会直接报错，因为未修正的衰减不过是换了个名字的 OLS。估计量从归一化交叉积中减去测量误差方差（`A = X'X/n - S`），这是消去衰减而不是重新加权；标准误来自 M 估计三明治，其中包含 `S` 对系数的导数——丢掉这一项会使标准误出错，尽管系数不变。

**签名**

```python
ws.Eivreg(endog: 'Any', exog: 'Any', reliab: 'Mapping[str, float]', *, param_names: 'Sequence[str]', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`Eivreg.from_formula`**

```python
Eivreg.from_formula(formula: 'str', data: 'pd.DataFrame', *, reliab: 'Mapping[str, float] | None' = None, subset: 'Any' = None) -> "'Eivreg'"
```

**`Eivreg.fit`**

```python
Eivreg.fit(self, cov_type: 'str | None' = 'nonrobust', cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None) -> 'StataResults'
```

**示例**

```python
ws.Eivreg.from_formula('y ~ x1 + x2', base, reliab={'x1': 0.9}).fit().summary(style='stata')
```

**输出**

```text
Errors-in-variables regression


     Assumed
----------------------------                    Number of obs     =        200
                                                F(2, 197)         =      97.02
                                                Prob > F          =     0.0000

------------------------------------------------------------------------------
             | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |    .934051    .078524    11.90   0.000     .7791954    1.088907
          x2 |  -.5117214   .0713415    -7.17   0.000    -.6524125   -.3710303
       _cons |   .9996233   .0743546    13.44   0.000     .8529901    1.146257
```

**注意:** 只提供未调整的 VCE：`cov_type='robust'` 会抛 `UnsupportedError`。若 `reliab` 中给出的名字不是回归元，同样会报错，且消息会列出它实际找到的回归元。

### `Sureg`

**Stata 指令:** `sureg (y1 x1) (y2 x2)`

似不相关回归，即 Stata 的 `sureg`。它接受一个双端公式的*序列*：`from_formula(['y1 ~ x1', 'y2 ~ x2'], data)`，而不是单个字符串。系数带方程标签前缀（`y1:x1`、`y2:_cons`），否则堆叠后的系数向量会含重复名字。跨方程协方差为 `Sigma_ij = e_i'e_j / n`——标准误取决于除数 `n` 而非 `n - k`——方程组按块求解，而不是把完整的 Kronecker 矩阵显式展开。

**签名**

```python
ws.Sureg(designs: 'Sequence[np.ndarray]', responses: 'Sequence[np.ndarray]', *, equation_names: 'Sequence[str]', param_names: 'Sequence[Sequence[str]]', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`Sureg.from_formula`**

```python
Sureg.from_formula(equations: 'Sequence[str]', data: 'pd.DataFrame', *, names: 'Sequence[str] | None' = None, subset: 'Any' = None) -> "'Sureg'"
```

**`Sureg.fit`**

```python
Sureg.fit(self, cov_type: 'str | None' = 'nonrobust', cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, iterate: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Sureg.from_formula(['y1 ~ x1', 'y2 ~ x2'], base).fit().summary(style='stata')
```

**输出**

```text
Seemingly unrelated regression
------------------------------------------------------------------------------
Equation             Obs   Params         RMSE  "R-squared"      chi2   P>chi2
------------------------------------------------------------------------------
y1                   200        1    1.0169773      0.2718      73.77   0.0000
y2                   200        1     .9704172      0.1092      24.73   0.0000
------------------------------------------------------------------------------

------------------------------------------------------------------------------
             | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
y1           |
          x1 |   .6028085   .0701866     8.59   0.000     .4652454    .7403717
       _cons |   .9131627    .072023    12.68   0.000     .7720002    1.054325
-------------+----------------------------------------------------------------
y2           |
          x2 |  -.3438779   .0691467    -4.97   0.000    -.4794029   -.2083529
       _cons |   1.958213   .0687593    28.48   0.000     1.823447    2.092978
```

**注意:** `iterate=True`（Stata 的 `isure`）被拒绝，因为其收敛规则未被精确文档化，而错误的规则会改变最后一次迭代后的每一个标准误。`cov_type='robust'` 走 `reg3` 的稳健三明治，确实会改变所报告的标准误，但 `e(vce)` 仍读作 `'unadjusted'`——若要知道拿到的是哪种 VCE，请读结果对象的 `cov_type` 属性。`cov_type='cluster'` 被拒绝；所有方程必须使用相同的观测，样本不一致会报错，而不是在各样本不等的条件下强行拟合。

### `Mvreg`

**Stata 指令:** `mvreg y1 y2 = x1 x2`

多元回归，即 Stata 的 `mvreg`，写作 `'y1 y2 = x1 x2'`，结果变量在 `=` 左侧，共用的回归元列表在右侧。每个方程都使用全部回归元，因此系数就是逐方程的 OLS；只有 VCE 用到跨方程残差协方差 `V = Sigma (x) (X'X)^-1`，其中 `Sigma_ij = u_i'u_j / (n - k)`。注意这里的除数是 `n - k`，而 `sureg` 用 `n`：当各方程含不同回归元时，两个命令在同一数据上因此结果不同，这是设定差异，不是不一致。

**签名**

```python
ws.Mvreg(designs: 'Sequence[np.ndarray]', responses: 'Sequence[np.ndarray]', *, outcome_names: 'Sequence[str]', param_names: 'Sequence[str]', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`Mvreg.from_formula`**

```python
Mvreg.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None) -> "'Mvreg'"
```

**`Mvreg.fit`**

```python
Mvreg.fit(self, cov_type: 'str | None' = 'nonrobust', cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None) -> 'StataResults'
```

**示例**

```python
ws.Mvreg.from_formula('y1 y2 = x1 x2', base).fit().summary(style='stata')
```

**输出**

```text
Equation             Obs   Parms        RMSE    "R-sq"          F      P>F
--------------------------------------------------------------------------
y1                   200       3   1.0243928    0.2722    36.8382   0.0000
y2                   200       3    .9740065    0.1160    12.9271   0.0000
--------------------------------------------------------------------------

------------------------------------------------------------------------------
             | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
y1           |
          x1 |   .6070447   .0707867     8.58   0.000     .4674478    .7466416
          x2 |  -.0243666   .0730832    -0.33   0.739    -.1684925    .1197593
       _cons |    .914468   .0726986    12.58   0.000     .7711007    1.057835
-------------+----------------------------------------------------------------
y2           |
          x1 |  -.0832083   .0673049    -1.24   0.218    -.2159389    .0495224
          x2 |  -.3430494   .0694885    -4.94   0.000    -.4800863   -.2060126
       _cons |   1.962931   .0691228    28.40   0.000     1.826615    2.099247
```

**注意:** 只实现了未调整的 VCE；`cov_type='robust'` 会抛 `UnsupportedError`，因为该命令只有未调整 VCE 经过与 Stata 的验证。

### `RobustRegression`

**Stata 指令:** `rreg y x1 x2`

稳健回归，即 Stata 的 `rreg`，按 Stata 的语句顺序执行：先拟合 OLS，剔除 Cook's D 大于 1 的观测后重拟合，迭代 Huber 权重，再迭代双权重，最后对调整后的响应变量做一次无权回归。`tune`（默认 7）缩放双权重常数 `c_b = 4.685 * tune / 7`，`tolerance` 默认为 0.01——即 Stata 自身的停止规则。该默认值有实质影响：迭代更紧会落到另一个点上，所以 `tolerance=1e-9` 得到的是完全收敛的估计，而非 Stata 的估计。

**签名**

```python
ws.RobustRegression(endog: 'Any', exog: 'Any', *, tune: 'float' = 7.0, param_names: 'list[str] | None' = None) -> 'None'
```

**`RobustRegression.from_formula`**

```python
RobustRegression.from_formula(formula: 'str', data: 'pd.DataFrame', *, tune: 'float' = 7.0, subset: 'Any' = None) -> "'RobustRegression'"
```

**`RobustRegression.fit`**

```python
RobustRegression.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 1000, tolerance: 'float' = 0.01, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.RobustRegression.from_formula('y ~ x1 + x2', base).fit().summary(style='stata')
```

**输出**

```text
Robust regression

Robust regression                               Number of obs     =        200
                                                F(  2,       197) =      81.25
                                                Prob > F          =     0.0000

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8381007   .0763045    10.98   0.000     .6876222    .9885792
          x2 |  -.5063725   .0787801    -6.43   0.000    -.6617331   -.3510119
       _cons |   1.019782   .0783655    13.01   0.000     .8652395    1.174325
```

**注意:** 报告的 `mss`/`rss`/`mse`/`rmse`/`r2`/`F` 描述的是对调整后响应变量的无权回归——也就是 Stata 真正发布的那个回归——而不是由最终双权重构造的加权平方和，后者是另一个量。

### `QuantileRegression`

**Stata 指令:** `qreg y x1 x2, quantile(.5)`

分位数回归，即 Stata 的 `qreg`。在 `quantile` 处最小化检验函数，用 IRLS 求解；由于目标函数是分段线性的，可能有多个参数向量达到同一最小值，求解器返回其中之一。标准误使用 Stata 的 `fitted` 稀疏度估计量——在 `quantile ± h` 处以 Hall-Sheather 带宽重新拟合回归——而不是对残差做核密度估计；后者会改变每一个标准误，却不动系数。两个报告细节容易误读：`e(sum_adev)`/`e(sum_rdev)` 是检验函数之和，而非绝对值之和；`e(q_v)`（打印为「(about q)」）是结果变量的第 `ceil(n*tau)` 个次序统计量，而不是插值分位数。

**签名**

```python
ws.QuantileRegression(endog: 'Any', exog: 'Any', *, quantile: 'float' = 0.5, param_names: 'list[str] | None' = None) -> 'None'
```

**`QuantileRegression.from_formula`**

```python
QuantileRegression.from_formula(formula: 'str', data: 'pd.DataFrame', *, quantile: 'float' = 0.5, subset: 'Any' = None) -> "'QuantileRegression'"
```

**`QuantileRegression.fit`**

```python
QuantileRegression.fit(self, cov_type: 'str | None' = 'iid', cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, reps: 'int' = 0, maxiter: 'int' = 2000, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.QuantileRegression.from_formula('y ~ x1 + x2', base, quantile=0.5).fit().summary(style='stata')
```

**输出**

```text

Median regression                                   Number of obs =        200
  Raw sum of deviations 112.7471 (about 1.0899456)
  Min sum of deviations   84.134                    Pseudo R2     =     0.2538

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8769292   .0996525     8.80   0.000     .6804067    1.073452
          x2 |  -.4154447   .1028856    -4.04   0.000    -.6183432   -.2125463
       _cons |   1.082302    .102344    10.58   0.000     .8804712    1.284132
```

**注意:** `e(marginsnotok)` 为 `"stdp stddp residuals"`，因此 `predict` 只提供 `xb`、`stdp` 与 `residuals`；`stddp` 是仅限 `sqreg` 的类型，在 `qreg` 之后调用会报错。

### `Bsqreg`

**Stata 指令:** `bsqreg y x1 x2, quantile(.5)`

自助法分位数回归，即 Stata 的 `bsqreg`。点估计就是普通的 `qreg` 估计，只有标准误来自自助法（对行重抽样并重拟合）。`reps` 默认为 200，`seed` 使抽样可复现；不给 seed 时会从系统熵中新抽一个种子，对应 Stata 使用会话当前的 `c(seed)` 而非固定常数——但抽到的值不会存到结果里，因此未给种子的拟合事后无法复现。结构上它就是单分位数的 `Sqreg`，这正是 `[R] qreg` 所述的等价关系，因此两者在重抽样方案上不可能不一致。

**签名**

```python
ws.Bsqreg(endog: 'Any', exog: 'Any', *, quantile: 'float' = 0.5, param_names: 'list[str] | None' = None) -> 'None'
```

**`Bsqreg.from_formula`**

```python
Bsqreg.from_formula(formula: 'str', data: 'pd.DataFrame', *, quantile: 'float' = 0.5, subset: 'Any' = None) -> "'Bsqreg'"
```

**`Bsqreg.fit`**

```python
Bsqreg.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, reps: 'int' = 200, seed: 'int | None' = None, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Bsqreg.from_formula('y ~ x1 + x2', base, quantile=0.5).fit(reps=50, seed=20261031).summary(style='stata')
```

**输出**

```text

Median regression, bootstrap(50) SEs                Number of obs =         200
  bootstrap(50) SEs                                   .5 Pseudo R2=      0.2538
              |               Bootstrap

------------------------------------------------------------------------------
             | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8769292   .1145663     7.65   0.000     .6509953    1.102863
          x2 |  -.4154447    .135737    -3.06   0.003    -.6831289   -.1477606
       _cons |   1.082302   .1080426    10.02   0.000     .8692332     1.29537
```

**注意:** bootstrap 种子记录在 `e(seed)` 中，因此**即使未指定种子**也可复现：把 `seed=result.e['seed']` 传回 `fit()`，各次重抽样会完全重复。Stata 通过 `c(seed)` 与 `set seed` 提供同样的保证。另外，行序在重抽样之间会延续（与 Stata 的 `bsampl_w`/`qreg_c` 循环一致），因此每次重抽样并非从原始数据顺序开始。

### `Sqreg`

**Stata 指令:** `sqreg y x1 x2, quantiles(.25 .5 .75)`

联合分位数回归，即 Stata 的 `sqreg`。一次拟合多个分位数，并对整个参数向量做自助法：每个重复抽一组样本，在该样本上拟合全部分位数，这正是协方差「联合」的含义——跨分位区块度量的是各分位数在同一次重抽样下如何一起变动，而独立抽样会破坏这一点。参数名带分位数前缀（`q0.25:x1`）。`predict(kind='stddp')` 给出前两个分位数线性预测之差的标差误，使用跨方程协方差，而不是把两个方差相加。

**签名**

```python
ws.Sqreg(endog: 'Any', exog: 'Any', *, quantiles: 'Sequence[float]' = (0.25, 0.5, 0.75), param_names: 'list[str] | None' = None) -> 'None'
```

**`Sqreg.from_formula`**

```python
Sqreg.from_formula(formula: 'str', data: 'pd.DataFrame', *, quantiles: 'Sequence[float]' = (0.25, 0.5, 0.75), subset: 'Any' = None) -> "'Sqreg'"
```

**`Sqreg.fit`**

```python
Sqreg.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, reps: 'int' = 200, seed: 'int | None' = None, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Sqreg.from_formula('y ~ x1 + x2', base, quantiles=(0.25, 0.75)).fit(reps=50, seed=20261031).summary(style='stata')
```

**输出**

```text

Simultaneous quantile regression                    Number of obs =         200
  bootstrap(50) SEs                                  .25 Pseudo R2=      0.2766
                                                     .75 Pseudo R2=      0.2865
              |               Bootstrap

---------------------------------------------------------------------------------
                | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
----------------+----------------------------------------------------------------
q25             |
          _cons |   .2844253   .1116347     2.55   0.012     .0642728    .5045777
             x1 |   .8812208   .0923716     9.54   0.000     .6990567    1.063385
             x2 |  -.5777819   .0853988    -6.77   0.000    -.7461952   -.4093686
----------------+----------------------------------------------------------------
q75             |
          _cons |   1.687185   .0782618    21.56   0.000     1.532847    1.841524
             x1 |   .8443399   .0861656     9.80   0.000     .6744144    1.014265
             x2 |  -.4856428   .0828392    -5.86   0.000    -.6490082   -.3222775
```

**注意:** `stddp` 需要至少两个分位数、且两个方程使用相同回归元的 `sqreg` 结果，否则会报错。默认 `reps=200` 使三分位数的拟合在 200 行数据上耗时约二十秒，因此上例传入了更小的 `reps`，并只取两个分位数。 种子记录在 `e(seed)` 中，把该值传回 `fit(seed=...)` 即可复现未指定种子的拟合。

### `NeweyWestResult`

**Stata 指令:** `newey y x1 x2, lag(2)`

`newey` 返回的对象：`regress` 的点估计套在 `newey` 更窄的结果表面之下。调用方会看到与普通 `StataResults` 的两处差别。`e()` 只含 `newey` 自身的标量与宏，因为 `newey` 发布的是全新的 `e()` 而非扩展 `regress` 的，所以 `r2`、`r2_a`、`rmse`、`sigma`、`rss`、`mss`、`ll` 都不存在。并且 `predict` 只接受 `xb`、`index`、`stdp` 与 `residuals`。直接构造它需要已拟合的结果与 `lag`；通常由 `ws.newey(formula, data, lag=...)` 返回。

**签名**

```python
ws.NeweyWestResult(results: 'Any', *, lag: 'int') -> 'None'
```

**示例**

```python
ws.newey('y ~ x1 + x2', base, lag=2).summary(style='stata')
```

**输出**

```text

Regression with Newey–West standard errors      Number of obs     =        200
Maximum lag = 2                                 F(  2,       197) =     107.69
                                                Prob > F          =     0.0000

------------------------------------------------------------------------------
             |             Newey–West
           y | Coefficient  std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8406445   .0662943    12.68   0.000     .7099069    .9713822
          x2 |   -.512087   .0707643    -7.24   0.000    -.6516398   -.3725342
       _cons |   1.005003   .0768392    13.08   0.000     .8534696    1.156535
```

**注意:** 请求其他任何预测类型都会抛错，并在消息中引述 `newey_p.ado` 的限制，而不是静默转发 `regress` 的十五种类型。

### `Nl`

**Stata 指令:** `nl y = {b0} * exp({b1} * x1)`

非线性最小二乘，即 Stata 的 `nl`。公式采用 Stata 的 `y = <表达式>` 形式，参数置于花括号中，例如 `'y = {b0} + {b1} * x1'`；系数表按 Stata 的 `/` 前缀显示它们（`/b0`）。初值来自 `start=` 或内联写法 `{b0=4}`；两者都没给时，用粗略的乘法网格搜索为优化器提供初值——这很重要，因为若某个参数的量级远离 1，否则根本无法到达。`has_constant` 由参数在表达式中的位置推断，决定总平方和是否中心化以及是否少算一个模型自由度。

**签名**

```python
ws.Nl(endog: 'Any', frame: 'pd.DataFrame', parameter_names: 'Sequence[str]', predictor: 'Callable[[Mapping[str, float], pd.DataFrame], np.ndarray]', *, has_constant: 'bool' = False, start: 'Mapping[str, float] | None' = None, formula: 'str' = '', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`Nl.from_formula`**

```python
Nl.from_formula(formula: 'str', data: 'pd.DataFrame', *, start: 'Mapping[str, float] | None' = None, subset: 'Any' = None) -> "'Nl'"
```

**`Nl.fit`**

```python
Nl.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 200, strict: 'bool' = True) -> 'StataResults'
```

**示例**

```python
ws.Nl.from_formula('y = {b0} + {b1} * x1', base).fit().summary(style='stata')
```

**输出**

```text

      Source |      SS            df       MS
-------------+----------------------------------   Number of obs   =       200
       Model |  148.66262         1  148.662619    F(1, 198)       =    109.75
    Residual |  268.20453       198  1.35456832    Prob > F        =    0.0000
-------------+----------------------------------   R-squared       =    0.3566
       Total |  416.86715       199  2.09480978    Adj R-squared   =    0.3534
                                                   Res. dev.       =    626.26

------------------------------------------------------------------------------
             | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         /b0 |   .9723568   .0824264    11.80   0.000     .8098105    1.134903
         /b1 |    .842525   .0804234    10.48   0.000     .6839286    1.001121
```

**注意:** `fit(strict=True)` 是默认值；优化器不收敛时它会抛 `ConvergenceError`，而不是返回未收敛的结果。该类不保留回归矩阵（`design_matrix` 为 `None`），且 `predict` 没有为 `nl` 注册任何类型，因此即使显式传入 `exog=` 也无法经由它做预测。


## 离散选择、计数与受限因变量

### `Logit`

**Stata 指令:** `logit yb x1 x2`

极大似然逻辑回归，即 Stata 的 `logit`。表头给出 LR 卡方模型检验、仅含常数项的对数似然与伪 R²，随后是对数几率尺度上的系数与 z 值。`e(aic)` 会被存下，且等于 `-2*e(ll) + 2*e(rank)`，与 `estat_ic` 打印的一致。`predict` 接受 `kind='xb'`、`'stdp'`、`'pr'`、`'residuals'` 与 `'score'`。

**签名**

```python
ws.Logit(endog: 'Any', exog: 'Any', *, param_names: 'list[str] | None' = None, freq_weights: 'Any' = None) -> 'None'
```

**`Logit.from_formula`**

```python
Logit.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None, freq_weights: 'str | None' = None) -> "'_DiscreteBase'"
```

**`Logit.fit`**

```python
Logit.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Logit.from_formula('yb ~ x1 + x2', base).fit().summary(style='stata')
```

**输出**

```text
Iteration 0:  Log likelihood = -137.98875  

Logistic regression                                     Number of obs =    200
                                                        LR chi2(2)    =  13.90
                                                        Prob > chi2   = 0.0010
Log likelihood = -131.03744                             Pseudo R2     = 0.0504

------------------------------------------------------------------------------
          yb | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .5292965   .1527099     3.47   0.001     .2299905    .8286024
          x2 |  -.1323303   .1488764    -0.89   0.374    -.4241227    .1594621
       _cons |   .1522173   .1474054     1.03   0.302    -.1366919    .4411265
------------------------------------------------------------------------------
```

**注意:** 预测类型写作单数 `score`，即方程级得分；复数 `scores` 在 Stata 中专用于多方程模型，因此 `predict(kind='scores')` 会报错而不是返回一列。频数权重经 `from_formula(..., freq_weights='w')` 支持，按行展开实现，这正是 Stata 此处 `fweight` 的含义。

### `Probit`

**Stata 指令:** `probit yb x1 x2`

极大似然 probit 回归，即 Stata 的 `probit`。表头区块与推断表的结构与 `Logit` 相同，但连接函数是标准正态分布函数，因此系数以潜在变量的标准差为单位，在同一数据上小于 logit 的系数。`predict` 提供 `'xb'`、`'stdp'`、`'pr'`、`'residuals'` 与 `'score'`。

**签名**

```python
ws.Probit(endog: 'Any', exog: 'Any', *, param_names: 'list[str] | None' = None, freq_weights: 'Any' = None) -> 'None'
```

**`Probit.from_formula`**

```python
Probit.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None, freq_weights: 'str | None' = None) -> "'_DiscreteBase'"
```

**`Probit.fit`**

```python
Probit.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Probit.from_formula('yb ~ x1 + x2', base).fit().summary(style='stata')
```

**输出**

```text
Iteration 0:  Log likelihood = -137.98875  

Probit regression                                       Number of obs =    200
                                                        LR chi2(2)    =  13.97
                                                        Prob > chi2   = 0.0009
Log likelihood = -131.00316                             Pseudo R2     = 0.0506

------------------------------------------------------------------------------
          yb | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .3293536   .0928298     3.55   0.000     .1474106    .5112966
          x2 |  -.0799735   .0916574    -0.87   0.383    -.2596188    .0996717
       _cons |   .0947076   .0908939     1.04   0.297    -.0834411    .2728564
------------------------------------------------------------------------------
```

**注意:** 与 `logit` 一样，得分类型写作单数 `score`；复数拼法是 Stata 的多方程形式，会被拒绝。与 `logit` 相同，频数权重按行展开实现。

### `OrderedLogit`

**Stata 指令:** `ologit yord x1 x2`

有序逻辑回归，即 Stata 的 `ologit`。有序结果的不同取值即类别（`e(k_cat)` 为其个数），表中给出斜率以及 `e(k_cat) - 1` 个切点，标记为 `/cut1`、`/cut2` 等，与 Stata 打印的标签一致。`predict(kind='pr')` 返回某一类别的概率；用 `outcome=` 指定是哪一类（默认为第一类），各概率之和为一。

**签名**

```python
ws.OrderedLogit(endog: 'Any', exog: 'Any', *, param_names: 'list[str] | None' = None, freq_weights: 'Any' = None) -> 'None'
```

**`OrderedLogit.from_formula`**

```python
OrderedLogit.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None, freq_weights: 'str | None' = None) -> "'_Ordered'"
```

**`OrderedLogit.fit`**

```python
OrderedLogit.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
d = base.assign(yord=base.ycnt.clip(0, 2))
ws.OrderedLogit.from_formula('yord ~ x1 + x2', d).fit().summary(style='stata')
```

**输出**

```text

Ordered logistic regression                             Number of obs =    200
                                                        LR chi2(2)    =  24.81
                                                        Prob > chi2   = 0.0000
Log likelihood = -202.12949                             Pseudo R2     = 0.0578

------------------------------------------------------------------------------
        yord | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .6788396   .1431435     4.74   0.000     .3982834    .9593957
          x2 |  -.0554988   .1342388    -0.41   0.679     -.318602    .2076043
-------------+----------------------------------------------------------------
       /cut1 |  -1.147585   .1700937                     -1.480962    -.814207
       /cut2 |   .2706843   .1493337                     -.0220043     .563373
------------------------------------------------------------------------------
```

**注意:** 模型假定结果是有序类别：若传入连续变量，每个不同取值都会成为一个类别，于是在连续响应上 `e(k_cat)` 等于样本量，并会与斜率一同估计 `k_cat - 1` 个切点。`predict` 不提供 `'score'` 类型——只有 `'xb'`、`'stdp'`、`'pr'` 和 `'residuals'`——因为得分会是每个方程一列。

### `OrderedProbit`

**Stata 指令:** `oprobit yord x1 x2`

有序 probit 回归，即 Stata 的 `oprobit`。它与 `OrderedLogit` 使用同一套有序类别机制，只是以正态分布代替逻辑分布，因此同样给出斜率与 `e(k_cat) - 1` 个切点；在同一数据上，LR 检验与对数似然处于可比量级，但系数系统性地更小（逻辑尺度相对正态尺度约需乘 1.6）。`predict(kind='pr')` 接受同样的可选参数 `outcome=`。

**签名**

```python
ws.OrderedProbit(endog: 'Any', exog: 'Any', *, param_names: 'list[str] | None' = None, freq_weights: 'Any' = None) -> 'None'
```

**`OrderedProbit.from_formula`**

```python
OrderedProbit.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None, freq_weights: 'str | None' = None) -> "'_Ordered'"
```

**`OrderedProbit.fit`**

```python
OrderedProbit.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
d = base.assign(yord=base.ycnt.clip(0, 2))
ws.OrderedProbit.from_formula('yord ~ x1 + x2', d).fit().summary(style='stata')
```

**输出**

```text

Ordered probit regression                               Number of obs =    200
                                                        LR chi2(2)    =  24.01
                                                        Prob > chi2   = 0.0000
Log likelihood = -202.52596                             Pseudo R2     = 0.0560

------------------------------------------------------------------------------
        yord | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .4063142   .0853207     4.76   0.000     .2390887    .5735398
          x2 |  -.0356842   .0825221    -0.43   0.665    -.1974245     .126056
-------------+----------------------------------------------------------------
       /cut1 |  -.6815503   .0986671                     -.8749343   -.4881663
       /cut2 |   .1743743   .0919787                     -.0059006    .3546491
------------------------------------------------------------------------------
```

**注意:** 只有离散结果才有意义：连续列会每个不同取值生成一个类别和 `k_cat - 1` 个切点，这样的模型不可识别。未注册 `'score'` 预测类型；可用类型为 `'xb'`、`'stdp'`、`'pr'` 与 `'residuals'`。

### `Poisson`

**Stata 指令:** `poisson ycnt x1 x2`

极大似然 Poisson 回归，即 Stata 的 `poisson`。系数在对数均值尺度上；表头打印 LR 卡方检验、`e(ll_0)` 与伪 R²。`predict(kind='n')` 与 `kind='ir'` 都给出预测均值（在本数据上数值完全相同），而 `kind='pr'` 需要指定计数：`predict(kind='pr', n=2)` 给出 `Pr(y = 2)`；不指定就调用 `'pr'` 会报错，而不是擅自假定一个取值。

**签名**

```python
ws.Poisson(endog: 'Any', exog: 'Any', *, param_names: 'list[str] | None' = None, freq_weights: 'Any' = None) -> 'None'
```

**`Poisson.from_formula`**

```python
Poisson.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None, freq_weights: 'str | None' = None) -> "'_DiscreteBase'"
```

**`Poisson.fit`**

```python
Poisson.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Poisson.from_formula('ycnt ~ x1 + x2', base).fit().summary(style='stata')
```

**输出**

```text

Poisson regression                                      Number of obs =    200
                                                        LR chi2(2)    =  35.40
                                                        Prob > chi2   = 0.0000
Log likelihood = -299.6568                              Pseudo R2     = 0.0558

------------------------------------------------------------------------------
        ycnt | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .3319199    .055595     5.97   0.000     .2229556    .4408841
          x2 |   .0025616   .0575378     0.04   0.964    -.1102105    .1153336
       _cons |   .3313258   .0621307     5.33   0.000     .2095519    .4530998
------------------------------------------------------------------------------
```

**注意:** `'pr'` 是 Stata 的 `pr(n)`/`pr(a, b)` 家族，而不是单一概率，因此需要传 `n=`。频数权重受支持，按行展开实现。

### `NegativeBinomial`

**Stata 指令:** `nbreg yod x1 x2`

负二项回归，NB2——Stata 默认的 `nbreg`。离散参数是 statsmodels 的 `alpha`，`Var = mu + alpha*mu^2`，与 Stata 默认的参数化相同；它在对数尺度上估计，报告为 `/lnalpha`，`e(alpha)` 为其指数。优化器用 BFGS 而非 Newton，因为一旦 `alpha` 进入参数向量，从 Poisson 初值出发的 Newton 步会发散。`e(aic)` 会被存下，且与 `estat_ic` 一致。

**签名**

```python
ws.NegativeBinomial(endog: 'Any', exog: 'Any', *, param_names: 'list[str] | None' = None, freq_weights: 'Any' = None) -> 'None'
```

**`NegativeBinomial.from_formula`**

```python
NegativeBinomial.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None, freq_weights: 'str | None' = None) -> "'_DiscreteBase'"
```

**`NegativeBinomial.fit`**

```python
NegativeBinomial.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
d = base.assign(yod=base.ycnt ** 2)
ws.NegativeBinomial.from_formula('yod ~ x1 + x2', d).fit().summary(style='stata')
```

**输出**

```text

Negative binomial regression                            Number of obs =    200
                                                        LR chi2(1)    =  33.33
Dispersion: mean                                        Prob > chi2   = 0.0000
Log likelihood = -472.86915                             Pseudo R2     = 0.0340

------------------------------------------------------------------------------
         yod | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .5272862    .090853     5.80   0.000     .3492176    .7053548
          x2 |   .0058862   .0894056     0.07   0.948    -.1693456    .1811179
       _cons |   1.194401   .0912093    13.10   0.000     1.015634    1.373168
-------------+----------------------------------------------------------------
    /lnalpha |   .2630268   .1342525                     -.0001032    .5261568
-------------+----------------------------------------------------------------
       alpha |   1.300862   .1746439                      .9998968    1.692416
------------------------------------------------------------------------------
LR test of alpha=0: chibar2(01) = 469.33              Prob >= chibar2 = 0.000
```

**注意:** `e(ll_c)` 是在 Poisson（即 `alpha = 0`）极限处的对数似然，`e(chi2_c)` 是它与 `e(ll)` 之差的二倍；在本数据上 `e(chi2_c)` 精确复现 `2*(e(ll) - e(ll_c))`，因此离散检验可直接由这些宏得到。未注册 `'score'` 预测类型，因为得分会是每个方程一列；`'pr'` 同样需要 `n=`。

### `NegativeBinomialNB1`

**Stata 指令:** `nbreg yod x1 x2, dispersion(constant)`

常数离散参数的负二项回归，NB1，即 Stata 的 `nbreg, dispersion(constant)`，其中 `Var = mu*(1 + delta)` 对均值是线性的，而非二次的。它与 `NegativeBinomial` 是不同的模型、不同的似然，因此单独成类，而不是后者上的一个选项。离散参数报告为 `/lndelta`，`e(delta)` 为其指数；似然依照 `nbreg_al.ado` 实现，其 gamma 项使用的是 `mu/delta` 而不是 `mu`。

**签名**

```python
ws.NegativeBinomialNB1(endog: 'Any', exog: 'Any', *, param_names: 'list[str] | None' = None, freq_weights: 'Any' = None) -> 'None'
```

**`NegativeBinomialNB1.from_formula`**

```python
NegativeBinomialNB1.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None, freq_weights: 'str | None' = None) -> "'_DiscreteBase'"
```

**`NegativeBinomialNB1.fit`**

```python
NegativeBinomialNB1.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Any' = None, ssc: 'Any' = None, *, maxiter: 'int' = 200, tol: 'float' = 1e-12, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
d = base.assign(yod=base.ycnt ** 2)
ws.NegativeBinomialNB1.from_formula('yod ~ x1 + x2', d).fit().summary(style='stata')
```

**输出**

```text

Negative binomial (NB1)                                 Number of obs =    200
                                                        LR chi2(2)    =  35.90
Log likelihood = -473.81429                             Prob > chi2   = 0.0000

------------------------------------------------------------------------------
             | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |    .439003   .0744434     5.90   0.000     .2930967    .5849093
          x2 |     .00851   .0752843     0.11   0.910    -.1390445    .1560645
       _cons |   1.242011   .0957695    12.97   0.000     1.054306    1.429715
    /lndelta |   1.602285   .1626415     9.85   0.000     1.283513    1.921056
------------------------------------------------------------------------------
```

**注意:** 没有 `e(aic)`、`e(bic)` 与 `e(r2_p)`，且 `e(cmd)` 是描述性字符串 `'Negative binomial (NB1)'` 而非 Stata 命令名，因此 `predict` 不为其注册任何类型：需要信息准则时请调用 `estat_ic`。稳健与聚类 VCE 会按名称被拒绝——只实现了 `cov_type='nonrobust'`。

### `ZeroInflatedPoisson`

**Stata 指令:** `zip ycnt x1 x2, inflate(x3)`

零膨胀 Poisson，即 Stata 的 `zip`。计数方程是 Poisson，膨胀方程是 `inflate=` 所指定变量上的二值 logit；两个区块出现在同一张表中，膨胀部分的行以 `inflate#` 为前缀。`e(n_zero)` 与 `e(n_nonzero)` 统计估计样本中的零与非零个数。当 `inflate=None` 时没有混合成分，模型就是普通 Poisson 回归，用同一似然拟合。

**签名**

```python
ws.ZeroInflatedPoisson(endog: 'Any', exog: 'Any', *, inflate_exog: 'Any' = None, inflate_names: 'list[str] | None' = None, param_names: 'list[str] | None' = None) -> 'None'
```

**`ZeroInflatedPoisson.from_formula`**

```python
ZeroInflatedPoisson.from_formula(formula: 'str', data: 'pd.DataFrame', *, inflate: 'str | None' = None, subset: 'Any' = None, freq_weights: 'str | None' = None) -> "'_ZeroInflatedBase'"
```

**`ZeroInflatedPoisson.fit`**

```python
ZeroInflatedPoisson.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.ZeroInflatedPoisson.from_formula('ycnt ~ x1 + x2', base, inflate='x3').fit().summary(style='stata')
```

**输出**

```text

Zero-inflated Poisson regression                        Number of obs =    200
Inflation model: logit                                  Nonzero obs   =    148
                                                        Zero obs      =     52
                                                        LR chi2(2)    =  33.03
Log likelihood = -299.2566                              Prob > chi2   = 0.0000

------------------------------------------------------------------------------
        ycnt | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
ycnt         |
          x1 |   .3327988   .0562969     5.91   0.000     .2224588    .4431387
          x2 |   .0039316   .0591759     0.07   0.947     -.112051    .1199142
       _cons |   .3681895   .0761911     4.83   0.000     .2188577    .5175212
-------------+----------------------------------------------------------------
inflate      |
          x3 |  -.3229142   1.022661    -0.32   0.752    -2.327293    1.681464
       _cons |  -3.374413   1.491321    -2.26   0.024    -6.297348   -.4514782
------------------------------------------------------------------------------
```

**注意:** `predict` 只支持 `'xb'`、`'n'`、`'pr'` 与 `'stdp'`；`'stdf'` 及其他任何类型都会被拒绝。`'pr'` 是膨胀概率 `logistic(w'g)`，不是计数概率；若模型以 `inflate=None` 拟合，调用它会报错。`e(df_m)` 只计主方程中常数项以外的回归元，因此打印的 `LR chi2` 自由度小于 `e(rank)`。

### `ZeroInflatedNegativeBinomial`

**Stata 指令:** `zinb yz x1 x2, inflate(x3)`

零膨胀负二项回归，NB2，即 Stata 的 `zinb`。它由一条负二项计数方程加一条 logit 膨胀方程组成；离散参数在对数尺度上估计为 `/lnalpha`，与 `nbreg` 的存法一致。出于与 `NegativeBinomial` 相同的原因，它使用准牛顿优化器。当 `inflate=None` 时退化为普通的 NB2 拟合。

**签名**

```python
ws.ZeroInflatedNegativeBinomial(endog: 'Any', exog: 'Any', *, inflate_exog: 'Any' = None, inflate_names: 'list[str] | None' = None, param_names: 'list[str] | None' = None) -> 'None'
```

**`ZeroInflatedNegativeBinomial.from_formula`**

```python
ZeroInflatedNegativeBinomial.from_formula(formula: 'str', data: 'pd.DataFrame', *, inflate: 'str | None' = None, subset: 'Any' = None, freq_weights: 'str | None' = None) -> "'_ZeroInflatedBase'"
```

**`ZeroInflatedNegativeBinomial.fit`**

```python
ZeroInflatedNegativeBinomial.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
rng = np.random.default_rng(0)
mu = np.exp(0.4 + 0.6 * base.x1 + 0.2 * base.x2)
cnt = rng.negative_binomial(1 / 0.8, 1 / (1 + 0.8 * mu))
d = base.assign(yz=np.where(rng.uniform(size=200) < 1 / (1 + np.exp(-base.x3)), 0, cnt))
ws.ZeroInflatedNegativeBinomial.from_formula('yz ~ x1 + x2', d, inflate='x3').fit().summary(style='stata')
```

**输出**

```text

Zero-inflated negative binomial regression              Number of obs =    200
Inflation model: logit                                  Nonzero obs   =     66
                                                        Zero obs      =    134
                                                        LR chi2(2)    =  25.08
Log likelihood = -235.1402                              Prob > chi2   = 0.0000

------------------------------------------------------------------------------
          yz | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
yz           |
          x1 |    .523662    .121246     4.32   0.000     .2860242    .7612998
          x2 |    .306602   .1051965     2.91   0.004     .1004207    .5127834
       _cons |   .5377467   .1747467     3.08   0.002     .1952495    .8802439
-------------+----------------------------------------------------------------
inflate      |
          x3 |   .8042556   .2505497     3.21   0.001     .3131873    1.295324
       _cons |    .290147   .2802637     1.04   0.301    -.2591598    .8394537
    /lnalpha |  -.9197992   .5600437    -1.64   0.101    -2.017465    .1778664
-------------+----------------------------------------------------------------
       alpha |   .3985991   .2232329                      .1329922    1.194666
------------------------------------------------------------------------------
LR test of alpha=0: chibar2(01) = 25.08               Prob >= chibar2 = 0.000
```

**注意:** `predict` 只提供 `'xb'`、`'n'`、`'pr'` 与 `'stdp'`；`'pr'` 是膨胀概率，当 `inflate=None` 时会报错。若数据没有真正的超额零，膨胀方程不可识别：估计会被推向边界，信息矩阵可能奇异，此时报告为 `ConvergenceError`，而不是给出一张满是巨大标准误的表。

### `TruncatedPoisson`

**Stata 指令:** `ztp ycnt x1 x2`

截断到 `[lower, upper]` 的 Poisson 回归——`lower=1` 且无上界时即 Stata 的 `ztp`，其他限制则为 `tpoisson`；`model_name`（`e(cmd)`）随之变化，因此同一个类会按限制的不同报告 `ztp` 或 `tpoisson`。每个观测都必须落在限制之内：构造函数会抛出 `ConsistencyError` 并说明有多少观测越界，而不是静默剔除。

**签名**

```python
ws.TruncatedPoisson(endog: 'Any', exog: 'Any', *, lower: 'float | None' = 1, upper: 'float | None' = None, param_names: 'list[str] | None' = None) -> 'None'
```

**`TruncatedPoisson.from_formula`**

```python
TruncatedPoisson.from_formula(formula: 'str', data: 'pd.DataFrame', *, lower: 'float | None' = 1, upper: 'float | None' = None, subset: 'Any' = None) -> "'_TruncatedBase'"
```

**`TruncatedPoisson.fit`**

```python
TruncatedPoisson.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
d = base[base.ycnt > 0]
ws.TruncatedPoisson.from_formula('ycnt ~ x1 + x2', d).fit().summary(style='stata')
```

**输出**

```text

Zero-truncated Poisson regression                       Number of obs =    148
                                                        LR chi2(2)    =  35.95
                                                        Prob > chi2   = 0.0000
Log likelihood = -187.9949                              Pseudo R2     = 0.0873

------------------------------------------------------------------------------
        ycnt | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .3660199     .07135     5.13   0.000     .2261764    .5058634
          x2 |   .0715222   .0709734     1.01   0.314     -.067583    .2106274
       _cons |   .3364123    .088168     3.82   0.000     .1636061    .5092184
------------------------------------------------------------------------------
```

**注意:** 截断意味着越界的行本就不在样本中，因此调用方必须先取子集——本类不会替你取子集，这与 `Truncreg` 的行为相反。`predict` 只提供 `'xb'`、`'n'` 与 `'stdp'`，没有 `'pr'`。

### `TruncatedNegativeBinomial`

**Stata 指令:** `tnbreg yod x1, ll(1)`

截断到 `[lower, upper]` 的负二项回归，即 Stata 的 `tnbreg`。NB2 离散参数与斜率一同估计，报告为 `/lnalpha`；与 `TruncatedPoisson` 一样，数据必须已经取到限制内部的子集，否则构造函数会报错。`predict` 只提供 `'xb'`、`'n'` 与 `'stdp'`。

**签名**

```python
ws.TruncatedNegativeBinomial(endog: 'Any', exog: 'Any', *, lower: 'float | None' = 1, upper: 'float | None' = None, param_names: 'list[str] | None' = None) -> 'None'
```

**`TruncatedNegativeBinomial.from_formula`**

```python
TruncatedNegativeBinomial.from_formula(formula: 'str', data: 'pd.DataFrame', *, lower: 'float | None' = 1, upper: 'float | None' = None, subset: 'Any' = None) -> "'_TruncatedBase'"
```

**`TruncatedNegativeBinomial.fit`**

```python
TruncatedNegativeBinomial.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
d = base[base.ycnt > 0].assign(yod=lambda f: f.ycnt ** 2)
ws.TruncatedNegativeBinomial.from_formula('yod ~ x1', d).fit().summary(style='stata')
```

**输出**

```text

Truncated negative binomial regression                  Number of obs =    148
Truncation point = 1                                    LR chi2(1)    =  19.31
                                                        Prob > chi2   = 0.0000
Log likelihood = -360.61677                             Pseudo R2     = 0.0261

------------------------------------------------------------------------------
         yod | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .5768012   .1302026     4.43   0.000     .3216088    .8319936
       _cons |   .8655364   .2784346     3.11   0.002     .3198146    1.411258
-------------+----------------------------------------------------------------
    /lnalpha |   .9244537   .4734482                     -.0034877    1.852395
-------------+----------------------------------------------------------------
       alpha |   2.520491   1.193322                      .9965184     6.37507
------------------------------------------------------------------------------
LR test of alpha=0: chibar2(01) = 19.31               Prob >= chibar2 = 0.000
```

**注意:** 当负二项退化到 Poisson 极限时，优化器会警告未完全收敛；系数表仍会输出，因此在采信离散参数之前请先看该警告。未注册 `'pr'` 或 `'score'` 类型。

### `Tobit`

**Stata 指令:** `tobit ycens x1 x2, ll(0)`

带左和/或右删失的 Tobit 回归，即 Stata 的 `tobit`，默认 `lower=0`，与 Stata 的 `ll(0)` 一致。删失观测贡献其删失概率，未删失观测贡献正态密度；`/sigma` 被估计，`e(sigma) = var_e` 由之导出。`e(N_lc)`、`e(N_rc)` 与 `e(N_unc)` 分别统计左删失、右删失与未删失观测个数，`e(df_r)` 为残差自由度。

**签名**

```python
ws.Tobit(endog: 'Any', exog: 'Any', *, lower: 'float | None' = 0.0, upper: 'float | None' = None, param_names: 'list[str] | None' = None) -> 'None'
```

**`Tobit.from_formula`**

```python
Tobit.from_formula(formula: 'str', data: 'pd.DataFrame', *, lower: 'float | None' = 0.0, upper: 'float | None' = None, subset: 'Any' = None) -> "'Tobit'"
```

**`Tobit.fit`**

```python
Tobit.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 1000, tol: 'float' = 1e-10, strict: 'bool' = False, start_params: 'np.ndarray | None' = None) -> 'StataResults'
```

**示例**

```python
ws.Tobit.from_formula('ycens ~ x1 + x2', base).fit().summary(style='stata')
```

**输出**

```text

Tobit regression                                    Number of obs     =    200
                                                           Uncensored =    158
Limits: Lower =    0                                    Left-censored =     42
        Upper = +inf                                   Right-censored =      0

                                                    LR chi2(2)        = 112.26
                                                    Prob > chi2       = 0.0000
Log likelihood = -252.67183                         Pseudo R2         = 0.1818

------------------------------------------------------------------------------
       ycens | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |    .849565   .0730346    11.63   0.000     .7055396    .9935905
          x2 |   .0661611   .0717033     0.92   0.357    -.0752391    .2075613
       _cons |   .9811939   .0729483    13.45   0.000     .8373387    1.125049
-------------+----------------------------------------------------------------
 var(e.ycens)|   .9382956   .1089417                      .7462808    1.179715
------------------------------------------------------------------------------
```

**注意:** `tobit` 不存 `e(aic)`；`estat_ic` 会按需计算 AIC 与 BIC，因此准则仍然可用。`predict` 提供 `'xb'`、`'stdp'`、`'stdf'`、`'e'`、`'ystar'`、`'pr'` 与 `'residuals'`；`'e'` 与 `'ystar'` 用 `lower=`/`upper=` 传入删失界限，而不传时 `'pr'` 对每个观测都返回 1，因为默认区间覆盖整条实轴。只实现了 `cov_type='nonrobust'`。

### `Truncreg`

**Stata 指令:** `truncreg y x1 x2, ll(0)`

截断回归，即 Stata 的 `truncreg`。每个观测贡献正态密度除以落在限制区间内的概率，且至少要给出一个限制。处于或超出限制的观测被从估计样本中剔除并报告个数：`e(N_bf)` 是截断前的观测数，因此 `e(N_bf) - e(N)` 即被剔除的个数，也就是 Stata 打印的 `(N obs truncated)`。

**签名**

```python
ws.Truncreg(endog: 'Any', exog: 'Any', *, lower: 'float | None' = None, upper: 'float | None' = None, param_names: 'list[str] | None' = None) -> 'None'
```

**`Truncreg.from_formula`**

```python
Truncreg.from_formula(formula: 'str', data: 'pd.DataFrame', *, lower: 'float | None' = None, upper: 'float | None' = None, subset: 'Any' = None) -> "'Truncreg'"
```

**`Truncreg.fit`**

```python
Truncreg.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Truncreg.from_formula('y ~ x1 + x2', base, lower=0.0).fit().summary(style='stata')
```

**输出**

```text

Limit: Lower =    0                             Number of obs     =  151
       Upper = +inf                             Wald chi2(2)      = 65.66
                                                Prob > chi2       = 0.0000
Log likelihood = -169.3211                      Pseudo R2         = 0.1635

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |    .663382   .0974009     6.81   0.000     .4724797    .8542842
          x2 |  -.4890241   .0918261    -5.33   0.000    -.6689999   -.3090484
       _cons |   1.234907   .1145149    10.78   0.000     1.010462    1.459352
-------------+----------------------------------------------------------------
      /sigma |   .9158148   .0717033                      .7752789    1.056351
------------------------------------------------------------------------------
```

**注意:** 与 `TruncatedPoisson` 相反，本类会自行剔除越界的行并给出警告，而不是报错，因为 `truncreg` 的样本本就由限制定义。被剔除的个数是模型上的属性（`_n_truncated`），并**不**存入 `e()`：`e()` 遵循 `truncreg.sthlp`，其中没有这个标量——请用 `e(N_bf) - e(N)`。`predict` 提供 `'xb'`、`'stdp'`、`'stdf'`、`'residuals'` 与 `'fitted'`。

### `Intreg`

**Stata 指令:** `intreg lo hi x1`

区间回归，即 Stata 的 `intreg`，用双边公式 `'lo hi ~ x'` 书写。每个观测只知区间：`lower < upper` 为区间删失，`lower == upper` 为精确观测值，某一端缺失表示该侧删失。`e(N_int)`、`e(N_lc)`、`e(N_rc)` 与 `e(N_unc)` 统计这四类，`/lnsigma` 被估计，`e(sigma)` 由之导出。

**签名**

```python
ws.Intreg(lower: 'Any', upper: 'Any', exog: 'Any', *, param_names: 'list[str] | None' = None) -> 'None'
```

**`Intreg.from_formula`**

```python
Intreg.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None) -> "'Intreg'"
```

**`Intreg.fit`**

```python
Intreg.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, tol: 'float' = 1e-10, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
d = base.assign(lo=np.floor(base.y1), hi=np.floor(base.y1) + 1)
ws.Intreg.from_formula('lo hi ~ x1', d).fit().summary(style='stata')
```

**输出**

```text

Interval regression                                 Number of obs     =    200
                                                           Uncensored =      0
                                                        Left-censored =      0
                                                       Right-censored =      0
                                                    Interval-cens.    =    200

                                                    LR chi2(1)        =  59.68
                                                    Prob > chi2       = 0.0000
Log likelihood = -292.97704                         Pseudo R2         = 0.0924

------------------------------------------------------------------------------
       lo hi | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .6033076   .0723365     8.34   0.000     .4615308    .7450845
       _cons |   .9150882   .0741438    12.34   0.000      .769769    1.060408
-------------+----------------------------------------------------------------
    /lnsigma |   .0063061   .0540379                     -.0996063    .1122185
-------------+----------------------------------------------------------------
       sigma |   1.006326   .0543798                      .9051937    1.118757
------------------------------------------------------------------------------
```

**注意:** 若所有 `hi - lo` 相对模型残差尺度都极小，`sigma` 会被推向零，其余参数的标准误将变得不可用（大于所报数值）——此时拟合本质上成了精确插值，请给出与结果尺度相称的区间宽度。`predict` 提供 `'xb'`、`'stdp'`、`'stdf'`、`'e'`、`'ystar'`、`'pr'` 与 `'fitted'`；在此 `'e'`、`'ystar'`、`'fitted'` 都返回线性预测值，且没有 `'residuals'` 类型。

### `Heckman`

**Stata 指令:** `heckman yselect x1, select(sel = x3)`

Heckman 样本选择模型，即 Stata 的 `heckman`，写作 `'y ~ x + select(s ~ z)'`。选择方程是在全样本上的 probit；结果方程只在 `s == 1` 处拟合，未选中的观测以 `y` 缺失为标志，因此结果列在未选中行上可以是 NaN。`fit(method='twostep')`（默认）是两步法，把逆米尔斯比作为回归元加入，并有 `e(lambda) = rho*sigma`；`fit(method='ml')` 拟合联合似然，参数为 `/athrho` 与 `/lnsigma`。

**签名**

```python
ws.Heckman(outcome: 'np.ndarray', outcome_exog: 'np.ndarray', selection: 'np.ndarray', selection_exog: 'np.ndarray', *, outcome_names: 'Sequence[str]', selection_names: 'Sequence[str]', selected: 'np.ndarray', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`Heckman.from_formula`**

```python
Heckman.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None) -> "'Heckman'"
```

**`Heckman.fit`**

```python
Heckman.fit(self, method: 'str' = 'twostep', cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 200, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Heckman.from_formula('yselect ~ x1 + select(sel ~ x3)', base).fit().summary(style='stata')
```

**输出**

```text

Heckman selection model                                 Number of obs =    124
                                                        Wald chi2(1)  = 228.30
                                                        Prob > chi2   = 0.0000
Log likelihood = -94.069719                             Pseudo R2     = 0.4061

------------------------------------------------------------------------------
             | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
-------------+----------------------------------------------------------------
y            |
          x1 |   .6667734    .044129    15.11   0.000     .5802821    .7532646
      lambda |  -.1293741   .1393244    -0.93   0.353    -.4024449    .1436968
       _cons |   1.026394   .0807803    12.71   0.000     .8680676     1.18472
-------------+----------------------------------------------------------------
select       |
          x3 |   .8066883   .1199658     6.72   0.000     .5715598    1.041817
       _cons |   .4765848    .105997     4.50   0.000     .2688345     .684335
------------------------------------------------------------------------------
-------------+----------------------------------------------------------------
         rho | -.24996944   (fraction of variance due to u_i)
------------------------------------------------------------------------------
```

**注意:** 两步法结果不存 `e(aic)`（两步估计量不是联合模型的 ML 拟合），`e(N_selected)`/`e(N_nonselected)` 划分样本，而 `e(N)` 是全样本量，尽管打印的 `Number of obs` 是选中样本数。在 ML 路径上 `e(aic)` 会被存下，且等于 `-2*e(ll) + 2*k`，与 `estat_ic` 一致。标准误是两步 M 估计的夹心形式，而非朴素 OLS 标准误。`predict` 提供 `'xb'`、`'stdp'`、`'stdf'`、`'fitted'`、`'psel'`、`'ycond'`、`'yexpected'`、`'mills'` 与 `'nshazard'`。

### `Etregress`

**Stata 指令:** `etregress ytreat x1, treat(treat = x3)`

用控制函数法实现的内生处理回归，即 Stata 的 `etregress`，写作 `'y ~ x + treat(t ~ z)'`。处理方程是在全样本上的 probit，其广义残差被加入结果回归，从而吸收两个误差项之间的相关；与 `Heckman` 不同，处理组与对照组的 `y` 都可观测，因此每一行都进入结果方程。处理指示变量会被自动加入结果方程，若在公式中再写一次也不会重复。

**签名**

```python
ws.Etregress(outcome: 'np.ndarray', outcome_exog: 'np.ndarray', treatment: 'np.ndarray', treatment_exog: 'np.ndarray', *, outcome_names: 'Sequence[str]', treatment_names: 'Sequence[str]', treated: 'np.ndarray', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`Etregress.from_formula`**

```python
Etregress.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None) -> "'Etregress'"
```

**`Etregress.fit`**

```python
Etregress.fit(self, method: 'str' = 'control', cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 200, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Etregress.from_formula('ytreat ~ x1 + treat(treat ~ x3)', base).fit().summary(style='stata')
```

**输出**

```text

Linear regression with endogenous treatment             Number of obs =    200
Estimator: Two-step                                     Wald chi2(2)  = 142.69
                                                        Prob > chi2   = 0.0000
Log likelihood = -281.73931                             Pseudo R2     = 0.2059

---------------------------------------------------------------------------------
                | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
----------------+----------------------------------------------------------------
----------------+----------------------------------------------------------------
y               |
          _cons |   .8850983   .1885729     4.69   0.000     .5155021    1.254694
             x1 |    .776287   .0686343    11.31   0.000     .6417662    .9108077
          treat |   1.539547   .2729629     5.64   0.000      1.00455    2.074545
         lambda |  -.0033512   .1888875    -0.02   0.986    -.3735639    .3668615
----------------+----------------------------------------------------------------
treat           |
          _cons |   .5573098   .1093099     5.10   0.000     .3430664    .7715532
             x3 |   .8581738   .1263191     6.79   0.000     .6105928    1.105755
---------------------------------------------------------------------------------
----------------+----------------------------------------------------------------
         rho | -.00342009   (fraction of variance due to u_i)
---------------------------------------------------------------------------------
```

**注意:** `e(N)` 是全样本量，`e(N_treated)` 是处理组个数。标准误是两步 M 估计的夹心形式（`e(se_corrected) = 1`），既非朴素 OLS 标准误，也非 Stata 的 ML 标准误。`predict` 提供 `'xb'`、`'stdp'`、`'stdf'`、`'fitted'`、`'cte'`、`'yctrt'`、`'ycntrt'`、`'ptrt'` 与 `'xbtrt'`，即处理组/对照组条件均值、处理概率与处理方程的线性指数。

### `Frontier`

**Stata 指令:** `frontier y x1 x2, distribution(halfnormal)`

随机生产前沿，即 Stata 的 `frontier`。模型为 `y = x'b + v - u`，其中 `v` 是对称的双边误差，`u` 是单边无效率项；接受 `distribution='halfnormal'`（默认）或 `'exponential'`，截断正态变体未实现，会抛出 `UnsupportedError`。表中按对数方差尺度报告 `/lnsig2v` 与 `/lnsig2u`，`e()` 中则带有导出的 `sigma_v`、`sigma_u`、`sigma2_v`、`sigma2_u`、`lambda` 与 `gamma`。

**签名**

```python
ws.Frontier(endog: 'Any', exog: 'Any', *, param_names: 'Sequence[str]', distribution: 'str' = 'halfnormal', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`Frontier.from_formula`**

```python
Frontier.from_formula(formula: 'str', data: 'pd.DataFrame', *, distribution: 'str' = 'halfnormal', subset: 'Any' = None) -> "'Frontier'"
```

**`Frontier.fit`**

```python
Frontier.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Frontier.from_formula('y ~ x1 + x2', base).fit().summary(style='stata')
```

**输出**

```text

Stoc. frontier normal/half-normal model                 Number of obs =    200
                                                        Wald chi2(2)  = 180.57
Log likelihood = -291.49689                             Prob > chi2   = 0.0000

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8347526   .0727708    11.47   0.000     .6921246    .9773807
          x2 |  -.5112541   .0737976    -6.93   0.000    -.6558948   -.3666135
       _cons |   1.768313   .3472315     5.09   0.000     1.087752    2.448874
    /lnsig2v |  -.2832233   .3824229    -0.74   0.459    -1.032758    .4663118
-------------+----------------------------------------------------------------
    /lnsig2u |  -.0917742   .8897731                     -1.835697    1.652149
-------------+----------------------------------------------------------------
     sigma_u |   .9551498                                        0           .
------------------------------------------------------------------------------
```

**注意:** 没有 `e(aic)` 或 `e(bic)`，模型检验是 Wald 统计量（`e(chi2type)='Wald'`），对「无无效率项」模型的似然比统计量存于 `e(lr_chi2)`，其 p 值存于 `e(p_lr)`。`predict` 未为 `frontier` 注册任何类型，因此预测只能经由宏或手工完成。

### `XtLogit`

**Stata 指令:** `xtlogit yb x1, fe`

面板 logit，即 Stata 的 `xtlogit`，`method` 可为 `'fe'`、`'re'` 或 `'pa'`，`model_name` 相应设为 `'xtlogit, fe'`、`'xtlogit, re'` 或 `'xtlogit, pa'`。`fe` 是 Chamberlain 条件似然：结果从不变化的组被剔除，`e(N_drop)` 与 `e(N_group_drop)` 记录其代价，模型无截距，检验为 LR。`re` 是带高斯随机效应的积分似然，用自适应 Gauss-Hermite 求积，把 `/lnsig2u` 作为参数存下，并给出 `e(sigma_u)` 与 `e(rho)`。

**签名**

```python
ws.XtLogit(endog: 'Any', exog: 'Any', groups: 'Any', *, param_names: 'Sequence[str]', method: 'str', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`XtLogit.from_formula`**

```python
XtLogit.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str | None' = None, method: 'str' = 're', subset: 'Any' = None) -> "'XtLogit'"
```

**`XtLogit.fit`**

```python
XtLogit.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, intpoints: 'int' = 12, maxiter: 'int' = 500, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.XtLogit.from_formula('yb ~ x1', base, entity='firm', method='fe').fit().summary(style='stata')
```

**输出**

```text
note: multiple positive outcomes within groups encountered.
note: 3 groups (15 obs) omitted because of all positive or
      all negative outcomes.

Conditional fixed-effects logistic regression        Number of obs    =    185
Group variable: firm                                 Number of groups =     37

                                                     Obs per group:
                                                                  min =      5
                                                                  avg =    5.0
                                                                  max =      5

                                                     LR chi2(1)       =  13.54
Log likelihood = -71.492167                          Prob > chi2      = 0.0002

------------------------------------------------------------------------------
          yb | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .6103907   .1789254     3.41   0.001     .2597033    .9610781
```

**注意:** `method='pa'` 是带可交换工作相关的 GEE，**不是**似然：它不存 `e(ll)`、`e(ll_0)`、`e(r2_p)`、`e(aic)` 或 `e(bic)`，模型检验是 Wald 统计量，并以 `e(estimator)='gee'` 标记。`fe` 路径上的 `e(ll_0)` 是在保留样本上仅含常数项的 logit，这正是 LR 统计量可比的原因。三者都没有注册任何 `predict` 类型。

### `XtProbit`

**Stata 指令:** `xtprobit ybp x1, re`

面板 probit，即 Stata 的 `xtprobit`，仅有随机效应。它与 `xtlogit, re` 是同一个积分，只把逻辑分布换成正态分布函数；把 `/lnsig2u` 作为参数存下，并给出 `e(sigma_u)` 与 `e(rho)`。其他任何 `method` 都会抛出 `UnsupportedError`：Stata 在此形态下没有固定效应或总体平均的 `xtprobit`，因为 probit 没有可供条件的充分统计量。

**签名**

```python
ws.XtProbit(endog: 'Any', exog: 'Any', groups: 'Any', *, param_names: 'Sequence[str]', method: 'str', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`XtProbit.from_formula`**

```python
XtProbit.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str | None' = None, method: 'str' = 're', subset: 'Any' = None) -> "'XtProbit'"
```

**`XtProbit.fit`**

```python
XtProbit.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, intpoints: 'int' = 12, maxiter: 'int' = 500, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
d = base.assign(ybp=(base.ypanel > 1).astype(int))
ws.XtProbit.from_formula('ybp ~ x1', d, entity='firm', method='re').fit().summary(style='stata')
```

**输出**

```text

Random-effects probit regression                     Number of obs    =    200
Group variable: firm                                 Number of groups =     40

Random effects u_i ~ Gaussian                        Obs per group:
                                                                  min =      5
                                                                  avg =    5.0
                                                                  max =      5

Integration method: mvaghermite                      Integration pts. =     12

                                                     Wald chi2(1)     =  31.47
Log likelihood = -115.42465                          Prob > chi2      = 0.0000

------------------------------------------------------------------------------
         ybp | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .6975478    .124335     5.61   0.000     .4538557    .9412399
       _cons |  -.1688001   .1333415    -1.27   0.206    -.4301446    .0925444
-------------+----------------------------------------------------------------
    /lnsig2u |  -1.211642   .6409358                     -2.467853    .0445689
-------------+----------------------------------------------------------------
     sigma_u |   .5456262                                        0           .
         rho |   .2294106                                        0           .
------------------------------------------------------------------------------
LR test of rho=0: chibar2(01) = 5.54                   Prob >= chibar2 = 0.009
```

**注意:** `method='fe'` 与 `method='pa'` 会按名称被拒绝，报错为 `method='fe' is not available; supported: 're'`（`'pa'` 同），而不是静默映射为 `re`。在组内相关很弱的二值结果上，`/lnsig2u` 的估计可能被推得很负，于是 `sigma_u` 实际为零——这是模型撞到边界，而非未收敛。

### `XtPoisson`

**Stata 指令:** `xtpoisson ycnt x1, fe`

面板 Poisson，即 Stata 的 `xtpoisson`，`method` 可为 `'fe'` 或 `'re'`。`fe` 以组总量 `S_i = sum_t y_it` 为条件，精确消去组效应（即多项分布归约），因此是真正的似然，`S_i = 0` 的组被剔除。`re` 使用与 Poisson 共轭的 gamma 随机效应，组似然有闭式，不涉及求积；离散参数报告为 `/lnalpha`，`e(alpha)` 为其指数。

**签名**

```python
ws.XtPoisson(endog: 'Any', exog: 'Any', groups: 'Any', *, param_names: 'Sequence[str]', method: 'str', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`XtPoisson.from_formula`**

```python
XtPoisson.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str | None' = None, method: 'str' = 're', subset: 'Any' = None) -> "'XtPoisson'"
```

**`XtPoisson.fit`**

```python
XtPoisson.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, intpoints: 'int' = 12, maxiter: 'int' = 500, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.XtPoisson.from_formula('ycnt ~ x1', base, entity='firm', method='fe').fit().summary(style='stata')
```

**输出**

```text

Conditional fixed-effects Poisson regression         Number of obs    =    200
Group variable: firm                                 Number of groups =     40

                                                     Obs per group:
                                                                  min =      5
                                                                  avg =    5.0
                                                                  max =      5

                                                     Wald chi2(1)     =  25.31
Log likelihood = -203.70227                          Prob > chi2      = 0.0000

------------------------------------------------------------------------------
        ycnt | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .3128918   .0621931     5.03   0.000     .1909956     .434788
```

**注意:** `method='pa'` 会被拒绝（`supported: 'fe', 're'`）。`fe` 似然没有截距——组效应吸收了它，似然在常数方向上完全平坦——因此 patsy 的截距被移除而不是估计。本类不填 `e(ivar)` 与 `e(tvar)`，因此打印的 `Group variable:` 行显示为 `.`，尽管 `e(N_g)` 确实统计了组数。`predict` 未注册任何类型。

### `XtNbreg`

**Stata 指令:** `xtnbreg ycnt x1, fe`

条件固定效应负二项，即 Stata 的 `xtnbreg, fe`，且只有 `fe` 可用。Stata 最大化的是 Hausman-Hall-Griliches（1984）条件核，而非条件 Poisson 核：离散参数通过 `ln Gamma(lam + y) - ln Gamma(lam)` 项进入，这正是同一数据上 `xtnbreg, fe` 与 `xtpoisson, fe` 报告不同标准误的原因。`e(estimator)='hhg_conditional_ml'` 与 `e(quasi_likelihood)=1` 标记这一点，`e(ll)` 是最大化的条件对数核——也就是 Stata 在 `Log likelihood` 下打印的值。

**签名**

```python
ws.XtNbreg(endog: 'Any', exog: 'Any', groups: 'Any', *, param_names: 'Sequence[str]', method: 'str', model_data: 'pd.DataFrame | None' = None) -> 'None'
```

**`XtNbreg.from_formula`**

```python
XtNbreg.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str | None' = None, method: 'str' = 'fe', subset: 'Any' = None) -> "'XtNbreg'"
```

**`XtNbreg.fit`**

```python
XtNbreg.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, intpoints: 'int' = 12, maxiter: 'int' = 500, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.XtNbreg.from_formula('ycnt ~ x1', base, entity='firm', method='fe').fit().summary(style='stata')
```

**输出**

```text

Conditional FE negative binomial regression          Number of obs    =    200
Group variable: firm                                 Number of groups =     40

                                                     Obs per group:
                                                                  min =      5
                                                                  avg =    5.0
                                                                  max =      5

                                                     Wald chi2(1)     =  25.31
Log likelihood = -203.70227                          Prob > chi2      = 0.0000

------------------------------------------------------------------------------
        ycnt | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .3128918   .0621933     5.03   0.000     .1909953    .4347884
       _cons |   23.17064   38374.96     0.00   1.000    -75190.37    75236.71
```

**注意:** 目标函数是准似然，因此 `e(ll)` 并非数据的似然，也不存 AIC/BIC。与 Stata 一样，`_cons` 属于参数向量，但目标函数在该方向上近乎平坦——给每个 `lam_it` 加常数几乎不改变核——所以它的估计值（及其巨大的标准误）依赖优化器，而斜率不然；去掉常数项会改变斜率，因此予以保留。

### `XtTobit`

**Stata 指令:** `xttobit ycensp x1, re`

随机效应 Tobit，即 Stata 的 `xttobit`，仅有随机效应。潜在结果为 `y*_it = x_it'b + u_i + e_it`，其中 `u_i ~ N(0, sigma_u^2)`、`e_it ~ N(0, sigma_e^2)`，并在 `lower` 和/或 `upper` 处删失。两个方差成分分别报告为 `sigma_u` 与 `sigma_e`，`e(rho) = sigma_u^2/(sigma_u^2 + sigma_e^2)`；积分使用 Stata 的自适应 Gauss-Hermite（`e(intmethod)='mvaghermite'`），`e(intpoints)` 默认为 12。

**签名**

```python
ws.XtTobit(*args: 'Any', lower: 'float | None' = 0.0, upper: 'float | None' = None, **kwargs: 'Any') -> 'None'
```

**`XtTobit.from_formula`**

```python
XtTobit.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str | None' = None, lower: 'float | None' = 0.0, upper: 'float | None' = None, method: 'str' = 're', subset: 'Any' = None) -> "'XtTobit'"
```

**`XtTobit.fit`**

```python
XtTobit.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, intpoints: 'int' = 12, maxiter: 'int' = 500, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
d = base.assign(ycensp=np.clip(base.ypanel, 0.0, None))
ws.XtTobit.from_formula('ycensp ~ x1', d, entity='firm', method='re').fit().summary(style='stata')
```

**输出**

```text

Random-effects tobit regression                      Number of obs    =    200
Group variable: .                                    Number of groups =     40

Random effects u_i ~ Gaussian                        Obs per group:
                                                                  min =      5
                                                                  avg =    5.0
                                                                  max =      5

Integration method: mvaghermite                      Integration pts. =     12

                                                     Wald chi2(1)     =  80.89
Log likelihood = -271.51546                          Prob > chi2      = 0.0000

------------------------------------------------------------------------------
             | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .7848094   .0872602     8.99   0.000     .6137825    .9558363
     sigma_u |   .8057239   .1338431     6.02   0.000     .5433962    1.068052
     sigma_e |   1.066372    .073274    14.55   0.000     .9227576    1.209987
       _cons |     .76606   .1536457     4.99   0.000       .46492      1.0672
-------------+----------------------------------------------------------------
-------------+----------------------------------------------------------------
         rho |   .3634196                                        0           .
------------------------------------------------------------------------------
LR test of rho=0: chibar2(01) = 89.96                  Prob >= chibar2 = 0.000
```

**注意:** `method='fe'` 会被拒绝（`supported: 're'`）。求积点数的收益是有上限的——12 是 Stata 的默认值，也是本模块的默认值，在性态良好的面板上增加点数并不改变估计。本类同样不填 `e(ivar)`，因此表头的 `Group variable:` 显示 `.`，而 `e(N_g)` 仍统计组数。`xttobit` 未注册任何 `predict` 类型。


## 时间序列、多元与面板单位根

### `ARIMA`

**Stata 指令:** `arima y, arima(1,1,0)`

以极大似然估计 ARIMA 与 SARIMA，底层封装 statsmodels 的状态空间 ARIMA。`fit()` 返回 `StataResults` 且 `dist="z"`，因此表头打印 Wald 卡方，与 Stata `arima` 一致，而不是 F 检验。`trend` 默认为 `"n"`（无常数项），因为 Stata 的默认也是如此；规格中含截距时请显式指定 `trend="c"`。`e(ar_max)` 与 `e(ma_max)` 是含季节阶数相乘后的有效滞后阶，与 Stata 的报告方式一致。

**签名**

```python
ws.ARIMA(endog: 'Any', order: 'tuple[int, int, int]' = (1, 0, 0), seasonal_order: 'tuple[int, int, int, int]' = (0, 0, 0, 0), *, trend: 'str' = 'n') -> 'None'
```

**`ARIMA.fit`**

```python
ARIMA.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, maxiter: 'int' = 500, strict: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.ARIMA(pd.Series(np.cumsum(np.random.default_rng(1).normal(size=120))), order=(1, 1, 0)).fit().summary(style='stata')
```

**输出**

```text
ARIMA regression

Sample: 2 thru 120                              Number of obs     =        119
                                                Wald chi2(1)      =       0.53
Log likelihood = -149.227                       Prob > chi2       =          .

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
-------------+----------------------------------------------------------------
ARMA         |
          ar |
         L1. |  -.0579121   .0795261    -0.73   0.466    -.2137804    .0979562
-------------+----------------------------------------------------------------
      /sigma |   .8479354   .0501061    16.92   0.000     .7497292    .9461416
------------------------------------------------------------------------------
```

**注意:** 对含差分的模型，`e(N)` 是差分之后估计样本的大小（对 120 点的 I(1) 序列这里是 119），而打印的 “Number of obs” 是序列全长——两者本就不同，Stata 自己的 `estat ic` 用的是较小的那个。此外 `fit()` 只接受 `cov_type="nonrobust"`，其他取值会直接报错。

### `GARCH`

**Stata 指令:** `arch y, arch(1) garch(1)`

GARCH 与 EGARCH 波动率模型，对应 Stata 的 `arch`。参数为 `alpha[i]`（ARCH）、`beta[i]`（GARCH）、`gamma[i]`（非对称项，`o=1` 时为 GJR-GARCH），另有 `mu` 与 `omega`；`vol="EGARCH"` 切换波动率方程，`dist="t"` 切换新息分布。有两个默认值刻意取 Stata 的约定而非底层 `arch` 包的约定：`cov_type="opg"`（BHHH 标准误，即 Stata 的 `e(vcetype) = "OPG"`）与 `presample="stata"`，后者把方差递推的起点取为去均值残差平方的均值，而不是指数加权的回推值。

**签名**

```python
ws.GARCH(endog: 'Any', *, p: 'int' = 1, q: 'int' = 1, o: 'int' = 0, vol: 'str' = 'GARCH', dist: 'str' = 'normal', mean: 'str' = 'Constant') -> 'None'
```

**`GARCH.fit`**

```python
GARCH.fit(self, *, strict: 'bool' = False, maxiter: 'int' = 1000, cov_type: 'str' = 'opg', presample: 'str' = 'stata', max_presample_iter: 'int' = 5) -> 'StataResults'
```

**示例**

```python
ws.GARCH(pd.Series(np.random.default_rng(1).normal(size=200) * 0.01), p=1, q=1).fit().summary(style='stata')
```

**输出**

```text
ARCH family regression

Sample: 1 thru 200                              Number of obs     =        200
                                                Wald chi2(.)      =          .
Log likelihood = 654.1268                       Prob > chi2       =          .

------------------------------------------------------------------------------
             |                 OPG
           y | Coefficient  std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
y            |
       _cons |  -.0007742   .0006318    -1.23   0.220    -.0020125    .0004641
-------------+----------------------------------------------------------------
ARCH         |
        arch |
         L1. |   .1118666   .0934251     1.20   0.231    -.0712433    .2949765
       garch |
         L1. |   .5372114   .3791344     1.42   0.156    -.2058783    1.280301
       _cons |   .0000304   .0000294     1.04   0.301    -.0000272    .0000879
------------------------------------------------------------------------------
```

**注意:** 这两个覆盖并不能让估计值与 Stata 完全相同：在参考 GARCH(1,1) 序列上，参数约 1.8e-4、标准误约 1.5e-4 的相对差距是有记录可查的；源码模块把原因追溯到 Stata 对第一个观测的处理与它的停机点，而不是公式差异。`e(archi)` 记录了实际使用的初始方差。

### `VAR`

**Stata 指令:** `var a b, lags(1)`

多序列向量自回归，对应 Stata 的 `var`。endog 参数是 `(n_obs, n_variables)` 数组，`names` 为方程命名。`fit()` 返回 `VARResults` 而非 `StataResults`：默认报告极大似然的残差协方差（每个方程的残差平方和除以 `T`，也就是 Stata 存入 `e(Sigma)` 与 `e(V)` 的量），`ml=False` 则复现 Stata 的 `var ..., dfk`。`select_order()` 返回 Stata `varsoc` 打印的 AIC/BIC/HQIC/FPE 表，而不是悄悄替你挑一个滞后阶。

**签名**

```python
ws.VAR(endog: 'Any', lags: 'int | None' = 1, *, trend: 'str' = 'c', names: 'Sequence[str] | None' = None) -> 'None'
```

**`VAR.fit`**

```python
VAR.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'Any' = None, *, ml: 'bool' = True) -> "'VARResults'"
```

**示例**

```python
d = np.column_stack([np.cumsum(np.random.default_rng(1).normal(size=100)), np.cumsum(np.random.default_rng(2).normal(size=100))])
ws.VAR(d, lags=1, names=['a', 'b']).fit().estat_ic()
```

**输出**

```text

Akaike's information criterion and Bayesian information criterion

-----------------------------------------------------------------------------
       Model |          N   ll(null)  ll(model)      df        AIC        BIC
-------------+---------------------------------------------------------------
           . |         99          .  -255.3751       6   522.7501   538.3208
-----------------------------------------------------------------------------
Note: BIC uses N = number of observations. See [R] IC note.
```

**注意:** `VARResults` 没有 `.summary()`，调用 `summary(style="stata")` 会抛 `AttributeError`；请改用 `estat_ic()`、`estat_summarize()`、`stata_params`/`stata_param_names`、`granger_causality()` 或 `irf()`。另外 `granger_causality()` 是在**拟合结果自身**的名字里解析 caused/causing，而不是在传给 `VAR` 的 `names=` 里——用 `names=['a','b']` 建的模型仍然只认 `y1`/`y2`，写成 `'a'` 会抛 `ValueError`。

### `VECM`

**Stata 指令:** `vec a b, rank(1) lags(1)`

向量误差修正模型，对应 Stata 的 `vec`。`coint_rank` 是协整秩，通常先由 `johansen()` 读出；该方法对应 `vecrank`，返回迹统计量与最大特征值统计量，并附上 Stata 打印的临界值。系数入口为 `beta`（协整向量）、`alpha`（调整速度）与 `gamma`（短期项），`e()` 载荷携带 Stata 的命名，如 `k_ce`、`k_ar_diff`、`r2_1`、`r2_2`。

**签名**

```python
ws.VECM(endog: 'Any', *, coint_rank: 'int' = 1, lags: 'int' = 1, deterministic: 'str' = 'co', names: 'Sequence[str] | None' = None) -> 'None'
```

**`VECM.fit`**

```python
VECM.fit(self, cov_type: 'str | None' = None, **_: 'Any') -> 'Any'
```

**示例**

```python
d = np.column_stack([np.cumsum(np.random.default_rng(1).normal(size=100)), np.cumsum(np.random.default_rng(2).normal(size=100))])
ws.VECM(d, coint_rank=1, lags=1, names=['a', 'b']).fit().beta
```

**输出**

```text
[[1.      ]
 [3.174865]]
```

**注意:** `johansen()` 中的临界值是 Stata 使用的 Johansen (1995) 表，而不是 statsmodels 的 Osterwald-Lenum 数值，这个差异会改变临界数据上选出的秩（5% 处 3.76 对 3.8415）。90% 那一列是 `nan`，因为 Stata 的 ado 只硬编码了 95% 与 99%。`VECMResults` 同样没有 `.summary()`——请直接读 `beta`/`alpha`/`gamma` 与 `e()`。

### `XtUnitroot`

**Stata 指令:** `xtunitroot ips ypanel   /   xtunitroot llc ypanel`

面板单位根检验，提供 Im-Pesaran-Shin（`test="ips"`）与 Levin-Lin-Chu（`test="llc"`）两种方法，对应 Stata 的 `xtunitroot`。`from_formula` 通过 `entity=` 与 `time=` 接收面板标识；`fit()` 返回 `PanelUnitRootResult`，携带 Stata 的 `r()` 标量，因为 `xtunitroot` 是 r 类命令，不存 `e(b)` 或 `e(V)`。两种方法都要求强平衡且无缺口的面板，本实现对非平衡面板直接拒绝，而不是悄悄采用恰好对齐的那部分行。

**签名**

```python
ws.XtUnitroot(values: 'Any', codes: 'np.ndarray', sizes: 'np.ndarray', *, test: 'str' = 'ips', lags: 'int | None' = None, trend: 'bool' = False, noconstant: 'bool' = False, demean: 'bool' = False, variable_name: 'str' = 'y') -> 'None'
```

**`XtUnitroot.from_formula`**

```python
XtUnitroot.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str | None' = None, test: 'str' = 'ips', lags: 'int | None' = None, trend: 'bool' = False, noconstant: 'bool' = False, demean: 'bool' = False, subset: 'Any' = None) -> "'XtUnitroot'"
```

**`XtUnitroot.fit`**

```python
XtUnitroot.fit(self, **_ignored: 'Any') -> 'PanelUnitRootResult'
```

**示例**

```python
ws.XtUnitroot.from_formula('ypanel ~ 1', base, entity='firm', time='year', test='ips').fit().r
```

**输出**

```text
{'test': 'ips', 't-bar': -2.443828941269152, 't-tilde-bar': -1.3462848052625447, 'Z-t-bar': -4.423070314618218, 'N_t': 5.0, 'lags': 0.0, 'Zt': -1.9851960161510092, 'p': 0.02356133369222999}
```

**注意:** `trend=True`、`noconstant=True` 与 `demean=True` 可以传入，但会抛 `UnsupportedError`，因为每一项都会改变查表值而这些表未被转录。短面板上 `lags=1` 还可能抛 `ConsistencyError`（“a panel has no residual degrees of freedom in its ADF regression”）——无滞后这一默认值才是复现 Stata 默认的取法。

### `IpsResult`

**Stata 指令:** `xtunitroot ips ypanel   (the r() results it returns)`

`im_pesaran_shin` 与 `XtUnitroot(test="ips").fit()` 返回的对象：t-bar、t-tilde-bar 与 Z-t-tilde-bar 统计量，附在 Z 统计量上的 p 值，面板维度以及插值所用的矩。`str()` 打印 Stata `xtunitroot ips` 打印的那段内容，`r` 属性把字段映射到 Stata 的 `r()` 标量（`r(N)`、`r(N_g)`、`r(N_t)`、`r(tbar)`、`r(ttildebar)`、`r(Zt)`、`r(p)`）。

**签名**

```python
ws.IpsResult(tbar: 'float', ttildebar: 'float', zt: 'float', zttildebar: 'float', pvalue: 'float', critical: 'dict[str, float]', n_groups: 'int', n_periods: 'int', observations: 'int', mean_t: 'float', var_t: 'float', mean_ttilde: 'float', var_ttilde: 'float') -> None
```

**示例**

```python
ws.im_pesaran_shin(base, variable='ypanel', entity='firm', time='year')
```

**输出**

```text
Im-Pesaran-Shin unit-root test
  t-bar     -2.4438
  t-tilde-bar -1.3463
  Z-t-tilde-bar -1.9852  p = 0.0236
  N = 40 panels x 5 periods
```

**注意:** `critical` 实际是空字典：固定 N 的精确临界值（IPS 表 2）尚未实现，代码宁可抛 `UnsupportedError` 也不从别的表里填一个数，因此 `r()` 里没有 `cv_1`/`cv_5`/`cv_10`。只有 t-bar 没有 p 值，这一点与 Stata 相同。

### `LlcResult`

**Stata 指令:** `xtunitroot llc ypanel   (the r() results it returns)`

`levin_lin_chu` 与 `XtUnitroot(test="llc").fit()` 返回的对象：未调整 t（`td`）、偏差调整后的 t*（`tds`）、长期与短期标准差之比 `sbar`、调整矩 `mu_adj`/`sig_adj`、Bartlett 核带宽 `hac_lags` 以及 ADF 滞后阶 `adf_lags`。`str()` 打印 Stata `xtunitroot llc` 打印的三行内容，`pvalue_tds` 是调整后统计量的左尾正态 p 值。

**签名**

```python
ws.LlcResult(tds: 'float', td: 'float', ttilde: 'float', sbar: 'float', nu_hat: 'float', nu_se: 'float', var_ep: 'float', mu_adj: 'float', sig_adj: 'float', pvalue_tds: 'float', pvalue_td: 'float | None', n_groups: 'int', n_periods: 'int', observations: 'int', adf_lags: 'int', hac_lags: 'int') -> None
```

**示例**

```python
ws.levin_lin_chu(base, variable='ypanel', entity='firm', time='year')
```

**输出**

```text
Levin-Lin-Chu unit-root test
  Unadjusted t   -15.8380
  Adjusted t*    -17.2339   p = 0.0000
  40 panels x 5 periods
```

**注意:** 与 `IpsResult` 和 `UnitRootResult` 不同，`LlcResult` 没有 `r` 属性，因此 Stata 的 `r()` 名字需要由字段手工拼出。`pvalue_td` 在未选择无常数模型时为 `None`，因为 Stata 只为那种情形列出 p 值。

### `UnitRootResult`

**Stata 指令:** `dfuller y   /   pperron y   /   kpss y`

`ws.unitroot` 为单变量序列返回的对象：`test`（描述性名称）、`statistic`、`pvalue`、实际使用的滞后阶或 Newey-West 截断、观测数、临界值区间以及用文字表述的原假设。`str()` 打印 Stata `dfuller`/`pperron`/`kpss` 打印的那段内容，`r` 属性按 Stata 的名字给出 `r()` 标量——`r(Zt)`、`r(N)`、`r(lags)`、`r(p)`（`pperron` 为 `r(pval)`，`kpss` 为 `r(stat)`）以及 `r(cv_1)`/`r(cv_5)`/`r(cv_10)`。

**签名**

```python
ws.UnitRootResult(test: 'str', statistic: 'float', pvalue: 'float', lags: 'int' = 0, observations: 'int' = 0, critical: 'Mapping[str, float]' = <factory>, null: 'str' = '', regression: 'str' = 'c') -> None
```

**示例**

```python
ws.unitroot(np.cumsum(np.random.default_rng(1).normal(size=120)), test='adf', regression='c')
```

**输出**

```text

Dickey–Fuller test for unit root          Number of obs  = 119
Variable: y                               Number of lags =   0

H0: Random walk without drift, d = 0

                                       Dickey–Fuller
                   Test      -------- critical value ---------
              statistic           1%           5%          10%
--------------------------------------------------------------
 Z(t)            -1.496       -3.504       -2.889       -2.579
--------------------------------------------------------------
MacKinnon approximate p-value for Z(t) = 0.5356.
```

**注意:** 打印的临界值并非 MacKinnon 响应面数值：`dfuller.ado` 对一张六点表按 N 线性插值，这里给的就是那张表的数，因此区间与 Stata 打印的一致。p 值仍然是 MacKinnon 的。`kpss` 的原假设是反的——其 `H0` 为平稳，Stata 也不为它存 MacKinnon p 值。

### `Manova`

**Stata 指令:** `manova y1 y2 = treat`

单因素多元方差分析，对应 Stata 的 `manova`。注意公式写法：两侧用 `=` 而不是 `~` 分隔，因此 `from_formula("y1 y2 = g", data)` 读起来与 Stata 命令完全一致。每个响应变量都对同一个因子回归，组间与组内平方和矩阵只构造一次；`fit()` 返回 `ManovaResult`，携带 Wilks' lambda 与 Rao F。Stata 通过 `manovatest` 打印的其余多元统计量也会计算，并可通过 `multivariate_tests()` 取出。

**签名**

```python
ws.Manova(responses: 'np.ndarray', groups: 'Any', *, term: 'str' = 'group') -> 'None'
```

**`Manova.from_formula`**

```python
Manova.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None) -> "'Manova'"
```

**`Manova.fit`**

```python
Manova.fit(self) -> 'ManovaResult'
```

**示例**

```python
ws.Manova.from_formula('y1 y2 = treat', base).fit()
```

**输出**

```text
MANOVA treat
  Wilks' lambda = 0.9925
  F(2, 197) = 0.75   p = 0.4740
```

**注意:** 只支持一个因子，不支持协变量：右端有多个项的公式会抛 `UnsupportedError`，而不是去拟合别的东西。非整数的数值型因子会被拒绝，诊断信息用的是 Stata 自己的说法，因为 Stata 的 `manova` 会把这种变量当作连续协变量，而本实现未实现协变量。

### `ManovaResult`

**Stata 指令:** `manova y1 y2 = treat   (followed by manovatest)`

单因素 MANOVA 的结果：`wilks_lambda`、Rao F 近似 `F` 及其分子分母自由度 `df1`/`df2` 与 p 值 `p`、因子名 `term`，以及逐方程的拟合行（`r2_by_equation`、`rmse_by_equation`、`f_by_equation`、`p_f_by_equation`），与 Stata 的逐方程表对应。`df` 是 Stata 为该因子给出的假设自由度（`g - 1`），与 `df1` 不是同一个量。`multivariate_tests()` 返回全部四个统计量——Wilks' lambda、Pillai 迹、Lawley-Hotelling 迹与 Roy 最大根——名称沿用 `manovatest` 的写法。

**签名**

```python
ws.ManovaResult(wilks_lambda: 'float', df: 'float', df1: 'float', df2: 'float', F: 'float', p: 'float', term: 'str' = 'group', nobs: 'int' = 0, groups: 'int' = 0, statistics: 'Mapping[str, float]' = <factory>, n_equations: 'int' = 0, r2_by_equation: "'Sequence[float]'" = <factory>, rmse_by_equation: "'Sequence[float]'" = <factory>, f_by_equation: "'Sequence[float]'" = <factory>, p_f_by_equation: "'Sequence[float]'" = <factory>, depvar_names: "'Sequence[str]'" = <factory>, design_matrix: 'Any' = None, coefficients: 'Any' = None, fitted_values: 'Any' = None, responses: 'Any' = None) -> None
```

**示例**

```python
ws.Manova.from_formula('y1 y2 = treat', base).fit().multivariate_tests()
```

**输出**

```text
{"Wilks' lambda": 0.9924503215154242, "Pillai's trace": 0.007549678484575742, 'Lawley-Hotelling trace': 0.007607109717137019, "Roy's largest root": 0.007549678484575742}
```

**注意:** `df_hypothesis`/`df_error`/`f_statistic`/`pvalue` 作为旧字段名的别名保留，因此 `df1` 与 `df_hypothesis` 不可互换。索取某次拟合未记录的统计量会抛 `KeyError` 而不是返回 `nan`，这样“不可得”就不会被误当成“算出来是零”。


## 处理效应与调查数据

### `Didregress`

**Stata 指令:** `didregress (y x1) (treat), group(firm) time(year)`

双向固定效应双重差分，对应 Stata 的 `didregress`。Stata 的两个括号块作为前两个参数传入：结果块（因变量，随后是协变量）与处理块（恰好一个 0/1 指示变量），`group=` 指定被吸收单位固定效应的单位，`time=` 指定时期。两个固定效应都是被吸收而非生成虚拟变量，报告的 ATET 即处理系数。默认协方差按 `group()` 聚类，与 Stata 相同。

**签名**

```python
ws.Didregress(outcome: 'Any', treatment: 'Any', data: 'pd.DataFrame', *, group: 'str', time: 'str', covariates: 'Any' = None) -> 'None'
```

**`Didregress.from_formula`**

```python
Didregress.from_formula(outcome: 'Any', treatment: 'Any', data: 'pd.DataFrame', *, group: 'str', time: 'str', covariates: 'Any' = None, subset: 'Any' = None) -> "'Didregress'"
```

**`Didregress.fit`**

```python
Didregress.fit(self, cov_type: 'str' = 'cluster') -> 'StataResults'
```

**示例**

```python
ws.Didregress.from_formula('y', 'treat', base, group='firm', time='year').fit().summary(style='stata')
```

**输出**

```text
Treatment and time information

Time variable: year
Control:       treat = 0
Treatment:     treat = 1
-----------------------------------
             |   Control  Treatment
-------------+---------------------
Group        |
        firm |         0         40
-------------+---------------------
Time         |
     Minimum |         .          1
     Maximum |         .          3
-----------------------------------

Difference-in-differences regression                       Number of obs = 200
Data type: Repeated cross-sectional

                                  (Std. err. adjusted for 40 clusters in firm)
------------------------------------------------------------------------------
             |               Robust
           y | Coefficient  std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
ATET         |
       treat |
   (1 vs 0)  |   .2838325   .2816428     1.01   0.320    -.2858438    .8535088
Note: ATET estimate adjusted for group effects and time effects.
```

**注意:** 单独的 `"y"` 表示**没有**协变量，与 Stata 的 `(y)` 块完全一致；要包含协变量请写 `"y x1"`，或用 `covariates="all"` 显式推断所有随时间变化的数值列。不随时间变化的处理指示变量就是单位固定效应本身，因此会被 `ConsistencyError` 拒绝（对应 Stata 的 r(198) 共线性诊断），而不是报告求解器剩下的值。`e(r2)`、`e(r2_a)`、`e(rmse)`、`e(F)`、`e(ll)` 是 westata 的增补，并非经 Stata 验证的输出。

### `Xtdidregress`

**Stata 指令:** `xtdidregress (y x1) (treat), group(firm) time(year)`

面板双重差分，对应 Stata 的 `xtdidregress`。估计量与 `Didregress` 是同一个双向固定效应回归，在平衡面板上 ATET 完全相同；差别只在自由度约定：面板单位效应嵌套在 `group()` 聚类变量之内，因此不计入 `(N-1)/(N-K)` 因子。于是聚类标准误是两者中较小的那个，这正是 Stata 自身输出里 `didregress` 与 `xtdidregress` 的差别。

**签名**

```python
ws.Xtdidregress(outcome: 'Any', treatment: 'Any', data: 'pd.DataFrame', *, group: 'str', time: 'str', covariates: 'Any' = None) -> 'None'
```

**`Xtdidregress.from_formula`**

```python
Xtdidregress.from_formula(outcome: 'Any', treatment: 'Any', data: 'pd.DataFrame', *, group: 'str', time: 'str', covariates: 'Any' = None, subset: 'Any' = None) -> "'Didregress'"
```

**`Xtdidregress.fit`**

```python
Xtdidregress.fit(self, cov_type: 'str' = 'cluster') -> 'StataResults'
```

**示例**

```python
ws.Xtdidregress.from_formula('y', 'treat', base, group='firm', time='year').fit().summary(style='stata')
```

**输出**

```text
Treatment and time information

Time variable: year
Control:       treat = 0
Treatment:     treat = 1
-----------------------------------
             |   Control  Treatment
-------------+---------------------
Group        |
        firm |         0         40
-------------+---------------------
Time         |
     Minimum |         .          1
     Maximum |         .          3
-----------------------------------

Difference-in-differences regression                       Number of obs = 200
Data type: Longitudinal

                                  (Std. err. adjusted for 40 clusters in firm)
------------------------------------------------------------------------------
             |               Robust
           y | Coefficient  std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
ATET         |
       treat |
   (1 vs 0)  |   .2838325   .2517466     1.13   0.266    -.2253731     .793038
Note: ATET estimate adjusted for panel effects and time effects.
```

**注意:** 与 `didregress` 不同，这种写法要求真正的面板：重复的 `(group, time)` 单元会抛 `ConsistencyError`，诊断沿用 Stata `xtset` 的说法（“repeated time values within panel”）。非平衡面板仍可估计；与 `Didregress` 完全一致这一点只在平衡面板上有保证并经过测试。

### `TreatmentEffectResult`

**Stata 指令:** `teffects ipw / aipw / nnmatch / ra   (the table they print)`

四个 `teffects` 估计量共用的结果类型：`estimand`（"ATE" 或 "ATET"）、效应值及其标准误、基于正态的 `z` 与双侧 `pvalue`、95% Wald 区间，以及 `potential_outcomes` 映射 `{"0": (mean, se), "1": (mean, se)}`，对应 Stata 的 `POmean` 行。`se_method` 记录标准误的产生方式，表头字段 `nobs`、`estimator`、`outcome_model`、`treatment_model` 复现 Stata 在表格上方打印的那一段。`e` 属性把表格映射成 `e()` 风格的标量，如 `atet`、`se_atet`、`ci_lower`。

**签名**

```python
ws.TreatmentEffectResult(estimand: 'str', estimate: 'float', std_err: 'float', z: 'float', pvalue: 'float', ci_lower: 'float', ci_upper: 'float', potential_outcomes: 'dict[str, tuple[float, float]]' = <factory>, se_method: 'str' = 'influence function', nobs: 'int' = 0, estimator: 'str' = '', outcome_model: 'str' = '', treatment_model: 'str' = '', weights: 'Any' = None) -> None
```

**示例**

```python
r = ws.IPW.from_formula('ytreat', 'treat', base, covariates=['x1', 'x2'], estimand='atet').fit()
r.e
```

**输出**

```text
{'N': 200.0, 'atet': 1.5423356362095726, 'se_atet': 0.1505521705826877, 'z_atet': 10.244526068539653, 'p_atet': 1.2523713423322037e-24, 'ci_lower': 1.2472588040731742, 'ci_upper': 1.837412468345971, 'estimator': 'inverse probability weighting', 'outcome_model': 'none', 'treatment_model': 'logit', 'se_method': 'influence function', 'po0': 0.8694097819079406, 'se_po0': 0.13823329458547928, 'po1': 2.4117454181175133, 'se_po1': 0.11180252552264044}
```

**注意:** `NearestNeighbourMatching` 的 `se_method` 是 `"Abadie-Imbens approximation"`，其余三个是 `"influence function"`，因此调用方不必猜某个标准误是否与 Stata 一致。潜在结果均值单独存储，而不是由效应反推，因为估计量可能在错误的水平上得到正确的差值。`with_header()` 需要四个表头关键字参数并返回副本，因为该 dataclass 是冻结的。

### `IPW`

**Stata 指令:** `teffects ipw (ytreat) (treat x1 x2) (x1 x2)`

逆概率加权，对应 Stata 的 `teffects ipw`。先用 logit 处理模型为每个观测给出倾向得分，报告的 ATE 是两个处理组**归一化**（Hajek）加权均值之差，而不是加权和的简单比值。ATET 则用几率 `p/(1-p)` 把对照组的权重重标定到处理组分布上。`estimand="atet"` 切换报告的是哪一个。

**签名**

```python
ws.IPW(y: 'Any', treated: 'Any', x: 'Any', *, estimand: 'str' = 'ate') -> 'None'
```

**`IPW.from_formula`**

```python
IPW.from_formula(outcome: 'str', treatment: 'str', data: 'pd.DataFrame', *, covariates: 'Sequence[str] | None' = None, estimand: 'str' = 'ate', subset: 'Any' = None) -> "'IPW'"
```

**`IPW.fit`**

```python
IPW.fit(self) -> 'TreatmentEffectResult'
```

**示例**

```python
ws.IPW.from_formula('ytreat', 'treat', base, covariates=['x1', 'x2']).fit()
```

**输出**

```text
Treatment-effects estimation
  Number of obs = 200
  Estimator      : inverse probability weighting
  Outcome model  : none
  Treatment model: logit
  ATE        1.559812   se 0.1452071   z   10.74   p 0.0000   [1.275212, 1.844413]
  POmean(0) 0.9280727   se 0.129868
  POmean(1)  2.487885   se 0.1024434
```

**注意:** 归一化不是为了好看：在参考数据上，不做归一化的加权和给出 1.9355，而 Stata 给 1.8882，因为估计出的倾向得分不会恰好加总到各组样本量。标准误是精确的影响函数方差，而不是从某个 OLS 协方差里读出来的数。

### `AIPW`

**Stata 指令:** `teffects aipw (ytreat x1 x2) (treat x1 x2) (x1 x2)`

增广逆概率加权，对应 Stata 的 `teffects aipw`。每个处理组的结果回归先给出一个预测值，再由倾向得分修正其残差，于是处理组插补出的潜在结果是 `yhat1 + t(y - yhat1)/p`，对照组对称。这使估计量具备双重稳健性：结果模型或处理模型之一设定错误，一致性依然成立。

**签名**

```python
ws.AIPW(y: 'Any', treated: 'Any', x: 'Any', *, estimand: 'str' = 'ate') -> 'None'
```

**`AIPW.from_formula`**

```python
AIPW.from_formula(outcome: 'str', treatment: 'str', data: 'pd.DataFrame', *, covariates: 'Sequence[str] | None' = None, estimand: 'str' = 'ate', subset: 'Any' = None) -> "'IPW'"
```

**`AIPW.fit`**

```python
AIPW.fit(self) -> 'TreatmentEffectResult'
```

**示例**

```python
ws.AIPW.from_formula('ytreat', 'treat', base, covariates=['x1', 'x2']).fit()
```

**输出**

```text
Treatment-effects estimation
  Number of obs = 200
  Estimator      : augmented inverse probability weighting
  Outcome model  : linear
  Treatment model: logit
  ATE        1.584398   se 0.1452802   z   10.91   p 0.0000   [1.299654, 1.869142]
  POmean(0) 0.8994778   se 0.1310723
  POmean(1)  2.483876   se 0.1025504
```

**注意:** 标准误包含结果回归与倾向得分模型两方面的贡献——增广后的潜在结果均值既有直接的非光滑部分，也有 `A^-1 m_i` 项；漏掉任何一块都会改变标准误，而点估计却纹丝不动。

### `NearestNeighbourMatching`

**Stata 指令:** `teffects nnmatch (ytreat x1 x2) (treat)`

基于协变量的最近邻匹配，对应 Stata 的 `teffects nnmatch`。每个观测的潜在结果由另一组中最近的 `nneighbor` 个观测插补，距离用协变量样本协方差下的马氏距离——这正是 `teffects nnmatch` 的默认做法。匹配是有放回的，因此一个对照观测可以服务多个处理观测。

**签名**

```python
ws.NearestNeighbourMatching(y: 'Any', treated: 'Any', x: 'Any', *, nneighbor: 'int' = 1, estimand: 'str' = 'ate') -> 'None'
```

**`NearestNeighbourMatching.from_formula`**

```python
NearestNeighbourMatching.from_formula(outcome: 'str', treatment: 'str', data: 'pd.DataFrame', *, covariates: 'Sequence[str] | None' = None, nneighbor: 'int' = 1, estimand: 'str' = 'ate', subset: 'Any' = None) -> "'NearestNeighbourMatching'"
```

**`NearestNeighbourMatching.fit`**

```python
NearestNeighbourMatching.fit(self) -> 'TreatmentEffectResult'
```

**示例**

```python
ws.NearestNeighbourMatching.from_formula('ytreat', 'treat', base, covariates=['x1', 'x2'], nneighbor=3).fit()
```

**输出**

```text
Treatment-effects estimation
  Number of obs = 200
  Estimator      : nearest-neighbor matching
  Outcome model  : none
  Treatment model: logit
  ATE         1.54307   se 0.1651077   z    9.35   p 0.0000   [1.219465, 1.866675]
  POmean(0) 0.9311758   se 0.09906461
  POmean(1)  2.474246   se 0.1320861
  note: standard errors are Abadie-Imbens approximation
```

**注意:** 这是该族中标准误为近似值的唯一一个估计量：`se_method` 显示为 `"Abadie-Imbens approximation"`，明确记录了与 Stata 约 2% 的差距，而不是把这个数当作一致结果呈现。距离度量不是细节——在参考数据上改用欧氏距离会让估计值偏移约 0.25%。

### `RegressionAdjustment`

**Stata 指令:** `teffects ra (ytreat x1 x2) (treat)`

回归调整，对应 Stata 的 `teffects ra`。结果模型在**每个处理组内分别**拟合，然后对每个观测预测两种潜在结果并取平均。允许斜率随组别不同，正是它与“带处理虚拟变量的单一回归”的区别所在；当协变量效应存在异质性时，这也决定了得到的是正确估计还是严重有偏的估计。

**签名**

```python
ws.RegressionAdjustment(y: 'Any', treated: 'Any', x: 'Any', *, estimand: 'str' = 'ate') -> 'None'
```

**`RegressionAdjustment.from_formula`**

```python
RegressionAdjustment.from_formula(outcome: 'str', treatment: 'str', data: 'pd.DataFrame', *, covariates: 'Sequence[str] | None' = None, estimand: 'str' = 'ate', subset: 'Any' = None) -> "'RegressionAdjustment'"
```

**`RegressionAdjustment.fit`**

```python
RegressionAdjustment.fit(self) -> 'TreatmentEffectResult'
```

**示例**

```python
ws.RegressionAdjustment.from_formula('ytreat', 'treat', base, covariates=['x1', 'x2']).fit()
```

**输出**

```text
Treatment-effects estimation
  Number of obs = 200
  Estimator      : regression adjustment
  Outcome model  : linear
  Treatment model: none
  ATE        1.583224   se 0.1452821   z   10.90   p 0.0000   [1.298476, 1.867971]
  POmean(0) 0.9002321   se 0.1309727
  POmean(1)  2.483456   se 0.1028893
```

**注意:** 标准误计入了插补预测本身的不确定性，而不只是拟合系数的不确定性：潜在结果均值中直接的非光滑部分属于影响函数的一部分，用 OLS 协方差会低估它。

### `SurveyDesign`

**Stata 指令:** `svyset firm [pweight=w], strata(strata) fpc(fpc)`

声明式复杂抽样设计——`svyset` 的对应物。它记录的是列**名**而非数据，因此同一个设计可以应用到任何带有这些列的帧上；`from_frame` 校验这些列存在，`layout(frame)` 把设计绑定到帧上并给出记账信息（`n_psu`、`n_strata`、`n_pop`、设计自由度 `df_r = #PSU - #strata` 以及逐观测权重）。`psu` 标签只在层内有意义，这一点与 Stata 的约定相同；`fpc` 大于 1 时读作总体 PSU 数，取值在 `(0, 1]` 时读作抽样比。

**签名**

```python
ws.SurveyDesign(psu: 'str', strata: 'str | None' = None, weights: 'str | None' = None, fpc: 'str | None' = None, singleunit: 'str' = 'certainty') -> None
```

**示例**

```python
ws.SurveyDesign.from_frame(base, psu='firm').layout(base).df_r
```

**输出**

```text
39.0
```

**注意:** `singleunit` 决定只含一个 PSU 的层如何处理：`"certainty"`（该 PSU 贡献为零，也是采集 fixture 时使用的设置）、`"missing"`、`"scaling"` 或 `"centered"`。只有 `certainty` 被采集到的 Stata 输出检验过，因此 `scaling` 与 `centered` 是有文档的近似，而非已验证的算术。未知取值在声明阶段就抛 `ValueError`，不会等到估计时才失败。

### `SvyMean`

**Stata 指令:** `svy: mean y`

抽样加权均值，对应 `svy: mean`。点估计是 `sum(w y) / sum(w)`；功夫全在方差上：方差由影响函数 `psi_i = w_i (y_i - ybar) / sum(w)` 线性化，再在层内按 PSU 汇总。权重和中心化都属于线性化的一部分，所报告的参考分布是自由度为 `df_r = #PSU - #strata` 的 t 分布。

**签名**

```python
ws.SvyMean(variables: 'Any', frame: 'pd.DataFrame', design: 'SurveyDesign', *, params: 'np.ndarray', layout: 'SurveyLayout', sample: 'pd.DataFrame') -> 'None'
```

**`SvyMean.fit`**

```python
SvyMean.fit(self) -> 'StataResults'
```

**示例**

```python
b = base.assign(w=np.random.default_rng(1).uniform(0.5, 2.0, len(base)))
ws.SvyMean.from_frame('y', b, ws.SurveyDesign(psu='firm', weights='w')).fit().summary(style='stata')
```

**输出**

```text
Survey: Mean estimation

Number of strata =   1            Number of obs   =        200
Number of PSUs   =  40            Population size = 252.433526
                                  Design df       =         39

--------------------------------------------------------------
             |             Linearized
             |       Mean   std. err.     [95% conf. interval]
-------------+------------------------------------------------
           y |    1.03929   .1025935      .8317748    1.246805
--------------------------------------------------------------
```

**注意:** 影响函数里漏掉任何一块，点估计都不变而方差会严重失真：在 fixture 设计上，`y_i / sum(w)` 低 65%，`w_i y_i / sum(w)`（有权重但不中心化）低 30%，而 `w_i (y_i - ybar) / sum(w)` 才复现 Stata。设计自由度 `df_r` 没有被强行截断到至少 1，因为那样会悄悄高估精度。

### `SvyTotal`

**Stata 指令:** `svy: total y`

抽样加权合计，对应 `svy: total`。估计为 `sum(w y)`，影响函数为 `psi_i = w_i y_i`；这里刻意**不**对权重归一化，因为合计估计的是一个总体量，它的量纲本身就是答案的一部分。其余部分——线性化、分层、`df_r = #PSU - #strata` 的参考分布与 fpc 因子——与 `SvyMean` 共用。

**签名**

```python
ws.SvyTotal(variables: 'Any', frame: 'pd.DataFrame', design: 'SurveyDesign', *, params: 'np.ndarray', layout: 'SurveyLayout', sample: 'pd.DataFrame') -> 'None'
```

**`SvyTotal.fit`**

```python
SvyTotal.fit(self) -> 'StataResults'
```

**示例**

```python
b = base.assign(w=np.random.default_rng(1).uniform(0.5, 2.0, len(base)))
ws.SvyTotal.from_frame('y', b, ws.SurveyDesign(psu='firm', weights='w')).fit().summary(style='stata')
```

**输出**

```text
Survey: Total estimation

Number of strata =   1            Number of obs   =        200
Number of PSUs   =  40            Population size = 252.433526
                                  Design df       =         39

--------------------------------------------------------------
             |             Linearized
             |      Total   std. err.     [95% conf. interval]
-------------+------------------------------------------------
           y |   262.3516   26.31999      209.1144    315.5888
--------------------------------------------------------------
```

**注意:** 与 `SvyMean` 不同，这里的影响函数**不**围绕估计值中心化。把它中心化成 `w_i (y_i - T)` 看似无害，实则不然：在 fixture 设计上会把方差放大约 244,000 倍，因为 `T` 是全局常数，它与不等权重之间的交互项在层内并不抵消。

### `SvyProportion`

**Stata 指令:** `svy: proportion yb`

一个或多个分类变量的加权比例，对应 `svy: proportion`。每个观测到的水平报告一个系数，按 Stata 的 `variable=level` 形式命名并按水平升序排列，同时给出联合协方差。二值变量的各水平会得到相同的标准误，且估计值完全负相关，因为每个比例就是其指示变量的均值。

**签名**

```python
ws.SvyProportion(variables: 'Any', frame: 'pd.DataFrame', design: 'SurveyDesign', *, params: 'np.ndarray', indicators: 'np.ndarray', layout: 'SurveyLayout', sample: 'pd.DataFrame', names: 'list[str]') -> 'None'
```

**`SvyProportion.fit`**

```python
SvyProportion.fit(self) -> 'StataResults'
```

**示例**

```python
b = base.assign(w=np.random.default_rng(1).uniform(0.5, 2.0, len(base)))
ws.SvyProportion.from_frame('yb', b, ws.SurveyDesign(psu='firm', weights='w')).fit().summary(style='stata')
```

**输出**

```text
Survey: Proportion estimation

Number of strata =   1            Number of obs   =        200
Number of PSUs   =  40            Population size = 252.433526
                                  Design df       =         39

--------------------------------------------------------------
             |             Linearized            Logit
             | Proportion   std. err.     [95% conf. interval]
-------------+------------------------------------------------
          yb |
          0  |   .4625698   .0374625      .3882178    .5386245
          1  |   .5374302   .0374625      .4613755    .6117822
--------------------------------------------------------------
```

**注意:** 系数名是形如 `yb=0`、`yb=1` 的字符串，因此按 `param_names` 定位是可靠的读取方式；这些水平是逐列删除缺失值后实际观测到的取值，而不是声明的值标签。

### `SvyRegress`

**Stata 指令:** `svy: regress y x1 x2`

抽样加权线性回归，对应 `svy: regress`。系数是加权最小二乘——与 `ws.OLS(..., weights=..., weight_type="pweight")` 得到的数值相同，差别只在顺序与命名（斜率在前，常数在最后记作 `_cons`）——因此抽样机制对点估计没有贡献，全部贡献都在方差上。影响函数为 `psi_i = (X'WX)^-1 w_i x_i e_i`，权重与逆交叉积都在线性化之内；模型的 F 统计量是按设计自由度报告的抽样调整 Wald 形式。

**签名**

```python
ws.SvyRegress(frame: 'pd.DataFrame', formula: 'str', design: 'SurveyDesign', *, X: 'np.ndarray', y: 'np.ndarray', names: 'list[str]', has_constant: 'bool', weights: 'np.ndarray', layout: 'SurveyLayout', sample: 'pd.DataFrame', params: 'np.ndarray', resid: 'np.ndarray', xtwx_inv_xt: 'np.ndarray') -> 'None'
```

**`SvyRegress.fit`**

```python
SvyRegress.fit(self) -> 'StataResults'
```

**示例**

```python
b = base.assign(w=np.random.default_rng(1).uniform(0.5, 2.0, len(base)))
ws.SvyRegress.from_frame(b, 'y ~ x1 + x2', ws.SurveyDesign(psu='firm', weights='w')).fit().summary(style='stata')
```

**输出**

```text
Survey: Linear regression

Number of strata =   1                            Number of obs   =        200
Number of PSUs   =  40                            Population size = 252.433526
                                                  Design df       =         39
                                                  F(2, 38)        =     101.48
                                                  Prob > F        =     0.0000
                                                  R-squared       =     0.4663

------------------------------------------------------------------------------
             |             Linearized
           y | Coefficient  std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8086296    .059397    13.61   0.000     .6884878    .9287714
          x2 |   -.513372   .0751049    -6.84   0.000     -.665286    -.361458
       _cons |   .9758079   .0831066    11.74   0.000     .8077089    1.143907
```

**注意:** 从影响函数里漏掉 `w_i` 或 `(X'WX)^-1`，系数仍然精确而方差会严重失准。基于设计的 `F` 是抽样调整形式 `(d - q + 1)/(d q) * W`，参考分布为 `F(q, d - q + 1)`，并非普通回归的 F；即使系数相同，它也不会等于未加权的 `regress` F。

### `SvyLogit`

**Stata 指令:** `svy: logit yb x1 x2`

抽样加权逻辑回归，对应 `svy: logit`。它属于伪极大似然：对数似然带上抽样权重，方差由 `psi_i = I^-1 w_i x_i (y_i - p_i)` 线性化，其中 `I = X' diag(w p(1-p)) X`。因变量必须编码为 0/1，常数结果会被拒绝而不是硬拟合。

**签名**

```python
ws.SvyLogit(frame: 'pd.DataFrame', formula: 'str', design: 'SurveyDesign', *, X: 'np.ndarray', y: 'np.ndarray', names: 'list[str]', has_constant: 'bool', weights: 'np.ndarray', layout: 'SurveyLayout', sample: 'pd.DataFrame', params: 'np.ndarray', probabilities: 'np.ndarray', information_inverse_xt: 'np.ndarray', loglik: 'float', iterations: 'int') -> 'None'
```

**`SvyLogit.fit`**

```python
SvyLogit.fit(self) -> 'StataResults'
```

**示例**

```python
b = base.assign(w=np.random.default_rng(1).uniform(0.5, 2.0, len(base)))
ws.SvyLogit.from_frame(b, 'yb ~ x1 + x2', ws.SurveyDesign(psu='firm', weights='w')).fit().summary(style='stata')
```

**输出**

```text
Survey: Logistic regression

Number of strata =   1                            Number of obs   =        200
Number of PSUs   =  40                            Population size = 252.433526
                                                  Design df       =         39
                                                  F(2, 38)        =       5.38
                                                  Prob > F        =     0.0087

------------------------------------------------------------------------------
             |             Linearized
          yb | Coefficient  std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .4975352   .1496844     3.32   0.002     .1947699    .8003005
          x2 |  -.0758175   .1557108    -0.49   0.629    -.3907723    .2391373
       _cons |   .1178733    .161271     0.73   0.469     -.208328    .4440747
```

**注意:** 在 fixture 设计上，估计值复现 Stata 到约 5e-10 的相对精度，而这个残差是 Stata 的停机容差、不是本代码的：把 Stata 自己发表的系数代回 Stata 自己的得分方程，残差梯度约 1.7e-8，而这里返回的是精确的极大化点。若把一致性容差收紧到 1e-9 以内，断言的就不是本实现而是 Stata 的不是了。


## 重抽样

### `Bootstrap`

**Stata 指令:** `bootstrap, reps(100): regress y x1 x2`

非参数自助法，对应 Stata 的 `bootstrap`。每个重复抽取一个重抽样样本并在其上重新拟合；`fit()` 返回 `StataResults`，标准误即各重复估计的标准差，同时记录 `e(reps)`、`e(seed)` 与 `e(vce) = "bootstrap"`。只支持 `regress` 与 `mean` 两种估计量——Stata 那种任意 `r()` 统计量需要一个小型表达式解释器，取不支持的名字会抛 `UnsupportedError`，而不是悄悄换成别的做法。

**签名**

```python
ws.Bootstrap(frame: 'pd.DataFrame', response: 'np.ndarray', design: 'np.ndarray', param_names: 'Sequence[str]', *, estimator: 'str') -> 'None'
```

**`Bootstrap.from_formula`**

```python
Bootstrap.from_formula(formula: 'str', data: 'pd.DataFrame', *, estimator: 'str' = 'regress', reps: 'int' = 200, seed: 'int | None' = None, subset: 'Any' = None, absorb: 'Any' = None) -> "'Bootstrap'"
```

**`Bootstrap.fit`**

```python
Bootstrap.fit(self) -> 'StataResults'
```

**示例**

```python
ws.Bootstrap.from_formula('y ~ x1 + x2', base, reps=100, seed=1).fit().bse
```

**输出**

```text
Intercept    0.066543
x1           0.076657
x2           0.068880
Name: bse, dtype: float64
```

**注意:** Stata 的随机数生成器不是 numpy 的，因此逐重复一致在构造上就不可能；可比的只有点估计与标准误的量级，`e(note)` 就在结果里写明了这一点。

### `Jackknife`

**Stata 指令:** `jackknife: regress y x1 x2`

逐一删除刀切法，对应 Stata 的 `jackknife`。N 个观测依次被剔除、估计量重拟合 N 次；默认使用伪值形式组合各重复，`fit(mse=True)` 则改用 Stata 更简单的 `(N-1)/N` 形式。`e(note)` 记录实际用了哪一种。Stata 的 `jackknife` 不论系数个数多少都报告 `e(df_r) = N - 1`，本实现同样如此。

**签名**

```python
ws.Jackknife(frame: 'pd.DataFrame', response: 'np.ndarray', design: 'np.ndarray', param_names: 'Sequence[str]', *, estimator: 'str') -> 'None'
```

**`Jackknife.from_formula`**

```python
Jackknife.from_formula(formula: 'str', data: 'pd.DataFrame', *, estimator: 'str' = 'regress', subset: 'Any' = None, absorb: 'Any' = None) -> "'Jackknife'"
```

**`Jackknife.fit`**

```python
Jackknife.fit(self, *, mse: 'bool' = False) -> 'StataResults'
```

**示例**

```python
ws.Jackknife.from_formula('y ~ x1 + x2', base).fit().bse
```

**输出**

```text
Intercept    0.074965
x1           0.071961
x2           0.072703
Name: bse, dtype: float64
```

**注意:** 刀切法不含随机数，因此被要求达到比自助法严格得多的标准——但回归方差与 Stata 仍有约 7e-6 的相对差距。这个差距是被诊断出来的而不是调出来的：缩放因子是对的，差异出在第六位之后的逐点剔除估计上，与 Stata 使用 Sherman-Morrison 更新而非重新拟合相符。`e(expected_gap)` 写明了这一点，免得有人再去追。

### `Permute`

**Stata 指令:** `permute x1, reps(200): regress y x1 x2`

置换检验，对应 Stata 的 `permute`。指定的回归元取值被随机打乱 `reps` 次，每次重新估计模型，p 值是置换统计量绝对值不低于观测值的比例——按 `(count + 1) / (reps + 1)` 计算，这也是它永远不可能恰好为零的原因。`e(permuted)` 给出被打乱的变量名，`e(pvalue_permuted)` 携带 p 值。

**签名**

```python
ws.Permute(*args: 'Any', permuted_column: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**`Permute.from_formula`**

```python
Permute.from_formula(formula: 'str', data: 'pd.DataFrame', *, permute: 'str', reps: 'int' = 200, seed: 'int | None' = None, subset: 'Any' = None) -> "'Permute'"
```

**`Permute.fit`**

```python
Permute.fit(self) -> 'StataResults'
```

**示例**

```python
ws.Permute.from_formula('y ~ x1 + x2', base, permute='x1', reps=200, seed=1).fit().e['pvalue_permuted']
```

**输出**

```text
0.004975124378109453
```

**注意:** 报告的协方差是解析的 OLS 协方差，而不是基于置换的估计：`permute` 产出的是 p 值，不是 VCE。`permute=` 必须写出公式中真实存在的回归元，否则 `ConsistencyError` 会列出实际可用的回归元。随机流不是 Stata 的，因此只有 p 值的量级可比。


## 后估计函数

### `testnl`

**Stata 指令:** `testnl exp(_b[x1]) = 2`

对一个或多个非线性约束做 Wald 检验（Stata `testnl`）。表达式按固定白名单解析——`_b[名称]`、七个函数 `exp`、`log`、`sqrt`、`sin`、`cos`、`logit`、`invlogit` 以及算术运算——从不交给 `eval`，因此 `__import__('os')` 这类文本会在校验阶段失败而不是被执行。裸表达式对零检验；以逗号或空白分隔的多个约束（包括 Stata 的成组写法 `(a = 0) (b = 0)`）联合检验。统计量为 chi2(q)，q 是约束个数，与 `[R] testnl` 一致；可调用为 `result.testnl(表达式)` 或 `ws.testnl(result, 表达式)`。

**签名**

```python
ws.testnl(result: 'Any', expression: 'str') -> 'WaldTest'
```

**示例**

```python
ws.testnl(ws.Logit.from_formula('yb ~ x1 + x2', base).fit(), 'exp(_b[x1]) = 2')
```

**输出**

```text

 ( 1)  exp(_b[x1]) = 2

       chi2(  1) =    1.36
            Prob > chi2 =    0.2437
```

**注意:** 进入方差的导数用中心差分，步长为 `1e-6 * max(1, |b_j|)`。因此线性约束与 `lincom` 完全一致，只有真正非线性的函数才是近似值。

### `lrtest`

**Stata 指令:** `lrtest unrestricted restricted`

对嵌套模型对做似然比检验（Stata `lrtest`）：`LR = 2(ll_非受限 - ll_受限)`，按参数个数之差服从 chi2 分布。两个拟合必须基于同一样本；参数个数相等时抛出 `ConsistencyError`——即 Stata 的 `r(498)`——因为 chi2(0) 是点质量分布，报 `p = 1` 会把一个 Stata 报错变成一次成功的命令。参数个数不同但并不嵌套的模型对仍会得到统计量，该值通常为负，此时照实报告并给出警告，而不是截断为零。可调用为 `ws.lrtest(非受限, 受限)` 或 `非受限.lrtest(受限)`。

**签名**

```python
ws.lrtest(unrestricted: 'Any', restricted: 'Any') -> 'LRTestResult'
```

**示例**

```python
full = ws.Logit.from_formula('yb ~ x1 + x2 + x3', base).fit()
full.lrtest(ws.Logit.from_formula('yb ~ x1 + x2', base).fit())
```

**输出**

```text
Likelihood-ratio test
Assumption: logit nested within logit
LR chi2(1) = 6.72
Prob > chi2 = 0.0095
```

**注意:** 是否嵌套仅由参数**个数**判定，与 Stata 的判定方式一致。系数名称不匹配只作为警告而非拒绝，因为 Stata 无论如何都会算出统计量。

### `hausman`

**Stata 指令:** `hausman fe re`

检验有效估计量一致性的 Hausman 设定检验（Stata `hausman`）。一致但非有效的拟合（如 `xtreg, fe`）在前，有效拟合（如 `xtreg, re`）在后，统计量为共同系数上的 `(b_c - b_e)'[V_c - V_e]^-1(b_c - b_e)`。除非 `constant=True`，常数项不参与比较，因为组内变换后的截距与 GLS 截距估计的是不同的总体量；`constant=True` 对应 Stata 的 `hausman ..., constant`。可调用为 `ws.hausman(一致, 有效)` 或 `一致.hausman(有效, constant=True)`。

**签名**

```python
ws.hausman(consistent: 'Any', efficient: 'Any', *, constant: 'bool' = False, sigmamore: 'bool' = False, sigmaless: 'bool' = False) -> 'HausmanResult'
```

**示例**

```python
fe = ws.PanelOLS.from_formula('ypanel ~ x1 + x2', base, entity='firm', time='year').fit()
re = ws.RandomEffects.from_formula('ypanel ~ x1 + x2', base, entity='firm', time='year').fit()
ws.hausman(fe, re)
```

**输出**

```text

                 ---- Coefficients ----
             |      (b)          (B)            (b-B)     sqrt(diag(V_b-V_B))
             |       fe           re         Difference       Std. err.
-------------+----------------------------------------------------------------
          x1 |    .7088762     .7277599       -.0188837         .016279
          x2 |   -.1092899     -.119745        .0104551        .0225716
------------------------------------------------------------------------------
                          b = Consistent under H0 and Ha; obtained from xtreg.
           B = Inconsistent under Ha, efficient under H0; obtained from xtreg.

Test of H0: Difference in coefficients not systematic

    chi2(2) = (b-B)'[(V_b-V_B)^(-1)](b-B)
            =   1.60
Prob > chi2 = 0.4501
```

**注意:** 当 `V_c - V_e` 非正定时，用伪逆求逆并把统计量取 0 作为下限，而 Stata 会打印它算出的负值。`df`、`p` 与警告仍然一致；不同的只是这个无法解释的统计量的符号，因为伪逆因舍入经常给出约 `-1e-18` 的结果。

### `contrast`

**Stata 指令:** `contrast r.treat`

对因子变量估计均值做方差分析式对比（Stata `contrast`）。`term` 是该因子在设计矩阵中的前缀（`'C(g)'` 或 `'g'`），按数据变量匹配，因此 `'C(g)'` 也能找到用 `C(g, base='a')` 拟合出的列。`method='r'` 把各水平与参照水平比较，`'g'` 与未加权总均值比较（L 行），`'a'` 比较相邻水平；其他取值抛出 `UnsupportedError`，不会静默退回某个默认值。返回每个对比一个 `LincomResult`，按水平顺序排列；可调用为 `ws.contrast(result, term, method)` 或 `result.contrast(term, method)`。

**签名**

```python
ws.contrast(result: 'Any', term: 'str', method: 'str' = 'r', *, level: 'float' = 0.95, nowald: 'bool' = False) -> 'list[LincomResult]'
```

**示例**

```python
ws.contrast(ws.OLS.from_formula('y ~ x1 + C(treat)', base).fit(), 'C(treat)')[0]
```

**输出**

```text

 ( 1)  C(treat)[T.1] - base(treat[0])

------------------------------------------------------------------------------
             | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   .3762187   .1705568     2.21   0.029     .0398673    .7125702
------------------------------------------------------------------------------
```

### `pwcompare`

**Stata 指令:** `pwcompare g`

因子变量各水平之间的全部两两差异（Stata `pwcompare`）。当 j > i 时第 (i, j) 行报告 `水平_j - 水平_i`，因此 L 个水平按此顺序给出 L*(L-1)/2 行；由于返回值是列表，示例取其中第一个。`adjust` 只接受 `None`——即 Stata 的 `adjust(none)`——因为所有真正的多重比较校正都未实现，传入其他值会抛出 `UnsupportedError`，而不是把校正悄悄丢掉。可调用为 `ws.pwcompare(result, term)` 或 `result.pwcompare(term)`。

**签名**

```python
ws.pwcompare(result: 'Any', term: 'str', *, level: 'float' = 0.95, adjust: 'str | None' = None) -> 'Any'
```

**示例**

```python
d = base.assign(g=pd.cut(base['x3'], 3, labels=['lo', 'mid', 'hi']))
ws.pwcompare(ws.OLS.from_formula('y ~ x1 + C(g)', d).fit(), 'C(g)')[0]
```

**输出**

```text

 ( 1)  C(g)[T.mid] - base(g[lo])

------------------------------------------------------------------------------
             | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   .0508092   .2187468     0.23   0.817    -.3805904    .4822089
------------------------------------------------------------------------------
```

### `nlcom`

**Stata 指令:** `nlcom exp(_b[x1])`

用 delta 方法给出系数非线性函数的点估计、标准误与置信区间（Stata `nlcom`）：`Var(g(b)) = G V G'`，G 为在估计值处取的雅可比矩阵。表达式语言为 `_b[名称]`、算术运算与白名单中的七个函数；返回值与 `lincom` 是同一个 `LincomResult` 容器，下游代码无需特判；`eform=True` 报告 `exp(g(b))` 并对区间取指数。可调用为 `ws.nlcom(result, 表达式)` 或 `result.nlcom(表达式)`。

**签名**

```python
ws.nlcom(result: 'Any', expression: 'str', *, level: 'float' = 0.95, eform: 'bool' = False) -> 'LincomResult'
```

**示例**

```python
ws.nlcom(ws.Logit.from_formula('yb ~ x1 + x2', base).fit(), 'exp(_b[x1])')
```

**输出**

```text

 ( 1)  exp(_b[x1])

------------------------------------------------------------------------------
             | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   1.697737   .2592614     6.55   0.000     1.189595     2.20588
------------------------------------------------------------------------------
```

**注意:** 与 Stata 的已知差异：推断沿用模型自身的分布——`regress` 之后用 t 与 `e(df_r)`，z 模型之后用正态——而 Stata 即使在 `regress` 之后也用 z 与正态区间打印 `nlcom`。点估计与标准误是一致的；Stata 的精确区间可由 `std_err` 重新算出。

### `margins`

**Stata 指令:** `margins, dydx(x1) at(x2=0)`

边际效应与预测边际。支持 Stata 的 `dydx`、`eyex`、`dyex`、`eydx` 四种导数形式，以及 `at()` 固定协变量、`over()` 分组。拟合离散选择模型后在响应尺度上求导，这与 Stata `margins` 一致——线性指数上的导数会是另一个量。

**签名**

```python
ws.margins(result: 'Any', *, dydx: 'Sequence[str] | str | None' = None, eyex: 'Sequence[str] | str | None' = None, dyex: 'Sequence[str] | str | None' = None, eydx: 'Sequence[str] | str | None' = None, at: 'Mapping[str, Any] | None' = None, atmeans: 'bool' = False, over: 'str | None' = None, exog: 'np.ndarray | None' = None, param_names: 'Sequence[str] | None' = None, level: 'float' = 0.95, predict_kind: 'str | None' = None, factor_vars: 'Sequence[str] | None' = None, outcome: 'int | None' = None) -> 'list[MarginsResult] | MarginsResult'
```

**示例**

```python
ws.margins(ws.Logit.from_formula('yb ~ x1 + x2', base).fit(), dydx='x1', at={'x2': 0.0})
```

**输出**

```text

Average marginal effects                                   Number of obs = 200
Model VCE: OIM

Expression: Pr(yb), predict()
dy/dx wrt:  x1

------------------------------------------------------------------------------
             |            Delta-method
             |      dy/dx   std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .1229425   .0311677     3.94   0.000      .061855      .18403
------------------------------------------------------------------------------
```

**注意:** `margins` 可用哪些预测类型由命令自身通过 `e(marginsok)`/`e(marginsnotok)` 声明；被命令禁止的类型会被拒绝，而不是静默计算。

### `estat_classification`

**Stata 指令:** `estat classification`

`logit`/`probit` 之后 Stata 打印的 2x2 分类表（Stata `estat classification`），把预测概率按 `cutoff` 阈值化（默认 0.5，即 Stata 的 `cutoff()`）。正类为结果非零，百分比采用 Stata 的 0-100 尺度。`to_dict()` 返回 `r()` 名称 `P_corr`、`P_p1`、`P_n0`、`P_p0`、`P_n1`、`P_1p`、`P_0n`、`P_0p`、`P_1n`。可调用为 `ws.estat_classification(result)` 或 `result.estat_classification()`。

**签名**

```python
ws.estat_classification(result: 'Any', *, cutoff: 'float' = 0.5, y: 'Any' = None, probabilities: 'Any' = None) -> 'ClassificationResult'
```

**示例**

```python
ws.estat_classification(ws.Logit.from_formula('yb ~ x1 + x2', base).fit())
```

**输出**

```text

Logistic model for yb

              -------- True --------
Classified |         D            ~D  |      Total
-----------+--------------------------+-----------
     +     |        76            44  |        120
     -     |        32            48  |         80
-----------+--------------------------+-----------
   Total   |       108            92  |        200

Classified + if predicted Pr(D) >= .5
True D defined as yb != 0
--------------------------------------------------
Sensitivity                     Pr( +| D)   70.37%
Specificity                     Pr( -|~D)   52.17%
Positive predictive value       Pr( D| +)   63.33%
Negative predictive value       Pr(~D| -)   60.00%
--------------------------------------------------
False + rate for true ~D        Pr( +|~D)   47.83%
False - rate for true D         Pr( -| D)   29.63%
False + rate for classified +   Pr(~D| +)   36.67%
False - rate for classified -   Pr( D| -)   40.00%
--------------------------------------------------
Correctly classified                        62.00%
--------------------------------------------------
```

**注意:** `cutoff` 超出 [0, 1] 或响应变量非二值时抛出 `ValueError`；对 Stata 不会转发该子命令的模型抛出 `NotImplementedError`，而不是为一个根本不存在的命令造出一张表。

### `estat_endogenous`

**Stata 指令:** `estat endogenous`

工具变量拟合之后的内生性检验（Stata `estat endogenous`）。返回哪一种检验取决于 VCE：未调整 VCE 给出 Durbin 与 Wu-Hausman 统计量，稳健 VCE 给出稳健得分检验，HAC VCE 给出 HAC 得分检验，聚类 VCE 给出基于回归的 F（`regF`），这也是 Stata 在那种情形下唯一报告的检验。`lags` 是 HAC 截断阶数，HAC VCE 下默认为 1，其余情形一律拒绝。LIML 与 GMM 拟合会以各自的理由被拒绝——Stata 只分派 2SLS 与 GMM，而 GMM 分支报告的是受约束重估得到的 `C` 统计量，本包未实现。可调用为 `ws.estat_endogenous(result)` 或 `result.estat_endogenous()`。

**签名**

```python
ws.estat_endogenous(result: 'Any', *, lags: 'int | None' = None) -> 'Any'
```

**示例**

```python
ws.estat_endogenous(ws.IV2SLS.from_formula('y ~ x1 + [x2 ~ x3]', base).fit())
```

**输出**

```text

  Tests of endogeneity
  H0: Variables are exogenous

  Durbin (score) chi2(1)          =  1.64391  (p = 0.1998)
  Wu-Hausman F(1,196)             =  1.62438  (p = 0.2040)
```

**注意:** 非 IV 结果、GMM 拟合，以及理由不同的 LIML 拟合，都会抛出 `NotImplementedError`；在非 HAC VCE 上给 `lags` 则抛出 `ValueError`（`lags() only valid with models with HAC VCEs`）。

### `estat_firststage`

**Stata 指令:** `estat firststage`

工具变量的一阶段诊断（Stata `estat firststage`）。对每个内生回归元，报告其对外生变量回归的 R² 与调整 R²、被排除工具变量的偏 R²，以及这些工具变量的联合 F。内生回归元多于一个时，报告会切换为 Shea 偏 R²，除非给出 `all=True`；`forcenonrobust=True` 使得在稳健 VCE 下仍报告最小特征值统计量及其临界值。可调用为 `ws.estat_firststage(result)` 或 `result.estat_firststage()`。

**签名**

```python
ws.estat_firststage(result: 'Any', *, all: 'bool' = False, forcenonrobust: 'bool' = False) -> 'FirstStageResult'
```

**示例**

```python
ws.estat_firststage(ws.IV2SLS.from_formula('y ~ x1 + [x2 ~ x3]', base).fit())
```

**输出**

```text

  First-stage regression summary statistics
  --------------------------------------------------------------------------
               |            Adjusted      Partial
      Variable |   R-sq.       R-sq.        R-sq.      F(1,197)   Prob > F
  -------------+------------------------------------------------------------
            x2 |  0.0008     -0.0093       0.0008      0.159849    0.6897
  --------------------------------------------------------------------------


  Minimum eigenvalue statistic = 0.159849    

  Critical Values                      # of endogenous regressors:    1
  H0: Instruments are weak             # of excluded instruments:     1
  ---------------------------------------------------------------------
                                     |    5%     10%     20%     30%
  2SLS relative bias                 |         (not available)
  -----------------------------------+---------------------------------
                                     |   10%     15%     20%     25%
  2SLS size of nominal 5% Wald test  |  16.38    8.96    6.66    5.53
  LIML size of nominal 5% Wald test  |  16.38    8.96    6.66    5.53
  ---------------------------------------------------------------------
```

**注意:** 未给出 `forcenonrobust` 时，稳健 VCE 下会抑制特征值统计量，因为其临界值假定误差独立同分布且服从正态。

### `estat_gof`

**Stata 指令:** `estat gof`

Poisson 拟合优度检验（Stata `estat gof`）：偏差（deviance）与 Pearson 卡方统计量，自由度均为 `N - df_m - 1`，按 Stata 的两块表格打印。它只对 `poisson` 存在——Stata 的 `nbreg` 不设 `e(estat_cmd)`，因此负二项拟合之后没有 `estat gof`，此处抛出 `NotImplementedError` 而不是凭空造一个。`to_dict()` 返回 `chi2_d`、`chi2_p`、`df`、`p_d`、`p_p`。可调用为 `ws.estat_gof(result)` 或 `result.estat_gof()`。

**签名**

```python
ws.estat_gof(result: 'Any', *, y: 'Any' = None, mu: 'Any' = None, group: 'int | None' = None) -> "'GofResult | LfitResult'"
```

**示例**

```python
ws.estat_gof(ws.Poisson.from_formula('ycnt ~ x1 + x2', base).fit())
```

**输出**

```text

Deviance goodness-of-fit           =   227.199
Prob > chi2(197)                   =    0.0690

Pearson goodness-of-fit            =   187.533
Prob > chi2(197)                   =    0.6740
```

**注意:** `y` 与 `mu` 默认取结果中保留的响应值与拟合均值；当结果未保留可用的残差时请显式传入。

### `estat_hettest`

**Stata 指令:** `estat hettest`

Breusch-Pagan / Cook-Weisberg 异方差检验（Stata `estat hettest`）。省略 `variables` 时，辅助回归是标准化残差平方对**拟合值**的回归——这正是 `hettest.ado:110-115` 在没有变量列表时的行为，它与 `rhs=True`（对 `e(b)` 全部回归元回归）**不是**同一个检验（`regress mpg weight foreign` 上分别为 chi2(1)=7.89 与 chi2(2)=24.17）。`variables` 优先按原始数据变量名解析（`'rep78'` 就是原始 1-5 那一列，自由度 1），否则按设计矩阵列名解析（`'foreign[T.Foreign]'`）。**默认**统计量是该辅助回归的 `e(mss)/2`，而不是 `N*R2`；`method='iid'` 给出 `N*R2`，`method='fstat'` 给出辅助回归的 F。估计权重会被辅助回归复用，因此 `fweight`/`iweight` 的 `e(N)` 是 `sum(w)`。稳健或聚类 VCE、pweight 数据以及 `noconstant` 拟合都会被拒绝，与 `hettest.ado` 一致。全部统计量已对照 StataNow 19.5 的实际输出验证。可调用为 `ws.estat_hettest(result)` 或 `result.estat_hettest()`。

**签名**

```python
ws.estat_hettest(result: 'Any', *, method: 'str' = 'normal', variables: 'Sequence[str] | None' = None, rhs: 'bool' = False, mtest: 'str | None' = None) -> 'Any'
```

**示例**

```python
ws.estat_hettest(ws.OLS.from_formula('y ~ x1 + x2', base).fit())
```

**输出**

```text

Breusch–Pagan/Cook–Weisberg test for heteroskedasticity 
Assumption: Normal error terms
Variable: Fitted values of y

H0: Constant variance

    chi2(1) =   0.27
Prob > chi2 = 0.6006
```

**注意:** `normal` 与 `iid`/`fstat` 组合会被拒绝，与 Stata 的 `opts_exclusive` 一致。`mtest()` 要求辅助回归有多个回归元：`hettest.ado:126-128` 在只有一个回归元时会静默丢弃它，因此无变量列表的调用只返回联合检验。

### `estat_ic`

**Stata 指令:** `estat ic`

单个已拟合模型的信息准则。依据 `estimates.ado:985-1002`，由对数似然与参数个数计算 AIC 与 BIC 并打印。与 `e(aic)` 宏不同——若干命令根本不存该宏，而 `regress` 存的还在另一个尺度上——本函数才是与 Stata `estat ic` 输出一致的入口。

**签名**

```python
ws.estat_ic(result: 'Any', *, n: 'float | None' = None, df: 'float | None' = None) -> 'IcResult'
```

**示例**

```python
ws.estat_ic(ws.OLS.from_formula('y ~ x1 + x2', base).fit())
```

**输出**

```text

Akaike's information criterion and Bayesian information criterion

-----------------------------------------------------------------------------
       Model |          N   ll(null)  ll(model)      df        AIC        BIC
-------------+---------------------------------------------------------------
           . |        200  -357.2327  -291.7997       3   589.5994   599.4944
-----------------------------------------------------------------------------
Note: BIC uses N = number of observations. See [R] IC note.
```

**注意:** `regress` 不存 `e(aic)`；本函数按需计算，因此在那里同样可用。

### `estat_imtest`

**Stata 指令:** `estat imtest, white`

`regress` 之后 Cameron & Trivedi 的信息矩阵检验（Stata `estat imtest`）：把得分检验分解为异方差、偏度、峰度三个分量以及总和。`white=True` 另外计算 White 的原始检验，即单独报告的异方差分量。权重与 `cluster()` 会被拒绝，因为 `imtest.ado` 也拒绝它们——否则统计量会由未加权的残差几何算出，含义已经不同——非 `regress` 结果抛出 `NotImplementedError`。可调用为 `ws.estat_imtest(result)` 或 `result.estat_imtest()`。

**签名**

```python
ws.estat_imtest(result: 'Any', *, white: 'bool' = False, design: 'Any' = None) -> 'ImTestResult'
```

**示例**

```python
ws.estat_imtest(ws.OLS.from_formula('y ~ x1 + x2', base).fit(), white=True)
```

**输出**

```text

White's test
H0: Homoskedasticity
Ha: Unrestricted heteroskedasticity

    chi2(5) =   0.85
Prob > chi2 = 0.9737

Cameron & Trivedi's decomposition of IM-test

--------------------------------------------------
              Source |       chi2     df         p
---------------------+----------------------------
  Heteroskedasticity |       0.85      5    0.9737
            Skewness |       1.19      2    0.5526
            Kurtosis |       2.04      1    0.1535
---------------------+----------------------------
               Total |       4.07      8    0.8503
--------------------------------------------------
```

### `estat_overid`

**Stata 指令:** `estat overid`

工具变量拟合之后的过度识别约束检验（Stata `estat overid`）。对 2SLS 与 LIML 报告 Sargan 及其有限样本 Basmann 变体，`N*(1 - rss_aux/rss_main)`，其中辅助回归是 IV 残差对**全部**工具变量回归——包含外生变量与被排除工具变量——若改用辅助回归的残差平方和作分母，得到的统计量没有分布。对 GMM 返回 `e(J)` 中的 Hansen J，因为 Sargan 假定同方差而 J 不假定。恰好识别模型、聚类 VCE，以及在非 HAC 拟合上给 `lags` 都会被拒绝。可调用为 `ws.estat_overid(result)` 或 `result.estat_overid()`。

**签名**

```python
ws.estat_overid(result: 'Any', *, lags: 'int | None' = None, forcenonrobust: 'bool' = False) -> 'OverIdResult'
```

**示例**

```python
ws.estat_overid(ws.IV2SLS.from_formula('y ~ x1 + [x2 ~ x3 + lo]', base).fit())
```

**输出**

```text

  Tests of overidentifying restrictions:

  Sargan (score) chi2(1) =  .141587  (p = 0.7067)
  Basmann chi2(1)        =  .138854  (p = 0.7094)
```

**注意:** `lags` 是 HAC 预白化截断阶数，HAC VCE 下默认为 1，否则为 0；Stata 拒绝负值，本实现同样拒绝。

### `estat_ovtest`

**Stata 指令:** `estat ovtest`

Ramsey 的 RESET 遗漏变量检验（Stata `estat ovtest`）。默认的辅助回归加入**重标度到 [0, 1]** 的拟合值的二、三、四次幂；重标度并非装饰——量级上百的拟合值的四次幂会让辅助设计矩阵病态，去掉它会改变统计量。检验是这三个新增项的联合 F。`rhs=True` 走 Stata 的另一条路径，改用右端变量的幂，其中示性变量只贡献自身水平。可调用为 `ws.estat_ovtest(result)` 或 `result.estat_ovtest()`。

**签名**

```python
ws.estat_ovtest(result: 'Any', *, rhs: 'bool' = False) -> 'OvTestResult'
```

**示例**

```python
ws.estat_ovtest(ws.OLS.from_formula('y ~ x1 + x2', base).fit(), rhs=True)
```

**输出**

```text

Ramsey RESET test for omitted variables
Omitted: Powers of independent variables

H0: Model has no omitted variables

F(6, 191) =   1.28
 Prob > F = 0.2694
```

### `estat_summarize`

**Stata 指令:** `estat summarize`

对估计样本逐变量给出描述统计（Stata `estat summarize`）：响应变量与所报回归元（或给定的 `variables`）的 Obs、Mean、Std. dev.、Min 与 Max。只有 `fweight` 会改变报告的 Obs，与 `estat_summ.ado` 一致；`aweight`、`pweight` 与 `iweight` 不会。数据默认取保留的 `model_data`，因此对未保留数据的估计量必须用 `data=` 传入；`cluster=` 会加上 Stata 的每簇观测数区块。可调用为 `ws.estat_summarize(result)` 或 `result.estat_summarize()`。

**签名**

```python
ws.estat_summarize(result: 'Any', *, data: 'pd.DataFrame | None' = None, variables: 'Sequence[str] | None' = None, weights: 'Any' = None, weight_type: 'str | None' = None, cluster: 'Any' = None, cluster_name: 'str' = '', noweights: 'bool' = False, noheader: 'bool' = False, labels: 'Mapping[str, str] | None' = None) -> 'SummarizeResult'
```

**示例**

```python
ws.estat_summarize(ws.OLS.from_formula('y ~ x1 + x2', base).fit(), variables=['y', 'x1'])
```

**输出**

```text

  Estimation sample regress                Number of obs =        200

  -------------------------------------------------------------------
      Variable |         Mean      Std. dev.         Min          Max
  -------------+-----------------------------------------------------
             y |     1.020667      1.447346    -2.752204     4.881985
            x1 |     .0573402      1.025868    -2.832134     3.278208
  -------------------------------------------------------------------
```

### `estat_vce`

**Stata 指令:** `estat vce, correlation`

Stata 的 `estat vce`：呈现系数协方差矩阵；`correlation=True` 时呈现系数相关矩阵，并把对角线按 Stata 的打印方式置零（不显示变量与自身的相关）。两者都是对 `cov_params()` 的纯呈现——不做任何重算——且只打印下三角。`to_frame()` 返回完整的对称矩阵 DataFrame。可调用为 `ws.estat_vce(result)` 或 `result.estat_vce(correlation=True)`。

**签名**

```python
ws.estat_vce(result: 'Any', *, correlation: 'bool' = False) -> 'VceMatrix'
```

**示例**

```python
ws.OLS.from_formula('y ~ x1 + x2', base).fit().estat_vce(correlation=True)
```

**输出**

```text

Correlation matrix of coefficients

        e(V) |         x1          x2       _cons 
-------------+-----------------------------------
          x1 |          0                         
          x2 |  .00379118           0             
       _cons | -.05607434  -.06408771           0
```

### `estat_vif`

**Stata 指令:** `estat vif`

方差膨胀因子（Stata `estat vif`）：对每个回归元，其对其余回归元的辅助回归给出 `VIF = 1/(1 - R2)`，其倒数即容忍度，另外报告 VIF 的算术平均。`uncentered=True` 在辅助回归中追加常数项，对无常数项的模型必须如此，与 `vif.ado` 的拒绝条件对应；`nosort=True` 保持模型自身的回归元顺序而不按 VIF 排序。非线性模型抛出 `NotImplementedError`，因为辅助回归在构造上就是普通最小二乘。可调用为 `ws.estat_vif(result)` 或 `result.estat_vif()`。

**签名**

```python
ws.estat_vif(result: 'Any', *, uncentered: 'bool' = False, nosort: 'bool' = False) -> 'VifResult'
```

**示例**

```python
ws.estat_vif(ws.OLS.from_formula('y ~ x1 + x2', base).fit())
```

**输出**

```text

    Variable |       VIF       1/VIF  
-------------+----------------------
          x1 |      1.00    0.999986
          x2 |      1.00    0.999986
-------------+----------------------
    Mean VIF |      1.00
```

### `estat_auc`

**Stata 指令:** `estat auc`

`logit` 或 `probit` 之后的 ROC 曲线下面积（Stata `estat auc`，`logit_estat.ado` 把它转给 `lroc, nograph`）。计算方式为 Mann-Whitney U 统计量除以两组样本量之积，并使用平均秩，这与 `lroc` 按不同预测概率迭代的做法对并列预测值的处理一致——因此两者完全相等，而不只是在预测值互不相同时相等。`to_dict()` 给出 `r(area)` 与 `r(N)`；`str(...)` 打印 Stata 的那一行说明。可调用为 `result.estat_auc()`，或模块函数 `westata.postestimation.auc.estat_auc(result)`——与其他 `estat_*` 命令不同，它没有被重新导出到顶层 `ws` 命名空间。

**签名**

```python
ws.estat_auc(self) -> 'Any'
```

**示例**

```python
ws.Logit.from_formula('yb ~ x1 + x2', base).fit().estat_auc()
```

**输出**

```text
Area under ROC curve = 0.6435
```

**注意:** 只接受 `logit`/`probit` 结果：`ologit` 不设 `e(estat_cmd)`，没有该子命令。

### `estat_esize`

**Stata 指令:** `estat esize`

线性模型的效应量（Stata `estat esize`）：默认给出 eta 平方，也就是 R 平方；`epsilon=True` 给出 epsilon 平方，`omega=True` 给出 omega 平方，这两个调整 R 平方的类比只在分母上不同，omega 多加上一个 `MS_error`，因此更小、在零效应下也不那么负。两个选项互斥，由 `esize.ado` 的 `opts_exclusive` 强制；只有 eta 平方有区间定义，来自非中心 F 分布。只服务 `regress`/`areg`/`anova` 拟合。请用 `result.estat_esize()` 或 `westata.postestimation.esize.estat_esize(result)`；与 `estat auc` 一样，它没有被重新导出到顶层 `ws` 命名空间。

**签名**

```python
ws.estat_esize(self, **kwargs: 'Any') -> 'Any'
```

**示例**

```python
ws.OLS.from_formula('y ~ x1 + x2', base).fit().estat_esize()
```

**输出**

```text

Effect sizes for linear models

         Eta-squared   [ 2.5% Conf. Interval]
            0.480210       0.379815     0.555658
```

**注意:** `to_dict()` 只返回实际算出的那个估计（`eta2`、`epsilon2` 或 `omega2`），调用方检查键名即可知道命令产出的是哪一个。epsilon 与 omega 没有区间，因为帮助文件并未记载。

### `estat_moran`

**Stata 指令:** `estat moran, errorlag(W)`

`regress` 之后对残差的空间相关做 Moran 检验（Stata `estat moran`）。必须提供空间权重矩阵——`errorlag=`（或其 Python 别名 `weights=`），重复给出时相加成一个滞后算子——因为本包没有 `spmatrix` 工具链。统计量是标准的随机化假设下的 Moran's I：`I = (n/S0)(e'We)/(e'e)`，`E[I] = -1/(n-1)`，方差由权重的一、二阶矩与残差峰度推出，并以 chi2 = z^2、单自由度报告。可调用为 `result.estat_moran(W)`，或模块函数 `westata.postestimation.moran.estat_moran(result, W)`；它没有被重新导出到 `ws` 中。

**签名**

```python
ws.estat_moran(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
W = (np.abs(np.subtract.outer(base['year'].to_numpy(), base['year'].to_numpy())) == 1).astype(float)
r.estat_moran(weights=W)
```

**输出**

```text

Moran test for spatial dependence
H0: error terms are independent
Ha: error terms are spatially correlated

  chi2(1)      =    7.1149
  Prob > chi2   =    0.0076

  Moran's I     = -0.031262
  E[I]          = -0.005025
  z             =   -2.6674
```

**注意:** Stata 在这里的算术在编译型 builtin 中，帮助文件里也没有对应的“方法与公式”一节，因此其中的 `r(chi2)` 无法凭本地证据复现。此处报告的是随机化假设下的标准 Moran 检验；对“无空间相关”这一原假设 p 值是精确的，但可能与 Stata 的数值不同。

### `estat_mundlak`

**Stata 指令:** `estat mundlak`

`xtreg` 之后的 Mundlak 设定检验（Stata `estat mundlak`）：把回归元的组内均值加入随机效应方程并联合检验其为零，因为这些均值正是承载不可观测效应与 `x` 之间相关性的载体；拒绝意味着随机效应估计量不一致。只接受 `xtreg, re`/`cre`/`fe`。必须传入 `entity=`（若原拟合用了 `time=` 也要传），因为拟合结果只保留了编码后的组标签，并不保留增广重估所需的列名。可通过 `result.estat_mundlak(entity='firm', time='year')` 调用，也可用 `westata.postestimation.mundlak.estat_mundlak(result, entity=..., time=...)`。

**签名**

```python
ws.estat_mundlak(self, **kwargs: 'Any') -> 'Any'
```

**示例**

```python
re = ws.RandomEffects.from_formula('ypanel ~ x1 + x2', base, entity='firm', time='year').fit()
re.estat_mundlak(entity='firm', time='year')
```

**输出**

```text

Mundlak specification test
H0: x1, x2 jointly zero (nu uncorrelated with x)

  chi2(2) =    1.6529
  Prob > chi2 =    0.4376
```

**注意:** 系数与标准误来自对增广方程的重新估计，而非原拟合，因此该检验不是 Hausman 统计量的重新标度。

### `estat_szroeter`

**Stata 指令:** `estat szroeter x1 x2`

Szroeter 的秩异方差检验（Stata `estat szroeter`），每个变量一个 chi2(1)：先对该变量取秩，再看残差平方是否集中在高秩或低秩，也就是方差是否随它单调变化。必须给出 `variables` 或 `rhs=True`——给出 `rhs` 时检验模型自身的回归元——单个变量返回一个 `SzroeterResult`，多个变量返回一张表。`mtest` 接受 `bonferroni`、`holm`、`sidak` 或 `noadjust`。只接受既未加权也未聚类的 `regress`/`areg` 拟合，其余情形 `szroeter.ado` 同样拒绝。请用 `result.estat_szroeter(rhs=True)` 或 `westata.postestimation.szroeter.estat_szroeter(result, rhs=True)`；它没有被重新导出到 `ws`。

**签名**

```python
ws.estat_szroeter(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```

**示例**

```python
ws.OLS.from_formula('y ~ x1 + x2', base).fit().estat_szroeter(rhs=True)
```

**输出**

```text

Szroeter's test for homoskedasticity
H0: Variance constant
Ha: Variance monotonic in variables

  Variable             chi2    df         p
  -----------------------------------------
  x1                 0.3760     1    0.5397
  x2                 0.1033     1    0.7479
  -----------------------------------------
```

**注意:** Stata 总是做多重检验但默认不做校正，因此单变量时丢弃校正后的 p 值并报告 `r(chi2)`/`r(df)`/`r(p)`，多变量时则把它放在 `r(mtest)` 矩阵里。此处单个变量返回单个结果，多个变量返回一张表；该表的 `to_dict()` 把 p 值列压平为 `{'p_<变量>': ...}`，`to_frame()` 则保留 chi2、df 与两个 p 值。

### `estat_weakrobust`

**Stata 指令:** `estat weakrobust`

对弱工具变量稳健的推断（Stata `estat weakrobust`）：报告 Anderson-Rubin 检验，其分布不依赖工具变量的强度。这是 Stata 在恰好识别模型下的默认；对过度识别模型，Stata 的默认是 Moreira 的条件似然比检验，而 CLR 需要模拟条件分布、本包未实现，因此过度识别拟合必须传入 `ar=True`，否则以该理由抛出 `NotImplementedError`。`betanull` 设定内生系数上的原假设：可用映射、按位置对应的序列，或 `None` 表示全为零。请用 `result.estat_weakrobust()` 或 `westata.postestimation.weakrobust.estat_weakrobust(result, ar=True)`；它没有被重新导出到 `ws`。

**签名**

```python
ws.estat_weakrobust(self, **kwargs: 'Any') -> 'Any'
```

**示例**

```python
ws.IV2SLS.from_formula('y ~ x1 + [x2 ~ x3]', base).fit().estat_weakrobust()
```

**输出**

```text

Weak-instrument-robust inference
Anderson-Rubin chi2(1) =    1.7538
               F(1,197) =    1.7538   Prob > F = 0.1869
```

**注意:** `betanull` 传序列时按**位置**对应，即 Stata 的 `copy` 子选项；长度不符会直接拒绝，而不是补齐或截断，因为两者都会检验与调用方书写不同的原假设。

### `linktest`

**Stata 指令:** `linktest`

Pregibon 的连接检验，用于发现设定错误（Stata `linktest`）。模型被**重新拟合**，用 `_hat`（拟合的线性指数）与 `_hatsq` 取代原有的回归元，报告的 t 为 `b[_hatsq]/se[_hatsq]`，自由度取**重估后**模型的残差自由度（`linktest.ado:131-132`）。由于该重估始终只有三个参数——`_hat`、`_hatsq` 与常数项——无论原模型如何，自由度都是 n-3，因此它并不等于原拟合的 `df_r`。平方项显著说明线性指数设定有误：函数形式不对，而不只是拟合不好。`xt*`、`iv*`、`sureg`、`mvreg`、`eivreg`、`zip` 与 `zinb` 会被拒绝，因为它们的线性指数不是单一的 `xb`。可通过 `result.linktest()` 调用。

**签名**

```python
ws.linktest(self) -> 'Any'
```

**示例**

```python
ws.OLS.from_formula('y ~ x1 + x2', base).fit().linktest()
```

**输出**

```text

Pregibon's link test
H0: the link is specified correctly

  _hat        =  1.092729
  _hatsq      = -0.041148
  t           =   -0.7700
  P>|t|       =    0.4422   (df = 197)
```

**注意:** 独立函数为 `westata.postestimation.linktest.linktest(result)`——不是 `ws.linktest`；它需要估计样本数据与响应变量名，而这些只在公式接口下才会被保留。

### `suest`

**Stata 指令:** `suest model1 model2`

把多个拟合堆叠成一个参数向量与一个同时的夹心协方差（Stata `suest`）。跨模型分块**不为零**：每个观测的堆叠得分把两个模型的贡献放在一起，这正是模型共享观测时协方差仍然有效的原因。`cluster=` 先按簇把每个观测的堆叠得分求和，再做外积。各模型样本量必须相同；无法给出得分的模型——得分不是残差的非线性估计量，或没有设计矩阵——会抛出 `NotImplementedError`。可调用为 `result.suest(其他, ...)`——注意 `suest` 在这里是**方法**，不是顶层的 `ws.suest`；模块函数为 `westata.postestimation.suest.suest((模型1, 模型2), cluster=...)`，它把各拟合作为一个序列传入。

**签名**

```python
ws.suest(self, *others: 'Any', **kwargs: 'Any') -> 'Any'
```

**示例**

```python
a = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
a.suest(ws.OLS.from_formula('y1 ~ x1 + x2', base).fit())
```

**输出**

```text

Simultaneous covariance matrix
  2 models, 6 parameters, 200 observations
  parameters: model1:Intercept, model1:x1, model1:x2, model2:Intercept, model2:x1, model2:x2
```

**注意:** 按 `suest.sthlp` 的建议，各分量应在**不带** `vce(robust)`/`vce(cluster)` 的情况下估计：`suest` 自身就给出稳健协方差，带稳健分量的结果会被重复校正。

### `newey`

**Stata 指令:** `newey y x1 x2, lag(2)`

Newey-West 回归，`newey 被解释变量 解释变量, lag(#)`（Stata `newey`）。它是一个**函数**而非类：传入公式与数据框，得到 NeweyWestResult。`lag` 必填且不得为负；`time=` 扮演 `tsset` 的角色并启用 `Checkt` 间隔校验，`force=True` 允许时间取值不等间隔；只接受 `aweight`。`e()` 集合被改写为 `newey` 自己的清单，因此 `regress` 会存而 `newey` 不存的键（`r2`、`rss`、`mss`、`rmse`、`sigma`、`ll` 等）会被**删除**，而不是留在里面。

**签名**

```python
ws.newey(formula: 'str', data: 'pd.DataFrame', *, lag: 'int', time: 'Any' = None, force: 'bool' = False, noconstant: 'bool' = False, weights: 'Any' = None, weight_type: 'str | None' = None) -> 'NeweyWestResult'
```

**示例**

```python
ws.newey('y ~ x1 + x2', base, lag=2, time=np.arange(1.0, 201.0)).summary(style='stata')
```

**输出**

```text

Regression with Newey–West standard errors      Number of obs     =        200
Maximum lag = 2                                 F(  2,       197) =     107.69
                                                Prob > F          =     0.0000

------------------------------------------------------------------------------
             |             Newey–West
           y | Coefficient  std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8406445   .0662943    12.68   0.000     .7099069    .9713822
          x2 |   -.512087   .0707643    -7.24   0.000    -.6516398   -.3725342
       _cons |   1.005003   .0768392    13.08   0.000     .8534696    1.156535
```

**注意:** 这里的 `time=` 数组扮演 `tsset` 的角色。`base` 自带的 `year` 列在各实体间重复 1..5，对整体样本并不等间隔，`newey` 会拒绝，除非给出 `force=True`；真正的时间序列两者都不需要。

### `unitroot`

**Stata 指令:** `dfuller y, lags(2)`

单位根检验，由 `test` 选择：`'adf'` 对应 Stata 的 `dfuller`，`'pp'` 对应 `pperron`，`'kpss'` 对应 `kpss`。`regression` 选择确定性项（`'c'`、`'n'`、`'ct'`、`'ctt'`），`autolag` 按 statsmodels 的方式选择滞后阶数，取 `None` 时则照用 `maxlag`，与 Stata 的 `lags(#)` 一致。对 ADF 而言，统计量与 MacKinnon p 值就是 Stata 跑的同一个检验，但**打印的临界值**不是 MacKinnon 响应面值：`dfuller.ado` 对一张六点表按 N 线性插值，本函数返回的正是该表的值，因此打印出的临界带与 Stata 相同。返回 UnitRootResult。

**签名**

```python
ws.unitroot(endog: 'Any', *, test: 'str' = 'adf', regression: 'str' = 'c', maxlag: 'int | None' = None, autolag: 'str | None' = None) -> 'UnitRootResult'
```

**示例**

```python
ws.unitroot(base['y'], test='adf', regression='c', maxlag=2)
```

**输出**

```text

Augmented Dickey–Fuller test for unit root
Variable: y                               Number of obs  = 197
                                          Number of lags =   2

H0: Random walk without drift, d = 0

                                       Dickey–Fuller
                   Test      -------- critical value ---------
              statistic           1%           5%          10%
--------------------------------------------------------------
 Z(t)            -6.706       -3.478       -2.884       -2.574
--------------------------------------------------------------
MacKinnon approximate p-value for Z(t) = 0.0000.
```

**注意:** p 值不受临界值约定的影响，仍是 MacKinnon 值；只有打印出的临界带来自 Stata 的那张表。

### `levin_lin_chu`

**Stata 指令:** `xtunitroot llc ypanel`

Levin-Lin-Chu 面板单位根检验（Stata `xtunitroot llc`）。数据框必须是长格式，面板列与时间列需显式命名。`constant` 与 `trend` 按 Stata 的 `noconstant` 与 `trend` 选择确定性设定：都不给为模型 1，只给常数为模型 2（默认），两者都给为模型 3。返回 LlcResult，打印内容为未调整 t、调整后的 t* 及其 p 值，以及面板规模。

**签名**

```python
ws.levin_lin_chu(frame: 'pd.DataFrame', *, variable: 'str', entity: 'str', time: 'str', trend: 'bool' = False, constant: 'bool' = True, demean: 'bool' = False, lags: 'int | None' = None, hac_lags: 'int | None' = None) -> 'LlcResult'
```

**示例**

```python
ws.levin_lin_chu(base, variable='ypanel', entity='firm', time='year')
```

**输出**

```text
Levin-Lin-Chu unit-root test
  Unadjusted t   -15.8380
  Adjusted t*    -17.2339   p = 0.0000
  40 panels x 5 periods
```

**注意:** `lags=None` 时 LLC 采用 Stata 的**一个**滞后差分，而不是零——`_xturllc.ado` 设 `adf_lags = 1`，这与 IPS 检验不同，后者默认为零。弄反会让每个面板的样本错开一个观测。`hac_lags` 设定 Bartlett 核带宽，默认 `int(3.21 * T^(1/3))`。

### `im_pesaran_shin`

**Stata 指令:** `xtunitroot ips ypanel`

Im-Pesaran-Shin 面板单位根检验（Stata `xtunitroot ips`）。`trend=True` 在每个面板的 ADF 回归中加入线性趋势；`demean=True` 先剔除截面均值再跑各面板回归（即剔除截面均值的 IPS 检验）；`lags` 设定滞后差分阶数，取 `None` 即 Stata 默认为零。返回 IpsResult，打印 t-bar、t-tilde-bar、标准化后的 Z 及其 p 值。

**签名**

```python
ws.im_pesaran_shin(frame: 'pd.DataFrame', *, variable: 'str', entity: 'str', time: 'str', trend: 'bool' = False, demean: 'bool' = False, lags: 'int | None' = None, level: 'float' = 0.05) -> 'IpsResult'
```

**示例**

```python
ws.im_pesaran_shin(base, variable='ypanel', entity='firm', time='year')
```

**输出**

```text
Im-Pesaran-Shin unit-root test
  t-bar     -2.4438
  t-tilde-bar -1.3463
  Z-t-tilde-bar -1.9852  p = 0.0236
  N = 40 panels x 5 periods
```

### `check_time_spacing`

**Stata 指令:** `(no Stata equivalent -- this is the Checkt routine inside newey.ado)`

判定时间变量的间隔：即 `newey.ado` 中的 `Checkt` 例程，之所以单独暴露，既因为 `newey` 需要它，也因为它是本包中唯一把 Stata 的 tsset 间隔规则写清楚的地方。返回 `(tflag, delta)`：`tflag` 为 0 表示间隔为 1 的等间隔，为 1 表示不等间隔，为 2 表示等间隔但间隔大于 1；等间隔时 `delta` 为公共间隔，否则为 `None`。差分按数组的**行顺序**计算，对应 Stata 的 `sort touse tvar`，所有比较都以浮点数进行。

**签名**

```python
ws.check_time_spacing(time_values: 'Any') -> 'tuple[int, float | None]'
```

**示例**

```python
ws.check_time_spacing(np.arange(1.0, 21.0))
```

**输出**

```text
(0, 1.0)
```

**注意:** 请传入整数（或浮点）时间序号，不要传日期：数组会被转成浮点数，因此 `datetime64` 列会变成**纳秒**，一天间隔的序列会被判为 `tflag=2`、`delta` 约 8.6e13，而不是间隔 1。

### `parse_exp_list`

**Stata 指令:** `(no single command -- implements the `exp_list` statistics syntax of [U] exp_list)`

按 `[U] exp_list` 的语法解析 `exp_list`，即 `table`、`estat` 与 `collect` 使用的统计表达式语言。既接受完全加括号的形式 `(mean=r(mean))`，也接受 Stata 允许的顶层裸形式（`or=(exp(_b[mpg]))`、`_b[x1]`）；结果是若干组构成的 `ParsedExpressionList`，每组包含具名的 `Statistic`，其表达式已被切分为 lookup/call/stat 节点。文本有误时抛出 `ExpListError`，而不是返回一个残缺的解析结果。

**签名**

```python
ws.parse_exp_list(text: 'str') -> 'ParsedExpressionList'
```

**示例**

```python
ws.parse_exp_list('(mean=r(mean)) (sd=r(sd))')
```

**输出**

```text
ParsedExpressionList(groups=(StatisticGroup(label=None, statistics=(Statistic(name='mean', expression=('stat', 'r', 'mean')),)), StatisticGroup(label=None, statistics=(Statistic(name='sd', expression=('stat', 'r', 'sd')),))))
```

### `normalize_weights`

**Stata 指令:** `[fw=], [aw=], [pw=], [iw=] -- the weight normalization rules of [U] 20.24`

按 Stata 对给定类型的规则归一化权重数组，返回 `WeightSpec`。四种类型的差异会改变数值：`fweight` 与 `iweight` 不做归一化，并以 `sum(w)` 作为有效样本量贡献；`aweight` 与 `pweight` 被重新缩放至和等于行数，并以 `N` 贡献；`pweight` 还会置 `implies_robust`，这是它与 `aweight` 唯一的区别。校验会拒绝未知类型、负/缺失/非有限权重，以及和为零的情况。

**签名**

```python
ws.normalize_weights(weights: 'Any', weight_type: 'str') -> 'WeightSpec'
```

**示例**

```python
s = ws.normalize_weights(np.array([1.0, 2.0, 3.0, 4.0]), 'fweight')
(s.effective_n, s.implies_robust, s.n_obs)
```

**输出**

```text
(10.0, False, 4)
```

**注意:** `effective_n` 才是自由度所用的量：这个四行 `fweight` 例子里为 10，因为每行按其权重计数。这正是频数权重下 `nobs` 与 `e(N)` 会分岔的原因。


## 已拟合结果对象

### `StataResults`

**Stata 指令:** `(the object every estimation command returns, analogous to Stata's e() results)`

每个 `fit()` 返回的对象。它带有 statsmodels 兼容接口（`params`、`bse`、`pvalues`、`conf_int`、`cov_params`）、以字典形式提供的 Stata `e()` 宏，以及作为方法提供的后估计命令——`test`、`lincom`、`margins`、`predict` 和二十个 `estat_*` 转发器。拟合之后的日常工作大多在这里完成。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.e['r2']
```

**输出**

```text
0.4802098236178376
```

**注意:** `e()` 的键集逐命令不同，并不统一；假定它统一正是凭空添加键的来源。`e(ic)`、`e(aic)`、`e(title)` 都是某些命令有、某些命令没有。

### `params`

**Stata 指令:** `regress -- the `Coef.` column, i.e. `e(b)``

点估计值，以 `param_names` 为索引的 `pandas.Series`，顺序与构建设计矩阵时的列序一致。该顺序是 patsy 的顺序，常数项排在最前（`Intercept`），而 Stata 的 `e(b)` 把 `_cons` 排在最后；两个向量数值相同但顺序不同。要与 Stata 的 `e(b)` 对比时请用 `stata_params`。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.params
```

**输出**

```text
Intercept    1.005003
x1           0.840645
x2          -0.512087
Name: params, dtype: float64
```

### `bse`

**Stata 指令:** `regress -- the `Std. err.` column`

标准误，即 `cov_type` 所选协方差矩阵对角线的平方根。它在构造时由 `cov` 一次算出，因此之后改动协方差不会改变 `bse`。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.bse
```

**输出**

```text
Intercept    0.074428
x1           0.072471
x2           0.074822
Name: bse, dtype: float64
```

### `pvalues`

**Stata 指令:** `regress -- the `P>|t|` column; logit -- the `P>|z|` column`

各系数为零这一原假设的双侧 p 值。参照分布由 `dist` 决定：`dist` 为 `"t"` 时用自由度为 `df_resid` 的 Student t 分布，为 `"z"` 时用标准正态分布。因此聚类 VCE 下 `df_resid` 为 `G - 1` 的模型用的是该计数，而不是 `nobs - k`。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.pvalues
```

**输出**

```text
Intercept    7.633899e-30
x1           4.755835e-24
x2           9.518706e-11
Name: pvalues, dtype: float64
```

### `tvalues`

**Stata 指令:** `regress -- the `t` column`

系数除以其标准误，即便对基于 z 的模型也始终命名为 `tvalues`。它是检验统计量，而非严格意义上的 t 统计量：对 logit 拟合，这些数字就是 Stata 打印的 z 值。`zvalues` 是同一对象，只是用了分布所对应的名字。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.tvalues
```

**输出**

```text
Intercept    13.502927
x1           11.599724
x2           -6.844043
Name: tvalues, dtype: float64
```

### `zvalues`

**Stata 指令:** `logit, probit, poisson -- the `z` column`

系数为零这一原假设的正态统计量。它与 `tvalues` 在数值上完全相同（两者都是 `params / bse`），之所以以这个名字暴露，是为了让下游代码能按推断分布分支而无需重算。它是别名而非独立计算，因此返回的 Series 仍带有 `tvalues` 这个名字。

**示例**

```python
r = ws.Logit.from_formula('yb ~ x1 + x2', base).fit()
r.zvalues
```

**输出**

```text
Intercept    1.032644
x1           3.466025
x2          -0.888860
Name: tvalues, dtype: float64
```

### `fvalue`

**Stata 指令:** `regress -- the `F(k, n-k)` model test`

模型的联合 F 统计量，取自 `e(F)`；命令不存该宏时为 NaN。基于似然的命令改报卡方模型检验，存为 `e(chi2)`，因此 logit 或 Poisson 拟合后 `fvalue` 是 NaN——这是正确的“不存在”，而非漏算。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.fvalue
```

**输出**

```text
90.99954130641439
```

### `llf`

**Stata 指令:** `e(ll) -- the log-likelihood`

最大化对数似然，取自 `e(ll)`；命令不记录该宏时为 NaN。`estat ic` 与 `lrtest` 的统计量都由这个数构成，因此此处报 NaN 的拟合无法支撑这两者。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.llf
```

**输出**

```text
-291.7997221200006
```

### `aic`

**Stata 指令:** `e(aic) -- note that `estat ic` is the surface that matches Stata's printed table`

取自 `e(aic)` 的 AIC；命令不记录该宏时为 NaN。对 `regress` 而言这不是 `estat ic` 打印的那个数：`regress` 根本不存 AIC，此处暴露的值在正态偏差尺度上，即 `N log(rss/N) + 2k`，与似然尺度的 `-2 logL + 2k` 是不同的量。对确实存 `e(aic)` 的命令，两者一致。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
(r.aic, ws.estat_ic(r).aic)
```

**输出**

```text
(589.5994442400012, 589.5994442400012)
```

**注意:** 不要把 `result.aic` 与 `regress` 拟合的 `estat ic` 输出相比；两者尺度不同。`e(_aic_is_stata)` 记录适用的是哪一种含义，摘要表仅在该标志表明该值即 Stata 的值时才打印 AIC。

### `bic`

**Stata 指令:** `e(bic) -- with the same caveat as `aic``

取自 `e(bic)` 的 BIC；命令不记录该宏时为 NaN。`aic` 的尺度注意事项同样适用：`regress` 之后存的值是正态偏差形式 `N log(rss/N) + k log(N)`，而非 `estat ic` 打印的似然尺度准则。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
(r.bic, ws.estat_ic(r).bic)
```

**输出**

```text
(599.4943963396453, 599.4943963396453)
```

### `nobs`

**Stata 指令:** `e(N) / the `Number of obs` line`

估计样本的行数，为整数。在频数权重下它刻意不等于 `e(N)`：使用 `fweight` 时行是重复计数，因此 `e(N)` 变为 `sum(w)`，而 `nobs` 仍是行数。有效样本量请用 `result.e['N']`。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
(r.nobs, r.e['N'])
```

**输出**

```text
(200, 200.0)
```

### `df_model`

**Stata 指令:** `e(df_m) -- model degrees of freedom`

模型自由度，即模型检验施加的约束个数：不含常数项的回归元个数。它取自 `e(df_m)`，因此单回归元模型的值为 1 而非 2，尽管系数向量有两个元素。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.df_model
```

**输出**

```text
2.0
```

### `df_resid`

**Stata 指令:** `e(df_r) -- residual degrees of freedom`

参照分布使用的分母自由度。它并不总是 `nobs - k`：聚类 VCE 下为 `G - 1`，即聚类数减一。本属性返回估计器在拟合时算出的值；`e(df_r)` 是另一回事，因为该宏只对非 z 模型存在（或由 `tobit`、`xtivreg` 等估计器显式提供），因此 logit 拟合在这里有 `df_resid`，却没有 `e(df_r)`。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1', base).fit(cov_type='cluster', cov_kwds={'groups': base['firm']})
(r.df_resid, r.cov_type)
```

**输出**

```text
(39.0, 'cluster')
```

### `dist`

**Stata 指令:** `(no Stata equivalent -- Stata picks t or z per command and stores it in e() indirectly)`

推断所使用的参照分布：`"t"` 表示 t 统计量与 t 置信区间，`"z"` 表示正态检验。它由估计器而非调用者决定，`pvalues` 与 `conf_int` 正是据此分支。线性拟合报 `t`；logit、probit 与计数模型报 `z`。

**示例**

```python
r = ws.Logit.from_formula('yb ~ x1 + x2', base).fit()
r.dist
```

**输出**

```text
z
```

### `model_name`

**Stata 指令:** `e(cmd) -- the command name, e.g. `regress``

产生该结果的估计器对应的 Stata 命令名——`regress`、`logit`、`reghdfe` 等——它同时也是摘要打印的标签。估计器未记录 `e(cmd)` 时，用该值合成，因此 `result.e['cmd']` 总是可用，尽管 Stata 本身总是会存它。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.model_name
```

**输出**

```text
regress
```

### `cov_type`

**Stata 指令:** `e(vce) / e(vcetype) -- the `Covariance Type` line of the summary`

所用协方差估计器的标签，用于展示与审计：默认 `nonrobust`，异方差稳健为 `HC1`，聚类为 `cluster`。它是记录下来的标签而非开关，事后修改不会改变任何东西，因为矩阵本身在拟合时已构建完成。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit(cov_type='HC1')
r.cov_type
```

**输出**

```text
HC1
```

### `cov`

**Stata 指令:** `e(V) -- the coefficient covariance matrix`

系数协方差矩阵，为 `k x k` 的 numpy 数组，已包含所请求的全部小样本修正。它与 `cov_params()` 返回的是同一对象；`cov_params()` 的存在只是为了满足按方法调用的 statsmodels 兼容代码。顺序遵循 `param_names`，即 patsy 顺序，而非 Stata 的 `e(V)` 顺序。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
np.round(r.cov, 4)
```

**输出**

```text
[[ 0.0055 -0.0003 -0.0004]
 [-0.0003  0.0053  0.    ]
 [-0.0004  0.      0.0056]]
```

### `e`

**Stata 指令:** `e() -- the stored estimation results`

以字典式对象提供的 Stata `e()` 宏，同时支持下标与属性访问。下标访问保持普通映射语义，命令未存的宏会抛 `KeyError`；属性访问则刻意宽容，返回 `None`。键集逐命令不同——假定它统一，正是手册里凭空多出键的原因：`regress` 在 `estat ic` 尺度上不存 `aic`，`tobit` 则根本不存 `aic`。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
sorted(r.e)
```

**输出**

```text
['F', 'N', '_aic_is_stata', 'aic', 'bic', 'cmd', 'depvar', 'df_m', 'df_r', 'estat_cmd', 'll', 'll_0', 'marginsok', 'model', 'mss', 'predict', 'properties', 'r2', 'r2_a', 'rank', 'rmse', 'rss', 'sigma', 'sum_w', 'title', 'vce', 'vcetype']
```

**注意:** 有两个宏是合成出来的，因为每条命令都记录它们：`e(depvar)` 取自响应变量名，`e(cmd)` 取自 `model_name`。`e(cmdline)` 刻意不合成——它记录的是用户键入的原样命令（含选项），本包没有该字符串，编造一个即为伪造值。

### `fittedvalues`

**Stata 指令:** `predict, xb -- the linear prediction`

估计样本上的拟合值，长度为 `nobs`、按行序排列的 numpy 数组。加权拟合时它们是加权最小二乘问题的预测值，而非加权后的拟合值。`predict(kind='xb')` 会从保留下来的设计矩阵重算同一个量，样本外数据应改用该入口。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.fittedvalues[:5]
```

**输出**

```text
[2.524914 0.144945 1.126616 2.107762 0.869349]
```

### `resid`

**Stata 指令:** `predict, residuals -- the residuals`

估计样本上的残差，即 `y - fittedvalues`，长度为 `nobs` 的 numpy 数组。它们是原始残差，而非标准化或学生化残差；加权拟合时它们是原始离差，而非加权残差。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.resid[:5]
```

**输出**

```text
[-0.245449 -0.01285  -1.598986  0.354059  0.682509]
```

### `rsquared`

**Stata 指令:** `e(r2)`

取自 `e(r2)` 的 R²；命令不记录该宏时为 NaN。它属于 OLS 一族；极大似然族改在 `e(r2_p)` 中存伪 R²，此处留作 NaN，因此这里的 NaN 表示“该模型没有 R²”，而非计算失败。logit、probit 与计数模型请读 `e['r2_p']`。

**示例**

```python
ols = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
lg = ws.Logit.from_formula('yb ~ x1 + x2', base).fit()
((ols.rsquared, lg.rsquared), lg.e['r2_p'])
```

**输出**

```text
((0.4802098236178376, nan), 0.05037590968400629)
```

### `rsquared_adj`

**Stata 指令:** `e(r2_a)`

取自 `e(r2_a)` 的调整 R²；命令不记录该宏时为 NaN。与 `rsquared` 一样它属于线性族，其自由度修正使用有效样本量，因此频数加权拟合按 `sum(w)` 而非行数修正。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.rsquared_adj
```

**输出**

```text
0.4749327659895922
```

### `model_data`

**Stata 指令:** `e(sample) / the estimation sample`

该拟合所用的估计样本数据框。Stata 通过 `e(sample)` 与保存的数据集提供同一信息；这里直接就是数据框，因此需要重读估计数据的后估计命令（如 `linktest`、`estat_mundlak`、需显式样本的 `predict`）可以直接从结果对象工作，无需调用者再次传入数据。

**示例**

```python
r = ws.OLS.from_formula("y ~ x1 + x2", base).fit()
list(r.model_data.columns)
```

**输出**

```text
['y', 'x1', 'x2', 'x3', 'yb', 'ycnt', 'ycens', 'lo', 'hi', 'y1', 'y2', 'firm', 'year', 'ypanel', 'treat', 'ytreat', 'sel', 'yselect']
```

**注意:** 只有保留了样本的估计器才带有它；假定它总是存在，会让后估计调用在深处失败而不是在调用处失败。

### `converged`

**Stata 指令:** `e(converged) -- 1 if the optimizer converged`

优化器是否报告收敛，取自 `e(converged)`。该宏缺失时报 `True` 而非 `False`：缺失意味着估计器根本不跑迭代优化，闭式解没有“收敛失败”可言。因此 `True` 在此有两种含义：对 ML 拟合是“已收敛”，对 OLS 是“不适用”。

**示例**

```python
r = ws.Logit.from_formula('yb ~ x1 + x2', base).fit()
(r.converged, r.iterations, r.grad_norm)
```

**输出**

```text
(True, 5.0, nan)
```

### `iterations`

**Stata 指令:** `e(ic) -- the number of iterations`

优化器迭代次数，取自 `e(ic)`；帮助文件将该宏说明为迭代次数。估计器非迭代或后端未报告计数时为 NaN。它是浮点数而非整数，因此非迭代拟合可以与“零次迭代即收敛”的拟合区分开。

**示例**

```python
r = ws.Logit.from_formula('yb ~ x1 + x2', base).fit()
r.iterations
```

**输出**

```text
5.0
```

### `grad_norm`

**Stata 指令:** `e(gradient_norm) -- the norm of the gradient at the reported estimate`

所报告估计值处的梯度范数，取自 `e(gradient_norm)`。数值很小是 `converged` 的数值对应物：优化器在它接近零时停止。后端未报告时它一律为 NaN，这是刻意的，因为 `0.0` 会被读作“精确收敛”，调用者将无法把它与“数据不可得”区分开。

**示例**

```python
r = ws.Logit.from_formula('yb ~ x1 + x2', base).fit()
r.grad_norm
```

**输出**

```text
nan
```

### `param_names`

**Stata 指令:** `row names of e(b) -- but in patsy spelling and order`

系数名，按系数顺序排列的字符串列表。拼写是 patsy 的：`Intercept` 在前，因子水平写作 `foreign[T.Foreign]`，交互项用 `:`。Stata 则把 `_cons` 排在最后、裸哑变量写作 `foreign`、交互项用 `#`；`stata_param_names` 负责该转换。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.param_names
```

**输出**

```text
['Intercept', 'x1', 'x2']
```

### `test`

**Stata 指令:** `test x1 = 0`

线性假设的 Wald 检验（Stata `test`）。裸系数列表（`'x1 x2'`）检验**每一个**都为零——这是联合检验，不是检验它们之和——`*` 按通配符展开；`'x1 = x2'` 以及链式写法 `'x1 = x2 = x3'` 检验相应的等式；多个约束可在同一个字符串里用逗号分隔，也可作为多个实参传入。`coef=True` 还会返回一个字典，包含 Stata 的受约束估计 `b_c = b - VR'(RVR')^-1(Rb - r)`、其协方差以及检验本身。统计量在模型报告 t 时为 F、报告 z 时为 chi2，分母自由度为 `df_resid`——聚类 VCE 下是 `G - 1`，不是 `nobs - k`。可通过 `result.test(...)` 调用。

**签名**

```python
ws.test(self, *specs: 'Any', **kwargs: 'Any') -> 'Any'
```

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.test('x1 = 0')
```

**输出**

```text

 ( 1)  x1 = 0

       F(  1,   197) =  134.55
            Prob > F =    0.0000
```

**注意:** 裸系数列表中出现常数是错误而非假设：Stata 对此返回 `r(111) '<n>' not found`，因此应把它放在 `=` 的右侧（`'x1 = 0'`），那里是允许的。

### `lincom`

**Stata 指令:** `lincom x1 - x2`

系数的一个线性组合（Stata `lincom`）：`L b`，标准误为 `sqrt(L V L')`，并沿用模型自身的参考分布。裸常数会被移到所打印约束的右端，且从不进入方差，因此 `lincom('2*x1 - x2 + 3')` 与 `lincom('2*x1 - x2')` 的标准误完全相同。`=` 号或逗号会被拒绝——Stata 的语法只有**一个**表达式，逗号分隔的是表达式与选项——多个组合应交给 `test`。`eform=True` 报告 `exp(Lb)` 并对区间取指数。返回 LincomResult，其 `to_dict()` 给出 Stata 的 `r()` 名称。

**签名**

```python
ws.lincom(self, expression: 'str', **kwargs: 'Any') -> 'Any'
```

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.lincom('x1 - x2')
```

**输出**

```text

 ( 1)  x1 - x2 = 0

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   1.352732   .1039679    13.01   0.000     1.147699    1.557764
------------------------------------------------------------------------------
```

**注意:** 对基于 z 的拟合，统计量存为 `r(z)`，而 `r(df)` 确实不存在——按另一个名字去取会一无所获，这与 Stata 一致，而不是强行造出一个值。

### `predict`

**Stata 指令:** `predict yhat, pr`

按类型预测（Stata `predict 新变量, <类型>`）。`kind` 选择类型，`exog` 默认取结果中保留的设计矩阵，因此常见的样本内调用无需参数。可用类型因估计量而异——`xb`、`stdp` 对每个模型都有；`pr` 用于 logit/probit/poisson/有序/tobit；`n`、`ir` 用于计数模型；`e`、`ystar` 用于 tobit/区间回归；`score` 用于 logit/probit/poisson；`hat`、`cooksd`、`dfbeta` 等用于 regress——权威清单是 `westata.postestimation.predict.supported_types(model_name)`。未保留设计矩阵的结果会抛出 `ConsistencyError`，并给出针对该估计量的具体解释，而不是笼统信息。

**签名**

```python
ws.predict(self, exog: 'Any' = None, kind: 'str | None' = None, **kwargs: 'Any') -> 'Any'
```

**示例**

```python
ws.Logit.from_formula('yb ~ x1 + x2', base).fit().predict(kind='pr')[:6]
```

**输出**

```text
[0.76494  0.407506 0.547025 0.670233 0.518222 0.633899]
```

**注意:** 在 `reghdfe` 之后不提供任何类型：被吸收的效应是否计入预测值是一个真实的选择，Stata 用不同选项表达它，而默默返回其中一种会是静默的错误。在 `xtreg` 之后各类型可用，`xb` 不含面板效应，这是 Stata 的默认（`xtreg_postestimation.sthlp:128`）；`xbu` 是把它加回来的变体。

### `summary`

**Stata 指令:** `(no Stata equivalent -- renders the estimation table Stata prints)`

把拟合结果渲染成文本摘要。`style='statsmodels'`（默认）打印 statsmodels 风格的表格，`style='stata'` 打印同一命令下 Stata 的版式——`regress` 为方差分析与拟合区块加系数表，离散选择模型为 LR 卡方表头与 z 统计量。它**返回**字符串而不是直接打印，因此在 notebook 中会显示，也可以原样写入文件。

**签名**

```python
ws.summary(self, style: 'str' = 'stata', *, odds_ratio: 'bool' = False, first: 'bool' = False) -> 'str'
```

**示例**

```python
ws.OLS.from_formula('y ~ x1 + x2', base).fit().summary(style='stata')
```

**输出**

```text

      Source |       SS           df       MS      Number of obs   =       200
-------------+----------------------------------   F(2, 197)       =     91.00
       Model |  200.183699         2  100.091849   Prob > F        =    0.0000
    Residual |  216.683448       197  1.09991598   R-squared       =    0.4802
-------------+----------------------------------   Adj R-squared   =    0.4749
       Total |  416.867146       199  2.09480978   Root MSE        =    1.0488

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8406445   .0724711    11.60   0.000     .6977258    .9835633
          x2 |   -.512087   .0748223    -6.84   0.000    -.6596425   -.3645315
       _cons |   1.005003   .0744285    13.50   0.000     .8582236    1.151781
------------------------------------------------------------------------------
```

### `wald_test`

**Stata 指令:** `(no Stata spelling -- Stata expresses the same restriction as 'test x1 = 0')`

与 statsmodels 兼容的联合 Wald 检验：`R b = q`，`R` 每行一个约束、**每列对应一个参数**，`q` 默认为零。`scalar=True` 返回 F 检验且 `df_denom = df_resid`，`scalar=False` 返回 chi2 检验，这正是 statsmodels 自身的行为；返回对象与 `test` 返回的是同一个 `WaldTest`，暴露 `statistic`、`df`（分子自由度）、`df_denom`、`pvalue` 与 `distribution`。需要 Stata 的约束语法时请用 `test`——这里是矩阵接口，形状不匹配会抛出 `ValueError`，而不是猜你想检验哪些参数。

**签名**

```python
ws.wald_test(self, R: 'Any', q: 'Any' = None, scalar: 'bool' = True) -> 'Any'
```

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.wald_test(np.array([[0.0, 1.0, 0.0]]), scalar=False)
```

**输出**

```text

 ( 1)  x1 = 0

       chi2(  1) =  134.55
            Prob > chi2 =    0.0000
```

### `cov_params`

**Stata 指令:** `matrix list e(V)`

系数协方差矩阵，以纯 numpy 数组返回，顺序为 `param_names` 的顺序。那是 patsy 的顺序，截距在前、因子水平按它自己的写法拼名，**不是** Stata `e(V)` 的顺序——Stata 把 `_cons` 放在最后，同一个约束的写法也不同。`stata_cov_params` 与 `stata_params` 给出置换并改名后的版本，`estat_vce` 则按 Stata 的样式打印。返回的是内部存储的数组本身，不是副本。

**签名**

```python
ws.cov_params(self) -> 'np.ndarray'
```

**示例**

```python
ws.OLS.from_formula('y ~ x1 + x2', base).fit().cov_params()
```

**输出**

```text
[[ 0.00554  -0.000302 -0.000357]
 [-0.000302  0.005252  0.000021]
 [-0.000357  0.000021  0.005598]]
```

### `conf_int`

**Stata 指令:** `(no Stata command -- the confidence-interval columns Stata prints in the estimation table)`

系数的置信区间，返回带 `lower` 与 `upper` 两列、以参数名为索引的 DataFrame。默认 `alpha=0.05`；`dist='t'` 时临界值取自由度为 `df_resid` 的 t 分布，`dist='z'` 时取标准正态，与系数表的选择一致。`alpha` 超出 (0, 1) 会抛出 `ValueError`。索引按 `param_names` 顺序，因此与 `params`、`bse` 对齐，而不是与 Stata 的 `e(b)` 对齐。

**签名**

```python
ws.conf_int(self, alpha: 'float | None' = None) -> 'pd.DataFrame'
```

**示例**

```python
ws.OLS.from_formula('y ~ x1 + x2', base).fit().conf_int(alpha=0.10)
```

**输出**

```text
              lower     upper
Intercept  0.882000  1.128005
x1         0.720877  0.960412
x2        -0.635740 -0.388434
```


## 错误与支撑类型

### `WestataError`

**Stata 指令:** `(no Stata equivalent -- an exception base class)`

本包所有专有失败的基类。捕获它即可一次性覆盖整个家族：`ConsistencyError`、`ConvergenceError`、`WeightError`、`FormulaError` 与 `UnsupportedError` 都是它的子类，因此只想把本包自身的拒绝与 numpy/pandas 的错误区分开的调用者，捕获这一个名字即可。它不会捕获对普通错误参数抛出的 `ValueError`——但它的两个子类 `ConsistencyError` 与 `FormulaError` 本身就是 `ValueError`，因此 `except WestataError` 会捕获所有 VCE 与公式拒绝，而一般类型错误仍会逃出。

**示例**

```python
[c.__name__ for c in (ws.ConsistencyError, ws.ConvergenceError, ws.WeightError, ws.FormulaError, ws.UnsupportedError) if issubclass(c, ws.WestataError)]
```

**输出**

```text
['ConsistencyError', 'ConvergenceError', 'WeightError', 'FormulaError', 'UnsupportedError']
```

### `ConsistencyError`

**Stata 指令:** `(no Stata equivalent -- Stata refuses with return codes such as r(498) or r(2000))`

当结果无法保证与 Stata 一致时抛出，本包选择拒绝，而不是返回一个它知道可能偏离的数字。这是刻意的失败通道：聚类数少于两个的聚类 VCE、不可识别的 IV 模型、参数个数相同的两个模型之间的 `lrtest`、`pweight` 拟合配非稳健协方差。消息会说明哪里不一致，因此该拒绝是可审计的而非莫名的。它同时刻意是 `ValueError`：`except ValueError` 仍能捕获它，而 `except WestataError` 现在也能捕获 VCE 的拒绝——正是这两个基类，使同一个用户错误在 OLS、logit 与各 IV 估计器上只对应一个异常类型，而不是两个。

**示例**

```python
msg = None
try:
    r = ws.OLS.from_formula('y ~ x1', base).fit(cov_type='cluster', cov_kwds={'groups': np.zeros(len(base), dtype=int)})
except ws.WestataError as exc:
    msg = (type(exc).__name__, isinstance(exc, ValueError))
msg
```

**输出**

```text
('ConsistencyError', True)
```

**注意:** `cov_kwds` 的键名并不统一：OLS 与各 ML 估计器用 `{'groups': 标签}`，而各 IV 估计器用 `{'clusters': 标签}`，会把 `groups` 当作多余条目拒绝。请按所使用估计器的文档传键，否则你拿到的错误将关于 kwds 名称而非你的聚类。

### `ConvergenceError`

**Stata 指令:** `(no Stata equivalent -- Stata reports `convergence not achieved` and returns estimates, r(430))`

当迭代估计器未收敛、且调用者通过 `strict=True` 要求硬失败时抛出。默认是 `strict=False`，此时仍返回估计值并记录 `converged=False`，与 Stata 报告未收敛却保留估计值的行为一致。当未收敛的估计比没有估计更糟时，请传 `strict=True`。

**示例**

```python
msg = None
try:
    ws.Logit.from_formula('yb ~ x1 + x2', base).fit(maxiter=1, strict=True)
except ws.ConvergenceError as exc:
    msg = str(exc)
msg
```

**输出**

```text
logit: optimiser did not converge after 1 iterations; estimates may be unreliable
```

### `FormulaError`

**Stata 指令:** `(no Stata equivalent -- Stata raises r(111) for an unknown variable)`

当模型公式无法在给定数据上求值时抛出：未知列、畸形项、不存在的因子。它把 patsy 自身的错误包装成普通异常，而不让 `PatsyError` 逃出，因为 patsy 会在回溯帧中保留命名空间对象，若干格式化器（包括 pytest 的）在自省它时会失败，从而掩盖真正的消息。与 `ConsistencyError` 一样，它既是 `WestataError` 的子类也是 `ValueError` 的子类，因此通用的参数错误处理仍能捕获它。

**示例**

```python
msg = None
try:
    ws.OLS.from_formula('y ~ nosuchcolumn', base)
except ws.FormulaError as exc:
    msg = str(exc).splitlines()[0]
msg
```

**输出**

```text
cannot evaluate formula 'y ~ nosuchcolumn': Error evaluating factor: NameError: name 'nosuchcolumn' is not defined
```

### `UnsupportedError`

**Stata 指令:** `(no Stata equivalent -- this means Stata CAN do it and westata cannot yet)`

当某项能力在 Stata 中存在但本包尚未实现时抛出——与 `ConsistencyError` 的区别很重要。它同时是 `NotImplementedError` 的子类，因此未知的 predict 类型、或 Stata 支持而本包未实现的 `estat` 子命令，会作为明确的缺口而非静默错误的数字抵达调用者。消息会说明请求了什么、以及实际可用什么。

**示例**

```python
msg = None
try:
    ws.Logit.from_formula('yb ~ x1', base).fit().predict(kind='zzz')
except ws.UnsupportedError as exc:
    msg = str(exc)
msg
```

**输出**

```text
predict: 'logit' supports ('xb', 'stdp', 'pr', 'residuals', 'score'); got 'zzz'
```

### `WeightError`

**Stata 指令:** `(no Stata equivalent -- Stata raises r(402) or r(401) for a bad weight)`

当 Stata 权重设定缺失、不受支持或含义不明时抛出：未知权重类型、负权重、缺失权重、权重和为零，或提供了 `weights` 却未给 `weight_type`（反之亦然）。权重在构建模型时一次性校验，因此错误会点明有问题的条件，而不会之后以协方差中的 NaN 形式浮现。

**示例**

```python
msg = None
try:
    ws.normalize_weights(np.array([1.0, -1.0]), 'aweight')
except ws.WeightError as exc:
    msg = str(exc)
msg
```

**输出**

```text
None
```

### `ExpListError`

**Stata 指令:** `(no Stata equivalent -- Stata's exp_list parser reports r(198))`

当文本不符合 Stata 的 `exp_list` 语法，或某个统计量引用了不存在的已存结果时抛出。它是 `ValueError` 的子类。两类失败在消息中被区分开：语法错误会引用有问题的文本与偏移量，而无法解析的统计量会列出实际可用的键——这正是让“只存 `r(N)` 的命令之后写 `r(mean)`”可诊断的原因。

**示例**

```python
msg = None
try:
    ws.parse_exp_list('(((')
except ws.ExpListError as exc:
    msg = str(exc)
msg
```

**输出**

```text
unbalanced parentheses in '((('
```

### `PackageNotFoundError`

**Stata 指令:** `(no Stata equivalent -- an importlib.metadata exception, re-exported)`

当找不到某发行版的元数据时由 `importlib.metadata` 抛出的异常。此处重新导出，使调用者无需自己导入 `importlib.metadata`；`westata.__version__` 也以它为回退：未安装而直接运行源码树时没有发行版元数据，版本号于是取字面回退值而非抛错。它本身不是 `WestataError`——它是标准库的 `ImportError` 子类。

**示例**

```python
(ws.PackageNotFoundError.__module__, issubclass(ws.PackageNotFoundError, ImportError))
```

**输出**

```text
('importlib.metadata', True)
```

**注意:** 由于它是 `importlib.metadata` 的类而非本包的异常，`except ws.WestataError` 捕不到它。请直接捕获 `ws.PackageNotFoundError` 或 `ImportError`。

### `StoredResults`

**Stata 指令:** `r(), e(), _b[] and _se[] -- the four tables a replicate leaves behind`

单次重复所运行命令留下的已存结果，是一个冻结 dataclass，含四个映射字段：`r` 与 `e` 存放标量，`b` 与 `se` 按名称存放系数。它们正好是 Stata 的 `exp_list` 能触及的四张表，这也是 bootstrap 或置换统计量针对该对象编写的原因。`lookup(table, key)` 返回浮点数，条目缺失时抛 `ExpListError` 并列出可用的键。

**签名**

```python
ws.StoredResults(r: 'Mapping[str, float]' = <factory>, e: 'Mapping[str, float]' = <factory>, b: 'Mapping[str, float]' = <factory>, se: 'Mapping[str, float]' = <factory>) -> None
```

**示例**

```python
s = ws.StoredResults(r={'mean': 1.5}, e={'N': 200.0})
ws.parse_exp_list('(n=e(N)) (m=r(mean))').evaluate(s)
```

**输出**

```text
{'n': 200.0, 'm': 1.5}
```

### `WeightSpec`

**Stata 指令:** `[fw=], [aw=], [pw=], [iw=] -- the parsed weight specification`

由 `normalize_weights` 从权重数组与 Stata 权重类型生成的、已归一化的权重及估计器所需的元数据。字段：`weights`（已按该类型规则归一化）、`weight_type`、`effective_n`（用于自由度的样本量）、`implies_robust`（对 `pweight` 为真，因为 Stata 的 `[pw=w]` 蕴含 `vce(robust)`）、`n_obs`（行数而非有效样本量）。它是冻结 dataclass，估计器无法在校验后悄悄改写该设定。

**签名**

```python
ws.WeightSpec(weights: 'np.ndarray', weight_type: 'str', effective_n: 'float', implies_robust: 'bool', n_obs: 'int', dropped: 'int' = 0, kept: 'np.ndarray | None' = None) -> None
```

**示例**

```python
ws.normalize_weights(np.array([1.0, 2.0, 3.0, 4.0]), 'pweight')
```

**输出**

```text
WeightSpec(weights=array([0.4, 0.8, 1.2, 1.6]), weight_type='pweight', effective_n=4.0, implies_robust=True, n_obs=4, dropped=0, kept=None)
```

### `WEIGHT_TYPES`

**Stata 指令:** `[fw=], [aw=], [pw=], [iw=]`

凡是接受权重之处都接受的四个 Stata 权重类型名，以元组给出：`fweight`、`aweight`、`pweight`、`iweight`。含义即 `[U] 20.24` 所述：频数权（整数重复计数，样本实际为 `sum(w)`）、解析/精度权（缩放至和等于 `N`，估计上按频数权使用）、抽样权（点估计同 `aweight`，但方差必须用线性化三明治——`[pw=w]` 蕴含 `vce(robust)`）、重要性权（按频数权使用且不归一化）。读者最常弄错的是：`fweight` 与 `aweight` 点估计相同、只差在自由度，而 `pweight` 才是改变方差的那一个。

**示例**

```python
ws.WEIGHT_TYPES
```

**输出**

```text
('fweight', 'aweight', 'pweight', 'iweight')
```

### `SSC`

**Stata 指令:** `the `, ssc(...)` style choices behind vce()'s small-sample corrections`

Stata 风格的小样本修正设定，为冻结 dataclass：`adj` 施加 `N/(N-K)` 自由度修正，`cluster_adj` 在 `adj` 之上再施加 `G/(G-1)` 聚类修正，`df` 选择 t/z 自由度遵循的约定（默认 `"stata"`，即遵循对应 Stata 命令自身的默认）。每个字段都默认为 Stata 的选择，因此接受 `ssc=` 的估计器在调用者另行指定前表现与 Stata 一致。

**签名**

```python
ws.SSC(adj: 'bool' = True, cluster_adj: 'bool' = True, df: 'str' = 'stata') -> None
```

**示例**

```python
ws.SSC(adj=False)
```

**输出**

```text
SSC(adj=False, cluster_adj=True, df='stata')
```

### `VceMatrix`

**Stata 指令:** `estat vce / estat vce, correlation`

打印出来的系数协方差距阵（或相关阵）。三个字段：`names`（按 Stata 拼写的行列标签）、`values`（完整对称矩阵的 DataFrame——机器可读接口）、`kind`（`"covariance"` 或 `"correlation"`）。`str(...)` 按 Stata 的方式只渲染下三角，`to_frame()` 返回完整对称矩阵。行序与系数表一致，`_cons` 在最后。

**签名**

```python
ws.VceMatrix(names: 'Sequence[str]', values: 'np.ndarray', kind: 'str', model_label: 'str' = '', response_label: 'str' = '') -> 'None'
```

**示例**

```python
ws.estat_vce(ws.OLS.from_formula('y ~ x1 + x2', base).fit())
```

**输出**

```text

Covariance matrix of coefficients of regress model

        e(V) |         x1          x2       _cons 
-------------+-----------------------------------
          x1 |  .00525206                         
          x2 |  .00002056   .00559838             
       _cons | -.00030246   -.0003569    .0055396
```

**注意:** 当 `kind == "correlation"` 时，对角线按 Stata 的打印方式置为 0——在那张表里变量不“与自身相关”。要取真实的对角线（其值为 1）请读 `values`。

### `ParsedExpressionList`

**Stata 指令:** `(the parsed form of an exp_list -- what Stata's parser builds internally)`

解析后的 Stata `exp_list`，可直接对单次重复的已存结果求值。它持有 `groups`（统计量组的元组，带标签的组保留其标签），并提供 `names`（展平后的统计量名，有标签的组加 `label:` 前缀）与 `evaluate(results)`——后者通过解析 `StoredResults` 中的 `r()`、`e()`、`_b[]`、`_se[]` 查表，把每个名字映射为浮点数。它是冻结 dataclass：解析产生一个值，而非可变构建器。

**签名**

```python
ws.ParsedExpressionList(groups: 'tuple[StatisticGroup, ...]') -> None
```

**示例**

```python
ws.parse_exp_list('(mean=r(mean)) (sd=r(sd))').names
```

**输出**

```text
['mean', 'sd']
```

### `WaldTest`

**Stata 指令:** `test x1 = 0`

对线性假设 `Rb = r` 的单次 Wald 检验，按 Stata 的方式渲染。字段：`statistic`（卡方推断时为 `W`，F 推断时为 `W/q`）、`df`（分子的 `q`）、`df_denom`（F 形式的分母，卡方形式为 `None`）、`pvalue`、`distribution`（`"F"` 或 `"chi2"`）与 `constraint`（被检验内容的可读说明，打印在统计量上方）。采用 F 还是卡方形式取决于模型的 `dist`，因此 logit 拟合给卡方、OLS 拟合给 F。

**签名**

```python
ws.WaldTest(statistic: 'float', df: 'float', df_denom: 'float | None', pvalue: 'float', distribution: 'str', constraint: 'str') -> None
```

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.test('x1 = 0')
```

**输出**

```text

 ( 1)  x1 = 0

       F(  1,   197) =  134.55
            Prob > F =    0.0000
```

**注意:** `to_dict()` 给出 Stata 的 `r()` 名称：统计量以其自身分布的名字出现（`F` 或 `chi2`），因此在卡方检验之后读 `r(F)` 会找不到——这是对的，因为 Stata 在那里存的是 `r(chi2)`。

### `LincomResult`

**Stata 指令:** `lincom x1 - x2`

参数的一个线性组合，形态与 Stata 的 `lincom` 输出一致。字段：`estimate`、`std_err`、`statistic`、`pvalue`、`ci_lower`、`ci_upper`、`distribution`（`"t"` 或 `"z"`）、`df`（基于 z 的拟合为 `None`，它确实没有自由度）、`expression`（所写假设），以及 `eform` 与 `label`。标准误由 `weights @ b` 上的 delta 方法给出，使用与系数表相同的协方差。

**签名**

```python
ws.LincomResult(estimate: 'float', std_err: 'float', statistic: 'float', pvalue: 'float', ci_lower: 'float', ci_upper: 'float', distribution: 'str', df: 'float | None', expression: 'str', eform: 'bool' = False, label: 'str' = '', response_name: 'str' = '') -> None
```

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.lincom('x1 - x2')
```

**输出**

```text

 ( 1)  x1 - x2 = 0

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         (1) |   1.352732   .1039679    13.01   0.000     1.147699    1.557764
------------------------------------------------------------------------------
```

**注意:** 对基于 z 的拟合，`to_dict()` 省略 `df`，而不是报 0 或 None，因为 Stata 在这种情况下不存 `r(df)`；统计量相应地以 `t` 或 `z` 出现。

### `IcResult`

**Stata 指令:** `estat ic`

单个已拟合模型的 `estat ic` 表。字段：`n`、`ll_null`、`ll_model`、`df`（前四列，对应 Stata 的 `r(S)` 布局）、四个准则 `aic`、`bic`、`aicc`、`caic`，以及 `model`（`Model` 列中的命令名）。缺失的统计量按 Stata 的方式打印为 `.` 而非 `nan`。对自身不存信息准则的 `regress` 来说，这才是正确的入口。

**签名**

```python
ws.IcResult(*, n: 'float', ll_null: 'float', ll_model: 'float', df: 'float', aic: 'float', bic: 'float', aicc: 'float', caic: 'float', model: 'str' = '') -> 'None'
```

**示例**

```python
ws.estat_ic(ws.OLS.from_formula('y ~ x1 + x2', base).fit())
```

**输出**

```text

Akaike's information criterion and Bayesian information criterion

-----------------------------------------------------------------------------
       Model |          N   ll(null)  ll(model)      df        AIC        BIC
-------------+---------------------------------------------------------------
           . |        200  -357.2327  -291.7997       3   589.5994   599.4944
-----------------------------------------------------------------------------
Note: BIC uses N = number of observations. See [R] IC note.
```

### `VifResult`

**Stata 指令:** `estat vif`

线性拟合的方差膨胀因子表。字段：`names`（按报告顺序排列的回归元，除非 `nosort=True`，否则按 VIF 降序排序）、数组形式的 `vif` 与其倒数 `tolerance`，以及 `mean_vif`。`to_frame()` 返回 DataFrame 形式的表，`to_dict()` 给出以变量名为键的 Stata `r(vif_#)` 标量。仅对线性族有定义：辅助回归按构造就是普通最小二乘。

**签名**

```python
ws.VifResult(names: 'Sequence[str]', vif: 'Sequence[float]', mean_vif: 'float') -> 'None'
```

**示例**

```python
ws.estat_vif(ws.OLS.from_formula('y ~ x1 + x2', base).fit())
```

**输出**

```text

    Variable |       VIF       1/VIF  
-------------+----------------------
          x1 |      1.00    0.999986
          x2 |      1.00    0.999986
-------------+----------------------
    Mean VIF |      1.00
```

### `MarginsResult`

**Stata 指令:** `margins, dydx(x1)`

`margins` 表的一行及其协方差。字段：`label`（行标签）、`estimate`、`std_err`、`statistic`、`pvalue`、`ci_lower`、`ci_upper` 与 `prediction`（所用的预测类型）。标准误是 delta 方法值：响应族有闭式解处用解析梯度，其余用自适应数值梯度。`to_dict()` 展平为 Stata `r(table)` 的行标签。

**签名**

```python
ws.MarginsResult(label: 'str', estimate: 'float', std_err: 'float', statistic: 'float', pvalue: 'float', ci_lower: 'float', ci_upper: 'float', prediction: 'str' = '', title: 'str' = '', nobs: 'float' = nan, vce_label: 'str' = '', expression: 'str' = '', variables: 'tuple[str, ...]' = (), column_label: 'str' = '', stat_label: 'str' = 'z', at_means: 'tuple[tuple[str, float], ...]' = ()) -> None
```

**示例**

```python
ws.margins(ws.Logit.from_formula('yb ~ x1 + x2', base).fit(), dydx='x1')
```

**输出**

```text

Average marginal effects                                   Number of obs = 200
Model VCE: OIM

Expression: Pr(yb), predict()
dy/dx wrt:  x1

------------------------------------------------------------------------------
             |            Delta-method
             |      dy/dx   std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .1226231   .0311214     3.94   0.000     .0616263    .1836198
------------------------------------------------------------------------------
```

**注意:** `to_dict()` 一律把统计量行标为 `t`，因为该对象不携带分布标志；logit 之后 Stata 会标为 `z`。当标签本身重要时请用 `str(...)`。

### `GofResult`

**Stata 指令:** `estat gof (after poisson)`

Poisson 拟合优度表：一个偏差（似然比）检验与一个 Pearson 检验，各带上尾卡方 p 值。字段：`deviance`、`deviance_pvalue`、`pearson`、`pearson_pvalue`、`df`（`e(N) - e(df_m) - 1`，两者共用）。Stata 仅对 `poisson` 定义 `estat gof`；`nbreg` 不设 `e(estat_cmd)`，因此在那里请求会抛 `NotImplementedError`，而不会返回编造的统计量。

**签名**

```python
ws.GofResult(*, deviance: 'float', deviance_pvalue: 'float', pearson: 'float', pearson_pvalue: 'float', df: 'float') -> 'None'
```

**示例**

```python
ws.estat_gof(ws.Poisson.from_formula('ycnt ~ x1', base).fit())
```

**输出**

```text

Deviance goodness-of-fit           =   227.201
Prob > chi2(198)                   =    0.0758

Pearson goodness-of-fit            =   187.551
Prob > chi2(198)                   =    0.6918
```

### `ClassificationResult`

**Stata 指令:** `estat classification`

`estat classification` 的 2x2 表及其九个百分比。字段：`table`（3x3 DataFrame，行为 `+`/`-`/`Total`，列为 `D`/`~D`/`Total`）、`cutoff`（概率阈值），以及 Stata 以 `r(P_*)` 返回的各项比率：`percent_correct`、`sensitivity`、`specificity`、`false_positive_rate`、`false_negative_rate`、`positive_predictive_value`、`negative_predictive_value`、`false_positive_classified`、`false_negative_classified`。注意两类“错误率”的条件不同——一类以真实类别为条件，另一类以判定的类别为条件。

**签名**

```python
ws.ClassificationResult(*, table: 'pd.DataFrame', cutoff: 'float', percent_correct: 'float', sensitivity: 'float', specificity: 'float', false_positive_rate: 'float', false_negative_rate: 'float', positive_predictive_value: 'float', negative_predictive_value: 'float', false_positive_classified: 'float', false_negative_classified: 'float', model_label: 'str' = '', depvar: 'str' = '') -> 'None'
```

**示例**

```python
r = ws.estat_classification(ws.Logit.from_formula('yb ~ x1 + x2', base).fit())
(r.cutoff, r.percent_correct)
```

**输出**

```text
(0.5, 62.0)
```

### `HetTestResult`

**Stata 指令:** `estat hettest`

Breusch-Pagan / Cook-Weisberg 异方差检验。字段：`method`（`"normal"`、`"iid"` 或 `"fstat"`）、`statistic`、`df`、`pvalue`、`df_denom`（仅 `method="fstat"` 时）与 `variables`（辅助回归所用的回归元——为空表示使用了拟合值，这是未给变量列表时 Stata 的默认）。与 Stata 一致，在稳健或聚类 VCE 之后以及 pweight 数据上它会拒绝。

**签名**

```python
ws.HetTestResult(*, method: 'str', statistic: 'float', df: 'float', pvalue: 'float', df_denom: 'float | None' = None, variables: 'Sequence[str]' = (), label: 'str' = '', plural: 'bool' = False) -> 'None'
```

**示例**

```python
ws.estat_hettest(ws.OLS.from_formula('y ~ x1 + x2', base).fit())
```

**输出**

```text

Breusch–Pagan/Cook–Weisberg test for heteroskedasticity 
Assumption: Normal error terms
Variable: Fitted values of y

H0: Constant variance

    chi2(1) =   0.27
Prob > chi2 = 0.6006
```

### `ImTestResult`

**Stata 指令:** `estat imtest`

Cameron & Trivedi 的信息矩阵检验分解，可选附带 White 原始检验。字段：`heteroskedasticity`、`skewness`、`kurtosis` 及其自由度 `df_heteroskedasticity`、`df_skewness`、`df_kurtosis`；`total` 与 `df_total`（三者之和）；以及 `white`、`white_df`、`white_pvalue`——除非请求了 `white=True`，否则为 `None`。

**签名**

```python
ws.ImTestResult(*, heteroskedasticity: 'float', skewness: 'float', kurtosis: 'float', df_heteroskedasticity: 'float', df_skewness: 'float', df_kurtosis: 'float', white: 'float | None' = None, white_df: 'float | None' = None) -> 'None'
```

**示例**

```python
ws.estat_imtest(ws.OLS.from_formula('y ~ x1 + x2', base).fit())
```

**输出**

```text

Cameron & Trivedi's decomposition of IM-test

--------------------------------------------------
              Source |       chi2     df         p
---------------------+----------------------------
  Heteroskedasticity |       0.85      5    0.9737
            Skewness |       1.19      2    0.5526
            Kurtosis |       2.04      1    0.1535
---------------------+----------------------------
               Total |       4.07      8    0.8503
--------------------------------------------------
```

### `OvTestResult`

**Stata 指令:** `estat ovtest`

Ramsey 的 RESET 遗漏变量检验。字段：`statistic`（F 统计量）、`df` 与 `df_denom`（分子与分母自由度）、`pvalue`，以及 `omitted`（对被取幂对象的可读描述，与 Stata 的 `Omitted:` 行一致）。默认路径是 Stata 的 `freset`，未给 `rhs` 时它就走这条路径。

**签名**

```python
ws.OvTestResult(*, statistic: 'float', df: 'float', df_denom: 'float', pvalue: 'float', omitted: 'str', dropped: 'Sequence[str]' = ()) -> 'None'
```

**示例**

```python
ws.estat_ovtest(ws.OLS.from_formula('y ~ x1 + x2', base).fit())
```

**输出**

```text

Ramsey RESET test for omitted variables
Omitted: Powers of fitted values of y

H0: Model has no omitted variables

F(3, 194) =   0.58
 Prob > F = 0.6277
```

### `EndogenousResult`

**Stata 指令:** `estat endogenous`

IV 拟合的 Durbin 与 Wu 内生性检验。字段：`durbin`、`durbin_pvalue`（Durbin 卡方，`df` 等于被检验的内生回归元个数）、`wu`、`wu_pvalue`、`wu_df`、`wu_df_denom`（Wu 的 F 及其两个自由度），以及 `tested`（所覆盖的内生回归元名称）。拒绝意味着该回归元内生，因此同方程上的 OLS 会不一致。

**签名**

```python
ws.EndogenousResult(*, durbin: 'float', durbin_pvalue: 'float', wu: 'float', wu_pvalue: 'float', wu_df: 'float', wu_df_denom: 'float', tested) -> 'None'
```

**示例**

```python
iv = ws.IV2SLS.from_formula('y ~ x1 + [x2 ~ x3]', base).fit()
ws.estat_endogenous(iv)
```

**输出**

```text

  Tests of endogeneity
  H0: Variables are exogenous

  Durbin (score) chi2(1)          =  1.64391  (p = 0.1998)
  Wu-Hausman F(1,196)             =  1.62438  (p = 0.2040)
```

### `FirstStageResult`

**Stata 指令:** `estat firststage`

IV 拟合的第一阶段诊断：每个内生回归元一行，外加一个总体 F。字段：`endog`（按报告顺序的名称）、`r2` 与 `adj_r2`（内生回归元对全部外生变量回归）、`partial_r2`（残差化后的内生回归元对残差化后的排除工具变量回归）、`f`、`df_num`、`df_denom`、`pvalue`（排除工具变量的联合检验），以及 `shea`、`shea_adj`（仅在内生回归元多于一个时计算，因为只有一个时它们与 `partial_r2` 相同）。

**签名**

```python
ws.FirstStageResult(*, endog: 'Sequence[str]', r2: 'Sequence[float]', adj_r2: 'Sequence[float]', partial_r2: 'Sequence[float]', f: 'Sequence[float]', df_num: 'Sequence[float]', df_denom: 'Sequence[float]', pvalue: 'Sequence[float]', shea: 'Sequence[float] | None' = None, shea_adj: 'Sequence[float] | None' = None, mineig: 'float | None' = None, mineigcv: 'Any' = None, show_all: 'bool' = False, force_nonrobust: 'bool' = False, robust_vce: 'bool' = False) -> 'None'
```

**示例**

```python
iv = ws.IV2SLS.from_formula('y ~ x1 + [x2 ~ x3]', base).fit()
ws.estat_firststage(iv)
```

**输出**

```text

  First-stage regression summary statistics
  --------------------------------------------------------------------------
               |            Adjusted      Partial
      Variable |   R-sq.       R-sq.        R-sq.      F(1,197)   Prob > F
  -------------+------------------------------------------------------------
            x2 |  0.0008     -0.0093       0.0008      0.159849    0.6897
  --------------------------------------------------------------------------


  Minimum eigenvalue statistic = 0.159849    

  Critical Values                      # of endogenous regressors:    1
  H0: Instruments are weak             # of excluded instruments:     1
  ---------------------------------------------------------------------
                                     |    5%     10%     20%     30%
  2SLS relative bias                 |         (not available)
  -----------------------------------+---------------------------------
                                     |   10%     15%     20%     25%
  2SLS size of nominal 5% Wald test  |  16.38    8.96    6.66    5.53
  LIML size of nominal 5% Wald test  |  16.38    8.96    6.66    5.53
  ---------------------------------------------------------------------
```

### `OverIdResult`

**Stata 指令:** `estat overid`

IV 拟合的过度识别约束检验。字段：`statistic`、`df`（过度识别约束个数）与 `pvalue`；`basmann` 与 `basmann_pvalue`，即有限样本变体 `Sargan*(N-k)/(N-Sargan)`，对 GMM 为 `None`；以及 `test`，为 `"Sargan"`（2SLS 与 LIML）或 `"Hansen J"`（GMM）。恰好识别的模型没有可检验的约束，会被拒绝，而不是报成统计量为零。

**签名**

```python
ws.OverIdResult(*, statistic: 'float', df: 'float', pvalue: 'float', test: 'str', basmann: 'float | None' = None, basmann_pvalue: 'float | None' = None) -> 'None'
```

**示例**

```python
iv = ws.IV2SLS.from_formula('y ~ x1 + [x2 ~ x3 + y1]', base).fit()
ws.estat_overid(iv)
```

**输出**

```text

  Tests of overidentifying restrictions:

  Sargan (score) chi2(1) = .00297725  (p = 0.9565)
  Basmann chi2(1)        = .00291775  (p = 0.9569)
```

### `LRTestResult`

**Stata 指令:** `lrtest unrestricted restricted`

受限模型与非受限模型之间的似然比检验。字段：`statistic`（`2 (lnL_unrestricted - lnL_restricted)`）、`df`（参数个数之差，始终为正，因为参数个数相同的两个模型会按 Stata 的 `r(498)` 被拒绝）、`pvalue`（`P > chi2(df)`，统计量为负时按 Stata 的打印恰为 1.0），以及 `ll_unrestricted`、`ll_restricted`——保留它们以便无需重新拟合即可审计该数值。

**签名**

```python
ws.LRTestResult(statistic: 'float', df: 'float', pvalue: 'float', ll_unrestricted: 'float', ll_restricted: 'float', restricted_name: 'str' = '', unrestricted_name: 'str' = '') -> None
```

**示例**

```python
u = ws.Logit.from_formula('yb ~ x1 + x2', base).fit()
r = ws.Logit.from_formula('yb ~ x1', base).fit()
u.lrtest(r)
```

**输出**

```text
Likelihood-ratio test
Assumption: logit nested within logit
LR chi2(1) = 0.79
Prob > chi2 = 0.3731
```

### `HausmanResult`

**Stata 指令:** `hausman fe re`

Hausman 设定检验：`(b_c - b_e)' [V_c - V_e]^-1 (b_c - b_e)`。字段：`statistic`、`df`（实际参与比较的系数个数，不含常数项）、`pvalue`（`P > chi2(df)`），以及 `excluded`（两个模型中不同时存在、因而不可比较的参数——这是本包增加的诊断，不是 `r()` 标量）。拒绝意味着有效估计量不一致，应改用一致估计量。

**签名**

```python
ws.HausmanResult(statistic: 'float', df: 'float', pvalue: 'float', excluded: 'list[str]' = <factory>, names: 'list[str]' = <factory>, consistent: 'np.ndarray | None' = None, efficient: 'np.ndarray | None' = None, standard_errors: 'np.ndarray | None' = None, consistent_label: 'str' = '', efficient_label: 'str' = '', consistent_command: 'str' = '', efficient_command: 'str' = '', tested_count: 'int | None' = None, not_positive_definite: 'bool' = False) -> None
```

**示例**

```python
fe = ws.PanelOLS.from_formula('ypanel ~ x1', base, entity='firm', time='year').fit()
re = ws.RandomEffects.from_formula('ypanel ~ x1', base, entity='firm', time='year').fit()
fe.hausman(re)
```

**输出**

```text

                 ---- Coefficients ----
             |      (b)          (B)            (b-B)     sqrt(diag(V_b-V_B))
             |       fe           re         Difference       Std. err.
-------------+----------------------------------------------------------------
          x1 |     .709774     .7287794       -.0190054        .0157648
------------------------------------------------------------------------------
                          b = Consistent under H0 and Ha; obtained from xtreg.
           B = Inconsistent under Ha, efficient under H0; obtained from xtreg.

Test of H0: Difference in coefficients not systematic

    chi2(1) = (b-B)'[(V_b-V_B)^(-1)](b-B)
            =   1.45
Prob > chi2 = 0.2280
```

**注意:** 差矩阵 `V_c - V_e` 在有限样本中可能奇异或出现负特征值；该检验只比较两个模型共有的系数，`df` 数的就是这些。


## 其他名称

### `Areg`

**Stata 指令:** `areg y x1 x2, absorb(firm)`

吸收类别指示变量的线性回归 —— Stata 的 `areg`。它与 `reghdfe` 是**两个不同的命令**，不是同义词：`absorb=` 是必填项，标题栏为「Linear regression, absorbing indicators」并在第二行注明被吸收的变量，`e(predict)` 是 `areg_p`，而且——实质差别在这里——**它报告截距**。`reghdfe` 只报斜率，所以 `Areg` 会追加 `_cons`，即被吸收组效应的**按组容量加权**均值（不是等权均值：两者只在平衡面板上巧合相等）。估计量本身是同一个组内变换，因此系数与 `FEOLS` 一致；不同的是命令身份、常数项和小样本口径。`fit` 接受 `level=` 控制打印的置信区间，并支持 `vce(hc2)`/`vce(hc3)`——因为它自己构造了完整哑变量设计，能算出 `FEOLS` 拿不到的杠杆值。

**签名**

```python
ws.Areg(formula: 'str', data: 'pd.DataFrame', *, absorb: 'Any' = None, weights: 'Any' = None, weight_type: 'str | None' = None, noabsorb: 'bool' = False) -> 'None'
```

**`Areg.from_formula`**

```python
Areg.from_formula(formula: 'str', data: 'pd.DataFrame', *, absorb: 'Any' = None, weights: 'Any' = None, weight_type: 'str | None' = None, subset: 'Any' = None, noabsorb: 'bool' = False) -> "'Areg'"
```

**`Areg.fit`**

```python
Areg.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'SSC | None' = None, level: 'float | None' = None) -> 'StataResults'
```

**示例**

```python
ws.Areg.from_formula('y ~ x1', base, absorb='firm').fit().summary(style='stata')
```

**输出**

```text
Linear regression, absorbing indicators            Number of obs     =     200
Absorbed variable: firm                            No. of categories =      40
                                                   F(1, 159)         =   86.12
                                                   Prob > F          =  0.0000
                                                   R-squared         =  0.4928
                                                   Adj R-squared     =  0.3652
                                                   Root MSE          =  1.1531

------------------------------------------------------------------------------
           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .8095565   .0872367     9.28   0.000     .6372645    .9818486
       _cons |   .9742472   .0816927    11.93   0.000     .8129043     1.13559
F test of absorbed indicators: F(39, 159) = 1.095       Prob > F = 0.340
```

**注意:** `absorb=` 是必填项——不给它就等同于 `regress`。`_cons` 是被吸收组效应的**按组容量加权**均值；固定效应模型无法把常数与组效应分开识别，只有归一化能定住它，这里采用的是 areg 的归一化。聚类协方差的 CR1 因子用 `K = k_reported + (k_absorb - 1)`，而 `reghdfe` 用 `k_reported`：在 areg 的口径里被吸收的水平也消耗自由度。这个差别**在聚类维度与吸收维度的不同配对下并不一致**，所以只有 `e(rank)` 显示哑变量尚未计入时才施加修正。多路聚类时两个命令的因子是**结构不同**、不是差一个参数计数，因此在完整设计上重算，而不是缩放。`noabsorb=True` 跳过被吸收指示变量的 F 检验（Stata 的 `noabsorbtest`），此时 `e(F_absorb)`/`e(p_absorb)` 不存在。

### `MarginsTable`

**Stata 指令:** `margins (table)`

当 `margins` 调用产生多行时返回的多行容器。它是 `list` 的子类，因此可以像 `MarginsResult` 列表一样迭代与索引；其 `__str__` 渲染 Stata 的单张表格（一个标题、一套横线、下方若干行）。只产生一行的调用则返回裸的 `MarginsResult`。

**签名**

```python
ws.MarginsTable(iterable=(), /)
```

**示例**

```python
print(ws.margins(ws.OLS.from_formula('y ~ x1 + x2', base).fit(), dydx=['x1','x2']))
```

**输出**

```text
None
```

### `MultinomialLogit`

**Stata 指令:** `mlogit depvar indepvars, base(#)`

多项 logit 回归。Stata 为每个结果类别存一组系数，**基准类别**那组为零但仍打印；statsmodels 只估计 `levels - 1` 组，所以这里把基准行**插入**而不是丢弃——丢弃会让后续每个系数整体错位一组，而结果看起来仍然合理。二值结果会被拒绝并提示改用 `Logit`。

**签名**

```python
ws.MultinomialLogit(endog: 'Any', exog: 'Any', *, param_names: 'list[str] | None' = None, freq_weights: 'Any' = None) -> 'None'
```

**`MultinomialLogit.from_formula`**

```python
MultinomialLogit.from_formula(formula: 'str', data: 'pd.DataFrame', *, subset: 'Any' = None, freq_weights: 'str | None' = None) -> "'_DiscreteBase'"
```

**`MultinomialLogit.fit`**

```python
MultinomialLogit.fit(self, cov_type: 'str | None' = None, cov_kwds: 'Any' = None, *, maxiter: 'int' = 200, **kwargs: 'Any') -> 'Any'
```

**示例**

```python
lv = (base['ycnt'] > base['ycnt'].median()).astype(int) + (base['ycnt'] > base['ycnt'].quantile(0.75)).astype(int)
f = ws.MultinomialLogit(lv.to_numpy(), np.column_stack([base['x1'], base['x2'], np.ones(len(base))]), param_names=['x1','x2','_cons']).fit()
np.round(f.e['ll'], 4)
```

**输出**

```text
-181.4397
```

### `PraisResult`

**Stata 指令:** `prais depvar indepvars`

Prais-Winsten AR(1) 估计的结果。除常规接口外，它还记录估计出的自相关系数 `rho`、迭代次数以及是否收敛，因为该估计量是迭代的，由 `maxiter`/`tolerance` 控制。

**签名**

```python
ws.PraisResult(**values: 'object') -> 'None'
```

**示例**

```python
r = ws.prais(base, 'y ~ x1')
print(round(r.rho, 6))
```

**输出**

```text
None
```

### `RandomEffectsMLE`

**Stata 指令:** `xtreg ypanel x1 x2, mle`

用**极大似然**估计的随机效应回归（Stata `xtreg, mle`）。`RandomEffects` 是代入 Swamy-Arora 方差成分的 GLS `xtreg, re`，而本类在同一随机效应似然中对系数与两个方差成分**同时**求极大，因此 `sigma_u` 会明显不同。底层用 statsmodels 的 `MixedLM` 以 ML（非 REML）配合 Powell 优化器拟合；标准误来自观测信息阵之逆。返回 `StataResults`；稳健或聚类 VCE 会被拒绝，因为 Stata 的 `mle` 也不接受它们。

**签名**

```python
ws.RandomEffectsMLE(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', weights: 'Any' = None, weight_type: 'str | None' = None) -> 'None'
```

**`RandomEffectsMLE.from_formula`**

```python
RandomEffectsMLE.from_formula(formula: 'str', data: 'pd.DataFrame', *, entity: 'str', time: 'str', subset: 'Any' = None, weights: 'Any' = None, weight_type: 'str | None' = None) -> "'PanelOLS'"
```

**`RandomEffectsMLE.fit`**

```python
RandomEffectsMLE.fit(self, entity_effects: 'bool' = False, time_effects: 'bool' = False, cov_type: 'str | None' = None, cov_kwds: 'Mapping[str, Any] | None' = None, ssc: 'SSC | None' = None) -> 'StataResults'
```

**示例**

```python
ws.RandomEffectsMLE.from_formula('ypanel ~ x1 + x2', base, entity='firm', time='year').fit().summary(style='stata')
```

**输出**

```text

Random-effects ML regression                         Number of obs    =    200
Group variable: firm                                 Number of groups =     40

Random effects u_i ~ Gaussian                        Obs per group:
                                                                  min =      .
                                                                  avg =      .
                                                                  max =      .

                                                     LR chi2(2)       =  82.25
Log likelihood = -308.62872                          Prob > chi2      = 0.0000

------------------------------------------------------------------------------
      ypanel | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .7282049    .072845    10.00   0.000     .5854313    .8709784
          x2 |  -.1199841   .0780419    -1.54   0.124    -.2729434    .0329752
       _cons |   .8439937   .1404197     6.01   0.000     .5687762    1.119211
------------------------------------------------------------------------------
-------------+----------------------------------------------------------------
    /sigma_u |  .77026643
    /sigma_e |  .98414177
         rho |  .37987779
------------------------------------------------------------------------------
LR test of sigma_u=0: chibar2(01) = 39.08             Prob >= chibar2 = 0.0000
```

### `SummarizeResult`

**Stata 指令:** `estat summarize -- summarize of the estimation sample`

`estat summarize` 表格：对估计样本（而非整个数据框）的逐变量摘要。字段：`table`（以变量为索引、含 `Obs`、`Mean`、`Std. Dev.`、`Min`、`Max` 的 DataFrame）、`n_obs`（估计样本量）、`labels`、`clusters` 与 `noheader`。由于它摘要的是拟合实际使用的行，列出删除后其 `Obs` 列可能小于 `len(base)`。

**签名**

```python
ws.SummarizeResult(table: 'pd.DataFrame', n_obs: 'float', clusters: 'Mapping[str, Any] | None' = None, noheader: 'bool' = False, labels: 'Mapping[str, str] | None' = None) -> 'None'
```

**示例**

```python
ws.estat_summarize(ws.OLS.from_formula('y ~ x1 + x2', base).fit()).to_frame()
```

**输出**

```text
            Obs      Mean  Std. Dev.       Min       Max
Variable                                                
y         200.0  1.020667   1.447346 -2.752204  4.881985
x1        200.0  0.057340   1.025868 -2.832134  3.278208
x2        200.0  0.063540   0.993631 -2.226171  3.259538
```

### `Synth2Result`

**Stata 指令:** `ereturn list -- the e() surface synth2 stores (synth's e() is wiped)`

保存 `synth2` 结果的冻结 dataclass（`synth` 的 e() 已被清空）。字段逐字对应 Stata 的 `e()`：标量 `N/T/T0/T1/K/J/rmse/r2/att`，矩阵 `bal`（K x 6）、`U_wt`、`V_wt`、`mspe`（仅空间安慰剂）、`pval`（仅空间安慰剂），轨迹 `Y_treated`/`Y_synthetic`/`time_index` 及 `gap` 属性，安慰剂字段（`placebo_probability`、`placebo_removed`、`mspe_names`、`time_placebo`），留一包络（`loo_min`/`loo_max`/`loo_effect_min`/`loo_effect_max`/`loo_n_runs`），以及审计字段（`synth_result`、`method`、`numerically_inactive`）。`summary()` 渲染紧凑的 Stata 风格报表。它不是 `StataResults`：不伪造 `params`、`bse`、`cov_params`、`df_resid`。`bal` 的 6 列为 `Vweight, Treated, ValueSyntheticControl, BiasSyntheticControl, ValueAverageControl, BiasAverageControl`；`Bias% = (Value/Treated - 1)*100`（ado:181）。

**签名**

```python
ws.Synth2Result(N: 'int', T: 'int', T0: 'int', T1: 'int', K: 'int', J: 'int', rmse: 'float', r2: 'float', att: 'float', bal: 'np.ndarray', U_wt: 'np.ndarray', V_wt: 'np.ndarray', mspe: 'np.ndarray | None' = None, pval: 'np.ndarray | None' = None, Y_treated: 'np.ndarray | None' = None, Y_synthetic: 'np.ndarray | None' = None, time_index: 'np.ndarray | None' = None, placebo_probability: 'float | None' = None, placebo_removed: 'tuple[Any, ...]' = (), mspe_names: 'tuple[str, ...]' = (), time_placebo: 'dict[float, dict[str, np.ndarray]]' = <factory>, loo_min: 'np.ndarray | None' = None, loo_max: 'np.ndarray | None' = None, loo_effect_min: 'np.ndarray | None' = None, loo_effect_max: 'np.ndarray | None' = None, loo_n_runs: 'int' = 0, loo_units: 'tuple[Any, ...]' = (), synth_result: 'SynthResult | None' = None, method: 'str' = 'regression', numerically_inactive: 'tuple[str, ...]' = ()) -> None
```

**示例**

```python
units = [1, 2, 3, 4, 5]
years = np.arange(1997, 2006)
u = np.repeat(units, len(years))
t = np.tile(years, len(units))
rng = np.random.default_rng(0)
x1 = 0.5 * u + rng.normal(size=len(u))
x2 = 0.2 * u + rng.normal(size=len(u))
y = 1.0 + 0.8 * x1 - 0.4 * x2 + 0.1 * (t - 1997) + 5.0 * (u == 1) * (t >= 2005) + rng.normal(size=len(u))
panel = pd.DataFrame({'unit': u, 'year': t, 'x1': x1, 'x2': x2, 'y': y})
res = ws.synth2(panel, unit='unit', time='year', depvar='y', predictors=['x1', 'x2'], trunit=1, trperiod=2005)
res.bal
```

**输出**

```text
[[  0.757511   0.851994   1.094422  28.454142   1.70489  100.105787]
 [  0.242489   0.870713   0.700712 -19.524351   0.678917 -22.02742 ]]
```

### `SynthResult`

**Stata 指令:** `ereturn list -- the e() matrices synth stores`

承载 `synth` 结果的冻结数据类。字段名逐字对应 Stata 的 `e()` 矩阵：`V_matrix`（K x K 对角预测变量权重——注意没有普通的 `e(V)`）、`W_weights`、`X_balance`、`RMSPE`、`Y_treated`、`Y_synthetic`，都可通过 `e` 属性一并取得。`gap` 是处理效应路径 `Y_treated - Y_synthetic`，`summary()` 给出紧凑的 Stata 风格报告，`W_exact` 保留未舍入的 QP 解供诊断。它**不是** `StataResults`，因此没有 `params`、`bse` 或 `cov_params`。

**签名**

```python
ws.SynthResult(V_matrix: 'np.ndarray', W_weights: 'np.ndarray', X_balance: 'np.ndarray', RMSPE: 'float', Y_treated: 'np.ndarray', Y_synthetic: 'np.ndarray', unit_names: 'tuple[Any, ...]', predictor_names: 'tuple[str, ...]', time_index: 'np.ndarray', treated_unit: 'Any', treatment_period: 'float', T0: 'int', W_exact: 'np.ndarray', missing_warnings: 'tuple[str, ...]' = (), method: 'str' = 'regression', optimizer_settings: 'dict[str, float | int] | None' = None, qp_solver: 'str' = 'SLSQP', unit_labels: 'tuple[str, ...]' = (), unit_codes: 'tuple[Any, ...]' = (), numerically_inactive: 'tuple[str, ...]' = (), stata_predictor_names: 'tuple[str, ...]' = ()) -> None
```

**示例**

```python
units = [1, 2, 3, 4, 5]
years = np.arange(1997, 2006)
u = np.repeat(units, len(years))
t = np.tile(years, len(units))
rng = np.random.default_rng(0)
x1 = 0.5 * u + rng.normal(size=len(u))
x2 = 0.2 * u + rng.normal(size=len(u))
y = 1.0 + 0.8 * x1 - 0.4 * x2 + 0.1 * (t - 1997) + 5.0 * (u == 1) * (t >= 2005) + rng.normal(size=len(u))
panel = pd.DataFrame({'unit': u, 'year': t, 'x1': x1, 'x2': x2, 'y': y})
res = ws.synth(panel, unit='unit', time='year', depvar='y', predictors=['x1', 'x2'], trunit=1, trperiod=2005)
res.V_matrix
```

**输出**

```text
[[0.757511 0.      ]
 [0.       0.242489]]
```

### `dfgls`

**Stata 指令:** `dfgls varname`

`ws.dfgls` 是 DF-GLS（Elliott-Rothenberg-Stock）单位根检验：经 GLS 去趋势的 Dickey-Fuller，在接近单位根时比普通 ADF 有更好的功效。报告的统计量与临界值来自 ERS 表，而非 Dickey-Fuller 表。

**签名**

```python
ws.dfgls(series: 'Any', *, maxlag: 'int | None' = None, trend: 'bool' = True, notrend: 'bool' = False, name: 'str' = '') -> 'DFGLSResult'
```

**示例**

```python
print(ws.dfgls(base['y'].to_numpy()))
```

**输出**

```text
None
```

### `dfuller`

**Stata 指令:** `dfuller y, lags(2)`

增广 Dickey-Fuller 单位根检验，采用 Stata 的拼写。`unitroot` 的薄封装：`dfuller(y, trend=True)` 等价于 `unitroot(y, test='adf', regression='ct')`，不带选项时 `regression='c'`，显式给 `regression=` 会覆盖 `trend`。`lags=` 对应 Stata 的 `lags(#)`（默认零阶滞后）。返回 `UnitRootResult`，打印形式为 Stata 的 Dickey-Fuller 表，统计量存于 `r(Zt)`，MacKinnon p 值存于 `r(p)`。

**签名**

```python
ws.dfuller(endog: 'Any', *, trend: 'bool' = False, lags: 'int | None' = None, autolag: 'str | None' = None, regression: 'str | None' = None) -> 'UnitRootResult'
```

**示例**

```python
ws.dfuller(base['y'], lags=2)
```

**输出**

```text

Augmented Dickey–Fuller test for unit root
Variable: y                               Number of obs  = 197
                                          Number of lags =   2

H0: Random walk without drift, d = 0

                                       Dickey–Fuller
                   Test      -------- critical value ---------
              statistic           1%           5%          10%
--------------------------------------------------------------
 Z(t)            -6.706       -3.478       -2.884       -2.574
--------------------------------------------------------------
MacKinnon approximate p-value for Z(t) = 0.0000.
```

### `ereturn_list`

**Stata 指令:** `ereturn list`

单个估计结果对应的 Stata `ereturn list` 四个区块：标量写作 `e(name) = value`，宏写作 `e(name) : "text"`，矩阵写作 `e(name) : r x c`，函数只列名字。某命令公布哪些标量由它自己决定，而 Stata 的选择比完整 `e()` 字典更窄，因此打印的是忠实子集而非全量转储。整段以空行开头，与 Stata 一致。

**签名**

```python
ws.ereturn_list(self) -> 'str'
```

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.ereturn_list()
```

**输出**

```text

scalars:
                  e(N) =  200
              e(sum_w) =  200
               e(df_m) =  2
               e(df_r) =  197
                  e(F) =  90.99954130641439
                 e(r2) =  .4802098236178376
               e(rmse) =  1.048768791147598
                e(mss) =  200.1836988170982
                e(rss) =  216.6834475251831
               e(r2_a) =  .4749327659895922
                 e(ll) =  -291.7997221200006
               e(ll_0) =  -357.2327276994888
               e(rank) =  3

macros:
              e(title) : "Linear regression"
          e(marginsok) : "XB default"
                e(vce) : "ols"
             e(depvar) : "y"
                e(cmd) : "regress"
         e(properties) : "b V"
            e(predict) : "regres_p"
              e(model) : "ols"
          e(estat_cmd) : "regress_estat"

matrices:
                  e(b) :  1 x 3
                  e(V) :  3 x 3
               e(beta) :  1 x 2

functions:
             e(sample)
```

### `estat_bgodfrey`

**Stata 指令:** `estat bgodfrey, lags(#)`

Breusch-Godfrey 的 `lags` 阶序列相关 LM 检验。该检验把残差对**原始回归元加上滞后残差**回归；包含原始回归元才能保证有效性，省掉它们就变成了另一个（无效的）检验。滞后取自**时间轴**而非行轴：传 `time=`（估计样本的列名，或期数值数组）即可复现 `tsset` 语义——某个期在样本里缺失时该行的滞后是缺失值，而不是把缺口两侧的行接起来。实测 `icecream.dta` 删去 `time == 5` 后，Stata 为 4.220，按行序则为 3.286。缺失的滞后默认补 0，`nomiss0` 则保持缺失。稳健或聚类 VCE 之后、跨多个单位的面板数据上、以及非 `regress` 拟合上都会拒绝，与 Stata 的 r(198)/r(459)/r(321) 一致。

**签名**

```python
ws.estat_bgodfrey(result: 'Any', *, lags: 'int | Sequence[int]' = 1, nomiss0: 'bool' = False, time: 'str | Sequence[Any] | None' = None, delta: 'float' = 1.0) -> 'BgTestResult'
```

**示例**

```python
print(ws.estat_bgodfrey(ws.OLS.from_formula('y ~ x1 + x2', base).fit(), lags=2))
```

**输出**

```text
None
```

### `estat_corrgram`

**Stata 指令:** `estat corrgram`

相关图：各滞后阶的自相关、偏自相关、Ljung-Box Q 统计量及其 p 值。默认阶数是 `max(1, min(int(n/2) - 2, 40))`（`corrgram.ado:18`）。偏自相关是残差对其前 k 阶滞后**带常数项**回归中第 k 阶的系数（`pac.ado:128-156`），不是 Yule-Walker 递推。`time=` 提供 `tsset` 时间轴，缺口处滞后即为缺失；Stata 此时打印 `(note: time series has 1 gap)`，本包一致。无法估计的偏自相关打印 `.` 且不画条形。

**签名**

```python
ws.estat_corrgram(result: 'Any', *, lags: 'int | None' = None, time: 'str | Any | None' = None, delta: 'float' = 1.0) -> 'Correlogram'
```

**示例**

```python
print(ws.estat_corrgram(ws.OLS.from_formula('y ~ x1 + x2', base).fit(), lags=4))
```

**输出**

```text
None
```

### `estat_dwatson`

**Stata 指令:** `estat dwatson`

残差一阶自相关的 Durbin-Watson d 统计量。Stata 把它打在一行里，形如 `Durbin–Watson d-statistic(  k,    N) = value`，标签里是 **en-dash（–）**，两个计数带填充；只给出数值不足以与打印输出比对。`time=` 即 `tsset` 的替代：跨缺口的那一对观测被跳过而不是相减，并先打印一行 `Number of gaps in sample = N`。实测 `icecream.dta` 删去 `time == 5` 后：`time=` 下为 .8930395，按行序为 1.0868969。

**签名**

```python
ws.estat_dwatson(result: 'Any', *, time: 'str | Any | None' = None, delta: 'float' = 1.0) -> 'DurbinWatsonResult'
```

**示例**

```python
print(ws.estat_dwatson(ws.OLS.from_formula('y ~ x1 + x2', base).fit()))
```

**输出**

```text
None
```

### `etable_as_stata`

**Stata 指令:** `esttab m1 m2, se r2 mtitle`

把 `etable` 按 Stata `esttab` 的打印形式渲染出来。它是 `render_esttab` 的便捷封装，转发 `model_names`、`digits`、`se`、`stars` 与 `stats`；`etable` 本身返回数据框（对程序更有用），而本函数返回与 Stata 对比所需的精确文本。两者刻意并存——互不替代。

**签名**

```python
ws.etable_as_stata(results: 'Sequence[Any]', *, model_names: 'Sequence[str] | None' = None, digits: 'int' = 3, se: 'bool' = True, stars: 'bool' = True, stats: 'Sequence[tuple[str, str]] | None' = None) -> 'str'
```

**示例**

```python
m1 = ws.OLS.from_formula('y ~ x1', base).fit()
m2 = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
ws.etable_as_stata([m1, m2], model_names=['m1', 'm2'], stats=[('N', 'nobs'), ('R-sq', 'r2')])
```

**输出**

```text

--------------------------------------------
                      (1)             (2)   
                       m1              m2   
                        y               y   
--------------------------------------------
x1                  0.843***        0.841***
                 (0.0804)        (0.0725)   

x2                                 -0.512***
                                 (0.0748)   

_cons               0.972***        1.005***
                 (0.0824)        (0.0744)   
--------------------------------------------
N                     200             200   
R-sq                0.357           0.480   
--------------------------------------------
Standard errors in parentheses
* p<0.05, ** p<0.01, *** p<0.001
```

### `f_test`

**Stata 指令:** `test x1 = 0, x2 = 0 (joint F)`

已拟合结果上的 statsmodels 风格联合 F 检验。`r_matrix` 接受一个或多个以逗号分隔的约束字符串（或一个序列），按与 `test` 相同的 Stata 约束语法解析后，以 F 统计量联合报告。返回 `ContrastResults`，含 `statistic`、`pvalue`、`df_num`/`df_denom` 以及 `fvalue`/`tvalue` 别名。这是面向 statsmodels 生态的接口，并非 Stata 自身的 `test` 方法。

**签名**

```python
ws.f_test(self, r_matrix: 'Any', cov_p: 'Any' = None, invcov: 'Any' = None) -> 'Any'
```

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.f_test('x1 = 0, x2 = 0')
```

**输出**

```text
<F contrast: statistic=90.9995, pvalue=1.02109e-28, df_denom=197>
```

### `first_stage`

**Stata 指令:** `ivregress 2sls ..., first`

IV 拟合的第一阶段回归表（Stata `ivregress ..., first`）。每个内生回归元一段，以「First-stage regressions」开头，给出该方程的 F、R 方、调整 R 方与 Root MSE，随后是含外生变量与排除工具的系数表。这与 `estat_firststage` 的输出不同——后者打印同一批方程的**诊断量**；Stata 两者都提供且互不替代，因此这里也都保留。

**签名**

```python
ws.first_stage(self) -> 'str'
```

**示例**

```python
r = ws.IV2SLS.from_formula('ytreat ~ x1 + [treat ~ x3]', base).fit()
r.first_stage()
```

**输出**

```text
First-stage regressions
-----------------------

                                                        Number of obs =    200
                                                        F(2, 197)     =  36.02
                                                        Prob > F      = 0.0000
                                                        R-squared     = 0.2678
                                                        Adj R-squared = 0.2603
                                                        Root MSE      = 0.4139

------------------------------------------------------------------------------
       treat | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
          x1 |   .0022204   .0291298     0.08   0.939    -.0552258    .0596667
          x3 |   .2398927   .0287406     8.35   0.000     .1832141    .2965714
       _cons |   .6613358   .0293962    22.50   0.000     .6033641    .7193075
------------------------------------------------------------------------------
```

### `fitted_series`

**Stata 指令:** `predict yhat, xb -- with the estimation sample's index`

以 `pandas.Series` 给出的拟合值，带估计样本索引、名称为 `fittedvalues`。它是普通 `fittedvalues` ndarray 的带标签版本：若把后者直接换成 Series 会破坏按位置访问，因此另设此名。当拟合未保留观测标签时，索引退化为 `RangeIndex`——这是诚实的做法：数值仍在，标签只表示「第 i 行」。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.fitted_series.head()
```

**输出**

```text
0    2.524914
1    0.144945
2    1.126616
3    2.107762
4    0.869349
Name: fittedvalues, dtype: float64
```

### `format_ereturn_list`

**Stata 指令:** `ereturn list`

Stata `ereturn list` 的四个块：scalars、macros、matrices、functions——顺序固定，每块后跟一个空行。标量与宏用**不同的分隔符**（` =  ` 两个空格 对 一个空格加冒号），两块按**类型**划分：字符串进 macros、数字进 scalars，因此以后新增的宏不会被错误归档。标量值打印至多 16 位有效数字，无前导零、无尾随零。

**签名**

```python
ws.format_ereturn_list(result: 'Any') -> 'str'
```

**示例**

```python
ws.format_ereturn_list(ws.OLS.from_formula('y ~ x1', base).fit())
```

**输出**

```text

scalars:
                  e(N) =  200
              e(sum_w) =  200
               e(df_m) =  1
               e(df_r) =  198
                  e(F) =  109.7490736414687
                 e(r2) =  .3566186969885982
               e(rmse) =  1.163859236421547
                e(mss) =  148.6626185459396
                e(rss) =  268.2045277963416
               e(r2_a) =  .3533692964683386
                 e(ll) =  -313.1309552730224
               e(ll_0) =  -357.2327276994888
               e(rank) =  2

macros:
              e(title) : "Linear regression"
          e(marginsok) : "XB default"
                e(vce) : "ols"
             e(depvar) : "y"
                e(cmd) : "regress"
         e(properties) : "b V"
            e(predict) : "regres_p"
              e(model) : "ols"
          e(estat_cmd) : "regress_estat"

matrices:
                  e(b) :  1 x 2
                  e(V) :  2 x 2
               e(beta) :  1 x 1

functions:
             e(sample)
```

### `get_influence`

**Stata 指令:** `(no Stata equivalent -- statsmodels OLSInfluence names)`

已拟合结果的影响度量，命名沿用 statsmodels 的 `OLSInfluence`。返回字典，含 `hat_matrix_diag`（杠杆）、`resid`、`resid_studentized_internal`、`resid_studentized`、`standardized_resid`、`cooks_distance`（数组加观测索引）与 `nobs`。只提供基于杠杆的度量：删除诊断需要逐观测重新拟合，返回「看起来合理但错误」的数组被认为比不提供该键更糟。

**签名**

```python
ws.get_influence(self) -> 'Any'
```

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.get_influence()['hat_matrix_diag'][:5]
```

**输出**

```text
[0.024179 0.010089 0.005392 0.011287 0.005195]
```

### `kpss`

**Stata 指令:** `kpss varname`

KPSS 平稳性检验——其原假设与旁边的单位根检验**相反**：`H0: 序列（趋势）平稳`。`trend=True` 围绕确定性趋势检验，`False` 围绕常数检验。p 值应读作**反对**平稳性的证据，这一点在紧邻 `dfuller` 使用时容易搞反。

**签名**

```python
ws.kpss(values: 'Any', *, trend: 'bool' = True, maxlag: 'int | None' = None, name: 'str' = '') -> 'KpssResult'
```

**示例**

```python
print(ws.kpss(base['y'].to_numpy()))
```

**输出**

```text
None
```

### `level`

**Stata 指令:** `regress ..., level(90)`

`conf_int()` 与打印区间使用的置信水平，以**分数**表示（Stata 的 `level(90)` 对应 `0.90`），默认 `0.95`。用 `level=` 拟合的模型会记录它，因此以 90% 估计的结果默认打印90% 区间，除非调用者显式传入 `alpha`。Stata 的 `level(#)` 收的是**百分数**；各估计器两种写法都接受并归一化——因为把 `0.90` 当百分数解读会悄悄产生 0.9% 的区间，而症状只是「区间看起来太窄」。

**示例**

```python
ws.OLS.from_formula('y ~ x1', base).fit(level=90).conf_int()
```

**输出**

```text
              lower     upper
Intercept  0.836140  1.108573
x1         0.709618  0.975431
```

**注意:** 该属性返回分数；`conf_int()` 仍接受显式 `alpha`，那一次调用里它会覆盖已记录的水平。它只影响区间的报告，**不改变任何系数或标准误**。

### `model`

**Stata 指令:** `(no Stata equivalent -- a statsmodels interop shim)`

以 statsmodels 消费者期待的形式给出的已拟合模型。`statsmodels.stats.anova.anova_lm` 会读取 `res.model` 并由其重新拟合；缺少该属性时该调用会抛 AttributeError。返回的 `ResultModel` 只暴露 `endog`、`exog`、`exog_names` 以及返回本结果自身的 `fit()`——足以满足 `anova_lm` 的契约，而不冒充真正的 statsmodels 模型。过度模仿会诱使调用方依赖并不存在的行为。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.model
```

**输出**

```text
<ResultModel: 200 obs, 3 params>
```

### `odds_ratios`

**Stata 指令:** `logit ..., or  (column, not a separate function)`

这不是一个导出的函数。`logit ..., or` 通过 `result.summary(style="stata", odds_ratio=True)` 表达：对估计值、其标准误和两个区间端点取指数，并把列名改为 `Odds ratio`；z 与 P>|z| **不变**，因为它们由原始尺度计算。

**示例**

```python
print(ws.Logit.from_formula('yb ~ x1 + x2', base).fit().summary(style='stata', odds_ratio=True)[-400:])
```

**输出**

```text
None
```

### `pperron`

**Stata 指令:** `pperron y, trend`

Phillips-Perron 单位根检验，采用 Stata 的拼写。`unitroot` 的薄封装：`pperron(y, trend=True)` 等价于 `unitroot(y, test='pp', regression='ct')`，选项映射与 `dfuller` 相同。返回 `UnitRootResult`；当 `lags=None` 时，Newey-West 滞后截断采用与 `dfuller` 相同的默认值。

**签名**

```python
ws.pperron(endog: 'Any', *, trend: 'bool' = False, lags: 'int | None' = None, autolag: 'str | None' = None, regression: 'str | None' = None) -> 'UnitRootResult'
```

**示例**

```python
ws.pperron(base['y'], trend=True)
```

**输出**

```text

Augmented Dickey–Fuller test for unit root
Variable: y                               Number of obs  = 199
                                          Number of lags =   4

H0: Random walk with or without drift

                                       Dickey–Fuller
                   Test      -------- critical value ---------
              statistic           1%           5%          10%
--------------------------------------------------------------
 Z(t)           -14.483       -4.007       -3.437       -3.137
--------------------------------------------------------------
MacKinnon approximate p-value for Z(t) = 0.0000.
```

### `prais`

**Stata 指令:** `prais depvar indepvars`

Prais-Winsten AR(1) 回归：用估计出的自相关系数变换数据后做 GLS，并迭代到 `rho` 稳定。该变换**保留第一个观测**（这正是 Prais-Winsten 与丢弃首观测的 Cochrane-Orcutt 的区别），因此样本量是完整的。

**签名**

```python
ws.prais(frame: 'pd.DataFrame', formula: 'str', *, maxiter: 'int' = 100, tolerance: 'float' = 1e-06, corc: 'bool' = False) -> 'PraisResult'
```

**示例**

```python
print(str(ws.prais(base, 'y ~ x1'))[:180])
```

**输出**

```text
None
```

### `rcm`

**Stata 指令:** `rcm depvar indepvars, trunit(#) trperiod(#)`

单个处理单元的回归控制法：用对照单元的结果变量拟合处理单元的处理前期结果，再对处理后期取平均。`selected=` 直接指定对照单元，选中同一集合时与 Stata 逐位一致。**前向搜索**并非 Stata 所用的同一算法——需要可复现结果请传 `selected=`。

**签名**

```python
ws.rcm(response: 'Any', predictors: 'Any', unit: 'Any', time: 'Any', *, treated_unit: 'Any', treatment_period: 'float', method: 'str' = 'forward', criterion: 'str' = 'aicc', selected: 'Sequence[Any] | None' = None, max_predictors: 'int | None' = None) -> 'RcmResult'
```

**示例**

```python
d = base.assign(unit=np.repeat(np.arange(20), 10), t=np.tile(np.arange(10), 20))
r = ws.rcm(d['y'], d[['x1']], d['unit'], d['t'], treated_unit=0, treatment_period=6, selected=[1, 2])
round(r.att, 6)
```

**输出**

```text
-0.596186
```

**注意:** 自动模型搜索**未**与 Stata 对齐。Stata 自身的选择在前期拟合上远差于穷举最优，因为它的回归设定与这里任何重建都不同；有一条测试显式断言这一不匹配，避免两条路径被误认为等价。

### `rdrobust`

**Stata 指令:** `rdrobust depvar runvar, c(#)`

尖锐断点回归估计：切点两侧各做一个局部多项式拟合，在切点处相减。**点估计精确**（在真实数据上验证到九位有效数字）。必须显式给出 `bandwidth`——Stata 的 MSE 最优带宽选择器未实现，所以 `None` 会报错而不是静默选一个。

**签名**

```python
ws.rdrobust(response: 'Any', running: 'Any', *, cutoff: 'float' = 0.0, bandwidth: 'float | None' = None, bias_bandwidth: 'float | None' = None, order: 'int' = 1, bias_order: 'int' = 2, kernel: 'str' = 'triangular', covs: 'Any' = None) -> 'RdRobustResult'
```

**示例**

```python
x = base['x1'].to_numpy()
y = 2.0 * (x > 0) + 0.5 * x + base['x2'].to_numpy() * 0.1
round(ws.rdrobust(y, x, cutoff=0.0, bandwidth=0.8).tau, 4)
```

**输出**

```text
1.9415
```

**注意:** 未复现稳健（CCT）标准误：`rdrobust_vce` 只存在于编译后的 lrdrobust.mlib 中，因此 se/z/ci 为 NaN。其外围的组装步骤已用 Stata 自身的 e(V_cl_l)/e(V_cl_r) 验证过。

### `render_estimates_table`

**Stata 指令:** `estimates table m1 m2 ..., b se`

把任意多个已拟合模型并排，采用 Stata `estimates table` 的布局——它与 `esttab` 是不同的形态：没有列号、没有依赖变量行，末尾以 `Legend: b/se` 收尾。每个模型一列，系数一行、标准误在下一行。

**签名**

```python
ws.render_estimates_table(results: 'Sequence[Any]', *, model_names: 'Sequence[str] | None' = None, se: 'bool' = True, star: 'Sequence[float] | None' = None, mtitle: 'bool' = False, stats: 'Sequence[str] | None' = None) -> 'str'
```

**示例**

```python
ws.render_estimates_table([ws.OLS.from_formula('y ~ x1', base).fit(), ws.OLS.from_formula('y ~ x1 + x2', base).fit()], model_names=['a', 'b'])
```

**输出**

```text

----------------------------------------
    Variable |     a            b       
-------------+--------------------------
          x1 |  .84252495    .84064454  
             |  .08042338    .07247108  
          x2 |              -.51208701  
             |                .0748223  
       _cons |  .97235675    1.0050025  
             |  .08242638    .07442849  
----------------------------------------
                            Legend: b/se
```

**注意:** 几何是推导出来的：14 字符标签槽，之后每个模型 13 列，规则线为 14 + 13*n 个字符、'+' 位于索引 13。名称居中，超过十个字符则缩写。数值在 |v|≥1 时用八位有效数字，否则八位小数。注意 VCE 约定：Stata 的 `regress, r` 是 HC1，`ivregress ..., r` 是 HC0；混合 OLS 与 IV 的表格需要对 OLS 用 cov_type='HC1'、对 IV 用 'HC0'。

### `render_esttab`

**Stata 指令:** `esttab m1 m2, se r2 mtitle`

把一个或多个已拟合结果渲染为 Stata `esttab` 打印的**文本**：`(1) (2)` 编号行、给出 `model_names` 时的 `mtitle` 行、带括号内标准误与显著性星号的系数行，以及统计量页脚。`stats` 是 `(标签, 来源)` 对序列，默认给出来自 `nobs` 的 `N`；`stats_digits` 控制页脚小数位。与返回数据框给程序用的 `etable` 不同，本函数返回可打印字符串。实测的 44 列版式最多容纳两个模型。

**签名**

```python
ws.render_esttab(results: 'Sequence[Any]', *, model_names: 'Sequence[str] | None', digits: 'int' = 3, se: 'bool' = True, stars: 'bool' = True, stats: 'Sequence[tuple[str, Any]] | None' = None, stats_digits: 'int' = 3) -> 'str'
```

**示例**

```python
m1 = ws.OLS.from_formula('y ~ x1', base).fit()
m2 = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
ws.render_esttab([m1, m2], model_names=['m1', 'm2'], stats=[('N', 'nobs'), ('R-sq', 'r2')])
```

**输出**

```text

--------------------------------------------
                      (1)             (2)   
                       m1              m2   
                        y               y   
--------------------------------------------
x1                  0.843***        0.841***
                 (0.0804)        (0.0725)   

x2                                 -0.512***
                                 (0.0748)   

_cons               0.972***        1.005***
                 (0.0824)        (0.0744)   
--------------------------------------------
N                     200             200   
R-sq                0.357           0.480   
--------------------------------------------
Standard errors in parentheses
* p<0.05, ** p<0.01, *** p<0.001
```

### `resid_series`

**Stata 指令:** `predict e, residuals -- with the estimation sample's index`

以 `pandas.Series` 给出的残差，带估计样本索引、名称为 `resid`，是普通 `resid` ndarray 的带标签版本。它为兼容 statsmodels 而设，同时不改动 ndarray 的按位置访问行为；拟合未保留行标签时索引退化为 `RangeIndex`。对应的拟合值见 `fitted_series`。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.resid_series.head()
```

**输出**

```text
0   -0.245449
1   -0.012850
2   -1.598986
3    0.354059
4    0.682509
Name: resid, dtype: float64
```

### `response_name`

**Stata 指令:** `e(depvar)`

被解释变量名，用于标题与列头。有的估计量把它设为实例属性，有的不设（ARIMA 与 GARCH 属于后者），因此该属性回退到 `e(depvar)`——每个拟合都会携带的宏——从而与基于公式的估计量给出相同答案。子类若赋值 `self.response_name` 仍然优先，因为先查实例字典。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.response_name
```

**输出**

```text
y
```

### `stata_cov_params`

**Stata 指令:** `e(V)`

按与 `stata_params` 相同的顺序重排后的协方差矩阵，为 numpy 数组。对含义不明确的因子项，它抛出与名称访问相同的拒绝，因此调用者不会拿到一个行序与 Stata 静默不一致的矩阵。比较对象是 Stata 的 `e(V)` 时请用它而非 `cov`。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
np.round(r.stata_cov_params, 4)
```

**输出**

```text
[[ 0.0053  0.     -0.0003]
 [ 0.      0.0056 -0.0004]
 [-0.0003 -0.0004  0.0055]]
```

### `stata_factor_terms`

**Stata 指令:** `i.<var> -- the explicit factor expansion in a varlist`

无法重建其 Stata `e(b)` 拼写的因子项，为排序后的列表；当每个系数都有明确的 Stata 拼写时为空。Stata 把显式 `i.foreign` 展开为每个水平一行再加一行显式零基准（`0b.foreign`），而本包只有一个水平列，故不存在一一映射。裸二元列不会出现在此，因为 Stata 用变量名为其单独一行命名。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + C(yb)', base).fit()
r.stata_factor_terms
```

**输出**

```text
['yb']
```

**注意:** 此处非空意味着该结果上的 `stata_param_names`、`stata_params` 与 `stata_cov_params` 都会抛 `NotImplementedError`；请改用 `params`/`bse` 对比，或把因子编码为哑变量后重新拟合。

### `stata_param_names`

**Stata 指令:** `the row names of e(b) as Stata spells and orders them`

按 Stata 的 `e(b)` 顺序与拼写给出的系数名：先是回归元，然后 `_cons` 在最后，最后是 `cut1` 或 `lnalpha` 等辅助参数。`Intercept` 变为 `_cons`，裸因子列取变量名而省略水平，`:` 变 `#`，前导 `/` 被去掉。对显式 `C(...)` 因子会抛 `NotImplementedError`，因为 Stata 为每个水平给一行再加一行零基准，不存在一一映射。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
(r.param_names, r.stata_param_names)
```

**输出**

```text
(['Intercept', 'x1', 'x2'], ['x1', 'x2', '_cons'])
```

### `stata_params`

**Stata 指令:** `e(b)`

重排为 Stata `e(b)` 顺序与拼写的系数向量：`_cons` 移到末尾而非开头，因子与交互项名称重写。它是属性而非方法——写 `r.stata_params`，不要写 `r.stata_params()`。数值与 `params` 完全相同，只是顺序与标签不同，与 Stata fixture 对比时应用此视图。

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.stata_params
```

**输出**

```text
x1       0.840645
x2      -0.512087
_cons    1.005003
Name: params, dtype: float64
```

### `summarize`

**Stata 指令:** `summarize varlist`

Stata 原生 `summarize` 表：每个变量的非缺失数、均值、标准差、最小值与最大值，一行 71 个字符。这是**六列**形式；`estat_summarize` 是估计样本变体（无 Obs 列、表头不同），两者不能互相替代。数值按 `%9.0g` 打印，所以同一列中 `106.2666` 与 `29.61636` 的小数位不同。`weights` 切换为频数权重，此时 Obs 单元格按 Stata 的做法报告权重和。

**签名**

```python
ws.summarize(data: 'Any', variables: 'Sequence[str] | None' = None, *, weights: 'Any' = None, detail: 'bool' = False) -> 'Any'
```

**示例**

```python
ws.summarize(base, ['y', 'x1'])
```

**输出**

```text

    Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
           y |        200    1.020667    1.447346  -2.752204   4.881985
          x1 |        200    .0573402    1.025868  -2.832134   3.278208
```

### `synth`

**Stata 指令:** `synth y x1 x2, trunit(1) trperiod(2005)`

合成控制估计（Stata `synth`，默认基于回归的 V 路径）。用若干控制单元构造加权组合，使其在干预前的预测变量均值上与处理单元匹配，再以之作为反事实。`unit`/`time`/`depvar` 指定面板列，`predictors` 为匹配变量，`trunit`/`trperiod` 为处理单元与干预时点。返回冻结的 `SynthResult` 数据类；由于 `synth` 没有系数向量、没有 `e(V)`、也没有自由度，这些 statsmodels 风格属性被刻意省略而不伪造。示例内联构造一个 5 单元 x 9 期的小平衡面板，因此不依赖任何外部 .dta 文件。

**签名**

```python
ws.synth(data: 'Any', *, unit: 'str', time: 'str', depvar: 'str', predictors: 'Sequence[str]', trunit: 'Any', trperiod: 'float', counit: 'Sequence[Any] | None' = None, xperiod: 'Sequence[float] | None' = None, mspeperiod: 'Sequence[float] | None' = None, resultsperiod: 'Sequence[float] | None' = None, customV: 'Sequence[float] | None' = None, unitnames: 'str | None' = None, margin: 'float' = 0.005, maxiter: 'int' = 1000, sigf: 'int' = 12, bound: 'float' = 10, nested: 'bool' = False, allopt: 'bool' = False, technique: 'str | None' = None, iterate: 'int | None' = None, nrtolerance: 'float | None' = None, nonrtolerance: 'bool' = False, difficult: 'bool' = False, search: 'str | None' = None, trace: 'bool' = False, keep: 'str | None' = None, replace: 'bool' = False, figure: 'bool' = False, placebo: 'bool' = False, **unsupported: 'Any') -> 'SynthResult'
```

**示例**

```python
units = [1, 2, 3, 4, 5]
years = np.arange(1997, 2006)
u = np.repeat(units, len(years))
t = np.tile(years, len(units))
rng = np.random.default_rng(0)
x1 = 0.5 * u + rng.normal(size=len(u))
x2 = 0.2 * u + rng.normal(size=len(u))
y = 1.0 + 0.8 * x1 - 0.4 * x2 + 0.1 * (t - 1997) + 5.0 * (u == 1) * (t >= 2005) + rng.normal(size=len(u))
panel = pd.DataFrame({'unit': u, 'year': t, 'x1': x1, 'x2': x2, 'y': y})
ws.synth(panel, unit='unit', time='year', depvar='y', predictors=['x1', 'x2'], trunit=1, trperiod=2005).RMSPE
```

**输出**

```text
1.3651043054505119
```

**注意:** 仅实现默认路径：`nested`/`allopt`、`figure`、`keep()`、`placebo` 以及任何推断量都会抛出 `UnsupportedError`。`customV` 提供对角预测变量权重；`xperiod`/`mspeperiod`/`resultsperiod` 覆盖预处理窗口。输出的 `W_weights` 按 Stata 做法四舍五入到 3 位小数，而 `RMSPE` 使用精确 QP 权重。

### `synth2`

**Stata 指令:** `synth2 y x1 x2, trunit(1) trperiod(2005)`

Stata `synth2` 的非图形功能实现。`synth2` 是包装器：全部优化委托给 `ws.synth`，自身增加拟合统计量（`rmse`/`r2`/`att`）、6 列协变量平衡表（`bal`）、非零单位权重（`U_wt`）、V 对角矩阵（`V_wt`）、空间安慰剂（假处理单位 -> `mspe` + 逐期双侧/右侧/左侧 p 值 `pval`，可选 `cut()`）、时间安慰剂（`placebo('period(2001)')`，仅重估、无推断）以及留一法逐期 min/max 包络（`loo=True`）。选项与 Stata 对应：`counit`、`preperiod`、`postperiod`、`xperiod`、`mspeperiod`、`customV`、`nested`、`allopt` 及四个优化器参数。返回冻结的 `Synth2Result`。纯图形选项（`figure`/`nofigure`/`savegraph`/`show`）与 Stata `frame()`/`symbol()` 明确拒绝/不实现，全部结果由结果对象属性承载。证据等级按量定为 E2：公式层（bal/U_wt/V_wt）达机器精度，重跑链路继承 `ws.synth` 秩亏 QP 的差异。

**签名**

```python
ws.synth2(data: 'Any', *, unit: 'str', time: 'str', depvar: 'str', predictors: 'Sequence[str]', trunit: 'Any', trperiod: 'float', counit: 'Sequence[Any] | None' = None, ctrlunit: 'Sequence[Any] | None' = None, preperiod: 'Sequence[float] | None' = None, postperiod: 'Sequence[float] | None' = None, xperiod: 'Sequence[float] | None' = None, mspeperiod: 'Sequence[float] | None' = None, customV: 'Sequence[float] | None' = None, nested: 'bool' = False, allopt: 'bool' = False, margin: 'float' = 0.005, maxiter: 'int' = 1000, sigf: 'int' = 12, bound: 'float' = 10, placebo: 'str | None' = None, loo: 'bool' = False, figure: 'bool' = False, nofigure: 'bool' = False, savegraph: 'str | None' = None, symbol: 'int' = 1, frame: 'str | None' = None, show: 'int | None' = None, **unsupported: 'Any') -> 'Synth2Result'
```

**示例**

```python
units = [1, 2, 3, 4, 5]
years = np.arange(1997, 2006)
u = np.repeat(units, len(years))
t = np.tile(years, len(units))
rng = np.random.default_rng(0)
x1 = 0.5 * u + rng.normal(size=len(u))
x2 = 0.2 * u + rng.normal(size=len(u))
y = 1.0 + 0.8 * x1 - 0.4 * x2 + 0.1 * (t - 1997) + 5.0 * (u == 1) * (t >= 2005) + rng.normal(size=len(u))
panel = pd.DataFrame({'unit': u, 'year': t, 'x1': x1, 'x2': x2, 'y': y})
ws.synth2(panel, unit='unit', time='year', depvar='y', predictors=['x1', 'x2'], trunit=1, trperiod=2005).rmse
```

**输出**

```text
1.3651043054505119
```

**注意:** `synth2.ado:330` 用 `ereturn clear` 清空 `synth` 的 e()，Python 结果面是 synth2 自己的，另附 `synth_result` 保留原始拟合。`R2` 分母是合成值的处理前离差（ado:647，非标准）。聚合概率与逐期 p 值把处理组自身计入分母且用 >= 比较（ado:441/940/947-949）。时间安慰剂不产生 mspe/pval。`frame()`/`symbol()` 是 Stata 特有的结果数据集机制，由 Synth2Result 属性替代。

### `t_test`

**Stata 指令:** `test x1 = 0 (single restriction, t form)`

单个线性约束的 statsmodels 风格 t 检验。`r_matrix` 是一个约束字符串（或单元素序列）；`use_t` 强制采用 t 形式，其统计量为相应 F 的平方根。返回 `ContrastResults`，含 `statistic`、`pvalue`、单约束可用的 `conf_int()` 以及 `fvalue`/`tvalue` 别名。若要检验多个联合约束，请改用 `f_test`。

**签名**

```python
ws.t_test(self, r_matrix: 'Any', cov_p: 'Any' = None, use_t: 'bool | None' = None) -> 'Any'
```

**示例**

```python
r = ws.OLS.from_formula('y ~ x1 + x2', base).fit()
r.t_test('x1 = 0')
```

**输出**

```text
<t contrast: statistic=11.5997, pvalue=4.75584e-24, df_denom=197>
```

### `tebalance_summarize`

**Stata 指令:** `tebalance summarize`

匹配后的协变量平衡性：每个变量的标准化差与方差比（原始与匹配后），以及四个观测计数。公式取自 `_tebalance_summarize.ado`：标准化差为 `(m1 - m0) / sqrt((v1 + v0)/2)`，方差比为 `v1 / v0`，匹配列用匹配权重重新加权对照组。把处理/对照权重作为 `match_weights` 传入。

**签名**

```python
ws.tebalance_summarize(treatment: 'Any', covariates: 'Any', *, match_weights: 'Any' = None, names: 'list[str] | None' = None, exact: 'tuple[str, ...]' = (), normalise_counts: 'bool' = False) -> 'BalanceSummary'
```

**示例**

```python
ps = np.asarray(ws.Logit.from_formula('treat ~ x1 + x2', base).fit().predict())
c = np.flatnonzero(base['treat'].to_numpy() == 0)
w = np.zeros(len(base))
[w.__setitem__(c[np.abs(ps[i] - ps[c]).argmin()], 1.0) for i in np.flatnonzero(base['treat'].to_numpy() == 1)]
ws.tebalance_summarize(base['treat'].to_numpy(), base[['x1','x2']].to_numpy(), match_weights=w, names=['x1','x2'])
```

**输出**

```text
Covariate balance summary

                         Raw      Matched
-----------------------------------------
Number of obs =          200          179
Treated obs   =          128          128
Control obs   =           72           51
-----------------------------------------

-----------------------------------------------------------------
                |Standardized differences          Variance ratio
                |        Raw     Matched           Raw    Matched
----------------+------------------------------------------------
             x1 |  -.1960176   -.0278721      1.036638   1.192021
             x2 |  -.3042354   -.2882414      .7707397   .8606013
-----------------------------------------------------------------
```

**注意:** 在参考的 nsw_dw 拟合上，八个协变量中有七个与 Stata 逐位一致；第八个（`age`）因并列权重的分配方式略有不同而有残差，该差异被记录而非隐藏。

### `teffects_nnmatch`

**Stata 指令:** `teffects nnmatch ..., atet`

在马氏距离（Stata 默认度量，`e(metric)` 为 "mahalanobis"）上做最近邻匹配的 ATET，使用样本协方差矩阵的逆。并列规则与 `teffects_psmatch` 相同：第 k 小距离上的所有对照都被使用并取平均。`caliper` 限定最大距离；处理组个体若在 caliper 内没有对照则报错，而不是借用远处的对照，这正是 Stata 的 "no matches" 情形。

**签名**

```python
ws.teffects_nnmatch(response: 'Any', treatment: 'Any', covariates: 'Any', *, nneighbor: 'int' = 1, caliper: 'float | None' = None, tie_tolerance: 'float' = 1e-10) -> 'NnmatchResult'
```

**示例**

```python
ws.teffects_nnmatch(base['ytreat'], base['treat'], base[['x1','x2','x3']].to_numpy(), nneighbor=4)
```

**输出**

```text
NnmatchResult(ate=1.4150728330730742, nobs=200, treated=128, controls=72, nneighbor=4, matches_min=4, matches_max=4, metric='mahalanobis', caliper=nan, se=nan, z=nan, pvalue=nan, ci_lower=nan, ci_upper=nan)
```

### `teffects_psmatch`

**Stata 指令:** `teffects psmatch ..., atet`

`teffects_psmatch` 做倾向得分最近邻匹配的 ATET。倾向得分来自处理变量对协变量的 logit（Stata 默认的处理模型）。并列（ties）是关键：第 k 小距离上的**所有**对照都算匹配，而不只取一个，这正是与 Stata 吻合的原因。参考数据上 `nneighbor=1` 的并列数可达 8、`nneighbor=4` 可达 11，这些计数会一并报告。

**签名**

```python
ws.teffects_psmatch(response: 'Any', treatment: 'Any', covariates: 'Any', *, nneighbor: 'int' = 1, tie_tolerance: 'float' = 1e-09, outcome: 'str' = '', caliper: 'float | None' = None) -> 'PsmatchResult'
```

**示例**

```python
ws.teffects_psmatch(base['ytreat'], base['treat'], base[['x1','x2','x3']].to_numpy(), nneighbor=1)
```

**输出**

```text

Treatment-effects estimation                   Number of obs      =        200
Estimator      : propensity-score matching     Matches: requested =          1
Outcome model  : matching                                     min =          1
Treatment model: logit                                        max =          1
------------------------------------------------------------------------------
             |              AI robust
             | Coefficient  std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
ATET         |
       treat |
   (1 vs 0)  |      1.852          .        .       .            .           .
------------------------------------------------------------------------------
```

**注意:** 未复现 Abadie-Imbens 的 AI 稳健标准误，因此 se/z/p/ci 为 NaN，而不是一个看似合理的猜测值。

### `teffects_ra`

**Stata 指令:** `teffects ra (depvar covars) (treat)`

回归调整法处理效应：在处理组与对照组分别拟合结果模型，再对估计量的总体平均预测差（`ate` 对所有个体、`atet` 对处理组）。点估计精确；标准误未复现，因为 sandwich 由 Stata 内置 `gmm` 计算，因此 se/z/p/ci 为 NaN。

**签名**

```python
ws.teffects_ra(response: 'Any', treatment: 'Any', covariates: 'Any', *, estimand: 'str' = 'ate', add_constant: 'bool' = True) -> 'TefectsRaResult'
```

**示例**

```python
print(ws.teffects_ra(base['y'] + base['yb'], base['yb'], base[['x1','x2']].to_numpy()).ate)
```

**输出**

```text
None
```

### `vargranger`

**Stata 指令:** `vargranger`

对已拟合 VAR 的格兰杰因果 Wald 检验，输出 Stata 的方框表：每个方程标题下逐个列出被排除变量，再加一行 ALL 做联合检验，每行 70 字符。统计量来自 `granger_causality` 所做的同一个 Wald 检验；注意 `granger_causality(a, b)` 表示**方程 a 排除 b**——参数名叫 `causing` 但装的是方程名，写反会静默对调两个 chi2 值。

**签名**

```python
ws.vargranger(fitted: 'Any') -> 'VarGrangerResult'
```

**示例**

```python
ws.vargranger(ws.VAR(base[['y1','y2']].to_numpy(), lags=1, names=['y1','y2']).fit())
```

**输出**

```text
   Granger causality Wald tests
  +------------------------------------------------------------------+
  |          Equation           Excluded |   chi2     df Prob > chi2 |
  |--------------------------------------+---------------------------|
  |                y1                 y2 | .025404     1    0.873    |
  |                y1                ALL | .025404     1    0.873    |
  |--------------------------------------+---------------------------|
  |                y2                 y1 |  .55296     1    0.457    |
  |                y2                ALL |  .55296     1    0.457    |
  +------------------------------------------------------------------+
```

**注意:** 即使只排除一个变量，每个方程块仍以 ALL 行结尾；Stata 就是这样打印的，该行与单行完全相同。

### `varlmar`

**Stata 指令:** `varlmar`

已拟合 VAR 的残差自相关拉格朗日乘数检验：对 1 到 `mlag` 的每一阶，检验残差在该阶是否存在自相关。原假设为该阶无自相关。

**签名**

```python
ws.varlmar(result: 'Any', *, mlag: 'int' = 2) -> 'LmarResult'
```

**示例**

```python
print(ws.varlmar(ws.VAR(base[['y1','y2']].to_numpy(), lags=1, names=['y1','y2']).fit(), mlag=1))
```

**输出**

```text
None
```

### `varsoc`

**Stata 指令:** `varsoc varlist, maxlag(#)`

`ws.varsoc` 给出 VAR 的滞后阶选择准则：对每个候选阶数给出最终预测误差、AIC、HQIC 与 SBIC，并按各准则分别给最小者加星。四个准则经常不一致，Stata 是各自加星而不是替你选一个。

**签名**

```python
ws.varsoc(data: 'np.ndarray | pd.DataFrame', *, maxlag: 'int' = 4, trend: 'str' = 'c', names: 'list[str] | None' = None) -> 'VarSocResult'
```

**示例**

```python
print(ws.varsoc(base[['y1','y2']].to_numpy(), maxlag=2))
```

**输出**

```text
None
```

### `varstable`

**Stata 指令:** `varstable`

`ws.varstable` 检查已拟合 VAR 的特征值稳定性条件：伴随矩阵各特征值的模与其特征值并列。当所有模都严格落在单位圆内时过程稳定，Stata 会把不满足的标出来。

**签名**

```python
ws.varstable(result: 'Any') -> 'VarStableResult'
```

**示例**

```python
print(ws.varstable(ws.VAR(base[['y1','y2']].to_numpy(), lags=1, names=['y1','y2']).fit()))
```

**输出**

```text
None
```

### `varwle`

**Stata 指令:** `varwle`

`ws.varwle` 给出 VAR 的 Wald 滞后排除检验：对每个方程，检验每一阶的系数是否联合为零，另加一行 ALL。统计量由系数协方差构造的 Wald chi2，因此需要整个滞后块，而不是逐个系数。

**签名**

```python
ws.varwle(result: 'Any') -> 'VarWleResult'
```

**示例**

```python
print(ws.varwle(ws.VAR(base[['y1','y2']].to_numpy(), lags=1, names=['y1','y2']).fit()))
```

**输出**

```text
None
```

### `veclmar`

**Stata 指令:** `veclmar`

已拟合 VECM 的残差自相关拉格朗日乘数检验，即 `varlmar` 的 VECM 对应物。它作用于差分后的残差，因此滞后编号指的是差分而不水平值。

**签名**

```python
ws.veclmar(result: 'Any', *, mlag: 'int' = 2) -> 'LmarResult'
```

**示例**

```python
print(ws.veclmar(ws.VECM(base[['y1','y2']].to_numpy(), coint_rank=1, lags=2, names=['y1','y2']).fit(), mlag=1))
```

**输出**

```text
None
```

### `vecrank`

**Stata 指令:** `vecrank varlist, lags(#)`

Johansen 协整秩检验，含 trace 与最大特征值两种形式及其 5% 临界值。`lags` 按 Stata 的习惯计 VAR 的滞后阶数；`lags(1)` 单独处理，因为一阶滞后没有可剔除的滞后差分，简秩回归两侧都只对常数取残差。

**签名**

```python
ws.vecrank(data: 'Any', *, lags: 'int' = 2, trend: 'str' = 'constant', show_max: 'bool' = False) -> 'VecRankResult'
```

**示例**

```python
print(ws.vecrank(base[['y1','y2']].to_numpy(), lags=1))
```

**输出**

```text
None
```

**注意:** `trend` 只接受 "constant"；Stata 的其他趋势设定会被拒绝，而不是近似处理。

### `vecstable`

**Stata 指令:** `vecstable`

已拟合 VECM 的特征值稳定性条件。注意该 VECM 必须确实存在短期块可供检验：`lags=1` 时没有短期块，检验会报错而不是返回空表。

**签名**

```python
ws.vecstable(result: 'Any') -> 'VecStableResult'
```

**示例**

```python
print(ws.vecstable(ws.VECM(base[['y1','y2']].to_numpy(), coint_rank=1, lags=2, names=['y1','y2']).fit()))
```

**输出**

```text
None
```

### `wntestq`

**Stata 指令:** `wntestq varname, lags(#)`

白噪声的混成（Ljung-Box）检验：序列到 `lags` 阶的自相关是否联合为零。与 `estat bgodfrey` 不同，它检验的是**序列**而非回归残差，因此传入数值序列，或一个你要取其残差的结果对象。默认阶数是 `max(1, min(int(n/2) - 2, 40))`（`ac.ado:71`），与上一次回归估了几个参数无关，且有 **40 的上限**——n = 758 时 Stata 报 chi2(40)，而不是 377。`time=` 给它 `tsset` 的时间轴；有缺口时那些配对从自协方差和中剔除，并多打一行 `(note: time series has N gaps)`。

**签名**

```python
ws.wntestq(result: 'Any', *, lags: 'int | None' = None, time: 'str | Any | None' = None, delta: 'float' = 1.0) -> 'PortmanteauResult'
```

**示例**

```python
print(ws.wntestq(ws.OLS.from_formula('y ~ x1 + x2', base).fit(), lags=2))
```

**输出**

```text
None
```

### `xttest0`

**Stata 指令:** `xttest0 (after xtreg, re)`

随机效应的 Breusch-Pagan 拉格朗日乘数检验：面板方差分量是否为零，即检验 `sigma_u = 0`。这是它必须跟在**随机效应**拟合之后的原因——固定效应已经吸收了 `u`，该检验不适用。

**签名**

```python
ws.xttest0(result: 'Any', *, entity: 'Any' = None) -> 'XtTest0Result'
```

**示例**

```python
d = base.assign(firm=np.repeat(np.arange(20), 10), t=np.tile(np.arange(10), 20))
f = ws.RandomEffects.from_formula('y ~ x1', d, entity='firm', time='t').fit()
print(ws.xttest0(f, entity=d['firm']))
```

**输出**

```text
None
```

