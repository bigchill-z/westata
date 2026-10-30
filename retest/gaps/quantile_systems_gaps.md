# westata 辅助统计与展示功能缺口：分位数 / 系统 / 因果 / 抽样 / 重抽样 / DID

> 本报告只覆盖任务书指定的六个家族。所有断言附 `path:line`。
> 未附行号的一律标注"未验证"或"手册知识（标注）"。
> 命令均在 `D:\myCodeProject\statapy` 下用 `./.venv/Scripts/python.exe` 运行。

---

## 0. 证据分级与方法

1. **A 级（最高）**：`tests/conformance/stata_fixtures/*.txt` 的活体 StataNow 19.5 控制台抄录。
2. **B 级**：`tests/conformance/stata_fixtures/*.json` 的 `scalars`/`macros`（`ereturn list` 全量）。
3. **C 级**：手册知识，明确标注。
4. **D 级**：本次实跑（`probe` 脚本 dump `e()` 键与对象属性），证明"westata 到底暴露什么"。

本次实跑覆盖：`qreg`(iid/robust)、`rreg`、`sqreg`、`bsqreg`、`sureg`、`mvreg`、`eivreg`、
`manova`、`teffects ra/ipw/aipw/nnmatch`、`didregress`、`xtdidregress`、`svy: mean/total/proportion/regress/logit`、
`jackknife`、`bootstrap`、`permute`、`frontier`、`nl`。

---

## 1. 逐估计器缺口表

图例：**已存** = westata `e()` 已有；**算了未存** = 源码里算出来了但没放进 `e` 字典；
**未算** = 源码里没有该计算；**无渲染** = 数据在 `e()` 里但 `summary()` 不打印。

### 1.1 `qreg`（`src/westata/estimators/quantile.py`）

| Stata 统计量（打印或 e()） | Stata 来源 | westata | 判定 |
|---|---|---|---|
| `Number of obs` | p8_outputs.txt:20 | `e(N)`（quantile.py:410） | ✅ |
| `Raw sum of deviations 438.3735 (about -.46158607)` | p8_outputs.txt:21 | `e(sum_rdev)` + `e(q_v)`（quantile.py:415-416）**已算已存** | ✅ 数据齐 |
| `Min sum of deviations 156.7522` | p8_outputs.txt:22 | `e(sum_adev)`（quantile.py:415） | ✅ 数据齐 |
| `Pseudo R2 = 0.6424` | p8_outputs.txt:22 | `e(r2_p)`（quantile.py:417） | ✅ 数据齐 |
| 系数表 + t + P + CI | p8_outputs.txt:25-30 | `StataResults` 全有 | ✅ |
| `e(sparsity)` / `e(f_r)` | 手册（[R] qreg） | quantile.py:419-420 | ✅ |
| `e(bwidth)` / `e(bwmethod)` / `e(denmethod)` | 手册 | quantile.py:421-423 | ✅ |
| **迭代轨迹 `Iteration k: Sum of abs. weighted deviations = ...`** | **p8_outputs.txt:3-18** | ❌ 未算未存 | ❌ 新计算+新渲染 |
| **`e(cmd)` / `e(cmdline)` / `e(depvar)` / `e(properties)` 等 macros** | var.json:98-113（A/B 级） | ❌ 全库无 macros（`EReturn` 只装标量，`results/stata_return.py:21-67`） | ❌ 架构级 |
| **`.25 Quantile regression` 标题块**（含 Raw/Min sum/Pseudo R2 两行右侧） | **p8_outputs.txt:20-22** | ❌ 走通用模板，见 §2.1 | ❌ 新渲染 |
| `vce(robust)` 的 `Robust` 表头 | p8_outputs.txt:211-212 | ❌ 无此标签 | ❌ 新渲染 |

### 1.2 `sqreg` / `bsqreg`（`src/westata/estimators/quantile_boot.py`）

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| `Number of obs` | p8_outputs.txt:125 | `e(N)`（quantile_boot.py:144） | ✅ |
| **`bootstrap(200) SEs` 标签** | p8_outputs.txt:126 | ❌ `cov_type="bootstrap"`（quantile_boot.py:163）但无渲染标签 | ❌ 新渲染 |
| **per-quantile `Pseudo R2` 块**（`.25 Pseudo R2 = 0.6424` × 3 行） | **p8_outputs.txt:126-128** | ❌ **完全没有**——`Sqreg.fit` 的 `e` 字典（quantile_boot.py:143-151）只有 `N/df_m/df_r/quantiles/reps/rank/vce`，**连 sum_adev/sum_rdev/r2_p 都没有** | ❌ **算了未存**（父类 `QuantileRegression.fit` 会算，见 quantile.py:397-403，但 `Sqreg` 自己不复用） |
| `q25/q50/q75` 分组表头 | p8_outputs.txt:134-147 | ❌ `param_names` 用 `q0.25:` 前缀（quantile_boot.py:135），无分组渲染 | ❌ 新渲染 |
| **`e(reps)`** | 手册 | ✅ quantile_boot.py:148 | ✅ |
| **`Bootstrap replications (200): ...done` 进度行** | p8_outputs.txt:120-123 | ❌ | ❌ 新渲染（纯 display） |
| `e(cmdline)` 等 macros | var.json:98-113 | ❌ | ❌ |

**`bsqreg` 额外**：Stata 打印 `Raw sum of deviations` / `Min sum of deviations` / `Pseudo R2` 完整块
（p8_outputs.txt:159-161），而 `Bsqreg.fit`（quantile_boot.py:333-348）只把 `q` 塞进父 `e`，
**同样一行都没有**（父 `Sqreg.e` 本就没有）。判定：**算了未存**（同一份残差，父类已有公式）。

### 1.3 `rreg`（`quantile.py:584-761`）

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| `Number of obs = 400` | p8_outputs.txt:181 | `e(N)`（quantile.py:723） | ✅ |
| `F(2, 397) = 688.92` / `Prob > F` | p8_outputs.txt:182-183 | `e(F)`（quantile.py:739） | ✅ |
| 系数表 | p8_outputs.txt:186-190 | ✅ | ✅ |
| **`Huber iteration k: Maximum difference in weights = ...` / `Biweight iteration k: ...`** | **p8_outputs.txt:174-179** | ❌ `e(iterations)` 只存总数（quantile.py:743），**逐次轨迹在 `_iterate` 里算出来了（quantile.py:804,842）但只 `len()`** | ❌ **算了未存** |
| `e(N_weighted)`（非零 biweight 数） | 手册 | ✅ quantile.py:724,687 | ✅ |
| `e(scale)`/`e(aa)`/`e(lambda)`/`e(tune)`/`e(tolerance)` | 手册 | ✅ quantile.py:729-733 | ✅ |
| `e(mss)`/`e(rss)`/`e(mse)`/`e(rmse)`/`e(r2)`/`e(r2_a)` | 手册 | ✅ quantile.py:734-740 | ✅ |
| `e(dropped)`（Cook's D 剔除数） | 手册 | ✅ quantile.py:728 | ✅ |
| **`e(huber_c)`/`e(biweight_c)` 命名**：Stata 用 `e(c_h)`/`e(c_b)` | 手册（标注 C 级） | 键名不同（quantile.py:741-742） | ⚠️ 命名分歧 |

### 1.4 `sureg`（`systems.py:272-611`）

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| 方程表 `Obs Params RMSE "R-squared" chi2 P>chi2` | **p10_outputs.txt:7-11** | 全部已算已存：`e(r2_1)`/`e(rmse_1)`/`e(chi2_1)`/`e(p_1)`（systems.py:580-588） | ✅ 数据齐 |
| `e(ll)` | 手册 | ✅ systems.py:556-560 | ✅ |
| `e(Sigma)` / `e(corr)` | 手册 | ✅ systems.py:547-548 | ✅ |
| `e(k_eq)` / `e(k)` / `e(rank)` / `e(df_m)` | 手册 | ✅ systems.py:541-545 | ✅ |
| `e(N)` | p10_outputs.txt:9 | ✅ systems.py:539 | ✅ |
| **`e(df_r)`** | 手册（sureg 不打印也不存） | ✅ systems.py:604（westata 自己设） | ✅ |
| **`isure` 迭代轨迹 `Iteration 1: Tolerance = .02286855`** | **p10_outputs.txt:32-34** | ❌ `iterate=True` 直接抛 `UnsupportedError`（systems.py:413-419） | ❌ 新计算 |
| **Breusch-Pagan 独立性检验**（`sureg, corr`） | 手册（C 级） | ❌ `sigma`/`corr` 已存但无 BP 检验统计量 | ❌ **算了未存**（BP 统计量 = `n * corr²`，可由 `e(corr)` 一步推出） |
| `e(cmdline)` 等 macros | var.json:98-113 | ❌ | ❌ |

### 1.5 `mvreg`（`systems.py:621-807`）

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| 方程表 `Obs Parms RMSE "R-sq" F P>F` | **p10_outputs.txt:63-66** | 已算已存：`e(r2_y1)`/`e(rmse_y1)`/`e(F_y1)`/`e(p_y1)` + macro 形态列表 `e(r2)`/`e(rmse)`/`e(F)`/`e(p_F)`（systems.py:763-787） | ✅ 数据齐 |
| `e(Sigma)` / `e(corr)` | 手册 | ✅ systems.py:749-750 | ✅ |
| `e(k_eq)`/`e(k)`/`e(rank)` | 手册 | ✅ systems.py:744-746 | ✅ |
| `e(N)` | p10_outputs.txt:65 | ✅ systems.py:742 | ✅ |
| **`e(df_r)` = 396** | 手册 | ✅ systems.py:746 | ✅ |
| **`e(ll)`** | 手册（mvreg 存 ll） | ❌ **未存**（`Sigma` 已算，ll 公式与 sureg 同构，systems.py:555-560 已有实现） | ❌ **算了未存** |
| **`e(k_eq_model)` / `e(k_dv)`** | var.json:60-96 | ❌ | ❌ |
| **`mvreg` 的多元整体检验（Wilks/Pillai/LR/H/Roy 四个 + F + p）** | 手册（C 级） | ❌ **完全未算** | ❌ 新计算 |

### 1.6 `eivreg`（`systems.py:811-1007`）

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| `Number of obs` / `F` / `Prob > F` / `R-squared` / `Root MSE` | 手册 | ✅ `e(N)`/`e(F)`/`e(p)`/`e(r2)`/`e(rmse)`（systems.py:979-990） | ✅ |
| `e(df_m)` / `e(df_r)` / `e(rank)` | 手册 | ✅ systems.py:981-983 | ✅ |
| `e(reliab)` | 手册 | ✅ systems.py:988 | ✅ |
| `e(vce)` | 手册 | ✅ systems.py:989 | ✅ |
| **`e(k)`** | 手册 | ❌ | ❌ 琐碎 |
| **实测异常**：`r2 = 1.12706`、`rmse = 0.0`（见 §5 附录） | 本次实跑 | ⚠️ 数值异常，非展示问题 | ⚠️ 需单独查 |

### 1.7 `manova`（`systems.py:1012-1208`）— **重点：Q5**

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| `Number of obs = 240` | p10_outputs.txt:123 | ✅ `ManovaResult.nobs`（systems.py:1030） | ✅ |
| `W 0.4290` Wilks' lambda | p10_outputs.txt:130 | ✅ `wilks_lambda`（systems.py:1161） | ✅ |
| `F(df1, df2) = 62.15 Prob>F e` | p10_outputs.txt:130 | ✅ `F`/`df1`/`df2`/`p`（systems.py:1177-1186） | ✅ |
| `Residual 237` / `Total 239` | p10_outputs.txt:135,137 | ✅ `df`(g-1) + `nobs`/`groups` 可推 | ✅ |
| `e = exact, a = approximate, u = upper bound` 图例 | p10_outputs.txt:139 | ⚠️ `statistics["exact"]`（systems.py:1191-1206）**已算已存**，但 `__str__` 不打印 | ⚠️ 无渲染 |
| **`P 0.5710` Pillai's trace** | **p10_outputs.txt:131** | ❌ **完全未算** | ❌ 新计算 |
| **`L 1.3307` Lawley-Hotelling trace** | **p10_outputs.txt:132** | ❌ **完全未算** | ❌ 新计算 |
| **`R 1.3306` Roy's largest root** | **p10_outputs.txt:133** | ❌ **完全未算** | ❌ 新计算 |
| **`.e`-like mapping** | — | ❌ `ManovaResult` 是 frozen dataclass，**没有 `.e`**（systems.py:1011-1057，实测 `hasattr(r,"e")==False`） | ❌ 架构 |
| **`.summary()`** | — | ❌ 无（实测 `hasattr(r,"summary")==False`），只有 `__str__` 三行（systems.py:1051-1057） | ❌ 新渲染 |

**Q5 完整答案见 §4.5。**

### 1.8 `teffects`（`treatreg_effects.py`）— **重点：Q3**

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| `ATE 2.034515 .0610149 33.34 0.000 [1.914928, 2.154102]` | **p11_outputs.txt:16-18** | ✅ `estimate`/`std_err`/`z`/`pvalue`/`ci_lower`/`ci_upper`（treatreg_effects.py:122-128） | ✅ |
| `POmean` 行（`treated 0` / `treated 1`） | **p11_outputs.txt:20-22** | ✅ `potential_outcomes = {"0": (mean, se), "1": (mean, se)}`（treatreg_effects.py:129） | ✅ |
| `ATET` 行 | p11_outputs.txt:91-93 | ✅ `estimand="atet"` | ✅ |
| `Number of obs = 1,200` | **p11_outputs.txt:8** | ❌ **完全没有**——`TreatmentEffectResult` 无 `nobs` 字段（treatreg_effects.py:122-130，实测 `hasattr(r,"nobs")==False`） | ❌ **算了未存**（`n = y.size` 在 `fit()` 里现成，如 treatreg_effects.py:1213） |
| `Estimator : regression adjustment` | **p11_outputs.txt:9** | ❌ 无字段（类名隐含） | ❌ 琐碎 |
| `Outcome model : linear` / `Treatment model: none` / `logit` / `matching` | **p11_outputs.txt:10-11, 35-36, 60-61** | ❌ 无字段 | ❌ 琐碎 |
| `Distance metric: Mahalanobis` / `Matches: requested/min/max` | **p11_outputs.txt:106-108** | ❌ 无字段（`nneighbor` 存在模型上但不进结果） | ❌ **算了未存** |
| `Robust` / `AI robust` 表头 | **p11_outputs.txt:13, 110** | ❌ 无渲染标签 | ❌ 新渲染 |
| `Iteration 0: EE criterion = 2.733e-31` | **p11_outputs.txt:5-6** | ❌ 无 | ❌ 新计算 |
| **`.e`-like mapping / `.summary()`** | — | ❌ 都没有（实测 `hasattr(r,"e")==False`、`hasattr(r,"summary")==False`），只有 `__str__`（treatreg_effects.py:132-144） | ❌ 架构 |

### 1.9 `svy:`（`survey.py`）— **重点：Q2**

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| `Number of strata = 8` | **p12_outputs.txt:8** | ✅ `e(N_strata)`（survey.py:636） | ✅ **已存** |
| `Number of PSUs = 48` | **p12_outputs.txt:9** | ✅ `e(N_psu)`（survey.py:637） | ✅ **已存** |
| `Number of obs = 576` | **p12_outputs.txt:8** | ✅ `e(N)`（survey.py:635） | ✅ **已存** |
| `Population size = 837.592551` | **p12_outputs.txt:9** | ✅ `e(N_pop)`（survey.py:638） | ✅ **已存** |
| `Design df = 40` | **p12_outputs.txt:10** | ✅ `e(df_r)`（survey.py:640，`SurveyLayout.df_r` = #PSU−#strata，survey.py:423-427） | ✅ **已存** |
| `Survey: Mean estimation` 标题 | p12_outputs.txt:6 | ❌ 无（实测打印 `Model: svy: mean`） | ❌ 新渲染 |
| `(running mean on estimation sample)` | p12_outputs.txt:4 | ❌ 无 | ❌ 新渲染 |
| `F(2, 39) = 238.36` / `Prob > F` | p12_outputs.txt:70-71 | ✅ `e(F)`/`e(p)`（survey.py:1070，`_survey_f_stat` survey.py:573-596） | ✅ |
| `R-squared = 0.5090` | p12_outputs.txt:72 | ✅ `e(r2)`（survey.py:1070，`_r_squared` survey.py:1052-1057） | ✅ |
| `Logit` 列（proportion 的 logit 变换 CI） | **p12_outputs.txt:51-56** | ❌ 无（实测只给线性化 CI） | ❌ 新计算+新渲染 |
| **`DEFF` / `SE` under SRS / `DEFF` mean 设计效应块** | 手册（C 级，`svy: mean, deff`） | ❌ **完全没有**——`survey.py` 全文搜 `deff` **零命中** | ❌ **算了未存**（= `V_design / V_srs`，`V_srs` 由 `w` 与样本即可算出） |
| **`N. of poststrata`** | 手册（C 级，`poststratify` 后） | ❌ 无（`poststratify` 本身未实现，GAP_REGISTER.md:159） | ❌ 新功能 |
| **`mse` 选项**（`svy, mse` → `e(mse)`） | 手册（C 级） | ❌ 全文搜 `mse` **零命中** | ❌ 新计算 |
| **`subpop()`** | GAP_REGISTER.md:158 已列 ⬜ | ❌ 全文搜 `subpop` **零命中** | ❌ 新计算（子总体方差处理） |
| `e(singleunit)` | 手册 | ✅ survey.py:642 | ✅ |
| `e(vce)="linearized"` | 手册 | ✅ survey.py:643 | ✅ |
| `e(ll)`（logit） | 手册 | ✅ survey.py:1291 | ✅ |

**Q2 完整答案见 §4.2。**

### 1.10 `bootstrap` / `jackknife` / `permute`（`resample.py`）— **重点：Q4**

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| `Number of obs = 400` | **p13_outputs.txt:16, 40, 71** | ✅ `e(N)`（resample.py:347） | ✅ |
| `Replications = 200` | **p13_outputs.txt:17, 41, 72** | ✅ `e(reps)`（resample.py:602, 688）**但 jackknife 不存 reps**（jackknife `e` 只有 `note/expected_gap/refit`，resample.py:477-486） | ⚠️ 部分 |
| `Observed coefficient` 列 | p13_outputs.txt:23, 49, 81 | ✅ `params` | ✅ |
| `Bootstrap std. err.` 列 | p13_outputs.txt:23, 49 | ✅ `bse` | ✅ |
| `z` / `P>|z|`（bootstrap 用正态） | p13_outputs.txt:26, 52 | ✅ `dist="z"`（resample.py:600） | ✅ |
| `t` / `P>|t|`（jackknife 用 t） | p13_outputs.txt:81-85 | ✅ `dist="t"`（resample.py:476） | ✅ |
| `[95% conf. interval]`（Normal-based） | **p13_outputs.txt:26, 52** | ✅ `conf_int()`（results/base.py:208-232） | ✅ |
| `Wald chi2(2) = 1127.33` / `Prob > chi2` / `R-squared` / `Adj R-squared` / `Root MSE`（bootstrap 内嵌 regress 的完整头部） | **p13_outputs.txt:42-46** | ❌ **未存**——`e` 字典（resample.py:346-353）只有 `N/df_m/df_r/rank/estimator/vce`，`_stored()` 明明算了 `rmse/rss/mss/r2`（resample.py:269-277）却没传出来 | ❌ **算了未存** |
| `Jackknife` 表头标签 | p13_outputs.txt:80 | ❌ `cov_type="jackknife"` 但无渲染标签 | ❌ 新渲染 |
| `Command: summarize y` / `_bs_1: r(mean)` 行 | **p13_outputs.txt:19-20** | ❌ 无 | ❌ 新渲染 |
| **`Bootstrap replications (200): .........10...done` 进度行** | **p13_outputs.txt:11-14** | ❌ 无 | ❌ 新渲染 |
| `warning: summarize does not set e(sample)...` | p13_outputs.txt:6-9 | ❌ 无 | ❌ 新渲染 |
| **`permute` 的 `p` 值表**（Stata 打印 `Statistic p-value` 表而非系数表） | 手册（C 级；本 fixture 的 permute 命令语法失败，p13_outputs.txt:92-93） | ❌ `e(pvalue_permuted)` 已算已存（resample.py:690），但**无渲染**（实测 `summary("stata")` 打印的是系数表） | ⚠️ 无渲染 |
| **`e(N_reps)` / `e(rng)` / `e(rngstate)`** | 手册（C 级） | ❌ 只有 `e(seed)`（resample.py:603, 689） | ❌ 琐碎 |
| **`e(bs_version)` 等 macros** | var.json:98-113 | ❌ | ❌ |

### 1.11 `didregress` / `xtdidregress`（`did.py`）

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| `ATET 1.18613 .0590465 20.09 0.000 [1.069692, 1.302567]` | **p11_did.txt:29-31** | ✅ `params["treated"]` + `bse` + `tvalues` + `conf_int()` | ✅ |
| `Number of obs = 1,600` | p11_did.txt:21 | ✅ `e(N)`（did.py:779） | ✅ |
| `(Std. err. adjusted for 200 clusters in unit)` | **p11_did.txt:24** | ✅ `e(N_clust)=200`（did.py:793）但**无该行渲染** | ⚠️ 无渲染 |
| **`Treatment and time information` 块**（`Time variable:`/`Control:`/`Treatment:`/`Group` 表 100/100/`Time Minimum 1 4`/`Maximum 1 4`） | **p11_did.txt:5-19** | ❌ **完全未算未存**（`self._group`/`self._time` 在模型上，did.py:447-448，但结果对象不暴露；实测 `[a for a in dir(r) if "group"/"time" in a]` 为空） | ❌ **算了未存** |
| `Data type: Repeated cross-sectional` / `Longitudinal` | **p11_did.txt:22, 57** | ✅ `e(data_type)`（did.py:786）但**无渲染** | ⚠️ 无渲染 |
| `Note: ATET estimate adjusted for covariates, group effects, and time effects.` | **p11_did.txt:33** | ❌ 无 | ❌ 新渲染 |
| `Robust` 表头 | p11_did.txt:26 | ❌ 无 | ❌ 新渲染 |
| `e(estimand)="ATET"` | 手册 | ✅ did.py:785 | ✅ |
| `e(r2)`/`e(r2_a)`/`e(rmse)`/`e(F)`/`e(ll)` | 手册（did.py:82-88 自述是 **westata additions**，非 Stata） | ✅ | ✅（标注为扩展） |

### 1.12 `nl` 与 `frontier`（`nonlinear.py`）

| Stata 统计量 | Stata 来源 | westata | 判定 |
|---|---|---|---|
| `nl` 的 `Source/SS/df/MS` 表 + `R-squared`/`Adj R-squared`/`Root MSE`/`Res. dev.` | **p10_outputs.txt:97-102** | `e(rss)`/`e(mss)`/`e(tss)`/`e(r2)`/`e(r2_a)`/`e(rmse)`/`e(deviance)` 全有（nonlinear.py:1327-1346） | ✅ 数据齐 |
| **`Res. dev. = -279.1251`** | **p10_outputs.txt:102** | ✅ `e(deviance)`（nonlinear.py:1340） | ✅ 数据齐 |
| `Iteration k: Residual SS = ...` | p10_outputs.txt:89-95 | ❌ 无 | ❌ 新渲染 |
| **`/b0` `/b1` 斜杠前缀** | p10_outputs.txt:107-108 | ✅ `param_names=[f"/{n}"]`（nonlinear.py:1350） | ✅ |
| **`frontier` 的 `/lnsig2v` `/lnsig2u` 两行** | **p10_outputs.txt:162-163** | ✅ `param_names` 追加（nonlinear.py:1655），实测 `params.index = ['Intercept','x1','x2','/lnsig2v','/lnsig2u']` | ✅ **Q6 已确认修复** |
| **`sigma_v`/`sigma_u`/`sigma2`/`lambda` 四行派生统计** | **p10_outputs.txt:165-168** | ✅ `e(sigma_v)`/`e(sigma_u)`/`e(sigma2)`/`e(lambda)`/`e(gamma)`（nonlinear.py:1640-1646） | ✅ 数据齐（**但系数表里没有这四行**——它们是 `e()` 标量而非参数，Stata 打印在表内，见 §2.1） |
| `Wald chi2(2) = 4462.60` / `Prob > chi2` | p10_outputs.txt:152-153 | ✅ `e(chi2)`/`e(p)`/`e(chi2type)`（nonlinear.py:1634-1636） | ✅ |
| `Log likelihood = -96.054179` | p10_outputs.txt:153 | ✅ `e(ll)`（nonlinear.py:1632） | ✅ |
| **`LR test of sigma_u=0: chibar2(01) = 23.60 Prob >= chibar2 = 0.000`** | **p10_outputs.txt:170** | ✅ `e(lr_chi2)`/`e(p_lr)`（nonlinear.py:1637-1639，`p_lr = 0.5*chi2.sf`，nonlinear.py:1638-1639 注释明确 `chibar2(01)`） | ✅ 数据齐（**无渲染**） |
| `e(ll_0)` | 手册 | ✅ nonlinear.py:1633 | ✅ |
| `Iteration k: Log likelihood = ...` | p10_outputs.txt:146-149 | ❌ 无 | ❌ 新渲染 |

---

## 2. 展示层缺口（display surface gaps）

### 2.1 **渲染层结构性问题：单一模板 vs Stata 的至少 6 种头部块形态**

`src/westata/results/summary.py:132-245` 的 `_stata_summary` 只有**两个**分支：
`chi2type`/`dist=="z"` → ML 式；否则 → `regress` 式（`F`/`Prob>F`/`R²`/`Adj R²`/`Root MSE`）。

实测 `qreg`（quantile=0.25）：

```
Number of obs         =        600
F(2, 597)             =          .        <- Stata 不打印 F
Prob > F              =          .
Log likelihood        =          .        <- Stata 不打印，qreg 非 ML
Pseudo R2             =     0.6424
```

而 Stata（p8_outputs.txt:20-22）：

```
.25 Quantile regression                             Number of obs =        600
  Raw sum of deviations 438.3735 (about -.46158607)
  Min sum of deviations 156.7522                    Pseudo R2     =     0.6424
```

**根因（契约耦合缺陷）**：`summary.py:216-220` 用 `e(r2_p)` **是否存在**判断"是否 ML 模型"。
`qreg` 为存 Stata 的 check-function 伪 R² 用了同一个键 `r2_p`（quantile.py:417），
渲染器于是误判为 ML，打印了一行空的 `Log likelihood = .`。
`sureg`/`mvreg`/`frontier`/`svy:` 的 `r2_p` 不冲突（它们不用该键），但 `sqreg`/`bsqreg`/`jackknife`/`bootstrap`/`permute` 实测打印的是 **R² 三行全空 + Root MSE**（数据本来就没有）。

**Stata 至少 6 种头部形态，westata 一个都没专门渲染**：
1. `regress` 式（已有）
2. ML 式（已有）
3. `qreg` 式（p8_outputs.txt:20-22）
4. `sqreg` 式（p8_outputs.txt:125-128，含 per-quantile Pseudo R² 块）
5. `rreg` 式（p8_outputs.txt:181-183，F/Prob>F 在右侧，无 R²/RMSE）
6. `svy` 式（p12_outputs.txt:6-10，strata/PSU/obs/pop/design df 五行）
7. `didregress` 式（p11_did.txt:5-33，time-info 块 + `Data type:` + `Note:`）
8. `teffects` 式（p11_outputs.txt:8-18，`Estimator:`/`Outcome model:`/`Treatment model:`）
9. `bootstrap`/`jackknife` 式（p13_outputs.txt:16-17, `Replications` 行）
10. `frontier` 式（p10_outputs.txt:151-170，含表内 4 行派生统计 + LR 测试行）
11. `manova` 式（p10_outputs.txt:123-139，四统计量 + 图例行）

### 2.2 `e()` 里已有但 `summary()` 不打印的项（零成本）

| 项 | 数据位置 | 应打印位置 |
|---|---|---|
| `e(sum_rdev)` / `e(sum_adev)` / `e(q_v)` | quantile.py:415-416 | qreg 头部两行 |
| `e(N_strata)`/`e(N_psu)`/`e(N_pop)`/`e(df_r)` | survey.py:636-640 | svy 头部五行 |
| `e(N_clust)` | did.py:793 | `(Std. err. adjusted for ... clusters in ...)` |
| `e(data_type)` | did.py:786 | `Data type: ...` |
| `e(lr_chi2)`/`e(p_lr)` | nonlinear.py:1637-1639 | `LR test of sigma_u=0: chibar2(01) = ...` |
| `e(pvalue_permuted)` | resample.py:690 | permute 的 p 值表 |
| `ManovaResult.statistics["exact"]` | systems.py:1206 | `e = exact, a = approximate, u = upper bound` 图例 |
| `e(iterations)` | quantile.py:743 | `Huber/Biweight iteration k:` 行 |

### 2.3 完全没有渲染器的项（需写新渲染器，数据部分就绪）

| Stata 展示 | 数据就绪度 | 说明 |
|---|---|---|
| `estat vce`（系数协方差矩阵） | 100%（`cov_params()`，base.py:203-206） | Lead 报告 §0 指认 2 已详述；本家族同样适用 |
| `estat vce, correlation` | 100% | `D^-1/2 V D^-1/2` |
| `estat ic`（AIC/BIC） | `frontier` 有 `ll`/`ll_0`，其余家族无 | 需新计算 |
| 各命令的 `Note:` 脚注行 | 0% | 需逐命令文案 |
| 重抽样进度行 `.........10.........20...done` | 0% | 纯 display，`reps` 已存 |
| 迭代轨迹行（qreg 14 行、rreg 6 行、nl 6 行、frontier 4 行、teffects 2 行） | rreg 逐次值**已算**（quantile.py:804,842）；qreg/nl/frontier/teffects **未存** | 混合 |
| `ManovaResult.summary()` | 数据齐 | `ManovaResult` 不是 `StataResults`，没有 `summary()` |
| `TreatmentEffectResult.summary()` | 数据齐（除 nobs） | 同上 |

### 2.4 `etable` 层

`src/westata/report/etable.py` 的 `_DEFAULT_STATS`（Lead 报告已引 `etable.py:28-31`）只含 `N`/`r2`。
本家族的 `rmse`/`F`/`chi2`/`ll`/`N_clust`/`N_psu`/`N_strata` 均需用户手写 `stats=`。
因为 `etable.py:144` 读 `res.e`，**只要 §3 的"算了未存"项补进 `e()`，etable 立刻可用**。

---

## 3. Q1–Q6 逐条回答

### Q1：Stata `qreg` 打印什么，westata 打印什么？

**Stata 打印（p8_outputs.txt:20-22，A 级证据）：**

```
.25 Quantile regression                             Number of obs =        600
  Raw sum of deviations 438.3735 (about -.46158607)
  Min sum of deviations 156.7522                    Pseudo R2     =     0.6424
```

即四行：**标题含 quantile 标签**、**Raw sum of deviations + (about q_v)**、
**Min sum of deviations**、**Pseudo R2**。median 时标题变为 `Median regression`
（p8_outputs.txt:45）。注意 Stata **不打印** F 检验、Log likelihood、R-squared、Root MSE。

**westata `e()` 已存**（quantile.py:409-426）：`sum_rdev`（438.373532 ✅）、`q_v`（-0.461586 ✅）、
`sum_adev`（156.752241 ✅）、`r2_p`（0.642423 ✅）、`N`、`q`、`sparsity`、`f_r`、`bwidth`、
`bwmethod`、`denmethod`、`vce`、`convcode`、`df_m`、`df_r`、`rank`。
**数值全部对得上 fixture。**

**westata `r.summary("stata")` 实测打印**：

```
Number of obs         =        600
F(2, 597)             =          .
Prob > F              =          .
Log likelihood        =          .
Pseudo R2             =     0.6424
```

**差距**：
1. `Raw sum of deviations ... (about ...)` / `Min sum of deviations` **两行完全没有渲染**（数据在 `e()` 里）。
2. 多打印了 3 行 Stata 不打印的（F / Prob>F / Log likelihood），其中 `Log likelihood` 是**误判**：
   `summary.py:216-220` 因 `e(r2_p)` 存在而走 ML 分支。
3. 标题 `.25 Quantile regression` 无（`Model: qreg` 太泛，丢失 quantile 标签）。
4. 系数表列名用 `Intercept`（patsy 拼写），Stata 用 `_cons`。

**结论：Q1 是纯数据可用 + 渲染错误（多打了 3 行 + 漏了 2 行 + 分支判错）。**

### Q2：`svy: mean` 的 design-effect 块与头部块？

**Stata 打印（p12_outputs.txt:6-10，A 级证据）：**

```
Survey: Mean estimation

Number of strata =  8             Number of obs   =        576
Number of PSUs   = 48             Population size = 837.592551
                                   Design df       =         40
```

**westata 已存**（survey.py:634-644，`_SvyCommand._results`）：
`e(N_strata)=8`（survey.py:636）、`e(N_psu)=48`（survey.py:637）、`e(N_pop)=837.592551`（survey.py:638）、
`e(N)=576`（survey.py:635）、`e(df_r)=40`（survey.py:640，`SurveyLayout.df_r` = `n_psu - n_strata`，survey.py:423-427）。

**答案：五个数字全部已算已存，一个都不缺；缺的只是渲染。**
实测 `summary("stata")` 只打印 `Number of obs = 576` 与 `F(0, 40)`/`Root MSE`（Root MSE 对 mean 无意义），
strata/PSU/pop/design df 四行**完全没有渲染**，`Survey: Mean estimation` 标题也没有。

**DEFF 块（用户重点问的）**：`survey.py` 全文搜 `deff` / `DEFF` **零命中**（本次 grep 实证）。
即 **westata 不计算也不暴露任何设计效应**。判定：**算了未存**——
`DEFF = V_design / V_srs`，`V_design` 就是 `linearized_variance()`（survey.py:444-520）的返回值，
`V_srs` 对 mean 是 `sum(w²(y-ȳ)²)/sum(w)² * n/(n-1)`，两者都只需已持有的 `w` 与样本。

**`mse`**：全文搜 `mse` 零命中（`mse` 只在 resample.py 里作 `mse=` 选项出现，非 survey）。
**`subpop`**：全文搜 `subpop` 零命中（GAP_REGISTER.md:158 已列为 ⬜）。
**`N. of poststrata`**：无（`poststratify` 未实现，GAP_REGISTER.md:159）。

### Q3：`teffects` 的 `TreatmentEffectResult` 暴露什么？有 `summary()` 吗？

**`TreatmentEffectResult` 是 frozen dataclass**（treatreg_effects.py:98-144），字段：

| 字段 | 行号 | 对应 Stata 打印 |
|---|---|---|
| `estimand` | 122 | `ATE`/`ATET` 行标签（p11_outputs.txt:16, 91） |
| `estimate` | 123 | 系数列 |
| `std_err` | 124 | `std. err.` 列 |
| `z` | 125 | `z` 列 |
| `pvalue` | 126 | `P>|z|` 列 |
| `ci_lower`/`ci_upper` | 127-128 | `[95% conf. interval]` |
| `potential_outcomes` | 129 | `POmean` 行（p11_outputs.txt:20-22） |
| `se_method` | 130 | 无对应（westata 自加的诚实标注） |

**没有 `.e`-like mapping**：实测 `hasattr(r,"e") == False`。也没有 `params`/`bse`/`pvalues`/
`conf_int()`/`test()`/`predict()` 等 `StataResults` 表面——即**后估计与 `etable` 全线不可用**。

**没有 `.summary()`**：实测 `hasattr(r,"summary") == False`。只有 `__str__`
（treatreg_effects.py:132-144），打印单行 `ATE ... se ... z ... p ... [lo, hi]` + 每水平一行 `POmean(k)`，
外加非精确 SE 时的 note 行。

**Stata 打印而 westata 完全没有的**：
- `Number of obs = 1,200`（p11_outputs.txt:8）——**`n = y.size` 在 `fit()` 里现成**
  （如 treatreg_effects.py:1213），却没进结果。判定：**算了未存**。
- `Estimator : regression adjustment` / `inverse-probability weights` / `augmented IPW` /
  `nearest-neighbor matching`（p11_outputs.txt:9, 34, 59, 106）
- `Outcome model : linear` / `weighted mean` / `linear by ML` / `matching`（p11_outputs.txt:10, 35, 60, 107）
- `Treatment model: none` / `logit`（p11_outputs.txt:11, 36, 61）
- `Distance metric: Mahalanobis`、`Matches: requested = 1`、`min = 1`、`max = 1`（p11_outputs.txt:106-108）
  ——`nneighbor` 存在模型上（treatreg_effects.py:1085）但不进结果。判定：**算了未存**。
- `Robust` / `AI robust` 表头（p11_outputs.txt:13, 110）
- `Iteration 0: EE criterion = ...`（p11_outputs.txt:5-6, 30-31, 54-55）

**`nnmatch` 的额外差异**：Stata 的 nnmatch **不打印 POmean 行**（p11_outputs.txt:113-116 只有 ATE），
而 westata 对 nnmatch 也填了 `potential_outcomes`（treatreg_effects.py:1293-1296），
其 docstring 自述"their reported standard errors are the effect's error rescaled by the share of
the sample each mean covers"（treatreg_effects.py:1070-1073）。这是一处**多打印**而非缺失。

### Q4：`bootstrap`/`jackknife`/`permute` 的 `Number of obs`、`Replications`、`r()` 表与 CI？

**Stata 打印（p13_outputs.txt:16-27，A 级证据）：**

```
Bootstrap results                                          Number of obs = 400
                                                           Replications  = 200

       Command: summarize y
         _bs_1: r(mean)

------------------------------------------------------------------------------
              |   Observed   Bootstrap                         Normal-based
              | coefficient  std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
        _bs_1 |   .7552786   .1314638     5.75   0.000     .4976142    1.012943
```

**westata 逐项核对**：

| Stata 项 | westata | 证据 |
|---|---|---|
| `Number of obs = 400` | ✅ `e(N)` | resample.py:347 |
| `Replications = 200` | ⚠️ bootstrap ✅ `e(reps)`（resample.py:602）；permute ✅（resample.py:688）；**jackknife ❌**（`e` 只有 `note/expected_gap/refit`，resample.py:477-486） | resample.py:477-486, 602, 688 |
| `Observed coefficient` | ✅ `params` | resample.py:381 |
| `Bootstrap std. err.` | ✅ `bse` | — |
| `z` / `P>|z|` | ✅ bootstrap `dist="z"`（resample.py:600）；jackknife `dist="t"`（resample.py:476） | resample.py:476, 600 |
| `[95% conf. interval]`（Normal-based） | ✅ `conf_int()` 返回 CI（base.py:208-232）；**但渲染时无 `Normal-based` 列标题**，且 bootstrap 的 z 临界值取自正态（`dist="z"` → `norm.ppf`，base.py:222），与 Stata 的 Normal-based 一致 | base.py:219-232 |
| **`Command: summarize y` / `_bs_1: r(mean)`** | ❌ 无 | — |
| **`Bootstrap replications (200): ...done` 进度行** | ❌ 无 | — |
| **bootstrap 内嵌 `regress` 的完整头部**（`Wald chi2(2) = 1127.33`/`Prob > chi2`/`R-squared`/`Adj R-squared`/`Root MSE`，p13_outputs.txt:42-46） | ❌ **未存**：`_stored()` 明明算出 `rmse`/`rss`/`mss`/`r2`（resample.py:269-277）与 `_stored_weighted()` 同样（resample.py:210-218），但 `_results` 的 `e`（resample.py:346-353）不接收它们 | resample.py:269-277, 346-353 |

**关于 `src/westata/expressions/explist.py`**：该文件是 `[U] exp_list` 文法
（explist.py:1-28 的 docstring 与 `__all__`，explist.py:37-43），
`StoredResults` 只暴露 `r`/`e`/`b`/`se` 四张表（explist.py:50-61），
`_STAT_TABLES = {"r", "e"}`（explist.py:85）。它**不涉及 CI**——
CI 在 `results/base.py:208-232`。`explist.py` 的相关缺口是：
`_stored()`/`_stored_weighted()` 只填了 `e(N)/e(rmse)/e(rss)/e(mss)/e(r2)` 与
`r(mean)/r(sd)/r(Var)/r(N)/r(sum)/r(min)/r(max)`（resample.py:208-240, 267-300），
所以 `bootstrap r(anything): <command>` 只能取到这些键——**这是 expression 侧的可用性问题，不是 CI 问题**。

**permute 的额外缺口**：Stata 的 `permute` 打印的是 `Statistic p-value` 表而非系数表
（本 fixture 的 permute 命令因语法失败未捕获到该表，p13_outputs.txt:92-93，故此处标 C 级）。
westata 的 `e(pvalue_permuted)` 已算已存（resample.py:675, 690），
但 `summary("stata")` 走的是系数表路径（实测确认）。

### Q5：`manova` 到底算了哪些统计量？缺的四个及其公式

**`src/westata/estimators/systems.py` 实际计算的**（逐行确认）：

1. **Wilks' lambda** `Λ = |W| / |W+B|`（systems.py:1161，`within`/`between` 在 systems.py:1143-1152 构建）。
2. **Rao's F 近似**（systems.py:1163-1186）：
   `s = sqrt((p²q²-4)/(p²+q²-5))`，`df1 = p q`，`df2 = s(v - (p-q+1)/2) - (pq-2)/2`，
   `F = ((1-Λ^(1/s))/Λ^(1/s))·(df2/df1)`，`p = F.sf(F, df1, df2)`。
   其中 `p` = 响应变量数，`q = g-1`（systems.py:1170），`v = n-g`（systems.py:1136）。
3. **exact/approximate 标记** `exact = s in (1,2)`（systems.py:1191），存入 `statistics["exact"]`。

**`ManovaResult.statistics` 实测内容**：`{'wilks_lambda': 0.6510, 's': 2.0, 'exact': 1.0}`——**只有三项**。

**模块 docstring 与类 docstring 均明确声明只做 Wilks**：
- systems.py:52-53："``manova`` statistics other than Wilks' lambda (Pillai, Lawley-Hotelling, Roy) and designs with covariates or more than one factor."
- systems.py:1064-1066："so are the Pillai, Lawley-Hotelling and Roy statistics."
- docs/GAP_REGISTER.md:139："`manova` 的 Pillai/Lawley-Hotelling/Roy | ⬜ | 仅实现 Wilks' λ 与 Rao F"

**结论：GAP_REGISTER.md:139 的记载与源码现状一致，仍未实现。**

**缺失的四个统计量及公式**

设 `B` = 假设 SSCP（`g-1` df），`W` = 误差 SSCP（`n-g` df），`p` = 响应变量数，
`q = g-1`，`v = n-g`，`s = min(p,q)`。

**本次已用 fixture 数值（p10_outputs.txt:130-133，`n=240, g=3` → `p=2, q=2, v=237`）
反解验证了后三条的 F 参数化**（临时脚本实算，已删除）：

| Stata 行 | 统计量 | 公式 | df1 | df2 | F | Stata 打印（p10_outputs.txt） |
|---|---|---|---|---|---|---|
| `W 0.4290` | Wilks' lambda `Λ` | `|W|/|W+B|`，`s_w=√((p²q²-4)/(p²+q²-5))`，`df2=s_w(v-(p-q+1)/2)-(pq-2)/2`，`F=((1-Λ^{1/s_w})/Λ^{1/s_w})(df2/df1)` | 4.0 | 472.0 | **62.16** ✅ | `62.15` |
| `P 0.5710` | **Pillai's trace** `V` | `tr[B(B+W)^-1]`；df 用 `n_p=(v-p-q+1)/2`、`df2=s(2n_p+s+1)` → **df1=4.0, df2=474.0 已验证吻合**；F 的确切参数化本次**未解出**（`(V/s)(df2/df1)` 得 33.83 vs 打印 47.35） | 4.0 | 474.0 | ⚠️ 47.35 | `47.35` |
| `L 1.3307` | **Lawley-Hotelling trace** `U` | `tr[B W^-1]`；`n_LH=(v-p-q-1)/2`、`df2=s(2n_LH+s+1)`、`F=(U/s)(df2/df1)` | 4.0 | 470.0 | **78.18** ✅ | `78.18` |
| `R 1.3306` | **Roy's largest root** `Θ` | 最大广义特征值 `λ_max`；`df1=max(p,q)`、`df2=v-max(p,q)+q`、`F=Θ(df2/df1)`（Stata 标 `u` = upper bound） | 2.0 | 237.0 | **157.68** ✅ | `157.67` |
| `e/a/u` 图例 | 精确性标记 | 同 Wilks 的 `e = exact, a = approximate, u = upper bound on F` | — | — | — | p10_outputs.txt:139 |

**结论（Q5）**：
- westata **只算 Wilks' lambda 与 Rao F**（systems.py:1161, 1163-1186），
  源码与 docstring（systems.py:52-53, 1064-1066）和 GAP_REGISTER.md:139 三方一致，**记载准确，仍未实现**。
- 缺失的四个统计量是：**Pillai's trace**、**Lawley-Hotelling trace**、**Roy's largest root**、
  以及每个统计量对应的 **F 与 Prob>F 行**（外加 `e/a/u` 精确性标记的渲染）。
- **成本判定修正**：`B`/`W` 矩阵 westata **已经算出来了**（systems.py:1143-1152），
  四个统计量本身都是 `B`/`W` 的简单函数（迹、行列式、广义特征值）——
  所以**统计量本身属于"算了未存"（A 级低成本）**。
  但 **Lawley-Hotelling 与 Roy 的 F 参数化需按上表实现（已实算验证），Pillai 的 F 参数化本次未解出，
  需再查证**。这是"新计算"里最便宜的一档。

---

## 3.7 Lead 追加核实：从 Stata **ado 源码**解出全部四条 F（A 级）

> 本节是 Lead 在收到本报告后的追加核实。它**推翻了上面"Pillai 的 F 未解出"的结论**，
> 并**修正了本节"统计量属于算了未存"的成本判定**。

### 3.7.1 权威实现的位置

StataNow 19 装在 `D:\Program Files\StataNow19`。`manova.ado` 只做语法解析与转发
（`manova.ado:74` 调 `_manova`），真正的统计量在 **`ado/base/_/_canon.ado:676-749`**——
`manova` 走典型相关分析路径，`_canon` 负责构造 `stat_m` 矩阵。

### 3.7.2 关键发现：四个统计量来自 `e(ccorr)`，不是 `B`/`W`

`_canon.ado:703-713`：

```stata
scalar `wilks'   = 1
scalar `pillai'  = 0
scalar `lawley'  = 0
scalar `roy'     = `ccorr'[1,1]^2
scalar `roy'     = `roy'/(1-`roy')
forvalues i = 1/`ncc' {
    scalar `fo' = `ccorr'[1,`i']^2
    scalar `wilks'   = `wilks'*(1-`fo')
    scalar `pillai'  = `pillai' + `fo'
    scalar `lawley'  = `lawley' + `fo'/(1-`fo')
}
```

**四个统计量全部由典型相关系数平方 `fo_i = ccorr_i²` 表出。**

`_canon.ado:682-686` 定义 df，`:714-736` 定义 F：

```stata
scalar `df1' = `p'*`q'
scalar `w'   = `n' - (`p' + `q' + 3)/2.0
scalar `t'   = sqrt(((`p')^2*(`q')^2 - 4)/((`p')^2 + (`q')^2 -5))
if (`p'*`q' == 2) scalar `t' = 1          ← 必须实现的特判
scalar `df2' = `w'*`t'- .5*`p'*`q' + 1.0

scalar `fo' = (`wilks')^(1/`t')
scalar `wilksF'  = ((1-`fo')/`fo')*(`df2'/`df1')
scalar `pillaiF' = `pillai'/`ncc'                          ← 先除以 ncc
scalar `dfp1' = `max'*`ncc'
scalar `dfp2' = (`n' - `max' - 1)*`ncc'
scalar `pillaiF' = (`pillaiF'*`dfp2')/(`dfp1'*(1 - `pillaiF'))   ← 比值变换
scalar `dfl2' = `ncc'*(`n'-`max'-`ncc' -2) + 2
scalar `lawleyF' = `lawley'*`dfl2'/(`ncc'*`dfp1')
scalar `dfr1' = `max'
scalar `dfr2' = `n' - `max' - 1
scalar `royF' = `roy'*`dfr2'/(`dfr1')
```

### 3.7.3 四条全部对上 Stata 打印值（Lead 实跑验证）

n=240、g=3、p=2、q=2、ncc=2，代入上式：

| 统计量 | 算出 | df1 | df2 | F 算出 | Stata 打印 | 判定 |
|---|---|---|---|---|---|---|
| Wilks | 0.4290 | 4 | 472 | 62.16 | 62.15 | ✅ |
| **Pillai** | 0.5710 | 4 | 474 | **47.35** | **47.35** | ✅ **解出** |
| Lawley | 1.3307 | 4 | 470 | 78.18 | 78.18 | ✅ |
| Roy | 1.3306 | 2 | 237 | 157.68 | 157.67 | ✅ |

**四个统计量、四个 F、八个 df 全部命中。**

### 3.7.4 Pillai 的 F 是怎么解出的（此前为何没解出）

本报告 §3 Q5 曾用 `(V/s)(df2/df1)` 得 33.83，与打印值 47.35 差 1.4 倍，
并诚实标注"未解出"、拒绝拟合常数。**那个方法判断是对的**（拟合常数确实不可用），
但真实公式一直在安装目录里：

```
pillaiF_raw = pillai / ncc
pillaiF     = (pillaiF_raw * dfp2) / (dfp1 * (1 - pillaiF_raw))
```

即 Pillai 的 F **不是** `(V/s)(df2/df1)`，而是**带分母修正的比值变换**。
1.3996 这个"看似像 7/5"的倍数只是 `dfp2/(dfp1(1-V/ncc))` 在该设计下的巧合取值——
Lead 复核时枚举了 7350 个简单闭式都等于 7/5，证实它**不具鉴别力**。

> **方法论教训（第五次，且最隐蔽）**：本项目已有"读 ado 源码胜过从输出反推"的先例
> （GAP_REGISTER §8.17 `heckman`、§8.19 `xtgls`）。这次同一教训以更隐蔽的形式重现：
> 先用数值反解，**正确地**判断"不能用拟合常数"，然后**停在那里**——没去读源码。
> **"拒绝伪造"和"已经查清"是两件不同的事**：前者只是不做错，后者才关闭问题。
> 正确的下一步始终是：装着的 Stata 里就有答案。

### 3.7.5 对成本判定的修正（重要）

本报告 §4 A10 与 SUMMARY.md 把 Pillai/LH/Roy 归为"**算了未存**（低成本）"，
理由写"`B`/`W` 已在 systems.py:1143-1152"。**这个理由不成立。**

源码证明四个统计量走 `e(ccorr)`（典型相关），而典型相关是 `W⁻¹B` 的
**广义特征值问题**；westata 现有的 Wilks 路径（`systems.py:1161`，直接 `|W|/|W+B|`）
**从不构造 `ccorr`**。修正后的判定：

| 项 | 本节原判 | 修正后 |
|---|---|---|
| 四个统计量 | 算了未存（i 级） | **需新计算（ii 级）**：`eig(W⁻¹B)` 广义特征问题，`scipy.linalg.eigh(a, b)` 可直接解，`B`/`W` 已有 |
| 四条 F + 八个 df | 部分未解出 | **全部解出，公式见 §3.7.2，A 级** |
| `p*q == 2` 时 `t = 1` | 未记录 | **必须实现**，否则 p=q=2 时 Wilks 的 `df2`/F 会错（本例正是 p=q=2） |

**顺带一个待查项**：`systems.py:1163-1186` 现有 Wilks 的 `df2` 是否处理了
`p*q == 2` 特判，Lead 未读该段，**不下结论**，建议实现前先核对。

### 3.7.6 本节对 Q5 结论的修正汇总

| 问题 | 本节原判 | Lead 核实后 |
|---|---|---|
| Pillai 的 F 参数化 | ❌ 未解出，建议留作已知差距 | ✅ **已解出**，`(V/ncc·dfp2)/(dfp1(1-V/ncc))` |
| 四个统计量的成本 | 算了未存（i 级） | 需新计算（ii 级），但公式已完全确定 |
| `B`/`W` 是否足够 | 是 | **否**，还需典型相关 |
| 是否值得做 | 存疑 | **值得**——公式来自 Stata 源码而非拟合，可对拍 |

### Q6：`frontier` 的 `/lnsig2v`、`/lnsig2u`、Wald chi2、边界 LR —— 当前源码状态

**GAP_REGISTER.md:303 记载 P11 已修复。本次逐行核实：确认已修复，且超出当时记录的范围。**

| Stata 项 | 源码状态 | 证据 |
|---|---|---|
| `/lnsig2v` 参数行 | ✅ | `param_names=list(self._param_names) + ["/lnsig2v", "/lnsig2u"]`（nonlinear.py:1655） |
| `/lnsig2u` 参数行 | ✅ | 同上 |
| 两者作为**可推断参数**参与系数表与 CI | ✅ | `theta = [beta..., lnsig2v, lnsig2u]`（nonlinear.py:1570），`covariance` 覆盖全部 `k+2`（nonlinear.py:1588） |
| `Wald chi2(2) = 4462.60` | ✅ | `e(chi2)` + `e(chi2type)="Wald"` + `e(p)`（nonlinear.py:1634-1636），`wald = b_s' V_ss^-1 b_s`（nonlinear.py:1622） |
| `Log likelihood = -96.054179` | ✅ | `e(ll)`（nonlinear.py:1632） |
| **`LR test of sigma_u=0: chibar2(01) = 23.60 Prob >= chibar2 = 0.000`** | ✅ | `e(lr_chi2)` + `e(p_lr)`，其中 `p_lr = 0.5*chi2.sf(lr, 1)`（nonlinear.py:1637-1639），注释明写 "Stata's test is one-sided at the sigma_u = 0 boundary: chibar2(01)"（nonlinear.py:1638） |
| `sigma_v`/`sigma_u`/`sigma2`/`lambda` 四行派生统计 | ✅ `e()` | nonlinear.py:1640-1646 |
| `e(ll_0)`（OLS 对数似然，LR 的分母） | ✅ | nonlinear.py:1633，`ll_ols` 在 nonlinear.py:1611-1615 |
| `Iteration k: Log likelihood = ...` | ❌ | 无 |
| **表内 4 行派生统计的渲染**（Stata 把它们打印在系数表内，p10_outputs.txt:165-168） | ❌ | 它们是 `e()` 标量而非参数，`summary()` 不渲染 |

**结论：Q6 的记载（GAP_REGISTER.md:303）与源码一致，`/lnsig2v`、`/lnsig2u`、Wald chi2、
边界 LR（chibar2）全部已在 `e()` 与参数表中。剩余缺口只在渲染层（LR 测试行不打印、
4 行派生统计不进表）。**

---

## 4. 优先级排序

### (i) 纯数据可用性（已计算，只差一个字典键 —— 零成本）

| # | 项 | 改动点 | 收益 |
|---|---|---|---|
| A1 | `sqreg`/`bsqreg` 补 `sum_adev`/`sum_rdev`/`r2_p`/`q` | quantile_boot.py:143-151 的 `e` 字典；公式已在 quantile.py:394-403 | 关闭 sqreg/bsqreg 的 Pseudo R² 块（p8_outputs.txt:126-128, 159-161） |
| A2 | `rreg` 存逐次迭代轨迹 | quantile.py:804,842 的 `iterations.append(...)` 已在 `_iterate` 里，`fit` 只取 `len()`（quantile.py:743） | 关闭 `Huber/Biweight iteration k:` 行（p8_outputs.txt:174-179） |
| A3 | `teffects` 补 `nobs` | treatreg_effects.py:1213 的 `n` 现成；`TreatmentEffectResult` 加字段 | 关闭 `Number of obs`（p11_outputs.txt:8） |
| A4 | `teffects nnmatch` 补 `nneighbor`/`distance_metric` | treatreg_effects.py:1085 | 关闭 `Matches:`/`Distance metric:` 块（p11_outputs.txt:106-108） |
| A5 | `jackknife` 补 `e(reps)`（= N） | resample.py:477-486 | 关闭 `Replications = 400`（p13_outputs.txt:72） |
| A6 | `bootstrap`/`jackknife` 把 `_stored()` 的 `rmse`/`rss`/`mss`/`r2` 传进 `e` | resample.py:269-277 → 346-353 | 关闭 bootstrap 内嵌 regress 头部（p13_outputs.txt:42-46） |
| A7 | `mvreg` 补 `e(ll)` | systems.py:741-752；公式同 systems.py:555-560 | 与 sureg 对称 |
| A8 | `didregress` 把 group/time 的名与最小最大期暴露到结果 | did.py:447-448（模型上已有） | 关闭 `Treatment and time information` 块（p11_did.txt:5-19） |
| A9 | `sureg` 补 Breusch-Pagan 独立性统计量 | `e(corr)` 已在 systems.py:548 | `sureg, corr` 的检验 |
| A10 | `manova` 先算典型相关 `ccorr = eig(W⁻¹B)`，再由 §3.7.2 的四条公式产出 Pillai/Lawley/Roy | **注意**：不是"B/W 已有所以免费"——`ccorr` 是新计算（`scipy.linalg.eigh`），公式已由 Stata 源码完全确定 | 关闭 p10_outputs.txt:131-133 三行 |

### (ii) 需要新计算

| # | 项 | 说明 |
|---|---|---|
| B1 | **`svy:` 的 `deff` / `DEFF` mean** | `V_srs` 需按 `w` 与样本算；`V_design` 已有（survey.py:444-520） |
| B2 | **`svy:` 的 `mse` 选项** | `e(mse)` = 用 `mse` VCE 变体重算线性化方差 |
| B3 | **`svy:` 的 `subpop()`** | 子总体方差需把非子总体观测的贡献置零再线性化；GAP_REGISTER.md:158 已列 |
| B4 | **`svy: proportion` 的 logit 变换 CI** | p12_outputs.txt:51-56 的 `Logit` 列 |
| B5 | `sureg, isure` 的迭代轨迹 | systems.py:413-419 目前直接拒绝 |
| B6 | `mvreg` 的四个多元整体检验（Wilks/Pillai/LR/H/Roy） | 与 B5/manova 共享同一套公式 |
| B7 | ~~`manova` 三个统计量的 F 近似参数化~~ | **已关闭**（§3.7）：从 `_canon.ado:676-749` 读出全部四条 F 与八个 df，并用 Stata 打印值逐条验证命中。剩余只是实现 |
| B8 | `teffects` 的 `EE criterion` 迭代行 | 需把优化器历史带出来 |

### (iii) 需要新 display / render

| # | 项 | 说明 |
|---|---|---|
| C1 | **`summary.py` 按 `model_name` 分支而非按 `r2_p` 键名分支** | 修 §2.1 的契约耦合缺陷；这是**所有家族**的根因 |
| C2 | `qreg` 头部块（Raw/Min sum + Pseudo R² 右侧布局 + quantile 标题） | 数据齐（A1 之外 qreg 本身已齐） |
| C3 | `sqreg` per-quantile Pseudo R² 块 + `q25/q50/q75` 分组表头 | 数据齐（A1 后） |
| C4 | `rreg` 头部块（F/Prob>F 右置，无 R²/RMSE） | 数据齐 |
| C5 | `svy:` 头部块（strata/PSU/obs/pop/design df 五行 + `Survey: X estimation` 标题） | **数据 100% 齐，只差渲染** |
| C6 | `didregress` 头部块（time-info 表 + `Data type:` + `Note:` + cluster 行） | 数据基本齐（A8 后） |
| C7 | `teffects` 头部块（`Estimator:`/`Outcome model:`/`Treatment model:` + `Robust`/`AI robust` 表头） | 数据基本齐（A3/A4 后） |
| C8 | `bootstrap`/`jackknife`/`permute` 的 `Replications` 行 + `Observed/Bootstrap` 列标题 + `Command:` 行 + 进度行 | 数据齐（A5/A6 后） |
| C9 | `frontier` 表内 4 行派生统计 + `LR test of sigma_u=0: chibar2(01)` 行 | 数据 100% 齐 |
| C10 | `manova` 的 `e/a/u` 图例行 + `.summary()` 方法 | 数据齐（A10 后） |
| C11 | 迭代轨迹行渲染（qreg/rreg/nl/frontier/teffects/sureg） | A2/B5/B8 后 |
| C12 | `ManovaResult`/`TreatmentEffectResult` 加 `.e`-like mapping 或 `.summary()` | 让二者进入 `etable`/后估计生态 |

**建议动手顺序**：C1（根因，一次性）→ C5 + A1/A2/A5/A6（数据+渲染都齐，纯赚）→ C2/C3/C4 →
C9 → A3/A4 + C7 → A8 + C6 → C8 → A10 + C10 → B1/B2（deff/mse）→ B3/B4/B5。
（B7 已由 §3.7 关闭，不再列入路线。）

> **给 Lead 的一条交叉提示**：A1（`sqreg`/`bsqreg` 的 Pseudo R²）与 Lead 报告 §1.2 指出的
> "`qreg` 的 `r2_p` 键名歧义"是同一处根因的两面——`quantile.py:417` 让 `r2_p` 同时承载
> "ML 的 McFadden"与"qreg 的 check-function 伪 R²"两种语义。
> 修 C1 时若只改渲染分支，A1 必须同时补上 `sqreg`/`bsqreg` 的 `r2_p`，否则 sqreg 头部块仍缺数据。

---

## 5. 附录

### 5.1 本次实跑的 `e()` 全量键（用于核对 §1 各表）

```
qreg .25 iid : N bwidth bwmethod convcode denmethod df_m df_r f_r q q_v r2_p rank
               sparsity sum_adev sum_rdev vce
rreg         : F N N_weighted aa biweight_c df_m df_r dropped huber_c iterations
               lambda mse mss r2 r2_a rank rmse rss scale tolerance tune
sqreg(2 q)   : N df_m df_r quantiles rank reps vce
bsqreg       : N df_m df_r q quantiles rank reps vce
sureg        : N chi2_1 chi2_2 cons_1 cons_2 cons_y1 cons_y2 corr df_m df_m_1 df_m_2
               df_m_y1 df_m_y2 df_r equations k k_eq ll method mss_1 mss_2 mss_y1
               mss_y2 p_1 p_2 r2_1 r2_2 r2_y1 r2_y2 rank rmse_1 rmse_2 rmse_y1
               rmse_y2 rss_1 rss_2 rss_y1 rss_y2 sigma vce
mvreg        : F F_y1 F_y2 N corr df_m df_r k k_eq outcomes p_F p_y1 p_y2 r2 r2_y1
               r2_y2 rank rmse rmse_y1 rmse_y2 sigma vce
eivreg       : F N df_m df_r p r2 rank reliab rmse vce
manova       : (无 .e) wilks_lambda df df1 df2 F p term nobs groups statistics
svy: mean    : N N_pop N_psu N_strata df_m df_r rank singleunit vce
svy: regress : F N N_pop N_psu N_strata df_m df_r p r2 rank singleunit vce
svy: logit   : F N N_pop N_psu N_strata df_m df_r iterations ll p rank singleunit vce
jackknife    : N df_m df_r estimator expected_gap note rank refit vce
bootstrap    : N df_m df_r estimator note rank reps seed vce
bootstrap(expr) : + expressions resid_kind
permute      : N df_m df_r estimator note permuted pvalue_permuted rank reps seed vce
didregress   : F N N_clust data_type df_m df_r estimand ll r2 r2_a rank rmse
frontier     : N chi2 chi2type cmd cmdline converged df_m df_r distribution gamma
               lambda ll ll_0 lr_chi2 p p_lr rank sigma2 sigma2_u sigma2_v sigma_u
               sigma_v
nl           : N cmd cmdline converged deviance df_m df_r has_constant ll mss
               n_parameters r2 r2_a rank rmse rss sigma tss
teffects     : (无 .e) estimand estimate std_err z pvalue ci_lower ci_upper
               potential_outcomes se_method
```

### 5.2 一处需要单独排查的数值异常（非展示问题）

实跑 `Eivreg.from_formula("y1 ~ x1 + x2", frame, reliab={"x1": 0.7}).fit()` 得到
`e(r2) = 1.12706`、`e(rmse) = 0.0`——R² > 1 且 RMSE 为 0。
根因在 `sigma2 = (y'y - n b'Ab)/(n-k)`（systems.py:962）与 `r2 = 1 - sigma2·dof/tss`（systems.py:964）：
当 `reliab` 使 `A = X'X/n - S` 的修正足够大时，`y'y - n b'Ab` 可能为负。
`max(sigma2, 0)` 只保护了 `rmse`（systems.py:985），没有保护 `r2`（systems.py:964 未 clamp）。
**这条不属于本报告范围（不是"缺失的辅助统计"），但是同一文件里的真缺陷，建议转交。**
（注：实跑用的 `_sur()` 帧与 `tests/test_systems.py:499` 的 `_eiv()` 不同，故未复现测试中的数值；
测试通过不代表该边界不存在。）

### 5.3 已清理

本次在仓库根创建的临时脚本 `probe_gaps_tmp.py`、`probe_gaps2_tmp.py`、`probe_gaps3_tmp.py`、
`probe_gaps4_tmp.py`、`probe_gaps5_tmp.py` 已全部删除。未修改 `src/`、`tests/`、`docs/` 下任何文件。
