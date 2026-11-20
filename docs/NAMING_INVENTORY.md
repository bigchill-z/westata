# westata 类与函数命名清单（全量盘点）

> 由 `ast` 扫描 `src/westata/**/*.py` 得到：**127 个模块**、**221 个类**（193 个公开）、
> **795 个模块级函数**（174 个公开），顶层 `westata.__all__` 导出 **253** 个名字。
> 目的：为「统一命名规范」的讨论提供**事实基线**。私有名（`_` 开头）按
> `docs/API_CONVENTIONS.md` R1.1 不受公共规范约束，这里只作计数，不逐条列出。

---

## 1. 顶层导出名按「词汇来源」分类 —— 问题的核心

同一层（`ws.*`）混用了**至少三套**词汇体系：Stata 命令原名、统计学术语、通用英文词，
且彼此没有稳定的取舍规则。下表是 253 个顶层导出按来源的归类。

| 类别 | 数量 | 代表名字 |
|---|---:|---|
| **A. Stata `estat` 命令** | 22 | `estat_vce` `estat_heteroskedasticity_test` `estat_summarize` `estat_auc` … |
| **B. Stata `xt*` 面板命令** | 23 | `PanelAbond` `PanelGls` `PanelLogit` `panel_set` `panel_describe` `panel_summarize` `panel_serial` … |
| **C. Stata `svy*` 调查命令** | 5 | `SurveyDesign` `SurveyMean` `SurveyRegress` `SurveyLogit` … |
| **D. Stata `var`/`vec` 时序命令** | 20 | `VAR` `VECM` `varbasic` `VarBasic` `var_select_order` `vec_rank` `VarGrangerResult` … |
| **E. 结果对象 `*Result(s)`** | 54 | `AucResult` `StataResults` `EReturn` `RdRobustResult` `TefectsRaResult` … |
| **F. 表格/行对象 `*Table/*Row`** | 5 | `IrFTable` `MarginsTable` `PwCompareTable` `SummarizeTable` `SzroeterTable` |
| **G. Stata `teffects` 族** | 6 | `TreatmentEffectsRa` `TreatmentEffectsPsmatch` `TreatmentEffectsNnmatch` + 同名函数 |
| **H. 其他（统计术语 / 通用词 / Stata 单词命令）** | 114 | `Regress` `Logit` `newey_west` `prais` `synth` `margins` `predict` `etable` … |

### 1.1 三套词汇并存的具体例子

| 概念 | Stata 原名写法 | 统计术语写法 | 通用/其他写法 |
|---|---|---|---|
| 最小二乘 | `regress` → 无 `Regress` 类 | — | `Regress` `PanelRegressFE` `RegressHighDimFixedEffects` `Areg` `PanelRegressBE` |
| 异方差稳健方差 | `vce(robust)` | `cov_hc` `cov_hac` | `cov_type="robust"` / 内部 `HC1` `ML_ROBUST` |
| 逻辑回归 | `logit` | `Logit` | `PanelLogit` `SurveyLogit` `MultinomialLogit` `OrderedLogit` |
| 汇总统计 | `summarize` | `SummarizeResult` | `summarize` `estat_summarize` `SummarizeTable` |
| 边际效应 | `margins` | `MarginsResult` | `margins` `MarginsTable` `MarginsResult` |
| 结果对象 | `e()` → `EReturn` | `*Result` | `StataResults` `*Results`（复数） |

**同一命令的函数/类/结果类三种写法**（`rd_robust` 一案）：

- 命令函数 `rd_robust()` → 命令类 `RDRobust` → 结果类 `RdRobustResult`（一个词三种大小写）。

---

## 2. 明显的不一致与拼写缺陷

以下问题**与采用哪种命名方案无关**，任何方案下都应作为「无争议修复」处理：

| 现象 | 位置 | 说明 |
|---|---|---|
| `TefectsRaResult`（缺一个 f） | `estimators/treatment_effects_ra.py` | 同模块命令类是 `TreatmentEffectsRa`，结果类拼成 `Tefects` |
| `IrFTable`（F 大写） | `varbasic.py` | 同模块 `IrfResult` 用 `Irf`，两者大小写不一致 |
| `RDRobust` / `RdRobustResult` / `rd_robust` | `estimators/rd_robust.py` | 一个词三种写法 |
| `SucestResult`（对应 `suest`） | `postestimation/suest.py` | 模块/函数叫 `suest`，结果类来源不明 |
| `VarBasic`（对应 `varbasic`） | `varbasic.py` | 与 `VAR` / `VARResults` 放在一起尤其突兀 |
| `VARResults` / `VECMResults` / `ContrastResults`（复数） | 多处 | 绝大多数结果类用单数 `*Result` |
| `StataResults` / `EReturn` | `results/base.py` `results/stata_return.py` | 基础类名直接以 Stata 命名，与 statsmodels 风格基类混用 |
| `statsmodels_wald_test` | `postestimation/api.py` | 以「来源库」命名的公共函数，全树独一份 |
| `test_as_stata` / `lincom_as_stata` / `etable_as_stata` | 多处 | `_as_stata` 后缀是第三套命名法 |
| `Mt19937_64` / `stata_runiform` | `random/stata_rng.py` | 一个用算法名，一个用 Stata 函数名 |
| 「方差协方差」四种叫法 | 全树 | `vce` / `cov` / `var` / `variance_covariance` |

---

## 3. 按族 grouping：命令类 / 命令函数 / 结果类 的配对关系

下表展示「一个 Stata 命令在本包里对应哪些 Python 名字」，是判断命名是否成体系的依据。
（命令函数即顶层可调用的函数；命令类即可 `.fit()` 的估计器；结果类即返回对象。）

| Stata 命令族 | 命令类 | 命令函数（薄包装/检验） | 结果类 |
|---|---|---|---|
| `regress` | `Regress` | — | `StataResults` |
| `xtreg` | `PanelRegressFE` `PanelRegressRE` `PanelRegressBE` `PanelRegressFD` | — | `StataResults` |
| `areg` / `reghdfe` | `RegressHighDimFixedEffects` `Areg` | — | `StataResults` |
| `ivregress` | `IV2SLS` `IVLIML` `IVGMM` | — | `StataResults` |
| `newey_west` | `Newey` | `newey_west()` | `NeweyWestResult` |
| `prais` | `Prais` | `prais()` `prais_winsten()` | `PraisResult` |
| `logit`/`probit`/`poisson`/`nbreg` | `Logit` `Probit` `Poisson` `NegativeBinomial` … | — | `StataResults` |
| `ologit`/`oprobit`/`mlogit` | `OrderedLogit` `OrderedProbit` `MultinomialLogit` | — | `StataResults` |
| `tobit`/`intreg`/`truncreg`/`heckman`/`etregress` | `Tobit` `IntervalRegress` `TruncatedRegress` `Heckman` `EndogenousTreatmentRegress` | — | `StataResults` |
| `qreg`/`rreg` | `QuantileRegress` `RobustRegress` | — | `StataResults` |
| `sqreg`/`bsqreg` | `SimultaneousQuantileRegress` `BootstrappedQuantileRegress` | — | `StataResults` |
| `xtlogit`/`xtprobit`/`xttobit`/`xtpoisson`/`xtnbreg` | `PanelLogit` `PanelProbit` `PanelTobit` `PanelPoisson` `PanelNbRegress` | — | `StataResults` |
| `xtivreg`/`xtgls`/`xtpcse`/`xtunitroot` | `PanelIVRegress` `PanelGls` `PanelPcse` `PanelUnitroot` | `levin_lin_chu()` `im_pesaran_shin()` | `PanelUnitRootResult` |
| `xtabond`/`xtdpd`/`xtdpdsys` | `PanelAbond` `PanelDpd` `PanelDpdSys` | — | `StataResults` |
| `didregress`/`xtdidregress` | `DIDRegress` `PanelDIDRegress` | — | `StataResults` |
| `teffects ra/ipw/aipw/nnmatch/psmatch` | `TreatmentEffectsRa` `IPW` `AIPW` `TreatmentEffectsNnmatch` `TreatmentEffectsPsmatch` | `teffects_*()` | `TefectsRaResult` `TreatmentEffectResult` `NnmatchResult` `PsmatchResult` |
| `synth`/`synth2` | `Synth` `Synth2` | `synth()` `synth2()` | `SynthResult` `Synth2Result` |
| `rd_robust` | `RDRobust` | `rd_robust()` | `RdRobustResult` |
| `var`/`vec`/`arima`/`garch` | `VAR` `VECM` `ARIMA` `GARCH` | `unitroot()` `dfuller()` `pperron()` | `VARResults` `VECMResults` `UnitRootResult` `JohansenResult` |
| `varbasic` | `VarBasic` | `varbasic()` | `IrfResult` `IrFTable` `ImpulseResponse` |
| `svy:*` | `SurveyMean` `SurveyTotal` `SurveyProportion` `SurveyRegress` `SurveyLogit` | — | `StataResults` |
| `suest`/`manova`/`sureg`/`mvreg`/`eivreg` | `Manova` `SeeminglyUnrelatedRegress` `MultivariateRegress` `ErrorsInVariablesRegress`（`suest` 仅有函数无类） | `suest()` | `SucestResult` `ManovaResult` `StataResults` |
| `bootstrap`/`jackknife`/`permute` | `Bootstrap` `Jackknife` `Permute` | — | — |
| `nl`/`frontier` | `Nl` `Frontier` | — | `StataResults` |
| `rcm` | `RegressionControlMethod` | `rcm()` | `RcmResult` |

观察：同一族内部基本能保持一致（如 `Xt*` 家族），但**族与族之间**的构造方式不统一
（有的既有类又有同名函数，如 `newey_west`/`prais`/`rd_robust`/`synth`；有的只有类；有的只有函数）。

---

## 4. 模块级公开函数按「命名动词」分类

去掉 `_` 私有函数后，174 个公开模块级函数大致分四种命名法：

| 命名法 | 例子 | 说明 |
|---|---|---|
| **Stata 命令原名（snake_case）** | `estat_*`（22）、`panel_set` `panel_describe` `panel_summarize`、`var_select_order` `vec_rank`、`dfuller` `kpss`、`margins` `predict`、`summarize` | 可与 Stata 反查，占多数 |
| **统计术语/动作（snake_case）** | `newey_west` `prais` `synth` `synth2` `rd_robust` `rcm` `suest` `linktest` `hausman` `lrtest`、`solve_lad` `build_design` `normalize_weights` | 无统一前缀，语义自解释 |
| **渲染/输出 `_as_stata` / `render_` / `to_`** | `etable_as_stata` `render_esttab` `render_estimates_table` `render_summary` `to_latex` `to_markdown` | 第三套后缀体系 |
| **`from_*` 构造入口** | `from_formula`（各估计类）、`from_variables`、`from_frame` | 与 R2.7/R3.2 相关，非顶层 |

---

## 5. 给「统一命名规范」讨论的事实约束

1. **不能脱离 Stata 反查性**：`docs/API_CONVENTIONS.md` R1.2 已定「有 Stata 对应命令时，
   Python 侧名字要能反查回 Stata」。因此 `estat_*` / `xt*` / `svy*` / `var*` 这类前缀不是
   随意风格，而是需求。任何方案必须保留「从 Python 名想到 Stata 命令」的路径。
2. **公共名受规范、私有名不受**：R1.1 明确 `_` 前缀函数/类、线性代数层 `vce.*` 允许保留
   数学记号。所以规范化对象应聚焦 253 个顶层导出 + 公共方法/属性。
3. **已有机械校验器**：`scripts/check_api_conventions.py` 覆盖 R1–R6，当前 `--strict` 通过
   （全部为已登记例外）。任何新命名规则应能落成一个可机械判定的检查，否则会重演
   「规范写了但自由生长 38 个版本」。
4. **253 个导出中约 54 个是 `*Result`**：结果类占最大单一类别，统一后缀（含单复数、
   `*Row`/`*Table` 的取舍）是收益最高、风险最低的切入点。
5. **破坏性变更需用户拍板**：大量顶层名字被 `retest/` 与外部脚本按名引用，改名必须
   考虑弃用别名（`docs/API_CONVENTIONS.md` 已有多处「保留一个版本弃用别名」的先例）。
