# westata 类/函数改名定稿映射

> 状态：**规则已定稿，映射已逐族核实，已于 0.3.41 实施**。每个 Stata 命令均由
> 5 个子智能体读取源码中的 `_stata_command` / `@model_name` / docstring 核实（非猜测）。
> 规则见 [NAMING_CONVENTIONS.md](NAMING_CONVENTIONS.md)；决策记录见
> [options.md 〇之十三](../options.md)；原始盘点见 [NAMING_INVENTORY.md](NAMING_INVENTORY.md)。
>
> **兼容**：下列所有改名，旧名保留一个版本 + `DeprecationWarning`；e-class 命令函数薄包装不删除。
>
> **机器化**：本表的机器可读真相源是 `scripts/naming_map.py`（`Rename` 数据表 +
> `result_rename()` 规则函数）；一致性由 `python scripts/check_naming.py --strict` 校验，
> 每条规则都能触发由 `tests/test_naming_conventions.py::test_every_rule_can_fire` 保证。
> 修改本表时**必须同步修改合同并跑校验器**，否则视为 drift。

---

## 0. 规则速查（定稿）

1. 类名/函数名以**对应 Stata 命令**为词源；生僻前缀展开、熟词保留。
2. 展开表：`xt→Panel`（函数 `xt`→`panel`）、`svy→Survey`、`teffects→TreatmentEffects`、
   `hdfe→HighDimFixedEffects`、`q→Quantile`、`r→Robust`。
3. **熟词保留（不展开）**：`DID`、`IV`、`RD`、`VECM`、`VAR`、`ARIMA`、`GARCH`、`ARCH`、`AR`、`MA`；
   `2SLS`/`LIML`/`GMM`、`FE`/`RE`/`BE`/`FD`/`MLE`、`nnmatch`/`psmatch`/`ra` 等作为熟词后缀保留。
   `estat_*` 后缀中的人名 `bgodfrey`/`moran`/`mundlak`/`szroeter`/`dwatson` 与
   `auc`/`gof`/`ic`/`vce`/`vif` 亦为熟词（规则 5）。
4. 特殊：`Regress→Regress`（Stata `regress`）；`regress→Regress`；`qreg→QuantileRegress`；
   `rreg→RobustRegress`。
5. **`estat_` 前缀保留，后缀按缩写展开**（用户 2026-11-18 澄清纠正）：
   `estat_parallel_trends→estat_parallel_trends`、`estat_effect_size→estat_effect_size`、
   `estat_first_stage→estat_first_stage`、`estat_heteroskedasticity_test→estat_heteroskedasticity_test`、
   `estat_information_matrix_test→estat_information_matrix_test`、`estat_overidentification→estat_overidentification`、
   `estat_omitted_variables_test→estat_omitted_variables_test`、`estat_correlogram→estat_correlogram`、
   `estat_weak_instrument_robust→estat_weak_instrument_robust`。
   后缀是人名（`bgodfrey`/`moran`/`mundlak`/`szroeter`/`dwatson`）或熟词
   （`auc`/`gof`/`ic`/`vce`/`vif`）者整体保留。
   **实施注意**：`estat_*` 同时是顶层函数与结果对象方法（API_CONVENTIONS R5.7 双向对齐），改名须两侧同步。
6. 结果类：**统一复数 `*Result→*Results`**（用户 2026-11-18 最终决定，推翻早先的统一单数）；
   `Xt*`/`Svy*` 前缀展开为 `Panel*`/`Survey*`；但不把估计类的新名联动到结果类。
   **顺序**：先前缀展开、再加 s（如 `XtSerialResult→PanelSerialResults`）。
   **撞名例外**：`ContrastResult`（postestimation/contrast.py:422，单行列对象）与
   `ContrastResults`（ecosystem.py:100，多行聚合）语义不同，两者**各自保留原名**，
   不适用本规则（用户决定登记为例外，见 §5）。
7. 撞名仲裁：teffects 族只改 `Teffects*` 包装类，`RegressionAdjustment`/
   `NearestNeighbourMatching`/`IPW`/`AIPW` 等独立估计器保留（合并属独立重构，见 options.md）。

---

## 1. 线性 / IV / HDFE / 面板 / Newey / Prais（fam-linear 核实）

| 现名 | Stata 命令（源码证据） | 定稿新名 |
|---|---|---|
| `Regress` | `regress`（linear.py:264 docstring, :845 model_name） | **`Regress`** |
| `IV2SLS` | `ivregress 2sls`（iv.py:407） | **`IV2SLS`**（保留，规则 3） |
| `IVLIML` | `ivregress liml`（iv.py:1377） | **`IVLIML`**（保留） |
| `IVGMM` | `ivregress gmm`（iv.py:1405） | **`IVGMM`**（保留） |
| `RegressHighDimFixedEffects` | `reghdfe`（hdfe.py:234 `_command_name`） | **`RegressHighDimFixedEffects`** |
| `Areg` | `areg`（hdfe.py:1667） | **`Areg`**（已是命令 PascalCase，保留） |
| `PanelRegressFE` | `xtreg, fe`（panel.py:396） | **`PanelRegressFE`** |
| `PanelRegressRE` | `xtreg, re`（panel.py:2060） | **`PanelRegressRE`** |
| `PanelRegressMLE` | `xtreg, mle`（panel.py:2555） | **`PanelRegressMLE`** |
| `PanelRegressBE` | `xtreg, be`（panel.py:2744） | **`PanelRegressBE`** |
| `PanelRegressFD` | `xtreg, fd`（panel.py:2827） | **`PanelRegressFD`** |
| `Newey` | `newey_west`（newey_west.py:419 docstring） | **`Newey`**（保留） |
| `Prais` | `prais`（prais.py:33 docstring） | **`Prais`**（保留） |
| `NeweyWestResult` / `PraisResult` | 结果类 | → `NeweyWestResults` / `PraisResults`（规则 6 统一复数，见 §5） |

## 2. 离散 / 计数 / limited / 分位数 / 非线性（fam-discrete 核实）

| 现名 | Stata 命令（源码证据） | 定稿新名 |
|---|---|---|
| `Logit` / `Probit` / `Poisson` | `logit` / `probit` / `poisson` | 保留（命令同形） |
| `NegativeBinomial` | `nbreg`（discrete.py:1605） | **`NegativeBinomial`**（已展开，保留） |
| `NegativeBinomialNB1` | `nbreg, dispersion(constant)`（discrete.py:1429） | **`NegativeBinomialNB1`**（保留，避免与上行撞名；NB1 是熟词变体后缀） |
| `OrderedLogit` / `OrderedProbit` / `MultinomialLogit` | `ologit` / `oprobit` / `mlogit`（discrete.py） | 保留（已按规则展开） |
| `ZeroInflatedPoisson` / `ZeroInflatedNegativeBinomial` | `zip` / `zinb`（counts.py:916/933） | 保留（已展开） |
| `TruncatedPoisson` / `TruncatedNegativeBinomial` | `ztp`/`tpoisson`、`tnbreg`（counts.py:1085/1094） | 保留（已展开） |
| `TruncatedRegress` | `truncreg`（limited.py:698） | **`TruncatedRegress`** |
| `IntervalRegress` | `intreg`（limited.py:915） | **`IntervalRegress`** |
| `Heckman` | `heckman`（limited.py:1572） | 保留（命令同形） |
| `EndogenousTreatmentRegress` | `etregress`（limited.py:2098） | **`EndogenousTreatmentRegress`** |
| `Tobit` | `tobit`（tobit.py:98） | 保留（命令同形） |
| `QuantileRegress` | `qreg`（quantile.py:313） | **`QuantileRegress`** |
| `RobustRegress` | `rreg`（quantile.py:625） | **`RobustRegress`** |
| `SimultaneousQuantileRegress` | `sqreg`（quantile_boot.py:56 "Simultaneous-quantile regression"） | **`SimultaneousQuantileRegress`** |
| `BootstrappedQuantileRegress` | `bsqreg`（quantile_boot.py:337 "Bootstrapped quantile regression"） | **`BootstrappedQuantileRegress`** |
| `Nl` | `nl`（nonlinear.py:943） | **`Nl`**（保留，命令即 `nl`） |
| `Frontier` | `frontier`（nonlinear.py:1336） | 保留（命令同形） |

## 3. 面板 xt* / DID / 调查 svy* / teffects / RD / synth（fam-panel 核实）

| 现名 | Stata 命令（源码证据） | 定稿新名 |
|---|---|---|
| `PanelAbond` | `xtabond`（panel_dynamic.py:203） | **`PanelAbond`** |
| `PanelDpd` / `PanelDpdSys` | `xtdpd` / `xtdpdsys`（:522/:534） | **`PanelDpd`** / **`PanelDpdSys`** |
| `DynamicPanelInstruments` | 无命令（helper） | 保留 |
| `PanelIVRegress` | `xtivreg`（panel_iv.py:531） | **`PanelIVRegress`** |
| `PanelGls` / `PanelPcse` / `PanelUnitroot` | `xtgls`/`xtpcse`/`xtunitroot`（:1074/:1568/:2120） | **`PanelGls`** / **`PanelPcse`** / **`PanelUnitroot`** |
| `PanelLogit`/`PanelProbit`/`PanelTobit`/`PanelPoisson` | `xtlogit`…（panel_nonlinear.py） | **`PanelLogit`** / **`PanelProbit`** / **`PanelTobit`** / **`PanelPoisson`** |
| `PanelNbRegress` | `xtnbreg`（:1936） | **`PanelNbRegress`**（nb→nb 熟词, reg→Regress） |
| `DIDRegress` | `didregress`（did.py:461） | **`DIDRegress`** |
| `PanelDIDRegress` | `xtdidregress`（did.py:975） | **`PanelDIDRegress`** |
| `SurveyDesign` | `svyset`（survey.py:149，设计对象） | **`SurveyDesign`**（保留，语义清晰；非估计类） |
| `SurveyLayout` | 无命令 | 保留 |
| `SurveyMean`/`SurveyTotal`/`SurveyProportion`/`SurveyRegress`/`SurveyLogit` | `svy: mean`/`total`/`proportion`/`regress`/`logit` | **`SurveyMean`**/ **`SurveyTotal`**/ **`SurveyProportion`**/ **`SurveyRegress`**/ **`SurveyLogit`** |
| `TreatmentEffectsRa` | `teffects ra`（treatment_effects_ra.py:140） | **`TreatmentEffectsRa`** |
| `TreatmentEffectsNnmatch` | `teffects nnmatch`（nnmatch.py:83） | **`TreatmentEffectsNnmatch`** |
| `TreatmentEffectsPsmatch` | `teffects psmatch`（psmatch.py:195） | **`TreatmentEffectsPsmatch`** |
| `IPW`/`AIPW`/`RegressionAdjustment`/`NearestNeighbourMatching` | teffects ipw/aipw/ra/nnmatch | 保留（独立估计器，见规则 7） |
| `RDRobust` | `rd_robust`（rd_robust.py:330） | **`RDRobust`**（RD 熟词保留） |
| `Synth` / `Synth2` | `synth` / `synth2`（命令本身） | 保留（命令同形） |
| `RegressionControlMethod` | `rcm` = regression control method（rcm.py:1,129） | **`RegressionControlMethod`** |

## 4. 时序 / 系统 / 重抽样 / varbasic（fam-ts 核实）

| 现名 | Stata 命令（源码证据） | 定稿新名 |
|---|---|---|
| `ARIMA` / `VAR` / `GARCH` | `arima` / `var` / `arch`（tseries.py） | 保留（熟词；GARCH 见下注） |
| `VECM` | `vec`（tseries.py:2809） | **`VECM`**（保留，规则 3 熟词扩充） |
| `SeeminglyUnrelatedRegress` | `sureg`（systems.py:407） | **`SeeminglyUnrelatedRegress`** |
| `MultivariateRegress` | `mvreg`（systems.py:774） | **`MultivariateRegress`** |
| `ErrorsInVariablesRegress` | `eivreg`（systems.py:1009） | **`ErrorsInVariablesRegress`** |
| `Manova` | `manova`（systems.py:1603） | 保留（命令同形） |
| `Bootstrap` / `Jackknife` / `Permute` | 同名命令（resample.py） | 保留 |
| `VarBasic` | `varbasic`（varbasic.py:1,476） | **`VarBasic`**（大小写规范；VAR 熟词保留） |

> **GARCH 辨析**：`GARCH._stata_command = "arch"`（tseries.py:4205），但 Stata 的 `arch` 命令覆盖
> ARCH/GARCH/EGARCH 整个条件异方差族，且 docstring 写明 "GARCH / EGARCH volatility model"；
> ARCH 与 GARCH 均在熟词保留表 → 保留 `GARCH`。

## 5. 结果类（跨族，统一规则 6：全部 `*Result → *Results`）

**63 个单数结果类全部加 s**（源码 AST 扫描确认，行号见下）。其中带 `Xt*` 前缀者先展开前缀；
拼写硬伤者一并修正。以下为完整清单：

| 现名 | 定稿新名 | 现名 | 定稿新名 |
|---|---|---|---|
| `NeweyWestResult` | `NeweyWestResults` | `NnmatchResult` | `NnmatchResults` |
| `PanelUnitRootResult` | `PanelUnitRootResults` | `PraisResult` | `PraisResults` |
| `PsmatchResult` | `PsmatchResults` | `RdRobustResult` | `RdRobustResults` |
| `SynthResult` | `SynthResults` | `Synth2Result` | `Synth2Results` |
| `ManovaResult` | `ManovaResults` | `TefectsRaResult` | `TreatmentEffectsRaResults` ¹ |
| `TreatmentEffectResult` | `TreatmentEffectResults` | `JohansenResult` | `JohansenResults` |
| `UnitRootResult` | `UnitRootResults` | `IpsResult` | `IpsResults` |
| `LlcResult` | `LlcResults` | `XtSetResult` | `PanelSetResults` ² |
| `XtDesResult` | `PanelDesResults` ² | `XtSumResult` | `PanelSumResults` ² |
| `AucResult` | `AucResults` | `BgTestResult` | `BgTestResults` |
| `ClassificationResult` | `ClassificationResults` | `ContrastResult` | **`ContrastResult`（保留，撞名例外）** ³ |
| `DFGLSResult` | `DFGLSResults` | `RegFResult` | `RegFResults` |
| `EndogScoreResult` | `EndogScoreResults` | `EndogenousResult` | `EndogenousResults` |
| `EffectSizeResult` | `EffectSizeResults` | `FirstStageResult` | `FirstStageResults` |
| `GofResult` | `GofResults` | `LfitResult` | `LfitResults` |
| `HetTestResult` | `HetTestResults` | `IcResult` | `IcResults` |
| `ImTestResult` | `ImTestResults` | `KpssResult` | `KpssResults` |
| `LincomResult` | `LincomResults` | `LinkTestResult` | `LinkTestResults` |
| `MarginsResult` | `MarginsResults` | `LRTestResult` | `LRTestResults` |
| `HausmanResult` | `HausmanResults` | `MoranResult` | `MoranResults` |
| `MTestResult` | `MTestResults` | `MundlakResult` | `MundlakResults` |
| `OverIdResult` | `OverIdResults` | `OvTestResult` | `OvTestResults` |
| `RcmResult` | `RcmResults` | `DurbinWatsonResult` | `DurbinWatsonResults` |
| `PortmanteauResult` | `PortmanteauResults` | `SucestResult` | `SuestResults` ⁴ |
| `SummarizeResult` | `SummarizeResults` | `SzroeterResult` | `SzroeterResults` |
| `VarGrangerResult` | `VarGrangerResults` | `LmarResult` | `LmarResults` |
| `VarSocResult` | `VarSocResults` | `VarStableResult` | `VarStableResults` |
| `VarWleResult` | `VarWleResults` | `VecRankResult` | `VecRankResults` |
| `VecStableResult` | `VecStableResults` | `VifResult` | `VifResults` |
| `WeakRobustResult` | `WeakRobustResults` | `XtOveridResult` | `PanelOveridResults` ² |
| `XtSerialResult` | `PanelSerialResults` ² | `XtTest0Result` | `PanelTest0Results` ² |
| `IrfResult` | `IrfResults` | — | — |

注释：
- ¹ `TefectsRaResult → TreatmentEffectsRaResults`：叠加拼写修正 + `teffects→TreatmentEffects`。
- ² `Xt*` 前缀按规则展开为 `Panel*` 后加 s。
- ³ **`ContrastResult` 保留**：`ContrastResults`（ecosystem.py:100）已存在且语义不同（多行聚合），
  若把 `ContrastResult`（contrast.py:422，单行列对象）改成 `ContrastResults` 会撞名。两者按用户决定
  各自保留原名，作为统一复数规则的**登记例外**。
- ⁴ `SucestResult → SuestResults`：叠加拼写硬伤修正。

**已是复数、保持不动**：`VARResults`、`VECMResults`、`StataResults`、`StoredResults`、
`ContrastResults`（例外）、私有 `_CountResults`/`_SystemResults`。

**非 `*Result` 的后缀不受本规则约束**：`SummarizeTable`、`PwCompareTable`、`MarginsTable`、
`SzroeterTable`、`IrFTable`（→ `IrfTable` 大小写规范）、`BalanceRow`、`*Table`/`*Row` 等保持各自后缀；
其中 `IrFTable → IrfTable` 是大小写规范（与 `IrfResults` 的 `Irf` 一致）。

## 6. 公共函数（fam-func 核实；estat_* 与熟词函数按定稿规则裁剪）

### 6.1 改名（缩写展开）

| 现名 | 定稿新名 | 依据 |
|---|---|---|
| `panel_set` / `panel_describe` / `panel_summarize` | **`panel_set`** / **`panel_describe`** / **`panel_summarize`** | xt→panel |
| `newey_west` | **`newey_west`** | 命令 `newey_west` = Newey-West（薄包装保留作弃用别名） |
| `treatment_effects_ra` / `treatment_effects_nnmatch` / `treatment_effects_psmatch` | **`treatment_effects_ra`** / **`treatment_effects_nnmatch`** / **`treatment_effects_psmatch`** | teffects→treatment_effects |
| `rd_robust` | **`rd_robust`** | rd→rd（熟词保留），加分隔（薄包装保留作弃用别名） |
| `panel_serial` / `panel_overid` / `panel_test0` | **`panel_serial`** / **`panel_overid`** / **`panel_test0`** | xt→panel |
| `var_select_order` | **`var_select_order`** | soc→select_order |
| `var_granger` / `var_lmar` / `var_stable` / `var_wle` | **`var_granger`** / **`var_lmar`** / **`var_stable`** / **`var_wle`** | 熟词 VAR 保留 + 分隔 |
| `vec_lmar` / `vec_rank` / `vec_stable` | **`vec_lmar`** / **`vec_rank`** / **`vec_stable`** | 熟词 VEC 保留 + 分隔 |
| `df_gls` | **`df_gls`** | DF 熟词 + 分隔 |
| `pairwise_compare` | **`pairwise_compare`** | pw→pairwise |
| `treatment_balance_summarize` | **`treatment_balance_summarize`** | te→treatment |

#### estat_* 后缀展开（`estat_` 前缀保留；用户 2026-11-18 澄清）

| 现名 | 定稿新名 | 依据 |
|---|---|---|
| `estat_parallel_trends` | **`estat_parallel_trends`** | ptrends→parallel_trends |
| `estat_effect_size` | **`estat_effect_size`** | esize→effect_size |
| `estat_first_stage` | **`estat_first_stage`** | firststage→first_stage |
| `estat_heteroskedasticity_test` | **`estat_heteroskedasticity_test`** | hettest→heteroskedasticity_test |
| `estat_information_matrix_test` | **`estat_information_matrix_test`** | imtest→information_matrix_test |
| `estat_overidentification` | **`estat_overidentification`** | overid→overidentification |
| `estat_omitted_variables_test` | **`estat_omitted_variables_test`** | ovtest→omitted_variables_test |
| `estat_correlogram` | **`estat_correlogram`** | corrgram→correlogram |
| `estat_weak_instrument_robust` | **`estat_weak_instrument_robust`** | weakrobust→weak_instrument_robust |

### 6.2 保持原名

- **`estat_*` 中后缀是人名或熟词者**（前缀 `estat_` 保留、后缀不动）：`estat_auc`、
  `estat_gof`、`estat_ic`、`estat_vce`、`estat_vif`（熟词）；`estat_bgodfrey`、
  `estat_moran`、`estat_mundlak`、`estat_szroeter`、`estat_dwatson`（人名）；
  `estat_classification`、`estat_endogenous`、`estat_summarize`（已是完整词）。
  注意 `estat_dwatson`/`estat_correlogram` 在 Stata 里其实是 `dwatson`/`corrgram`（无 estat 前缀），
  本包沿用了 `estat_` 命名；dwatson 属人名保留，corrgram 后缀按上表展开。
- 熟词/命令同形/描述性函数：`dfuller` `pperron` `kpss` `unitroot` `levin_lin_chu`
  `im_pesaran_shin` `prais` `prais_winsten` `synth` `synth2` `rcm` `linktest` `suest`
  `margins` `contrast` `predict` `summarize` `supported_types` `lincom_as_stata`
  `test_as_stata` `endog_regf_test` `endog_score_test` `check_time_spacing`
  `normalize_weights` `parse_exp_list` `render_*` `to_*` `etable` `ssc` `wntestq`。

---

## 7. 尚需领队/用户留意的核实发现（未在本次改名内）

- **`GARCH` 的 `_stata_command` 是 `arch`**：已辨析保留 `GARCH`（见 §4 注）。
- **`TruncatedPoisson` 有两个命令**：`ztp`（默认）与 `tpoisson`（有上下界时动态切换，
  counts.py:990-994）；现名覆盖两者，不改名。
- **`RDRobust`/`rd_robust` 是完整 Stata 命令/包名**（Calonico-Cattaneo-Titiunik），
  本按熟词 RD 处理（`RDRobust`/`rd_robust`）；若希望完全保留命令同形，可维持 `RDRobust`，
  待用户确认。
- **teffects 合并**：用户选「合并重复类」，但源码证实是两个独立实现（见 options.md 纠偏记录），
  本次按「只改 Teffects* 包装类」执行，合并列为独立功能重构任务。

---

## 8. 实施校验清单（未执行）

- [ ] 机械校验器：把规则 0 的 7 条落成可判定检查（仿 check_api_conventions.py）。
- [ ] 冲突检查：全树 grep 新旧名，确认无未预期撞名（尤其 `*Result` 单复数、`RDRobust`）。
- [ ] 每批改名后：`pytest -q` 全量（当前基线 6655 passed）+ `retest/` 约 180 处按名调用。
- [ ] 文档：README/README.zh-CN、GUIDE 中英、API 中英（生成器重渲染）、API_CONVENTIONS R1.2 改写。
- [ ] 发布：bump 版本，PyPI + GitHub；弃用别名保留一个版本。

**本文件状态的诚实声明：映射已核实、未实施、未跑改名后测试、未发布。**
