# westata 命名规范（讨论稿）

> 状态：**规则已定稿并于 0.3.41 实施**。本文件记录 2026-11-18 多轮讨论达成的命名规则共识。
> 盘点基线见 [NAMING_INVENTORY.md](NAMING_INVENTORY.md)；需用户拍板的取舍见
> [options.md 〇之十三](../options.md)。与既有 [API_CONVENTIONS.md](API_CONVENTIONS.md) 的关系：
> 后者管参数/签名/结果对象能力（R1–R6），本文件补充「类与函数名字怎么取」这一层，
> 定稿后其中与 R1.2 冲突处需回头修订 R1.2。

---

## 1. 核心原则：功能 → Stata 命令 → Python 改造

命名的唯一推导链（用户原话：*「同样的功能，先从 stata 的名称出发进行改造，熟词可以缩写。
因为 Regress 和 reg 是同样的回归，所以从 stata 的 reg 出发，然后改写 Regress」*）：

```
功能语义  →  找到对应的 Stata 命令名  →  按 Python 习惯改造
```

三步的含义：

1. **先定功能**：这是一个什么估计/检验（如「普通最小二乘」）。
2. **再找 Stata 命令**：该功能在 Stata 里叫什么（如 `regress`，缩写 `reg`）。
3. **最后按 Python 习惯改造**：snake_case（函数）/ PascalCase（类），生僻缩写展开为
   可读英文词，熟词允许保留缩写。例：`reg` → `Regress`（动词化，非 `Reg`）。

### 1.1 为什么改：现状的病

顶层 253 个名字混用三套体系：Stata 命令原名（`estat_vce`/`panel_set`/`newey_west`）、
statsmodels 风格类名（`Regress`/`PanelRegressFE`/`RegressHighDimFixedEffects`）、自造统计词（`QuantileRegress`/
`render_esttab`/`etable_as_stata`）。同一个「最小二乘」有 `Regress`/`PanelRegressFE`/`RegressHighDimFixedEffects`/
`Areg`/`PanelRegressBE` 五种叫法且无取舍规则；`RDRobust`/`RdRobustResult`/`rd_robust`
一个词三种写法。详见盘点文档 §1–§2。

### 1.2 与原定位的关系

README 的定位是「Stata-conformant numerics + statsmodels-style API」。本次改动**只动名字，
不动接口形态、不动数值**：

| 轴 | 权威 | 本次 |
|---|---|---|
| 名字（词源） | Stata 命令 + Python 改造 | ✅ 改 |
| 接口形态（`from_formula().fit()`、duck typing） | statsmodels | ❌ 不变 |
| 数值口径 | Stata 对齐 | ❌ 不变 |

---

## 2. 规则

### R-N1 类名以 Stata 命令为词源，生僻缩写展开、熟词保留

- 类名用 PascalCase，来自对应 Stata 命令。
- 生僻前缀/缩写展开：

| Stata 缩写/前缀 | 展开 | 例 |
|---|---|---|
| `xt` | `Panel` | `PanelLogit → PanelLogit`、`Xtset → Panel...` |
| `svy` | `Survey` | `SurveyRegress → SurveyRegress` |
| `rd` | `RegressionDiscontinuity` | `RDRobust → RegressionDiscontinuityRobust` |
| `teffects` | `TreatmentEffects` | `TreatmentEffectsRa → TreatmentEffectsRegressionAdjustment` |
| `hdfe` | `HighDimFixedEffects` | `RegressHighDimFixedEffects → HighDimFixedEffectsRegress` |
| `q`（qreg） | `Quantile` | `QuantileRegress → QuantileRegress` |

- **熟词保留**：`VAR` / `ARIMA` / `GARCH` 等 Stata 命令本身即如此命名者保持原样；
  **`DID` / `IV` 也算熟词，保留缩写、不展开**（用户 2026-11-18 明确）。因此
  `DIDRegress → DIDRegress`、`IV2SLS → IV2SLS`（原名保留）、`PanelIVRegress → PanelIVRegress`
  （xt 展开、iv 保留），而不是展开成 DifferenceInDifferences / InstrumentalVariables。
- 判据：熟词 = 学界与 Stata 命令里都以该缩写通用、展开后反而更长更难用者；生僻前缀
  （xt/svy/rd/teffects/hdfe）才展开。
- **已明确的特殊决定**：`Regress → Regress`。理由：Regress 与 `regress` 是同一功能，从 Stata
  的 `reg` 出发动词化为 `Regress`，而不是保留 `Regress` 这个与 Stata 命令不同形的统计缩写。

### R-N2 函数名也按缩写展开（有记录的风险）

用户最终决定：函数名也按同一规则展开，例如 `panel_set → panel_set`、
`var_select_order → var_select_order`。**`estat_*` 特例（用户 2026-11-18 澄清纠正）**：
`estat_` 前缀保留不展开，只展开后缀——`estat_effect_size → estat_effect_size`、
`estat_first_stage → estat_first_stage`、`estat_weak_instrument_robust → estat_weak_instrument_robust`；
后缀是人名（`bgodfrey`/`moran`/`dwatson`…）或熟词（`auc`/`gof`/`ic`/`vce`/`vif`）者保留。
（此前误写成 `estat_* → estimate_*`、`estat_vce → estimate_variance_covariance`，已作废。）

> **已记录的风险（纠偏义务已履行，用户坚持）**：函数名展开会破坏
> `docs/API_CONVENTIONS.md` R1.2「Python 名可反查回 Stata」的基础——例如改名后用户无法从
> `panel_set` 反查 Stata 的 `panel_set`（`estat_` 前缀保留使 `estat_*` 一族受影响较小）。替代方案
> （只改类名、函数名保留）已提出，用户明确选择坚持展开。最小缓解（**尚未实施**）：① 旧名保留
> 一个版本 + `DeprecationWarning`，兼容期内仍可反查；② 新函数 docstring 首行写明对应 Stata 命令；
> ③ 提供「Stata 命令 → Python 名」对照表作为长期反查入口。落地时必须同步改写 R1.2。

### R-N3 结果类保留统计学术名，统一复数后缀

- 结果类**不**跟随命令新名改造（不做 `RdRobustResult → RegressionDiscontinuity...Results`
  这类联动）。
- **统一后缀为复数 `*Results`**（用户 2026-11-18 最终决定，推翻早先的统一单数）：
  63 个单数 `*Result` 全部加 s（`AucResult → AucResults`、`NeweyWestResult → NeweyWestResults`…）；
  已是复数者（`VARResults`/`VECMResults`/`StataResults`/`StoredResults`）不动。
  `Xt*` 前缀先展开再加 s（`XtSerialResult → PanelSerialResults`）。
  **撞名例外**：`ContrastResult` 与 `ContrastResults` 语义不同（单行列对象 vs 多行聚合），
  各自保留原名，不适用本规则。
- 表格/行对象用 `*Table` / `*Row`，不并入 `*Results`。

### R-N4 拼写硬伤无条件修复

与方案无关的拼写错误直接改（旧名仍按 R-N5 兼容）：
`TefectsRaResult → （随 TreatmentEffectsRa 新名）`、`IrFTable → IrfTable`、
`SucestResult → SuestResult`、`VarBasic → VARBasic`、`RDRobust/RdRobustResult/rd_robust`
三者大小写统一。

### R-N5 兼容：旧名保留一个版本 + DeprecationWarning

- 所有被改的公共类名、函数名保留旧名为别名，调用时发 `DeprecationWarning`，一个版本后移除。
- e-class 函数薄包装（`newey_west`/`prais`/`synth`/`synth2`/`rd_robust`）同样标 deprecated，
  不立即删除。**已于 0.3.43 实施**：上述五个函数发出 `DeprecationWarning`，指向等价类入口
  （`Newey`/`Prais`/`Synth`/`Synth2`/`RDRobust`）。`prais_winsten` 是底层数值求解器、
  无等价类入口，故不弃用。
- 项目已有多处「保留弃用别名一个版本」的先例（见 API_CONVENTIONS 阶段三）。

### R-N6 一个概念只有一个名字

沿用 API_CONVENTIONS R1.3。改名后不得出现新旧名在顶层同时作为「主名」存在；旧名只以
弃用别名形式存在。

---

## 3. 命名推导示例（**草案，需逐条核实 Stata 语义**）

> ⚠️ 下表是规则落地的**示例**，不是定稿映射。每个新名都必须先核实对应 Stata 命令的
> 真实名称与语义后才能采用——讨论中已经在 `RegressionControlMethod` 上出现一次猜测错误：
> `RegressionControlMethod` 是 **regression control method**（源码 [rcm.py:1](src/westata/postestimation/rcm.py#L1)），
> 曾被误猜为 regression discontinuity curve。这类错误正是「不能凭描述定名」的例证。

### 3.1 高置信（Stata 命令语义明确、规则直接适用）

| 现名 | Stata 命令 | 推导 | 候选新名 |
|---|---|---|---|
| `Regress` | `regress` | reg → Regress | `Regress` |
| `PanelRegressFE` | `xtreg` | xt→Panel, reg→Regress | `PanelRegress` |
| `PanelLogit` | `xtlogit` | xt→Panel | `PanelLogit` |
| `PanelProbit` | `xtprobit` | xt→Panel | `PanelProbit` |
| `SurveyRegress` | `svy: regress` | svy→Survey | `SurveyRegress` |
| `SurveyMean` | `svy: mean` | svy→Survey | `SurveyMean` |
| `DIDRegress` | `didregress` | did 是熟词保留 | `DIDRegress` |
| `RDRobust` | `rd_robust` | rd→展开 | `RegressionDiscontinuityRobust` |
| `IV2SLS` | `ivregress 2sls` | IV 熟词保留，2SLS 熟词保留 | 原名保留 `IV2SLS` |
| `PanelIVRegress` | `xtivreg` | xt→Panel，IV 保留 | `PanelIVRegress` |
| `QuantileRegress` | `qreg` | q→Quantile | `QuantileRegress` |
| `RobustRegress` | `rreg` | r→Robust | `RobustRegress` |
| `TreatmentEffectsRa` | `teffects ra` | teffects→展开 | `TreatmentEffectsRegressionAdjustment` |
| `RegressionControlMethod` | `rcm` | regression control method | `RegressionControlMethod` |

### 3.2 待核实 Stata 语义后才能定名

`Synth`/`Synth2`（Stata 命令即 synth，是否展开为 SyntheticControl 待定）、
`SimultaneousQuantileRegress`/`BootstrappedQuantileRegress`（sqreg 的准确全称）、`Nnmatch`/`Psmatch`（对应 ado 的确切命令名）、
`ErrorsInVariablesRegress`/`MultivariateRegress`/`SeeminglyUnrelatedRegress`、`Frontier`/`Nl`、`TruncatedRegress`/`IntervalRegress`/`EndogenousTreatmentRegress`
等，需逐一读源码 docstring 与 Stata 命令对照后确定，不在讨论稿阶段拍板。

---

## 4. 实施路线（多发布周期工程，非单个 PR）

按仓库规则（全量测试 → 更新中英文文档 → bump 版本 → 推 PyPI + GitHub），改动规模决定了
它必须分批：

1. **规则定稿**：本讨论稿 → 定稿，修订 API_CONVENTIONS R1.2，更新 README 命名说明。
2. **机械校验**：把 R-N1~R-N6 落成校验规则（仿 `scripts/check_api_conventions.py`），
   先能判定、再执行——避免「规范写了但自由生长」。
3. **分批改名**：按族推进（先 xt/svy 前缀族，再 e-class 函数，再结果类统一复数），每批
   全量测试 + 弃用别名 + 文档同步。
4. **发布**：每批一个版本，`DeprecationWarning` 保留一个版本后清理。

### 4.1 规模提示

- 顶层 253 导出；候选受影响：类名约 60、函数名约 60–174、结果类 54。
- `retest/` 有约 180 处按名调用；6655 项测试、中英文 README/GUIDE/API 文档需同步。
- 当前测试基线：**尚未在改名后运行**（本次讨论未修改任何公共名字）。

---

## 5. 状态

**已实施（0.3.41）**：全部 136 项改名已落地，旧名保留为弃用别名一个版本。
全量 `pytest -q`：6764 passed, 23 skipped, 0 failed；改名检查器 136/136 done、0 findings。
