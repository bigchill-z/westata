# westata 设计规格

- 日期：2026-10-22
- 状态：待用户审阅
- 项目性质：个人自用工具（非对外发布、不承诺向后兼容）
- 包名 / import 名：`westata`（示例中缩写为 `ws`）

---

## 1. 概述

`westata` 是一个建立在 statsmodels、linearmodels、PyFixest、arch 之上的计量经济学库。它遵守两条硬约束：

1. **函数用法以 statsmodels 为准**——建模入口、`fit()`、结果对象属性、`cov_type`/`cov_kwds` 的组织方式，与 statsmodels 保持一致，结果对象可直接被现有 statsmodels 下游代码消费。
2. **数值行为以 Stata 为准**——点估计、标准误、自由度修正、权重语义、因子变量编码、后估计结果，按 Stata 对应命令的手册定义实现。

由于开发环境没有 Stata（用户选择方案 C），"与 Stata 一致"被定义为**可审计的证据等级**（见第 11 节），而不是未经证实的口号。

---

## 2. 背景与问题

现有 Python 生态已经覆盖了大部分估计算法，但直接使用会在以下地方与 Stata 对不上：

- 小样本自由度修正与聚类修正系数（HC1、CR1、双向聚类、HDFE 的 `dofadjustments`）
- 权重语义（`aweight`/`pweight`/`fweight`/`iweight` 的归一化差异）
- 因子变量编码（base level、交互展开、共线省略、`(omitted)` 报告）
- `xtreg` 默认估计量与自由度、`xtreg, re` 的 Swamy-Arora 默认
- MLE 的标准误来源（observed vs expected information）、收敛准则
- 后估计（`test`、`lincom`、`margins`）与 Stata 风格报表

这些差异单点都能修，但分散在每次分析里重复修。`westata` 把它们收敛成一次性的、被证据表覆盖的实现。

---

## 3. 目标与非目标

### 3.1 目标

- G1：用一套 statsmodels 风格的 API，完成横截面/面板/IV 回归、离散选择、受限因变量、时间序列分析。
- G2：对范围内的每个命令，给出与 Stata 对应的**数值一致性等级**（E1/E2/E3）与可复现证据。
- G3：产出可直接用于论文的 Stata 风格回归表（`esttab` 等价物）。
- G4：当一致性无法保证时**显式报错或警告**，绝不静默给出"看起来对"的结果。

### 3.2 非目标

- N1：不实现 Stata 命令语法解析器（不解析 `reg y x, cluster(id)` 字符串）。API 只有 Python 风格。
- N2：不做数据管理工具（`gen`/`replace`/`merge`/`reshape`），数据操作交给 pandas。
- N3：不实现 Mata、`dsge`、`mgarch`、`sspace`、`ucm`、`heckman`、`csdid`、`rdrobust`、`synth`、`xtabond2`。
- N4：不做 GUI、不做 CLI。
- N5：不承诺跨版本向后兼容（0.x 阶段 API 可变）。
- N6：在 `tobit`、`margins` 等自研模块上，不承诺 bit-level 一致，只承诺声明容差内一致。

### 3.3 成功标准

- S1：`regress`（含 robust / cluster / 因子变量 / 三种权重）在至少 5 个公开数据集上，与公开 Stata 输出点估计及标准误相对误差 ≤ 1e-10。
- S2：范围内每个命令的证据等级 ≥ E1；进入 MVP 交付的命令 ≥ E2。
- S3：一条命令从"写代码"到"拿到 Stata 风格表"不需要读源码，仅靠 API 文档。
- S4：不收敛、共线、完美预测等异常被显式报告，测试覆盖这些路径。

---

## 4. 范围

### 4.1 纳入（MVP）

| 族 | 命令 | 目标等级 |
|---|---|---|
| A 应用微观 | `regress`（OLS/WLS，robust/cluster，`i.` 因子，三种权重） | E3 |
| A | `ivregress 2sls` / `liml` / `gmm` | E2 |
| A | `areg`、`reghdfe` | E3 |
| A | `xtreg, fe` / `re` / `be` / `fd` | E2 |
| A | `test`、`lincom`、`predict` | E2 |
| A | `esttab` 风格输出 | E1（纯展示） |
| B 时间序列 | `arima` / `sarima` | E1 |
| B | `var`（滞后选择、Granger、IRF、FEVD） | E2 |
| B | `vec`（Johansen） | E1 |
| B | `arch`（GARCH/EGARCH/GJR） | E1 |
| B | `dfuller` / `pperron` / `kpss` | E2 |
| B | `newey` | E3 |
| D 离散/受限 | `logit`、`probit` | E2 |
| D | `tobit`（左右截断，自研） | E2 |
| D | `poisson`、`nbreg` | E2 |
| D | `ologit` / `oprobit`、`mlogit` | E1 |
| D | `margins` 子集：线性/logit/probit/poisson 的 `dydx`、`at()`、`over()` | E1 |

### 4.2 排除（首期）

`xtabond2`、`csdid`、`rdrobust`、`synth`、`heckman`、`margins` 全组合（`contrast`/`expression`/任意 `predict` 组合）、`dsge`、`mgarch`、`sspace`、`ucm`、Mata、Stata 命令语法解析、数据管理命令。

---

## 5. 用户与核心流程

单一用户：作者本人（应用研究者）。核心流程：

1. 用 `pandas.read_stata`（或 `ws.read_stata`）读入数据。
2. 用 statsmodels 风格公式建模：`ws.OLS.from_formula("y ~ x1 + C(region)", data=df)`。
3. `.fit(cov_type="cluster", cov_kwds={"groups": ...})` 得到结果。
4. 用 results 的 statsmodels 属性做后续分析；用 `ws.etable([...])` 拼论文表。
5. 需要时 `results.margins(...)`、`results.wald_test(...)`、`results.predict(...)`。

---

## 6. 总体架构

```
src/westata/
├── __init__.py          # 公开 API 导出
├── data/
│   ├── frame.py         # StataData：DataFrame + 权重 + 元数据
│   ├── weights.py       # aweight/pweight/fweight/iweight 归一化
│   ├── panel.py         # PanelData：entity/time、组内 lag/lead/diff
│   └── timeseries.py    # TSData：频率、tsset 语义、时间算子
├── formula/
│   ├── parse.py         # patsy 集成
│   ├── factor.py        # C(x, base=)、交互展开、共线 omit
│   └── transform.py     # L./F./D. 算子 → 显式列
├── vce/
│   ├── robust.py        # HC0–HC3
│   ├── cluster.py       # 一维/二维/多维聚类（CGM）
│   ├── hac.py           # Newey-West、Driscoll-Kraay
│   ├── ssc.py           # Stata 小样本自由度修正
│   └── dispatch.py      # cov_type/cov_kwds → 具体估计器
├── estimators/
│   ├── base.py          # 统一模型协议
│   ├── linear.py        # OLS/WLS/IV2SLS/IVLIML/IVGMM
│   ├── hdfe.py          # FEOLS（PyFixest 后端）
│   ├── panel.py         # PanelOLS/RandomEffects/BetweenOLS/FirstDifferenceOLS
│   ├── discrete.py      # Logit/Probit/Poisson/NegativeBinomial/Ordered*/MNLogit
│   ├── tobit.py         # 自研 Tobit
│   └── tseries.py       # ARIMA/VAR/VECM/ARCH + 单位根
├── results/
│   ├── base.py          # 结果协议：statsmodels duck-typing
│   ├── stata_return.py  # e()/r() 标量容器
│   └── summary.py       # summary(style="stata"|"statsmodels")
├── postestimation/
│   ├── tests.py         # wald_test / lincom
│   ├── margins.py       # dydx / at() / over()
│   └── predict.py       # predict(type=...)
├── report/
│   └── etable.py        # esttab 风格表格
└── conformance/
    ├── registry.py      # 读取并校验证据登记表
    ├── oracles/         # published / r_crosscheck / invariants
    └── tolerances.py    # 容差策略
```

分层原则：`estimators` 只负责估计，`vce` 只负责推断，`results` 只负责呈现与返回，三者不互相渗透。更换后端不影响 VCE 与报表。

---

## 7. 模块设计

### 7.1 data

- `ws.read_stata(path)`：`pyreadstat` 的薄封装，读入 `.dta` 并保留变量标签、值标签（供 `etable` 使用）。
- `StataData`：包装 DataFrame，携带 `weights`（列名或数组）、`weight_type`、`panel`（entity/time）、`ts`（频率）。
- `weights.py` 实现四种权重的 Stata 语义：
  - `fweight`：频数权重，有效样本量取 `sum(w)`，等价于重复观测。
  - `aweight`：解析权重，Stata 在回归中归一化到 `sum(w) = N`。
  - `pweight`：概率权重，点估计同 `aweight`，方差按线性化（sandwich）计算。
  - `iweight`：重要性权重，不做归一化。
  - 具体公式以 `[R] regress` 的 Methods and formulas 为准，实现时逐项核对并写入证据表。
- `panel.py`：`PanelData(df, entity=..., time=...)`，提供 `.lag()`、`.lead()`、`.diff()`、`.groupby_entity()`；对不平衡面板与缺失做显式校验。
- `timeseries.py`：`TSData`，记录频率；提供 `L`/`F`/`D` 时间算子。

### 7.2 formula

- 语法与 statsmodels 一致：`y ~ x1 + x2 + C(region) + x1:x2`。
- 使用 `patsy` 作为解析引擎（statsmodels 自带依赖），确保设计矩阵与 statsmodels 完全一致；`formulaic` 作为 P1 的可选加速引擎，不在 P0 引入。
- `C(x, base=...)` 指定参照组；默认规则对齐 Stata `i.` 的 base level 规则（数值取最小值、字符串按排序），实现时以证据表为准。
- 共线列自动 omit，并在 `summary` 中标注 `(omitted)`，不静默丢弃。
- `transform.py`：`L(x)`、`F(x)`、`D(x)` 生成显式列；面板数据下自动按 entity 分组。

### 7.3 vce

| 估计器 | 公式/规则 | 对应 Stata |
|---|---|---|
| HC0 | `(X'X)^-1 [Σ x_i x_i' û_i²] (X'X)^-1` | `vce(robust)` 的未修正版本 |
| HC1 | `N/(N-K) · HC0` | `regress, robust` 默认 |
| HC2 | `û_i²/(1-h_i)` | — |
| HC3 | `û_i²/(1-h_i)²` | — |
| Cluster (CR1) | `(N-1)/(N-K) · G/(G-1) · sandwich` | `vce(cluster id)` |
| 双向聚类 | CGM：`V1 + V2 - V12` | `vce(cluster id1 id2)` |
| HAC | Newey-West + Bartlett 核，滞后按 Stata 默认规则 | `newey` |
| Driscoll-Kraay | 面板 HAC | `xtscc`（仅作参考） |

- `ssc` 对象集中管理小样本修正：`ws.ssc(adj=True, cluster_adj=True, df="stata")`。
- 所有模型共用同一套 VCE 实现，禁止各模型自写方差。

### 7.4 estimators

统一协议：
- `Model(endog, exog, ...)` 与 `Model.from_formula(formula, data, ...)`。
- `.fit(cov_type="nonrobust", cov_kwds=None, ssc=None)`。
- 内部分派到后端（statsmodels / linearmodels / PyFixest / arch / 自研），再由 `vce` 统一重算或校正协方差。

各后端映射见第 8 节。

### 7.5 results

结果对象实现 statsmodels 协议：
`params`、`bse`、`tvalues`、`zvalues`（离散模型）、`pvalues`、`conf_int()`、`cov_params()`、`resid`、`fittedvalues`、`llf`、`aic`、`bic`、`rsquared`、`rsquared_adj`、`fvalue`、`wald_test()`。

附加：
- `.e`：Stata `e()` 标量字典（`N`、`r2`、`r2_a`、`F`、`df_m`、`df_r`、`rmse`、`ll`、`rank` 等）。
- `.r`：`r()` 结果字典（后估计用）。
- `.summary(style="stata")`：输出 Number of obs / F / Prob > F / R-squared / Adj R-squared / Root MSE，以及系数表（Coef. / Std. Err. / t 或 z / P>|t| / [95% Conf. Interval]）。
- 收敛信息：`converged`、`iterations`、`grad_norm`。

### 7.6 postestimation

- `wald_test()` / `lincom()`：线性组合、联合检验，支持 Stata 的 `test` 语义。
- `margins()`：首期只支持 `dydx`（平均边际效应、`at()` 指定协变量值、`over()` 分组），覆盖线性、logit、probit、Poisson；其他模型显式抛 `NotImplementedError` 并给出原因。
- `predict()`：按 Stata 的 `xb`、`pr`、`residuals` 等类型实现子集。

### 7.7 report

- `ws.etable([res1, res2], ...)`：输出 Stata `esttab` 风格的 DataFrame / Markdown / LaTeX，含系数、标准误、显著性星号、观测数、R²。
- 不做 `estimates store/table` 的完整状态机，只做结果对象列表。

### 7.8 conformance

- `registry.yml` 记录每条命令的证据；`registry.py` 在测试启动时校验 schema。
- Oracle 适配器见第 11 节。

---

## 8. 命令映射表

| Stata 命令 | westata 入口 | 后端 | 关键一致性点 |
|---|---|---|---|
| `regress` | `ws.OLS` / `ws.WLS` | statsmodels + `ws.vce` | HC1、cluster、权重、df |
| `ivregress 2sls` | `ws.IV2SLS` | linearmodels `IV2SLS` | df、R² 定义、小样本修正 |
| `ivregress liml` | `ws.IVLIML` | linearmodels `IVLIML` | 同上 |
| `ivregress gmm` | `ws.IVGMM` | linearmodels `IVGMM` | 加权矩阵、两步估计 |
| `areg` | `ws.FEOLS(absorb=...)` | PyFixest | 单固定效应 df |
| `reghdfe` | `ws.FEOLS` | PyFixest | `ssc`、`dofadjustments` |
| `xtreg, fe` | `ws.PanelOLS` | linearmodels | 组内变换、df、F 检验 |
| `xtreg, re` | `ws.RandomEffects` | linearmodels | Swamy-Arora 默认 |
| `xtreg, be` | `ws.BetweenOLS` | linearmodels | 组间变换 |
| `xtreg, fd` | `ws.FirstDifferenceOLS` | linearmodels | 一阶差分 |
| `logit` | `ws.Logit` | statsmodels `Logit` | Newton、observed info、z 检验 |
| `probit` | `ws.Probit` | statsmodels `Probit` | 同上 |
| `tobit` | `ws.Tobit` | 自研 | 左右截断、Newton、观测信息 |
| `poisson` | `ws.Poisson` | statsmodels `Poisson` | 稳健/聚类 VCE |
| `nbreg` | `ws.NegativeBinomial` | statsmodels | 离散参数参数化 |
| `ologit` / `oprobit` | `ws.OrderedLogit` / `ws.OrderedProbit` | statsmodels `OrderedModel` | 阈值参数化 |
| `mlogit` | `ws.MultinomialLogit` | statsmodels `MNLogit` | 参照组 |
| `arima` | `ws.ARIMA` | statsmodels | 常数项、季节项、报告 |
| `var` | `ws.VAR` | statsmodels | 滞后选择、IRF、FEVD |
| `vec` | `ws.VECM` | statsmodels | Johansen 秩检验 |
| `arch` | `ws.ARCH` | arch | 分布假设、报告 |
| `dfuller`/`pperron`/`kpss` | `ws.unitroot` | arch / statsmodels | 滞后默认值、临界值 |
| `newey` | `ws.OLS(...).fit(cov_type="HAC")` | `ws.vce` | Bartlett 核、默认滞后 |
| `test` / `lincom` | `results.wald_test()` / `lincom()` | — | 线性组合语义 |
| `margins` | `results.margins()` | 自研 | delta method、因子变量 |
| `esttab` | `ws.etable()` | — | 表格格式 |

---

## 9. API 契约

```python
import westata as ws

# 1) 基本 OLS，statsmodels 风格
m = ws.OLS.from_formula("wage ~ tenure + C(race) + hours", data=df)
res = m.fit()
res.params; res.bse; res.tvalues; res.conf_int()
res.summary()                  # statsmodels 风格
res.summary(style="stata")     # Stata 风格
res.e["r2_a"], res.e["df_r"]   # Stata e() 标量

# 2) 稳健/聚类 + 权重
m = ws.OLS.from_formula(
    "wage ~ tenure + C(race)", data=df,
    weights=df["w"], weight_type="pweight",
)
res = m.fit(
    cov_type="cluster",
    cov_kwds={"groups": df["industry"]},
    ssc=ws.ssc(adj=True, cluster_adj=True),
)

# 3) 面板
m = ws.PanelOLS.from_formula("wage ~ tenure + union", data=panel)
res = m.fit(entity_effects=True, time_effects=False,
            cov_type="cluster", cov_kwds={"groups": panel["industry"]})

# 4) 高维固定效应
res = ws.FEOLS.from_formula("y ~ x1 + x2 | firm + year", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["firm"]})

# 5) 离散选择 + 后估计
m = ws.Logit.from_formula("union ~ wage + tenure + C(race)", data=df)
res = m.fit()
res.margins(dydx=["wage"], at={"tenure": [1, 5, 10]})
res.margins(dydx="wage", over="race")

# 6) 时间序列
ws.ARIMA(endog=y, order=(1, 1, 1)).fit()
v = ws.VAR(endog=df[["y1", "y2"]], lags=2).fit()
ws.irf(v, periods=10)

# 7) 论文表
ws.etable([res1, res2, res3], se=True, stars=True)
```

契约要点：
- 权重属于模型（影响点估计），放在构造函数 / `from_formula`。
- `cov_type` / `cov_kwds` / `ssc` 属于推断，放在 `fit()`。
- `cov_type` 取值：`"nonrobust"`、`"HC0"`、`"HC1"`、`"HC2"`、`"HC3"`、`"cluster"`、`"twoway"`、`"HAC"`。
- 结果对象对 statsmodels 下游代码保持 duck-typing 兼容。

---

## 10. 关键语义规格

### 10.1 小样本修正

- `ssc` 默认 `adj=True, cluster_adj=True, df="stata"`，与 Stata 对应命令默认一致。
- 每个命令在文档中标注其默认修正；显式覆盖必须写参数。

### 10.2 权重

- 必须显式声明 `weight_type`；未声明时直接 `ValueError`，不猜测。
- 三种以上权重组合（如 `pweight` + 因子变量）在 Stata 中受限的场景，`westata` 同样拒绝并说明原因。

### 10.3 因子变量

- `C(x, base=...)` 默认 base 规则对齐 Stata `i.`。
- 共线列 omit 并在 summary 标 `(omitted)`。
- 交互 `x1:x2` 与 statsmodels/patsy 语义一致；Stata 的 `#` 与 `##` 不引入，避免两套语法。

### 10.4 推断分布

- OLS / WLS / IV / 面板 FE：t 分布，`df_resid = N - K`（聚类时按 `G - 1`）。
- logit / probit / Poisson / NB / tobit / ordered：z 分布（正态）。
- 具体以 Stata 对应命令的输出为准，写入证据表。

### 10.5 MLE 容差与收敛

- 默认 `tol=1e-8`（参数变化）、`maxiter=1000`。
- 结果对象记录 `converged`、`iterations`、`grad_norm`。
- `strict=True` 时不收敛抛 `ConvergenceError`；默认 `warn`。
- 与 Stata 的一致性承诺是"同一最优点 + 容差内一致"，不承诺逐位相同。

---

## 11. 验证与证据体系

### 11.1 证据等级

| 等级 | 定义 | 可验证性 |
|---|---|---|
| E0 | 无对标证据 | — |
| E1 | 严格按 Stata 手册公式/默认值实现，有推导与单元测试 | 读手册可复核 |
| E2 | 与公开 Stata 输出数值吻合（点估计 + 标准误） | 数据集 + 输出公开 |
| E3 | E2 + 与至少两个独立 R 实现一致 | 完全可复现 |

### 11.2 Oracle

- `PublishedOutputOracle`：Stata 手册 Examples、Wooldridge 教材、UCLA IDRE 等公开输出，手工录入为 fixture（含出处、页码、数据集、命令）。
- `RCrossCheckOracle`：通过 rpy2 调用 `estimatr`（`se_type="stata"`）、`fixest`、`plm`、`margins`/`marginaleffects`，在 CI 中比对。
- `InvariantOracle`：FWL 定理、`Σresid=0`、自由度关系、方差分解、系数与 SE 的尺度变换等恒等式。
- `StataOracle`：预留接口，将来获得 Stata 环境时直接接入，不改业务代码。

### 11.3 证据登记表

`tests/conformance/registry.yml`：

```yaml
- command: regress
  level: E3
  tolerance: 1.0e-10
  sources:
    - type: published_output
      ref: "Stata 18 [R] regress, Example 1"
      dataset: auto.dta
    - type: r_crosscheck
      package: estimatr
      call: 'lm_robust(mpg ~ weight + foreign, se_type = "stata")'
  verified_by: lead
  date: 2026-10-22
```

### 11.4 容差策略

| 模型类型 | 点估计 | 标准误 | 额外要求 |
|---|---|---|---|
| 线性 / IV / 面板 | 1e-10（相对） | 1e-10（相对） | — |
| MLE | 1e-5（相对） | 1e-4（相对） | 符号、显著性一致 |

### 11.5 CI 分档

- 无 R 环境：跑 E1 单元测试 + InvariantOracle + 已入库 fixture。
- 有 R 环境：额外跑 `RCrossCheckOracle`。
- 数据集：`scripts/fetch_stata_datasets.py` 从 Stata 公开地址下载到 `tests/data/`（gitignore），不随仓库分发。

---

## 12. 错误处理与降级

- 后端不支持的能力：抛 `NotImplementedError`，消息中说明"Stata 有、westata 未实现"。
- 后端可能偏离 Stata 而无法校正：抛 `ConsistencyError`（自定义），不静默返回。
- 共线、完美预测、分离、组数不足（聚类 `G` 太小）、不收敛：给出明确 warning 或异常，并在结果对象中标记。
- 权重与命令组合不受 Stata 支持：直接拒绝。

---

## 13. 依赖与许可证

| 依赖 | 用途 | 许可证 |
|---|---|---|
| numpy / pandas / scipy | 基础 | BSD |
| statsmodels | OLS/离散/时序 | BSD-3 |
| linearmodels | IV/面板 | BSD-3 |
| pyfixest | HDFE | MIT |
| arch | GARCH/单位根 | NCSA |
| patsy | 公式解析（statsmodels 自带） | BSD-2 |
| pyreadstat | 读 `.dta` | Apache-2.0 |
| rpy2（可选） | 交叉验证 | GPL-2（仅测试环境使用，不随发布分发） |

说明：复用这些库的**算法与行为**不构成侵权；不复制 Stata 的代码或文档原文。`rpy2` 只在开发环境使用，不作为运行时依赖。

---

## 14. 性能

- HDFE 走 PyFixest 的 demean 实现。
- VCE 全部向量化，禁止逐观测循环。
- 面板 lag/diff 用 groupby + shift，避免 apply。
- 时间序列直接复用 statsmodels/arch 的 C 实现。

---

## 15. 开发环境与工程规范

- Python：当前机器为 3.10.11；3.10 已接近 EOL，建议后续安装 3.12。首期在 3.10 下开发。
- 隔离环境：项目根目录 `.venv`（已创建）。
- 版本控制：本地 git，`main` 分支；每阶段结束提交。
- 代码规范：类型标注 + `ruff` + `pytest`；关键模块要求 docstring 写明对应的 Stata 命令与手册章节。
- 目录：`src/westata/`（src layout），`tests/`，`scripts/`，`docs/`。

---

## 16. 里程碑

| 阶段 | 产出 | 周期（1 人） |
|---|---|---|
| P0 骨架 | 包结构、`vce/`、`OLS` 全链路、结果容器、conformance harness、2 条 E3 证据 | 3–4 周 |
| P1 微观 | IV / HDFE / 面板 FE/RE/BE/FD、因子变量、权重 | 2–3 月 |
| P2 离散 | logit/probit/poisson/nb/ordered、自研 tobit、margins 子集 | 2–3 月 |
| P3 时序 | ARIMA/VAR/VECM/GARCH/单位根 | 1.5–2 月 |
| P4 交付 | etable、文档、证据表完善 | 1 月 |

**P0 是硬门**：若 OLS + 聚类 SE 无法与公开 Stata 输出在 1e-10 内吻合，后续一致性目标整体降级重估。

---

## 17. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 无 Stata，证据只能点状覆盖 | 长尾命令无法升级到 E2 | 证据表显式标注等级；不夸大承诺 |
| R 实现自身偏离 Stata | 交叉验证失真 | 三角验证（手册 + 公开输出 + ≥2 个 R 包） |
| 上游库 API 变动 | 适配层失效 | 锁定版本范围 + 适配测试 |
| `tobit` 自研收敛问题 | MLE 不一致 | 明确容差；提供多起点与收敛诊断 |
| `margins` 组合爆炸 | 覆盖不足 | 首期只做子集，其余抛 `NotImplementedError` |
| Python 3.10 EOL | 依赖升级受限 | 尽早迁移到 3.12 |

---

## 18. 未决问题

无。范围、包名、项目性质、验证策略均已确认。

---

## 19. 附录：术语

- VCE：方差-协方差估计（variance-covariance estimator）。
- HC0–HC3：异方差稳健估计的四种小样本变体。
- CR1：一维聚类稳健估计（Stata `vce(cluster)`）。
- CGM：Cameron–Gelbach–Miller 双向聚类。
- HDFE：高维固定效应。
- ssc：small-sample correction。
