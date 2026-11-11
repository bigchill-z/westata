# westata `synth2` 非图形功能设计规格（待用户审阅）

- 日期：2026-11-11
- 状态：**已实现并验证**（P0 设计 → P1 捕获 → P2 实现 → P3 对拍 → P4 文档 全部完成）；
  设计阶段经用户批准，实现按 TDD 推进
- 最终实测（5 场景，见 §8）：`e(N/T/T0/T1/K/J)` 精确；`e(bal) ≤ 6.6e-14`、
  `e(U_wt)` 精确、`e(V_wt) ≤ 1.2e-14`（机器精度）；`e(r2) ≤ 1.4e-8`、
  `e(att) ≤ 2.7e-7`、`e(rmse) ≤ 2.7e-5`；`e(pval) ≤ 3.8e-6`（无 cutoff）/
  `≤ 1.8e-2`（cut(2)）；`e(mspe)` 逐行中位 1.5e-3、80% ≤ 1e-2、39 行中 2 行
  退化假单位最大 2.1e1。全量 `pytest -q` 绿（含既有 218 项 synth 测试无回归）。
- 决策依据：本轮三次用户拍板
  1. 范围 = **一次做完整非图形 synth2**（含单位/时间安慰剂、loo、拟合统计、平衡表、p 值）
  2. 验收 = **逐量对拍活体 Stata，如实定级**（公式层 E3 子集、重跑链路 E2）
  3. `frame()` / `symbol()` **不实现**，由结果对象承载结果
- 前置：`ws.synth`（默认路径 + 扩展功能，E2）已落地，是 `synth2` 的引擎
- 依据源码：`C:\Users\zjl\ado\plus\s\synth2.ado`（968 行，v2.1.0，2023-10-05）
  与 `synth2.sthlp`（325 行），已全文读毕并交叉验证

---

## 1. 目标与非目标

### 1.1 目标

- G1：新增 `ws.synth2`，覆盖 Stata `synth2` 的**全部非图形功能**：
  主拟合、拟合统计、6 列协变量平衡表、单位/预测变量权重、
  in-space 单位安慰剂（含 `cutoff`）、in-time 时间安慰剂、loo 留一稳健性。
- G2：逐量对拍活体 StataNow 19.5 捕获；公式层可到机器精度的标 E3 子集，
  重跑链路的累积差异标 **E2** 并写明实测值；不先拍死容差再凑。
- G3：无法保证一致的量显式拒绝或标注 `matched=False`，绝不静默给
  「看起来对」的数。
- G4：中英文 README / GUIDE / API / USING_WITH_STATA / CHANGELOG /
  GAP_REGISTER 文档同步校对（工作区规则）。

### 1.2 非目标（明确不做，均附理由）

| 不做 | 理由 / 源码 |
|---|---|
| `figure`、`nofigure`、`savegraph()`、`show()` | 纯图形；`synth2.ado:234,300,320,460,476,528` 门控；本轮目标明确排除图形 |
| `graphlist`、`e(graph)` | 仅为图形名列表；`:336` |
| 四个临时绘图 frame（`balanceFrame` 等） | `:184-196,204-213,218-227,470-483`，纯绘图 |
| `frame()` 结果数据集、`symbol()` 命名 | **用户拍板不实现**。Stata frame 是其特有概念；Python 里所有轨迹 / p 值 / 包络放进 `Synth2Result` 属性，信息不丢、语义更清晰 |
| 参数推断 / 标准误 / 置信区间 | `synth2` 本身只给**置换 p 值**（排序 + 点态），无 `e(V)`；伪造违反 G3 |
| 全部数值机器精度 / E3 | 底层 `ws.synth` 本身是 E2（QP 秩亏），placebo 最多重跑 39 次会放大差异；物理上不可达 |
| `synth2` 的图形渲染 | 同上 |

### 1.3 成功标准

- S1：≥5 个活体 Stata 捕获场景（主拟合 / `placebo(unit)` /
  `placebo(unit cut(2))` / `placebo(period(1985))` / `loo` / 组合），
  覆盖 `e()` 全部数值面 + loo 包络。
- S2：公式层（RMSE / R2 / ATT / bal / mspe / pval）实测到机器精度或
  如实标注；重跑链路量给出实测界。
- S3：变异测试——回退公式实现对拍测试必须转红。
- S4：全量 `pytest -q` 绿；两个「钉住 synth2 缺席」守卫按各自注释主动改写；
  registry 追加 `synth2` 条目；E 级计数三处一致。
- S5：不把未跑过的内容写成「已验证」。

---

## 2. 用户与核心流程

单一用户（作者本人）。核心流程：

```python
import pandas as pd
import westata as ws

df = pd.read_stata("smoking.dta", convert_categoricals=False)

res = ws.synth2(
    df, unit="state", time="year", depvar="cigsale",
    predictors=["lnincome", "age15to24", "beer", "retprice",
                "cigsale(1975)", "cigsale(1980)", "cigsale(1988)"],
    trunit=3, trperiod=1989, xperiod=list(range(1980, 1989)),
    placebo="unit cut(2)",        # in-space 安慰剂 + cutoff
    loo=True,
)
res.rmse, res.r2, res.att          # 拟合统计
res.bal                           # K x 6 平衡表
res.U_wt, res.V_wt                # 非零单位权重 / V 对角
res.mspe                          # (1+P) x 4
res.pval                          # T1 x 4 逐期 p 值
res.loo_min, res.loo_max          # 留一包络
```

结果对象为**冻结 dataclass**（照 `SynthResult` / `RcmResult` /
`RdRobustResult` 先例），**不套 `StataResults`**：`synth2` 无系数向量、
无 `cov_params`、无 t/z 分布。

---

## 3. 总体架构与互斥写入范围

```
src/westata/estimators/synth2.py   # 唯一实现文件（新）
src/westata/estimators/__init__.py # barrel 导入 + __all__
src/westata/__init__.py            # 顶层导出
tests/test_synth2.py               # 单测（新）
tests/conformance/test_stata_synth2.py        # fixture 对拍（新）
tests/conformance/stata_fixtures/synth2_*.json# 活体捕获（新）
scripts/capture_synth2_fixtures.py            # 捕获脚本（新）
tests/conformance/registry.yml     # 追加 synth2 条目
tests/test_the_two_missing_features.py        # 守卫改写
tests/test_zero_scored_but_present.py         # 守卫改写
文档（README ×2 / GUIDE ×2 / API ×2 / USING_WITH_STATA / CHANGELOG /
      GAP_REGISTER / STATA_ALIGNMENT_SUMMARY / options.md / error.md）
```

**分层**（`synth2.py` 内部三段互不渗透）：
① 输入校验 + 数据筛选（`preperiod`/`postperiod`/`counit` 按 Stata 语义**删行**）；
② 引擎编排（4 个 `ws.synth` 调用点：主拟合 / loo / in-space / in-time）；
③ 汇总与结果组装（拟合统计、平衡表、mspe、pval、loo 包络、e() 面）。

**并行写入纪律**（`AGENTS.md §6.1`）：若实现阶段再委派，Lead 不写已分配文件；
共享文件（三个 `__init__`、`registry.yml`）只设一个 owner；交付按所有权清单核对。

---

## 4. 输入契约（字段级，源码为唯一真相）

### 4.1 主选项（对齐 `synth2.ado:11-31`）

| 参数 | 类型 | 语义 / 校验 |
|---|---|---|
| `data` | `pd.DataFrame` | 长面板；`unit`/`time`/`depvar`/`predictors` 列必须存在 |
| `unit` / `time` | `str` | 面板 / 时间变量（Stata 用 `xtset`，DataFrame 显式给列名） |
| `depvar` | `str` | 结果变量 |
| `predictors` | `Sequence[str]` | 可含 `var(period)`（复用 `ws.synth` 的解析） |
| `trunit` / `trperiod` | 编码 / 数值 | 必须能在相应列取到（`:61-66,85-90`），否则 `ValueError` |
| `counit` | `Sequence \| None` | 列出的单位必须存在、不得含 trunit（`:68-78`）；**未列出的单位从数据删除**（`:79-82`） |
| `ctrlunit` | `Sequence \| None` | **弃用别名**：与 `counit` 同时给出报错（`:39-48`） |
| `preperiod` | `Sequence \| None` | 每个值必须 `< trperiod`（`:92-98`）；未列出的前期行删除（`:99-102`） |
| `postperiod` | `Sequence \| None` | **首元素必须等于 trperiod**（`:107-111`）；其余 `>= trperiod`；未列出后期行删除（`:117-120`） |
| `xperiod` / `mspeperiod` / `customV` | passthru | 透传给 `ws.synth`（`:157` 等） |
| `nested` / `allopt` | `bool` | 透传 |
| `margin` / `maxiter` / `sigf` / `bound` | float/int | 透传；注意 `synth2.sthlp` 默认值 0.05/1000/7/10 与 `synth.ado` 的 0.005/1000/12/10 **不同**，以各自 ado 为准 |
| `placebo` | `str \| None` | 子选项见 §4.2 |
| `loo` | `bool` | 留一稳健性 |

### 4.2 `placebo` 字符串子选项（对齐 `synth2_placebo:378-390`）

- `unit` / `unit(.)` → 全部对照单位；`unit(1 2 3)` → 显式假处理单位
- `period(1985)` → in-time 假处理期，**必须 `< trperiod`**（syntax 级 + `:505-510`）
- `cut(2)` / `cutoff(2)` → 单值 ≥1，剔除 `PreMSPE_fake/PreMSPE_treated > cutoff` 的假单位
- 可组合，如 `unit cut(2)`、`unit(5 21) period(1985) cut(3)`
- **拒绝**：`show()`（纯图形）、`nofigure`（纯图形）

### 4.3 关键语义陷阱（必须照抄）

- `counit/preperiod/postperiod` 是**删行**实现，不是传参；因此 `e(T0)/e(T1)/e(T)/e(J)` 反映筛选后的样本（`:351-355`）。
- `customV` 在 placebo 重跑时**未被转发**（`:253-256` 漏传，虽 `:383` 声明）—— 这是 Stata 的真实行为，实现按「placebo 重跑不带 customV」对齐，并在文档中注明。
- `synth2` 用 `ereturn clear`（`:330`）**清空 synth 的 e()**；Python 里结果对象只暴露 synth2 自己的面，不伪装成 synth 结果。

---

## 5. 数值内核（全部可读源码，逐行转写）

### 5.1 引擎：4 个 `ws.synth` 调用点

| 场景 | Stata 行 | 调用 |
|---|---|---|
| 主拟合 | `:157` | `ws.synth(...)` 全参数 |
| loo 每轮 | `:269` | `ws.synth(..., counit=其余供体)` |
| in-space placebo 每个假单位 | `:417` | `ws.synth(..., trunit=假单位, trperiod 不变)` |
| in-time placebo 每个假期 | `:516` | `ws.synth(..., trunit 不变, trperiod=假期)` |

**`synth2` 自身没有任何优化器**（`:34/157/269/417/516` 全部委托 `synth`），
只消费 `e(V_matrix)` / `e(Y_synthetic)` / `e(X_balance)` / `e(RMSPE)` /
`e(W_weights)`。这是本设计成立的基础。

### 5.2 拟合统计（`synth2_sum:636-672`、`synth2_print:753`）

```
indexPre  = (time <  trperiod) & (unit == trunit)
indexPost = (time >= trperiod) & (unit == trunit)
effect    = Y_treated - Y_synthetic
MSPE_pre  = synth.RMSPE ** 2                 # :640，注意不是重算残差
MSPE_post = mean(effect[indexPost] ** 2)     # :641
RMSE      = synth.RMSPE                      # :645
R2        = 1 - sum(effect[indexPre]**2)
                / sum((Y_synthetic[indexPre] - mean(Y_synthetic[indexPre]))**2)
                                             # :647 分母是【合成值】的离差，非标准，必须照抄
ATT       = mean(effect[indexPost])          # :753
mspe 行   = [MSPE_pre, MSPE_post, MSPE_post/MSPE_pre, 1]   # :660（treated 行）
```

### 5.3 平衡表 6 列（`:174-183`）

```
bal[:,0] = diag(V)                       # Vweight
bal[:,1] = X_balance[:,0]               # Treated
bal[:,2] = X_balance[:,1]               # ValueSyntheticControl
bal[:,3] = (bal[:,2]/bal[:,1] - 1)*100  # BiasSyntheticControl
bal[:,4] = mean(controls 的 predictor)  # ValueAverageControl（`:576` 等权均值）
bal[:,5] = (bal[:,4]/bal[:,1] - 1)*100  # BiasAverageControl
```

`U_wt` = 非零单位权重（降序，`:673-690`）；`V_wt` = V 对角矩阵（`:163`）。

### 5.4 in-space placebo（`:409-459`、Mata `:935-960`）

对每个假单位重跑 `synth`，得 `PreMSPE_fake / PostMSPE_fake`：

```
ratio_post_pre = PostMSPE / PreMSPE
ratio_tr_ctrl  = PreMSPE_fake / PreMSPE_treated          # :664
mspe 追加行 [PreMSPE, PostMSPE, ratio_post_pre, ratio_tr_ctrl]
cutoff: 若 ratio_tr_ctrl > cutoff 则标记剔除（仍留在 mspe 表，但不进 p 值）
```

- **聚合概率** = `mean(ratio_treated <= ratio_all)`（`:441/448`），
  **含 treated 自身**做分母；cutoff 分支用 `mspe_cut` 子集。
- **逐期点态 p 值**（`:947-949`），`eff` 矩阵含 treated 自身（`:940`）：

```
two_sided   = mean(|tr_eff[t]| <= |eff_all[t]|)   # >= 语义
right_sided = mean( tr_eff[t]  <=  eff_all[t] )
left_sided  = mean( tr_eff[t]  >=  eff_all[t] )
pval[t] = [tr_eff[t], two, right, left]          # T1 x 4
```

- `e(mspe)` = `(1+P) × 4`，行名 `[treated, 假单位...]`，列名
  `PreMSPE PostMSPE RatioPostPre RatioTrCtrl`（`:435-436`）。
- `e(pval)` 仅在 in-space 运行时存在；`T1 × 4`，列名
  `Tr.Eff. two-sided right-sided left-sided`（`:953`）。

### 5.5 in-time placebo（`:502-552`）

对每个假处理期重跑，产出该期之后 `Actual / Synthetic / Effect` 轨迹表。
**无 MSPE、无 ratio、无 p 值、无 e()**（`:502-552` 未调用 `synth2_sum`）。
结果放 `Synth2Result.time_placebo`（按假期的有序映射）。

### 5.6 loo（`:261-299`）

- 迭代集 = **仅有非零 W 权重**的供体，按权重降序（`:676-682`）。
- 每轮 `counit = 全部供体 − 该单位`，重跑 `synth`。
- 收集所有轮的 `Y_synthetic` 与 `effect`，取**逐期 min/max**：
  `loo_min / loo_max`（`Y_synthetic` 包络）、`loo_effect_min / loo_effect_max`。
- **无 e()/矩阵**，只有结果变量（Python 里即结果对象属性）。

---

## 6. 输出 / 结果对象

`Synth2Result` 冻结 dataclass，字段逐字对齐 `e()`（`:331-358`）：

| 组 | 字段 |
|---|---|
| 标量 | `N, T, T0, T1, K, J, rmse, r2, att` |
| 矩阵 | `bal(K×6)`, `U_wt`, `V_wt`, `mspe`, `pval` |
| 轨迹 | `Y_treated, Y_synthetic, gap, time_index` |
| 派生 | `loo_min/loo_max`, `loo_effect_min/loo_effect_max`, `time_placebo`, `placebo_probability`, `placebo_removed` |
| 审计 | `method`, `numerically_inactive`, `synth_result`（主拟合的原始 `SynthResult`） |

**明确不提供** `params/bse/pvalues/conf_int/cov_params/df_resid`。
`summary(style="stata")` 渲染紧凑 Stata 风格报表。

---

## 7. 错误处理与降级（对齐 `:61-132`）

| 情形 | 行为 |
|---|---|
| `trunit`/`trperiod` 不在数据中 | `ValueError`（附实际/期望） |
| `counit` 含不存在单位 / 含 trunit | `ValueError` |
| `ctrlunit` 与 `counit` 同时给出 | `ValueError` |
| `preperiod` 有 `>= trperiod` 的值 | `ValueError` |
| `postperiod` 首元素 ≠ trperiod | `ValueError` |
| placebo `period()` 有 `>= trperiod` 或不存在 | `ValueError` |
| `cutoff < 1` / 非单值 | `ValueError` |
| `show()` / `nofigure` / `savegraph()` | `UnsupportedError`（纯图形，明确不做） |
| placebo/loo 重跑中某次 `ws.synth` 失败 | 传播异常 + 指明是哪个假单位/假期/loo 单位（Stata `:418-421` 是 `error _rc` 直接退出） |
| `preperiod`/`postperiod` 删行后样本不足 | `ConsistencyError` |
| 任一量超出实测容差 | 标注 `matched=False` + 实测差 |

---

## 8. 测试与证据

1. `scripts/capture_synth2_fixtures.py`：活体 StataNow 19.5 捕获 ≥5 场景的
   `e()` 全精度（主拟合 / `placebo(unit)` / `placebo(unit cut(2))` /
   `placebo(period(1985))` / `loo` / 组合），参照 `synth2.sthlp:216-231`
   的 California 规格（`trunit(3) trperiod(1989) xperiod(1980(1)1988)`）。
   **注意帮助文件没有任何印刷数值**，所以参照值必须自己从活体 Stata 采集。
2. `tests/test_synth2.py`（TDD）：公式层单测（R2 分母是合成值、Bias 以
   treated 为分母、p 值含自身、cutoff 剔除规则、loo 只迭代非零权重、
   in-time 无 p 值）+ 全部校验路径 + `ctrlunit` 冲突。
3. `tests/conformance/test_stata_synth2.py`：读 fixture（无需 Stata），
   逐量比对，容差**先测后写**。
4. 变异测试：回退 `R2` 分母为实际值 / p 值不含自身 / cutoff 用 `>` 而非 `>=`
   —— 对拍测试必须转红。
5. 守卫改写：`test_the_two_missing_features.py:44-47`、
   `test_zero_scored_but_present.py:74` 由「断言缺席」改为「存在性检查」，
   并保留 `xtserial` 仍缺席的部分。
6. registry 追加 `command: synth2`，`level: E2`，notes 逐量写实测差；
   `scripts/audit_registry_coverage.py` 的 `NO_TABLE` 需加 `synth2`（无系数表）。
7. 全量 `pytest -q` + registry schema 校验 + 每 20 轮修改一次全量测试。

---

## 9. 文档与交付（工作区规则）

- 中英文 README / GUIDE / API（由渲染器生成）/ USING_WITH_STATA /
  CHANGELOG / GAP_REGISTER / STATA_ALIGNMENT_SUMMARY 同步校对。
- `options.md` 记录本轮三项决策；`error.md` 记录通用教训（尤其
  「`synth2` 的 e() 会清空 synth 的 e()」「R2 分母非标准」「p 值含自身」）。
- 版本号与 PyPI/GitHub 推送**另行单独确认**（当前外部推送仍被环境阻断）。
- 推送前完成中英文文档校对（工作区规则）。

---

## 10. 里程碑

| 阶段 | 产出 | 状态 |
|---|---|---|
| P0 设计 | 本文档 → 用户批准 | ✅ 完成 |
| P1 证据 | 活体捕获脚本 + 5 份全精度 fixture（含 loo frame 包络） | ✅ 完成 |
| P2 实现 | `synth2.py` + 35 个单测（TDD RED→GREEN）+ 两级 `__init__` 导出 | ✅ 完成 |
| P3 对拍 | fixture 比对、容差实测确认、registry E2 登记、两处守卫改写、NO_TABLE | ✅ 完成 |
| P4 交付 | 中英文文档校对、CHANGELOG、options/error、全量测试绿、本地提交 | ✅ 完成（推送另行确认） |

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| placebo 最多重跑 39 次 `ws.synth`，嵌套时很慢 | 复用已有 QP 自适应；提供 `max_fits` 上限与显式拒绝；实测计时 |
| 底层 `ws.synth` 是 E2，重跑会放大 RMSPE 差异 | 逐量实测定级，不承诺 E3；公式层单独标机器精度 |
| `synth2` 默认值与 `synth` 不同（0.05/7 vs 0.005/12） | 以各自 ado 为准，测试固定 |
| 「R2 用合成值离差」非常规，易被「好心修正」 | 单测 + 注释 + 变异测试锁死 |
| p 值含 treated 自身、用 `>=` | 单测 + 变异测试锁死 |
| Stata frame / graphlist 概念误移植 | 用户已拍板不实现，结果对象承载 |
| 并发会话持续改其他文件 | 互斥写入范围 + 提交前核对 `git status` |

---

## 12. 未决问题

- 仅 **P4 推送 PyPI / GitHub 前的单独确认**。其余本轮范围内全部已拍板。
