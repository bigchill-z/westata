# westata 缺口登记册（Gap Register）

本文件回答一个问题：**Stata 有、westata 没有的，具体是哪些？**

这不是一个 TODO 列表，而是一份**边界声明**。westata 的目标是在**可检验的计量命令集合**上达到与 Stata 数值一致，不是复刻 Stata 的全部功能。凡是不在范围内（Out of Scope）的条目，都会说明**为什么不做**，而不是含糊地留白。

- 基线：StataNow 19.5（MP），`ado/base` 下共 **3,962 个 .ado 文件**
- 已实现：**32 条命令条目**（见 `tests/conformance/registry.yml`）
- 覆盖率：统计估计与后估计领域的**子集**，不是 Stata 的替代品

状态图例：

| 标记 | 含义 |
|---|---|
| ✅ | 已实现，且经真实 Stata 验证（E2/E3） |
| ⚠️ | 已实现但验证不完整，或只实现了子集 |
| ⬜ | 计划实现（见路线图 P5–P13） |
| ❌ | **明确不做**，附理由 |

---

## 1. 已实现（32 条）

| Stata | westata | 等级 | 备注 |
|---|---|---|---|
| `regress` | `OLS` | E3 | 含 robust / cluster / 因子变量 / 4 种权重 |
| `ivregress 2sls` | `IV2SLS` | E3 | |
| `ivregress liml` | `IVLIML` | E3 | |
| `ivregress gmm` | `IVGMM` | E2 | 系数精确，稳健标准误差 2.9e-2 |
| `areg` | `FEOLS` | E3 | |
| `reghdfe` | `FEOLS` | E3 | 用户包，从 GitHub 安装 |
| `xtreg, fe` | `PanelOLS` | E3 | |
| `xtreg, re` | `RandomEffects` | E3 | Swamy-Arora |
| `xtreg, be` | `BetweenOLS` | E3 | |
| `xtreg, fd` | `FirstDifferenceOLS` | E3 | **StataNow 19 已移除该选项**，改用 `regress D.` 对拍 |
| `logit` | `Logit` | E3 | |
| `probit` | `Probit` | E3 | |
| `poisson` | `Poisson` | E3 | |
| `nbreg` | `NegativeBinomial` | E3 | |
| `ologit` | `OrderedLogit` | E3 | |
| `oprobit` | `OrderedProbit` | E3 | |
| `tobit` | `Tobit` | E3 | 自研似然 |
| `arima` | `ARIMA` | E3 | |
| `var` | `VAR` | E3 | |
| `vecrank` | `VECM.johansen` | E2 | `lags(1)` 边界情况差 0.4% |
| `arch` | `GARCH` | E2 | |
| `dfuller` | `unitroot("adf")` | E2 | |
| `pperron` | `unitroot("pp")` | E2 | 差 1.6e-2 |
| `test` | `results.test` | E2 | |
| `lincom` | `results.lincom` | E2 | |
| `margins, dydx` | `results.margins` | E3 | 仅 `dydx`/`at()`/`over()` |
| `predict` | `results.predict` | E3 | 仅 `xb`/`pr`/`n`/`e`/`ystar`/`residuals` |
| `esttab`/`estout` | `etable` | E3 | 仅系数表，无 `estadd`/`estpost` |

---

## 2. 后估计缺口（P5，最高优先级）

| Stata | 状态 | 说明 |
|---|---|---|
| `nlcom` | ⬜ | 非线性组合，delta method |
| `testnl` | ⬜ | 非线性 Wald 检验 |
| `lrtest` | ⬜ | 似然比检验，需要模型间比较 |
| `hausman` | ⬜ | 固定效应 vs 随机效应检验 |
| `contrast` | ⬜ | ANOVA 式对比 |
| `pwcompare` | ⬜ | 成对比较 |
| `margins, expression()` | ⬜ | 任意表达式 |
| `margins, contrast()` | ⬜ | |
| `marginsplot` | ❌ | 绘图，超出范围 |
| `estat` | ⚠️ | 只支持部分：`estat ic` 可由 `e(aic)`/`e(bic)` 替代；`estat vce`、`estat summarize`、`estat firststage`、`estat classification` 未实现 |
| `suest` | ⬜ | 联合估计，跨模型对比 |
| `linktest` | ⬜ | 模型设定检验 |
| `estat gof` | ⬜ | |
| `estat hetest` | ⬜ | |
| `predict, stdp/stdp/stdf` | ⬜ | 预测的标准误 |
| `predict, scores` | ⬜ | 得分向量 |

---

## 3. 估计命令缺口

### 3.1 受限因变量（P6）

| Stata | 状态 | 说明 |
|---|---|---|
| `heckman` | ⬜ | Heckman 选择模型（两步 + MLE） |
| `truncreg` | ⬜ | 截断回归 |
| `intreg` | ⬜ | 区间回归 |
| `etregress` | ⬜ | 内生处理效应 |
| `treatreg` | ⬜ | `etregress` 的旧名 |

### 3.2 计数模型（P7）

| Stata | 状态 |
|---|---|
| `zip` / `zinb` | ⬜ 零膨胀 |
| `ztp` / `ztnb` | ⬜ 零截断 |
| `tpoisson` / `tnbreg` | ⬜ 截断计数 |
| `gnbreg` | ⬜ 广义负二项 |

### 3.3 分位数与稳健回归（P8）

| Stata | 状态 |
|---|---|
| `qreg` / `sqreg` / `bsqreg` | ⬜ 分位数回归 |
| `rreg` | ⬜ 稳健回归（Huber） |
| `qreg2` | ❌ 用户包，不在 base |

### 3.4 面板扩展（P9）

| Stata | 状态 | 说明 |
|---|---|---|
| `xtlogit` / `xtprobit` | ⬜ | 面板二值（FE/RE/PA） |
| `xttobit` | ⬜ | 面板 Tobit |
| `xtpoisson` / `xtnbreg` | ⬜ | |
| `xtivreg` | ⬜ | 面板 IV |
| `xtabond` | ⬜ | Arellano-Bond 动态面板 |
| `xtdpd` / `xtdpdsys` | ⬜ | 动态面板 GMM |
| `xtgls` / `xtpcse` | ⬜ | 面板 GLS / PCSE |
| `xtheckman` | ⬜ | |
| `xtcloglog` / `xtmelogit` | ⬜ | |
| `xtunitroot` | ⬜ | 面板单位根 |
| `xtcointreg` | ⬜ | |
| `xtfrontier` | ⬜ | |
| `xtmixed` / `mixed` | ❌ | 混合模型，需要专用混合效应引擎，与 lme4 生态重叠 |

### 3.5 离散选择扩展

| Stata | 状态 | 说明 |
|---|---|---|
| `clogit` | ⬜ | 条件 logit |
| `mlogit` | ⚠️ | **未实现**（登记表中无），需要补 |
| `mprobit` | ⬜ | |
| `cloglog` | ⬜ | 互补双对数 |
| `nlogit` | ❌ | 嵌套 logit，实现复杂、使用率低 |
| `asmprobit` | ❌ | 多维 probit，需要模拟似然，极专业 |
| `cmxtmixlogit` | ❌ | 混合 logit，同上 |
| `hetprobit` | ⬜ | |

### 3.6 多方程与系统（P10）

| Stata | 状态 |
|---|---|
| `sureg` | ⬜ SUR |
| `mvreg` | ⬜ 多元回归 |
| `nl` | ⬜ 非线性回归 |
| `nlsur` | ⬜ 非线性似然无关 |
| `gmm` | ⬜ 广义矩估计 |
| `eivreg` | ⬜ 误差变量回归 |
| `manova` | ⬜ 多元方差分析 |
| `frontier` | ⬜ 随机前沿 |
| `sem` / `gsem` | ❌ 结构方程，见 §7 |

### 3.7 因果推断（P11）

| Stata | 状态 | 说明 |
|---|---|---|
| `didregress` / `xtdidregress` | ⬜ | 双重差分（Stata 17+ 内置） |
| `teffects` | ⬜ | 处理效应（IPW/AIPW/匹配/回归调整） |
| `stteffects` | ⬜ | 生存处理效应 |
| `csdid` | ❌ | 用户包，非 base |
| `drdid` | ❌ | 用户包 |
| `rdrobust` / `rdplot` | ❌ | 用户包 |
| `synth` / `synthdid` | ❌ | 用户包 |
| `psmatch` / `pstest` | ❌ | 用户包 |
| `pscore` | ❌ | 用户包 |

> 说明：这些用户包在计量实践中很重要，但它们不是 Stata 内置命令，且各自有独立的 Python 实现（`rdrobust`、`csdid`、`pysynth`）。westata 不会重新实现它们，但会在需要时提供**适配层**。

### 3.8 复杂抽样与重抽样（P12/P13）

| Stata | 状态 |
|---|---|
| `svy:` 前缀 | ⬜ 调查设计与线性化方差 |
| `svyset` / `svydescribe` / `svy: tabulate` | ⬜ |
| `bootstrap` / `jackknife` / `permute` | ⬜ |
| `bs` | ⬜ |

### 3.9 时间序列扩展

| Stata | 状态 | 说明 |
|---|---|---|
| `vec` | ⚠️ | 秩检验已到 E2，**系数估计未验证** |
| `var` 的 `irf`/`fevd`/`vargranger` | ⚠️ | 有封装，未与 Stata 对拍 |
| `svar` | ⬜ | 结构 VAR |
| `vecm` 的 `vecrank` 之外的 `vec stable`/`vecnorm` | ⬜ | |
| `arima` 的 `sarima` 季节性 | ⚠️ | 参数已支持，未验证 |
| `arch` 的 EGARCH/GJR | ⚠️ | 能跑，未单独验证 |
| `newey` | ⬜ | HAC 回归 |
| `prais` | ⬜ | |
| `actest` | ❌ | 用户包 |
| `xpose`/`tsappend` | ❌ | 数据管理 |
| `dfgls` / `ers` / `kpss`（已被 Stata 移除） | ❌ | |
| `vecstable` / `vecnorm` | ⬜ | |
| `sspace` / `ucm` / `mgarch` / `dsge` | ❌ | 见 §7 |

---

## 4. 数据管理与 I/O

**全部 ❌**。理由：pandas 已覆盖且更强大。

| Stata | 替代方案 |
|---|---|
| `gen` / `replace` / `egen` | pandas 赋值与 `transform` |
| `merge` / `joinby` / `append` | `pd.merge` / `concat` |
| `reshape` | `melt` / `pivot` |
| `collapse` | `groupby().agg()` |
| `expand` / `duplicates` | |
| `collapse` / `contract` | |
| `frames` / `frame` | 多 DataFrame |
| `use` / `save` / `export` | `read_stata` / `to_stata` |
| `import delimited` / `import excel` | `read_csv` / `read_excel` |
| `label` / `notes` / `char` | DataFrame attrs |
| `codebook` / `describe` | `df.info()` / `describe()` |
| `tabulate` / `table` | `pd.crosstab` / `pivot_table` |
| `list` / `browse` | 直接打印 DataFrame |
| `by` 前缀 | `groupby` |
| `if` / `in` | 布尔索引 / `.iloc` |

唯一例外：`ws.read_stata()` 提供读取 `.dta` 的薄封装（保留变量标签），供 `etable` 使用。

---

## 5. 图形

**全部 ❌**。

| Stata | 替代方案 |
|---|---|
| `graph twoway` / `scatter` / `line` | matplotlib / seaborn |
| `graph bar` / `box` / `histogram` | 同上 |
| `marginsplot` | 手动绘制 `margins` 结果 |
| `coefplot` | 用户包 |
| `graph combine` / `graph export` | matplotlib subplots / `savefig` |
| `twoway` 的所有子选项 | |

理由：Stata 的图形是独立的渲染引擎，与计量实现无关；matplotlib 生态远超 Stata。

---

## 6. 编程与自动化

**全部 ❌**，除 `etable` 外。

| Stata | 状态 | 理由 |
|---|---|---|
| `mata` | ❌ | 图灵完备语言 + 编译器，不是计量功能 |
| `program` / `syntax` / `args` | ❌ | Stata 宏语言，Python 是更好的编程语言 |
| `foreach` / `forvalues` / `while` | ❌ | Python 原生循环 |
| `local` / `global` / `scalar` | ❌ | Python 变量 |
| `do` / `run` / `include` | ❌ | Python 脚本 |
| `ado` 包管理（`ssc install`） | ❌ | pip |
| `log` / `cmdlog` | ❌ | logging 模块 |
| `simulate` / `permute` | ⚠️ | `permute` 计划（P13）；`simulate` ❌ |
| `postfile` / `post` | ❌ | 直接用 Python 收集结果 |
| `estimates store/table` | ⚠️ | `etable` 接受结果列表，无状态机 |

---

## 7. 明确排除的高级子系统

| 子系统 | 文件数 | 为什么不做 |
|---|---|---|
| **Mata 语言** | 独立运行时 | 图灵完备 + 编译器，等于重写一门语言 |
| **SEM / GSEM** | 62 + 32 | 需要专用引擎；`semopy`/`lavaan` 已存在，语义与 Stata 差异大，对拍成本极高 |
| **生存分析 `st*`** | 724 | lifelines 的语义体系与 Stata 完全不同；`stcox`/`streg` 的基线风险参数化差异巨大 |
| **贝叶斯 `bayes:`** | 96 | 需要 Stan/PyMC，属于另一套推断哲学（MCMC vs MLE） |
| **多重插补 `mi`** | 210 | 210 个文件；且 `statsmodels`/`miceforest` 生态更成熟 |
| **DSGE** | 19 | 极专业，用户极少，且需要专用求解器 |
| **`sspace` / `ucm` / `mgarch`** | — | 状态空间 / 不可观测成分 / 多元 GARCH；专业且 R 生态更强 |
| **绘图** | `graph*` 30+ | 见 §5 |
| **数据管理** | 数百 | 见 §4 |
| **Stata 宏语言** | — | 见 §6 |
| **`svy` 完整体系** | 90 | 计划做核心子集（P12），不做完整 survey 生态 |

---

## 8. 已知的"实现但未完全验证"（⚠️）

这些已实现，但证据等级不足或有已知差距，**不应被当作可靠**：

| 命令 | 问题 |
|---|---|
| `ivregress gmm` | 系数精确（4.9e-16），但稳健标准误差 **2.9e-2**；GMM 权重矩阵构造与 Stata 不同 |
| `arch` | alpha 差 6.9e-3，均值常数差 **3.6e-2**；Stata 默认用 OPG 标准误 |
| `pperron` | 差 1.6e-2，超出 Stata 显示精度，可能是默认滞后截断不同 |
| `vecrank, lags(1)` | 差 0.4%；无滞后差分时的边界处理 |
| `vec`（系数估计） | **完全未验证** |
| `arima` | 4.8e-5，优化器停止点差异 |
| `var` 的 IRF/FEVD/Granger | 有封装，未对拍 |
| `margins` 的因子变量 | 支持 `factor_vars` 参数，但未与 Stata 的离散变化定义对拍 |
| `predict` 的 `stdp`/`stdf`/`scores` | 未实现 |

---

## 9. 环境层面的已知限制

| 限制 | 说明 |
|---|---|
| StataNow 19 **移除了** `xtreg, fd` | 实测 `r(198): option fd not allowed` |
| StataNow 19 **移除了** `kpss` | 实测 `r(199): command kpss is unrecognized` |
| SSC 网络不可达 | `fmwww.bc.edu` 超时；`reghdfe`/`estout` 从 GitHub 源码安装 |
| 无 Stata 许可证时 | E3 测试仍可跑（fixture 已入库），但无法生成新证据 |

---

## 10. 路线图

| 批次 | 内容 | 工时 | 状态 |
|---|---|---|---|
| P5 | 后估计补完（`nlcom`/`testnl`/`lrtest`/`hausman`/`contrast`/`pwcompare`/`predict stdp`） | 2–3 周 | **进行中** |
| P6 | 受限因变量（`heckman`/`truncreg`/`intreg`/`etregress`） | 3–4 周 | 计划 |
| P7 | 计数扩展（`zip`/`zinb`/`ztp`/`ztnb`/`tpoisson`/`tnbreg`/`gnbreg`） | 2–3 周 | 计划 |
| P8 | 分位数与稳健（`qreg`/`sqreg`/`bsqreg`/`rreg`） | 2–3 周 | 计划 |
| P9 | 面板扩展（`xtlogit`/`xtprobit`/`xttobit`/`xtivreg`/`xtabond`/`xtdpd`/`xtgls`/`xtunitroot`） | 4–6 周 | 计划 |
| P10 | 多方程（`sureg`/`mvreg`/`nl`/`nlsur`/`gmm`/`eivreg`/`frontier`） | 3–4 周 | 计划 |
| P11 | 因果推断（`didregress`/`xtdidregress`/`teffects`） | 4–6 周 | 计划 |
| P12 | 复杂抽样核心（`svy:` 子集） | 4–6 周 | 计划 |
| P13 | 重抽样（`bootstrap`/`jackknife`/`permute`） | 2–3 周 | 计划 |

**合计约 26–38 周。** 完成后预计覆盖 Stata 估计类命令的 60–70%（按使用频率加权），仍然不含 §7 的所有子系统。

---

## 11. 如何理解这份登记册

- **westata 不试图替代 Stata。** 它是一个在可验证范围内、能证明数值一致的 Python 计量库。
- **每一条 ✅ 都有对应的自动化测试和证据记录。** 没有测试支撑的功能不算实现。
- **每一条 ❌ 都有替代方案或理由。** 不是遗漏，是取舍。
- **每一条 ⚠️ 都在文档中明确标注。** 不把"能跑"说成"已验证"。

如果你需要某个 ❌ 的功能，正确的做法通常是：用现有的 Python 库（matplotlib、lifelines、semopy），或继续用 Stata。westata 不会为了"覆盖所有功能"而牺牲它唯一的核心价值——**可审计的数值一致性**。
