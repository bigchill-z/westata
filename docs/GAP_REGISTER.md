# westata 缺口登记册（Gap Register）

本文件回答一个问题：**Stata 有、westata 没有的，具体是哪些？**

这不是一个 TODO 列表，而是一份**边界声明**。westata 的目标是在**可检验的计量命令集合**上达到与 Stata 数值一致，不是复刻 Stata 的全部功能。凡是不在范围内（Out of Scope）的条目，都会说明**为什么不做**，而不是含糊地留白。

- 基线：StataNow 19.5（MP），`ado/base` 下共 **3,962 个 .ado 文件**
- 已实现：**77 条命令条目**（见 `tests/conformance/registry.yml`），其中 53 条 E3
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

## 2. 后估计（P5 已完成）

| Stata | 状态 | 说明 |
|---|---|---|
| `nlcom` | ✅ E3 | delta method；**用 z 分布**，与 lincom 不同 |
| `testnl` | ✅ E3 | 非线性 Wald；支持逗号/空格/括号三种联合写法 |
| `lrtest` | ✅ E3 | 嵌套检验，`df==0` 返回 p=1.0 |
| `hausman` | ✅ E3 | **排除常数项**，否则 `V_c−V_e` 不定 |
| `contrast` | ✅ E3 | `r.`/`g.`/`a.`；`a.` 是 low-vs-high |
| `pwcompare` | ✅ E3 | 无多重比较校正 |
| `predict, stdp/stdf` | ✅ E3 | **`stdf` 仅线性族**，Stata 对 logit/poisson 报 `r(198)` |
| `margins, expression()` | ⬜ | 任意表达式 |
| `margins, contrast()` | ⬜ | |
| `marginsplot` | ❌ | 绘图，超出范围 |
| `estat` | ⚠️ | `estat ic` 可由 `e(aic)`/`e(bic)` 替代；`estat vce`、`summarize`、`firststage`、`classification` 未实现 |
| `suest` | ⬜ | 联合估计，跨模型对比 |
| `linktest` | ⬜ | 模型设定检验 |
| `estat gof` / `estat hetest` | ⬜ | |
| `predict, scores` | ⬜ | 得分向量 |

---

## 3. 估计命令缺口

### 3.1 受限因变量（P6 已完成）

| Stata | 状态 | 说明 |
|---|---|---|
| `truncreg` | ✅ E3 | 截断回归，自研似然 |
| `intreg` | ✅ E3 | 区间回归，报 `/lnsigma` 如 Stata |
| `heckman` | ⚠️ E2 | **仅两步法**；标准误**未做 Heckman 校正**（`e(se_corrected)=0`）；`rho` 是插件估计而非 ML |
| `etregress` | ⚠️ E1 | 仅控制函数法，未与 Stata 对拍 |
| `treatreg` | ❌ | `etregress` 的旧名，不单独实现 |

### 3.2 计数模型（P7 已完成）

| Stata | 状态 | 说明 |
|---|---|---|
| `zip` | ✅ E3 | 四个系数与 Stata 精确一致 |
| `zinb` | ⚠️ E2 | 收敛且参数有限；`/lnalpha` |
| `ztp` / `tpoisson` | ✅ E3 | 自研零截断似然 |
| `tnbreg` / `ztnb` | ✅ E3 | 自研截断负二项 |
| `gnbreg` | ⬜ | 广义负二项（方差方程含协变量） |

### 3.3 分位数与稳健回归（P8 已完成）

| Stata | 状态 | 说明 |
|---|---|---|
| `qreg` | ✅ E3 | 系数、标准误、Pseudo R² 全部与 Stata 精确一致 |
| `qreg, vce(robust)` | ⚠️ E2 | 差 1.05%；手册未给闭式公式 |
| `rreg` | ✅ E3 | 完整实现 Cook's D + Huber + biweight |
| `sqreg` | ⚠️ E1 | **系数精确，但 bootstrap 标准误偏小 5–15%** |
| `bsqreg` | ⚠️ E1 | 同上，标准误偏小 12% |
| `iqreg` | ⬜ | 分位差回归，未实现 |
| `qreg2` | ❌ 用户包，不在 base |

> **P8 的关键陷阱**：`qreg` 默认的稀疏度估计是 `fitted` 法（在 τ±h 处重估），
> 不是残差核密度。用核密度会让标准误偏 29% 而系数完全不变。
>
> **`sqreg`/`bsqreg` 为什么只到 E1**：系数是 `qreg` 的拟合值（已验证精确），
> 但 bootstrap 标准误系统性偏小。Stata 用自己的 RNG，逐格对比不可能；
> 但偏差是系统性的，说明抽样方案不同。不声称 E3。

### 3.4 面板扩展（P9 已完成）

| Stata | 状态 | 说明 |
|---|---|---|
| `xtlogit, fe` | ✅ **E3** | 条件似然；系数与 N=912 都与 Stata 一致 |
| `xtlogit, re` | ⚠️ E1 | quadrature 已实现，未逐项对拍 |
| `xtlogit, pa` | ⚠️ E1 | GEE，**不是似然**，`e(ll)` 故意缺席 |
| `xtprobit, re` | ⚠️ E1 | 已实现未对拍 |
| `xttobit` | ⚠️ E1 | 已实现未对拍 |
| `xtpoisson, fe` | ⚠️ E2 | 真似然（多项条件化），差 <5% |
| `xtpoisson, re` | ⚠️ E1 | 已实现未对拍 |
| `xtnbreg, fe` | ⚠️ E1 | **准似然**，`e(quasi_likelihood)=1` |
| `xtivreg, fe` | ✅ **E3** | 三项系数全精确 |
| `xtivreg, re` | ⚠️ E2 | 差 0.42%（theta 分解差异） |
| `xtgls` | ✅ **E3** | 精确匹配；**不等于 OLS** |
| `xtpcse` | ✅ **E3** | 精确匹配 |
| `xtabond` | ⚠️ E2 | 工具数精确 =30；系数差 0.2% |
| `xtdpd` / `xtdpdsys` | ⚠️ E1 | 别名；**系统 GMM 附加矩未实现** |
| `xtunitroot` | ⚠️ E1 | IPS/LLC 近似，Stata 用不可复现的临界值表 |
| `xtheckman` | ⬜ | |
| `xtcloglog` / `xtmelogit` | ⬜ | |
| `xtcointreg` | ⬜ | |
| `xtfrontier` | ⬜ | |
| `xtmixed` / `mixed` | ❌ | 混合模型，需要专用混合效应引擎，与 lme4 生态重叠 |

> **P9 的关键教训**：`xtabond` 的判别式不是系数，而是 **`e(instruments)` 必须等于 30**。
> 这个数字抓出了三个 bug（H 矩阵维度错误、工具块从 y2 而非 y1 开始、缺标准工具），
> 而这三个 bug 发生时**系数看起来都还合理**。工具数错了，系数一定错；工具数对了，才有资格谈系数。
>
> **P9 的另一教训**：我两次写错了**测试**而不是代码。
> `xtgls` 的测试断言"等于 OLS"——那是错的，`xtgls` 是真 GLS（0.679847 vs OLS 0.753313），
> 而 westata 的实现本来就是对的。若按错误测试去"修"实现，会把正确的代码改坏。

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

### 3.6 多方程与系统（P10 已完成）

| Stata | 状态 | 说明 |
|---|---|---|
| `sureg` | ✅ **E3** | 系数与标准误都与 Stata 逐位一致 |
| `mvreg` | ✅ **E3** | 系数 = 逐方程 OLS；VCE 用跨方程协方差 |
| `eivreg` | ✅ **E3** | `A = X'X/n − S`，差 1.8e-3 |
| `nl` | ✅ **E3** | 系数与标准误全精确；ast 白名单，无 eval |
| `frontier` | ✅ **E3** | 半正态前沿，精确 |
| `manova` | ⚠️ E2 | 仅 Wilks' lambda（Rao F 近似） |
| `sureg, isure` | ⬜ | 迭代 FGLS，收敛判据 Stata 未明确 |
| `nlsur` | ⬜ | 用户自定义似然 |
| `gmm` | ⬜ | `gmm (expr)` 语法等价于一个 DSL |
| `frontier, distribution(truncated/exp)` | ⬜ | 需不同似然，非 flag 切换 |
| `manova` 其他统计量 | ⬜ | Pillai/Lawley-Hotelling/Roy |
| `sem` / `gsem` | ❌ 结构方程，见 §7 |

> **P10 的关键教训（`eivreg`）**：`[R] eivreg` 写 `A = X'X − S`，但**没说明 X'X 是归一化的**。
> 我试了三次都错：按 `1/λ` 加权 → 1.214；按 `λ` 加权 → 1.735；用原始 X'X → 1.452。
> 正确是 `X'X/n − S` → 2.075。根因是 `S` 由**样本方差**构成，是逐观测量，
> 所以叉积必须也是逐观测量，否则减法项小 200 倍、等于没减。
>
> **P10 的第二个教训**：我一次粗心的切片替换把 `Sureg.fit` 覆盖成 `Eivreg.fit`，
> 还删掉了 `Mvreg` 和 `Eivreg` 两个类。文件从未提交，git 无法恢复，只能重建。
> **新模块要先提交再打补丁。**

### 3.7 因果推断（P11 已完成）

| Stata | 状态 | 说明 |
|---|---|---|
| `teffects ra` | ✅ **E3** | ATE/ATET/POmean 与其标准误**全部精确** |
| `teffects ipw` | ✅ **E3** | 同上；倾向得分误差已传播 |
| `teffects aipw` | ✅ **E3** | 同上 |
| `teffects nnmatch` | ⚠️ E2 | **点估计精确**；标准误差 +2.2%（`sigma²(Xᵢ)` 未定） |
| `didregress` | ✅ **E3** | ATET 与标准误**全精确**（1e-8） |
| `xtdidregress` | ✅ **E3** | 同上，但 S.E. **不同**（见下） |
| `teffects psmatch` / `ipwra` | ⬜ | 倾向得分匹配 / 加权回归调整 |
| `didregress` 多期交错 | ⬜ | 仅做标准 2×2 |
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

## 8.1 P5 期间发现并修正的真实缺陷

| 缺陷 | 性质 |
|---|---|
| `nlcom` 用了模型的 t 分布 | Stata 的 `nlcom` 恒用 z / 正态区间，即使 `regress` 后 |
| `contrast a.` 方向错误 | Stata 是 low-vs-high，与 `r.` 的 high-vs-low 相反 |
| `hausman` 包含常数项 | 使 `V_c−V_e` 不定（特征值含 −0.0212），Stata 默认排除 |
| `predict stdf` 对 logit 可用 | Stata 报 `r(198)`；westata 现已同样拒绝 |
| `results/base.py` 引用未导入的 `ConsistencyError` | ordered 模型路径才触发 |

## 8.2 P11 期间发现并修正的真实缺陷

| 缺陷 | 性质 |
|---|---|
| DID 协变量列索引错位 | 从已过滤的名字列表取索引，导致截距（全 1）成为唯一协变量；吸水后零方差，x1 系数变 0，`treated` 从 1.186 悄悄变成 1.154 |
| `_absorb` 的转置启发式 | `shape[0] != n` 时转置，把 `(n,1)` 变成 `(1,n)`，吸收了错误的轴 |
| `teffects ra` 影响函数 | OLS 残差与设计正交，模型误差项**恒为零**；只用它们会使标准误小 33 倍 |
| `nl` 的 Total SS 应**不中心化** | Stata 只在有加性参数时中心化；`{b0}*exp(...)` 没有，故 Total = `sum(y²)` = 1017.4143，非 406.7919 |
| `frontier` 缺 `/lnsig2v`、`/lnsig2u` | 方差分量的推断行完全缺失，也无 Wald χ² 与边界 LR |

## 8.3 P11 的两处结论修正（都是我先错、子智能体对的）

**修正 1：`didregress` 与 `xtdidregress` 的标准误不同。**
我曾断言两者相同，并把 6.7% 的差距解释成"Stata 聚类口径不明"——一个无法证实的猜测。
证据其实就在同一份 fixture 里，第二块明写着：

| 命令 | `Data type` | ATET | 标准误 |
|---|---|---|---|
| `didregress` | Repeated cross-sectional | 1.18613 | **.0590465** |
| `xtdidregress` | Longitudinal | 1.18613 | **.055228** |

差异来自自由度口径：重复截面把单位效应当普通回归元（K=209，N−K=1391）；
纵向把单位效应嵌套进聚类变量，只计时期效应（K=10，N−K=1590）。
`sqrt(1391/1590) = 0.9353302985`，实测比值 `.055228/.0590465 = 0.935330629`。

**教训：我错在没读 fixture 的第二块。** 用"手册下载失败"给错误结论披上诚实的外衣，
而需要的证据一直在手边。

**修正 2：`teffects` 的标准误可以做到精确。**
我的版本 RA 偏 29%、IPW 偏 74%。正确做法是把每个估计量写成关于**完整参数向量**的
M 估计量：`psi = −A⁻¹m`，`Var = mean(psi²)/n`（用 Stata 的 `1/n`，不是 `ddof=1`）。

| 估计量 | 我的错值 | 正确值 | 错在 |
|---|---|---|---|
| RA ATE se | 0.06063 | **0.06101488** | 漏了系数项 |
| RA PO(0) se | 0.0436 | **0.06196974** | 同上 |
| IPW ATE se | 0.1632 | **0.09392416** | 把倾向得分当固定 |
| AIPW ATE se | 0.06416 | **0.06250844** | 漏了系数影响函数 |

关键细节：OLS 残差与设计**正交**，所以 RA 的 plug-in 项**恒为零**——
只用那些项的影响函数无论怎么缩放都不可能对。

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
| P5 | 后估计补完（`nlcom`/`testnl`/`lrtest`/`hausman`/`contrast`/`pwcompare`/`predict stdp`） | 2–3 周 | **已完成** |
| P6 | 受限因变量（`heckman`/`truncreg`/`intreg`/`etregress`） | 3–4 周 | **已完成** |
| P7 | 计数扩展（`zip`/`zinb`/`ztp`/`ztnb`/`tpoisson`/`tnbreg`） | 2–3 周 | **已完成**（`gnbreg` 除外） |
| P8 | 分位数与稳健（`qreg`/`sqreg`/`bsqreg`/`rreg`） | 2–3 周 | **已完成**（`sqreg`/`bsqreg` 止于 E1） |
| P9 | 面板扩展（`xtlogit`/`xtprobit`/`xttobit`/`xtivreg`/`xtabond`/`xtdpd`/`xtgls`/`xtunitroot`） | 4–6 周 | **已完成** |
| P10 | 多方程（`sureg`/`mvreg`/`nl`/`eivreg`/`frontier`/`manova`） | 3–4 周 | **已完成**（`nlsur`/`gmm` 除外） |
| P11 | 因果推断（`didregress`/`xtdidregress`/`teffects`） | 4–6 周 | **已完成**（部分） |
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
