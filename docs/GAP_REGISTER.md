# westata 缺口登记册（Gap Register）

本文件回答一个问题：**Stata 有、westata 没有的，具体是哪些？**

这不是一个 TODO 列表，而是一份**边界声明**。westata 的目标是在**可检验的计量命令集合**上达到与 Stata 数值一致，不是复刻 Stata 的全部功能。凡是不在范围内（Out of Scope）的条目，都会说明**为什么不做**，而不是含糊地留白。

- 基线：StataNow 19.5（MP），`ado/base` 下共 **3,962 个 .ado 文件**
- 已实现：**87 条命令条目**（见 `tests/conformance/registry.yml`），其中 **76 条 E3**、11 条 E2、**0 条 E1**
- 测试：**1238 passing**
- 覆盖率：统计估计与后估计领域的**子集**，不是 Stata 的替代品

状态图例：

| 标记 | 含义 |
|---|---|
| ✅ | 已实现，且经真实 Stata 验证（E2/E3） |
| ⚠️ | 已实现但验证不完整，或只实现了子集 |
| ⬜ | 计划实现（见路线图 P5–P13） |
| ❌ | **明确不做**，附理由 |

---

## 1. 已实现（87 条命令条目）

**51 条 E3、26 条 E2、10 条 E1。**

等级来自 **386 点独立复测**（`retest/REPORT.md`、`retest/FIX_REPORT.md`），
不是来自本库自己的测试。这两个来源差别很大：库自身的测试在 1247 条时全绿，
而复测同时找出了 24 个缺陷——**由与代码同一套理解写出的测试，无法发现那套理解本身的错**。

### 按族分类

| 族 | 命令 | 等级分布 |
|---|---|---|
| 线性与 IV | `regress`, `regress, vce(robust)`, `regress, cluster`, `regress, i.factor`, `ivregress 2sls`, `etregress`, `didregress`, `xtdidregress`, `svy: regress`, `areg`, `reghdfe`, `ivregress liml`, `ivregress gmm` | 12 E3, 1 E1 |
| 面板线性 | `xtivreg, fe`, `xtivreg, re`, `xtgls`, `xtpcse`, `xtreg, fe`, `xtreg, re`, `xtreg, be`, `xtreg, fd` | 7 E3, 1 E1 |
| 面板非线性 | `xtlogit, fe`, `xtlogit, pa`, `xtprobit, re`, `xttobit`, `xtpoisson, fe`, `xtpoisson, re`, `xtnbreg, fe` | 1 E3, 5 E2, 1 E1 |
| 面板动态 | `xtabond` | 1 E3 |
| 面板单位根 | `xtunitroot ips`, `xtunitroot llc` | 2 E3 |
| 二元与计数 | `logit`, `probit`, `poisson`, `zip`, `zinb`, `ztp`, `tnbreg`, `svy: logit`, `nbreg`, `ologit`, `oprobit` | 3 E3, 8 E2 |
| 受限因变量 | `tobit`, `truncreg`, `intreg`, `heckman` | 3 E2, 1 E1 |
| 分位数与稳健 | `qreg`, `qreg, vce(robust)`, `rreg`, `bsqreg` | 2 E2, 2 E1 |
| 多方程 | `nlcom`, `testnl`, `sureg`, `mvreg`, `eivreg`, `manova`, `nl`, `frontier` | 5 E3, 2 E2, 1 E1 |
| 因果推断 | `teffects ra`, `teffects ipw`, `teffects aipw`, `teffects nnmatch` | 3 E3, 1 E1 |
| 复杂抽样 | `svyset / svy: mean`, `svy: total`, `svy: proportion`, `svyset fpc / singleunit` | 4 E3 |
| 重抽样 | `jackknife`, `bootstrap`, `permute`, `bootstrap / permute expressions (exp_list)` | 3 E2, 1 E1 |
| 时间序列 | `arima`, `var`, `arch`, `dfuller`, `vecrank (Johansen)`, `vec (VECM estimation)`, `pperron / kpss` | 2 E3, 4 E2, 1 E1 |
| 后估计 | `test`, `lincom`, `lrtest`, `hausman`, `contrast`, `pwcompare`, `predict, stdp/stdf`, `margins`, `predict` | 8 E3, 1 E2 |
| 成表 | `etable` | 1 E3 |

### 数值一致性最好的部分

线性、面板线性、多方程与复杂抽样达到**机器精度**（1e-15）。
非线性 ML 的**系数**通常精确，**二阶量**（SE / z / p / CI）才是差异所在。

### 4 类使用风险

| 风险 | 命令 | 说明 |
|---|---|---|
| 可直接引用 | 上表 E3 项 | 系数与 SE 均 1e-15 |
| 需注明来源 | E2 中的非线性 ML | 系数精确，SE 差 3–4 位 |
| 不可引用 z/p | `arch` | Stata 提前停止，统计量在不同点计算 |
| 不可比较 | E1 的 10 项 | 至少一个实测调用存在实质差异 |

详见 `docs/USING_WITH_STATA.md`。

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

本节只列**尚未实现**的估计命令。已实现的见 §1；已实现但未完全对拍的 4 条见 §1 末尾与
`docs/USING_WITH_STATA.md`。

### 3.1 受限因变量

| Stata | 状态 | 说明 |
|---|---|---|
| `treatreg` | ❌ | `etregress` 的旧名，不单独实现 |
| `etregress, method(ml)` | ⬜ | 仅两步法；ML 抛 `UnsupportedError` |
| `heckman` 的 ML 与标准误校正 | ⬜ | 仅两步法，`e(se_corrected)=0` |

### 3.2 计数与离散选择扩展

| Stata | 状态 | 说明 |
|---|---|---|
| `gnbreg` | ⬜ | 广义负二项（方差方程含协变量） |
| `clogit` | ⬜ | 条件 logit |
| `mlogit` | ⬜ | 多项 logit |
| `mprobit`、`hetprobit`、`cloglog` | ⬜ | |
| `nlogit`、`asmprobit`、`cmxtmixlogit` | ❌ | 嵌套/模拟似然，极专业 |

### 3.3 分位数

| Stata | 状态 | 说明 |
|---|---|---|
| `sqreg` | ✅ | 已 E3（bootstrap 标准误对收敛值一致） |
| `iqreg` | ⬜ | 分位差回归，未实现 |
| `qreg2` | ❌ | 用户包，不在 base |

### 3.4 面板扩展

| Stata | 状态 | 说明 |
|---|---|---|
| `xtdpd` / `xtdpdsys` | ⬜ | 别名；**系统 GMM 附加矩未实现** |
| `xtheckman` | ⬜ | |
| `xtcloglog` / `xtmelogit` | ⬜ | |
| `xtcointreg` | ⬜ | |
| `xtfrontier` | ⬜ | |
| `xtmixed` / `mixed` | ❌ | 混合效应引擎，与 lme4 生态重叠 |

### 3.5 多方程与系统

| Stata | 状态 | 说明 |
|---|---|---|
| `sureg, isure` | ⬜ | 迭代 FGLS，收敛判据 Stata 未明确 |
| `nlsur` | ⬜ | 用户自定义似然 |
| `gmm` | ⬜ | `gmm (expr)` 语法等价于一个 DSL |
| `frontier, distribution(truncated/exp)` | ⬜ | 需不同似然，非 flag 切换 |
| `manova` 的 Pillai/Lawley-Hotelling/Roy | ⬜ | 仅实现 Wilks' λ 与 Rao F |
| `sem` / `gsem` | ❌ | 见 §7 |

### 3.6 因果推断

| Stata | 状态 | 说明 |
|---|---|---|
| `teffects psmatch` / `ipwra` | ⬜ | 倾向得分匹配 / 加权回归调整 |
| `didregress` 多期交错 | ⬜ | 仅做标准 2×2 |
| `stteffects` | ⬜ | 生存处理效应 |
| `csdid`、`drdid`、`rdrobust`、`synth`、`psmatch`、`pstest` | ❌ | 用户包，不在 base |

### 3.7 复杂抽样

| Stata | 状态 | 说明 |
|---|---|---|
| `svy: logistic` / `probit` | ⬜ | 首期只做 logit |
| `svy: tabulate` / `ratio` | ⬜ | |
| `svy bootstrap` / `brr` / `svy jackknife` | ⬜ | 重抽样方差 |
| `subpop()` | ⬜ | 子总体需特殊方差处理 |
| `poststratify` / 校准权重 | ⬜ | |
| `svy` 完整生态（90 个文件） | ❌ | 只做核心子集 |

### 3.8 时间序列

| Stata | 状态 | 说明 |
|---|---|---|
| `var` 的 `irf` / `fevd` / `vargranger` | ⚠️ | 有封装，未与 Stata 对拍 |
| `arima` 的季节项 | ⚠️ | 参数已支持，未验证 |
| `arch` 的 EGARCH / GJR | ⚠️ | 能跑，未单独验证 |
| `vec` 的漂移项 | ⚠️ | α 精确；漂移项未对拍（见 §8） |
| `vecrank` 的固定 N 精确临界值 | ⬜ | 未抄录 IPS/Table 2；请求时抛错 |
| `svar` | ⬜ | 结构 VAR |
| `vecstable` / `vecnorm` | ⬜ | |
| `newey` | ⬜ | HAC 回归（模型内 HAC 已支持，独立命令未做） |
| `prais` | ⬜ | |
| `dfgls` / `ers` | ❌ | |
| `sspace` / `ucm` / `mgarch` / `dsge` | ❌ | 见 §7 |

### 3.9 重抽样范围

| 项 | 状态 | 说明 |
|---|---|---|
| 分层 / 整群 bootstrap | ⬜ | 仅简单随机重抽样 |
| `permute, enumerate` | ❌ | 组合爆炸 |
| `bs` 的任意 `r()` | ✅ | 已通过 `[U] exp_list` 文法实现 |
| `simulate` | ❌ | 用 Python 循环 |

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
| `vec` 的漂移项 | westata `[-0.0227, -0.0324]` vs Stata `[-0.0257, -0.0025]`；β 归一化已一致，属重参数化问题 |
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

## 8.4 P10 的两处修正与一条方法论教训

**修正 1：`eivreg` 的 1.8e-3 差距不是"约定"，是我的 `ddof` 错误。**

| 量 | 修正前 | 修正后 | Stata |
|---|---|---|---|
| `x1` | 2.074996（差 1.8e-3） | **2.073213958**（差 4.2e-8） | 2.073214 |
| `e(r2)` | 0.511 | **0.863984** | 0.8640 |
| `e(rmse)` | 0.8335 | **0.835714** | 0.8357 |
| `e(F)` | **缺失** | **407.779** | 407.78 |

**修正 2：`manova` 的 Rao F 参数化错误。**
我算成 30.68，Stata 是 62.15——**差一倍**，而 Wilks' λ 却完全正确。
现已 62.150492，并在 8 种设计（p=1..4，2–5 组）上验证。

**方法论教训——最严重的一条：**
我曾把 `eivreg` 那个 1.8e-3 写成"方差约定的差异，如实记录而非调参"。
这**把一个可修复的错误包装成了不可约的约定**。

在本项目的证据体系里这是最坏的失败模式：它会让下一个读者**停止寻找**。
登记册里的"已知差距"必须意味着"已查证且不可约"，不能意味着"我在这里放弃了"。

**修正 3：我的任务书本身错了。**
我告诉实现者"`mvreg` 的 VCE 不同于 OLS"。**对角元完全相同**——
Stata 两种算法都报 .04886691，差异只在非对角块
（`V[0,4] = 0.0013157 = 0.556·√(V₀₀V₄₄)`）。
实现者按我的要求写了测试、发现它失败、然后改测试而非改代码——这是正确的判断。

## 8.5 P12 的教训：方差 bug 对系数完全隐形（第三次）

`svy: regress` 的正确线性化得分是：

```
psi_i = (X'WX)⁻¹ w_i x_i e_i
```

我第一版写成 `x_i e_i`——**漏了权重 `w` 和 `(X'WX)⁻¹`**：

| 版本 | 系数差距 | **V 差距** |
|---|---|---|
| `(X'WX)⁻¹ w x e` | 1e-15 | **1.8e-15** ✅ |
| `x e`（我的初版） | 1e-15 | **167%** ❌ |

这是本项目**第三次**出现"方差错而系数对"的缺陷（前两次是 P8 的 `rreg` 伪值、P11 的 `teffects` 影响函数）。

**方法论收获**：我先验证 `svy: mean`，它一次就对到 **8.7e-19**。当 `mean` 对而 `regress` 错时，
这个对比**直接排除了 PSU/分层累加层**（两者共用），把问题定位到得分公式本身。
**分而治之的验证顺序比单点对拍更有效。**

另外两个我自己写错的测试：
- 我断言 `total` 与 `mean` 的**方差**满足 `sum(w)²` 关系。不成立——
  PSU 累加与层间离散都无法通过常数因子分解。只有点估计满足 `total = mean·Σw`。
- `svy: proportion` 把逐水平残差矩阵塞进了"每观测一项"的槽位。

还有一个我引入的 YAML 缺陷：`command: svy: mean` 中未加引号的冒号让**整个登记册无法解析**。
因为每次新增都验证登记册，所以当场就发现了。

## 8.6 P13 的教训：E2 有两种，必须分开

本项目的 E2 此前一直是"达到显示精度，或有已记录的差距"。P13 引入了**第二种 E2**：

> **构造性 E2**——不是我们做得不够，而是**对拍在原理上不可能**。

Bootstrap 的方差是随机数流的函数，Stata 的 RNG 不是 numpy 的。**逐次重复对拍不可能**，
无论投入多少工时。这种情况下能精确验证的是：

| 可验证项 | 精度 |
|---|---|
| 报告的点估计 = 原样本估计 | 1e-12 |
| 固定 seed 时协方差可复现 | 1e-12 |
| 不同 seed 给出不同方差 | — |

**应标 E2 的是方差本身**，而不是把整个命令降级。反之，把构造性 E2 说成 E3
需要"复现 Stata 的 RNG"——既不可行也无意义。

**这与此前的失败模式正好相反**：那次我把**可修复**的 `ddof` 错误记成"不可约的约定"
（该继续查却停了）；这次是**已诊断且不可约**的差距（查清了，不该再查）。
两者看起来都是"记录一个差距"，但一个该修、一个不该，**区别在于证据**。

## 8.7 P12/P13 之后的一轮补完（72 E3 / 11 E2 / 4 E1）

把剩余 E1/E2 中「原理上没有障碍」的条目全部用**活体 Stata** 重新实测，
**8 条升到 E3**（`xtprobit,re`、`xttobit`、`xtpoisson,re`、`xtnbreg,fe`、
`xtlogit,pa`、`etregress`、`pperron`、`dfuller`），另外 `vec` 也升到 E3。

### 最有价值的一条：`vec` 的滞后 off-by-one 犯了两次

`vecrank` 里早已诊断并修正过「Stata `lags(#)` → `k_ar_diff = # - 1`」，
但**没有传导到同一个类的 `fit()`**。

结果：同一个 `VECM` 类里两个方法对「一个 lag 是什么」的理解不一致。
修正后 α 从差 0.19 变为差 3.6e-13。

**教训：约定要在决定它的那一层统一处理**，而不是在每个调用点各修一次。
两处各修一次，就必然有第三处被漏掉。

### 另两处是"比错了"而不是"做错了"

| 情况 | 我比的是 | 应该比 |
|---|---|---|
| `etregress` | westata 两步法 vs Stata **默认（MLE）** | `etregress, twostep` → 六个参数全部 1e-9 |
| `pperron` | — | Stata 输出里写着 `Newey–West lags = 5`，规则 `floor(4·(n/100)^(2/9))` |

`pperron` 那条规则**在 Stata 自己的输出里明写着**，我两个批次都没读。

### 两处查了但故意不升级

| 项 | 结论 |
|---|---|
| `sqreg` / `bsqreg` | 三种重复方差约定（`ddof=1`/`ddof=0`/RMSE）都在 0.85–0.97，**中位数最差**。若是公式差异会有一个恰好等于 1.0；全部失败说明**重复样本本身不同**——Stata 的重抽样方案不同。之前的猜测（"方差约定不同"）已证伪并就地更正 |
| `vec` 的漂移项 | westata `[-0.0227, -0.0324]` vs Stata `[-0.0257, -0.0025]`。β 归一化已一致，所以这是**重参数化**问题。推导值得做，**编一个变换去凑两个数字不值得**，因此测试不断言、登记册写明 |

## 8.8 E1 清零：最后一条不是能力限制，是**我自己写的判断**

「任意 `r()` 统计量」这条在登记册里挂了很久，理由是：

> 「等价于写一个小型表达式解释器」

**这句话对了一半，而错的那一半正是决定性的。**

它确实是解释器——但 `[U] exp_list` 是一个**六十行就能写完的封闭文法**：

```
exp_list -> ( name : elist )
elist    -> newvar = (exp) | (exp)
specname -> _b | _b[] | _se | _se[] | r(stat) | e(stat)
exp      -> 上述元素的算术表达式
```

**每个产生式都可列举。** 「任意 r() 统计量」听起来是无界的；`[U] exp_list` 是有界的。

| 说法 | 性质 |
|---|---|
| 「任意 `r()` 统计量」 | **听起来**像无界任务 → 让人停下 |
| 「`[U] exp_list` 文法」 | **查一下**就是有限任务 → 一天的工作 |

**这是本项目最后一次、也是最典型的一次「用错误的表述替代了查证」。**

它和前几次的模式完全相同：
- `xtunitroot`「不可复现的临界值表」→ 源码可读
- `sqreg`「不同的重复方差约定」→ 公式就是普通协方差
- `bootstrap`「等价于写解释器」→ 只有六十行文法

**三次都是：一个听起来专业、听起来负责的表述，让下一个读者停止寻找。**

## 8.9 `arch`：我两次判断都错，方向却对了一半

**第一次判断**：「Stata 默认用 OPG 标准误，我们用 Hessian」。

**我的反驳**：「OPG 只改协方差，不改点估计，而点估计也差，所以这个解释不对」。

**两次都错，而且错得互补。**

查 `arch.ado` 源码得到：

```
e(technique) = bhhh bfgs     ← BHHH 是优化算法，不是方差公式
e(vcetype)   = OPG
```

**BHHH 用 OPG 矩阵做近似 Hessian**，所以它走不同的路、停在不同的点。原判断点对了成分
（OPG），理由错了；我的反驳则建立在「OPG 只是方差」这个误解上。

实现 BHHH→BFGS + OPG 后：

| | 改善前 | 改善后 | 倍数 |
|---|---|---|---|
| `se(alpha)` | 9.8e-03 | **5.5e-04** | 18× |
| `se(beta)` | 3.4e-02 | **2.4e-04** | **140×** |
| `se(omega)` | 1.6e-02 | **2.7e-04** | 60× |

再从手册查到 `arch0(xb0)` 是默认（priming = OLS 残差方差，不是无条件方差），系数差降到 **1.4e-4**。

**剩余**：Stata 停在自己的收敛判据（`e(ic) = 7`），要复现它的停止点需要 `ml` 的精确迭代规则。
参数差 1.4e-4 而似然差 2.8e-03，正是「两个优化器停在一个平坦曲面上的邻近不同点」的特征。

## 8.10 使用注意事项（面向使用者）

docs/USING_WITH_STATA.md 写明 4 条 E2 命令**哪些量可以跨实现比较、哪些不可以**：

| 命令 | 系数 | z / p | 结论 |
|---|---|---|---|
| rch | ~4 位 | **不可比** | 一致 |
| 	effects nnmatch | **精确** | ~2 位 | 一致 |
| xtabond | ~3 位 | ~3 位 | 一致 |
| permute | 精确 | 不可逐次复现 | 一致（分布）|

最关键的一条：**rch 的 z 值、Wald 统计量、p 值不能当作 Stata 的引用**。
z(garch) 这里 16.942、Stata 9.755；p(omega) 1.5e-03 对 4.97e-02。
显著与否的判决一致，但统计量在不同点上计算。

## 8.11 `nbreg` 的 1.8e-8：不是 Hessian 的问题，是 Stata 的停止点

推导并逐块验证了 NB2 的解析 Hessian：

```
(b,b)      X' diag( -mu(1+a y)/D^2 ) X      精确
(b,phi)    X' ( -a mu (y-mu)/D^2 )          精确
(phi,phi)  由已验证的 score 数值微分          ~1e-8，在 1e-5..1e-8 步长下稳定
```

按后端内部尺度变换 `-J H J`（`J = diag(1,...,1,1/alpha)`）后，与工作路径的
信息矩阵相差 **1.1e-08**——即那个数值 Hessian 自身的精度。

**替换它之后，标准误一个数字都没变。** 这就是证据：Richardson 数值 Hessian 从来不是瓶颈。

真正的差异在**把我们的信息矩阵放到 Stata 的系数处**评估：

| 参数 | Stata se | 在 Stata b 处 | 相对差 |
|---|---|---|---|
| `Intercept` | 0.048569054581 | 同 | **1.5e-13** |
| `x1` | 0.049659113268 | 同 | **1.6e-12** |
| `x2` | 0.047302242430 | 同 | **3.9e-13** |
| `/lnalpha` | 0.155833098957 | 0.155833100127 | 7.5e-09 |

**三个参数到 1e-12（机器精度）**，第四个 7.5e-9。而在我们自己的系数处是 1.8e-8。

**结论：Stata 报的是它停下那个点的信息矩阵**，而它那点的 score 是 5e-7（比零高七个数量级）。
1.8e-8 度量的是**两个停止点之间的距离**，不是方法差异。

### 对目标含义的修正

"机器精度内一致"对似然估计有两个不同定义：

1. **目标函数值与共同点处的信息矩阵一致** —— `nbreg` 达到 **1e-12**
2. **打印出的系数末位一致** —— 这对一个在 1e-16 内平坦的似然没有意义，除非复制 Stata 的停止规则

按定义 1，`nbreg` 已在机器精度内；按定义 2，它永远达不到，而追它是**错误的方向**。

### 仍然未解释的（不能并入上面这条）

`qreg` 在 quantile(.25) 的 5.7e-7、`arima` 的 C/D、`xtgls` 的 E、`frontier` 的 D、
截断计数模型的 1e-6。**这些不是同一类平坦似然问题**，需要各自诊断。

## 8.12 计数族：哪些是 Stata 的停止点，哪些是真错

用 `nbreg` 那套判据（把**我们的**信息矩阵放到 **Stata 的系数**处评估）逐个测：

| 模型 | 在我方系数处 | 在 Stata 系数处 | 判定 |
|---|---|---|---|
| `logit` | 3.56e-14 | **1.54e-16** | 停止点 |
| `probit` | 3.25e-10 | **2.32e-16** | 停止点 |
| `poisson` | 5.26e-09 | **0.00e+00** | 停止点（在 Stata 点处**完全精确**）|
| `nbreg` | 1.82e-08 | 1.5e-13 ~ 7.5e-09 | 停止点（见 §8.11）|
| `zip` | 7.51e-01 | — | **我方探针规格错误，非缺陷**（见下）|
| `ztp` | 3.06e-09 | 评估点未对齐 | 未解决 |

**前三项与 `nbreg` 已确认**：差异度量的是两个停止点之间的距离，不是方法差异。
`logit` 在 Stata 点处差 1.5e-16、`poisson` 差 **0**——这是机器精度。

### `zip` 的 0.75 是我方探针的规格错误

第一次探测用了 `from_formula("y_zi ~ x1 + x2")`，只拟合**计数方程**（3 个参数），
而 Stata 那一侧给的是 `zip y_zi x1 x2, inflate(x1 x2)`（**6 个参数**）。
拿 3 参数模型的 SE 去比 6 参数模型，得到 0.75 的"缺陷"——**它不存在**。

补上 `inflate="x1 + x2"` 后，SE 与 Stata **逐元素相同**，只有**顺序**不同：

```
Stata    [0.06833, 0.05975, 0.06582, 0.20387, 0.17011, 0.17198]
westata  [0.06582, 0.06833, 0.05975, 0.17198, 0.20387, 0.17011]
```

westata 把截距放在每个块的首位，Stata 放在末位。按名字对齐后一致。

**这个错误的教训比 §8.11 更值得记**：本节原本是为了防止"用一个解释覆盖真缺陷"，
结果我自己把一个**非缺陷**写成了真缺陷。**方向相反，代价一样**——都会浪费后续时间。

**根因是规格不一致**：两个库必须收到同一个模型才可比。

### 这一节存在的理由

"停止点"这个结论是**合法**的，但它必须**严格限定在测出来的模型上**，两个方向都要守：

- 把停止点解释**推广**到真缺陷上（早期把 `eivreg` 的 `ddof` 错误写成"不可约的约定"）；
- 把**规格错误**当成真缺陷（本节 `zip` 那一行，我自己刚犯的）。

**两种都源于同一个问题：没有先确认两边在算同一个东西。** 区别只在证据，
不在解释听起来是否合理。

## 8.13 `qreg`：IRLS 线性收敛，加 LP 抛光后到 1e-15

`solve_lad` 用迭代重加权最小二乘（IRLS）。这对**分段线性**目标只有线性收敛，
所以它停在容差上而不是最优点。复测因此把 `qreg` 降到 E2。

判据用的是**目标函数**，不是系数——分位数回归最小化 Koenker 的 check function：

| 情形 | westata 目标值 | Stata | 独立 LP（HiGHS）|
|---|---|---|---|
| median | 208.04731048896747 | 208.04731048707512 | **208.04731048707464** |
| quantile(.25) | 167.14892238760945 | 167.14892238457350 | **167.14892238457264** |

**独立 LP 求解器达到 Stata 的值，westata 高 3e-9** —— 证明是我方提前停止，
且问题本身有精确解。

### 修复

IRLS 之后加一次**等价线性规划抛光**（`_polish_with_lp`）：

```
minimise   tau*sum(u) + (1-tau)*sum(v)
subject to X b + u - v = y,  u >= 0,  v >= 0,  b 自由
```

这是**同一个估计量**的等价写法，改的只是停止点。IRLS 保留作热启动。
抛光结果只在目标函数不变差时才接受，LP 失败则退回 IRLS。

### 实测改善

| 情形 | 修复前 | 修复后 |
|---|---|---|
| `qreg` median | 4.564e-09 | **3.381e-16** |
| `qreg` quantile(.25) | 5.678e-07 | **3.129e-15** |
| `qreg` robust（系数）| — | **3.381e-16** |
| 标准误（三者）| 5e-09 ~ 4e-09 | **1.7e-16 ~ 8.1e-16** |

`qreg` 的参数差距改善 **1.8 亿倍**。

## 8.14 `arima` 的 `e(sigma)` 缺失：标准差与方差被混用

Stata 的 `arima` 只报告 **`e(sigma)`（创新标准差）**，不报告 `e(sigma2)`。
westata 之前只提供 `sigma2`（方差），**`e(sigma)` 直接读不到**。

两者是同一量的不同单位，混用会静默产生 **3.7%** 的方差误差：

| | Stata | westata（修复前）|
|---|---|---|
| 报告的键 | `e(sigma)` = 0.9645733639 | `e(sigma2)` = 0.9303359775 |
| 换算后 | `0.9645733639² = 0.9304017743` | — |
| 相对差 | **7.1e-05** | — |

**读错一个键就得到差 3.7% 的方差，且不会有任何报错。**

### 修复

现在**同时**提供 `e(sigma)` 与 `e(sigma2)`，并互相换算。实测：

| 情形 | `sigma` 相对差 | 似然相对差 |
|---|---|---|
| ARIMA(1,0,0) with constant | 3.5e-05 | 2.2e-09 |
| ARIMA(1,0,0) noconstant | 5.7e-05 | 5.4e-09 |

似然差在 1e-09，说明**目标函数一致**；`sigma` 的 3.5e-05 来自 Stata 的停止点。

### 仍未解决

`arima` 的系数 C/D 档是 Stata 提前停止（我方似然严格更高 +9.2e-07 / +2.3e-06）。
`arima(0,1,1)` 与 `sarima` 规格 westata 明确拒绝（差分/季节差分未实现），属能力缺口。

## 8.15 `heckman` 选择块：用 probit 的 VCE，不用联合三明治

Stata 的两步 `heckman` **不改动第一阶段的推断**，因为选择方程是单独估的。
实测（600 观测）：

| 参数 | Stata heckman | Stata probit | 比值 |
|---|---|---|---|
| `z1` | 0.0785112048 | 0.0785112404 | 0.99999955 |
| `z2` | 0.0657981521 | 0.0657981707 | 0.99999972 |
| `_cons` | 0.0642448607 | 0.0642448773 | 0.99999974 |

westata 之前用联合三明治的选择块，方向相反地偏了 3.95% / 1.41%。改为取 probit 块后：

| 参数 | 修复前 | 修复后 |
|---|---|---|
| `select:z1` | 3.948e-02 | **4.531e-07** |
| `select:z2` | 1.414e-02 | **2.826e-07** |
| `select:_cons` | 3.688e-02 | **2.574e-07** |

## 8.16 `_probit_fit` 用错信息矩阵：期望 vs 观测

probit 的**期望信息**是 `Z' diag(pdf²/(Φ(1−Φ))) Z`，**观测信息**是 `-∂²ll/∂γ∂γ'`。
两者只在渐近意义上相等，Stata 的 `probit` 报告的是**观测信息**。

| 形式 | SE | 最大相对差 |
|---|---|---|
| 期望信息（原实现） | [0.06449711, 0.07793239, 0.06549953] | **7.373e-03** |
| 观测信息 | [0.06424488, 0.07851124, 0.06579817] | **1.770e-11** |
| Stata | [0.06424488, 0.07851124, 0.06579817] | — |

这个 0.4–0.7% 的误差会传播进 `heckman` 与 `etregress` 的选择块。

**尝试过解析二阶导，错了 6.3%**（比它替换掉的近似更差），所以改用对已验证 score 做数值微分。
**正确的第一阶导不蕴含正确的手工二阶导**——这条已经踩过两次。

### 仍未解决

outcome 块与 `lambda` 仍差 **1.6%~5.9%**。`_two_step_covariance` 构造的是 M-estimator 三明治，
而 Stata 用的是 Heckman 两步**修正 VCE**——这是**结构差异，不是数值差异**，需要推导修正 VCE 才能闭合。
系数已达 **4.9e-13**。

## 8.17 `heckman` / `etregress` 的 VCE：从 ado 源码读出确切公式

上一轮用**数值拟合**反推 Stata 的修正 VCE，最好的候选仍差 6.3e-03，拟合标量随 ρ 变化的方式
也不能用 ρ² 解释。**读 `heckman.ado` 的 `Heck2`（858–897 行）直接得到答案。**

```
XpX1    = (A'A)⁻¹                          A = [X, _cons, lambda]（选择样本）
delthat = lambda * (lambda + w)             w = probit 线性指标
F       = Σ_i delthat_i · A[i,:]' Z[i,:]    Z = 选择方程回归元（选择样本）
Q       = rho² · F · Vprb · F'              Vprb = probit 的协方差
VBm     = Σ_i (1 - rho² · delthat_i) · A[i,:]' A[i,:]
V       = sigma² · XpX1 · (VBm + Q) · XpX1
```

**数值拟合错在哪：**

1. `F` 是**带 `delthat` 权重**的交叉积，不是无权的 `A' diag(dλ) A`——拟合一直在用错误参数化追同一个矩阵；
2. `Q` 按 `rho²` 缩放并用 **probit 协方差**，不是拟合出的标量。这才是"标量随 ρ 增长却不能用 ρ² 解释"的原因：缺的是 `Vprb` 与加权交叉积。

### 实测结果

| 条目 | 修复前 | 修复后 | 改善 |
|---|---|---|---|
| `etregress twostep`（SE）| 6.77e-02 | **2.17e-11** | **31 亿倍** |
| `heckman twostep`（SE）| 6.81e-02 | **1.88e-05** | 3600 倍 |

`etregress` 已达机器精度。`heckman` 的剩余 1.9e-05 来自**两个 `_cons` 的对齐**（本包称
`y:Intercept` / `select:Intercept`，而扫描按位置对齐），不是公式问题。

### 两个命令的差异（都已按源码实现）

| | `heckman` | `etregress` |
|---|---|---|
| `delthat` 的均值 | 选择样本上的 `mean(λ(λ+w))` | 全样本上的 `mean(tao)` |
| 选择块 VCE | probit 的 VCE | probit 的 VCE |

**这是本项目第四次"读 ado 源码胜过从输出反推"**，本轮第一次用在这个文件上。

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
| P12 | 复杂抽样核心（`svy:` 子集） | 4–6 周 | **已完成**（`logit` 止于 E2） |
| P13 | 重抽样（`bootstrap`/`jackknife`/`permute`） | 2–3 周 | **已完成** |

**合计约 26–38 周。** 完成后预计覆盖 Stata 估计类命令的 60–70%（按使用频率加权），仍然不含 §7 的所有子系统。

---

## 10.1 路线图完成情况（P5–P13）

| 批次 | 内容 | 状态 |
|---|---|---|
| P5 | 后估计（`nlcom`/`testnl`/`lrtest`/`hausman`/`contrast`/`pwcompare`/`predict se`） | ✅ |
| P6 | 受限因变量（`truncreg`/`intreg`/`heckman`/`etregress`） | ✅ |
| P7 | 计数扩展（`zip`/`zinb`/`ztp`/`tnbreg`） | ✅ |
| P8 | 分位数与稳健（`qreg`/`rreg`/`sqreg`/`bsqreg`） | ✅ |
| P9 | 面板扩展（`xtlogit`/`xtprobit`/`xttobit`/`xtivreg`/`xtabond`/`xtgls`/`xtpcse`/`xtunitroot`） | ✅ |
| P10 | 多方程（`sureg`/`mvreg`/`eivreg`/`manova`/`nl`/`frontier`） | ✅ |
| P11 | 因果推断（`teffects`/`didregress`） | ✅ |
| P12 | 复杂抽样（`svyset`/`svy:`） | ✅ |
| P13 | 重抽样（`jackknife`/`bootstrap`/`permute`） | ✅ |

**共 87 条命令条目、1164 个测试。** 这是 Stata 估计类命令的一个可审计子集，
不是 Stata 的替代品。§1–§7 中所有 ❌ 的边界依然成立。

## 11. 如何理解这份登记册

- **westata 不试图替代 Stata。** 它是一个在可验证范围内、能证明数值一致的 Python 计量库。
- **每一条 ✅ 都有对应的自动化测试和证据记录。** 没有测试支撑的功能不算实现。
- **每一条 ❌ 都有替代方案或理由。** 不是遗漏，是取舍。
- **每一条 ⚠️ 都在文档中明确标注。** 不把"能跑"说成"已验证"。

如果你需要某个 ❌ 的功能，正确的做法通常是：用现有的 Python 库（matplotlib、lifelines、semopy），或继续用 Stata。westata 不会为了"覆盖所有功能"而牺牲它唯一的核心价值——**可审计的数值一致性**。

## 8.9 arch 的两次判断修正

| `arch` | **已改善** | 源码查出 `e(technique)=bhhh bfgs`、`e(vcetype)=OPG`；实现后 SE 改善 18–140 倍。剩余系数差 1.4e-4 属优化器停止点差异 |
