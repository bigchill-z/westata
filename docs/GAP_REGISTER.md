# westata 缺口登记册（Gap Register）

本文件回答一个问题：**Stata 有、westata 没有的，具体是哪些？**

这不是一个 TODO 列表，而是一份**边界声明**。westata 的目标是在**可检验的计量命令集合**上达到与 Stata 数值一致，不是复刻 Stata 的全部功能。凡是不在范围内（Out of Scope）的条目，都会说明**为什么不做**，而不是含糊地留白。

- 基线：StataNow 19.5（MP），`ado/base` 下共 **3,962 个 .ado 文件**
- 已实现：**93 条命令条目**（见 `tests/conformance/registry.yml`），其中 **52 条 E3**、26 条 E2、**15 条 E1**
- 测试：**3866 passing**
- 覆盖率：统计估计与后估计领域的**子集**，不是 Stata 的替代品

状态图例：

| 标记 | 含义 |
|---|---|
| ✅ | 已实现，且经真实 Stata 验证（E2/E3） |
| ⚠️ | 已实现但验证不完整，或只实现了子集 |
| ⬜ | 计划实现（见路线图 P5–P13） |
| ❌ | **明确不做**，附理由 |

---

## 1. 已实现（93 条命令条目）

**52 条 E3、26 条 E2、9 条 E1。**

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
| `predict, ptrt` / `xbtrt` / `yctrt` / `ycntrt` / `cte` | ✅ 已实现（`etregress`） | 依据 `etregress.ado:423`（`marginsok "XB CTE YCTrt YCNTrt PTrt XBTrt default"`）与 `etregress_postestimation.sthlp:136-167`。`ptrt`=`Pr(treatment=1)`，`xbtrt`=处理方程线性预测，`yctrt`/`ycntrt`=`E(y\|t=1)`/`E(y\|t=0)`，**`cte` = 两者之差**（定义即「潜在结果均值之差」，故是导出量而非独立公式——恒等式 `cte == yctrt−ycntrt` 已验证到 1e-12）。处理方程设计需从 `e(treat_varlist)` + `model_data` 重建 |
| `predict, psel` / `mills` / `nshazard` / `ycond` / `yexpected` | ✅ 已实现（`heckman`） | 全部五个选择统计量，依据 `heckman.ado:491`（`marginsok default XB PSel YCond YExpected Mills NShazard`）与 `heckman_postestimation.sthlp:201-221`。**行数不同是语义要求**：`psel`/`mills`/`nshazard` 出自**全样本** probit（400 行），`ycond`/`yexpected` 属**结果方程**（仅选中行 256）。`mills` 与 `nshazard` 是**同义词**（同一组数）。核心恒等式 `ycond = xb + lambda·mills` 已验证到 1e-8 |
| `predict, score` | ✅ 已实现（`regress`/`logit`/`probit`/`poisson`） | `result.predict(kind="score")`；`predict.sthlp:188-189`：**「对线性预测的对数似然导数」**，故单方程模型只有**一个**变量（复数 `scores` 保留给多方程模型）。**Logit 的 `∂ll/∂η = y−p`（即残差），但 Probit 是 `(y−p)φ(η)/(p(1−p))`**——早期把两者合并成一个分支导致 probit 值错。验收判据：**有截距时 `Σ∂llᵢ/∂ηᵢ = 0`**（截距一阶条件），修复前 probit 给出 0.2504，修复后 3e-15 |
| `predict, e` / `predict, ystar` | ✅ 已实现（`tobit`/`truncreg`/`intreg`） | `kind="e"` / `kind="ystar"`；定义见 `tobit_postestimation.sthlp:143-148`：`e` = `E(xb+u \| a < xb+u < b)`（**条件**期望），`ystar` = `E(y*)`。左删失时 `e > ystar` 恒成立，已验证 |
| `predict, fitted` | ✅ 已实现 | `result.predict(kind="fitted")`；Stata 的**同义词**之一（与 `xb` 并列出现在 `regress`/`xtreg` 等多个命令的 `predict` 语法中）。本包把 `fitted` 与 `xb` 分开登记，因为 `SUPPORTED_TYPES` 是逐命令列出的：`regress` 列 `fitted`，`logit`/`poisson` 不列 |
| `predict, xb` | ✅ 已实现 | `result.predict(kind="xb")`；线性指数。`newey` 的 `index` 与它同义（`newey_p.ado:54` 送入同一分支） |
| `predict, residuals` | ✅ 已实现 | `result.predict(kind="residuals")`；`y - xb` |
| `predict, pr` / `predict, n` / `predict, ir` | ✅ 已实现（计数/二值族） | `pr` 的计数形式**需要参数**（`poisson_postestimation.sthlp:147-148` 的 `pr(n)`）；`nbreg` 用负二项 pmf 而非 Poisson（`alpha=exp(lnalpha)`）|
| `predict, hat` / `leverage` / `rstandard` / `rstudent` / `cooksd` / `dfbeta` / `stdr` / `covratio` / `dfits` / `welsch` | ✅ 已实现（`regress` 且 `vce(ols)`） | 影响诊断；`regress_postestimation.sthlp:152-153` 明确：除 `vce(ols)` 外的任何 VCE 下均不可用，因此按 VCE 拒绝而非静默计算 |
| `predict, u` / `xbu` / `ue` | ✅ 已实现（`xtreg, fe`） | `xtreg_postestimation.sthlp:127-134`：`u` = 固定效应估计，`xbu` = `x·u`，`ue` = 组合残差 |
| `predict, score` | ✅ 已实现（`poisson` 专属） | `poisson_postestimation.sthlp:163-164`：`∂ll/∂eta`。**`nbreg` 刻意拒绝**——它产生两个变量（`scores` 复数），单数组 API 无法承载 |
| `estat vce` | ✅ 已实现（渲染层） | `result.estat_vce()` / `ws.estat_vce(result)`；**纯渲染 `cov_params()`，零新计算**。常数按 Stata 拼写显示为 `_cons`；有序模型切点显示为裸 `cut1` 而非 `/cut1`；只打印下三角 |
| `estat vce, correlation` | ✅ 已实现（渲染层） | `result.estat_vce(correlation=True)`；`D^-1/2 V D^-1/2`，**对角线置 0**（Stata 语义：变量与自身不相关） |
| `margins, expression()` | ⬜ | 任意表达式 |
| `margins, contrast()` | ⬜ | |
| `marginsplot` | ❌ | 绘图，超出范围 |
| `estat gof` | ✅ 已实现（`poisson` 专属） | `result.estat_gof()`；公式抄自 `poisgof.ado`，含易漏的 `Σ lngamma(1+y)` 项；**`nbreg` 刻意拒绝**（Stata 无此命令） |
| `estat ic` | ✅ 已实现 | `result.estat_ic()`；四准则公式抄自 `estimates.ado:985-1002`；`df` 默认 `rank(V)`。**注意 `e(aic)` 与 `estat ic` 的 AIC 本就不同**（差 `N(1+log2π)`），见 §8.20.12 |
| `estat vif` | ✅ 已实现（线性模型） | `result.estat_vif()`；公式抄自 `vif.ado`，支持 `uncentered`/`nosort`；无常数模型与非线性模型按 Stata 行为拒绝 |
| `estat hettest` | ✅ 已实现（线性模型，已对 Stata 逐值验证） | `result.estat_hettest()`；公式抄自 `hettest.ado`，支持 `normal`（默认）/`iid`/`fstat`。**无变量列表 = 拟合值**（`hettest.ado:110-115`），不是“全部回归元”——这曾是与 Stata 不一致的根因；`rhs=True` 才是全部 `e(b)` 回归元。`variables` 按**原始数据变量名**解析（`rep78` 是原始 1-5 列，df=1），否则按设计列名。估计权重被辅助回归复用（`fweight`/`iweight` 的 `e(N)=sum(w)`）。robust/cluster VCE、pweight 与 `noconstant` 按 Stata 行为拒绝。**`mtest()` 已实现**（bonferroni/holm/sidak/noadjust，含 Holm 单调下限），见 §8.20.29；单回归元时按 `hettest.ado:126-128` 静默丢弃。90 个用例对照 StataNow 19.5 实跑输出（`tests/conformance/stata_fixtures/hettest_auto.json`） |
| `estat ovtest` | ✅ 已实现（线性模型） | `result.estat_ovtest()`；公式抄自 `ovtest.ado`，含决定性的 `[0,1]` 标准化步骤；支持 `rhs=True` |
| `estat firststage` | ✅ 已实现 | `result.estat_firststage()`；列定义抄自 `ivregress_estat.ado`，含 Shea's partial R²（多内生变量时）。**`r(mineig)` 与 `r(mineigcv)` 已实现**（Cragg-Donald 最小特征值 + Stock-Yogo 三张表逐字抄录），见 §8.20.48 |
| `estat overid` | ✅ 已实现 | `result.estat_overid()`；Sargan + Basmann（2SLS/LIML）、Hansen J（GMM）、**Wooldridge score（稳健/HAC）**；辅助回归须用**全部工具**（含外生）。LIML 在稳健/HAC 下按 Stata 拒绝，见 §8.20.14、§8.20.30、§8.20.46 |
| `estat endogenous` | ✅ 已实现（全内生变量） | `result.estat_endogenous()`；Durbin + Wu-Hausman、**robwool/acwool（稳健/HAC）**、**regF（增广回归，解锁 cluster）**。LIML 拒绝、GMM 的 C 统计量未实现（有阻塞证据），见 §8.20.15、§8.20.35、§8.20.36、§8.20.47 |
| `estat classification` | ✅ 已实现（logit/probit） | `result.estat_classification()`；2×2 表 + 9 个百分比，抄自 `lstat.ado`；支持 `cutoff()`，见 §8.20.18 |
| `estat imtest` | ✅ 已实现（线性模型） | `result.estat_imtest()`；Cameron-Trivedi 三项分解 + 可选 White 检验，抄自 `imtest.ado`；white 与 hetero 分量实测相等（7e-15），见 §8.20.19 |
| `estat summarize` | ✅ 已实现 | `result.estat_summarize()`；列定义照 `estat_summ.ado`；`fweight` 改变 Obs 且 sd 等于展开数据。**aweight/pweight/iweight 归一化到 `sum(w)=N`（尺度不变），估计权重自动沿用**，见 §8.20.20、§8.20.34、§8.20.49 |
| `estat auc` | ✅ 已实现（logit/probit） | `result.estat_auc()`；AUC = Mann-Whitney U/(n₁n₀)，与 `lroc.ado:47-49` 的梯形法等价；`r(N)` 已补，见 §8.20.74 |
| `estat weakrobust` | ⚠️ 部分实现 | `result.estat_weakrobust()`；**Anderson-Rubin 已实现**（受限 F，闭式可验；其分布不依赖工具强度）。**CLR 刻意拒绝**（需模拟条件分布，Stata 为无源码内置）。过度识别模型默认拒绝（Stata 默认报 CLR），须显式 `ar=True`。`betanull()` 支持按名/按位置，见 §8.20.83 |
| `estat mundlak` | ✅ 已实现（xtreg） | `result.estat_mundlak(entity=..., time=...)`；把**组内均值**加入方程后联合检验（`xtreg_postestimation.sthlp:366-370`）。`xtreg_estat.ado:5` 的五个子命令至此齐备。**须显式传 `entity`/`time`**：结果对象保存的是编码后的组标签，不含列名，见 §8.20.84 |
| `estat szroeter` | ✅ 已实现（线性模型） | `result.estat_szroeter()`；公式逐字抄自 `szroeter.ado:158-178`（`Q = n²/(2(n-1)) · (h-mh)²/vh`，用**秩**而非水平值，因备择假设是「方差随变量**单调**变化」）。支持 `rhs=True` 与 `mtest()` 四种调整（bonferroni/holm/sidak/noadjust）。加权、聚类、非 regress 一律拒绝（`szroeter.ado:13-22`） |
| `estat esize` | ✅ 已实现（线性模型） | `result.estat_esize()`；`eta2 = R²`（默认）、`epsilon2`、`omega2`。**注意**：`esize.ado` 把算式交给编译内置，公式取自帮助文件所引文献（Kerlinger 1964；Grisson & Kim 2012），已注明来源；eta² 的置信区间用非中心 F（Smithson 2001）。epsilon 与 omega 互斥（`esize.ado:14`） |
| `estat moran` | ⚠️ 已实现（**与 Stata 可能数值不同**） | `result.estat_moran(errorlag=W)`；**本机无 Stata 依据**——`estat_moran.sthlp:56-59` 的 Methods and formulas 不在帮助文件内，且无 `estat_moran.ado`（编译内置）。实现的是**标准随机化 Moran 检验**（Moran 1950；Cliff & Ord 1981），p 值对「无空间相关」精确但可能与 Stata 不同。本包无 `spmatrix` 工具链，权重矩阵须由调用者传入 |
| `suest` | ✅ 已实现（线性模型） | `result.suest(other1, other2, ...)`；把多个拟合堆叠为**一个参数向量 + 联合 sandwich 协方差**（`suest.sthlp:84-88`）。**关键在于交叉块非零**——两个模型用同一批观测时得分相关，块对角堆叠会错误地报零协方差；sandwich 用 `V = B(Σsᵢsᵢ')B` 按**观测逐行**配对才捕获到。前置条件按 `sthlp:99-102`：各模型须能 `predict, score`，线性模型中它等于残差（`regress_postestimation.sthlp:193`），故已为 `regress` 补上 `score` 类型。样本量必须一致，否则拒绝 |
| `suest, vce(cluster)` | ✅ 已实现 | `result.suest(other, cluster=labels)`；`sthlp:108-109` 允许 `vce(cluster clustvar)`。meat 改为**按簇先聚合得分再取外积**（`Σ_c (Σ_{i∈c}sᵢ)(…)`），这正是捕捉簇内相关的关键。**边界验证**：每个观测自成一簇时结果与不聚类**完全一致**，证明这是真泛化而非另一套公式 |
| `test` / `lincom` after `suest` | ✅ 已实现 | `suest.sthlp:91-94` 把 `test`/`testnl`/`lincom`/`nlcom` 列为 `suest` 的**典型用途**（如广义 Hausman 检验）。已补 `cov_params()`（**方法**，四个命令都要调用它）、`test()`、`lincom()`、`df_resid`（联合检验分母取各分量中**最小**的残差自由度）。跨模型系数相等检验已实测可通过 |
| `linktest` | ✅ 已实现 | `result.linktest()`；Pregibon 链检验，流程逐字对照 `linktest.ado:106-134`：取 `hat = predict(xb)`，生成 `hat²`，**用同一估计器重拟合**含 `_hat`/`_hatsq` 的方程，报告 `_hatsq` 的 t 统计量。**`r(t)` 与 `r(df)` 是 Stata 仅有的两个返回值**（`ado:131-132`），`df` 取重拟合后的 `e(df_r)` = n−3。`ado:52-66` 明确拒绝的命令（`zip`/`zinb`/`sureg`/`mvreg`/`eivreg`/`iv*`/`xt*`）同样拒绝——它们的线性指数不是单一 `xb` |
| `predict, scores` | ✅ 已实现（logit/probit/poisson） | `result.predict(kind="scores")` 返回 **(n × k) 二维矩阵**——行是观测、列是参数，**每个方程的得分都在**（对应 Stata 「每个方程写一个新变量」）。**关键验证**：MLE 处各列得分之和为 **0**（`Σᵢ ∂llᵢ/∂b_j = 0`），符号、链接函数导数或离散因子写错都会破坏它。`logit_p.ado:6` 测的是 **`scores` 复数**，`poisson` 文档用 `score` 单数——这个不对称是 Stata 自身的，两者都注册且不等义。**`nbreg` 仍拒绝**：其离散方程不是设计矩阵的列，第二个得分无处安放 |

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
| `manova` 的 Pillai/Lawley-Hotelling/Roy | ✅ | 四个统计量均从一次特征分解导出；`estat mvtest` 的 F 近似未做（`_manovatest` 为内置无源码）。见 §8.20.51 |
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
| `newey` | ✅ 已实现 | `ws.newey(y ~ x, lag=#)`；`lag()` 必需，Bartlett 核；`Checkt` 三种间隔判定（`tflag==2` 即便 force 也拒绝）；`e()` 只有 7 个标量；`predict` 限 `xb/index/stdp/residuals`。见 §8.20.50 |
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

## 8.18 `sqreg` / `bsqreg`：系数精确，VCE 的差异是蒙特卡洛噪声（并撤回此前四轮的错误归因）

### 结论

按**名称**对齐后（Stata 的 `e(V)[0,0]` 是 `x1`，我们的 `cov_params()[0,0]` 是截距）：

| reps | 统计量 | Stata | 我们 | 相对差 |
|---|---|---|---|---|
| 100 | x1 | 0.049002338388 | 0.057881445137 | 1.8e-01 |
| 100 | x2 | 0.017389717194 | 0.012257626173 | 3.0e-01 |
| 100 | 截距 | 0.032763586567 | 0.030847725297 | 5.8e-02 |
| 400 | x1 | 0.054044288862 | 0.052543112373 | **2.8e-02** |
| 400 | x2 | 0.014543903782 | 0.014795971245 | 1.7e-02 |
| 400 | 截距 | 0.028780064439 | 0.029569181761 | 2.7e-02 |
| 800 | x1 | 0.051563237347 | 0.057322475338 | 1.1e-01 |

**系数在以上每个 reps 下都是 1.8e-16 ~ 4.2e-16。**

而 **Stata 自己的 VCE 在 reps 从 20 到 400 之间变化 42%**（0.037930 → 0.054044）。bootstrap 方差在这些
reps 下**两侧都尚未收敛**，所以逐点比较本身就不是一个良定义的检验。

### 必须撤回的：本轮之前四轮的归因

`e(V)[0,0]` 在 Stata 是 `x1` 的方差（`_cons` **在最后**），而 `cov_params()[0,0]` 是**截距**（截距在最前）。
把两者相比得到「我们的方差小了 1.83 倍」，然后依次被解释为：

| 轮次 | 当时给出的归因 | 实际 |
|---|---|---|
| 5 | `qreg [fweight]` 不是行展开 | **错**：是行序错位 |
| 11 | `qreg` 消耗 RNG | **错**：uniform 前后逐位相同 |
| 12 | 行序跨重复累积 | **真实现象**，但不是这个差距的原因 |
| 13 | LP 顶点选择 / `sqreg` 不是它的文档循环 | 观察为真，但差距量级是索引错位造成的 |

**四轮假设都在追一个在对齐后的量上并不存在的差距。**

### 这一系列工作真正确定的东西（仍然有效）

- `bsampl_w` 的权重与 Stata **逐位相等**（n=20/40/200，抽样 1..4）；
- 单次加权拟合 **3.5e-16**；
- 同一展开数组送进 Stata，`solve_lad` 在 6 次重复上 **≤1e-13**；
- 携带置换确实改善方差：**有 0.052543 / 无 0.049441，Stata 0.054044**，即**有**比**无**更接近。

### 分级建议

bootstrap VCE 在数百 reps 下**不可能**与 Stata 逐位一致——估计量本身未收敛。因此：

- **系数**：确定性 `qreg` 拟合，**E3**；
- **VCE**：应当表述为蒙特卡洛误差范围内的分布性一致，而不是逐点一致。

这不是「已修复」，而是**把一个错误的差距归因纠正为正确的量级判断**。

## 8.19 `xtgls corr(ar1)`：从 ado 源码实现，达到机器精度

这条此前是 **E1 能力缺口**（抛 `UnsupportedError`）。现在已实现，实测**系数 6.6e-16、标准误 1.3e-16**。

### 算法（`xtgls.ado` `mc1` + `_crcar1.ado` `regress` 分支）

```
1. 原始数据 OLS，取残差 e
2. 每个面板：rho_i = Σ(e_t · e_{t-1}) / Σ(e_{t-1}²)      ← 无截距最小二乘
3. 跨面板合并成【一个】共同 rho，按面板观测数加权：
       rho = Σ_i rho_i · N_i / Σ_i N_i
4. Prais-Winsten 变换（逐面板）：
       首期   q = sqrt(1-ρ²) · x
       其余   q = x_t − ρ · x_{t-1}
5. 对【变换后】的数据走原有 GLS 累加
```

**三个容易搞错的点，都按源码处理：**

1. `rho` 是**按面板样本量加权合并**，不是等权平均；
2. 变换是 **Prais-Winsten**（首期缩放保留），不是丢掉首期的 Cochrane-Orcutt；
3. 变换在 **GLS 加权之前**，所以估计量不变、变的是数据。

### 实测

| 量 | 相对差 |
|---|---|
| 系数 | **6.640e-16** |
| 标准误 | **1.253e-16** |

数据：25 面板 × 8 期，带持续误差分量（ρ≈0.6）。

### `corr(psar1)` 仍然拒绝

`mc2` 对**每个面板**单独估 rho，Omega 的块结构不同。把共同 rho 套上 `psar1` 标签是
**用正确的名字装错误的估计**，所以明确抛错并说明原因。

## 8.20 辅助指标补齐的第一轮：`e()` 存储 + 渲染层分支

本轮针对"Stata 输出的辅助信息 westata 没有"做了一次系统盘点与首批修复。
完整证据见 `retest/gaps/`（`SUMMARY.md` 为总装，另有四份分族报告）。

### 8.20.1 已修复：跨命令的 `e()` 存储

| 命令 | 新增键 | 判据 |
|---|---|---|
| `reghdfe` | `tss`、`mss`、`ll_0` | `reghdfe_firm*.json` 三个键都在，且 `rss+mss=tss` |
| `xtreg, fe` | `N_g`、`r2_w` | `xtreg_fe.json`；`r2_w` **仅 fe**（见 8.20.3） |
| `xtreg, re/be/fd` | `N_g` | 同上；`r2_w` 刻意不加 |
| `xtpcse` | `tss` | 已算出，只是没存 |
| `xtivreg` | `tss`、`tss_within`、`tss_between`、`mss` | 已算出 |
| `svy: regress` | `rss`、`tss`、`mss` | 已算出（`_sums_of_squares()`） |

这些全部是**内存里已有、只是没落进 `e` 字典**的量，因此没有引入任何新计算，
也没有 conformance 风险。

### 8.20.2 已修复：渲染层按估计器分支（原根因）

`results/summary.py` 过去用 **`e(r2_p)` 是否存在** 推断"是不是 ML 模型"。
但 `qreg` 为存 Stata 的 check-function 伪 R² 复用了同一个键
（`quantile.py:417`），于是被误判成 ML 模型，打印了一行 Stata 从不打印的
空 `Log likelihood = .`。

现在 `_HEADER_BLOCK` 把**每种命令对应哪种头部块**显式列出（`ols` / `ml` /
`quantile` / `svy` / `none`），未登记的命令退回"有 `e(ll)` 即 ML"的诚实判断。

顺带修掉三处渲染缺陷：

| 命令 | 修复前 | 修复后 |
|---|---|---|
| `qreg` | 空 `F = .` / 空 `Log likelihood = .` | Stata 的 `Raw sum of deviations ... (about q_v)` / `Min sum of deviations` / `Pseudo R2` |
| `svy: mean/total/proportion` | `F(0,40)=.` / `R-squared=.` / **`Root MSE=4.6094`（语义错误）** | `Number of strata` / `Number of PSUs` / `Population size` / `Design df` |
| `arima`/`arch`/`xtgls` | `R-squared = .` 噪音行 | 不打印拟合块（Stata 也不打印） |

`svy` 那处 `Root MSE` 尤其危险：`summary.py` 的兜底在 `e(rmse)` 缺失时
用 `sqrt(rss/df_resid)` 现算，而对 survey 估计量 `df_resid` 是 design df，
于是得到一个既不是 RMSE 也不是标准差的数（`svy: total` 打印 `0.0000`）。
**宁可打印 `.`，也不能打印一个看起来合理、实际语义错误的数字。**

### 8.20.3 一处被否掉的"廉价修复"

盘点曾建议把 `e(r2_w)` 一律别名到 `e(r2)`。**这是错的**，fixture 直接反证：

| fixture | `e(r2)` | `e(r2_w)` | 关系 |
|---|---|---|---|
| `xtreg_fe.json` | 0.9299367179801631 | 0.9299367179801631 | 相等 |
| `xtreg_re.json` | **不存在** | 0.9299308456407561 | 无 `e(r2)` |
| `xtreg_be.json` | 0.28158005317996504 | 0.7540434430300107 | **不相等**（`r2` 其实是 `r2_b`） |

无条件别名会让 `re`/`be` 在 Stata 的名字下报出错误的数——**比缺键更糟**。
最终只对 `fe` 加 `r2_w`，依据是独立复算确认该估计器的 backend `rsquared`
就是组内 R²（测试面板上手算 0.838219091483203，`e(r2)` 0.838219091483202）。
教训与 §8.4 修正 1 同源：**"看起来同义的两个键"必须逐命令验证，不能推广**。

### 8.20.4 第 2 轮：`xtreg` 方差分量块（全库最大的单块缺口）

`xtreg` 四个估计器过去返回**逐字节相同的 12 个键**，而 `xtreg_fe.json` 有 37 个。
本轮补齐，覆盖度：

| 估计器 | 修复前 | 修复后 | Stata fixture |
|---|---|---|---|
| `xtreg, fe` | 12 | **28** | 37 |
| `xtreg, re` | 12 | **18** | 20 |
| `xtreg, be` | 12 | **20** | 23 |

新增：`Tbar`、`Tcon`、`g_min`、`g_avg`、`g_max`、`r2_w`、`r2_b`、`r2_o`、
`sigma_u`、`sigma_e`、`sigma`、`rho`、`corr`、`ui`、`tss`/`tss_within`/
`tss_between`/`r2_within`（仅 fe）、`mss`（fe/be）。

**实现依据是 `xtreg_fe.ado:358-430`（StataNow 19.5, v2.0.2），不是从输出反推。**

#### 一个静默错误差点被写进去

第一版把 `sigma_u` 写成"组内残差的实体均值的标准差"。**但组内残差的实体均值
按构造恒为 0**，于是 `sigma_u ≡ 0`、`rho ≡ 0`——一个看起来完全合理、
任何系数或标准误检查都发现不了的错值。

`xtreg_fe.ado` 的实际做法是：

```stata
quietly _predict double `XB' if `touse', xb     // 水平拟合值
quietly gen double `U' = `y' - `XB'            // 水平残差（不是组内残差）
`BY' replace `U' = cond(_n==_N,sum(`U')/`Ti',.) // 面板均值
quietly summ `U'
ereturn scalar sigma_u = r(sd)                  // r(sd)，不是总体 sd
ereturn scalar ui      = r(sd)
quietly corr `XB' `U' if `touse'
ereturn scalar corr = r(rho)
```

第二版还错了一次：用 westata 的 `fittedvalues` 当 `XB`。但对 FE，
`fittedvalues` 是 **`y - 组内残差`（去均值拟合）**，其面板均值同样恒为 0。
必须用 **`设计矩阵 @ 系数`** 重建水平预测量。实测修正后
`sd(U_panel) = 0.9877`（真实效应 sd ≈ 1.0），`rho = 0.798`。

**测试钉死在独立复算上**（`test_xtreg_sigma_u_is_the_sd_of_panel_means_of_the_level_residual`），
并在 `sigma_u` 为 0 时给出明确的失败信息——因为这是"错得像个正常值"的那类缺陷。

#### `e(r2_w)` 不是 `e(r2)` 的别名（第三次确认）

| fixture | `e(r2)` | `e(r2_w)` | 关系 |
|---|---|---|---|
| `xtreg_fe.json` | 0.9299367179801631 | 0.9299367179801631 | 相等 |
| `xtreg_re.json` | **不存在** | 0.9299308456407561 | — |
| `xtreg_be.json` | 0.28158005317996504 | 0.7540434430300107 | **不等**（`r2` 是 `r2_b`）|

所以 `r2_w` 是**计算出来的**（组内平方相关），不是赋值。`be` 的 `r2_w`
需要长期面板数据，因此 `_long_response` 在 `_design()` 里保留原始响应。

#### `e(tss)` 也逐命令不同

| fixture | `rss` | `mss` | `tss` |
|---|---|---|---|
| `xtreg_fe.json` | ✅ | ✅ | ✅ |
| `xtreg_be.json` | ✅ | ✅ | **❌** |
| `xtreg_re.json` | ❌ | ❌ | ❌ |

因此 `_stores_totals`（fe 专属）与 `_stores_mss`（be 专属）分开控制，
**不是统一加**。

### 8.20.5 仍未做（已登记，未声称完成）

- `e(tss)` 对 `regress`：Stata **自己不存**（三个 fixture 实证），**不应添加**。
- `reghdfe` 的 `tss_within`/`r2_within`/`r2_a_within`：需要 FE-demeaned 响应，
  PyFixest 适配层当前不保留；**刻意留空而非近似**。
- `xtreg` 剩余：`F_f`/`p_f`（吸收效应联合检验，需新计算）、`df_a`/`k_absorb`/
  `r2_c`、`theta`（re）、`chi2`（re 的 Wald，Stata 报 chi2 而非 F）、
  `ll_0`、`sum_w`、`rc`。
- ML 的 `k*` 块、`ologit`/`heckman`/`etregress` 的 `aic`/`bic`、
  面板族 `e(ic)` 语义冲突（是迭代次数，不是 AIC 惩罚项）。
- `estat gof`/`hettest`/`ovtest`/`vif`/`imtest`/`firststage` 表。
- `manova` 的 Pillai/Lawley-Hotelling/Roy：四条 F 的精确公式已从
  `_canon.ado:676-749` 读出并逐条验证命中 Stata 打印值，但四个统计量需要
  典型相关（`W^-1 B` 广义特征问题），westata 现有 Wilks 路径不构造它。

---

### 8.20.6 第 3 轮：ML 的 `e(ic)` / `k*` / AIC-BIC

#### 最危险的一条：面板族的 `e(ic)` 是迭代次数

Stata 的 `e(ic)` 是 **AIC/BIC 的惩罚项**（参数数 + 1），fixture 实证都是小整数：
`logit_sim.json` 4.0、`probit_sim.json` 3.0、`poisson_sim.json` 1.0、
`nbreg.json` 4.0、`ologit.json` 4.0。

而 `panel_nonlinear.py:725` 存的却是**优化器迭代次数**（实测 `xtlogit fe` 7、
`xtlogit re` 156、`xttobit` 162）。后果：任何把 Stata `estat ic` 逻辑
（`-2*ll + 2*e(ic)`）移植过来的调用者，会算出一个**看起来正常、实际毫无意义**的数，
而且不会报错。

现在 `e(ic) = k + 1`，迭代次数另存为 `e(iterations)`。
测试 `test_panel_e_ic_is_not_the_iteration_count` 直接断言 `e(ic) < 20` 并
检查 `e(ic) == rank + 1`。

#### 补齐的 `k*` 块与信息准则

| 命令 | 新增 |
|---|---|
| `logit`/`probit`/`poisson`/`nbreg`/`zip`/`zinb`/`ztp`/`tnbreg` | `ic`、`k`、`k_dv`、`k_eq` |
| `zip`/`zinb`（有 `inflate` 时） | `k_eq = 2`（两个线性预测子） |
| `ologit`/`oprobit` | `aic`、`bic`、`ic`、`converged`、`k`、`k_dv`、`k_eq`、`k_cat` |
| `heckman, method(ml)`、`etregress, method(ml)` | `aic`、`bic`、`ic`、`k`、`k_dv`、`k_eq` |
| `heckman` 两步 / `etregress` 控制函数 | `k`、`k_dv`、`k_eq`（**刻意不给 AIC**）|

**两点刻意的取舍：**

1. **`ologit`/`oprobit` 之前连 `aic`/`bic` 都没有**（全包唯一没有信息准则的 ML 家族），
   而 `ll`/`rank`/`N` 都已在手——这是两行就能补的。
2. **两步法与控制函数法不给 AIC**：它们不是联合最大似然拟合，没有单一的 `e(ll)`
   可以构造准则。Stata 同样只对 `heckman, method(ml)` 报信息准则。
   **给一个不存在的似然配一个 AIC，比缺少该键更糟**——这与 §8.4 修正 1 同源。

#### 顺带修好的渲染

`xtlogit, pa` 是 GEE（无似然），此前因 `e(r2_p)` 缺失而落进 OLS 分支、
打印空的 `R-squared` / `Root MSE`。第 1 轮的 `_HEADER_BLOCK` 已把它标为
`none`，本轮实测确认它现在只打印 `Wald chi2` 与系数表。

**仍未做**：`e(k_aux)` 对 `logit` 等无辅助参数的模型、`e(rc)`（Stata 返回码）、
`e(crittype)`/`ml_method`/`technique`/`opt` 等宏块。

---

### 8.20.7 第 4 轮：`estat gof`（Poisson 拟合优度）

`result.estat_gof()` / `ws.estat_gof(result)` 现在提供 Stata 的 `estat gof`，
公式逐行抄自 `poisgof.ado`（StataNow 19.5, v1.3.3）：

```
Pearson   chi2_p = Σ (y - ŷ)² / ŷ
Deviance  chi2_d = -2·(e(ll) - llperf)
          llperf = Σ[y·ln y - y] - Σ lngamma(1+y)
          df     = e(N) - e(df_m) - 1
```

**`Σ lngamma(1+y)` 这一项是最容易漏的。** 我第一版漏了它，deviance 从
533.51 变成 1551.62——一个**没有任何报错、看起来仍在合理量级**的错值
（p 值从 0.125 变成 0.0000，会让使用者得出"模型拟合很差"的错误结论）。

发现方式是**逐行手抄 ado 公式做独立复算**，而不是"看起来差不多"。
最终用两条独立路径验证：
1. 手抄公式复算（1e-12）；
2. 逐观测 deviance 残差平方和恒等式 `Σ 2[y·ln(y/μ) − (y−μ)]`（1e-8）。

#### 只对 `poisson` 开放

`poisson.ado:255` 记录 `estat_cmd = "poisson_estat"`，该程序把 `gof` 转给
`poisgof.ado`；**`nbreg` 不设 `estat_cmd`，因此 Stata 没有 `estat gof`**。
所以 `estat_gof` 对非 Poisson 结果抛 `NotImplementedError` 并说明原因，
而不是返回一个"看起来能用"的统计量。测试
`test_estat_gof_refuses_nbreg` 钉住这一点。

---

### 8.20.8 第 5 轮：`estat vif`（方差膨胀因子）

`result.estat_vif()` / `ws.estat_vif(result)` 提供 Stata 的 `estat vif`。
定义抄自 `vif.ado`（StataNow 19.5, v1.5.4）：

```
对每个回归元 j：把它当因变量、其余当自变量重跑 _regress
VIF_j   = 1/(1 - R²_j)
1/VIF_j = 容差
Mean VIF = 各 VIF 的算术平均
```

三个易错点，都按源码处理：

1. **辅助回归的常数项**：`_regress` 默认含常数，但设计矩阵里的常数列
   在去掉目标列时会被一并保留，若不剔除会让辅助 R² 恒为 1。实现里把
   常数列从"others"中剔除、再显式加回截距。
2. **排序**：`vif.ado` 用 `gsort -vvv nms`（VIF 降序，同值按名字）。
   `nosort=True` 保留模型顺序。
3. **`uncentered` 的常数行**：`vif.ado` 在该分支追加 `_cons` 作为一行。

#### 一个方向容易搞反的不等式

`uncentered` 的 R² 分母是**未中心化**的总平方和（数值更大），
所以 R² **更小**、VIF **更大**。我第一版验证脚本把不等号写反了，
断言 `uncentered <= centred` 直接失败——**实现是对的，测试假设错了**。
这个方向性已在测试注释里写明。

#### 两条拒绝规则，都照 vif.ado

- **无常数模型**：`regress, nocons` 后 Stata 报
  `not appropriate after regress, nocons; use option uncentered`。
  westata 同样抛 `ValueError` 并提示 `uncentered=True`。
- **非线性模型**：`estat vif` 经 `regress_estat` 到达 `vif.ado`，
  辅助回归按构造是 OLS，所以对 `logit` 等抛 `NotImplementedError`。

近共线时 `1-R²` 在双精度下留下极小正残差，VIF 得到 4.5e15 这样的
**有限大数**而非 `inf`，因此 `Mean VIF` 仍可计算、输出仍可读
（若返回 `inf`，均值会变成 `nan`）。

---

### 8.20.9 第 6 轮：`estat hettest`（Breusch-Pagan / Cook-Weisberg）

`result.estat_hettest()` 提供 Stata 的 `estat hettest`。算法抄自
`hettest.ado`（StataNow 19.5, v3.4.4）：

```
ui = resid² / (e(rss)/e(N))            ← 标准化平方残差
辅助回归 ui ~ 回归元（无 varlist 时用拟合值）
normal（默认）: chi2 = e(mss)/2
iid          : chi2 = e(N)·e(R²)
fstat        : F = 辅助回归的 F
df = e(df_m)
```

#### 两个决定对错的分歧点

**① 默认统计量是 `e(mss)/2`，不是 `N·R²`。** 二者只有在辅助回归**未中心化**时
才相等。在 auto 数据上，同一个拟合给出 **22.53（默认）** vs **9.54（iid）**——
差 2.4 倍。把 `iid` 的公式当成默认，**不会有任何报错**，只会得到一个
"看起来合理"的错值。

测试 `test_default_is_not_n_times_r2` 直接断言两者不相等，把这个陷阱钉住。

**② 标准化用的是 ML 残差方差 `e(rss)/e(N)`，不是 `rss/(N−K)`。**
分母差一个 `N/(N−K)` 因子，会整体缩放统计量。

#### 三种形态都已验证

| 形态 | 手算 | westata | 一致 |
|---|---|---|---|
| `normal`（默认）| `e(mss)/2` = 22.5346701368 | 同 | 1e-10 |
| `iid` | `N·R²` = 9.5350641274 | 同 | 1e-10 |
| `fstat` | F(3,70) = 3.4512534082 | 同 | 1e-10 |
| 无 varlist（拟合值）| 14.9761469385 | 同 | 1e-10 |

#### 三条拒绝规则，照 `hettest.ado:22-41`

- **robust / cluster VCE**：Stata 报
  `hettest not appropriate after estimation using robust or clustered VCE`
- **`pweight` 数据**：Stata 报 `hettest not appropriate with pweighted data`
- **`normal` 与 `iid`/`fstat` 同用**：Stata 报 `option normal may not be specified`

**仍未做**：`estat hettest` 的 `mtest()` 多重检验与 p 值校正
（`hettest.ado:163-190` 的多变量检验矩阵）。

---

### 8.20.10 第 7 轮：`estat ovtest`（Ramsey RESET 设定检验）

`result.estat_ovtest()` 提供 Stata 的 `estat ovtest`。算法抄自
`ovtest.ado`（StataNow 19.5, v2.7.0）的默认 `freset` 路径：

```
yh  = 拟合值
yh  = (yh - min)/(max - min)        ← 标准化到 [0,1]
yh2, yh3, yh4 = yh², yh³, yh⁴
回归 lhs ~ 原回归元 + yh2 + yh3 + yh4
test yh2 yh3 yh4                    → F(3, df_r)
```

#### 那一步 `[0,1]` 标准化是决定性的

同一份数据（auto，`mpg ~ weight + length`）：

| 版本 | F 统计量 |
|---|---|
| 标准化到 [0,1]（**正确**）| **1.8374159072** |
| 未标准化 | **16672.5498712278** |

**差 9000 倍**，而且两个都"看起来像个数"。未标准化的版本在任何显著性水平
都会拒绝，得出"模型设定有误"的相反结论。

`ovtest.ado` 里那行 `replace yh = (yh-r(min))/(r(max)-r(min))` 极易被当作
无关的数值整理而省略。测试 `test_rescaling_changes_the_statistic` 直接断言
两种算法结果不相等，把这个陷阱钉住。

#### `, rhs` 变体

按 `ovtest.ado` 的 `else` 分支：每个**连续**回归元标准化到 [0,1] 后贡献
`x², x³, x⁴`；指示变量（取值全在 {0,1}）只贡献自身。若无连续回归元，
Stata 报 `powers of all explanatory variables collinear` 且"test not possible"，
westata 抛 `ValueError` 并给出同一提示。

#### 已验证

| 项 | 手算 | westata | 一致 |
|---|---|---|---|
| 默认（拟合值）| F(3,68) = 1.8374159072 | 同 | 1e-10 |
| `, rhs` | F(6,65) = 1.9481714166 | 同 | 1e-10 |
| 含遗漏二次项 | p ≈ 0 | p = 0.000000 | ✓ 拒绝 |
| 设定正确 | — | p = 0.141204 | ✓ 不拒绝 |

---

### 8.20.11 第 8 轮：`predict` 的影响诊断量

`regress` 后可用的 `predict` 类型从 5 种扩到 11 种，新增
`hat`、`leverage`、`rstandard`、`rstudent`、`cooksd`、`dfbeta`。

全部公式都用**留一法暴力复算**验证（不是与另一个闭式比对，避免"实现与测试
共享同一个误解"）：

| 量 | 验证方式 | 一致度 |
|---|---|---|
| `hat` | Sherman-Morrison 恒等式 `h_j = q/(1+q)`，`q = x_j'(Xⱼ'Xⱼ)⁻¹x_j` | 1.6e-14 |
| `rstandard` | `e_j/(s√(1−h_j))` | 0（逐位相同）|
| `rstudent` | **显式重拟合去掉观测 j** | 9.6e-16 |
| `cooksd` | `(b−bⱼ)'(X'X)(b−bⱼ)/(k s²)` | 2.8e-8 |
| `dfbeta` | `(b−bⱼ)/se(b)`，逐观测逐系数 | 7.0e-13 |

#### `rstudent` 是第 6 个"看似合理的错版"

第一版写成 `e_j / (sⱼ·√(1−h_j))`——即 `rstandard` 的形式但换成留一方差。
**这是错的**：它保留了 `√(1−h_j)` 缩放，却漏掉了留一预测残差的 `1/(1−h_j)` 放大。

在 auto 数据上偏差 **9.0e-2**（9%）。正确形式是：

```
e_j^(j) = e_j / (1 − h_j)                     ← 留一预测残差
s²ⱼ     = (rss − e_j²/(1−h_j)) / (df_resid − 1)
r_j     = e_j^(j) / sⱼ
```

修正后与显式重拟合一致到 **9.6e-16**。

这个错版的危险在于**只差 9%，不是量级错误**——任何只看"数量级合理"的检查
都会放过它。测试 `test_rstudent_is_not_rstandard_with_a_leave_out_variance`
直接断言两者不相等。

#### 为什么不用 `cov_params()/s2` 之外的途径

实现从 `cov_params()/s2` 反推 `(X'X)⁻¹`，在 `vce(ols)` 下这**恰好**等于
`(X'X)⁻¹`。这也解释了为什么这些诊断量只对 `vce(ols)` 定义——
`regress_postestimation.sthlp:152-153` 明写 robust/cluster 下不可用，
westata 对上表六种量在非 `vce(ols)` 时抛 `UnsupportedError`。

**仍未做**：`stdr`、`covratio`、`dfits`、`welsch`（Stata 同样列在
`regress_postestimation.sthlp:148-152` 的 vce(ols) 限定组里）；
`score`（线性回归里等价于残差）。

---

### 8.20.12 第 9 轮：`estat ic`（AIC / BIC / AICc / CAIC）

`result.estat_ic()` 提供 Stata 的 `estat ic`。四个公式抄自
`estimates.ado:985-1002`（`estat ic` 背后的实现）：

```
AIC  = -2·ll + 2·df
BIC  = -2·ll + log(N)·df
AICc = -2·ll + 2·df·(N/(N-df-1))
CAIC = -2·ll + df·(log(N)+1)
```

`df` 默认 **`rank(V)`**（系数协方差矩阵的秩），优先级 `df(#)` > `e(df_ic)` >
`rank(V)`（`estimates.ado:955-978`）。`N` 是 `e(N_ic)`（若记录）否则 `e(N)`。

#### 顺带查清一处既有不一致

`e(aic)`（`linear.py`）的值与 `estat ic` 的 AIC **本来就不同**：

| 量 | 公式 | auto 数据上的值 |
|---|---|---|
| `e(aic)` | `N·log(rss/N) + 2k` | 184.6506 |
| `estat ic` AIC | `-2·e(ll) + 2·df` | 394.6535 |

差值恒为 **`N(1+log 2π) = 210.00`**（`e(ll)` 含该常数，前者省略）。

**查证结论：`estimates.ado:953-954` 只读 `e(ll)`/`e(ll_0)`/`e(N)`，
从不读 `e(aic)`。** 所以 `estat ic` 的 AIC 就该是 `-2·ll+2·df`。
且 **Stata 的 `regress` fixture 根本没有 `e(aic)`/`e(bic)` 这两个键**——
它们是 westata 的自有扩展。

已在 `linear.py` 就地更正注释（原文声称"Stata e(aic)/e(bic) for regress"，
与 fixture 不符），并加测试
`test_estat_ic_aic_is_not_the_stored_e_aic` 明确记录两者差
`N(1+log2π)`、**不应相等**。`result.estat_ic()` 是与 Stata 一致的那一个。

#### 已验证

- 四个公式与手算逐项一致（1e-14）
- AICc 的 ado 写法 `2df·N/(N-df-1)` 与教材写法 `AIC+2df(df+1)/(N-df-1)` **等价**（1e-12）
- `CAIC − BIC = df`（代数必然，作为交叉校验）
- `df(#)`/`n(#)` 覆盖生效
- `N-df-1 ≤ 0` 时 AICc 返回 `NaN` 而非编造数值
- 无似然的模型（`xtunitroot` 等）抛 `ValueError`

---

### 8.20.13 第 10 轮：`estat firststage`（IV 第一阶段诊断）

`result.estat_firststage()` 提供 Stata 的 `estat firststage`。列定义抄自
`ivregress_estat.ado` 的 `FirstStage` 程序：

| 列 | 定义 |
|---|---|
| R-sq. | 内生变量对**全部**外生变量（含排除工具）回归的 R² |
| Adjusted R-sq. | 上式按 `N−K` 调整 |
| Partial R-sq. | 残差化内生变量对残差化排除工具回归的 R² |
| F / Prob > F | 排除工具的联合检验 |

#### Shea's partial R²

多个内生变量时另报 Shea's R²（`ivregress_estat.ado:700-736`），三步：
`vhat_i` = 内生 i 对全部外生变量的拟合值；`x1tilde` = 内生 i 对**其余内生**
+ 包含外生的残差；`x1bar` = `vhat_i` 对其余 `vhat` + 包含外生的残差；
`Shea_R2_i` = `x1tilde` 对 `x1bar` 的 R²。

**Shea 的意义必须用相关内生变量才能验证。** 我先用独立的内生变量测试，
Shea 与普通 partial R² 只差 1e-4——**这个测试是空的**。改造成 `corr(x1,x2)=0.56`
后两者差 **0.20**：

| 量 | x1 | x2 |
|---|---|---|
| partial R² | 0.4092 | 0.4747 |
| Shea's R² | **0.5832** | **0.6765** |

Shea 去除内生变量间相关性后重新衡量工具强度，数值明显更高。
测试 `test_shea_differs_from_partial_r2_when_endogenous_are_correlated`
断言差异 > 1e-3，并先检查 fixture 的相关性 > 0.4——**否则测试会因数据
退化而变成空断言**。

#### 已验证

| 项 | 手算 | westata | 一致 |
|---|---|---|---|
| R²（对全部外生）| 0.8955893735 | 同 | 1e-12 |
| Adj R² | 0.8926482291 | 同 | 1e-12 |
| Partial R² | 0.0062533056 | 同 | 1e-10 |
| F(1,72) | 0.4530711941 | 同 | 1e-8 |
| p | 0.5030349229 | 同 | 1e-10 |

弱工具（`length ~ displacement` 净掉 `weight` 后）得 partial R²=0.006、
F=0.45；强工具场景得 partial R²=0.90、F=4445。方向正确。

#### 顺带的结构改动

`IV2SLS` 之前只把合并后的 `X` 交给结果，`estat firststage` 无法区分
"包含外生 / 内生 / 排除工具"三类列。现在结果上保留 `iv_structure`
（三块矩阵 + 各自列名），由 `from_formula` 分别填充名称。

---

### 8.20.14 第 11 轮：`estat overid`（过度识别检验）

`result.estat_overid()` 提供 Stata 的 `estat overid`。公式抄自
`ivregress_estat.ado` 的 `OverID2sls`：

```
sargan  = N·(1 − rss_aux/rss_main)      辅助回归：残差 ~ 全部工具变量
basmann = sargan·(N−k)/(N − sargan)
GMM     → Hansen's J（即 e(J)，estimator 已存）
```

#### 辅助回归的设计矩阵决定对错

`allinst = e(insts)` = **包含外生变量 + 排除工具**，不是只有排除工具。

我先写成"const + 仅排除工具"，得 **Sargan = 0.1783**；正确构造得
**0.4353**——差 2.4 倍，且两者都落在合理的卡方范围里。

诊断方式是逐个试三种设计（含外生+工具 / 仅工具 / const+工具），
看哪一个与手算一致。结论是**含外生 + 全部工具**。

#### 一个容易误解的恒等式

`ado` 写的是 `N·(1 − rss_aux/rss_main)`，而不是常见的 `N·R²`。
二者在**此处恰好相等**，因为 `rss_main` 是残差序列的**中心化总平方和**
（OLS 的 `e(rss)` 就是 `Σe²`，而辅助回归的 TSS 也是 `Σ(e−ē)²`，
在 `ē≈0` 时两者相同）。测试
`test_sargan_equals_n_times_r2_of_the_auxiliary` 显式验证了这个恒等式。

#### GMM 报的是 Hansen J，不是 Sargan

Sargan 假设同方差，J 不假设。`ivregress_estat.ado:428-436` 对 GMM 报
`e(J)`。所以 `estat_overid` 在 GMM 结果上返回 `test="Hansen J"` 且
`basmann=None`——**用 Sargan 冒充 J 会是错误的**。

#### 已验证

| 项 | 手算 | westata | 一致 |
|---|---|---|---|
| Sargan χ²(1) | 0.4352732082 | 同 | 1e-10 |
| Basmann χ²(1) | 0.4200980433 | 同 | 1e-10 |
| p | 0.5094127444 | 同 | 1e-12 |
| Hansen J（GMM）| 0.6709016459 (= `e(J)`) | 同 | 1e-14 |

#### 拒绝规则

- **恰好识别**（无过度识别约束）：抛 `ValueError`
- **robust/cluster VCE**：`OverID2sls` 在该情形把两个统计量设为缺失
  （只报 Wooldridge score 检验，westata 未实现），故抛 `ValueError` 说明原因
- **非 IV 模型**：`NotImplementedError`

---

### 8.20.15 第 12 轮：`estat endogenous`（Durbin-Wu 内生性检验）

`result.estat_endogenous()` 提供 Stata 的 `estat endogenous`。公式抄自
`ivregress_estat.ado` 的 `Endog2sls`（Durbin-Wu 分支）：

```
uc    = 2SLS 残差（一致模型）
ue    = 对 exog + endog 做 OLS 的残差（有效模型，当全部内生变量都被检验时）
USSR  = Σue²
uePue = (ue ~ exog + endog + addlinst, noconstant) 的 e(mss)
ucPuc = (uc ~ exog + addlinst, noconstant) 的 e(mss)
Durbin = (uePue − ucPuc) / (USSR/N)
Wu     = ((uePue−ucPuc)/df) / ((USSR−(uePue−ucPuc))/dfFdenom)
dfFdenom = N − k − df
```

#### 三个必须照抄的细节

1. **`nocons` 是显式的。** `exog` 已含常数项，再让 `_regress` 加一个就重复了。
2. **`USSR` 是有效模型的 rss**，不是一致模型的。用错会让统计量无意义。
3. **`dfFdenom = N − k − df`**，其中 `k` 是结构方程系数个数（含常数），
   `df` 是被检验的内生变量个数。

#### 已验证

| 项 | 手算 | westata | 一致 |
|---|---|---|---|
| Durbin χ²(1) | 0.8296078797 | 同 | 1e-10 |
| Wu F(1,70) | 0.7936618884 | 同 | 1e-10 |
| p_durbin | 0.3623859392 | 同 | 1e-12 |
| p_wu | 0.3760470226 | 同 | 1e-12 |

方向也对：真内生变量（与结构误差相关）检出 **p = 1.3e-147**；
外生变量不误报 **p = 0.5636**。

#### 拒绝规则

- **robust/cluster VCE**：ado 在该情形走**回归型（regression-based）**变体，
  是另一个统计量；westata 未实现，故抛 `ValueError` 说明原因，而**不是**
  用 Durbin-Wu 的名字报一个别的数。
- **非 IV 模型**：`NotImplementedError`。

**仍未做**：`estat endogenous` 的**部分集检验**
（`estat endogenous varlist`，只检验部分内生变量 —— 此时有效模型是受限 2SLS
而非 OLS，代码路径在 `Endog2sls:1160-1170`）；robust 情形的
`regF`/`robwool`/`acwool` 变体（`:1253-1300`）。

---

### 8.20.16 第 13 轮：`xtreg` 三个估计器 **100% 覆盖** fixture

补完剩余键后：

| 估计器 | 覆盖 | 缺失 |
|---|---|---|
| `xtreg, fe` | **37/37** | 无 |
| `xtreg, re` | **20/20** | 无 |
| `xtreg, be` | **23/23** | 无 |

新增：`theta`（re）、`chi2`/`chi2type`/`p`（re）、`k_absorb`/`df_a`/`r2_c`/`F_f`/`p_f`（fe）、
`ll_0`（fe/be）、`sum_w`、`rc`、`p`（fe/be）。

#### 一处我写反了的结论

我曾在第 2 轮的注释里写：

> `e(F_f)`/`e(p_f)`：需要实体效应本身，linearmodels 不返回，故留空而非近似。

**这是错的。** 读 `xtreg_fe.ado:505-524` 后发现三者都来自**一次普通 OLS**：

```stata
scalar `r2'  = 1 - e(rss)/e(tss)              // 组内 R²
quietly _regress `y' `xvars' if `touse'       // 普通 OLS
local r2c = e(r2)
ereturn scalar r2_c = `r2c'
ereturn scalar F_f = ((`r2'-`r2c')/e(df_a))/((1-`r2')/e(df_r))
ereturn scalar p_f = fprob(e(df_a), e(df_r), e(F_f))
```

`F_f` 是**组内 R² 与混合 OLS R² 的比较**，按吸收自由度缩放——**完全不需要实体效应**。

**教训（第 8 次同型）**：我曾用"linearmodels 不返回实体效应"给一个**可实现的量**
披上了"结构限制"的外衣。这与 §8.4 修正 1（把 `ddof` 错误写成"不可约的约定"）
是同一个失败模式：**一个听起来专业的限制，让下一个读者停止寻找。**

#### `r2_c` 不能与 `r2` 简单比较

`xtreg, fe` 的 `e(r2)` 是**组内** R²，而 `r2_c` 是**混合 OLS** R²。
我最初的测试断言 `r2_c >= r2`，在测试面板上失败（0.717 < 0.919）——
**实现是对的，断言错了**。已改为直接复算混合 OLS 的 R² 来对照。

#### 新增的结构标志

为避免"哪些键属于哪个估计器"散落各处，用类属性集中声明：

| 标志 | 含义 | 为 True 者 |
|---|---|---|
| `_stores_totals` | `tss`/`tss_within`/`tss_between` | fe |
| `_stores_mss` | 仅 `mss` | be |
| `_stores_ll_0` | `ll_0` | fe, be |
| `_stores_theta` | `theta` | re |
| `_stores_wald_chi2` | `chi2`/`p` | re |
| `_stores_absorbed` | `k_absorb`/`df_a`/`r2_c`/`F_f`/`p_f` | fe |

---

### 8.20.17 第 14 轮：`predict` 剩余四个诊断量

`regress` 的 `predict` 类型从 11 增至 **15**，补齐 `stdr`、`dfits`、`welsch`、`covratio`。
公式抄自 `regres_p.ado:158-191`（Stata 自己的实现）：

```
dfits    = rstudent · √(h/(1−h))
welsch   = dfits · √((N−1)/(1−h))
covratio = (((N − df_m − t² − 1)/(N − df_m − 2))^(df_m+1)) / (1−h)
           where t = resid/(rmse·√(1−h))
stdr     = rmse · √(1−h)          ← fpredict.ado:62 "Std. err. of the residual"
```

#### 已验证（与手算逐项一致）

| 量 | 最大绝对差 |
|---|---|
| `stdr` | 6.2e-15 |
| `dfits` | 1.2e-14 |
| `welsch` | 1.0e-13 |
| `covratio` | 4.4e-15 |

#### `welsch` 用了两条独立路径验证

`welsch = dfits · √((N−1)/(1−h))`，与 `dfits` 共享中间量。若直接比"welsch 是否
等于该式"，**共享的错误会在两边同时出现而抵消**。所以另加一条：
`welsch/dfits` 必须等于 `√((N−1)/(1−h))`——这条关系不经过 `rstudent`，
能独立发现 `rstudent` 或 `hat` 的问题。

#### 合理性检查

`COVRATIO` 是协方差行列式之比，应围绕 1：实测均值 **1.045**、
范围 [0.510, 1.157]，全部为正。这条不依赖公式复算，能发现量纲或符号错误。

---

### 8.20.18 第 15 轮：`estat classification`（分类表）

`result.estat_classification()` 提供 Stata 的 `estat classification`。实现抄自
`lstat.ado`（StataNow 19.5, v3.2.0；`logit_estat` 把 `classification` 转给它）。

核心是 `cutoff()`（默认 0.5）下的 2×2 表与 9 个百分比：

| `r()` | 含义 | 公式 |
|---|---|---|
| `P_corr` | 正确分类 | `(a+d)/n·100` |
| `P_p1` | 敏感度 `Pr(+\|D)` | `a/(a+c)·100` |
| `P_n0` | 特异度 `Pr(−\|~D)` | `d/(b+d)·100` |
| `P_p0` | 假阳（真 ~D） | `b/(b+d)·100` |
| `P_n1` | 假阴（真 D） | `c/(a+c)·100` |
| `P_1p` | 阳性预测值 `Pr(D\|+)` | `a/(a+b)·100` |
| `P_0n` | 阴性预测值 `Pr(~D\|−)` | `d/(c+d)·100` |
| `P_0p` | 假阳（分类 +） | `b/(a+b)·100` |
| `P_1n` | 假阴（分类 −） | `c/(c+d)·100` |

**注意 `P_p0` 与 `P_0p` 的分母不同**（`b+d` vs `a+b`）——这两个键名极易混淆，
是最可能写错的一处。

「True D defined as `y' != 0」：正类是**非零**响应，对因子编码与 0/1 都成立。

#### 用四组互补恒等式交叉验证

9 个百分比两两构成 4 组互补关系，**每组必须恰好等于 100**：

```
敏感度 + 真 D 假阴率        = 100
特异度 + 真 ~D 假阳率       = 100
阳性预测值 + 分类 + 假阳率   = 100
阴性预测值 + 分类 − 假阴率   = 100
```

这四组覆盖了全部 8 个比率。若任何一个分子/分母写反，配对关系立刻破裂——
比"逐项比对公式"多一层保障，因为**配对的正确性不依赖于我对 `lstat.ado` 的解读**。

#### 已验证

- 混淆矩阵 `a=13, b=6, c=9, d=46`，行列边际全对
- 9 个百分比与手算逐项一致（1e-12）
- 4 组互补恒等式全部 = 100
- `cutoff=0` 时敏感度 100%、特异度 0%（钉住 `>=` 的比较方向）
- cutoff 升高 → 阳性分类数下降、敏感度下降（单调性）
- `cutoff` 越界抛 `ValueError`；非二值模型抛 `NotImplementedError`

---

### 8.20.19 第 16 轮：`estat imtest`（Cameron & Trivedi IM 检验）

`result.estat_imtest()` 提供 Stata 的 `estat imtest`。实现抄自
`imtest.ado`（StataNow 19.5, v2.1.5）：

```
s2     = e(rss)/e(N)                          ← ML 残差方差
rhs2   = 中心化的 x_i·x_j（j ≤ i）

white  : chi2 = N·R² of (res² ~ rhs + rhs2)          ← 中心化 R²
hetero : chi2_h = N·(1 − rss/uss), y = res² − s2, 回归 y ~ rhs + rhs2
skew   : chi2_s = N·(1 − rss/uss), y = res³ − 3·s2·res, 回归 y ~ rhs
kurt   : chi2_k = N·(1 − rss/uss), y = res⁴ − 6·s2·res² + 3·s2², 回归 y ~ 常数
total  = chi2_h + chi2_s + chi2_k
```

#### 三个必须照抄的细节

1. **`s2` 是 ML 残差方差 `e(rss)/e(N)`**，不是 `rss/(N−K)`。skew 与 kurt 分量对
   这个尺度敏感，`ddof` 差一点两者都会偏。
2. **`chi2 = N·(1 − rss/uss)` 用的是「未中心化」平方和**。换成中心化 TSS 得到
   的是另一个统计量。
3. **`rhs2` 的交叉项是中心化的**（`(x_i−mean_i)(x_j−mean_j)`）。

#### 一个不依赖公式抄写的内部一致性检验

**White 检验与 Heteroskedasticity 分量必须相等。** 二者是同一个检验的两种写法，
实测差 **7.1e-15**。

这不是我抄对公式的证据，而是**两条独立构造路径互相印证**：若任一辅助回归的
设计矩阵写错（漏掉交叉项、漏掉常数、多余项），两者会立刻分离。
测试 `test_heteroskedasticity_equals_whites_statistic` 钉住这一点。

#### 已验证

| 项 | 手算 | westata | 一致 |
|---|---|---|---|
| White χ²(5) | 10.3001557434 | 同 | 1e-10 |
| Hetero χ²(5) | 10.3001557434 | 同 | 1e-10 |
| Skew χ²(2) | 6.6695363586 | 同 | 1e-10 |
| Kurt χ²(1) | 1.5316770316 | 同 | 1e-10 |
| Total χ²(8) | 三项之和 | 同 | 精确 |

方向也对：异方差数据（σ ∝ x）hetero p = **1.1e-20**；
同方差数据 p = **0.8794**。

#### 拒绝规则（照 `imtest.ado:13-24`）

- 非 `regress`：`NotImplementedError`
- 加权或 `cluster()`：`ValueError`

---

### 8.20.20 第 17 轮：`estat summarize`（估计样本描述统计）

`result.estat_summarize()` 提供 Stata 的 `estat summarize`。列定义来自
`estat_summ.ado:706-710`：

```
Variable | Obs  Mean  Std. dev.  Min  Max
```

默认汇总**因变量 + 报告出的回归元**，顺序以 `e(depvar)` 打头
（`estat_summ.ado:86,96,503` 都从它开始）。

#### 权重规则（`estat_summ.ado:222-233`）

- **只有 `fweight` 改变 `Obs`**——原文注释是 "a, i, and p weights do not
  change N"，因为 `fweight` 才是字面复制。
- `pweight` 被当作 `aweight` 处理，注释理由："produces okay mean and std. dev."。

#### 加权方差的约定，以及我第一版断言写错了

我先断言"常数权重下 sd 应等于无权重 sd"——**失败**（5.7593 vs 5.7855）。

推导后发现正确关系是：

```
Var_w / Var_unw = c(N−1)/(cN−1)     （常数权重 c）
```

实测方差比 **0.9909502262443438**，公式值 **0.9909502262443439**——
精确到 1e-16。这确认了 replication 方差
`Σw(x−x̄)²/(Σw−1)` 是正确的，也确认了**我的断言错、实现对**。

#### 一处刻意不做声称

`aweight`/`pweight` 的 sd 与 Stata 是否逐位一致，我**没有证据**：
描述统计的加权约定不在 386 点复测范围内。所以：
- 代码里明确写出所用公式；
- docstring 说明"需要引用 aweight 标准差时应在 Stata 中核对"；
- **不声称任何证据等级**。

这与 §8.4 修正 1 的纪律一致：**没有证据就写明没有，而不是让读者以为已经对拍过。**

#### 已验证

- 表内 5 列与 numpy 逐项一致（1e-12）
- 因变量在首行
- `fweight` 使 `Obs = Σw`；`aweight` 保持 `N`
- 常数权重的方差比精确等于 `c(N−1)/(cN−1)`
- 非常数权重的加权均值与手算一致
- cluster 块的组数、min/avg/max 正确
- 未知变量、缺 `weight_type` 均抛 `ValueError`

---

### 8.20.21 第 18 轮：ML 共用 `k*` 块 + `areg` 补充

本轮先做了一次全族覆盖盘点，找出最大公约数缺口。

#### ML 族共缺的 5 个键（全部补齐）

`consonly`、`k_autoCns`、`k_eq_model`、`noconstant`、`rc` —— 每个 ML 估计器都缺。

| 键 | 定义 | 来源 |
|---|---|---|
| `consonly` | 除常数外无回归元时为 1 | 参数名里无斜率 |
| `noconstant` | 无截距时为 1 | 参数名里无 `_cons` |
| `k_eq_model` | 携带模型的方程数（零膨胀为 2） | `k_eq` 的模型部分 |
| `k_autoCns` | 自动生成的约束数（通常 0） | 常数 |
| `rc` | Stata 返回码（成功为 0） | 常数 |

覆盖度变化：

| 估计器 | 修复前 | 修复后 |
|---|---|---|
| `poisson` | 12/18 | **18/18** |
| `ologit` | 15/22 | **22/22** |
| `oprobit` | 15/22 | **22/22** |
| `logit` | 12/20 | 18/20 |
| `probit` | 12/20 | 18/20 |
| `nbreg` | 13/23 | 20/23 |

顺带补了 `nbreg` 的 `e(alpha)`（`npm.exp(/lnalpha)`）与 `k_aux`，
以及 `ologit`/`oprobit` 的 `N_cd`/`version`。

#### `areg`：13/22 → 19/22

新增 `ar2`（= `r2_a` 的另一个拼写）、`k_absorb`、`df_a`、`sum_w`、`rc`、`p`。

#### 三处当时未做 —— 后续记录

> **后续状态（第 22、24 轮）：第 1、2 项均已实现，当时的理由都是错的。**
> - `F_absorb`/`p_absorb`/`r2_c`：第 22 轮补齐（`areg` 达 22/22），见 §8.20.25。
>   "PyFixest 不返回吸收效应"是真的，但**不需要**吸收效应——只需混合 OLS 的 R²。
> - `N_cdf`/`N_cds`：第 24 轮补齐（`logit`/`probit` 达 20/20），见 §8.20.27。
>   语义在 `logit.sthlp:243-244` 里写得很清楚，不是"需先定义"。
>
> 这两条都属同一类失败：**用一个听起来专业的理由，掩盖"我还没查"**。
> 保留原文是为了让这个模式可见。

1. ~~**`F_absorb`/`p_absorb`/`r2_c`**：…PyFixest 不返回~~ → 第 22 轮已实现
2. ~~**`logit`/`probit` 的 `N_cdf`/`N_cds`**：这两个是分层计数，语义需先定义~~ →
   第 24 轮已实现
3. **`nbreg` 的 `ll_c`/`chi2_c`/`rank0`**：需要**仅含常数的额外拟合**。
   → **第 25 轮已实现**，理由同样错：comparison model 是 **Poisson** 而非
   "仅含常数"，见 §8.20.28。

> **三条全部实现，三条理由全错。** 这不是巧合：三次都是
> **在没读源码的情况下给出一个"技术上听起来合理"的推断**。
> 这些记录保留在此，正是为了让这个模式无法被忘记。

#### 一个死代码 bug

我给 `_stata_scalars` 加 `k_aux` 时，把 `return scalars` 插在了新代码**之前**，
导致整块变成死代码。发现方式是覆盖盘点显示 `k_aux` 仍未出现——**覆盖面
盘点本身就是回归测试**。

---

### 8.20.22 第 19 轮：`var` 的 `e()`（36/36 全覆盖）

`VARResults` 是包装类（不是 `StataResults`），**此前完全没有 `.e`**——从 Stata
迁移的代码读任何一个 scalar 都会 `AttributeError`。本轮补齐 `var.json` 全部
**36 个键**。

#### 两个我写反了的量

**① AIC/BIC 是「除以 N」而非「乘以 N」。**

```
fixture: N=399, ll=−1144.5503147111253, k=6
e(aic) = 5.767169497298874

(−2ll+2k)/N = 5.76716949729887   ← 精确匹配
−2ll+2k     = 2301.10062942225   ← 399 倍
```

**② 更重要的是：不能直接用 statsmodels 的 `.aic`。**

statsmodels 的 VAR `aic` 是 `log|Σ| + 2k/N`，**省略了 `(1+log2π)·neqs` 常数项**。
同一份拟合上：

| 来源 | 值 |
|---|---|
| statsmodels `.aic` | **0.003442** |
| Stata `(-2ll+2k)/N` | **5.679196** |

差一个常数，**且两个都是"看起来正常"的数值**。所以实现改为用 Stata 自己的
对数似然重算：

```
e(ll)   = −N/2·(neqs·(1+log2π) + log|Σ_ml|)     ← 与 fixture 对到 1e-13
e(aic)  = (−2ll + 2k)/N
e(sbic) = (−2ll + k·ln N)/N
e(hqic) = (−2ll + 2k·ln ln N)/N
e(fpe)  = |Σ|·((N+k)/N)^neqs
```

#### 一个命名陷阱

**`df_m1`/`df_m2` 没有额外下划线**，因为前缀本身已以 `_m` 结尾；
其余 per-equation 键都是 `k_1`/`obs_1`/`r2_1`/`chi2_1`/`p_1` 形式。
第一版我统一写成 `df_m_1`，覆盖盘点立刻显示缺 2 个键。

#### 已验证

- 36/36 键齐全
- `e(ll)` 与 fixture 的 ll 公式对到 1e-13
- `e(aic)`/`e(sbic)` 用 fixture 自身数字复算对到 1e-12
- per-equation `chi2_i` 与 `b_s'V_s⁻¹b_s` 对到 1e-10
- `r2_i`/`rmse_i` 从残差复算对到 1e-12
- `detsig == |Σ|`、`detsig_ml == detsig`、`tmin == lags+1`、`tmax == N+lags`

---

### 8.20.23 第 20 轮：`teffects` 的头部块与 `.e`

`TreatmentEffectResult` 是 frozen dataclass（不是 `StataResults`），**此前既无 `.e`
也无头部块**。而 Stata 在系数表**之前**打印四行（`p11_outputs.txt:8-11`）：

```
Treatment-effects estimation    Number of obs     =      1,200
Estimator      : regression adjustment
Outcome model  : linear
Treatment model: none
```

#### 新增字段

| 字段 | RA | IPW | AIPW | nnmatch |
|---|---|---|---|---|
| `estimator` | regression adjustment | inverse probability weighting | augmented inverse probability weighting | nearest-neighbor matching |
| `outcome_model` | linear | **none** | linear | **none** |
| `treatment_model` | **none** | logit | logit | logit |

**注意 `ra` 与 `ipw` 在两张表里互补**：`ra` 建结果模型但不建处理模型，`ipw` 相反。
这两个 "none" 是 `p11_outputs.txt` 原文，不是占位。

#### `.e` 的键按估计量命名

`ate`/`se_ate`/`z_ate`/`p_ate`，ATET 时换成 `atet`/…。
**ATE 与 ATET 不共用同一个键**——否则混在一张表里就无法区分。

#### 实现细节

`TreatmentEffectResult` 是 frozen dataclass，所以头部用
`dataclasses.replace` 附加（`with_header` 方法），而不是在四个估计器构造函数里
各传四个字符串。

`nobs` 从 influence function 的行数推出（`_summarise` 已有该数组）；
`nnmatch` 走的是 `_summarise_from_standard_error`（无 influence 数组），
所以那里显式传 `nobs=int(self._y.size)`。

#### 已验证

- 四个估计器的三行元数据全部与 fixture 一致
- `nobs` 均为 1200
- `.e` 的 `ate`/`se_ate`/`z_ate`/`p_ate`/`ci_*` 与结果对象字段逐一相等
- 两个 POmean 的均值与标准误都在 `.e` 里
- `str()` 打印完整四行头部
- `with_header` 返回独立副本（frozen dataclass 的要求）

---

### 8.20.24 第 21 轮：r-class 检验结果的 `r()` 映射

`dfuller`/`pperron`/`kpss`/`xtunitroot` 在 Stata 里是 **r-class 命令**（检验，
不是估计），结果存在 `r()` 而非 `e()`。对应的结果类此前**只有属性、没有 `r` 映射**。

#### 单序列：`UnitRootResult.r`

规范来自 `dfuller.ado:134-166` 与 `pperron.ado:136-155`：

| 键 | 含义 | adf | pp | kpss |
|---|---|---|---|---|
| `Zt` | 检验统计量 | ✅ | ✅ | ❌ |
| `stat` | 检验统计量（kpss 专用名） | ❌ | ❌ | ✅ |
| `p` / `pval` | MacKinnon p 值 | `p` | 两者都有 | ❌ |
| `N` / `lags` | 观测数、滞后阶 | ✅ | ✅ | ✅ |
| `cv_1`/`cv_5`/`cv_10` | 临界值 | ✅ | ✅ | ✅ |

**`kpss` 没有 p 值**——它的原假设是平稳，与其他三个相反，Stata 只存 `r(stat)`。
`pperron` 用 `r(pval)` 而非 `r(p)`，但两种拼写都提供，因为迁移动代码可能用任一个。

#### 面板：`PanelUnitRootResult.r`

规范来自 `xtunitroot.sthlp:755-870`。两个检验的头部统计量**不共用键**：

- `ips` → `r(Zt)`（`Z-t-tilde-bar`）+ `r(p)`
- `llc` → `r(tds)`（`t*`）+ `r(p_tds)`

共用键会让合并报表产生歧义，测试
`test_the_two_panel_tests_do_not_share_statistic_names` 直接断言互斥。

`details` 里已有的表内其他行（`t-bar`/`t-tilde-bar`/`delta`/`sbar`/`hac_lags` 等）
原样带入 `r`。

#### 一个 API 陷阱

`XtUnitroot` 的 `test=` 是**构造函数**参数，不是 `fit()` 的。

我第一版写 `fit(method='llc')`，Python 静默接受了未知关键字（被 `**kwargs` 吞掉），
**返回了 `ips` 的结果**——因为默认值就是 `ips`。这类"参数名错了但没报错"的
情况只能靠断言 `r["test"] == "llc"` 发现，所以测试里包含这一条。

---

### 8.20.25 第 22 轮：`areg` 的 `F_absorb`/`p_absorb`/`r2_c`（收回一处"已知缺口"）

第 18 轮我把这三个键登记为"需要混合 OLS R²，PyFixest 不返回吸收效应，故留空"。
**这个判断错了两次：**

1. 第 13 轮已在 `xtreg, fe` 里实现过同一套（那里叫 `F_f`/`p_f`/`r2_c`）。
2. fixture 直接证明两者是同一个量：

| fixture | `F_absorb`/`F_f` | `r2_c` |
|---|---|---|
| `areg.json` | 28.085188842551904 | 0.6955194140102255 |
| `xtreg_fe.json` | **28.085188842551904** | **0.6955194140102255** |

**逐位相同。** 公式（已用手算验证到 1e-9）：

```
F_absorb = ((r2 − r2_c)/df_a) / ((1 − r2)/df_r)
p_absorb = fprob(df_a, df_r, F_absorb)
```

`areg` 现在 **22/22 完全覆盖**。

#### 一个"看起来对"的错值

第一版我用 `model._X` 当设计矩阵，得 `r2_c = 0.9187`。

**但真正的 pooled OLS R² 是 0.7174。** 原因：PyFixest 的 `_X` 已经**去掉吸收效应**
（demeaned），在它上面回归得到的是**组内** R²，不是混合 R²。而 0.9187 恰好接近
`e(r2) = 0.9497`，所以看起来完全合理。

修正后 `r2_c = 0.7173575810880002`，与独立计算的 `0.7173575810880001` 精确一致（1e-16）。

**教训（第 9 次同型）**：又一次是"用错了后端对象上的字段，而它恰好给出一个
数量级合理的数"。检测方法是**独立复算**，不是看值是否合理。

#### 顺带

原来那条"断言三个键不存在"的测试已删除并替换为**断言它们存在且公式正确**。
这正是第 18 轮写下那条测试的目的——**当缺口被填补时，测试会失败并提醒更新文档**。

---

### 8.20.26 第 23 轮：`reghdfe` 的 within 分解块

第 1 轮我把 `tss_within`/`r2_within`/`r2_a_within` 留空，理由是"需要被 FE 去均值的
响应，故意留空"。**实际上只需要一次额外拟合**，本轮补齐。

#### 定义与验证

`tss_within` 是**仅去掉固定效应之后** y 的平方和——等价于
`y ~ 1 | absorb` 的残差平方和。fixture 自身的数字直接证明：

```
1 − 131.2293504838976/418.6969854497428 = 0.6865767964798271
```

正是 `reghdfe_firm.json` 的 `e(r2_within)`。

```
r2_within   = 1 − rss/tss_within
r2_a_within = 1 − (1 − r2_within)·(N−1)/(N − rank)
```

#### 三个容易混淆的"总和"

| 量 | 含义 | 用它算会得到 |
|---|---|---|
| `tss` | y 的**中心化**平方和 | `r2`（普通 R²） |
| `tss_within` | y **仅去 FE** 后的平方和 | `r2_within` |
| 去均值后的**回归**残差平方和 | `rss` | — |

把 `tss_within` 写成 `tss` 会得到普通 R²，**数值上完全正当**，只是不是
`r2_within`。测试 `test_tss_within_differs_from_the_centred_total` 钉住这一点。

#### 多路吸收必须用后端重拟合

单路 FE 可以手写一次 groupby 去均值，但两路及以上**一次去均值不够**。
所以 `_tss_within` 统一用 `feols("y ~ 1 | <absorb>")` 重拟合，
两路吸收自然精确（探针确认与迭代去均值一致到 1e-6）。

#### 顺带补齐的 4 个键

`sumweights`（= `sum_w`）、`converged`、`ic`、`report_constant`。
覆盖度 17/29 → **21/29**。

#### 剩余 8 个刻意不伪造 —— **第 35 轮已全部实现**

> **后续状态：这 8 个键在第 35 轮全部补齐，`reghdfe` 达 29/29 与 32/32。**
> 当时的理由是错的：fixture 自身给出了足够恒等式
> （`df_a = df_a_initial − (df_a_nested + df_a_redundant)`，
> `N_hdfe` = 吸收维度数，`num_singletons = 0`）。见 §8.20.38。
>
> 保留原文是因为这已是**第 8 次**同类失败：把可推导的量登记为"只能编造"。

`N_hdfe`、`N_hdfe_extended`、`N_full`、`df_a_initial`、`df_a_nested`、
`df_a_redundant`、`drop_singletons`、`num_singletons` —— 它们记录 reghdfe
**迭代求解过程中**的单例剔除与嵌套吸收计数。westata 用闭式投影，
这些计数器只能编造。测试 `test_the_eight_singleton_diagnostics_are_not_faked`
显式断言它们不存在。

> ~~测试 `test_the_eight_singleton_diagnostics_are_not_faked` 显式断言它们不存在。~~
> 该测试已改为 `test_the_absorbed_bookkeeping_block_is_recorded`，断言它们**存在且满足恒等式**。

---

### 8.20.27 第 24 轮：`logit`/`probit` 的 `N_cds`/`N_cdf`

第 18 轮我把这两个键列为"分层计数，语义需先定义"。语义其实在帮助文件里：

`logit.sthlp:243-244`：

```
e(N_cds)   number of completely determined successes
e(N_cdf)   number of completely determined failures
```

`frac_154.ado:679` 给出判据：

```stata
if (e(N) == e(N_cds) + e(N_cdf))  →  "outcome is perfectly predicted"
```

即：某个**协变量模式**（回归元的一个不同取值组合）如果结果**全为成功**，
其观测计入 `N_cds`；**全为失败**计入 `N_cdf`；**有变异**则两者都不计。

#### 一个陷阱：fixture 里两个值都是 0.0

**硬编码 0 也能"通过" fixture。** 所以本轮验证刻意分三组：

| 数据 | 模式数 | `N_cds` | `N_cdf` | sum/N |
|---|---|---|---|---|
| auto（`dom~weight+mpg`） | **74 = 行数** | 22 | 52 | 74/74（平凡）|
| 混合模式（稀少+常见 x） | 7 | **37** | 0 | **37/400** |
| 完美预测 | 2 | 25 | 25 | 50/50 |

**第二组是关键**——`37` 既不是 0 也不是 N，证明计数真的在被计算。

第一组之所以 `sum == N`，是因为 auto 里每个 `(weight, mpg)` 组合只出现一次，
**每个模式都是单观测，因而平凡同质**。这不是实现正确的证据，
所以不能只靠它。

#### 已验证

- 三组数据均与**独立的 dict 分组实现**逐项一致
- `mixed_patterns` 的 sum 严格落在 (0, N) 开区间
- 完美预测时 `N_cds + N_cdf == N`（复现 `frac_154.ado:679` 的判据）
- `logit`/`probit` 双双达到 **20/20** fixture 覆盖

---

### 8.20.28 第 25 轮：`nbreg` 的 `ll_c`/`chi2_c`/`rank0` + 修复一个负数卡方

第 18 轮登记的第三条理由是「需要仅含常数的额外拟合」。**又错了。**

#### 定义（`nbreg.ado:215-228` + `nbreg.sthlp:414-422`）

```stata
poisson `y' `xvars' ... , nodisplay      // 拟合 Poisson
scalar `llp' = e(ll)
...
est scalar ll_c = `llp'                  // nbreg.ado:377
est scalar chi2_c = 2*(e(ll)-e(ll_c))    // nbreg.ado:384
```

所以 comparison model 是**同一组回归元的 Poisson 拟合**（α=0 的特例），
不是"仅含常数"的模型。`e(rank0)` 才是 constant-only 模型的秩。

fixture 自身算术确认：`2(ll − ll_c) = 96.45036022672957` 正是 `e(chi2_c)`。

`nbreg` 现达 **23/23 完全覆盖**。

#### 顺带发现一个既有 bug：负数卡方

诊断时注意到 auto 数据上 `e(ll) = −302.36` 比 `e(ll_0) = −231.07` **更负**，
于是 `e(chi2) = −142.59` —— **一个不可能的卡方值被原样报出**。

Stata 的处理在 `nbreg.ado:378-384`：`e(ll) < e(ll_c)` 时**报 0**。

我在 `_model_test` 加了同样的钳制。理由：LR 统计量不能为负——它只说明
数值最优不是真最优，不该把一个不可能的统计量递给用户。

#### 根因核查：是数据问题还是实现问题？

我模拟了**真正来自 NB2** 的数据（α=0.4, n=800）验证：

| 检查 | 结果 |
|---|---|
| α 恢复 | 0.4552（真值 0.4）✓ |
| NB2 优于 Poisson | ✓ |
| `chi2 ≥ 0`、`chi2_c ≥ 0` | ✓ |
| `chi2_c == 2(ll−ll_c)` | ✓ |

auto 数据的 `converged=False` 是 statsmodels NB2 在 74 观测上的**局部最优**，
不是本实现的缺陷。这条已写入 docstring，避免后人误判。

---

### 8.20.29 第 26 轮：`estat hettest, mtest()` 多重检验校正

抄自 `hettest.ado:160-190` 与 `_mtest.ado:189-296`。

#### 算法

`mtest()` 对**每个回归元各做一次**辅助回归（`hettest.ado:170-176`）：

```stata
foreach v of local varlist {
    qui _regress `ui' `v' `wgt' if `touse'
    mat `mtest'[`i',1] = e(mss)/2        // 统计量
    mat `mtest'[`i',2] = e(df_m)
    mat `mtest'[`i',3] = chi2tail(e(df_m), e(mss)/2)
}
_mtest adjust `mtest', mtest(`mtmethod') pindex(3) append
```

**注意统计量是「该单变量回归自己的 `e(mss)/2`」**，不是全模型的那个。

#### 三种校正公式

```
bonferroni: adjp_i = min(1, m·p_i)
sidak:      adjp_i = 1 − (1−p_i)^m
holm:       按 p 升序，乘数 m, m−1, …, 1，并**以上一个校正值为下限**
```

`noadjust` 原样返回。

#### Holm 的下限是最容易漏的一步

我构造了 `p = [0.01, 0.013, 0.02]`、`m = 3`：

```
朴素 step-down: [0.03, 0.026, 0.02]   ← 递减，不是合法校正
带下限:         [0.03, 0.03, 0.03]
```

`_mtest_holm` 与的是**上一个校正值**（不是上一个原始 p 值）。
如果只测 p 值间距大的样本，**这个缺陷完全看不出来**——所以专门加了一条
构造该场景的测试。

#### 已验证

| 检查 | 结果 |
|---|---|
| bonferroni / sidak 与手算公式 | 1e-12 |
| holm 与逐步推导 | 一致 |
| **holm 下限确实生效**（构造递减场景）| `[0.03,0.03,0.03]` |
| 三种方法都不改变检验顺序 | ✓ |
| 统计量与 `e(mss)/2` 逐变量复算 | 1e-10 |
| `df` 每个单变量回归为 1 | ✓ |
| p 值是 `chi2.sf` 上尾 | 1e-12 |
| `r(mtest)` 形状 nvar×3（校正后 ×4）| ✓ |

#### 拒绝规则（`hettest.ado:49-61`）

`iid` 与 `fstat` **都不能与 `mtest()` 同用**，均抛 `ValueError`。

---

### 8.20.30 第 27 轮：`estat overid` 的 Wooldridge score（稳健/HAC）

第 12 轮我让 `estat overid` 在稳健 VCE 下**直接报错**，理由是"score 检验尚未实现"。
本轮补齐，改为**报告 score 检验**。抄自 `ivregress_estat.ado:184-261`。

#### 算法

```
1. endoghat_i = 每个内生变量对「全部工具变量」（含外生回归元 + 排除工具）回归的拟合值
2. khat_j     = 每个排除工具变量对 (endoghat..., 含内生成分) 回归的残差 × IV 残差
3. score      = N − rss，其中把 ones 对 khat 回归（nocons）
   HAC 且 lags>0 时先对 khat 做 prewhitening 再第 3 步
```

`khat` 只取**过度识别约束个数**列——ado 注释（199-201）：
"which Q excluded exog. vars is irrelevant -- statistic is the same"。

#### 决定性的验证：异方差下 score 必须与 Sargan **不同**

`ado:181-182` 自己写着：

> NB, if we assume homoskedastic errors, then it's the same as Sargan's test.
> Otherwise, it's not.

于是分两组数据验证：

| 数据 | Sargan | Wooldridge score | 差 |
|---|---|---|---|
| 同方差 | 0.0006254 | 0.0006162 | 9e-6（一致）|
| **异方差** | **0.1411847** | **0.0784173** | **6.3e-2**（显著不同）|

**如果两条路径永远给同一个数，说明稳健分支没有真的在加权**——而只用同方差数据
测试的话，这个缺陷完全不可见。所以异方差那组是必要的。

两组数据均与**独立逐步转录**一致到 1e-10。

#### cluster 仍拒绝

`ivregress_estat.ado:290-296`：clustered 误差下不提供过度识别检验。保持抛错。

---

### 8.20.31 第 28 轮：修正 prewhitening 的样本处理，并标注 HAC 路径不可达

第 27 轮我实现了 Wooldridge score 的 HAC prewhitening 分支，**但从未验证过它**。
本轮去验证，发现两个问题。

#### 问题一：零填充是错的

`ado:255-259` 用：

```stata
qui mvreg `khat' = L(1/`lags').(`khat') if `touse' `wt', nocons
```

`mvreg` **丢弃前 `lags` 行**（它们没有滞后值），所以第 3 步的样本是 `N − lags`。

我上一轮写的是**用零填充前 `lags` 行，保留 N 行**。这是结构性错误：
那些行不是"残差为 0"，而是**根本没有做回归**。零行会同时进入设计矩阵和 N。

#### 问题二：这个 bug 在 Q=1 时完全不可见

对比两种处理：

| Q（过度识别约束数）| drop | pad | 相对差 |
|---|---|---|---|
| **1** | 3.2848405774 | 3.2848405774 | **0（逐位相同）** |
| 2 | — | — | 4.0% |
| 5 | — | — | 6.9% |

原因：`Q=1` 时第 3 步是「ones 对**单列**无常数回归」，闭式解为
`(Σx)²/Σx²`——零行对分子分母**同时贡献 0**，恰好抵消。

而 **Q=1 正是最常见的情形**（一个过度识别约束），所以我最初用 `Q=1` 数据做
A/B 对比时看到"两法一致"，差点得出"零填充无害"的结论。

测试 `test_padding_is_neutral_only_at_one_restriction` 把边界两侧都钉住：
`Q=1` 必须相同，`Q≥2` 必须不同。

#### 问题三：HAC 路径根本不可达

`SUPPORTED_COV_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster")` ——
**没有 HAC**。所以 `estat_overid` 永远收不到 HAC 结果，prewhitening 分支
无法通过公开 API 触达。

处理方式：保留该分支（它是 ado 规范的一部分），在函数层验证其算术，
并**明确标注没有与 Stata 的 HAC 输出对拍过**。测试
`test_no_hac_vce_can_be_fitted` 直接断言这一点，避免后人以为已验证。

同时补齐 `lags` 的 adoc 语义：HAC 默认 **1**（`ado:87-95`），
非 HAC 时给 `lags` 应报错（`ado:80-83`）。

---

### 8.20.32 第 29 轮：Newey-West HAC 协方差（`westata.vce.hac`）

第 28 轮确认 `SUPPORTED_COV_TYPES` 里**没有 HAC**，导致 prewhitening 不可达。
本轮实现 HAC 本身。抄自 `newey.ado:133-183`。

#### 公式

```
g_t   = x_t · e_t                                    （得分贡献）
cross_j = Σ_t g_t g_{t−j}'
tt_j  = (cross_j + cross_j') · (1 − j/(1+lag))       ← Bartlett 核
j = 0:  xtx = tt_0 · 0.5     → 等于 cross_0（它对称）
j > 0:  xtx += tt_j
V = (X'X)^-1 · xtx · (X'X)^-1 × N/(N−K)
```

`j=0` 的 `0.5` 只是抵消重复的转置。实现里直接写 `meat = cross_0 + Σ_{j≥1} w_j(cross_j + cross_j')`，
避免因子 2 的记账错误——已与 ado 逐步转录逐位对齐（5 个 lag 全部 0 误差）。

#### 一个必须区分的参照物

`newey.sthlp:113-114` 说 `lag(0)` 等价于 `vce(robust)`。

**Stata 的 `vce(robust)` 带 `N/(N−K)` 因子，所以 `lag(0)` 对应 HC1，不是 HC0。**

我第一版拿 HC0 对比，看到比值恰好 `N/(N−K) = 100/98 = 1.0204`，
**看起来像核函数 bug**，实际是缩放约定不同。改成对比 HC1 后：
`max|HAC(lag=0) − HC1| = 3.5e-18`。

#### 方向性验证做了双向

最初我写「加大 lag → 方差增大」，**测试失败**（0.00457 > 0.00389）。
原因是 HAC 加滞后**不必然**增大方差——交叉项带自相关的符号。

改为两个方向分别构造：

| 场景 | lag 增加时 |
|---|---|
| 得分正自相关（x 与 e 同向 AR） | 方差**增大** ✓ |
| 得分负自相关（e 交替符号） | 方差**减小** ✓ |

**只做正向测试的话，核权重符号写反也能通过。**

#### `IV2SLS` 仍不接受 HAC

IV 的协方差来自 linearmodels，没有 Newey-West 选项。所以：

```python
ws.IV2SLS(...).fit(cov_type="HAC")
# UnsupportedError: ... the IV covariance comes from linearmodels, which has
# no Newey-West option here ... The HAC machinery itself exists and is
# exercised by compute_cov (westata.vce.hac); wiring it into IV2SLS needs the
# IV VCE path rewritten.
```

**明确告知"能力存在但未接线"，而不是假装支持。** 这意味着第 28 轮标注的
prewhitening 不可达状态**仍然成立**。

---

### 8.20.33 第 30 轮：把 HAC 接入 `IV2SLS`

第 29 轮实现了 HAC 却标注"未接线"。本轮接上。

#### 为什么不需要重写 IV 的 VCE

2SLS 是 `kappa == 1` 的 k-class 估计量，其 bread 就是 `Xhat'Xhat`——
**这正是 `compute_cov` 已实现的 OLS 形状三明治**。所以只需把带宽转发过去。

修改点：`_prepare_cov_kwds` 接受 `maxlags`（存到 `self._hac_kwds`），
`_shared_vce` 把它转发给 `compute_cov`。

#### 一个 0.995 的比值

第一版验证：手算 sandwich 与 `fit.cov_params()` 的比值**恰好 0.995 = 398/400**。

起初看起来像核函数写错。实际是 **`N/(N−K)` 因子没应用**：

`newey.sthlp:113-114` 说 `lag(0)` 等于 `vce(robust)`，而 Stata 的 `vce(robust)`
**无条件**带 `N/(N−K)`。`_shared_vce` 里 `cov_ssc` 只在 `not settings.adj`
时才重建，导致 HAC 拿到的因子为 1。

修正为：HAC 单独一支，**始终** `adj=True`。修正后手算与实现差 **0.000e+00**。

#### `ssc(adj=False)` 对 HAC 无效——这是对的

我一开始写测试断言 `ssc(adj=False)` 会去掉因子。**失败。**

`newey` 的文档里**没有**关闭该因子的选项，所以 westata 也不该允许。
测试改为：
- 用 `cov_hac(..., ssc=SSC(adj=False))` 拿原始 sandwich，确认比值恰为 `N/(N−K)`；
- 确认 `ssc(adj=False)` 对 IV 的 HAC 拟合**没有任何影响**。

#### 端到端可达性验证

`estat_overid` 现在真的走 prewhitening 分支了：

| `lags` | score |
|---|---|
| 0（不 prewhiten） | 4.0290915289 |
| **1（HAC 默认，`ado:87-95`）** | **5.3013504157** |
| 3 | 4.0651290190 |

默认值等于 `lags=1`，且与 `lags=0` 不同——**证明分支被执行**。
第 28 轮那条"不可达"的测试已改为断言**可达**。

#### 方向性又是数据决定的

我第三次尝试断言"HAC 增大方差"，**又失败了**：该数据里 `x` 的 lag-1 得分
自协方差是 **−53.5**，所以 HAC 方差**减小**（0.00716 → 0.00525）。

改为检查机制（得分自协方差的符号）而非假设方向。

---

### 8.20.34 第 31 轮：修复一个回归 + 澄清 aweight sd 的真实状况

#### 修复：`IVLIML`/`IVGMM` 的裸 `KeyError`

第 30 轮把 HAC 加进 `_IV_COV_TYPES` 后，`IVLIML`/`IVGMM` 走到
`_LM_COV_TYPES[canonical]` 时抛出**裸 `KeyError: 'HAC'`**——用户看到的是
一个字典键错误，而不是"这个估计器不支持"。

修正为 `UnsupportedError`，说明 HAC sandwich 是针对 2SLS 的 `Xhat'Xhat`
bread 实现的，而 LIML/GMM 的 bread 是 k-class 形式、来自 linearmodels。

测试 `test_liml_and_gmm_never_raise_a_bare_keyerror` 显式守住这一点。

#### `estat summarize` 的 aweight sd：第 17 轮的谨慎是对的，但理由不全

第 17 轮我写"aweight 的 sd 与 Stata 是否逐位一致，我没有证据"，然后留白。
本轮去查，发现状况比我当时描述的更具体：

**公式本身是对的。** `s2_a = Σw(x−x̄)²/(Σw−1)`，aweight 与 fweight **共用这个分母**。

**但当前实现有一个真实的可伸缩性问题：**

| 权重 | 均值 | sd |
|---|---|---|
| `w` | 10.0… | 1.7851545623 |
| `7w` | **相同** | **1.7577445527** |

均值不变（正确），**sd 变了**。而解析权重本应是**尺度不变**的。

原因：`Σw` 随尺度变化，`Σw−1` 也跟着变，两者之比不守恒。若把 `Σw` 归一化
到 `N`，分母就固定为 `N−1`，sd 即可不变。

**但我没有据此修改实现。** 因为"Stata 对描述统计表采用哪种归一化"我没有证据——
第 17 轮拒绝声称的那条结论，本轮依然无法确认。改成猜测的归一化，只是把一个
未验证的数字换成另一个未验证的数字。

所以本轮做的是：
1. 把这段性质**写进 docstring**，说明未归一化、以及原因；
2. 加测试 `test_aweight_sd_is_not_scale_invariant_documented_limitation`
   **把"sd 会随尺度变化"这件事固定下来**，避免后人误以为不变；
3. 加测试 `test_fweight_sd_matches_expanded_data` 确认 fweight 路径
   **确实等于展开数据的 ddof=1 sd**（这是可验证的）。

**这是本轮更重要的判断：知道一个缺陷存在，但不假装已经修好。**

---

### 8.20.35 第 32 轮：`estat endogenous` 的稳健/HAC score 检验

第 12 轮我让 `estat endogenous` 在稳健 VCE 下**直接报错**（"Durbin-Wu 假设未调整
VCE，Stata 在那里换成基于回归的变体"）。本轮补齐 score 检验，改为**报告**。

#### VCE 决定报告哪个检验（`ivregress_estat.ado:1128-1145`）

| `e(vce)` | 检验 |
|---|---|
| `unadjusted` | Durbin-Wu |
| `robust` | `regF` + `robwool` |
| `cluster` | **只** `regF` |
| `HAC` | `acwool` + `regF` |

`cluster` 因此仍然拒绝：只有 `regF`，而 `regF` 需要增广回归，未实现。
**拒绝理由已精确到"因为只有 regF 而 regF 未实现"，不再是笼统的"不支持"。**

#### 公式（`ivregress_estat.ado:1187-1263`）

```
uhat     = y 对 (exog + endog) 的 OLS 残差
endhat_i = endog_i 对 (exog + addlinst) 的拟合值
rhat_i   = endhat_i 对 (exog + endog) 的残差 × uhat
robwool  = N − rss（ones 对 rhat 回归，nocons）
acwool   = 对 rhat 先 prewhiten，再 N − rss
```

#### 两个结构性检查

**① score 必须与 Durbin-Wu 不同。** 它们是不同的检验；若结果相同，说明分派是装饰性的。

实测：Durbin-Wu = 59.9913，robust score = **53.1399**，确实不同。

**② HAC 必须与未 prewhiten 的版本不同。**

| 版本 | 值 |
|---|---|
| `lags=1`（HAC 默认） | **39.7984** |
| `lags=0`（不 prewhiten） | 53.1399 |

若两者相同，说明 prewhitening 分支是死代码——这正是第 28 轮在 overid 上
修过的同一个坑。

#### 已验证

- `robwool` / `acwool` 与**独立逐步转录**一致到 1e-10
- Durbin-Wu 路径未受影响
- `df` = 内生变量个数
- p 值是 `chi2.sf` 上尾（1e-12）
- `lags()` 仅在 HAC 下有效
- HAC 默认 `lags=1`（`ado:1059-1063`）

---

### 8.20.36 第 33 轮：`regF`（增广回归内生性检验）

第 32 轮唯一遗留的缺口。抄自 `ivregress_estat.ado:1285-1325` 与 `RobReg`
（`ado:2406-2470`）。

#### 算法

```
x2hat_i = endog_i 对 (exog + addlinst) 的拟合值
增广模型：y ~ exog + endog + x2hat
test x2hat  → 联合 F
```

#### 一个把 F 算成 0 的错误

我第一版把增广模型当作**真正的 2SLS**（用工具变量投影）。结果 **F = 0.0**，
而且 p = 1.0。

诊断：

```
rank(aug) = 3     rank(P) = 2     cond(P'P) = 1.7e16
```

`endhat` **按构造就是工具变量的线性组合**，所以它在投影后仍落在工具张成空间内，
其系数**不可识别**，F 退化为 0。

**根因**：`RobReg` 调用的是

```stata
ivregress 2sls `dep' `indep' ..., `vceopt' small
```

其中 `indep = exog test x2hat`，**没有 `(endog = instr)` 块**。
没有那个块时，`ivregress 2sls` 没有内生变量需要工具化，**退化为普通 OLS**。

改正为 OLS 后：**F = 70.636，p = 4.6e-11**，与手算 Wald 形式一致到 1e-10。

#### 解锁 cluster

`ado:1128-1145` 里 cluster **只报 `regF`**，所以第 32 轮只能拒绝。
现在 cluster 可用，`df_denom = G−1 = 49`。

#### robust 的取舍（已记录）

Stata 在 robust 下**同时**报 `regF` 与 `robwool`，而 westata 的
`estat_endogenous()` 只返回一个对象。当前返回 `robwool`（score），
`regF` 通过 `endog_regF_test` 单独可达。测试
`test_robust_regF_differs_from_the_robust_score` 确认两者是**不同的数**。

---

### 8.20.37 第 34 轮：`ManovaResult` 与 `JohansenResult` 的 `e()`

这两个类此前都只有属性、没有 `e` 映射。

#### `manova`（`manova.sthlp:216-225`）

8 个标量：`N`、`k`、`k_eq`、`df_m`、`df_r`、`df#`、`rank`。

- `df_m` = 项的假设自由度 = `g−1`
- `df_r` = `N − g`
- **`df#` 没有额外下划线**（`df1` 而非 `df_1`）——与 `var` 的 `df_m1` 同一约定
- `k` = 每方程 `g` 个参数（截距 + `g−1` 虚拟变量）× 方程数

为此给 `ManovaResult` 加了 `n_equations` 字段，避免从 `statistics` 里猜。开发过程中，我第一版确实在猜，已改正。

#### `vecrank`（`vecrank.sthlp:179-210`）

**标量**：`N`、`k_eq`、`k_dv`、`tmin`、`tmax`、`n_lags`、`k_ce95`、`k_ce99`、`k_cesbic`、`k_cehqic`

**向量**（每个候选秩一项）：`max`、`trace`、`ll`、`lambda`、`k_rank`、`aic`、`sbic`、`hqic`

共 **18 个键**。

```
ll(r) = −T/2·(neqs·(1+ln2π) + ln|S00| + Σ_{i≤r} ln(1−λ_i))
```

#### 决定性验证：`trace` 必须能由同一组特征值重建

```
trace(r) = −N·Σ_{i>r} ln(1−λ_i)
```

实测在 `k_ar_diff = 0,1,2` 三档下均**精确重建到 1e-10**。若 `ll` 与 `trace` 用了
不同的 `N`，这条立刻破裂——**它检验的是两者读的是同一个拟合**，而不仅仅是我抄对了公式。

#### 两个我写错的量

**① `e(n_lags)` 报成 0。** `resolved` 是 `k_ar_diff`（= `lags−1`），而 Stata 的
`n_lags` 是 **VAR 滞后阶**，即 `resolved+1`。`lags=1` 时 `k_ar_diff=0`，
所以应报 1 而非 0。

**② 我一度以为 `e(N)` 错了。** 脚本里写 `expected_n = n-1-1`，而实际
`lags=1 → k_ar_diff=0 → N = T−1 = 199`。**是断言错了，实现对**。

这与第 13 轮的 `r2_c` 断言、第 17 轮的加权 sd 断言是同一类：**先怀疑实现，
但先复算自己的期望值。**

---

### 8.20.38 第 35 轮：`reghdfe` 补齐 → **13 个估计器全部 100% 覆盖**

本轮先做全量覆盖盘点，发现 **12/13 已经 100%**，只剩 `reghdfe` 的 8 个键。
第 23 轮我判定这 8 个"只能编造"。**又错了。**

#### fixture 自身给出了全部恒等式

```
N_full              = 180 = N          （无观测被剔）
N_hdfe              = 1 / 2            （吸收维度数：firm 或 firm+year）
N_hdfe_extended     = 1 / 2            （无连续交互，故相同）
num_singletons      = 0                （无单例）
drop_singletons     = 1                （这是选项标志，非计数）
df_a_initial        = 30 / 36          （各维度水平数之和）
df_a                = 30 / 35
df_a_nested         = 0
df_a_redundant      = 0 / 1
```

决定性恒等式（`reghdfe_firm_year.json`）：

```
df_a_initial − df_a == df_a_nested + df_a_redundant == 1
```

双向吸收时有**恰好一个冗余划分**（共享常数），所以 `36 − (0+1) = 35`。
`N_hdfe` 直接由 PyFixest 的 `_k_fe` 长度给出（1 或 2 个吸收效应）。

#### 一个命令间真实存在的语义差异

`areg` 与 `reghdfe` 对 `df_a` **定义不同**，两个 fixture 各自证明：

| fixture | `df_a` | 含义 |
|---|---|---|
| `areg.json` | **29** | `k_absorb − 1`（截距单独报告）|
| `reghdfe_firm.json` | **30** | `= df_a_initial`（截距被吸收）|

`FEOLS` 对外声称 `model_name="reghdfe"`，故采用 **reghdfe 语义**。
为此加了类标志 `_absorb_df_drops_constant`（`FEOLS` 为 `False`）。

#### 一个连带 bug

`F_absorb` 内部原本重算 `df_a = absorbed_levels − 1`，**绕过了上面算好的
`e["df_a"]`**。双向吸收时它用的是错的 36−1=35 之外的 35 → 实际是 35 vs 36 的
不一致，且 `F_absorb` 与 `e(df_a)` 互相矛盾。改为复用 `e["df_a"]`。

#### 一处必须承认的测试缺陷

我发现第 22 轮起的 `test_areg_*` 系列**用 auto 数据却断言 `areg` 公式**。
而 `areg.json` 的 `cmdline` 是 `areg y x1 x2, absorb(firm)`——**180 行 panel 数据**，
不是 auto。那些测试一直是在**自证**。

已改为：用 auto 数据只验证 `FEOLS`（reghdfe 语义）的内部一致性，
并注明 `areg` 的 29 应在 `test_hdfe.py` 里对照正确的 fixture。

#### 五个 reghdfe fixture 全部 100%

```
reghdfe_firm                 29/29
reghdfe_firm_robust          29/29
reghdfe_firm_cluster         32/32
reghdfe_firm_year            29/29
reghdfe_firm_year_cluster    32/32
```

顺带补了 cluster 版的 `N_clust1`（首聚类变量组数）与
`N_clustervars`（聚类变量个数，本实现恒为 1）。

---

### 8.20.39 第 36 轮：验证面的真实边界（清单式披露）

上一轮发现一类**测试自证**缺陷，本轮系统性排查全库，得到一个比缺陷本身更重要的结论。

#### 事实一：21 个 fixture 里只有 3 个能复现数据

| fixture 组 | 数据集 | 能否数值对拍 |
|---|---|---|
| `regress_auto` / `_factor` / `_robust` | `auto.dta`（**随仓库发布**）| **能** |
| 其余 **18 个** | 合成数据（**未发布**）| 不能 |

每个 fixture 都带 `cmdline` 与（多数带）`datasignature`（如
`180:4:580615418:931814339`），但它们只**证明**数据存在过，**无法重建**。

#### 事实二：fixture 驱动的测试只比对键名，不比对数值

`test_ml_e_block.py`、`test_panel.py`、`test_xtreg_remaining_keys.py` 等
全部只做 `set(fixture["scalars"]) - set(result.e)` 式的**键集合**比对。

这**不是缺陷**——键名正确、存在，是有价值的检查。但它**不证明数值正确**。

#### 事实三：真正比对数值的，是恒等式检验

`test_nbreg_comparison_block.py` 用 `2(ll − ll_c) = chi2_c`、
`test_reghdfe_within.py` 用 `1 − rss/tss_within = r2_within`。

这类测试拿 fixture **自带的数字**验证**它们之间**的关系——不声称复现拟合。
所以它们合法，且价值不低（能同时校验 fixture 与公式）。

#### 结论：必须用清单式披露，而不是笼统说"已对拍"

本仓库 `e()` 表面的验证强度实际分三层：

| 强度 | 覆盖 | 说明 |
|---|---|---|
| **数值对拍** | 3 个 `regress_auto*` fixture | 与 Stata 记录值直接比对 |
| **恒等式校验** | 若干（nbreg、reghdfe、var 等）| 校验 fixture 内部关系 |
| **键名覆盖** | 全部 21 个 | 只保证键存在 |

**此前我在 CHANGELOG 与文档里多次写"100% 覆盖 fixture"——那指的是键名覆盖，
不是数值对拍。** 这个区别此前没有被明确写出，容易被误读。

#### 本轮产出

新增 `tests/test_fixture_verification_scope.py`（5 条），把上述三层固定下来：

1. `REPRODUCIBLE` 集合恰为 3 个 auto fixture——若将来发布更多数据，测试会失败并要求更新
2. 每个 fixture 的 `N` 可恢复
3. **fixture 数值只能出现在恒等式检验中**（用一个"孤立常量"检测判定，
   而非逐项枚举白名单——后者不可扩展，我第一版就是这么写的）
4. auto fixture 确实被数值比对
5. 本节的编号必须存在于文档中

#### 一处我自己的误判

第一版审计脚本用 `f"{value:.6f}"` 匹配，短整数（`3`、`500`、`2.0`）到处误命中，
一度得出"没有任何测试比对数值"的错误结论。加入"跳过整数值"过滤后才看清真相。

**审计工具本身的假阳性，会伪装成被审计对象的缺陷。**

---

### 8.20.40 第 37 轮：收益系数顺序（`stata_params` / `stata_cov_params`）

#### 一个差点被误判为"实现错误"的差异

上一轮确认只有 3 个 fixture 可数值对拍。本轮**逐个 scalar 比对**这三个
fixture，结果：**13/13 全部一致**。

但接着比对 `b` 与 `V` 时看到：

```
regress_auto.json:  max|b − ours| = 41.6797      V 相对差 = 1.16e+07
```

**看起来像拟合完全错误。** 实际原因是**系数排列顺序不同**：

| 来源 | 顺序 |
|---|---|
| Stata `e(b)` | `weight`、`foreign`、`_cons`（**常数在最后**，按公式书写顺序）|
| patsy（westata） | `Intercept`、`foreign[T.Foreign]`、`weight`（**常数在最前**，字母序）|

按 Stata 顺序重排后：**`b` 差 1.6e-12，`V` 差 1.6e-15**。数值一直是正确的。

#### 这是一个真实的可用性缺陷

用户若直接 `result.params` 对比 Stata 的 `e(b)`，会看到首元素差 41.68，
**完全有理由认为结果错了**。

因此新增：

```python
result.stata_param_names    # ['weight', 'foreign', '_cons']
result.stata_params         # e(b) 顺序
result.stata_cov_params     # e(V) 顺序
```

#### 关键：顺序必须从公式恢复

patsy 会**字母序重排**且把截距放首位，**原始书写顺序被销毁**。
所以 `OLS.from_formula` 记录 `_formula_rhs_order` 并传到结果对象。

验证：`mpg ~ weight + foreign` 与 `mpg ~ foreign + weight` 给出**不同**的
`stata_param_names`——证明顺序来自公式，不是字母序或数据顺序。

#### 顺带清理

`linear.py:156` 有一段 `return` 之后的**死代码**（重复的 `cls(...)` 构造），已删除。

---

### 8.20.41 第 38 轮：`stata_params` 的因子变量边界（拒绝而非猜测）

第 37 轮我加了 `stata_params`，只在简单回归上验证。本轮检查**因子变量**这个边界，
发现上一轮的实现会给出**含混且有误导性的名称**。

#### 问题

`regress_auto_factor.json` 有 **4 个**系数：

```
names = ['weight', '0b.foreign', '1.foreign', '_cons']
b     = [-0.00659, 0.0, -1.65003, 41.6797]      ← 基准项系数为 0
```

而 westata 只有 **3 个**（patsy 把基准并入截距）。我上一轮的实现会把
`C(foreign)[T.Foreign]` 映射成 `C(foreign)`——**一个在 Stata 里不存在的名字**。

#### 处理：拒绝，而不是构造一个看起来合理的行

```python
result.stata_factor_terms      # ['foreign']
result.stata_param_names       # raises NotImplementedError
result.stata_cov_params        # raises NotImplementedError
```

理由：给出一行**看似对应 Stata 的 `e(b)` 但实际不存在**的内容，
比报错更危险。

#### 一个必须区分的细节

- **`mpg ~ weight + foreign`**（裸列名）：Stata 报**单行** `foreign`，
  即使该列是字符串（`auto.dta` 的 `foreign` 是 `"Domestic"`/`"Foreign"`），
  而 `regress_auto.json` 确实只有 3 个名字。→ **可以对齐**
- **`mpg ~ weight + C(foreign)`**（显式因子）：Stata 报**逐水平多行 + 基准行**。
  → **拒绝**

判据是**调用者写了什么**（是否 `C(...)`），不是 patsy 产出了什么——
后者对两者都生成 `[T.Foreign]`，无法区分。

**我第一版用"该列是否为 0/1 数值"判断，失败**：`foreign` 是字符串列，
patsy 必须编码成因子，但 Stata 视其为单变量。改为纯语法判据后正确。

#### 已验证

| 情形 | `stata_param_names` | `b` 差 |
|---|---|---|
| 裸列名 `foreign` | `['weight','foreign','_cons']` 与 fixture **一致** | 1.6e-12 |
| 显式 `C(foreign)` | 抛 `NotImplementedError`，消息点名 `foreign` | — |

---

### 8.20.42 第 39 轮：多水平裸变量也必须拒绝（一处重复命名错误）

第 38 轮的判据是"参数名是否以 `C(` 开头"。本轮检查 `reghdfe` + 因子时发现
该判据**漏掉了一个真实错误**。

#### 错误

`y ~ x1 + g | firm`，其中 `g` 是**五水平的字符串列**（裸列名，无 `C()`）：

```
param_names: ['x1', 'g[T.b]', 'g[T.c]', 'g[T.d]', 'g[T.e]']
stata names: ['x1', 'g', 'g', 'g', 'g']         ← 四个系数同一个名字
```

`g[T.b]`、`g[T.c]`、`g[T.d]`、`g[T.e]` 全部折叠成 `g`。**四个不同的系数
共享一个标签**——这不是报错，而是**输出了一个错误的表**，比报错更糟。

#### 修正后的两个判据

一个项在下列任一情形下无法对齐：

1. **展开成多于一个水平列**（`g` → 4 列）——这是裸列名也会触发的
2. **调用者写了显式 `C(...)`**——即使只有两个水平

第 2 条必要，因为二值因子只产生**一个**水平列（基准并入截距），
单靠"多列"判据会放行它。而 `regress_auto_factor.json` 证明
`i.foreign` 有 **4 行**（含系数为 0 的 `0b.foreign`）。

#### 三种情形

| 写法 | 水平列数 | 判定 |
|---|---|---|
| `mpg ~ weight + foreign`（二值裸列）| 1 | ✅ 可对齐，与 fixture 一致 |
| `mpg ~ weight + C(foreign)`（二值显式）| 1 | ❌ 拒绝（含零基准行）|
| `y ~ x1 + g \| firm`（五水平裸列）| 4 | ❌ 拒绝（重复命名）|

#### 已验证

- 三种情形均按上表行为
- 裸二值列的 `stata_param_names` 与 `regress_auto.json` 的 `names` **完全一致**
- `reghdfe` 纯数值回归不受影响（patsy 不重排，天然正确）

---

### 8.20.43 第 40 轮：交互项分隔符（`:` → `#`）

第 37–39 轮在同一个功能上接连暴露三个边界缺陷，说明**我对新功能的边界验证不足**。
本轮改为系统扫描公开属性在 17 种模型形态下的表现。

#### 探针的结论与它的局限

17 个模型**全部返回有限值或清晰拒绝**，0 个静默错误。但探针本身**判别力不足**——
它检测"重复命名""非有限值"，检测不出"名字与 Stata 不一致"这类差异。

补充人工核对后发现一处真实差异：

```
patsy 名: x1:x2
Stata 名: x1#x2        ← 交互项分隔符不同
```

`stata_param_names` 承诺给出"Stata 拼写"，这是遗漏。

#### 修正

```python
elif ":" in name:
    renamed.append(name.replace(":", "#"))
```

安全性依据：**位置已经正确**，只有分隔符不同。所以这是**改名**，不是猜测布局。

#### 验证

| 公式 | `stata_param_names` |
|---|---|
| `y ~ x1 * x2` | `['x1','x2','x1#x2','_cons']` |
| `y ~ x2 * x1` | `['x2','x1','x2#x1','_cons']` |

顺序随公式书写变化，分隔符统一为 `#`。

#### 同时确认一处正确行为

数值型三值列（`rep78` 取值 1–5）在 Stata 里是**连续变量**，不会展开水平。
westata 也如此：`stata_factor_terms == []`，单行 `rep78`。这是对的——
同类列若以**字符串**传入，patsy 会因子编码，那时才应拒绝。

---

### 8.20.44 第 41 轮：辅助参数的位置与斜杠（`lnalpha` / `cut*`）

第 40 轮的扫描覆盖了 `OLS` 族，但**没测 ML 模型**。本轮补上，发现两处差异。

#### 差异一：斜杠是显示格式，不是 `e(b)` 的存储名

| 模型 | fixture 的 `names` |
|---|---|
| `nbreg.json` | `['x1', '_cons', 'lnalpha']` |
| `ologit.json` | `['x1', 'x2', 'cut1', 'cut2']` |
| `oprobit.json` | 同上 |

**两处都没有斜杠。** 而 westata 内部用 `/lnalpha`、`/cut1`——那是 Stata
**打印表格时**的格式（`/` 表示辅助方程），不是 `e(b)` 里的名字。

#### 差异二：辅助参数排在最后

`nbreg.json` 的顺序是 `slopes → _cons → lnalpha`，即**离散度在常数之后**，
而不是混在回归元里。

我第一版只去掉斜杠、没改顺序，得到 `['weight','lnalpha','_cons']`——
**方向反了**。

修正后的 Stata 布局：

```
1. 回归元（按公式书写顺序）
2. _cons（若有）
3. 辅助参数（cut1…、lnalpha）
```

#### 已验证

| 模型 | `stata_param_names` |
|---|---|
| `nbreg` | `['weight','_cons','lnalpha']` |
| `ologit` | `['weight','cut1','cut2','cut3','cut4']` |
| `logit` | `['weight','mpg','_cons']`（无辅助参数，不受影响）|
| `OLS` | 仍与 `regress_auto.json` 一致 |

#### 一条写错的测试

我第一版写 `assert "/lnalpha" in ordered`，把结果对象当作可迭代对象，
抛 `TypeError`。已改为检查 `param_names` 列表。

---

### 8.20.45 第 42–43 轮：IV 的内生变量优先 + 跨估计器一致性固化

#### 第 42 轮：`IV2SLS` 的 `e(b)` 顺序

fixture 的 `cmdline` 与 `names` 对照给出决定性证据：

```
iv_liml.json: cmdline = ivregress liml y x2 (x = z1 z2)
              names   = ['x', 'x2', '_cons']          ← 内生 x 在外生 x2 之前
```

而 westata 的内部设计是 `names = exog + endog`，**恰好相反**。

修正后：

| 公式 | `stata_param_names` |
|---|---|
| `y ~ z1 + [x ~ z2]` | `['x', 'z1', '_cons']` |
| `y ~ [x ~ z1 + z2]` | `['x', '_cons']` |

`_formula_rhs_order` 已排除 `Intercept`，语义干净。

#### 第 43 轮：确认 `reghdfe` 无需额外处理

第 39 轮我判断 `reghdfe` 因无截距、patsy 不重排，所以天然正确。
本轮去验证这个**当时未验证的判断**：

| 公式 | `param_names` |
|---|---|
| `y ~ x1 + x2 \| firm` | `['x1','x2']` |
| `y ~ x2 + x1 \| firm` | `['x2','x1']`（保留书写顺序）|
| `y ~ 1 + x1 + x2 \| firm` | `['x1','x2']`（截距被丢弃）|

**判断成立。**

#### 跨估计器一致性：14 个估计器，0 问题

新增 `tests/test_stata_surface_family.py`（37 条），对全部 14 个估计器断言：

1. 名称**无重复**（第 39 轮缺陷类）
2. `stata_params` 是 `params` 的**置换**（改名而非重算）
3. `stata_cov_params` 对称且对角元素多重集相同
4. `_cons` 在辅助参数**之前**
5. 无法映射时抛 `NotImplementedError`，不编造行

#### 一次自我纠正

扫描时我怀疑 `PanelOLS fe` 不该有 `_cons`。查 `xtreg_fe.json` 后确认
其 `names` 正是 `['x1','x2','_cons']`——**实现是对的，我的怀疑没有依据**。

---

### 8.20.46 第 44–46 轮：LIML/GMM 的 HAC 与过度识别检验分派

#### 第 44 轮：HAC 接入 LIML/GMM——不需要自己实现

我此前判断「LIML/GMM 的 HAC 需替换两步权重矩阵构造」。**探针推翻了这个判断**：
linearmodels 自带 `KernelCovariance`（Bartlett 核 Newey-West），且接受 `kappa`，
所以 k-class（LIML）与 GMM **都能用**。

接入方式只是转发带宽：

```python
config["kernel"] = "bartlett"
config["bandwidth"] = lags
```

#### 一条被推翻的注释

`iv.py` 模块注释写着：

> for LIML `kappa > 1` makes the k-class bread indefinite

**实测不成立**：`B(κ)` 特征值 `[113.84, 312.19]` 全正，条件数 2.74。
`κ = 1.00057`，远不足以让 bread 失去定性。这条注释当初是我为"不实现 HAC"
提供的理由，**现在看也是错的**。

#### 第 45 轮：小样本因子必须统一

接入后实测发现两路径行为不同：

| 估计器 | `HAC(lag=0)` vs `HC0` |
|---|---|
| `liml` / `gmm` | **完全相等**（缺因子）|
| `2sls` | 差 1.0025（有因子）|

`newey.ado:187` 是决定性的：

```stata
mat `v' = (`xtxi'*`xtx'*`xtxi')*(`nobs'/`tdf')
```

**`newey` 无条件乘 `nobs/tdf`，没有 `small` 选项。** 而 `ivregress` 的默认是
`SSC(adj=False)`，HAC 落入 `else` 分支时跟着丢了因子。

为 HAC 加独立分支强制 `debiased=True` 后，三个估计器一致：

```
HAC(lag=0)/HC0 = 1.00250941 = √(400/398)   （三个精确相同）
```

#### 第 46 轮：`estat overid` 的分派矩阵

`liml` 在稳健/HAC 下**错误地返回了 Wooldridge score**。`ivregress_estat.ado:354-357`：

> tests not available after LIML estimation with robust VCE
> use {cmd:forcenonrobust} to override

**LIML 没有稳健/ HAC 的过度识别检验**，不像 2SLS 会退回 score。
为 LIML 报 score 等于把 2SLS 的 `Xhat'Xhat` bread 套到 k-class 拟合上——
**一个没有推导的数字**。

`OverIDgmm`（`ado:408-436`）则**无条件**报 `e(J)`，当前实现与之一致。

修正后的完整分派：

| 估计器 | nonrobust | HC0/HC1 | HAC | cluster |
|---|---|---|---|---|
| `2sls` | Sargan | Wooldridge score | Wooldridge score | 拒绝 |
| `liml` | Sargan | **拒绝** | **拒绝** | 拒绝 |
| `gmm` | Hansen J | Hansen J | Hansen J | Hansen J |

#### 这三轮的模式

44 轮：实现了功能，但只验证「SE 有限、与 robust 不同」。
45 轮：发现**跨路径**行为不一致——单看一条路径看不出来。
46 轮：发现**跨估计器**分派错误——单看 `2sls` 看不出来。

**每轮都在上一轮验证不足的地方继续深入。** 而三次的关键证据都来自
**对照另一个参照物**（另一条路径、另一个估计器、ado 源码）。

---

### 8.20.47 第 47–49 轮：`estat endogenous` 的分派修正与 GMM C 统计量的阻塞证据

#### 第 47 轮：LIML/GMM 被错误地套用了 2SLS 的检验

实测发现 `liml` 与 `gmm` 都返回了 Durbin-Wu / robwool / acwool。

`ivregress_estat.ado:1002-1019`：

```stata
if "`e(estimator)'" == "2sls"        Endog2sls
else if "`e(estimator)'" == "gmm"    Endoggmm
else  di as error "not available after ... estimator" ; exit 321
```

- **LIML 落入最后的 `else`**：完全没有 `estat endogenous`
- **GMM 走 `Endoggmm`**：报告 **GMM C 统计量**，不是 2SLS 的任何检验

#### 第 48 轮：GMM C 统计量的两条路都走不通

`Endoggmm`（`ado:1490-1524`）的算法：

```
S     = invsym(e(W))               // S = W^-1
DropRowCol S : S <被检验变量>        // 删除该矩条件的行/列
Wcon  = invsym(S)
mata: _iv_gmm_cstat(..., Wcon, ..., Jprime)
C     = e(J) - Jprime
```

我尝试用 linearmodels 实现，**两个独立阻塞点**：

1. **后端不支持降维权重矩阵**：
   ```
   ValueError: initial_weight must be a 3 by 3 array
   ```
   删除矩条件后是 2×2，linearmodels 拒绝。
2. **无源码可抄**：`_iv_gmm_cstat` 是 **Mata 内置**，`ado/base/i/` 下无对应实现。

为此新增测试 `test_the_gmm_c_statistic_cannot_be_built_from_linearmodels`，
**显式断言这条错误**。若将来 linearmodels 支持降维 `initial_weight`，该测试失败，
即提示可以重新考虑实现。

#### 第 49 轮：`estat firststage` 无估计器限制（确认实现正确）

三个估计器给出**完全相同**的一阶段结果（`partial_r2 = 0.23428551`）。

查 `FirstStage`（`ado:468-500`）确认：它只依赖 `e(instd)`/`e(insts)`/`e(exogr)`
的划分，**没有任何估计器检查**。一阶段回归与估计方法无关，所以相同是**正确的**。

#### 但发现一个真实缺口

`FirstStage` 返回的是**矩阵**而非 scalar（`ado:514-522`）：

```
r(singleresults)   r(multiresults)   r(mineig)   r(mineigcv)
```

当前 `FirstStageResult` 只有 `r2`/`adj_r2`/`partial_r2`/`f`/`shea` 等，
**缺少 `mineig`（Cragg-Donald 最小特征值）与 `mineigcv`（Stock-Yogo 临界值）**。

`mineig` 是弱工具变量的核心判据。它由 Mata 内置 `_iv_craggdonald_wrk` 计算，
同样**无源码可抄**，且无法对拍。

**本轮不实现它**：在没有 Stata 参照值的情况下新增一个数值，
是第 17 轮（aweight sd）已经拒绝过的那种做法。缺口记录在此，待有参照值时再补。

---

### 8.20.48 第 50–52 轮：`r(mineig)` 与 `r(mineigcv)`（弱工具变量判据）

第 49 轮我把这两个键记为"无参照值、不实现"。**第 50 轮找到了一条不依赖 Stata 的验证路径。**

#### 第 50 轮：`mineig` —— 恒等式提供了参照物

Cragg-Donald 统计量是内生变量与残差化工具变量之间**典型相关的平方**的最小特征值。

关键洞察：**单内生变量时矩阵是 1×1**，其唯一特征值**就是** partial R²。
而 partial F 有恒等式：

```
partial_r2 = F / (F + N − k)
```

于是三个独立计算量必须一致：

```
mineig = partial_r2 = F/(F+N−k) = 0.2111030772778157   （1e-16 内一致）
```

**这就是参照物**，无需 Stata fixture。

#### 由此抓出一个真实 bug

首版得到 `mineig = 0.21434` vs `partial_r2 = 0.21110`，差 **1.5%**。

根因：模型无已含外生回归元时，`included_matrix` 是空矩阵；`_ols` 对空 design
直接返回零拟合，`_residualise` **没做去均值**。

```python
if design.size == 0:
    return np.asarray(y) - np.asarray(y).mean()   # 修复
```

**单独看 0.21434 是个完全合理的数**——只有恒等式能暴露它。

#### 第 51 轮：确认修复把数值移到了「对」而非「不同」

`_residualise` 也被 `partial_r2`/`f` 共用，所以这次修复**改动了既有数值**。
用完全不经过 `_residualise` 的教科书定义复算：

```
partial_R2 = 1 − RSS_full / RSS_restricted
```

四种模型形态，全部 **1e-10** 一致。`f` 同样验证。

**一个修复改变了数字，必须证明新数字是对的**，否则只是换了个数。

#### 第 52 轮：`mineigcv` —— 完整表格就在 ado 里

`Calcmineigcv`（`ado:2483-2668`）把三张硬编码表堆叠：

```stata
mat `cv' = `relbias' \ `tsls' \ `liml'
```

返回 **3×4**：相对偏误（5/10/20/30%）、TSLS 尺寸、LIML 尺寸（各 10/15/20/25%）。

行选择（`ado:2642-2668`）：

| 条件 | 结果 |
|---|---|
| `nendog > 3` 或 `ninst > 30` | 全缺失 |
| `nendog == 1` | 列 1–4 |
| `nendog == 2` | 列 5–8 |
| `nendog == 3`（仅偏误表）| 列 9–12 |
| `nendog > 2` | TSLS/LIML 尺寸行缺失 |

三张表**逐字抄录**（`stock_yogo.py`），不重新推导——它们是 Stock-Yogo 的
已发表模拟值，任何闭式都是"对近似的再近似"。

#### 一个实质结论

在测试数据上 `mineig = 0.203`，而最弱阈值是 `7.25`，**判定为弱工具变量**。

我最初断言"工具足够强能过 25% 列"，**失败**——那是关于数据的假设，不是关于代码的。
改为断言"比较能给出确定结论，且在本数据上不过阈值"。

---

### 8.20.49 第 53–55 轮：`estat summarize` 的权重语义（收回第 17/31 轮的遗留）

第 17 轮记「aweight 的 sd 与 Stata 是否一致，我没有证据」，第 31 轮确认它会随权重
尺度变化但**未修**，理由是「Stata 对描述统计表采用哪种归一化未验证」。

**第 53 轮发现这个理由不成立：Stata 的规则早就写在仓库里。**

#### 第 53 轮：aweight 归一化——参照物就在代码里

`src/westata/data/weights.py:125`：

```python
# aweight and pweight are both normalized so that the weights sum to N;
# normalized = w * (n_obs / total)
```

而实测两条路径的约定**互相矛盾**：

| 路径 | 归一化 | 尺度不变 |
|---|---|---|
| 估计器（`bse`）| ✅ | ✅ `w` 与 `7w` 的 bse 完全相同 |
| 描述统计（`sd`）| ❌ | ❌ `1.85307` vs `1.85071` |

**同一个 `aweight`，两种行为。** 这不是"Stata 未知"，而是内部不一致。

修正后 `aweight`/`pweight`/`iweight` 先归一化到 `sum(w)=N`：

| 情形 | 修正前 | 修正后 |
|---|---|---|
| `aweight` sd @ `w` | 1.85307 | 1.85495 |
| `aweight` sd @ `7w` | 1.85071 | **1.85495**（一致）|

`fweight` 不受影响（它是字面复制，不该归一化）。

#### 连带修正三条旧测试

第 17 轮的 `test_constant_weight_scales_variance_by_the_known_factor`
用 **aweight** 断言尺度因子 `c(N−1)/(cN−1)`——**该恒等式只对 fweight 成立**。
已拆成：

- `fweight` → 保留尺度因子断言
- `aweight` → 断言**不变**（归一化后常数权重无信息）
- 新增 `pweight`/`iweight` 同规则、`Obs` 不变

**旧测试断言旧行为；修了实现必须同步修测试**，否则"测试通过但行为错误"。

#### 第 54 轮：自动沿用估计权重

`estat_summ.ado:208-231` 在 `e(wtype)` 非空时**自动重建权重表达式**并汇总。
而 westata 需要调用者显式传 `weights=`，否则返回**未加权**统计量。

| 情形 | 修正前 | 修正后 |
|---|---|---|
| 加权拟合，不传权重 | 未加权均值 `-0.07361` | **加权均值 `0.00762`** |
| 手算加权均值 | `0.00762` | 一致 |

根因：权重规格没传到结果对象。修法是 `linear.py` 附加
`results.weight_spec`，`summarize.py` 在 `weights is None` 时采用。

显式 `weights` 仍优先；未加权拟合不受影响。

#### 第 55 轮：`fweight` 与展开数据逐列一致

这是对整条加权路径最强的独立验证——**五项全部匹配**（1e-12）：

| 列 | fweight | 展开数据 |
|---|---|---|
| `Obs` | 123 | 123 |
| `Mean` | −0.157677806783 | 同 |
| `Std. Dev.` | 0.919349283257 | 同 |
| `Min` / `Max` | −2.5168 / 2.0004 | 同 |

同时确认 `estat_summ.ado` **不存任何 `e()`/`r()`**，所以 `estat_summarize`
没有 `e()` 缺口是对的。

#### 一个第三方库问题（记录，不修）

`FEOLS` 在**同时使用权重与吸收固定效应**时，PyFixest 在 Rust 层 panic：

```
thread '<unnamed>' panicked at src\demean.rs:71:48:
called `Option::unwrap()` on a `None` value
```

这是 PyFixest 的内部缺陷，与 westata 代码无关。`PanelOLS` 与 `IV2SLS` 本身
不接受权重，所以 `OLS` 是唯一受影响路径，本轮修复已完整覆盖。

---

### 8.20.50 第 58–60 轮：`newey` 独立命令

§3.8 里 `newey` 一直标 ⬜「HAC 回归（模型内 HAC 已支持，独立命令未做）」。
HAC 引擎第 27–30 轮已完成并对拍，**缺的只是命令外壳**。

#### 第 58 轮：实现

关键判断：`newey` **不是新估计器**。`newey.ado:96` 点估计直接调 `_regress`，
只有方差不同，而 Bartlett 核 HAC 已验证。所以实现是复用 `OLS` + HAC VCE。

移植的规范细节：

| 规范 | 来源 | 实现 |
|---|---|---|
| `lag()` 必需、无默认 | `ado:15`、`sthlp:55` | `lag is None` → `ValueError` |
| 负 lag 拒绝 | `ado:24-27` | `ValueError` |
| `tflag==2`（间隔>1）**即使 force 也拒绝** | `ado:74-77` | 无条件拒绝 |
| `tflag==1`（不规则）需 `force` | `ado:78-81` | 有 force 才放行 |
| 仅 `aweight` | `ado:15` `[aw/]` | 其他抛 `NotImplementedError` |

`e()` 必须**删减**：`newey.ado:191` 用 `mat post` 创建全新 `e()`，
`newey.sthlp:156-187` 只有 7 个标量（`N df_m df_r F p lag rank`）。
`regress` 的 `r2`/`rss`/`mss`/`rmse`/`sigma`/`ll` **在 `newey` 里不存在**，
保留会凭空造出表面。

`lag(0)` = **HC1**（非 HC0），实测比值正好 `√(N/(N−K))`，与第 30 轮发现互证。

#### 第 59 轮：predict 与 estat 表面必须收窄

`newey_p.ado:20` 只列四种：`XB Index STDP REsiduals`，而 `regress` 有 15 种。
`regress_postestimation.sthlp:152-153` 明确影响诊断量只在 `vce(ols)` 下可用，
而 `newey` 是 HAC。

`newey_p.ado:54`：

```stata
if "`type'" == "" | "`type'" == "index" | "`type'" == "xb" {
```

**`index` 与 `xb` 同义**——拒绝它就等于拒绝 Stata 接受的东西，已改为映射。

`newey_estat.ado:5-20` 只有一个专用分支 `durbinalt`（要求 `force`），
其余走 `estat_default`。

**一处自造回归**：第 58 轮把 `model_name` 改为 `"newey"`，
而 `predict` 按 `model_name` 查注册表 → 所有 predict 类型报
`no prediction types registered`。已注册 `"newey"`。

#### 第 60 轮：表头必须换成 `newey` 自己的

`newey.ado:245-258` 打印的表头**不是 `regress` 的**：

```
Regression with Newey–West standard errors   Number of obs = N
Maximum lag = lag                            F(df_m, df_r) = F
                                             Prob > F      = p
```

**没有 R²、没有 Adj R²、没有 Root MSE**，左栏多一行 `Maximum lag`。

新增 `_HEADER_BLOCK["newey"] = "newey"`，该种类不打印拟合块，并在 F 行前
输出 `Maximum lag`。

另一处自造回归：`NeweyWestResult.__str__` 先委托 `StataResults.summary()`，
那是 **statsmodels 布局**，会打印 `R-squared: .` / `Adj R-squared: .`——
正是 Stata 从不显示的行。改为调 `render_summary(..., style="stata")`。

#### 一次方向性断言的失败（第 4 次）

写了「lag 增大 → 所有方差增大」，**失败**：

```
截距方差  0.00503 → 0.01695  （增大）
斜率方差  0.00539 → 0.00454  （减小）
```

lag-j 项是**带符号交叉乘积**，方向取决于变量。改为断言「至少发生变化」。

---

### 8.20.51 第 61–63 轮：`manova` 的四统计量与 `e()` 表面

§3.5 里 `manova` 的 Pillai/Lawley-Hotelling/Roy 一直标 ⬜「仅实现 Wilks' λ 与 Rao F」。

#### 第 61 轮：四个统计量，全部从一次特征分解导出

`mv_glossary.sthlp` 给出四个统计量的定义，都是 `E⁻¹H` 特征值 λ 的函数：

| 统计量 | 公式 |
|---|---|
| Wilks' λ | `∏ 1/(1+λᵢ)` |
| Pillai's V | `Σ λᵢ/(1+λᵢ)` |
| Lawley-Hotelling U | `Σ λᵢ` |
| Roy's θ | `λ_max/(1+λ_max)` |

**发现一处真实的数值稳定性问题**：Wilks 原本用 `det(E)/det(E+H)`。
当 `cond(E) ≈ 3e12` 时该式与特征值乘积**相差 2e-4**，而其他三个都精确。
**这是第 4 位数字上的真实误差**，只有对照两条路径才能暴露。

现在四个统计量统一从同一次特征分解导出，不可能互相不一致。

#### 三个我自己犯的错误假设

| 假设 | 实际 |
|---|---|
| `L + V = 1` 精确成立 | 仅**恰好一个**特征值时成立；否则差 1.4e-4 |
| `eigvals(E⁻¹H)` 可靠 | 非对称矩阵会返回**错误符号**的实根 |
| `cond=3e12` 存在精确答案 | 最小特征值 3.4e-11，`E^{-1/2}` 使对称阵**失去对称性**，无路线可得真值 |

第三条尤其重要：我原想断言「某路径正确」，但**该数据不支持**。
改为断言「实现与自身特征值精确一致」——这是能保证的性质。

#### 第 62 轮：`e()` 从 7 个键补到 21 个

`manova.sthlp:216-249` 的完整清单，此前缺 20 个。关键理解：
`e(r2)`/`e(rmse)`/`e(F)`/`e(p_F)` **是逐方程向量**，不是多元检验：

> e(r2) — R-squared for **each equation**
> e(F) — F statistic for **each equation**

用独立 `OLS` 逐方程交叉验证，**精确一致（1e-10）**：

| 响应 | manova R² | OLS R² | manova F | OLS F |
|---|---|---|---|---|
| y1 | 0.36940469 | 0.36940469 | 25.482435 | 25.482435 |
| y2 | 0.17319174 | 0.17319174 | 9.111956 | 9.111956 |

修正旧测试 `test_manova_values_are_floats`：「`e()` 全是 float」过宽，
`manova.sthlp` 明确有字符串宏和向量。

#### 第 63 轮：撤掉一个「声明了但没实现」的宏

第 62 轮我在 `e()` 里写了 `e(predict) = "manova_p"`，但 **`ManovaResult` 没有 predict 方法**。

`manova_p.ado:16` 用的是 `_pred_me`（方程感知），支持 `equation()`/`stddp`/`difference`，
需要设计矩阵与系数矩阵；本结果两者都没保存。

**声明一个会抛 `AttributeError` 的函数是错的。** 已移除 `e(predict)` 与两个
margins 宏（它们描述的正是 predict 选项），并加测试锁住「现在没有 predict」。

#### 一处拒绝做的事

`manovatest` 打印四个统计量各自的 F 近似与 p 值，但 `_manovatest` 是
**内置命令**（无 ado、无 mata、无文档）。**没有公式就不编 F 和 p 值**——
四个统计量的**数值**有 `mv_glossary.sthlp` 的定义，故已实现；F 近似无依据，故不做。

---

### 8.20.52 第 64–66 轮：`manova` 的 predict 与 `mvreg` 的 `e(Sigma)`

#### 第 64 轮：实现 `manova` 的 predict（撤回第 63 轮的「不声明」）

第 63 轮撤掉 `e(predict)` 的理由是「结果不保存设计矩阵与系数，无法实现」。
**那个判断当时是对的。** 本轮补齐数据后即可实现：

- `design_matrix`：截距 + g−1 组虚拟变量（首水平为基准）
- `coefficients`：`(k, p)`、`fitted_values`、`responses`

实现的 6 种类型（`manova_p.ado`）：

| 类型 | 语义 | 验证 |
|---|---|---|
| `xb`/`index` | 拟合值（同义）| 等于各组均值 |
| `stdp` | 预测标准误 | 见第 65 轮修正 |
| `residuals` | 残差 | 组内和为零、`xb+resid` 重构原值 |
| `difference` | `xb1 − xb2` | `ado:100-106` |
| `stddp` | 两方程差的标准误 | 见第 65 轮修正 |

`equation()` 为 **1-based**；缺省取第一个方程（`manova_p.ado:38-44`）。

#### 第 65 轮：两处语义错误

**错误一**：`stdp` 我实现为「组内 `rmse/√n_cell`」。
`manova_postestimation.sthlp:151-154` 的定义是 **`√(x'Vx)`**：

> stdp 计算估计期望值（或均值）的标准误，即拟合值的标准误

在**饱和单因子设计**下两者数值恰好相等（`0.15582963`），
所以捷径「看起来是对的」，但它不是定义。已改为 sandwich 形式。

**错误二**：`stddp` 我实现为 `√(se1²+se2²)`，这**假设两方程误差不相关**。

实测残差相关 **0.17（非零）**，必须含：

```
Var(xb₁−xb₂) = x'V₁₁x + x'V₂₂x − 2x'V₁₂x
```

| 版本 | 值 |
|---|---|
| 忽略协方差（错误）| 0.23204 |
| 含协方差（正确）| **0.21149** |

**相差 9%**，无任何报错、量级正常。独立构建 `V₁₂` 验证，精确匹配 1e-10。

#### 第 66 轮：`mvreg` 的 `e(Sigma)`

`mvreg.sthlp:194` 拼作 **`e(Sigma)`**（「Sigma hat matrix」），
而实现用的是小写 `sigma`。已改为两者并存：

- `e["Sigma"]` 与 Stata 一致
- `e["sigma"]` 保持向后兼容

Σ̂ 公式 `σᵢⱼ = uᵢ'uⱼ/(n−k)` 经独立重算**精确匹配**。

同时补齐 8 个宏：`cmd`、`depvar`、`eqnames`、`properties`、`estat_cmd`、
`asbalanced`、`asobserved`。

#### 第 63 轮的规则再次生效

我起初给 `mvreg` 也加了 `e(predict) = "mvreg_p"`，但实测其 `predict` 方法
**一调用就抛 `ConsistencyError`**（未保存设计矩阵）。

**方法存在 ≠ 方法可用。** 已撤销 `e(predict)` 与两个 margins 宏，
并加测试锁住这一状态。这是同一个规则的第 2 次应用。

---

### 8.20.53 第 67–69 轮：多方程 predict 与脚本宏审计

#### 第 67 轮：`mvreg` 的方程感知 predict

`mvreg` 结果是**堆叠布局**（`nobs = 2n`、`k` 为各方程之和），基类 `predict`
的 `X @ b` 在此**无法定义**。重写 `_SystemResults.predict`：

- `equation()` 为 1-based，缺省取第一个方程
- 按方程分别预测
- `stdp` 用 `σ²ⱼ(X'X)⁻¹` 的 sandwich 形式（沿用第 65 轮教训，不用捷径）

**验证**：方程 1/2 的 `xb` 与独立 OLS 拟合值**精确一致**；`xb+resid` 重构原值；
单方程诊断量正确拒绝（堆叠拟合没有形状正确的 `X`）。

#### 第 68 轮：一个我编造的程序名

第 67 轮我在 `e(predict)` 里写了 `"mvreg_p"`。查证发现：

**`ado/base/m` 下根本没有 `mvreg_p.ado`。**

`mvreg.ado:124` 的真实值：

```stata
est local predict "reg3_p"
```

`reg3_p.ado` 是**多方程共享**的 predict 程序。

**这个错误完全不会报错**——`e(predict)` 只是个字符串，写什么都不影响执行。
这类「从命令名推出程序名」的编造，正是本会话反复警惕的模式。

顺带补上 `sureg` 的 `e(predict) = "reg3_p"`（`sureg.ado:14,45` 也走 `reg3`），
此前**完全没有**。

`sureg` 各方程可有**不同回归元**，故 `_equation_design` 是**矩阵列表**
（不能 `np.stack`），系数是**长度不等的向量列表**。

**又一次列序教训**：我手工构造 `X = [ones, x1]` 验证，但 `_design()` 已把常数
移到**最后**（`[x1, _cons]`，与 Stata `e(b)` 一致），导致误判实现错误。
这是本会话第 3 次在**验证脚本**里犯列序错误。

#### 第 69 轮：脚本宏审计

写脚本抓取所有 `e(predict)`/`e(estat_cmd)`/`e(cmd)`，逐个在 ado 树里查文件是否存在：

| 宏 | 值 | 存在 |
|---|---|---|
| `manova` 的 `predict` | `manova_p` | ✅ |
| `manova` 的 `estat_cmd` | `manova_estat` | ✅ |
| `mvreg` 的 `predict` | `reg3_p` | ✅ |
| `mvreg` 的 `estat_cmd` | `mvreg_estat` | ✅ |
| `sureg` 的 `predict` | `reg3_p` | ✅ |

**0 个编造。** 但审计暴露了反向缺口：`newey.ado:218` 有
`eret local predict newey_p`，而我们的实现把它只存在 Python 属性
`predict_cmd` 上，**`e()` 里没有**。

`newey.sthlp:168-180` 明确列出 8 个宏。现已全部写入 `e()`，
`e()` 从 7 个键增至 15 个：

`cmd`、`title`、`vcetype`、`estat_cmd`、`predict`、`properties`、
`asbalanced`、`asobserved`（+ `wtype`，仅加权时）

**教训**：属性存在 ≠ `e()` 存在。Stata 形状的消费者读的是 `e()`。

---

### 8.20.54 第 70–72 轮：默认路径审计与「静默错误」清理

#### 第 70 轮：`Sqreg` 的默认路径崩溃

审计 `e()` 键时（初版报 245 个「缺失」，实为把矩阵/函数/条件键误判），
意外抓到一个真实 bug：

```python
generator = Mt19937_64(DEFAULT_SEED if seed is None else seed)
# NameError: name 'DEFAULT_SEED' is not defined
```

**`Sqreg(...).fit()` 不传 `seed` 就崩溃。**

而**所有既有测试都传了 `seed=`**，所以从未被发现——
功能「已测试、已文档、却在最常见的用法下不可用」。

修复：无种子时每次不同（符合 Stata `c(seed)` 语义），有种子可复现。
**没有改成固定常量**——那会把「崩溃」变成「每次都一样」，是另一种错答案。

#### 第 71 轮：建立「默认路径」测试维度

对 21 个估计器 + 2 个 bootstrap 变体做最小合法调用：

| 结果 | 数量 |
|---|---|
| 正常 | 22 |
| `Eivreg` 拒绝 | 1（**正确**）|
| 崩溃 | **0** |

`Eivreg` 无 `reliab` 时拒绝是对的（否则退化为 OLS 却自称 EIV）。
测试中明确区分了 **「默认跑不起来」与「默认不该跑」**。

#### 第 72 轮：`estat` 交叉乘积审计

对 8 个估计器 × 12 个 `estat` 入口做全矩阵，发现 **2 个静默错误**：

| 方法 | 问题 | 证据 |
|---|---|---|
| `estat_hettest` | logit/poisson 上**返回数值** | `hettest.ado:108` 用 `e(rss)/e(N)`，`:155` 用 `e(mss)/2`，非线性模型**两者都没有** |
| `estat_ovtest` | 同上 | 回归残差对拟合值幂次，是比较平方和的 **OLS 设定检验** |

**最危险的地方：错误数值与正确数值几乎相同。**

| 检验 | logit（错误）| OLS（正确）|
|---|---|---|
| `hettest` | 0.00496 | 0.00487 |
| `ovtest` | 1.3478 | 1.3508 |

两个都「量级正常、无报错」，只有对照才能发现。

同时 `estat_hettest` 在 `nbreg` 上抛 **`IndexError`**（未捕获异常），
已一并变成 `ValueError`。

修正方式：显式要求 `e(rss)` 与 `e(mss)` 存在，否则按估计器名拒绝并说明原因。

**同时加反向保护**：`estat ic`、`estat vce` 对任何模型都适用，
已有测试确保它们**不会**被「一律拒绝非线性模型」的粗暴改法误伤。

---

### 8.20.55 第 73–75 轮：predict 全矩阵审计与面板误差成分

#### 第 73 轮：`IV2SLS` 的 predict 整列不可用

对 **18 种 predict 类型 × 9 个估计器**做矩阵审计，发现 `IV2SLS` 整列 `CRASH`：

```
xb/stdp/stdf/residuals/fitted/hat/... 全部 ConsistencyError
```

根因：`IV2SLS` 从未把 `design_matrix` 传给结果。

**为何长期未被发现**：每个单独的类型测试都断言了「正确拒绝」。
**一列 17 次拒绝，看起来像审慎的设计，而不是缺失的功能。**

这与第 72 轮正好互补：

| 轮次 | 形态 |
|---|---|
| 72 | **该拒绝的却返回了数值**（静默错误）|
| 73 | **该工作的却全部拒绝**（静默缺失）|

两者都只能靠**横向对比整列**发现。

#### 第 74 轮：反转上一轮的判断，实现 `xtreg` predict

第 73 轮拒绝 `xtreg` 的理由是「面板效应语义有分支」。
**查证后发现 Stata 已有明确默认值** — `xtreg_postestimation.sthlp:128-134`：

| 类型 | 语义 |
|---|---|
| `xb` | `a + xb`，**不含** `u_i`，**这就是默认** |
| `xbu` | `a + xb + u_i` |
| `ue` | `u_i + e_it` |
| `u` | `u_i` |
| `e` | `e_it` |

**默认值是写下来的，不是需要我判断的分支。**

同时发现两处「注册了却不能工作」：`ue`（永远抛 `UnsupportedError`）与
`reghdfe`（无设计矩阵）。**注册了却不能运行，比没注册更糟**——
它宣称存在一个不存在的功能。

#### 第 75 轮：实现面板误差成分

第 74 轮我把 `ue` **删掉**了（因为未实现）。本轮查证：**`ue` 是 Stata 真实存在的类型**，
删掉是错的。**正确做法是实现它。**

实现中的关键发现：我最初假设 `fittedvalues = linear`，实测：

```
fittedvalues = y − resid，且 resid 组内均值为 0
```

所以 **`fittedvalues` 已包含 `u_i`**。正确分解：

```
u_i  = mean_by_entity(fittedvalues) − mean_by_entity(a + xb)
e_it = resid
xb   = fittedvalues − u_i
```

**五条恒等式全部验证通过**：

```
xbu == xb + u            ✅
ue  == u + e             ✅
xb + ue == y             ✅
u 组内恒定               ✅
sd(u) == e[sigma_u]      ✅ 0.36129872 精确一致
```

最后一条是最强的独立验证：**从拟合值反推的 `u_i`，其标准差与 Stata 存储的
`e(sigma_u)` 完全相同**。

#### 一个由此产生的回归

第 74 轮删除 `reghdfe` 注册项时，没有检查既有测试
`test_stdp_is_registered_wherever_xb_is_and_stdf_only_for_linear_models`，
它仍在断言 `reghdfe` 有 predict 类型 → 全套出现 1 个失败。

已修正该测试并**新增一条测试锁住「reghdfe 不再注册」**，
避免未来有人误加回去。

**教训：删除注册/功能时，必须搜索既有测试是否依赖它。**

---

### 8.20.56 第 76–78 轮：注册表审计与选项审计

第 72–77 轮的审计都覆盖「方法 × 估计器」和「predict 类型 × 估计器」两个维度，
但**只测默认路径**。本轮补上第三个维度：**可选参数**。

#### 第 76 轮：predict 注册表 3 处「注册了但不可用」

对 24 个模型的注册表逐条验证，发现 6 个类型被登记却无法运行：

| 模型 | 类型 | 根因 |
|---|---|---|
| `nbreg` | `n` | `params.size(3) != exog.shape[1](2)`，被参数数量检查拦截 |
| `newey` | `index` | 基础分派器不支持；它只是 `xb` 的别名 |
| `xtreg, fd` | `xb/stdp/stdf/fitted` | **设计矩阵 2 列 but 只有 1 个系数** |
| `zip`/`zinb`/`ztp`/`tnbreg` | `residuals` | Stata 文档里**没有** `residuals` |

**最严重的两个**：

**① `xtreg, fd`**：差分去掉截距，`param_names` 只剩 `x1`，但传入的 `X` 仍是
2 列 → **每个 predict 类型都报 matmul 维度错误**。

**② `nbreg` 的 `n` 是默认类型**：`nbreg_postestimation.sthlp:91` 写着
「number of events; **the default**」，却被参数检查拦住。
`n = exp(xb)` 只需设计列，`params[:k]` 已能正确切片 —— **纯属检查顺序问题**。

#### 第 77 轮：补上 7 个未验证模型

第 76 轮审计时，7 个模型因**我构造数据的失误**被跳过（`heckman` 缺 `select()`、
`intreg` 缺两列 LHS、`ztp` 数据违反截断）。**跳过 ≠ 验证。**

本轮用正确的构造方式逐个覆盖，全部可用，**没有新的注册表欺骗**：

| 模型 | 正确构造 |
|---|---|
| `heckman` | `w ~ x1 + select(yb ~ x1 + x2)` |
| `etregress` | `w ~ x1 + treat(yb ~ x1 + x2)` |
| `intreg` | `lo hi ~ x1`（两列 LHS）|
| `ztp`/`tnbreg` | 数据需 `y >= 1` |

#### 第 78 轮：26 个选项值 —— 能跑，且真的有影响

对每个 `estat` 方法的**文档化选项**逐一取值：

- `mtest()`：`_mtest.ado:84` 的 `bonferroni/holm/sidak/noadjust`
- `hettest method()`：normal/iid/fstat
- `vif`：`uncentered`/`nosort`
- `imtest`：`white`
- `ic df()`、`classification cutoff()`、`vce correlation=True`

**26 个全部能跑。** 并进一步断言**选项真的改变输出**——一个被接受却
被忽略的选项是最安静的 bug。

#### 一次误报，原因还是测试数据

首轮报告 `vif nosort` 被忽略。查证 `vif.ado:121-122`：

```stata
if !`fvops' & "`nosort'" == "" {
    gsort -`vvv' `nms'
```

默认确实按 VIF 降序排。**问题在我的测试数据**：x1 与 x2 的 VIF 完全相同
（59.41），排序后与未排序看起来一样。

换成 VIF 明显不同的数据后立刻显现：默认 `[x2, x3, x1]`，`nosort` 为
`[x1, x2, x3]`。

**这是本会话第 3 次「测试数据没能区分要测的性质」。**

---

### 8.20.57 第 79–81 轮：`estat` 全估计器审计与三处过度放行

#### 第 79 轮：23 个估计器 × 12 个 `estat`

第 72 轮只覆盖 8 个估计器。本轮扩到 23 个，发现 **2 处未捕获异常**：

```
estat_summarize / Mvreg  -> KeyError
estat_summarize / Sureg  -> KeyError
```

**根因**：多方程模型系数名带方程前缀（`y:x2`/`x1:_cons`），与数据列名
（`x2`/`y`）**永不匹配** → 变量列表为空 → 固定列选择抛 `KeyError`。

依据：`estat_summ.ado:15` 接受 **`eqlist`（方程列表）**，说明本就支持多方程。

修法分三层：解析冒号**后**的部分作回归元、冒号**前**的部分作响应名、
并读取 `e(outcomes)`（`Mvreg` 有，`Sureg` 没有，只能靠前缀解析）。

| 模型 | 修复前 | 修复后 |
|---|---|---|
| `Mvreg` | `KeyError` | `['x1','y']` |
| `Sureg` | `KeyError` | `['x1','y','x2']` |

同时加兜底：变量列表为空时报**清晰的 `ValueError`**（列出可用列），
不让 `KeyError` 穿透——**未捕获异常不是拒绝，它不告诉调用者原因**。

#### 第 80 轮：两处「返回了不该有的数」

**① `estat vif` 在 IV 上返回 OLS 的 VIF**

`IV2SLS` 的 `estat vif` 给出 `1.000203`，与 OLS **完全相同**。

`ivregress_estat.ado:20-56` 只分派 firststage / overid / endogenous /
weakrobust / sbknown|sbsingle，其余走 `estat_default`；而
`estat_default.ado` 只提供 ic / summarize / bootstrap / **vce**——**没有 vif**。

那个数值只是**对同一设计矩阵重算了一遍 OLS 的 VIF**。它看起来合理，正因为
IV 与 OLS 共用设计。

**② `rreg` 后 `estat` 全部不可用**

`rreg.ado:214`：

```stata
est local estat_cmd ""        // reset to empty
```

显式清空。但 `rreg` **有** `e(rss)`/`e(mss)`，所以第 72 轮加的
「有 rss/mss 就放行」检查**放行了** `hettest`/`ovtest`，返回数值。
那些检验假设 **OLS 残差**，而 `rreg` 的残差来自迭代重加权。

**两处的共同点：不是崩溃，而是静默返回一个不该有的数。**

#### 第 81 轮：反向缺口——`reghdfe` 该有的却没有

第 79 轮矩阵里 `FEOLS` 的 `estat_summarize`/`estat_vif` 是 `ref`，
我当时当作合理拒绝。查证发现 **`reghdfe` 在 Stata 中支持这两个命令**，
拒绝理由是「未保留估计样本数据」——**而估计器其实一直持有那份数据**。

修法：`FEOLS.fit` 现在保留 `model_data`、`design_matrix` 与
`response_name`/`e(depvar)`。

**教训**：「拒绝」有两种——**不该支持**（正确）与**本可支持但缺数据**（缺口）。
第 79 轮的矩阵把两者都标成 `ref`，只有逐个追问原因才能分辨。

---

### 8.20.58 第 82–85 轮：按 `*_estat.ado` 的允许清单判定资格

第 83 轮发现同一个估计器上**两种错误方向同时存在**：

- `estat_summarize` **本来可以支持却拒绝**（缺口）
- `estat_hettest`/`ovtest` **本不该支持却返回数值**（过度放行）

原因：我的实现用**单一启发式依据**（「有 `rss`/`mss` 就放行」）判断，
而面板模型恰好两者都有 → 前者被误拒、后者被误放。

#### 权威依据：`xtreg_estat.ado:15-28`

```stata
if (substr(`"`subcmd'"',1, 2) == "su" | `"`subcmd'"' == "ic") {
    estat_default `ZERO'          // summarize / ic 可用
}
local allowed_butsum "hausman mundlak vce"
if strpos(`"`allowed_butsum'"',`"`subcmd'"') == 0 {
    display as error "estat `subcmd' not valid"   // 其余一律拒绝
}
```

`xtreg` 只允许 **5 个**：`summarize`、`ic`、`hausman`、`mundlak`、`vce`。

| 项 | 修复前 | 修复后 |
|---|---|---|
| `estat_summarize` | 拒绝（「未保留数据」）| ✅ 可用 |
| `estat_hettest` | 返回数值 | ❌ 拒绝 |
| `estat_ovtest` | 返回数值 | ❌ 拒绝 |

修法：面板 `fit` 保留 `model_data` 与 `depvar`（使 summarize 可用），
同时在 `hettest`/`ovtest` 中按 `model_name` 显式拒绝。

#### 第 84 轮：其他估计器的允许清单

| 命令 | 允许 | 依据 |
|---|---|---|
| `xtreg` | summarize, ic, hausman, mundlak, vce | `xtreg_estat.ado:15-28` |
| `logit` | classification, gof, auc, + estat_default | `logit_estat.ado:8-24` |
| `rreg` | **无**（`e(estat_cmd)` 被清空）| `rreg.ado:214` |
| `ivregress` | firststage, overid, endogenous, … | `ivregress_estat.ado:20-56` |

**`logit` 的实现完全正确**：`classification`/`ic`/`summarize`/`vce` 允许，
`hettest`/`ovtest`/`vif` 拒绝。

#### 识别出一个真实缺口（未实现，已记录）

`logit_estat.ado:17-19`：

```stata
else if "`key'" == bsubstr("gof",1,max(3,`lkey')) {
    CheckForBad `rest'
    lfit `rest'
}
```

**`logit` 支持 `estat gof`**，走的是 `lfit.ado` —— **Pearson 检验或
Hosmer-Lemeshow**，与 `poisgof` 是完全不同的检验：

| 检验 | 统计量 | df |
|---|---|---|
| `poisgof`（Poisson）| deviance | — |
| `lfit` Pearson | `Σ(yobs−yexp)²/(yexp(1−yexp/w))` | `m − k` |
| `lfit` HL | 分组 χ² | `m − 2` |

**当前拒绝它的理由是「gof 只用于 poisson」——这是错的。**

**本轮不实现**：`lfit.ado:199-250` 的 HL 分支需要复现
`Quantile`/`Qportion`/`Eportion`（`lfit.ado:252-320`）的 quantile 折叠算法，
在预算已用满时仓促加入一个无对拍测试的统计量，正是本会话反复拒绝的做法。

测试 `test_logit_gof_refusal_is_currently_misattributed` **显式断言了
「拒绝理由与真实情况不符」**，防止后人误以为这是正确的拒绝。
公式位置已写入 docstring，供后续实现者查阅。

---

### 8.20.59 第 86–88 轮：`logit` 的 `estat gof`（`lfit` Pearson 检验）

第 85 轮把 `logit` 的 `estat gof` 记为缺口（拒绝理由「gof 只用于 poisson」是错的）。
本轮实现。

#### 依据

`logit_estat.ado:17-19` 把 `gof` 派给 `lfit`。默认分支（无 `group()`）：

```
chi2 = Σⱼ (obsⱼ − expⱼ)² / (expⱼ (1 − expⱼ/wⱼ))
```

按**协变量模式**分组（`lfit.ado:66-73`）。

#### 细节一：`df` 里的 `nvar` 含常数项

首次实现断言 `df = m − 回归元数`，测试失败。查 `Getivars`（`lfit.ado:480-504`）：

```stata
local ivars : colnames `beta'
local nvar  : word count `ivars'      // ← 含 _cons
if "``nvar''"' == `"_cons"' {
    local `nvar'                       // 只从 ivars 里删
}
ret local N_vars = `nvar'              // nvar 不减
```

**`nvar` 取 `e(b)` 的列数（含 `_cons`），只从变量列表移除。**
故 `df = m − (回归元数 + 1)`。

#### 细节二：越界组仍计入 `m`（一处真实错误）

首版实现：

```python
if expected <= 0.0 or expected >= weight:
    continue          # patterns 不增
patterns += 1
```

`lfit.ado:83-92`：

```stata
replace `touse' = (`w'>0 & `w'<.)
gen double `x' = sum(cond(`yexp'>0 & `yexp'<`w' & `touse', ...))
count if `touse'
ret scalar m = r(N)
```

| 项 | 条件 |
|---|---|
| `m`（进入 `df`）| 只看 `w>0` |
| χ² 项 | 还要求 `yexp ∈ (0,w)` |

**越界组只跳过统计量贡献，仍计入 `m` 与 `df`。**

**为何危险**：`df` 直接决定 p 值，少算一个会让 p 值偏大**但数值依然合理**，
不报错、不荒谬——典型静默错误。

修正后：

| 场景 | 修正前 | 修正后 |
|---|---|---|
| 常规（4 模式）| m=4, df=2 | m=4, df=2（不变）|
| 完全分离（2 模式）| m=0, df=−2 | **m=2, df=0** |

#### 细节三：加权情形的诚实拒绝

`lfit_p.ado:123-128` 表明 `lfit` **支持加权**。但本仓库中
`freq_weights` 拟合会把结果数组扩展（`nobs = sum(w)`），而设计矩阵仍是
一行一个观测——实测 `resid` 只有 5 个元素对 300 行设计。

**这些片段已无法可靠重建协变量模式分组**，猜一个对齐会静默地把行分错组。
故**明确拒绝并说明**，而非抛出数组长度错误。

#### 未实现：`group(#)`（Hosmer-Lemeshow）

需要 `lfit.ado:252-367` 的 `Quantile`/`Qportion`/`Eportion` 分位数折叠算法
（含 ties 处理与累积计数分配）。**继续显式拒绝**，不在 HL 标签下返回 Pearson 数值。

#### 一个设计确认

`lfit.sthlp:6-8`：`lfit` 是 **Stata 9 起过时的命令，已被 `estat gof` 取代**。
故**不暴露为独立入口**是正确的，已加测试锁定。

---

### 8.20.60 第 89–91 轮：`e(depvar)` 的跨族缺失与兜底机制

#### 第 89 轮：`lfit` 表头暴露了 `e(depvar)` 缺失

`lfit.ado:116-120` 打印两行：

```stata
di _n in smcl as txt "Goodness-of-fit test after " ...
di as smcl "Variable: " as res "`y'"
```

我只实现了第一行。补第二行时**取不到响应名**，才发现 `logit` 结果
**既无 `response_name` 属性，也无 `e(depvar)` 宏**。

而 `logit.sthlp:264` 明确列出「**e(depvar)** — name of dependent variable」。

顺带修正模型标签（`lfit.ado:117-119`）：原实现硬编码 `"logistic"`，
`probit` 也印成 logistic。

#### 第 90 轮：这是跨族的普遍缺失

审计结果：

| 估计器 | 修复前 |
|---|---|
| `Tobit`/`Truncreg`/`Intreg` | ❌ |
| `OrderedLogit` | ❌ |
| `QuantileRegression`/`RobustRegression` | ❌ |
| `FEOLS`/`PanelOLS` | ✅（第 81 轮）|
| `Logit`/`Probit`/`Poisson` | ❌（第 89 轮修）|

**六份帮助文件（`tobit`/`truncreg`/`intreg`/`ologit`/`qreg`/`rreg`）
全部文档化了这个宏。**

**关键发现**：`Tobit` 的 MRO 是 `(Tobit, object)`——**不继承 `_DiscreteBase`**。
**同类估计器未必共享基类**，所以上层的修复不会自动传播。

#### 两层修复

**① 在 `StataResults.e` 加兜底**：

```python
if "depvar" not in self._e:
    response = getattr(self, "response_name", None)
    if response:
        self._e["depvar"] = str(response)
```

**放在 `e` 属性而非构造时**：`response_name` 是在结果构造**之后**才赋值的
（各 `fit` 末尾），构造时兜底会取到 `None`。

**② 各估计器只需记录 `response_name`**——一处一行。

#### 第 91 轮：逐个补齐

补齐过程中发现**两个估计器绕过共用路径**：

| 估计器 | 绕过的原因 |
|---|---|
| `_Ordered`（`ologit`/`oprobit`）| 重写了 `fit`（`discrete.py:1725`），不走 `_DiscreteBase.fit` |
| `QuantileRegression`/`RobustRegression` | 各自构造 `StataResults` |

最终 6 个全部覆盖：

```
Tobit    depvar='y'
Truncreg depvar='w'
Intreg   depvar='lo hi'    ← 区间由两列给出，Stata 记录两列
OrderedLogit depvar='rep'
QuantileRegression depvar='y'
RobustRegression depvar='y'
```

#### 一条可复用的教训

**「基类修复」不等于「全族修复」**：需要确认每个估计器是否真的走共用路径。
`_Ordered` 重写 `fit`、`Tobit` 不继承基类，都是反例。

兜底机制放在 `e` 属性（消费端）而非各处 `fit`（生产端），
正是为了让「忘记设置」不再成为缺陷。

---

### 8.20.61 第 92–94 轮：`e()` 的宏与标量一致性

#### 第 92 轮：`e(cmd)`/`e(properties)` 全族缺失

审计 `e()` 的常用宏，发现除 `depvar` 外，`regress.sthlp:435-441` 文档化的
其余宏**在全部估计器上都不存在**：

| 宏 | 处理 |
|---|---|
| `cmd` | ✅ **补齐**（`model_name` 就是 Stata 命令名）|
| `properties` | ✅ **补齐**（文档写作**恒定值** `b V`）|
| `title` | ❌ **拒绝** |
| `vcetype` | ❌ **拒绝** |
| `cmdline` | ❌ **拒绝** |

拒绝理由：`title`/`vcetype` 是**逐命令的英文字符串**（"Linear regression"、
"Robust"），`cmdline` 记录用户**实际键入**的命令。**编造一个英文标题或命令行，
等于在调用者信任的位置放一个看起来合理的假值。**

顺带补上 `IV2SLS` 的 `e(depvar)`（上一轮遗漏）。

#### 第 93 轮：`e(df_r)` 在 ML 模型上是编造的

跨估计器一致性检查暴露：

| 模型 | `N − df_r` | `rank` |
|---|---|---|
| `OLS` | 2 | 2 ✅ |
| `Logit` | **0** | 2 ❌ |
| `Poisson` | **0** | 2 ❌ |

查证帮助文件：

| 文件 | 有 `e(df_r)` |
|---|---|
| `logit`/`poisson`/`nbreg`/`ologit` | ❌（只有 `e(df_m)`）|
| `tobit` | ✅ |

**原因**：ML 模型的检验是 **z 检验**，没有残差自由度。

而 `StataResults.__init__` **无条件注入** `df_r`，等于给 logit 结果塞了一个
Stata 根本没有的 `200.0`。

修正：按 `dist` 判断——`dist != "z"` 才注入。

**重要边界**：只移除 `e()` 里的键，**不动 `df_resid` 属性**。
后者是 statsmodels 面向的，t/F 分布需要它。
「Stata 的 `e()` 没有」不等于「内部不该有」。

#### 第 94 轮：恒等式复核（一次误报）

用 `N − df_m − df_r = 1` 检查，`Tobit` 报 `0`。

查 `tobit.py:34-40` 的注释——公式抄自 `tobit.ado`：

```stata
df_m = e(rank) - 1 - hascons
df_r = e(N) - e(df_m)
```

**`df_m + df_r = N` 对 tobit 是正确的**（`df_m` 既不含截距也不含 `lnsigma`）。
**是我的恒等式不适用，不是实现有误。**

其余恒等式精确成立：

| 恒等式 | 结果 |
|---|---|
| `r2 = mss/(mss+rss)` | ✅ 精确 |
| `ll = −N/2·(ln2π + ln(rss/N) + 1)` | ✅ 精确 |

---

### 8.20.62 第 95–97 轮：恒等式验证层

第 92–94 轮的两个缺陷都只能靠**比对 `e()` 内部数值**发现，存在性检查永远不会暴露。
本轮把这一方法固化为独立的测试层。

#### 第 95 轮：`e()` 标量恒等式

| 恒等式 | 适用 | 结果 |
|---|---|---|
| `r2 = mss/(mss+rss)` | OLS | ✅ 精确 |
| `ll = −N/2·(ln2π + ln(rss/N) + 1)` | OLS | ✅ 精确 |
| `chi2 = 2·(ll − ll_0)` | Logit/Poisson | ✅ 精确 |
| `ll_0` = 常数模型对数似然 | Logit | ✅ 与直接拟合一致 |
| `N − df_r = rank` | 含 `df_r` 且非 tobit/面板 | ✅ |

#### 第 96 轮：恒等式的**适用域**

三种模型不满足上述恒等式，**但都不是实现错误**：

| 模型 | 现象 | 真相 |
|---|---|---|
| `Tobit` | `N−df_r=1` vs `rank=3` | `tobit.ado` 定义 `df_r = N − df_m`，`rank` 含 `lnsigma` |
| `PanelOLS` | `N−df_r=41` vs `rank=2` | `df_r` **吸收组效应**（60 组 + 1）|
| 面板的 `r2` | `0.0088` vs `mss/tss=0.2244` | `r2`=**组内**，`r2_o`=**整体** |

**共同教训：「恒等式不成立」的第一反应应当是「这个恒等式适用吗」，而不是「实现错了」。**

三次都是我把一个模型的约定套到了另一个模型上。

已把每种约定**显式写进测试**：

```python
assert e["df_m"] + e["df_r"] == e["N"]      # tobit
assert e["N"] - e["df_r"] == groups + 1     # 面板 fe
assert e["r2_o"] == e["mss"] / e["tss"]     # 面板的整体 R²
```

面板的约定**从实现记录而非对照参考断言**（`xtreg.sthlp:1023` 只写
"residual degrees of freedom"，未展开），已在 docstring 标注。

#### 第 97 轮：后估计量与 `e()` 的一致性

| 检查 | 结果 |
|---|---|
| `estat ic` 的 AIC `= −2·ll + 2·rank` | ✅ 精确 |
| `estat ic` 的 BIC `= −2·ll + ln(N)·rank` | ✅ 精确 |
| `estat_ic.aic − e(aic) = N(1+ln2π)` | ✅ 精确（**已知且已记录的差异**）|
| `estat summarize` 的均值 = 原始数据均值 | ✅ 精确 |
| `estat summarize` 的标准差 = `ddof=1` 样本标准差 | ✅ 精确 |
| `tolerance = 1/VIF` | ✅ 精确 |
| `mean_vif` = VIF 列均值 | ✅ 精确 |
| 单回归元 VIF 恰为 1 | ✅ |
| 共线回归元 VIF 显著大于 1 | ✅ |
| `estat vce` 的对角元 = `bse²` | ✅ 精确 |

**`e(aic)` 与 `estat ic` 的 AIC 本就不同**（差 `N(1+log2π)`），
这不是缺陷而是 Stata 的两个不同量——第 12 轮已记录，本轮把**差值本身**用测试锁定，
使其成为「有记录的约定」而非「可疑的不一致」。

---

### 8.20.63 第 98–100 轮：面板 `e()` 的恒等式与互斥分支

#### 第 98 轮：面板恒等式全部成立

| 恒等式 | 结果 |
|---|---|
| `rho = σ_u²/(σ_u²+σ_e²)` | ✅ fe 与 re 均精确 |
| `theta = 1 − σ_e/√(T·σ_u²+σ_e²)` | ✅ 精确（`xtreg_re.ado:265-266`）|
| `sigma = √(σ_u²+σ_e²)` | ✅ 精确（`xtreg.sthlp:902`）|
| `r2 == r2_w` | ✅ |
| `r2_o ≠ r2` | ✅（否则其一标错）|
| `r2_b == 1`（单回归元）| ✅ 饱和拟合的必然结果 |

**`r2_b = 1` 单回归元即可精确拟合组均值，记录以防止将来被当 bug「修掉」。**

#### 第 99 轮：`theta` 与 `thta_*` 是**互斥**的

`xtreg_re.ado:368-379`：

```stata
if e(Tcon) {
    est scalar theta = `thetav'          // 组大小恒定 → 单一 theta
}
else quietly {
    summ `thetav', d
    est scalar thta_min/p5/p50/p95/max   // 组大小可变 → theta 的分布
}
```

| 面板 | `Tcon` | 修复前 | 修复后 |
|---|---|---|---|
| 平衡 | 1 | `theta` ✓ | 不变 |
| **不平衡** | 0 | `theta`（**不该有**）、无 `thta_*` | `theta` 移除、`thta_*` 五项 |

**为何隐蔽**：不平衡面板上给出的 `theta` 由**平均组大小**算得——数值合理、在 [0,1] 内、无报错。
但**它不对应任何实体**：每个实体的 `T_i` 不同，`theta` 本就该是一组值。
**报告单一均值等于发明了一个不存在的人群参数。**

`thta_*` 与逐实体重算精确一致（1e-12），并验证了排序性。

#### 第 100 轮：继承导致的 `theta` 泄漏

`xtreg_be.ado` **完全没有 `theta`**（grep 计数为 0），但 `BetweenOLS` 有。

根因：`BetweenOLS(RandomEffects)` **继承**了 `_stores_theta = True`。

修正：显式设 `_stores_theta = False`，并加测试锁定 `fe`/`be`/`fd` 都无 `theta`。

**这与第 91 轮 `_Ordered` 重写 `fit` 同类——继承会把父类的能力带到一个不该有的子类上。**

#### 一条关于「依据位置」的教训

`xtreg.sthlp` 把 `fe` 与 `re` 的键**混排在同一区块**，我最初把 `thta_*` 误当作 `fe` 的键。

**真正的位置在 `xtreg_re.ado`，而不是帮助文件的分节。**

> 帮助文件按「主题」组织，ado 按「执行」组织——**后者才是可靠依据**。

---

### 8.20.64 第 101–103 轮：面板继承泄漏与「假朋友」识别

#### 第 101 轮：`fd` 的 `theta` 缺失是**偶然**的

`FirstDifferenceOLS` 没有重设任何标志，全部继承 `RandomEffects`：

```python
fd._stores_theta = True       # 继承了 re
```

实测 `fd` 的 `e()` **没有 `theta`**，原因：

```python
if np.isfinite(sigma_u) and np.isfinite(sigma_e):   # ← 前置条件
    ...
    if self._stores_theta and t_bar > 0:            # ← 未执行
        out["theta"] = theta
```

`fd` 恰好没有 `sigma_u`，整个块被跳过。**这不是设计，是巧合**——
若将来给 `fd` 加上 `sigma_u`，`theta` 会**静默出现**。

修法：显式声明四个估计器的全部标志，不依赖继承。

#### 第 102 轮：`be` 的整个 sigma 家族都是编造的

`xtreg_be.ado:148-166`：

```stata
corr `XB' `depvar'
est scalar r2_o = r(rho)^2          ← r(rho) 来自 corr 命令
corr `XB' `T'
est scalar r2_w = r(rho)^2
```

**`corr` 是命令，`r(rho)` 是它的返回值**——用来算 `r2_o`/`r2_w`，**都不是 `e()` 键**。

`xtreg_be.ado` 全文只设置 `df_*`、`g_*`、`p`、`r`、`rmse`、`r2_*`、`Tbar`/`Tcon`、`N_g`。

**`be` 是对实体均值做 OLS，没有面板/误差分解可报。**

修正后：

| 估计器 | sigma 家族 |
|---|---|
| `fe` / `re` | `sigma_u`/`sigma_e`/`sigma`/`rho`/`corr`/`ui` ✅ |
| `be` / `fd` | **无** ✅（修复前 `be` 有全部 6 个）|

#### 一份「假朋友」清单

| 文件中的字符串 | 真实含义 |
|---|---|
| `corr` | **命令名**（计算相关系数）|
| `r(rho)` | `corr` 命令的返回值 |
| `e(rho)` | 存在，但由**其他分支**设置 |

**在 ado 里 grep 一个词，命中的可能是命令、选项、甚至注释——必须看上下文。**

#### 第 103 轮：保留 `be` 的 `r2_b`，记录 `fd` 的缺口

**`r2_b` 有依据**（`xtreg_be.ado:132`：`est scalar r2_b = e(r2)`）。
移除继承的 sigma 家族时**不能连带移除它**——已加测试锁定。

**`fd` 的 `r2_w`/`r2_o` 是缺口，但本轮不实现**：

- `xtreg.ado` 全文**没有** `r2_w`/`r2_o`；
- 它们由**各方法子程序** `xtreg_fe`/`xtreg_re`/`xtreg_be` 设置；
- **`xtreg, fd` 没有自己的子程序**，故其 `r2_w`/`r2_o` 的定义**无法从 ado 读到**；
- `xtreg.sthlp` 把它们列在**混合各变体**的通用区块里。

**凭猜测从混合回归推一个公式，会得到一个没有出处的数。** 已在测试中如实记录为缺口。

---

### 8.20.65 第 104–106 轮：`ereturn local` 的「清空」与 `k_aux` 覆盖

#### 第 104 轮：`qreg` 的缺失是「清空宏」而非「未实现」

`qreg.ado` 里两种 `ereturn local` 含义**相反**：

| 形式 | 含义 |
|---|---|
| `ereturn local name "value"` | 设置宏 |
| `ereturn local name`（**无值**）| **清空**宏 |

`qreg.ado:175-194` 的 `fvops` 分支有一整块**无赋值**：

```stata
ereturn local r2
ereturn local mss
ereturn local rss
ereturn local rmse
ereturn local r2_a
ereturn local ll
ereturn local ll_0
ereturn local title
ereturn local model
```

**把这些当成「文档化的存储结果」，正是编造键的来源。**

条件键也已查证（`qreg.ado:266-273`）：`vcetype` 仅当 `vce != iid`，`kernel`/`kbwidth`
仅当 `method(kernel)`。默认下本就不存在。

**结论：`qreg` 的 `e()` 表面基本正确**，缺的都是清空宏或条件键。

#### 第 105–106 轮：`k_aux` 的覆盖率

| 估计器 | `k_aux` 在 ado | 修复前 |
|---|---|---|
| `nbreg` | 有 | ✅ |
| `ologit` | 有 | ✅ |
| `tobit` | `tobit.ado:153` | ❌ **缺** |
| `truncreg` | `truncreg.ado:252` | ❌ **缺** |
| `intreg` | `intreg.ado:418` | ❌ **缺** |
| `logit`/`poisson` | **无** | ✅ 正确（本就不应有）|

三个缺口都因**独立模块**（`tobit.py` 的 MRO 是 `(Tobit, object)`；
`truncreg`/`intreg` 在 `limited.py` 有自己的 e() 构造）。

#### 三个模型、两种 `rank` 组成

我最初用统一公式 `rank = df_m + 1 + k_aux`，**两次断言失败**，据此查清：

| 模型 | `rank` 组成 | 公式 |
|---|---|---|
| `tobit` | 斜率 + 截距 + `lnsigma` | `df_m + 1 + k_aux` |
| `nbreg` | 斜率 + 截距 + `lnalpha` | `df_m + 1 + k_aux` |
| **`ologit`** | 斜率 + **cut points**（**无截距**）| `df_m + k_aux` |

另一次失败：我以为 3 个类别有 3 个 cut point，**实际是 `K−1 = 2`**。

**两次都是我的模型知识不准，代码是对的。**

#### 本轮的「假朋友」清单（累积）

| 轮次 | 误读对象 | 真相 |
|---|---|---|
| 102 | `xtreg_be.ado` 的 `corr` | **命令名** |
| 103 | `xtreg.sthlp` 的混合区块 | 混合各变体 |
| 104 | `ereturn local` 无值形式 | **清空**，非设置 |

---

### 8.20.66 第 107–109 轮：静态审计脚本与键名不一致

#### 第 107 轮：写了一个「赋值 vs 清空」的审计脚本

用正则机械区分 ado 中的三种形式：

| 形式 | 判定 |
|---|---|
| `ereturn local NAME "value"` | **SET** |
| `ereturn local NAME`（无值）| **CLEAR** |
| `ereturn scalar NAME = expr` | **SET** |

结果分两向：

- **正向**：8 个估计器**全部没有**实现「只被清空」的宏（0 问题）；
- **反向**：发现 4 处数值缺口。

#### 第 107 轮：`tobit` 的删失计数用了自创键名

| 我们的实现 | Stata（`tobit.sthlp:254-256`）|
|---|---|
| `n_uncensored` | **`N_unc`** |
| `n_left` | **`N_lc`** |
| `n_right` | **`N_rc`** |

**大写 `N` + 缩写 `unc`/`lc`/`rc`。** 原名字描述性很强，但**没有任何 Stata 消费者会去找它们**——
按文档读 `e(N_unc)` 会得到 `KeyError`。已补正确名，旧名保留为别名。

验证：`N_unc + N_lc + N_rc == N`，且 `N_rc` 与右删失计数精确一致。

#### 第 108 轮：`truncreg.N_bf` 与 `intreg` 的区间计数

| 键 | 依据 |
|---|---|
| `truncreg.e(N_bf)` | `truncreg.sthlp:247`、`ado:232` |
| `intreg.e(N_unc/N_lc/N_rc/N_int)` | `intreg.sthlp:242-245` |
| `intreg.e(se_sigma)` | `intreg.ado:416-417` |

**`N_bf` 的陷阱**：它是**截断前**的观测数，必须在 `from_formula` 裁剪数据**之前**记录
`y.size`，晚一步就永远拿不到原值。

**`se_sigma` 的公式**：

```stata
sigma    = exp([lnsigma]_cons)
se_sigma = exp([lnsigma]_cons) * [lnsigma]_se[_cons]     // = σ · se(lnσ)
```

**取 σ 自己的对角元素会得到完全不同的数**，必须取 `/lnsigma` 位置的标准误。
`truncreg` 没有 `/lnsigma` 参数，故 `se_sigma` 不该存在（已加反向测试）。

#### 第 109 轮：`qreg.sum_w`

`qreg.ado:279` 无条件设置 `sum_w`，`qreg.sthlp:507` 文档为「sum of weights」。
无权重时等于 `N`。`rreg` **没有**该键，已加反向测试防止复制。

`kbwidth` 是 `method(kernel)` 的条件键（`qreg.ado:270-272`），默认下缺失**正确**。

#### 一个反复出现的模式

本轮四个缺口全部是**同一个性质**：**我们有这个数，但键名与 Stata 不同**（或根本没用 Stata 的名字）。

这类错误比「缺功能」更隐蔽——功能在、数算对了，**但调用方按文档取不到**。

---

### 8.20.67 第 110–112 轮：反向审计（多给了什么）

前几轮查的是「少给了什么」。本轮反过来查：**我们报告了哪些 Stata 并不存在的键**。

#### 第 110 轮：`rreg` 的 11 个编造键（正确移除）

`rreg.ado` 全文**只有一个 `ereturn` 语句**（第 219 行的 `cmdline`），其余全部经
`ereturn repost` 从底层回归继承。

而我们暴露了 11 个：

| 键 | 在 `rreg.ado` 中的真实身份 |
|---|---|
| `scale` / `aa` / `lambda` | **`tempname` 局部标量**（第 61 行）|
| `tune` / `tolerance` | 选项值 |
| `iterations` / `dropped` / `mse` / `huber_c` / `biweight_c` / `N_weighted` | 内部实现细节 |

修法：从 `e()` 移除，值保留为 `results.rreg_detail` **属性**（诊断有用，但不得冒充 Stata 标量）。

#### 第 111 轮：一次**错误移除**，已自行纠正

我依据「不在 `logit.sthlp` 中」移除了 `consonly`/`k_autoCns`/`noconstant`。

**这是错的。** 查证 fixtures：

```
logit_sim.json   consonly=0.0  k_autoCns=0.0  noconstant=0.0  k_eq_model=1.0  rc=0.0
nbreg.json       同样 5 个键
ologit.json      同样 5 个键
```

**fixtures 是从真实 Stata 运行捕获的，里面确实有这些键。**

> **帮助文件不是唯一权威。fixtures 的证据等级更高。**

这些键来自**共享的估计机制**，不写在各命令自己的帮助文件里。

**已全部恢复**，并在代码注释中写明依据。

#### 区分「真编造」与「真宏」

| 键 | fixture 佐证 | ado `ereturn` | 判定 |
|---|---|---|---|
| `rreg` 的 `scale`/`aa`/`lambda` 等 | ❌ | ❌（全文仅 1 个）| **编造**，移除 |
| `consonly`/`k_autoCns`/`noconstant` | ✅ 有真实值 | 共享机制 | **真宏**，保留 |

**关键判据是「有没有实跑证据」，不是「有没有写进帮助文件」。**

#### 第 112 轮：`tobit` 的 `lower`/`upper` → `llopt`/`ulopt`

`tobit.ado:131-132`：

```stata
ereturn local llopt `ll'
ereturn local ulopt `ul'
```

**Stata 用选项名 `llopt`/`ulopt`，我们用描述性的 `lower`/`upper`** ——
按 ado 读 `e(llopt)` 会得到 `KeyError`。已改。

#### 第 112 轮：`N_cds`/`N_cdf` 经查证**实现正确**

fixture 中两者均为 0.0，但那是数据的性质（协变量模式内都有变异），不是硬编码。

| 数据 | `N_cds` | `N_cdf` |
|---|---|---|
| 离散回归元（模式内有变异）| 0.0 | 0.0（**与 fixture 一致**）|
| 连续回归元（每个观测自成一模式）| 307 | 293（和 = N）|

**`test_the_N_cds_block_counts_every_singleton_pattern` 用来证明这不是硬编码的 0。**

---

### 8.20.68 第 113 轮：剩余编造键的清理与连带修复

#### 按证据优先级逐个查证

第 112 轮的教训是「先查 fixture」。本轮照做：

| 键 | fixture | ado | sthlp | 判定 |
|---|---|---|---|---|
| `N_cds`/`N_cdf` | ✅ 有 | — | `logit.sthlp:243-244` | **正确**，不动 |
| `n_truncated` | ❌ | ❌ | ❌ | **编造**，移除 |
| `lower`/`upper`（truncreg）| ❌ | ❌ | 有 `llopt`/`ulopt` | **改名** |
| `lnsigma`（intreg）| ❌ | ❌ | ❌ | **编造**，移除 |
| `n_left`/`n_right`/`n_interval`/`n_uncensored` | ❌ | ❌ | 有 `N_lc`/`N_rc`/`N_int`/`N_unc` | **改名** |

#### `truncreg` 的 `_extra_scalars`

原返回 4 个自创键，改为只返回 Stata 的：

```python
return {"llopt": self._lower, "ulopt": self._upper}
```

截断计数保留为内部属性 `_n_truncated`（诊断有用），不再冒充 `e()` 标量。

#### `intreg` 的 `_extra_scalars`

原返回 `lnsigma` + 4 个小写计数，**全部无依据**。改为返回 `{}` ——
正确名称已由 `_interval_counts` 提供（`N_unc`/`N_lc`/`N_rc`/`N_int`），
`se_sigma` 由 delta 方法提供。

#### 连带发现：`xttobit` 也用了旧名

`panel_nonlinear.py:1532-1536` 同样用 `lower`/`upper`/`n_uncensored`/`n_left`/`n_right`。
`xttobit.sthlp:241-243`、`:274-275` 确认使用 `N_unc`/`N_lc`/`N_rc`/`llopt`/`ulopt`。已改。

#### 一次关于「类别互斥」的自我纠正

`intreg` 的混合删失测试我最初按 `isneginf(lower)` 期望 `N_lc=80`，实现给出 68。

原因：**12 行同时满足 `lower=-inf` 且 `upper=+inf`**（索引 %5 与 %7 同时为 0），
这些行**不含任何信息**，既非左删失也非右删失。

`_interval_counts` 的互斥分类是**正确的**，我的期望值算错了。测试已改为
按互斥类别计算，并显式断言「四类之和 < n」（重叠行不计入任何一类）。

#### 改名必须全局搜索消费者

`tobit`/`truncreg` 的 `lower`→`llopt` 改名，导致 `predict.py:1127` 仍读旧键，
`predict, e` 与 `predict, ystar` **静默退化为「无限制」**，4 个测试失败。

修法：优先读 `llopt`，保留旧名作回落。并更新 6 处测试引用。

> **这是第 74 轮（删除 `reghdfe` 注册）之后，同一类错误的第二次：**
> **生产者改名/删能力，必须搜索全部消费者。**

---

### 8.20.69 第 117–119 轮：`F_f` 的 R² 陷阱与检验行渲染

#### 第 117 轮：`F_f` 用了错误的 R²（真缺陷）

**表象**：随机面板数据上 `F_f = -0.00891`，**负值**。

F 统计量是均方之比，**不可能为负**——这个判据不需要任何 fixture。

**根因**（`xtreg_fe.ado:505-506`），Stata 作者自己写了警告注释：

```stata
/* not the same as e(r2) returned from _regress */
scalar `r2' = 1 - e(rss)/e(tss)
ereturn scalar F_f = ((`r2'-`r2c')/e(df_a))/((1-`r2')/e(df_r))
```

**公式里的 `r2` 是 `1 − rss/tss`，不是 `e(r2)`。**

| 量 | 值（`xtreg_fe.json`）|
|---|---|
| `1 − rss/tss` | **0.9531797** |
| `e(r2)` | 0.9299367 |

代入结果：

| 用哪个 | `F_f` |
|---|---|
| `e(r2)` | 17.075 |
| `1 − rss/tss` | **28.085188842551904** ← 与 fixture 逐位一致 |

**独立确认**：`f.sf(28.085, 29, 148) = 1.8906548563177605e-46`，
与 fixture 记录的 `p_f` 精确相同 —— 这同时证明自由度取法（`df_a=29`、`df_r=148`）正确。

#### 第 118 轮：检验行的措辞必须用原文

我最初写 `F test of u_i=0`，依据是 `xtreg.sthlp:1098` 的**描述**。

`whatsnew13.sthlp:1165-1168` 给出了**原文**：

> 「...this statistic reported **below the results table as "F test that all u_i=0"**」

**`that all` 是输出文本的一部分。** 已改，并加了双向断言：

```python
assert "F test that all u_i=0" in rendered
assert "F test of u_i=0" not in rendered   # 转述不得复现
```

#### 一条关于「找措辞」的教训

| 来源 | 提供的是 |
|---|---|
| `xtreg_fe.ado` | 公式 |
| `xtreg.sthlp:1098` | **描述**（"F statistic for test of u_i=0"）|
| `whatsnew13.sthlp:1168` | **原文**（"F test that all u_i=0"）|

**拿描述当原文，就会漏掉 `that all` 这样的词。**

#### 第 119 轮：值域压力测试

24 个不同 DGP（6 seed × 4 实体效应强度）下 `F_f` **全部非负**，
确认修复不是靠单一数据集凑对的。

---

### 8.20.70 第 120–122 轮：后估计量的显示层

#### 第 120 轮：`estat ic` 的 Model 列是硬编码的 `.`

修前：

```
       Model |          N     ll(null) ...
           . |        200    -269.7290 ...
```

`'.'` 读起来是「信息不可得」，但**命令名一直已知**（`e(cmd)`）。

修后：`regress` / `logit` / `poisson`。

**能力边界如实标注**：`estat ic` 是 **builtin，无 `e/estat_ic.ado`**，所以
单元格的**精确对齐宽度无法从 Stata 文件验证**。能确定的是该列携带命令名。

保留退化路径：结果完全无命令名时仍显示 `.`——**渲染成空白比显示「不可得」更差**。

#### 第 121 轮：`estat summarize` 遇分类变量即崩溃

```
ValueError: could not convert string to float: 'b'
```

模型含**一个**分类变量，整张表消失，连带丢失调用者关心的数值变量。

修后：

| 列 | 分类变量行 |
|---|---|
| `Obs` | ✅ 200（可数）|
| `Mean`/`Std. dev.`/`Min`/`Max` | `.`（Stata 缺失标记）|

依据 `estat_summarize.sthlp:38`（*factor-variable labeling*）——Stata 报告
**底层变量**而非虚拟展开，且无法对字符串求均值。

顺带把 `nan` 改为 **`.`**——**Stata 从不打印 `nan`**。

#### 第 122 轮：分类变量下行序与常数标签

**发现的内部不一致**：`stata_param_names` 对因子项抛异常，
渲染回退后用 `Intercept`——于是**同一个命令**在数值模型显示 `_cons`、
在因子模型显示 `Intercept`。

修法分两步：

1. **降级重命名**：异常时仍把常数项改 `_cons`，其余保持原名；
2. **降级排序**：`_stata_order` 同样会拒绝，此时用 `_constant_last_order`
   把常数项移到最后。

#### 三次错位：一次真实的教训

修复过程中我犯了**同一个错误的三种变体**：

| 尝试 | 症状 |
|---|---|
| 只在异常分支重命名 | 名称对，行序错 |
| 用 `order` 再索引一次 | **数值整体错位**（每行显示邻居的系数）|
| 白名单式斜杠 | `nl` 的 `/b0` 丢失 |

最终正确的规则是：

1. `stata_param_names` **已按 Stata 顺序返回**，**不再二次置换**；
2. 只有**回退路径**（异常时构造的名称）才需要按 `order` 置换；
3. 斜杠不靠白名单，而是看**原始拼写**里有没有 `/`。

> **「重排」与「重命名」是两条独立的轴，混在一起就会错位。**
> 而且错位的表格结构完全正常、行数正确——**只有逐个核对标签与数值才能发现**。

---

### 8.20.71 第 123–125 轮：参数命名转换的重复实现

#### 症状：三张表两种命名

| 命令 | 修复前 |
|---|---|
| 主表 | `x1#x2` ✅ |
| `estat vce` | `x1:x2` ❌（第 123 轮修）|
| `estat vif` | `x1:x2` ❌（第 124 轮修）|

同一次分析里，**同一个参数**有两种拼写。

#### 合并后的**行序**问题

`estat vce` 不只是分隔符错，**行序**也不同：

| | 主表 | `estat vce`（修复前）|
|---|---|---|
| 顺序 | `x1`, `x2`, `x1#x2`, `_cons` | `_cons`, `x1`, `x2`, `x1:x2` |

**为何危险**：协方差矩阵是**方阵**，无论怎么排都「结构正常」。
而 `names[i]` 在两处指向**不同参数**——交叉引用会静默取错。

修复：应用 `_stata_order` 重排名称**和矩阵**（`cov[np.ix_(order, order)]`）。

#### 根因：同一个转换逻辑有三份实现

| 位置 | 处理内容 |
|---|---|
| `results/base.py:378` | `:` → `#`（`e(b)` 用）|
| `postestimation/vce.py` | 常数项 + 斜杠 + 冒号 |
| **`postestimation/vif.py`** | **只有常数项**（漏了冒号）|

**每份实现各自演化，于是漂移。**

修法：抽出 `postestimation/names.py`，`vce`/`vif` 改为**委托**，
保留 `_display_name` 作为薄壳以免破坏既有导入。

#### 一个新模块明确的两条边界

`stata_display_name` 与 `stata_param_names` **不是同一个函数**：

| 函数 | 用途 | 斜杠 |
|---|---|---|
| `stata_param_names` | 生成 `e(b)` **存储**的名字 | **去掉**（`ologit.json` 是 `cut1`）|
| `stata_display_name` | 生成**表格打印**的名字 | 也去掉（矩阵命令不用斜杠）|

**系数表**另行给辅助参数补回斜杠——这是**第三个独立决定**，
混进前两者之一就会出错（第 122 轮 `nl` 的 `/b0` 消失即此）。

#### 第四次「同一概念多份实现」

| 轮次 | 重复实现 |
|---|---|
| 91 | `e(depvar)` 各估计器自行设置 |
| 102 | `corr` 被误读为 e() 键 |
| 123 | `estat vce` 的行序 |
| **124–125** | **参数命名转换** |

**重复实现不会自己保持一致**；只有抽出单一实现才能真正消除漂移。

---

### 8.20.72 第 126–128 轮：计数家族的 `predict` 类型

#### 第 126 轮：`pr(n)` —— 一个文档化但缺失的类型

`poisson_postestimation.sthlp:105, 147-148`：

```stata
pr(n)      probability Pr(y = n)
pr(a,b)    probability Pr(a < y < b)
```

`pr` **需要参数**，我们完全没有提供。调用 `predict(kind="pr")` 只报
「poisson 支持 xb/stdp/n/residuals」——**看起来像命令没有这个功能，
实际是调用不完整**。

#### 两个错误

**① 只按拼写分派**：`pr` 在 logit 是**概率本身**（无参），在 poisson 是
**`Pr(y=n)`**（需参）。第一版按拼写判断，导致 **logit/probit 的 `pr` 全部回归失败**。
改为按**家族**判断。

**② `nbreg` 用了泊松分布**：`nbreg` 是**负二项**。

```
Stata:  Var(y) = mu + alpha*mu²,  alpha = exp(lnalpha)
scipy:  nbinom.pmf(k, size=1/alpha, p=1/(1+alpha*mu))
```

验证：

| 对照 | 结果 |
|---|---|
| `nbreg pr(2)` vs 泊松 | **False** |
| `nbreg pr(2)` vs 负二项 | **True** |

**用泊松会得到看似合理的数，但完全忽略离散参数**——误差恰是模型要捕捉的过度离散。

#### 第 127 轮：`pr(a,b)` 的端点归属

`poisson_postestimation.sthlp:106` 原文：

```stata
{opt pr(a,b)}   probability Pr(a {ul:<} y {ul:<} b)
```

**两侧都是 `{ul:<}`**，即**严格开区间**。

| 类型 | 形式 |
|---|---|
| 点质量 | `pmf(k)` |
| 区间 | **`cdf(b-1) − cdf(a)`**（不是 `cdf(b) − cdf(a)`）|

三条自洽验证：

- `pr(k) == pr(k-1, k+1)` ✅
- `pr(-1, 1000) == 1` ✅
- **`pr(2,3) == 0`** ✅ —— 若含右端点会返回 `P(y=3)`

**端点归属的详细说明在 `pr_options.sthlp` 中不存在**，判据是帮助文本的
`{ul:<}` 标记，代码注释已如实标注。

#### 第 128 轮：`ir` 与 `score`

| 类型 | 定义 | 实现 |
|---|---|---|
| `ir` | `exp(xb)`（`sthlp:141-142`）| **等价于 `n`**（无 offset 时）|
| `score` | `d ll / d eta`（`sthlp:163-164`）| Poisson 为 **`y − mu`** |

`score` 的均值在 MLE 处为 0——这是错误公式无法满足的性质。

**`nbreg` 的 `score` 不注册**：其 score 含离散因子，未实现。
**注册一个会抛异常的类型，正是注册表测试要抓的缺陷。**

#### 一次分派顺序的教训

`ir` 和 `score` 首次实现后仍报「registered but has no implementation」——
因为分支写在了 `normalized in ("pr","n","ir")` 块**内部**，而 `score` 进不去。

**注册表的一致性检查替我发现了两处错误的分派位置**——它比手工测试更可靠。

---

### 8.20.73 第 129–131 轮：`scores` 的形状不匹配与命名审计

#### 第 129 轮：`nbreg` 的 `scores` 是「形状不匹配」，不是「缺功能」

`nbreg_postestimation.sthlp:172-181`：

> `scores` 计算**两个**变量：
> 1. 对**回归方程**的对数似然导数
> 2. 对**离散方程**的对数似然导数

**我们的 `predict` 每次返回一个数组。**

只返回第一个 → **静默丢掉离散分数**；用泊松的 `y−mu` → **完全忽略离散**。

故不注册，但拒绝时**说明形状原因**：

```
predict, scores after nbreg produces TWO variables -- the derivative of the log
likelihood with respect to the regression equation AND with respect to the
dispersion equation -- while this method returns one array.
```

#### 第 130 轮：这是系统性问题，不止 `nbreg` 一处

| 命令 | 帮助文件拼写 | 变量数 |
|---|---|---|
| `poisson` | **`score`**（单数）| 1（单方程）|
| `logit` | `score`（单数）| 1 |
| `nbreg` | `scores` | 2（回归 + 离散）|
| `ologit` | `scores` | **= 结局数** |
| `tobit`/`truncreg`/`intreg` | `scores` | 2+（回归 + 辅助参数）|

`ologit_postestimation` 的原文最能说明：

> "The number of score variables created will equal the number of outcomes"

**修复前**：`predict: 'tobit' supports (...); got 'scores'` ——
读起来像「本包没有这个功能」。

**修复后**：明确说明「产生多个变量，本方法返回一个数组」。

**通用消息与真实原因不符，就是误导**，即使拒绝本身是对的。

#### 拼写不对称是 Stata 自身的特点

`poisson` 用 `score`，其余用 `scores`。已加测试**直接比对帮助文件**，
防止将来「统一命名」让其中一个偏离。

#### 三次修改同一逻辑的教训

我改了三处才让说明生效：

| 尝试 | 失败原因 |
|---|---|
| 在 `declared` 分支加 | `nbreg` 的 `declared` 是 `None` |
| 在 `else` 分支加 | 前面的 `dispatch` 检查先拦截 |
| **抽出 `_unsupported_reason` 供两处共用** | ✅ |

**同一逻辑散在两处判断分支里，必须两处都改**——同第 125 轮。

#### 第 131 轮：`alpha`/`lnalpha`/`stdplna` 属 `gnbreg`，不属 `nbreg`

`nbreg_postestimation.sthlp`：

> "alpha, lnalpha, and stdplna are relevant after **gnbreg** estimation only"

普通 `nbreg` 的 `alpha` 是**常数**，逐观测的 `alpha` 列会是一列重复值——
提供它会暗示离散随观测变化，而模型说它不变。

本包未实现 `gnbreg`，故这三个类型**正确缺失**，已加测试锁定。

---

### 8.20.74 第 132–134 轮：`estat classification` 与 `estat auc`

#### 第 132 轮：`estat classification` 已正确实现

五个比率全部与 2×2 单元格自洽（1e-12）：

| 指标 | 公式 |
|---|---|
| `sensitivity` | `100·TP/(TP+FN)` |
| `specificity` | `100·TN/(TN+FP)` |
| `PPV` | `100·TP/(TP+FP)` |
| `NPV` | `100·TN/(TN+FN)` |
| `percent_correct` | `100·(TP+TN)/N` |

**行分母与列分母容易混淆**——互换后两者都在 0–100 内，看不出错误。

**适用性判断完全正确**：

| 模型 | 支持 | 依据 |
|---|---|---|
| `logit` | ✅ | `logit_estat.ado:11` |
| `probit` | ✅ | `probit.ado:259` 设 `estat_cmd` |
| **`ologit`** | ❌ | **`ologit.ado` 无 `estat_cmd`** |

`ologit` 是**最易误加**的一个——多类别分类表看似二元分类的自然推广。
测试同时断言**行为**和**ado 证据**，将来要加这个功能必须与证据辩论。

#### 第 133 轮：`estat auc` 是真实缺口，已实现

`logit_estat.ado:19-21` 把 `estat auc` 分派给 `lroc, nograph`。
`lroc.ado:47-49` 用**梯形积分**算面积：

```stata
replace `p' = sum((`spec'-`spec'[_n-1])*(`sens'+`sens'[_n-1]))
return scalar area = `p'[_N]/2
```

数学上等价于 **Mann-Whitney U/(n₁·n₀)**。用秩统计量实现（数值更稳），
并用 `scipy.mannwhitneyu` **独立交叉验证**：`0.942624 == 0.942624`。

`lroc` 只返回两个标量（`N`、`area`），我们的 `to_dict()` 完全对应。

#### 两次测试假设错误，暴露的是我的理解偏差

| 我的假设 | 实际 |
|---|---|
| 翻转结局 → AUC 变 `1−AUC` | **AUC 不变**——模型会重新拟合，系数符号翻转（实测 `-2.4234`）|
| 取负预测变量 → 同理 | 同样重新拟合，排序不变 |

**正确理解：AUC 只依赖拟合后的排序，任何单调变换都不改变它。**

改为测试**真正的不变性**（负值、线性、指数三种变换），另加**敏感性**测试
（噪声越大 AUC 越低）——不变性测试无法发现「实现写反」。

#### 一个有意义的交叉验证

`probit` 与 `logit` 的 AUC **完全相同**（1e-10）：

**链接函数改变概率值但不改变其排序，而 ROC 曲线只依赖排序。**
这同时验证了两条实现路径。

#### 第 134 轮：拒绝路径补全

- `ologit` 拒绝（依据：`ologit_postestimation.sthlp` 中无 `auc`）；
- **仅单一类别时拒绝**——返回 0.5 会把「无从判别」伪装成「无判别力」。

---

### 8.20.75 第 135–137 轮：Hosmer-Lemeshow 检验

#### 第 135 轮：`group(#)` 从拒绝变为实现

`lfit.ado:268` 的核心公式：

```stata
local q = `group' * `w' / `obs'
```

按预测概率排序后，第 i 行属于第 `floor(q)+1` 类（上限 `group`），
再按类计算 Pearson，`df = group − 2`。

#### 三个 bug，全部在分组边界

| 尝试 | 组大小 |
|---|---|
| `scaled[i] >= cur+1` 递增 | `[199, 100, …, 1]` |
| 用前一行判断 | 仍偏移 |
| **`floor(scaled)+1` 直接算** | **`[99, 100, …, 101]`** ✅ |

**第一次的 `[199,…,1]` 仍能算出看似合理的 chi2**——
只有检查**组大小分布**才发现。因此新增测试直接断言每组大小接近 `n/group`。

#### 一个测试设计要点

用**连续协变量**测试：

| 数据 | 可形成的类数 |
|---|---|
| 离散协变量（4 水平）| 4 类，`df=2` |
| 连续协变量（1000 obs）| 完整 10 类，`df=8` |

**在离散数据上断言「10 类」是在断言一个虚构的事实。**

#### 第 136 轮：显示层的两处遗漏

| 缺陷 | 表现 |
|---|---|
| `model_label` 硬编码 | `probit` 报 “Goodness-of-fit test after **logistic** model” |
| `depvar` 未传入 | `Variable:` 行缺失 |

`lfit.ado:116-120` 的标题对两个分支是同一模板，按 `e(cmd)` 区分
（`cond(...=="probit", "probit","logistic")`）。

**这是关于「拟合了什么模型」的陈述，不是装饰性差异。**

新增双向断言：

```python
assert collapsed.chi2 != pytest.approx(uncollapsed.chi2, rel=1e-6)
assert collapsed.df != uncollapsed.df
```

**若 `group()` 静默返回未折叠的 Pearson 数值，两者会相等**——
那时数字本身没错，只是标签用错了。

#### 第 137 轮：`r()` 返回值缺 `N`

`lfit.ado:91-98` 返回 **5 个**标量：`N`、`m`、`chi2`、`df`、`p`。

我们的 `to_dict()` 只给 4 个——**docstring 里引用的是第 88-96 行，但漏掉了第 91 行的 `N`**。

修复后两个分支（Pearson / HL）都返回全部 5 个。

#### 本轮的模式

第 135 轮新增一条分支时，**只复制了计算部分，没有复制显示与返回契约**。

> **新增分支时，「算法」「标签/变量名」「`r()` 键集」三样都要一并复制。**

这与第 106 轮、111 轮的教训同源。

---

### 8.20.76 第 138–140 轮：选项「被静默吃掉」与「不生效」

#### 第 138 轮：`poisson` 的 `estat gof, group()` 被静默忽略

`poisgof.ado:3`：

```stata
syntax [, Pearson]
```

**只有 `Pearson`，没有 `group()`。** 而我们的实现**接受了 `group` 参数却忽略它**，
返回未折叠的 deviance 检验。

**调用者以为拿到了 Hosmer-Lemeshow 折叠检验，实际是未折叠的。**
两者不可互换——数字合理，只有标签暗示区别。

修复：明确抛 `ValueError` 并指出 HL 只属 `lfit`（logit/probit）。

#### 第 139 轮：审计「选项是否真的生效」

| 检查 | 结果 |
|---|---|
| 未知参数被 `**kwargs` 吸收 | ❌ 三方法均正确 `TypeError` |
| `nosort` 生效 | ✅ 模型序 vs VIF 降序 |
| `correlation` 生效 | ✅ 与协方差不同 |
| `variables=` 生效 | ✅ |

**`nosort` 的隐蔽性值得记录**：第一次用的数据里 `x1`/`x2` 的 VIF **都是 80 左右**，
排序与不排序结果相同——**测试会「通过」但什么都没证明**。

**这是第 33 轮踩过的同一个陷阱。**

改用 VIF 差异明显的数据（329 vs 1.0），并新增**守护测试**：

```python
assert sorted_names != unsorted_names   # 若巧合相同，下面的测试毫无意义
```

#### 第 140 轮：`estat summarize, noweights` 未实现

`estat_summarize.sthlp:31-35` 列出四个选项，其中 `nowights` 定义最明确：
**"ignore weights"**。

我们的默认行为是**自动采用估计权重**（`estat_summ.ado:208-231`），
所以 `fweight` 拟合下 `Obs` 是**权重之和**。`noweights` 应恢复原始计数。

| 模式 | `Obs` | `Mean` |
|---|---|---|
| 默认（加权）| 400 | -0.227511 |
| `noweights` | **200** | **-0.181012** |
| 原始数据均值 | — | **-0.181012** ✓ |

#### 一次语义判断的自我修正

我最初认为「显式传入的 `weights=` 应当胜过 `noweights`」。

**但帮助文本写的是 "ignore weights"**——这是对**输出**的陈述，不是对权重来源的陈述。
若声称忽略权重却又采用显式传入的权重，正是本轮要消除的那类**静默矛盾**。

已按文档语义实现，并在测试注释中记录了这次判断过程。

#### 累计四次的同一类问题

| 轮次 | 被误接/未生效的参数 |
|---|---|
| 84 | `estat hettest` 在 logit 上 |
| 102 | `estat vif` 在 ivregress 上 |
| 130 | `predict, scores` 的形状 |
| 138 | `estat gof, group()` 在 poisson 上 |
| **140** | **`estat summarize, noweights`** |

> **「静默接受并给出合理数字」比「崩溃」危险得多。**

---

### 8.20.77 第 141–143 轮：`estat summarize` 的三个选项

`estat_summarize.sthlp:31-35` 列出五个选项，此前全部未实现。

#### 第 141 轮：权重变量必须**非加权**汇总

`estat_summarize.sthlp:96-99`：

> 加权拟合时，对**除权重变量本身之外**的所有变量做加权 summarize，
> **对权重变量本身做非加权 summarize**。

| 变量 | 修复前 | 修复后（= Stata）| 原始值 |
|---|---|---|---|
| `y` | `Obs=393` | `Obs=393` | — |
| **`w`** | **`Obs=393`, `Mean=2.303`** | **`Obs=200`, `Mean=1.965`** ✅ | `200`, `1.965` |

**为何不是装饰性规则**：用权重加权权重本身，计算的是 `Σw²/Σw`——**无解释意义的数**。

**实现难点**：`weight_spec` 只有 `weights`，**没有列名**（权重可能以数组传入）。
改用**数值匹配**识别权重列。

**防止「修一处坏一片」**：

```python
assert table.loc["y", "Obs"] == weights.sum()   # 其他变量仍加权
assert table.loc["w", "Obs"] == len(frame)      # 仅权重列豁免
```

#### 第 142 轮：`noheader`

`estat_summarize.sthlp:93`：**"noheader suppresses the header"**

| 模式 | 输出 |
|---|---|
| 默认 | 空行 + `Number of obs = 150` + 空行 + 表格 |
| `noheader` | 空行 + 表格 |

**关键语义**：header = 表格**上方**的 `Number of obs` 行，**不含列标题**。
若连 `Variable`/`Obs`/`Mean` 也抑制，会得到**一堆没有标签的数字**——
选项「生效」了，输出却不可用。已加测试锁定六个列标题全部保留。

簇块（表格**之后**）也不属于 header，同样保留。

#### 第 143 轮：`labels`

`estat_summarize.sthlp:90`：**"labels displays variable labels"**

两个「不发明信息」的设计点：

1. **标签列追加在统计量之后**——放在变量名旁会**推移所有其他列**；
2. **无标签变量留空**——回退到变量名会让「标签列」显示一个名字，
   读起来像真实元数据。

**数据源说明**：`pandas.read_stata` 默认不保留变量标签，所以标签由调用者提供。
实测 dta 文件里确实有标签（`make` → "Make and Model"），只是读取时丢弃。

#### 选项完成度

| 选项 | 状态 |
|---|---|
| `noweights` | ✅ 第 140 轮 |
| `noheader` | ✅ 第 142 轮 |
| `labels` | ✅ 第 143 轮 |
| `equation` | ❌ 多方程场景，单方程下无意义 |
| `group` | ❌ 仅 `sem`/`gsem` 可用 |

---

### 8.20.78 第 144–146 轮：选项组合测试与适用性矩阵

#### 第 144 轮：为什么必须测组合

第 143 轮的真 bug 无法被任何单选项测试发现：

```python
labels = np.asarray(cluster)   # 局部变量覆盖了 labels= 参数
```

**`cluster=` 与 `labels=` 各自单独测试都通过，只有同时使用才出错。**

> **N 个布尔选项产生 2^N 个组合。** 单独测 N 个，覆盖率是 `N/2^N`。
> `estat summarize` 的 3 个选项，单测只覆盖 3/8 = 37.5%。

实现：对 `noweights`/`noheader`/`labels` 的 **8 种组合**各测两遍，
且**始终传入触发 bug 的那个参数 `cluster=`**。

#### 第 145 轮：`estat vif` 的两个选项会交互

`vif.ado:4`：`syntax [, UNCentered NOSORT]`

组合中才显现的两个行为：

**① `uncentered` 增加一行 `_cons`**（`vif.ado:115-118` 把它写进名字向量）

**② 排序时 `_cons` 参与排序**（`vif.ado:121-123`：`gsort -vvv nms`，`_cons` 已在 `nms` 中）

| 选项 | 顺序 |
|---|---|
| `nosort=False, uncentered=True` | `x2, x1, _cons, x3`（**中间**）|
| `nosort=True, uncentered=True` | `x3, x1, x2, _cons`（末尾）|

**两种都正确**——排序规则不同。

**`Mean VIF` 的分母**（`vif.ado:145`）是**报告的行数**，
所以 `uncentered` 会**两次**改变均值：多一项、分母也大 1。

跨全部组合的恒等式：`tolerance = 1/VIF` 恒成立。

#### 第 146 轮：`estat` 适用性矩阵

| 命令 | OLS | Logit | Poisson | IV2SLS |
|---|---|---|---|---|
| `summarize` | ✅ | ✅ | ✅ | ✅ |
| `vce` | ✅ | ✅ | ✅ | ✅ |
| `ic` | ✅ | ✅ | ✅ | ✅ |
| **`vif`** | ✅ | ❌ | ❌ | ❌ |

**依据**（不是从 `e()` 内容推测）：

- `logit_estat.ado` 只处理 `classification`/`gof`/`auc`，其余转 `estat_default`；
- **`estat_default.ado` 中完全没有 `vif`**（grep 计数 0）；
- `ivregress_estat.ado` 也不分派 `vif`。

**`ivregress` 是最容易误判的一个**：它有 `e(r2)` 和完整的拟合统计量，
看起来「应该」支持 VIF。第 102 轮曾发现它返回的正是 **OLS 的 VIF**——
一个看似合理但拟合于另一个模型的数字。

测试同时断言**行为**与**ado 证据**，将来若要加必须与证据辩论。

---

### 8.20.79 第 147–149 轮：`test` / `lincom` 的语法与渲染

#### 第 147 轮：`test` 的逗号形式解析失败（真缺陷）

```python
r.test('x1 = 0, x2 = 0')     # KeyError: coefficient '0,x2' not found
r.test('x1 = 0', 'x2 = 0')   # ✅
```

**Stata 用户写的是逗号形式**，我们只支持多参数形式。

修复：入口处按逗号拆分。两种写法结果**完全相同**（`F=156.355926`，`df=(2,297)`）。

**防伪测试**：

```python
assert float(joint.df) != float(single.df)
assert float(joint.statistic) != pytest.approx(float(single.statistic))
```

**若逗号被拆成两个独立检验但只保留一个，结果会等于单约束统计量**——数值合理，完全错误。

#### 第 148 轮：`lincom` 的逗号含义**相反**

| 命令 | 语法 | 逗号含义 |
|---|---|---|
| **`test`** | `test exp1, exp2` | **分隔多个假设** |
| **`lincom`** | `lincom exp [, options]` | **分隔表达式与选项** |

依据 `lincom.sthlp`：**只有一个 `exp`**。

修复前的错误信息毫无帮助：

```
KeyError: coefficient 'x2,x1' not found
```

**描述的是症状，不是问题。** 修复后说明「`lincom` 只有一个线性组合」并**指名正确的命令 `test`**。

**防止「解析器统一」的测试**：直接断言两者对同一字符串的**相反行为**，
避免将来有人合并解析器而悄悄改变逗号语义。

#### 第 149 轮：`test` 的约束行从未被渲染

`WaldTest.constraint` **早就有**该文本（带编号、多条换行），但 `__str__` 从未使用它：

```
F(2, 197) =       0.25
Prob > F        =     0.7777
```

**一个没有假设陈述的统计量，读者无法核对。**

修复后：

```
  ( 1) x1 = 0
  ( 2) x2 = 0

F(2, 197) =       0.25
Prob > F        =     0.7777
```

**首个修复版本重复编号**：`constraint` 已带 `( 1)`，我又加了一层前缀，会印出
`( 1) ( 1) x1 = 0`。已加 `assert rendered.count("( 1)") == 1` 锁定。

#### 一次测试假设错误

我最初断言 `test("x1 = x2")` 的输出含 `"x1 = x2"`。

**实际输出是 `x1 - x2 = 0`**——约束被移到左侧，这正是 Stata 的写法。
**规格化是正确的，是我的断言错了。**

---

### 8.20.80 第 151–153 轮：后估计结果对象的 `r()` 表面

#### 起点：第 149 轮发现「`constraint` 一直没被渲染」

这提示了一个更普遍的问题：**对象上有字段，但调用者够不到**。

#### 审计结果：四个结果对象缺 `to_dict()`

| 结果对象 | 修复前 | 修复轮次 |
|---|---|---|
| `WaldTest`（`test`）| ❌ | 151 |
| `LincomResult`（`lincom`）| ❌ | 151 |
| `LRTestResult`（`lrtest`）| ❌ | 152 |
| `HausmanResult`（`hausman`）| ❌ | 152 |
| `MarginsResult`（`margins`）| ❌ | 153 |

其余模块已有 `to_dict`/`to_frame`。

#### 依据与形状

| 命令 | 依据 | `r()` 键 |
|---|---|---|
| `test`（F）| `regress_postestimation.sthlp:770-785` | `F`、`df`、`df_r`、`p` |
| `test`（chi2）| 同上 | **`chi2`**、`df`、`p` |
| `lincom` | `lincom.sthlp:182-189` | `estimate`、`se`、`df`、`t`/`z`、`p`、`lb`、`ub` |
| `lrtest` | `lrtest.sthlp:216-218` | `chi2`、`df`、`p` |
| `hausman` | 同 lrtest（都是 chi2 检验）| `chi2`、`df`、`p` |
| `margins` | `margins.sthlp:1101`（`r(table)` 的行标签）| `b`、`se`、`t`、`p`、`ll`、`ul` |

**统计量以自己分布的名字存储**——`logit` 的 `test` 没有 `r(F)`，
`OLS` 的没有 `r(chi2)`。

#### 三处「不发明表面」的克制

**① `logit.lincom` 的 `df` 是 `None`**

测试暴露直接 `float(None)` 会崩。正确做法是**不包含该键**——
z 基拟合**确实没有** `r(df)`，强制转换等于发明一个 Stata 不存储的值。

**② `LRTestResult` 的两个对数似然不是 `r()` 标量**

Stata 把它们存在两个拟合模型的 `e(ll)` 里。放进 `to_dict()` 会让调用者
以为是本命令的返回值。

**③ `HausmanResult.excluded` 与 `MarginsResult` 的分布标签**

`excluded` 是本包增加的诊断量，保留为属性。

`MarginsResult` **不携带分布信息**，因此统计量键固定为 `t`，
**docstring 明确说明这是无法验证的边界**——命名成我们查不到的分布是猜测。

#### 一条可复用的验证方式

不重复实现自身的算术，而是验证**字段之间的关系**：

```python
assert table["lb"] < table["estimate"] < table["ub"]     # 区间含点估计
assert table["t"] == table["estimate"] / table["se"]     # 三个标量间的恒等式
assert table["p"] == chi2.sf(table["chi2"], table["df"]) # 上尾
assert table["chi2"] == 2*(ll_unrestricted - ll_restricted)  # LR 定义
```

---

### 8.20.81 第 154–156 轮：`margins` 的四种导数形式

#### 起点：三种弹性形式只报 `TypeError`

`margins.sthlp:265-266` 定义四种形式：

| 选项 | 含义 |
|---|---|
| `dydx()` | `∂E[y]/∂x` |
| `eyex()` | `∂lnE[y]/∂lnx`（弹性）|
| `dyex()` | `∂E[y]/∂lnx`（半弹性）|
| `eydx()` | `∂lnE[y]/∂x`（半弹性）|

修复前调用后三种得到：

```
TypeError: margins() got an unexpected keyword argument 'eyex'
```

**这暗示「本包不认识这个参数」，而 Stata 确有该功能。**

#### 第 154 轮：先让拒绝变得诚实

三种形式都**识别**，并说明为何暂不实现：

> 需要**逐观测**的缩放因子，必须同时作用于估计值（在平均之前）与**梯度**。

**先平均再乘系数会得到不同的数字**，且标准误 `√(g'Vg)` 也须按链式法则修正。

同时加了一项检查：**一次只能指定一种形式**；未知参数仍报 `TypeError`。

#### 第 155 轮：`dyex` 可实现，另两种不可

**关键区别**：

| 形式 | 缩放因子 | 是否随观测变化 |
|---|---|---|
| **`dyex`** | `x_i` 单独 | 对线性预测，导数恒定 → **可提出平均** |
| `eyex` | `x_i/E[y_i]` | 含 `E[y_i]`，**逐观测变化** |
| `eydx` | `1/E[y_i]` | 同上 |

验证：`dyex = dydx × mean(x)` 精确成立（1e-10），
且**标准误按同一因子缩放**（delta 方法）。

#### 第 155 轮发现的真实 bug：`at()` 顺序错误

```
修复前: margins(dyex='x1', at={'x1':2}) == margins(dyex='x1')   ← 完全相同
修复后: at(x1=2) 时 dyex == 2 × dydx                            ← 正确
```

**根因**：`at()` 在弹性分支**返回之后**才应用，`_scale` 读到**未调整**的 `exog`。

**危险之处**：返回一个看起来正常的数，只是用错了因子。

**这个 bug 是被我之前写的 `at` 测试抓出来的**——若只测未调整情形（`dyex == dydx × mean(x)`），它会完全逃逸。

#### 第 156 轮：`over` 分支用**组内**均值

| 组 | `dyex` | `dydx` | 组内 `mean(x)` | 乘积 |
|---|---|---|---|---|
| a | -0.034986 | 0.802512 | -0.043595 | -0.034986 |
| b | 3.954848 | 0.802512 | 4.928085 | 3.954848 |

**每组用自己的因子**。若用合并均值，两组会得到相同结果——
而这里 `dydx` 恰好恒定，正是会让弱断言通过的场景，因此新增了**守护测试**。

#### 一条通用教训

**「拒绝」也分质量等级**：

| 拒绝方式 | 传递的信息 |
|---|---|
| `TypeError: unexpected keyword` | 「本包不认识」 |
| `NotImplementedError` + 原因 | 「是 Stata 功能，但需逐观测缩放」 |

后者能让后来者判断如何实现；前者只会让人以为是拼写错误。

---

### 8.20.82 第 157–159 轮：三种弹性形式全部实现

#### 第 157 轮：`eyex`/`eydx` 实现（我上一轮的判断是错的）

第 154 轮我以「需要逐观测缩放，属结构性改动」为由拒绝实现。

**实际只有「除以 E[y_i]」这一步必须逐观测**，而 `predict()` 已经返回逐观测值。

| 形式 | 因子 | 实现值 | 独立验证 |
|---|---|---|---|
| `dyex` | `x_i` | -0.00077237 | `b·mean(x)` ✓ |
| `eydx` | `1/E[y_i]` | 0.27625217 | `b·mean(1/E)` ✓ |
| `eyex` | `x_i/E[y_i]` | -0.09102117 | `b·mean(x/E)` ✓ |

**全部精确到 1e-10。**

**我把「不能把因子提出平均」误解成了「无法实现」。**

#### 第 158 轮：与 `at()` / `over()` 的组合

| 场景 | 验证 |
|---|---|
| `at(x1=2)` | `eydx`/`eyex` 均与独立重算精确一致 |
| `over` 分组 | **每组使用自己的 `E[y_i]`** |

**为何要专门测组合**：第 155 轮的真 bug 正是**在 `at()` 组合中发现**的
（`at()` 曾应用在表单分派之后）。单独测每个形式时完全不可见。

#### 第 159 轮：负预测值的定义域问题

`eyex`/`eydx` 是 **`ln E[y]` 的导数**。当 `E[y] < 0` 时：

- `ln(E[y])` **无定义**
- 但 `1/E[y]` **仍有值** → 实现会返回**一个看起来正常的数**

修复前：35 个负预测值下 `eydx = 3.21518656`（照常返回）。

**Stata 的行为无法从 `ado/base` 查证**（因子在无源码的 Mata 例程中计算），
因此按诚实原则**拒绝并说明**：

```
margins, eydx() is a derivative of ln(E[y]), and 35 fitted value(s) are
negative, where the log is undefined.  Use dydx() for the derivative of
the level, or model a strictly positive response.
```

`dyex` 不受影响（因子为 `x_i`，无对数）。

#### 一条测试设计原则（本轮反复使用）

> **「A 等于 B」的测试必须同时断言「A 不等于 C」**，
> 否则当 A、B、C 三者被同一个错误影响时，测试仍会通过。

本轮三个守护测试都在执行这条：

```python
assert adjusted != pytest.approx(plain)       # at() 真的生效
assert values[0] != pytest.approx(values[1])  # 组因子真的不同
assert np.any(prediction < 0.0)               # 拒绝测试真的会触发
```

#### 「断言某功能不可用」的测试必须随实现更新

本轮按新事实重写了四个测试。这在本会话已出现 **6 次**：

| 轮次 | 功能 | 测试从「断言拒绝」改为「断言可用」 |
|---|---|---|
| 86 | `logit` 的 `estat gof` | ✓ |
| 107 | `tobit` 的 `N_unc` | ✓ |
| 135 | `estat gof, group()` | ✓ |
| 157 | `margins, eyex/eydx` | ✓ |

---

### 8.20.83 第 160–169 轮：IV 后估计命令的完整审计

#### 第 160 轮：`estat firststage` 的 F 检验**分母自由度错误**

`ivregress_estat.ado:581`：

```stata
local dfdenom = e(N) - `rank' - `df_a' - `nendog'
```

而我们实现的是增广设计矩阵的列数：

| | 值 |
|---|---|
| 实现（旧）| 398 |
| **Stata** | **396** |

**偏差来源**：增广矩阵只含截距与工具变量，**没有扣除被解释的内生变量本身**；
Stata 的 `rank`（含内生变量）与 `nendog` 各扣一次。

后果：F 统计量偏小 0.25%（1407.34 vs 1403.79）。

**连带修正 3 处既有测试**，其中恒等式 `partial_r2 = F/(F + df_denom)`
必须使用**与 F 相同的分母**，否则恒等式失效。

#### 第 161 轮：表格形态应随内生变量数量切换

`ivregress_postestimation.sthlp:408-414`：

> 默认情况下，**单个**内生变量报 R-squared/Adjusted/Partial/F；
> **多个**内生变量改报 **Shea 的 partial R-squared**。

**Shea 值早已算好（`fs.shea`），只是从未渲染**——与第 149 轮 `constraint`
未被渲染是同一类问题。

#### 第 162 轮：`all` 选项

`sthlp:408-412`：`all` 请求「无论一个还是多个内生变量」都报告全部统计量，
即**覆盖多内生变量下的 Shea 默认**。

验证点：`all` 只改变**显示哪组统计量**，不改变数值（`array_equal` 全部成立）。

#### 第 163 轮：`forcenonrobust` + 稳健 VCE 检测

`sthlp:416-422`：

> `forcenonrobust` 请求**即便用了稳健 VCE** 仍报告最小特征值统计量及其临界值，
> 因为临界值假设误差独立同分布正态。

因此**稳健 VCE 下默认抑制**。用属性 `reports_mineig` 表达，而非把 `mineig` 置 `None`：

> **`mineig=None` 无法区分「该模型形状不可用」与「当前 VCE 不适用」——
> 这是两类不同的信息。**

#### 第 164 轮：`estat overid, forcenonrobust`

| 情形 | 默认 | `forcenonrobust=True` |
|---|---|---|
| 普通 VCE | Sargan | Sargan |
| **稳健 VCE** | Wooldridge score | **Sargan** |
| **LIML + 稳健 VCE** | **报错** | **Sargan** |

**最有价值的验证**：Sargan/Basmann 只依赖残差，与 VCE 无关，
强制计算的值必须与普通 VCE **完全相同**（1e-10）。

聚类 VCE 仍然拒绝——`forcenonrobust` 针对的是分布假设，不是聚类误差。

#### 第 165 轮：`estat endogenous` 数值验证

Wu-Hausman F = 0.030506 与独立重建的**控制函数 F** 精确一致。

**双向方向性验证**：

```python
# 无内生性数据 → 不拒绝；有内生性数据 → 强烈拒绝
assert quiet > 0.05 > loud
```

**只验「数值等于公式」不够**——若检验从未检测到内生性，公式相符也无意义。

#### 第 166–168 轮：新建 `estat weakrobust`

此前**完全不存在**（同组的 `firststage`/`overid`/`endogenous` 都已实现）。

`sthlp:364-375` 的默认检验**取决于识别状态**：

| 模型 | 默认 | 本轮 |
|---|---|---|
| **恰好识别** | Anderson-Rubin | ✅ 已实现 |
| 过度识别 + 同方差 | CLR (Moreira 2003) | ❌ 明确拒绝 |
| 过度识别 + 稳健 | CLR 推广 | ❌ 明确拒绝 |

**只实现 AR 的理由**：AR 是受限 F 检验，**有闭式公式可独立验证**，
且其分布**不依赖工具变量强度**——这正是该命令存在的目的。

CLR 需要模拟分布，且 Stata 实现是**无 `ado/base` 源码的编译内置**。

**过度识别模型默认拒绝**，而不是返回 AR：
**拿 AR 冒充 CLR 就是把另一个检验的数字贴上 CLR 标签。**

#### 第 167–168 轮：`betanull()` 零假设

`sthlp:457-465`：支持按名称（`varname=#`）或按位置（`numlist ..., copy`），
**未指定的系数取 0**。

**最关键的结构恒等式**：

```python
# AR 在点估计处取极小值 → 统计量恰为 0
statistic = estat_weakrobust(betanull={"e": estimated}).f_statistic
assert statistic == pytest.approx(0.0, abs=1e-8)
```

**若实现忽略了 `betanull`，两次调用会返回相同的非零值**，测试立即失败。

多内生变量下同样验证：联合零假设 → 0；仅指定一个 → 非零。

#### 一条贯穿本轮的教训

**同一条命令的差异分布在不同层次**：数值（160）→ 显示形态（161）→ 选项（162/163）
→ 其他子命令（164/165）→ 整个命令缺失（166）。

**逐层核对才能穷尽**；只查数值会漏掉前四类中的三类。

---

### 8.20.84 第 170–179 轮：面板与时间序列的 `e()` 表面

#### 第 170–171 轮：新建 `estat mundlak`，补齐 `xtreg` 唯一缺失的子命令

`xtreg_estat.ado:5`：

```stata
local allowed_butsum "hausman mundlak vce"
```

加上 `:20-25` 路由到 `estat_default` 的 `su`/`ic`，共**五个**子命令。
`mundlak` 是唯一未实现的。

**检验定义**（`xtreg_postestimation.sthlp:366-370`）：把各回归元的**组内均值**加入方程，
联合检验其系数为零——组均值正是承载 `nu` 与 `x` 相关性的部分。

| 数据 | p 值 |
|---|---|
| 有相关性 | < 0.001（拒绝）|
| 无相关性 | > 0.05（不拒绝）|

**实现中遇到的两个 API 事实**：

1. `model_name` 是 Stata 写法 `'xtreg, re'`，不是描述性词汇；
2. `panel_entity` 存的是**编码后的组标签**，不是列名——因此**无法仅凭结果对象重建方程**，
   `estat_mundlak` 必须要求 `entity=` 参数并说明原因，而不是假装可用。

#### 第 172 轮：`xtreg` 的 `e()` 恒等式

| 键 | 定义 |
|---|---|
| `e(sigma)` | **平方根**为 `sigma_e² + sigma_u²` |
| `e(rho)` | `sigma_u²/(sigma_u² + sigma_e²)` |
| `e(Tcon)` | T 是否恒定（1/0）|

**`e(sigma)` 不是两个标准差之和**——后者更大，且看起来仍像个标准差，因此加了守护断言。

`e(rc)` 是**返回码**（`xtreg.sthlp:1222`），不是统计量；与 `arima.sthlp:579` 同义。

#### 第 173 轮：继承导致的 `e(chi2)` 误报

```python
class RandomEffects(PanelOLS):
    _stores_wald_chi2 = True       # ← 被子类继承
class BetweenOLS(RandomEffects):   # 未覆盖
class FirstDifferenceOLS(RandomEffects):  # 未覆盖
```

`be`/`fd` 因此报告了 Stata **不存储**的 `e(chi2)`，其值就是 `F × df_m`。

**权威依据**（`xtreg.sthlp` 各估计器块）：

| 块 | 行号 | `e(chi2)` |
|---|---|---|
| `re` | 891–951 | ✅ |
| `be` | 1014–1067 | ❌ |
| `fe` | 1068–1144 | ❌ |

**危险之处**：看到 `be` 上有 `e(chi2)` 会合理推断模型检验是 Wald 检验；
实际是 F 检验，那个 chi-square 只是复制。

#### 第 174 轮：面板 `e()` 键集全面核对

| 估计器 | mss | rss | tss | sigma 族 | theta | chi2 |
|---|---|---|---|---|---|---|
| `re` | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `be` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `fe` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

**实际输出全部符合。**

本轮还有一个方法论收获：我的审计脚本误报「`PanelOLS._stores_mss` 泄漏」——

**实际语义是「存 `mss` 但不存 `tss`」**，专为 `be` 设计：

```python
if self._stores_totals or self._stores_mss:   # 两条路径都产出 mss
```

`fe` 通过 `_stores_totals` 产出 `mss`，所以 `_stores_mss=False` 是**正确的**。
已把语义写进注释并加测试锁定。

新增 `test_the_flag_matrix_is_pinned`：**4 个类 × 7 个标志全部固定**，
第 173 轮那种「子类意外继承」会直接出现在 diff 中。

#### 第 175–176 轮：`arima` 的 `e()` 标量与宏

补齐 13 项：`k`、`ar_max`、`ma_max`、`rc`、`N_gaps`、`ic`、`depvar`、`eqnames`、
`covariates`、`tmin`、`tmax`、`tmins`、`tmaxs`。

**最容易错的一条**：`ar_max`/`ma_max` 报的是**有效**阶数，季节项要乘进：

```
ARIMA(1,0,0)(1,0,0,4) → ar_max = 1 + 1×4 = 5
```

**只报非季节阶会得到 1——一个看似合理但错误的数。**

`e(depvar)` 此前**完全缺失**，即使传入带名称的 pandas Series。

`e(covariates)` 用**空字符串**而非 `None`——后者表示「未存储」，`""` 是 Stata 实际记录的值。

#### 第 177 轮：`var` 的 9 个宏

| 键 | 值 |
|---|---|
| `e(depvar)` | `'y1 y2'` —— **列表**，不是单个名字 |
| `e(eqnames)`/`e(endog)` | 同上 |
| `e(lags)` | `'1 2'` —— **滞后列表** |
| `e(df_m)`/`e(df_r)` | 每方程回归元数 / 残差自由度 |
| `e(nocons)` | 0/1 |

**修正了一个过严的既有测试**：它要求 `e()` 中**所有值都是 float**——
这在宏缺失时才成立。Stata 的宏本来就是字符串。改为按 `DOCUMENTED_MACROS` 集合区分。

#### 第 178 轮：`VECM` 缺少结果包装类

```python
VAR.fit()   → VARResults            # 本包定义，有 e()
VECM.fit()  → statsmodels 原生对象   # 无 e()
```

**任何按 `e()` 泛化的后估计代码，在 `var` 上工作、在 `vec` 上抛 `AttributeError`。**

新增 `VECMResults`，含 `e(k_ce)`（协整秩）与 `e(k_ar_diff)`（**= `lags − 1`**，
Stata 与 statsmodels 的滞后约定不同）。

**包装的代价由测试暴露**：conformance 测试报
`AttributeError: 'VECMResults' object has no attribute 'stderr_alpha'`。

**修复方式：转发而非逐个补齐**

```python
def __getattr__(self, name):
    if name == "raw": raise AttributeError(name)
    return getattr(self.raw, name)
```

> **包装的目的是「增加 Stata 界面」，不是「替换 statsmodels」。**
> 逐个绑定属性意味着每新增一种原生用法都会再次失败。

#### 本轮最重要的方法论

**同一条命令的差异分布在多个层次**（这是第 160–169 轮已总结的规律，本轮再次验证）：

| 层次 | 本轮实例 |
|---|---|
| 数值 | `e(sigma)`/`e(rho)` 的定义 |
| 显示形态 | 无 |
| 选项 | `estat mundlak` 的 `entity=` |
| **整条命令缺失** | `estat mundlak` |
| **结果对象缺接口** | `VECMResults` |
| **继承泄漏** | `_stores_wald_chi2` |

**逐层核对才能穷尽。**

---

### 8.20.85 第 180–189 轮：渲染、访问器与「注册但未实现」

#### 第 180 轮：`VECM` 的摘要表

修复前 `str(vecm_fit)` 返回 **dataclass 的字段 repr**：

```
VECMResults(raw=<statsmodels object>, names=['y','x'], lags=2, ...)
```

**这不是结果，是对象的内部结构。**

按 `vec.sthlp:61-71` 实现两张表：

| 表 | 默认 | 控制选项 |
|---|---|---|
| 协整方程（beta） | ✅ | `nobtable` 关闭 |
| 调整参数（alpha） | ✅ | `noetable` 关闭 |

**首版渲染错误**：beta 表里混入了 `chi2`/`nan` 两列（误把方程检验列当成系数表）。

**动态列数验证**：`rank(1)` → 仅 `ce1`；`rank(2)` → `ce1`、`ce2`。

#### 第 181 轮：`stata_params` 是方法而非属性

| 访问器 | 修复前 |
|---|---|
| `stata_param_names` | 属性 ✅ |
| `bse` | 属性 ✅ |
| `aic` | 属性 ✅ |
| **`stata_params`** | **方法** ❌ |

`result.stata_params[0]` 报 `TypeError`——**对象看起来正确，索引时才崩**。

修复后更新 3 处消费者（源码 1 处、测试 2 处）。

#### 第 182 轮：`arima` 缺 `forecast`

```python
r.predict(n=5)
# ConsistencyError: the design matrix was not retained
```

**这个错误描述的是错误的问题**——时间序列模型**本来就没有设计矩阵**。

`predict, xb` 是回归接口；Stata 在 `arima` 之后的后估计命令是 **`forecast`**。

修复：`r.forecast(5)`。**`predict` 仍不提供**——提供它就得发明一个设计矩阵。

#### 第 183 轮：`VAR.forecast` 完全不可用

```python
r.forecast(3)
# AttributeError: 'VARResults' object has no attribute 'y'
```

根因：statsmodels 暴露的是 **`endog`**，我们读了 **`.y`**。

**报错来自 statsmodels 的属性包装器**，看起来像兼容性问题，实际是本包写错属性名。

**为何能存活这么久**：同族的 `irf`/`fevd` **一直是好的**——只用过它们的调用者永远遇不到。

#### 第 184–185 轮：整个 `qreg` 系列的 `predict` 缺失，以及 `stddp`

`SUPPORTED_TYPES` 中**没有 `qreg` 系列的任何条目**，所有类型全部拒绝——包括
只需设计矩阵的 `xb`。

依据 `qreg_postestimation.sthlp:86-105`：

| 命令 | 类型 |
|---|---|
| `qreg`/`iqreg`/`bsqreg` | `xb`、`stdp`、`residuals` |
| **`sqreg`** | 另有 **`stddp`** |

**第 184 轮注册了 `stddp` 却没实现**，第 185 轮调用时被自检机制抓住：

```
UnsupportedError: type 'stddp' is registered for 'sqreg' but has no
implementation; this is a westata bug
```

`stddp` 实现要点（`sthlp:130-132`）：

$$\text{stddp} = \sqrt{V_1 + V_2 - 2\text{Cov}_{12}}$$

**交叉协方差项是关键**——忽略它会系统性高估差值（两方程通常正相关）。
已加守护测试确认结果**不等于** `sqrt(V1+V2)`。

#### 第 186–188 轮：三张注册表的一致性防守

**第 186 轮：`SUPPORTED_TYPES` ↔ 实现**

「注册但未实现」在本会话出现 **3 次**（`ir`、`score`、`stddp`），每次靠人工发现。

新增结构性检查：

```python
declared = {kind for kinds in SUPPORTED_TYPES.values() for kind in kinds}
missing = [kind for kind in declared if f'"{kind}"' not in source]
assert missing == []
```

**新增类型必须同时在 dispatch 中出现字符串字面量。**

**第 187 轮：三个诊断命令 ↔ 同一适用性规则**

`regress_postestimation.sthlp:42-44` 把 `hettest`/`imtest`/`ovtest` 都列为
regress 后估计命令，因此只适用于线性模型。

三者实现在**独立模块**中，没有结构强制一致，因此直接断言一致性：

```python
assert len(set(outcomes)) == 1   # 三种命令对同一模型必须给出相同的是/否
```

**若某个接受了 logit 而另两个拒绝，那就是它错了。**

**第 188 轮：`_LINEAR_MODELS` 的两个排除**

| 排除项 | 依据 |
|---|---|
| `ivregress` | `ivregress_estat.ado:20-56` → `estat_default`（无 vif）|
| **`newey`** | **`newey_estat.ado:11-17` 只处理 `durbinalt`** |

**最值得记录的对比**：

| 命令 | `predict` | `estat vif` |
|---|---|---|
| `newey` | ✅ | ❌ |

**因为两者由不同的 ado 服务**（`newey_p.ado` vs `newey_estat.ado`）。

> **同一估计器的两个后估计命令可以有完全不同的可用性。**

#### 本轮的方法论总结

前 10 轮（160–169）发现差异分布在**数值/显示/选项/命令缺失**四层；
本轮又确认了两层：

| 层次 | 本轮实例 |
|---|---|
| **结果对象缺接口** | `VECMResults` 无 `str` |
| **访问器形态不一致** | `stata_params` 是方法 |
| **注册表与实现脱节** | `stddp` 注册未实现 |
| **同族 API 单点失效** | `VAR.forecast` |

> **「看起来能用」的接口最容易长期失效**——因为无人调用就不会暴露。

---

### 8.20.86 第 190–199 轮：`estat endogenous` 分支、`VAR`/`VECM` 后估计

#### 第 190 轮：`estat endogenous` 有四条 VCE 分支

| VCE | 检验 | 结果类型 | `r()` 键 |
|---|---|---|---|
| 普通 | Durbin + Wu-Hausman | `EndogenousResult` | `durbin`、`wu` |
| 聚类 | 稳健回归 F | `RegFResult` | `regF` |
| 稳健 | 稳健 score | `EndogScoreResult` | `r_score` |
| HAC | HAC score | `EndogScoreResult`（含 `lags`）| `hac_score` |

**四种是真正不同的统计量。** 统一的是契约（都有 `to_dict()`），不是字段名——
`EndogenousResult` 有两组 p 值，加一个单一 `pvalue` 会制造虚假对称。

#### 第 191–193 轮：`VAR` 的三个 `estat` 命令

`var_postestimation.sthlp:45-47` 列出 `estat ic`/`summarize`/`vce`。
`VARResults` 是**独立 dataclass，不继承 `StataResults`**，三者全缺。

实现中发现三处真实缺陷：

| 缺陷 | 表现 |
|---|---|
| `e(cmd)` 从未存储 | `estat ic` 的 Model 列显示 `.` |
| `cov_params` 是属性 | 共享实现按方法调用 → `TypeError` |
| `ll(null)` 缺失渲染成 `nan` | 应为 Stata 的 `.` |

**`nan` 读作一个数字，`.` 明确表示不可用。**

第 193 轮又发现 `estat_vce` 的**索引重复**：两方程 VAR 的 `L.y1`/`_cons` 各出现两次，
`frame.loc["L.y1"]` 返回两行——**矩阵无法回答「这个系数的方差是多少」**。
加方程前缀后唯一。

**没有发明 Stata 的多方程排版**（`post_estatvce.sthlp` 无依据），只解决数据正确性。

#### 第 194–198 轮：`VECM` 的后估计

`vec_postestimation.sthlp:44-46` 与 `var` 相同的三个命令，`VECMResults` 同样全缺。

**`estat ic` 与 `estat summarize` 已实现；`estat vce` 明确拒绝**：

> VECM 参数分布在三个形状不同的块（`alpha` 2×1、`beta` 2×1 仅差归一化、`gamma`），
> 而 `cov_params_default` 是 10×10。**没有任何单一展平方式能与之对应。**
> 在错误的归一化下标注矩阵，比不提供矩阵更糟。

#### `vec` 的 `e()` 与 `var` **不同**

| 键 | `var` | `vec` |
|---|---|---|
| 参数个数 | `e(k)` | **`e(k_rank)`**（无 `e(k)`）|
| 协整秩 | — | `e(k_ce)` |
| 逐方程 | `k_#`/`df_m#`/`r2_#`/`rmse_#` | 同（但 VECM 的 `r2` 相对**差分响应**）|

**一处 Stata 自身的拼写不对称**：`df_m1`（索引直接跟）vs `k_1`（带下划线）。
已加测试锁定，防止后来者「统一拼写」时破坏文档键名。

#### 一个负行列式暴露的错误来源

`detsig_ml`（`vec.sthlp:365`）首版取自 `cov_var_repr`：

```
-9.308115442380354e-50     ← 负值
```

**协方差矩阵的行列式不可能为负** —— 这个不可能的值直接指向来源错误。
正确来源是 `raw.sigma_u`，修正后为 `0.2433`。

#### 一条被一致性检查抓住的回归

第 196 轮用 `k_rank` 计算了 `e(aic)`，**却让 `estat_ic` 用了 `e(k)`**：

```
estat_ic 表 AIC : 1310.9062
e(aic) × N      : 1310.9062   ← 第 198 轮修复后一致
```

**修复方式是把 `df` 改为 `k_rank`**，并新增交叉验证：

```python
assert table.aic == e(aic) * N
assert table.bic == e(sbic) * N
```

#### 一条刻意的「不提供」

`vec.sthlp:357-358` 只写「`e(chi2_#)` chi-squared statistic for equation #」，
**未说明检验什么假设，ado 中也无公式**。

首版用 `nan` 占位，随后改为**不提供该键**：

> **`nan` 读作「计算失败」，缺失键读作「未实现」。后者才是诚实的，
> 也是调用者能无歧义检测的。**

#### 本轮的元教训

**「注册但未实现」「属性/方法不一致」「字段从未渲染」三类问题反复出现**，
每次都要靠单独测试或全套测试才能发现。

> **同族 API 中的单个成员失效是最难发现的一类** ——
> 因为只用过其他成员的调用者永远不会遇到它
> （`VAR.forecast` 崩溃而 `irf`/`fevd` 正常；`stddp` 注册未实现而 `xb` 正常）。

---

### 8.20.87 第 200–209 轮：补齐 `regress` 后估计的全部 7 个 `estat`，以及文档/证据一致性

#### 第 200–202 轮：三个缺失的 `estat` 子命令

`regress_postestimation.sthlp` 共列出 **7 个** `estat`，此前只有 4 个（`hettest`/`imtest`/`ovtest`/`vif`）。

**`estat szroeter`（E1）** —— 公式**逐字抄自可读 ado 源码** `szroeter.ado:158-178`：

```stata
egen `ht' = rank(`var')
scalar `mh' = r(mean)      // 秩均值
scalar `vh' = r(Var)       // 秩的样本方差
summ `ht' [aw=`res'^2]
scalar `h' = r(mean)       // 残差平方加权的秩均值
Q = (`n'^2 / (2*(`n'-1))) * (`h'-`mh')^2 / `vh'
```

**用秩而非水平值**：备择假设是「方差随变量**单调**变化」。
三类拒绝对齐 `szroeter.ado:13-22`（仅 regress / 不加权 / 不聚类）。

**`estat esize`（E1）** —— `esize.ado` 把算式交给**编译内置**，公式取自帮助文件所引文献：

| 类型 | 公式 |
|---|---|
| `eta2`（默认）| `SS_effect/SS_total`（**= R²**）|
| `epsilon2` | `(SS_effect − df_eff·MS_err)/SS_total` |
| `omega2` | `(SS_effect − df_eff·MS_err)/(SS_total + MS_err)` |

`regress` **不存 `e(tss)`**，故 `SS_total = mss + rss`。
eta² 的置信区间用非中心 F（Smithson 2001）。

**`estat moran`（E1，且有重要保留）** —— `estat_moran.sthlp:56-59` 明确：

> The above sections **are not included in this help file**（Methods and formulas）

且 **`ado/base` 中无 `estat_moran.ado`**（编译内置）。

**因此 Stata 的 `r(chi2)` 精确形式在本机无法查证。** 实现的是**标准随机化 Moran 检验**
（Moran 1950；Cliff & Ord 1981），p 值对「无空间相关」精确，**但可能与 Stata 不同**。
本包亦无 `spmatrix`/`spset` 工具链，权重矩阵须由调用者传入。

#### 第 203–204 轮：文档覆盖的自动检查

**第 203 轮**发现 6 个 `estat` **在详细章节有记录但主表无行**：
`auc`、`weakrobust`、`mundlak`、`szroeter`、`esize`、`moran`。

**第 204 轮**把同样的检查用于 `predict`，又发现 8 个未登记，其中
**`ystar` 与 `fitted` 在文档中完全零命中**。

两者都固化为 `@pytest.mark.parametrize` 检查：

```python
@pytest.mark.parametrize("method", _implemented_estat_methods())   # 18 个
@pytest.mark.parametrize("kind", _predict_kinds())                 # 25 种
```

> **一条能工作但不在清单里的命令，不是「漏了文档」，而是「一份写着假话的文档」。**

#### 第 205–206 轮：证据注册表

`README.md:41-42` 承诺「every command carries an evidence level」，
但 **`estat` 系列在 `registry.yml` 中完全没有登记**（87 条全为估计命令）。

补齐 6 条，等级**如实评定为 E1**（均**未与 Stata 实跑对比**）：

| 命令 | 关键说明 |
|---|---|
| `estat szroeter` | 公式可读，但无 fixture |
| `estat esize` | 算式在编译内置，公式取自文献 |
| `estat moran` | **Stata 依据完全不可得** |
| `estat auc` | 早期「翻转结局 → 1−AUC」测试**失败**（模型重拟合，AUC 不变）|
| `estat weakrobust` | **仅 AR 部分**；CLR 拒绝 |
| `estat mundlak` | 公式标准，无 fixture |

**同时发现 `GAP_REGISTER.md:8` 的等级分布完全错误**：写 76/11/0，实际 52/26/12。

**schema 校验拦截了一次错误**：我用了 `ado_source`/`manual`/`published_formula`，
但注册表只允许 `invariant`/`published_output`/`r_crosscheck`/`statsmodels_reference`。

#### 第 207–208 轮：结果对象的标签一致性

**第 207 轮**：`EndogenousResult` **唯独缺少 `test` 属性**，而两个兄弟类型都有。
`getattr(result, "test", None)` 在三分支中恰好一个返回 `None`。

**第 208 轮**把完整标签表钉进测试：

| 场景 | 标签 |
|---|---|
| `estat overid`（普通）| `Sargan` |
| `estat overid`（稳健/HAC）| `Wooldridge score` |
| `estat endogenous`（普通）| `Durbin-Wu-Hausman` |
| `estat endogenous`（稳健/HAC）| **`robwool`**（Stata 自己的缩写）|
| `estat endogenous`（聚类）| `regF` |
| `estat weakrobust` | `Anderson-Rubin` |

**关键验证**：同一命令在不同 VCE 下标签**不同**，因此调用者**不能**硬编码
`estat overid` == Sargan。

#### 本周期的方法论

**「已实现但不可见」有三个层次，本周期各遇到一次**：

| 层次 | 实例 |
|---|---|
| 未实现 | `szroeter`/`esize`/`moran` 三个 `estat` |
| 已实现但未文档化 | 6 个 `estat` + 8 个 `predict` 类型 |
| 已实现但无证据登记 | 全部 `estat` 系列 |

> **三者都是「看起来完整」的系统**——只有对着权威清单逐项核对才能发现。

---

### 8.20.88 第 210–219 轮：`linktest`、`suest`、`score` 语义与两次自我纠正

#### 第 210 轮：`linktest`（Pregibon 链检验）

`logit_postestimation.sthlp` 列出三个后估计命令，前两个早已实现，`linktest` 一直缺失。

`linktest.ado:106-134` 是可读源码：

```stata
quietly _predict `hat', xb          // 拟合模型的线性指数
quietly gen `hat2' = `hat'^2
quietly `nxtcmd' `lhs' _hat _hatsq  // ★ 用同一估计器重拟合
ret scalar t  = _b[_hatsq]/_se[_hatsq]
ret scalar df = e(df_r)
```

**关键：不是对原模型做检验，而是重拟合含 `_hat`/`_hatsq` 的方程。**

**一处易错点**：`df` 取**重拟合后**的 `e(df_r)` = **n − 3**（原模型是 n − 2）。

双向诊断已验证：线性设定正确时不拒绝；含真实二次项时 p < 0.001。

`ado:52-66` 明确拒绝的命令（`zip`/`zinb`/`sureg`/`mvreg`/`eivreg`/`iv*`/`xt*`）同样拒绝。

#### 第 211–213 轮：`suest` 及其实用性

`suest.sthlp:84-88`：把多个拟合堆叠为一个参数向量 + **联合 sandwich 协方差**，
「**即使估计来自同一批或重叠的数据，该矩阵仍然适用**」。

**核心：交叉块必须非零。** 两个模型用同一批观测时得分相关，
块对角堆叠会错误地报零协方差。sandwich 用 `V = B(Σsᵢsᵢ')B` 按**观测逐行**配对。

**前置条件的发现与补齐**：`sthlp:99-102` 要求各模型能 `predict, score`，
而 **`regress` 原本没有 `score` 类型**。查证
`regress_postestimation.sthlp:193`：**「score is equivalent to residuals in linear
regression」**，据此补上。

第 212 轮补 `test`/`lincom`/`cov_params()`/`df_resid`（联合检验分母取各分量**最小**者），
第 213 轮补 `vce(cluster)`（meat 按簇先聚合再取外积）。
**边界验证**：每个观测自成一簇时与不聚类**完全一致**，证明这是真泛化而非另一套公式。

#### 第 214–216 轮：`score` 语义的三次澄清（含两次自我纠正）

**第 214 轮的错误**：把 `logit_p.ado:4` 的 `[, SCores *]` 读成「复数拼写」，
注册了 `scores`（返回 n×k 矩阵）。

**第 216 轮纠正**——这是双重错误：

1. **`SCores` 是 Stata 的解析器缩写规则**，匹配的是文档化的 `score`，不是给选项改名；
2. 即使存在 `scores`，`predict.sthlp:192-195` 明确它是**多方程模型**的选项——
   「**每个方程**一个得分变量」，而 `logit` 是单方程。

**因此改为注册单数 `score`（`∂ll/∂η`）。** 注意 `logit` 的 `score` 此前**从未存在**。

**第 217 轮的真 bug**：`probit` 复用了 `logit` 的 `y − p`。

Probit 的正确得分是：

$$\frac{\partial ll}{\partial \eta} = \frac{(y-p)\,\phi(\eta)}{p(1-p)}$$

**用一阶条件发现的**：有截距时 `Σ ∂llᵢ/∂ηᵢ` 必须为 0，
而 probit 给出 **0.2504**。修正后为 3e-15。

**第 218 轮**：把该恒等式确立为通用验收判据，四个模型
（`regress`/`logit`/`probit`/`poisson`）全部满足，并加配对测试
（无截距时和**不为**零，防止全零假实现通过）。

**`ologit` 的 `scores` 再次搁置，这次有具体证据**：模型 `predict(pr)` 给出
P₁ 之和 249.06，而按 `F(cut₁−η)` 手算得 232.6——**本包的有序 logit 内部参数化与标准形式不同**，
因此无法可靠推导得分分母。**保持缺失，并把该对比写进测试。**

#### 第 215 轮：一次代价高昂的返工

该轮尝试实现 `tobit` 的 `scores`（`tobit_p.ado:295-330` 有闭式公式），
**实现后列和不等于零**——判据不成立，于是撤下。

**但撤下时同时删除了大块代码，导致 `predict.py` 语法损坏**，
随后十余次修补尝试**每次都在制造新的重复定义**，最终用事先验证过的完整版本覆盖恢复。

**教训**：发现公式不对时，**只做「取消注册」这一处最小改动**（一行），
不要同时删除实现函数。**删除大块代码正是损坏的开始。**

#### 本周期的方法论

**`Σ score = 0`（有截距）是一个极强的判据**：

| 属性 | 说明 |
|---|---|
| 无需 fixture | 是最大似然的一阶条件 |
| 能抓住符号错误 | Logit/Probit 混用立即暴露 |
| 能抓住链接函数错误 | Probit 漏掉密度比立即暴露 |
| 能抓住离散因子错误 | nbreg 类模型会失败 |

**但它对「全零」是恒真的**，因此必须配「无截距时和为 0 **不**成立」的对照测试。

---

### 8.20.89 第 220–229 轮：`e()` 键的反向审计，以及「宏当能力清单」这条方法

#### 第 220–221 轮：`e(aic)` 不是 Stata 的键

`regress.sthlp` 的返回值区块中 grep `aic` **计数为 0**；整个 `ado/base` 里只有
`binreg`/`glm`/`factor` 等少数命令文档化 `e(aic)`——**本包实现的估计器一个都不在其列**。

本包仍暴露 `aic`/`bic`（调用者依赖），但它们在高斯偏差尺度上：

```
aic = N·log(rss/N) + 2k
```

**省略了 `N(1 + log 2π)` 常数**，这正是它与 `estat ic` 的 AIC 相差一个常数的原因：

| 量 | 公式 | 值 |
|---|---|---|
| `e(aic)`（本包便利值）| `N·log(rss/N) + 2k` | 14.8511 |
| `estat_ic` 表 AIC（**符合 Stata**）| `-2ll + 2k` | 866.2142 |

**处理**：不移除，但加注释说明，并新增测试直接**读取 Stata 帮助文件**断言 `aic` 不存在——
将来 Stata 升级若加上该键，测试会提醒重新评估。

**第 221 轮**逐项核对了 `regress` 的 14 个标量键，确认本包只多出
`sigma`（`e(rmse)` 的别名）、`aic`、`bic` 三个，且**正确地没有 `tss`**。

> **一次自我纠正**：最初用 `grep -oE "cmd:e\([a-z_0-9]+\)"` 提取，得到 17 个键，
> **其中没有 `N` 和 `F`**，差点误判它们也是多余的。实际帮助文件写成
> `{synopt:{cmd:e(N)}}`，大写字母不在 `[a-z_0-9]` 内。
> **模式匹配的字符集限制会静默漏掉匹配项，看起来就像「Stata 没有这个键」。**

#### 第 222–223 轮：`ereturn local` 块

`logit.ado:252-268` 与 `probit.ado:252-259` 逐字设置：

```stata
ereturn local title "Logistic regression"
ereturn local marginsnotok stdp DBeta DEviance DX2 DDeviance Hat Number ///
                           Residuals RStandard SCore
ereturn local marginsok default Pr
ereturn local predict logit_p
ereturn local estat_cmd logit_estat
```

**这些都不在帮助文件的返回值区块中**——帮助文件记录的是**估计量**，而它们是 ado 里的
**管道接线**，只有读源码才能得到。

**一处既有测试的前提被修正**：`test_underivable_macros_are_absent_not_invented`
原本把 `title` 归入「不可推导」并断言其不存在。但 `logit.ado:253` 写的是**字面字符串**，
逐字抄录**不是编造**。已把 `title` 移出该清单，`vcetype`/`cmdline` 保留
（它们确实随调用而变）。

> **「从源码抄录字面量」与「凭印象拼接字符串」是两件事。**

#### 第 224–227 轮：`e(marginsok)` 是一份能力清单

**第 224 轮**发现 `margins` **从不读取** `e(marginsok)`/`e(marginsnotok)`——
只在注释里提过。而 Stata 依赖这两个宏禁止特定预测类型：

```stata
areg.ado:255    marginsnotok Residuals SCore D XBD
arch.ado:1531   marginsnotok Residuals YResiduals
arima.ado:727   marginsnotok stdp mse Residuals YResiduals
```

修复后行为与宏完全一致，且拒绝信息会指出两个宏的内容。
另修正 `predict_kind="default"` 被当作字面类型名拒收的问题。

**第 225 轮**把宏推广到 `regress`/`tobit`/`qreg`。三者取值**完全不同**：

| 命令 | `marginsok` | `marginsnotok` |
|---|---|---|
| `regress` | `XB default` | **未设置** |
| `tobit` | `default XB Pr(passthru) E(passthru) YStar(passthru)` | **空字符串** |
| `qreg` | 空（已清除）| `stdp stddp residuals` |

> **「未设置」与「设为空」是两种不同的陈述。**

#### 第 226–228 轮：用宏反查缺失的 predict 类型

**这是本周期最有价值的方法**：

> **`e(marginsok)` 不只是给 `margins` 用的——它是一份「本命令支持哪些预测类型」
> 的权威声明，可以逐项对照 `SUPPORTED_TYPES` 找出实现缺了什么。**

三轮各命中一个命令：

| 轮次 | 命令 | 缺失类型 |
|---|---|---|
| 226 | `tobit` | `pr(a,b)` |
| 227 | `intreg` | `pr`、`e`、`ystar` |
| **228** | **`heckman`** | **`psel`、`mills`、`nshazard`、`ycond`、`yexpected`** |

**第 227 轮的尺度陷阱**：`tobit` 存 `sigma`（直接值），`intreg` 存 **`/lnsigma`（对数）**。
若直接用 `params[-1]` 作 sigma，`intreg` 得到 ~0.0006，正态概率**饱和成 0 或 1**——
「看起来像概率」但完全错误，且不报错。新增 `_scale_sigma` 按参数名判断，
并加测试断言 `pr` 的标准差 > 1e-3（尺度误读会把它压成常数）。

**第 228 轮的两处结构问题**：

1. `heckman` 的选择方程在**全样本**拟合（400 行），结果方程只在**选中行**（256 行）。
   两个统计量集合的行数不同是语义要求。
2. `ycond = xb + λ·mills` 的恒等式在**第三位小数**处不符——根因是回退逻辑取了
   「前 256 行」，而**真实选中的行是交错分布的**。改为从 `model_data` 推导
   （`y` 缺失处即未选中），恒等式随即精确成立。

#### 本周期固化的判据

**有截距时 `Σ ∂llᵢ/∂ηᵢ = 0`**（截距的一阶条件），无需 fixture：

| 属性 | 说明 |
|---|---|
| 能抓符号错误 | Logit/Probit 混用立即暴露（probit 曾给 0.2504）|
| 能抓链接函数错误 | Probit 漏掉密度比立即暴露 |
| **但对「全零」恒真** | 故必须配「无截距时和为 0 **不**成立」的对照测试 |

---

### 8.20.90 第 230–239 轮：把宏变成可执行的自检，以及两处「含义搞错」

#### 第 230 轮：`etregress` 的 5 个处理效应统计量

依据 `etregress.ado:423`（`marginsok "XB CTE YCTrt YCNTrt PTrt XBTrt default"`）
与 `etregress_postestimation.sthlp:136-167`：

| 类型 | 含义 |
|---|---|
| `ptrt` | `Pr(treatment=1)` |
| `xbtrt` | 处理方程线性预测 |
| `yctrt` / `ycntrt` | `E(y\|t=1)` / `E(y\|t=0)` |
| **`cte`** | **两者之差**（定义即「潜在结果均值之差」）|

**`cte` 是导出量而非独立公式**，因此恒等式 `cte == yctrt − ycntrt` 可验证到 1e-12。

**一处被既有测试抓住的假设错误**：最初用 `("t","treat","treatment")` 猜处理变量名，
但 `test_predict_registry` 用了一列叫 **`yb`** 的处理变量。改为按**参数位置**推导——
`Etregress.from_formula` 把处理变量追加为结果设计的最后一列（`names = list(...) + [treatment_lhs]`），
**这是结构事实，不是命名约定**。

#### 第 231 轮：把「宏即能力清单」变成自动测试

前四轮靠人工逐条核对 `e(marginsok)` 与 `SUPPORTED_TYPES`。第 231 轮固化为参数化测试，
**立即发现三个我自己造成的缺口**：

| 模型 | 问题 |
|---|---|
| `heckman` | **完全没有 `e(marginsok)`** |
| `etregress` | 同上 |
| `intreg` | 同上 |

第 228/230 轮实现了预测类型，却**忘记把宏写进 `e()`**。

> **人工检查连续 4 轮只看「宏 → 实现」方向，没人检查「实现 → 宏」方向。**

#### 第 232–233 轮：`e(estat_cmd)` / `e(title)`

| 键 | 处置 | 理由 |
|---|---|---|
| `e(estat_cmd)` | ✅ 加（`regress_estat`）| `regress_estat.ado:2` 存在 + `estat.ado:10-11` 实际消费该宏 |
| **`e(predict)`** | **刻意不加** | `regress_p.ado` **不存在**（编译），值不可确认 |

> **缺失的键至少表示「未知」；错误的字符串会静默调用错误的程序。**

**第 233 轮我也犯了一处语义错误**：看到 `qreg.ado:189` 是裸 `ereturn local title`，
就记录成 `title: ""`。既有测试指出：

> **裸 `ereturn local name` 会清除宏，键随之不存在。**

与 `tobit` 显式赋空字符串的 `marginsnotok` 恰成对照——**看起来一样，语义完全不同**。

#### 第 234–237 轮：展示层与 Stata 对齐

**第 234 轮**：`regress` 的 summary 打印了 Stata 表格**没有**的 AIC/BIC，
且数值在高斯偏差尺度上（`N·log(rss/N)+2k`），与 `estat ic` 的 `-2ll+2k` 不同。
改为按 `e(_aic_is_stata)` 条件显示：`regress` 不显示，ML 族继续显示。

**第 235 轮**：ML 模型的表头是 OLS 样式，显示
`R-squared: .` / `Adj. R-squared: .` / `F-statistic: .` / `Df Residuals: 250`
——**一行的 `.` 让表格看起来像坏掉的回归表**。改为按命令实际报告的内容选区块。

**判据的关键选择**：`tobit` 的 `dist` 是 `"t"`，但它报告 LR chi2 而无 F，
所以按 `dist == "z"` 判断会重犯原错误。改为：

```python
reports_lr = "chi2" in e and "F" not in e and "r2" not in e
```

**第 236 轮**：`style="stata"` 的标题是固定字符串，而 Stata 的标题随命令变化
（`Linear regression` / `Logistic regression` / …）。改为读 `e(title)`，
`poisson` 因 ado 无该宏而回退到中性标题。

**第 237 轮**：系数表头写成 `Std. err.` / `[95% conf. interval]`，
**与项目自己的规范 `design.md:209` 矛盾**（规范要求 `Std. Err.` / `[95% Conf. Interval]`）。
同处还有第二个缺陷：**`95%` 是硬编码的**，`level=0.90` 的结果仍显示 95%。

> **错误的表头会静默地误标它下面的每一个数。**

8 个既有测试断言了旧的小写表头，依规范一并更新。

#### 第 238 轮：`e(ic)` 的含义搞错了

`logit.sthlp:257` 原文：**`e(ic)` number of iterations**。
本包实现为 `params.size + 1`（当作 AIC 惩罚项）。

**fixture 直接推翻**：

| fixture | `e(ic)` | 参数数 | `k+1` 预测 |
|---|---|---|---|
| `logit_sim` | 4.0 | 3 | 4 ✓ |
| **`poisson_sim`** | **1.0** | **2** | **3** ✗ |

**一个公式无法同时产生两者**，而「迭代次数」能解释两者。
值得注意的是：既有测试名就叫
`test_e_ic_is_the_aic_penalty_term_not_the_iteration_count`，
而**同文件另一个测试的失败信息写着「看起来像迭代次数，不是惩罚项」**——
它已经怀疑过正确答案，却被前一个测试压制了。

同时补上 `design.md:209` 要求但**完全缺失**的三个属性：

| 属性 | 缺失时的行为 |
|---|---|
| `converged` | `True`（闭式解无「收敛失败」可言）|
| `iterations` | 取自 `e(ic)`，无则 NaN |
| **`grad_norm`** | **NaN，而非 0.0** |

**返回 `0.0` 会让调用者误以为「精确收敛」。**

#### 本周期的三条方法论

1. **宏是能力清单**：`e(marginsok)` 可以逐项对照实现，反向发现缺失（4 轮命中 4 个命令）。
2. **宏本身也要双向核对**：实现了类型不等于登记了宏。
3. **「看起来一样」的 Stata 语义必须分开处理**：
   裸 `ereturn local` = 清除（键不存在）vs 显式赋空 = 键存在且为空。

---

### 8.20.91 第 240–249 轮：键的适用范围、间接设置路径，以及一次被真实输出推翻的修正

#### 第 240 轮：`N_cds`/`N_cdf` 只属于二值模型

`logit.sthlp:243-244` 定义「完全确定的成功/失败数」，`probit.sthlp` 同样文档化，
而 **`poisson.sthlp` 完全没有**。本包却给 `poisson` 也加了这两个键，实测给出
`N_cds = 179`——**计数模型里「全部成功的协变量模式」需要先指定目标计数值，无意义**。

用构造数据验证了正确语义：x=0 组全成功（5）、x=1 组全失败（5）、x=2 组混合（不计）
→ `N_cds=5`、`N_cdf=5`，且 `frac_154.ado:679` 的分离诊断恒等式
`e(N) == e(N_cds) + e(N_cdf)` 成立。

#### 第 241–242 轮：`e(title)` 的第二条设置路径

**第 233 轮的结论是错的**：当时在 `poisson.ado` 里 grep 不到 `ereturn local title`，
就判定它没有该宏。第 241 轮找到了真正的机制：

```stata
poisson.ado:241    */ nopreserve `mlopts' title(Poisson regression) `scopt' `robust' /*
nbreg.ado:352      */ title(`title') diparm(ln`parm', exp label("`parm'")) `negh' /*
nbreg.ado:71       local title  "Negative binomial regression"
```

**它们把标题当作 `title()` 选项传给 `ml`**，而 `poisson.sthlp:247` 明确文档化该键。

> **「在 ado 里 grep 不到」不等于「不存在」**——当值通过**被调用程序的选项**传递时。

第 242 轮补上 `nbreg`（间接路径）与 `ologit`/`oprobit`（直接路径，`ologit.ado:270`）。

#### 第 243–244 轮：`e(estat_cmd)` 与 `e(predict)`

| 命令 | `estat_cmd` | `e(predict)` |
|---|---|---|
| `regress` | `regress_estat` | **无**（`regress_p.ado` 不存在）|
| `logit` | `logit_estat` | `logit_p` |
| `probit` | `probit_estat` | `probit_p` |
| `poisson` | `poisson_estat` | **`poisso_p`** |
| `nbreg` | **无** | `nbreg_p` |
| `ologit` | **无** | `ologit_p` |
| `oprobit` | **无** | **`oprobi_p`** |

**两个程序名被截断到 Stata 的 8 字符限制**：`poisso_p`（非 `poisson_p`）、
`oprobi_p`（非 `oprobit_p`）。按 `command + "_p"` 构造会得到两个不存在的名字，
因此映射是**抄录而非推导**，并配有「查文件系统」的测试。

#### 第 245–246 轮：`e(vce)` / `e(vcetype)` 是两个不同的字符串

| 宏 | 含义 | 形式 |
|---|---|---|
| `e(vce)` | `vce()` 中的 vcetype | 小写 token（`ols`/`hc2`/`cluster`）|
| `e(vcetype)` | 标注 Std. err. 的标题 | 大写显示名（`Ols`/`Robust HC2`/`OIM`）|

**Stata 自己的代码依赖这个区别**（`arch.ado:1470`）：

```stata
if "`e(vcetype)'" == "Robust" { ... }
```

**第 246 轮的族间差异**：线性族的非稳健默认是 `ols`/`Ols`
（`regress.ado:68`），而 **ML 族是 `oim`/`OIM`**（`arch.ado:1474`、`arima.ado:741`：
`est local vcetype "OIM"`）。

**HC 形式是两个词**（`didregress.ado:477`：`ereturn local vcetype "Robust HC2"`），
`token.capitalize()` 会给出 `"Hc2"`，匹配不上任何东西。

#### 第 247 轮：移除 Stata 输出中不存在的两行

`style="stata"` 曾在标题后打印 `Model:` 和 `Covariance type:`。
真实捕获（`p10_outputs.txt:97-98`）显示 Stata 直接以 `Source | SS df MS` 开头：

```
      Source |      SS           df       MS
-------------+----------------------------------    Number of obs =        300
```

且 `regress.ado`/`regress.sthlp` 中搜不到这两个标签。已移除。

#### 第 248 轮：真实输出推翻了第 237 轮的修正

**第 237 轮依据 `design.md:209` 把表头改成大写，是错的。** 真实捕获显示：

| 文件 | 命令 | 表头 |
|---|---|---|
| `p8_outputs.txt:25` | `qreg` | `y \| Coefficient  Std. err.      t    P>|t|     [95% conf. interval]` |
| `p10_outputs.txt:106` | `nl` | 同上 |
| `p6p7_outputs.txt:18` | `logit` | `y \| Coefficient  Std. err.      z    P>|z|     [95% conf. interval]` |

| 项 | 第 237 轮（错） | 第 248 轮（对） |
|---|---|---|
| 第一列 | `Coef.` | **`Coefficient`** |
| 标准差 | `Std. Err.` | **`Std. err.`** |
| 置信区间 | `[95% Conf. Interval]` | **`[95% conf. interval]`** |

> **提交在仓库中的真实 Stata 捕获，优先于项目的设计文档。**
> 文档写的是**意图**，捕获记录的是 **Stata 实际打印的内容**。
> 当两者冲突，以捕获为准——这正是「以 stata 为准」的直接含义。

第 237 轮发现的**硬编码 `95%`** 问题是对的，本轮保留（`_ci_label` 仍随 `level` 变化）。

#### 本周期的三条方法论

1. **键的适用范围要双向核对**：既查「帮助文件有而实现无」，也查「实现有而帮助文件无」
   ——后者发现了 `poisson` 的 `N_cds`/`N_cdf` 与 `aic`/`bic`。
2. **grep 不到 ≠ 不存在**：宏可能通过被调用程序的 `选项()` 设置。
3. **真实捕获 > 设计文档 > 个人记忆**：第 237→248 轮的反转是这条层级的直接验证。

---

### 8.20.92 第 250–256 轮：「副本错误」与展示层的逐行核对

#### 第 250 轮：`xtpcse`/`xtgls` 的面板设计块

真实捕获（`p9_outputs.txt:496-506`）显示 `xtpcse` 在估计表**上方**印一个设计块
（`Group variable`/`Time variable`/`Panels`/`Obs per group`/`Estimated ...`）。
本包**一行都没有**，读者无法知道相关结构估计在哪些面板上。

**实现陷阱**：首次把设计块追加到 `lines`，测试却发现输出中没有它——
`_stata_summary` 有一个分支（ANOVA 表可用时）会**整体重建** `lines`，而
**这恰是 `xtpcse` 必经的路径**。修复需在两处都追加。

> **修改「累积型」输出时必须检查是否有重建该累加器的分支。**

#### 第 251 轮：`xtpcse` 的模型检验应为 `Wald chi2`

`p9_outputs.txt:504-506` 显示 `xtpcse` **同时**使用 `Wald chi2` 作模型检验、
`z`/`P>|z|` 作系数检验。渲染器按 `dist == "z"` 判断，而 `xtpcse` 的 `dist` 是 `"t"`，
于是印出 `F(1, 1398)`——一个该命令不报告的统计量。

**本包早已存有 `e(chi2type) = "Wald"`，只是展示层忽略了它。**
判据改为读取该宏，而非按分布推断。

#### 第 252 轮：两种设计块形状

第 250 轮的触发条件是**两个命令名的固定清单**，导致所有
`xtlogit`/`xtprobit`/`xtpoisson` 输出**完全没有面板结构**。
放宽后出现反向错误：给 `xtlogit` 套用 `xtpcse` 的完整块，印出三行 `.`。

真实捕获（`p9_outputs.txt:14-22`）显示 `xtlogit, fe` 的是**精简块**——
无 `Time variable:`、无 `Panels:`、无 `Estimated ...` 行。最终按命令族分两种形状。

> **把「某命令有某特性」写成命令名清单时，清单的边界就是错误的边界。**
> 放宽后必须重新确认：新纳入的命令是否真的需要同样的输出。

#### 第 253–254 轮：同一个错误结论的四个副本

**`e(ic)` 的含义被误读为「AIC 惩罚项 = 参数个数+1」**，而帮助文件明写
「number of iterations」（`logit.sthlp:257`、`xtlogit.sthlp:587`），
fixture 也数值印证（`poisson_sim`：`ic=1.0` 而 `k=2.0`）。

| 轮次 | 发现位置 |
|---|---|
| 238 | `discrete.py`（`logit`/`probit`/`poisson` 主路径）|
| 253 | `panel_nonlinear.py`（`xtlogit` 等）|
| **254** | **`limited.py` × 2**（`heckman, method(ml)`、`etregress, method(ml)`）|

**每处都有相同的错误注释、相同的 `+1` 公式，以及断言该错误的测试**
——即「错误结论 + 错误的测试预期」一起被复制。

第 254 轮改为**扫查**而非抽查，并加守卫测试防止扫查在数据上失效。

> **一个错误结论被复制到多个模块时，逐点修正是不够的。**

#### 第 253 轮（另一半）：设计块印出 `Group variable:   .`

`model_data` 存的是**编码后的组标签**，不含列名。修正为估计器在 `from_formula`
中记录 `_entity_name`/`_time_name`，写入 `e(ivar)`/`e(tvar)`。

#### 第 255 轮：`e(aic)` 在三个族中三种情形

| 估计器 | `e(aic)` | `estat_ic()` |
|---|---|---|
| `regress` | **有，但非 Stata 的**（高斯偏差尺度）| ✅ |
| `logit`/`probit`/`poisson` | **有，与 `estat ic` 一致** | ✅ |
| **`tobit`** | **完全没有**（`tobit.sthlp` 零命中）| ✅ |

> **「不一致」不自动等于「缺陷」。**
> 建立「哪一种读法才对」才是工作本身；**把差异抹平会毁掉信息**。

已用精确断言锁定 `regress` 的差值恰为 `N(1 + log 2π)`，证明是省略常数而非算错。

#### 本周期的「宏存在性」完整图景

「哪些宏存在」是**逐命令**问题，不存在安全默认值：

| 情形 | 实例 |
|---|---|
| 文档有、值可查 | `e(estat_cmd)` for `regress` |
| 文档有、值不可查 | `e(predict)` for `regress`（编译实现）|
| 文档有、值间接设置 | `e(title)` for `poisson`/`nbreg`（经 `ml` 的 `title()`）|
| 文档无、实现也不该有 | `e(aic)` for `tobit` |
| 文档无、实现却有 | `e(aic)`/`e(bic)` for `regress`（已标注非 Stata）|

#### 第 247–248 轮的回看：真实输出推翻设计文档

第 237 轮依 `design.md:209` 把表头改成大写；第 248 轮被三份真实捕获推翻：

```
p8_outputs.txt:25   (qreg)   y | Coefficient  Std. err.      t  P>|t|  [95% conf. interval]
p10_outputs.txt:106 (nl)     y | Coefficient  Std. err.      t  P>|t|  [95% conf. interval]
p6p7_outputs.txt:18 (logit)  y | Coefficient  Std. err.      z  P>|z|  [95% conf. interval]
```

> **证据层级：真实捕获 > 设计文档 > 个人记忆。**
> 文档记录意图，捕获记录 Stata 实际打印的内容。

第 237 轮发现的**硬编码 `95%`** 问题是对的，保留（`_ci_label` 随 `level` 变化）。

---

## 8.5 表格布局对齐的边界（第 18–46 轮）

从第 18 轮起，用 `scripts/scan_all_layouts.py` 逐命令比对 westata 渲染与真实 Stata
输出。它把「还有多少没对上」变成一个可复现的数字，而不是靠记忆。

**当前结果：59 个扫描规格，59 个匹配（100%）；注册表 94 条全部有归属（59 条被扫描，
35 条在 `scripts/audit_registry_coverage.py` 的 `NO_TABLE` 清单里逐条写明理由）。** 匹配分两档：
`MATCH` 指逐字符一致，`SKELETON` 指文本骨架一致、只有数字位数不同（不同随机样本的
必然结果）。判定带一道**自检验**：两侧表格线数量不同即判为 `SUSPECT` 而不计入匹配，
因为被过滤掉的一行会伪装成布局缺陷——这道守卫在第 42 轮抓到过一个静默回归
（`ologit`/`oprobit` 从 MATCH 退化为 7 条线）。

### 曾经的最后一项：`xtivreg, fe` 的 F 检验（第 57 轮已解决）

这一行**已渲染**，第 57 轮的完整过程见下（保留它是因为其中的错误很有代表性）。
`xtivreg, fe` 现已与 Stata 一致，扫描 59/59。

关键的两个错误都是同一类——**假设宏的含义，而不是观察它**：

1. `end1_tpb`（ado:375-377）是 `predict xb` **加** `predict u`。实测：`xb` 是原始水平的
   一阶段拟合加常数；`u` 是**组常数**（同一实体内每行相同）。两者之和 = 水平拟合 + 组效应。
2. 组内回归的因变量应为 `y`，不是内生变量。

修正后 `F_f = 1.102521616`，Stata 为 `1.1025216`。

### 覆盖审计（第 52 轮加入的守卫）

`scripts/audit_registry_coverage.py` 把「哪些命令根本没被检查过」变成可失败的检查：
注册表每一条必须**要么被扫描覆盖，要么在 `NO_TABLE` 里写明不覆盖的理由**，否则退出码 1。
`tests/test_registry_coverage_audit.py` 把它接进测试套件。

加这个守卫的原因很具体：本目标两次报出过**分母错误**的完成度——

1. 用 `_HEADER_BLOCK` 的 53 个条目当作「全部命令」，而注册表有 94 条；
2. `areg`/`xtabond`/`sureg` 三个 **E3** 命令压根没有扫描规格，静默走了通用渲染路径，
   从未被比对过。

审计首轮就抓出一条我自己写错的账：`estat hettest`（E3）没有归属。查证后确认它打印的是
三行**结果文本**（`H0: Constant variance` / `chi2(1) = 1.60` / `Prob > chi2 = 0.2063`），
不是回归表格，本就不属于表格布局范畴；数值另行动态核对过，与 Stata 一致。

同轮还纠正了一条描述错误：`jackknife` 一度被写成「未实现」，实际包里有一个 `Jackknife`
**辅助类**（接收已构建好的设计矩阵），与 Stata 的 `jackknife:` 前缀不是一回事。

### `xtivreg, fe` 的 F 检验：已定位到 Stata 源码的确切公式

第 46–47 轮把这一项查到底了。结论是**可实现但属于估计器工作，不是排版工作**：

* 点估计**完全一致**（`x2` .5068329 / `x1` .0908546 / `_cons` −.0373916 与 westata 逐位相同），
  所以这不是「模型算错了」；
* westata 的 `e(rss)` 是**原始残差平方和** 131.801，Stata 是 **100.26647**。
  实测口径：**组内去均值、只用斜率（去掉截距）** 的残差平方和——
  `100.26647116`，与 Stata 一致到小数点后 5 位；
* 而 `sqrt(100.26647/78) = 1.1337846` **精确等于** Stata 的 `e(sigma_e)`，口径由此确认；
* **第 47 轮拿到了 F 本身的真正公式**（`xtivreg.ado:653`）——它不是方差分量之比，
  而是**受限与不受限模型的残差平方和之差**：

  ```
  F_f = ((rss_rlm − rss_ulm) / e(rss)) * (df_wald / df_a)
  ```

  其中（`xtivreg.ado:437-458`）：
  * `rss_rlm` = 对**变换后变量** (`lhs_t`, `end1_tpb`) 做普通 `_regress` 的 `e(rss)`；
  * `rss_ulm` = 对同样变量做 `xtreg ..., fe` 的 `e(rss)`；
  * `end1_tpb` 是内生变量第一阶段的拟合值。

  由此反推：`rss_rlm` 应为 **127.194**，`rss_ulm` = 100.266。

* **卡住的地方（第 46 轮以为是 theta 变换，第 48 轮已否定）**：

  **第 48 轮的实测把这一条推进了，并且否定它**：

  * `tsrevar`/`fvrevar`（`xtivreg.ado:323-333`）只是因子变量展开，不改数值，所以
    `lhs_t` 就是 `y` 本身；
  * `end1_tpb = xbt + uit`（`xtivreg.ado:375-377`，`xb`/`u` 都取自 `xtreg, fe`），
    而 `xbt + uit` 恒等于**组内去均值的内生变量**——已用 `np.allclose` 验证；
  * 于是我在 Stata 里**直接复现这两个回归**并打印 `e(rss)`：

    ```
    rss_rlm = 95.414339     (regress y x1 x2hat)
    rss_ulm = 69.261849     (xtreg y x1 x2hat, fe)
    ```

    两个数与我按「水平/去均值」手算的完全一致——**说明变量变换那段我理解对了**。

  * 但把它们代进 `F_f` 公式**得不到** 1.1025216：用 `e(rss)=100.26647` 当分母得 **1.0708**，
    用 `rss_ulm` 当分母得 1.5501。反解 `df_wald` 得 **80.31**，不是整数。

  也就是说：**公式的文字我已经拿到，但其中某个量的口径仍有偏差**，而剩余候选不是靠
  读源码能唯一确定的。再往下就是在猜系数，**停手**。

* **第 53 轮的补充（把公式本身钉死了，也把卡点缩到一个数上）**：

  * `df_wald` 不是什么未知量。`xtivreg.ado:644-650` 写得很清楚：

    ```
    if "`small'" != ""  scalar `df_wald' = e(df_r)
    else                scalar `df_wald' = e(df_rz)
    ```

    实测 `e(df_rz) = 78`、`e(df_a) = 19`、`e(F_f) = 1.1025216`。我第 48 轮说
    「反解 df_wald 得 80.31，不是整数」是把别处的误差算到了 df_wald 头上——**那个结论
    作废**。

  * 于是公式里只剩 `e(rss)` 一个未知量。用实测的
    `rss_rlm = 95.414339`、`rss_ulm = 69.261849`、`df_rz = 78`、`df_a = 19` 反解：

    ```
    所需分母 = (rss_rlm − rss_ulm) × (78/19) / 1.1025216 = 97.379
    ```

    而任何我能在 Stata 里复现的估计之后，`e(rss)` 都是 **100.26647**（xtivreg 自己）
    或 **69.261849**（`xtreg ..., fe`），都不是 97.379。

  * 结论：`F_f` 在 653 行处读到的 `e(rss)` 是那一刻的**中间状态**，之后才被覆盖成
    100.26647；这个中间值无法通过事后调用来观察，也推不出唯一来源。
    公式到此为止是完整的，**只剩这一个数没有着落**——而它不能靠猜。

**已停手，没有拿近似值去冒充一个 F 统计量。**

* **第 54 轮：把缺口压到一个具体差额，并排除了所有外部可复现的路径**

  用 Stata 自己的 `e()` 值反解公式右边需要的差额：

  ```
  (rss_rlm − rss_ulm) = F_f × e(rss) × df_a / df_rz = 26.92786
  ```

  而我能复现的是 `95.414339 − 69.261849 = 26.15249`，**相差 0.775**。
  也就是说 `rss_rlm` 应为 **96.189709**，而任何组合都给出 95.414339：

  | 复现方式 | rss | 需要 |
  |---|---|---|
  | `regress y x1 x2hat` | 95.414339 | 96.189709 |
  | `regress y x1 z x2hat` | 95.414339 | 96.189709 |
  | `_regress y x2hat x1`（ado 原命令） | 95.414339 | 96.189709 |

  同时确认 653 行与 678 行**不在同一分支**（673 行有 `exit`）。当时据此推断「`e(rss)` 在
  那一刻应是 `rss_ulm`」——**第 55 轮证明这个推断是错的**，正确值是 100.26647（见下）。
  但同一段的另一个结论是对的：`rss_rlm` 用的 `end1_tpb` 与按 `xbt + uit` 从 `regress`
  重建的不同，构造在 375-377 行之外还有一步。

  **这一步没有继续追。** 从外部只能观察到 96.189709 这个目标值，构造方式无法反推唯一解；
  再往下就是枚举猜测。**仍未打印、近似或占位任何 F 值。**

* **第 55 轮：找到缺失的那一步，公式跑通到 1e-08，但实现精度只到 0.5%，于是回退**

  缺失的一步是 `end1_tpb` 的构造。ado:375-377 用的是 **`xtreg, fe`** 的 `predict xb` 与
  `predict u` 相加，而我此前用 `regress` 的预测值——两者不同：

  | 端到端复现（在 Stata 里照 ado 做） | `rss_rlm` |
  |---|---|
  | `regress x2 x1 z` → `predict xb` | 95.414339 ✗ |
  | **`xtreg x2 x1 z, fe` → `predict xb` + `predict u` → 相加** | **96.189709 ✓** |

  用后者，公式在 Stata 端**精确复现**：

  ```
  F = ((96.189709 − 69.261849) / 100.26647) × (78/19) = 1.102521626   (Stata 1.1025216)
  ```

  也就是说：**公式、`end1_tpb` 的构造、`e(rss)` 的取值（拟合系数下的组内残差
  100.26647）三者全部确认。**

  但在 westata 里按同样三步实现，只得到 **1.0973**（差 0.5%，p 值 0.3703 vs 0.3653）。
  差异来自 `xtreg, fe` 的 `predict xb` 所携带的**组截距**与我的重建不完全一致
  （实测 `endpb − x2` 均值为 0 但极差达 −2.79..3.17，说明它不是简单的去均值关系）。

  **已回退这份实现。** 本库里其他所有量都精确到 1e-8，混进一个差 0.5% 的统计量会让
  「匹配」这个词失去意义——宁可这一行继续缺着。

  留给后续：**只要把 `xtreg, fe` 的 `predict xb`/`predict u` 语义在 westata 内精确复刻，
  这条公式就是现成的。**

* **第 56 轮：把 `predict xb + predict u` 的性质测清楚了（仍未复刻）**

  在 Stata 里对 `xtreg x2 x1 z, fe` 分别 `predict xb` / `predict u` 后实测：

  | 量 | 值 |
  |---|---|
  | `mean(xb)` | **.09701**，恰好等于 `mean(x2)` |
  | `mean(u)` | 0 |
  | `_cons` | .12561 ≠ `mean(x2)` |
  | `xb + u` 减 `x2` 的**均值** | 0 |
  | `xb + u` 减 `x2` 的**标准差** | **.9103**（极差 −2.79..3.17）|

  也就是说 `xb + u` 与 `x2` 的差别**不是一个常数偏移，而是纯组内变异**：两者组均值一致，
  逐元素不同。逐一排除的关系：

  * `xb + u ≠ x2`
  * `xb + u ≠ x2 − 组效应`
  * `xb ≠ x2 − u`
  * 中心化后仍不等（`max|combo_c − x2_c| = 3.17`）

  结论：`predict u` 在 `xtreg, fe` 里不是"原始残差"，`predict xb` 也不等于"组内拟合”。
  要复刻必须先精确确定这两个 predict 的定义（可能涉及 `e(sample)` 与组内变换的交互），
  **本轮没有做到，因此没有代码改动**——再次说明：不拿 0.5% 的近似值凑数。

* 顺带确认：`sigma_u` 来自 `xtivreg.ado:806-838` 的 **Swamy–Arora** 公式
  `sig_u2 = (ubPub − (n−K)·sig_e2) / (N′ − tr)`，`tr` 是 `glsaccum` 的矩阵迹；
  用 `tr = K` 近似会得到**负方差**，所以这条也依赖 theta/GLS 那一套。

### 这一节曾经写错过的三件事（保留，作为记录）

前几轮把下面三项记为「功能缺失」，**三项都是误判**，后来都做成了：

| 当时的错误结论 | 实际情况 |
|---|---|
| `reghdfe` 没实现吸收固定效应 | 实现了，走的是 PyFixest 公式的 `\|` 块（`y ~ x1 \| unit`）。**是扫描器拿无固定效应的模型去比** |
| `etregress` 需要实现 athrho 参数化 | ML 路径早就有（`fit(method="ml")`）。**是扫描器拿 Stata 的默认（ML）比 westata 的默认（two-step）** |
| `tnbreg` 的 chi-bar-squared 检验「无数据可算」 | `e(ll)`/`e(ll_0)` 一直都在；`LR = 2(ll−ll₀)`、`p = 0.5·chi2.sf(LR,1)` 与 Stata 逐位吻合 |

共同教训：**「需要实现」不等于「无法实现」；而在断言「无法实现」之前，先确认两边比的是
同一个模型、同一个估计器。** 这三项里有两项的根本原因是扫描器，不是代码。

### 关于「匹配」的口径

`SKELETON` 与 `MATCH` 都不代表**数值正确**。`xtlogit, re` 的布局已对齐，但它的
`e(sigma_u)` 是 `.0037` 而 Stata 是 `.0000243`，`/lnsig2u` 是 `−11.18` 而 Stata 是
`−21.25`。**布局一致不等于结果正确**，两者必须分开报告。

### 另一个没有妥协的地方

第 44 轮我试过把 `Etregress.fit()` 的默认值从 `control` 改成 `ml`（与 Stata 的默认一致），
结果 **16 个测试失败**——两条路径写入的 `e()` 宏不同（`r2_p`、`cmd`、五种 margins）。
已撤回并在 docstring 里写明原因：**改变一个包默认跑哪个估计器是估计器决策，不是排版
改动**，不能凭一次布局比对就发布。

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

**共 93 条命令条目、3866 个测试。** 这是 Stata 估计类命令的一个可审计子集，
不是 Stata 的替代品。§1–§7 中所有 ❌ 的边界依然成立。

## 11. 如何理解这份登记册

- **westata 不试图替代 Stata。** 它是一个在可验证范围内、能证明数值一致的 Python 计量库。
- **每一条 ✅ 都有对应的自动化测试和证据记录。** 没有测试支撑的功能不算实现。
- **每一条 ❌ 都有替代方案或理由。** 不是遗漏，是取舍。
- **每一条 ⚠️ 都在文档中明确标注。** 不把"能跑"说成"已验证"。

如果你需要某个 ❌ 的功能，正确的做法通常是：用现有的 Python 库（matplotlib、lifelines、semopy），或继续用 Stata。westata 不会为了"覆盖所有功能"而牺牲它唯一的核心价值——**可审计的数值一致性**。

## 8.9 arch 的两次判断修正

| `arch` | **已改善** | 源码查出 `e(technique)=bhhh bfgs`、`e(vcetype)=OPG`；实现后 SE 改善 18–140 倍。剩余系数差 1.4e-4 属优化器停止点差异 |
