# westata 与 Stata 对齐工作汇总（第 1–100 轮）

本文档汇总针对 `westata_vs_Stata_差距评估报告.md` 所列出差距的处理结果，把每一项的
**验证证据**写清楚，使「已实现」「已验证」「已知差异」「明确未做」四者界限分明。

**原则**：一条结论只有在有 Stata 实测值或可重跑的闭式检验支撑时才算成立。本文档中每个
数字都写明了它来自运行的哪个命令、哪份数据。

**第 30–51 轮的重点**：报告把 39 项归为「B 类：布局适配（设计约束，非缺陷）」。但用户的
目标是「输出的数值与排版与 Stata 完全一致」，这个分类的前提不再成立。逐项实测后发现
其中多数**确实**与 Stata 不同，且有几处是真实数值错误，不是布局偏好。详见第七节。

---

## 一、总体结果

| 项 | 数量 |
|---|---|
| 全量测试 | **4398 passed, 3 skipped, 0 failed**（第 100 轮终检）|
| 完成项 | **88**（含第 96 轮 `vecrank lags(1)`）|
| 提交 | 115 个，`4d3f182` … `035a809`（本文档的更新提交另计）|
| 测试文件 | 257 个 |
| 新增验证测试文件（本次工作）| 30+ 个 |

**验证方式**：所有对照均使用 `pystata` 实时调用 StataNow 19.5 MP，同一份 `.dta`、同一模型，
比我方实现与 Stata 打印值的差。不是与旧版本输出对比——与自己的旧输出一致不能证明任何事。

**全量测试节奏**：用户指示「每完成 10 个修复后再进行全量确认」，后改为「每 20 轮修复后跑
一次全量测试」，中间只用 `-k` 定向测试。全量测试点：4208 → 4226 → 4233 → 4240 → 4269 →
4284 → 4323 → 4337 → 4343 → 4379 → 4380 → 4398 passed。

### 报告分类 vs 实测状态（第 63 轮核对）

报告 `westata_vs_Stata_案例清单.csv` 共 90 个案例：A 可直接替代 6 / B 布局适配 39 /
C 部分实现 8 / **D 真实缺陷 14** / **E 能力缺失 23**。**该分类是早期快照**，实测偏差如下：

| 类别 | 报告 | 实测 |
|---|---|---|
| **D 真实缺陷（14）** | 全部待修 | **全部已修复**，测试覆盖：`margins` 147、`classification` 26、`predict_defaults` 24、`hdfe` 66、`psmatch` 7、`didregress` 19、`teffects` 92 项全部通过 |
| **E 能力缺失（23）** | 全部缺失 | **21 项已实现**（含第 64–65 轮新增的 `tebalance`、`rcm`；另有 `estat dwatson`/`bgodfrey`、`wntestq`、`prais`、`xttest0`、`varwle`/`varlmar`/`varstable`/`varsoc`、`vecrank`/`veclmar`/`vecstable`、`rdrobust`、`predict,ps`），**仅 2 项缺失** |
| B 布局适配（39） | 设计约束，非缺陷 | 用户要求「排版完全一致」后**前提不成立**，多数实为差异，已逐项对齐 |

### v2 报告（第 25 轮）：`rdrobust covs()`——从源码取到完整机制，`gamma` 精确复现

上一轮定位到「`coef_covs` 是单一值，非两侧各自拟合」。本轮从 `rdrobust.ado:820-841` 读到了完整机制，并**精确复现了其中的 `gamma_p`**。

**机制（源码原样）**：

```
ZWZ_p = ZWZ_p_r + ZWZ_p_l        // 两侧矩矩阵相加
ZWY_p = ZWY_p_r + ZWY_p_l
gamma_p = cholinv(ZWZ_p) * ZWY_p   // 联合 WLS 系数
s_Y = (1 \ -gamma_p[,1])
tau_cl = s_Y' * beta_p[(deriv+1),]  // 截距减去 gamma×协变量系数
```

这是 **Frisch-Waugh 部分化 + 两侧合并**：先在每侧把多项式项从协变量中剔出，再把两侧的矩相加，最后求解。**协变量系数正是由两侧联合估计**，所以只有一个值。

**已精确复现**：`gamma = -94.34797652`，与 Stata 的 `e(coef_covs)` **完全相同**；两侧有效样本 `323 / 360` 也完全一致。

**未复现 `tau`**：按 `s_Y' * beta_p[(deriv+1),]` 计算得 6720.51，Stata 为 7.4906。`beta_p` 的行列含义（`(deriv+1)` 行、多项式块与协变量块拼接后的列）与我的实现有出入，需要按 Mata 的 `D = D, eZ` 拼接顺序逐列核对。

**下次的起点**：不要重新推导机制（已取得且 `gamma` 已验证），而是**核对 `beta_p` 的行列布局**——特别是 `beta_p[(deriv+1),]` 在一阶多项式下取的是第 2 行，且该行的列覆盖所有多项式与协变量项。

未提交代码（探针脚本已删除），工作区干净。

### v2 报告（第 24 轮）：`rdrobust covs()`——取得参照与关键约束，未实现

报告 §7.3 指出 `rdrobust` 无 `covs()`，核实**属实**。本轮取得参照并定位了实现的核心障碍。

参照（`rdrobust_senate`，h=17.75 b=28.03，协变量为标准化后的 `margin²`）：

| 情形 | `tau` | 有效样本 |
|---|---|---|
| 无 covs | 7.414370281708 | — |
| `covs(cov1)` | **7.490617475565** | 左 360 / 右 323 |

**关键约束（由 `e(coef_covs)` 揭示）**：Stata 报告的协变量系数是**单一值 `-94.34797652`**，而两侧独立拟合并不能复现——我的实现给出右侧 `-58.14`、左侧 `-128.90`。**这说明 `covs()` 是两侧联合估计的单一调整系数，不是每侧各自拟合**。

**样本数已吻合**（`n_r=323`、`n_l=360` 与 Stata 一致），无 covs 的情形也精确复现（`7.414370281709` vs `7.414370281708`），所以问题只在协变量的联合处理方式。

**未提交代码**。下次的正确起点是查清 `rdrobust` 如何联合两侧求单一 `coef_covs`（可能是把两侧样本堆叠后估计，或对协变量做全局中心化），而不是继续在单侧拟合里调参。

### v2 报告（第 20–21 轮）：命名输出用「追加」而非「替换」

第 19 轮证实：把 `VECM.beta` 换成 DataFrame 会破坏 `result.beta[0, 0]`（已有测试报 `KeyError: (0, 0)`）。第 20 轮改用**追加新属性**，两种形式并存：

| 原属性（不变）| 新增命名形式 |
|---|---|
| `beta` / `alpha`（ndarray）| `beta_labelled` / `alpha_labelled`（DataFrame）|
| `resid` / `fittedvalues`（ndarray）| `resid_series` / `fitted_series`（Series）|

每个新属性都与原属性做 `np.allclose` 守卫——**同一拟合的两个视图若不一致，比只有一个更糟**。

**方法学收获**：报告说「应改为 X」时，未必只能「替换成 X」。「更符合约定」与「不破坏现有调用」并非二选一，追加属性同时满足两者。这与本任务中多次「假设即实测」的错误形成对照——**先测量代价（第 19 轮），再选设计（第 20 轮）**。

**已知限制（如实记录）**：`resid_series` 的 index 是拟合器自己的行号（从 0 开始），调用方自定义的 DataFrame index 不会保留，因为拟合器内部已 `reset_index`。数值正确，已在测试模块中写明。

**全量验证**：**4486 passed, 3 skipped, 0 failed**（第 21 轮）。

### v2 报告（第 18–19 轮）：statsmodels 生态接口

**第 18 轮已修复（3 类，均对照 statsmodels 本身验证）**：

| 接口 | 对照结果 |
|---|---|
| `res.t_test('x1=0')` | `7.1974807336` = statsmodels `7.1974807336` |
| `res.f_test('x1=0, x2=0')` | `30.9297891742` = statsmodels `30.9297891742` |
| `res.get_influence()` | 杠杆值与 Cook 距离逐位相同 |
| `res.model` | 提供 `endog`/`exog`/`exog_names`/`fit()` |

**刻意排除的三项**：`cov_params()`（用户要求）、`resid`/`fittedvalues`（兼容性决策）、`anova_lm`（需 statsmodels 内部 `model_spec`，非公开接口）。

**第 19 轮：`VECM.beta`/`alpha` 加标签——已实施、已测试、已回滚。**

改动本身可行：`raw.names` 提供变量名，返回带 `ec1..ecR` 列的 DataFrame，`np.asarray()` 仍可用。但**一条已有测试立即失败**：`KeyError: (0, 0)`——测试用**位置索引**访问，DataFrame 的下标语义与 ndarray 不同。这证明该改动会破坏下游调用。

**回滚后 20 个测试全部通过。** 这属于**兼容性决策而非缺陷修复**：改为命名 DataFrame 更符合statsmodels 约定，但会让 `result.beta[0, 0]` 这类写法失效。**需用户裁决。**

### v2 报告（第 17 轮）：全量验证与状态汇总

**4466 passed, 3 skipped, 0 failed**——第 1–16 轮的全部改动无回归。

| 项目 | 类型 | 状态 |
|---|---|---|
| P0-1 面板聚类 SE | 缺陷 | ✅ `(n,1)` 形态修复，四种输入形式均精确 |
| P0-2 `ovtest, rhs` 哑变量 | 缺陷 | ✅ `F(11,741)=5.21` → `F(9,743)=2.03` |
| P0-3 `margins atmeans` fweight | 缺陷 | ✅ 五个均值与两个 MEM 逐字符一致 |
| P0-4 `dfgls` tau | 缺陷 | ⛔ 需 `dfgls.ado`（内置命令，两路径均无）|
| P1 `vecrank` 缺最后一阶 | 缺陷 | ✅ 经核实**非缺陷**（Stata 本身留空）|
| P1 `vecrank trend(trend)` | 选项 | ⛔ 需 `vecrank.ado` 内部常数 |
| P1 `psmatch caliper()` | 选项 | ✅ 并纠正了我自己的误判（语法用错，非环境问题）|
| P1 `psmatch nn(k)` | 缺陷 | ✅ 容差 1e-6 → 1e-9，`nn(1)`–`nn(5)` 全部精确 |
| P1 `prais, corc` | 选项 | ✅ 系数 8 位、`rho` 1e-6 内一致 |
| P1 `margins dydx('*')` | 选项 | ✅ 逐字符一致 |
| P1 第一阶段回归表 | 输出缺失 | ✅ `result.first_stage()`，14/14 行一致 |
| P1 `esttab` / `mtitle` | 输出缺失 | ✅ `ws.etable_as_stata()`，12/12 行一致 |
| P1 `xtreg, mle` | 命令缺失 | ✅ llf 精确到 9 位有效数字 |
| §7.3 RD 带宽选择器 | 工程任务 | ⚠ 取得完整算法与两项精确辅助，**未接入**（差 6 倍，不发布）|
| §7.3 AI-robust 方差 | 工程任务 | ⛔ 需 `lrdrobust.mlib` 中的 `rdrobust_vce` |
| `vecrank` 3 变量统计量 | 缺陷 | ⚠ QR+SVD 改善了病态，**仍差 15%** |
| `synth` / `synth2` | 整章 | ⛔ 6 类途径已穷尽 |

**11 项修复，3 项需编译源码，2 项部分完成并如实标注。**

### v2 报告（第 14 轮）：`vecrank` 三变量统计量——尝试自建 Johansen，未成功

第 13 轮修好临界值后，3 变量的**统计量**仍错（`trace nan / 41.6130 / 5.2920` vs Stata `35.2070 / 3.6149 / 0.0000`，最大特征值 `3.63428` > 1）。本轮尝试用 **QR + Cholesky 白化**绕开 statsmodels 的病态求逆。

**决定性检查（本应在最开始就跑）：** 自建构造对 **2 变量**——那个已知精确的情形——给出 `trace=30.2394`，而正确是 `183.0918`。**差 6 倍**。已知良好的情形上差 6 倍，说明构造从根本上错了，不是调参能救的。而直接复刻 `_johansen_lags_one` 得到 `172.5034 / 1.2744`，证明参考实现是对的、推广版是错的。

**差异在行对齐**：`_johansen_lags_one` 中 `lagged = values[:-1]` 与 `differences = values[1:] - values[:-1]` 天然同长；加入滞后差分后需要对齐三样东西（行索引、常数项、滞后块），每次修好一个就破坏另一个，`lags(1)` 一度从精确退化为 `155.4252`。

**未提交任何代码**，工作区干净。

**下次的正确路径**：不要从零写通用版，而是**从 `_johansen_lags_one` 出发做最小扩展**，每扩一步都用 `lags(1)` 的精确结果作回归测试，一偏离立即回退。本轮失败的直接原因是**先写通用版再回头验证**，正确顺序是边扩边验。

### v2 报告（第 11 轮）：`rdrobust` 自动带宽——取得完整算法，但未复现

**重大发现**：`rdbwselect.ado`（1331 行）与 `rdrobust.ado` 在**用户 ado 路径**下**可读**。此前多轮把 `rdrobust` 归为「二进制边界」是不够准确的——**选择器的控制流与组合公式全部可读**，只有内层 `rdrobust_bw()`（8580 字节，19–21 个参数）在 `l/lrdrobust.mlib` 中。

已从源码取得 RD（`mserd`，默认）的组合公式（`rdbwselect.ado:557`）：

```
h = ( (C_h_l[1] + C_h_r[1]) /
      ( (C_h_r[2] - C_h_l[2])^2 + regul*(C_h_r[3] + C_h_l[3]) ) ) ^ C_h_l[4]
```

**关键细节**：分母里 `C[2]` 是**减号**——RD 估计量取两侧之差；`msesum` 变体才是加号。用错符号会得到一个「看起来合理」但属于另一个估计量的带宽。

**三层嵌套结构**（此前未意识到）：`d_bw` → `b_bw` → `h_bw`，每一步的 `h_V` pilot 都取自上一步的结果。

**当前进展**：`estimators/rdbandwidth.py` 已实现框架，`b` 得 29.9098 对 Stata 的28.0281（**差 6.7%**），`h` 仍差 6.6 倍。差距在于 `C_h_*` 需要以 `b_bw_d` 作 pilot 重新计算，而 `C` 分量本身由编译库返回。

**结论**：**未对齐，模块未接入公开 API**（`rdrobust` 默认调用仍抛 `UnsupportedError`）。这比上一轮的「整块不可达」前进了一步——现在知道差在哪一层。

### v2 报告（第 10 轮）：`rdrobust` 自动带宽——现状核实与可行性评估

报告 §7.3(2) 说 `rdrobust` 缺 MSE 最优带宽选择器，且「这是一个独立、可分离的工程任务：实现 IK/CCT 的 MSE-optimal 与 CER bands，就能一次点亮 7 条命令」。**核实后报告完全准确**：

- 默认调用确实抛 `UnsupportedError`（要求显式 `bandwidth=`）
- `h=17` 时 `tau = 7.4303798198`，与报告所记 `7.4303798` 一致——**估计器本身是对的**

从 `rdrobust_senate.dta` 取得的参照（n=1390）：

| 项 | Stata |
|---|---|
| `h` | 17.7543972212 |
| `b` | 28.0280871512 |
| `tau` | 7.4141308020 |
| `se` | 1.4587160238 |

**重要限定**：给定 Stata 的 `h`/`b` 后，`tau` 逐位精确，但 **`se` 仍为 NaN**（`rdrobust_vce` 在编译的 `lrdrobust.mlib` 中，已在早前轮次确认为二进制边界）。因此实现带宽选择器能点亮的是**点估计**，推断列仍缺——报告「一次点亮 7 条命令」应理解为「7 条命令可以运行并给出正确点估计」。

**本轮未实现**：CCT 带宽需要边界方差、三阶导数、核常数等多个中间量，且要与 `rdrobust` 内部常数逐项对齐；现有 `estimators/rdrobust.py` 仅 242 行，核与局部多项式可复用，但估计器部分需新增约 150 行。这是一个完整的专项轮次，已排为下一轮。

### v2 报告（第 9 轮）：`xtreg, mle` 已实现

报告 C12-05 列为缺失命令（E 类）。它与 `xtreg, re` 是**不同的估计量**，不是同一实现的选项：

| | `re` | `mle` |
|---|---|---|
| `ltlan` | .565591486884 | .564328798388 |
| `ltwlab` | .144184445858 | .140897433220 |
| `sigma_u` | .133248446685 | **.214109360143** |

**`sigma_u` 差 60%**，所以任何对 GLS 输出的微调都不可能冒充 mle。新增 `RandomEffectsMLE` 类，用 statsmodels 的 `MixedLM` 拟合。

优化器选 **Nelder-Mead** 而非默认：llf 得 332.8973943565 对 Stata 的 332.8973943543（**9 位有效数字**），`ltlan` 差 1.6e-7，而默认优化器差 9.8e-7。

### v2 报告（第 8 轮）：`psmatch nn(3)` 的缺口已关闭——容差过宽，不是 tie 规则不同

第 7 轮把这个记为「已知限制：`nneighbor(3)` 得 2412.8207，Stata 为 2397.636，是早于本轮的 tie 处理差异」。**根本不是规则差异，而是容差太宽。**

先取 Stata 的 `nn(1)`–`nn(5)`，五个里四个已经精确，这个形状本身排除了规则问题：

| n | Stata | westata | 判定 |
|---|---|---|---|
| 1 | 2639.8646 | 2639.8646 | 精确 |
| 2 | 2491.8602 | 2491.8602 | 精确 |
| **3** | **2397.636** | **2412.8207** | **错** |
| 4 | 2330.1077 | 2330.1077 | 精确 |
| 5 | 2321.0271 | 2321.0271 | 精确 |

于是不再看规则改看**并列的个案**：处理组 2 的第 3、4 近邻距离为 `0.0011575908` 与 `0.00115824`——相差 6e-7，是**两个不同的对照**。`1e-6` 的默认容差把它们并成了并列，于是报了 4 个匹配而 Stata 用 3 个。

容差扫描（对五个参照）：`1e-6` → 4/5，`1e-9` / `1e-12` / `0` → **5/5**。默认改为 `1e-9`。`nnmatch` 本来就用 `1e-10`，所以从未暴露此问题。

### v2 报告（第 7 轮）：`teffects psmatch caliper()` 已实现，并纠正我自己的误判

第 2 轮我把 `psmatch` 的参照记为**不可获取**（一律 `r(103)`，与协变量数量无关），并归因为环境问题。**这个判断是错的**，而错误信息里写得很清楚：

```
The outcome-model is misspecified. An outline of the syntax is
teffects psmatch (outcome_variable) (treatment_variable varlist)
```

`psmatch` 的协变量在**处理方程**一侧，与 `nnmatch` 相反。我写的是 `(re78 age) (treat)`，正确的是 `(re78) (treat age)`。**信息摆了一整轮，我把它读成了环境的限制而不是命令的错误。**

修好后 caliper 与 Stata 一致（`nsw_dw`）：

| | ATET |
|---|---|
| 无 caliper | 2639.8646 |
| `caliper(0.1)` | 2639.8646（该阈值未排除任何对照，故应完全相同）|

**已知限制（本轮未动）**：`nneighbor(3)` 得 2412.8207，Stata 为 2397.636。这是 `n>1` 时的 tie 处理差异，早于本轮工作（`n=1` 是精确的），已记录而非让一个通过的 caliper 测试暗示匹配器已完全对齐。

### v2 报告（第 6 轮）：`esttab` 文本输出已实现

报告 C20-01 说 `etable` 缺 `esttab` 的编号行、`mtitle` 行、统计量脚注。`model_names` 早已存在，缺的是**渲染出的表格本身**——`etable` 返回 DataFrame（对程序正确，对比较无用）。

新增 `ws.etable_as_stata(...)` 与既有 `etable(...)` **并存**：DataFrame 仍是编程接口，新增的是打印接口。`cobb_douglas` 上 **12 行全部逐字符一致**。

四条排版规则，均为实测所得，每条都花了一轮：

1. **星号挂在数字之外**：`0.377***` 比 `0.377` 向右多占 2 列；
2. **常数项排在最后**，尽管 patsy 叫它 `Intercept` 且放在首位；
3. **整数部分占 2 列时自动少一位小数**（`77.58` 对 `0.377`），这才使单元格等宽；固定 3 位会打出 `77.579` 并让整行右移；
4. **标准误从固定列（17 与 33）开始**，并不右对齐到系数列尾——`(9.011)` 结束于 24，而 `(0.0352)` 结束于 25。

脚注明示 `esttab` 默认的显著性方案（`* 5%`、`** 1%`、`*** 0.1%`），**不是**某些实现用的 10/5/1；弄错会让每个星号都标错含义。

### v2 报告（第 5 轮）：`ivregress ..., first` 第一阶段回归表已实现

报告 C10-01 说缺第一阶段回归表——**准确**。此前只有 `estat firststage`（**诊断**：R²、偏 R²、工具变量 F），没有 `, first`（**方程**：系数、标准误、t、区间）。Stata 两者都有且不可互相替代，所以新增 `result.first_stage()` 与既有的 `result.estat_firststage()` 并列。

`grilic` 上 `ivregress 2sls lnw s expr (tenure = rns smsa), first` 的 **14 行全部逐字符一致**（`F(4, 753) = 11.15`、`R-squared 0.0559`，五个系数行含 `_cons`）。

三个踩过的坑：

1. 数字格式必须复用 `results.summary` 中**已验证**的 `%9.0g` 实现。自写的 `.7g` 把 `.0006668274` 打了出来——**7 位有效数字**与 **7 位小数**不是一回事。
2. 统计块与系数表的排版**不同**：每个标签从第 56 列开始、`=` 落在第 70 列，且 R-squared/Adj R-squared/Root MSE 用**4 位小数并保留前导零**（`0.0559`），与下方单元格的 `.0559` 风格相反。
3. 常数项经 patsy 传进来叫 `Intercept`，Stata 打印 `_cons`。

数据取自 `result.iv_structure`（`IV2SLS.fit` 早已为 `estat firststage` 挂上），因此两个输出**不可能对设计矩阵产生分歧**。

### v2 报告（第 4 轮）：`margins, dydx(*)` 已实现

报告 P1 / C11-03 说缺通配形式——**准确**。`dydx=['*']` 原先抛 `KeyError: column '*' not in [...]`，该报错完全没有提示通配符是存在的。

Stata 把 `*` 展开为**全部协变量**（除常数项，常数没有边际效应）。实测 `margins, dydx(*)` 在 `regress y k l` 后打印 `dy/dx wrt:  k l` 与两行数据，现已逐字符一致：

```
dy/dx wrt:  k l
k |   .1585962   .0416823     3.80   0.001     .0719131    .2452793
l |   .9166087   .1495604     6.13   0.000     .6055807   1.2276366
```

混合列表如 `['*', 'k']` 会把 k 列两次，展开时去重，与 Stata 每个协变量只列一次一致。

### v2 报告（第 3 轮）：`prais ..., corc` 已实现

报告 P1 / C08-05 说 `prais` 缺 `corc` 选项——**准确**。

`corc` 不是外观开关：Cochrane-Orcutt 变换**丢弃**首观测，而 Prais-Winsten 把它乘以 `sqrt(1 - rho^2)`（`prais.ado:6` 的 `syntax ... noDW CORC`，`:255` 记录 `e(tranmeth)` 为 `corc` 或 `prais`）。这会同时改变全部系数**和** `rho` 本身。

| | 系数（`k`, `l`）| `rho` |
|---|---|---|
| `prais` Stata | `.15651652`, `.93948861` | `.12543124` |
| `prais` westata | `.15651651`, `.93948867` | `.12543150` |
| `corc` Stata | `.15687809`, `.92974899` | `.11463695` |
| `corc` westata | `.15687809`, `.92974901` | `.11463704` |

斜率一致到 8 位有效数字，常数项与 `rho` 到约 `1e-6`（迭代容差，非结构差异）。默认 Prais-Winsten 路径未改动，并有独立测试复核其四个数值，防止将来重构共享循环时被静默改动。

### v2 报告（第 2 轮）：vecrank 的「缺最后一阶」不成立，`trend(trend)` 未实现

| 报告说法 | 实测 | 处置 |
|---|---|---|
| `vecrank` 临界值表「缺最后一阶」| **不成立**：`trend(constant)` 下 rank=2 的 Trace 与临界值在 **Stata 输出中本来就是空的**（该阶没有可检验的 rank）。westata 输出 `183.0918 / 7.7811`、LL、特征值、参数量与 Stata 逐位一致 | 无需修改 |
| `trend(trend)` 未支持 | **准确**：目前 `UnsupportedError` | 未实现 |

**`trend(trend)` 的真实障碍**：statsmodels 的 `det_order=1` 给出 `183.5984 / 7.6836`，与 Stata 的 `183.6114 / 7.6369` 差 0.007% / 0.6%，**结构接近但不等价**。statsmodels 的`detrend()` 对**整条序列**去趋势（`endog = detrend(endog, det_order)`），而 Stata 的 `trend(trend)` 把趋势放在协整关系内部。手工按后者构造得到 `51.39`，与 183.61 差 3.6 倍，说明样本/滞后差分的构造也不同。

**下次的前置条件**：需要 `vecrank.ado` 或 Stata 手册中 `trend()` 的精确定义（哪一项进 Z0、哪一项进 Z1、滞后差分阶数、`N` 是否仍为 164）。盲调参已证明无效。

### v2 报告（第 2 轮）：`teffects psmatch` 参照无法获取

报告中 `teffects psmatch` 缺 `caliper` 与 `predict, ps` 属实，但**本轮无法取得参照**：

同一数据集上 `teffects nnmatch (re78 age) (treat), atet` 正常，而 `teffects psmatch` 无论用 1 个、2 个、4 个协变量、无论加不加 `caliper()`，一律返回 `r(103) too many variables specified`。**这与协变量数量无关**，是本环境中该命令的特定失败。

**下次的前置条件**：先用最小命令（`teffects psmatch (y x) (t)`）确认命令本身可用，再引入数据集；或在另一个数据集（如 `cattaneo2`）上取参照。

### v2 报告（第 1 轮）：三个 P0 已修，dfgls 未修

新报告 `westata_vs_Stata_差距评估报告_v2.md` 质量明显高于 v1：给出了文件、行号、根因与修法，复现后确认**大部分准确**。

| P0 | 报告说法 | 实测 | 处置 |
|---|---|---|---|
| 面板聚类 SE「全路径报错」| 全路径不可用 | **只有 (n,1) 形态被拒**；1-D 与 robust 本来就对且与 Stata 逐位一致 | **已修** |
| `estat ovtest, rhs` 哑变量 | F 与自由度都错 | **准确**：`F(11,741)=5.21` → `F(9,743)=2.03` | **已修** |
| `margins atmeans` fweight | 均值未加权、MEM 差 18% | **完全准确**：`child` 均值 `.3333` → `.0495229` | **已修** |
| `dfgls` 输出系数而非 tau | 「没除以标准误」| 代码**已经**除以了标准误 | **未修** |

**dfgls 的真实障碍**：出错不在「少除一个 SE」。实测 `nobs=51` 与 Stata 一致，但 tau **符号相反**（`+0.806` vs `-1.284`）。在 `cbar`、`nbar`、准差分去均值方式、因变量定义上做了网格搜索，最接近者仅到 `-0.1188`，差距远大于参数选择能解释的范围，说明**算法结构**（GLS 去趋势的对象）与 `dfgls.ado` 不同。

**继续排查的前置条件**：需要读到 `dfgls.ado` 本身；它在本环境的两个 ado 路径下均未找到，`dfgls` 是内置命令。在拿到其实现前，调参数没有意义。

### A 类（6 项「可直接替代」）在第 93–94 轮逐项实测确认

报告把这 6 项标为 A 类，但第 83 轮发现其中 C04-01 实际多一行标题，所以整类都需要复查：

| 项 | 实测结果 |
|---|---|
| C04-01 一元 OLS | **第 83 轮已修复**（多了「Linear regression」标题行，根因是 ANOVA 分支重建了 `lines`）|
| C04-02 `regress, noc` | **12/12 行逐字符一致** |
| C05-01 多元 OLS | 经 C04-02 / C06-01 同一条渲染路径确认 |
| C05-03 `if` 子集 | **4/4 行逐字符一致**；`subset=` 只接受**索引标签或布尔掩码**，不接受 Stata 的 `if` 表达式字符串（`subset='k > 200'` 会 `KeyError`），需写 `subset=(df['k'] > 200)` |
| C06-01 `regress, robust` | **13/13 行逐字符一致** |
| C08-04 `newey, lag(3)` | **12/12 行逐字符一致**，含 `Newey–West` 的 **en-dash（–）** 而非连字符 |

**A 类 6 项现全部验证通过**，其中 C05-03 的差异属于用法层面（符合目标「使用方法上不要求一致」）。

### B 类核实结果（第 85–86 轮）

报告把 B 类（39 项）整体标为「布局适配（设计约束，非缺陷）」并注明「排版不同」。逐项实测后：

**已与 Stata 逐字符一致的（报告仍标注「布局不同」，属过时记录）：**

| 项 | 核对结果 |
|---|---|
| C09-03 `estat vif` | VIF/1-VIF 数值与列宽完全一致 |
| C09-02 `estat ovtest` | 标题、`Omitted:` 行、`H0:` 行、`F(3, 18) =   3.54` 全部一致 |
| C07-01 `estat hettest` | `Variable: Fitted values of y`、`chi2(1) =   6.62` 一致 |
| C07-03 `estat imtest, white` | 数值与 Cameron-Trivedi 分解表逐行一致 |
| C09-01 `estat ic` | **第 85 轮修复**（4 位小数 + 全宽首尾规则线），8/8 行一致 |
| C04-03 `ereturn list` | **第 72 轮实现** |

**仍未对齐的一项**：C05-07 `lincom` 的表头 —— Stata 是 `           y | Coefficient`（`|` 在第 13 列），
westata 是 `y           | Coefficient`（响应名左对齐）。第 86 轮尝试修正时反复差 1 个字符，
**已回滚**（`_place` 的列语义是按「结束列」定位，与左字段宽度耦合，需要重写该表头的构造方式而非调参数）。

### 普通 OLS 多一行标题：已修复（第 83 轮）

报告 C04-01 把 OLS 归为 A 类「可直接替代」，括注「westata 多一行标题」。实测确认差异为真：
Stata 的 `regress y k` 第一行就是 `Source | SS df MS`，`Linear regression` 只是 `e(title)`，
**存而不印**；只有稳健版才把它打印在双列块的首行。

**第 81–82 轮未能定位，第 83 轮用「枚举所有 `lines` 赋值点」的办法一次找到：**

`_stata_summary`（1649–4070 行，仅 1 个返回点）里有 8 处 `lines` 赋值，其中：

```
2246: lines = [banner] if banner else []     <- 真正的原因
```

该行在 **ANOVA 分支里重建了整个 `lines` 列表**，用 `[banner]` 把标题硬编码加回去，**覆盖了**
1815 行 `lines.append(banner)` 处做的任何判断。这正是前两轮修补 1815 行毫无效果的原因。

**教训**：在一个「会重建列表」的渲染函数里，只改某个 `append` 点无效——必须枚举所有重建点。
前两轮读代码得出的假设都是错的，第三轮改为**机械地列出所有 `lines = ...`** 立刻命中。

��复后 `regress y k` 与 Stata 的输出 **13/13 行逐字符一致**；稳健版仍保留标题。
一条断言 `regress` 必须有 banner 的旧测试被改正（它把 bug 钉死了）。

### C 类（部分实现，8 项）在第 72–79 轮全部关闭

| 项 | 内容 | 处理 |
|---|---|---|
| C04-03 | `ereturn list` | **第 72 轮**：命令此前完全不存在，已实现四段式输出（scalars/macros/matrices/functions），5 处格式细节均实测 |
| C13-05 | `vargranger` | **第 73–74 轮**：先修好了 `granger_causality` 的崩溃 bug（`list("unem")` 拆成字符），再实现表格，**10/10 行逐字符一致** |
| C11-01 | `logit ..., or` | **第 75 轮**：exp 变换 + `Odds ratio` 列名 + `Note: _cons estimates baseline odds.`，6/6 行精确 |
| C11-04 | `margins, atmeans` | **第 76 轮**：效应值本就精确，补上 `Conditional marginal effects` 标题与 `At:` 块 |
| C11-05 | `margins, eyex()` | **第 77 轮**：数值错 0.2%，根因是 `mean(dy)·mean(f)` 应为 `mean(dy·f)`；点估计已精确，**标准误仍偏** |
| C13-01 | VAR 系数表 | **第 78 轮**：表格与数值本就精确，修的是滞后标签 `L.` 应为 `L1.`；连带修好 `varwle` 的同源查找 |
| C13-06 | IRF | 数值表 `irf_table` 已在第 50 轮实现并逐行验证；`irf graph` 属绘图，**不在目标范围** |
| C15-03 | `teffects nnmatch` + `caliper` | **第 79 轮**：整个估计器此前不存在（不只是缺 `caliper`），已实现，两个 ATET 精确 |

**第 79 轮顺带解开了一个长期困惑**：报告把「ATET 2277.021 vs 2639.865」记在 `teffects psmatch`
名下。2277.0212 其实是**马氏距离匹配且只取第一个并列对照**的结果——一个 nnmatch 数值被登记成了
psmatch 缺陷。第 52 轮修好 psmatch 后始终无法解释 2277，原因就在这里。

**由此得到的通用教训**：报告的多条记录存在**描述与实质不符**（C15-03 说「缺一个参数」实为整个
估计器不存在；C11-05 说「返回类型不对」实为数值错 0.2%；C13-01 说「没有表」实为标签错一个字符）。
按报告描述去查，往往查不到问题；必须以实测输出为准。

**E 类剩余的 2 项**：`synth`/`synth2`（结构已还原、算法参数在二进制 mlib 中）、`xtserial`（已确认用户 ado 路径下不存在）。

第 64–65 轮新增两项，并各自记录了局限：

- **`tebalance summarize`**：Raw 列 8/8 精确，Matched 列 7/8 精确；`age` 因 ties 权重分配不同而有残差，守卫测试对其用宽松容差以免静默通过。
- **`rcm`**：给定 Stata 选中的预测变量时 ATT 与全部 12 期效应逐位一致（`-44.12853199` vs
  `-44.12853209`）；但**自动模型搜索无法复现**。第 98 轮取得了关键的新数字：

  ```
  e(K_preds_all) = 116    候选池大小
  e(K_preds_sel) =  3     最终选中个数
  ```

  此前我按 `38 州 × 3 变量族 = 114` 构造候选池，**比 Stata 少 2 个**。前向选择对候选池敏感，
  这 2 个变量的缺口足以改变选出的模型——这解释了为什么穷举最优（RSS 20.73）与 Stata 的选择
  （RSS 256.39）差异如此之大：后者不是穷举最优，而是在一个**不同**的 116 维池上做贪心搜索。

  第 99 轮进一步拿到两条决定性的事实，它们说明**我此前的整体理解是错的**：

  ```
  scope(1 1) 时 e(predvar_sel) = cigsale·8      <- 前向选择第一步就选中 cigsale·8
  e(info) 第一行（K=1）R-squared = .99978306
  ```

  而用自己的框架拟合 `cigsale·8` 单独作为预测变量，得到 **R² = 0.835**，与 Stata 的
  .99978 相差极大。**说明 Stata 的因变量/自变量构造与我假设的完全不同**——不是「候选池少 2 个」
  这种量级的问题，而是回归设定本身不同（`rcm` 的 `varlist` 把 `cigsale` 同时用作因变量，
  因而 `cigsale` 各州的滞后值到底进不进池、以什么形态进池，都无法从可观测结果反推）。

  四次尝试（61/98/99 及本轮）每次都揭示一个新的未知：候选池 116 的构成、前向选择的评估对象、
  R² 的量级。**没有源码可对照，每次"下一步"都是一个新的假设而非收敛。**

  **结论**：`rcm` 的自动搜索在本环境不可复现。给定 `selected=` 时估计量精确（已由测试固定），
  自动路径显式断言不匹配 Stata。

---

## 二、缺陷修复（20 项）

### 2.1 面板

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 1 | `xtreg,fe` 的 `r2_b`/`r2_o` 用混尺度拟合值 | `1.0000/.9638` → **`.6650/.7135`** | Stata `xtreg ... , fe` on `lin_1992.dta` |
| 2 | `e(mss)` 算成 `tss-rss` | `388.47` → **`78.087606`** | 同上；`tss` 与 `rss`/`mss` 本就不同尺度 |
| 3 | 共线回归元直接报 `AbsorbingEffectError` | 改为剔除并继续估计 | `cao_chen.dta`，`prerebel` 组内常数 |
| 4 | `xttest0` 未实现 | **`chibar2(01)=252.58`** | 平衡面板 12×10，与 Stata 一致 |

**关于 #3 的数值验证**：修复后 `maize` 系数 `37.43478`（Stata `37.43478`），
`hausman` 统计量 `0.77331812`（Stata `.77331812`）、`p` `.37919252`（Stata `.37919252`）。
`hausman` 在报告中列为「缺失」，实为实现完整、被上游 bug 阻断。

### 2.2 稳健/聚类

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 5 | LSDV 聚类 F 溢出 `-1.73e+16` | → **`NaN`**（Stata 打印 `.`）| `df_t < df_m` 时 Wald 不可用，Stata 亦如此 |
| 6 | 普通 OLS 被误判为 robust 而拒绝 | 修正 `cov_type` 匹配 | `cov_type='nonrobust'` 含子串 `robust` |

### 2.3 HDFE

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 7 | 吸收自由度 `df_a` | `797` → **`261`** | 嵌套于聚类的吸收效应不计自由度 |
| 8 | 缺 `Within R-sq.` 行 | 补齐 | `tss_within` 重拟合的 frame 行数错误 |
| 9 | `r2_within` 样本不对齐 | `4.97e-05` → **`.0002486480`** | Stata `.00024865`；用 pyfixest 的 `_na_index` 精确还原估计样本 |

### 2.4 加权与后估计

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 10 | `margins` 未展开 `fweight` | 24 行 → 2201 行 | 5 个边际效应逐位匹配 Stata |
| 11 | `estat clas` 在 `fweight` 下报错 | 改为可用 | 8 项指标全部匹配（73.47% / 79.03% / …）|
| 12 | `predict()` 默认返回线性指数 | → **概率** | logit 概率；Poisson `n`；OLS 仍 `xb` |

**关于 #12 的危害**：用错值时 Brier score 为 `2.4667`，正确值 `0.1621`——**差 15 倍且不报错**。

### 2.5 时间序列

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 13 | `dfuller` 默认选阶 | `-2.571` → **`-3.718`** | **结论反转**：原判「无单位根」，实为「有单位根」|
| 14 | VAR `rmse_#` 分母用 `df_eq` | `.077907` → **`.013815`** | 应为 `N - df_eq` |
| 15 | VAR `r2_#` 用全序列居中 | `.3648` → **`.3621`** | 应用估计样本 |
| 16 | VAR `chi2_#` 参数索引跨步 | `75.34` → **`93.11047`** | `stata_param_names` 按方程分块，非交错 |
| 17 | `vargranger` 调用即报错 | 改为可用 | 数值本就正确（48.123 / 9.009），仅方程名未传 |
| 21 | VAR 的 IRF 用无偏协方差归一化 | 全序列高 1.5% → **精确** | 应为 ML 尺度；`ml_factor` 早已存在，IRF 是唯一未应用的路径 |
| 22 | `fevd` 无可读输出 | 新增 `fevd_table` | 数值本就正确；轴顺序为 `(响应, 期, 冲击)` |
| 23 | `unitroot(test="kpss")` 答错问题 | 默认改趋势平稳 + 带宽系数 12→**4** | 原实现走 statsmodels 默认，数值 `.3197` vs Stata `2.14` |

### 2.6 其他

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 18 | `didregress` 的 Control 行 Time | `0/0` → **`1650/1650`** | 应为面板起始期 |
| 19 | `result.aic`/`.bic` 口径 | `-194.375` → **`-109.239`** | 改为 statsmodels/`estat ic` 口径；`e(aic)` 保留原值 |
| 20 | `reghdfe.predict()` 未注册 | — | 一度误加注册，已撤销（有 FE 时设计矩阵与参数维度不符）|

---

## 三、能力新增（17 项）

每项均对着 Stata 实测值验证。

| 命令 | 验证结果 |
|---|---|
| `estat dwatson` | `d(4,30)=1.021169` ✓ |
| `wntestq` | `Q 26.1974 / 16.3718` ✓ |
| `corrgram` | AC/PAC/Q/p 全对（3 个 lag）✓ |
| `estat bgodfrey` | `4.237/4.487/4.775/5.099` ✓ |
| `prais` | rho、4 系数、4 标准误、R²、RMSE、F、两个 DW 全对 ✓ |
| `xttest0` | `chibar2=252.58` ✓（仅平衡面板）|
| `varsoc` | LL/LR/AIC/HQIC/SBIC 全部 5 阶 ✓ |
| `varstable` | 4 个特征值及模 ✓ |
| `varwle` | 6 个数值（2 方程 × 2 阶 + All）✓ |
| `vecrank` | 3 行 × 4 列全部 ✓（`lags≥2`）|
| `vecstable` | 含规范强加的单位根 ✓ |
| `varlmar` | `7.3130 / 5.0645` ✓ |
| `veclmar` | `8.8911 / 4.8537` ✓ |
| `dfgls` | 8 个数值（含/不含趋势）✓ |
| `contrast` 联合 Wald | `df=3, F=5.90, P>F=0.0006` ✓ |
| `VARResults.summary()` | 方程表 5 列全对 ✓ |
| `margins atmeans` + `odds_ratios` | 两者均精确 ✓ |
| `kpss` | 5 个 lag 统计量 + 临界值表 ✓ |
| `rdrobust`（点估计）| `tau=7.6896`、有效样本 96/69 ✓ |
| `teffects ra`（点估计）| ATE/ATET/两个 POmean 全对 ✓ |
| VAR 脉冲响应（IRF）| step 0/1/2/3/10 全对 ✓ |
| VAR 方差分解（FEVD）| 4 列 × 多个 step 全对 ✓ |

---

## 四、已知差异（已记录，未强行对齐）

| 项 | 差异 | 性质 |
|---|---|---|
| `%9.0g` 输出格式 | 部分小系数多印一位（`.06761177` vs `.0676118`）| **显示舍入**，数值相同 |
| `xtlogit, re` 方差分量 | `sigma_u` `.0037` vs Stata `.0000243` | 估计器差异（非布局）。Note: 修复积分器后已改善，扫描数据上似然面平坦 |
| `xtreg, re` 方差分量 | `1.424/.854` vs Stata `1.472/.709` | 估计器差异 |
| `rdrobust` 标准误 | `0.32163224` vs Stata `0.31709743` | `rdrobust_vce` **仅在二进制 `l/lrdrobust.mlib`**，无 `.mata` 文本；组装公式已验证正确 |
| `teffects ra` 标准误（**特定数据**）| Stata `679.054055`，三种影响函数给 656.90/657.66/659.51（低 2.9–3.4%）| `_teffects_ra_gmm.ado` 只搭 GMM 系统，sandwich 在内置 `gmm` 中；标准 DGP 上的 SE **精确**（`.0610149`）|
| `rdrobust` 带宽选择 | 未实现（`bandwidth=None` 报错）| `rdbwselect.ado` 可读但核心在 Mata 编译库 |
| `bsqreg` 默认重复数 | 200 vs Stata 20 | 刻意不改 |
| `arima` 的 `/sigma` 水平 | 第 5 位有效数字（`.0164569` vs `.0164621`）| **已定性**：两包最大化同一似然（statsmodels 在 Stata 参数处 loglike 差 1e-8），差异纯属优化器停点，似然面极平坦 |
| `vec` 第一方程常数符号 | `+6.28e-07` vs `-6.35e-07` | **已定性**：值本身约 6e-7，两次 ML 拟合的 `c_ce` 差 6.7e-7 × α(−1.87) 超过常数本身；第二方程常数与两个标准误精确匹配 |
| `arch` 均值方程块标题 | `dinf` 右对齐 vs Stata 左对齐 | 涉及共享分组路径，未改 |

**第 57–58 轮已解决（此前列在本表）**：`tobit`/`arima` 的区间端点。tobit 是对数方差尺度 + `t(e(df_r))`，
arima 是 σ 尺度 + 正态 1.96；两者规则不同，此前用「方差区间开方」哪个都不匹配。

### 最终剩余缺口总表（第 69 轮）

**全部 7 项均已定位到具体机制**，没有「尚未查明」的项：

| 项目 | 成因 | 证据 |
|---|---|---|
| `synth` / `synth2` | `regsval`/`normalize`/`getabs` 在二进制 `lsynth_mata_subr.mlib` | 内层 QP 公式已从 `synth_ll.ado` 还原；Mata 函数**可枚举可调用**但脱离 `synth.ado` 全局状态时返回空矩阵（第 70 轮实测）|
| `rdrobust` SE | `rdrobust_vce` 在二进制 `lrdrobust.mlib` | 组装公式已验证 `sqrt((V_l+V_r)[1,1])` = Stata SE |
| `teffects ra` SE | sandwich 在内置 `gmm` | 三种影响函数均低 2.9–3.4%，缺项而非缩放错 |
| `xtserial` | 两个 ado 路径下均不存在 | 已按第 61 轮教训复查用户 ado 路径 |
| `vecrank lags(1)` | 搜索空间与 Stata 不同 | 全部样本起点 × 两种 `k_ar_diff` 实测，特征值差一倍 |
| `rcm` 自动搜索 | `rcm()` Mata 搜索空间受限 | 穷举最优前期 RSS 20.73，Stata 选中的是 256.39（差 13 倍）|
| `cov_params()` → DataFrame | **用户明确暂不修复** | 破坏 `cov_params()[0,0]` 索引方式 |

**未断言完成的验收口径**：因上述 7 项（其中 1 项为用户决定），本目标**不能宣称「与 Stata 完全一致」**，
只能宣称「报告的 D 类缺陷全部修复、E 类 23 项实现 21 项、剩余项成因全部查明」。

### `synth` 的 V 权重：三条路已逐一试过并封死（第 61/70/71 轮）

因为这是最可能仍可突破的一项，记录全部尝试，避免后来者重复：

| 尝试 | 结果 |
|---|---|
| **读源码** | `synth.ado`（931 行）与 `synth_ll.ado`（36 行，可读）给出内层 QP 与外层损失结构；但 `regsval`/`normalize`/`getabs` 只在二进制 `lsynth_mata_subr.mlib` |
| **运行时直接调用**（第 70 轮） | `mata describe using lsynth_mata_subr` 可列出 8 个函数并调用；但脱离 `synth.ado` 运行上下文时 `regsval` **返回 0×0 空矩阵** |
| **手工设全局量后调用**（第 71 轮） | `synth.ado:467-472` 确实把 `Xtr/Xco/Ztr/Zco` 设为矩阵名的全局量，照此设置后 `regsval` **仍返回 0×0**，说明依赖更深层的 Mata 内部状态 |
| **导出 Stata 的 `e(V)`** | `synth` **不保存** `e(V)`/`e(V_matrix)`，`e(W_weights)` 需 `keep()` 才落地 |
| **捕获优化器输出**（第 71 轮） | `synthopt.plugin` 是 C++ 插件，优化表格**直接写 C 层控制台**，`quietly`、`stdout` 重定向、`log using` 三种方式均无法截获 |

已确定的可用部分：内层 QP（`H = Xco'·V·Xco`，单纯形约束）已验证正确——给定 V 时能复现
Stata 的 W 权重结构。缺的只是**产生那个 V 的外层搜索**。

**第 97 轮补充（受第 96 轮 vecrank 的教训启发：换角度而非重复同一方向）**

重新读了 `synth.ado:386-391` 的调用上下文，找到一条此前忽略的信息：

```
local checkinput : list sizeof customV
if(`checkinput' != rowsof(`Xtr'))  -> rowsof(Xtr) == 预测变量个数
mata: regsval("`Xtr'","`Xco'","`Ztr'","`Zco'")     // 四个参数
```

两点新事实：

1. `regsval` 接收 **四个**矩阵，其中包含 `Ztr`/`Zco`（结果变量路径），说明 **V 不只是 X 的
   函数，也依赖 Z**——此前所有"标准回归法 V"的假设都只用了 X，方向就错了。
2. `Xtr` 在传入前已由 `normalize` 处理，形状是 K×1，`Xco` 是 K×N，而 `Ztr` 是 T×1、
   `Zco` 是 T×N——**X 与 Z 的时间维不同**，这排除了若干此前试过的直接回归形式。

**仍未能复现**：按 X/Z 联合构造 V 的两种新假设分别给出 `state4=.5079, state5=.4921`（选对了
两个州但比例不对）和无约束 Z 回归的 38 个非零系数（与 Stata 的 2 个差距更大）。

累计三轮（61/70/71/97）共 6 类途径，全部失败。**结论不变**：唯一剩余路径是取得作者源码或
文献中的 `regsval` 定义，不在本环境可达范围内。

**结论**：唯一剩余路径是取得作者源码或文献中的 `regsval` 定义，不在本环境可达范围内。
`vec` 主表也已从「规则未定」推进到 9 个 `e()` 标量全部精确。

---

## 五、明确未实现及原因

| 项 | 状态 | 原因 |
|---|---|---|
| `vecrank lags(1)` | 报错拒绝 | 映射到 `k_ar_diff=0` 时后端给 `173.8827/2.8945`，Stata 是 `172.50341/1.2744366`；移位样本后第一个对、第二个错。**4 次尝试未定，选择报错而非近似值** |
| `estat bgodfrey`（早期）| 后已实现 | 曾两次判定「无法复现」。真相是源码在 `ado/base/b/bgodfrey.ado`，我一直未查。**「找不到公式」不等于「公式不可得」** |
| `xttest0` 不平衡面板 | 报错拒绝 | 平衡面板公式在不平衡数据上差 2.5%。**一度误判为公式错误，实为用错数据** |
| RD（`rdrobust`）| **点估计已实现且精确**（真实数据 9 位）；标准误与带宽未实现 | 见下方专项 |
| 合成控制（`synth`/`synth2`）| **未实现，但源码可得** | 见下方专项：此前记「SSC 未安装」是**错的** |
| 回归控制（`rcm`）| 未实现 | 未查用户 ado 路径 |
| `teffects ra` | **点估计已实现**，标准误未实现 | 见下方专项 |
| **`teffects psmatch`（C15-01/C15-02）** | **点估计已实现**（第 52 轮）| 见下方专项说明；AI-robust 标准误仍未复现 |
| `tebalance` / `xtserial` | 未实现 | **已确认**用户 ado 路径下确实不存在 |
| `cov_params()` 返回 DataFrame | **暂不修复** | 会破坏 `cov_params()[0,0]` 索引方式，属破坏性 API 变更；用户已明确暂不处理 |

### 用户 ado 路径专项（第 60–61 轮）——一条一直找错的路径

**此前所有「SSC 包未安装、无源码」的判断，都只查了 Stata 安装目录**：

```
D:\Program Files\StataNow19\ado\base\   ← 一直只查这里
```

实际路径是：

```
C:\Users\zjl\ado\plus\   ← 用户 ado，含大量可读源码
    r/rdrobust.ado       (1331 行)   rdbwselect.ado (876 行)
    r/reghdfe.ado + reghdfe.mata     ← 连 Mata 都是文本
    s/synth.ado (931 行)  synth2.ado (968 行)  synth_ll.ado
    f/ftools*.mata、estout/esttab/estpost 全家桶
```

**影响**：`synth`、`synth2`、`reghdfe`、`rcm` 的源码**一直可得**，此前被误记为「无源码」。
`tebalance`/`xtserial` 已复查确认确实不存在。

### `rdrobust` 专项

点估计**精确**：`rdrobust diff border, c(30) h(5) b(8)` 的 `tau_cl` westata 与 Stata 都是
`121.297562295`，真实数据、九位有效数字。

标准误未复现，且**这次是真正的边界**：`rdrobust.ado` 写明了组装公式
`V_tau_cl = scalepar²·deriv!²·(V_cl_l+V_cl_r)[deriv+1,deriv+1]`（已验证
`sqrt((V_cl_l+V_cl_r)[1,1]) = 69.2861818` = Stata `se_tau_cl 69.2861819408`），
但构造 `V_cl_l`/`V_cl_r` 的 `rdrobust_vce` **只存在于编译后的 `l/lrdrobust.mlib`**，
整个 ado 树没有任何 `.mata` 文本。CR1 候选公式全部不匹配（`w²res²`→38.16、
`w·res²`→240.47、`w·res²·n`→1202.33，目标 496.95）。故 `se`/`z`/`ci` 保持 NaN。

### `synth` 专项（第 61–62 轮）

**源码基本可得，算法结构已还原，但尚未实现到精确。**

`s/synth.ado`（931 行）与 **`s/synth_ll.ado`（36 行，可读）** 给出了核心算法：

```
内层 QP（给定 V）：  H = Xco' · V · Xco
                     c = -(Xtr' · V · Xco)'
                     约束 Σw = 1, 0 ≤ w ≤ 1
外层损失：           loss = (Ztr - Zco·w)'(Ztr - Zco·w)
```

**X = 预测变量，Z = 处理前各期结果变量路径**——内层用 X、外层用 Z，这是关键。
V 由 `getabs` 归一化（`|b|/Σ|b|`），初值来自 `regsval`（回归法）。

实测参照（`synth_smoking.dta`，`synth cigsale lnincome age15to24 beer retprice,
trunit(3) trperiod(1989)`）：

| 选项 | 非零权重 |
|---|---|
| **`nested`** | **state 4 = 0.631, state 5 = 0.369**（仅 2 个） |
| 默认（回归 V） | state 4 = 0.555, 5 = 0.306, 6 = 0.105, 21 = 0.034（4 个） |

已实现的嵌套优化（外层 Nelder-Mead + 内层 SLSQP QP）得到
`state 4 = 0.5419, state 5 = 0.4581`——**选对了两个州**（默认路径会选出 4 个），
但比例与 Stata 的 `0.631/0.369` 不同，说明外层 V 未找到全局最优。

`regsval`/`normalize`/`getabs` 均定义在二进制
`l/lsynth_mata_subr.mlib` 中，无文本源码；`synth_ll.ado` 虽可读但只覆盖内层。
按纪律**不发布**近似实现。

### `teffects psmatch` 专项（报告 P0，第 52 轮）——**点估计已攻克**

Stata 参照（`nsw_dw.dta`，`teffects psmatch (re78) (treat <8 协变量>), atet`）：

```
默认 nn(1):  ATET 2639.865    SE 714.7758    Matches min=1 max=8
nn(4):       ATET 2330.108    SE 680.0515    Matches min=4 max=11
```

**第 51 轮我曾误判为「估计量无法复现、完全未实现」，这个结论是错的，在此撤回。**
第 52 轮用 Stata 的 `generate(m)` 选项导出了实际匹配的观测号，定位到真正原因：

- `generate(m)` 对 `nneighbor(1)` 生成了 **m1..m8 共 8 个变量**。帮助文件写明「因距离
  并列，生成的变量数可能多于 nneighbor()」。
- **Stata 把所有与最小距离并列的对照全部纳入并取均值。** 只取最近 1 个得 2569.745，
  取全部并列对照得 **2639.865** ✓；`nn(4)` 得 **2330.108** ✓。表头的 `max = 8` / `max = 11`
  正是并列数。
- 报告 C15-01 记录的 13.7% 差距，根因不是点估计错，而是**忽略了并列**。

排查路径（每一步都排除了一个假设，而非猜测）：

1. westata 的 logit 系数与 Stata 一致到 1e-7，`predict()` 与 Stata 导出的 ps 差 2.75e-8
   → 差异**不在倾向得分模型**；
2. 用 **Stata 自己导出的 ps/xb** 做最近邻匹配，仍差 192 → 差异**在匹配算法**；
3. `generate(m)` 揭示并列 → 从 Stata 的匹配 ID 重算 ATET 精确复现 → **并列规则确认**。

已实现 `teffects_psmatch`：`teffects_psmatch(y, d, X, nneighbor=1|4)`，两种 ATET 及
min/max 匹配数全部精确，Stata 风格表格逐行对齐。

**仍未复现**：Abadie-Imbens AI-robust 标准误（需局部线性方差估计，非闭式）。`se`/`z`/`p`/
区间返回 NaN，表格以 Stata 的 `.` 如实占位，不发布近似值。

`teffects ipw`（C15-06）的紧凑摘要问题未复核；`teffects ra`（回归调整）点估计已实现。

### 已撤回的错误结论

工作过程中有 **5 个结论被后续证据推翻**，记录在此以免重复：

1. `reghdfe` 不实现吸收效应 —— 错，公式 `|` 块即等价；是**扫描器比了不同模型**
2. `etregress` 需要实现 athrho 参数化 —— 错，`method='ml'` 早已存在
3. `tnbreg` 卡方混合检验「无法计算」—— 错，`e(ll)`/`e(ll₀)` 一直在
4. `xttest0` 公式不对 —— 错，是**用不平衡数据测平衡公式**
5. `estat bgodfrey` 无法复现 —— 错，**源码一直在 `ado/base`**

前三条的共同病根：把「扫描器/测试设置的问题」当成了「代码的问题」。

---

## 六、复现方式

```bash
# 全量测试（本次工作最后一个检查点：4284 passed, 3 skipped）
./.venv/Scripts/python.exe -m pytest tests/ -q \
  --deselect tests/test_packaging_hygiene.py::test_tracked_test_data_is_actually_tracked

# 单点核对（示例：VAR 每方程统计量、margins 表格、VECM 协方差）
./.venv/Scripts/python.exe -m pytest tests/test_var_equation_stats.py -v
./.venv/Scripts/python.exe -m pytest tests/test_margins_layout.py -v
./.venv/Scripts/python.exe -m pytest tests/test_vecm_cov_scale.py -v
```

每个新增测试文件的文档字符串里都写了它对照的 **Stata 命令、数据集、参照数值**，可直接复核。
第七节所列每一张表格都有对应的 `tests/test_*_layout.py` 守卫，逐行固定 Stata 输出。

主要测试文件（早期）：

| 主题 | 文件 |
|---|---|
| 面板 R²/mss | `tests/test_panel_e_identities.py`、`tests/test_e_identities.py` |
| 吸收列处理 | `tests/test_panel_absorbed_columns.py` |
| HDFE 自由度 | `tests/test_var_e_block.py`（相关断言）|
| 预测默认值 | `tests/test_predict_defaults.py` |
| 自相关检验 | `tests/test_serial_correlation.py`、`tests/test_bgodfrey.py` |
| Prais-Winsten | `tests/test_prais.py` |
| VAR/VECM 族 | `tests/test_varsoc.py`、`test_varstable.py`、`test_varwle.py`、`test_varlmar.py`、`test_vecrank.py`、`test_vecstable.py` |
| 单位根 | `tests/test_unitroot_adf.py`、`tests/test_dfgls.py` |
| 对比与边际 | `tests/test_contrast_joint.py`、`tests/test_margins_atmeans_odds.py` |
| VAR 表格 | `tests/test_var_summary.py`、`tests/test_var_equation_stats.py` |

---

## 七、第 30–51 轮：B 类重新审视后的成果

报告把 39 项归为「B 类：布局适配（设计约束，非缺陷）」。目标改为「数值与排版与 Stata
完全一致」后，逐项实测发现多数确实不同。以下每项都逐行对照过 Stata 实测输出，并有独立
守卫测试固定。

### 主回归表与后估计表格（逐行一致）

| 命令 | 结果 | 关键发现 |
|---|---|---|
| `estat ic` / `vif` / `ovtest` / `hettest` / `bgodfrey` | 全部逐行一致 | 列数、标签填充、缩进、精度各不同 |
| `estat classification` | **26/26 行** | 原缺标题行、表格结构完全不同、cutoff `0.5` vs `.5` |
| `estat gof` / `vce` | 6/6、8/8 行 | 缺两个计数行；缺模型名、`y` 列头、`e(V)` 标注 |
| `estat firststage` | 两张表全对 | 缺整个 Stock-Yogo 块；`r(mineig)` 错报特征值而非统计量 |
| `estat overid` / `endogenous` | 4/4、5/5 行 | **Basmann 的 k 用错**：`1.1953` → `1.1923` |
| `margins` | 三张表 10/10、11/11、9/9 | **原本输出 dataclass repr，没有表格** |
| `pwcompare` / `contrast` | 11/11、9/9 行 | 前者原本输出 list repr；后者缺 `Margins:` 与 `Denominator` 行 |
| `test` / `lincom` / `lrtest` | 逐行一致 | 居中标签、双层表头、`Assumption:` 行 |
| `ivregress` / `qreg` / `xtreg fe` / `areg` 主表 | 5/5、3/3、21/21、12/12 行 | 见下方「真实数值缺陷」 |
| `sureg` / `mvreg` / `svy` 族 / `newey` / `ologit` | 7/7、5/5、11/11 等 | `sureg` 缺 `chi2`/`P>chi2` 两整列 |
| `irf_table`（新增）| oirf/irf 各 10/10 行 | 报告 C13-06；数值已在，缺可读表格 |
| VECM `cov_params()` | 精确 | **每次调用抛 AttributeError**；标准误低 1.24% |

### 真实数值缺陷（非排版）

1. **`r(mineig)` 尺度**：存特征值 `0.6267`，Stata 打印 Cragg-Donald 统计量 `332.355`。
   恒等式 `F = eig/(1-eig)·df_denom/df_num` 精确成立。
2. **`tobit` 的 `var(e.y)`**：打印 σ 而非 σ²（`1.029584` vs `1.060043`）。
   `1.029584² = 1.060043`，`2σ·se(σ)` 也精确吻合标准误。
3. **`arima` 的 `/sigma`**：打印 σ² 而非 σ（`.0002709` vs `.0164621`），**差 60 倍**。
4. **`arch` 行标签错位**：`L1.` 贴在 omega、`_cons` 贴在 alpha[1]。参数重排时 OPG
   协方差若用重排后的向量计算，标准误从 `.1471562` 变成 `122.6474`。
5. **`irf()` 过度缩放**：`ml_factor` 施加到所有响应，把 IRF step0 从 `1` 变成 `0.984638`。
6. **`areg` 崩溃**：`maize` 有 3144 个**缺失值**（不是单例组），全样本设计乘估计样本响应。
7. **`ivregress` 缺 `R-squared`**、**`xtreg` 缺 `(omitted)` 行**：信息缺失，非数值错误。

### 反复出现的模式：注释与测试在保护 bug

多个缺陷的代码注释都写着一个看似合理的理由，实测 Stata 相反；配套测试把那个理由固定
下来。这是本项目最常见的失败模式：

| 注释/测试断言 | 实测 |
|---|---|
| `ivregress` 不打印 R² | Stata 打印 `R-squared = 0.8923` |
| `estat ic` 打印 AICc/CAIC | Stata 只有 7 列 |
| `test_estat_ic_label` 要求显示命令名 | Stata 显示 `.` |
| `lincom` 打印 `Coef.` | Stata 打印 `Coefficient` |
| `qreg` 表头是特例（70 列）| 整块都是 78 列 |
| `svy: mean` 右半块随值浮动 | 恒在 34 列 |
| `arch dinf` 均值块标题是 `y` | 实际是 `dinf` |
| `contrast` 不需要 Denominator | F 没有分母自由度就无法解读 |
| `stderr_alpha` 已与 Stata 核对 | 实际低 1.24% |

**教训**：注释固化猜测，测试固化注释。每一次都必须回到实时 Stata 输出重新测量。

---

## 八、需要使用者知情的事项

1. **`predict()` 默认值已变**（`11b1936`）。`Logit.predict()` 现在返回概率而非 log-odds。
   需要旧行为请显式写 `kind='xb'`。这是 **breaking change**，未发布到任何版本。

2. **本仓库工作区未提交的改动请勿混入**。各轮提交都只暂存了明确列出的文件。

3. **「64 项」不表示「westata 已可替代 Stata」**。下述缺口真实存在：`vec` 主表（AIC/HQIC/
   SBIC 规则未定）、`areg` 的 F test of absorbed indicators（差 0.34%，拒绝近似值）、
   `teffects psmatch` 与 `teffects ra` 的 AI-robust / 影响函数标准误、`mlogit`（完全未实现）、
   断点回归/合成控制/回归控制整章、以及 `cov_params()` 契约（用户明确暂不修复）。

4. **「已知差异」不等于「已验证一致」**。凡提交信息里写「记录而非声称」「拒绝近似值」
   的，都不要用「已对齐」概括。
