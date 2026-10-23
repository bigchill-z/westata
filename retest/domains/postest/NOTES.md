# postest 领域复测结论（westata vs StataNow 19.5 MP）

- 复测方式：本机内嵌 StataNow 19.5 (MP) 会话 + 同进程 westata 实跑，逐项比对。
  **未**引用 `tests/conformance/` 下任何 fixture / registry / csv，也未引用 `docs/GAP_REGISTER.md`。
- 脚本：`retest/domains/postest/run.py`（`cd D:\myCodeProject\statapy && .venv\Scripts\python.exe retest/domains/postest/run.py`）
- 原始记录：`results.jsonl`（77 条），汇总：`summary.json`
- Stata 版本：`c(stata_version)` = 19.5，edition `mp`；`esttab` 已安装（2.1.4），无需退化为 `estimates table`
- 整体计数：**77 条，pass 45 / fail 32 / error 0 / skip 0**；其中 A 档（≤1e-9）43 条

判定口径沿用 BRIEF §2.3：A ≤1e-9、B ≤1e-6、C ≤1e-3、D ≤1e-2、E >1e-2，
相对差 = `max(|westata - stata| / max(|stata|, 1e-12))`。

> **重要口径说明（本领域特有）**：`test` / `margins` / `contrast` / `pwcompare` / `lincom` / `nlcom`
> 的 Stata 结果都存在 `r()` 里，**没有 `e(b)/e(V)`**，而 `r(b)/r(V)` 在 `pwcompare` 里存的是
> *单元格均值*、对比量在 `r(b_vs)`。所以本领域的"统计量"必须从 `r(F)/r(chi2)/r(df)/r(p)`
> 或 `r(table)`（9×k：b、se、stat、p、ll、ul、df、crit、eform）取全精度，不能解析打印表。
> 每条 `test`/`margins` 记录都同时比了**统计量、df、p 值**三项（见 `extra.part_rel`）。

---

## 覆盖矩阵

| # | Stata 命令 | westata 调用 | 数据 | 判定 | 系数相对差 | 标准误相对差 |
|---|---|---|---|---|---|---|
| 1 | `regress mpg weight foreign` | `ws.OLS.from_formula` | auto(N=74) | **pass A** | 0 | — (样本核对) |
| 2 | `logit foreign weight mpg` | `ws.Logit.from_formula` | auto | **pass A** | 0 | — (样本核对) |
| 3 | `poisson rep78 mpg weight` | `ws.Poisson.from_formula` | auto dropna(N=69) | **pass A** | 0 | — (样本核对) |
| 4 | `tobit mpg weight foreign, ll(17)` | `ws.Tobit.from_formula(lower=17)` | auto | **pass A** | 0 | — (样本核对) |
| 5 | `test weight` (regress) | `res.test("weight")` | auto | **pass A** | 2.66e-16 | — |
| 6 | `test weight foreign` | `res.test("weight","foreign")` | auto | **pass A** | 1.02e-15 | — |
| 7 | `test weight=0.5` | `res.test("weight=0.5")` | auto | **pass A** | 9.21e-16 | — |
| 8 | `test weight=foreign` | `res.test("weight=foreign")` | auto | **pass A** | 1.34e-14 | — |
| 9 | `regress …, vce(robust)` + `test weight foreign` | `fit(cov_type="HC1")` + `test` | auto | **pass A** | 4.81e-15 | — |
| 10 | `test weight` (logit) | `lg.test("weight")` | auto | **pass A** | 1.91e-10 | — |
| 11 | `test weight mpg` (logit) | `lg.test("weight","mpg")` | auto | **pass A** | 4.62e-11 | — |
| 12 | `test weight=0` (logit) | `lg.test("weight=0")` | auto | **pass A** | 1.91e-10 | — |
| 13 | `lincom weight` | `res.lincom("weight")` | auto | **pass A** | 6.81e-16 | — |
| 14 | `lincom 2*weight - 3*foreign` | `res.lincom(...)` | auto | **pass A** | 1.40e-14 | — |
| 15 | `lincom 2*weight - 3*foreign + 10` | `res.lincom(...)` | auto | **fail** | — (westata 拒绝) | — |
| 16 | `lincom …, level(90)` | `lincom(..., level=0.90)` | auto | **pass A** | 6.24e-14 | — |
| 17 | `lincom weight + mpg` (logit) | `lg.lincom(...)` | auto | **pass A** | 7.81e-10 | — |
| 18 | `nlcom _b[weight]/_b[foreign]` | `res.nlcom(...)` | auto | **fail C** | 1.34e-05 | — |
| 19 | `nlcom exp(_b[weight])` | `res.nlcom(...)` | auto | **pass A** | 5.87e-11 | — |
| 20 | `nlcom exp(_b[weight])` (logit) | `lg.nlcom(...)` | auto | **pass A** | 1.13e-11 | — |
| 21 | 上条 #18 的解析仲裁 | 解析 delta method | auto | **pass A** | 6.79e-12 | — |
| 22 | `testnl (_b[weight]=1) (_b[foreign]=0)` | `res.testnl(...)` | auto | **pass A** | 5.86e-11 | — |
| 23 | `testnl exp(_b[weight])=1` | `res.testnl(...)` | auto | **pass A** | 8.13e-11 | — |
| 24 | `testnl _b[weight]=1, _b[foreign]=0` | `res.testnl(...)` | auto | **fail** | — (语法接受度) | — |
| 25 | `testnl _b[weight]=1 _b[foreign]=0` | `res.testnl(...)` | auto | **fail** | — (语法接受度) | — |
| 26 | `lrtest LU LR`（嵌套 logit） | `ws.lrtest(big, small)` | auto | **pass A** | 9.11e-15 | — |
| 27 | `lrtest RU RR`（嵌套 regress） | `ws.lrtest(big, small)` | auto | **pass A** | 7.03e-15 | — |
| 28 | `lrtest E1 E2`（同模型，df=0） | `ws.lrtest(m2, m1)` | auto | **fail** | — (接受度) | — |
| 29 | `lrtest N1 N2`（非嵌套） | `ws.lrtest(b, a)` | auto | **fail** | — (接受度) | — |
| 30 | `hausman FEm REm`（正定差） | `ws.hausman(fe, re)` | 自造面板 40×8 | **pass A** | 2.60e-13 | — |
| 31 | `hausman FEm REm, constant`（正定） | — 无对应开关 | 同上 | **fail** | — (API 面) | — |
| 32 | `hausman FEm REm`（非正定差） | `ws.hausman` | 同上 corr=0.15 | **fail** | — (统计量符号) | — |
| 33 | `hausman FEm REm, constant`（非正定） | — | 同上 | **fail** | — (API 面) | — |
| 34 | `contrast r.foreign` | `m2.contrast("C(foreign)","r")` | auto | **pass A** | 5.38e-15 | — |
| 35 | `contrast g.foreign` | `contrast(..., "g")` | auto | **pass A** | 5.38e-15 | — |
| 36 | `contrast a.foreign` | `contrast(..., "a")` | auto | **pass A** | 5.38e-15 | — |
| 37 | `contrast r.r3`（3 水平） | `contrast("C(r3)","r")` | 自造 r3(N=69) | **pass A** | 5.35e-14 | — |
| 38 | `contrast g.r3` | `contrast(..., "g")` | 同上 | **pass A** | 3.81e-15 | — |
| 39 | `contrast a.r3` | `contrast(..., "a")` | 同上 | **pass A** | 5.35e-14 | — |
| 40 | `pwcompare r3` | `m.pwcompare("C(r3)")` | 同上 | **pass A** | 5.35e-14 | — |
| 41 | `margins, dydx(weight)`（logit） | `lg.margins(dydx="weight")` | auto | **fail D** | 7.15e-03（SE 分量 7.10e-03） | 同左 |
| 42 | 上条 #41 的解析仲裁 | 解析 delta method | auto | **fail D** | 7.10e-03 | — |
| 43 | `margins, dydx(weight) at(mpg=20)` | `margins(dydx=..., at=...)` | auto | **fail E** | 5.38e-02（SE 分量） | 同左 |
| 44 | `margins`（regress 预测边际） | `m.margins()` | auto | **pass A** | 4.08e-12 | — |
| 45 | `margins, over(foreign)` | `m.margins(over="foreign")` | auto | **fail B** | 1.85e-09（SE 分量） | 同左 |
| 46 | `margins, dydx(foreign)` | `margins(dydx=..., factor_vars=...)` | auto | **fail B** | 7.74e-09（SE 分量） | 同左 |
| 47 | 上条 #46 的解析仲裁（=系数） | 系数即解析解 | auto | **fail B** | 1.79e-09 | — |
| 48 | 上条 #45 的解析仲裁 | 解析 delta method | auto | **fail B** | 1.85e-09 | — |
| 49 | 上条 #43 的解析仲裁 | 解析 delta method | auto | **fail E** | 5.38e-02 | — |
| 50 | `predict, n`（poisson）恒等式 | exp(X b_Stata) | auto | **pass A** | 2.64e-16 | — |
| 51 | `predict, xb`（tobit）恒等式 | X b_Stata | auto | **pass A** | 1.73e-16 | — |
| 52 | `predict, stdp`（tobit）恒等式 | sqrt(x'V_Stata x) | auto | **pass A** | 1.32e-15 | — |
| 53 | `predict double, xb`（regress） | `m.predict(kind="xb")` | auto | **pass A** | 1.45e-15 | — |
| 54 | `predict double, stdp`（regress） | `predict(kind="stdp")` | auto | **pass A** | 3.95e-15 | — |
| 55 | `predict double, stdf`（regress） | `predict(kind="stdf")` | auto | **pass A** | 2.56e-16 | — |
| 56 | `predict double, residuals`（regress） | `predict(kind="residuals")` | auto | **pass A** | 5.11e-13 | — |
| 57 | `predict double, pr`（logit） | `predict(kind="pr")` | auto | **pass A** | 4.59e-10 | — |
| 58 | `predict double, xb`（logit） | `predict(kind="xb")` | auto | **pass A** | 1.11e-10 | — |
| 59 | `predict double, stdp`（logit） | `predict(kind="stdp")` | auto | **pass A** | 6.80e-11 | — |
| 60 | `predict, stdf`（logit） | `predict(kind="stdf")` | auto | **pass** | — (双方均拒绝) | — |
| 61 | `predict double, n`（poisson） | `po.predict(kind="n")` | auto N=69 | **fail B** | 2.44e-09 | — |
| 62 | `predict, stdf`（poisson） | `po.predict(kind="stdf")` | auto N=69 | **pass** | — (双方均拒绝) | — |
| 63 | `tobit …` 设计系数 | `tb.params` | auto | **fail B** | 4.06e-08 | — |
| 64 | `tobit …` 设计系数标准误 | `tb.bse` | auto | **fail C** | 8.88e-05 | 同左 |
| 65 | `tobit …` sigma 点估计 | `tb.params["/sigma"]` | auto | **fail B** | 6.93e-09 | — |
| 66 | `tobit …` sigma 的标准误 | `tb.bse["/sigma"]` | auto | **fail E** | 7.36e-01 | 同左 |
| 67 | `tobit …` `e(df_r)` | `tb.df_resid` | auto | **fail E** | 1.39e-02 | — |
| 68 | `predict double, e(17,.)` | `tb.predict(kind="e")` | auto | **fail B** | 5.78e-09 | — |
| 69 | `predict double, ystar(17,.)` | `tb.predict(kind="ystar")` | auto | **fail B** | 5.79e-09 | — |
| 70 | `predict double, stdp`（tobit） | `tb.predict(kind="stdp")` | auto | **fail C** | 7.17e-05 | — |
| 71 | `predict double, xb`（tobit） | `tb.predict(kind="xb")` | auto | **fail B** | 1.50e-08 | — |
| 72 | `predict double, stdf`（tobit） | `tb.predict(kind="stdf")` | auto | **fail E** | 7.71e-02 | — |
| 73 | 上条 #72 的归因（rss/N vs sigma²） | — | auto | **fail E** | 1.52e-01 | — |
| 74 | `predict double, pr(17,.)`（tobit） | `tb.predict(kind="pr")` | auto | **fail** | — (westata 拒绝) | — |
| 75 | `esttab m1 m2, b(%12.6f) se(%12.6f) nostar …` | `ws.etable(...)` + `to_markdown` | auto | **pass A** | 0 | 0 |
| 76 | `pwcompare foreign, mcompare(bonferroni)` | `pwcompare(adjust="bonferroni")` | auto | **fail** | — (westata 拒绝) | — |
| 77 | `contrast h.foreign` | `contrast(..., "h")` | auto | **fail** | — (westata 拒绝) | — |

---

## 逐项说明（不达 A 档 / 非纯数值条目）

### 1. `lincom` 不接受裸常数项（#15）

- Stata：`lincom 2*weight - 3*foreign + 10` 正常返回 `r(estimate)=14.9369…`、`r(se)=3.2272…`、`r(df)=71`。
  常数项不参与方差，所以 SE 与不带常数时完全一致。
- westata：`res.lincom("2*weight - 3*foreign + 10")` 抛
  `ValueError: constant '10' is not a coefficient reference; put constants on the right-hand side of '='`。
- 归因：`postestimation/api.py::_split_terms` 明确把裸数字当作非法项（注释说常数应放在 `=` 右边）。
  这是**语法子集**问题，不是算术问题——去掉常数项后 #14 的相对差是 1.40e-14。
- 影响：任何从 Stata 脚本直接搬过来的 `lincom a*x + b*y + c` 会直接报错而不是给错数。

### 2. `nlcom` 比值（#18）的 1.34e-05 是 Stata 侧的数值噪声

- `nlcom _b[weight]/_b[foreign]`：b 相对差 4.6e-15，SE 相对差 **1.996e-06**，
  p 6.93e-06、区间端点 1.34e-05 全部由 SE 传导。
- 我用**解析 delta method**（`g = [1/f, -w/f², 0]`，用 Stata 自己的 `e(b)/e(V)` 全精度）做仲裁（#21）：
  解析 SE = 0.0023949589179793032，**westata = 0.0023949589179955693（相对差 6.79e-12）**，
  Stata = 0.002394963698728649（相对差 **2.00e-06**）。
- 即：westata 精确等于解析值，Stata 自己的 `nlcom` 偏离解析值 2e-6。
  用 `Fraction` 做精确有理数运算也得到同一个解析值（5.735828218808595e-06 方差），
  且 Stata 的 `r(V)` 除以它恒为 1.00000399…，说明这是 Stata `nlcom` 的差分梯度噪声，不是 westata 算错。
- 结论：**westata 更接近真值**；但按"与 Stata 对齐"的口径仍记 fail C。
  判定不影响显著性（p 都在 0.0955 附近）。

### 3. `testnl` 的逗号 / 空格联合写法（#24、#25）

- 实测 StataNow 19.5：**只有括号形式** `testnl (_b[w]=1) (_b[f]=0)` 被接受。
  逗号形式报 `option _b not allowed r(198)`；空格形式被解析成一条畸形约束后报 `1_b invalid name r(198)`。
- westata 三种写法全部接受，且数值完全一致（与 #22 相同，5.86e-11）。
- 归因：`nonlinear.py::_split_constraints` 有意接受逗号与空格分隔（docstring 是这么写的）。
  这是 **westata 是超集**：在 westata 能跑通的脚本搬到 Stata 会 r(198)。
- 影响方向是"西→东移植失败"，不是"算错"。已在 NOTES 明确标注。

### 4. `lrtest` 的 df=0 与嵌套校验（#28、#29）

- **df=0**：`lrtest E1 E2` 用同一个模型拟合两次（参数数相同）。StataNow 19.5 直接拒绝：
  `df(unrestricted) = df(restricted) = 2 r(498)`。westata 返回 `chi2=0, df=0, p=1.0`。
  `p=1.0` 对 `chi2(0)` 是数学上正确的生存函数值（`modeltests.py` 显式写了这条分支），
  但**命令成功而 Stata 报错**是可观测的行为差异。
- **非嵌套但参数数不同**：`regress mpg weight` vs `regress mpg foreign length`。
  Stata **接受**并返回 `LR chi2(1) = -2.6835, Prob > chi2 = 1.0000`（负的 LR！）。
  westata 抛 `ConsistencyError: the models are not nested …`。
- 归因：westata 按系数名检查嵌套，Stata 只检查参数个数。**westata 更严格，且它给出的判断在统计上是对的**
  （负 LR 本身说明这对模型不可比）。但仍是行为差异，记 fail 并在 NOTES 说明。
- 正常嵌套场景（#26、#27）两边在 1e-14 / 1e-15 完全一致。

### 5. `hausman`：数值对，但 API 面与退化情形不一致（#31–#33）

- **正定差**（40 实体 × 8 期，α 与 x1 不相关）：Stata `chi2(2)=0.9525, p=0.6211`；
  westata `2.60e-13` 相对差。**常数项双方都排除**，df 都等于 2，一致。
- **`constant` 选项**：Stata 允许 `hausman FEm REm, constant`，此时比较 3 个参数
  （df=3）。westata 的 `HausmanResult` 没有 `constant` 开关，永远只比 2 个斜率（`excluded=[]`）。
  这是**能力缺口**而非算错：`modeltests.py` 的 docstring 说明排除常数项是刻意的。
- **非正定差**（α 与 x1 相关 0.15）：Stata 打印 `chi2(2) = -2.3542` 并警告
  "fails to meet the asymptotic assumptions"，`r(p)=1`。westata 用广义逆后把二次型
  **夹到 0**（`statistic = max(statistic, 0.0)`），报 `chi2=0, p=1`，并发出 RuntimeWarning。
  两者 p 都是 1，统计量符号不同。这是已文档化的退化处理选择，不是算错，但记 fail 以便暴露。

### 6. `margins` 的三个层次（#41–#49）

这是本次**唯一发现真实数值缺陷**的地方，分三层：

**(a) regress 的预测边际（#44）完全正确**：4.08e-12。

**(b) `margins, over(foreign)`（#45）与 `dydx(foreign)`（#46）只有 SE 偏 1e-9 量级**：
点估计相对差 1e-13～1e-14，但 SE 是 1.85e-09 / 1.79e-09。
用解析解仲裁（#47、#48）：`over()` 的 SE 就是 `sqrt(g'Vg)`，`g` 是组内设计矩阵列均值——
这是**精确线性组合**，解析值无歧义；Stata 与解析值一致到 0（或 3e-16），
westata 偏 1e-11～1e-9。`dydx(foreign)` 在线性模型里离散变化**就等于系数本身**，
所以 SE 的解析解就是 `bse`；Stata 相对差 0，westata 1.79e-09。

**(c) logit 的 AME（#41、#43）是量级问题**：
- `margins, dydx(weight)`：Stata `-4.569448e-4, se 5.706055e-5`；westata 点估计相对差 7.15e-08、
  但 **SE 相对差 7.10e-03**（4.4%→0.7% 级别的绝对误差：5.6655e-5 vs 5.7061e-5）。
- `margins, dydx(weight) at(mpg=20)`：**SE 相对差 5.38e-02**（1.876e-6 vs 1.780e-6），
  点估计仍只差 1.08e-07。
- 仲裁（#42、#49）：用解析梯度
  `G_j = mean(s_i(1-2p_i)x_ij·b_w) + [j=w]·mean(s_i)`，`s_i=p_i(1-p_i)`，
  代入 Stata 自己的 `e(b)/e(V)`：
  - 全样本 AME：解析 SE = 5.70605521e-05，**Stata 5.70605515e-05（差 1e-8）**，
    westata 5.66553325e-05（差 7.1e-3）。
  - `at(mpg=20)`：解析 SE = 1.7799669233e-06，**Stata 1.7799669228e-06**，
    westata 1.8757698e-06（差 5.4e-2）。
  我还读了 Stata 的 `r(Jacobian)`：`[-1.10858047e-01, -6.64926510e-04, -4.79699615e-05]`，
  与解析梯度逐位吻合；westata 内部梯度是 `[-1.10859349e-01, -6.65530910e-04, -4.78880938e-05]`。

**归因（可验证）**：`postestimation/margins.py::_statistic_and_gradient` 对协变量用
**固定步长 h = 1e-6 的中心差分**（`shifted[:, i] += h`），而不是解析导数 `p(1-p)`。
我复现了步长扫描：h=1e-3 时 SE 相对差 4.9e-6、h=1e-4 时 8.0e-5、h=1e-6 时 7.1e-3、
h=1e-7 时 4.9e-2、h=1e-8 时 0.70——**误差随 h 减小反而变大**，典型的
"差分步长小于函数值量级导致相消"模式。AME 本身只有 3.5e-4 量级，`p_i` 约 0.1，
`(p(x+h)-p(x-h))/(2h)` 在 h=1e-6 时高位有效数字已被吃掉。

**影响**：`margins, dydx()` 的**点估计可信（1e-7）**，但 **SE、z、p、置信区间不可信**，
在 `at()` 定值时误差可达 5%（跨档到 E）。这会改变"边际效应显不显著"的判断：
AME 的 z 从 Stata 的 -8.01 变成 westata 的 -8.07（差 0.7%），at(mpg=20) 时
z 从 -197.7 变成 -187.6（差 5%）。p 值仍远小于 0.001，但区间宽度会偏。

### 7. `predict` 的 logit / poisson `stdf` 一致性（#60、#62）

- Stata：`predict, stdf` after `logit` / `poisson` 均报
  `option stdf not allowed r(198)`。
- westata：`UnsupportedError: predict: 'logit' supports ('xb','stdp','pr','residuals'); got 'stdf'`，
  poisson 同理。
- **行为一致**，记 pass。这验证了 BRIEF 里"westata 是否一致地拒绝"的问题：答案是**是**。
  （`predict.py` 的 `SUPPORTED_TYPES` 把 stdf 只给了线性族 + tobit。）

### 8. `predict, n`（poisson，N=69）2.44e-09（#61）

- 归因（#50）：`exp(X·e(b)_Stata)` 复现 Stata 自己的 `n` 到 2.64e-16，说明 **exp 链接精确**。
  差异全部来自系数本身：Stata `[0.0098722, -8.7587e-05, 1.27358]`，
  westata `[0.00987220, -8.75874e-05, 1.273580]`，相对差 1.4e-9～1.5e-9（ML 收敛点差异）。
- 属于 B 档"数值实现差异"，不影响任何结论。

### 9. `tobit`：两个**真实缺陷** + 若干精度差异（#63–#73）

**(a) 缺陷一：`/sigma` 的标准误是 log 尺度的 SE（#66，E 档，7.36e-01）**

- Stata：`var(e.mpg) = 14.297844`，SE 2.7179562，故 `sigma = 3.781249`，
  delta method 得 `se(sigma) = 2.7179562/(2·3.781249) = 0.3593993`。
- westata：`/sigma = 3.781249`（点估计相对差 6.93e-09，正确），但 `bse = 0.09504725`。
  **`0.3593993 × (1/3.781249) = 0.09504726`** ——比例恰好 `1/sigma`（实测比值 1/sigma = 0.264514）。
- 归因：westata 用 `log(sigma)` 参数化求 Hessian（`tobit.py` 里 `theta[k] = log(sigma)`），
  得到的 `cov` 是 log 尺度的；`params` 换回 `sigma` 时**没有对协方差做 Jacobian 修正**
  `V_sigma = sigma² · V_log`。我独立用 sigma 参数化重算 OIM 得到 `se(sigma)=0.3593976`，
  用 log 参数化得到 `se(log sigma)=0.0950474`，westata 报的正是后者。
- 影响：`/sigma` 行的 t 值变成 3.781/0.0950 = **39.79**，Stata 是 3.781/0.3594 = **10.52**。
  任何对尺度参数的检验/区间都是错的。系数块不受影响（见 (b)）。

**(b) 缺陷二：`predict, stdf` 用 rss/N 代替 sigma²（#72、#73，E 档）**

- Stata：`stdf² - stdp² = sigma² = 14.297844`（每个观测都相同）。
- westata：`stdf² - stdp² = rss/N = 12.127405`（相对差 1.52e-01），所以 `stdf` 整体偏小 7.7%。
- 归因：`predict.py::_residual_variance` 先找 `e(rmse)`，找不到就退回 `rss/N`。
  tobit 记录的是 `e(sigma)`（=√14.2978=3.7812）而**不是** `e(rmse)`，
  于是 fallback 取到了删失模型的 `rss/N`——这不是创新方差（删失观测的残差被截断在限值处）。
- 影响：tobit 的**预测区间**宽度全错（偏小 7.7%），`stdp` 本身没问题（见 #70 的归因）。

**(c) `e(df_r)` 差 1（#67，E 档 1.39e-02）**

- Stata：`df_m = 2`（只数斜率），`df_r = N - df_m = 72`；**sigma 不消耗自由度**。
- westata：`df_resid = N - (斜率 + 常数) = 71`。
- 影响：tobit 的每个 t 统计量、p 值、t 区间都用 71 而不是 72。
  在 N=74 时 t(71)/t(72) 的分位数差约 1e-4 量级，属可见但小的偏差；
  但这是自由度口径错误，会随样本变小而放大。

**(d) 精度类差异（#63、#64、#65、#68–#71）**

- 系数点估计 4.06e-08（weight 1.21e-08、foreign 4.06e-08、_cons 7.76e-09）。
- 系数 SE 8.88e-05（weight 3.70e-05、foreign 8.88e-05、_cons 4.55e-05）。
- 归因（#52 的恒等式证明）：`sqrt(x'V_Stata x)` 复现 Stata 自己的 `stdp` 到 1.32e-15，
  所以 stdp 的公式没错；7.17e-05 的 stdp 偏差来自**V 的系数块**。
  westata 的 `_observed_information` 是**中心差分数值 Hessian（step=1e-5）**，
  Stata 用解析 observed information。我做步长扫描：
  step=1e-4 → se(weight)=8.6970e-4、1e-5 → 8.7016e-4（Stata 8.70169e-4）、1e-6 → 8.6988e-4、
  1e-7 → 8.0673e-4（崩）。**step=1e-5 恰好最接近 Stata，但仍有 ~1e-5 相对误差**。
- `predict, e(17,.)` / `ystar(17,.)` / `xb` 的 5.8e-09 / 1.5e-08 完全继承系数差（#51 证明 X·b 精确）。
- **注意**：这条与 BRIEF 提到的老结论"tobit e()/ystar() 5.8e-10"不一致——本次是 5.8e-09，
  差一个数量级（见下节）。

### 10. `tobit` 不支持 `predict, pr(a,b)`（#74）

- Stata：`predict x, pr(17,.)` 返回"未被删失的概率"，前三个值 `[0.9106, 0.6838, 0.9740]`。
- westata：`UnsupportedError: predict: 'tobit' supports ('xb','stdp','stdf','e','ystar','residuals'); got 'pr'`。
- 这是**能力缺口**：`margins, predict(pr(0,.))` 这类 Stata 工作流无法复现。
  记 fail，但注意它是"明确拒绝"而非"给错数"。

### 11. `pwcompare` 的多重比较校正与 `contrast h.`（#76、#77）

- Stata：`pwcompare foreign, mcompare(bonferroni)` 正常返回校正后 p 值；
  `contrast h.foreign`（Helmert）也可用。
- westata：两者都抛 `UnsupportedError`（`contrast.py::_check_adjust` / `_KNOWN_UNSUPPORTED_METHODS`）。
- 属**已文档化的能力缺口**，且拒绝而非静默给出未校正 p 值，方向是对的；记 fail 以计入覆盖。

### 12. `etable` vs `esttab`（#75）

- `esttab` 2.1.4 已安装（`C:\Users\zjl\ado\plus\esttab.ado`），**未**需要退化到 `estimates table`。
- 命令：`esttab m1 m2, b(%12.6f) se(%12.6f) nostar stats(N r2 r2_a, labels(...))`，
  两列 = `regress mpg weight foreign` 与同式 `vce(robust)`。
- 逐格比对 20 个数值单元（3 系数 + 3 SE + 3 统计量，×2 列），**全部相对差 0**。
- 三个已知差异（不影响数值）：
  1. 行标：Stata `_cons` vs westata `Intercept`（比对时按别名映射）；
  2. 行序：westata 用设计矩阵顺序（`Intercept` 在首），esttab 用估计顺序（`_cons` 在末）；
  3. `N` 的格式：esttab 按 `b()` 格式打印 `74.000000`，westata 打印整数 `74`，解析后都是 74.0。
- 这是**打印层**对比，`e(b)/e(V)` 的数值对比在其他条目里。

---

## 与旧结论的出入（`tests/conformance/registry.yml`，只读引用，不作证据）

| 旧条目 | 旧结论 | 本次实测 | 一致？ |
|---|---|---|---|
| `test` | E3，容差 1e-12，`test mpg weight` 差 1.6e-14 | 8 个变体全部 A 档（2.7e-16～1.3e-14）；**logit 的 Wald chi2 形式也已单独验证**（1.9e-10） | **一致**，并扩了覆盖 |
| `lincom` | E3，1e-11，1e-12 量级 | 4 个数值变体 A 档（6.8e-16～7.8e-10）；**但裸常数项被拒绝**（旧条目未提） | 数值一致；**新增能力缺口** |
| `nlcom` | E3，1e-6，"westata identical" | `exp()` 两种场景 A 档；**比值场景 SE 相对 Stata 1.996e-06，但 westata 更接近解析值** | **需要修正表述**：不是"identical"，是"westata 精确、Stata 有 2e-6 噪声" |
| `testnl` | E3，1e-6，"Joint constraints are accepted as comma-, whitespace- or parenthesised lists" | 括号形式 A 档；**StataNow 19.5 不接受逗号/空格形式**（r(198)） | **冲突**：旧条目的"accepted"只是 westata 侧的行为，未在 Stata 侧验证 |
| `lrtest` | E3，1e-3，"df == 0 返回 p = 1.0" | 嵌套场景 A 档；**Stata 对 df=0 直接 r(498) 报错**；**非嵌套（k 不同）Stata 接受并返回负 LR，westata 拒绝** | **冲突**：df=0 分支在 Stata 侧不存在 |
| `hausman` | E3，容差 5e-2 | 正定场景 2.60e-13；**非正定场景 Stata 报负 chi2、westata 夹到 0**；**`constant` 选项无对应** | 数值一致；**新增两条行为差异**，旧条目未记 |
| `contrast` | E3，2e-6 | r./g./a. 各两档全 A（3.8e-15～5.35e-14）；**已知 a. 是 low-vs-high，本次独立复核确认** | **一致** |
| `pwcompare` | E3，2e-6 | A 档 5.35e-14；**注意 Stata 把对比量放在 `r(b_vs)`，`r(b)` 是单元格均值** | 一致，补了取数口径 |
| `margins` | E3，1e-6，"Marginal effects 1.6e-7，standard errors 5.2e-6" | **未复现该等级**：regress 预测边际 A 档；但 logit AME 的 SE 偏 7.1e-3、`at()` 偏 5.4e-2，`over()`/`dydx()` 偏 1.9e-9 | **冲突**：旧结论的 5.2e-6 本次未重现，实测最差 5.4e-2 |
| `predict` | E3，1e-9，"e(a,b) and ystar(a,b) both 5.8e-10" | regress/logit 各类型 A 档；tobit `e()`/`ystar()` 5.78e-09（**差一个数量级**）；**tobit stdf 偏 7.7e-2**（旧条目未提） | **部分冲突**：数量级不同 + stdf 缺陷未被记录 |
| `predict, stdp/stdf` | E3，1e-10，"stdf is offered only for the linear families … so westata refuses it too" | 复核**成立**（logit/poisson 双方都拒绝）；但 **tobit 的 stdf 有量纲错误** | 部分一致；**tobit stdf 是旧条目遗漏** |
| `etable` | E3，5e-7，"gap 0.0" | 20 格全部 0 | **一致** |

**最值得注意的三条冲突**：
1. `margins` 的 E3 等级在本次**不成立**（logit AME SE 偏 5.4e-2），根因是固定步长数值微分；
2. `testnl` 的"逗号/空格形式被接受"只在 westata 侧成立，StataNow 19.5 报 r(198)；
3. `lrtest` 的 df=0 分支在 Stata 侧是**硬报错**，不是"返回 p=1.0"。

---

## 未覆盖

| 项 | 原因 |
|---|---|
| `margins, dydx()` 的 `vce(unconditional)` | westata 明确列为未实现；本次未构造可比的 Stata 命令 |
| `margins` 的 `expression()` / `atmeans` 与因子组合 | 同上，westata 抛 `UnsupportedError`；未做系统覆盖 |
| `margins, at()` 对因子变量的组合（`at(foreign=(0 1))`） | 需要 `margins` 的因子展开层，westata 无对应；只测了连续变量 `at()` |
| `pwcompare` 的 Holm/Šidák/Tukey 等 | 只测了 bonferroni 一种（验证拒绝行为），未逐一测全 |
| `contrast m.`（multiple-comparison） | 同上 |
| `esttab` 的 `star()` / `mtitles` / 多模型 4 列以上 | 只测了 `nostar` + `stats()` 的两列场景（打印层足够） |
| `predict` 的 `xb`/`stdp` 在 `xtreg`/`reghdfe` 后 | 属 panel/linear 领域，本领域只测了 regress/logit/poisson/tobit |
| `test` 对 `ologit`/`nbreg` 等 z 族 | 只测了 logit 一个 z 族代表 |
| `hausman` 用 `suest` 实现 | westata 无 `suest`；只在 `xtreg, fe/re` 上测 |
| `tobit` 的 `ul()` 上删失场景 | 只用 `ll(17)` 左删失；上删失与双侧删失未测 |
| `tobit` 的 `margins, predict(e(17,.))` | westata 的 `DEFAULT_PREDICT["tobit"] = "e"` 存在，但未与 Stata 的 `margins, predict(e(0,.))` 对比 |

另外：本领域**没有** `status=error` 或 `status=skip` 条目——所有想跑的路径都跑到了
（westata 拒绝的路径记成 `fail` 并附异常原文，按要求没有为了跑通去改参数）。
