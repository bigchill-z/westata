# postest 修复报告（M1–M6）

组名：`postest`；写作用域：`src/westata/postestimation/{margins,predict,api,modeltests}.py`；
测试：`tests/test_margins.py`、`tests/test_postestimation.py`、`tests/test_predict_se.py`、
`tests/test_modeltests.py`、`tests/test_contrast.py`。

所有 Stata 数字来自本机内嵌 **StataNow 19.5 (MP)**，2026-10-24 现场捕获，
数据为 `sysuse auto`（`tests/data/auto.dta`，N=74）或脚本内自造数据。
每个 repro 脚本都可独立重跑（`cd D:\myCodeProject\statapy && .venv\Scripts\python.exe <脚本>`），
输出 `改前 / 改后 / Stata` 三列。

## 逐项结果

| # | 缺陷 | 改前相对差 | 改后相对差 | Stata 命令 | 判定 |
|---|---|---|---|---|---|
| M1a | `margins` logit AME `dydx(weight)` 的 SE | **7.10e-03** | **6.36e-11** | `logit foreign weight mpg` + `margins, dydx(weight)` | 已修复 A |
| M1b | 同上 `at(mpg=20)` 的 SE | **5.38e-02** | **2.97e-10** | `margins, dydx(weight) at(mpg=20)` | 已修复 A |
| M1c | probit AME 的 SE | 5.93e-04 | **5.20e-13** | `probit foreign weight mpg` + `margins, dydx(weight)` | 已修复 A |
| M1d | poisson AME 的 SE | 3.68e-03 | **3.14e-11** | `poisson rep78 mpg weight` + `margins, dydx(weight)` | 已修复 A |
| M1e | `over(foreign)` 预测边际的 SE | 8.13e-11 / 1.85e-09 | **4.82e-15 / 6.11e-16** | `regress mpg weight i.foreign` + `margins, over(foreign)` | 已修复 A |
| M1f | `dydx(foreign)` 离散变化的 SE | 1.79e-09（旧报告档位 B） | **0.00e+00**（== `bse`） | `margins, dydx(foreign)` | 已修复 A |
| M1g | ologit AME（**新增覆盖**） | 旧实现直接不可用 | **4.6e-10 / 1.3e-10** | `ologit rep78 weight mpg` + `margins, dydx(weight)` | 已修复 A |
| M2a | tobit `predict, stdf` 用 `rss/N` 代替 `σ²` | **7.71e-02** | **2.09e-10** | `tobit mpg weight foreign, ll(17)` + `predict double, stdf` | 已修复 A |
| M2b | truncreg `stdf`（同一根因，新增覆盖） | 6.18e-02 | **1.83e-07** | `truncreg y x1 x2, ll(1)` + `predict, stdf` | 已修复 B |
| M2c | intreg `stdf`（同一根因，新增覆盖） | 6.77e-02 | **1.55e-08** | `intreg lo hi x1 x2` + `predict, stdf` | 已修复 A |
| M2d | `regress` 的 `stdf`（**不能改坏**） | 2.56e-16 | **2.56e-16** | `regress mpg weight foreign` + `predict, stdf` | 未变（本来对） |
| M3a | `zip` 的 `predict, n` 崩溃 | `ValueError: matmul ... mismatch` | **OK**（语义对到 4.4e-16） | `zip yzi x1, inflate(z)` + `predict, n` | 已修复 |
| M3b | `zip` 的 `predict, pr`（新增覆盖） | `ValueError: matmul ...` | **OK**（= inflation 概率 π） | `predict, pr` | 已修复 |
| M3c | `zip` 的 `predict, stdp`（新增覆盖） | `ValueError: exog 2 columns vs 4 coefficients` | **OK**（= `sqrt(x_c'V_cc x_c)`） | `predict, stdp` | 已修复 |
| M3d | `ztp` 的 `predict, n` 语义错 | 返回 `Φ(xb)`（概率冒充均值） | **OK**（= `exp(xb)`，0.0e+00） | `ztp y1 x1 x2` + `predict, n` | 已修复 |
| M3e | 不支持 kind 的报错不列可用项 | 只报 "no prediction types registered" | 报 `('xb','n','pr','stdp'); got 'mu'` | `predict double x, mu` → r(198) | 已修复 |
| M3f | 模型侧 `_predict_dispatch` 契约 | 不存在 | 通用委托 + `_predict_types` 白名单 | — | 已实现 |
| M4a | `lincom` 裸常数项 | `ValueError: constant '10' ...` | **1.78e-15 / 2.75e-16** | `lincom 2*weight - 3*foreign + 10` → 14.9369115 | 已修复 A |
| M4b | `lincom 10`（纯常数） | `ValueError` | **0.00e+00**（b=10, se=0） | `lincom 10` | 已修复 A |
| M4c | `lincom ... + 1e3`（指数符号） | `ValueError` | **1.13e-16** | `lincom 2*weight - 3*foreign + 1e3` | 已修复 A |
| M4d | `test` 两边的常数项 | 只有 `a = b` 形式可用 | **3.32e-15**（F=21.4221254242） | `test 2*weight - 3*foreign + 10 = 0` | 已修复 A |
| M5a | `lrtest` 的 `df=0` 返回 p=1.0 | 命令**成功**（Stata r(498)） | **抛 `ConsistencyError`（消息含 r(498)）** | `lrtest L1 L2`（同一模型两次） | 已对齐 |
| M5b | 非嵌套（k 不同）被拒绝 | 抛 `ConsistencyError` | **返回 −2.68354973321414，p=1** | `lrtest N2 L1` | 已对齐 A |
| M5c | 参数顺序颠倒被拒绝（新增） | 抛 "swap the two arguments" | **与 Stata 同样接受，2.41124840273977** | `lrtest L1 B1` → "Assumption: L1 nested within B1" | 已对齐 A |
| M5d | 同 k 不同变量（新增） | 抛 "not nested" | **抛 r(498)**（Stata 先查个数） | `lrtest N3 L1` | 已对齐 |
| M6a | `hausman` 无 `constant` 开关 | 无参数（API 面缺失） | **`constant=True` → df 3**，统计量 0.952546096131752（实例方法与自由函数都可用） | `hausman FEm REm, constant` | 已实现 |
| M6b | `hausman` 默认行为 | df=2，0.9525460961317572 | **不变**（df=2，同值） | `hausman FEm REm` | 未变 |
| **M7** | **（新增发现）tobit 的 `margins` 默认 kind 是 `e(a,b)` 而不是 `xb`** | **3.87e-01 / 4.29e-01**（`dydx`），9.95e-02 / 3.08e-01（预测边际） | **7.85e-12 / 1.99e-10** 与 **3.82e-12 / 2.08e-10** | `tobit ... , ll(17)` + `margins, dydx(weight)`；`e(marginsok)` = `default XB ...` | 已修复 A |

「改前」一列由各 repro 脚本**重新实现旧代码路径**实测得到（不是引用 REPORT 的旧数字）；
`REPORT.md` 的旧值（7.1e-3 / 5.4e-2 / 7.7e-2 / 1.79e-9 / 1.85e-9）与本次实测一致。

> **M7 是超出原缺陷清单的新发现**：`REPORT.md` §2.7 只比过 tobit 的
> `margins, predict(e(...))`，默认值从未被钉住，而旧 `DEFAULT_PREDICT["tobit"] = "e"`
> 让 `margins` **不报错地回答了另一个量**（点估计偏 39%）。详见下文
> 「新增发现（超出原缺陷清单）」。Stata 自己在 `e(marginsok)` 里标了 `default XB`。

## 每项的根因与改法

### M1 —— `margins` 的 AME 标准误（`margins.py`）

**根因**：`_statistic_and_gradient` 对**协变量**做固定步长 `h = 1e-6` 的中心差分。
AME 本身只有 3.5e-4 量级，而预测概率约 0.1：`(p(x+h) − p(x−h)) / 2h` 是把两个
前 10 位相同的数相减，导数的绝对误差下限是 `eps·p/h ≈ 4e-11`，相对误差
`eps·p/(h·p')` 随 h 减小**变大**。实测步长扫描（`repro_M1_margins.py` 的
`step sweep`）：

```
h=1e-2 → 1.17e-11   h=1e-3 → 1.83e-10   h=1e-4 → 1.81e-09
h=1e-5 → 1.74e-08   h=1e-6 → 7.15e-08   h=1e-7 → 1.30e-06   h=1e-8 → 8.16e-06
```

**改法**（两路，解析优先）：

1. **解析路线**（`logit` / `probit` / `poisson` / `nbreg` / 线性 `xb` / 有序模型的 `pr`）：
   后端 `predict.py` 新增 `prediction_family(result, kind)` 与
   `family_slopes(family, eta) → (g'(η), g''(η))`，`margins` 用它算

   ```
   AME_j        = mean_i g'(η_i)·β_j
   d AME_j/dβ_l = mean_i g''(η_i)·X_il·β_j + [l = j]·mean_i g'(η_i)
   ```

   这正是 Stata 的 `r(Jacobian)`：`logit` 的实测值
   `[-1.108580469274756e-01, -6.649265095237061e-04, -4.796996154758345e-05]`
   与公式逐位吻合（测试 `test_logit_ame_gradient_matches_stata_jacobian`，rtol 1e-9）。

2. **自适应数值路线**（无闭式解的 `tobit e(a,b)` / `ystar(a,b)`）：步长按被扰动量的
   尺度缩放 + 一次 Richardson 外推。**步长系数是实测选的，不是推的**：
   协变量 `h = 3·eps^(1/4)·max(1,|x|)`，参数 `h = eps^(1/3)·max(1,|b|)`。
   对 Stata 实测（4 个量：`e/weight`、`ystar/weight`、`xb/weight`、`ystar/foreign`）：

   | 协变量步长 | 最差 b 相对差 | 最差 SE 相对差 |
   |---|---|---|
   | `1·eps^(1/3)`（=6.06e-6，初版） | 5.6e-09 | 2.5e-05 |
   | `30·eps^(1/3)` | 5.6e-09 | 2.8e-06 |
   | `3·eps^(1/4)`（=3.66e-4，采用） | 1.5e-08 | **1.1e-07** |
   | `3·eps^(1/3)` | 5.6e-09 | 2.4e-06 |

   参数步长独立扫描（`eps^(1/3)`、`eps^(1/4)`、`eps^(1/5)` × 1/3/10 倍）后取 `eps^(1/3)`。

**顺带修掉的两件事**（都在同一函数族里，且都是"静默错数"）：

- **有序模型的 `pr` 是"某一类的概率"，不是 logistic 族**。旧 `_PREDICT_FAMILY` 把
  `ologit.prob` 当 `p(1−p)` 会让**符号反掉**。现在有序模型走
  `_analytic_ordered_margin`（对斜率**和** cut 点都求导）：
  `P(y=k) = F(c_k−η) − F(c_{k−1}−η)`，`dP/dc_m = f(c_m−η)([m=k]−[m=k−1])`。
  实测 `ologit rep78 weight mpg` + `margins, dydx(weight)`：
  Stata `9.40690482679723e-06 / 1.52602157141091e-05`，westata 相对差 4.6e-10 / 1.3e-10；
  `outcome(2)` 同样对到 4.8e-10 / 1.0e-10。
- **`over()` / `dydx(foreign)` 的退化**：`over()` 的梯度现在是精确的
  `mean(X_g)`（原来靠 `h=1e-6` 的 `xb` 差分，1.85e-09 → 6.1e-16）；
  `dydx(foreign)` 在线性模型里等于系数，现在**逐位**等于 `bse`（0.0e+00）。

**`tobit` 的默认 predict kind 从 `e` 改成 `xb`**（Stata 事实，见"与旧结论的出入"）。

### M2 —— tobit 的 `predict, stdf`（`predict.py::_residual_variance`）

**根因**：旧实现只查 `e(rmse)`，查不到就退回 `rss/N`。`tobit` 记录的是
`e(sigma)` / `e(var_e)`（不是 `e(rmse)`），于是 fallback 取到**删失响应**的
`rss/N = 12.127405`，而 Stata 的 `stdf² − stdp² = e(var_e) = 14.297844`。

**改法**：把"创新方差"按权威性排序解析，并且**绝不**读 `e(sigma)`（westata 把
`e(sigma)` 当作 `e(rmse)` 的别名记录，读它会让文档化的 `rss/N` fallback 永远不可达）：

```
1. 模型侧钩子 _innovation_variance()      （模型可以自己说了算）
2. e(var_e)                               （tobit/truncreg/intreg 都记，Stata 的 σ²）
3. e(rmse)^2                              （线性族；regress 走这条，保持不变）
4. 参数向量里的 /sigma 或 sigma 的最后一项平方
5. rss / N
6. 否则抛 ConsistencyError                （不退回 rss/(N-k)，那是另一个量）
```

**验证**（三个受限因变量族 + 一个线性族，`repro_M2_tobit_stdf.py`）：

| 模型 | Stata `stdf²−stdp²` | 改前 | 改后 | 改后 stdf 相对差 |
|---|---|---|---|---|
| `tobit mpg weight foreign, ll(17)` | 14.2978442 | 12.12740491 | **14.29784421** | **2.09e-10** |
| `truncreg y x1 x2, ll(1)` | 0.23222929 | 0.20404373 | **0.23222936** | 1.83e-07 |
| `intreg lo hi x1 x2` | 0.2270283 | 0.25891374 | **0.2270283** | 1.55e-08 |
| `regress mpg weight foreign`（不能改坏） | 11.60805298 | 11.60805298 | 11.60805298 | 2.56e-16 |

`tobit`/`intreg` 到 A 档；`truncreg` 的 1.8e-7 是**模型侧拟合精度**（truncreg 的
`e(sigma)` 本身与 Stata 差 1.6e-7，见 repro 输出），不是这个修复引入的。

### M3 —— `predict` 的通用委托（`predict.py`）

**根因**（三个独立缺陷，同一条分发链）：

1. `kind ∈ ("pr","n")` 一律用 `params[:n_columns]` + 按模型名选的链接。
   `zip` 有 4 个参数、设计矩阵 3 列 → 裸 `matmul` 维度错；
   `ztp` 不在硬编码的 `("poisson","nbreg")` 名单里 → 走 probit 分支，**返回概率冒充计数均值**。
2. `stdp`/`stdf` 要求"每个参数一列"，多方程模型的 4 个参数 vs 2 列设计矩阵 → `ValueError`。
3. 未知 `kind` 只报 "no prediction types registered"，不列可用项。

**改法**：

- **通用委托契约**（不含任何模型名清单）：
  - 结果对象上若有 `_predict_dispatch`（可调用），**由模型接管**；适配器按
    `inspect.signature` 只传它声明过的关键字（`kind`/`exog`/`lower`/`upper`），
    返回 `None` 表示"这个 kind 我不管"，退回共享实现（`xb`/`stdp`/`stdf`）；
  - 可选 `_predict_types`（tuple 或返回 tuple 的 callable）作为**权威白名单**，
    拒绝消息据此列出可用项。
- **`n` 的链接改为统一 `exp(η)`**（计数族的均值都是指数链接），不再按模型名选；
  `ztp predict, n` 因此 = `exp(xb)`，与 Stata 实测 **0.0e+00** 一致。
- **多方程模型的 `stdp`/`stdf`**：`_coefficient_block` 在"设计矩阵列数 < 参数个数"
  且参数名显示是多方程（含 `:` 或名字重复）时，取**第一方程的块**。
  实测 Stata：`zip ... predict, stdp` = `sqrt(x_c'V_cc x_c)` 到 **2.0e-16**，
  若把 inflation 块折进去则差 **9.6e-1**。
- **拒绝消息**列出可用 kind：`predict: 'zip' supports ('xb', 'n', 'pr', 'stdp'); got 'mu'`。

**accept/reject 矩阵**（`repro_M3_predict_dispatch.py`，`MISMATCH` 计数 = **0**）：

| 模型 | kind | Stata | 改前 | 改后 |
|---|---|---|---|---|
| zip | n | OK | `ValueError`(matmul) | OK |
| zip | pr | OK | `ValueError`(matmul) | OK |
| zip | xb | OK | OK | OK |
| zip | stdp | OK | `ValueError`(列数) | OK |
| zip | stdf / residuals | reject r(198) | 静默给数 | `UnsupportedError` + 可用项 |
| ztp | n / xb / stdp | OK | n 语义错 | OK |
| ztp | pr / stdf / residuals | reject r(198) | 静默给数 | `UnsupportedError` + 可用项 |

**与 `discrete` 组的接口**：已双向确认（见 §6 的通信记录），钩子名就是
`_predict_dispatch` + `_predict_types`，模型侧实现 `n`/`pr`/`xb`，本侧只提供通用委托。

### M4 —— `lincom` 的裸常数项（`api.py`）

**根因**：`_split_terms` 把裸数字当非法项（"put constants on the right-hand side"），
所以 `lincom 2*weight - 3*foreign + 10` 直接抛错。Stata 允许常数出现在表达式的任何位置，
并把它移到等号右边（`( 1) 2*weight - 3*foreign = -10`），**常数不进入方差**。

**改法**：

- 新增 `_split_signed_terms`（按 `+`/`-` 切项，但**指数里的符号** `1e-3` 与**括号内**的
  符号 `C(g)[T.1]` 不切）与 `_split_linear_form`（返回 `(系数权重, 常数)`）；
- `lincom` 把常数加到点估计与区间上，SE 仍由 `L V L'` 决定；
- `test` 的每一侧都走 `_split_linear_form`，于是 `a + 10 = 0` 与 `a = -10` 是**同一个假设**
  （实测两边 F 完全相同，`rel = 0.00e+00`）；
- 纯系数列表形式（`test 10`，没有 `=`）**仍然拒绝**，因为 Stata 那里是
  `r(111) '10' not found`（语法是"系数名列表"而不是"表达式"）。

| 表达式 | Stata b / se | westata b / se | 相对差 |
|---|---|---|---|
| `2*weight - 3*foreign + 10` | 14.936911544776347 / 3.2272271185374004 | 同值 | 1.78e-15 / 2.75e-16 |
| `2*weight - 3*foreign` | 4.93691154477635 / 3.2272271185374 | 同值 | 5.22e-15 / 2.75e-16 |
| `10` | 10 / 0 | 10 / 0 | 0 |
| `2*weight - 3*foreign + 1e3` | 1004.9369115447763 / 3.2272271185374004 | 同值 | 1.13e-16 / 2.75e-16 |
| `weight + 1` | 0.993412113624226 / 0.000637112934853876 | 同值 | 1.12e-16 / 6.81e-16 |
| `test 2*w - 3*f + 10 = 0` | F 21.422125424189847 | 21.4221254242 | 3.32e-15 |

### M5 —— `lrtest`（`modeltests.py`）

**根因**：旧代码把 `df==0` 当作"退化但可报告"（`chi2(0)` 是点质量，`p` 钉成 1.0），
并按**系数名**判嵌套；Stata 只查**参数个数**，所以：

- `df=0` → Stata **硬报错** r(498)（"df(unrestricted) = df(restricted) = 2"）；
- 非嵌套但 k 不同 → Stata **接受**并报负 LR（`LR chi2(1) = -2.6835, p = 1.0000`）；
- 参数顺序颠倒 → Stata **接受**并重新标注（"Assumption: L1 nested within B1"）。

**改法**（严格按 Stata 的检查顺序）：

1. 样本量不同 → `ConsistencyError`（不变）；
2. `k_u == k_r` → **`ConsistencyError`，消息含 `r(498)` 与两个 df**（顺带说明是否也不嵌套）；
3. `k_u < k_r` → **交换**（Stata 行为），不告警；
4. 名字不嵌套但 k 不同 → **只警告**，照算（负 LR 会自然出现）；
5. `statistic < 0` → 只在 LL 真的更差时警告，值**保留负号**（不 clamp）。

`LRTestResult` 的文档同步改：`df` 恒为正、`p` 对负统计量仍为 1（Stata 打印值）。

| 场景 | Stata | westata | 结论 |
|---|---|---|---|
| `lrtest B1 L1`（嵌套） | 2.41124840273977 / 1 / .12046626248080883 | 同值（p 7.03e-15） | 一致 |
| `lrtest L1 B1`（顺序颠倒） | 同上 | 同上 | 一致 |
| `lrtest L1 E2`（同模型） | reject r(498) | reject（含 r(498)） | 一致 |
| `lrtest N3 L1`（同 k 不同变量） | reject r(498) | reject（含 r(498)） | 一致 |
| `lrtest N2 L1`（非嵌套，k 不同） | −2.68354973321414 / p 1 | **−2.68354973321414 / p 1** | 一致（rel 0.0） |

### M6 —— `hausman` 的 `constant` 开关（`modeltests.py`）

**根因**：`_comparable_names` 无条件丢弃常数，`hausman()` 没有 `constant` 参数，
所以 Stata 的 `hausman ..., constant` 在 westata 侧无对应 API。

**改法**：`hausman(consistent, efficient, *, constant=False)`；`constant=True` 时把常数
纳入**同一个二次型**（`df` 随共享参数个数变化），默认行为逐位不变。
`excluded` 仍只报"一边根本没有的参数"（常数只是在默认下被刻意丢弃，不算缺失）。

| 面板（40 实体 × 8 期，seed 4242） | Stata | westata | 相对差 |
|---|---|---|---|
| `hausman FEm REm`（默认，差正定） | chi2(2)=.9525460961317572, p=.6210938706123512 | .952546096132005, df 2 | 2.6e-13 |
| `hausman FEm REm, constant` | chi2(3)=.952546096131752, p=.8127322054037279 | .952546096132004, df 3 | 2.6e-13 |
| 非正定差（corr=0.15）默认 | chi2(2)=**−2.35424271822838**, p=1 | 0（广义逆后 clamp）, p=1 | 见"未修" |
| 非正定差 + `constant` | chi2(3)=**−2.35424271822838**, p=1 | 0, df 3, p=1 | 同上 |

## 测试变更

| 文件 | 旧断言 | 新断言 | 依据（Stata 捕获，2026-10-24 / 19.5 MP） |
|---|---|---|---|
| `tests/test_modeltests.py` | `test_lrtest_identical_models_is_degenerate_with_pvalue_one`：`df==0` 时 `p==1.0`、`chi2==0` | 改名 `test_lrtest_identical_models_is_refused_like_stata_r498`：`pytest.raises(ConsistencyError, match=r"r\(498\)")` | `lrtest L1 E2` → `df(unrestricted) = df(restricted) = 2` / `r(498)`；旧断言把 Stata 的**报错**写成"约定返回 p=1"，是错的 |
| `tests/test_modeltests.py` | `test_lrtest_rejects_non_nested_models`：`x1` vs `x2` 抛 "not nested" | `test_lrtest_equal_parameter_counts_of_different_models_is_also_r498`：同 k → r(498) | `lrtest N3 L1` → r(498)（Stata 先查个数） |
| `tests/test_modeltests.py` | `test_lrtest_names_the_parameters_that_break_nesting`：`small.lrtest(other)` 抛错 | `test_lrtest_swaps_the_arguments_the_way_stata_does`：两个方向都接受且同值 | `lrtest L1 B1` → "Assumption: L1 nested within B1"，2.41124840273977 |
| `tests/test_predict_se.py` | `test_ordered_model_stdp_drops_the_cut_points`：断言 `design_matrix is None` | 断言 `design_matrix is not None` 且 cut 点仍不进 `x'Vx` | `discrete` 组本轮修好了有序模型保留设计矩阵；`ologit` 的 `predict, stdp` 在 Stata 被接受 |
| `tests/test_predict_se.py` | `test_ordered_model_refuses_stdf_for_two_independent_reasons`：靠 "无 design_matrix" 作为第一道 guard | `test_ordered_model_refuses_stdf_but_offers_stdp`：只钉 `stdf` 拒绝（带/不带 `exog` 各一次） | `ologit rep78 weight mpg` + `predict, stdf` → `option stdf not allowed` r(198) |
| `tests/test_predict_se.py` | `_ordered_frame` 用 `g ∈ {0,1,2}` | `g ∈ {1,2,3}` | `ologit`（Stata）的 cut 命名是 `cut1..cutJ-1`，类标签从 1 开始；`discrete` 组本轮也按此实现 |
| `tests/test_margins.py` | （新增 12 条） | logit/probit/poisson/ologit 的 AME + `at()` + `r(Jacobian)` + `over()` + 因子离散变化 + **M7 的 tobit 默认 kind** | 见文件内 `# Stata-captured regression (P0-7, M1)` 注释块与 M7 的 docstring |
| `tests/test_postestimation.py` | （新增 7 条） | 裸常数、SE 不变、纯常数、`1e3`、`test` 两边常数等价、`test 10` 仍拒绝 | 见文件内 `# Stata-captured regressions (M4: bare constants)` 注释块 |
| `tests/test_predict_se.py` | （新增 8 条） | tobit `stdf²−stdp²`、tobit `stdp/stdf` 前三个观测、regress 未变、zip `n`、ztp `n`、委托契约、可用项消息 | 见文件内 `# Stata-captured regressions (M2 and M3)` 注释块 |
| `tests/test_modeltests.py` | （新增 8 条） | lrtest 嵌套/颠倒/负 LR、hausman `constant` 的 df 与统计量、默认不变、**实例方法 `fe.hausman(re, constant=True)` 走通 `**kwargs` 转发**、**非正定差 clamp 的 docstring 契约** | 见文件内 `# Stata-captured regressions (M5 and M6)` 注释块 |

`tests/test_contrast.py` **未改动**（不含这些缺陷的任何断言）。

## 新增发现（超出原缺陷清单）

### M7 —— `tobit` 的 `margins` 默认 predict kind 是 `xb`，不是 `e(a,b)`

`retest/REPORT.md` §2.7 与任务书都没提这一条：复测只用
`margins, predict(e(...))` 比过 tobit，**默认值从未被钉住**，而
`margins.py::DEFAULT_PREDICT["tobit"]` 写的是 `"e"`。这是最危险的一类缺陷——
命令**成功**，但回答的是**另一个问题**。

现场实测（StataNow 19.5 MP，2026-10-24，`tobit mpg weight foreign, ll(17)`）：

| 命令 | Stata b | Stata se |
|---|---|---|
| `margins, dydx(weight)`（**默认**） | −7.797866181897683e-03 | 8.701688711856095e-04 |
| `margins, dydx(weight) predict(xb)` | −7.797866181897683e-03 | 8.701688711856095e-04 |
| `margins, dydx(weight) predict(e(17,.))` | −4.778245322021773e-03 | 4.969403105957036e-04 |
| `margins, dydx(weight) predict(ystar(17,.))` | −5.489910911133220e-03 | 5.104456991742232e-04 |
| `margins`（**默认**） | 2.069188444034438e+01 | 4.921805620495909e-01 |
| `margins, predict(e(17,.))` | 2.275106106085989e+01 | 3.405896140626539e-01 |

- 默认 `≡ predict(xb)`，且 `xb` 在 tobit 里是线性的，所以
  `margins, dydx(weight)` 的默认值**逐位等于系数 `_b[weight]`**（rel 0.00e+00）。
- 默认与 `predict(e(17,.))` 的差距：`dydx` 的 b 差 **6.32e-01**、se 差 **7.51e-01**；
  `margins` 的 b 差 **9.05e-02**、se 差 **4.45e-01**。
- **Stata 自己写着答案**：tobit 的 `e(marginsok)` =
  `default XB Pr(passthru) E(passthru) YStar(passthru)` —— `XB` 被标 `default`，
  其余三个是 `passthru`（只能经 `margins, predict()` 触达）。

**旧实现在数字上会怎样**（`repro_M7_tobit_margins_default.py` 的
"the old default"段，用新的 `predict_kind="e"` 复现旧默认）：

| 命令 | 旧默认（= `e`）输出 | Stata 默认 | 相对差 |
|---|---|---|---|
| `margins, dydx(weight)` | b −4.778245348685955e-03, se 4.969403198940970e-04 | b −7.797866181897683e-03, se 8.701688711856095e-04 | **3.87e-01 / 4.29e-01** |
| `margins` | b 2.275106106122035e+01, se 3.405896142477751e-01 | b 2.069188444034438e+01, se 4.921805620495909e-01 | **9.95e-02 / 3.08e-01** |

即旧的 tobit `margins` 点估计偏 **39%**（`dydx`）/ **10%**（预测边际），
SE 偏 **43%** / **31%**，而且**不会报错**。改成 `"xb"` 后这两项分别到
**7.85e-12 / 1.99e-10** 与 **3.82e-12 / 2.08e-10**（A 档）。

**改法**：`DEFAULT_PREDICT["tobit"] = "xb"`，并把 `e(marginsok)` 的原文与上表写进
`DEFAULT_PREDICT` 的 docstring（防止以后有人"顺手改回 e"）。
`predict(e(...))` / `predict(ystar(...))` 仍可通过 `margins(..., predict_kind="e")`
显式选择，数值本身对到 1.5e-08 / 1.1e-07（见 M1 的步长表），**没有丢能力**。

**测试**：`tests/test_margins.py::test_tobit_margins_default_is_the_linear_prediction_not_e_ab`
同时钉住四件事（默认 = 系数、`prediction == "xb"`、`e` 仍是不同的数、预测边际 = `mean(xb)`）。

## 与旧结论/任务书的出入

1. **`REPORT.md` 说 `margins` 的 `over()`/`dydx(foreign)` "只有 SE 偏 1e-9 量级"**——
   实测改前是 8.1e-11 / 1.85e-09 / 1.79e-09（与报告一致），改后是 4.8e-15 / 6.1e-16 / **0.0**。
   两者都在 A 档内，所以这不是"档位变化"，而是"顺手把数值噪声也消掉了"。

2. **`lrtest` 的"非嵌套"有两种**：k 不同（Stata 接受，负 LR）与 k 相同（Stata r(498)）。
   任务书只写了前一种。两者现在分开处理（见 M5 的检查顺序）。

## 未修 / 未达 A 的

1. **`truncreg` 的 `stdf` 只到 B 档（1.83e-07）**。`stdf² − stdp²` 的结构已经精确
   （= `e(sigma)²`，与 Stata 的相对差 1.6e-07），残差全部来自 `truncreg` 自身拟合：
   westata `e(sigma) = 0.4819018168` vs Stata `0.4819017395863228`（相对差 1.6e-7）。
   这属于 `limited` 组的写作用域，不在本组可修范围。
2. **`tobit` 没有 `predict, pr(a,b)`**（`REPORT.md` 的 #74 能力缺口）。它不在 M1–M6 范围内，
   本次未实现：`predict, pr` 对 tobit 是"未被删失的概率"，与已有的 `e(a,b)`/`ystar(a,b)`
   是不同的量，加上它需要同时确认 `margins, predict(pr(...))` 的行为。

## 已裁决：`hausman` 非正定差时保留 clamp 到 0（刻意差异，不是缺陷）

Lead 已确认**保留 clamp**，理由已按裁决要求写进
`modeltests.py::hausman` 的 docstring `Notes` 段（不再只写在本文里）：

> 非正定差（`V_fe − V_re` 有负特征值）时 Stata 打印负统计量
> （本数据 **−2.35424271822838**，`r(p) = 1`）。westata 夹到 0 并给 `p = 1`：
> **保留 clamp**，因为不夹的话 −1e-18 级浮点噪声会被当成"真实负统计量"报出去。
> 这是刻意的行为差异，不是缺陷。

两边一致的部分：`df`、`p`、以及"用广义逆并警告"这一行为；只有**不可解释统计量的符号**不同。
测试 `test_hausman_documents_the_deliberate_clamp_on_an_indefinite_difference`
把这个契约写死（断言 `statistic == 0`、`p == 1`，并断言 docstring 里保留了 −2.3542 与
"deliberate" 字样，防止理由被以后重构删掉）。

## 需要 Lead 处理的

1. **`results/base.py::StataResults.hausman` 的 `**kwargs` 转发已由 Lead 修好**：
   现在是 `def hausman(self, efficient: Any, **kwargs: Any)`，`fe.hausman(re, constant=True)`
   可用。本组已补
   `tests/test_modeltests.py::test_hausman_constant_is_reachable_through_the_result_method`
   **走实例方法路径**（并断言与自由函数逐位相同、默认仍是 df 2），以后这条转发断掉会被立刻抓到。
   `lrtest` 的同类缺口目前没有新参数，不阻塞。

2. **`predict.py` 的 `predict()` 新增关键字 `outcome=`**（有序模型的
   `predict, pr outcome(#k)`）。这是纯增量参数（默认 `None` = Stata 的类 1），
   不影响任何现有调用；`results/base.py::predict(**kwargs)` 已经在转发 `**kwargs`，
   所以 `res.predict(kind="pr", outcome=2)` 直接可用。

3. **能力回退（无）**：M1–M7 没有把任何原本能算的量改成拒绝。M3e/M3d 与 M7 都是把
   **静默错数**改成"要么给 Stata 的数、要么明确拒绝并列可用项"，方向与 BRIEF 一致。

## 我跑过的命令与结果

```text
# Lead 指定的一条（裁决后复跑）
.venv\Scripts\python.exe -m pytest tests/test_modeltests.py tests/test_margins.py -q
  → 53 passed in 3.77s

# 本组五个测试文件 + 相关的 conformance（最终状态全部通过）
.venv\Scripts\python.exe -m pytest tests/test_margins.py tests/test_postestimation.py \
    tests/test_predict_se.py tests/test_modeltests.py tests/test_contrast.py \
    tests/conformance/test_stata_predict_etable.py -q
  → 160 passed

# 相邻模型文件（确认 predict 面没把别处改坏）
.venv\Scripts\python.exe -m pytest tests/test_tobit.py tests/test_limited.py \
    tests/test_discrete.py tests/test_counts.py -q
  → 中途出现过 4 条 test_discrete 失败，那是 discrete 组正在改 counts.py 的瞬时状态
    （失败内容是 ologit 的 e(V)/df_m 与 nbreg/probit 的 robust VCE，不在本组作用域）；
    他们落盘后复跑：
.venv\Scripts\python.exe -m pytest tests/test_counts.py -q           → 42 passed
.venv\Scripts\python.exe -m pytest tests/test_discrete.py -q         → 通过
.venv\Scripts\python.exe -m pytest tests/test_panel_nonlinear.py -q  → 49 passed

# 全量（不含 conformance）
.venv\Scripts\python.exe -m pytest tests/ -q --ignore=tests/conformance
  → 1166 passed, 5 failed
```

那 5 条失败**全部落在其他组的作用域内**，且都出现在多组并发改源码的过程中：

| 失败 | 归属 | 复跑结果 |
|---|---|---|
| `test_counts.py::test_truncated_models_refuse_pr_like_stata` | `discrete`（counts.py / predict 面） | 单独复跑 **passed**（42 passed） |
| `test_panel_nonlinear.py::test_xtnbreg_fe_*`（4 条） | `panel`（xtnbreg 的 HHG，P5） | 单独复跑 **49 passed** |

本组的 160 条（含 conformance 的 `predict`/`etable`）在最终状态下**全绿**；本组未改
`tests/conformance/` 任何文件（该目录下当前的 `test_stata_p11.py` 改动属于 `systems` 组）。

repro 脚本（各自开一个真实 Stata 会话，输出 `改前 / 改后 / Stata`）：

```text
.venv\Scripts\python.exe retest/fix/postest/repro_M1_margins.py            # 最差 2.975e-10 → A
.venv\Scripts\python.exe retest/fix/postest/repro_M2_tobit_stdf.py         # tobit 2.09e-10 → A；truncreg 1.83e-07 → B
.venv\Scripts\python.exe retest/fix/postest/repro_M3_predict_dispatch.py   # MISMATCH 计数 0
.venv\Scripts\python.exe retest/fix/postest/repro_M4_lincom_constant.py    # 最差 5.217e-15 → A
.venv\Scripts\python.exe retest/fix/postest/repro_M5_lrtest.py             # 5/5 accept-reject 一致
.venv\Scripts\python.exe retest/fix/postest/repro_M6_hausman_constant.py   # df 2→3，统计量 2.6e-13
.venv\Scripts\python.exe retest/fix/postest/repro_M7_tobit_margins_default.py
                                                          # 旧默认(e) 4.289e-01 → 新默认(xb) 2.082e-10 → A
```

`retest/fix/postest/` 下的 `_probe*.py` 是取证过程的临时脚本（保留以便复核），
不是交付物；交付物是七个 `repro_M*.py` 与本文。
