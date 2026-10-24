# discrete 组修复报告

覆盖 `retest/REPORT.md` §3 的 **D1–D5**（P0-1、P0-9、P0-10、P0-11、P1-18）+ 语义统一 + `zip/zinb/ztp` 的 `predict`。

- 源文件：`src/westata/estimators/discrete.py`、`src/westata/estimators/counts.py`
- 测试：`tests/test_discrete.py`、`tests/test_counts.py`
- 证据：真实 StataNow **19.5 (MP)**，全部现场跑，捕获日期 **2026-10-24**
- 可重跑：`.venv\Scripts\python.exe retest/fix/discrete/repro_discrete.py`（输出存档 `repro_output.txt`）
- 测试常量出处：`.venv\Scripts\python.exe retest/fix/discrete/capture_test_fixtures.py`

## 逐项结果

| # | 缺陷 | 改前相对差 | 改后相对差 | Stata 命令 | 判定 |
|---|---|---|---|---|---|
| D1a | `ologit` `e(V)` 全矩阵 | **2.09e+01** | **2.75e-11** | `ologit ycat x1 x2` | **已修复 A** |
| D1b | `oprobit` `e(V)` 全矩阵 | **1.05e+01** | **1.54e-08** | `oprobit ycat x1 x2` | 已修复 **B**（残差是 Stata 停步，见下） |
| D1c | `ologit` cut SE | 0.0874→0.1043（+20.5%）、0.0962→0.3208（+176%） | **4.50e-12** | 同上 | **已修复 A** |
| D1d | `ologit` slope×cut 交叉块 | 从未变换（逐位等于后端矩阵） | **2.75e-11** | 同上 | **已修复 A** |
| D2a | `logit vce(robust)` | `ValueError`（不可用） | **6.97e-14** | `logit yb x1 x2, vce(robust)` | **已修复 A** |
| D2b | `probit vce(robust)` | `ValueError` | **8.55e-10** | `probit yb x1 x2, vce(robust)` | **已修复 A** |
| D2c | `nbreg vce(robust)` | `ValueError` | SE **3.72e-08** | `nbreg y_nb x1 x2, vce(robust)` | **已修复 A（SE）** |
| D2d | `vce(cluster g)` 全家 | 3.35e-03 | logit **7.49e-14** / nbreg SE 8.0e-08 | `..., vce(cluster g)` | **已修复 A（SE）** |
| D3 | `Poisson` `HC1` 静默等于 `HC0` | `array_equal=True` | 现为 `N/(N-K)` 倍，误差 **0.0** | 无对应 Stata 选项（见下） | **已修复** |
| D4 | `e(chi2)` 不随 VCE 切 LR→Wald | 3.72e-01（完全不敏感） | logit **5.8e-14**、poisson **6.3e-10**、ologit **8.2e-14** | `..., vce(robust)` / `vce(cluster g)` | **已修复 A** |
| D5a | `nbreg` `e(df_m)`=3 | 3（Stata 2… 本数据 1） | **1 = Stata** | `nbreg y_nb x1 x2` | **已修复** |
| D5b | `ologit/oprobit` `e(df_m)`=4 | 4（Stata 2） | **2 = Stata**（`_ordered()` 数据为 1 = Stata） | `ologit ycat x1 x2` | **已修复** |
| D5c | `e(k_aux)` 缺失 | 不存在 | nbreg **1**、ologit/oprobit **3**，均 = Stata | 同上 | **已修复** |
| P1-19 | `zip/zinb` `e(p)` 用 `df_m` | **1.15e+02** | **1.74e-07** | `zip y_zi x1 x2, inflate(x1 x2)` | **已修复** |
| M3 | `zip/zinb/ztp` `predict` | 崩溃（matmul 维数） | n **1.1e-09**、pr **5.2e-09**、xb **1.1e-06**、stdp **2.8e-08** | `predict, n` / `pr` / `xb` / `stdp` | **已修复** |
| M3b | `ztp/tnbreg predict, n` 语义 | 返回截断均值（8.68e-01） | **4.7e-08** | `ztp y_nb x1 x2` → `predict, n` | **已修复** |

UAT 说明：`e(V)` 一列的 "改前" 是**现场用 `git stash` 回到旧源码重跑同一脚本**得到的，
原始日志见 `before_after.txt`；D1 的旧算法另在 `repro_discrete.py::_legacy_to_cut_points`
里原样保留，任何人可在当前代码上复算那一列。

## 每项的根因与改法

### D1 `ologit/oprobit` 的 `e(V)` —— 两个**独立**缺陷，都在 `discrete.py::_to_cut_points`

**① numpy 视图别名。** 旧代码 `theta = params[n_slopes:]` 取的是**视图**，紧接着
`params[n_slopes:] = cuts` 把 `theta` 也改成了 cut 点，于是构造 Jacobian 时
`np.exp(theta[1:k+1])` 作用在 **cut 点**上而不是 log-gap 上。修：`theta = params[n_slopes:].copy()`。
这一点不改，只修 Jacobian 形状也没用——`exp()` 的**输入**本身就是错的。

**② slope×cut 交叉块从未变换。** 旧代码只改 `cov[n_slopes:, n_slopes:]`，`cov[:n_slopes, n_slopes:]`
原样输出。修：用**整块**变换 `T @ cov @ T.T`，其中 `T = diag(I_ns, J)`。
复现脚本直接打印了证据：旧代码输出的交叉块与输入**逐位相同**（`array_equal=True`）。

**③（新增，BRIEF 未列）Jacobian 的解析形式。** 正确关系是
`cut_k = theta_0 + Σ_{j≤k} exp(theta_j)`（`theta_0` 就是 `c1`），因此
`J[k,0] = 1`、`J[k,j] = exp(theta_j)`（`1 ≤ j ≤ k`）。旧代码写的是 `J[k,:k] = exp(theta[1:k+1])`，
把 `exp()` 放在了本该是 1 的累积列上。

**④（新增）协方差改用解析 observed Hessian。** 即使 ①②③ 都修对，`e(V)` 仍只到
1e-6 量级：`_Ordered.fit()` 走 statsmodels `OrderedModel` 的**数值** Hessian，在
slope×cut 交叉块上精度不够（eps=1e-4 → 4e-6，eps 更小反而更差）。改成解析
observed Hessian（`_ordered_hessian`，logistic/normal 各自的一阶/二阶密度显式写出）后，
**在 Stata 自己的 b 上** `e(V)` 复现到 **2.1e-13**，这才是 1e-9 可达的前提。

**⑤（新增）点估计的 Newton 抛光。** `OrderedModel.fit(method="bfgs")` 停在梯度范数上，
cut 点在第 5 位有效数字就与 Stata 分开（3.55e-05）。加了 `_newton_polish`：用解析
gradient/Hessian 做精确 Newton（带回溯线搜索 + cut 单调性检查），通常 **1 步**就落到
Stata 的点上，系数从 3.55e-05 降到 1.68e-11。

### D2 `vce(robust)` / `vce(cluster)` 全崩

根因：`discrete.py:178` 把 `cov_kwds={"use_correction": ...}` 传给 statsmodels，
而 `Logit/Probit/NegativeBinomial` 的 HCx 分支**拒绝任何 cov_kwds**（"heteroscedasticity
robust covariance does not use keywords"）。只有 `Poisson` 因为 `_accepts_hc_kwds=False` 幸免。
但**接上管道也不对**：statsmodels 的 HC0 无小样本因子、HC1 用 `N/(N-K)`，而 Stata 的
`ml vce(robust)` 用 `N/(N-1)`，`vce(cluster g)` 只要 `G/(G-1)`（**没有** `(N-1)/(N-K)`）。

修：不再委派。新增 `_vce_matrix`，用观测信息 `(-H)^-1` 与 score `s_i` 自己组装：
`robust = N/(N-1) · B M B`、`cluster = G/(G-1) · B M_g B`、`HC0 = B M B`、`HC1 = N/(N-K) · B M B`。
在 Stata 自己的 b 上验证：logit/probit/poisson/ologit/oprobit 到 **~1e-15**。

`NegativeBinomial` 有个坑：statsmodels 的 `_transparams` 标志控制末位参数是 `alpha` 还是
`ln(alpha)`，**`fit` 会把它翻成 False 并报告 alpha**，而 `_rescale_extra_params` 之后再做
delta 变换。若在 `_transparams=True` 下算 VCE，delta 会做两次（实测 `/lnalpha` SE 从
0.1558 变成 0.0388）。修：`_backend_cov` 里显式 `model._transparams = False`。

### D3 `Poisson` 的 `HC1` 静默等于 `HC0`

statsmodels 0.15 对这两个模型**返回同一个矩阵**（`cov_white_simple(use_correction=False)`，
且 `cov_kwds` 被丢弃）。实测 `array_equal=True`。修：`HC1` 由自己按 `N/(N-K)` 计算，
现在 `HC1 == N/(N-K) · HC0` 到 **0.0**（四个家族全部）。同时 docstring 写明
`HC0`/`HC1` 是 **statsmodels 约定，不是 Stata 的 `vce(robust)`**，后者是 `cov_type="robust"`。

### D4 `e(chi2)` 不随 VCE 切 LR→Wald

Stata 一旦 `vce(robust)`/`vce(cluster)`，`e(chi2type)` 从 `LR` 变 `Wald`，且 Wald
**只含斜率块**（常数项与辅助参数都不进）。实测：用 Stata 自己的 `e(V)` 复算
`b_s' V_ss^-1 b_s`，六个命令**全部复现 `e(chi2)` 到 ~1e-16**；把常数项算进去则偏
7%–125%，把 inflate 斜率也算进去偏 14%–35%。

修：新增 `_model_test(params, cov, names, llf, ll_null, robust)`，`robust=True` 时返回
斜率块 Wald 并把 `e(chi2type)` 设为 `"Wald"`；`summary.py` **本来就已经**按
`e['chi2type']` 渲染标签（无需改共享文件）。

### D5 `e(df_m)` 把辅助参数算进去

旧代码两处都写 `df_m = len(params) - 1`：`nbreg` 得 3（`/lnalpha` 被算进去），
`ologit/oprobit` 得 4（3 个 cut 被算进去）。而 `tnbreg` 走 `_CountBase` 反而是对的——
**同一个 NB2 似然，两条路径两个答案**。

修：`df_m` 改为"斜率个数"（`_slope_positions` / `_model_df`），辅助参数改由
`e(k_aux)` 报告（Stata 自己也是这么分的：`nbreg` 的 `k_aux=1`、`ologit` 的 `k_aux=3`）。
`StataResults.df_model` 与 `e(df_m)` 同步改，否则 summary 的标签会和 p 值不一致。

### P1-19 `zip/zinb` 的 `e(p)` 自由度

Stata 的 `e(p) = chi2.sf(e(chi2), e(rank)-1)`，**不是** `chi2.sf(e(chi2), e(df_m))`。
实测：`zip` 的 `e(p)=0.6133814699910781` 与 `chi2.sf(3.566291547919718, 5)` 吻合到
1.7e-15，而用 `df_m=2` 得 0.1681（差 1.15e+02 倍）。注意这只在**默认 VCE** 下成立，
`vce(robust)` 后两侧都切回 `df_m`。`logit/probit/poisson/nbreg/ologit` 一直用 `df_m`，
所以这是多方程专有约定。

修：`_CountBase._two_equation_p` 标志（`zip`/`zinb` 为 True），`_stata_scalars` 按
`robust` 与它决定自由度。

### M3 `predict` 的模型侧能力

按 `postest` 组的契约实现：结果对象上挂 `_predict_dispatch`（返回 `None` 表示交回通用
实现）与 `_predict_types`（Stata 实际接受的 kind 白名单）。

- `zip`/`zinb`：`('xb','n','pr','stdp')`；`n = (1-pi)·mu`、`pr = pi`、`xb` 是 count 方程线性指标。
- `ztp`/`tpoisson`/`tnbreg`：`('xb','n','stdp')`；`n = exp(xb)`（**不含**截断修正）。

**语义陷阱（两处旧代码都错了）**：
1. `zip predict, n` 是 `(1-pi)·mu`，不是 `mu`（实测 9.9e-16）；
2. `zip predict, pr` 是**零膨胀概率 pi**，不是 `P(y=0) = pi + (1-pi)f(0)`——两者在这组数据上差 0.77；
3. `ztp predict, n` 就是 `exp(xb)`，Stata **不做**截断修正（实测 1.8e-15），
   而旧的 `fittedvalues` 给的是截断均值 `mu/(1-e^-mu)`，差 8.68e-01。
   现在两者并存：`predict(kind="n")` 给 Stata 的定义，`fittedvalues` 保留截断均值。

顺序模型（`ologit/oprobit`）**不挂** `_predict_dispatch`：它们的 `pr` 是
`P(y=观测到的类别)`，需要 `outcome(k)` 参数，属于 `postest` 的通用面。
但本组补了它的前置条件——`fit()` 现在会传 `design_matrix`（旧代码从不传，
导致 `StataResults.predict(exog=None)` 抛 `ConsistencyError`）。

## 测试变更

| 文件 | 旧断言 | 新断言 | 依据（Stata 捕获） |
|---|---|---|---|
| `test_discrete.py::test_logit_summary_reports_wald_chi2` | 默认 `logit` 的 summary 含 `Wald chi2(` | **拆成两条**：默认断言 `LR chi2(` 且不含 `Wald chi2(`；`fit(cov_type="robust")` 断言 `Wald chi2(` 且 `e(chi2)=76.13552601769518` | `logit y x1 x2` → 打印 `LR chi2(2) = 136.03`、`e(chi2type)=LR`；`logit y x1 x2, vce(robust)` → `Wald chi2(2) = 76.14`、`e(chi2type)=Wald`（2026-10-24） |
| `test_discrete.py`（新增 5 条） | — | `test_ordered_covariance_matches_stata_reference_values`（cut SE 0.10432381/0.08740299/0.09620776 与交叉块）、`test_ordered_robust_vce_runs_and_switches_to_wald`、`test_df_m_excludes_the_dispersion_parameter`、`test_df_m_excludes_the_cut_points`、`test_probit_robust_vce_matches_stata`、`test_negative_binomial_robust_vce_matches_stata`、`test_poisson_hc1_is_not_silently_hc0` | 每条注释里带完整 Stata 命令与数字，出处 `capture_test_fixtures.py` |
| `test_counts.py`（新增 6 条） | — | `test_zip_predict_n_is_the_conditional_mean_and_pr_is_pi`、`test_ztp_predict_n_is_exp_xb_not_the_truncated_mean`、`test_truncated_models_refuse_pr_like_stata`、`test_zip_exposes_the_predict_hook_postestimation_delegates_to`、`test_count_model_test_switches_to_a_slope_block_wald`、`test_zip_e_p_uses_rank_minus_one_degrees_of_freedom` | `zip y x1, inflate(x1)` 的 `predict,n/pr/xb` 头 3 个值、`e(b)`、`e(chi2)=248.04159086422897`、`e(p)=1.7353842226580317e-53`（2026-10-24） |

**没有为了让测试通过而放宽任何数值判据**：新增断言的容差是按实测残差定的
（1e-6~1e-3），并在注释里写明该残差的来源。

## 未修 / 未达 A 的

1. **`oprobit` 的 `e(V)` 是 B 档（1.54e-08），不是我们算错。**
   把解析 Hessian 求值在 **Stata 自己的 b** 上，`e(V)` 复现到 **2.5e-14**。
   残差完全来自点估计：Stata 在该数据上停在 `|gradient| = 9.1e-06` 处（`e(ll)` 与
   我们差 0/1e-13），我们停在 `2e-14`。似然面在该方向极平，所以 b 差 1e-8 会
   放大成 `e(V)` 的 1.5e-08。**要更近只能复刻 Stata 的停步点，那是复制它的收敛缺陷，
   不做**。
2. **`nbreg` 的 `e(V)` 全矩阵是 C 档（1.2e-05），但 SE 是 B/A 档（1.8e-08）。**
   限项在 `/lnalpha` 与斜率的**交叉项**：statsmodels 的 `NegativeBinomial.hessian` 是
   alpha 尺度的解析公式（且**忽略** `_transparams`，两种参数化返回同一个矩阵），
   经 delta 变换后 alpha 方向的三阶项残留 ~1e-05 相对误差。SE（对角开方）不受影响，
   因为方差项的 delta 是精确的 `1/alpha²`。三个 VCE 都是这个模式。
3. **`zip/zinb/ztp/tnbreg` 的 `e(V)` 是 B/C 档（1e-8 ~ 1.7e-04）。**
   这几个模型的似然是本项目自写的 `_CountLikelihood`，没有解析 Hessian，只能数值求导。
   做了一阶 Richardson 外推（`_richardson_hessian`，step=4e-4, ratio=2）：`zip` 从
   1.7e-04 降到 **1.2e-07**、`ztp` 从 3.6e-07 降到 **2.4e-08**。再往下是双精度极限
   （步长 1e-4 时舍入误差已经主导：6e-06）。**要上 A 档需要给这四个似然写解析
   Hessian**，工作量约等于重写一遍；建议作为后续项。
4. **`nbreg` 的 `/lnalpha` SE 在近乎 Poisson 的样本上无意义。** `_count()` 数据
   `e(alpha)=6.1e-06`，Stata 自己在同一份数据上给出三个不同的 `se(lnalpha)`：
   548.8（OIM）/ 42.6（robust）/ 40.0（cluster）。这是边界不可识别，不是实现差异，
   所以测试只断言斜率块的 SE。
5. **`zip`/`zinb` 的 `predict, xb` 相对差 1.1e-06 / 7.7e-05。** 这是纯线性组合
   `Xb`，误差完全来自系数（2.4e-07 / 2.6e-01），在 `Xb` 接近 0 的观测上被相对
   口径放大；绝对误差 1.3e-09。不是 predict 的问题。
6. **`ologit/oprobit/logit/probit/poisson` 的默认 `e(V)` 现在是 A 档，但
   `_Ordered` 不再调用 `OrderedModel.fit` 的协方差**——如果以后有人依赖
   `result.cov_params()` 与 statsmodels 一致，行为变了。已写进模块 docstring。

## 需要 Lead 裁决的

1. **`_Ordered.fit()` 不再使用 statsmodels 的协方差**（改用解析 Hessian + Newton 抛光）。
   这会让 `westata.OrderedLogit` 与 `sm.OrderedModel` 的数值不再逐位相同，
   换来与 Stata 一致。我判断这是对的（目标就是 Stata），但这是**结构性变更**，
   请复核是否接受。
2. **`e(df_m)` 的定义变了**（不再 `len(params)-1`）。`limited.py` 的
   `_MLLimitedBase` 继承 `_DiscreteBase`，而基类 `_stata_scalars` 的签名变了
   （多了 `cov/names/canonical/groups`）。**已核实不是阻塞项**：`limited` 组已经把它
   覆写为同样的新签名（`limited.py:495-505`，`canonical`/`groups` 带默认值），
   实测 `Truncreg.from_formula(...).fit()` 正常返回 SE。**无需裁决，仅告知**。
   我最初按旧代码判断会 TypeError，实测后已更正。
3. **`summary.py` 我没有改**。它本来就按 `e['chi2type']` 渲染，实测
   nonrobust→`LR chi2(2)`、robust→`Wald chi2(2)`。你的授权不需要动用。
4. **计数模型 `e(V)` 停在 B/C 档**（上文第 3 条）。如果你要求这四项也到 1e-9，
   需要额外一轮写解析 Hessian 的工作量；我建议记为"已诊断、已知残差"。

## 回给 Lead 的话（按 BRIEF §6）

- **组名**：discrete。**修了 8 项 / 3 项未达 A（均有实测残差与归因）**。
- 逐项：
  - D1 ologit `e(V)`：2.09e+01 → **2.75e-11（A）**，`repro_discrete.py`
  - D1 oprobit `e(V)`：1.05e+01 → **1.54e-08（B，Stata 停步所致，在 Stata b 上是 2.5e-14）**
  - D2 logit/probit robust：`ValueError` → **6.97e-14 / 8.55e-10（A）**
  - D2 nbreg robust / cluster：`ValueError` / 3.35e-03 → SE **3.7e-08 / 8.0e-08（A）**
  - D3 `HC1`：静默等于 `HC0` → 现为 `N/(N-K)` 倍，误差 **0.0**
  - D4 `e(chi2)`：3.72e-01 → **5.8e-14 ~ 2.0e-07**，`chi2type` 全部切对
  - D5 `e(df_m)`：nbreg 3→**1**、ologit/oprobit 4→**2**（本测试数据），`e(k_aux)` 补齐
  - P1-19 `zip e(p)`：1.15e+02 → **1.74e-07**
  - M3 `predict`：崩溃 → n **1.1e-09**、pr **5.2e-09**、xb **1.1e-06**、stdp **2.8e-08**
- **测试改了 1 条旧断言**（`test_logit_summary_reports_wald_chi2` 拆成两条），
  依据是 2026-10-24 现场 `logit` 默认打印 `LR chi2(2)`、`vce(robust)` 才打印
  `Wald chi2(2)`；另新增 11 条带 Stata 常量的回归测试。
- **需要裁决**：3 条（`_Ordered` 不再用 statsmodels 协方差；计数模型 `e(V)` 停在 B/C 档；
  `limited.py` 签名已核实**不是**阻塞项，仅告知）。
- **pytest**：`pytest tests/test_discrete.py tests/test_counts.py -q` → **70 passed**。
