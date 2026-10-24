# panel 修复报告

真值来源：本机 StataNow 19.5 (MP)，捕获日期 **2026-10-24**。
可重跑：`.venv\Scripts\python.exe retest/fix/panel/repro_panel.py`（脚本内现场启动 Stata）。
判定：相对差 ≤1e-9 记 A 档；SE 用 `e(V)` 对角开方。

## 逐项结果

| # | 缺陷 | 改前相对差 | 改后相对差 | Stata 命令 | 判定 |
|---|---|---|---|---|---|
| P1 | `xtreg, fe vce(robust)` 被映射成 HC1（常数项 SE 差 18.4 倍） | **3.821e+00** | **6.24e-16** | `xtreg y x1 x2, fe vce(robust)` | 已修复 A |
| P2 | `xtreg, fe vce(cluster id)` 的小样本因子用 `K=32`（含被吸收效应） | **3.821e+00**（同一因子） | **6.24e-16** | `xtreg y x1 x2, fe vce(cluster id)` | 已修复 A |
| P3a | `xtpcse` 非平衡 casewise：Σ 与 meat 按行号截断 | **7.032e-01** | **2.29e-16** | `xtpcse y x1 x2` | 已修复 A |
| P3b | `xtpcse` pairwise 未实现（与 casewise 逐位相同） | **6.626e-01** | **1.97e-15** | `xtpcse y x1 x2, pairwise` | 已修复 A |
| P3c | 平衡面板（回归防护，不能改坏） | 5.2e-18（全矩阵） | **5.56e-16**（SE）/5.2e-18（`e(V)`） | `xtpcse y x1 x2`（30×6） | 保持 A |
| P4a | `xtabond` 的 `e(sig2)` 未除以 2 | **1.993×**（相对差 1.000） | **2.21e-16** | `xtabond y x1 x2, lags(1)` | 已修复 A |
| P4b | `xtabond lags(2)` 的 `param_names` 整体错位一位 | 命名错位（数值另计） | 名字正确且系数 **4.63e-15** | `xtabond y x1 x2, lags(2)` | 已修复 A |
| P4c | `xtabond` 的 `vce(robust)` 没有对应实现 | — | **1.37e-15** | `xtabond y x1 x2, lags(1) vce(robust)` | 新增并达 A |
| P5 | `xtnbreg, fe` 只做条件 Poisson（HHG 未实现） | 系数 3.41e-05 / **SE 1.17e-03** | 系数 **1.39e-05** / **SE 5.17e-05** | `xtnbreg yc x1 x2, fe` | **B 档，判据已修正**（见下） |

> **P5 判定已由 Lead 批准改为**："**B 档；判据 = 对 `xtnbreg, fe ltolerance(1e-8)`
> 比斜率到 1e-6**"。按该判据，实测斜率差 **6.98e-09**（fixture）/ **6.2e-8 与
> 1.7e-6**（30×6），即已达 B 档目标。改用此判据的理由见 P5 小节。

补充（本轮修复顺带做的、超出 BRIEF 清单但同源）：
`xtabond` 的仪器集合多了一个常数列，导致 `lags(1)` 系数差 2.0e-3、`lags(2)` 差 3.5e-2。
删掉该列后系数 4.6e-15。这一项不修就无法让 P4a/P4b 真正达 A。

## 每项的根因与改法

### P1 / P2 — `panel.py`

`[XT] xtreg` 原文："Specifying `vce(robust)` is equivalent to specifying
`vce(cluster panelvar)`"。本轮实测复核：30×6 面板上
`max|V_robust − V_cluster(id)| = 0.0`（严格 0，见 repro 输出）。旧代码把
`cov_type="robust"/"HC1"` 映射到 linearmodels 的 `cov_type="robust"` = **HC1**，
那恰好等于 `areg y x1 x2, absorb(id) vce(robust)`（常数项 SE `.0402173` vs
Stata 的 `.0083416`，18.4 倍）。

改法：
- `_BACKEND_COV_TYPES`：`"robust"`/`"hc1"` → `"clustered"`；`_resolve_cov` 在
  robust 分支自动用 **panel 变量本身**做 cluster，并**禁止**调用方再传
  `cov_kwds`（否则就是另一个估计量）。
- `_extra_fit_kwargs`：`auto_df` 由 `False` 改为 **`True`**，并重写注释——
  旧注释断言"`auto_df=True` 会漏掉被吸收效应"，实测相反：`auto_df=True` 只数
  报告系数（Stata 的 K），`auto_df=False` 才把 29 个效应算进 K，正是 Stata
  **未文档化**的 `dfadj` 口径（差 9.36%）。
- `_stata_clustered_df_resid`：聚类时 `e(df_r) = G − 1`（旧代码留 `N − K`）。
  `RandomEffects` 覆写为"保持 `N − K`"，因为 `xtreg, re` 根本不报 `e(df_r)`。
- `_stata_clustered_scalar_df`：`e(rmse)`/`e(r2_a)` 在聚类下用
  `N − rank`（实测 `rss/rmse² = 177.0000`），非聚类用 `df_r`。
- 新增 `e(df_b)`、`e(rss)`。

**未修（需 Lead 裁决）**：Stata 聚类时 `e(df_m)` 报的是"报告 rank − 1"
（2 个斜率报 1，3 个报 2），但它的 F 检验 `Prob > F` 用的是 `F(df_b, df_r)`。
westata 保留 `df_m = 斜率数`，使打印的 `F(2, 29)` 标签与统计量自洽；若按 Stata
写 `e(df_m)`，summary 会打印成 `F(1, 29)` 而数值是 `F(2, 29)` 的——那是新的
静默错数。见"需要 Lead 裁决"。

### P3 — `panel_iv.py::XtPcse`

旧 `_aligned_slices()`：`common = min(len)` 再 `index[:common]`，按**行号**截断，
Σ 和 meat 都被截。且 `pairwise` 分支根本不存在。

改法（照 `xtpcse.ado::getSigma` 与 `matrix glsaccum ..., row(ivar) group(timevar)`）：
- `_sigma_casewise`：取**所有面板共同期**的残差估 Σ，除以 `n_sigma`；无共同期才
  报 r(459) 等价错误（并提示 pairwise）。
- `_sigma_pairwise`：逐对用 `T_ij`（两面板共同期数）。
- `_omega_meat`：按**期**分组累加 `Σ_t X_t' Σ[该期在场面板] X_t`，而不是
  `Σ_{i,j} σ_ij X_i'X_j`（后者只在平衡时等价）。
- 新增 `e(n_sigma)`/`e(missmeth)`；`periods=` 由 `from_formula` 传入。
- **回归修复**：`time=None` 且面板平衡时按行序推断期（Stata 无 `tis` 时的行为），
  不再误报 r(459)；非平衡且无 `time` 才拒绝并提示传 `time=`。

### P4 — `panel_dynamic.py::XtAbond`

- `e(sig2)`：`rss / (2 * (N − rank))`。差分误差 `Δε` 方差是 `2σ²`，旧代码少除 2，
  整个 `V` 放大 2 倍、SE 放大 √2。
- `param_names`：`tuple(f"L{lag}.y" for lag in range(1, lags+1))`。旧代码
  写死 `(f"L{lags}.y",)`，`lags=2` 时得 `['L2.y','x1','x2','extra_0']`，
  设计矩阵却是 `[L1.y, L2.y, x1, x2]`——按名取系数**静默取错**。现在列数不符
  会直接 `ConsistencyError`，不再编造 `extra_i` 名字。
- `vce`：新增 `cov_type="gmm"`（默认，`sig2 * W1⁻¹`）与 `"robust"`
  （`W1⁻¹ Q A1 [Σ_i Z_i'e_i e_i'Z_i] A1 Q' W1⁻¹`，无 sig2、无小样本因子）。
  两者都对 Stata 1.4e-15。
- **仪器集合**：删掉追加在**差分**工具变量里的常数列。Stata 的
  `_cons` 在 **level** 方程（`Instruments for level equation / Standard: _cons`），
  差分 GMM 不用它。`e(instruments)`/`e(zrank)` 报 Stata 的
  "Number of instruments"，`e(instruments_difference)` 报实际使用的矩阵列数。

### P5 — `panel_nonlinear.py::XtNbreg`

旧实现最大化的是**条件 Poisson** 核（`Σy·x'b − S·ln Σexp(x'b)`），所以它与
自己的 `xtpoisson, fe` 逐位相同（`.35845684 / .06584633`），而 Stata 的
`xtnbreg, fe` 是 `.35844983 / .05989124`。

改法：实现真正的 HHG 条件核
`Σ_t lnΓ(λ+y) − lnΓ(λ) − lnΓ(y+1) + lnΓ(Σλ) + lnΓ(S+1) − lnΓ(Σλ+S)`，
`λ=exp(x'b)`；保留 `_cons`（Stata 报 `.2.97667093`），解析梯度 + 解析 Hessian，
牛顿逐步减半，**按梯度收敛**（Stata 的 `nrtolerance` 规则）。数值上用
`Σ_{j<y} log(λ+j)` 代替 `lgamma` 差，避免 λ→1e7 时的灾难性抵消。

## 测试变更

| 文件 | 旧断言 | 新断言 | 依据（Stata 捕获 2026-10-24） |
|---|---|---|---|
| `tests/test_panel.py::test_auto_df_off_reproduces_stata_clustered_standard_errors` | 要求 `auto_df=False` 复现 Stata，并断言 `auto_df=True` 是错的 | 改名 `test_auto_df_on_...`；`auto_df=True` 复现 `[.2024145, .1204298]`，`auto_df=False` 是 `dfadj` 口径（14.3% 大） | `xtreg y x1 x2, fe vce(cluster firm)`，5×4 面板，`e(df_r)=4`、`se=(.2024145,.1204298,.03893807)` |
| `tests/test_panel.py::test_cluster_standard_errors_match_stata_cr1` | `cr1_cov(..., df_resid=FE_DF_R)`（K=7） | `df_resid=N−3`（Stata 的 K=3）；补 `e(df_r)=G−1` | 同上 |
| `tests/test_panel.py::test_robust_standard_errors_match_stata_hc1` | 断言 HC1 三明治 `(N/df_r)·bread·meat·bread` | 改名 `test_robust_is_statas_clustered_vce_not_hc1`；断言 `robust` 与 `cluster` **逐位相同**且等于 Stata 的 `[.03893807, .2024145, .1204298]`（westata 名序） | 同上；另 `areg ..., absorb(id) vce(robust)` 给出的是 HC1，18.4 倍 |
| `tests/test_panel.py::test_ssc_always_applies_stata_defaults` | 同一个 K=7 期望 | K=3 期望 | 同上 |
| `tests/test_panel.py::test_cluster_labels_survive_listwise_deletion` | 同一个 K=7 期望 | K=3；期望 `[.19931035, .12090296]`，`e(df_r)=4` | 同面板去掉第 3 行 x1 后 `xtreg y x1 x2, fe vce(cluster firm)`：N=19，`se(x1)=.19931035` |
| `tests/test_panel.py::cr1_cov`（docstring） | 未说明 `df_resid` 语义 | 写明它是 `(N−1)/(N−K)` 的**分母**，`xtreg, fe` 的 K 是报告系数 | — |
| `tests/test_panel_dynamic.py::test_xtabond_reports_the_instrument_count` | `e["instruments"] == 30` | `e["instruments"]==30`（Stata 的 zrank）+ `e["instruments_difference"]==29`（实际矩阵）+ 系数 `0.45495392 / 0.77623398` | `xtabond y x1, lags(1)` 90×9：`Number of instruments = 30`、`e(zrank)=30`、`GMM-type L(2/.).y`、`Standard: D.x1`、level 方程 `Standard: _cons` |
| `tests/test_panel_dynamic.py::test_xtabond_lag2_parameter_names_line_up_with_the_design`（新增） | — | `param_names == ['L1.y','L2.y','x1']` 且系数等于 Stata | `xtabond y x1, lags(2)`：`.43055232 / -.01727483 / .76238076` |
| `tests/test_panel_dynamic.py::test_xtabond_instrument_columns_account_for_statas_total` | 28+2=30（把常数算进差分块） | 28+1=29（差分块）+1（level 方程）=30 | 同上 |
| `tests/test_panel_nonlinear.py::test_xtnbreg_fe_matches_stata` | `.5041909`(abs 1e-4)/`se .0344353`(1e-5)，并称"核就是 Poisson 核" | 对 Stata **收紧容差**的 `.50418027`(1e-6)/`se .034435192`(1e-6)/`ll −775.440679`(1e-5)；另外保留对默认打印值 `abs=5e-5` 的显式残差 | `xtnbreg y x1, fe` 与 `xtnbreg y x1, fe ltolerance(1e-8)`，100×7 |
| `tests/test_panel_nonlinear.py::test_xtnbreg_fe_is_labelled_a_quasi_likelihood` | `estimator == "conditional_quasi_ml"` | `estimator == "hhg_conditional_ml"` | 目标函数换成了真 HHG 核 |
| `tests/test_panel_nonlinear.py::test_xtnbreg_fe_agrees_with_the_poisson_kernel` | 断言与 `xtpoisson, fe` 的系数、`ll` 各差 ≤1e-5（"就是 Poisson 核"） | 改名 `..._differs_from_the_poisson_kernel`：系数只做 1e-3 的粗对照，**严格断言两者是不同估计量**（label 不同、NB 有 `_cons` 而 Poisson 没有） | `xtnbreg` vs `xtpoisson`, 30×6：`.35844983/.05989124` vs `.35845684/.05784733` |
| `tests/test_panel_iv.py::test_xtpcse_balanced_without_time_matches_stata`（新增） | — | 平衡面板不传 `time=` 与传 `time="year"` 逐位相同，且等于 Stata | `xtset firm year` + `xtpcse y x1`，100×7：`x1 .75331259`(se `.05643178`)、`_cons 1.57268563`(se `.03729976`)、`e(n_sigma)=7` |
| `tests/test_panel_iv.py::test_xtpcse_unbalanced_without_time_is_rejected`（新增） | — | 非平衡且无 `time=` 时 `ConsistencyError`（提示传 `time=`） | 位置配对会拿甲的第 3 期对乙的第 5 期，不是协方差 |

## 未修 / 未达 A 的

### P5 `xtnbreg, fe`：SE 5.17e-5、系数 1.39e-5（B/C 档，非 A）

这是本轮唯一未到 A 的项。已确认**不是**公式问题：

- **信息矩阵是对的**。把 westata 的解析 Hessian 在 **Stata 自己的 `e(b)`** 处求值，
  得到的 SE 与 Stata 的 `e(V)` 相对差
  **3.67e-13 / 2.97e-12 / 7.02e-9**（30×6 面板三个参数）。也就是说：给定同一个 b，
  两边的观测信息矩阵逐位相同。
- **残差全部来自"停在哪一点"**。`xtnbreg, fe` 的 HHG 核在截距方向近乎平坦
  （Stata 自己报 `_cons` SE = **641**，`z` = 0.02），所以：
  - Stata 默认（`ltolerance(1e-7)`）停在 `b = [.35844983, −.22522002, 7.10468266]`；
  - Stata 的 `ltolerance(1e-8)` 停到 `[.35845171, −.22524727, 7.18542849]`；
  - 两者的**系数本身**就差到 1.1e-2（截距），斜率差 4.7e-6 / 1.05e-4。
  这解释了为什么"对 Stata 默认"的残差（1.39e-05）比"对 Stata 收紧容差"还大：
  两个 Stata 答案彼此之间就差这么多。
- westata 现在落在 `[.35845154, −.22524414, 7.18518766]`，即 Stata **收紧容差**的
  解（斜率相对差 6.2e-8 / 1.7e-6），`e(ll)` 差 **3.09e-09**。
- 100×7 fixture 上同理：斜率相对 Stata `ltolerance(1e-8)` 为 **6.98e-09**，
  `ll` 差 3.5e-09；对 Stata 默认打印值的残差是 2.1e-05，同样是 Stata 自己的停止规则。

**结论（Lead 已批准）**：HHG 已经实现且矩阵精确；剩余残差是 Stata 的收敛判据，
不是本实现的错。judgment 记为 **B 档，判据已修正为"对
`xtnbreg, fe ltolerance(1e-8)` 比斜率到 1e-6"**。按该判据 fixture 斜率差
**6.98e-09**、`e(ll)` 差 **3.5e-09**，已达标。**这不是能力回退**，不需要
`UnsupportedError`。

这个 `ltolerance` 依赖**已经写进 `XtNbreg` 的 docstring**（"Agreement, and why
the tolerance matters" 一节），并明确写了"不要当 bug 再修一遍"，避免下一个人
重复这一轮。

### P1 附带：聚类下 `e(df_m)` / `e(rank)` 与 Stata 不同（**Lead 已裁决：保留 westata 做法**）

verifier2 复核指出第二处未声明偏差：**`e(rank)` 也不等于 Stata**（westata=3 vs
Stata=2）。已在本轮补进文档，数值**不改**（`e(F)` 已是 6.5e-16，改 `rank` 会连带
影响读它的其他标量）。

Stata 在聚类 VCE 下把这两个标量**重新定义在"聚类问题的自由参数"上**，而不是
定义的模型列数上：

- `e(df_m) = 斜率数 − 1`（2 个斜率 → 1，3 个 → 2，1 个 → 0）
- `e(rank)  = 斜率数`（2 → 2，3 → 3，1 → 1）

而它打印的 `F` 与存的 `e(p)` 用的是 `F(df_b, df_r)`。所以 Stata 自己的 `e(df_m)`
就与它自己那行检验的自由度矛盾。`e(rank_mb)=3`（两个斜率时）保留了未聚类的值，
说明这个位移是约定而非意外。

westata 报参数向量自身的量：`e(df_m)` = 斜率数（2）、`e(rank)` = `len(e(b))` = 3
（含常数）、另加 `e(df_b)` = 2。因此**两个标量都恰好比 Stata 的聚类值大 1**，
其余全对：

| 量 | Stata | westata | 差 |
|---|---|---|---|
| `e(df_m)`（2 斜率） | 1 | 2 | +1 |
| `e(rank)`（2 斜率） | 2 | 3 | +1 |
| `e(df_b)` | 2 | 2 | 0 |
| `e(df_r)` | 29 | 29 | 0 |
| `e(F)` | 303.80 | 303.80 | 0（相对差 6.5e-16） |

**Lead 裁决：保留此做法。** `panel.py` 的模块 docstring 与 `PanelOLS` 类 docstring
各有一节 "Known deviations: `e(df_m)` and `e(rank)` under a clustered VCE"，
含三组实测：

```
command                                  e(df_m) e(df_b) e(rank)  prints
xtreg y x1,     fe vce(cluster id)          0      1       1      F(1,29)=211.29
xtreg y x1 x2,  fe vce(cluster id)          1      2       2      F(2,29)=303.80
xtreg y x1 x2 z, fe vce(cluster id)         2      3       3      F(3,29)=254.47
```

并新增回归测试
`tests/test_panel.py::test_clustered_df_m_and_rank_are_documented_deviations`
（两个斜率数 × 断言 `df_m`/`rank`/`df_b`/`df_r`，外加未聚类路径的 `rank`），
防止以后有人把它当 bug 悄悄"修"掉。

## 未做的部分

- `xtabond` 的 Sargan / Hansen / AR(1) / AR(2)：仍未报。本轮拿到了
  `e(sargan)` 的数值（Stata 25.12653980191442 / 18.655638669119067），但没有
  可靠公式能复现，按 BRIEF 的"诚实优于静默"原则继续留空。
- `xtabond` 的 `xtdpdsys`（system GMM）：仍只有差分部分，`system=True` 只做标记。

## 需 Lead 裁决的（4 条均已裁决，记录如下）

| # | 事项 | 裁决 | 落地 |
|---|---|---|---|
| 1 | 聚类下 `e(df_m)` 与 `e(rank)`：保留 westata 自洽口径，还是复刻 Stata（`df_m`=斜率−1、`rank`=斜率） | **保留 westata 做法**（`df_m` = 斜率数、`rank` = `len(e(b))` = 斜率数+1，另加 `e(df_b)`）。理由：Stata 的 `e()` 在此自相矛盾（它的 `e(df_m)` 与它自己打印的 `F(df_b, df_r)` 打架），复刻会让 summary 打出 `F(1,29)` 而数值是 `F(2,29)`——把标签错换成数错更糟；`e(rank)` 被结果容器与 postestimation 当作 `e(b)`/`e(V)` 的尺寸使用，缩小它会静默改变其他标量 | `panel.py` 模块 docstring 与 `PanelOLS` 类 docstring 的 "Known deviations: `e(df_m)` and `e(rank)` under a clustered VCE"，含 `e(df_m)/e(df_b)/e(rank)` 三组对照；回归测试 `test_clustered_df_m_and_rank_are_documented_deviations` |
| 2 | `xtnbreg, fe` 判据 | **改为"对 `ltolerance(1e-8)` 比斜率到 1e-6"，B 档** | 本文件 P5 条目；`XtNbreg` docstring "Agreement, and why the tolerance matters" |
| 3 | `test_stata_p9.py` 里 xtabond 仪器数的旧注释「28 GMM + D.x1 + 常数」是错的 | **Lead 自己改**（不在我作用域） | — |
| 4 | `xtpcse` 的 `time=` 是否一律必填 | **保持"平衡可省、非平衡必填"**，与 Stata 无 `tis` 时的行为一致 | `XtPcse` docstring "When ``time=`` may be omitted"：平衡按行序推断期（依赖 entity-major/period-minor 行序，由 `_as_frame` 保证）、非平衡无 `time` 报错并提示、有 `time` 但无共同期时报 Stata `r(459)` 等价错误并建议 `pairwise` |

另：`xtabond` 的 `e(instruments)`（= Stata 的 `e(zrank)`，"Number of
instruments"）与 `e(instruments_difference)`（实际矩阵列数）的区别，已写进
`panel_dynamic.py` 模块 docstring 新增的 "The two instrument counts are *not* the
same number" 一节，含 28 + 1 + 1 = 30 的拆解、以及"曾为凑 30 往差分工具变量塞
常数列 → `lags(2)` 系数偏 3.5e-2"的告诫。
