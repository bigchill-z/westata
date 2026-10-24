# systems 组修复报告

范围：`nl` / `frontier`（`nonlinear.py`）、`teffects`（`treatreg_effects.py`）、
`jackknife`/`bootstrap`（`resample.py`）、`didregress`（`did.py`）。
另加 Lead 追加的 **T8**（`frontier halfnormal` 的 SE，原属 limited 组，因文件归属冲突转入）。

所有 Stata 数字都来自本次现场 StataNow 19.5（MP）嵌入式会话，脚本在
`retest/fix/systems/`，每条命令都能重跑。

## 逐项结果

| # | 缺陷 | 改前相对差 | 改后相对差 | Stata 命令 | 判定 |
|---|---|---|---|---|---|
| S1 | `nl` 协方差用默认 `pinv` 截断 | SE **1.000e+00**（差 7 个数量级） | SE 复现 **1.848e-12**（Stata 锚定）；秩守卫另验：缩放不变性 **1.646e-11** + 精确共线三种缩放全拒 | `nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)` | **A 指 SE 复现**；秩守卫判定**另有独立验收**，见下方 S1 节（两个维度分开写，不要只看"A"就认为判据没问题） |
| S2 | `nl` 默认初值全 1、不收敛仍返回 | 参数 1.42e+01 / RSS 差 7.5 倍 | 参数 **1.57e-07**、RSS 6.3e-15、`converged=1` | `nl (price = {b0}*exp({b1}*mpg))` | 已修复 **B**（A 档不可达，见下） |
| S3 | `teffects aipw, atet` 增广项结构错 | ATET 5.2e-01 / SE 2.93e-02 | ATET **7.34e-16** / SE **1.17e-15** | `teffects aipw (y x1 x2) (t x1 x2), atet` | 已修复 **A** |
| S3' | **ATE 分支护栏** | 1.11e-16 | **1.18e-16**（未动） | `teffects aipw (y x1 x2) (t x1 x2)` | 未破坏，已加回归测试钉住 |
| S4 | `nnmatch` 的 ATET SE 与 ATE 逐位相同 | ATET/ATE 比值 = **1.000000**（逐位相同） | 比值 **1.109128**（Stata 1.133439）；SE 差 1.0e-01 | `teffects nnmatch (y x1 x2) (t), atet` | **结构性缺陷已消除、数值仍是 C 档**（见「未达 A」第 1 条，不写作"已修复"） |
| S5 | `didregress` 的 `covariates=None` 自动加协变量 | ATET 位移 1.85e-04（含无关列时） | `(y)` 与 Stata **0.0** | `didregress (y) (treated), group(unit) time(period)` | 已修复 **A**（语义反转） |
| S6 | `jackknife` 的 t 检验 `df_r` 用 72 | 区间宽 2.34e-04 | `df_r` = **73**（与 Stata 相等） | `jackknife: regress mpg weight` | 已修复 **A** |
| S7 | `bootstrap` 用 t 而 Stata 用 z | 临界值差 1.71e-02；区间宽 1.7% | CI 与 Stata **1.4e-9** | `bootstrap, reps(100) seed(20241024): regress mpg weight` | 已修复 **A** |
| T8 | `frontier halfnormal` 的 SE（数值 Hessian 步长） | 旧 step=1e-5 目标函数差分 **4.027e-03**；出厂 SE 现 **C 档 2.5e-5** | **Hessian 本身** 1.722e-07（fixture）/ 4.6e-06（n=400 DGP，在 Stata 的 `e(b)` 上） | `frontier y x1 x2` | **A 仅指 Hessian 本身；出厂 SE 仍是 C 档 2.5e-5**，归因于停机点，不是本轮可修项 |

## 每项的根因与改法

**S1（`nonlinear.py`）**
错在 `covariance = sigma2 * np.linalg.pinv(jacobian.T @ jacobian)`（旧 `:1024`）。
`J` 的条件数是 1.6e8，平方成 `J'J` 后是 **2.6e16**，最小奇异值 3.39e-01 对 8.68e15
（`s_min/s_max = 3.91e-17 < max(shape)*eps = 6.66e-16`），`pinv` 的默认 rcond 把这一维
**静默置零**，于是 `se(/b0)` 报 3.95e-07 而 Stata 是 5.77——点估计却全对。
改法（`_gauss_newton_covariance`）：**不构造 `J'J`**，对 `J` 做 `qr` 用 `R⁻¹ R⁻ᵀ`；
秩检验作用在 `J` 上（不是 `J'J`）。注意：本模型 `J` 满秩、`J'J` 数值上秩 2，
所以按 `J'J` 判秩会把一个可估模型误杀。
**没有**用 `rcond=0`：Task 明确禁止，且它只会把病态方向的数值噪声放大成"看起来合理"的 SE。
另外把 Jacobian 的差分步长从优化器的 1e-6 换成独立的 `_JACOBIAN_STEP = 1e-5`：
协方差是 `J'J` 的**逆**，误差被条件数放大，1e-5 把 SE 从 7.1e-11 压到 1.4e-12。

**S1 复核（verifier2 发现真实回退，已修）——判据必须无量纲，容差也是。**
第一版的秩判据用 `max(shape)*eps*s_max/h`，**随量纲翻转**：同一个可识别的二次式，
`u = weight*1e3` 被**误拒**（`rank 2/3`），`u = weight*1e-6` 却能过。verifier2 从
缩小方向看到同样的翻转（他们用 ×0.1…×1e-6 全 OK 而原单位边缘）。实测三条事实：
- 列归一化后 `cond(J_n)` 在 8 个缩放下**逐位恒为 65.2558**，而原始 `s_min/s_max` 从
  `6.25e-09` 变到 `4.97e-03` —— **矩阵根本没病态，是判据在量单位**。
- auto.dta 原单位当时只以 **1.4 倍余量**过关（`6.25e-9` vs 阈值 `4.44e-9`），不是罕见角落。
- 真共线仍被抓（可识别子模型仍可拟合）→ 是"过度敏感"，不是"不安全"。

修法分**两层**，缺一不可：

1. **判据无量纲**：令 `D = diag(‖J[:,k]‖)`，对 `J_n = J D⁻¹` 做 SVD 判秩与算 `cond`；
   协方差由 `(J'J)⁻¹ = D⁻¹(J_n'J_n)⁻¹D⁻¹` 还原，**代数上与原式等价**，不改任何 SE。
2. **容差也不能只看 `s_max`**——这是我在修第一层时**自己踩出来的第二个坑**（由新增的
   `test_nl_scale_free_rank_test_still_catches_exact_collinearity` 抓到）：
   只做列归一化后，`z = x*1e-6` 这种**精确共线**反而漏过了（伪奇异值 8.2e-6 > 容差 4.4e-10）。
   根因是归一化同时改变了噪声的传播：每列是中心差分，**绝对**误差约
   `eps·‖f‖/h`（与参数尺度无关），除以 `‖J[:,k]‖` 后变成**相对**误差
   `eps·‖f‖/(h·‖J[:,k]‖)`。`z=x*1e-6` 时 `‖col_b‖≈2.7e-5`，相对噪声被放大到 3.3e-5。
   所以容差改为**逐列**由该噪声地板构造、取归一化矩阵的 `‖E‖_F`，
   **不再**是 `s_max` 乘一个相对常数。

四条验收（`repro_S1_scale.py`，全部 PASS）：

| 验收 | 结果 |
|---|---|
| 原单位 `weight` 不再被拒 | OK，`se(b0) = 5.767884338`（Stata 同值） |
| ×0.1 / ×0.01 / ×0.001 / ×1e-6 与原单位一致 | **worst SE rel = 1.646e-11**（目标 1e-10） |
| 扩到 ×10 / ×1e3 / ×1e6 也一致（放大方向原来会误拒） | 同上，8 组全过 |
| `{a}*x + {b}*z`（z≡x）仍抛 `ConsistencyError` | 是，且在 ×1e-6 / ×1 / ×1e6 三种缩放下都拒 |
| 原 S1 的 Stata 锚定 SE 不退回 pinv 截断 | `se(b0) = 5.767884338136`，rel 1e-12 |

`se(b2)*scale²` 在 8 组缩放下**逐位相同**（6.2575944815e-07），这是协方差真正与量纲无关的直接证据。系数本身有 1.17e-07 的散布，那是优化器停点差异（与 T8 同源），不是判据性质。
另外确认没有过度拒绝：exp decay / quad / linear / power / exp / frontier 六个正常模型全部照常
拟合，SE 与既有 Stata 值一致。

**S1 的档位怎么读（两个维度，不要混为一谈）。** 上面逐项表里的 **A** 只指 **SE 复现**：
`se` 与 Stata 的 1.848e-12。**秩守卫的判定本身**是另一个维度，它另有独立验收：
缩放不变性 1.646e-11（8 组 ×1e-6…×1e6）、精确共线在 ×1e-6 / ×1 / ×1e6 三种缩放下全拒、
且六个正常模型不被误拒。verifier2 的建议是"修好守卫之前不要把 S1 当 A 档进 README"——
守卫现已修好，但**以后引用 S1 请同时引用这两组数字**：不要只看到"A"就以为判据没问题，
也不要只看到"1.8e-12"就以为那是判据的验收。

**判据余量（防止将来又变成 flaky）。** 决定"是否共线"的余量不能贴边：真共线的
`s_min/tol` 实测只有 0.049x（`z=x`）、**0.46x（`z=x*1e-6`）**、0.012x（`z=x*1e6`）——
0.46x 意味着离翻转只有 2.2 倍，不同的 BLAS 路径或求和顺序就可能翻转，这正是它曾在
**全量并发**时偶发失败的原因。已排除模块级全局状态污染：`_JACOBIAN_STEP` / `_HESSIAN_STEP`
/ `_DIFFERENCE_STEP` 没有任何测试或生产代码改写（`grep` 全仓确认）。
修法是给容差乘 `_RANK_NOISE_FACTOR = 10`：之后共线余量为 0.0049x / 0.046x / 0.0012x
（离翻转 20–850 倍），而合法模型本就在噪声地板之上 1e7–1e10 倍（quad 3.3e8、
exp decay 3.0e10、linear 1.1e7）——**加这个系数不损失任何判别力**，只消掉刀锋。
验证：该测试单独连跑 10 次 10/10 通过，整文件 5/5 通过。

**S2（`nonlinear.py`）——选了「复刻 Stata 网格起步」**
两者择一，选网格的理由：直接抛 `ConvergenceError` 会让
`nl (price = {b0}*exp({b1}*mpg))` 这类**Stata 能跑通的常规命令**在 westata 上永远失败，
用户没有一个"正确的默认"可用；把默认修好才是根因修复。实现见 `Nl._starting_values`：
按 `_START_LADDER`（1e-3…1e4，**乘性**，因为 `nl` 参数多为尺度参数）做逐坐标搜索，
只在 RSS 下降时替换，调用者给的 `start=` 只作为起点、不会被覆盖。实测
`b0` 从 2.27e-10（RSS 3.42e9）到达 19581.414（RSS 453839393.06，Stata 19581.4174/同 RSS）。
同时满足 Task 的另一个要求：**未收敛默认抛** `ConvergenceError`（`strict` 默认改为 `True`），
`strict=False` 保留为查看最后迭代点的逃生口。
档位 B 而非 A：残差 1.57e-07 来自 Stata 自己的 Gauss-Newton 与 westata 的 Levenberg-Marquardt
在**同一个最优点**上的最后几位差异（RSS 一致到 6.3e-15，说明点估计相同），不是公式差异。

**S3（`treatreg_effects.py`，`AIPW.fit` 的 `atet` 分支）**
旧代码对 ATET 直接套用了 ATE 的平均：`PO(1) = mean_t(m1 + t·r1/p)`（多了一个 1/p；
treated 侧的增广项代数上恒等于 y，根本不该出现），`PO(0) = mean_t(m0)`
（`augmented0` 带 `(1−t)`，取 treated 均值后**控制组增广项全被乘成 0**）。
Stata 的 ATET 两个均值都以 `n1 = sum(t)` 归一，且控制组增广按 odds 重加权：
`PO(1) = (1/n1)Σ t·y`，`PO(0) = (1/n1)Σ [t·m0 + (1−t)·odds·(y−m0)]`。
影响函数同步修正：直接项 `(a0_i − t_i·μ0)/n1`，theta 项用
`∂a0/∂β = (1−t)·r0·odds·x`、`∂a0/∂b0 = (t − (1−t)·odds)·x`。
ATE 分支**一行未动**，并新增 `test_aipw_ate_is_unaffected_by_the_atet_repair` 钉死。

**S4（`treatreg_effects.py`，`NearestNeighbourMatching.fit`）**
旧代码两种估计量共用 `V = Σ(1+reuse)²σ²/n²`，`share0 = share1 = 1` 只是症状。
按 Abadie-Imbens 写成线性权重：
ATE `α_i = (2t_i−1)(1+K_i)/n`；ATET `α_i = t_i/n1 − (1−t_i)K_i/n1`
（treated 每个只进一次、**没有** reuse；控制按被复用次数 `K_i` 进）。
改后 ATET/ATE 比值从 1.000000 变成 1.109128（Stata 1.133439）。
**判定口径**：**结构性缺陷已消除（ATET 不再借用 ATE 的方差），但数值仍是 C 档**
（SE 残差 1.0e-1），残余在 `σ²_i` 而非权重，详见「未修 / 未达 A」第 1 条。
不要把这个记成"已修复"。

**S5（`did.py`）**
`covariates=None` 以前走 `_infer_covariates()`，把除 y/t/group/time 外**所有时变数值列**
当协变量，只发一条 warning。Stata 的 `(y)` 一个协变量都不加。反转默认：`None` → 不加；
旧行为改为显式 `covariates="all"`（仍发 warning，`inferred_covariates` 暴露清单）。
实测 `(y)` 与 Stata 逐位相同（0.0），`(y x1)` 仍 1.81e-16。

**S6（`resample.py`）**
`Jackknife.fit` 复用了基类的 `df_resid = n − k`。Stata 的 jackknife 用 `e(df_r) = N − 1`
（每个 replicate 一个自由度，与系数个数无关）。现场实测三组：`regress mpg weight`（k=2）、
`regress mpg weight foreign`（k=3）、`mean mpg`，Stata **全部**是 73（N=74）。
改为 `df_resid = n − 1`，`dist="t"`。

**S7（`resample.py`）**
`Bootstrap.fit` 通过基类拿到 `dist="t"`。Stata 的 bootstrap 打印 "Normal-based" 表、
用 `P>|z|`，且**不存 `e(df_r)`**。改为 `dist="z"`。`_cons` 的 95% CI 从
[35.22343, 43.65714] 变成 [35.29429, 43.58628]，与 Stata 的 [35.29429, 43.58628] 差 1.4e-9。

**T8（`nonlinear.py`，`Frontier.fit`）——选了「解析梯度 + 对梯度做差分」**
half-normal 似然在 `sigma_u → 0` 方向极平坦，**对目标函数**做二阶差分会在那里损失一半有效位。
新增 `frontier_gradient`（闭式推导，并与 `frontier_loglike` 的中心差分核对到 1e-8 相对），
`_numerical_hessian` 增加 `gradient=` 参数，改为差分**梯度**。

**归因已按 limited 组的复核改写（2026-10-24 二次复核，`repro_T8_stopping_point.py`）。**
原来的「步长缺陷」在换成梯度路线后就不存在了；剩下的差距是**两侧停机点不同**，不是
Hessian 精度。用**同一份 Hessian 代码**在两个参数点上算（n=400、seed=99、
`u ~ |N(0,0.7²)|`，即 limited 组 `repro_frontier.py` 与 FIXES.md 用的那个 DGP）：

| Hessian | step | 在 **Stata 的 `e(b)`** 上 | 在 **westata 的 `params`** 上 |
|---|---|---|---|
| 梯度（出厂的路径） | 1e-3 | 8.168e-06 | 2.201e-05 |
| 梯度（出厂的路径） | 1e-4 | **4.600e-06** | 2.502e-05 |
| 梯度（出厂的路径） | 1e-5 | 4.565e-06 | 2.505e-05 |
| 梯度（出厂的路径） | 1e-6 | 4.567e-06 | 2.505e-05 |
| 目标函数（旧路线） | 1e-3 | **9.697e-07** &nbsp;← 复现 limited 的 9.7e-7 | 2.844e-05 &nbsp;← 复现 2.8e-5 |
| 目标函数（旧路线） | 1e-5 | 4.021e-04 | 8.126e-03 |
| 目标函数（旧路线） | 1e-6 | 1.208e-01（发散） | 2.317e-01 |

即 **① Stata 点上 1e-6 量级、② westata 点上 2.5e-5 量级**，与 limited 组的结论一致：
**Hessian 已够准，θ̂ 才是瓶颈**。两侧的停机点证据：n=400 上
`ll(westata) = -602.6873375121513` 比 `ll(Stata e(b)) = -602.6873375123615` **高 2.1e-10**，
而 Stata 的梯度无穷范数是 `1.385e-04`、westata 是 `3.155e-06` —— Stata 先停了。
这与 REPORT §2.5 对 ARIMA 的结论同构。

**按队长要求：没有为了压这个数去改优化器默认**（风险大于收益，且会动到其它似然路径）。
改为在 `Frontier` 的模块 docstring 里写明：SE 在 Stata 参数点上与 Stata 差 1e-6，
在 westata 最优点上差 2.5e-5，差异来自两侧停机点而不是 Hessian 精度。

**保留原结果作为「Hessian 本身」的证据**（`repro_T8.py`，fixture DGP 两边落到同一点）：
SE 相对差 **4.027e-03 → 1.722e-07**。

**关于 n=500（任务书里写的样本量）的补充观察——"不要盲目追 Stata 数"的例证。**
同一个 seed=99 换成 n=500 时，即使在 Stata 自己的 `e(b)` 上，SE 仍有 6.5e-4 的差。
追查后确认**这是 Stata 侧的问题，三条独立证据**：

1. **Stata 的落脚点根本没收敛。** 在它自己的 `e(b)` 上，Stata 的梯度无穷范数是
   **6.162e-03**，westata 是 **8.219e-06**（差 3 个数量级）。
2. **westata 的似然更高。** `ll(westata) = -745.9410188288551` vs
   `ll(Stata e(b)) = -745.9410189040755`，westata **高 7.5e-08**。
   即 westata 停在一个**更好**的点上，Stata 的不是最优点。
3. **Stata 的 `e(V)` 与 `e(b)` 自相矛盾**（最直接的一条）。在 Stata 自己的 `e(b)` 上，
   若 Hessian 正确，则 `diag(H @ V_Stata)` 应恒等于 1。实测
   （`repro_T8_stopping_point.py` 第 3 段，**脚本实算**，不是硬编码）：

   | 参数 | `diag(H @ V_Stata)` | 偏离 1 的量 |
   |---|---|---|
   | `Intercept` | 0.999999999 | 1.7e-10 |
   | `x1` | 1.000000000 | 1.7e-11 |
   | `x2` | 1.000000000 | 2.0e-11 |
   | `/lnsig2v` | 0.999999953 | 4.7e-08 |
   | `/lnsig2u`（平坦方向） | **1.001296184** | **1.296e-03** |

   四个参数都到 1e-8 以内（三个斜率到 1e-10），**只有平坦方向 `/lnsig2u` 偏 1.296e-03**。
   **这意味着：在 n=500 上不存在任何一个参数点能同时复现 Stata 的 `e(b)` 与 `e(V)`** ——
   二者的关系本身不满足"V 是 H 的逆"。这比"两侧停在不同点"更强：后者只说明两侧位置不同，
   前者说明**目标本身不自洽**。追这个数等于复刻一个非最优点、外加一个不自洽的 `e(V)`。

   脚本把这条写成了断言（`stata_v_consistent` 必须为 `False`，否则 `RESULT: CHECK`），
   所以上面这行数字不是注释，是每次重跑都会验证的结论。

复现：`repro_T8_stopping_point.py` 第 3 段。本组以 limited 组的 n=400 DGP 为准
（两边都收敛，且 `diag(H @ V_Stata)` 为 1），n=500 的现象记录在此备查。

## 测试变更

| 文件 | 旧断言 | 新断言 | 依据（Stata 捕获） |
|---|---|---|---|
| `tests/test_teffects.py` | `test_aipw_atet_is_reported` 只查"有限且 > 0" | 新增 `test_aipw_atet_matches_stata_on_the_po_means_and_the_effect`：ATET 2.046425868899699 / se .06553792047575811 / PO(1) 3.539389746575276 / PO(0) 1.4929638776755771 | `teffects aipw (y x1 x2) (treated x1 x2), atet`（本模块 n=1200 seed=211 样本，`capture_aipw_tests.py`，2026-10-24） |
| `tests/test_teffects.py` | 无 ATE 护栏 | 新增 `test_aipw_ate_is_unaffected_by_the_atet_repair`：ATE 2.0188360574679662 / se .06250843751655079 / PO(0) 0.9801460721600921 | 同上，`teffects aipw (y x1 x2) (treated x1 x2)` |
| `tests/test_teffects.py` | nnmatch 只有点估计测试 | 新增 `test_nearest_neighbour_matching_gives_the_atet_its_own_variance`：ATET SE ≠ ATE SE，比值 ≈ 1.133439（rel 5e-2） | `teffects nnmatch (y x1 x2) (t), atet`（`repro_S4.py`） |
| `tests/test_teffects_did.py` | `test_didregress_matches_stata` 用裸 `"y"` 却断言 `(y x1)` 的 1.18613 | 改用 `"y x1"`（该数字所属的设定），并**新增** `test_didregress_bare_outcome_uses_no_covariate` 断言 `"y"` 的 1.1543177866493608 / se .07809416248412232 | `didregress (y) (treated)` 与 `didregress (y x1) (treated)`，P11 面板，`capture_did_tests.py`，2026-10-24 |
| `tests/test_teffects_did.py` | `test_didregress_standard_error_is_the_right_order` 用裸 `"y"` | 改用 `"y x1"`（与 0.0590465 同一设定） | 同上 |
| `tests/test_teffects_did.py` | `test_didregress_absorbs_rather_than_dummies` 期望 design 2 列 | 改用 `"y x1"`，仍然 2 列 | 旧断言依赖"裸 y 会推断 x1"这一缺陷 |
| `tests/test_did.py` | `test_bare_outcome_infers_time_varying_covariates_with_a_warning`（断言 `covariates == ["x1"]` 且 ATET ≈ 1.18613） | 拆成 `test_bare_outcome_adds_no_covariate`（`covariates == []`、无 warning、ATET 1.1543177866493608）＋ `test_covariates_all_opts_into_the_old_inference_with_a_warning`（`"all"` 仍选中 x1、ATET 1.1861296209175876） | 同上（同一面板，`capture_did_tests.py` 第二段） |
| `tests/test_nl.py` | 无 | 新增 `test_nl_gauss_newton_covariance_survives_a_conditioned_design`（SE 5.767884338 / .003883535774 / 6.257594481e-07）、`test_nl_refuses_an_unidentified_model_instead_of_inventing_a_standard_error`、`test_nl_finds_stata_start_values_without_being_given_them`、`test_nl_non_convergence_raises_by_default` | `nl (mpg = {b0}+{b1}*weight+{b2}*weight^2)` 与 `nl (price = {b0}*exp({b1}*mpg))`（`repro_S1_S2.py`） |
| `tests/test_nl.py` | `test_frontier_standard_errors_match_stata` 保持 | 新增 `test_frontier_observed_information_comes_from_the_analytic_gradient`（含**梯度 vs 数值梯度**的 1e-6 一致性断言） | `frontier y x1 x2`，u ~ \|N(0,0.7²)\|，n=500 seed=99（`repro_T8.py`） |
| `tests/test_resample.py` | 无 | 新增 `test_jackknife_uses_n_minus_one_degrees_of_freedom`（三组 k 都是 73）、`test_bootstrap_uses_a_normal_reference_distribution`（`dist == "z"`、`_cons` CI [35.29428573, 43.58628133]） | `jackknife: regress ...` ／ `bootstrap, reps(100) seed(20241024)`（`repro_S6_S7.py`） |

## 未修 / 未达 A 的

1. **S4 的 nnmatch SE：结构性缺陷已消除、数值仍是 C 档。**
   **不是"已修复"**。改前 ATET 的 SE 与 ATE **逐位相同**（比值 1.000000），因为两种估计量
   共用一条 `Σ(1+reuse)²σ²/n²`；这是本轮的真缺陷，已消除——现在按 Abadie-Imbens 的线性
   权重分开：ATE 用 `α_i = (2t_i−1)(1+K_i)/n`，ATET 用 `α_i = t_i/n1 − (1−t_i)K_i/n1`
   （treated 每个只进一次、无 reuse；控制按被复用次数 `K_i` 进）。改后 ATET/ATE 比值
   **1.109128**（Stata 1.133439）。
   但**数值仍停在 C 档**：SE 残差 1.0e-01（ATET, confounded nn=1）/ 5.5e-02
   （ATE, homoskedastic nn=1），目标是 ≤1e-2。剩余差距在 `σ²_i` 的估计量，不在权重。
   证据（`probe_S4_sigma.py`）：
   - 用**已知条件方差**的数据（`y = 2t + N(0,1)`，三组种子，残差方差恒为 1）做校准：
     如果 Abadie-Imbens 的"邻域样本方差"形式是对的，Stata 的
     `V_atet·n1² / (n1 + Σ K_c²)` 在 9 个组合上应当**全都等于 1.0**。
     实测 0.8990…1.0008（例：homo3 的 nn=1/2/3 = 0.9244 / 0.9608 / 0.9907；
     homo5 = 0.9929 / 0.9817 / 1.0008；confounded = 0.9030 / 0.9626 / 0.8990）。
     同一批权重下 Stata 的方差对应**不同**的 σ²，说明 `_matchnn_estimates`
     用的不是本文档化的那种邻域方差。
   - 用**已知条件方差**的数据（`y = 2t + N(0,1)`，三组种子，残差方差恒为 1）做校准：
     如果 Abadie-Imbens 的"邻域样本方差"形式是对的，Stata 的
     `V_atet·n1² / (n1 + Σ K_c²)` 在 9 个组合上应当**全都等于 1.0**。
     实测 0.8990…1.0008（例：homo3 的 nn=1/2/3 = 0.9244 / 0.9608 / 0.9907；
     homo5 = 0.9929 / 0.9817 / 1.0008；confounded = 0.9030 / 0.9626 / 0.8990）。
     同一批权重下 Stata 的方差对应**不同**的 σ²，说明 `_matchnn_estimates`
     用的不是本文档化的那种邻域方差。
   - 我把 4 种邻域定义 × 2 种 ddof × 2 种邻域来源（同臂 / 对臂）× nn=1,2,3 ×
     3 组数据全扫过（`probe_S4_sigma.py` 第 2 段），**最优组合**仍是
     ATE 5.01%、ATET 7.83%（h=1, ddof=1）；没有一组能同时压到 1e-2。
   - Stata 的 `vce(robust, nn(#))` 会改 SE（`nn(2)`→0.10827、`nn(3)`→0.10943、
     `nn(5)`→0.11038，数据同上），证明 σ² 的邻域大小本来就独立于 `nneighbor()`；
     我按 `vce` 的默认 `nn(2)` 实现，但文档没有给出 `nn(#)` 到邻域构造的确切映射。
   - 现状（`repro_S4.py` 实测）：`nn=2,3` 多数落在 2e-4–6e-3（达标或接近），
     `nn=1` 最差 ATE 5.5e-2 / ATET 1.0e-1。
   建议下一步：反编译 Mata `_matchnn_estimates`，或按 Abadie-Imbens (1996)
   式 4.4 的显式形式逐项对照。**未调参凑数**。

2. **S2 只到 B 档（1.57e-07）。** 原因已定位：同一个最优点（RSS 差 6.3e-15），
   差异来自 Stata Gauss-Newton 与 westata Levenberg-Marquardt 的收敛容差。
   要进 A 档需要复刻 Stata 的 GN 迭代与停止规则，收益低、风险高，未做。

3. **T8：Hessian 已修好；残余 2.5e-5 归因于停机点，不是本轮可修项。**
   判定保持 **A**，但要说清层次：
   - **Hessian 本身**（这是 T8 的被修对象）：在 Stata 自己的 `e(b)` 上评估，n=400 DGP
     差 **4.6e-6**、`frontier_0` fixture 差 **1.7e-07**；旧的目标函数路线在同一 fixture 上
     是 4.027e-03（换掉了）。**步长缺陷已不存在**——梯度路线在 1e-3…1e-6 全程稳定，
     目标函数路线在 1e-6 直接发散为 NaN。
   - **出厂结果报告的 SE**（在 westata 自己的 `params` 上）差 **2.5e-5**：这不是 Hessian
     精度问题，而是 `Frontier.fit` 的停机点。同一份 Hessian 代码在 Stata 点上 4.6e-6、
     在 westata 点上 2.5e-5（差 5.4 倍），而 westata 的似然比 Stata 高 2.1e-10、
     Stata 的梯度范数 1.4e-4 对 westata 的 3.2e-6。
   - **未改优化器**（按队长指示：风险大于收益），改为在 `Frontier` 的 docstring 里写明这
     两层差别。要追平 Stata 的 SE，应在共同参数点上比较，或复刻 Stata 的停机规则——
     后者是独立议题，超出本轮缺陷修复范围。
   - n=500 的额外现象（Stata 该点上未收敛、`e(V)` 与 `e(b)` 自相矛盾）见「每项的根因与改法」
     的 T8 段末，备查。

## 需要 Lead 裁决的

1. **`tests/conformance/test_stata_p11.py` 里两条 DID 测试现在会失败**，
   而 `tests/conformance/` 不在我的写作用域内。两条都是
   `Didregress.from_formula("y", "treated", did_frame(), ...)`（**裸 `"y"`**），
   却断言 `(y x1)` 的 `1.18613 / .0590465`。它们以前能过，**正是因为 S5 这个缺陷**：
   裸 `"y"` 会静默把 `x1` 当协变量。S5 修好后裸 `"y"` 是对的（1.1543177866493608），
   该断言必然失败。

   **请 Lead 改两处**（把裸 `"y"` 换成 `"y x1"`，即该数字所属的设定）：
   - `test_didregress_matches_stata`（`retest` 前 `:182-192`）
   - `test_the_two_did_spellings_share_the_atet_but_not_the_standard_error`（`:196-233`）

   本次现场 Stata 证据（`retest/fix/systems/capture_did_tests.py`，2026-10-24，P11 面板）：

   | 命令 | ATET | se |
   |---|---|---|
   | `didregress (y) (treated), group(unit) time(period)` | 1.1543177866493608 | .07809416248412232 |
   | `didregress (y x1) (treated), ...` | 1.1861296209175876 | .05904651458904048 |
   | `xtdidregress (y) (treated), ...` | 1.1543177866493608 | .07304712028056734 |
   | `xtdidregress (y x1) (treated), ...` | 1.1861296209175876 | .05522799411606224 |

   **自由度机制没有退化**：两种拼写都用 `(y x1)` 时，比值
   `0.05522799411606224 / 0.05904651458904046 = 0.9353302985018701`
   = `sqrt(1391/1590)` 到 1.7e-16；用 `(y)` 时比值 0.9353723499 也几乎相同
   （差 4.2e-5，因为去掉协变量后 `df_r` 的分子分母各差 1）。所以那条"比值等于闭式
   自由度因子"的断言在 `(y x1)` 下仍然成立到 1e-9。
   同文件 `test_didregress_rejects_a_time_invariant_treatment` 用裸 `"y"` 但只断言抛异常，
   不受影响。

2. **`retest/domains/systems_causal/run.py` 的 `didregress` 两行会变。**
   `run_did` 里为了绕开 S5，先把 frame 裁成 4 列再传裸 `"y"`；现在裁不裁都一样，
   结果与 Stata 仍是 0.0，不需要改，但那段注释（"westata cannot be told no covariates"）
   已经过时，`retest/` 归 Lead。

3. **能力回退：无。** 没有把任何命令降级为 `UnsupportedError`；S5 的旧行为通过
   `covariates="all"` 完整保留。

4. **与 `limited` 组的交接**：T8 只改了 `Frontier.fit`、`_numerical_hessian` 的签名
   （新增关键字参数，向后兼容）与模块 docstring；`frontier_loglike` 未改。
   **归因已按 limited 组的复核改写**（见上）：Hessian 已修好，残余 2.5e-5 是停机点，
   **未改优化器**。他们 FIXES.md 里"转交 systems 的 T8 方案"第 1 条建议（给 BFGS 传
   `jac=` 或加牛顿抛光以收停机点）**本轮未采纳**，理由按队长指示：风险大于收益，
   且会动到其它似然路径。如果 Lead 后续想追这 2.5e-5，那是独立议题。

## 跑过的 pytest

```
.venv\Scripts\python.exe -m pytest tests/test_nl.py tests/test_nonlinear.py \
    tests/test_teffects.py tests/test_teffects_did.py tests/test_resample.py \
    tests/test_did.py tests/test_systems.py -q
# 287 passed, 1 warning

.venv\Scripts\python.exe -m pytest tests -q
# 51 failed, 1257 passed —— 其中 49 条在 discrete / limited / panel / postest / tseries
# 的写作用域里（并发修改中的其它组），本组作用域内 0 条失败；
# 剩下 2 条是上面第 1 条要 Lead 改的 conformance 测试。
```

六个 `repro_*.py` 全部 PASS（`repro_S1_S2`、`repro_S3`、`repro_S4`、`repro_S5`、
`repro_S6_S7`、`repro_T8`），外加二次复核用 `repro_T8_stopping_point.py` PASS。

## 复现脚本

| 脚本 | 覆盖 |
|---|---|
| `repro_S1_S2.py` | S1 协方差（含秩亏拒绝）、S2 初值与 `strict` 默认 |
| `repro_S1_scale.py` | S1 复核：8 组缩放的秩判据不变性、SE 不变性、精确共线仍被拒、Stata SE 保持 |
| `repro_S3.py` | S3 ATET 修复 + ATE 护栏 |
| `repro_S4.py` | S4 nnmatch ATE/ATET SE 与结构性缺陷消除 |
| `repro_S5.py` | S5 `covariates=None` 语义反转 + `"all"` 逃生口 |
| `repro_S6_S7.py` | S6 `df_r`、S7 正态参考分布 |
| `repro_T8.py` | T8 的 Hessian 证据（fixture DGP 两边同点，1.722e-07） |
| `repro_T8_stopping_point.py` | T8 二次复核：同一 Hessian 在 Stata `e(b)` vs westata `params` 上的对比；n=500 的 Stata 侧异常 |
| `capture_did_tests.py` | 为 `tests/test_teffects_did.py`、`tests/test_did.py` 捕获 DID 真值 |
| `capture_aipw_tests.py` | 为 `tests/test_teffects.py` 捕获 AIPW 真值 |
| `probe_S4_sigma.py` | S4 的 σ² 加权方案扫描（保留作为"为什么停在 C 档"的证据） |
| `_common.py` | Stata 会话、数据构造、打印辅助 |
