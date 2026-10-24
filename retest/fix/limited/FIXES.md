# limited 组修复报告

组名：`limited`（tobit / truncreg / intreg / heckman / etregress / qreg / rreg；frontier 转交 systems）

本轮全部数字都由本机 StataNow **19.5 MP** 的实时会话捕获（脚本内嵌 `pystata`），
不是从已提交 fixture 抄的。每个修复都能用一条命令复现：

```
.venv\Scripts\python.exe retest\fix\limited\repro_tobit.py
.venv\Scripts\python.exe retest\fix\limited\repro_qreg.py
.venv\Scripts\python.exe retest\fix\limited\repro_heckman.py
.venv\Scripts\python.exe retest\fix\limited\repro_rreg.py
.venv\Scripts\python.exe retest\fix\limited\repro_frontier.py
```

## 逐项结果

| # | 缺陷 | 改前相对差 | 改后相对差 | Stata 命令 | 判定 |
|---|---|---|---|---|---|
| T1 | tobit `/sigma` SE 实为 `se(lnσ)` | 5.4e-2 (ll) / 1.6e-2 (ul) / 6.6e-2 (ll+ul) | **5.1e-8 / 1.8e-8 / 5.7e-8** | `tobit y x1 x2, ll(0)`、`ul(2.5)`、`ll(0) ul(2)` | 已修复 A |
| T2 | tobit `e(df_r)` 496 vs 498 | 1.4e-2（71 vs 72，auto 数据） | **0（逐位相等）** | `tobit ...` 六组 N/k + `noconstant` | 已修复 A |
| T3 | tobit 系数 SE 来自数值 Hessian(1e-5) | 8.9e-5 | **2.8e-8**；stdp **3.0e-8** | 同上 + `predict double s, stdp` | B（受 Stata 自身停机容差限制，见下） |
| T4 | qreg `vce(robust)` 在 τ≠.5 | 4.6e-2 (τ=.25) / 最高 1.28（报告值） | **1.5e-8**（τ=.25/.5/.75 × 同/异方差 6 组全过） | `qreg y x1 x2, quantile(τ) vce(robust)` | 已修复 A |
| T5 | heckman `e(sigma)`/`e(rho)` 公式错 | 6.7e-2 / 7.2e-2 | **1.8e-10 / 9.6e-10** | `heckman y x, select(s = z1 z2) twostep` | 已修复 A |
| T6 | etregress `e(rho)` 反号 | 2.05（反号） | **8.0e-12 / 2.4e-11**（两个 DGP） | `etregress y x1, treat(t = z1 z2) twostep` | 已修复 A |
| T7 | rreg SE 用 STR 伪值 | 2.06e-1 / 2.97e-1（最高） | **2.7e-11**；`rss/mse/rmse/r2/F` **≤3.2e-15** | `rreg y x1 x2` | 已修复 A |
| T8 | frontier 数值 Hessian 步长 | SE 4.0e-3（旧 step=1e-5，目标函数差分） | 现况：SE 2.5e-5 / 系数 9.9e-5；**归因已改写为 `Frontier.fit` 的停机点**（详见末节） | `frontier y x1 x2` | **转交 systems** |

补充档位说明：

- **T3 到 3e-8 而不是 1e-9，是 Stata 的容差不是我们的误差。** 证据：把我们的解析 score 在
  **Stata 自己的参数点**上求值，`|score|∞ = 1.5e-2…6e-6`（两侧 tobit）而不是 0；我们的参数点上
  是 `1e-13`。从 Stata 的点做一次牛顿迭代**精确落回我们的点**（`repro_tobit.py` 与其历史探针
  `_probe_two.py` 记录：`newton - stata = [+8.0e-10, -1.4e-8, +9.9e-9, -1.8e-8]` 的相反数）。
  另外 `technique(nr/dfp/bhhh)` 与 `tolerance(1e-14)` **不改变** Stata 的答案，而 `dfp`/`bhhh`
  会把它移动 5e-5 / 7e-5 —— 说明 Stata 的默认 `technique(nr)` 停在一个固定点上，但它与我们
  的精确最优差 1e-8 量级。这是**同一似然面上的停机点差**，与 REPORT §2.5 对 ARIMA 的结论同构。

## 每项的根因与改法

### T1 tobit `/sigma` 的 SE

`tobit.py::fit` 里 `cov = self._observed_information(theta, k)` 返回的是
`(β, lnσ)` 参数化下的协方差；`params` 最后一位却换成了 `σ`。所以报告的
`se(/sigma)` 是 `se(lnσ)`，比值恰为 `1/σ̂`（报告实测 0.945719091 vs `1/0.945721083`）。

改法：`cov[k, :] *= sigma; cov[:, k] *= sigma`（`dσ/dlnσ = σ` 的 Jacobian）。
三变体都到 5e-8。

### T2 tobit `e(df_r)`

`tobit.ado`（StataNow 19.5）第 103-105 行：

```stata
local df_m = `e(rank)' - 1 - `hascons'
ereturn scalar df_m = `df_m'
ereturn scalar df_r = e(N) - e(df_m)
```

即**尺度参数不耗自由度，有常数项时常数项也不耗**。westata 用 `N - k`（k 含常数与 σ），
所以 74-3=71 而 Stata 是 74-2=72。

反推数据（六组，`noconstant` 也测了）：

| 数据 | N | k | Stata df_m | Stata df_r | N−k | N−df_m |
|---|---|---|---|---|---|---|
| auto `tobit mpg weight foreign, ll(17)` | 74 | 3 | 2 | **72** | 71 | 72 |
| 合成 n=200 k=2 `ll(0)` | 200 | 2 | 1 | **199** | 198 | 199 |
| 合成 n=350 k=4 `ll(0)` | 350 | 4 | 3 | **347** | 346 | 347 |
| 合成 n=500 k=6 `ul(1)` | 500 | 6 | 5 | **495** | 494 | 495 |
| 合成 n=120 k=3 `ll(0) ul(2)` | 120 | 3 | 2 | **118** | 117 | 118 |
| 合成 n=300 k=2 `ll(0) noconstant` | 300 | 2 | 1 | **299** | 298 | 299 |

六组**全部**满足 `df_r = N − df_m`，且 `df_m = k − hascons`（`noconstant` 时 `hascons=0`，
`df_m` 仍等于斜率数 —— 因为 `rank` 也少 1）。改法照抄这条规则，并把 `df_resid`、
`df_model`、`e(df_r)`、`e(df_m)`、LR 的 p 值一处统一。

### T3 tobit 系数 SE

旧实现用 `_numerical_hessian(negative_loglike, theta, step=1e-5)`。删失项
（`lnΦ(±z)`）与 `β`、`lnσ` 的交叉二阶导不消失，中心差分在 1e-5 步长下把这块
丢到只剩 5 位（`repro_tobit.py` 的 `_probe_hessian` 记录：解析 vs 数值协方差在
step=1e-5 差 2.4e-5，step=1e-4 差 4.3e-7）。

改法：**解析 observed information**（`Tobit._information`，含未删失密度块、
左右删失块、以及 `β×lnσ` 交叉块）+ 解析 score（`Tobit._score`）+ 用它们做
一次带回溯的牛顿抛光（`Tobit._newton_polish`）。score 与 Hessian 都在
`tests/test_tobit.py::test_analytic_score_and_hessian_agree_with_finite_differences`
里对中心差分做了断言（rel 1e-6 / 1e-5）。

### T4 qreg `vce(robust)`

`qreg.ado`（StataNow 19.5）的 `GetVCE`：

```stata
mat accum `J' = `indep'            // J = X'X
... VCE_fitted ... (robust)
    fi  = (xb_u - xb_l)            // xb at tau+h minus xb at tau-h
    fi  = cond(fi>sqrt(c(epsdouble)), (2*band)/fi, 0)
    mat accum `H' = `indep' [iw=`fi']
    mat `H' = invsym(`H')
mat `V' = `tau'*(1-`tau')*`H'*`J'*`H'
```

所以 **① meat 是 `τ(1−τ)X'X`（不是 `Σψ_i²xx'`）；② bread 保留逐点密度 `f_i`**。
旧实现 meat 用 `ψ²`、bread 也保留 `f_i` —— 只有 τ=.5 时 `ψ²≡.25` 才碰巧对，
这正是"median-only 测不出来"的原因。verifier 的更正（"换 meat 后在异方差上仍差 15%–147%"）
经复核**不成立**：换 meat 后若 bread 仍留 `f_i`，异方差也到 1e-8（见
`repro_qreg.py`；用 Stata 自己的 τ±h 拟合代入同一公式可复现 `e(V)` 到 **7e-16**）。
余下的 1.5e-8 来自我们 `solve_lad` 在 τ±h 的停机精度（check 函数值比 Stata 高 1e-9），
不是公式。

### T5 heckman `e(sigma)` / `e(rho)`

`heckman.ado:700-746`：

```stata
gen double `delthat' = `mills' * (`mills' + `pxb')
sum `delthat' if `touse' & `seldep', meanonly
scalar `deltbar' = r(mean)
...
scalar `ebar'  = `rss' / e(N)                       // m = 选择样本量
scalar `sigma' = sqrt(`ebar' + `deltbar'*_b[`mills']^2)
scalar `rho'   = _b[`mills'] / `sigma'
```

westata 用 `sqrt(rss/(m-k))`。改法：`sigma = sqrt(rss/m + b_λ²·mean(λ(λ+w)))`，
`rho = b_λ/σ`，并补上 Stata 的 `rhometh=2` 截断（`|rho|>1` 时 `rho=±1` 且 `sigma=|b_λ|`）。
到 1.8e-10（与报告给的公式一致）。

### T6 etregress `e(rho)` 反号

`etregress.ado:490-492` 与 heckman 同形：

```stata
local sigma = sqrt(e(rss)/e(N) + `mtao'*`blambda'^2)
local rho   = `blambda'/`sigma'
```

`blambda` 是控制函数项 `λ` 的系数，`e(lambda) = [hazard]_b[lambda] = ρσ = b_ctrl`。
westata 写 `rho = -b_ctrl/sigma`，两个 DGP 都恰好互为相反数。改法：去掉负号，
并把 `sigma` 一起换成上面的公式（旧公式同时错 4.5%–7.0%）。到 1e-11。

### T7 rreg SE（与连带标量）

报告给的 `se(b) = e(rmse)·sqrt(diag((X'X)⁻¹))` 是对的，但**只改这一条只会从 2e-1 降到 4e-4**，
因为 `e(rmse)` 本身也错。`rreg.ado` 的真实流程有三处细节：

1. **迭代停止规则**是 `tolerance(.01)` 默认值：Huber 阶段 `while max|Δw| > 5*tolerance`，
   biweight 阶段 `while (max|Δw| > tolerance) | notyet`。跑到机器精度会停在**另一个点**
   （系数差 2e-4），因为 Stata 自己提前停了。
2. **`scale` 落后一轮**：biweight 循环体是
   `sum absdev` → `scale = r(p50)/.6745` → `gen weight`(用旧 res) → `_regress` →
   `drop res; predict res` → `replace absdev`。出循环时 `weight`/`absdev` 由**不同残差**构成，
   而最后的调整式

   ```stata
   replace absdev = (1-(res/scale)^2/tune^2)*(1-5*(res/scale)^2/tune^2)
   scalar aa = r(mean) ; lambda = 1+((e(df_m)+1)/e(N))*(1-aa)/aa
   ```

   混用**最后一轮 res** 与**上一轮 scale**。改成"用最终残差重算 scale"（看起来更自然）
   在 `tolerance(1e-9)` 下无差别，在默认 `tolerance(.01)` 下把 `aa` 移动 1.7e-4。
3. **`e(N)` 是加权回归的观测数**。`lambda` 里的 `e(N)` 取值时，最近一次回归是
   `_regress ... [aw=weight]`，零权重行被丢掉，所以它是**非零权重个数**（本例 386，不是 400）。
   用全样本 N 会差 1.7e-4。反解：`3/m = 386.0000001`（n=400 数据），四组数据 × 两档容差
   都是这个结论。

改法：`RobustRegression._iterate` 按 `rreg.ado` 逐句转录（含上面的顺序），
`fit` 用 `y_adj = xb + (λ·s/aa)(e/s)w` 做**未加权**回归，`rss/mss/rmse/r2/F` 全部取自它，
`cov = rmse²(X'X)⁻¹`。自洽约束 `rmse²·(n−k) = rss` 由构造保证（测试断言到 1e-12）。
`tolerance` 现在是一个参数，默认 0.01（Stata 默认值）。

顺带记录两条被推翻的旧注释：

- 旧 docstring 说"Huber 常数 1.345、`w = c_h/|u|`"。`rreg.ado` 实际写的是
  `abs(res) > 2*r(p50)`（`r(p50)` 是 `|res − median(res)|` 的中位数）。两者差 0.6745，
  **是不同估计量**。已改为 Stata 的写法。
- 旧 docstring 说 SE 用 Street-Carroll-Ruppert 伪值、残差 15%–30% 是"文档没写清"。
  实际 Stata 有确定闭式，复现到 3e-11；伪值**不是** Stata 报的量。`_pseudovalue_covariance`
  保留在文件里（`[R] rreg` 引用它），但 `fit` 不再调用，docstring 写明它不是 Stata 的 VCE。

## 测试变更

所有新断言都写死"现场捕获的 Stata 数字"，注释里写明命令、DGP、seed 与捕获日期（2026-10-24）。

| 文件 | 旧断言 | 新断言 | 依据（Stata 捕获） |
|---|---|---|---|
| `tests/test_tobit.py` | 无 `/sigma` SE 断言（只有"有限且为正"） | `test_sigma_standard_error_uses_the_sigma_jacobian`：三变体 `se(/sigma)` 对 Stata 1e-6；`STATA_TOBIT` 常量含全部 b/SE | `tobit y/y u/yb x1 x2, ll(0)/ul(2.5)/ll(0) ul(2)`，n=500 seed=2024 |
| `tests/test_tobit.py` | 无 `df_r` 断言 | `test_residual_dof_is_n_minus_df_m`：`df_m=2, df_r=498` 且 `df_r+df_m=N` | 同上；另六组 N/k 表的规则来自 `tobit.ado:103-105` |
| `tests/test_tobit.py` | "SE 正且有限" | `test_coefficient_standard_errors_match_stata` + `test_stdp_uses_the_coefficient_block_of_the_covariance`（stdp 逐点 1e-6） | `predict double _stdp, stdp`；`stdp[0]=.08376789256819156`、`max=.19301463582789766` |
| `tests/test_tobit.py` | 无导数一致性断言 | `test_analytic_score_and_hessian_agree_with_finite_differences` | 自校验（解析 vs 中心差分） |
| `tests/test_quantile.py` | `test_qreg_robust_meets_*` 只测 τ=.5（fixture 里只有 median） | `test_qreg_standard_errors_match_stata_at_every_quantile`：τ=.25/.5/.75 × 同方差/异方差 × iid/robust = 12 组，1e-6 | `qreg y x1 x2, quantile(τ) [vce(robust)]`，n=400 seed=31 / seed=77 |
| `tests/test_quantile.py` | 无 structural 断言 | `test_qreg_robust_meat_is_tau_one_minus_tau_xpx`：`V = J⁻¹(τ(1−τ)X'X)J⁻¹` 到 1e-12，且**不等于** `ψ²` meat | `qreg.ado` GetVCE（同上捕获） |
| `tests/test_quantile.py` | `test_rreg_standard_errors_are_close_to_stata`（rel=6e-2，把 15–30% 的差当"文档模糊"） | `test_rreg_standard_errors_are_rmse_times_the_ols_law`（1e-6）+ `test_rreg_reported_sum_of_squares_match_stata`（1e-9）+ `test_rreg_lambda_uses_the_nonzero_weight_count` | `rreg y x1 x2`，n=400 seed=67，5% 污染；`e(rss)=471.32198880778736` 等 |
| `tests/test_limited.py` | 无 heckman/etregress 派生量断言（旧测试只查 `rho = lam/sigma` 的内部自洽） | `test_heckman_sigma_uses_the_two_step_formula`（1e-8 + 公式重算 1e-12）、`test_etregress_rho_has_stata_sign`（1e-8 + 符号） | `heckman ..., twostep`、`etregress ..., twostep`，n=600 seed=7 |

## 未修 / 未达 A 的

1. **T3 的 tobit stdp 停在 3.0e-8，不是 1e-9。** 原因如上：Stata 自己的参数点上有
   `|score|∞ ≈ 6e-6`，即它并未停在我们这种"精确最优"。我们从它的点牛顿一步就精确回到
   我们的点，且似然完全相同（`-554.988723306911` vs `-554.988723306912`）。
   要更强只能把 Stata 的迭代也逐句复刻（`gsem` 内部，不可见）。**建议接受 B 档**。
2. **T4 余下 1.5e-8 来自 `solve_lad`。** 我们的 check 函数值比 Stata 的 LP 解高 1e-9
   （`repro_qreg.py` 里 `_probe_qreg2` 记录过 `check(b_up)` ws vs stata）。要更小需要
   复刻 Stata 的 `_qreg` 单纯形路径。**建议接受**：1.5e-8 远在 B 档内。
3. **T7 的两档容差。** verifier 用 Stata **默认** `tolerance(.01)` 的环境比，我们默认也是
   .01，到 1e-11。但 `tolerance(1e-9)` 下 Stata 自己会停在另一个点（我们与它一致到 1e-11）。
   两者不是同一目标点，报告里不要混用。
4. **`truncreg`/`intreg` 未在本轮重测**（原报告 C 档，BRIEF 未列入 T1–T8）。它们与 tobit
   共用 `_MLLimitedBase`；本轮把它的 `_stata_scalars` 签名同步到了 `discrete.py` 的新基类
   （见下"需要 Lead 裁决"第 2 条），`tests/test_limited.py` 里 truncreg/intreg 的既有断言全过。

## pytest 记录

```
.venv\Scripts\python.exe -m pytest tests/test_tobit.py tests/test_limited.py \
    tests/test_quantile.py tests/test_quantile_boot.py -q -p no:randomly
  -> 121 passed

# 所有引用本组估计量的 conformance / 下游测试（p6p7、etregress、p8、e_upgrade、
# predict_etable、predict_se、panel_nonlinear）：
.venv\Scripts\python.exe -m pytest tests/conformance/test_stata_p6p7.py \
    tests/conformance/test_stata_etregress.py tests/conformance/test_stata_p8.py \
    tests/conformance/test_stata_e_upgrade.py \
    tests/conformance/test_stata_predict_etable.py tests/test_predict_se.py \
    tests/test_panel_nonlinear.py -q -p no:randomly
  -> 112 passed

# 全量（Lead 会另跑；这里只记录本轮观察，避免误判为本组回归）：
.venv\Scripts\python.exe -m pytest tests/ -q -p no:randomly
  -> 10 failed, 1311 passed in 224s
```

那 10 个失败**没有一个引用本组的估计量**（逐个 grep：`test_stata_p11.py`、
`test_stata_p9.py`、`test_discrete.py`、`test_panel.py`、`test_panel_dynamic.py` 里
`Tobit/QuantileRegression/RobustRegression/Heckman/Etregress/Truncreg/Intreg` 的引用数为 0）。
按归属：

| 失败 | 归属组 | 现场 |
|---|---|---|
| `test_stata_p11.py::test_didregress_matches_stata`、`..._two_did_spellings_...` | systems | `didregress` |
| `test_stata_p9.py::test_xtpcse_matches_stata_coefficients` | panel | `xtpcse`：`ConsistencyError: no time periods are common to all panels` |
| `test_stata_p9.py::test_xtabond_matches_stata_...`、`test_panel_dynamic.py::test_xtabond_reports_the_instrument_count` | panel | `xtabond` |
| `test_discrete.py::test_logit_summary_reports_wald_chi2` | discrete | logit 的模型检验 |
| `test_panel.py` 四组（cluster CR1 / robust HC1 / SSC 默认 / cluster labels） | panel | SE 对齐 |

本组**修过的文件**只有 6 个，全在写作用域内（`git diff --stat`）：
`src/westata/estimators/{tobit,limited,quantile}.py`、
`tests/{test_tobit,test_limited,test_quantile}.py`。
`src/westata/estimators/nonlinear.py` 显示为 modified 是 `systems` 组的改动，**不是我改的**。

## 已裁决 / 已处理

1. **T8 归属**：队长已裁决 `nonlinear.py` 归 `systems`（同文件并发写会静默覆盖）。
   本节末给出实测与可落地方案。**我没有改 `nonlinear.py`**（`git status` 可验证）。
   补充：`systems` 已把 Hessian 换成"解析梯度的中心差分"，原来的步长缺陷消失了；
   现在 frontier 余下 2.5e-5 的归因是**优化器停机点**（见末节表格），请在复核时按新证据判。
2. **跨组契约变更**：队长裁决 `_DiscreteBase._stata_scalars` 的签名变更由 `discrete` 组负责，
   他们必须同时改 `limited.py` 的调用点。我已在 `limited.py` 同步过一次，够了
   （`_MLLimitedBase._stata_scalars` 现签名为
   `(result, params, cov, names, llf, nobs, canonical, groups)`）。
3. **`quantile.py` 归属**：队长确认归我，并已在 BRIEF §3 补上该行。
4. **`rreg` 的三处行为变更已批准**，并按队长要求写进 `quantile.py` 模块 docstring 的
   **"Deviations from the previous westata behaviour"** 一节（不只是本文件）。
   实测幅度（`_measure_rreg_changes.py` 的量，脚本已随探针删除，数字保留于此）：
   - 默认 `tolerance=0.01` vs `1e-9`：系数 **1.9e-4** 相对差，迭代 6 次 vs 23 次；
   - Huber 阈值 `2·median(|e−med e|)` vs 旧的 `1.345·scale = 1.9941·median`：
     阈值相对差 3.0e-3，**系数 7.9e-3** 相对差（`x1` 2.0142 → 2.0002）；
   - `e(mss/rss/rmse/r2/F)` 语义：旧 `e(rss)=330.3` vs Stata `471.3`。
   - 另有 `tolerance` 关键字参数、`e(mse)` 新增、`_pseudovalue_covariance` 保留但不再被调用。
   这三条已用 `tests/test_quantile.py` 的三条测试锁住
   （`test_rreg_default_tolerance_is_statas_not_machine_precision`、
   `test_rreg_sums_of_squares_describe_the_adjusted_regression`、
   `test_rreg_huber_threshold_is_two_mad_not_1345`），防止以后被"清理"回去。
5. **T3 停在 B 档已接受**。归因已按队长要求写进 `tobit.py` 模块 docstring 的
   **"Accuracy of the reported standard errors"** 一节，并明确写了"不要再去调容差，
   若差 <1e-8 说明在追 Stata 的迭代噪声，若 >1e-6 说明解析 Hessian/score 坏了"。

## 转交 systems 的 T8 方案

`src/westata/estimators/nonlinear.py`（systems 组所有）。复现脚本：
`retest/fix/limited/repro_frontier.py`（只测不改）。

**先说现状（`systems` 已改过后的实测）**：`systems` 已经把 `_numerical_hessian` 换成
"对解析梯度做中心差分"（`nonlinear.py:534-564` 新增 `gradient=` 分支），Hessian 步长问题
**已经不在了**。当前 westata vs Stata（n=400 seed=99 DGP）：

| 量 | 改后的 rel |
|---|---|
| 系数 `Intercept / lnsig2v / lnsig2u` | 1.1e-5 / 9.9e-5 / 4.1e-5 |
| SE `Intercept / lnsig2v / lnsig2u` | 2.2e-6 / 2.5e-5 / 1.8e-5 |
| 其余系数与 SE | ≤4e-7 |

**新的归因不是我原来写的"步长"，而是 `Frontier.fit` 的停机点**。证据（同一 DGP）：

| Hessian 的取值点 | step | max rel SE |
|---|---|---|
| **Stata 的参数点** | 1e-3 | **9.7e-7** |
| **Stata 的参数点** | 1e-4 | 3.1e-6 |
| **westata 自己的参数点** | 1e-3 | 2.8e-5 |
| **westata 自己的参数点** | 1e-4 | 6.4e-5 |

即：**同一份 Hessian 代码，在 Stata 的参数点上算能到 9.7e-7，在 westata 的参数点上算只有
2.8e-5**。所以剩下的 2.5e-5 里，Hessian 公式不是瓶颈，`θ̂` 才是 —— westata 的 `θ̂` 与 Stata 差
1e-5（`Intercept` 9.1e-6、`/lnsig2v` 5.6e-6、`/lnsig2u` 4.1e-5）。两者似然几乎相同
（`-602.687337512151` vs `-602.687337512362`，rel 3.5e-13），是同一个似然面上的停机点差异，
与 REPORT §2.5 对 ARIMA 的结论同构。

**建议（按推荐顺序）**：

1. **优先收 `Frontier.fit` 的停机点**，不是换步长：给 BFGS 传解析梯度（`jac=`），
   或加一步基于梯度的牛顿抛光（本组在 `tobit.py::_newton_polish` 用的就是这个办法，
   把 tobit 的停机差从 3e-8 压到 1e-13）。只要 `θ̂` 落到 1e-9 内，SE 会跟着落到 1e-6 内。
2. **不要**为了 SE 去调 `_HESSIAN_STEP`：现在的 `gradient=` 分支已经足够，调步长只会动
   Hessian 而 `θ̂` 不动，实测收益 ≤1 个数量级。
3. **原先的"步长"证据**（保留给需要回退的场景）：如果 `gradient=` 分支被绕过、
   退回对**目标函数**做中心差分，那么步长必须取 1e-3：

   | step | max rel SE vs Stata（在 Stata 点上） |
   |---|---|
   | 1e-1 | 4.0e-2 |
   | 1e-2 | 3.8e-4 |
   | **1e-3** | **9.7e-7** |
   | 1e-4 | 3.1e-6 |
   | **1e-5（旧默认）** | **4.0e-4** |
   | 1e-6 | 1.3e-1 |

   非单调：大步长是截断误差，小步长是相消误差，1e-5 正好落在两者之间的谷底。

4. **参考 DGP + Stata 真值**（n=400, seed=99；`x1,x2 ~ N(0,1)`，
   `y = 1 + 0.8x1 − 0.4x2 + N(0,1) − |N(0,0.7²)|`）：

   ```
   . frontier y x1 x2
   e(b)（打印顺序 x1, x2, _cons, /lnsig2v, /lnsig2u）
      x1        .79924180908
      x2       -.31679671345
      _cons     .90568547579
      lnsig2v   .056605938964
      lnsig2u  -.99766713314
   e(V) 的对角开方（同序）
      .052614014357  .056187964936  .55848529858  .29607949853  2.2928505474
   e(ll) = -602.687337512362
   ```

   验收：`frontier_loglike(theta=上面 e(b) 重排为 (cons,x1,x2,lnsig2v,lnsig2u))`
   应得 `-602.687337512362`（1e-12）；在该点上用 1e-3 步长的目标函数 Hessian
   求逆开方，5 个 SE 都 ≤1e-6（实测 9.7e-7）。**注意这条验收只考 Hessian**；
   要考 `Frontier.fit` 整体，则要求 `θ̂` 与上表 b 的 rel ≤1e-8，SE ≤1e-6 ——
   那是第 1 条建议要解决的。
