# verifier 独立复核报告（AUDIT）

> 复核对象：`retest/domains/` 下 7 个领域的 `NOTES.md` / `summary.json` / `results.jsonl`。
> 复核方式：**不复用** `retest/lib/stata_probe.py`，**不 import** 任何领域 `run.py`。
> 自建最小 Stata 桥（`my_stata.py`，62 行）+ 每个断言一个独立探针脚本（`p1*`–`p23*`）。
> 全部数字取自本次现场 StataNow 19.5 MP 会话（`config.init("mp")`，`c(stata_version)=19.5`）
> 与 `src/westata`（editable）实跑。数据**全部由我自造**（除 `auto.dta` 用 `tests/data/auto.dta`），
> 因此上游的"数据/样本不同"类错误会被直接暴露。
>
> 判定口径沿用 BRIEF §2.3。**A 档 ≤1e-9 / B ≤1e-6 / C ≤1e-3 / D ≤1e-2 / E >1e-2。**

## 0. 结论摘要（给 lead 的 5 行）

1. **7 个领域里 6 个的主线结论站得住**，但**上游有 3 处归因是错的**（不是数值错，是"为什么错"错了），
   其中 1 处（ologit）会把手修引向错误位置。
2. **最大的误判：ologit/oprobit 的 `e(V)` 根因不是"Jacobian 转置"**。我用 `sys.settrace`式插桩把
   `_to_cut_points` 的输入输出抓出来，逐步复算，**真实根因是 numpy 视图别名**
   （`theta = params[n_slopes:]` 之后 `params[n_slopes:] = cuts` 把 `theta` 也改成了 cut 点，
   于是 `exp()` 作用在**cut 点**上而不是 log-gap 上）。两个 Jacobian 都不是对方的转置
   （`max|J_code − J_ok^T| = 1.25`）。
3. **未被上游发现的第 2 个独立缺陷：ologit/oprobit 的 slope×cut 交叉块根本没做变换**
   （`cov[:ns, ns:]` 前后逐位相同）。任何 `lincom`/`test` 用到 cut 与斜率的组合都会错，
   而修好 Jacobian 也修不好它。
4. **tobit / qreg / heckman / etregress / rreg / nls / margins / xtpcse / xyabond / e(F) 的失败项
   全部复现**，数值档位与上游一致或更差。
5. **A 档抽查 12 条全部通过**（含 `e(V)` 全矩阵、GMM 两步 VCE、权重四型、cluster）= 无假阳性；
   **1 条"对齐遗漏"**：上游 `linear_iv` 的 `e(N)` 对 `fweight` 记录的是 369，而 westata 的
   `results.nobs` 是 140 —— 这只是 `nobs` 语义（不是 `e(N)`），上游没错，但**"样本量是否一致"的
   自动检查会误报**，需要在下游用 `e(N)` 而不是 `nobs`。

---

## 1. 逐条仲裁：所有 fail / error 条目

判定分类：
- **(a) 真实数值差异** —— 独立复算确认 westata 与 Stata 不同；
- **(b) 上游调用/参数没对齐** —— 差异来自上游脚本的调用方式，不是 westata 的错；
- **(c) 数据/样本不同** —— 两侧不是同一份数据。

### 1.1 discrete（上游 70 条：pass 15 / fail 39 / error 13 / skip 3）

| # | 上游条目 | 上游结论 | 我的独立复核 | 仲裁 |
|---|---|---|---|---|
| D1 | `logit vce(robust)` error | statsmodels 拒绝 `cov_kwds` | 复现：`ValueError: heteroscedasticity robust covariance does not use keywords`，Logit/Probit/NegativeBinomial 三条全崩 | **(a) 真能力缺失** |
| D2 | logit robust 诊断 8.34e-4 | Stata ML robust = sandwich × N/(N−1) | 复现：我手搭 sandwich，`ψ=(y−μ)x`，**×N/(N−1) 后与 Stata 差 6.4e-09**；不加因子差 1.05e-01 | **(a) 真差异**，归因正确 |
| D3 | logit cluster SE 1.67e-3 | 比值恒为 sqrt((N−K)/(N−1)·(G−1)/G) | 复现：测得比值 **0.99832916（G=30,N=600）**，与预测 `sqrt((597/599)·(29/30))=0.99832916` 逐位相同 | **(a) 真差异**，归因正确 |
| D4 | logit cluster `e(chi2)` 37.2% | Stata 换成斜率块 Wald | 复现：Stata `e(chi2)=129.49224689281`，我用 `b_s'V_s⁻¹b_s`（斜率块）算得 **129.4922468928**，差 4e-13；westata 的 `e(chi2)` 在 ml/cluster 下**都是 93.47889189865225**（逐位相同） | **(a) 真 bug**，归因正确 |
| D5 | probit 1.79e-9 / poisson 2.91e-8 | 停步点不同 | 复现 B 档量级 | **(a)** 精度差 |
| D6 | poisson HC1 == HC0 | 静默相等 | 复现：`np.array_equal(...) = True` | **(a) 真 bug** |
| D7 | nbreg 1.82e-4 | bfgs `gtol=1e-5` | 复现 C 档 | **(a)** 精度差 |
| D8 | `e(df_m)`：nbreg 报 3（Stata 2）、ologit 报 4（Stata 2） | 辅助参数被算进 df_m | 复现：Stata `e(df_m)=2, e(rank)=4, e(k_aux)=1`（nbreg）；`2/5/3`（ologit）。westata `e['df_m']` = **3 / 4** | **(a) 真 bug**，归因正确 |
| D9 | zip `e(p)` 差 1.15e+2 | Stata 用 `rank−1` 而非 `df_m` | 复现：Stata `chi2=89.9411565109`, `rank=6`, `df_m=2`, `e(p)=6.913314029e-18`。`χ².sf(chi2, rank−1) = 6.91331403e-18`（差 2.8e-15），`χ².sf(chi2, df_m) = 2.948e-20` | **(a) 真 bug**，归因正确 |
| D10 | **ologit/oprobit `e(V)` 2.09e+01** | **"`_to_cut_points` 的 Jacobian 恰是正确矩阵的转置"** | **复现数值（报告 8.43e-01 相对差 / 全线 E 档），但推翻归因。见 §2。** | **(a) 真 bug，(b) 上游归因错误** |
| D11 | ztp `predict,n` 8.68e-01 | Stata 给 `exp(xb)`，westata 给截断均值 | 可复现（同一语义分歧） | **(a) 语义不一致** |
| D12 | zip `predict,n` 崩溃 | `StataResults.predict` 分发器错 | 复现：`ValueError: matmul ... (size 6 is different from 3)`；同时 `model.fittedvalues` 与 Stata `predict,n` 差 **1.02e-09** | **(a) 真 bug**，归因正确（模型对、分发器错） |
| D13 | ologit `predict,pr` 崩溃 | 没传 `design_matrix=` | 复现：`ConsistencyError: the design matrix was not retained` | **(a) 真 bug** |
| D14 | `logit`, `logit,or`, `predict pr/xb` A 档 | ≤1e-13 | 抽查通过（见 §3） | **A 档成立** |

### 1.2 limited_quantile（上游 25 条：pass 0 / fail 22 / error 3）

| # | 上游条目 | 上游结论 | 我的独立复核 | 仲裁 |
|---|---|---|---|---|
| L1 | **tobit /sigma SE 5.4e-2 / 6.6e-2** | westata 报的是 `se(lnσ)`，cov 未做 delta 变换 | **复现并独立证明**。见 §4.1 | **(a) 真 bug**，归因正确 |
| L2 | tobit 系数 1e-8（B） | 优化器收敛 | 复现 B 档 | **(a)** 精度差 |
| L3 | truncreg 1e-8~1.6e-6（C） | 同 | 复现 | **(a)** 精度差 |
| L4 | intreg 2.9e-9/2.0e-6 | 同 | 复现 | **(a)** 精度差 |
| L5 | **heckman `e(sigma)`/`e(rho)` 错**（6.7%/7.2%） | Stata `σ=sqrt(RSS/m + b_λ²·mean(λ(λ+w)))` | **复现到 1.09e-10**：我独立用 Probit+OLS 手工重建，得 `1.127010047942`，Stata `e(sigma)=1.127010048066`，**相对差 1.09e-10**；`sqrt(RSS/(m−3))=1.024342147693`（差 9.1e-2）。**且 Stata `e(rho) == e(lambda)/e(sigma)` 精确到 0.0**，westata 报 `rho = b_λ/σ = −0.637064`（错）。 | **(a) 真 bug**，归因正确 |
| L6 | heckman twostep select 块 SE 高 6.8% | Stata 沿用 probit VCE | **复现**：Stata `heckman` 的 select 块 SE 与独立 `probit` 的 SE 比值 = `[0.9999891, 0.9999904, 0.9999955]`（即同一 VCE）；westata 的 `select:z1` = 0.0698215 vs Stata 0.0635998（高 9.8%） | **(a) 真 bug**，归因正确 |
| L7 | **etregress `e(rho)` 符号错** | westata 用 `rho = −b_ctrl/σ` | **两个独立 DGP × 两个 seed 全部复现**：westata 的 `e(rho)` 恒等于 `−b_ctrl/σ`（差 <1e-12），而 Stata `e(rho) = e(lambda)/e(sigma)` 精确到 0.0，两者**逐 DGP 反号**。见 §4.4 | **(a) 真 bug**，归因正确 |
| L8 | heckman/etregress ML、frontier exponential = error | 未实现 | **复现**：`UnsupportedError: heckman: only method='twostep' is implemented`、`etregress: only method='control'`、`frontier: distribution='exponential' is not implemented` | **(a) 真能力缺失** |
| L9 | **qreg vce(robust) τ≠.5 真 bug（E）** | westata meat=`Σψ²xx'`，Stata=`τ(1−τ)X'X`；换后复现到 9.0e-9 | **数值复现；归因部分成立、不完全成立**。见 §4.2 | **(a) 真 bug；(b) 归因需限定条件** |
| L10 | **rreg SE 15–30%（E）** | Stata 用 `e(rmse)·sqrt(diag((X'X)⁻¹))`，不是 pseudovalue | **复现到 2.24e-16（机器精度）**：`rmse*sqrt(diag(inv(X'X))) = [0.06203408, 0.06218766, 0.05841324]` 与 Stata 完全相同；westata `[0.06497951, 0.06594866, 0.05935866]`，差 5.7e-2。同时 `rmse²(n−k) == rss` 成立 | **(a) 真 bug**，归因正确 |
| L11 | frontier D 档 | 数值 Hessian 步长 | 复现 D 档 | **(a)** 精度差 |
| L12 | sqreg/bsqreg SE 2–3% | 重抽样方案不同 | 未独立复跑（bootstrap 耗时长且需 Stata RNG 对齐）；**上游自己已标注为"无法进一步归因"** | **未复核**（见 §6） |

### 1.3 linear_iv（上游 65 条：pass 46 / fail 15 / skip 4 / error 0）

| # | 上游条目 | 上游结论 | 我的独立复核 | 仲裁 |
|---|---|---|---|---|
| I1 | 21 `areg ... vce(cluster gid)` 4.86e-2（E） | **不是 westata 的错**，是 Stata `areg` 与 `reghdfe` 的 K 口径差 | 复现方向与量级：`areg` 用 K=16（含 13 个 gid 虚拟），`reghdfe` 用 K=3 | **(b) 上游判断正确**（这条是"防误判"记录，很关键） |
| I2 | 34/40/46 IV `vce(robust)` 1.09e-2（E） | 文档把 `HC1` 说成 Stata 的 robust 默认，实际是 HC0 | **复现**：`ivregress gmm ..., vce(robust)` vs westata `HC0` = **5.6e-16**；vs `HC1` = **1.089e-02**，且 `sqrt(N/(N−K)) = 1.0108896` 与实测比值逐位相同 | **(a) 真语义标签错**，归因正确 |
| I3 | 41 LIML cluster 4.53e-2 | Stata 默认 CR0，westata 恒 CR1 | 未逐条复跑；但 I2 的因子机制已确认，同源 | **归因可信** |
| I4 | 47 GMM cluster 点估计 3.58e-2 | Stata 把 W 也聚类 | 未独立复跑（需自建 cluster-GMM）；上游手算三候选矩阵表已给出可复现证据 | **归因可信但未复核** |
| I5 | 5/7/9/14 OLS `e(F)` 不随 cov_type 变 | **复现并强化**：见 §4.5 | **(a) 真 bug**，归因正确 |
| I6 | 52–56 `e()` 覆盖缺口 | Stata 有、westata 无 | 复现（`rss/mss/sum_w/N_clust/kappa/J` 均缺） | **(a) 真缺口** |

### 1.4 panel（上游 42 条：pass 22 / fail 18 / skip 1 / error 1）

| # | 上游条目 | 上游结论 | 我的独立复核 | 仲裁 |
|---|---|---|---|---|
| P1 | **12 `xtreg,fe vce(robust)` 17.36×** | Stata 的 `vce(robust)` **等价于** `vce(cluster panelvar)`；westata 映射成观测级 HC1 | **完全复现**：Stata 侧 `max|V_robust − V_cluster| = 0.0`（严格 0）；`areg absorb(id) vce(robust)` 的 SE 与 westata 的 HC1 差 **1.4e-16**。westata 的 `Intercept` SE = 0.00161116（cluster）vs Stata 0.00147327 | **(a) 真 bug**，归因正确 |
| P2 | **13 cluster SE 9.36%** | 小样本因子用错 K | **完全复现**：实测 Stata/westata SE 比值 = `[0.91441686×3]`，与 `sqrt((179/177)/(179/148)) = 0.914416859037` **逐位相同**；westata 的 cluster SE 与 Stata 的 `vce(cluster id) dfadj` 差 **1.6e-15** | **(a) 真 bug**，归因正确 |
| P3 | `e(df_r)`：Stata=29(G−1)，westata=148(N−K) | 同 | 复现：Stata `e(df_r)=29`，westata `df_resid=148` | **(a) 真 bug** |
| P4 | 23/24 `xtpcse` 非平衡 28.1%/41.9%（E） | `_aligned_slices` 按行号截断；pairwise 未实现 | **完全复现**：Stata casewise ≠ pairwise（比值 0.94/1.04/1.19），而 westata 两者**逐位相同**（`array_equal=True`） | **(a) 真 bug**，归因正确 |
| P5 | 20 `xtgls panels(correlated)` 30×6 秩亏 | 不可比 | 复现（Σ 秩 ≤ T） | **(b) 上游判断正确**（"不可比"） |
| P6 | 27–34 xt 非线性 B/C/D 档 | 数值 Hessian / 实现差异 | 未逐条复跑；`xtnbreg,fe` 的"实际是条件 Poisson"与 `xtabond` 的 `e(sig2)` 2 倍已在源码层确认存在 | **归因可信** |
| P7 | 37 `xtabond lags(2)` 参数名错位 | `param_names=(f"L{lags}.y",)` | **复现**：`lags=1 -> ['L1.y','x1','x2']`；`lags=2 -> ['L2.y','x1','x2','extra_0']`（真值应为 `L1.y, L2.y, x1, x2`） | **(a) 真 bug（最危险的一条）** |
| P8 | 21 `xtgls, corr(ar1)` error | 未实现 | 复现：`UnsupportedError: xtgls: corr(ar1) is not implemented` | **(a) 真能力缺失** |

### 1.5 postest（上游 77 条：pass 45 / fail 32）

| # | 上游条目 | 上游结论 | 我的独立复核 | 仲裁 |
|---|---|---|---|---|
| O1 | **41/43 margins logit AME SE 7.10e-3 / 5.38e-2** | `margins.py::_statistic_and_gradient` 用固定 h=1e-6 中心差分 | **完全复现并用解析解仲裁**：解析梯度 SE = **0.000057060551**，Stata = **0.000057060551**（差 **6.4e-11**），westata = **5.66553325e-05**（差 7.10e-3）。源码第 66 行 `h: float = 1e-6` 确认 | **(a) 真 bug**，归因正确 |
| O2 | **66 tobit /sigma SE 7.36e-1（E）** | 与 L1 同一个 bug | 复现（auto.dta 上 7.36e-1） | **(a) 真 bug** |
| O3 | **67 tobit `e(df_r)=71` vs 72** | Stata 的 df_m 只数斜率 | **复现**：Stata `e(df_m)=2, e(df_r)=72`，westata `df_resid=71` | **(a) 真 bug** |
| O4 | **72/73 tobit `predict stdf` 7.71e-2** | `_residual_variance` fallback 到 `rss/N` | **复现**：Stata `stdf²−stdp² = 14.2978442032 = σ²`；westata `= 12.1274048661 = rss/N`（rss=897.42796, N=74） | **(a) 真 bug** |
| O5 | 64 tobit 系数 SE 8.88e-5（C） | 数值 Hessian vs 解析 OIM | 复现 C 档 | **(a)** 精度差 |
| O6 | 15 `lincom` 拒绝裸常数 | 语法子集 | 复现（`ValueError`） | **(a) 能力缺口** |
| O7 | 28/29 `lrtest` df=0 / 非嵌套 | 行为差异 | 未独立复跑；上游给出 Stata 原始报错文本 | **可信但未复核** |
| O8 | 76/77 `pwcompare mcompare` / `contrast h.` | 明确拒绝 | 未独立复跑 | **可信但未复核** |
| O9 | 18/21 `nlcom` 比值 1.34e-5（C） | **westata 更接近解析值**，Stata 有 2e-6 噪声 | 未独立复跑该条；但同类"解析仲裁"方法在 O1 已证明上游做对了 | **方法可信** |

### 1.6 systems_causal（上游 56 条：pass 36 / fail 16 / error 2 / skip 2）

| # | 上游条目 | 上游结论 | 我的独立复核 | 仲裁 |
|---|---|---|---|---|
| S1 | **8 `nl` 二次式 SE 相对差 1.00（E）** | `pinv` 的默认 rcond 丢掉了最小奇异方向 | **完全复现到每一位**：`J'J` 奇异值 `[8.6797e15, 3.1939e7, 0.33908]`，`s_min/s_max = 3.907e-17 < 6.661e-16`。`inv` 给 Stata 的 `[5.76788434, 3.8835e-3, 6.2576e-7]`（差 1.37e-12）；`pinv` 默认给 westata 的 `[3.95e-7, 5.94e-4, 1.71e-7]`（差 1.0）；`pinv(rcond=0)` 差 1.89e-07 | **(a) 真 bug**，归因精确 |
| S2 | 11 `nl` 默认初值不收敛 1.42e+01 | 初值全 1 | 未独立复跑 | **可信** |
| S3 | **17 `teffects aipw, atet`** | ATE 精确、ATET 分支有缺陷 | **复现方向**：ATE `rel=1.1e-16` 级；ATET 点估计差 3.1e-3、SE 差 2.9e-2 | **(a) 真 bug** |
| S4 | **19–21 `nnmatch` SE 全错，且 ATET SE == ATE SE** | share0=share1=1 | **完全复现**：westata ATE SE 与 ATET SE **逐位相等**（`==` True），Stata 比值 **1.1947**；点估计 westata 与 Stata 差 1.2e-15 / 7.5e-16（A 档） | **(a) 真 bug**，且这是上游的**独立新发现**，成立 |
| S5 | **25 `didregress (y)` 自动加协变量（C）** | `covariates=None` 时推断 | **完全复现**：用上游同款 DGP，westata 得到 `param_names=['treated','x1']`、`treated=1.30572003`（= Stata 的 `(y x1)` 规格），而 Stata `(y)` 是 `1.3079574`；窄表后 westata 与 Stata `(y)` 差 **1.70e-16**。`covariates=[]` 抛 `ValueError: covariates must name at least one variable` | **(a) 危险默认值**，归因正确 |
| S6 | **45–47 jackknife SE 5e-7：Stata float32** | westata 更接近真值 | **完全复现**：用 `Fraction` 精确有理数算 74 次留一 OLS，真值 SE = `[2.02203366, 5.948844e-04]`。westata 差 **6.4e-15**；Stata 差 **5.11e-7**；把精确 replicate 过 `float32` 再套公式 → 复现 Stata 到 **2.20e-16** | **(b) 上游判断正确，westata 确实更准** |
| S7 | 48/49 jackknife/bootstrap 参考分布 df 与 t vs z | 口径差异 | 未独立复跑 | **可信** |
| S8 | 2 `sureg vce(robust)` error | 未实现 | 复现：`UnsupportedError` | **(a) 真缺口** |
| S9 | 22 `nnmatch biasadj` error | 能力缺失 | 复现：`TypeError: fit() got an unexpected keyword argument 'bias_adjust'` | **(a) 真缺口** |
| S10 | 44 `svy: mean, over()` skip | 无对应 | 复现：`SvyMean.from_frame() got an unexpected keyword argument 'over'` | **(a) 真缺口** |

### 1.7 tseries（上游 56 条：pass 35 / fail 21 / error 0）

| # | 上游条目 | 上游结论 | 我的独立复核 | 仲裁 |
|---|---|---|---|---|
| T1 | VAR 系数 A 档 8.45e-14 | 需按名字逐项翻译 | **复现**：正确的 permute（每方程 `[L1.y1, L1.y2, const]`）下 rel = **1.03e-14** | **A 档成立** |
| T2 | **VAR SE / `e(Sigma)` / Granger 的 E 档全是每方程自由度** | `sqrt((T−k_eq)/T)`，k_eq 为每方程回归元数 | **完全复现到 1e-16**：`e(N)=299`，比值实测 `[0.99497063 ×6]`（ptp=3.3e-16）对 `sqrt((299−3)/299)=0.9949706303122452`；归一后 SE rel **2.86e-16**。`e(Sigma)` 比值 `0.9899665551839465` 对 `(299−3)/299`，归一后 **2.13e-16**。VAR(2) 同（k_eq=5） | **(b) 归因正确，不是 westata 的错** |
| T3 | **dfuller 统计量 A 档；打印临界值 D 档** | `dfuller.ado` 硬编码旧表 + 线性插值 | **统计量复现 A 档**（我的探针第一版用了 `autolag` 默认导致 3e-2 假差异；改成 `autolag=None` 后 rel = **3.8e-16 / 4.0e-15 / 5.3e-15**）。Stata 打印的 5% 值随 N 微动（295→297→299 时 −3.4564→−3.45624→−3.45608），与"硬编码插值"一致 | **(a) 真差异，但归因正确**；**注意：这一条的第一版探针暴露了"用错 API 会造出 3e-2 的假差异"的风险** |
| T4 | **vecrank 选阶会翻（3.8369 vs 3.76/3.8415）** | Stata 用 Johansen(1995) 表，statsmodels 用 Osterwald-Lenum | **部分复现**：我在自己的数据上 Stata 打印 r=1 的 5% 临界值 = **3.76**，而 `VECM.johansen()` 返回的 `trace_critical[1] = [2.7055, **3.8415**, 6.6349]`。**两个临界值表不同这一事实成立**；但"选阶翻转"我只在统计量落在 (3.76, 3.8415) 区间时才发生，我的数据 trace(r=1)=2.074 不在该区间，**未能独立重现翻转** | **(a) 表确实不同；翻转结论未被独立验证** |
| T5 | ARIMA 系数 1e-3~5e-3（C/D），ll 1e-9 | 同一似然面停步点 | 未独立复跑 | **归因可信** |
| T6 | arch GARCH(1,1) E 档 | SE 从 OPG→robust；presample h₀ 不同 | 未独立复跑 | **归因可信（含交叉验证表）** |

---

## 2. 最重要的误判：ologit/oprobit `e(V)` 的根因

### 2.1 上游的说法
> "`src/westata/estimators/discrete.py:396-398` 写的是 `J = eye(n_cut)` + `J[k, :k] = exp(theta[1:k+1])`。
> 把两者按实测 theta 排出来 …… **westata 的 J 恰好是正确 J 的转置**"

### 2.2 实测（`p1i` / `p1j` / `p1M`）
插桩 `_to_cut_points` 拿到真实输入输出（ologit，n=600，4 类）：

```
params_in  = [ 0.82804695 -0.28833656 -1.39170366  0.13636621  0.22202295]
theta(真)  = [-1.39170366  0.13636621  0.22202295]      <- log-gap 参数化
cuts(输出) = [-1.39170366 -0.24560213  1.0029979 ]

J_code (源码)      J_correct (正确)         J_correct.T
[[1,      0,     0],  [[1, 0,       0],      [[1,        1,        1],
 [1.1461, 1,     0],   [1, 1.1461,  0],       [0,        1.1461,   1.1461],
 [1.1461, 1.2486,1]]   [1, 1.1461,  1.2486]]  [0,        0,        1.2486]]
```

- `max|J_code − J_correct| = 0.2486`，`max|J_code − J_correct^T| = **1.2486**` → **不是转置**。
- **决定性检验**：用**插桩抓到的真实 cut 点**构造 `J_alias`（即把 `exp()` 作用在 cut 点上），
  则 `V_out == J_alias · V_θ · J_alias'` **成立，max|diff| = 0.000e+00**（ologit 与 oprobit 都是）。
  用 `J_code`（按 theta 算）或 `J_correct` 都**不成立**。

### 2.3 真实机制（numpy 视图别名，`discrete.py:386-398`）
```python
theta = params[n_slopes:]        # ← 视图，不是拷贝
...
params[n_slopes:] = cuts         # ← 把 theta 也一起改成了 cut 点
...
J[k, :k] = np.exp(theta[1 : k + 1])   # ← 这里 exp() 看到的是 cut 点
```
于是 `J_code(源码)` 实际不是 `J_code(theta)` 而是 `J_code(cuts)`。
**这就是为什么同一个数据上前两版探针凑不出 `J_code` 的效果**——源码的 `exp` 参数被别名改了。

### 2.4 第二个独立缺陷（上游未发现）
```python
block = J @ cov[n_slopes:, n_slopes:] @ J.T   # 只改了 (cut, cut) 块
cov[n_slopes:, n_slopes:] = block
```
**slope × cut 交叉块 `cov[:ns, ns:]` 从头到尾没动**。实测 `cov_in[:2,2:]` 与 `cov_out[:2,2:]`
**逐位相同**（`np.allclose` True）。正确应做 `cov[:ns, ns:] @ J_correct.T`。
→ 任何 `lincom`/`test` 混合斜率与 cut 的假设都会被算错，**光修 Jacobian 修不好这一块**。

### 2.5 修复后的数值
用 westata **自己的**后端协方差 `V_θ`，只把 Jacobian 换成正确的（并补齐交叉块）：

| | cut1 SE | cut2 SE | cut3 SE | 全 `e(V)` rel |
|---|---|---|---|---|
| Stata | 0.10432381 | 0.08740299 | 0.09620776 | — |
| westata 报告 | 0.10432379 | **0.07143547** | **0.17728862** | **3.944** |
| 正确 J 重算 | 0.10432379 | 0.08740305 | 0.09620791 | **3.269e-05** |
| 正确 J 重算（oprobit） | 0.06006359 | 0.05350361 | 0.05700968 | **5.384e-05** |

即：**只有 Jacobian（+交叉块）错，后端协方差本身是对的**。上游"点估计对、协方差错"的大方向正确，
但"转置"这个具体描述是错的，会让修复者去改对角线结构而不是改 `theta` 的别名/取值。

---

## 3. A 档抽查（12 条，全部重跑）

数据全部自造（`auto.dta` 除外），**每条都核对两侧样本量**，按名对齐（不按位置）。

| # | 条目 | N(Stata)/N(westata) | 系数 rel | SE rel | 结论 |
|---|---|---|---|---|---|
| 1 | `regress mpg weight foreign` | 74 / 74 | 5.114e-15 | 6.807e-16 | 真 A |
| 2 | `regress mpg weight i.foreign` | 74 / 74 | 3.699e-14 | 5.105e-16 | 真 A（因子对齐正确） |
| 3 | `regress y x1 x2, vce(cluster gid)`（14 组） | 140 / 140 | 1.129e-15 | 6.504e-16 | 真 A |
| 4 | `regress ... [fweight=w]` | **364** / 140 | 8.176e-16 | 2.011e-16 | 真 A；`e(N)=364` 双方一致（见 §5.1） |
| 5 | `regress ... [pweight=w]` | 140 / 140 | 3.865e-16 | 4.587e-16 | 真 A |
| 6 | `ivregress 2sls yiv x1 (e1=z1)` | 140 / 140 | 2.629e-15 | 2.674e-15 | 真 A |
| 7 | `ivregress gmm yiv x1 (e1=z1 z2)` | 140 / 140 | 1.374e-16 | 3.288e-16 | 真 A；**全 `e(V)` rel = 5.53e-15** |
| 8 | `eivreg ..., reliab(x1 .9 x2 .95)` | 140 / 140 | 9.289e-16 | 1.056e-15 | 真 A |
| 9 | `xtreg y x1 x2, fe` | 140 / 140 | 2.854e-16 | 1.653e-16 | 真 A |
| 10 | `xtreg y x1 x2, re` | 140 / 140 | 6.765e-16 | 5.544e-16 | 真 A |
| 11 | `areg y x1 x2, absorb(gid)` | 140 / 140 | 2.854e-16 | 1.653e-16 | 真 A |
| 12 | `logit foreign weight mpg` | 74 / 74 | 6.578e-11 | 5.299e-11 | 真 A/B 边界（上游报 1.09e-13，我的数据 6.6e-11，同档） |

补充（非抽查，为验证全矩阵）：`sureg` 2 方程全 `e(V)` 6×6 rel = **3.86e-11**；
`ivregress gmm` 全 `e(V)` = **5.53e-15**。两者都支持上游"跨方程协方差全对"的说法。

**A 档假阳性检查结论：未发现假阳性。** 特别核了三项常见假阳性来源：
1. **样本量一致**：12 条里 11 条 N 完全相等；第 4 条 `fweight` 的 `results.nobs=140` 与
   `e(N)=364` 不同，但**两边 `e(N)` 都是 364**，属 `nobs` 语义而非样本差异（§5.1）。
2. **是否真用 `e(V)` 全精度**：我全部用 `sfi.Matrix.get("e(V)")` 与 `res.cov_params()`，
   未解析任何打印表。
3. **系数是否被对齐到错误位置**：我按名对齐；尝试按位置对齐时**立即出现 inf**（如
   `poisson` 的 Stata 序是 `(x1,x2,_cons)`、westata 是 `(Intercept,x1,x2)`），
   说明上游那几条"按位置比"的条目若未显式声明 `perm` 就会假失败——上游确实都声明了。

---

## 4. 高风险点专项复核

### 4.1 tobit 的 `/sigma` vs `var(e.y)` 换算 ✅ 确认是真 bug

任务点名要求复核。我做了**三条互相独立**的验证：

```
Stata: e(b) 最后一行是 var(e.mpg) = 14.2978442032, SE = 2.71795624
       ⇒ sigma = sqrt(14.2978442032) = 3.7812490269
       ⇒ se(sigma) = 2.71795624/(2·3.7812490269) = 0.3593992646
       ⇒ se(ln sigma) = 2.71795624/(2·14.2978442032) = 0.0950477639
westata: /sigma = 3.7812490007（点估计对，rel 6.9e-09），bse = 0.0950472499
```

1. **比值恒等式**：`westata_se / se(sigma) = 0.264461448`，`1/sigma = 0.264462880`
   （差 5.4e-6，即数值 Hessian 步长残差）。
2. **对 Stata 自己的 `se(lnsigma)`**：`rel(westata_se, se(ln sigma)) = 3.07e-04`。
3. **对独立中心差分 Hessian**（我自己写 `tobit_ll`，在 `(b, ln σ)` 下做 4 点差分）：
   `se(ln sigma) = 0.0950180678`，**rel(westata, 我的 Hessian) = 3.07e-4**；
   `se(sigma) = sigma·se(lnσ) = 0.3592869740`，**rel(westata, 我的 sigma 尺度) = 7.36e-01**。

→ **westata 报的是 log 尺度的 SE，未做 `V_σ = σ²·V_logσ` 的 Jacobian 修正。确认为真 bug。**
副作用：σ 的 t 值从 10.52 变成 39.79（放大 1/σ = 3.78 倍）。
上游对"Lead 看到的 7.9e-2 是 transform 双向错配"的解释也成立（我自己也踩过同样的坑，
见 `p2` 第一版）。

### 4.2 qreg `vce(robust)`：数值对，但上游的归因只在部分数据上成立 ⚠️

上游说："westata meat = `Σψ_i² x x'`，Stata meat = `τ(1−τ)X'X`；换后 τ=.25 复现到 9.0e-9。"

我在**四个数据集 × 三个 τ** 上测这个具体断言：

| 数据集 | τ | Stata SE | westata | `τ(1−τ)X'X` meat 换后 |
|---|---|---|---|---|
| **上游原 DGP**（seed 31） | .25 | [0.09707325, 0.08136206, 0.09471935] | rel **4.63e-2** | rel **1.93e-8** ✅ |
| 上游原 DGP | .5 | [0.08224835, 0.06947293, 0.07994508] | rel **5.84e-9** | （ψ²≡τ(1−τ)，同一个矩阵） |
| 上游原 DGP | .75 | [0.09827913, 0.08423291, 0.09758433] | rel 3.54e-2 | rel **1.44e-8** ✅ |
| 我的同方差 DGP | .25 | — | rel 2.39e-2 | rel **8.84e-9** ✅ |
| 我的同方差 DGP | .75 | — | rel 1.13e-1 | rel **1.59e-8** ✅ |
| **我的异方差 DGP** | **.25** | [0.07738305, 0.08379782, 0.09690685] | rel **2.25e-1** | rel **1.54e-1** ❌ |
| 我的异方差 DGP | **.75** | [0.07349862, 0.0715494, 0.10526457] | rel **1.28** | rel **1.47** ❌ |

**结论**：
- **"westata 在 τ≠.5 用错了 variance 定义"这个大方向成立**（换 meat 后在上游数据 + 同方差数据上掉到 1e-8）；
- **但"换 meat 就能复现 Stata"不是普适的**：在异方差数据上仍差 15%–147%，说明**bread（逐点密度 `f_i`）也参与了差异**；
- 另外，**上游"westata 在 τ=.25 偏高 3.5–4.6%"是不完整的**：我在异方差 DGP 上看到西侧偏高 22%，在 τ=.75 甚至偏高 128%（westata 0.163 vs Stata 0.0715）。上游只报了他们的数据，**低估了量级**。

**这条 lead 必须谨慎表述**：不要说"换成 `τ(1−τ)X'X` 就修好了"，应说
"variance 定义与 Stata 不同，且**即使换掉 meat 仍有 bread 层面的差异**，需要同时核 `f_i` 的构造"。

### 4.3 cluster 的自由度 ✅ 三个域全部确认

| 场景 | Stata | westata | 我的独立证据 |
|---|---|---|---|
| `logit, vce(cluster)`（ML，G 组） | 只有 `G/(G−1)` | 多乘 `(N−1)/(N−K)` | 比值 `0.99832916` = `sqrt((597/599)(29/30))` 逐位相同 |
| `regress, vce(cluster)`（OLS，G=14） | `CR1 = G/(G−1)·(N−1)/(N−K)`，`df_r=G−1=13` | SE 完全一致；但 `e(df_r)` 存 `N−K=137` | `results.df_resid=13`，`test x1` 的 p 与 Stata 差 7.1e-15 |
| `xtreg, fe vce(cluster)`（G=30） | `df_r = G−1 = 29`，小样本因子 K 只数报告系数 | 与 Stata `dfadj` 逐位相同（1.6e-15），与默认差 9.36% | 比值 `0.914416859037` = `sqrt((179/177)/(179/148))` |
| `ivregress, vce(cluster)` | 2SLS/2sls 默认 **CR0**；LIML/GMM 恒 **CR1** | 2SLS 走 CR0（对），LIML/GMM 走 CR1（错） | `rel(GMM HC0 vs Stata robust)=5.6e-16`；`HC1` 差 1.089e-2 = `sqrt(N/(N−K))` |

**关掉 3 个域后的一致结论：Stata 的小样本因子随**命令**变化，不是全局常量。**
所以"统一加 CR1"或"统一不加"都会在某一族上错。这是本次最容易被误归纳的一点。

### 4.4 aweight/pweight 语义 ⚠️ 上游只测了"点估计+SE 一致"，但没测**语义混淆**

我的复核（4 型权重 × 1 个 DGP，n=140，`w∈{1..4}`）：

| 权重 | Stata `e(N)` | Stata `e(df_r)` | westata `e(N)` | westata `e(df_r)` | 系数 rel | SE rel |
|---|---|---|---|---|---|---|
| `fweight` | **364**（=Σw） | 361 | **364** | 361 | 9.52e-16 | 1.98e-16 |
| `iweight` | **364** | 361 | **364** | 361 | 9.52e-16 | 1.98e-16 |
| `aweight` | 140 | 137 | 140 | 137 | 9.93e-16 | 1.34e-16 |
| `pweight` | 140 | 137 | 140 | 137 | 9.93e-16 | 3.43e-16 |

- **复制型（f/i）与精度型（a/p）的 N 语义完全正确**（`fweight`/`iweight` 的 `e(N)` 是权重和，
  `aweight`/`pweight` 是观测数）。
- **`pweight` 隐含 `vce(robust)` 也对**：`[pweight=w]` 直跑与 `[pweight=w], vce(robust)` 的
  westata SE 都是 `rel=3.43e-16`（即 westata 的 pweight 默认就走 HC1）。
- ⚠️ **但 `results.nobs` 在 `fweight` 下是 140 而不是 364**，而 `e('N')` 是 364。
  任何用 `res.nobs` 当 `N` 算自由度的下游代码都会错（上游 `linear_iv` 已在 NOTES 里点出
  `sum_w` 缺失，但**没有说 `nobs ≠ e(N)` 这个更直接的坑**）。**建议 lead 把这条写成显式警告。**

### 4.5 二值模型用 z 而非 t ✅ 确认

- Stata `logit` 的 `e(df_r)` **不存在**（`None`），`test weight` 返回 `r(chi2)=14.913858384769`、
  `r(F)=None`、`r(df)=1`。而 `z(weight)² = 14.913858384769` —— **`chi2 == z²` 精确成立，
  证明用的是标准正态而非 t**。
- westata `Logit` 的 `dist = "z"`、`conf_int()` 用 1.959964（我按 z 手算的 CI 与 Stata 打印一致）。
- **结论：二值模型两侧都用 z，一致。** 上游 `discrete` 没专门测这一条，但行为是对的。

### 4.6 GMM 的两步 VCE ✅ 确认 A 档

- `ivregress gmm yiv x1 (e1=z1 z2)` vs `ws.IVGMM(...)`: 系数 rel **1.51e-16**、SE rel **3.05e-16**。
- **全 `e(V)` 9 个元素** rel = **5.53e-15**，逐元素最大绝对差 **6.94e-18**。
- `vce(robust)` 的正确映射是 `HC0`（rel **5.64e-16**），`HC1` 差 **1.089e-02**（= `sqrt(140/137)`）。
- Stata `e(J)=0.7525403756`，westata **不存 `e(J)`** —— 上游 `linear_iv` §5 的覆盖度缺口成立。

---

## 5. 我发现的**误判**（最重要的一节）

### 5.1 ⚠️ `fweight` 的样本量检查会假报"样本不同"（工具层面，非 westata 缺陷）

上游 `linear_iv` 的 10 号记 `fweight` 为 A 档（正确）；但如果下游加一条
`assert st.nobs() == res.nobs` 的通用检查（BRIEF §3.4 建议这么做），**`fweight` 会失败**：

```
Stata e(N) = 364   st.nobs() = 140（原始行数）
westata res.nobs = 140,  res.e['N'] = 364
```

**这不是样本不同**——两侧用的都是同样 140 行、同样的权重。**正确做法是比较 `e(N)`，
不是 `nobs`**。我在 §3 的抽查里对这条做了单独说明。**建议 lead 在报告里显式写这一句**，
否则后续复核者会把它当成新的失败。

### 5.2 ⚠️ ologit 根因："转置"是错的（见 §2）—— 会误导修复

后果：如果按"转置"去改 `J[k,:k] = exp(...)` 的下标顺序，**仍然不会对**，因为真正错的是
`theta` 被别名改写成了 cut 点。我在 `p1j` 里验证了：只有把 `exp()` 的参数换回**真正的 theta**
（即 `theta = params[n_slopes:].copy()` 或先取 `theta` 再赋 `params`），正确 Jacobian 才能成立。

### 5.3 ⚠️ qreg 的归因被过度推广（见 §4.2）

上游原文："换成后者后 τ=.25 复现到 9.0e-9、τ=.75 到 1.4e-8"——这在**他们的数据**上成立，
但**不是普适公式**。而且"westata 在 τ=.25 偏高 3.5–4.6%"这句在异方差数据上偏低估计
（实测偏高 22% 到 128%）。

### 5.4 ✅ `xtpcse` 的"只在非平衡时才是缺陷"—— 我确认了上游的隐含限定，并补了一条证据

上游只在**非平衡**面板上报了 casewise/pairwise 差异。我在**平衡**面板（20×6）上补测：

```
Stata  casewise se = [0.1349119  0.07035166 0.04822153]
Stata  pairwise se = [0.1349119  0.07035166 0.04822153]   -> 完全相同
westata casewise/pairwise 也相同                          -> 与 Stata 一致
按名对齐后 rel(westata, Stata) = 1.439e-15               -> A 档！
```

**结论：平衡面板上 westata 的 xtpcse 是精确的（1.44e-15）；"pairwise 静默等于 casewise"
只在非平衡面板上构成缺陷。** 上游的表述没有明确这个限定，建议 lead 补上
"仅在非平衡面板上"这五个字，否则会被读成"xtpcse 整体不可用"。

**顺带暴露一个通用陷阱（值得写进报告）**：这条的 westata `param_names` 是
`['Intercept','x1','x2']` 而 Stata 是 `['x1','x2','_cons']`。**按位置比较会得到
rel = 9.18e-01 的假失败**；按名对齐后是 1.44e-15。**A 档与否完全取决于对齐方式**，
这也解释了为什么我在自己探针的第一版里出现过多次"inf / 0.9 量级"的假差异。

### 5.5 ✅ 一个上游**做对了但我一开始做错**的地方（写下来防后人踩）

- **`ws.unitroot(..., maxlag=2)` 默认 `autolag="AIC"`**，会重新选滞后阶。
  我第一版没传 `autolag=None`，得到 dfuller 统计量差 **3.4e-2** 的假失败。
  上游 `tseries/run.py:901` **显式传了 `autolag=None`**，是对的。**这是 westata API 的陷阱，
  不是上游的错**，但值得写进文档。
- **`ws.VAR(...)` 的 `res.params` 是 statsmodels 的 DataFrame**（行 `[const, L1.y1, L1.y2]`），
  而 Stata `e(b)` 是 `[L1.y1, L1.y2, _cons]`。按位置对齐会得到 **rel = 3.58e+01** 的假失败。
  上游用了显式 `map_stata_var_names()`，是对的。

---

## 6. 未复核 / 复核不充分（如实记录）

| 项 | 原因 |
|---|---|
| `sqreg`/`bsqreg` 的 2–3% SE 差 | Stata RNG 无法在 numpy 复现；上游已做 4×500 池化，我只能引用其表，未独立重跑 |
| `ivregress gmm vce(cluster)` 点估计（I4） | 需自建"W 也聚类"的 GMM，工作量超出本次复核预算；上游给了三候选矩阵对照表，方法可信 |
| `xtnbreg, fe` 的 HHG 似然、`xtabond` 的 `e(sig2)` 2 倍 | 未逐条复跑数值；`e(sig2)` 的 1.993 倍在源码层可见 |
| `lrtest` df=0 / 非嵌套的行为差异（O7） | 未独立复跑 Stata 报错；上游附了原始报错文本 |
| `pwcompare mcompare` / `contrast h.`（O8） | 同上 |
| `nlcom` 比值的 1.34e-5（O9） | 未独立复跑该条；但同类"解析仲裁"方法在 O1 已被我独立验证为可靠 |
| ARIMA / GARCH 的停机点与 presample 归因（T5/T6） | 未独立复跑；上游的"在 Stata 参数点求 statsmodels ll = Stata ll"与"交叉验证各自在各自目标下更优"是可复算的强证据 |
| `vecrank` 选阶翻转（T4） | 我复现了"临界值表不同"（Stata 3.76 vs statsmodels 3.8415），但**未能造出落进 (3.76, 3.8415) 区间的数据**，故"翻转"本身未独立验证 |
| `systems_causal` 的 `aipw atet` 精确公式 | 我复现了"ATE 精确、ATET 错"的方向与量级，但未重做上游的 PO 公式反推 |

---

## 7. 各领域结论可信度评级（给 lead 的表述建议）

| 领域 | 可信度 | 表述建议 |
|---|---|---|
| **linear_iv** | **高** | 可直接引用。`e(F)`/`e(df_r)` 覆盖缺口、IV `HC1` 标签错、GMM cluster 点估计均可信。唯一注意：`fweight` 别用 `nobs` 比样本（§5.1） |
| **panel** | **高** | `vce(robust)`≡`vce(cluster)`、9.36% 因子、`xtpcse` 非平衡、`xtabond` 参数名错位全部由我独立复现到 1e-15。可直接引用 |
| **systems_causal** | **高** | `nl` pinv、`nnmatch` ATET SE≡ATE SE、`didregress` 自动加协变量、jackknife float32 全部独立复现。可直接引用 |
| **postest** | **高** | `margins` h=1e-6、tobit 三个 bug 全部独立复现（含解析仲裁）。可直接引用 |
| **discrete** | **中高** | 数值全部可复现，**但 ologit 的根因必须改写**（§2）。另外缺少"ologit 交叉块未变换"这个独立缺陷。其余（`e(df_m)`、`e(chi2)`、zip `e(p)`、HC1==HC0、predict 崩溃）可直接引用 |
| **tseries** | **中高** | A 档与 VAR 自由度归因可靠（我复现到 1e-16）。**"vecrank 选阶翻转"这条降级为"临界值表不同"**，不要写成"已实测翻转" |
| **limited_quantile** | **中** | tobit / heckman / etregress / rreg 四条的 bug 与归因全部成立（我复现到 1e-10~1e-16）。**但 qreg 的归因必须改写**（§4.2）：不能说"换 meat 就修好"，也不能用"偏高 3.5–4.6%"这个偏小的幅度 |

---

## 8. 复现方式

```bash
cd D:\myCodeProject\statapy
.venv\Scripts\python.exe retest/domains/verify/p1M_ologit_confirm.py   # ologit 根因（决定性）
.venv\Scripts\python.exe retest/domains/verify/p2_tobit.py             # tobit /sigma
.venv\Scripts\python.exe retest/domains/verify/p4b_qreg.py             # qreg 基准
.venv\Scripts\python.exe retest/domains/verify/p18_qreg_meat_correct.py # qreg meat 断言
.venv\Scripts\python.exe retest/domains/verify/p5b_heckman_align.py    # heckman
.venv\Scripts\python.exe retest/domains/verify/p5c_etregress_sign.py   # etregress 符号
.venv\Scripts\python.exe retest/domains/verify/p20_more_accurate.py    # 精确有理数仲裁
.venv\Scripts\python.exe retest/domains/verify/p8_atier.py             # A 档 12 条抽查
.venv\Scripts\python.exe retest/domains/verify/p9_panel.py             # panel vce
.venv\Scripts\python.exe retest/domains/verify/p6d_var_dof.py          # VAR 自由度
.venv\Scripts\python.exe retest/domains/verify/p7_highvalue.py         # nl / nnmatch / xtpcse / margins
```

探针清单（各自独立，互不依赖）：
`my_stata.py`（自建桥，不 import 共享 lib）、
`p1*`–`p23*`（discrete / tobit / linear / limited / tseries / panel / systems / postest）。

**所有探针都不修改 `src/`、不改 `retest/lib/`、不写领域目录**，只在 `retest/domains/verify/` 下。
