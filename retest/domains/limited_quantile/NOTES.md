# limited_quantile 领域复测结论

工具：`retest/lib/stata_probe.py` + 本目录 `run.py`（可重跑：
`cd D:\myCodeProject\statapy && .venv\Scripts\python.exe retest/domains/limited_quantile/run.py`）。
Stata 侧：内嵌 StataNow 19.5 MP（`config.init("mp")`，`c(stata_version)=19.5`）。
全部数字取自 `e(b)/e(V)/e()` 全精度，未使用任何打印值，未读取仓库内既有 fixture / registry / GAP_REGISTER。

判定口径（BRIEF §2.3）：A ≤1e-9，B ≤1e-6，C ≤1e-3，D ≤1e-2，E >1e-2。
系数差与标准误差分别取「逐系数相对差的最大值」；SE 用 `e(V)` 对角开方。

结果：**25 个对比点，pass 0 / fail 22 / error 3**（`summary.json` 计数）。
这里的 fail 是**数值门槛判定**，不是"命令不可用"——大多数条目系数本身在 B 档以内，
失败几乎都来自标准误或派生统计量。下文逐条区分"真 bug"与"口径问题"。

## 覆盖矩阵

| # | Stata 命令 | westata 调用 | 数据 | 判定 | 系数相对差 | 标准误相对差 |
|---|---|---|---|---|---|---|
| 1 | `tobit y x1 x2, ll(0)` | `ws.Tobit(lower=0)` | 自造 500 obs，134 左删失 | fail | 1.38e-08 (B) | **5.43e-02 (E)** |
| 1b | 同上（诊断项） | 同上 | 同上 | fail | 5.43e-02 (E) | **5.43e-02 (E)** |
| 2 | `tobit yu x1 x2, ul(2.5)` | `ws.Tobit(upper=2.5)` | 70 右删失 | fail | 1.06e-08 (B) | **1.62e-02 (E)** |
| 3 | `tobit yb x1 x2, ll(0) ul(2)` | `ws.Tobit(0, 2)` | 134 左 + 116 右 | fail | 3.37e-08 (B) | **6.65e-02 (E)** |
| 4 | `truncreg y x1 x2, ll(0)` | `ws.Truncreg(lower=0)` | 366 保留（134 截断） | fail | 1.05e-07 (B) | 6.74e-06 (C) |
| 5 | `truncreg yu x1 x2, ul(2.5)` | `ws.Truncreg(upper=2.5)` | 430 保留 | fail | 1.91e-08 (B) | 1.63e-06 (C) |
| 6 | `truncreg yb x1 x2, ll(0) ul(2)` | `ws.Truncreg(0, 2)` | 250 保留 | fail | 1.63e-06 (C) | 8.28e-05 (C) |
| 7 | `intreg lo hi x1 x2` | `ws.Intreg.from_formula("lo hi ~ …")` | 500 obs 区间删失 | fail | 2.88e-09 (B) | 2.01e-06 (C) |
| 8 | `heckman y_sel x1, select(s = z1 z2) twostep` | `ws.Heckman(...).fit()` | 600 obs，377 入选 | fail | 1.34e-09 (B) | **6.81e-02 (E)** |
| 9 | `heckman …, `（默认 = ML） | `fit(method="ml")` | 同上 | **error** | — | — |
| 10 | `etregress y x1, treat(s = z1 z2) twostep` | `ws.Etregress(...).fit()` | 600 obs | fail | **3.84e-11 (A)** | **6.77e-02 (E)** |
| 11 | `etregress …, `（默认 = ML） | `fit(method="ml")` | 同上 | **error** | — | — |
| 12 | `frontier y x1 x2` | `ws.Frontier(distribution="halfnormal")` | 500 obs，真无效项 | fail | 1.08e-03 (D) | 4.03e-03 (D) |
| 13 | `frontier …, distribution(exponential)` | `distribution="exponential"` | 同上 | **error** | — | — |
| 14 | `qreg y x1 x2` | `QuantileRegression(quantile=.5)` | 400 obs 异方差 | fail | 4.56e-09 (B) | 5.15e-09 (B) |
| 15 | `qreg y x1 x2, quantile(.25)` | `QuantileRegression(quantile=.25)` | 同上 | fail | 5.68e-07 (B) | 4.22e-09 (B) |
| 16 | `qreg y x1 x2, vce(robust)` | `.fit(cov_type="robust")` | 同上，τ=.5 | fail | 4.56e-09 (B) | 5.84e-09 (B) |
| 17 | `qreg y x1 x2, quantile(.25) vce(robust)` | 同上，τ=.25 | 同上 | fail | 5.68e-07 (B) | **4.63e-02 (E)** |
| 18 | `sqreg y x1 x2, quantile(.25 .5) reps(500)` | `ws.Sqreg(reps=500)` | 同上，4 seed 池化 | fail | 5.68e-07 (B) | **3.35e-02 (E)** |
| 19 | `bsqreg y x1 x2, quantile(.5) reps(500)` | `ws.Bsqreg(reps=500)` | 同上，4 seed 池化 | fail | 4.56e-09 (B) | **2.19e-02 (E)** |
| 20 | `rreg y x1 x2, gen(wt)` | `ws.RobustRegression` | 400 obs 异方差 | fail | 2.70e-04 (C) | **2.06e-01 (E)** |
| 21 | 同上（权重对比） | `_biweight_weights` | 同上 | fail | 1.35e-03 (D) | — |
| 22 | `rreg y x1 x2` | 同上 | 300 obs 干净正态 | fail | 2.15e-03 (D) | **2.97e-01 (E)** |
| 23 | `rreg y x1 x2` | 同上 | 300 obs + 6 个离群 | fail | 1.40e-03 (D) | **1.54e-01 (E)** |
| 24 | `rreg mpg weight length` | 同上 | auto.dta, 74 obs | fail | 1.83e-03 (D) | **2.29e-01 (E)** |

**A 档（≤1e-9）条目：0 条**。最接近的是 `etregress twostep` 的**系数**（3.84e-11，属 A），
但其 SE 在 E 档，整条判 fail。没有任何一个对比点在"系数 + SE + 标量"三项同时 ≤1e-9。

---

## 逐项说明

### 1. tobit（ll / ul / 双边）：**westata 的 `/sigma` 标准误是真 bug**

三条 tobit 的系数全部在 B 档（1e-8 量级，纯优化器收敛差异），`ll`、`ll_0`、`chi2`、
`df_m`、`r2_p`、`N_unc/N_lc/N_rc` 与 Stata 完全一致（相对差 0 或 1e-12）。
唯一失败项是 `/sigma` 的 SE，而且**不是**先生说的 delta 法/对齐问题：

```
stata: var(e.y) = 0.8589706557, se(var) = 0.0858683272
       → se(sigma) = se(var)/(2·sigma) = 0.0406037441
       → se(lnsigma) = se(var)/(2·var) = 0.0383998173
westata: se(/sigma) = 0.0383997360
```

三条独立证据（`extra` 见 `results.jsonl` 第 1b 条）：

1. **对独立的中心差分 Hessian**：westata 报的 SE 与 `(b, lnσ)` 参数化下的 Hessian 给出的
   SE 相对差 **0.00e+00**；与 `(b, σ)` 参数化下的 SE 相对差 **5.43e-02**。
2. **对 Stata 自己的 se(lnsigma)**：相对差 2.1e-6（残余差异只是 Hessian 数值步长）。
3. **比值恒等式**：`westata_se / stata_se(sigma) = 0.945719091`，`1/σ̂ = 0.945721083`，
   相对差 2.1e-6。也就是 westata 报的正是 `se(lnσ)`，而 σ 点估计已乘回 σ 尺度。

结论：`Tobit.fit` 里 `params` 的最后一个元素换成了 σ（`np.exp(theta[k])`），
但 `cov` 仍是 `theta`（=lnσ）参数化下 `_observed_information` 的结果，没有做 delta 变换。
后果：σ 的 z 统计量被**放大 1/σ 倍**（本 DGP 下 7.9%），即 σ 的显著性被高估；
而 σ 点估计本身正确（相对差 1e-8）。三个变体（ll、ul、ll+ul）的偏差分别是 5.4%、1.6%、6.6%，
幅度随 σ̂ 变化，与 `1/σ̂` 完全吻合——所以不是"某一条命令算错"，是同一个公式错误。

对照 Lead 提示的 7.9e-2：那是 `compare()` 里 `stata_transform` 只对 σ 用了 sqrt+delta，
而 westata 侧 `bse` 仍是 lnσ 尺度造成的**双向错配**。正确做法有两种，本次都验证过：
(a) 只对 Stata 的 `var(e.y)` 做 sqrt + delta（得到 se(σ)），与 westata 的 raw bse 比 → 差 5.4%，
这正是真 bug 的量级；(b) 把 westata 的 `bse` 也乘 σ（得 se(σ)）再与 (a) 比 → 差 1.9e-6（A/B 边界）。
`run.py` 采用 (a)，故 fail；同时 1b 条把 (b) 的证据写进 `extra`。

### 2. truncreg（ll / ul / 双边）：C 档，可接受

系数 1e-8～1.6e-6，SE 1.6e-6～8.3e-5，全部 C 档（1e-3 以内）。属于不同优化器收敛路径
（westata 走 statsmodels Newton + 放宽 tol；Stata 用自己的 Newton）导致的数值实现差异，
**不影响任何系数第 4 位有效数字**。双边截断（250 观测、`chi2=4.09` 极不显著）的系数差
稍大（1.6e-6）且 SE 有 8e-5，是因为该点 Hessian 条件数差、两个优化器停在略不同的点；
`ll` 差 2e-7 也印证是收敛精度而非公式错误。
样本一致性已核对：Stata `e(N)=250`，`st.nobs()` 仍是 500（截断观测留在数据集里），
westata `nobs=250`、`e(n_truncated)=250`——**不是样本不同**。

### 3. intreg：B/C 档，可接受

系数 2.9e-9（B），SE 2.0e-6（C）。区间构造：`[floor(y*), floor(y*)+1)`，y*<0 的观测
左删失（`lo` 缺失、`hi=0`）。Stata 侧 `e(b)` 列名是 `['x1','x2','_cons','_cons']`——
`/lnsigma` 行复用了 `_cons` 标签，这是 Stata 的命名怪癖；`compare` 靠 dict rename
无法区分两个同名键，所以 `run.py` 用打印表格的行序（x1, x2, _cons, /lnsigma）重写列名，
**b/V 数值未做任何改动**，并在 `notes` 里注明。同理适用于 `frontier`（3 个 `_cons`）和
两方程模型。

### 4. heckman twostep：系数 A/B，SE 在 E 档，**e(sigma)/e(rho) 定义错**

（a）**回归系数**（y|x1、y|_cons、s|z1、s|z2、s|_cons、λ）相对差 1.3e-9——λ 系数本身
两边的确一致（`0.5301503851` vs `0.5301503857`）。

（b）**`e(sigma)` / `e(rho)` 的定义是错的**。Stata 的两步法 `e(sigma)` 在本 DGP 上等于

```
sqrt( RSS/m + b_λ² · mean( λ(λ + w) ) )      w = probit 线性指数, m = 入选样本量
```

复现到相对差 **1.8e-10**（`extra.sigma_reconstruction`）。westata 用的是
`sqrt(RSS/(m-3)) = 0.904786`，而 Stata 是 `0.970088`——差 6.7%。连带 `e(rho)` 也错
（westata 0.58594 vs Stata 0.54650，差 7.2%），因为 westata 用 `rho = b_λ/σ`。
`e(lambda)` 反而对（两边都是 `b_λ`，因为 Stata 的 λ 系数就是 `rho·sigma` 的乘积）。
影响：`e(rho)`、`e(sigma)` 常被下游拿去做 LR 检验和报告，6–7% 的偏差不可忽略。
注意这不是"westata 换了个估计量"——它报的就是 twostep，λ 系数一致，只是派生量算错。

（c）**标准误**：
- 选择方程块：Stata 与普通 `probit s z1 z2` 的 VCE 一致（比值 0.99998），
  即 Stata 在 twostep 里**保留 probit 的方差**；westata 用联合 sandwich 的 bread，
  把结果方程块混进了 probit 块，使 `select:z1` 的 SE 高 6.8%（0.076046 vs 0.071195）、
  `select:z2` 高 1.3%、`select:_cons` 高 1.0%。我用独立的 M-estimator sandwich
  手工重算 westata 的联合 VCE，6 个 SE 全部复现到 1e-8，确认这就是它的算法，
  而不是数值噪声。
- 结果方程块：westata 的三个 SE 比 Stata 高 1.7%–3.1%。我试过 Greene 的经典修正
  `s²(X'X)^-1 + (X'X)^-1 A'D V_g D'A (X'X)^-1` 的多种 `s²` 变体（RSS/m、RSS/(m-3)、
  `e(sigma)²`、含 `ρ²·mean(δ)` 的版本，以及全 M-estimator sandwich），**没有一个能复现
  Stata 的数值**（最接近的差 2.0%）。Stata 的 twostep VCE 在其文档里没有给出闭式，
  我只能记录"无法从公开定义复现，差异 2–3%"，不猜测。

### 5. heckman ML / etregress ML：**不支持（error）**

两者都抛 `UnsupportedError`，原文见 `results.jsonl` 的 `error` 字段：

- `heckman`：`heckman: only method='twostep' is implemented; 'ml' is not.`
- `etregress`：`etregress: only method='control' is implemented; 'ml' is not.`

Stata 侧两者都能跑（`heckman` 默认就是 ML，`etregress` 默认也是 ML），参考值已全文记录在
对应 `notes` 里，供后续实现比对。**注意 `etregress` 的默认估计量是 ML**，
westata 唯一可用的 control-function 路径**不是** Stata 的默认——这一点在文档里应写明，
否则用户按 `etregress y x, treat(...)` 的直觉调用会拿到不同的估计量。

### 6. etregress twostep：系数 A 档（3.8e-11），SE E 档，**e(rho) 符号错**

- 系数（含 `y:s` 处理效应）相对差 3.8e-11，是全场最好的结果。
- **`e(rho)` 符号错误**：本 DGP Stata 报 `+0.42835`，westata 报 `−0.44860`。
  在第二个（真 ρ = −0.55）的 DGP 上，Stata 报 `−0.57154`，westata 报 `+0.62258`——
  两次都是**互为相反数**。原因：westata 用 `rho = -b_ctrl/sigma`，而 Stata 的
  `e(lambda) = b_ctrl = rho·sigma`（已验证 `0.41420433 = 0.42835 × 0.96699`），
  所以正确关系是 `rho = +b_ctrl/sigma`。westata 多了一个负号。
- `e(sigma)` 也有 3–4% 偏差：Stata 的 etregress `e(sigma)` 恰好等于
  `sqrt(RSS/n + b_ctrl²·mean(c² + w·c))`（复现到 1.8e-10，c 是广义残差），
  westata 用 `sqrt(RSS/(n-4))`。
- `e(sigma1)/e(sigma2)`：**Stata 19.5 的 etregress 根本不存这两个标量**
  （完整 `get_ereturn()` dump 里只有 `e(rho) e(sigma) e(lambda) e(selambda) e(ll_p)`）。
  必测清单里的这一项在 Stata 侧不可测，按"未覆盖"处理，不编数字。
- SE：全部 7 个都在 6.6e-3–6.8e-2，与 heckman 同源问题（联合 sandwich 的 bread 与
  Stata 的 corrected VCE 不同）。

### 7. frontier halfnormal：D 档

系数：x1 2.2e-7、x2 2.9e-7（B），`_cons` 1.0e-4、`/lnsig2v` 1.1e-3、`/lnsig2u` 2.9e-4。
`ll` 两侧差异 5.9e-8（−745.941018904 vs −745.941018829，westata 略高 7.4e-11）。
`chi2` 相对差 2.4e-6。SE 全部 ≤4.1e-3（D 档）。

归因：**数值 Hessian 的步长**。westata 用 `_numerical_hessian(..., step=1e-5)`，
而该似然在 `sigma_u→0` 附近极平坦。我固定 westata 的最优点不变，只用不同步长重算 Hessian：
`h=1e-3` 与 `h=1e-4` 给出 `se(/lnsig2u) ≈ 2.7106/2.7107`，`h=1e-5` 给 2.7323，`h=1e-6`
给 6.9587（已发散）。Stata 报 2.7044——即 **h=1e-3 时 westata 的 SE 与 Stata 差 0.2%**，
h=1e-5 时差 1.0%。也就是说 westata 默认步长偏小，SE 里混入了步长误差，
而系数本身只差 1e-7。这一项是"可修的实现细节"，不是模型公式错误。
另外我没有用"无无效项"的 DGP：那种数据下两个估计器都会落到 `sigma_u = 0` 边界，
`/lnsig2u` 的 SE 失去意义（westata 报 209.7、Stata 报 61.2，两边都不可信），
所以本项刻意选了有效无效项（`u ~ |N(0,0.7²)|`）的数据。

### 8. frontier exponential：**不支持（error）**

`frontier: distribution='exponential' is not implemented; only 'halfnormal' is.`
Stata 同数据可跑（`ll = -745.9702703`）。必测清单只要求 halfnormal / exponential
"至少一项"，halfnormal 已完整测，exponential 记为 error。

### 9. qreg（中位数 / .25）：B 档，可接受

系数 4.6e-9 / 5.7e-7，SE 5.1e-9 / 4.2e-9——都远在 B 档内。
`sum_adev`、`sum_rdev`、`r2_p`、`q`、`q_v`、`sparsity`、`bwidth` 全部在 1e-9 内一致。

`.25` 的系数差 5.7e-7 出在 `_cons` 上（−0.00388141，两边同为 −0.00388141），
其绝对值接近 0（0.0039），`rel_diff` 用 `max(|stata|, 1e-12)` 作分母时对小数值放大——
这是 BRIEF §2.3 的定义本身，不改。该系数在 25% 分位处本来就接近 0（真实截距 1.0，
但 25% 分位下 LAD 的截距就是分位点，DGP 下约为 0），不是实现错误。
`vce(robust)` 在 τ=.5 时两边的 Ψ² 都是常数 0.25，**无法区分** sandwich 的两种 meat，
所以 τ=.5 的 robust 一致（5.8e-9）并不证明公式正确；真正有区分力的是下一条。

### 10. qreg vce(robust) 在 τ=.25：**真 bug（E 档）**

τ=.25 时 westata 的稳健 SE 高 3.5%–4.6%（x1 0.101566 vs 0.097073）。

拆解：设 `J = Σ f_i x_i x_i'`（`f_i = 2h/(x_i'(b(τ+h) − b(τ−h)))`，观测特定密度），
westata 用 `M = Σ ψ_i² x_i x_i'`，`ψ_i = τ − 1{u_i<0}`；`V = J⁻¹ M J⁻¹`。
我手工重算 confirm westata 就是这个公式（复现到 1e-12）。而 **Stata 的 meat 是
`τ(1−τ)·X'X`**：把 `M` 换成 `τ(1−τ)X'X` 后，τ=.25 的 SE 与 Stata 复现到 **9.0e-9**，
τ=.75 复现到 **1.4e-8**，τ=.5 自然仍一致（两者在该点等价）。
所以 westata 的 sandwich 用了"逐点 Ψ²"，Stata 用"常数 τ(1−τ)"——在 τ≠.5 时差 3–5%。
后果：非中位数分位的 qreg robust 推断偏保守（SE 偏大 3–5%），t 值偏小。
这条也被 `QuantileRegression` 的 docstring 自己点名（"only that special case is verified
against Stata"），本次实测确认：**确实只在 τ=.5 成立**，一般 τ 是错的。

### 11. sqreg / bsqreg：系数确定，SE 差 2–3%，**主要是 bootstrap 方案差异 + Stata 自身的 MC 波动**

先把方法论说清：Stata 的 RNG 流无法在 numpy 里复现，单次 `reps(500)` 的 SE 本身有
Monte-Carlo 误差。为把 MC 噪声压下去，两边都**对 4 个 seed × 500 reps 的方差取平均**：

| | x1 | x2 | _cons |
|---|---|---|---|
| Stata bsqreg 池化 SE | 0.102453 | 0.088847 | 0.084768 |
| westata bsqreg 池化 SE | 0.102590 | 0.087713 | 0.082913 |
| 相对差 | 0.13% | 1.3% | **2.2%** |

单次 500 reps 时 Stata 自己在两个 seed 间的 `_cons` SE 就差 1.0%（0.08397 vs 0.08407），
1000 reps 下仍有 0.4%。把方差池化 4×500 后，westata 与 Stata 的差（2.2%）大于剩余
MC 噪声（约 0.7%），说明**两者的重抽样方案确有差别，但不大**：
- 系数完全相同（sqreg 的点估计就是 qreg，1e-9 内一致）；
- 跨分位相关结构：Stata `corr(_cons@.25, _cons@.5) = 0.532`，westata `0.538`——两边都是
  "同一组重抽样指标同时拟合各分位"的同时（simultaneous）方案，方向一致；
- 但**bootstrap SE 与解析 robust SE 的比值**两边不一致：
  Stata `bsqreg/robust = 1.199 / 1.224 / 1.051`，
  westata `bsqreg/robust = 1.034 / 1.248 / 1.304`（顺序 x1/x2/_cons）。
  Stata 的 bootstrap SE 在 x1 上比解析 robust 高 20%，westata 只高 3%；
  在 `_cons` 上反过来。这不是"某个数差 2%"，而是**分布形状不同**，说明 resampling
  细节（是否按行、是否用 m-out-of-n、是否用同一 draw 拟合所有分位、偏差修正）不同。
  我在同方差 DGP 上重复了同一实验，比值同样不一致（Stata 1.32/0.95/1.07，
  westata 1.10/1.28/0.96），排除了"DGP 特例"。
- 结论：**westata 的 sqreg/bsqreg 不可判为与 Stata 数值等价**；2–3% 的差距不是舍入，
  是重抽样方案差异，需要拿到 Stata 的 seed/重抽样实现才能进一步归因。
  `sqreg` 的 `seed()` 参数 Stata 不接受（`option seed() not allowed`，r(198)），
  只能 `set seed`，这一点也记下来避免后续误用。

### 12. rreg：系数 C 档，SE **E 档，且已定位 Stata 的公式**

（a）**系数**：2.7e-4 – 2.2e-3（C/D）。三个 DGP（干净正态、含离群、auto.dta）都是这个量级。
不是收敛问题：`maxiter` 从 50 加到 2000，westata 的估计完全不变（打印值是同一串数字）。

（b）**权重**：`rreg ..., gen(wt)` 与 westata 的 `_biweight_weights` 对比，
`sum(w)` 368.6395 vs 368.5906（差 0.013%），`max|Δw| = 1.35e-3`，
相关系数 0.99999897。两次方都用了 c=4.685、MAD/0.6745 的尺度——**权重定义一致**，
残差 1e-4 量级的差异被权重放大成了 1e-3，所以这一条判 D 而非 A。

（c）**标准误：Stata 根本不用 pseudovalue**。这是本次最有价值的发现之一。
Stata 的 rreg SE 精确等于

```
se(b_j) = e(rmse) · sqrt( [(X'X)⁻¹]_jj )       （X 是未加权的原始设计矩阵，含常数）
e(rmse) = sqrt(e(rss)/(n−k))
```

在四个数据集上复现到 **1.9e-16 / 1.1e-16 / 1.2e-16 / 2.3e-14**（`extra.stata_se_law`），
即这是 Stata 的定义，不是巧合。westata 用的是 Street-Carroll-Ruppert 的
leave-one-out pseudovalue，**这是两个不同的估计量**：我在 westata 之外独立实现了一遍
STR（完整 Huber→biweight 两阶段，逐个留一重拟合），得到的 SE 与 westata 报的
完全一致（0.06712246 / 0.08658141 / 0.06741934），因此 westata 的伪值实现本身没错，
错的是**它选的方差定义与 Stata 不同**。离差 15%–30% 完全来自这里。
附带效应：`e(rss)`、`e(mss)`、`e(rmse)`、`e(r2)`、`e(F)` 也全不一致（westata 报加权平方和，
Stata 报的是能让 `rmse²·(n−k) = rss` 成立的那一套），F 差 24%–38%。这一点旧 registry
把差异归因为"`[R] rreg` 没把留一构造写清楚"，**本次实测证明不是文档模糊，是 Stata
用的是闭式 OLS 型方差**——旧结论需要修正。

同时对 `e(rss)` 本身：我试了 `Σw^c e²` 形式的各种权重归一化（c=0,1,2、除以 mean(w)、
乘 n/sum(w)、`Σe²/w` 等 12 种），没有一个能同时复现四个数据集；`auto.dta` 上甚至能解出
c=0.1985 但其他数据集无解。这一项**未能归因**，如实记录为未解决。

### 13. 其他被验证为"正确"的口径（不算独立对比点，但写下来避免后人重复踩）

- tobit 的 `e(N_unc)/e(N_lc)/e(N_rc)`：Stata 用 `N_unc/N_lc/N_rc`，westata 用
  `n_uncensored/n_left/n_right`，数值全等（500 观测下 366/134/0，三变体均核对）。
- `ll`/`ll_0`/`chi2`/`r2_p`/`df_m`：tobit、truncreg、intreg、frontier、qreg 全部 ≤1e-9 一致，
  其中 tobit 的 `ll` 是 **0.0**（完全相同的浮点数）。
- `qreg` 的 `sparsity`、`bwidth`（Hall-Sheather）：1e-9 内一致，说明拟合密度算法正确。
- 因子变量：本域全部命令都用连续变量，未涉及基准水平对齐问题。

---

## 与旧结论的出入

旧证据按 BRIEF §0 只读、不作依据，但需要指出冲突点（这是最有价值的部分）：

1. **`tobit` 旧 registry 记 E3 / 1e-6，notes 只说"Slopes 8.5e-8，标准误差 6.4e-7，
   sigma 比较时平方处理"**。
   → **冲突**：那 6.4e-7 是所有系数 SE 的最大值，但**恰好漏掉了 `/sigma` 的 SE**
   （它比其余大 5 个数量级）。本次实测 `/sigma` SE 差 5.4%（ll）、1.6%（ul）、6.6%（双边），
   是**真 bug**（cov 未随参数化做 delta 变换）。旧结论把这条命令判为通过，属漏检。

2. **`heckman` 旧 registry 记 E3 / 5e-2，notes 主动说"rho 是 plug-in 值，排除比较"**。
   → **部分冲突**：排除 `rho` 是对的，但 `e(sigma)` 同样错（westata `sqrt(RSS/(m-3))`
   vs Stata `sqrt(RSS/m + b_λ²mean(δ))`，差 6.7%），旧 notes 只提了 rho。
   另外旧 notes 说"corrected SEs 落在 Stata **ML** SE 的 3% 内"——拿 two-step 的 SE 去比
   ML 的 SE，本身就是跨估计量比较；本次与 **twostep** 比，结果方程块差 1.7–3.1%，
   选择方程块差到 6.8%（westata 的联合 bread 把两个方程混了）。

3. **`etregress` 旧 registry 记 E3 / 1e-6，ref 写"westata matches all six to 1e-9"**。
   → **冲突**：本次对**系数**确认到 3.8e-11（比旧记录还准），但旧记录只比了系数，
   没比 SE（实测 6.8e-2）也没比 `e(rho)` 的**符号**（westata 与 Stata 反号，两个 DGP 都反）。
   旧 notes 里"5.3e-2 的差是因为拿 ML 比了"的教训是对的，但"1e-9 全部匹配"这句话
   只对系数成立。

4. **`qreg, vce(robust)` 旧 registry 记 E3 / 1e-5，notes 自己写了 CAVEAT：
   "the fixture only exercises tau=.5 … only the .5 case is verified"**。
   → **确认并升级**：本次在 τ=.25/.75 上实测，westata 的 meat 用 `ψ_i²`、
   Stata 用 `τ(1−τ)X'X`，差 3.5–4.6%。旧的 CAVEAT 是对的，而且**确实有 bug**——
   应把这一条从 E3 降级，或在实现里改成 `τ(1−τ)X'X`。

5. **`rreg` 旧 registry 记 E3 / 2e-2，notes 说"SE 差 −2.8%/+3.4%/+0.1%，
   原因是 `[R] rreg` 没写清留一构造，documented rather than tuned"**。
   → **冲突（最重要的一条）**：Stata 的 rreg SE 不是 pseudovalue，而是
   `e(rmse)·sqrt(diag((X'X)⁻¹))`，四个数据集上复现到机器精度。所以
   (i) 不是"文档没写清"，是 Stata 有确定公式；
   (ii) westata 的 STR 实现本身正确（我用独立实现复现了它的数），但**方差定义选错了**；
   (iii) 本次实测的 SE 差是 15%–30%，比旧 notes 的 3% 大一个量级——旧 notes 报的
   那组数（−2.8%/+3.4%/+0.1%）在本次的四个 DGP 上没有一个能复现，
   怀疑旧 fixture 的样本恰好落在两个估计量接近的区域。

6. **`frontier` 旧 registry 记 E3 / 1e-4，notes 说"lambda 的 SE 仍不报"**。
   → **部分冲突**：本次系数差 1e-3（`/lnsig2v`）/ 1e-4（`_cons`），比 1e-4 的门槛大；
   SE 差 4e-3。归因是 westata 的数值 Hessian 步长（1e-5）在平坦方向上不够小，
   改成 1e-3 后差降到 2e-3 以内。旧的 1e-4 等级偏乐观。

7. **`sqreg`/`bsqreg` 旧 registry 记 E3 / 2e-2，notes 说"2e-2 是 Stata 自身 seed 间波动"**。
   → **部分确认**：seed 间波动确实存在（单次 500 reps 约 1%），但**池化 4×500 后**
   （把 MC 噪声压到 0.7% 以下）westata 与 Stata 仍差 2.2%（`_cons`），且
   `bootstrap/robust` 比值在两边系统性不同（Stata 1.20/1.22/1.05 vs westata 1.03/1.25/1.30）。
   所以 2e-2 的容差不是"MC 波动"，而是掩盖了一个真实的重抽样方案差异。

8. **`truncreg`/`intreg` 旧 registry 记 E3 / 1e-3**。
   → **一致**：本次系数 1e-8–1.6e-6、SE 1.6e-6–8.3e-5，全部在 1e-3 内。这两条旧结论站得住。

---

## 未覆盖

1. **`etregress` 的 `e(sigma1)` / `e(sigma2)`**：必测清单要求核这两个标量，
   但 Stata 19.5 的 etregress **不存它们**（`get_ereturn()` 全量 dump 里没有；
   只有 `e(rho) e(sigma) e(lambda) e(selambda) e(ll_p)`）。Stata 只在 ML 下存
   `e(athrho)/e(lnsigma)`；two-step 下连 `e(athrho)` 都没有。因此这一项**在 Stata 侧
   无法取到参考值**，记为不可测，不伪造。
2. **`etregress` 的 `athrho` VCE**：同上——two-step 下 Stata 没有 `e(athrho)` 及其 VCE；
   westata 的 `Etregress` 也只报 `lambda` 的 SE（`e(selambda)` 对应 Stata 的
   `e(selambda)=0.11402906`，westata `bse_lambda` 可对上但差 3.4%）。ML 下的 `athrho`
   因 westata 无 ML 路径而无法比。
3. **`heckman` / `etregress` 的 ML 数值**：westata 无实现（error），只能记录 Stata 参考值。
4. **`frontier` 的 truncated 分布**：必测只要求 halfnormal/exponential 之一；
   halfnormal 已测，exponential 记为 error。truncated 未测。
5. **`rreg` 的 `e(rss)` 精确公式**：试了 12 种加权平方和形式，四个数据集无法统一复现，
   只确认了 `e(rmse)²·(n−k) = e(rss)` 这一自洽关系。归因失败，如实记录。
6. **`sqreg`/`bsqreg` 的重抽样细节**：2–3% 的差异已定位到"重抽样方案不同"，
   但无法进一步确定是 m-out-of-n、偏差修正还是别的（Stata 不接受 `seed()` 选项，
   两边 RNG 流不可共享）。
7. **tobit 的权重/稳健方差**：westata `Tobit.fit` 明确只支持 `cov_type='nonrobust'`，
   `vce(robust)` 等其他变体未测（westata 侧会抛 `ValueError`）。
8. **truncreg/intreg 的 `vce(robust)`**：westata `_MLLimitedBase.fit` 同样只支持
   `nonrobust`，未测。
9. **`qreg` 的 `quantile(.75)/.9`、`rreg` 的 `tune()` 变体、`sqreg` 的三分位以上**：
   为控制 Stata 会话时长（单条 bootstrap 已到 45 s）未展开，但 τ=.75 的 robust
   仅在单独的诊断脚本里测过（结果已写进第 10 条）。

---

## 附：本目录文件

- `run.py` —— 主脚本，25 个对比点，全部可重跑（约 3–4 分钟，含 4×500 次 bootstrap）。
- `results.jsonl` / `summary.json` —— 由 `Record` 落盘，`extra.detail` 含逐系数 b/se 对。
- `probe*.py` —— 归因用的一次性探针脚本（probe2–probe39），保留以便复核每一条归因结论；
  它们不参与 `summary.json`，也不修改任何 `src/` 内容。
- `NOTES.md` —— 本文件。
