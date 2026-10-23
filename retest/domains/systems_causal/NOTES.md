# systems_causal 复测结论（多方程 / 处理效应 / DID / svy / 重抽样）

- 复测时间：2026-10-24
- Stata：StataNow 19.5 MP（`pystata` 内嵌，`config.init("mp" , splash=False)`，实测 `c(stata_version)` = 19.5）
- westata：仓库内 editable 安装的 `src/westata`（`import westata`）
- 脚本：`retest/domains/systems_causal/run.py`（单进程内串行、一个 Stata 会话）
- 原始记录：`results.jsonl`（56 条）、`summary.json`
- 判定档位按 BRIEF §2.3：A ≤1e-9、B ≤1e-6、C ≤1e-3、D ≤1e-2、E >1e-2；
  相对差 = `max|w-s| / max(|s|, 1e-12)`，逐元素取最大（含 `e(V)` 全矩阵复查的条目另行标注）。

**总计 56 条对比：pass 36 / fail 16 / error 2 / skip 2。**

所有数字都来自本次现场 Stata 会话 + 本次 westata 实跑；未引用 `tests/conformance/`、
`registry.yml`、`conformance*.csv`、`docs/GAP_REGISTER.md` 中的任何既有结论。

---

## 覆盖矩阵

| # | Stata 命令 | westata 调用 | 数据 | 判定 | 系数相对差 | 标准误相对差 |
|---|---|---|---|---|---|---|
| 1 | `sureg (price weight mpg) (mpg weight foreign)` | `Sureg.from_formula([...]).fit()` | auto (N=74) | **A** | 5.75e-13 | 6.09e-15 |
| 2 | `sureg ..., vce(robust)` | `fit(cov_type="robust")` | auto | **error** | — | — |
| 3 | `mvreg price mpg = weight foreign` | `Mvreg.from_formula("price mpg = weight foreign")` | auto | **A** | 7.18e-15 | 9.12e-14 |
| 4 | `manova price mpg = foreign` (p=2,q=1) | `Manova.from_formula(...)` | auto | **A** | 1.96e-14 | n/a（无 SE） |
| 5 | `manova y1 y2 y3 = g` (p=3,q=2) | 同上 | 自造 N=90 | **A** | 4.68e-15 | n/a |
| 6 | `manova z1 z2 z3 = g` (p=3,q=3, s=2.4337) | 同上 | 自造 N=100 | **A** | 7.55e-14 | n/a |
| 7 | `nl (price = {b0}*exp({b1}*mpg))`（给初值） | `Nl.from_formula(..., start=...)` | auto | **B** | 1.57e-07 | 1.58e-07 |
| 8 | `nl (mpg = {b0}+{b1}*weight+{b2}*weight^2)` | `Nl.from_formula(..., "**")` | auto | **E**（SE） | 6.33e-08 | **1.00e+00** |
| 9 | `nl (price = {b0}+{b1}*mpg)`（良态） | 同上 | auto | **A** | 2.38e-16 | 8.20e-11 |
| 10 | `nl (price = {b0}*mpg^{b1})` | 同上 | auto | **B** | 8.02e-08 | 9.23e-08 |
| 11 | `nl (...exp...)` **不给初值** | `Nl.from_formula(...)` | auto | **E** | 1.42e+01 | — |
| 12 | `teffects ra (y x1 x2) (t)` | `RegressionAdjustment(estimand="ate")` | 自造 N=400 | **A** | 3.84e-15 | 1.21e-15 |
| 13 | `teffects ra ..., atet` | `RegressionAdjustment(estimand="atet")` | 同上 | **A** | 3.28e-14 | 3.40e-16 |
| 14 | `teffects ipw (y) (t x1 x2)` | `IPW(estimand="ate")` | 同上 | **A** | 1.04e-15 | 3.59e-16 |
| 15 | `teffects ipw ..., atet` | `IPW(estimand="atet")` | 同上 | **A** | 6.20e-16 | 1.11e-15 |
| 16 | `teffects aipw (y x1 x2) (t x1 x2)` | `AIPW(estimand="ate")` | 同上 | **A** | 1.11e-16 | 1.08e-15 |
| 17 | `teffects aipw ..., atet` | `AIPW(estimand="atet")` | 同上 | **E** | **5.23e-01** | **2.93e-02** |
| 18 | 同上，PO 均值公式反推核对 | 手工重建 Stata 的 PO(0)/PO(1) | 同上 | **A** | 4.41e-16 | n/a |
| 19 | `teffects nnmatch (y x1 x2) (t)` | `NearestNeighbourMatching(nneighbor=1)` | 同上 | **A / E(SE)** | 4.50e-16 | **5.09e-02** |
| 20 | `teffects nnmatch ..., nneighbor(3)` | `nneighbor=3` | 同上 | **A / E(SE)** | 8.86e-16 | **4.53e-02** |
| 21 | `teffects nnmatch ..., atet` | `nneighbor=1, estimand="atet"` | 同上 | **A / E(SE)** | 0（位相同） | **1.63e-01** |
| 22 | `teffects nnmatch ..., biasadj(x1 x2)` | 无对应能力 | 同上 | **error** | — | — |
| 23 | `didregress (y x1) (treated), group(unit) time(period)` | `Didregress.from_formula("y x1",...)` | 自造面板 N=120 | **A** | 1.93e-15 | 8.69e-16 |
| 24 | `didregress (y) (treated), ...`（窄表） | 同上（4 列子表） | 同上 | **A** | 0（位相同） | 0（位相同） |
| 25 | `didregress (y) (treated), ...`（含无关列 x1） | 同上（全表） | 同上 | **C** | 1.85e-04 | —（规格问题） |
| 26 | `didregress ..., vce(robust)` | `fit(cov_type="HC1")` | 同上 | **A** | 1.93e-15 | 1.06e-15 |
| 27 | `didregress ..., vce(unadjusted)` | `fit(cov_type="nonrobust")` | 同上 | **skip** | — | — |
| 28 | `didregress (y) (treated)`，5 期交错采纳 | `Didregress("y",...)` | 自造面板 N=600 | **A** | 1.67e-15 | 3.39e-16 |
| 29 | `xtdidregress (y) (treated)`，5 期 | `Xtdidregress("y",...)` | 同上 | **A** | 1.67e-15 | 2.53e-16 |
| 30 | `xtdidregress (y x1) (treated)`，2 期 | `Xtdidregress("y x1",...)` | N=120 | **A** | 1.93e-15 | 8.86e-16 |
| 31 | did vs xtdid 的嵌套自由度约定 | 两个类的 SE 比值 | N=120 | **A** | 1.81e-16 | n/a |
| 32 | `svyset psu [pw=w], strata(h)` + `svy: mean y x1` | `SurveyDesign.from_frame` + `SvyMean` | 自造设计 N=576 | **A** | 3.84e-15 | 0（位相同） |
| 33 | `svy: total y x1` | `SvyTotal` | 同上 | **A** | 3.90e-15 | 0（位相同） |
| 34 | `svy: proportion grp`（3 水平） | `SvyProportion` | 同上 | **A** | 0（位相同） | 1.47e-16 |
| 35 | `svy: proportion f`（二值） | `SvyProportion` | 同上 | **A** | 0（位相同） | 0（位相同） |
| 36 | `svy: regress y x1` | `SvyRegress` | 同上 | **A** | 9.77e-16 | 0（位相同） |
| 37 | `svy: regress y x1 i.f` | `SvyRegress("y ~ x1 + C(f)")` | 同上 | **A** | 3.46e-15 | 1.86e-16 |
| 38 | `svy: logit gbin x1` | `SvyLogit` | 同上 | **A** | 1.85e-14 | 1.51e-15 |
| 39 | `svyset psu`（**self-weight**）+ `svy: mean y` | `SurveyDesign(psu="psu")` | 同上 | **A** | 1.27e-16 | 1.26e-16 |
| 40 | `svyset psu, strata(h)`（无权重）+ `svy: mean y` | `SurveyDesign(psu,strata)` | 同上 | **A** | 1.27e-16 | 0（位相同） |
| 41 | `svyset psu [pw=w], strata(h) fpc(fpc)` + `svy: mean y` | `SurveyDesign(..., fpc="fpc")` | 同上 | **A** | 1.28e-16 | 0（位相同） |
| 42 | lone PSU + `singleunit(missing)`（Stata 默认） | `singleunit="missing"` | 自造设计（2 个单 PSU 层） | **A** | 0（位相同） | 双方皆不可估 |
| 43 | lone PSU + `singleunit(centered)` | `singleunit="centered"` | 同上 | **A** | 0（位相同） | 0（位相同） |
| 44 | `svy: mean y, over(f)` | 无对应 | 同上 | **skip** | — | — |
| 45 | `jackknife: regress mpg weight` | `Jackknife.from_formula("mpg ~ weight")` | auto | **B**（SE） | 1.15e-15 | 5.11e-07 |
| 46 | `jackknife, mse: regress mpg weight` | `.fit(mse=True)` | auto | **B**（SE） | 1.15e-15 | 5.12e-07 |
| 47 | `jackknife: mean mpg` | `Jackknife(estimator="mean")` | auto | **C**（SE） | 0（位相同） | 4.58e-06 |
| 48 | `jackknife` 的 t 检验自由度 | `res.df_resid` / `conf_int()` | auto | **E** | 1.37e-02（df 73 vs 72） | n/a |
| 49 | `bootstrap` 的参考分布（z vs t） | `res.dist` / `conf_int()` | auto | **E** | 1.71e-02（临界值） | n/a |
| 50 | `bootstrap, reps(100) seed(20241024): regress` **逐次复现** | `Bootstrap(reps=100, seed=...)` 的每个 replicate | auto | **A** | 1.41e-15（100×2 个 replicate） | n/a |
| 51 | `bootstrap, reps(100) seed(20241024): regress` 的 SE | `.fit().bse` | auto | **B** | 1.15e-15 | 8.70e-09 |
| 52 | `bootstrap, reps(100) seed(20241024): mean` 的 SE | `.fit().bse` | auto | **B** | 0（位相同） | 1.12e-07 |
| 53 | `permute weight bweight = _b[weight], reps(200)` 的观测统计量 | `Permute.from_formula(...)` | auto | **A** | 2.80e-15 | n/a |
| 54 | `permute ..., reps(200) seed(20241024)` 的 p 值 | `.fit().e["pvalue_permuted"]` | auto | **D** | abs 差 4.98e-03 | n/a |
| 55 | `permute` 在弱效应下的 p 值（6 seed × 400 reps） | 同上 | 自造 N=80 | **pass**（阈值 0.05） | 均值差 1.21e-02 | n/a |
| 56 | `bootstrap, reps(200) seed(101/202/303)` 的 SE | 同上 | auto | **B** | 4.13e-08 | 4.13e-08 |

> 说明：第 2、22 条是"westata 无对应能力"的 error 条目，没有可比的数值，故相对差为 n/a；
> 第 27、44 条是 skip（Stata 自己拒绝 / westata 未实现），也没有数值。

---

## 逐项说明（凡不达 A 档者）

### 1. `nl` 二次式 SE 完全失效（第 8 条，SE 相对差 = 1.0，最严重）

**现象**：`nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)` 的系数、R²、RSS、df 全部一致
（6.3e-08），但标准误完全对不上：

| 参数 | Stata SE | westata SE |
|---|---|---|
| b0 | 5.76788434 | 3.95431143e-07 |
| b1 | 3.88353577e-03 | 5.94300115e-04 |
| b2 | 6.25759448e-07 | 1.71020052e-07 |

相对差 1.0 意味着 westata 的 SE 只有真值的 7 个数量级分之一，不是"轻微不准"，是**结论不同**。

**归因（已验证，非猜测）**：`src/westata/estimators/nonlinear.py` 用
`sigma2 * np.linalg.pinv(jacobian.T @ jacobian)` 求 Gauss-Newton 协方差。
本模型的 Jacobian 由 `weight` 与 `weight^2` 构成，`J'J` 的条件数是 **6.13e9**
（`X'X` 对角：74 / 7.19e8 / 2.74e10），其最小奇异值 / 最大奇异值 =
**3.91e-17**，低于 numpy `pinv` 的默认截断 `max(shape)*eps = 6.66e-16`。
于是最小特征方向被静默丢弃，`pinv` 退化成秩 2 的伪逆：

```
svd(J'J)                = [8.68e15, 3.19e7, 3.39e-1]
s_min / s_max           = 3.91e-17   <  6.66e-16  → 被丢弃
pinv(默认) 的 se        = [3.95431143e-07, 5.94300115e-04, 1.71020052e-07]  ← westata 报的
inv(J'J)   的 se        = [5.76788434e+00, 3.88353577e-03, 6.25759448e-07]  ← Stata 报的
```

三条独立证据：
1. `np.linalg.inv(J'J)` 与 `np.linalg.pinv(J'J, rcond=0.0)` 都给出 Stata 的数（相对差 <1e-9），
   只有默认 `rcond` 的 `pinv` 给出 westata 的数；
2. `_jacobian()` 是中心差分，步长 `1e-6*max(1,|θ|)`，对 `weight^2` 这类量级到 1e5 的项
   精度足够（`max|J - X_analytic| = 5.2e-09`，相对 5e-11），**不是**差分误差；
3. 同一模型的 OLS（`y ~ weight + I(weight**2)`）两边 SE 完全一致，说明数据/设计没问题。

**影响**：任何"变量量级差异大"的 `nl` 模型（多项式、幂函数、指数衰减）都会踩到；
一旦最小特征方向被截断，SE 会小几个数量级，t 值虚高，显著性判断完全错误。
这属于**必须修**的缺陷，不是可接受的数值差异。

**建议**：改用 `np.linalg.solve` / `pinv(..., rcond=<很小>)`，或者在 `J'J` 的秩不足时
报 `ConsistencyError`（Stata 在这种情况下会提示共线），而不是返回一个数值上"看起来合理"的 SE。

### 2. `nl` 默认初值不收敛（第 11 条，相对差 1.42e+01）

**现象**：不给 `start=` 时，westata 的每个参数初值都是 1.0，LM 停在一个离解很远的点：

```
Stata  (grid search 起步) RSS = 4.538e+08,  b = [19581.42, -0.0569546]
westata (初值全 1)         RSS = 3.417e+09,  b = [2.27e-10,  0.750895441]
```

RSS 差 7.5 倍。westata 会发 `RuntimeWarning: nl: the optimiser did not converge`，
但**默认仍然返回这个结果**（只有 `strict=True` 才抛异常）。

**归因**：Stata 的 `nl` 在迭代前自己做网格搜索选初值；westata 明确写"默认 1.0"。
这是设计选择，但后果是：用户按 Stata 写法跑 `nl (...)` 会拿到一个 Warning + 一组错的系数，
若把 warning 过滤掉就会静默出错。

**建议**：至少让"未收敛"默认抛 `ConvergenceError`；更好的做法是复刻 Stata 的起步网格搜索。

### 3. `nl` 其余两个良态模型的 B 档差异（第 7、10 条）

- 第 7 条（指数衰减，给了 Stata 的初值）：系数 1.57e-07、SE 1.58e-07；
- 第 10 条（幂函数 `{b0}*mpg^{b1}`）：系数 8.02e-08、SE 9.23e-08。

**归因**：这是两个优化器的停止点差异（Stata 的 Gauss-Newton/DFP vs westata 的
Levenberg-Marquardt `xtol=ftol=gtol=1e-15`），RSS 相对差在 1e-15 量级、系数差 1e-7。
**不影响任何显著性判断**（t 值第 3 位有效数字才可能变）。第 9 条（线性参数化、
条件数 ~1e2）落在 A 档，说明 westata 的 Gauss-Newton 协方差公式本身是对的，
问题只在病态设计（见第 1 点）。

### 4. `teffects aipw ..., atet` 的 ATET 与两个 PO 均值都错（第 17 条）

**现象**：

| 量 | Stata | westata | 相对差 |
|---|---|---|---|
| ATET | 2.05300232 | 2.04669372 | 3.07e-03（相对 ATET） |
| PO(0) | 0.02362706 | 0.01125893 | — |
| PO(1) | 2.07662938 | 2.05795265 | — |

（顶层 `param_rel` 记的是 5.23e-01，因为 PO(0) 本身接近 0，相对差被放大；
按 ATET 的绝对差算是 6.3e-03。）

**归因（用 westata 自己的中间量反推 Stata 的公式，误差 4.4e-16，已反推成功）**：

Stata 的 ATET 是
```
PO(0) = (1/n1) [ Σ_{t=1} x_i'b0 + Σ_{t=0} odds_i (y_i - x_i'b0) ]      ← 增广
PO(1) = mean_{t=1} y_i                                                ← 未增广
ATET  = PO(1) - PO(0)
```
用 westata 的 `_LogitPropensity` 得到的 p_i、以及两臂 OLS 的 b0，可以把这个 PO(0)
重算到 `0.023627059413063943`（Stata 是 `0.02362705941306397`，相对差 2.8e-17），
PO(1) 重算到 `2.0766293842852925`（与 Stata 位相同）。**所以 Stata 的公式已被完全确定。**

westata 的 ATET 分支（`AIPW.fit` 的 `else` 分支）做的是：
```
mean1 = mean_{t=1}(augmented1)   where augmented1 = x1'b1 + t*(y-x1'b1)/p
mean0 = mean_{t=1}(augmented0)   where augmented0 = x0'b0 + (1-t)*(y-x0'b0)/(1-p)
```
问题有两处，且**不互相抵消**：
- `augmented0` 里带 `(1-t)`，在 treated 子样本上恒为 0，于是控制组的增广项整个消失，
  PO(0) 退化成 `mean_{t=1}(x0'b0)`（未增广的回归预测）；
- `augmented1` 在 treated 上多了 1/p 的权重，PO(1) 变成 `mean_t[x1'b1 + r1/p]`
  而不是 `mean_t(y)`。

而 **ATE 分支是对的**（1.11e-16），说明倾向得分模型、两臂结果回归、影响函数都没有问题，
缺陷**只存在于 `estimand="atet"` 这条代码路径**。

**影响**：ATET 偏差 0.6%（2.053 vs 2.047），SE 偏差 2.9%（0.1203 vs 0.1236）。
在 n=400 的样本里两者都显著（t≈17），所以"是否为 0"的结论不变，
但**系数值的第 3 位有效数字就错了**，且 PO 表格完全不能看。

### 5. `teffects nnmatch` 的点估计精确、SE 全错（第 19–21 条）

| 命令 | Stata 估计 | westata 估计 | Stata SE | westata SE | SE 相对差 |
|---|---|---|---|---|---|
| ATE, nn(1) | 1.971774291 | 1.971774291 | 0.11993310 | 0.11383146 | 5.09e-02 |
| ATE, nn(3) | 2.005731686 | 2.005731686 | 0.11005426 | 0.10506617 | 4.53e-02 |
| ATET, nn(1) | 1.893276132 | 1.893276132 | 0.13593684 | 0.11383146 | 1.63e-01 |

**点估计**：三条全是 A 档（Mahalanobis 距离 + 有放回匹配，与 Stata 完全一致），
说明距离度量、邻居选择、有放回语义都对。

**SE**：westata 自己在源码里就标注了 `se_method="Abadie-Imbens approximation"`，
即它知道这个数不是精确复现。本次实测确认了这一点，并补充了一个**新的独立缺陷**：

> `nnmatch` 的 **ATET** SE 与 **ATE** SE **完全相同**（都是 0.11383146295432885，逐位相同）。
> Stata 的两个数不同（0.13593684 vs 0.11993310，比值 0.8823）。
> 机制：ATET 分支里 `share0 = share1 = 1.0`，而 ATE 分支里 `share0 = n0/n`、
> `share1 = n1/n`；`potential_outcomes` 的 SE 用 `std_err*sqrt(share)` 缩放，
> share 取 1 时两者退化到同一个数。也就是说 **ATET 的方差根本没有单独计算**。

**归因（SE 差距的来源）**：Abadie-Imbens 方差里
`V = (1/n^2) Σ (1+K_i)^2 σ²_i`，`K_i`（复用次数）westata 是精确的，
`σ²_i` 用"同臂最近邻的样本方差"估。Stata 的 `_matchnn_estimates` 是 Mata 字节码，
没有可读源码；westata 试过的 5 种 `σ²_i` 变体把 Stata 的数夹在 0.0691–0.0706 之间
（源码注释记录），本次在 n=400、nn∈{1,3} 上测得的相对差 4.5%–5.1% 与之一致。
**这一项是本领域唯一"知道差在哪一层、但无法给出精确公式"的**，如实记为 E 档。

**关于 `biasadj()`**（第 22 条）：westata 的 `NearestNeighbourMatching.fit()` **不接受任何参数**，
`fit(bias_adjust=[...])` 直接 `TypeError`。Stata 加上 `biasadj(x1 x2)` 后估计从
1.97177429 变成 1.95603462（差 8.05e-03）。这是**能力缺失**，不是数值问题。

### 6. `didregress` 的"无协变量"写法会被自动加协变量（第 25 条）

**现象**：Stata 写 `didregress (y) (treated), group(unit) time(period)` 表示**不加协变量**；
westata 的 `Didregress.from_formula("y", "treated", panel2, ...)` 在 `covariates=None` 时
会把**所有其余"组内时变"的数值列**当协变量（并发 `RuntimeWarning` 提示）。
本次面板里有一列无关的 `x1`，于是：

```
Stata  ATET = 1.22558603   (无协变量)
westata ATET = 1.22581273   (自动加上了 x1)
相对差 1.85e-04  → C 档
```

（把表窄化成 4 列 `unit/period/treated/y` 后，两者**位相同**，见第 24 条。）

**归因**：这是 **API 语义差异 + 危险默认值**，不是数值 bug。但它很危险：
Stata 用户按习惯写 `(y)`，数据里恰好有一列时变数值（面板数据里几乎总有），
结果会悄悄变成"带协变量的 DID"，而且**只发一个 RuntimeWarning**。
westata 也**无法表达"显式不要协变量"**（`covariates=[]` 抛 `ValueError`）。

**建议**：`covariates=None` 时应当**默认不加协变量**（与 Stata 一致），
需要推断时用显式开关（如 `infer_covariates=True`）。目前的行为与 Stata 相反。

### 7. `jackknife` / `bootstrap` 的 SE 差 1e-7 ~ 5e-6：**是 Stata 的 float32 存储**

这是本次最有意思的一条，因为它看起来像"westata 算错了"，实测证明**恰恰相反**。

**做法**：`auto.dta` 全是整数，所以 74 次 leave-one-out OLS 可以用
`fractions.Fraction` **精确有理数**求解，得到一个与两个实现都无关的真值。

| 量 | 精确有理数 | westata | Stata |
|---|---|---|---|
| jackknife SE（weight） | 5.94884416e-04 | 5.94884416e-04 | 5.94884535e-04 |
| jackknife SE（_cons） | 2.02203366e+00 | 2.02203366e+00 | 2.02203469e+00 |
| jackknife SE（mean，= s/√n） | 0.6725510870764972 | 0.6725510870764978 | 0.6725541686341475 |

- **westata 与精确解一致**（1.15e-15 / 0.0）；
- **Stata 偏离精确解 5.11e-07（回归）/ 4.58e-06（均值）**；
- **把精确的 replicates 过一遍 `float32` 再套同一公式，能复现 Stata 的数到 2.2e-16**。

结论：`jackknife` 与 `bootstrap` 在 Stata 内部**以单精度存储每个 replicate**，
所以它的 SE 天然带 ~1e-7 相对误差。westata 用双精度，因此两者的 B/C 档差异
**是 Stata 的舍入，不是 westata 的错误**。

同一机制也解释了 bootstrap 的 SE 差（第 51 条）：

```
Stata 保存的 100 个 replicate（双精度）算协方差  → 与 Stata e(V) 差 8.70e-09，与 westata 差 4.2e-16
同一批 replicate 先过 float32 再算协方差          → 与 Stata e(V) 差 2.10e-16
```

即 westata 的 bootstrap 协方差与"Stata 的 replicate 的双精度协方差"一致到 1e-15，
两者报出的 SE 差异全部来自 Stata 的单精度存储。

> 这里推翻了 westata 源码注释里的旧说法（"Stata 用 Sherman-Morrison 更新，
> 所以第六位不同"）。实测：Sherman-Morrison 解析更新与直接重拟合在双精度下
> 只差 7.1e-14，解释不了 5e-7；**float32 存储能解释到 1e-16**。

### 8. `jackknife` 的 t 检验自由度用错（第 48 条）

```
Stata  e(df_r) = 73   (N - 1：每个 replicate 贡献一个自由度)
westata df_resid = 72 (N - k)
```

于是 95% 临界值不同（t(73)=1.992997 vs t(72)=1.993464），`_cons` 的置信区间：

```
westata: [35.40943, 43.47113]
Stata  : [35.41037, 43.47019]
```

**归因**：自由度约定不同（df_r 相对差 1.37e-02）。数值影响很小（区间宽度差 0.02%），
但属于"参考分布不同"，不是舍入。注意 `bootstrap` 的 Stata 侧根本没有 `e(df_r)`
（表格打 `z`、`P>|z|`），westata 却给了 `t(72)`：

```
t(72) 临界值 = 1.993464，  z 临界值 = 1.959964，  相对差 1.71e-02
westata bootstrap 的 _cons CI: [35.22342, 43.65714]
Stata    bootstrap 的 _cons CI: [35.29429, 43.58628]
```

即 **westata 的 bootstrap 区间系统性偏宽约 1.7%**，p 值偏保守。
这属于"统计推断口径"差异，会影响边界显著性的判断。

### 9. `bootstrap` 的"能否逐次复现"：**能，精确到机器精度**（第 50 条）

这是给 BRIEF 里"随机方法必须写清是否可逐次复现"的正面回答。

做法：在 Stata 里用 `postfile` + 自定义 program 把每个 replicate 的
`_b[weight]`、`_b[_cons]` 写盘，再在 Python 侧用 `Mt19937_64(seed)` 复刻
`bsampl_w` 的权重流，逐次重拟合：

```
100 个 replicate × 2 个系数 = 200 个数
最大相对差 = 1.41e-15
最大绝对差 = 7.8e-18 (weight) / 5.0e-14 (_cons)
```

**逐位级别一致**。原因：Stata 的 bootstrap 不是"重抽样后重拟合"，而是把
`bsampl_w` 生成的频数权重传给命令；westata 精确复刻了 `bsampl_w`
（包括第二个 uniform 流做 tie-break），并从同一个 MT19937-64 种子出流。
所以对 `regress` / `mean` 这类线性统计量，**同种子下可以逐次复现**，
不是"只能比分布"。

（这也是为什么第 51/52 条的 SE 差异可以断定是 Stata 的 float32 存储——
replicate 本身已经对到 1e-15，剩下的只能是求协方差那一步。）

### 10. `permute`："逐次复现"**不能**，"分布层面"**一致**（第 53–55 条）

- **观测统计量**（未置换的 `_b[weight]`）：A 档（2.80e-15）。这是确定量，必须一致。
  额外发现：Stata 的 `r(b)` 副本以 **float32** 存储（`1.746559143066406` vs
  `e(b)` 的 `1.7465591583457587`，相对差 8.75e-09），所以用 `r(b)` 去比会误判为 B 档；
  用 `e(b)`（或直接跑一次 `regress`）才是对的。
- **同种子 p 值**：**不可逐次复现**。Stata 的 shuffle 顺序未公开（源码注释也这么说），
  westata 用自己的 Fisher-Yates + MT19937-64，置换序列不同。
  本次实测：`weight` 对 `price` 的置换，Stata 200 次里上侧 0 次命中
  → 双侧 p = 0.0000；westata 用 `(c+1)/(reps+1)` 下限为 1/201 = 0.004975。
  绝对差 4.98e-03 = 恰好 1 个蒙特卡洛网格。记 D 档（按绝对差判，因为 Stata 侧可以是 0，
  相对差无定义）。
- **分布层面**：在 `y = 0.05*x + noise`（n=80，弱效应）下，6 个种子 × 400 次置换：

```
Stata   6 个 p: [0.880, 0.825, 0.810, 0.790, 0.845, 0.795]   均值 0.8242
westata 6 个 p: [0.8254, 0.7880, 0.8229, 0.8978, 0.8130, 0.8703]  均值 0.8362
均值差 1.21e-02
```

  两边都落在 p≈0.82 附近，均值差 0.012，小于单个均值的蒙特卡洛标准误（~0.04）。
  **结论：数值量级一致，逐次不可复现**；westata 的 `(c+1)/(reps+1)` 规则会让 p 值
  系统性偏大 ~1/(reps+1)（本次 0.0025），属于保守侧。
  **这只是量级检查，不是精度检查**：6 个样本无法检出细微偏差，NOTES 不据此宣称"分布完全一致"。

（另外记一笔：westata 的 `Permute` 只支持 `permute=<某个回归自变量>` 这一种用法，
Stata 的 `permute <varlist> <exp_list>` 通用形式不支持。）

### 11. `sureg` 的 `vce(robust)` 与 `didregress` 的 `vce(unadjusted)`

- `sureg ..., vce(robust)`：Stata 支持，SE = `[0.75993745, 91.70326703, 4053.05585619,
  0.00053137, 1.09265040, 1.74656612]`；westata 对 `cov_type="robust"` 抛
  `UnsupportedError`（只实现 unadjusted）。**能力缺失**。
  注：westata 的 default `cov_type="nonrobust"` 与 Stata 的 default `small` 一致到 A 档，
  这一条是唯一不覆盖的 VCE 变体。
- `didregress ..., vce(unadjusted)`：**Stata 19.5 自己拒绝**（`vcetype unadjusted not allowed`，
  r(198)），而 westata 实现了 `cov_type="nonrobust"`。方向反过来了：
  westata 比 Stata 多一个选项，无从对比。此条记 skip 并附 westata 的数值
  （ATET 1.22581273，SE 0.209524096）。

### 12. `svy` 系列：全部 A 档，包括 self-weight / strata / fpc / lone PSU

这一块是本次最干净的：

- 点估计相对差 ≤ 3.90e-15，SE 相对差 **大量为 0（逐位相同）**；
- `e(N)/e(N_psu)/e(N_strata)/e(N_pop)/e(df_r)` 全部逐位一致；
- self-weight（`svyset psu`，无 pw）、`strata()`、`fpc()`（>1 读作总体 PSU 数）
  三种设定都对；
- lone PSU：Stata 默认 `singleunit(missing)` 打 `.` 且 **`e(V)` 存 0**
  （不是 NaN！），westata 返回 NaN。两者语义相同（"不可估"），比较时把 0 映射为"不可估"后记 A 档；
  `singleunit(centered)` 两边 SE 也完全一致（0）；
- `svy: proportion` 的基准水平：Stata 用 `0bn.grp` 存**真实估计值**（不是基准占位），
  `Stata.estimate()` 会把它当基准行删掉——本次用 `capture_raw()` 保留全部行并显式 rename，
  才拿到完整对比。**这是一个工具层面的坑，写进"未覆盖/注意事项"**。
- `svy: logit` 的线性化 VCE 与 Stata 一致到 1.5e-15（westata 没有走"拟似然/robust"替代）。

**唯一未覆盖**：`svy: mean y, over(f)`（子总体估计）。westata 的 `SvyMean`/`SvyProportion`
没有 `over()` 概念，无法表达。

### 13. `sureg` / `mvreg` 的 `e(V)` 交叉项（BRIEF 明确要求）

不是只比 SE——把整块 `e(V)` 拉平后逐元素比：

| 命令 | `e(V)` 全矩阵相对差 | 档位 |
|---|---|---|
| `sureg`（6×6，含两方程间交叉协方差） | 1.76e-12 | A |
| `mvreg`（6×6） | 3.61e-12 | A |

`sureg` 的跨方程块（如 `Cov(b_price,weight , b_mpg,weight) = -5.45875582e+02`）
与 `mvreg` 的 `Σ ⊗ (X'X)^-1` 全部对上。系数逐位（5.75e-13 来自 `price:weight = 1.33525098`
这类量的舍入）。

### 14. `manova`

Wilks λ、Rao F、df1、df2、p 五个量在三种 (p,q) 组合下全部 A 档（≤7.6e-14）：

| (p, q) | λ | F | df1 | df2 | 备注 |
|---|---|---|---|---|---|
| (2, 1) | 0.7756360582941422 | 10.268888153646719 | 2 | 71 | `p²+q²=5` → s=1，Stata 标 "e"（精确） |
| (3, 2) | 0.8061... | 3.23... | 6 | 170 | s=2，Stata 标 "e" |
| (3, 3) | 0.5617098732060969 | 6.802226667428487 | 9 | **228.92190582578982** | df2 是**分数**，s=2.4337，Stata 标 "a"（近似） |

第 6 条最能说明问题：df2 是 228.9219…，只有正确实现 Rao 变换才能得到这个分数；
任何"数格子"的自由度公式都会错。westata 对上了（df2 相对差 0）。
另外 `e(df_1)`（= q）与 `e(df_r)`（= n - g）也一致。

**限制**：westata 只实现 Wilks；Stata 的 `e(stat_m)` 还有 Pillai / Lawley-Hotelling /
Roy 三行（同一组 df），westata 抛 `UnsupportedError`。本次只比 Wilks 行。
`manova` 也只支持单因子、无协变量。

### 15. `didregress` / `xtdidregress`：ATET 与 SE 全部 A 档

- 两组两期（带/不带协变量、`vce(cluster)` / `vce(robust)`）：ATET 与 SE 均 ≤2.08e-15；
- 5 期交错采纳（cohort 2 组，N=600）：`didregress` ATET 1.67e-15、`xtdidregress` 1.67e-15；
- `e(N)`、`e(df_r)`（= 119）、`e(N_clust)`（= 120）全部逐位一致；
- **两个拼写的自由度约定**：`didregress` SE = 0.29746353，`xtdidregress` SE = 0.20851734，
  比值 0.7012；westata 复现这个比值到 1.81e-16（即两者的 `(N-1)/(N-K)` 因子差
  被正确实现，不是简单缩放）。

---

## 与旧结论的出入

本次**不引用**旧 registry / GAP_REGISTER 作为证据，但既然要写"出入"，
只陈述一个**能从 westata 源码注释直接读到的旧结论**以及本次实测的反驳：

| 旧说法（出处） | 本次实测 | 结论 |
|---|---|---|
| `jackknife` 的回归方差与 Stata 差 ~7e-6，原因是"Stata 用 Sherman-Morrison 解析更新而不是重拟合"（`src/westata/estimators/resample.py` 模块 docstring + `e(note)`） | 用精确有理数求真值：westata 1.15e-15 命中真值，Stata 偏 5.11e-07。Sherman-Morrison 与直接重拟合在双精度下只差 7.1e-14，**解释不了 5e-7**；把 replicates 过 float32 能复现 Stata 到 2.2e-16 | **旧归因错误**。真正原因是 Stata 的 replicate 单精度存储。westata 的数是**更准**的那个 |
| `nnmatch` 的 SE 是"Abadie-Imbens 近似"，与 Stata 差 +2.2%（源码注释，基于其内部 fixture） | 本次在独立数据（n=400, 2 个协变量）上测得 ATE 差 5.09%（nn=1）/4.53%（nn=3）、ATET 差 16.3% | **差距比旧记录更大**；且发现**新缺陷**：ATET 的 SE 与 ATE 的 SE 完全相同（share 取 1），ATE 与 ATET 没有分别估计方差 |
| `sureg`/`mvreg`/`manova`/`svy`/`didregress`/`ra`/`ipw` 未见"已知问题"记录 | 全部 A 档 | 一致（本次独立确认，不是沿用旧结论） |

---

## 未覆盖

1. **`sureg ... , vce(robust)`**：westata 只实现 unadjusted VCE。
   Stata 侧数值已记录（6 个 SE），westata 侧 `UnsupportedError`。
2. **`sureg, isure`（迭代 FGLS）**：westata 明确 `UnsupportedError`。未测（没有可比的第二组解）。
3. **`teffects nnmatch biasadj()`**：westata 的 `fit()` 无参数，能力缺失。
   Stata 侧数值已记录（1.95603462，相对无调整差 8.05e-03）。
4. **`svy: mean, over()`**：westata 无子总体估计。
5. **`manova` 的 Pillai / Lawley-Hotelling / Roy 统计量与多因子、协变量模型**：
   westata 只做单因子 Wilks。
6. **`teffects` 的 `vce()` 变体**（`vce(iid)` vs 默认 robust）：westata 无此选项，
   两边都只跑默认。
7. **`teffects` 的 PO 均值 SE 全量**：`ra`/`ipw`/`aipw` 的 `POmean(1)` 行 Stata 存了，
   但只有 `aipw atet` 单独做了公式核对；其余条的 `POmean(1)` 在 `effect_compare` 里
   依 `po_levels` 参数只比了 `POmean(0)`（`aipw ATE` 也只比 PO(0)）。
   这是权衡：Stata 的 `e(b)` 把 nuisance 模型混在同一向量里，比太多行会让
   "配不上"的信号淹没真正的差异。PO(0) 已足以暴露 ATET 分支的问题。
8. **`nl` 的 `vce(robust)`**：Stata 的 `nl` 本身没有 `vce()`，westata 也拒绝；
   未设为该领域的失败项。
9. **`bootstrap` 的 BC/BCa 置信区间**：westata 只给 percentile 协方差
   （`np.cov(replicates, ddof=1)`），Stata 的 `bca`/`bc` 选项未测。
10. **`permute` 的通用 `exp_list`**：westata 只支持 `permute=<回归自变量>`（置换某个系数），
    不支持 Stata 的 `permute <varlist> <exp_list>`（如 `permute foreign r(F)`）。
    本次两组数据都用的单系数形式。

---

## 复现方式

```bash
cd D:\myCodeProject\statapy
.venv\Scripts\python.exe retest/domains/systems_causal/run.py
```

脚本会覆盖 `results.jsonl` / `summary.json`，并在 `_tmp/` 下写三个 Stata 侧
replicate 转储（`bsrep.dta` 等，用于逐次复现核对，可随时删除）。

注意事项（本次踩过的坑，写给后来者）：
- `svy: proportion` 的基准水平 `0bn.<var>` 在 Stata 里是**真实估计值**，
  `Stata.estimate()` 会把它当基准删掉；本脚本用 `capture_raw()` 绕开。
- `permute` 的 `r(b)` 是 float32，`e(b)` 是 double；比观测统计量要用 `e(b)`。
- `permute` 不留 `e(b)`，`st.estimate()` 会抛 "did not leave e(b)"，必须用 `st.output()`。
- Stata 的 `didregress` 不接受 `vce(unadjusted)`（r(198)）。
- `jackknife`/`bootstrap` 的 `e(df_r)`：jackknife 是 N-1（bootstrap 没有）。
