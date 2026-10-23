# discrete 领域复测结论

> 复测对象：本机 **StataNow 19.5 (MP)** vs 本机 **westata**（`src/`，editable 安装）
> 证据来源：`retest/domains/discrete/run.py` 现场启动的 pystata 会话 + 本机 westata 实跑。
> 未读取 `tests/conformance/*`、`docs/GAP_REGISTER.md`、`conformance*.csv`。
> 数据：`numpy.default_rng(20261024)` 自造，n=600，`st.send(df)` 送入 Stata；
> 每个对比点都断言 Stata `nobs()` 与 westata `nobs` 相同。
> 判定口径按 BRIEF：A ≤1e-9、B ≤1e-6、C ≤1e-3、D ≤1e-2、E >1e-2。
> `summary.json` 计数：**70 个对比点，pass 15 / fail 39 / error 13 / skip 3**。

## 覆盖矩阵

`系数相对差` / `标准误相对差` 均取逐系数最大值；`nan` 表示该侧不产出该量。
`perm` 见下文"对齐说明"。

| # | Stata 命令 | westata 调用 | 数据 | 判定 | 系数相对差 | 标准误相对差 |
|---|---|---|---|---|---|---|
| 1 | `logit yb x1 x2` | `Logit.from_formula("yb~x1+x2").fit()` | 自造 n=600 | **pass A/A** | 1.09e-13 | 3.56e-14 || 2 | `logit …, or` | `exp(params)`, `exp(b)*se` | 同上 | **pass A/A** | 8.53e-14 | 1.21e-13 |
| 3 | `logit …, vce(robust)` | `fit(cov_type="HC0")` | 同上 | **error** | — | — |
| 3b | 同上（诊断：statsmodels HC0 直调） | — | 同上 | fail C/C | 8.34e-04 | 8.34e-04 |
| 4 | `logit …, vce(cluster g)` | `fit(cov_type="cluster", cov_kwds={"groups": g})` | 同上 | fail A/**D** | 1.09e-13 | 1.67e-03 |
| 4b | 同上 e() 标量 | — | 同上 | fail **E** | 3.72e-01 (`e(chi2)`) | — |
| 5 | `probit yb x1 x2` | `Probit…fit()` | 同上 | fail **B**/A | 1.79e-09 | 3.25e-10 |
| 6 | `probit …, vce(robust)` | `fit(cov_type="HC0")` | 同上 | **error** | — | — |
| 7 | `probit …, vce(cluster g)` | 同 4 | 同上 | fail B/**D** | 1.79e-09 | 1.67e-03 |
| 8 | `poisson y_c x1 x2` | `Poisson…fit()` | 同上 | fail **B**/B | 2.91e-08 | 5.26e-09 |
| 9 | `poisson …, vce(robust)` | `fit(cov_type="HC0")` | 同上 | fail B/**C** | 2.91e-08 | 8.34e-04 |
| 10 | `poisson …, vce(cluster g)` | 同 4 | 同上 | fail B/**D** | 2.91e-08 | 1.67e-03 |
| 11 | `poisson` 的 `e(deviance)`/`e(alpha)` | — | 同上 | **skip** | — | — |
| 12 | `poisson cov_type="HC1"` | `fit(cov_type="HC1")` | 同上 | **error** | 0（与 HC0 逐位相同） | — |
| 13 | `logit … [fweight=w]` | `fit(weights=…)` | 同上（w∈{1,2,3}） | **error** | — | — |
| 14 | `logit … if x1>0` | `from_formula(..., subset=)` | 同上（N=290） | **pass A/A** | 1.32e-10 | 3.69e-11 |
| 15 | `nbreg y_nb x1 x2` | `NegativeBinomial…fit()` | 同上（真 alpha=0.5） | fail **C**/C | 1.82e-04 | 2.80e-05 |
| 16 | 同上 `/lnalpha`、`alpha` 与其 SE | `params[-1]`, `bse[-1]` | 同上 | fail C/C | 1.82e-04 | 4.59e-06 |
| 17 | 同上（紧容差归因） | statsmodels bfgs `gtol=1e-10` | 同上 | pass **B** | 1.50e-08 | — |
| 18 | `nbreg …, dispersion(constant)`（NB1） | 无对应 API | 同上 | **error** | — | — |
| 19 | 同上（独立 NB1 参考实现） | 手写 likelihood | 同上 | fail B/B | 2.20e-08 | 5.38e-07 || 20 | `nbreg …, vce(robust)` | `fit(cov_type="HC0")` | 同上 | **error** | — | — |
| 21 | `nbreg …, vce(cluster g)` | 同 4 | 同上 | fail C/**D** | 1.82e-04 | 2.52e-03 |
| 22 | `zip y_zi x1 x2, inflate(x1 x2)` | `ZeroInflatedPoisson…fit()` | 同上 | fail B/**C** | 2.46e-08 | 6.76e-06 |
| 23 | 同上 `e(V)` 全矩阵 | `cov_params()` | 同上 | fail **C** | 5.35e-05 | — |
| 24 | 同上 `e(p)` | `e["p"]` | 同上 | **error** | 1.15e+02 | — |
| 25 | `zip y_zi x1 x2`（无 inflate） | `inflate=None` | 同上 | **skip** | — | — |
| 26 | `zip …, inflate() vce(robust)` | `fit(cov_type="HC0")` | 同上 | fail B/**C** | 2.46e-08 | 8.47e-04 |
| 27 | `zinb y_zinb x1 x2, inflate(x1 x2)` | `ZeroInflatedNegativeBinomial…fit()` | 同上 | fail **D**/C | 1.06e-03 | 9.42e-05 |
| 28 | 同上 `e(V)` 全矩阵 | `cov_params()` | 同上 | fail **C** | 5.17e-04 | — |
| 29 | 同上（紧容差归因） | `_CountLikelihood` `gtol=1e-10` | 同上 | pass **B** | 9.96e-07 | — |
| 30 | `ztp y_nb x1 x2` | `TruncatedPoisson…fit()` | 仅 y>0，N=365 | fail **B**/B | 3.60e-08 | 4.05e-08 |
| 31 | 同上 `e(ll_0)` 约定 | 手算 `mean(y)` 单参数 null | 同上 | **pass A** | 1.68e-16 | — |
| 32 | `ztp …, vce(robust)` | `fit(cov_type="HC0")` | 同上 | fail B/**D** | 3.60e-08 | 1.37e-03 |
| 33 | `tnbreg y_nb x1 x2` | `TruncatedNegativeBinomial…fit()` | 同上，N=365 | fail **C**/C | 1.08e-05 | 2.22e-06 |
| 34 | 同上 `e(V)` 全矩阵 | `cov_params()` | 同上 | fail **C** | 8.11e-05 | — |
| 35 | `ologit ycat x1 x2` | `OrderedLogit…fit()` | 4 类有序 | fail **C**/**E** | 3.55e-05 | **1.76e+00** |
| 36 | 同上 `e(V)` 全矩阵 | `cov_params()` | 同上 | fail **E** | **2.09e+01** | — |
| 37 | 同上（正确 cut Jacobian 参考） | 手算 J | 同上 | **pass** C/A | 1.01e-06 | 2.28e-10 |
| 38 | `oprobit ycat x1 x2` | `OrderedProbit…fit()` | 同上 | fail **C**/**E** | 3.80e-06 | **1.37e+00** |
| 39 | 同上 `e(V)` 全矩阵 | `cov_params()` | 同上 | fail **E** | **1.05e+01** | — |
| 40 | 同上（正确 cut Jacobian 参考） | 手算 J | 同上 | **pass** C/B | 7.36e-06 | 3.33e-09 |
| 41 | `ologit / oprobit …, vce(robust)` | `fit(cov_type="HC0")` | 同上 | **error**×2 | — | — |
| 42 | `logit` → `predict, pr` | `predict(kind="pr")` | 同上 | **pass A** | 2.34e-13 | — |
| 43 | `logit` → `predict, xb` | `predict(kind="xb")` | 同上 | **pass A** | 2.88e-12 | — |
| 44 | `poisson` → `predict, n` | `predict(kind="n")` | 同上 | fail **B** | 2.41e-08 | — |
| 45 | `ztp` → `predict, n` | `predict(kind="n")` | N=365 | fail **E** | 8.68e-01 | — |
| 46 | `ztp` → `fittedvalues` | `result.fittedvalues` | N=365 | fail **E** | 1.71e+00 | — |
| 47 | `zip` → `predict, n` | `predict(kind="n")` | 同上 | **error** | — | — |
| 48 | `zip` → `fittedvalues` | `result.fittedvalues` | 同上 | fail B | 1.09e-09 | — |
| 49 | `ologit` → `predict, pr` | `predict(kind="pr")` | 同上 | **error** | — | — |
| 50 | `ologit` → 期望类别 | 由 4 个 `predict, pr outcome(k)` 重建 | 同上 | fail **C** | 2.08e-05 | — |
| 51 | `logit yb x1 i.grpf` | `Logit("yb~x1+C(grpf)")` | 同上 | fail **B**/B | 1.02e-07 | 2.06e-08 |
| 52 | 迭代/收敛诊断（`e(ic)`、`e(converged)`） | — | — | **skip** | — | — |
| 53 | `vce()` 下 `e(chi2)` LR→Wald | `e["chi2"]` | 5 条命令 | **error** | 3.72e-01 | — |
| 54 | `e(df_m)` 排除辅助参数 | `e["df_m"]` | nbreg/ologit/oprobit | **error** | 1.00e+00 | — |

> 表里 `#` 为记录序号（`results.jsonl` 同名条目可直接查原始 `stata_b/westata_b/V`）。

## 对齐说明（为什么不用 `compare` 的按名对齐）

`retest/lib/stata_probe.compare` 按系数规范名配对，但本领域两个模型让它失效：

1. **zip/zinb 的 `e(b)` 列名按方程局部命名且重复**：`zip y_zi x1 x2, inflate(x1 x2)`
   的 `colnames e(b)` 是 `x1 x2 _cons x1 x2 _cons`（`matrix list e(b)` 里前三个带
   `yzi:` 前缀、后三个带 `inflate:` 前缀，但 `colnames` 局部宏不带）。按名对齐会把
   主方程的 `_cons` 配到 inflate 的 `_cons`，得到假的巨大差异。
2. **列序不同**：Stata `e(b)` 是 `[x1 x2 _cons /lnalpha]`，westata 是
   `[Intercept, x1, x2, /lnalpha]`（patsy 常数项在前）。

所以本文件用**位置对齐** `perm`（`stata[perm]` 与 `westata.params` 同序），
并把 `perm` 原样写进每条记录的 `notes` 与 `extra`，不做任何猜测。
`logit/probit/poisson` 这类无辅助参数的模型另外用官方 `compare` 复核过（结果一致）。

**Stata 侧开了 `set type double`**：`predict` 默认把新变量建成 `float`，会白白丢掉
有效位。对照实验：`logit` 的 `predict, pr` 在 `float` 变量上最大相对差 5.43e-8
（B 档），在 `double` 变量上是 2.34e-13（A 档）。这是 Stata 的存储精度，不是
westata 的误差，必须区分开，否则会把 A 档误判成 B 档。

## 逐项说明（所有不达 A 档的条目）

### 0. 先说结论：哪些是**真错**，哪些只是"数值实现不同"

| 类别 | 条目 | 归因 |
|---|---|---|
| **真错 —— 必须修** | #24 zip/zinb `e(p)` 自由度、#35–39 ologit/oprobit `e(V)`、#47 zip `predict,n` 崩溃、#49 ologit `predict,pr` 崩溃、#12 `HC1` 静默返回 HC0、#53 `e(chi2)` 在 `vce()` 下没换成 Wald、#54 `e(df_m)` 把辅助参数算进去、#23/#3 稳健 VCE 全路径不可用 | 见各小节 |
| **未实现** | #1(3) `vce(robust)` 对 Logit/Probit/nbreg、#41 有序模型 robust、#18 NB1、#13 权重 | API 缺失，非数值误差 |
| **数值实现差异（B/C/D 档）** | #5 #8 #15 #16 #22 #30 #33 #51 + 全部 cluster 的 SE | 停步准则（bfgs `gtol=1e-5`）/ VCE 约定（`(N-1)/(N-K)` vs `G/(G-1)`） |
| **约定差异（skip）** | #11 `e(deviance)` 两侧都没有、#25 zip 无 inflate 时 Stata 直接拒绝 | — |

**A 档（≤1e-9）条目清单**（可以声称"完全一致"）：

| 命令 | 最大相对差 |
|---|---|
| `logit yb x1 x2`（系数 A 1.09e-13 / SE A 3.56e-14） | 1.09e-13 |
| `logit yb x1 x2, or`（OR 与 OR 的 SE） | 8.53e-14 / 1.21e-13 |
| `logit` 的 `e()` 标量（N/ll/ll_0/chi2/r2_p/df_m） | 3.64e-10 |
| `probit` 的 `e()` 标量 | 2.67e-10 |
| `poisson` 的 `e()` 标量 | 8.36e-16 |
| `logit … if x1>0`（子样本） | 1.32e-10 / 3.69e-11 |
| `ztp` 的 `e()` 标量 | 1.30e-15 |
| `ztp` 的 `e(ll_0)` 约定（独立手算复核） | 1.68e-16 |
| `logit → predict, pr` | 2.34e-13 |
| `logit → predict, xb` | 2.88e-12 |
| `ologit / oprobit` 用**正确 Jacobian** 复算的 cut SE | 2.28e-10 / 3.33e-09 |

### 1. `logit` 默认：A 档，没有问题
系数 1.09e-13、SE 3.56e-14，六个 `e()` 标量最差 3.64e-10（`e(chi2)`）。
westata 用 statsmodels `Logit` + `newton tol=1e-10`，和 Stata 的 `technique: nr` 同路。

### 2. `logit, or`：Stata 的 `or` 只改显示，A 档
关键事实（已断言，不是看输出）：`logit yb x1 x2, or` 的 `e(b)`/`e(V)` 与不带 `or` 的
**逐位相同**。所以 odds ratio 是 `exp(b)`、其 SE 是 `exp(b)*SE(b)`（delta method，
Jacobian 恰为 `exp(b)`）。westata 没有 `, or` 显示开关（也不需要），但按这个关系算出的
OR 与 SE 和 Stata 打印表一致到 8.53e-14 / 1.21e-13。**结论：`or` 不需要额外实现，只要
派生量对得上即可，实测对得上。**

### 3. `logit/probit/nbreg, vce(robust)`：**westata 直接抛异常（真错，全路径不可用）**

```
ValueError: heteroscedasticity robust covariance does not use keywords
```
根因（源码定位）：`src/westata/estimators/discrete.py:178` 对 **HC0 和 HC1 都**传入
`cov_kwds={"use_correction": canonical == "hc1"}`，而 statsmodels 的 `Logit/Probit/
NegativeBinomial` 一旦拿到 `cov_kwds` 就报这句。`_accepts_hc_kwds` 只有 `Poisson`
是 `False`，所以**只有 Poisson 幸免**。已实测：`Logit/Probit/NegativeBinomial` 的
`HC0` 与 `HC1` 两条路都 100% 抛异常。

诊断路径（绕开 TypeError 直调 statsmodels HC0）：SE 差 **8.34e-04**，且
**每个元素的 `V_stata/V_westata` 比值恒等于 `N/(N-1)` = 1.001669449**（标准差
1.4e-13）。即 Stata 的 ML robust VCE = 原始 sandwich × `N/(N-1)`，而 statsmodels
HC0 不打任何小样本因子、HC1 打 `N/(N-K)`。**Stata 没有对应选项，westata 两条路都
不对**。修完 TypeError 之后这一项仍然到不了 A 档，需要单独的 Stata-ML-robust 因子。

### 4. `vce(cluster g)`：系数 A 档，SE 差 D 档 —— VCE 约定问题
系数与 VCE 无关，仍是 1.09e-13（logit）。SE 差 1.67e-03，同样**逐元素比值恒定**：
`V_stata/V_westata = 0.996661`，而 `(N-1)/(N-K) = 1.003350`，`G/(G-1)=1.111111`。
手搭两套 sandwich 验证：
- 只用 `G/(G-1)` → 复现 Stata `e(V)` 到 **1.7e-7**；
- 用 `G/(G-1)*(N-1)/(N-K)` → 偏 **3.35e-3**。

即 **Stata 的 ML cluster 只有 `G/(G-1)`，没有 `(N-1)/(N-K)`**；westata 走的是
`vce/ssc.py` 的 CR1 公式（`logit` 这类非线性 ML 的 cluster 与 `regress` 不同）。
**归因：VCE 类型（cluster 小样本因子），不是 score/bread 算错。**

### 5. `logit vce(cluster) 的 e()` 标量差 37%：真错（`e(chi2)` 没换 Wald）
Stata 一旦 `vce(robust)`/`vce(cluster)`，`e(chi2type)` 宏从 `LR` 变成 `Wald`，
`e(chi2)` 也随之换成 **去掉常数项的 Wald 统计量**。五个命令实测验证：
把 `b'V⁻¹b` 只作用在斜率块上，**全部复现 `e(chi2)`**（logit-robust 复现到 8e-16 量级），
把常数项也算进去则偏到 1.4–3.4 倍。westata 的 `e(chi2)` **完全不受 `cov_type` 影响**
（`logit` cluster 的 `e(chi2)` 仍是 89.218111，与 `oim` 时一模一样），所以与 Stata
差 37.2%。**这是报错了统计量，不是精度。**

### 6. `probit` 默认：B 档（1.79e-09），但是**收敛点不同**、结论无影响
已证明：从 **Stata 的 b 出发走一步精确 Newton** 只移动 8.55e-10，落到距 statsmodels
解 1e-11 以内，两侧 `ll` 差 1.6e-16。即两条解在同一个极平坦的似然面上，
1e-9 的 b 差异代价为零。**归因：迭代收敛。不影响显著性。**

### 7. `poisson` 默认：B 档（2.91e-08）
`newton tol=1e-13`、`bfgs gtol=1e-12` 都停在同一个点（相对差不变），再加一步 Newton
步长只有 2.3e-17 —— 说明 statsmodels 已经到自己的定点，Stata 停在另一点，
`ll` 差 1.3e-16。**归因：迭代收敛（两侧停步准则不同）。** 注意 `poisson` 的 `e()`
标量是 A 档（8.36e-16），只有系数轻微不同。

### 8. `poisson vce(robust)`：B/C 档，同 #3 的 `N/(N-1)` 因子
Poisson 能跑通（`_accepts_hc_kwds=False`），系数 2.91e-08，SE 8.34e-04。**归因：VCE 约定。**

### 9. `poisson e(deviance) / e(alpha)`：**skip —— 两侧都没有**
Stata 19.5 的 `poisson` `ereturn list` 里**没有** `e(deviance)`、也没有 `e(alpha)`；
deviance 只以 `r(chi2_d)` 出现在 `estat gof` 里（本文件已记录该 `r()` 值）。
westata `Poisson.e` 也没有 `deviance`/`alpha` 键。也就是说这一条**不是 westata 漏了**，
是 Stata 本身不存；真要 deviance 得走 `estat gof`，而 westata 没有 Poisson 版的
`estat gof`。**这一项写进"未覆盖"，不算 westata 的缺陷，但调用方要注意不能假设
`e(deviance)` 存在。**

### 10. `poisson cov_type="HC1"` 与 `"HC0"` **逐位相同**（真错，静默给错答案）
`fit(cov_type="HC1")` 不报错，但返回的 SE 与 `HC0` **完全相同**（`array_equal` 为
True），只有 `cov_type` 标签不同。根因：`Poisson._accepts_hc_kwds = False`，
`cov_kwds` 从不下传，statsmodels 于是用它的 HC0 默认值。**没有任何路径能产出 Stata 的
`N/(N-1)` 因子。** 这种"接受参数但静默忽略"比抛异常更危险。

### 11. `logit [fweight=w]`：**未实现**
`_DiscreteBase.from_formula` / `.fit` 都没有 `weights`/`fweight` 参数，抛
`TypeError`。Stata 正常跑出 `N=1191`（权重和）与系数。离散/计数模型全家
（logit/probit/poisson/nbreg/zip/ologit）都没有权重支持。

### 12. `nbreg`：C 档（1.82e-04）—— **/lnalpha 参数化是对的，差在 bfgs 停步**
这一条是任务书点名的历史难点，所以拆成三层核：

1. **参数化正确**：westata 估 statsmodels 的 `alpha`（`Var = mu + alpha*mu²`），再用
   delta method `J = 1/alpha` 转成 `ln(alpha)`，**值和方差都一起转**。实测
   `alpha(westata) = 0.493656...` vs `e(alpha) = 0.493656...`，相对差与 `/lnalpha`
   完全同级。**不存在单位错配**（这正是历史上最容易错的地方）。
2. **`/lnalpha` 的 SE 也对**：SE 相对差 4.59e-06，比系数本身还小。
3. **残差归因**：同一条 statsmodels 似然，`bfgs` 把 `gtol` 从默认 `1e-5` 收到 `1e-10`，
   系数从 **C 档 1.82e-04 降到 B 档 1.50e-08**，`ll` 本来就已经一致到 1e-16。
   **归因：优化器停步准则（statsmodels bfgs 默认 `gtol=1e-5`，westata 没有下传更紧的
   值），不是似然、不是参数化、不是 delta method。**
   到不了 A 档的原因也说清楚：bfgs 按梯度范数停，收到 `gtol=1e-12` 也只到 1e-8；
   statsmodels 的 `newton` 在这个似然上直接崩（alpha 方向 Hessian 不定）。

> 数据是故意造得"真有过度分散"的（真 `alpha=0.5`）。顺带记录一个 Stata 行为：
> 如果数据其实是 Poisson（真 alpha→0），`nbreg` 会把 `lnalpha` 压到 `-22.64`、
> `e(alpha)` 触底，`e(df_m)` 也跟着变成 1 —— 那种数据上比 `alpha` 没有意义。

### 13. `nbreg, dispersion(constant)`（NB1）：**未实现**，但**可复现**
`NegativeBinomial.fit()` 没有 `dispersion=` 参数，NB1/NB2 无法切换，抛 `TypeError`。
为了区分"没实现"和"不能实现"，本文件**手写了一份 NB1 参考似然**
（`Var = mu*(1+delta)`，`size = mu/delta`）：它把 Stata 的 NB1 系数复现到 **2.20e-08**
（B 档，SE 5.38e-07），`delta` = 0.8157014790 vs Stata `e(delta)` = 0.8157014827
（相对差 4.5e-09）。**结论：模型良定、公式可写，westata 是缺功能而非算法无解。**
（参考实现自己也是 bfgs 收在 `gtol=1e-10`，所以同样停在 B 档而非 A 档——
这不影响"能复现"的结论。）

### 14. `zip`：B/C 档；`e(p)` 是**真错**（自由度用错）
- 系数 2.46e-08、SE 6.76e-06、`e(V)` 全矩阵 5.35e-05 —— 都只是停步准则。
- **`e(p)` 差 1.15e+02（两个数量级以上）**：Stata 的 `e(p) = chi2.sf(e(chi2), e(rank)-1)`，
  **不是** `chi2.sf(e(chi2), e(df_m))`。实测 `chi2.sf(55.4118, 5) = 1.073973e-10`
  与 `e(p)` 吻合到 6.7e-16，而 `chi2.sf(55.4118, 2) = 9.278e-13`。（`df_m=2`、`rank=6`，
  所以 Stata 用的自由度是 5 —— 即"总参数个数减一"。）westata 用 `df_m`，所以打印的
  `Prob > chi2` 不是 Stata 那个数。**`zinb` 同病。**
  > 注：`logit/probit/poisson/nbreg/ologit/oprobit` 的 `e(p)` 用 `df_m` 是对的（实测到 1e-16），
  > 所以这是 zip/zinb（多方程）专有的约定差异，不是全局问题。

### 15. `zip` 不带 `inflate()`：**skip（两侧不是一个模型）**
Stata 直接 `r(198) option inflate() required`。westata 的
`ZeroInflatedPoisson(inflate=None)` 按 `counts.py` 的文档就是"退化成普通 Poisson"，
所以**两边没有可比对象**。任务书要求的"默认两步法"在 Stata 19.5 里根本不存在
（`zip` 强制写 `inflate()`），westata 也没有"两步法"这个概念。如实记为 skip。

### 16. `zinb`：**D 档（1.06e-03）**，同样是停步准则被平坦似然放大
紧容差（`gtol=1e-10`，用 westata 自己的 `_CountLikelihood`）把它从 1.06e-03 拉到
**9.96e-07**。主方程截距的 westata 值是 0.003 量级，所以 1e-6 的绝对误差在相对
口径下就很大。**归因：迭代收敛（bfgs `gtol=1e-5`），不达 A 档是因为西格玛方向病态。**

### 17. `ztp`：B 档（3.60e-08），**`e(ll_0)` 约定正确**
`e()` 标量 A 档（1.30e-15），系数 B 档。`e(ll_0)` 已单独用独立算法复核：
`sum(logpmf(y, mu=mean(y)) - logsf(0, mu))` 与 Stata 一致到 1.68e-16。说明 westata
对 `ztp` 采用"`mean(y)` 单参数 null + 保留截断"的约定，**和 Stata 一致**，
不是重新拟合一个常数项截断模型（那样会差到 246.25 vs 270.42 那种量级）。

### 18. `tnbreg`：C 档（1.08e-05），同 `nbreg`，但样本更小更敏感
系数 C 档、`e(V)` 8.11e-05。`e(alpha)` 相对差 4.36e-06。**归因：bfgs 停步准则。**

### 19. `ologit / oprobit`：**真错 —— cut-point 协方差的 delta method Jacobian 错了**
这是本次最重的数值错误。现象：

| 量 | Stata | westata | 正确 Jacobian 复算 |
|---|---|---|---|
| `cut1` SE | 0.09878516 | 0.09878512（**对**） | 0.09878516 |
| `cut2` SE | 0.08645192 | 0.10420782（**错**，+20.5%） | 0.08645192 |
| `cut3` SE | 0.11621549 | **0.32076898**（**错**，+176%） | 0.11621549 |
| 全 `e(V)` 最大相对差 | — | **2.09e+01（ologit） / 1.05e+01（oprobit）** | 1.01e-06 / 7.36e-06 |

**根因（精确定位到"转置"）**：statsmodels 的阈值参数是 `[c1, log(c2-c1), log(c3-c2)]`，
正确关系是 `cut_k = theta[0] + Σ_{j≤k} exp(theta_j)`，因此
`J[k,0] = 1`（每个 cut 都继承 theta[0]）、`J[k,j] = exp(theta_j)`（只影响 `k ≥ j`）。
而 `src/westata/estimators/discrete.py:396-398` 写的是
`J = eye(n_cut)` + `J[k, :k] = exp(theta[1:k+1])`。把两者按实测 theta 排出来：

```
westata J（错）              正确 J
[[1,        0,        0],    [[1,        0,        0],
 [1.467488, 1,        0],     [1,        1.467488, 0],
 [1.467488, 1.491588, 1]]     [1,        1.467488, 1.491588]]
```

**westata 的 J 恰好是正确 J 的转置** —— `exp()` 放在了本该是 1 的累积列上，
1 放在了本该是 `exp()` 的对角线上。这也是为什么 `cut1` 的 SE "碰巧"还对：
`J[0,0]` 在两个版本里都是 1。

**证据**：用正确 Jacobian 复算，`e(V)` 全矩阵（含斜率×cut、cut×cut 交割项）
复现 Stata 到 **1.01e-06（ologit） / 7.36e-06（oprobit）**，cut SE 到
**2.28e-10 / 3.33e-09**。同一个 BFGS 解，只换 Jacobian，误差从 2.1e+01 掉到 1.0e-06。
**点估计本身是对的**（一样的递归重建），错的只有协方差。

**是否影响结论/显著性**：严重影响。`cut2`/`cut3` 的 z 值和置信区间会被报成错的；
cut 与斜率的协方差错了，任何 `lincom`/`test` 用到 cut 的组合都会错。

### 20. `ologit/oprobit, vce(robust)`：**未实现**
`_Ordered.fit` 对非 `nonrobust` 一律抛 `ValueError`（源码里明说了 robust sandwich
还没接）。Stata 正常给出 robust 结果。有序模型的稳健 VCE 完全没有。

### 21. `predict` 面：两条 A、一条 B、三条崩/错
- **`logit predict, pr` / `, xb`：A 档**（2.34e-13 / 2.88e-12，已开 `set type double`）。
- `poisson predict, n`：B 档 2.41e-08，只是系数停步准则的传导。
- **`ztp predict, n`：E 档（8.68e-01）—— 语义不一致（真错）**。
  Stata 的 `predict, n` 在 `ztp` 后返回 `exp(xb)`（实测 `max|n - exp(xb)| == 0.0`，
  365 个观测全部相等）；westata 的 `fittedvalues` 返回**截断均值** `mu/Pr(y>0)`。
  两者是不同定义。注意 Stata 的 `predict, xb`/`n` 在这里同值，说明 Stata 不做截断修正，
  而 westata 做了——**问题不在数学，在于 `predict, n` 的定义没对齐**。
- **`zip predict, n`：崩溃（真错）**。
  `StataResults.predict` 的 `kind in ("pr","n")` 分支只认 `model_name in ("logit","ologit")`
  才走 logit 链接，然后对非 `poisson/nbreg` 的模型一律走 `_predict_binary`，
  执行 `exog @ params`。zip 的参数向量长 6、设计矩阵宽 3，于是
  `ValueError: matmul: ... (size 6 is different from 3)`。即使不崩，把零膨胀模型当
  二值模型套 sigmoid 也没有数学意义。
  **但 westata 的 `ZeroInflatedPoisson.fittedvalues`（模型层）是对的**：
  `(1-pi)*mu` 与 Stata `predict, n` 一致到 1.09e-09。所以**模型对，公开的 `predict`
  分发器错**。
- **`ologit predict, pr`：崩溃（真错）**。`OrderedLogit.fit()` 从没把
  `design_matrix=` 传给 `StataResults`，于是 `StataResults.predict()` 在
  `exog is None` 时抛 `ConsistencyError: the design matrix was not retained`。
  另外语义也不对：Stata 的 `ologit predict, pr` 是 `P(y = 观测到的类别)`，而
  westata 的分发器对 `ologit` 走的是二值 sigmoid，两者都不是同一个量。
- `ologit fittedvalues`：C 档 2.08e-05。westata 给的是"期望类别"
  `Σ k·P(y=k)`，Stata 没有单一的 predict 类型对应它，本文件用 4 个
  `predict, pr outcome(k)` 重建后比较——**这是可解释的定义差异**，
  残留 2e-5 来自系数/阈值的停步准则（oprobit 同理）。

### 22. `logit i.grpf` 因子：B 档（1.02e-07）
系数 1.02e-07、SE 2.06e-08。因子对齐（`1.grpf`/`2.grpf` ↔ `C(grpf)[T.1]`/`[T.2]`、
`_cons` ↔ `Intercept`）完全正常，无 `mismatched`。**归因：迭代收敛**（同一份牛顿解，
只是停步点不同；纯连续变量时是 A 档 1.09e-13，加了因子后 statsmodels 的 Newton
条件数变差）。**不影响结论。**

### 23. `e(df_m)` 把辅助参数算进去：**真错**
Stata：`nbreg` 的 `df_m=2`（`/lnalpha` 不算，另给 `k_aux=1`、`rank=4`）；
`ologit` 的 `df_m=2`（`cut1..cut3` 不算，另给 `k_aux=3`、`rank=5`）。
westata：`nbreg` 的 `_DiscreteBase._stata_scalars` 用 `df_m = len(params) - 1`，报 **3**；
`ologit/oprobit` 报 **4**，相对差 **100%**。
注意 `tnbreg` 是**对的**（`df_m=2`）—— 因为 `tnbreg` 走的是 `_CountBase`，
`_CountBase._model_df()` 只数主方程斜率；而 `nbreg` 走 `discrete.py` 的
`_DiscreteBase`，没有这层修正。**同一个 nb2 似然，两条路径给出两个 df_m**，
这是最清楚的 bug 定位。
连锁后果：`e(p)`、打印的 `LR chi2(df)` 标签、任何按 df 的检验全错。
注意 `counts.py` 的 `_CountBase._model_df` 对 zip/zinb/ztp **是对的**，
所以 bug 只出在 statsmodels-backed 那条路径上。

## 与旧结论的出入

旧证据（`tests/conformance/registry.yml`、`docs/GAP_REGISTER.md`、`conformance*.csv`）
本次**未作为证据引用**，下面只按"本次实测 vs 仓库里 `counts.py` 文档字符串的自述"对齐，
因为源码注释是对外宣称的一部分：

| 源码自述（`src/westata/estimators/counts.py` 文档） | 本次实测 | 结论 |
|---|---|---|
| "Robust/cluster VCEs are passed through to statsmodels' sandwich and are **not yet verified** against Stata `vce(robust)`" | 实测：Stata ML robust = 原始 sandwich × `N/(N-1)`，statsmodels HC0 无因子 / HC1 用 `N/(N-K)`，**两种都不对**；cluster 上 Stata 只要 `G/(G-1)`，statsmodels 多一个 `(N-1)/(N-K)` | **自述低估了问题**：不只是"未验证"，是"两边的约定系统性地不同" |
| "the default observed-information covariance is **verified** (ztp: .0350626/.04102 vs Stata .0350626/.04102)" | ztp 默认 SE 差 **4.05e-08**（B 档）—— 与"verified"相容（打印到 7 位确实一样） | 一致，但只能说 B 档，不能说 A 档 |
| `counts.py`："`e(df_m)` counts only the main-equation regressors beyond the constant…`zip y x1, inflate(x1)` prints `LR chi2(1)`" | zip/ztp 确实对；**但 nbreg/tnbreg/ologit/oprobit 全错**（`df_m` 多算） | **部分正确、未覆盖全家族** |
| `counts.py`："the reported coefficient is `/lnalpha` exactly as `nbreg`/`zinb`/`tnbreg` print it; `e(alpha)` is the exponentiated value" | 对 —— 但这条只对 `counts.py` 自己写的似然成立；`discrete.py` 的 `NegativeBinomial` 走 statsmodels，另外做了 delta method（也是对的） | 一致 |
| `discrete.py`："`e(r2_p)` is McFadden's pseudo R-squared" | 全部命令 A 档（最差 3.6e-10） | 一致 |
| `discrete.py`：`_to_cut_points` 注释"verified against StataNow 19.5, where cut1 matches exactly and cut2 = cut1 + exp(param) to 2e-5" | **点估计**确实对（10 位全等），但注释只验了点估计、**没验协方差**；`e(V)` 因此错 2.1e+01 | **注释给了虚假的安全感**：cut2 的*值*对，cut2 的 *SE* 错 |
| `discrete.py`：`_SUPPORTED_COV = ("nonrobust","HC0","HC1","cluster")` | `HC0/HC1` 对 Logit/Probit/NegativeBinomial 全崩；Poisson 的 `HC1` 静默等于 `HC0` | **声称支持 ≠ 能用** |
| `discrete.py`："Robust VCEs are delegated to statsmodels' own machinery" | 委派本身不是错，但 statsmodels 的因子不是 Stata 的因子 | 需明确写成"近似" |

## 未覆盖

1. **`tpoisson`（一般截断，`ll()`/`ul()` 都指定）**：Stata 要求样本满足截断，
   本文件的数据只有 `ztp`（`ll(1)`）能直接对上；`ll(2)`/`ul(6)` 需要另外造满足条件的
   样本。westata 侧 `TruncatedPoisson(lower=2, upper=6)` 的代码路径存在但本次**未实测**。
   建议下一个迭代补（一条命令 + 一条 VCE 即可）。
2. **权重（`fweight`/`pweight`/`aweight`）**：离散/计数家族全部无权重 API。
   记录为 error 后没有继续尝试绕路（BRIEF 第 6 条禁止为跑通改参数）。
3. **`estat gof` / `lroc` / `linktest` / `margins` 等 `estat` 后估计**：不属于"估计命令
   数值"，本领域只记了 `poisson` 的 `r(chi2_d)` 作为 deviance 的替代来源。
4. **`ml` 的 `technique(bhhh)`/`difficult` 等选项**：Stata 侧未穷举算法变体；
   westata 的优化器也不可换（`_optimizer` 是类属性，不是 `fit()` 参数）。
5. **多分类 `mlogit`**：任务书未要求（只要求 `ologit/oprobit`），westata 也没有对应类。
6. **`e(V)` 在 zip/zinb 下的稳健版**：跑过 `vce(robust)` 但只比了系数/SE，
   没逐元素比 `e(V)`（默认版已逐元素比过）。
7. **`i.grpf` 与其它命令（probit/poisson/nbreg）的因子组合**：只测了 `logit` 一条；
   因子编码由共享的 `build_design` 处理，风险低但未穷举。

## 复现

```powershell
cd D:\myCodeProject\statapy
.venv\Scripts\python.exe retest/domains/discrete/run.py
```

产物：`results.jsonl`（逐条含 `stata_b/westata_b/stata_se/westata_se/e(V)` 原始数组）、
`summary.json`（计数与分级）。目录里的 `probes/probe_*.py` 是归因用的探针脚本，
单独可跑，不属于交付口径的一部分；`run.py` 不依赖它们。
