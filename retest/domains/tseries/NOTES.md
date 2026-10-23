# tseries 领域复测结论

**被测对象**：westata（本仓库 `src/westata`，editable 安装）
**对照基准**：StataNow 19.5 MP —— `retest/lib/stata_probe.py` 现场启动的独立会话，`c(stata_version)` 实测 = `19.5`
**证据来源**：仅本次实跑。未读取 `tests/conformance/**`、`registry.yml`、`conformance*.csv`、`docs/GAP_REGISTER.md`。
**复现**：`cd D:\myCodeProject\statapy && .venv\Scripts\python.exe retest/domains/tseries/run.py`
**数据**：全部自造、固定 seed。
- 主表 300 obs，`numpy.random.default_rng(20261024)`：`y_ar`（平稳 AR(1), φ=0.6）、`y_rw`（随机游走，增量为 MA(1) θ=0.5 → 真模型 ARIMA(0,1,1)）、`y_111`（差分服从 ARMA(1,1) 后累积 → 真模型 ARIMA(1,1,1)）、`y_s`（含季节 MA(1)@4 的平稳序列）、`y1/y2`（双变量 VAR(1)）、`yc1/yc2`（协整对，协整向量 (1, −1.5)，AR(1) 误差 φ=0.4）、`yb1/yb2`（边界协整对，seed=248，专门用来暴露选阶翻转）
- 波动率表 600 obs，`default_rng(777)`：GARCH(1,1) 真值 ω=0.05, α=0.10, β=0.85
- 送入后 `tsset t`，并断言 `st.nobs()==300`、`TSERIES` 所有列均存在
**判定口径**：A ≤1e-9 / B ≤1e-6 / C ≤1e-3 / D ≤1e-2 / E >1e-2；相对差 = `max(|westata−stata| / max(|stata|,1e-12))`，逐参数取最大；SE 用 `e(V)` 对角线开方（全精度，**不用**打印值）。

## 总计

**56 个对比点：pass 35 / fail 21 / error 0 / skip 0。** 最大参数相对差 **1.706e-02**（VAR(2) 的 `e(Sigma)`，已完全归因，见 §2.2）。

> **关于 `stata_probe.compare`（偏离共享工具的理由）**
> 时间序列命令的 `e(b)` 布局与 `regress` 不同：`arima`/`arch` 的 `e(b)` 里有**两行同名 `_cons`**（均值方程常数 vs 方差方程 σ/ω）；`var`/`vec` 的 `e(b)` 是 1×k 且名字按方程重复；`var, lags(1/2)` 的块内顺序是**变量优先**（`L.y1 L2.y1 L.y2 L2.y2 _cons`）；`dfuller`/`pperron`/`kpss`/`vecrank` 根本不产生估计对象。按名字对齐会错配或直接判 fail。
> 因此本领域改为**显式声明配对**（`extra["pairing"]`）、并按全精度向量比较；每条记录另附 `extra["per_param_rel"]` 与 `extra["worst_param"]` 可逐参数复核。共享工具一行未改。这个偏离本身就是一条发现（§2.7）。

---

## 覆盖矩阵

| # | Stata 命令 | westata 调用 | 数据 | 判定 | 系数相对差 | 标准误相对差 |
|---|---|---|---|---|---|---|
| 1 | `arima y_ar, arima(1,0,0)` | `ARIMA(y_ar,(1,0,0),trend="c")` | y_ar | fail / **D** | 2.735e-03 | 6.678e-05 (C) |
| 2 | `arima y_ar, arima(1,0,0) noconstant` | `trend="n"` | y_ar | fail / C | 1.987e-05 | 5.949e-05 (C) |
| 3 | `arima y_rw, arima(0,1,1)` | `trend="t"` | y_rw | fail / C | 6.350e-04 | 7.885e-05 (C) |
| 4 | `arima y_111, arima(1,1,1)` | `trend="t"` | y_111 | fail / **D** | 1.391e-03 | 4.188e-05 (C) |
| 5 | `arima y_s, arima(1,0,0) sarima(0,0,1,4)` | `seasonal_order=(0,0,1,4)` | y_s | fail / **D** | 5.461e-03 | 5.306e-05 (C) |
| 6 | 同上两条的 `e(N)` | — | y_rw / y_111 | fail | 299 vs 300 | — |
| 7 | `var y1 y2, lags(1)` | `VAR(Y,lags=1)` | y1 y2 | **pass / A** | 8.451e-14 | — |
| 8 | 同上，SE（原样） | — | — | fail / D | 8.451e-14 | 5.055e-03 |
| 9 | 同上，SE（每方程 ML 归一后） | — | — | **pass / A** | — | **2.899e-16** |
| 10 | 同上，`e(Sigma)`（原样） | — | — | fail / **E** | 1.014e-02 | — |
| 11 | 同上，`e(Sigma)`（归一后） | — | — | **pass / A** | 3.252e-15 | — |
| 12 | `vargranger`（VAR(1)） | `VARResults.granger_causality` | — | fail / **E** | 1.003e-02 | — |
| 13 | 同上，归一后 | — | — | **pass / A** | 2.057e-15 | — |
| 14 | `var y1 y2, lags(1/2)` | `VAR(Y,lags=2)` | y1 y2 | **pass / A** | 1.030e-14 | — |
| 15 | 同上，SE（原样） | — | — | fail / D | 1.030e-14 | 8.496e-03 |
| 16 | 同上，SE（归一后） | — | — | **pass / A** | — | **1.224e-16** |
| 17 | 同上，`e(Sigma)`（原样） | — | — | fail / **E** | 1.706e-02 | — |
| 18 | 同上，`e(Sigma)`（归一后） | — | — | **pass / A** | 1.084e-14 | — |
| 19 | `vargranger`（VAR(2)） | — | — | fail / **E** | 1.678e-02 | — |
| 20 | 同上，归一后 | — | — | **pass / A** | 2.485e-15 | — |
| 21 | `varsoc y1 y2, maxlag(3)` FPE | `VAR.select_order` | y1 y2 | pass / C | 2.297e-06 | — |
| 22 | 同上 AIC（去尺度常数） | — | — | pass / B | 7.581e-07 | — |
| 23 | 同上 BIC（去尺度常数） | — | — | pass / B | 8.467e-07 | — |
| 24 | 同上 HQIC（去尺度常数） | — | — | pass / B | 7.412e-07 | — |
| 25 | 同上 选阶（整数比较） | — | — | **pass** | AIC/BIC/HQIC/FPE 两侧全选 lag 1 | — |
| 26 | `vec yc1 yc2, rank(1) lags(2) trend(constant)` | `VECM(rank=1,lags=2,deterministic="co")` | yc1 yc2 | **pass / A** | 3.047e-10 | — |
| 27 | `vec ..., trend(rconstant)` | `deterministic="ci"` | yc1 yc2 | **pass / A** | 1.493e-10 | — |
| 28 | `vec ..., lags(1)` | `lags=1` | yc1 yc2 | **pass / A** | 4.210e-11 | — |
| 29 | `vecrank yc1 yc2, lags(2) trend(constant) max` trace | `.johansen(lags=2)` | yc1 yc2 | pass / C | 7.792e-06 | — |
| 30 | 同上 max-eigen | — | — | pass / C | 7.792e-06 | — |
| 31 | `vecrank ..., lags(1)` trace | `.johansen(lags=1)` | yc1 yc2 | pass / C | 5.430e-06 | — |
| 32 | 同上 max-eigen | — | — | pass / C | 5.430e-06 | — |
| 33 | `vecrank` 边界对，lags(2) trace | — | yb1 yb2 | **fail / C** | 1.005e-05 | **选阶 2 vs 1** |
| 34 | 同上 max-eigen | — | — | pass / C | 1.005e-05 | — |
| 35 | `dfuller y_rw, lags(2)` | `unitroot(test="adf",regression="c",maxlag=2)` | y_rw | **pass / A** | 1.179e-12 | — |
| 36 | 同上，打印的 1/5/10% 临界值 | — | — | fail / **D** | 2.362e-03 | — |
| 37 | `dfuller y_rw, lags(2) trend` | `regression="ct"` | y_rw | **pass / A** | 1.216e-13 | — |
| 38 | 同上，临界值 | — | — | fail / **D** | 1.853e-03 | — |
| 39 | `dfuller y_rw, lags(2) noconstant` | `regression="n"` | y_rw | **pass / A** | 1.461e-15 | — |
| 40 | 同上，临界值 | — | — | fail / **D** | 4.133e-03 | — |
| 41 | `dfuller y_ar, lags(4)` | `maxlag=4` | y_ar | **pass / A** | 0.000e+00 | — |
| 42 | 同上，临界值 | — | — | fail / **D** | 2.367e-03 | — |
| 43 | `dfuller y_rw, lags(0)` | `maxlag=0` | y_rw | **pass / A** | 7.203e-16 | — |
| 44 | 同上，临界值 | — | — | fail / **D** | 2.357e-03 | — |
| 45 | `pperron y_rw`（默认 NW 滞后） | `unitroot(test="pp")` | y_rw | **pass / A** | 2.055e-13 | — |
| 46 | `pperron y_rw, lags(3)` | `maxlag=3` | y_rw | **pass / A** | 1.786e-13 | — |
| 47 | `pperron y_rw, trend` | `regression="ct"` | y_rw | **pass / A** | 5.007e-14 | — |
| 48 | `pperron y_ar` | — | y_ar | **pass / A** | 1.561e-15 | — |
| 49 | `kpss y_ar, notrend maxlag(5)` | `test="kpss", regression="c", maxlag=5` | y_ar | pass / B | 6.097e-09 | — |
| 50 | `kpss y_ar, maxlag(5)` | `regression="ct"` | y_ar | pass / B | 4.497e-09 | — |
| 51 | `kpss y_rw, maxlag(5)` | `regression="ct"` | y_rw | pass / B | 8.244e-09 | — |
| 52 | `kpss y_rw, notrend maxlag(5)` | `regression="c"` | y_rw | pass / B | 6.740e-09 | — |
| 53 | `arch yg, arch(1) garch(1)` | `GARCH(yg,p=1,q=1)` | yg | fail / **E** | 1.068e-02 | 2.782e-01 (E) |
| 54 | `arch yg, arch(1)` | `GARCH(yg,p=1,q=0)` | yg | fail / C | 4.856e-04 | 1.017e-01 (E) |
| 55 | 同上，SE 归因（robust vs classic/OPG） | — | yg | **pass** | westata ≡ arch robust（相对差 0.0） | — |
| 56 | `arch ... estat ic`：`e(ll)`/`e(aic)`/`e(bic)` | — | yg | fail（随 #53） | ll 1.640e-05 / aic 1.633e-05 / bic 1.615e-05 | — |

---

## 逐项说明

### 2.1 ARIMA（#1–#6）：ll/aic/bic 到 1e-9，系数在第 3–4 位分叉 —— 已证明是**同一似然面上的停机点**

| 模型 | 最大系数相对差 | 出现在 | 绝对差 | ll 相对差 | aic 相对差 | bic 相对差 | SE 相对差 |
|---|---|---|---|---|---|---|---|
| ARIMA(1,0,0)+c | 2.735e-03 | 常数项 | 1.92e-05 | 3.79e-10 | 3.76e-10 | 3.71e-10 | 6.68e-05 |
| ARIMA(1,0,0) noc | 1.987e-05 | /sigma | 1.93e-05 | 1.49e-10 | 1.48e-10 | 1.47e-10 | 5.95e-05 |
| ARIMA(0,1,1)+drift | 6.350e-04 | 漂移项 | 1.03e-04 | 5.56e-10 | 5.53e-10 | 5.46e-10 | 7.89e-05 |
| ARIMA(1,1,1)+drift | 1.391e-03 | 漂移项 | 3.98e-05 | 1.45e-09 | 1.44e-09 | 1.41e-09 | 4.19e-05 |
| SARIMA(1,0,0)(0,0,1,4) | 5.461e-03 | 常数项 | 1.26e-04 | 1.43e-09 | 1.41e-09 | 1.39e-09 | 5.31e-05 |

**归因（可复算，不是推测）**：

1. **不是似然函数不同。** `run.py` 调 statsmodels 自己的 `model.loglike()`，在**Stata 报的参数点**上求值：ARIMA(1,0,0) 得 `-417.888634159056`，与 Stata 的 `e(ll) = -417.888634159056` **逐位相同**；westata 最优点是 `-417.88863400082`，高出 1.582e-07。五个模型全部如此（`extra["ll_surface_gap"]` ∈ [6.2e-08, 6.2e-07]）。→ **两者最大化的是同一个面**，statsmodels 停在略靠近真最大的一点。
2. **不是标准误公式不同。** SE 相对差 4.2e-05～7.9e-05，与"Stata 用 OPG（`e(vcetype)="OPG"`，`technique: bhhh bfgs`）、statsmodels 用解析 Hessian"一致；且 ll 一致说明这不是估计量差异。
3. **是平坦方向被放大。** 相对差最大的是**常数项/漂移项**——正是 ll 最平坦方向的参数。ARIMA(1,0,0) 的常数项绝对差 1.92e-05，其 z 从 1.71 动到 ~1.71（<1e-4），**不改变任何显著性判断**。ARIMA(1,1,1) 与 SARIMA 上到 D 档（1.4e-3 / 5.5e-3），但 ll 相对差仍 ≤1.5e-9，属同一机制。

**季节项（任务必测项 2）：westata 支持 `seasonal_order`，且与 Stata 的 `sarima` 对得上。**
Stata 名 `L.ar` / `L.ma`，westata 名 `ar.L1` / `ma.S.L4`，按位置配对后系数相对差 5.461e-03、ll 相对差 1.427e-09。支持成立，数值差异机制同上。

**#6 `e(N)` 约定差异（新增发现）**：`d=1` 时 Stata `e(N)=299`，westata 报 `300`（= 传入序列长度），相对差 3.344e-03。ll/aic/bic 仍一致（同一批差分新息），但**任何用 `e(N)` 归一化的下游计算都会偏 1/300**。这是报告约定，不是估计差异。

### 2.2 VAR / vargranger / varsoc（#7–#25）：系数 A 档；SE / `e(Sigma)` / Granger 的 E 档差异**只是每方程 ML 自由度**

**系数（A 档）**：VAR(1) **8.451e-14**，VAR(2) **1.030e-14**。
配对是**逐名字翻译**：Stata `e(b)` 是 1×k 行、按方程分块、块内按"变量优先、滞后在后"（`lags(1/2)` 实测为 `L.y1 L2.y1 L.y2 L2.y2 _cons`）；statsmodels 是 `vec(B)` 列主序。`map_stata_var_names()` 把每个 Stata 名字翻译成显式 statsmodels 下标，`extra["pairing"]` 逐条记录。**能到 1e-14 本身就证明翻译正确**，不是假定。

**SE / `e(Sigma)` / Granger 的差异全部是同一个自由度约定**：
- Stata 的 `var` 用**每方程** ML 残差方差（除以 T），statsmodels 的 `cov_params()`/`sigma_u` 用无偏版本（除以 `T − k_eq`，`k_eq` = 每方程回归元个数）。
- 实测比值与 `sqrt((T−k_eq)/T)` **每一位都相同**：VAR(1) `0.9949706303122452`（T=299, k_eq=3，6 个系数全一致）；VAR(2) `0.9915752500491665`（T=298, k_eq=5，10 个系数全一致）。
- 归一后：SE **2.899e-16 / 1.224e-16 → A**；`e(Sigma)` **3.252e-15 / 1.084e-14 → A**（sigma 比值 0.989966555183948 对 (T−k_eq)/T = 0.9899665551839465；VAR(2) 0.9832214765100616 对 0.9832214765100671）。
- **所以"E 档"完全由约定造成，不是算错。**
- ⚠️ **坑（我第一版踩了）**：拿**全系统**参数个数（6 或 10）去除，会得到 0.9–9.6 的假差异。必须用**每方程**的 `k_eq`。

**Granger 因果（#12/#13/#19/#20）—— westata 有，但默认接口返回的统计量与 Stata 不同**：
- 任务允许 skip，**但不需要**：`VARResults.granger_causality()` 存在且可算。
- 但它返回 statsmodels `kind="f"` 的 **F 统计量**（VAR(1)：F=20.6804，df=(1, 592)），而 Stata `vargranger` 报 **Wald chi2**（20.8900，df=1）。直接对比就是拿 F 比 chi2。
- 改用 `kind="wald"`：chi2 `20.6804` vs `20.8900`，比值 `1.0101351351351362`，恰好等于 `T/(T−k_eq) = 1.010135135135135`（T=299, k_eq=3）。VAR(2)：比值 `1.0170648464163805` 对 `T/(T-k_eq) = 1.0170648464163823`。
- 归一后 **2.057e-15 / 2.485e-15 → A 档**。
- **p 值**：VAR(1) 4.864e-06 vs 5.427e-06（相对差 1.16e-01），VAR(2) 4.600e-06 vs 5.653e-06（2.29e-01）、2.027e-09 vs 2.836e-09（3.99e-01）。**相对差看着大，是因为 p 已极小**；四个 p 全部远低于 0.001，**显著性结论不变**。
- **结论**：`granger_causality()` 有两个可操作缺陷 —— (a) 默认 kind 与 Stata 不同（F vs Wald chi2），(b) 残差方差用无偏自由度，与 Stata 的统计量差 `T/(T−k_eq)`。本数据上不翻结论，边界情形会翻。

**`varsoc` 选阶（#21–#25）：数值尺度不同，但选阶一致**
- Stata 的 AIC/BIC/HQIC 是 `(−2·ll + penalty)/T`（**水平尺度**，含正态化常数 `neqs·(ln2π+1) = 5.675754132818691`），statsmodels 是 `ln|Sigma_ml| + penalty/T`。**原样比会看到 6.27 vs 0.596**，看起来灾难性，其实是同一量平移。
- 去掉该常数后：AIC **7.581e-07**、BIC **8.467e-07**、HQIC **7.412e-07** → **B 档**（Stata 只打 5 位小数，这是可达下限）。
- FPE 定义里没有正态化常数，可直接比：**2.297e-06 → C**（同受打印精度限制：Stata `1.00221` vs westata `1.0022076982`）。
- **选阶完全一致**：AIC/BIC/HQIC/FPE 四个准则两侧都选 **lag 1**（整数比较，容差 0）。
- ⚠️ **陷阱**：Stata 列名是 **`SBIC`**，westata/statsmodels 是 **`BIC`**。我第一版按 `BIC` 去取 Stata 列 → 得到"缺失"的假失败。必须显式映射。

### 2.3 VECM（`vec`，#26–#28）：A 档，alpha / beta / ll 全对齐

| 设定 | alpha（Stata） | beta（Stata，含常数行） | ll（Stata / westata） | 最大相对差 |
|---|---|---|---|---|
| `trend(constant) lags(2)` | `[0.316463475974, 0.702572964961]` | `[1, −1.499553214200, −2.176261714337]` | 85.475133278552 / 85.475133263580 | **3.047e-10** |
| `trend(rconstant) lags(2)` | `[0.349434310767, 0.724583202637]` | `[1, −1.499553811419, −1.993397736583]` | 84.082390137213 / 84.082390140285 | **1.493e-10** |
| `trend(constant) lags(1)` | `[−0.952948354021, −0.124940385095]` | `[1, −1.499501026011, −1.839337667614]` | 84.826039207265 / 84.826039204929 | **4.210e-11** |

**确定性设定映射是实测出来的**（`vec` 的 `trend()` 五个取值全跑过）：
- `trend(constant)`（Stata 默认）↔ statsmodels **`co`**：常数在协整关系**外**（unrestricted）。
- `trend(rconstant)` ↔ **`ci`**：常数被限制在协整关系**内**。
- 另三个：`trend(none)` ↔ `nc`、`trend(trend)` ↔ `lo`、`trend(rtrend)` ↔ `li`（ll 差 ≤2e-3，未列入矩阵）。
- GETTING IT WRONG MATTERS：`co` 与 `ci` 的 alpha 差 0.316 vs 0.349（约 10%），选错就是另一个模型。

**beta 的第三行只存在于 Stata**：`e(beta)` 末位是无约束常数的规范化值；statsmodels 把它放进确定性项并用不同规范化，**因此只比前两行（两个斜率）**，`alpha·beta_const` 记在 `extra` 供手工还原漂移。
**滞后映射**：Stata `lags(#)` 数的是 VAR 滞后，`k_ar_diff = # − 1`。这条在 `vec` 与 `vecrank` 上分别验证过。

### 2.4 vecrank（#29–#34）：统计量 C 档（打印精度地板），但**临界值不同，选阶会翻**

- 统计量：`trace r=0` 92.1075 vs 92.10750911，`r=1` 1.384 vs 1.38398922（**7.792e-06**）；lags(1) 时 145.6968 / 1.3049（**5.430e-06**）；max-eigen 同量级。
- **为什么只能到 C**：`[TS] vecrank` **只留 `r(trendnumber)` / `r(ttext)`**，statistic 和临界值都不在 `r()`/`e()` 里，唯一数据源是**打印表（4 位小数）**。以 92.1075 计，4 位小数 ⇒ ~1e-5 相对精度，C 是这条路径的地板，**不是实现差异**。这是本领域唯一用打印输出当证据的地方，已在代码与 `extra` 标注。
- **临界值确实不同**：Stata 打 Johansen (1995) 表（r=1 时 **3.76**），statsmodels 用 Osterwald-Lenum（**3.8415**），差 2.2%。
- **后果：选阶可以翻。** 我扫 seed 造了一对协整序列（`yb1/yb2`，seed=248），其 r=1 的 trace 统计量 ≈ **3.8369**，正落在 3.76 与 3.8415 **之间**：

  | | trace(r=0) | trace(r=1) | 5% 临界值(r=1) | 选阶 |
  |---|---|---|---|---|
  | Stata | 95.9056 | 3.8369 | **3.76** | 3.8369 > 3.76 → 拒绝 → **rank 2** |
  | westata | 95.90558322 | 3.83693856 | **3.8415** | 3.8369 ≤ 3.8415 → 接受 → **rank 1** |

  同一份数据、同一个统计量（相对差 **1.005e-05**），**结论不同**。这是本次最有价值的结论级差异之一。选阶规则本身两侧**完全相同**（最小的使 statistic ≤ 5% 临界值的 r，同一函数 `select_rank()` 应用于双方），差异 100% 来自临界值表。
- 常规协整对（`yc1/yc2`）两侧都选 rank 1（`rank_decision_agrees=true`），无分歧。

### 2.5 单位根：dfuller / pperron 统计量与 p 值 A 档；**dfuller 打印临界值不同，已完全归因**

**统计量与 p 值**：

| 命令 | Stata Z(t) | westata Z(t) | 相对差 | p（Stata / westata） |
|---|---|---|---|---|
| `dfuller y_rw, lags(2)` | 0.3506012025850289 | 0.3506012025854422 | 1.179e-12 | 0.9795505857606045 / 0.9795505857606216 |
| `dfuller y_rw, lags(2) trend` | −1.848114142925153 | −1.8481141429253778 | 1.216e-13 | 0.6811091473177239 / 0.6811091473176115 |
| `dfuller y_rw, lags(2) noconstant` | 1.367836775420685 | 1.367836775420683 | 1.461e-15 | **Stata 无 p** |
| `dfuller y_ar, lags(4)` | −6.397424253725925 | −6.397424253725925 | **0.000e+00** | 2.03449475472e-08 / 2.03449475472e-08 |
| `dfuller y_rw, lags(0)` | 1.233071737453916 | 1.2330717374539168 | 7.203e-16 | 0.9962065990535565 / 同 |
| `pperron y_rw`（NW=5） | 0.4445865349487736 | 0.4445865349486822 | 2.055e-13 | 0.983087891438 / 同 |
| `pperron y_rw, lags(3)` | 0.5326207862777599 | 0.5326207862776647 | 1.786e-13 | 0.985831570512 / 同 |
| `pperron y_rw, trend` | −1.747100856507414 | −1.7471008565075015 | 5.007e-14 | 0.72968976187 / 同 |
| `pperron y_ar` | −9.105653076770674 | −9.105653076770688 | 1.561e-15 | 0.0 / 0.0 |

- `pperron` 能到 A 档的关键：westata 主动套用 **Stata 的 Newey-West 带宽规则** `floor(4·(n/100)^(2/9))`，默认 n=299 时两侧都取 5。不套用（让 arch 自选）时 Z(t) 差 ~2e-2 —— 这条经验本次被独立复现。
- `dfuller ..., noconstant` 时 **Stata 不返回 `r(p)`**（`dfuller.ado` 只在有 MacKinnon 曲面的 case 设 `r(p)`），westata 会返回（0.95665）。故这一条只比统计量，p 记在 `extra`。

**打印的 1% / 5% / 10% 临界值：D 档，且 100% 归因到表本身**：

| 命令 | r(N) | Stata 打印 | statsmodels | 相对差 | 照抄 dfuller.ado 复算值 | 复算相对差 |
|---|---|---|---|---|---|---|
| `lags(2)` | 297 | −3.45624 / −2.87812 / −2.57 | −3.45256 / −2.87132 / −2.57198 | **2.362e-03** | −3.45624 / −2.87812 / −2.57 | **1.285e-16** |
| `lags(2) trend` | 297 | −3.98812 / −3.42812 / −3.13 | −3.98958 / −3.42538 / −3.13580 | **1.853e-03** | −3.98812 / −3.42812 / −3.13 | **1.295e-16** |
| `lags(2) noconstant` | 297 | −2.58 / −1.95 / −1.62 | −2.57331 / −1.94194 / −1.61596 | **4.133e-03** | −2.58 / −1.95 / −1.62 | **0.0** |
| `y_ar lags(4)` | 295 | −3.4564 / −2.8782 / −2.57 | −3.45271 / −2.87139 / −2.57202 | **2.367e-03** | −3.4564 / −2.8782 / −2.57 | **0.0** |
| `lags(0)` | 299 | −3.45608 / −2.87804 / −2.57 | −3.45241 / −2.87126 / −2.57195 | **2.357e-03** | −3.45608 / −2.87804 / −2.57 | **0.0** |

**证据**：`dfuller.ado` 的 `GetCrit` **硬编码**一张 6 点表（N = 25/50/100/250/500/∞），按 N **线性插值**。`run.py` 把这张表和插值规则照抄成 `stata_dfuller_cv()`，从实测 `r(N)` 复算 → **相对差 ≤1.3e-16**（机器精度）。
也就是说：差距**不是**检验不同、**不是**统计量不同，而是 Stata 用旧表+线性插值、statsmodels 用 MacKinnon (2010) 响应面。**p 值一致（≤1e-12）恰恰证明两者都出自同一个 MacKinnon 曲面**，所以这只影响"用户拿打印临界值肉眼比"的场景，不影响任何 p 值或结论。指纹验证：N 从 295→297→299 变化时 Stata 的 cv 随之微动（−3.4564 → −3.45624 → −3.45608），正是硬编码插值的特征。

**kpss（#49–#52）—— 三个必须先讲清的前提**
1. **`kpss` 不在 StataNow 19.5 里**：`which kpss` → `command kpss not found as either built-in or ado-file`。本次用 `ssc install kpss, replace` 装了 **SSC 包 `kpss` v1.2.3（C.F. Baum, 13may2018）**，装后指向 `C:\Users\zjl\ado\plus\k\kpss.ado`。**所以比的是第三方包，不是官方命令** —— 与 Lead 汇总时必须标明。
2. **原假设方向相反，且与 dfuller/pperron 相反**：`kpss` 默认 H0 = **趋势平稳**，`notrend` 才是 **水平平稳**；而 `dfuller`/`pperron` 的 H0 是**单位根**。实测确认映射：`notrend ↔ regression="c"`、默认 `↔ regression="ct"`。**搞反就把所有结论颠倒。**
3. **p 值不可比，故不比较**：Baum 的 `kpss` **只给临界值、不给 p**；statsmodels 插值出 p 并**在 0.1 处截断**，且只在统计量落出查表范围时才截断。实测：

   | 数据 | Stata 统计量（lag 5） | westata | 相对差 | westata p |
   |---|---|---|---|---|
   | y_ar, notrend（水平平稳 H0） | 0.068212770644898 | 0.06821277022902042 | 6.097e-09 | 0.1（**截断**，有 InterpolationWarning） |
   | y_ar, 默认（趋势平稳 H0） | 0.0664433248632958 | 0.06644332516209592 | 4.497e-09 | 0.1（**截断**） |
   | y_rw, 默认（趋势平稳 H0） | 0.2703623699282585 | 0.2703623721571065 | 8.244e-09 | 0.01（未截断） |
   | y_rw, notrend（水平平稳 H0） | 3.981726948354534 | 3.981726975190205 | 6.740e-09 | 0.01（未截断） |

   四个**临界值在两侧完全相同**（水平：0.347/0.463/0.574/0.739；趋势：0.119/0.146/0.176/0.216，相对差 0.0）。统计量 B 而非 A 源于求和次序，1e-9 相对 / ~1e-10 绝对，不影响任何判定。
   带宽必须**人为对齐**：statsmodels 的 `nlags="auto"` 会选 8，Stata 用 Schwert 规则选 5（`r(maxlag)=5`）。不对齐则完全不同。

### 2.6 ARCH / GARCH（#53–#56）：已知分歧被独立复现，**两个独立成因**都量化到可复算

**参数（600 obs，GARCH(1,1) 真值 ω=0.05, α=0.10, β=0.85）**：

| 参数 | Stata | westata | 相对差 | 绝对差 |
|---|---|---|---|---|
| μ（`_cons` ↔ `mu`） | −0.038559286118 | −0.038147532424 | **1.068e-02** | 4.118e-04 |
| ω（方差方程 `_cons` ↔ `omega`） | 0.058185818505 | 0.057708836452 | 8.201e-03 | 4.770e-04 |
| α（`L.arch` ↔ `alpha[1]`） | 0.102107636138 | 0.101628194149 | 4.697e-03 | 4.794e-04 |
| β（`L.garch` ↔ `beta[1]`） | 0.838357327224 | 0.839182353004 | 9.841e-04 | 8.250e-04 |

整体最大 **1.068e-02 → E 档**（越过 1e-2）。`arch yg, arch(1)` 时最大 **4.856e-04 → C 档**（μ 2.222e-05 / ω 3.697e-05 / α 1.045e-05）。

**成因 (a)：标准误 —— cov_type 约定不同（SE 差 2.78e-01 的全部原因）**
- Stata 的 `arch` 明确报 `e(vcetype)="OPG"`；westata 的 `GARCH.fit()` 调 `model.fit(disp="off")`，**从不传 `cov_type`**，于是继承 arch 包的默认 `"robust"`（Bollerslev-Wooldridge）。
- **证据**：westata 报的 SE `[0.03576651, 0.02515056, 0.02740584, 0.04158110]` 与 `arch` 包 `cov_type="robust"` 的 SE **逐位相同（相对差 0.0）**。
- Stata SE `[0.03740327, 0.03264898, 0.03725869, 0.05761018]`；arch `cov_type="classic"` 给 `[0.03648679, 0.02850537, 0.03176655, 0.04878139]`（与 westata 相对差 **1.476e-01**）。→ **OPG 既不等于 robust 也不等于 inverse-Hessian**，换 classic 也复现不了 Stata。
- SE 相对差 **2.782e-01**（GARCH(1,1)）、**1.017e-01**（ARCH(1)）。这个量级足以让 z 统计量跨过显著性边界，是最需要向用户告警的一条。

**成因 (b)：似然/点估计 —— presample h₀ 不同，两侧最大化的不是同一个目标**
GARCH 方差递推 `h_t = ω + α·ε²_{t−1} + β·h_{t−1}` 需要 `h_0`。Stata 用 `e(archi) = 0.9690947048974677`（= 去均值后残差平方的均值），arch 包用指数加权 backcast。**决定性证据（交叉验证，不是"收敛没收敛"的猜测）**：

| 目标函数（h₀ 来源） \ 参数点 | Stata 参数 | westata 参数 |
|---|---|---|
| **Stata 的 h₀**（= e(archi)） | **−824.3780305031632** ← Stata 报的 ll 就在这里附近 | −824.3783118143474 |
| **arch 的 backcast** | −824.3647119597724 | **−824.3644064194082** ← westata 报的 ll |

→ **在各自的目标函数下各自都更优**（`each_side_better_under_own_objective` 两项均为 `true`）。若只是停机点差异，同一目标函数下不可能出现这种交叉。**所以这是两个略微不同的目标函数**，`objective_level_gap = 1.331854e-02`（在 Stata 参数点上，arch 的 h₀ 比 Stata 的 h₀ 高这么多）。

**h₀ 的搜索验证**：把 h₀ 当自由参数扫，`e(archi)` 给出 ll `−824.37803050`，与 Stata 报的 `−824.37793001` 差 **1.005e-04**；`var(y)` 差 −1.297e-04、`mean(y²)` +3.884e-04、`omega/(1−α−β)` +1.532e-03 —— **e(archi) 是候选中最近的**（`closest_presample="stata e(archi)"`）。剩余 1e-4 来自 Stata 对首期/条件观测的处理（`e(condobs)=0`），**未能进一步分解** —— 这是本领域唯一没分解到 1e-9 的环节，如实记录。
参数影响：ω 最敏感（吸收 presample 水平），μ 次之，α/β 最小；β 的绝对差 8.25e-04 / SE 0.0576 ≈ 0.014 个 SE，**α、β 两侧都显著，结论不变**。

**`e(ll)` / `e(aic)` / `e(bic)`（任务必测项 11）**：Stata `ll=−824.377930010551` vs westata `−824.364406419408`（相对差 **1.640e-05**，绝对差 1.352e-02）；`aic` 1656.755860021102 vs 1656.728812838816（**1.633e-05**）；`bic` 1674.343578641966 vs 1674.316531459681（**1.615e-05**）。
⚠️ **Stata 的 `arch` 不存 `e(aic)/e(bic)`**（`arima` 也一样）—— 直接读 `e(aic)` 得 missing，必须 `estat ic` 从 `r(S)` 全精度取（`r(S)` 列序 `[N, ll0, ll, df, AIC, BIC]`）。

### 2.7 顺带发现（对下游使用者最有用）

1. **`var ..., lags(2)` 在 Stata 里不是 VAR(2)。** `lags()` 收**数字列表**，`lags(2)` = 只用第 2 阶滞后；VAR(2) 必须写 `lags(1/2)`。实测（`_verify_trap` 复现）：
   - `var y1 y2, lags(2)` 的 `e(b)` 名字是 `L2.y1 L2.y2 _cons`（×2 方程），**共 6 个系数**；
   - westata `VAR(lags=2)` 是 `(1+2·2)×2 = 10` 个系数（`L1.*` 与 `L2.*` 都在）；
   - 两侧**参数个数都不一样，按位置比毫无意义**；硬按前 6 个比会得到 1.9e+02 这种数。
   - 对照组 `var y1 y2, lags(1/2)` 的名字是 `L.y1 L2.y1 L.y2 L2.y2 _cons`（×2），共 10 个，与 westata `lags=2` 一一对应，系数相对差 **1.030e-14**。
   → westata 的 `lags=2` 语义是 VAR(2)，**只有 Stata `lags(1/#)` 才对得上**。这解释了为什么"系数差 10 倍/百倍"类报告要格外小心：先确认模型参数个数一致。
2. **`e(b)` 布局必须逐名字验证，不能按块假定。** 逐名字翻译后 1e-14；按块假定则可能"看起来对但有偏移"。本领域所有 `var`/`vec`/`arch` 条目都记 `extra["pairing"]` + `extra["per_param_rel"]`，可逐条复核。
3. **双行同名 `_cons`**：`arima` 与 `arch` 的 `e(b)` 里有两行都叫 `_cons`（均值方程常数 vs 方差方程 σ/ω）。任何按名字对齐的工具（含 `stata_probe.compare`）都会取错行。本领域按位置配对并把规则写入 `extra["pairing"]`。
4. **`e(Sigma)`/`e(V)` 的 ML 归一化**：Stata 的 `var` 用**每方程** `T`，statsmodels 用 `T − k_eq`。用全系统 `k` 会得到 0.9–9.6 的假差异。同理 Granger 的 Wald 统计量要乘 `T/(T−k_eq)`。
5. **`e(N)` 在差分模型上差 d**：`d=1` 时 Stata 报 299，westata 报 300。
6. **`kpss` 不在 Stata 19.5 里**，且其 H0 与 `dfuller`/`pperron` **相反**；`notrend` 关掉趋势项后是水平平稳；statsmodels 的 p 会在表边界被截断到 0.1，**不可直接引用**。
7. **`dfuller` 的 `noconstant` 情形不返回 `r(p)`**，不要当 0 或 missing 处理。
8. **`arima`/`arch` 不存 `e(aic)/e(bic)`**，须 `estat ic` → `r(S)`。
9. **接口缺口**：`ARIMA.fit(cov_type=...)` 只接受 `"nonrobust"`（其余 `ValueError`），`GARCH.fit()` 根本不接受 `cov_type`；Stata 侧两者都支持 `vce(robust)`/`vce(opg)`。用户想复现 Stata 的 OPG 或做稳健推断时无路可走。

---

## 与旧结论的出入

本次**未读取** `registry.yml` / `GAP_REGISTER.md` / `conformance*.csv`（BRIEF §0 要求）。下表对照的是**源码 docstring 内嵌的旧结论**（`src/westata/estimators/tseries.py`）——它同样是被检验对象：

| 旧表述（源码 docstring） | 本次实测 | 判定 |
|---|---|---|
| `ARIMA.fit`："剩余差距是停机点差异，不是公式差异；在 Stata 参数处 statsmodels 的 ll 更低，说明 statsmodels 停得略靠近真实最大" | **独立复现并证实**：在 Stata 参数点 statsmodels ll = −417.888634159056（= Stata 的 ll），westata 最优点高 1.582e-07。结论**正确** | **一致**（本次给了可复算证据） |
| `_johansen_statistics`："在我方数据上每个滞后阶都与 StataNow 19.5 匹配到全部打印位数" | 统计量确实到打印精度（7.792e-06 = 4 位小数地板），**成立**；但"匹配打印位数 ⇒ 可互换"**不成立**——临界值不同会让选阶翻转（3.8369 例） | **部分更正** |
| `VECM.johansen`："`lags(2)` 匹配到 195.9673；`lags(1)` 时 statsmodels 287.2485 vs Stata 288.3450，0.4% 差异，rank 判定在 lags(1) 不可互换" | 本次换数据（yc1/yc2）：**lags(2) 与 lags(1) 都是 C 档（7.792e-06 / 5.430e-06），没有出现 0.4% 的边界差异**。旧描述的 0.4% 差异在本次数据上**未复现**；若真实存在，应限于特定数据/`k_ar_diff=0` 边界，本领域**未能独立验证** | **未证实**（旧值可能是真的，但本次数据不支持"lags(1) 必然差 0.4%"）。不过旧注释"rank 判定不可互换"的**警告方向是对的**——真正让选阶翻的是**临界值表**（3.76 vs 3.8415），与滞后阶无关 |
| `GARCH`："arch 包估计同一模型但默认优化器与 rescaling 不同，故显式暴露收敛状态" | **归因需修正**：`rescale` 已被 westata 显式关掉（`rescale=False`），**不是 rescaling**。真正两个成因是 (a) SE 从 OPG→robust（westata 与 arch robust 逐位相同）；(b) presample h₀ 不同导致**目标函数不同**（交叉验证：各自在各自目标下更优）。"优化器不同"也不是主因——两侧是**不同目标**，不是同一目标的不同搜索 | **需修正** |
| `unitroot`："`pperron` 让 arch 自选带宽得 −1.994881 vs Stata −1.953（2.1e-2），套用 `floor(4(n/100)^(2/9))` 后达 1.1e-4" | **独立复现**：套用规则后 Z(t) 相对差 **2.055e-13**（比旧记录还好两个量级，因为本次用全精度 `r(Zt)` 而非打印值）。结论**成立** | **一致且更强** |
| `var`："`select_order` 返回全部准则，避免静默选 AIC" | 功能成立，但**数值尺度与 Stata 差常数** `neqs·(ln2π+1)`，直接拿 `select_order` 对 Stata 的 `varsoc` 表会误判。选阶结论一致（都选 lag 1） | **需补充**：应文档化尺度差 |
| （旧代码未提及） | **新发现**：`var` 的 SE/`e(Sigma)`/Granger 与 Stata 差**每方程**自由度因子 `(T−k_eq)/T`；`arima`/`arch` 不存 `e(aic)/e(bic)`；`e(N)` 在差分模型上差 d；`var, lags(#)` 是 numlist 陷阱；`kpss` 需外部安装且 H0 相反；`dfuller` 临界值是硬编码粗表 | **新增** |

---

## 未覆盖

1. **`arch` 的 EGARCH / GJR-GARCH / t 分布**：Stata 侧 `arch yg, egarch(1)` 报 `garch(), egarch(), sdgarch(), and pgarch() options require that other`（需与 `arch()` 同时给出）；`arch yg, arch(1) garch(1) distribution(t)` 报 `flat log likelihood encountered`（该数据上 t 分布无法收敛）。**未测** → "EGARCH 是否一致"**无结论**。westata 的 `vol="EGARCH"`、`o=1`、`dist="t"` 路径**未与 Stata 对照**。
2. **`arch` 的 `vce(robust)`**：Stata 能跑（`e(vcetype)="Semirobust"`），但 **westata 的 `GARCH.fit()` 不接受 `cov_type`**（签名只有 `strict`/`maxiter`），无法选择估计量。**记为接口缺口**，未做数值对比。
3. **`arch` 的 `archm`（ARCH-in-mean）**：westata 的 `GARCH(mean=...)` 只接受 `"Constant"`。**未测**。
4. **`vecrank` 的 `trend(none)` / `trend(rtrend)`**：VECM 侧跑过（ll 对得上），但 `vecrank` 侧只测 `trend(constant)`。其余确定性设定下的 Johansen 统计量**未测**。
5. **`vec` 的 `rank(2)`**：只测 `rank(1)`（任务要求）。rank=2 的规范化与 beta 解释不同，**未测**。
6. **ARIMA 的 `vce(robust)` / `vce(opg)`**：westata 的 `ARIMA.fit()` 只接受 `cov_type="nonrobust"`，其余直接 `ValueError`；Stata 支持。**记为接口缺口**，未做数值对比。
7. **`dfgls`**：Stata 有，westata 无对应实现；`ersle`/`zandrews` 两侧都没有。**未测**。
8. **`vecrank` 的 max-eigen 选阶**：只比了统计量（C 档），**没有**像 trace 那样单独造边界数据验证 max-eigen 选阶翻转。原理上同样会翻（14.07 vs 14.2639），**未实测**。
9. **`varsoc` 的 LR 检验**：表里有 `LR/df/p` 三列，本次只比 LL / 信息准则 / 选阶，**LR 统计量未单独对比**。
10. **多变量（3+ 方程）VAR**：只测 2 方程。方程数增加时 `k_eq` 与 `e(Sigma)` 维度都变，**未测**。
11. **`arch` 的 presample 剩余 1e-4 差**：如 §2.6，`e(archi)` 是最接近的 h₀，但仍有 −1.005e-04 的 ll 残差未能分解到 Stata 的条件观测处理细节（`e(condobs)=0` 的确切含义）。**未完全分解**。
12. **`kpss` 用的是 SSC 第三方包（v1.2.3, Baum）而非 Stata 官方命令**（19.5 未内置）。若未来 Stata 内置 `kpss`，需重新对照。
