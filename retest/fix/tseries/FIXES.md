# tseries 修复报告

修复对象：`src/westata/estimators/tseries.py`（唯一改动源文件）
测试：`tests/test_tseries.py`（唯一改动测试文件）
验证：StataNow 19.5 MP 现场会话（`retest/lib/stata_probe.py`，未改动），数据与
`retest/domains/tseries/run.py` 完全一致（`SEED=20261024`, `N=300`, `N_GARCH=600`, GARCH seed 777）。

复现命令：

```bat
cd D:\myCodeProject\statapy
.venv\Scripts\python.exe retest\fix\tseries\run_all.py
```

单个脚本也可单独跑：`repro_v1v2_var.py` / `repro_v3_dfuller.py` / `repro_v4_vecrank.py` /
`repro_v5_arima_n.py` / `repro_v6_arch.py`。

## 逐项结果

| # | 缺陷 | 改前相对差 | 改后相对差 | Stata 命令 | 判定 |
|---|---|---|---|---|---|
| V1a | VAR 标准误（每方程 ML 自由度） | 5.055e-03 / 8.496e-03 | **2.760e-16 / 5.019e-16** | `var y1 y2, lags(1)` / `lags(1/2)` | 已修复 A |
| V1b | VAR `e(Sigma)` | 1.014e-02 / 1.706e-02 | **4.770e-15 / 1.677e-13** | 同上 | 已修复 A |
| V1c | VAR `e(V)` 全矩阵 | 1.014e-02 / 1.706e-02 | **6.760e-15 / 1.757e-13** | 同上 | 已修复 A |
| V2 | Granger 默认返回 F，Stata 是 Wald chi2 | F(1,592)=20.68 vs chi2=20.89；chi2 未归一 1.003e-02 / 1.678e-02 | **1.570e-15 / 1.689e-15**（p 值 1.54e-14） | `vargranger` | 已修复 A |
| V3 | `dfuller` 打印的 1/5/10% 临界值 | 2.362e-03～4.133e-03 | **≤1.295e-16**（6 组；3 组为 0.0） | `dfuller ..., lags(2) [trend|noconstant]` 等 6 条 | 已修复 A |
| V3b | `pperron` 打印临界值（同一张表） | 2.3e-03 | **0.0**（4 组） | `pperron y_rw [trend|lags(3)]` | 已修复 A |
| V4 | `vecrank` 临界值表（Stata Johansen 1995 vs Osterwald-Lenum） | trace 5% 差 2.168e-02（constant）/ 7.542e-02（none）/ 2.714e-02（trend）；max-eig 同量级 | **0.000e+00**（constant / none / trend × trace / max-eig，6 组全 0） | `vecrank yc1 yc2, lags(2) trend(...) max` | 已修复 A |
| V4b | `vecrank` 选阶在边界数据上翻 | 8/8 组边界数据翻（statsmodels 表选 1，Stata 选 2） | **0/8 翻**（与 Stata 一致） | `vecrank yb1 yb2, lags(2) trend(constant) max` | 已修复（结论级） |
| V5 | `arima` 的 `e(N)` 在 `d=1` 差 1 | 300 vs 299（差 d+D·s） | **精确相等**（6 个模型：300/299/299/298/298/296 全中） | `arima y_rw, arima(0,1,1)` 等 6 条 | 已修复 A |
| V6a | `arch` 标准误：继承 arch 包 robust，Stata 是 OPG | 2.782e-01 / 1.017e-01 | **1.511e-04 / 3.165e-04** | `arch yg, arch(1) garch(1)` / `arch(1)` | 已修复 B（目标 ≤1e-2） |
| V6b | `arch` presample h₀ 不同导致目标函数不同 | 参数 1.068e-02 | **1.780e-04**（GARCH(1,1)）/ 4.433e-04（ARCH(1)，原本就是 C 档） | 同上 | 已修复 B |

档位：A ≤1e-9，B ≤1e-6，C ≤1e-3（BRIEF §2.2）。

## 每项的根因与改法

### V1 VAR 的每方程 ML 自由度

**根因**：`VARResults` 直接透传 statsmodels 的 `cov_params()` / `sigma_u`，二者除以
`T − k_eq`（`k_eq` = **每方程**回归元个数 = `k_ar*neqs + k_trend`）。Stata 的 `var` 除以 `T`。
两者比值是 `(T−k_eq)/T`：VAR(1) 上 `T=299, k_eq=3` → `0.9899665551839465`（SE 比
`sqrt(...)=0.9949706303122452`，与 REPORT 实测值逐位相同）；VAR(2) 上 `T=298, k_eq=5`
→ `0.9832214765100671`。

**改法**：`VARResults` 增加 `k_per_equation` / `ml_factor`，`sigma_u`、`cov_params` 乘
`ml_factor`；`bse`、`stata_cov_params`、`stata_params()`、`stata_param_names` 按 **Stata 的
`e(b)` 顺序**（每方程内"变量优先、滞后在后"）重排。`fit(ml=False)` 关闭该因子——实测它
**精确等于 Stata 的 `var ..., dfk`**（SE 1.373e-16、`e(Sigma)` 4.722e-15），不是我方臆造的选项。

> 踩坑记录（写进代码注释）：用**全系统**参数个数（6 或 10）去除会得到 0.9–9.6 的假差异。

### V2 Granger 默认统计量

**根因**：`granger_causality()` 硬写 `kind="f"`。Stata `vargranger` 打印 **Wald chi2**
（`df = #causing × lags`）。还有第二层：Wald 形式用 ML 残差协方差，statsmodels 用无偏的，
所以还要乘 `T/(T−k_eq)`。

**改法**：默认走 `kind="wald"`，统计量乘 `1/ml_factor`，p 值与临界值用 `chi2(df)` 重算，
返回 `CausalityTestResults(method="Wald")`。`statistic="F"` 保留 statsmodels 原样（`df2 = k(T−k_eq)`），
并在 docstring 说明它**不是** `vargranger` 的量。实测 VAR(1) `18.1080077246501` vs Stata
`18.10800772465012`。

### V3 dfuller 打印的临界值

**根因**：statsmodels 用 MacKinnon (2010) 响应面；Stata 的 `dfuller.ado::GetCrit` **硬编码**
一张 6 点表（N = 25/50/100/250/500/∞）按 N **线性插值**。`pperron.ado::GetCrit` 的 Z(t)
三行是同一张表。

**改法**：把 `dfuller.ado` 的三个 case（`n`/`c`/`ct`）× 三行 × 6 点原样抄进
`_DFULLER_TABLE`，`_dfuller_critical()` 复刻插值分支；`unitroot(test="adf")` 与
`test="pp"` 都改用它。**统计量与 MacKinnon p 未动**（实测改动 <1e-12）。

`repro_v3_dfuller.py` 同时断言"模块里的表 == ado 里的表"（54 个数逐个比对），所以改表会立刻失败。

### V4 vecrank 临界值表

**根因**：`VECM.johansen()` 的临界值取自 `statsmodels.coint_johansen().cvt/.cvm`
（Osterwald-Lenum）；Stata 的 `_vecgetcv.ado::_VECGETCV_` 硬编码 **Johansen (1995)** 的
11×5 矩阵。K−r=1 且含常数时 5% 值：Stata **3.76** vs statsmodels **3.8415**；max-eig
K−r=2：**14.07** vs **14.2639**。

**改法**：抄入 `_VECRANK_TRACE_CV95/99` 与 `_VECRANK_MAXEIG_CV95/99`（列序按 ado 注释：
none / rconstant / constant / rtrend / trend），`_vecrank_critical()` 按行 `K−r` 取表。
**统计量公式一行未动**。新增 `JohansenResult.select_rank(statistic=..., level=...)`，
把 `vecrank` 的打星规则（最小的使统计量 ≤ 5% 临界值的 r）显式化。

**边界数据**：扫 seed 得到 8 组 `trace(r=1) ∈ (3.76, 3.8415)` 的协整对（seed 5/128/211/237/248/287/320/470）。
用 statsmodels 表 → 8 组全部选 rank 1；换 Stata 表 → 8 组全部选 rank 2，与 `vecrank` 一致。
其中 seed=248 是现场 Stata 证据：`trace = [95.9056, 3.8369]`，临界值 `[15.41, 3.76]`，r=1 无星号。

**顺带修掉的一处**（同一函数内、属 V4 的"只换表不换公式"必读项）：`det_order=1` 时原实现
只在残差里去了趋势、**没把趋势列放进差分回归**，于是 `vecrank ..., trend(trend)` 给
92.1075 而 Stata 是 92.2870。补上趋势列后 92.2870（max-eig 90.6871）逐位一致。这一改
只影响 `det_order=1`；`det_order=-1/0` 的统计量不变（`repro_v4` 里三档都打印了统计量相对差）。

**副作用（明确声明）**：Stata 的 ado 只硬编码 95% 与 99%，因此 `trace_critical[:, 0]`
（90% 列）现在是 `nan`，不再回落到另一张表的值。`to_frame()` 只用第 1 列（5%），不受影响。

### V5 arima 的 e(N)

**根因**：`ARIMA.fit()` 用 `result.nobs`（= 传入序列长度）填 `e["N"]`。Stata 的
`e(N) = tmax − tmin + 1 − d − D·s`，即估计样本量。

**改法**：改用 `result.nobs_effective`（statsmodels 的 `loglikelihood_burn` 正是这个定义），
填 `e["N"]` 与 `df_resid`；`StataResults.nobs` 仍保持残差长度（构造器的硬约束）。
六个模型实测：`d=0`→300、`d=1`→299（两组）、`d=2`→298（两组）、SARIMA(0,1,1,4)→296，全部精确相等。
并现场验证 Stata 的 `estat ic` 的 BIC 惩罚确实用这个 N（用 `r(S)` 反解 N 得 299.0000000000028）。

### V6 arch 的 OPG 与 presample h₀

**根因（两条独立）**：

1. `GARCH.fit()` **从不传 `cov_type`**，于是继承 arch 包的默认 `"robust"`
   （Bollerslev-Wooldridge sandwich）。Stata 报 `e(vcetype)="OPG"`（BHHH）。
   实测 westata 旧 SE 与 `arch.fit(cov_type="robust")` 的相对差 **0.000e+00**，
   而 arch 的 `"classic"`（逆 Hessian）与 Stata 也差 1.5e-01 / 3.0e-01 —— 三种估计量互不相等。
2. 方差递推需要 `h₀`：Stata 用 `e(archi) = mean(去均值残差²)`，arch 包用 0.94 指数加权 backcast。
   **不同的 h₀ = 不同的目标函数**，所以最优点不同（μ 相对差 1.068e-02）。

**改法**：

1. `fit(cov_type="opg")` 成为默认。arch 包没有 OPG，因此 `_opg_covariance()` 自己算
   BHHH：`inv(Σ_t s_t s_t')`，`s_t` 是**逐观测**对数似然对参数的中心差分
   （步长 `1e-5·max(|θ|,1e-3)`；固定步长 1e-6 会丢 4 位）。
2. `fit(presample="stata")` 成为默认：把 `h₀ = mean((y−μ̂)²)` 迭代到不动点，并把最终的
   h₀ 暴露为 `e["archi"]`。`presample="arch"` 保留包的行为，`cov_type="robust"/"classic"` 也保留。

**结果**：SE 2.782e-01 → **1.511e-04**（GARCH(1,1)）/ 1.017e-01 → **3.165e-04**（ARCH(1)）；
参数 1.068e-02 → **1.780e-04**；`e(archi)` 相对差 1.55e-07。

**为什么没到 1e-9（如实记录，未掩盖）**：见下节。

## 未修 / 未达 A 的

| 项 | 实测残差 | 原因 | 建议下一步 |
|---|---|---|---|
| V6 参数与 SE 停在 ~2e-4 | 参数 1.780e-04，SE 1.511e-04 | **不是公式差异**。`repro_v6_arch.py` 用 Stata 的递推在两组参数点上求值：Stata 自己报的 `e(ll)=-824.3779300106`，比它自己的递推在**它自己参数点**上给出的 `-824.3780305032` **高 1.00e-04**；westata 在另一参数点上给出 `-824.3780321`，与 Stata 的递推只差 1.6e-06。即：两边都在同一个最优附近，Stata 停得更早、且它报的 ll 与朴素递推略有 1e-4 的口径差（Stata 首期条件观测处理，`e(condobs)=0` 的确切含义仍未完全分解）。 | 若 Lead 要求 ≤1e-6，需要拿到 Stata `arch` 的首期处理细节（可能要 `set trace on` 逐行读 `arch_*` mata），成本高、收益是第 4 位小数。**建议保持 B 档并保留这段归因**。 |
| `vecrank` 的 `trend(rconstant)` / `trend(rtrend)` 统计量 | 未实现 | 这两种设定把确定性项**限制在协整关系内**，是另一个估计问题；`_johansen_statistics` 只做无约束版。REPORT 也未测。 | 若要支持，需要按 Johansen 的 "case 2/case 4" 做受限回归，工作量独立成项。 |
| `vecrank` 的 90% 临界值列 | `nan` | Stata 的 ado 只有 95%/99% 两套矩阵。填别的表就是"把错数换成另一个错数"。 | 保持 `nan`（已文档化，见 `_vecrank_critical` docstring）；`select_rank(level=0.10)` 显式 `ValueError`。打印层用 `to_stata_table()` 渲染 `None`。**Lead 已裁决：同意保留 NaN** |
| `arima` 的 `cov_type` | 仍只接受 `"nonrobust"` | 属接口缺口（旧 NOTES §未覆盖 6），不在这 6 个缺陷里。 | 另开任务；Stata 侧 `vce(robust)/vce(opg)` 可测。 |

> **V6 判定的一句话**：目标 **SE ≤1e-2 已达成**（2.782e-01 → 1.511e-04），**1e-9 不可达且原因已知**
> （Stata 自己报的 `e(ll)` 比它自己的方差递推在它自己参数点上高 1.00e-04，westata 在另一参数点
> 与 Stata 递推只差 1.6e-06）。**Lead 已裁决：接受 B 档。**

## 测试变更

`tests/test_tseries.py` —— 新增 13 条断言 Stata 数值的回归测试（每条注释里写明命令、数据、捕获日期）：

| 测试 | 旧断言 | 新断言 | 依据（Stata 捕获，2026-10-24） |
|---|---|---|---|
| `test_var_granger_default_is_the_stata_wald_chi2_not_f` | 只断言 `outcome is not None`（旧默认返回 F 也算通过） | 默认 `method=="Wald"`、`df==1`；`chi2 == F·T/(T−k_eq)`（rel 1e-12）；`statistic="F"` 仍是 F 且 `df2 = 2(T−k_eq)` | `var y1 y2, lags(1)` + `vargranger`：chi2 = 18.10800772465012 / 33.61050577858357，df = 1 |
| `test_var_reports_stata_ml_degrees_of_freedom` | （无） | `k_per_equation==3`、`ml_factor==(T−3)/T`、`bse` 比 == `sqrt((T−k_eq)/T)`、`e(Sigma)` 比 == `(T−k_eq)/T` | `var y1 y2, lags(1)`：`e(N)=299`，SE 比 0.9949706303122452 |
| `test_var_ml_false_reproduces_stata_dfk` | （无） | `ml=False` 的 `e(Sigma)` 与 `bse` 等于 Stata `var ..., dfk`（rtol 1e-13）；系数与默认完全相同 | `var y1 y2, lags(1)` / `var y1 y2, lags(1) dfk`（600 obs, seed 7）：`e(Sigma)` = 0.8549362620862754 … vs 0.8592396325330185 … |
| `test_johansen_uses_stat_critical_values_for_the_constant_case` | （无） | 5% trace cv = 15.41 / 3.76；max-eig = 14.07 / 3.76；99% = 20.04；90% 列是 `nan` | `vecrank ..., lags(2) trend(constant) max` |
| `test_select_rank_returns_the_rank_and_refuses_untabulated_levels` | （无） | 默认用 5% 列；返回值是 r（∈{0,1,2}）而非索引；`level=0.10` 抛 `ValueError` | 按 Lead 裁决 4 的接口契约；斜率来自 Stata 表 |
| `test_to_stata_table_renders_missing_critical_values_as_none` | （无） | 5% 数值与 `to_frame()` 相同；非有限值渲染为 `None`，有限值仍是 `float`；`to_frame()` 仍是 `float64` + `np.isnan` 可用 | 按 Lead 裁决 1②；数值层来自 Stata 表 |
| `test_johansen_rank_decision_matches_vecrank_on_borderline_data` | （无） | `3.76 < trace[1] < 3.8415` 且 `select_rank()==2`（trace 与 max-eig 都是） | `vecrank yb1 yb2, lags(2) trend(constant)`：trace = 95.9056 / 3.8369，cv 15.41 / 3.76 |
| `test_johansen_trend_specification_adds_the_trend_to_the_regression` | （无） | `det_order=1` 时 trace[0] ≈ 92.2870、max-eig[0] ≈ 90.6871、cv = 18.17 | `vecrank yc1 yc2, lags(2) trend(trend) max` |
| `test_adf_printed_critical_values_follow_dfuller_ado_table` | （无） | `regression="c"`：N=297 时 cv = −3.45624 / −2.87812 / −2.57；`"ct"`：−3.98812 / −3.42812；统计量与 p 仍等于 statsmodels 的原值 | `dfuller y_rw, lags(2)` / `lags(2) trend` |
| `test_adf_critical_values_interpolate_like_the_ado` | （无） | N=299 → −3.45608；N=295 → −3.4564（同一线性段的两个点） | `dfuller y_rw, lags(0)` / `dfuller y_ar, lags(4)` |
| `test_arima_reports_statas_estimation_sample_size` | （无） | `d=1`→`e["N"]==299` 且 `nobs==300`；SARIMA(0,1,1,4)→296；`d=0`→==残差长度 | `arima y_rw, arima(0,1,1)`；`arima y_s, ... sarima(0,1,1,4)` |
| `test_garch_defaults_are_statas_opg_and_presample` | `test_garch_recovers_persistence` 只查 α+β≈0.96 | 默认 `e["vcetype"]=="OPG"`；与 `cov_type="robust"` 的 SE 差 >1e-2；`e["archi"] == mean((y−μ̂)²)`（rel 1e-9）；与 `classic` 也差 >1e-2 | `arch yg, arch(1) garch(1)`：`e(vcetype)="OPG"`，`e(archi)=0.9690947048974677` |
| `test_garch_rejects_unknown_cov_type_and_presample` | （无） | 未知 `cov_type` / `presample` 抛 `ValueError` | —（接口纪律） |

**没有放宽任何旧断言**；原有 21 条全部保留并通过。`tests/conformance/test_stata_vecrank.py`
（其中一条断言 `trace_critical[0,1]` 是 15.41 的注释此前就是凭 Stata 打印值写的）与
`tests/conformance/test_stata_e3_ext.py` 的 VAR/vec 条目未改，全部通过。

## Lead 裁决与落实（4 条，已全部执行）

| # | 裁决 | 落实位置 |
|---|---|---|
| 1 | vecrank 90% 列保留 `nan`；①docstring 写明"Stata 不提供该列，不是缺失计算" ②打印层用 `None` 渲染而非 `nan` 字符串 | ① `_vecrank_critical()` docstring 新增五句说明；② 新增 `JohansenResult.to_frame(render_missing=False)` + `to_stata_table()`，数值层仍是 `np.nan`（`dtype=float64`），渲染层才转 `None`。测试：`test_to_stata_table_renders_missing_critical_values_as_none` |
| 2 | V6 留 B 档；docstring 写清残差；FIXES 标注"≤1e-2 达成、1e-9 不可达且已知原因" | `GARCH` class docstring 改写：给出 1.8e-4/1.5e-4 实测残差、Stata `e(ll)` 比自身递推高 1.00e-04、westata 距 Stata 递推 1.6e-06 三点证据；本文件「未修 / 未达 A」表加粗结论行 |
| 3 | `cov_params` 默认 ML 批准；docstring 第一行写明，并加 `ml=False == dfk` 的测试 | `VARResults.cov_params` docstring 首行即"默认与 Stata `var` 一致（每方程 ML 自由度）；要 `var ..., dfk` 用小样本版本请 `fit(ml=False)`"；测试：`test_var_ml_false_reproduces_stata_dfk`（Stata 全精度常量，rtol 1e-13） |
| 4 | `select_rank()` 批准；docstring 写明用 5% 临界值、返回 `r` 而非索引；`to_frame()` 旁加注释指向它 | `select_rank` docstring 重写（明确"返回 `r` 的值，不是索引"、"5% 与 `vecrank` 默认一致"）；`to_frame` docstring 首段指向 `select_rank`。测试：`test_select_rank_returns_the_rank_and_refuses_untabulated_levels` |

## pytest 结果

**本组范围（稳定，可复现）**：

```text
.venv\Scripts\python.exe -m pytest tests/test_tseries.py -q
34 passed, 1 warning        # 21 旧 + 13 新；warning 是 kpss 既有的 p 值截断警告

.venv\Scripts\python.exe -m pytest tests/conformance/test_stata_vecrank.py -q
9 passed

.venv\Scripts\python.exe -m pytest tests/conformance/test_stata_e3_ext.py tests/conformance/test_stata_e_upgrade.py -q
17 passed, 1 warning
```

**全量 `pytest tests -q`（仅供参考，数字随其他组的并行改动变化）**：
本组最后一次全量观察到 `7 failed / 1364 passed`，失败全部落在
`tests/test_panel_nonlinear.py`(4)、`tests/conformance/test_stata_p9.py`(2)、
`tests/conformance/test_stata_e_upgrade.py`(1) —— 均属 `src/westata/estimators/panel*.py`
（panel 组）正在修改的范围。本组只改 `tseries.py` 与 `test_tseries.py`，从未写上述文件，
因此**全量的通过/失败数不作为本组的验收指标**，以 Lead 在各组收敛后统一重跑的
`pytest tests` 为准。

另注：`retest/domains/discrete/results.jsonl` 显示为 deleted，是 discrete 组跑 `Record()` 时
被清空重写所致，非本组所为（Lead 已确认由他在最后统一重跑恢复）。
