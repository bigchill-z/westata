# linear-iv 修复报告（BRIEF §4.1 的 L1–L6）

环境：`D:\myCodeProject\statapy`，`.venv\Scripts\python.exe`（3.10.11），
StataNow **19.5 (MP)**，对比工具 `retest/lib/stata_probe.py`（只读）。
所有"Stata"数字都是本脚本内新起的 Stata 会话实测，未用 fixture / registry。
捕获日期：**2026-10-24**。

复现命令（cwd = 仓库根）：

```bat
.venv\Scripts\python.exe retest\fix\linear-iv\repro_l1_iv_hc_labels.py
.venv\Scripts\python.exe retest\fix\linear-iv\repro_l2_liml_cluster.py
.venv\Scripts\python.exe retest\fix\linear-iv\repro_l3_gmm_cluster.py
.venv\Scripts\python.exe retest\fix\linear-iv\repro_l4_ols_e_f.py
.venv\Scripts\python.exe retest\fix\linear-iv\repro_l5_l6_ols_e.py
.venv\Scripts\python.exe retest\fix\linear-iv\repro_l6_iv_e.py
```

每条脚本都打印 `改前 / 改后 / Stata` 三列（改前一律由**当前代码重算**——
L1 用旧文档规定的映射、L3/L4 用修复前的估计量/公式——所以不是抄来的常数），
并在末尾输出 `PASS`/`FAIL`，退出码 = 失败数是否为 0。

---

## 逐项结果

| # | 缺陷 | 改前相对差 | 改后相对差 | Stata 命令 | 判定 |
|---|---|---|---|---|---|
| L1 | IV `vce(robust)` 标签：文档说 HC1，实为 HC0 | 1.09e-02（HC1 vs `vce(robust)`） | 5.4e-15（HC0 vs `vce(robust)`）；5.4e-15（HC1 vs `vce(robust) small`） | `ivregress 2sls y x1 (e1 = z1 z2), vce(robust)` / `… small` | 已修复 A |
| L2 | `IVLIML` 的 `vce(cluster)` 恒加 CR1（Stata 默认 CR0） | 4.53e-02（140×14）；2.58e-02（240×24） | 1.15e-14 / 1.27e-14 / 2.68e-15（三组数据） | `ivregress liml y x1 (e1 = z1 z2), vce(cluster gid)` | 已修复 A |
| L3 | `IVGMM` + `vce(cluster)`：Stata 连 `W` 一起聚类 | 点估计 **3.583e-02**（在 retest 的 12d 原始数据上复现，与报告 3.58e-2 逐位吻合）；SE 1.736e-01 | 系数 4.575e-16；SE 6.362e-16（同一数据）；另三组数据系数 6.7e-16 / 8.2e-16 / 1.4e-15，`e(V)` 全矩阵 1.3e-14 | `ivregress gmm yiv x1 (e1 = z1 z2), vce(cluster gid)` | 已修复 A（不是回退成 `UnsupportedError`） |
| L4 | `OLS.e['F']` 不随 `cov_type` 变 | nonrobust 0（本来对）；robust 5.497e-02；cluster 1.489e-01；pweight 1.279e-01；`noconstant` 与截距-only 改前是 **NaN**（Stata 分别 216.97 与 0） | 1.2e-15 / 4.6e-15 / 4.0e-15 / 1.4e-14（另 aweight/fweight/noconstant/noconstant-robust 也 ≤2.3e-15） | `regress mpg weight foreign[, vce(robust) / vce(cluster gid)]`、`[pweight=w]` | 已修复 A |
| L5 | `OLS.e['df_r']` 在 cluster 下存 `N-K` | 9.54（存 71，Stata 存 11） | 0（存 11 = G-1；`e(N_clust)=12`） | `regress mpg weight foreign, vce(cluster gid)` | 已修复 A |
| L6 | `e()` 覆盖缺口 `rss`/`mss`/`sum_w`/`N_clust`（+ `ll_0`） | 缺失（`get_scalar` 返回 None） | 0 / ≤6.1e-16（八种权重与 VCE 组合）；`e(ll_0)` 六种情形 rel = 0.0（`aweight` 1.81e-02，见"未修"） | `regress …`、`… [fweight=w]`、`… vce(cluster gid)` | 已修复 A |

**L1–L6 全部修到 A 档（≤1e-9），无一项降级到 B/C。**
L3 按 BRIEF 的"优先实现"路线完成，没有使用 `UnsupportedError` 回退。

---

## 每项的根因与改法

### L1（P1-13）IV `vce(robust)` = HC0、`vce(robust) small` = HC1

- **根因**：`src/westata/estimators/iv.py` 的模块 docstring 映射表把
  `HC1` 写成 "linearmodels `robust` with `debiased=True`" 并暗示它等于
  Stata 的 `vce(robust)`。Stata 的 `ivregress` 默认 robust **不加**
  `N/(N-K)`（HC0），只有 `small` 才加。实测比值恰为 `sqrt(N/(N-K))`
  = 1.01088961，与公式逐位吻合。代码本身（`_shared_vce` / `_backend_fit`）
  的分支是对的，**错的是对外承诺的语义**，因此按 BRIEF 要求
  （"不要改 `_canonical_cov_type` 的名字"）只修文档与 `_canonical_cov_type`
  的 docstring，并补断言 Stata 数值的回归测试。
- **改法**：
  - `iv.py` 模块 docstring 的映射表重写为
    `HC0 ↔ vce(robust)`、`HC1 ↔ vce(robust) small`，并明确写出
    "这与 OLS/`regress` 相反"；
  - `-- cov_type` 参数文档、`_canonical_cov_type` docstring 同步；
  - `_backend_fit` 的注释从"robust variants keep their own convention"
    改成三条明确分支（nonrobust/HC0/HC1/cluster）。
- **验证**：`repro_l1_iv_hc_labels.py`：`HC0` vs `vce(robust)` = 5.444e-15，
  `HC1` vs `vce(robust) small` = 5.386e-15；`HC1/HC0` = 1.01088961 = `sqrt(140/137)`。

### L2（P1-14）`IVLIML` 的 `vce(cluster)` 恒加 CR1

- **根因**：`iv.py::_backend_fit` 里 `else: debiased = True` 这一支同时吃掉了
  `HC1`、`cluster` 与 `HC2/HC3`。对 `cluster` 而言 `debiased=True` 就是
  linearmodels 的 CR1，于是 LIML 的聚类 SE 无条件乘了
  `G/(G-1) * (N-1)/(N-K)`。2SLS 走 `compute_cov` + `SSC(adj=False)`，所以没病。
- **改法**：把分支拆开——`cluster` 用 `debiased = bool(settings.adj)`，
  即默认 CR0、`ssc=SSC(adj=True)` 才是 Stata 的 `small`/CR1。LIML 与 2SLS
  因此共享同一个 VCE 默认。
- **验证**：`repro_l2_liml_cluster.py`，三组数据（140×14×2instr、140×14×3instr、
  240×24×2instr）：
  - 默认 vs Stata：1.149e-14 / 1.274e-14 / 2.680e-15；
  - `SSC(adj=True)` vs Stata `small`：1.153e-14 / 1.268e-14 / 2.732e-15；
  - Stata 的 `small/默认` 比值 = 1.045296408535（240×24 时 1.025808941563），与
    `sqrt(G/(G-1)·(N-1)/(N-K))` 逐位一致；2SLS 未被改坏（5.7e-15 / 6.7e-15 / 5.1e-15）。

### L3（P1-17）`IVGMM` + `vce(cluster)`：Stata 把 `W` 也聚类

- **根因**：`ivregress gmm` 的 `W` 由"该 `vce()` 蕴含的矩协方差"决定。
  `vce(cluster g)` 下 Stata 用**聚类** `S` 的逆做 `W`，因此**点估计**也随之变化
  （不只是方差）。westata 之前无论 `cov_type` 都用 linearmodels 默认的
  heteroskedastic `W`，属"估计量选择缺失"。
- **改法**（BRIEF 的"优先实现"路线）：新增
  `IVGMM._backend_kwargs(canonical, clusters)`，`cov_type="cluster"` 时返回
  `{"weight_type": "clustered", "clusters": ...}`，否则返回 `{}`（保持
  heteroskedastic 默认）。为此把 `IV2SLS._backend_model()` 改为接受
  `canonical`/`clusters` 并转发 `_backend_kwargs`（2SLS/LIML 的实现返回 `{}`，
  行为不变）。`_backend_fit` 同步传入。
- **验证**：`repro_l3_gmm_cluster.py` 先复现 retest 的 12d 原始数据
  （`retest_iv_panel()`，与 `retest/domains/linear_iv/run.py` 的 `make_panel` 逐字相同）：
  - **改前**（用 heteroskedastic `W`，即修复前的返回值）：系数相对差 **3.583e-02**、
    SE 1.736e-01 —— 与 `retest/REPORT.md` 的 3.58e-2 **逐位吻合**，确认修复方向对；
  - **改后**：系数 4.575e-16、SE 6.362e-16。
  再在三组数据上做完整核验：
  - 系数 6.709e-16 / 8.182e-16 / 1.403e-15；
  - SE 7.008e-16 / 2.174e-16 / 5.759e-16；
  - `e(V)` 全矩阵 1.320e-14 / 2.382e-15 / 5.632e-15（逐元素）；
  - `e(J)` 4.0e-16 / 2.35e-15 / 1.68e-15；
  - **反向对照**：用 heteroskedastic `W`（改前估计量）对同一条 Stata 结果，
    系数仍差 2.895e-02 / 2.028e-02 / 1.482e-02 —— 证明上面的 1e-15 确实来自
    `W` 换成了聚类版本，不是别的原因；
  - 非聚类 GMM 未被改坏：系数 1.04e-15 / 6.0e-16 / 4.4e-16。

### L4（P1-15）`OLS.e['F']` 不随 `cov_type` 变

- **根因**：`linear.py` 里 `fvalue = (r2/df_m)/((1-r2)/df_resid)` 是经典 R² 型 F，
  只在 nonrobust 下等于 Stata 的 `e(F)`。Stata 的 `e(F)` 是
  **用实际报告的 `e(V)` 对全部非常数系数做的 Wald/k**。
- **改法**：新增模块级 `_wald_f(params, cov, tested)`，用
  `b_s' V_ss⁻¹ b_s / df_m` 计算；`has_intercept` 时 `tested` 去掉常数项，
  `noconstant` 时包含全部列；截距-only 模型（`df_m == 0`）显式给 `0.0`
  （Stata `regress mpg` → `e(F) = 0`，不是缺）。
- **验证**：`repro_l4_ols_e_f.py`：nonrobust 1.222e-15、`vce(robust)` 4.621e-15、
  `vce(cluster gid)` 3.988e-15、pweight 1.428e-14、aweight 4.642e-15、
  fweight 2.140e-15、`noconstant` 2.620e-16、`noconstant vce(robust)` 2.288e-16；
  并打印"用 westata 自己的 V 复算的 Wald/k"以把"V 错"与"F 定义错"分开
  ——两者相对差完全相同，说明 V 一直是 A 档、错的只是定义。

### L5（P1-16）`OLS.e['df_r']` 在 cluster 下应为 `G-1`

- **根因**：`linear.py` 的 `e` 字典写入 `"df_r": df_resid`（`N-K`），而
  `StataResults(df_resid=df_t)` 已经在用 `G-1`。两个"自由度"概念静默分叉。
- **改法**：`e["df_r"] = df_t`（即 Stata 实际存的那个数：cluster 下 `G-1`，
  其他 `N-K`），并在 cluster 时额外写 `e["N_clust"]`。`e(rmse)` 仍用
  `N-K` 分母（Stata 就是这样：`e(df_r)=11` 但 `e(rmse)=sqrt(rss/(N-K))`）。
  `results.df_resid` 完全没动。
- **验证**：`repro_l5_l6_ols_e.py`：auto.dta + 12 组，Stata `e(df_r)=11`、
  `e(N_clust)=12`，westata 11 / 12；并断言 `rmse == sqrt(rss/(N-K))`、
  `e(df_r) != N-K`。

### L6（15a/15b/15c/15d/15e）`e()` 覆盖缺口

- **根因**：`linear.py` / `iv.py` 的 `e` 字典没有 `rss`/`mss`/`sum_w`/`N_clust`；
  IV 侧还缺 `kappa`/`k_endog`/`iterations`/`J`/`tss`/`rank_mb`。
- **改法**：
  - **OLS**：补 `rss`（`Σ w e²`）、`mss = tss − rss`（有常数时 tss 为去均值平方和，
    `noconstant` 时为 `Σ w y²`，与 Stata 一致）、`sum_w`（**原始**权重和：
    `aweight`/`pweight` 的估计权重被归一化到和为 N，必须用归一化前的向量才
    得到 185 而不是 74）、`N_clust`（cluster 时）。为此在 `OLS.__init__` 里
    新增 `_raw_weight_sum`。
  - **IV**：补 `rss`/`mss`/`tss`/`k_endog`/`iterations`；
    `_extra_estimator_scalars(ref, canonical)` 钩子：k-class 给 `kappa`，
    GMM 给 `J`（并**不**给 `kappa`，与 Stata 一致）；`vce(cluster)` 时给 `N_clust`；
    `ivregress 2sls` 在 `vce(robust|cluster)` 下额外给 `e(rank_mb)`（= `e(rank)`，
    LIML/GMM 从不给）。
  - **`e(ll_0)`**（BRIEF 允许留，但成本很低所以实现了）：Stata 的 `e(ll_0)` 就是
    "仅常数模型的高斯对数似然"，即用 `TSS/N` 代替 `RSS/N` 的同一个公式。
    实测（auto.dta）：nonrobust / `vce(robust)` / `vce(cluster gid)` 全是
    -234.39433764823468；`[pweight=w]` -238.86834919298698；`[fweight=w]`/`[iweight=w]`
    -597.1708729824675；`noconstant` **不存**（westata 也不给）。
    nonrobust/robust/cluster/pweight/fweight/iweight 全部 **rel = 0.0**（逐位）。
    **一个已知偏差**：`aweight` 下 Stata 的 `e(ll)` **和** `e(ll_0)` 都带同一个
    权重约定偏移（westata 的 `e(ll)` 差 2.160e-02、`e(ll_0)` 差 1.810e-02，
    绝对差都是 4.4044），这**不是本轮引入的**（`e(ll)` 一直如此），
    且 `e(ll) − e(ll_0)`（LR 统计量）仍然精确。已在代码注释与本文件标注。
- **验证**：`repro_l5_l6_ols_e.py`（OLS 八种情形）与 `repro_l6_iv_e.py`
  （`ivregress 2sls/liml/gmm` × `default/vce(cluster)/vce(robust)`）：
  Stata 存过的标量 westata 一个不缺，数值全部 ≤ 6.1e-16（fweight 的 rss 6.0e-16、
  LIML `kappa` 5.7e-15、GMM `J` 1.1e-15）。

### 顺带修正（同属 e() 语义，已用 Stata 验证）

- **`noconstant` 的 `r2_a`**：原实现恒用 `(N-1)/(N-K)`，Stata 无常数时用
  `N/(N-K)`。实测 `regress mpg weight foreign, noconstant` 的
  `e(r2_a) = 0.8537391937770055 = 1 − (1−0.8576921885397891)·74/72`，
  已改为 `(N − c)/df_resid`，`c = 1 if has_intercept else 0`。

---

## 测试变更

| 文件 | 旧断言 | 新断言 | 依据（Stata 捕获） |
|---|---|---|---|
| `tests/test_cluster_df.py` | `e(df_r) == N - K`（并在 docstring 里论证"Stata 不重新定义 e(df_r)"） | `e(df_r) == G - 1` 且 `e(df_r) == df_resid`；新增 `e(N_clust) == G`、`e(rmse)` 仍用 `N-K` | **旧断言是错的**。2026-10-24 实测 `regress mpg weight foreign, vce(cluster gid)`（`gen byte gid = mod(_n,12)`，G=12）：`ereturn list` → `e(N)=74, e(df_r)=11, e(N_clust)=12, e(rmse)=3.4070592856515844`。`11 = G-1 ≠ N-K = 71`。 |
| `tests/test_cluster_df.py` | 无 | 新增 `test_cluster_e_f_is_the_wald_statistic_of_that_vce` | Stata `e(F)` 在 cluster 下 = CR1 V 的 Wald/k（81.951329431322），与 nonrobust 的 69.74846262000308 明显不同 |
| `tests/test_iv.py` | `test_cluster_for_liml_matches_linearmodels` 断言 `debiased=True` | 默认对齐 `debiased=False`（CR0），并**额外**断言 `ssc=SSC(adj=True)` 对齐 `debiased=True`（`small`），两者比值 = `sqrt(G/(G-1)·(N-1)/(N-K))` | **旧断言是把 bug 固化**。实测 `ivregress liml …, vce(cluster gid)` 与 linearmodels `debiased=False` 差 1.1e-14；`debiased=True` 差 4.53e-02 |
| `tests/test_iv.py` | `test_cluster_adj_off_drops_g_over_g_minus_1` 断言 `SSC(adj=True, cluster_adj=False)` 只去掉 `G/(G-1)` | 改为断言 `SSC(adj=True)`（Stata 的 `small`）= 默认 × CR1/CR0 | 同一次捕获：Stata `small` 才引入两个因子，默认是 CR0，没有"只加 `(N-1)/(N-K)`"的 Stata 选项 |
| `tests/test_iv.py` | 无 | 新增 `test_iv_vce_robust_is_hc0_and_small_is_hc1`（钉住 `HC0 = [.05607703848079, .05246009123669, .07300778706249]`、`HC1 = [.05668769574323, .05303136134241, .07380281362936]`） | 140×14 IV 面板（与 retest 同 seed 20261024），`ivregress 2sls y x1 (e1 = z1 z2), vce(robust)` / `… small`，2026-10-24 |
| `tests/test_iv.py` | 无 | 新增 `test_liml_cluster_defaults_to_cr0_like_stata` | 同面板 `ivregress liml …, vce(cluster gid)` → SE `[.04174899098950, .05239940413109, .06219927341224]` |
| `tests/test_iv.py` | 无 | 新增 `test_gmm_cluster_reproduces_the_retest_panel`（用 `retest/domains/linear_iv/run.py` 的 `make_panel` 逐字复刻 140×14 数据，并断言 heteroskedastic-W 版本仍差 1e-2..6e-2 —— 即修复前那个 3.58e-2） | 同面板 `ivregress gmm yiv x1 (e1 = z1 z2), vce(cluster gid)` → b `[1.0126539516130684, 2.0935721060105577, -.48536211184222755]` |
| `tests/test_ols.py` | 无（只有 statsmodels 对比） | 新增 14 条 Stata 断言：`e(F)` × {nonrobust, robust, cluster, pweight, aweight, fweight, 截距-only}、`e(F) = Wald/k` 恒等式、`e(df_r)=G-1`+`e(N_clust)`、`e(rss)/e(mss)/e(sum_w)` × {nonrobust, 三种权重, noconstant} | auto.dta + `gen byte gid = mod(_n,12)` + `gen int w = mod(_n,4)+1`，2026-10-24（数值见脚本 `STATA_E_F` / `STATA_SUMS`） |

**没有改动任何 fixture、registry、README、docs、`results/`。**

---

## 未修 / 未达 A 的

L1–L6 的验收项全部 ≤1e-9，且都有可重跑脚本。

**一个如实的例外（BRIEF 允许的"给出实测残差与原因"）**：`aweight` 下的
`e(ll)` / `e(ll_0)` 差 2.160e-02 / 1.810e-02。

**这不是本轮引入的**：`e(ll)` 的公式本轮一行未改（diff 里只有 `rss/mss/sum_w/
N_clust/ll_0/F/df_r` 与其依赖的计算顺序），`e(ll_0)` 是新增字段但复用了同一个
公式（只把 `rss` 换成 `tss`），所以两者的偏差同源、同向、同大小。
复测（`retest/domains/linear_iv`）从未把 `e(ll)` 记成 fail，正是因为它在
aweight 下一直是这个值。

原因：Stata 的 aweighted 高斯对数似然带一个**权重约定的常数偏移**。

**可复算判据（Lead 复核用，两条，互相独立）**：

1. **反解隐含 σ² 的倍数**（`σ²` 是 Stata 与 westata 之间唯一的自由度）：

   ```
   σ²_implied/σ²_westata = exp( 2·(ll_westata − ll_stata)/N )
                         = exp( 2·4.4043928716/74 )
                         = 1.1264123214829
   ```

   即 Stata 的 `e(ll)` 等价于把残差平方和乘 1.1264123214829 再取同一公式；
   `e(ll_0)` 给出**同一个**倍数（差 1.4e-15 的最后一位），说明偏移作用于整个
   似然、不作用于残差。

2. **`e(ll) − e(ll_0)` 两边精确相等**：aweight 下两个绝对差都是 4.4043928716，
   相减后为 **0.000e+00** —— 所以 LR 统计量与伪 R²（任何用到 `ll − ll_0` 的
   下游量）都不受影响，这也是"可接受为已知偏差"的关键理由。

实测覆盖（`verify_aweight_ll.py` 一条命令复算全部）：

| 情形 | `e(ll)` \|diff\| | `e(ll_0)` \|diff\| | 反解 σ² 倍数 | `(ll−ll_0)` 差 |
|---|---|---|---|
| nonrobust | 0 | 0 | 1.0000000000000 | 0 |
| `vce(robust)` | 0 | 0 | 1.0000000000000 | 0 |
| `vce(cluster gid)` | 0 | 0 | 1.0000000000000 | 0 |
| pweight | 0 | 0 | 1.0000000000000 | 0 |
| **aweight** | **4.4043928716** | **4.4043928716** | **1.1264123214829** | **0.000e+00** |
| fweight | 0 | 0 | 1.0000000000000 | −1.1e-13（浮点噪声） |
| iweight | 0 | 0 | 1.0000000000000 | −1.1e-13（浮点噪声） |
| `noconstant` | 0 | 两边都不存 | — | — |

即：**只有 aweight 一种情形带这个偏移**；其余六种（含 `noconstant` 的 `e(ll)`）
逐位相同。

不在 L1–L6 的验收项里（L6 只要求 `rss`/`mss`/`sum_w`/`N_clust`），因此不改。
下一轮若要对接 Stata 的 aweighted `e(ll)`，需要先弄清 Stata 对 aweight 的
似然归一化（`[R] regress` 未给出，只能用上面第 1 条那样黑箱反解）。

`retest/domains/linear_iv/run.py` 里 L6 的覆盖项中还列了几个
westata **故意多存**的量（`aic`/`bic`/`sigma` for OLS；`F`/`F_first`/`ll`/`tss`
for IV）。这些是便利字段，Stata 不存，属"多"而不是"少"；`repro_l6_iv_e.py`
会把它们作为 `westata extra (documented conveniences)` 打印出来。

复测域脚本重跑（把 recorder 重定向到 `retest/fix/linear-iv/_rerun/`，
不动 `retest/domains/`）：`{'pass': 56, 'fail': 5, 'skip': 4}`，5 条 fail 全部是
**本轮刻意保留的"旧文档映射/旧默认"证据**，不是未修项：

| 条目 | 为什么仍是 fail |
|---|---|
| `10d / 11d / 12c …vs westata HC1（文档映射）` | 这三条正是把"HC1 当成 `vce(robust)`"的错误映射固定成 fail 的证据。修复后 HC1 对的是 `vce(robust) small`（`10e` 就是这条，pass）。要它们转 pass 只能把比较对象改掉 → 需 Lead 改 `run.py`。 |
| `11f Stata liml cluster **small** vs westata cluster` | westata 的 `cluster` 默认现在是 CR0（= Stata 无 `small`），与 Stata 的 `small` 当然不同。改成 `ssc=SSC(adj=True)` 即 pass。 |
| `7c areg absorb(gid) vce(cluster gid) vs FEOLS` | **与本组无关，且是防误判项**：`retest/REPORT.md` §2.1 的"防误判提示"已说明 areg 把 13 个虚拟变量计入 K（K=16）而 reghdfe 记 K=3，闭式复算三方都对得上 1e-15。这条 fail 是**预期**的。 |

## 需要 Lead 裁决的

1. **无能力回退**。L3 按优先路线实现成功（不用 `UnsupportedError`），
   所以 BRIEF 里"做不到就显式拒绝"的分支没有触发。
2. **`retest/domains/linear_iv/run.py` 需要跟着更新**（该文件由 Lead 拥有，
   我不改）。它有几条记录是专门"按旧文档映射去比"的，修完后这些条目的
   *语义*变了（不是数值变好）：

   | run.py 条目 | 修复后状况 | 建议改法 |
   |---|---|---|
   | `10d / 11d / 12c …vs westata HC1（文档映射）` | **仍 fail，而且应该 fail** —— HC1 现在对应 `vce(robust) small`，不是 `vce(robust)`。这三条是"错误标签被记成 fail"的证据，不该转 pass。 | 若要它们转 pass，把比较对象改成 `…, vce(robust) small`（并另留一条 `HC1 vs vce(robust)` 作为**反例**，预期 fail） |
   | `10e / 11d …small vs HC1` | 仍 pass（数值没变） | 保留 |
   | `11f Stata liml cluster small vs westata cluster` | **fail** —— westata 的 `cluster` 默认现在是 CR0 | 改成 `ssc=SSC(adj=True)` 才对应 Stata 的 `small` |
   | `11e Stata liml cluster vs westata cluster` | **pass**（1.5e-15 系数 / 9.1e-16 SE） | 保留；原文"SE 系统性大 4.53%"的归因段落应改写为"已修复" |
   | `12d gmm vce(cluster)` | **pass**（点估计 3.58e-2 → 4.6e-16） | 保留；notes 里"westata 固定用 heteroskedastic W"要改成"已按 `vce()` 选择 W" |
   | `4b / 4d / 5e / 3b`（`e(df_r)`、`e(F)`） | **pass** | 保留 |
   | `15a–15e` 覆盖项 | **pass** | 保留 |

3. **`e(df_r)` 对 IV 的语义 —— 已裁决：保持存 `N`，只写文档。**
   Stata 在没加 `small` 时存缺失。westata 往同一个槽位放有限数，因为该槽位
   同时是推断自由度 `results.df_resid`。**实测证据（为什么不能填 NaN）**：
   同一批数据里把 `df_resid` 置为 `nan` 后，`pvalues` 变成 `[nan, nan]`、
   `conf_int()` 的上下界全 NaN —— 比"语义略偏但已注明"危险得多。
   要逐位贴合需要 results 层把"存下来的 `e(df_r)`"与"推断用的 `df_resid`"
   拆成两个概念，属 `results/base.py` 改动，本轮不做。
   已按裁决把这段话说满，写在 `iv.py` 的 "Known deviations from Stata" 里
   （Lead 要原样引用）。同时修正了同一文件更早的
   "Degrees of freedom (``e(df_r)``)" 小节 —— 它原本写"Stata reports
   `e(df_r) = N - K`"，也是错的（无 `small` 时缺失），一并改成正确表述。

---

## 跑过的 pytest

```bat
.venv\Scripts\python.exe -m pytest tests\test_ols.py tests\test_ols_weights.py ^
  tests\test_iv.py tests\test_cluster_df.py tests\test_vce_cluster.py ^
  tests\test_vce_dispatch.py tests\test_vce_robust.py tests\test_ssc.py ^
  tests\test_weights.py tests\test_fweight_expansion.py tests\test_results.py -q
-> 185 passed, 2 warnings in 10.24s
```

（那 2 条 warning 是 linearmodels 自己的 `MissingValueWarning`，来自
`test_clusters_align_on_index_after_listwise_deletion`，修复前就存在。）

全量：

```bat
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
-> 10 failed, 1311 passed, 13 warnings in 229.02s
```

10 条失败**全部在别的组的文件里**，与本组无关：

| 文件 | 条数 | 归属 |
|---|---|---|
| `tests/test_panel.py` | 4 | `panel` 组（正在改 `panel.py` 的 `auto_df`/robust 语义） |
| `tests/test_panel_dynamic.py` | 1 | `panel` 组（xtabond 工具变量计数） |
| `tests/conformance/test_stata_p9.py` | 2 | `panel` 组（xtpcse / xtabond） |
| `tests/conformance/test_stata_p11.py` | 2 | `systems` 组（didregress，BRIEF §4.5 的 S5） |
| `tests/test_discrete.py` | 1 | `discrete` 组（logit 的 Wald chi2，BRIEF §4.3 的 D4） |

本组拥有的所有测试文件全绿：
`test_ols`(25) + `test_ols_weights`(9) + `test_iv`(47) + `test_cluster_df`(9)
+ `test_vce_cluster` / `test_vce_dispatch` / `test_vce_robust` / `test_ssc` /
`test_weights` / `test_fweight_expansion` / `test_results` = **185 passed**。
