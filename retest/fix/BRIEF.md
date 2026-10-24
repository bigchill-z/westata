# 修复任务书（westata 复测缺陷修复轮）— 所有子智能体必读

## 0. 背景与目标

`retest/REPORT.md` 记录了 391 个对比点、**24 个独立缺陷**。本轮的唯一目标是：
**把这些缺陷修到与真实 Stata 数值一致，或用显式拒绝替代静默的错数。**

两条铁律：

1. **不许为了"修好"去改测试的判据**。旧测试里有一些断言的是**错误行为**
   （例如 `tests/test_panel.py` 里有一个"证明 auto_df=False 能复现 Stata 聚类 SE"的测试，
   该结论已被实测推翻）。改测试是允许的，但**必须附上本次现场 Stata 证据**，
   写清"旧断言错在哪、新断言依据什么"。改测试而不给证据 = 本轮失败。
2. **不许把错数换成另一个错数**。每一处修改都必须用一个**新的 Stata 实测**验证：
   改之前先跑一次记下差异，改之后再跑一次证明 ≤1e-9（或给不出 1e-9 的项，明确写出实测残差与原因）。
   报告里"已修复"三个字必须能被一条命令复现。

## 1. 环境（已就绪）

- 仓库根：`D:\myCodeProject\statapy`（所有命令以此为 cwd）
- Python：`.venv\Scripts\python.exe`（3.10.11，westata 已 editable，`import westata` 直接改源码生效）
- Stata：`D:\Program Files\StataNow19`，`pystata` 可用（已把 `python310.dll`/`python3.dll` 拷进 `.venv/Scripts`）
- 对比工具：`retest/lib/stata_probe.py`（**只读，不要改**）

```python
import sys; sys.path.insert(0, "retest/lib")
from stata_probe import Stata, compare, rel_diff, canon_stata, canon_westata
```

## 2. 工作方式

### 2.1 每个缺陷的固定流程

1. **先复现**：写一个 `retest/fix/<你的组>/repro_<n>.py`，用真实 Stata 证明缺陷存在，记录改前数字。
2. **读源码**：定位到具体行，读懂为什么错（不要凭现象猜）。
3. **改代码**：最小改动。修根因，不要为了过一条测试加特例。
4. **改后验证**：同一个脚本再跑，输出 `改前 / 改后 / Stata` 三列 + 相对差。
5. **补/改测试**：在 `tests/` 下加一条**断言 Stata 数值**的回归测试（用现场捕获的数字当常量，
   注释里写明 `regress mpg weight, vce(robust)` + 数据 + 捕获日期）。
6. **写进交付报告**：`retest/fix/<你的组>/FIXES.md`（格式见 §5）。

### 2.2 判据

| 档位 | 相对差上限 |
|---|---|
| A | 1e-9 ← **绝大多数修复的目标** |
| B | 1e-6 ← 迭代优化器类（ML/GMM）的合理目标 |
| C | 1e-3 ← 只能到此的，必须写清原因 |

相对差 = `max(|westata − stata| / max(|stata|, 1e-12))`，逐系数取最大；SE 用 `e(V)` 对角开方。

### 2.3 纪律

1. **只改你的写作用域内的文件**（见 §3 的归属表）。越界修改 = 本轮失败。
2. **不 `git commit`**，不 `git add`。Lead 统一提交。
3. **不碰**：`tests/conformance/stata_fixtures/`（fixture 是 Stata 真值，绝不改）、
   `tests/conformance/registry.yml`、`README.md`、`docs/`、`retest/lib/`、`retest/REPORT.md`。
   这些由 Lead 统一改。
4. Stata 会话串行用；单条超过 3 分钟就缩小样本并在报告里说明。
5. 跑 `pytest tests -x -q` 相关文件确认没有回归；最终全量由 Lead 跑。
6. 遇到**互相冲突**的两条要求（例如"修复 A 会让 B 变差"），**不要自己二选一**：
   在 `FIXES.md` 里写清冲突，选一个你认为更对的，并明确标注"需 Lead 裁决"。

## 3. 写作用域归属表（互斥，不得越界）

| 组 | 拥有的源文件 | 拥有的测试文件 |
|---|---|---|
| `linear-iv` | `src/westata/estimators/linear.py`, `src/westata/estimators/iv.py`, `src/westata/vce/*` | `tests/test_ols.py`, `tests/test_ols_weights.py`, `tests/test_iv.py`, `tests/test_cluster_df.py`, `tests/test_vce_*.py`, `tests/test_weights.py` |
| `panel` | `src/westata/estimators/panel.py`, `panel_dynamic.py`, `panel_nonlinear.py`, `panel_iv.py` | `tests/test_panel*.py` |
| `discrete` | `src/westata/estimators/discrete.py`, `src/westata/estimators/counts.py` | `tests/test_discrete.py`, `tests/test_counts.py` |
| `limited` | `src/westata/estimators/tobit.py`, `src/westata/estimators/limited.py`, `src/westata/estimators/quantile.py` | `tests/test_tobit.py`, `tests/test_limited.py`, `tests/test_quantile*.py` |
| `systems` | `src/westata/estimators/nonlinear.py`, `treatreg_effects.py`, `resample.py`, `did.py` | `tests/test_nl.py`, `tests/test_nonlinear.py`, `tests/test_teffects*.py`, `tests/test_resample.py`, `tests/test_did.py`, `tests/test_systems.py` |
| `postest` | `src/westata/postestimation/margins.py`, `api.py`, `modeltests.py`, `predict.py` | `tests/test_margins.py`, `tests/test_postestimation.py`, `tests/test_predict_se.py`, `tests/test_modeltests.py`, `tests/test_contrast.py` |
| `tseries` | `src/westata/estimators/tseries.py` | `tests/test_tseries.py` |

共享但**谁都不许改**的：`src/westata/results/*`、`src/westata/protocols.py`、`src/westata/errors.py`。
如果你的修复**必须**改 `results/base.py`（例如 `e()` 的默认值），**停下来写进 FIXES.md 标注"需 Lead 改"**，
用能够工作的本地绕法（在 estimator 里显式传 e 字典）先把功能修好。

## 4. 各组的具体缺陷（编号与 `retest/REPORT.md` §3 对应）

### 4.1 `linear-iv`

- **L1 (P1-13)** IV 的 `vce(robust)` 标签错。Stata：`ivregress 2sls y x (e=z), vce(robust)` = **HC0**；
  `..., vce(robust) small` = **HC1**。实测：westata `HC0` 与 Stata 差 6.1e-15，`HC1` 差 1.09e-2。
  修：模块 docstring 的映射表 + `HC0`/`HC1` 的 docstring 写清各自对应哪个 Stata 选项；
  加回归测试断言两件事实。**不要**改 `_canonical_cov_type` 的名字。
- **L2 (P1-14)** `IVLIML` 的 `vce(cluster)` 恒加 CR1；Stata 默认 CR0（`small` 才是 CR1）。
  实测比值 1.045296408534764 = `sqrt((N-1)/(N-K))`，Stata 加 `small` 后与 westata 一致到 8.7e-16。
  `IV2SLS` 没这问题（走 `compute_cov` 且默认 `SSC(adj=False)`）——所以修法是让 LIML 走同一条 VCE 路径/同一 `SSC` 默认。
- **L3 (P1-17)** `IVGMM` + `vce(cluster)` 点估计差 3.58e-2：Stata 在该 VCE 下把 GMM 权重矩阵 `W` 也聚类。
  **优先实现**（W 用聚类 S 的逆），用 140×14 组数据验证到 ≤1e-9；
  若实现后无法达到 1e-6，**改为显式 `UnsupportedError`** 并写清"为什么不能给 Stata 的数"。
- **L4 (P1-15)** `OLS.e['F']` 不随 `cov_type` 变。Stata 的 `e(F)` = 用该 VCE 的 `e(V)` 对**全部斜率**做的 Wald/k。
  实测：把 westata 自己的 `V` 代入 `b'V⁻¹b/k` 可复现 Stata 到 1.9e-15。修：按 `cov` 计算。
  必须验证 nonrobust / HC1 / cluster / pweight 四种情形的 `e(F)` 都到 1e-9。
- **L5 (P1-16)** `OLS.e['df_r']` 在 cluster 下存 `N-K`，Stata 存 `G-1`（`results.df_resid` 已经是对的）。
- **L6** `e()` 覆盖缺口：补 `rss`、`mss`、`sum_w`（fweight 下 = `e(N)`）、`N_clust`（cluster 时）。
  `ll_0`/`kappa`/`J`/`k_endog`/`iterations` 若成本高可留，写进 FIXES 的"未修"。

### 4.2 `panel`

- **P1 (P0-4)** `xtreg, fe vce(robust)`：Stata 里**严格等于** `vce(cluster panelvar)`
  （Lead 已独立验证 `max|V_robust − V_cluster| = 0.0`，30×6 平衡面板）。westata 映射成 HC1 → 常数项 SE 差 18 倍。
  修：`xtreg` 族的 `cov_type="robust"/"HC1"` 一律映射为 cluster(panelvar)。
  **注意**：`reghdfe` 的 robust 是 HC1，`areg` 也是 HC1——只有 `xtreg` 不同。不要改 `FEOLS`/`areg` 路径。
- **P2 (P0-5)** `xtreg, fe vce(cluster id)` SE 差 9.36%：小样本因子的 `K` 算成 32（含被吸收效应）。
  `panel.py:403-408` 强制 `auto_df=False`，且**那段注释的论断是错的**（说 auto_df=True 会漏效应）。
  用现场 Stata 验证 `auto_df=True` 是否逐位复现，然后改代码 + 改注释。
- **P3 (P0-6)** `xtpcse` 非平衡：`_aligned_slices` 按行号取前 k 行，Stata 按"所有面板共同期"估 Σ；
  `pairwise` 分支根本未实现（与 casewise 逐位相同）。修 casewise；pairwise 要么实现要么 `UnsupportedError`。
  **平衡面板现在是 A 档（1.44e-15），不能改坏。**
- **P4 (P0-12)** `xtabond`：`e(sig2)` 未除 2（差 1.993 倍）；`lags(2)` 时 `param_names` 整体错位一位
  （实际 `['L2.y','x1','x2','extra_0']`，真值 `['L1.y','L2.y','x1','x2']`）。命名错位会让**按名字取系数静默取错**，最优先。
- **P5 (P1-24)** `xtnbreg, fe` 只做了条件 Poisson OIM（与自己的 `xtpoisson, fe` 逐位相同），SE 差 6.79e-2。
  实现 HHG 到 ≤1e-6 就实现；做不到就 `UnsupportedError`（诚实优于静默，但**必须在 README 层面让 Lead 知道这是能力回退**）。

### 4.3 `discrete`

- **D1 (P0-1)** `ologit/oprobit` 的 `e(V)` 差 20 倍。verifier 插桩确认**两个独立缺陷**：
  ① `discrete.py:386` `theta = params[n_slopes:]` 是 **numpy 视图**，`:392` `params[n_slopes:] = cuts` 把 `theta`
  也改成了 cut 点，于是 `:398` 的 `np.exp(theta[...])` 作用在 **cut 点**上（应为 log-gap）；
  ② `:399-400` 只改了 `(cut,cut)` 块，**slope×cut 交叉块 `cov[:ns, ns:]` 从未变换**。
  正确做法：`theta = params[n_slopes:].copy()`；`J` 用 log-gap 构造；交叉块做 `cov[:ns, ns:] @ J.T`。
  验证：Lead/verifier 的参考值 `cut SE = [0.0874, 0.0962]`（另一组数据）、`e(V)` 全矩阵 ≤1e-9；oprobit 同理。
- **D2 (P0-9)** `vce(robust)` 对 Logit/Probit/NegativeBinomial 全崩（`cov_kwds` 传给 statsmodels 就报错）。
- **D3 (P0-10)** `Poisson` 的 `cov_type="HC1"` 静默等于 `HC0`（`array_equal=True`）。
- **D4 (P0-11)** `e(chi2)` 在 `vce(robust)/vce(cluster)` 下没从 LR 换成 Wald。
  Stata：`chi2type` 变 Wald，且 Wald **只含斜率块**（实测 `b_s'V_ss⁻¹b_s` 复现到 1e-16）。
- **D5 (P1-18)** `e(df_m)` 把辅助参数算进去（nbreg 3 vs Stata 2；ologit 4 vs 2）。`tnbreg` 走另一条路径是对的。
- **语义统一要求**（本轮的设计决定，按此实现）：
  - `cov_type="robust"`（新别名）= **Stata 的 `vce(robust)`**：ML 模型的原始 sandwich × `N/(N-1)`；
  - `cov_type="cluster"` = **Stata 的 `vce(cluster g)`**：只有 `G/(G-1)`，**没有** `(N-1)/(N-K)`
    （与 `regress` 的 CR1 不同，这是 ML 模型的 Stata 约定）；
  - `HC0`/`HC1` 保留，但 docstring 必须写明"这是 statsmodels 的约定，**不是** Stata 的 `vce(robust)`"。
  - 三者都要有 Stata 对照测试；`HC1==HC0` 的静默必须消失（要么真的不同，要么拒绝）。

### 4.4 `limited`

- **T1 (P0-3)** `tobit` 的 `/sigma` SE 是 `se(lnσ)`：`cov` 用 log 尺度参数化求 Hessian，换回 σ 时没做 Jacobian。
  实测比值 `westata/stata_se(σ) = 0.945719091 ≈ 1/σ̂`。修：`se_σ = se_lnσ · σ̂`（或对 cov 做 `diag(1,1,1,σ)` 变换）。
  三个变体（ll / ul / ll+ul）都要到 1e-6。
- **T2 (P0-?/P1)** `tobit` 的 `e(df_r)`=71 vs Stata 72：**用 ≥3 组不同 N/k 的数据反推 Stata 的规则**，别只看一个数字。
- **T3** `tobit` 数值 Hessian(step=1e-5) 导致系数 SE 偏 8.9e-5：换解析 observed information 或调步长，
  验证 stdp `sqrt(x'Vx)` 到 1e-9。
- **T4 (P1-21)** `qreg, vce(robust)` 在 τ≠.5 错。**注意 verifier 的更正**：换成 `τ(1−τ)X'X` 只在同方差数据上到 1e-8，
  异方差数据上仍差 15%–147%，说明 **bread（逐点密度 `f_i`）也参与**。必须让 τ=.25/.5/.75 × 同方差/异方差 **全部** ≤1e-6。
- **T5 (P1-20a)** `heckman` 的 `e(sigma)`/`e(rho)`：Stata 的 `e(sigma)=sqrt(RSS/m + b_λ²·mean(λ(λ+w)))`
  （复现到 1.8e-10）；`e(rho)` 跟着改。
- **T6 (P1-20b)** `etregress` 的 `e(rho)` **反号**：westata 用 `rho = −b_ctrl/σ`，应为 `+b_ctrl/σ`
  （Stata 的 `e(lambda) = b_ctrl = ρσ`，已验证 `0.41420433 = 0.42835 × 0.96699`）。两个 DGP 都要对。
- **T7 (P1-22)** `rreg` 的 SE：Stata **不用 pseudovalue**，是 `se(b)=e(rmse)·sqrt(diag((X'X)⁻¹))`，
  四个数据集复现到 1e-16。改方差定义；连带 `e(rss)/e(mse)/e(rmse)/e(r2)/e(F)` 一起对齐
  （约束 `rmse²·(n−k) = rss`）。
- ~~**T8** `frontier` halfnormal 的数值 Hessian 步长~~ → **已转给 systems 组**（`Frontier` 与 `Nl` 同在 `nonlinear.py`，避免并发写）。limited 组把方案写进 FIXES.md 最后一节。

### 4.5 `systems`

- **S1 (P0-1)** `nl` 的协方差用 `pinv` 默认截断，SE 差 7 个数量级：
  `J'J` 条件数 6.13e9，`s_min/s_max = 3.91e-17 < max(shape)*eps = 6.66e-16`。
  修：用 `np.linalg.solve`/`lstsq`，或在秩亏时抛 `ConsistencyError`。**不要**简单地 `rcond=0`
  （会放大噪声）；先判秩，秩不足就报错。验证：`nl (mpg = {b0}+{b1}*weight+{b2}*weight^2)` 的 SE 到 1e-9。
- **S2 (P0-?)** `nl` 默认初值全 1，LM 停在离解很远的点（RSS 差 7.5 倍），只发 warning 仍返回。
  修：默认 `strict=True` 语义（未收敛抛 `ConvergenceError`），或复刻 Stata 的网格起步。二者择一，写清理由。
- **S3 (P0-8)** `teffects aipw, atet`：ATET 分支的增广项结构错（`augmented0` 带 (1−t)、`augmented1` 多带 1/p）。
  **ATE 分支现在是精确的（1e-16），绝对不能改坏**。ATET 校正后用 `atet` 的 ATET/PO(0)/PO(1) 三个量与 Stata 对齐。
- **S4 (P1-?)** `nnmatch`：ATET 的 SE 与 ATE 的 SE **逐位相同**（`share0=share1=1`），ATET 方差没单独算。
  Stata 两者比值 0.8823（另一数据 1.1947）。修 `share` 逻辑；目标 ATE/ATET 的 SE 差 ≤1e-2
  （Abadie-Imbens 本身是近似，能用 Stata 公式就到 1e-6）。
- **S5 (P1-?)** `didregress`：`covariates=None` 会**自动把其他时变数值列当协变量**（Stata 的 `(y)` 不会）。
  反转语义：默认不加协变量；需要"全部时变列"时用显式开关（如 `covariates="all"`），并保留 warning。
- **S6 (P1-23a)** `jackknife` 的 t 检验用 `df_r=72`，Stata 用 73。
- **S7 (P1-23b)** `bootstrap` 用 `t(72)` 而 Stata 用 **z**（区间宽 1.7%、p 偏保守）。
- **S8（T8 转入）** `Frontier` halfnormal 的 `_numerical_hessian(step=1e-5)` 太粗（SE 差 4.0e-3）：改自适应步长或解析梯度。详见 limited 组 FIXES 的「转交方案」。

### 4.6 `postest`

- **M1 (P0-7)** `margins` 的 AME 标准误：`margins.py::_statistic_and_gradient` 用**固定 h=1e-6 中心差分**。
  实测步长扫描 h=1e-3→4.9e-6、1e-6→7.1e-3、1e-8→**0.70**（误差随 h 减小而变大，典型相消）。
  修：对 logit/probit 用解析 `p(1−p)x`；其他模型用**自适应步长**（`h ∝ max(1,|θ|)·eps^{1/3}`）。
  验证：`margins, dydx(weight)` 的 SE 与 `at(mpg=20)` 都到 1e-6，且 `over()`/`dydx(foreign)` 不退化。
- **M2 (P0-?)** `predict, stdf`（tobit）：用 `rss/N` 代替 `sigma²`（`stdf²−stdp²=12.127` vs `σ²=14.298`）。
  修：tobit 的 `stdf² = stdp² + e(sigma)²`（依赖 `limited` 组把 `e(sigma)` 修对；
  若 `e(sigma)` 尚未修好，用设计矩阵与参数自己算 σ）。
- **M3 (P1-?)** `predict, n`（zip/zinb）崩溃：`StataResults.predict` 对非 poisson/nbreg 一律走二值分支。
  **契约（与 `discrete` 组约定）**：结果对象上若有 `_predict_dispatch` 回调则由模型接管；
  模型侧由 `discrete` 组为 `zip/zinb/ztp` 实现 `predict(kind)` 的 `n`/`pr`/`xb`。
  你负责**通用委托逻辑 + 明确的错误信息**（不支持的 kind 列出可用项），不要硬编码模型名清单。
- **M4** `lincom` 拒绝裸常数项（Stata：`lincom 2*weight - 3*foreign + 10` 返回 14.9369，SE 不变）。
  实现：允许项里出现纯数字常数（进入 estimate，不进方差）。
- **M5** `lrtest` 的 `df=0` westata 返回 p=1.0，Stata 报 `r(498)`；非嵌套 Stata 接受并返回**负 LR**，westata 拒绝。
  按 Stata 对齐：`df=0` 报错（`ConsistencyError`，消息说明）；非嵌套允许负 LR。
- **M6** `hausman` 缺 `constant` 开关（Stata 的 `hausman ..., constant` 把常数项纳入比较）。
  加参数；默认行为不变。

### 4.7 `tseries`

- **V1 (P0-?)** VAR 的 SE / `e(Sigma)` / Granger 差 `(T−k_eq)/T`（k_eq 是**每方程**回归元数）。
  实测比值 0.9949706303122452 = `sqrt(296/299)`。修：按 Stata 的每方程 ML 归一化输出。
- **V2** Granger 默认返回 F 统计量（statsmodels `kind="f"`），Stata `vargranger` 是 **Wald chi2**；改默认 + 文档。
- **V3** `dfuller` 打印的临界值：Stata 的 `dfuller.ado` 硬编码 6 点表按 N 线性插值，照抄后能复算到 1.28e-16。
  修：实现同一张表（统计量与 MacKinnon p 已是 A 档，别改坏）。
- **V4** `vecrank` 的临界值表：Stata 用 Johansen(1995)（r=1 时 **3.76**），statsmodels 用 Osterwald-Lenum（**3.8415**）。
  修：换成 Stata 的表（trace 与 max-eigen 都要），并造一组边界数据证明选阶与 Stata 一致。
  注意：**只换表，不换统计量公式**（统计量已是 A 档）。
- **V5** `arima` 在 `d=1` 时 `e(N)` 差 1（Stata 299 vs westata 300）。
- **V6** `arch`：① `fit()` 从不传 `cov_type`，继承 arch 包默认 `robust`，而 Stata 是 **OPG**；
  ② presample `h₀` 不同导致**目标函数不同**（Stata 用 `e(archi)=mean(去均值残差²)`）。
  先修 ①（低风险、可验证），再试 ②（若能把参数差从 1.07e-2 压到 ≤1e-3 就保留）。
  目标：SE 从 2.78e-1 降到 ≤1e-2。

## 5. 交付物（每组）

`retest/fix/<组名>/` 下：

1. `repro_*.py` —— 可重跑的复现+验证脚本（改前/改后/Stata 三列）。
2. `FIXES.md` —— 固定结构：

```markdown
# <组> 修复报告

## 逐项结果
| # | 缺陷 | 改前相对差 | 改后相对差 | Stata 命令 | 判定 |
|---|---|---|---|---|---|
| L1 | IV vce(robust)=HC0 标签 | 1.09e-02 | 6.1e-15 | `ivregress 2sls ... , vce(robust)` | 已修复 A |

## 每项的根因与改法
（一两句：错在哪个文件的哪一行，为什么错，怎么改的）

## 测试变更
| 文件 | 旧断言 | 新断言 | 依据（Stata 捕获） |

## 未修 / 未达 A 的
（逐条写清：为什么修不了、实测残差多少、建议下一步）

## 需要 Lead 裁决的
（冲突、需要改共享文件 `results/`、需要能力回退的）
```

3. 回给 Lead 的一段话（按 §6）。

## 6. 回给 Lead 的格式

- 组名；修了几项 / 未修几项；
- **每一项一行**：编号 + 改前 → 改后 + 档位 + 验证命令；
- **测试改了什么**（哪条旧断言、依据哪次 Stata 捕获）；
- **需要 Lead 裁决**的（尤其：能力回退、需要改 `results/`、与其他组冲突）；
- 你跑过的 `pytest` 命令与结果。

## 7. 已知陷阱

- `from pystata import stata` 必须在 `config.init()` 之后。
- Stata 缺失值 `8.988e+307`；`st.var()/st.frame()` 已转 NaN。
- `stata.get_ereturn()` 会保留上一个估计——先跑估计再 capture。
- `fweight` 下**不能用 `nobs` 比样本量**（`e(N)=sum(w)` ≠ `nobs`）。
- `esttab`/`kpss` 等是外部安装的；报告里要标明。
- `sysuse` 只有 Stata 自带数据集；自造数据用 `st.send(df)` 送。
- Python 侧改完源码立即生效（editable），**不需要**重装。
