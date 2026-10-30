# westata 辅助指标与展示功能缺口：Lead 主线盘点

> 本文是 Lead 亲自读源码 + 实跑验证的部分，覆盖**跨家族共性**与**展示层**。
> 三个子智能体分别负责：线性/IV/时序、离散/受限/面板非线性、分位数/系统/因果/抽样/重抽样。
> 所有断言附 `path:line`。未经实跑确认的一律标注"未验证"。

## 0. 用户的三个具体指认，逐条核实

### 指认 1：Stata 输出的 RSS、TSS 在 westata 结果里不存在

**部分成立，且比用户说的更微妙——`rss` 大部分有，`tss` 大面积缺，而且是"算了但不存"。**

实测（`tests/data/auto.dta`，`mpg ~ weight + foreign`）：

| 估计器 | `rss` | `mss` | `tss` | 证据 |
|---|---|---|---|---|
| `OLS` | ✅ | ✅ | ❌ | `estimators/linear.py:311-327` 的 `e` 字典含 `rss`/`mss`，无 `tss` |
| `ivregress 2sls` | ✅ | ✅ | ✅ | `estimators/iv.py:824-826` |
| `nl` | ✅ | ✅ | ✅ | `estimators/nonlinear.py:1332-1334` |
| `reghdfe` (`FEOLS`) | ❌ | ❌ | ❌ | `estimators/hdfe.py:520,522` **算出了 `tss`，但 `e` 字典（536-546 行）三个都没存** |
| `svy: regress` | ❌ | ❌ | ❌ | `estimators/survey.py:1055-1056` **算出 `tss`，`fit()` 只回 `r2`/`F`/`p`（1071 行）** |
| `xtreg, fe/re/be/fd` | ✅ | ❌ | ❌ | `estimators/panel.py:753` |
| `xtivreg` | ✅ | ✅ | ❌ | `estimators/panel_iv.py:1550-1551` **算出 `tss`，没存** |
| `xtabond` | ✅ | ❌ | ❌ | `estimators/panel_dynamic.py`（扫描见下） |
| `qreg`/`sqreg`/`rreg` | ✅ | ✅ | ❌ | `estimators/quantile.py` |
| `sureg`/`mvreg` | ✅ | ✅ | ❌ | `estimators/systems.py` |
| `didregress`/`xtdidregress` | ❌ | ❌ | ❌ | `estimators/did.py` |

**关键判据（来自 Stata 自己的 fixture，不是我的猜测）：**

`tests/conformance/stata_fixtures/regress_auto.json` 的 `scalars` 里**确实没有 `tss`**
（只有 `rss`、`mss`）——所以 `regress` 本身**不存** `e(tss)`，westata 不存它是**符合 Stata 的**。
但：

- `areg.json` **有** `tss`（`['F','F_absorb','N','ar2','df_a','df_m','df_r','k_absorb','ll','ll_0','mss','p','p_absorb','r2','r2_a','r2_c','rank','rc','rmse','rss','sum_w','tss']`）
- `reghdfe_*.json` **有** `tss` 与 `tss_within`
- `xtreg_fe.json` **有** `tss`
- `xtreg_be.json` **有** `tss`

**结论：`tss` 在 Stata 里是按命令存的，不是全局约定。** westata 的问题是**方向一致地缺失**：
凡是 Stata 存 `tss` 的命令，westata 都没存。这不是"漏了一个键"，是**一个跨家族的系统性遗漏**，
而且其中 4 处（`hdfe`、`survey`、`panel_iv`，以及 `linear`）`tss` **已经在内存里算出来了**，
只是没往 `e` 字典里放——属于**零成本可修**。

### 指认 2：回归方程中打印回归系数的协方差矩阵

**完全成立，且是整个库里最大的单块展示缺口。**

Stata 有两条路径拿到协方差矩阵：
- `estat vce` —— 打印 `e(V)` 全矩阵（系数协方差）
- `estat vce, correlation` —— 打印相关系数矩阵

westata 的现状：
- `results.cov_params()` 返回矩阵（`results/base.py:203-206`），**但没有任何 display/render 路径**。
- 全库搜索 `correlation matrix|Covariance matrix|estat vce` 在 `src/` 下**零命中**
  （唯一命中在 `conformance/oracles/stata.py`，是测试夹具的字段名，不是产品功能）。
- `results/summary.py` 两个 renderer 都不打印矩阵。
- statsmodels 有 `normalized_cov_params()` 与完整的 `summary()`；westata 两者皆无
  （实测 `hasattr(r, "normalized_cov_params") == False`）。

**这不是"数据拿不到"，是"完全没有渲染器"。** `e(V)` 数据 100% 可用，缺的是
`estat vce` 与 `estat vce, correlation` 两个输出形态。

### 指认 3：还有很多功能缺少

成立，见 §2 的跨家族共性清单与三个子报告。

---

## 1. 展示层缺口（`results/summary.py` + `report/etable.py`）

### 1.1 `summary(style="stata")` 已渲染的块

实测 `OLS` 与 `logit` 的 `stata` 风格输出（`results/summary.py:132-245`）：

```
Number of obs         =         74
F(2, 71)              =      69.75        <- regress
Prob > F              =     0.0000
R-squared             =      0.6627
Adj R-squared         =      0.6532
Root MSE              =      3.4071
```

`logit`（`dist="z"` 分支，`summary.py:196-208`）：

```
Number of obs         =         74
LR chi2(2)            =      35.72
Prob > chi2           =     0.0000
Log likelihood        =   -27.1752
Pseudo R2             =      0.3966
```

**四个 Stata 行都在**（Q1 的子智能体会逐族复核）。

### 1.2 展示层系统性缺口（按影响排序）

| # | Stata 展示 | westata | 证据 / 说明 |
|---|---|---|---|
| D1 | `estat vce`（系数协方差矩阵） | ❌ 无 | 见 §0 指认 2 |
| D2 | `estat vce, correlation`（相关系数矩阵） | ❌ 无 | 同上；需要 `D⁻¹ᐟ² V D⁻¹ᐟ²` 变换 |
| D3 | `estat ic`（AIC/BIC 表） | ❌ 无独立入口 | `e(aic)`/`e(bic)` 只在 `statsmodels` 风格 summary 里出现；**`stata` 风格完全不显示 AIC/BIC** |
| D4 | `etable` 底部统计块默认只有 N 与 R² | ⚠️ 极薄 | `report/etable.py:28-31` `_DEFAULT_STATS = (("N","Observations"),("r2","R-squared"))`。Stata 的 `esttab` 默认给 `N r2_a` 或用户 `stats()` 列表；这里连 `rmse`/`F`/`ll` 都不在默认里 |
| D5 | 残差诊断表（Omnibus/JB/Skew/Kurtosis/Durbin-Watson/Cond.No.） | ❌ 无 | 实测 `hasattr(r,"omnibus"/"jb"/"skew"/"kurtosis"/"durbin_watson"/"condition_number")` **全部 False**。statsmodels 的 `summary()` 打印这些（`statsmodels` 风格 renderer 也没抄） |
| D6 | `e(tss)`/`e(rss)`/`e(mss)` 的方差分解行 | ❌ 无 | 即便 `e()` 里有（IV/nl 有），summary 与 etable 都不展示 |
| D7 | `estat summarize`（估计样本的描述统计） | ❌ 无 | `results` 不带原始变量数据（只有 `model_data` 在部分估计器保留，见 `base.py:151`） |
| D8 | `estat firststage`（IV 第一阶段诊断表） | ❌ 无 | `e(F_first)` 有（`iv.py:807`），但**没有展示路径**，也没有 Shea partial R² |
| D9 | 各命令的脚注/说明行（Stata 输出的 `Note:` 段） | ❌ 无 | 如 `xtreg` 的 `rho` 说明、`ivregress` 的弱工具警告 |

### 1.4 已发现的一处**渲染错误**（不只是缺失，是打印了错的东西）

`qreg` 走 `summary(style="stata")` 实测输出：

```
Number of obs         =         74
F(1, 72)              =          .        <- Stata 不打印 F 检验
Prob > F              =          .
Log likelihood        =          .        <- Stata 不打印，qreg 不是 ML 模型
Pseudo R2             =      0.4934
```

而 Stata 的 `qreg` 头部块**结构完全不同**（`tests/conformance/stata_fixtures/p8_outputs.txt:20-22`）：

```
.25 Quantile regression                             Number of obs =        600
  Raw sum of deviations 438.3735 (about -.46158607)
  Min sum of deviations 156.7522                    Pseudo R2     =     0.6424
```

**根因**：`summary.py:216-224` 用 `e(r2_p)` 是否存在来决定走哪个分支。
`qreg` 为了存 Stata 的伪 R² 而用了同一个键名 `r2_p`（`quantile.py:417`），
于是渲染器误判"这是 ML 模型"，打印了一行空的 `Log likelihood = .`。

**这是一条契约耦合缺陷**：`r2_p` 一个键被两种语义共用（ML 的 McFadden vs qreg 的
check-function 伪 R²），而渲染器按键名推断模型类别。修法有两个方向：
(a) 渲染器改为按 `model_name`/`dist` 分支而非按 `r2_p` 是否存在；
(b) 给 qreg 的伪 R² 换键名。**(a) 正确**，因为 Stata 自己也是按命令打印不同头部块。

同时 **`qreg` 的 `Raw sum of deviations` / `Min sum of deviations` 两行完全没渲染**，
而这两个数 `e()` 里都有（`sum_rdev`/`sum_adev`，`quantile.py:415-416`）。
`sqreg` 还多一个 per-quantile Pseudo R2 块（`p8_outputs.txt:126-128`）。

`rreg` 的 Stata 头部块又是另一种（`p8_outputs.txt:181-183`）：F 与 Prob>F **在右侧**，
没有 R²/Adj R²/Root MSE 三行。westata 实测把 `rreg` 当 OLS 渲染，打印了 R² 三行。

**这一节的结论**：westata 的 `stata` 渲染器是**单一模板**（R²/RMSE/F 或
ll/PseudoR²/chi2 二选一），而 Stata 至少有 **5 种**头部块形态：
`regress` 式、ML 式、`qreg` 式、`rreg` 式、`sqreg` 式。**这是渲染层的结构性缺口。**

### 1.5 第二处渲染错误：`svy: mean` 打印了 5 行没有意义的值

实测 `svy: mean`（设计：8 层 × 6 PSU × 12 obs，`svyset psu [pw=w], strata(stratum)`）：

```
Number of obs         =        576
F(0, 40)              =          .        <- 无意义：mean 没有模型 F 检验
Prob > F              =          .
R-squared             =          .        <- 无意义：mean 没有 R²
Adj R-squared         =          .        <- 无意义
Root MSE              =     4.6094        <- 无意义：这是 y 的 sd，不是残差 RMSE
```

Stata 的 `svy: mean` 头部块（`p12_outputs.txt:6-17`）：

```
Survey: Mean estimation

Number of strata =  8             Number of obs   =        576
Number of PSUs   = 48             Population size = 837.592551
                                   Design df       =         40

--------------------------------------------------------------
              |             Linearized
              |       Mean   std. err.     [95% conf. interval]
--------------+------------------------------------------------
            y |   1.061347   .0873402      .8848254    1.237868
--------------------------------------------------------------
```

**根因同上**：`summary.py:132-245` 只有一个模板，不认 `svy:` 家族。
而 `N_strata`/`N_psu`/`N_pop`/`df_r` **四个数都已在 `e()` 里**
（`estimators/survey.py:634-644` 实证，实测 `sorted(m.e.keys())` 确认）。
**所以这是纯渲染缺口，零计算成本。**

`Root MSE = 4.6094` 那一行还额外有问题：`summary.py:140-144` 的兜底逻辑
在 `e(rmse)` 缺失时用 `sqrt(rss/df_resid)` 现算。对 `svy: mean`，
`resid` 是 `y - mean(y)`，`df_resid` 是 design df=40，于是得到一个
**既不是 RMSE 也不是 sd 的数**（真实 sd(y) ≈ 4.6，数值巧合接近，
但语义完全错误——它除的是 design df 而不是 N）。

**这条比"缺一行"更严重：它在打印一个看起来合理、实际语义错误的数字。**
按本项目的纪律（`GAP_REGISTER.md` 反复强调"不能把能跑的写成对的"），
这属于**必须修**的一类：宁可打印 `.`，也不能打印错数。

### 1.6 `bootstrap`/`jackknife` 的 `Replications` 行缺失

Stata（`p13_outputs.txt:40-46`）：

```
Linear regression                                      Number of obs =     400
                                                        Replications  =     200
                                                        Wald chi2(2)  = 1127.33
                                                        Prob > chi2   =  0.0000
                                                        R-squared     =  0.8057
                                                        Adj R-squared =  0.8048
                                                        Root MSE      =  1.0860
```

`jackknife` 多一个 `Replications = 400` 且用 t 分布（`p13_outputs.txt:71-77`），
`bootstrap` 用 Normal-based z（`p13_outputs.txt:49-54`）。

westata：`estimators/resample.py:601-609` 把 `reps`/`seed`/`note` 存进了 `e()`，
**但 `summary.py` 不渲染 `Replications` 行**，所以 `Number of obs` 之后直接跳 F/R²。
系数表头也缺 `Observed / Bootstrap / Jackknife / Normal-based` 那一段——
Stata 用表头第二行区分估计量来源，westata 的表头只有一行。

### 1.7 `teffects` 的头部三行完全没渲染

Stata（`p11_outputs.txt:8-11`）：

```
Treatment-effects estimation                    Number of obs     =      1,200
Estimator      : regression adjustment
Outcome model  : linear
Treatment model: none
```

westata 的 `TreatmentEffectResult.__str__`（`estimators/treatreg_effects.py:132-144`）
打印的是**完全不同的自有格式**：

```
Treatment-effects estimation
  ATE    -0.009573963   se 0.02461756   z   -0.39   p 0.6973   [-0.05782349, 0.03867557]
  POmean(0) -0.007750909   se 0.01655147
  POmean(1) -0.01732487   se 0.04098977
```

**数据都在**（`potential_outcomes`、`estimand`、`se_method`），缺的是
`Estimator / Outcome model / Treatment model` 三行与 Stata 的表头列名
（`Coefficient / std. err. / z / P>|z|`）。
注意这个类**不走 `StataResults`**，所以 `summary()` 完全不可用——
它是唯一一个没有 `e()` 也没有 statsmodels 兼容面的结果对象。

### 1.8 `manova` 只打印 Wilks' λ，Stata 打印四种统计量 + 完整表

Stata（`p10_outputs.txt:125-139`）：

```
                        W = Wilks' lambda      L = Lawley-Hotelling trace
                        P = Pillai's trace     R = Roy's largest root

               Source | Statistic        df    F(df1,     df2) =   F   Prob>F
           -----------+-------------------------------------------------------
                    g |W   0.4290         2      4.0    472.0    62.15 0.0000 e
                      |P   0.5710                4.0    474.0    47.35 0.0000 a
                      |L   1.3307                4.0    470.0    78.18 0.0000 a
                      |R   1.3306                2.0    237.0   157.67 0.0000 u
                      |-------------------------------------------------------
             Residual |                 237
           -----------+-------------------------------------------------------
                Total |                 239
           -------------------------------------------------------------------
                        e = exact, a = approximate, u = upper bound on F
```

westata：`ManovaResult.__str__`（`estimators/systems.py:1051-1057`）只打印
Wilks' λ 与 Rao F 两行；`Manova` 的 docstring（`systems.py:1065-1066`）
**明确声明** Pillai / Lawley-Hotelling / Roy `raise UnsupportedError`。
`ManovaResult.statistics` 字典（`systems.py:1203-1207`）只装
`wilks_lambda`/`s`/`exact`。

**注意：这条 GAP_REGISTER §3.5 已列为 ⬜，不是新发现。**
但本次补上了 Stata 的**完整输出形态证据**（四行统计量各自的 df1/df2 与
exact/approximate/upper-bound 标记），说明这不是"补三个数"而是
"补一张表 + 三种不同的 F 近似"。

### 1.9 `nl` 的 ANOVA 表完全没渲染

Stata（`p10_outputs.txt:97-102`）：

```
       Source |      SS            df       MS
-------------+----------------------------------    Number of obs =        300
        Model |  1010.4869          2  505.243457    R-squared     =     0.9932
     Residual |  6.9274269        298  .023246399    Adj R-squared =     0.9931
-------------+----------------------------------    Root MSE      =   .1524677
        Total |  1017.4143        300  3.39138114    Res. dev.     =  -279.1251
```

westata：`nl` 的 `e()` 有 `rss`/`mss`/`tss`（`estimators/nonlinear.py:1332-1334`），
**但 ANOVA 三行（Model/Residual/Total 的 SS/df/MS）没有任何渲染路径**，
`summary.py` 只打印 R²/RMSE。`Res. dev.` 那一行也没有
（`e(deviance)` 有，`nonlinear.py:1340`）。

### 1.10 `sureg`/`mvreg` 的方程表没有渲染

Stata（`p10_outputs.txt:7`、`:63`）：

```
Equation             Obs   Params         RMSE  "R-squared"      chi2   P>chi2
Equation             Obs   Parms        RMSE    "R-sq"          F      P>F
```

这是 `sureg`/`mvreg` **特有的顶部方程汇总表**（每方程一行），
westata 的 `etable` 是跨模型的列式表，`summary()` 是单模型，
**两者都不对应 Stata 的这个形态**。

**但数据全在 `e()` 里，这是纯渲染缺口**（我读了源码，可以下结论）：

- `sureg`：`estimators/systems.py:538-588` 存了
  `k_eq`、`ll`，以及每方程的 `rss_{i}`、`mss_{i}`、`r2_{i}`、`rmse_{i}`、
  `df_m_{i}`、`chi2_{i}`、`p_{i}`（位置式与标签式双拼写，见 580-582 行注释）。
- `mvreg`：`estimators/systems.py:741-767` 存了 `k_eq`、`sigma`、`corr`，
  并按 `p10_outputs.txt` 的需求把每方程拟合值存成 **macro 形态的列表**
  （`e(r2) = " .8134 .6004"`，见 753-757 行注释），另有 `r2_y1` 式标量别名。
- `e(sigma)` 与 `e(corr)` 存成**矩阵的 `.tolist()`**（547-548、749-750 行），
  这正是 Stata 的 `e(sigma)`/`e(corr)` 形态。

**结论：Stata 那张 `Equation Obs Params RMSE "R-squared" chi2 P>chi2` 表，
westata 需要的一切数都已在 `e()` 中，缺的只是一个渲染器。**

### 1.3 `etable` 的一个具体不一致

`report/etable.py:144` 读 `stats` 键时走 `res.e`，所以只要 `e()` 补上 `tss`/`rmse`/`ll`，
`etable` 立刻可用。**这意味着 D6 的修复是"存一个键"而非"写一个渲染器"。**

---

## 2. 跨家族共性缺口（三个子报告的公约数）

以下条目在多个家族重复出现，属于**架构级**而非某命令的疏漏：

### C1. ML 模型的 `e(k)` / `e(k_dv)` / `e(k_eq)` / `e(k_eq_model)` / `e(k_aux)` 体系

Stata 对每个 ML 模型都存这套参数计数（fixture 实证：`logit_sim.json` 有
`k`/`k_dv`/`k_eq`/`k_eq_model`/`k_aux`/`k_autoCns`；`nbreg.json` 还有 `k_aux`；
`ologit.json` 有 `k_cat`）。westata 全库只存 `rank`（且语义是 `len(params)`，
与 Stata 的 `e(rank)` 不一定同义）。

**这条不只是"少几个键"**：`retest/domains/verify/AUDIT.md` D8 记录了一个真实 bug：
Stata `nbreg` 报 `e(df_m)=2, e(rank)=4, e(k_aux)=1`，而 westata 的 `e['df_m']` 是 **3**
（把辅助参数算进了 `df_m`）。**缺 `k_aux` 与 `df_m` 口径不统一是同一个病的两个症状。**

### C2. `e(ic)` 与 `e(aic)`/`e(bic)` 的命名分裂

Stata 存 `e(ic)`（AIC）与 `e(ic)` 相关的一套，`estat ic` 显示 AIC/BIC。
westata 的 ML 估计器存的是 `aic`/`bic`（`discrete.py`/`counts.py` 扫描确认）。
fixture 里 `logit_sim.json` 只有 `ic`，没有 `aic`/`bic`。

**这是契约命名不一致，不是数值问题**：一个从 Stata 迁移的脚本写 `e.ic` 会 KeyError。
需要在 `e()` 里同时提供（或文档明确映射）。

### C3. `e(converged)` / `e(rc)` / `e(crittype)` / `e(ml_method)` / `e(technique)` / `e(opt)`

Stata 的 ML fixture 全带这套（`logit_sim.json` macros 有 `crittype`/`ml_method`/`opt`/`technique`/`singularHmethod`/`deriv_useminbound`）。westata 只有部分估计器有 `converged`（`discrete.py`/`tobit.py`/`tseries.py` 有），且**没有 `rc`、没有优化器元数据**。

**可复现性影响**：`arch` 的 E2 归因（GAP_REGISTER §8.9）正是"Stata 用自己的停止点"。
没有 `e(crittype)`/`e(technique)`，使用者无法判断两个库是否该逐位一致。

### C4. `e(properties)` / `e(predict)` / `e(cmd)` / `e(model)` / `e(title)` 等 macros

Stata 的每个估计结果都带这组字符串（fixture macros 列实证）。westata 的 `EReturn`
只装标量；`model_name`/`cov_type` 是 Python 属性，不在 `e` 里。
**后果**：`etable` 无法按 Stata 惯例自动取列名；用户必须手写 `model_names=`。

### C5. `predict` 的诊断类型全面缺失

`postestimation/predict.py:50-81` 的 `SUPPORTED_TYPES` 表覆盖 `xb`/`stdp`/`stdf`/`pr`/`n`/`residuals`/`fitted`/`ue`/`e`/`ystar`。
Stata `regress` 后可用但 westata 没有的类型：

`cooksd`、`cd`（Cook 距离）、`dfbeta`、`dfbetas`、`hat`（杠杆）、`rstandard`、`rstudent`、
`stdresid`（标准化残差）、`scores`（得分向量）、`lev`（leverage 别名）、`resid`（有，作 `residuals`）。

`logit` 后 Stata 有 `cooksd`/`dfbeta`/`dx2`/`d2`/`hat`/`rstandard`/`rstudent`/`score`，全部缺失。

**注意**：`stdp`/`stdf` 在 P5 已 E3 对拍（GAP_REGISTER §2），所以这不是"没做过后估计"，
而是**诊断类 `predict` 整块没做**。

### C6. `estat` 家族整体缺席

GAP_REGISTER §2 自己列了 `estat vce`/`summarize`/`firststage`/`classification` 未实现。
本次盘点的补充：

| `estat` 子命令 | Stata 用途 | westata | 备注 |
|---|---|---|---|
| `estat vce` | 打印 `e(V)` | ❌ | 见 D1 |
| `estat vce, correlation` | 相关系数矩阵 | ❌ | 见 D2 |
| `estat ic` | AIC/BIC | ❌ | 见 D3 |
| `estat summarize` | 估计样本描述统计 | ❌ | 见 D7 |
| `estat firststage` | IV 第一阶段 | ❌ | 见 D8 |
| `estat gof` | Poisson/NB 拟合优度（deviance/Pearson） | ❌ | 子报告 2 复核 |
| `estat hettest` | Breusch-Pagan | ❌ | |
| `estat ovtest` | Ramsey RESET | ❌ | |
| `estat vif` | 方差膨胀因子 | ❌ | |
| `estat szroeter` | 异方差检验 | ❌ | |
| `estat imtest` | 信息矩阵检验 | ❌ | |
| `estat classification` | 分类表（logit/probit） | ❌ | |
| `estat Pearson` / `estat Spearman` | 相关 | ❌ | |
| `linktest` | 模型设定检验 | ❌ | GAP_REGISTER §2 已列 |

**共 14 个 `estat` 入口，westata 有 0 个同名 API。** 现有的是 `results.test()`/`lincom()`/
`lrtest()`/`hausman()`/`contrast()`/`margins()`/`nlcom()`/`testnl()`——这些对应的是
**独立命令**，不是 `estat` 子命令。

### C7. 残差诊断与影响分析整块缺失

实测 `OLS` 结果对象：`ssr`/`ess`/`centered_tss`/`uncentered_tss`/`mse_model`/`mse_resid`/
`mse_total`/`scale`/`durbin_watson`/`omnibus`/`jb`/`skew`/`kurtosis`/`condition_number`/
`normalized_cov_params`/`resid_pearson`/`wresid`/`influence`/`get_influence` **全部 `False`**。

statsmodels 全有。**这是 statsmodels 兼容面的缺口**——`westata` 声称
"implements the surface statsmodels downstream code expects"（`results/base.py:1-11` 的 docstring），
但残差诊断这块没有跟上。

---

## 3. 优先级建议（按"成本/价值比"排序）

### 第一优先：零成本可修（只差一个字典键，数据已在内存里）

| 项 | 改动 | 收益 |
|---|---|---|
| `OLS` 存 `tss` | `linear.py:311-327` 加一行 | 与 `areg`/`reghdfe`/`xtreg` 的 Stata 行为对齐 |
| `reghdfe` 存 `tss`/`tss_within`/`mss`/`rss` | `hdfe.py:536-546` 加三行 | Stata fixture 实证有这三个键 |
| `svy: regress` 存 `tss` | `survey.py:1055,1071` | 同上 |
| `xtreg` 存 `tss` | `panel.py:741-754` | `xtreg_fe.json`/`xtreg_be.json` 实证有 |
| `xtivreg` 存 `tss` | `panel_iv.py:1551` | 已算出 |
| ML 模型补 `k`/`k_dv`/`k_eq`/`k_aux` | `discrete.py`/`counts.py`/`limited.py` | 关闭 AUDIT D8/D9 那类 `df_m` 口径 bug 的**根因** |
| ML 模型补 `ic`（与 `aic`/`bic` 并存） | 同上 | 关闭 `estat ic` 迁移断层 |

### 第二优先：需要新渲染器，但数据 100% 就绪

| 项 | 说明 |
|---|---|
| `estat vce` | 一个 30 行的矩阵打印器，读 `cov_params()` |
| `estat vce, correlation` | 同一个打印器 + `D⁻¹ᐟ² V D⁻¹ᐟ²` |
| `estat ic` | 一个表：Model/Obs/ll/null ll/AIC/BIC |
| `summary(stata)` 补 AIC/BIC 行 | ML 模型 Stata 不打印这两行，但 `estat ic` 用 |
| `etable` 默认 stats 扩展 | `_DEFAULT_STATS` 加 `rmse`/`ll`，让 `etable` 直接可用 |
| `e()` 补 macros（`cmd`/`model`/`title`/`predict`/`properties`/`vce`） | 让 `etable` 能自动命名列 |

### 第三优先：需要新计算（工作量以天计）

| 项 | 说明 |
|---|---|
| `predict` 诊断类型（`cooksd`/`dfbeta`/`hat`/`rstudent`/`scores`…） | 每族一套公式，需要对拍 |
| `estat hettest`/`ovtest`/`vif`/`imtest` | 各有明确公式，可对拍 |
| `estat gof`（deviance/Pearson） | Poisson/NB 专用 |
| `estat firststage` 完整版（Shea partial R²） | IV 专用 |
| 残差诊断属性（`omnibus`/`jb`/`skew`/`kurtosis`/`durbin_watson`/`condition_number`） | statsmodels 兼容面 |

### 明确不建议做（或需先讨论）

| 项 | 理由 |
|---|---|
| 复刻 Stata 的完整 `estat` 家族（14 个全做） | 应按**使用频率**取子集。建议先做 `vce`/`ic`/`gof`/`hettest`，其余按用户反馈 |
| `marginsplot` | GAP_REGISTER §2 已判 ❌，维持 |
| 逐位复刻 Stata ML 停止点以获得 `crittype`/`opt` | GAP_REGISTER §8.11/§8.12 已论证这是错误方向；应改为**记录**我们的收敛信息而非追逐它的 |

---

## 4. 与现有文档的关系（需要同步修改）

- `docs/GAP_REGISTER.md` §2 已列 `estat vce`/`summarize`/`firststage`/`classification` 未实现，
  但**没有记录 `tss` 的系统性缺失**、**没有记录 `k_aux`/`ic` 的契约分裂**、
  **没有记录 `estat` 家族共 14 个入口全缺**。建议新增一节。
- `docs/GAP_REGISTER.md` §8.2 说 `frontier` 的 `/lnsig2v`、`/lnsig2u` 已在 P11 修复——
  子报告 3 会复核当前源码状态（我未亲自读该文件的 `e` 字典，**不下结论**）。
- `README.md:156-168` 的命令覆盖表**没有一行提到 `estat`/`vce`/`ic`**，
  读者会以为后估计已完整。

## 5. 方法论备注

本次盘点的三条判断依据，按可靠性排序：

1. **Stata 自己的 fixture**（`tests/conformance/stata_fixtures/*.json` 的 `scalars`/`macros`）
   —— 最高证据，是 `pystata` 从活体 StataNow 19.5 抓的 `ereturn list` 全量。
2. **Stata 控制台输出的抄录**（`*.txt`）—— 次高，证明"Stata 打印什么"。
3. **手册知识** —— 仅在没有前两者时使用，且必须标注。

**一条重要的边界**：fixture 里没有某个键，**不等于** Stata 不存它。
`regress_auto.json` 没有 `tss` 而 `areg.json` 有，正说明这点。
判"缺失"必须逐命令找证据，不能用一个 fixture 推全部。这正是用户指认 1 需要小心的地方：
**`regress` 不存 `e(tss)` 是符合 Stata 的，但 `reghdfe`/`xtreg`/`areg` 不存是真缺口。**
