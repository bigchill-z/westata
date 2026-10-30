# westata 辅助指标与展示功能缺口：汇总报告（Lead 总装）

> **实施状态**：
> **第 1 轮**：`estat vce`/`estat vce, correlation`、6 个命令的 `e()` 补键、
> `summary()` 按估计器分支（修掉 qreg/svy/arima 三处渲染缺陷）。
> **第 2 轮**：`xtreg` 方差分量块（fe 12→28、re 12→18、be 12→20 键），
> 依 `xtreg_fe.ado:358-430` 实现。
> **第 3 轮**：ML 的 `e(ic)`（面板族原来存的是迭代次数 → 改为 `k+1`）、
> `k*` 块、`ologit`/`oprobit`/`heckman ml`/`etregress ml` 的信息准则。
> **第 4 轮**：`estat gof`（Poisson 拟合优度，公式抄自 `poisgof.ado`）。
> **第 5 轮**：`estat vif`（方差膨胀因子，公式抄自 `vif.ado`）。
> **第 6 轮**：`estat hettest`（Breusch-Pagan/Cook-Weisberg，公式抄自 `hettest.ado`）。
> **第 7 轮**：`estat ovtest`（Ramsey RESET，公式抄自 `ovtest.ado`）。
> **第 8 轮**：`predict` 影响诊断量（`hat`/`leverage`/`rstandard`/`rstudent`/`cooksd`/`dfbeta`）。
> **第 9 轮**：`estat ic`（AIC/BIC/AICc/CAIC，公式抄自 `estimates.ado`），并纠正 `e(aic)` 的注释。
> **第 10 轮**：`estat firststage`（IV 第一阶段 + Shea's partial R²，抄自 `ivregress_estat.ado`）。
> **第 11 轮**：`estat overid`（Sargan/Basmann/Hansen J，抄自 `ivregress_estat.ado`）。
> **第 12 轮**：`estat endogenous`（Durbin-Wu，抄自 `Endog2sls`）。
> **第 13 轮**：`xtreg` 三个估计器 **100% 覆盖** fixture（fe 37/37、re 20/20、be 23/23）。
> **第 14 轮**：`predict` 剩余四个诊断量（`stdr`/`dfits`/`welsch`/`covratio`）。
> **第 15 轮**：`estat classification`（2×2 分类表 + 9 个百分比，抄自 `lstat.ado`）。
> **第 16 轮**：`estat imtest`（Cameron-Trivedi IM 检验，抄自 `imtest.ado`）。
> **第 17 轮**：`estat summarize`（估计样本描述统计，抄自 `estat_summ.ado`）。
> **第 18 轮**：ML 共用 `k*` 块 5 键 + `areg` 补充（poisson/ologit/oprobit 达 100%）。
> **第 19 轮**：`var` 的 `e()`（36/36 全覆盖，含 AIC 必须用 Stata ll 重算的教训）。
> **第 20 轮**：`teffects` 的头部块与 `.e`（四个估计器的 Estimator/Outcome/Treatment 标签）。
> **第 21 轮**：r-class 检验结果的 `r()` 映射（`dfuller`/`pperron`/`kpss`/`xtunitroot`）。
> **第 22 轮**：`areg` 的 `F_absorb`/`p_absorb`/`r2_c`（收回第 18 轮的"已知缺口"，areg 达 22/22）。
> **第 23 轮**：`reghdfe` 的 within 分解块（`tss_within`/`r2_within`/`r2_a_within`，17/29 → 21/29）。
> **第 24 轮**：`logit`/`probit` 的 `N_cds`/`N_cdf`（两个估计器达 20/20）。
> **第 25 轮**：`nbreg` 的 `ll_c`/`chi2_c`/`rank0`（达 23/23），并修复负数卡方被原样报出的既有 bug。
> **第 26 轮**：`estat hettest, mtest()` 多重检验校正（bonferroni/holm/sidak）。
> **第 27 轮**：`estat overid` 的 Wooldridge score（稳健/HAC），收回第 12 轮的"直接报错"。
> **第 28 轮**：修正该 score 的 HAC prewhitening 样本处理（零填充 → 丢弃前 lags 行），并标注 HAC 路径不可达。
> **第 29 轮**：实现 Newey-West HAC 协方差（`westata.vce.hac`），`lag(0) == HC1` 已验证。
> **第 30 轮**：把 HAC 接入 `IV2SLS`，修正 `N/(N−K)` 因子，prewhitening 端到端可达。
> **第 31 轮**：修复 LIML/GMM 的裸 KeyError 回归；澄清 `estat summarize` 的 aweight sd 随权重尺度变化（已知限制，故意不猜修）。
> **第 32 轮**：`estat endogenous` 的稳健/HAC score 检验（`robwool`/`acwool`），收回第 12 轮的"直接报错"。
> **第 33 轮**：`regF`（增广回归内生性检验），解锁 cluster；修正一个把 F 算成 0 的设计错误。
> **第 34 轮**：`ManovaResult` 与 `JohansenResult` 的 `e()`（manova 8 键、vecrank 18 键）。
> **第 35 轮**：`reghdfe` 补齐 → **13 个估计器全部 100% 覆盖 fixture**；修正 `df_a` 命令间语义差异与一个测试自证缺陷。
> **第 36 轮**：披露验证面的真实边界（21 个 fixture 中仅 3 个可数值对拍），新增 5 条守护测试。
> **第 37 轮**：新增 `stata_params`/`stata_cov_params`：修正系数排列顺序与 Stata `e(b)` 不一致导致的误判陷阱。
> **第 38 轮**：修正上一轮的因子变量边界——显式 `C()` 现在拒绝而非编造不存在的 `e(b)` 行。
> **第 39 轮**：修正多水平裸变量的重复命名错误（四个系数共享一个标签）。
> **第 40 轮**：系统扫描 17 种模型形态，修正交互项分隔符 `:` → `#`。
> 测试 1458 → **1896 passing**，零回归。详见 `CHANGELOG.md` 与
> `docs/GAP_REGISTER.md` §8.20。本文件余下内容仍是**待办清单**。


> **本文件是四份盘点的结论汇总。** 详细证据在：
> - `retest/gaps/lead_cross_cutting.md` —— Lead 主线：跨家族共性 + 展示层
> - `retest/gaps/linear_iv_ts_gaps.md` —— 线性/IV/HDFE/面板/时序（522 行）
> - `retest/gaps/discrete_limited_gaps.md` —— 离散/计数/受限/面板非线性（620 行）
> - `retest/gaps/quantile_systems_gaps.md` —— 分位数/系统/因果/抽样/重抽样/DID（614 行）
>
> 全部断言有 `path:line` 或实测输出支撑。方法：读源码 → 读 Stata  fixture → 实跑拟合 → 求差。

---

## 1. 先纠正三个指认（用户报告"半对"，且对错两边修法不同）

### 指认 1："RSS、TSS 在 westata 结果里不存在" —— **对 `regress` 是错的，对 `xtreg`/`areg`/`reghdfe` 是对的**

这是本次盘点最需要小心的地方，因为**用一个 fixture 推全部会得出错误结论**。

| 命令 | Stata 存 `e(tss)`? | 证据 | westata 现状 |
|---|---|---|---|
| `regress` | **不存** | `regress_auto.json` / `_robust` / `_factor` 三个 fixture 的 `scalars` 都只有 `rss`+`mss`，无 `tss` | **已符合 Stata**（`linear.py:322-323` 存 `rss`/`mss`） |
| `ivregress 2sls` | 存 | `iv.py:821-823` 注释已记录对拍 | ✅ 已存（`iv.py:826`） |
| `areg` | **存** | `areg.json` 有 `tss` | ❌ 缺 |
| `reghdfe` | **存**（外加 `tss_within`） | 5 个 `reghdfe_*.json` 全有 | ❌ 缺 |
| `xtreg, fe` | **存** | `xtreg_fe.json` 有 `tss`/`mss` | ❌ 缺 |
| `xtreg, be` | **存 `mss`** | `xtreg_be.json` 有 `mss` | ❌ 缺 |
| `xtgls` / `xtpcse` | 存 | p9 输出 + 已算 | ❌ 缺 |

**`rss` 反而大面积已有**（OLS/IV/nl/qreg/rreg/sureg/mvreg/xtreg/xtivreg/xtabond 都有）。

**关键点：这不是"漏了一个键"，是三件不同的事：**

1. **`regress` 不缺** —— 用户看到的"没有 TSS"在 `regress` 上是**符合 Stata 的**。给 `regress` 加 `e(tss)` 是存一个 Stata 没有的数。
2. **`areg`/`reghdfe`/`xtreg fe` 是真缺口** —— 而且**数已经算出来了被丢掉**：
   `hdfe.py:522` 算出 `tss`，`linear.py:249,255` 算出 `tss`（只用它推 `mss`），
   `survey.py:1056` 算出 `tss`，`panel_iv.py:1524` 算出 `tss`。**四处都是"算了未存"。**
3. **`xtreg` 四个估计器返回完全相同的 12 个键** —— 我独立复跑确认：
   ```
   xtreg fe : ['F','N','df_b','df_m','df_r','ll','n_entities','r2','r2_a','rank','rmse','rss']
   xtreg re : （同上，逐字节相同）
   xtreg be : （同上）
   xtreg fd : （同上）
   ```
   而 `xtreg_fe.json` 有 **38 个** `e()` 键，`xtreg_re.json` 有 **20 个**。
   这是**全库最大的单块 `e()` 缺口**。

### 指认 2："打印回归系数的协方差矩阵" —— **完全成立，是纯展示缺口**

- `results.cov_params()` 对每个估计器都返回完整 `k×k` 矩阵（`results/base.py:203-206`），实测 shape `(3,3)`。
- **全库没有任何渲染路径**：`src/` 下搜 `estat` 只命中注释；`summary.py` 两个 renderer 都不碰 `cov`；
  `etable.py:147` 只能吐标量。
- **零新数学**：矩阵、参数名、`dist` 全在结果对象上，缺的只是一个矩阵打印器 + 入口。
- Stata 的两个形态：`estat vce`（协方差）与 `estat vce, correlation`（`D⁻¹ᐟ²VD⁻¹ᐟ²`）。

### 指认 3："还有很多功能缺少" —— 成立，见 §3–§5

---

## 2. 最大的单块缺口：`xtreg` 的方差分量块（16 个键）

`xtreg_fe.json` 有而 `xtreg` 四个估计器都没有的键：

`tss`、`tss_within`、`mss`、`r2_w`、`r2_b`、`r2_o`、`r2_c`、`sigma_e`、`sigma_u`、`sigma`、`rho`、`theta`、`corr`、`N_g`、`Tbar`、`Tcon`、`g_min/g_avg/g_max`、`k_absorb`、`df_a`、`F_f`、`p_f`、`ui`、`chi2`、`p`。

**三个键是"同名不同值"陷阱**（`retest/domains/panel/NOTES.md:314` 已实测 1e-16）：
- `e["r2"]` 实际上就是 Stata 的 `e(r2_w)`（within R²）
- `e["rmse"]` 实际上就是 Stata 的 `e(sigma_e)`
- `e["n_entities"]` 实际上就是 Stata 的 `e(N_g)`

**所以 `e["r2_w"]` 返回 `None` 而不是 within R² —— 一个从 Stata 迁移的脚本会静默拿到空值。**

### 这条缺口为什么便宜：`xtivreg` 已经全做了

同一个文件 `panel_iv.py:864-903` 的 `xtivreg` 存了 `xtreg` 缺的**全部**：
`r2_w`、`r2_b`、`r2_o`、`sigma_u`、`sigma_e`、`rho`、`corr`、`N_g`、`g_min/g_avg/g_max`、`theta`、`chi2type`、`p`。

**同文件、同后端族、同一套 Swamy–Arora 机制。这是搬运，不是新研究。**

---

## 3. 展示层：一个渲染器对 Stata 的 11 种头部块形态

### 3.1 结构性缺陷：按 `e(r2_p)` 键名判断模型类别

`results/summary.py:216-224`：

```python
pseudo_r2 = _e_float(result, "r2_p")
if np.isfinite(pseudo_r2):
    → 打印 Log likelihood + Pseudo R2
else:
    → 打印 R-squared / Adj R-squared / Root MSE
```

**这个判断是错的**，因为 `r2_p` 一个键被两种语义共用：
- ML 的 McFadden 伪 R²
- `qreg` 的 check-function 伪 R²（`quantile.py:417`）

**实测后果（我独立复跑确认）**——`qreg`：

```
Number of obs         =        500
F(1, 498)             =          .      ← Stata 不打印 F
Prob > F              =          .
Log likelihood        =          .      ← Stata 不打印，qreg 不是 ML 模型
Pseudo R2             =     0.0934
```

而 Stata（`p8_outputs.txt:20-22`）：

```
.25 Quantile regression                    Number of obs =        600
  Raw sum of deviations 438.3735 (about -.46158607)
  Min sum of deviations 156.7522           Pseudo R2     =     0.6424
```

**修法：按 `model_name` 分支，不按键名分支。** 这是一次性修好所有家族的根因。

### 3.2 `svy:` 家族：五个数全在 `e()` 里，渲染的是一堆无意义的行

实测 `svy: mean`：

```
Number of obs         =        576
F(0, 40)              =          .       ← 无意义
Prob > F              =          .
R-squared             =          .       ← 无意义
Adj R-squared         =          .       ← 无意义
Root MSE              =     4.6094       ← 语义错误（见下）
```

Stata（`p12_outputs.txt:6-10`）：

```
Survey: Mean estimation

Number of strata =  8             Number of obs   =        576
Number of PSUs   = 48             Population size = 837.592551
                                   Design df       =         40
```

**`N_strata`/`N_psu`/`N_pop`/`df_r` 四个数全在 `e()` 里**（`survey.py:636-640`），
**缺的只是渲染**。

**`Root MSE = 4.6094` 这一行比"缺一行"更严重**：`summary.py:140-144` 的兜底在 `e(rmse)` 缺失时
用 `sqrt(rss/df_resid)` 现算。对 `svy: mean`，`resid = y - mean(y)`，`df_resid = design df = 40`，
于是得到一个**既不是 RMSE 也不是 sd(y) 的数**（与 sd(y)≈4.6 数值巧合接近，语义完全错误）。

**这条按本项目纪律属于"必须修"**：宁可打印 `.`，也不能打印一个看起来合理、实际语义错误的数字。
`svy: total` 更糟——实测打印 `Root MSE = 0.0000`。

### 3.3 其余头部块形态缺口（数据大部分已就绪）

| Stata 形态 | 数据就绪度 | 证据 |
|---|---|---|
| `qreg` 式（Raw/Min sum + Pseudo R² 右置 + quantile 标题） | **100%** | `quantile.py:415-416` 已存 `sum_rdev`/`sum_adev`/`q_v` |
| `svy` 式（strata/PSU/obs/pop/design df 五行） | **100%** | `survey.py:636-640` |
| `sureg`/`mvreg` 式（每方程一行 `Obs Params RMSE R² chi2 P>chi2`） | **100%** | `systems.py:580-588`（sureg）、`:763-787`（mvreg）已存 `r2_1`/`rmse_1`/`chi2_1`/`p_1` 等 |
| `frontier` 式（表内 4 行派生统计 + LR 测试行） | **100%** | `nonlinear.py:1637-1646` 已存 `lr_chi2`/`p_lr`/`sigma_v`/`sigma_u`/`sigma2`/`lambda` |
| `bootstrap`/`jackknife` 式（`Replications` 行 + `Observed/Bootstrap` 列头） | 90% | `e(reps)` 已有（bootstrap/permute）；jackknife 缺 |
| `didregress` 式（time-info 表 + `Data type:` + `Note:`） | 70% | `e(data_type)` 已存；group/time 最小最大期未暴露 |
| `teffects` 式（`Estimator:`/`Outcome model:`/`Treatment model:`） | 60% | `n = y.size` 在 `fit()` 里现成但没进结果 |
| `nl` 式（ANOVA 表 `Source/SS/df/MS`） | **100%** | `nonlinear.py:1332-1334` 已存 `rss`/`mss`/`tss` |
| `manova` 式（四统计量 + `e/a/u` 图例） | 25% | 只有 Wilks' λ；`statistics["exact"]` 已算但未渲染 |
| `xtreg` 式（方差分量 + group header） | 30% | `xtivreg` 已有，`xtreg` 无 |
| `arima`/`arch`/`xtgls` 的 `R-squared = .` 噪音行 | — | Stata 根本不打印 R² 块，westata 打三个点 |

---

## 4. `estat` 家族：14 个入口，westata 有 0 个

| `estat` 子命令 | 用途 | 状态 |
|---|---|---|
| `estat vce` | 打印 `e(V)` 系数协方差矩阵 | ❌ **数据 100% 就绪** |
| `estat vce, correlation` | 相关系数矩阵 | ❌ **数据 100% 就绪** |
| `estat ic` | AIC/BIC 表 | ❌ 数据部分就绪（见 §5 的 `e(ic)` 陷阱） |
| `estat gof` | Poisson/NB 拟合优度（deviance/Pearson） | ❌ 需新计算 |
| `estat hettest` | Breusch-Pagan | ❌ 需新计算 |
| `estat ovtest` | Ramsey RESET | ❌ |
| `estat vif` | 方差膨胀因子 | ❌ |
| `estat imtest` | 信息矩阵检验 | ❌ |
| `estat summarize` | 估计样本描述统计 | ❌ |
| `estat firststage` | IV 第一阶段诊断表 | ⚠️ `e(F_first)` 有（`iv.py:807`），无展示 |
| `estat classification` | 分类表（logit/probit） | ❌ |
| `estat Pearson`/`Spearman` | 相关 | ❌ |
| `linktest` | 模型设定检验 | ❌（GAP_REGISTER §2 已列） |

现有 API（`results.test/lincom/lrtest/hausman/contrast/margins/nlcom/testnl`）对应的是
**独立命令**，不是 `estat` 子命令——`GAP_REGISTER.md:81` 已承认 `estat vce`/`summarize`/`firststage`/`classification` 未实现，
但**没记录共 14 个入口全缺**，也没记录 `vce`/`ic`/`gof` 是最便宜的三条。

---

## 5. 三个契约级缺陷（比"缺键"更危险）

### 5.1 `e(ic)` 在面板家族是**迭代次数**，不是 AIC 惩罚项

`panel_nonlinear.py:725` 设 `"ic": int(iterations)`。实测：`xtlogit fe ic=7`、`xtlogit re ic=156`、
`xttobit ic=162`、`xtpoisson re ic=116`。

而 Stata 的 `e(ic)` 是"参数数 + 1"的 AIC 惩罚项（fixture 实证：`logit_sim.json:27` 的 `ic=4.0`）。

**一个把 Stata `estat ic` 逻辑（`-2*ll + 2*e(ic)`）移植过来的调用者会得到毫无意义的数，
而且不会报错。** 这比键缺失更糟。

### 5.2 `e(aic)`/`e(bic)` vs `e(ic)` 的命名分裂

- Stata 存 `e(ic)`（6 个 ML fixture 全部实证），`estat ic` 现场算 AIC/BIC 进 `r(S)`。
- westata 的 ML 估计器存 `aic`/`bic`，**不存 `ic`**（`discrete.py:616-617`、`counts.py:572-573`、
  `limited.py:533-534`、`panel_nonlinear.py:739-741`）。
- **`ologit`/`oprobit`/`heckman`/`etregress` 连 `aic`/`bic` 都没有**——`ologit` 的
  `ll`/`rank`/`N` 都在 `e()` 里，这是全审计最便宜的修复之一。

### 5.3 `ologit`/`oprobit` 的切点用 `/cut1` 而非 Stata 的 `cut1`

`discrete.py:1541` 生成 `/cut{i}` 前缀；Stata fixture（`ologit.json:80-85`）用裸 `cut1`。
`/lnalpha` 的斜杠前缀**是对的**（匹配 Stata），但切点的斜杠**是错的**。
后果：`r.params["cut1"]`  KeyError。

### 5.4 相关的：ML 的 `k*` 块全缺

`k`、`k_dv`、`k_eq`、`k_eq_model`、`k_aux`（部分有）、`k_autoCns`、`consonly`、`noconstant`、`rc`。
**这不是"少几个键"**：`retest/domains/verify/AUDIT.md` D8 记录了一个真 bug——
Stata `nbreg` 报 `e(df_m)=2, e(rank)=4, e(k_aux)=1`，而 westata 的 `e['df_m']` 是 **3**
（把辅助参数算进了 `df_m`）。**缺 `k_aux` 与 `df_m` 口径不统一是同一个病的两个症状。**

---

## 6. 优先级建议（按"成本/价值比"）

### 第一优先：零成本，只差一个字典键（数据已在内存里）

| # | 项 | 改动点 | 收益 |
|---|---|---|---|
| 1 | `xtreg` 四个估计器补方差分量块（16 键） | `panel.py:741-757`，从 `panel_iv.py:864-903` 搬运 | 关闭全库最大 `e()` 缺口 |
| 2 | `tss`：`areg`/`reghdfe`/`xtreg`/`xtgls`/`xtpcse`/`svy: regress` | `hdfe.py:536`、`panel.py:741`、`survey.py:1071`、`panel_iv.py:1551` | 关闭用户指认 1 的真缺口部分 |
| 3 | `tss_within`/`r2_within`/`r2_a_within` | `hdfe.py:522` | reghdfe fixture 实证有 |
| 4 | `mss`：`areg`/`reghdfe`/`xtreg fe/be`/`xtgls` | `tss - rss`，两个都有了 | |
| 5 | ML 补 `k`/`k_dv`/`k_eq`/`k_eq_model`/`k_aux` | `discrete.py`/`counts.py`/`limited.py` 四个装配点 | 关闭 §5.4 的 `df_m` 口径 bug 根因 |
| 6 | `ologit`/`oprobit` 补 `aic`/`bic`/`converged` | `discrete.py:1562-1583` | 全审计最便宜的修复之一 |
| 7 | `heckman`/`etregress` 补 `aic`/`bic` | `limited.py` 四个装配点 | |
| 8 | NB2 补 `e(alpha)` = `exp(/lnalpha)` | `counts.py:580` 已有同样写法 | |
| 9 | **面板家族 `e(ic)` 改名**（`ic` → `iterations`） | `panel_nonlinear.py:725` | 关闭 §5.1 的静默错误 |
| 10 | `ologit` 切点改名 `/cut1` → `cut1` | `discrete.py:1541` | 关闭 §5.3 |
| 11 | `teffects` 补 `nobs` | `treatreg_effects.py:1213` 的 `n` 现成 | |
| 12 | `jackknife` 补 `e(reps)` | `resample.py:477-486` | |
| 13 | `bootstrap`/`jackknife` 把 `rmse`/`rss`/`mss`/`r2` 传出 | `resample.py:269-277` → `346-353` | |
| 14 | `sqreg`/`bsqreg` 补 `sum_adev`/`sum_rdev`/`r2_p` | 公式已在 `quantile.py:394-403` | |
| 15 | `mvreg` 补 `e(ll)` | 公式同 `systems.py:555-560` | |

### 第二优先：新渲染器，但数据 100% 就绪

| # | 项 | 说明 |
|---|---|---|
| 16 | **`estat vce` + `estat vce, correlation`** | 读 `cov_params()`；用户指认 2 |
| 17 | **`summary.py` 按 `model_name` 分支替代按 `r2_p` 分支** | **一次性修好所有家族的根因**（§3.1） |
| 18 | `svy:` 头部块（5 行 + 标题） | 数据 100%（§3.2） |
| 19 | `qreg` 头部块 | 数据 100% |
| 20 | `sureg`/`mvreg` 方程表 | 数据 100% |
| 21 | `frontier` LR 测试行 + 表内 4 行派生统计 | 数据 100% |
| 22 | `nl` ANOVA 表 | 数据 100% |
| 23 | `bootstrap`/`jackknife` `Replications` 行 + 列头 | 数据 90% |
| 24 | `estat ic` 表 | 依赖 #5/#9 |
| 25 | `etable` 默认 stats 扩展（加 `rmse`/`ll`） | `etable.py:28-31` |
| 26 | `e()` 补 macros（`cmd`/`model`/`title`/`predict`/`properties`） | 让 `etable` 能自动命名列 |
| 27 | 抑制 `arima`/`arch`/`xtgls` 的 `R-squared = .` 噪音行 | `summary.py:216-224` |

### 第三优先：需要新计算（以天计）

| # | 项 | 说明 |
|---|---|---|
| 28 | `estat gof`（deviance/Pearson/dispersion） | 每族方差函数不同 |
| 29 | `estat hettest`/`ovtest`/`vif`/`imtest` | 公式明确，可对拍 |
| 30 | `estat firststage` 完整版（Shea partial R²） | IV |
| 31 | `predict` 诊断类型（`cooksd`/`dfbeta`/`hat`/`rstudent`/`scores`…） | `predict.py:50-81` 的表完全没这些 |
| 32 | 残差诊断属性（`omnibus`/`jb`/`skew`/`kurtosis`/`durbin_watson`/`condition_number`） | statsmodels 兼容面 |
| 33 | `svy:` 的 `deff`/`mse`/`subpop()` | `V_srs` 可由已有 `w` 与样本算 |
| 34 | `manova` 的 Pillai/Lawley-Hotelling/Roy | `B`/`W` 已算（`systems.py:1143-1152`），但**还需典型相关**（见 §3.7） |
| 35 | `heckman` 的 `LR test of indep. eqns. (rho = 0)` | Stata `p6p7_outputs.txt:97` 打印 |
| 36 | `xtreg fe` 的 `F_f`/`p_f`（吸收效应联合检验） |  genuinely absent |
| 37 | `xtabond` 的 Sargan/Hansen/AR(1)/AR(2) | **已文档化刻意省略**（`panel_dynamic.py:455-457`），不建议为补全而补 |

### 明确不建议做

| 项 | 理由 |
|---|---|
| 复刻全部 14 个 `estat` | 按频率取子集：先 `vce`/`ic`/`gof`/`hettest` |
| 逐位复刻 Stata ML 停止点以获得 `crittype`/`opt` | GAP_REGISTER §8.11/§8.12 已论证这是错误方向；应**记录**我们的收敛信息而非追逐它的 |
| 给 `regress` 加 `e(tss)` | Stata 不存，加了是存一个 Stata 没有的数 |

---

## 7. 顺手发现的两处真缺陷（转交，不属于"缺失统计"范畴）

### 7.1 `svy: mean`/`svy: total` 打印语义错误的 `Root MSE`

见 §3.2。`summary.py:140-144` 的兜底 `sqrt(rss/df_resid)` 对 survey 估计量给出
**既非 RMSE 亦非 sd 的数**（`svy: total` 实测 `0.0000`）。
**修法：兜底应在 `e(rmse)` 缺失时打印 `.`，而不是现算一个数。**

### 7.2 `eivreg` 的 `r2 > 1`

`systems.py:962` 的 `sigma2 = (y'y - n b'Ab)/(n-k)` 在 `reliab` 修正足够大时可负；
`systems.py:985` 的 `max(sigma2,0)` 只保护了 `rmse`，`systems.py:964` 的 `r2` 未 clamp。
实测某帧下 `e(r2)=1.12706`、`e(rmse)=0.0`。
（测试通过不代表边界不存在——测试用的帧与该实跑不同。）

---

## 8. 收尾时解决的一个悬案：`manova` 四条 F 全部从 Stata 源码解出

审计中 `manova` 的 Pillai F 参数化一度"未解出"（数值反解得 33.83 vs Stata 打印 47.35），
当时正确地**拒绝拟合常数**，但停在了那里。

**StataNow 19 就装在 `D:\Program Files\StataNow19`，答案在 `ado/base/_/_canon.ado:676-749`。**
四个统计量全部由典型相关系数平方 `fo_i = ccorr_i²` 表出（`:703-713`），F 在 `:714-736`：

```
df1 = p*q ;  w = n - (p+q+3)/2 ;  t = sqrt((p²q²-4)/(p²+q²-5))  [p*q==2 时 t=1]
df2 = w*t - .5*p*q + 1
wilksF  = ((1-Λ^{1/t})/Λ^{1/t}) · (df2/df1)
pillaiF = (V/ncc · dfp2) / (dfp1 · (1 - V/ncc))     dfp1 = max·ncc, dfp2 = (n-max-1)·ncc
dfl2    = ncc·(n-max-ncc-2) + 2
lawleyF = U · dfl2 / (ncc · dfp1)
royF    = Θ · (n-max-1) / max
```

**实跑验证，四条全部命中 Stata 打印值（p10_outputs.txt:130-133）：**

| 统计量 | 算出 | df1/df2 | F 算出 | Stata |
|---|---|---|---|---|
| Wilks | 0.4290 | 4 / 472 | 62.16 | 62.15 |
| **Pillai** | 0.5710 | 4 / 474 | **47.35** | **47.35** |
| Lawley | 1.3307 | 4 / 470 | 78.18 | 78.18 |
| Roy | 1.3306 | 2 / 237 | 157.68 | 157.67 |

**但成本判定被推翻**：四个统计量走 `e(ccorr)`（典型相关），是 `W⁻¹B` 的广义特征问题；
westata 现有 Wilks 路径（`systems.py:1161`）直接 `|W|/|W+B|`，**从不构造 `ccorr`**。
所以这不是"算了未存"，而是**需新计算但公式已完全确定**——比"需查证"好得多。

**方法论教训（本项目第五次"读源码胜过反推"）**：先数值反解 → 正确判断"不能用拟合常数" →
**停在那里**。而"拒绝伪造"和"已经查清"是两件事。1.3996 这个看似像 7/5 的倍数，
经枚举 7350 个简单闭式都等于 7/5，证实**不具鉴别力**——巧合数字会制造"已接近"的错觉。

---

## 9. 需要同步修改的文档

- **`docs/GAP_REGISTER.md` §2** 已列 `estat vce`/`summarize`/`firststage`/`classification`，
  但**没有记录**：`tss` 的系统性缺失（4 处"算了未存"）、`xtreg` 16 键缺口、
  `e(ic)` 语义冲突、`estat` 共 14 个入口全缺、`e()` macros 全缺。
- **`docs/GAP_REGISTER.md` §3.5** 说 `manova` 只实现 Wilks' λ 与 Rao F —— 本次核实**与源码一致**，
  并补上了 Stata 完整输出形态证据（`p10_outputs.txt:130-133` 四行统计量 + `e/a/u` 图例）。
- **`docs/GAP_REGISTER.md` §8.2** 说 `frontier` 的 `/lnsig2v`、`/lnsig2u` 已在 P11 修复 ——
  本次逐行核实**确认已修复且超出当时记录**（`lr_chi2`/`p_lr`/`sigma_v`/`sigma_u`/`lambda`
  全在 `e()` 里），剩余只在渲染层。
- **`README.md:156-168`** 的命令覆盖表**没有一行提到 `estat`/`vce`/`ic`**，
  读者会以为后估计已完整。

---

## 10. 方法论：本次盘点的三条判断依据

按可靠性排序：

1. **Stata 自己的 fixture**（`*.json` 的 `scalars`/`macros`）——最高证据，
   是 `pystata` 从活体 StataNow 19.5 抓的 `ereturn list` 全量
   （`conformance/oracles/stata.py:200-243` 实证抓取机制）。
2. **Stata 控制台输出抄录**（`*.txt`）——次高，证明"Stata 打印什么"。
3. **安装目录里的 Stata ado 源码**（`D:\Program Files\StataNow19\ado\base\`）——
   与 1 同级，且是唯一能回答"公式是什么"的来源。本项目 §8.17/§8.19 两次靠它定案。
4. **手册知识** —— 仅在没有前两者时使用，且必须标注 C 级。

**一条重要的边界，也是本次最需要守住的纪律**：

> **fixture 里没有某个键，不等于 Stata 不存它。**

`regress_auto.json` 没有 `tss` 而 `areg.json` 有，正说明这点。
判"缺失"必须**逐命令**找证据，不能用一个 fixture 推全部——
这正是用户指认 1 需要小心的地方：**`regress` 不存 `e(tss)` 是符合 Stata 的，
但 `reghdfe`/`xtreg`/`areg` 不存是真缺口。** 两者修法完全不同：
前者不该动，后者是一个字典键。

---

## 11. 交付物清单

| 文件 | 内容 |
|---|---|
| `retest/gaps/lead_cross_cutting.md` | Lead 主线：跨家族共性、展示层、9 处渲染缺陷 |
| `retest/gaps/linear_iv_ts_gaps.md` | 15 个估计器表、219 行、Q1–Q5 答案、32 项优先级 |
| `retest/gaps/discrete_limited_gaps.md` | 3 家族 20+ 实跑、Q1–Q5 答案、`e(ic)` 三形态分析 |
| `retest/gaps/quantile_systems_gaps.md` | 6 家族实跑、Q1–Q6 答案、`e()` 全量键附录、**§3.7 manova 四条 F 的源码解出** |
| 本文件 | 汇总与优先级 |

`git status` 确认除 `retest/gaps/` 外无任何改动；所有临时探针脚本已删除。

**一处需要你决定的**：§6 第 17 项（`summary.py` 按 `model_name` 分支替代按 `r2_p` 分支）
会引入约 6–11 个 per-command 渲染分支。子智能体建议**先做 6 个高频形态**
（qreg/sqreg/rreg/svy/didregress/bootstrap），其余留给 `e()` 数据补齐——
我同意：这样维护成本可控，且第一批就能覆盖全部已知缺陷。
