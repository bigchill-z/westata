# westata 与 Stata 对齐工作汇总（第 1–22 轮）

本文档汇总针对 `westata_vs_Stata_差距评估报告.md` 所列出差距的处理结果，把每一项的
**验证证据**写清楚，使「已实现」「已验证」「已知差异」「明确未做」四者界限分明。

**原则**：一条结论只有在有 Stata 实测值或可重跑的闭式检验支撑时才算成立。本文档中每个
数字都写明了它来自运行的哪个命令、哪份数据。

---

## 一、总体结果

| 项 | 数量 |
|---|---|
| 全量测试 | **4208 passed, 3 skipped, 0 failed** |
| 完成项 | **37**（20 项缺陷修复 + 17 项能力新增）|
| 提交 | 24 个，`4d3f182` … `f8d7495` |
| 新增验证测试文件 | 19 个 |

**验证方式**：所有对照均使用 `pystata` 实时调用 StataNow 19.5 MP，同一份 `.dta`、同一模型，
比我方实现与 Stata 打印值的差。不是与旧版本输出对比——与自己的旧输出一致不能证明任何事。

---

## 二、缺陷修复（20 项）

### 2.1 面板

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 1 | `xtreg,fe` 的 `r2_b`/`r2_o` 用混尺度拟合值 | `1.0000/.9638` → **`.6650/.7135`** | Stata `xtreg ... , fe` on `lin_1992.dta` |
| 2 | `e(mss)` 算成 `tss-rss` | `388.47` → **`78.087606`** | 同上；`tss` 与 `rss`/`mss` 本就不同尺度 |
| 3 | 共线回归元直接报 `AbsorbingEffectError` | 改为剔除并继续估计 | `cao_chen.dta`，`prerebel` 组内常数 |
| 4 | `xttest0` 未实现 | **`chibar2(01)=252.58`** | 平衡面板 12×10，与 Stata 一致 |

**关于 #3 的数值验证**：修复后 `maize` 系数 `37.43478`（Stata `37.43478`），
`hausman` 统计量 `0.77331812`（Stata `.77331812`）、`p` `.37919252`（Stata `.37919252`）。
`hausman` 在报告中列为「缺失」，实为实现完整、被上游 bug 阻断。

### 2.2 稳健/聚类

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 5 | LSDV 聚类 F 溢出 `-1.73e+16` | → **`NaN`**（Stata 打印 `.`）| `df_t < df_m` 时 Wald 不可用，Stata 亦如此 |
| 6 | 普通 OLS 被误判为 robust 而拒绝 | 修正 `cov_type` 匹配 | `cov_type='nonrobust'` 含子串 `robust` |

### 2.3 HDFE

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 7 | 吸收自由度 `df_a` | `797` → **`261`** | 嵌套于聚类的吸收效应不计自由度 |
| 8 | 缺 `Within R-sq.` 行 | 补齐 | `tss_within` 重拟合的 frame 行数错误 |
| 9 | `r2_within` 样本不对齐 | `4.97e-05` → **`.0002486480`** | Stata `.00024865`；用 pyfixest 的 `_na_index` 精确还原估计样本 |

### 2.4 加权与后估计

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 10 | `margins` 未展开 `fweight` | 24 行 → 2201 行 | 5 个边际效应逐位匹配 Stata |
| 11 | `estat clas` 在 `fweight` 下报错 | 改为可用 | 8 项指标全部匹配（73.47% / 79.03% / …）|
| 12 | `predict()` 默认返回线性指数 | → **概率** | logit 概率；Poisson `n`；OLS 仍 `xb` |

**关于 #12 的危害**：用错值时 Brier score 为 `2.4667`，正确值 `0.1621`——**差 15 倍且不报错**。

### 2.5 时间序列

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 13 | `dfuller` 默认选阶 | `-2.571` → **`-3.718`** | **结论反转**：原判「无单位根」，实为「有单位根」|
| 14 | VAR `rmse_#` 分母用 `df_eq` | `.077907` → **`.013815`** | 应为 `N - df_eq` |
| 15 | VAR `r2_#` 用全序列居中 | `.3648` → **`.3621`** | 应用估计样本 |
| 16 | VAR `chi2_#` 参数索引跨步 | `75.34` → **`93.11047`** | `stata_param_names` 按方程分块，非交错 |
| 17 | `vargranger` 调用即报错 | 改为可用 | 数值本就正确（48.123 / 9.009），仅方程名未传 |

### 2.6 其他

| # | 缺陷 | 修复前后 | 证据 |
|---|---|---|---|
| 18 | `didregress` 的 Control 行 Time | `0/0` → **`1650/1650`** | 应为面板起始期 |
| 19 | `result.aic`/`.bic` 口径 | `-194.375` → **`-109.239`** | 改为 statsmodels/`estat ic` 口径；`e(aic)` 保留原值 |
| 20 | `reghdfe.predict()` 未注册 | — | 一度误加注册，已撤销（有 FE 时设计矩阵与参数维度不符）|

---

## 三、能力新增（17 项）

每项均对着 Stata 实测值验证。

| 命令 | 验证结果 |
|---|---|
| `estat dwatson` | `d(4,30)=1.021169` ✓ |
| `wntestq` | `Q 26.1974 / 16.3718` ✓ |
| `corrgram` | AC/PAC/Q/p 全对（3 个 lag）✓ |
| `estat bgodfrey` | `4.237/4.487/4.775/5.099` ✓ |
| `prais` | rho、4 系数、4 标准误、R²、RMSE、F、两个 DW 全对 ✓ |
| `xttest0` | `chibar2=252.58` ✓（仅平衡面板）|
| `varsoc` | LL/LR/AIC/HQIC/SBIC 全部 5 阶 ✓ |
| `varstable` | 4 个特征值及模 ✓ |
| `varwle` | 6 个数值（2 方程 × 2 阶 + All）✓ |
| `vecrank` | 3 行 × 4 列全部 ✓（`lags≥2`）|
| `vecstable` | 含规范强加的单位根 ✓ |
| `varlmar` | `7.3130 / 5.0645` ✓ |
| `veclmar` | `8.8911 / 4.8537` ✓ |
| `dfgls` | 8 个数值（含/不含趋势）✓ |
| `contrast` 联合 Wald | `df=3, F=5.90, P>F=0.0006` ✓ |
| `VARResults.summary()` | 方程表 5 列全对 ✓ |
| `margins atmeans` + `odds_ratios` | 两者均精确 ✓ |

---

## 四、已知差异（已记录，未强行对齐）

| 项 | 差异 | 性质 |
|---|---|---|
| `%9.0g` 输出格式 | 部分小系数多印一位（`.06761177` vs `.0676118`）| **显示舍入**，数值相同 |
| `xtlogit, re` 方差分量 | `sigma_u` `.0037` vs Stata `.0000243` | 估计器差异（非布局）。Note: 修复积分器后已改善，扫描数据上似然面平坦 |
| `xtreg, re` 方差分量 | `1.424/.854` vs Stata `1.472/.709` | 估计器差异 |
| `bsqreg` 默认重复数 | 200 vs Stata 20 | 刻意不改 |

---

## 五、明确未实现及原因

| 项 | 状态 | 原因 |
|---|---|---|
| `vecrank lags(1)` | 报错拒绝 | 映射到 `k_ar_diff=0` 时后端给 `173.8827/2.8945`，Stata 是 `172.50341/1.2744366`；移位样本后第一个对、第二个错。**4 次尝试未定，选择报错而非近似值** |
| `estat bgodfrey`（早期）| 后已实现 | 曾两次判定「无法复现」。真相是源码在 `ado/base/b/bgodfrey.ado`，我一直未查。**「找不到公式」不等于「公式不可得」** |
| `xttest0` 不平衡面板 | 报错拒绝 | 平衡面板公式在不平衡数据上差 2.5%。**一度误判为公式错误，实为用错数据** |
| RD（`rdrobust`×3）| 未实现 | SSC 第三方包，无 `ado/base` 源码；整章级新功能 |
| 合成控制（`synth`/`synth2`）| 未实现 | 同上 |
| 回归控制（`rcm`）| 未实现 | 同上 |
| `tebalance` / `psmatch` | 未实现 | 无源码；且 `psmatch` 的匹配距离为倾向得分，与既有马氏距离实现不是同一估计量 |
| `xtserial` | 未实现 | SSC 包未安装，无源码 |
| `cov_params()` 返回 DataFrame | **暂不修复** | 会破坏 `cov_params()[0,0]` 索引方式，属破坏性 API 变更；用户已明确暂不处理 |

### 已撤回的错误结论

工作过程中有 **5 个结论被后续证据推翻**，记录在此以免重复：

1. `reghdfe` 不实现吸收效应 —— 错，公式 `|` 块即等价；是**扫描器比了不同模型**
2. `etregress` 需要实现 athrho 参数化 —— 错，`method='ml'` 早已存在
3. `tnbreg` 卡方混合检验「无法计算」—— 错，`e(ll)`/`e(ll₀)` 一直在
4. `xttest0` 公式不对 —— 错，是**用不平衡数据测平衡公式**
5. `estat bgodfrey` 无法复现 —— 错，**源码一直在 `ado/base`**

前三条的共同病根：把「扫描器/测试设置的问题」当成了「代码的问题」。

---

## 六、复现方式

```bash
# 全量测试
./.venv/Scripts/python.exe -m pytest tests/ -q

# 单点核对（示例：VAR 每方程统计量）
./.venv/Scripts/python.exe -m pytest tests/test_var_equation_stats.py -v
```

每个新增测试文件的文档字符串里都写了它对照的 **Stata 命令、数据集、参照数值**，可直接复核。

主要测试文件：

| 主题 | 文件 |
|---|---|
| 面板 R²/mss | `tests/test_panel_e_identities.py`、`tests/test_e_identities.py` |
| 吸收列处理 | `tests/test_panel_absorbed_columns.py` |
| HDFE 自由度 | `tests/test_var_e_block.py`（相关断言）|
| 预测默认值 | `tests/test_predict_defaults.py` |
| 自相关检验 | `tests/test_serial_correlation.py`、`tests/test_bgodfrey.py` |
| Prais-Winsten | `tests/test_prais.py` |
| VAR/VECM 族 | `tests/test_varsoc.py`、`test_varstable.py`、`test_varwle.py`、`test_varlmar.py`、`test_vecrank.py`、`test_vecstable.py` |
| 单位根 | `tests/test_unitroot_adf.py`、`tests/test_dfgls.py` |
| 对比与边际 | `tests/test_contrast_joint.py`、`tests/test_margins_atmeans_odds.py` |
| VAR 表格 | `tests/test_var_summary.py`、`tests/test_var_equation_stats.py` |

---

## 七、需要使用者知情的事项

1. **`predict()` 默认值已变**（`11b1936`）。`Logit.predict()` 现在返回概率而非 log-odds。
   需要旧行为请显式写 `kind='xb'`。这是 **breaking change**，未发布到任何版本。

2. **本仓库工作区未提交的改动请勿混入**。各轮提交都只暂存了明确列出的文件。

3. **本文档的「37 项」不表示「westata 已可替代 Stata」**。报告中 E 类 6 项（RD、合成控制、
   回归控制、倾向得分匹配等）属整章级新功能，**完全未实现**。
