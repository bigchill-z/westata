# westata 与 Stata 对齐工作汇总（第 1–51 轮）

本文档汇总针对 `westata_vs_Stata_差距评估报告.md` 所列出差距的处理结果，把每一项的
**验证证据**写清楚，使「已实现」「已验证」「已知差异」「明确未做」四者界限分明。

**原则**：一条结论只有在有 Stata 实测值或可重跑的闭式检验支撑时才算成立。本文档中每个
数字都写明了它来自运行的哪个命令、哪份数据。

**第 30–51 轮的重点**：报告把 39 项归为「B 类：布局适配（设计约束，非缺陷）」。但用户的
目标是「输出的数值与排版与 Stata 完全一致」，这个分类的前提不再成立。逐项实测后发现
其中多数**确实**与 Stata 不同，且有几处是真实数值错误，不是布局偏好。详见第七节。

---

## 一、总体结果

| 项 | 数量 |
|---|---|
| 全量测试 | **4284 passed, 3 skipped, 0 failed**（4288 收集；第 51 轮末检查点。第 52 轮新增 `teffects_psmatch` 后只跑了 102 个 teffects 定向测试，全量留待下一个 20 项节点）|
| 完成项 | **64**（含第 52 轮 `teffects_psmatch` 点估计）|
| 提交 | 66 个，`4d3f182` … `9cc3db7`（本文档的更新提交另计）|
| 测试文件 | 248 个 |
| 新增验证测试文件（本次工作）| 30+ 个 |

**验证方式**：所有对照均使用 `pystata` 实时调用 StataNow 19.5 MP，同一份 `.dta`、同一模型，
比我方实现与 Stata 打印值的差。不是与旧版本输出对比——与自己的旧输出一致不能证明任何事。

**全量测试节奏**：用户指示「每完成 10 个修复后再进行全量确认」，后改为「每 20 轮修复后跑
一次全量测试」，中间只用 `-k` 定向测试。全量测试点：4208 → 4226 → 4233 → 4240 → 4269 →
4284 passed。

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
| 21 | VAR 的 IRF 用无偏协方差归一化 | 全序列高 1.5% → **精确** | 应为 ML 尺度；`ml_factor` 早已存在，IRF 是唯一未应用的路径 |
| 22 | `fevd` 无可读输出 | 新增 `fevd_table` | 数值本就正确；轴顺序为 `(响应, 期, 冲击)` |
| 23 | `unitroot(test="kpss")` 答错问题 | 默认改趋势平稳 + 带宽系数 12→**4** | 原实现走 statsmodels 默认，数值 `.3197` vs Stata `2.14` |

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
| `kpss` | 5 个 lag 统计量 + 临界值表 ✓ |
| `rdrobust`（点估计）| `tau=7.6896`、有效样本 96/69 ✓ |
| `teffects ra`（点估计）| ATE/ATET/两个 POmean 全对 ✓ |
| VAR 脉冲响应（IRF）| step 0/1/2/3/10 全对 ✓ |
| VAR 方差分解（FEVD）| 4 列 × 多个 step 全对 ✓ |

---

## 四、已知差异（已记录，未强行对齐）

| 项 | 差异 | 性质 |
|---|---|---|
| `%9.0g` 输出格式 | 部分小系数多印一位（`.06761177` vs `.0676118`）| **显示舍入**，数值相同 |
| `xtlogit, re` 方差分量 | `sigma_u` `.0037` vs Stata `.0000243` | 估计器差异（非布局）。Note: 修复积分器后已改善，扫描数据上似然面平坦 |
| `xtreg, re` 方差分量 | `1.424/.854` vs Stata `1.472/.709` | 估计器差异 |
| `rdrobust` 标准误 | `0.32163224` vs Stata `0.31709743` | 稳健方差估计器未复现（CCT 形式）|
| `teffects ra` 标准误 | `0.1827790` vs Stata `0.2430606` | 同上；试过 6 种影响函数形式 |
| `rdrobust` 带宽选择 | 未实现（`bandwidth=None` 报错）| MSE 最优选择器是独立工程 |
| `teffects ra` 标准误（第 2 次尝试）| 试 6 种影响函数；最好 0.2436296 vs Stata 0.2430606，但另两个样本分别偏高 2.0% / 偏低 2.5% | **一个样本上吻合、其他样本双向错 = 巧合，不发布** |
| `bsqreg` 默认重复数 | 200 vs Stata 20 | 刻意不改 |
| `arima` 的 `/sigma` 水平 | 第 5 位有效数字（`.0164569` vs `.0164621`）| **已定性**：两包最大化同一似然（statsmodels 在 Stata 参数处 loglike 差 1e-8），差异纯属优化器停点，似然面极平坦 |
| `vec` 第一方程常数符号 | `+6.28e-07` vs `-6.35e-07` | **已定性**：值本身约 6e-7，两次 ML 拟合的 `c_ce` 差 6.7e-7 × α(−1.87) 超过常数本身；第二方程常数与两个标准误精确匹配 |
| `arch` 均值方程块标题 | `dinf` 右对齐 vs Stata 左对齐 | 涉及共享分组路径，未改 |

**第 57–58 轮已解决（此前列在本表）**：`tobit`/`arima` 的区间端点。tobit 是对数方差尺度 + `t(e(df_r))`，
arima 是 σ 尺度 + 正态 1.96；两者规则不同，此前用「方差区间开方」哪个都不匹配。
`vec` 主表也已从「规则未定」推进到 9 个 `e()` 标量全部精确。

---

## 五、明确未实现及原因

| 项 | 状态 | 原因 |
|---|---|---|
| `vecrank lags(1)` | 报错拒绝 | 映射到 `k_ar_diff=0` 时后端给 `173.8827/2.8945`，Stata 是 `172.50341/1.2744366`；移位样本后第一个对、第二个错。**4 次尝试未定，选择报错而非近似值** |
| `estat bgodfrey`（早期）| 后已实现 | 曾两次判定「无法复现」。真相是源码在 `ado/base/b/bgodfrey.ado`，我一直未查。**「找不到公式」不等于「公式不可得」** |
| `xttest0` 不平衡面板 | 报错拒绝 | 平衡面板公式在不平衡数据上差 2.5%。**一度误判为公式错误，实为用错数据** |
| RD（`rdrobust`）| **点估计已实现**，推断与带宽未实现 | 标准误为 CCT 形式，6 次尝试未复现 |
| 合成控制（`synth`/`synth2`）| 未实现 | 同上 |
| 回归控制（`rcm`）| 未实现 | 同上 |
| `teffects ra` | **点估计已实现**，标准误未实现 | 同上 |
| **`teffects psmatch`（C15-01/C15-02）** | **点估计已实现**（第 52 轮）| 见下方专项说明；AI-robust 标准误仍未复现 |
| `tebalance` / `xtserial` | 未实现 | SSC 包未安装，无源码 |
| `cov_params()` 返回 DataFrame | **暂不修复** | 会破坏 `cov_params()[0,0]` 索引方式，属破坏性 API 变更；用户已明确暂不处理 |

### `teffects psmatch` 专项（报告 P0，第 52 轮）——**点估计已攻克**

Stata 参照（`nsw_dw.dta`，`teffects psmatch (re78) (treat <8 协变量>), atet`）：

```
默认 nn(1):  ATET 2639.865    SE 714.7758    Matches min=1 max=8
nn(4):       ATET 2330.108    SE 680.0515    Matches min=4 max=11
```

**第 51 轮我曾误判为「估计量无法复现、完全未实现」，这个结论是错的，在此撤回。**
第 52 轮用 Stata 的 `generate(m)` 选项导出了实际匹配的观测号，定位到真正原因：

- `generate(m)` 对 `nneighbor(1)` 生成了 **m1..m8 共 8 个变量**。帮助文件写明「因距离
  并列，生成的变量数可能多于 nneighbor()」。
- **Stata 把所有与最小距离并列的对照全部纳入并取均值。** 只取最近 1 个得 2569.745，
  取全部并列对照得 **2639.865** ✓；`nn(4)` 得 **2330.108** ✓。表头的 `max = 8` / `max = 11`
  正是并列数。
- 报告 C15-01 记录的 13.7% 差距，根因不是点估计错，而是**忽略了并列**。

排查路径（每一步都排除了一个假设，而非猜测）：

1. westata 的 logit 系数与 Stata 一致到 1e-7，`predict()` 与 Stata 导出的 ps 差 2.75e-8
   → 差异**不在倾向得分模型**；
2. 用 **Stata 自己导出的 ps/xb** 做最近邻匹配，仍差 192 → 差异**在匹配算法**；
3. `generate(m)` 揭示并列 → 从 Stata 的匹配 ID 重算 ATET 精确复现 → **并列规则确认**。

已实现 `teffects_psmatch`：`teffects_psmatch(y, d, X, nneighbor=1|4)`，两种 ATET 及
min/max 匹配数全部精确，Stata 风格表格逐行对齐。

**仍未复现**：Abadie-Imbens AI-robust 标准误（需局部线性方差估计，非闭式）。`se`/`z`/`p`/
区间返回 NaN，表格以 Stata 的 `.` 如实占位，不发布近似值。

`teffects ipw`（C15-06）的紧凑摘要问题未复核；`teffects ra`（回归调整）点估计已实现。

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
# 全量测试（本次工作最后一个检查点：4284 passed, 3 skipped）
./.venv/Scripts/python.exe -m pytest tests/ -q \
  --deselect tests/test_packaging_hygiene.py::test_tracked_test_data_is_actually_tracked

# 单点核对（示例：VAR 每方程统计量、margins 表格、VECM 协方差）
./.venv/Scripts/python.exe -m pytest tests/test_var_equation_stats.py -v
./.venv/Scripts/python.exe -m pytest tests/test_margins_layout.py -v
./.venv/Scripts/python.exe -m pytest tests/test_vecm_cov_scale.py -v
```

每个新增测试文件的文档字符串里都写了它对照的 **Stata 命令、数据集、参照数值**，可直接复核。
第七节所列每一张表格都有对应的 `tests/test_*_layout.py` 守卫，逐行固定 Stata 输出。

主要测试文件（早期）：

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

## 七、第 30–51 轮：B 类重新审视后的成果

报告把 39 项归为「B 类：布局适配（设计约束，非缺陷）」。目标改为「数值与排版与 Stata
完全一致」后，逐项实测发现多数确实不同。以下每项都逐行对照过 Stata 实测输出，并有独立
守卫测试固定。

### 主回归表与后估计表格（逐行一致）

| 命令 | 结果 | 关键发现 |
|---|---|---|
| `estat ic` / `vif` / `ovtest` / `hettest` / `bgodfrey` | 全部逐行一致 | 列数、标签填充、缩进、精度各不同 |
| `estat classification` | **26/26 行** | 原缺标题行、表格结构完全不同、cutoff `0.5` vs `.5` |
| `estat gof` / `vce` | 6/6、8/8 行 | 缺两个计数行；缺模型名、`y` 列头、`e(V)` 标注 |
| `estat firststage` | 两张表全对 | 缺整个 Stock-Yogo 块；`r(mineig)` 错报特征值而非统计量 |
| `estat overid` / `endogenous` | 4/4、5/5 行 | **Basmann 的 k 用错**：`1.1953` → `1.1923` |
| `margins` | 三张表 10/10、11/11、9/9 | **原本输出 dataclass repr，没有表格** |
| `pwcompare` / `contrast` | 11/11、9/9 行 | 前者原本输出 list repr；后者缺 `Margins:` 与 `Denominator` 行 |
| `test` / `lincom` / `lrtest` | 逐行一致 | 居中标签、双层表头、`Assumption:` 行 |
| `ivregress` / `qreg` / `xtreg fe` / `areg` 主表 | 5/5、3/3、21/21、12/12 行 | 见下方「真实数值缺陷」 |
| `sureg` / `mvreg` / `svy` 族 / `newey` / `ologit` | 7/7、5/5、11/11 等 | `sureg` 缺 `chi2`/`P>chi2` 两整列 |
| `irf_table`（新增）| oirf/irf 各 10/10 行 | 报告 C13-06；数值已在，缺可读表格 |
| VECM `cov_params()` | 精确 | **每次调用抛 AttributeError**；标准误低 1.24% |

### 真实数值缺陷（非排版）

1. **`r(mineig)` 尺度**：存特征值 `0.6267`，Stata 打印 Cragg-Donald 统计量 `332.355`。
   恒等式 `F = eig/(1-eig)·df_denom/df_num` 精确成立。
2. **`tobit` 的 `var(e.y)`**：打印 σ 而非 σ²（`1.029584` vs `1.060043`）。
   `1.029584² = 1.060043`，`2σ·se(σ)` 也精确吻合标准误。
3. **`arima` 的 `/sigma`**：打印 σ² 而非 σ（`.0002709` vs `.0164621`），**差 60 倍**。
4. **`arch` 行标签错位**：`L1.` 贴在 omega、`_cons` 贴在 alpha[1]。参数重排时 OPG
   协方差若用重排后的向量计算，标准误从 `.1471562` 变成 `122.6474`。
5. **`irf()` 过度缩放**：`ml_factor` 施加到所有响应，把 IRF step0 从 `1` 变成 `0.984638`。
6. **`areg` 崩溃**：`maize` 有 3144 个**缺失值**（不是单例组），全样本设计乘估计样本响应。
7. **`ivregress` 缺 `R-squared`**、**`xtreg` 缺 `(omitted)` 行**：信息缺失，非数值错误。

### 反复出现的模式：注释与测试在保护 bug

多个缺陷的代码注释都写着一个看似合理的理由，实测 Stata 相反；配套测试把那个理由固定
下来。这是本项目最常见的失败模式：

| 注释/测试断言 | 实测 |
|---|---|
| `ivregress` 不打印 R² | Stata 打印 `R-squared = 0.8923` |
| `estat ic` 打印 AICc/CAIC | Stata 只有 7 列 |
| `test_estat_ic_label` 要求显示命令名 | Stata 显示 `.` |
| `lincom` 打印 `Coef.` | Stata 打印 `Coefficient` |
| `qreg` 表头是特例（70 列）| 整块都是 78 列 |
| `svy: mean` 右半块随值浮动 | 恒在 34 列 |
| `arch dinf` 均值块标题是 `y` | 实际是 `dinf` |
| `contrast` 不需要 Denominator | F 没有分母自由度就无法解读 |
| `stderr_alpha` 已与 Stata 核对 | 实际低 1.24% |

**教训**：注释固化猜测，测试固化注释。每一次都必须回到实时 Stata 输出重新测量。

---

## 八、需要使用者知情的事项

1. **`predict()` 默认值已变**（`11b1936`）。`Logit.predict()` 现在返回概率而非 log-odds。
   需要旧行为请显式写 `kind='xb'`。这是 **breaking change**，未发布到任何版本。

2. **本仓库工作区未提交的改动请勿混入**。各轮提交都只暂存了明确列出的文件。

3. **「64 项」不表示「westata 已可替代 Stata」**。下述缺口真实存在：`vec` 主表（AIC/HQIC/
   SBIC 规则未定）、`areg` 的 F test of absorbed indicators（差 0.34%，拒绝近似值）、
   `teffects psmatch` 与 `teffects ra` 的 AI-robust / 影响函数标准误、`mlogit`（完全未实现）、
   断点回归/合成控制/回归控制整章、以及 `cov_params()` 契约（用户明确暂不修复）。

4. **「已知差异」不等于「已验证一致」**。凡提交信息里写「记录而非声称」「拒绝近似值」
   的，都不要用「已对齐」概括。
