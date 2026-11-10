# v3 报告核实清单

本文件记录对 `westata_vs_Stata_替代评估报告_v3.md` 各条目的实测核实结果。
**报告中的部分结论与当前代码不符**，逐条核实后记录如下。

> **2026-11-09 更正**：下文「`synth`/`synth2` 未实现，核心在 Mata 字节码（A 类：
> 信息不可得）」是 v3 报告当时的快照，现已失效。`ws.synth` 默认路径已落地（E2），
> P0 实测证伪了「信息不可得」：`e(V_matrix)` 可导出、`regsval` 是确定性纯函数、
> `normalize`/`regsval` 两个内核被精确求解。`synth2` 仍不做。详见
> `docs/GAP_REGISTER.md` §3.6.4 与同日 B 档设计规格。原文保留为历史。

## 一、报告成立并已修复

| 条目 | 报告内容 | 核实 | 处理 |
|---|---|---|---|
| **N14** | `prais, corc` 标准误 4.7% 偏差 | ✅ 成立：系数与 rho 精确，`se(price)` `.8491466` vs `.8108501` | **已修复**：最终方差用了错误的变换（PW 而非 corc），且 corc 的残差自由度应少 1 |
| **P0-2** | `rdrobust` 缺自动带宽 | ✅ 成立：`bandwidth=None` 抛 `UnsupportedError` | **部分**：三种组合规则（mserd/msesum/msetwo）已从 `rdbwselect.ado` 精确转写；`C[1..3]` 在 `lrdrobust.mlib` 字节码中，不可得 |

## 二、报告不成立（当前代码已正确）

| 条目 | 报告内容 | 实测结果 |
|---|---|---|
| **P0-1** | `PanelOLS.fit(cov_type=)` 在 linearmodels 7.x 下全部 `ValueError` | ❌ **不成立**。linearmodels **7.1** 下 `robust` / `HC1` / `cluster` 三种写法**全部成功**，且数值不同（`0.065784` vs `0.116169`），证明确实在计算 |
| **C13-08** | `reg airpassengers i.m` 整型因子未展开，`F(0,143)=0.00`，只有常数项 | ❌ **不成立**。`C(m)` 正常展开为 12 个参数，`F=1.4245`（Stata 1.42）、`R²=0.1061`（Stata 0.1061），全部精确 |

## 三、已记录但仍未完成

| 条目 | 状态 | 原因 |
|---|---|---|
| `teffects psmatch` 推断量 | ATET 精确（`2639.8646` vs `2639.865`），`se`/`z`/`p`/CI 为 NaN | 需要完整的 Abadie-Imbens 影响函数。模块 docstring 已排除两种简化路线：直接对处理效应取样本方差会小 1.2948 倍；该比值既非 `n/n_t` 也非 `(n-1)/(n_t-1)`，说明携带一阶推断函数修正 |
| `rdrobust` 自动带宽 | 组合规则已完成 | `C[1..3]` 在 Mata 字节码中 |
| `synth` / `synth2` | 未实现 | 核心在 Mata 字节码（A 类：信息不可得）|

## 四、`nnmatch` 标准误的诚实标注

`NearestNeighbourMatching` 的 `std_err` 使用的是 **Abadie-Imbens approximation**，
不是精确影响函数，且与 Stata 有 0.78% 偏差（`777.4784` vs `771.4713`）。

**该近似已在输出中明确标注**：

```
note: standard errors are Abadie-Imbens approximation
```

这与其他条目「宁可标 NaN 也不给看似合理的数」的处理一致。

## 五、核实方法

每条都遵循同一原则：**拿实时 Stata 对照，不以报告描述为准**。

- P0-1：构造真实面板（30 组 × 8 期），逐个 `cov_type` 实测
- C13-08：用报告的原始数据（`airpassengers`）+ 从 `time` 提取月份构造 `m`，复现全部三个统计量
- N14：用报告的原始配置（`icecream` + `consumption ~ temp + price + income, corc`）逐项比对
- `psmatch`：先确认点估计精确，再确认识别 NaN 的确切字段

## 七、第 9–10 轮的逐行相同率复核（报告自己的度量）

用报告 §6.1 定义的**逐行相同率**（空白归一化后 `difflib.SequenceMatcher` 最长匹配块 /
max(行数)）直接重测，比逐条读报告描述快得多，也更硬。

| 条目 | 报告分数 | 实测 | 结论 |
|---|---|---|---|
| C10-03 `estat overid` | 0.00 | **1.0000**（5/5） | 缺标题后空行；已修 |
| C10-05 `estat endogenous` | 0.00 | **1.0000**（6/6） | 本就是对的，报告过时 |
| C10-02 `estat firststage` | 0.28 | **0.9565**（22/23） | 变宽统计量被 `.3f` 截断 + 缺 2 个空行；已修 |
| C09-04 `predict lev` | 0.00 | **1.0000**（4/4） | 本就是对的 |
| C11-07 `margins, dydx(*)` | 0.48 | **1.0000**（15/15） | 本就是对的 |
| C11-08 `predict p` | 0.00 | 三种写法一致 | 本就是对的 |
| C12-09 `reg y i.year` | 0.00 | 42 行，`C(year)` 可用 | 本就是对的 |
| C04-04/05-08/09-04 `summarize` | 0.00 | **0.7500**（3/4） | 缺前导空行；已修 |

### 这一轮修的实际缺陷只有三处

1. `estat overid` 缺标题与首个统计量之间的空行；
2. `estat firststage` 用 `.3f` 截断变宽统计量（`20.0861` → `20.086`），F 列同病，
   且**普通表与 Shea 表各有一份**；
3. `summarize` 缺 Stata 打印的前导空行——这一处影响所有 `predict … ; summarize …` 案例。

其余六项的 0.00 都是**报告快照过时**，不是当前缺陷。

### 两个排查教训

- **陈旧的 `__pycache__` 会让修复看起来无效**：源码已是 `:.6g`，输出仍是 `20.086`。
  清缓存后测量才有意义。
- **死代码会伪装成权威**：`_render_r2_table` 里有两份连续的 `lines = [...]`，前一份被后一份
  立即覆盖，且两者相差一个前导空格——正是读者会以为"有讲究"的那种差异。已删除。

## 八、最终验证

```
5137 passed, 3 skipped, 0 failed
```

## 九、计数与受限因变量模型的最终逐行比对（第 12–19 轮）


breg 经六轮修复后，与 Stata 逐行比对（Stata 用 
olog，排除迭代日志）：

| 案例 | Stata 行数 | westata 行数 | 逐行相同率 |
|---|---|---|---|
| 
breg（边界解） | 18 | 18 | 0.6667 |
| 
breg（内部解） | 17 | 17 | 0.7059 |
| 	obit | 19 | 19 | **1.0000** |
| logit / probit / poisson | 13 | 13 | **1.0000** |

### 
breg 剩余差异全部是优化器精度

唯二不一致的行是 Log likelihood 与系数末位：

`
Stata  : Log likelihood = -196.12988
westata: Log likelihood = -196.13029
`

差 4e-4。第 13 轮已证明：所有起点的似然在此一致到 4e-4，无法用似然区分“真边界”与数值噪声。

### 一个重要的方法论教训

**	obit 曾被误判为“表格位置错位”**（第 14 轮），因为它与 Stata 的 26 行对不上。

真相是：Stata 的 	obit 默认输出 **7 行迭代日志**（Refining starting values: / Grid node 0: / Fitting full model:），加上这些正好 26 行。

用 
olog 后，两边都是 **19 行且逐行相同（1.0000）**。

> **比对 Stata 输出时必须先排除迭代日志**，否则会把日志行数误算成结构差异，进而去修一个不存在的缺陷。

## 十、第 21–24 轮：statsmodels 协议与剩余条目

### 已修复

| 报告项 | 问题 | 修复 |
|---|---|---|
| §4 (1) | `wald_test("s = 0")` 抛 `ValueError: could not convert string to float` | 字符串形式委托给已存在的 `test()` 约束解析器 |
| §4 (5) | `cov_type="robust"` 不被接受，且同一词在三个命令族含义不同 | `regress` 家族 → HC1，`ivregress` → **HC0**，分两处映射；结果存**规范名**而非回显别名 |
| §4 (4) | `PraisResult` 有私有 `_conf_int` 却无公开 `conf_int()`，抛 `AttributeError` | 补 `conf_int` / `cov_params` / `summary`；非对角用 **NaN 而非 0** |

### 核实后不成立

**§4 (2)「7 个 `from_formula` 首参不是公式字符串」—— 对 `Sureg` 是合理设计。**

`Sureg.from_formula(['lnw ~ s + expr', 'iq ~ s + expr'], d)` 接受**公式列表**，
因为多方程系统本来就需要多个公式，单个 `formula` 无法表达。其余 6 个（`IPW`/`AIPW`/
`RegressionAdjustment`/`NearestNeighbourMatching`/`Didregress`/`Xtdidregress`）是
`(outcome, treatment, data, ...)` 形态，对应 Stata 的 `teffects (y) (t) (x)` 双公式语法——
单公式同样无法表达。

**结论**：这不是 API 缺陷，而是这三类命令的固有结构。改动会破坏语义。

### 核实后确认为功能缺失（非报告描述的形态）

**C13-06 `irf table oirf` 的 4 列**：报告称「只给 1 列」。

实测 `ws.VAR` **没有 `irf` 方法**（只有 `fit` 与 `select_order`），所以不是「输出列数不足」，
而是**整个 IRF 表面不存在**。这是能力缺口，与 §7.3 列出的未实现项同级。

### `xtreg, mle` 的 2.6e-6「收敛差」

实测 `xtreg y k1 i.year, mle`（munnell.dta）：

| 量 | westata | Stata | 差 |
|---|---|---|---|
| `k1` | `2.4012813` | `2.4012813` | **完全一致** |
| `_cons` | `-1209.2341856` | `-1209.2335` | `7e-4`（相对 **6e-7**）|

主系数全位一致，`_cons` 的相对差在浮点量级。属于优化器停止点差异，与 `logit, or`、
`ologit` cut 点同类。

## 十一、第 30–32 轮：再核实三项报告断言

### C11-01 / C11-02（加权 logit/probit）——不成立

报告的 §5.5 记录「C11-01 缺表尾分隔线」。实测 `odds_ratio=True` 的输出：

```
       _cons |   1.101192   .1112681     0.95   0.340     .90334
----------------------------------------------------------------
Note: _cons estimates baseline odds.
```

**闭合边框与脚注都在**，且 `or` 变体的每个数字与 Stata 逐位相同：

```
Stata  : _cons |   1.107758   .0782241     1.45   0.147     .96457
westata: _cons |   1.107758   .0782241     1.45   0.147     .96457
```

一个容易搞错的细节，已实测确认：**`Note:` 脚注只在 `, or` 下出现**。普通
`logit y x1 [fweight=freq]` 末尾只有分隔线，而 `logit ..., or` 才有脚注——本包两种情形都对。

### C13-06（`irf table oirf` 的 4 列）——第 25–27 轮已修

| 轮次 | 内容 |
|---|---|
| 25 | 核实：方法存在，只是 `impulse`/`response` 默认取第一个变量 |
| 26 | 默认改为显示全部列；**修正列序**（impulse 外层，非数组内存序）|
| 27 | 表头与分隔线几何逐字符对齐 |

最终 8 行（2 表头 + 2 分隔线 + 5 数据行）与 Stata **逐字符相同**。

### C17-02/03（`didregress` 分组口径）——无法复现

报告称「控制/处理组 478/72 vs Stata 465/71」。仓库内没有报告所用的
`rebel`/`canal_post` 数据，因此用合成面板复核：ATET 0.4229（真值 0.5），**结果合理**。

但审计顺带发现一个真实的易用性缺陷（第 30 轮已修）：

```python
Didregress.from_formula(...).n_used     # 返回 0，即使数据有 2000 个完整行
```

样本计数在 `__init__` 里置 0、只在 `fit()` 时填充。现在构造时即计算，且与 `fit` 后一致
（2000/0 与 1945/55 两种情形均验证）。

### 一条通用教训（已完成第 8 条的第 4 次验证）

写测试参考值时**必须实时捕获**。我的第一版 OLS 全精度 fixture 是编造的，
测试立即失败。`error.md` 第 8 条至此已命中四次。

## 六、最终验证

```
5137 passed, 3 skipped, 0 failed
```
