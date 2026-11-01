# westata 可优化点清单 · Nunn & Qian (2011)

**来源**：Nunn & Qian (2011) 复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/nq2011/nq2011_probe_gaps.py`

> 配套报告：[REPORT.md](../papers/nq2011/REPORT.md)

---

## 0. 结论：0 个缺口（2 条均已修复）

westata 新增了 `Areg`（提交 `56304d4`），本文件随之更新。

| 条目 | 性质 | 状态 |
|---|---|---|
| `Areg` 的 `_cons` 与 Stata `areg` 不一致 | 缺口（P1）| **已修复** |
| `Areg` 的聚类标准误遵循 `reghdfe` 而非 `areg` | 缺口 | **已修复**（见 §1.5）|
| 系数（斜率） | 正确 | 6 列全部吻合 |

三条数值现在都与 Stata 的 `areg` **逐位一致**：

| 量 | westata `Areg` | Stata `areg` |
|---|---|---|
| 斜率标准误 | 0.0171579390 | 0.0171579390 |
| `_cons` | 12.64269254 | 12.64269254 |
| `_cons` 标准误 | 0.03383845 | 0.03383845 |

修法见 §1.5（聚类标准误）与 §4（常数项）。**这两条此前被记为"约定差异"和"未修复缺口"，
现在都是已修复项**——它们是同一类问题：`Areg` 继承 `reghdfe` 的方差路径，而 `areg` 的
自由度记账不同。

## 1. 最关键的一点：判定差异的归属需要对照

### 1.1 现象

同一条命令链、同一份数据、同一个系数，标准误不同：

| 路径 | 系数 | 标准误 |
|---|---|---|
| Stata `areg`（作者当年跑的） | 0.1213388 | **0.0171579** |
| Stata `xtreg, fe` | 0.1213388 | **0.0164182** |
| westata `FEOLS` | 0.1213388 | **0.0164182** |
| westata `PanelOLS` | 0.1213388 | **0.0164182** |
| **westata `Areg`（修复后）** | 0.1213388 | **0.0171579** ← 与 `areg` 一致 |

> 修复前 `Areg` 报的是 0.0164182——即 `xtreg`/`reghdfe` 的值。见 §1.5。

如果只对照"论文报告的 0.0171579"，会得出"westata 标准误错 4.5%"的结论。**但这个结论是错的。**

### 1.2 正确的判定方式

关键在于：**Stata 自己就不同意自己**。`areg` 与 `xtreg, fe` 给出同样的系数、不同的标准误。所以"对照论文"这个单一基准不足以判定谁对。

正确的做法是**同时对照两个 Stata 命令**：

```
westata vs xtreg, fe  →  se 相对差 1e-13      （机器精度）
westata vs areg       →  se 相对差 4.3e-02    （4.5%）
```

这就把问题从"westata 是否算错"变成"westata 实现的是哪一种约定"。

`FEOLS`/`PanelOLS` 的答案是 **`xtreg`/`reghdfe`**，这是它们的正确目标。
`Areg` 的答案必须是 **`areg`**——它修好之后报 0.0171579，与 Stata `areg` 逐位相同（§1.5）。

### 1.3 差异的机制（实测定位，纠正了初版说法）

初版本文档把差异归因为「聚类稳健方差的小样本调整因子对不同 K 的计数」。**这个解释是错的**——实测推翻了它。

三步定位：

**第 1 步：残差完全相同。** 两个命令拟合的是同一个模型：

```
areg  e(rss) = 95.281035
xtreg e(rss) = 95.281035     ← 逐位相同
逐行残差差异 = 0 行
```

所以不是模型不同，也不是聚类调整不同。

**第 2 步：`rmse` 不同，且比值和标准误比值完全相等。**

```
areg  se=0.0171579390  rmse=0.2608790559
xtreg se=0.0164181944  rmse=0.2496315592
se 比值   = 1.0450563895
rmse 比值 = 1.0450563895     ← 完全一致
```

标准误 = `rmse × (聚类调整项)`。既然比值相等，说明**聚类调整项两边一样**，差异全部来自 `rmse`。

**第 3 步：`rmse` 的分母不同。** 反解 `rss/rmse²`：

| 命令 | 隐含分母 | 构成 |
|---|---|---|
| `areg` | **1400** = 1552 − 152 | 152 = 22 个回归元 + 1 常数 + **129 个被吸收的国家虚拟变量** |
| `xtreg, fe` | **1529** = 1552 − 23 | 23 = 22 个回归元 + 1 常数，**不计被吸收的固定效应** |

验证：

$$\sqrt{rac{1529}{1400}} = 1.0450563895$$

与实测比值**精确到 10 位小数相同**。

### 1.3.1 一句话说清

> `areg` 在算残差方差时，把「被吸收的国家虚拟变量」也算作消耗的自由度；`xtreg, fe` 不算。
> 两者残差一模一样，只是除以不同的分母，于是 `rmse` 差 4.5%，标准误就差 4.5%。

**这不是统计方法的对错之争**，而是关于"被吸收的固定效应算不算参数"的记账约定。LIML/IV 里没有这个问题，所以只有面板固定效应模型会碰到。

**注意区分两个自由度**：两者都报 `e(df_r) = 129`（130 个聚类 − 1），那是**推断用**的自由度；而 `rmse` 的分母是另一个量，这才是产生 4.5% 的地方。初版之所以写错，就是把这两个概念混为一谈。

### 1.4 为什么这是有价值的发现

**这段结论对使用者的实际意义**：如果你的论文用 `areg` 报告标准误，而你想用 westata 复现，你会看到 4.5% 的差距——**不要以为是 westata 算错了，这是 `areg` 特有的约定**。反过来，如果论文用 `xtreg` 或 `reghdfe`，westata 会精确吻合。

这个区别此前没有记录在任何地方，包括 westata 自己的文档。

### 1.5 聚类标准误的修复：`Areg` 现在与 `areg` 一致

**修复前**：`Areg` 继承 `FEOLS` 的 `reghdfe` 方差路径，报 0.0164182。

**根因**：两个命令用同一个 CR1 公式 `G/(G-1) · (N-1)/(N-K)`，**只有 K 不同**：

| 命令 | K | 值 |
|---|---|---|
| `reghdfe` | `k_reported` = 23 | **0.0164182** |
| `areg` | `k_reported + (k_absorb − 1)` = 23 + 129 = 152 | **0.0171579** |

在本文数据上直接代入两个 K 验证：

```
K = 23  → 0.0164181944   （= reghdfe，westata 修复前的值）
K = 152 → 0.0171579390   （= Stata areg，逐位相同）
```

`k_absorb`（130）数的是**吸收的水平数**，其中常数项本身占一个，所以是 `k_absorb − 1`。
用 `K = 153` 会得到 0.0171640701——差 0.036%，看起来像舍入误差，实际是错的。

**注意这个修正只作用于聚类**：`areg.json` 是 `vce(ols)` 抓取，非聚类路径本来就对，
第一版修正误伤了它（两个元素偏 9%），测试立刻抓到。
---

## 2. 探针结果

```
-- Areg against Stata areg --
  [OK  ] Areg coefficient       coefficient 0.1213388（与作者日志一致）
  [GAP ] Areg _cons             _cons=12.63564782 (se 0.07604164)
                                Stata areg: 12.64269254 (se 0.03383845)
                                点估计差 5.57e-04，标准误差 125%
  [NOTE] Areg clustered SE      SE 0.0164182 = reghdfe 值，非 areg 的 0.0171579
  [OK  ] Areg identity          e(cmd)='areg', e(title)='Linear regression, absorbing indicators'
-- controls --
  [OK  ] public surface         143 public names
```

**3 ok、1 note、1 gap、0 error。**

| 类别 | 含义 |
|---|---|
| `OK` | 正确 |
| `NOTE` | 已归因的预期差异，**不是缺陷** |
| `GAP` | 该修的问题 |

`Areg` 新功能让 `e(cmd)`、`e(title)` 这些**身份**字段对上了，这部分是 `OK`。
但 `_cons` 的**数值**和聚类标准误的**口径**是另外两件事，分别记在 §4 与 §1。

---

## 3. 曾记为缺口的一条：`rmse` 与 `df_r` 分母不同（已撤销）

这一条曾以"P2 内部不一致"记入本文档，后在**与真实 Stata `reghdfe` 逐位对照**后被撤销。
保留全过程，因为撤销的理由比原来的判断更有价值。

### 3.1 现象：确实存在两个不同的分母

同一次 `FEOLS` 拟合（cluster 于 `isocode`），三个量的分母不一致：

```
westata FEOLS (cluster):
    e(rmse) = 0.2608791     ← 隐含分母 rss/rmse² = 1400
    e(df_r) = 129           ← G−1 = 130−1
    bse     = 0.0164182     ← 与 xtreg, fe 相同
```

**现象本身是真的**：`e(rmse)` 的分母是 1400，`e(df_r)` 是 129，标准误的缩放因子又不同。
下面的问题不是"有没有这个现象"，而是**这个现象该归给谁**。

### 3.2 关键对照：真实 `reghdfe` 也是这么算的

判定"是否缺陷"的唯一依据是**对照实现**。用真实 Stata（StataNow 19.5 MP）跑同一份数据、
同一个设定：

```stata
. reghdfe ln_wpot … , absorb(isocode) vce(cluster isocode)
```

| 量 | 真实 `reghdfe` | westata `FEOLS` |
|---|---|---|
| `e(rmse)` | 0.2608790670546 | 0.2608790559441 |
| 隐含 `rmse` 分母 | **1400** | **1400** |
| `e(df_r)` | 129 | 129 |
| `b` | 0.1213387966 | 0.1213387957 |
| `se` | 0.0164181947 | 0.0164181944 |

**`reghdfe` 自己就同时报出 `rmse` 分母 1400 与 `df_r` 129。** 这不是 westata 的混用，
而是 `reghdfe` 的行为，westata 逐位复现了它。

在仓库自带的、可离线回放的 Stata 抓取上，这一点更清楚——
两个 fixture 的 `rmse` **完全相同**，而 `df_r` 从 148 变到 29：

| fixture | vce | `e(df_r)` | `e(rmse)` | `rss/rmse²` |
|---|---|---|---|---|
| `reghdfe_firm.json` | ols | 148 | 0.9416394217357702 | 148 |
| `reghdfe_firm_cluster.json` | cluster | **29** | **0.9416394217357702** | **148** |

`rmse` 不随聚类变化，`df_r` 随聚类变化——两者按设计就是不同口径。

`reghdfe` 源码对此有明确说明（`reghdfe.mata`，版本 6.14.1）：

```mata
// :3742
used_df_r = sol.N - S.df_a - sol.df_m - S.df_a_nested
// :3751
sol.rmse = used_df_r ? sqrt(sol.rss / used_df_r) : sqrt(sol.rss)
```

并在 `:4061-4066` 自陈：这样会 "breaks consistency with areg but ensures internal
consistency between vce(robust) and vce(cluster _n)"。**它是清醒的取舍，不是疏漏。**

### 3.3 所以原来错在哪

原判断的推理是：

> 因为 `rmse` 只对上 `areg`、标准误只对上 `xtreg`，所以两者不属于同一个模型，故为缺陷。

错在**把"我预期的自洽关系"当成了判据，而没有对照实现**。
真实参照系（`reghdfe`）给出的正是同一对数字。westata 的目标是复现 Stata，不是构造一个
比 Stata 更"自洽"的新约定。

**若按原建议去"修"**——让 `rmse` 改用嵌套排除口径——会直接破坏逐位一致性：

| fixture | 改为 `sqrt(rss/(N−rank))` | Stata `reghdfe` | 误差 |
|---|---|---|---|
| `reghdfe_firm_cluster` | 0.8610509917 | 0.9416394217357702 | **8.6%** |
| `reghdfe_firm_year_cluster` | 0.3751496271 | 0.41143452536210567 | **8.8%** |

现在的误差是 **0.0**。也就是说，原建议的"修复"会制造真正的缺陷。

### 3.4 源码位置与结论

- `src/westata/estimators/hdfe.py` —— `rmse = sqrt(rss / r2_denominator)`，
  其中 `r2_denominator = n_effective - rank_full`；`rank_full` 计入全部被吸收的
  固定效应水平。这正是 `reghdfe` 的 `used_df_r`。
- 同一文件的 `vcov` 复用 PyFixest 的协方差矩阵，其小样本因子按 `reghdfe` 的
  `dofadjustments(nested)` 排除嵌套固定效应——同样与 `reghdfe` 一致。

**结论：不是缺口，是忠实复现。** 已改为 `NOTE`，并由探针锁定这个口径。

### 3.5 顺带修正的文档错误

`hdfe.py` 的模块文档原先有两处与实测不符，已随本次修正更新：

| 原说法 | 实测 |
|---|---|
| `e(rmse) = sqrt(rss / e(df_r))` | 仅非聚类时成立；聚类时 `e(df_r)=29` 而 `rmse` 分母是 1400 |
| 嵌套固定效应会让 `e(rmse)` 一起缩小 | `rmse` 不随聚类变化（0.9416394217358 两种口径相同） |

`e(rank)` 也补了一条澄清：它是**自由度修正口径**的参数数，与 `reghdfe` 的 `e(rank)`
（文档定义为 `e(V)` 的秩，即只计报告出来的回归元）**不是同一个量**。要看后者的口径，
用 `e(df_m)`。

---

## 4. `Areg` 的 `_cons` 与 Stata `areg` 不符 —— 已修复

### 4.1 现象与修复结果

同一份数据、同一个设定（11 个马铃薯交互项 + 11 个年份虚拟变量 + 130 个国家固定效应，
聚类于 `isocode`）：

| | `_cons` 点估计 | `_cons` 标准误 |
|---|---|---|
| Stata `areg` | **12.64269254** | **0.03383845** |
| westata `Areg`（修复前） | 12.63564782 | 0.07604164 |
| **westata `Areg`（修复后）** | **12.64269254** | **0.03383845** |

**6 列的全部斜率参数一直是对的**（与作者日志 1e-6 内吻合），问题只在常数项。

### 4.1.1 根因：等权平均 vs 按组容量加权

`Areg` 原先取「组效应**等权**平均」。正确的规则是**按组容量加权**：

| 规则 | 结果 |
|---|---|
| 等权平均 | 12.63564782 ← 旧实现 |
| **按组容量加权** | **12.64269254** ← Stata |

两者只在**各组容量不等**时才不同。`areg.json` 是 30 个组 × 6 期的平衡面板，
**等权与加权完全相同**，所以那个 fixture 无法区分两种规则——这正是缺陷通过审查的原因。

标准误同理：不能用同方差的 `σ²(X'X)⁻¹`，必须用**聚类稳健**三明治。
本文数据上同方差给出 0.02306936，聚类给出 0.03383845（Stata 的值）。

### 4.2 为什么这算缺口

`Areg` 存在的理由就是提供 Stata `areg` 的语义。提交 `56304d4` 已经做到：

- `e(cmd) = "areg"`
- `e(title) = "Linear regression, absorbing indicators"`
- `e(predict) = "areg_p"`
- 报告 `_cons`（`FEOLS` 不报告）

但**报告一个与 Stata 数值不符的 `_cons`，比不报告更糟**——使用者会以为它可信。
标准误相差 125% 尤其严重：这会让常数项的置信区间宽约 2.2 倍。

### 4.3 可能的成因

提交说明提到，常数项的正确算法是「拟合完整设计（斜率 + 每组一个虚拟变量），再取
`c = (1/G)·[0,0,1,…,1]`」，这样才能同时得到 Stata 的方差与斜率协方差。

本篇的设定比该 fixture 复杂：**除被吸收的 130 个国家效应外，还有 11 个显式的年份虚拟变量**。
推测常数项的构造未正确处理"既有 absorbed 效应、又有显式虚拟变量"的情形。**这是推测，
需要看 `hdfe.py` 的 `_cons` 代码确认。**

### 4.4 修法

`Areg._with_intercept` 现在：

1. 拟合完整设计（斜率 + 每组一个虚拟变量）；
2. 取 `c = (1/Σn)·[0,…,0, n₁,…,n_G]`——**按组容量加权**的组效应均值；
3. 用**聚类稳健**三明治（同一个 `c` 向量）得到方差与协方差。

验证：点估计与标准误都与 Stata 逐位相同。

### 4.5 分级：P1

| 判据 | 情况 |
|---|---|
| 是否静默出错 | **是**——输出看起来正常，`_cons` 有值、有标准误，不报错 |
| 影响面 | 中：常数项通常不是研究关注点，但它是 `predict` 与 `margins` 的输入 |
| 是否有替代 | 有（`FEOLS` 不报常数项，反而不会误导） |

主结果（斜率）不受影响，但会误导需要常数项的使用者。**P1。**
（**已修复**，见 §4.1/§4.4。）

### 4.6 顺带修复：`vce(hc2)` / `vce(hc3)`

`areg.sthlp:47-49` 把 `vce(hc2 [clustvar])` 列为它的 SE/Robust 选项之一。
`Areg` 原先继承 `FEOLS` 的限制而**拒绝** HC2/HC3：

```
ValueError: cov_type 'HC2' is not supported by FEOLS: HC2 needs the leverage of
the full dummy design, which PyFixest does not expose for fixed-effect models
```

**这个理由对 `FEOLS` 成立，对 `Areg` 不成立**——`Areg` 为算 `_cons` 本来就构造了完整
设计矩阵，杠杆值 `h_jj` 就在手里。`areg.sthlp:202` 写得很清楚：

> `vce(hc2)` instead uses `u_j^2/(1-h_jj)` as the observation's variance estimate

现在按此实现：HC2 用 `u²/(1−h)`，HC3 用 `u²/(1−h)²`，两者都在完整设计上重算协方差
（含 `_cons` 那一行/列）。实测四者严格递增（HC0 < HC1 < HC2 < HC3），符合文档预期。

`FEOLS` 仍然拒绝 HC2/HC3——它的理由没有被削弱，测试里也锁住了这一点。

---

## 5. 复核方式

```bash
.venv/Scripts/python.exe replication/papers/nq2011/nq2011_probe_gaps.py
```

探针有四种结论：`OK`、`NOTE`（已知约定差异，不计为缺口）、`GAP`、`ERROR`。
**本篇当前预期是 `0 gap` + `1 note`。**

**测试守护**（`replication/tests/test_nq2011_replication.py`）：

| 测试 | 守住什么 |
|---|---|
| `test_westata_differs_from_areg_by_the_convention_ratio` | 每列 `areg/xtreg` 标准误比值落在 **[1.03, 1.06]** |
| `test_the_gap_probe_reports_the_rmse_convention_and_keeps_the_notes` | `rmse` 口径探针为 `OK` 且点明 `reghdfe` 来源；`areg` 的 `NOTE` 仍在 |
| `test_the_areg_xtreg_gap_comes_from_the_rmse_denominator` | `sqrt(1529/1400) = 1.0450563895` 这套算术成立 |
| `test_rmse_uses_the_full_rank_denominator_like_reghdfe` | `rmse` 不随聚类变化、分母为 N−rank_full |
| `test_feols_matches_the_reghdfe_fixture_bit_for_bit` | 与离线 Stata 抓取逐位一致（含 `rmse`/`ll`/`r2_a`/`df_r`）|

这些断言都是**双向**的：

- 比值跑出区间 → 可能是 westata 的真实缺陷，或 Stata 改了约定；
- `rmse` 口径探针转为 `GAP` → 说明 `rmse` 不再跟随 `reghdfe`，本文件的"忠实复现"结论失效；
- 与 fixture 的逐位一致被破坏 → `reghdfe` 抓取或 westata 实现有一方变了。

**没有使用"rmse 必须等于 sqrt(rss/e(df_r))"这种断言**——那正是本文档曾经犯的错：
把一条自己臆想的恒等式当成契约。

---
