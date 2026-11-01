# westata 可优化点清单 · Nunn & Qian (2011)

**来源**：Nunn & Qian (2011) 复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/nq2011/nq2011_probe_gaps.py`

> 配套报告：[REPORT.md](../papers/nq2011/REPORT.md)

---

## 0. 结论：0 个缺口

本论文复现过程中，westata 的**系数**全部正确。两条一度引起怀疑的现象，经对照后**都不是
westata 的缺陷**：

| 条目 | 性质 | 探针结论 |
|---|---|---|
| 与 `areg` 的 4.5% 标准误差异 | Stata 内部约定分歧 | `NOTE`（§1）|
| `rmse` 与 `df_r` 分母不同 | **reghdfe 自身的行为**，westata 忠实复现 | `OK`（§3）|

第二条曾是本文档记录的"缺口"。**经与真实 Stata 的 `reghdfe` 逐位对照后，这个判定被推翻**——详见 §3。

> **判据教训**：把"我认为不一致"当成"确实有缺陷"，是本文档犯过的最大错误。
> 唯一能判定的是**对照实现**：把 westata 与真实 `reghdfe` 的输出逐位比较，而不是与
> "我预期的自洽关系"比较。§3 记录了这个纠正过程。

## 1. 最关键的一点：判定差异的归属需要对照

### 1.1 现象

同一条命令链、同一份数据、同一个系数，标准误不同：

| 路径 | 系数 | 标准误 |
|---|---|---|
| Stata `areg`（作者当年跑的） | 0.1213388 | **0.0171579** |
| Stata `xtreg, fe` | 0.1213388 | **0.0164182** |
| westata `FEOLS` | 0.1213388 | **0.0164182** |
| westata `PanelOLS` | 0.1213388 | **0.0164182** |

如果只对照"论文报告的 0.0171579"，会得出"westata 标准误错 4.5%"的结论。**但这个结论是错的。**

### 1.2 正确的判定方式

关键在于：**Stata 自己就不同意自己**。`areg` 与 `xtreg, fe` 给出同样的系数、不同的标准误。所以"对照论文"这个单一基准不足以判定谁对。

正确的做法是**同时对照两个 Stata 命令**：

```
westata vs xtreg, fe  →  se 相对差 1e-13      （机器精度）
westata vs areg       →  se 相对差 4.3e-02    （4.5%）
```

这就把问题从"westata 是否算错"变成"westata 实现的是哪一种约定"，而后者可以回答：**westata 与 `xtreg`/`reghdfe` 一致**。

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

---

## 2. 探针结果

```
-- estimator surface --
  [OK  ] rmse vs se convention      e(rmse) keeps the full-rank denominator 1400, as
                                    reghdfe does, while e(df_r)=129.0 follows the
                                    clustered one; this is reghdfe's documented
                                    behaviour, not an inconsistency
  [OK  ] areg-equivalent (FEOLS)    coefficient 0.1213388
  [OK  ] xtreg-equivalent (PanelOLS) coefficient 0.1213388
-- agreement --
  [OK  ] Stata labels               22 parameters carry Stata-style labels
  [NOTE] areg versus xtreg          SE 0.0164182 与 xtreg 一致；areg 的 0.0171579
                                    差 4.5%，属 Stata 内部约定，不是 westata 缺陷
-- controls --
  [OK  ] public surface             142 public names
```

**5 ok、1 note、0 gap、0 error。**

`OK` / `NOTE` / `GAP` 的区别是本文件的关键：

| 类别 | 含义 | 例 |
|---|---|---|
| `OK` | 与参照实现一致 | `rmse` 口径——与真实 `reghdfe` 逐位相同（§3）|
| `NOTE` | 已归因的预期差异，**不是缺陷** | 与 `areg` 的 4.5% 差距——Stata 自己两个命令就不一致 |
| `GAP` | 该修的问题 | —— 本文件目前没有 |

**注意**：`rmse vs se convention` 这一条最初不存在。初版探针只对比系数和 `xtreg` 标准误，
全部通过，于是文档写了"0 gap"。是追问"是否该实现 `areg`"才促成了这项检查——
**"每个数字都能对上某个 Stata 命令"不等于"输出内部自洽"**，这两件事必须分开验证。

但**这项检查本身后来又犯了一次方向相反的错**：它把"我自己预期的自洽关系"当成了契约，
于是把 `reghdfe` 的正常行为报成了缺口。真正的判据只有对照实现——见 §3。

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

## 4. 复核方式

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
