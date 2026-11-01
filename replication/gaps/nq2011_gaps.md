# westata 可优化点清单 · Nunn & Qian (2011)

**来源**：Nunn & Qian (2011) 复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/nq2011/nq2011_probe_gaps.py`

> 配套报告：[REPORT.md](../papers/nq2011/REPORT.md)

---

## 0. 这一篇的结论：没有新缺口

本论文复现过程中，westata 的估计层没有出现问题。

**这一篇（NQ）没有发现新的 westata 问题。**

这不是因为没测——恰恰相反，这次测的是 westata 声称达标的**面板估计器**（`areg` / `xtreg` 路径），是前三次都没碰过的表面。结果是：

- 系数：**全部正确**；
- 标准误：与 Stata 的 `xtreg, fe` **精确一致**（1e-13）；
- 与 `areg` 的 4.5% 差异：**是 Stata 内部两个命令的约定分歧，不是 westata 的缺陷**。

**所以本文件记录的是"为什么这次没有缺口"，而不是硬凑几条。** 这本身有价值：它说明 westata 在加权 OLS + 高维固定效应这条路径上是干净的。

---

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
  [OK] areg-equivalent (FEOLS)      coefficient 0.1213388
  [OK] xtreg-equivalent (PanelOLS)  coefficient 0.1213388
  [OK] Stata labels                 stata_param_names exposes _cons
-- agreement --
  [OK] vs xtreg, fe                 SE rel diff 2.1e-13
  [NOTE] vs areg                    SE rel diff 4.3e-02 (convention, not a defect)
-- controls --
  [OK] public surface intact        142 public names
```

**0 gap、0 error。** `NOTE` 是这一篇特设的类别：**已知且已归因的预期差异**，不计为缺口，但必须显式记录——否则下一次复现的人会把它当成新缺陷重新调查一遍。

---

## 3. 一个建议：把约定写进文档

虽然这不是缺口，但值得做一件小事：

> **在 `docs/USING_WITH_STATA.md` 里加一节，说明 westata 的面板标准误跟随 `xtreg`/`reghdfe` 而非 `areg`。**

理由：

- 这是一个**会被误判为缺陷**的差异，而且误判方向是低估 westata；
- 已有文档列了"已知偏差"清单（`xtgls`、`teffects nnmatch` 等），但那些是 westata 的数值偏差；这一条相反——是 westata 正确、而与某个 Stata 命令的约定不同；
- 成本极低（几行文字），收益是省掉未来每次复现的一轮排查。

这条建议不是"缺口"，所以不影响探针的 `0 gap` 计数。

---

## 4. 复核方式

```bash
.venv/Scripts/python.exe replication/papers/nq2011/nq2011_probe_gaps.py
```

探针有五种结论：`OK`、`NOTE`（已知约定差异）、`GAP`、`BOUNDARY`、`ERROR`。本篇预期是 `0 gap`。

**测试守护**：`replication/tests/test_nq2011_replication.py` 中的
`test_westata_differs_from_areg_by_the_convention_ratio` 会检查每列的 `areg/xtreg` 标准误比值落在 **[1.03, 1.06]** 区间内。

这个断言的作用是双向的：

- 若某列跑出区间外 → 可能是 westata 的真实缺陷，或 Stata 改了约定；
- 若某列**接近 1.0**（westata 同时匹配 `areg`）→ 说明约定故事已过时，本文件需重写。

---
