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

### 1.3 差异的机制

`areg` 与 `xtreg` 在聚类稳健方差的**小样本调整因子**里对 `K` 的计数不同：

$$\text{因子} = \frac{G}{G-1} \cdot \frac{N-1}{N-K}$$

- `xtreg`（以及 `reghdfe`、westata）：`K` 只计显式回归元（本例 23 个）；
- `areg`：`K` 把被吸收的 130 个国别固定效应也计入（`K ≈ 153`）。

代入 N = 1552、G = 130：

$$\sqrt{\frac{N-23}{N-153}} = 1.0454$$

与实测的 4.505% 吻合（各列 1.0451–1.0465，随列的控制变量数变化）。

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
