# westata 可优化点清单 · Nunn & Qian (2011)

**来源**：Nunn & Qian (2011) 复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/nq2011/nq2011_probe_gaps.py`

> 配套报告：[REPORT.md](../papers/nq2011/REPORT.md)

---

## 0. 结论：1 个缺口，1 条已知约定差异

本论文复现过程中，westata 的**系数**全部正确，但发现了两件事：

| 条目 | 性质 | 状态 |
|---|---|---|
| **`rmse` 与 `df_r` / 标准误使用两套自由度约定** | **内部不一致（P2）** | **未修复** |
| 与 `areg` 的 4.5% 标准误差异 | Stata 内部约定分歧 | 不是缺陷，但暴露了上一条 |

**第一条是本文件唯一记录的缺口**，而且它是我在追问"是否需要实现 `areg` 效果"时才挖出来的——
初版本文档只写了"没有缺口"，因为当时只对照了系数和 `xtreg`。

细节见 §3。

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
-- probes --
  [GAP ] rmse vs se convention      e(rmse)=0.2608791 隐含分母 1400 = N−152（areg 口径），
                                    而标准误跟随 xtreg（分母 N−23 = 1529）
  [OK  ] areg-equivalent (FEOLS)    coefficient 0.1213388
  [OK  ] xtreg-equivalent (PanelOLS) coefficient 0.1213388
-- agreement --
  [OK  ] Stata labels               22 parameters carry Stata-style labels
  [NOTE] areg versus xtreg          SE 0.0164182 与 xtreg 一致；areg 的 0.0171579
                                    差 4.5%，属 Stata 内部约定，不是 westata 缺陷
-- controls --
  [OK  ] public surface             142 public names
```

**1 gap、1 note、0 error。**

`NOTE` 与 `GAP` 的区别是本文件的关键：

| 类别 | 含义 | 例 |
|---|---|---|
| `NOTE` | 已归因的预期差异，**不是缺陷** | 与 `areg` 的 4.5% 差距——Stata 自己两个命令就不一致 |
| `GAP` | 该修的问题 | `rmse` 与标准误用了两套自由度，输出内部矛盾 |

**注意**：`rmse vs se convention` 这一条最初不存在。初版探针只对比系数和 `xtreg` 标准误，
全部通过，于是文档写了"0 gap"。是追问"是否该实现 `areg`"才促成了这项检查——
**"每个数字都能对上某个 Stata 命令"不等于"输出内部自洽"**，这两件事必须分开验证。

---

## 3. 缺口：`rmse` 与标准误使用两套自由度约定（P2，未修复）

这一条是在追问"westata 是否需要实现 `areg` 的效果"时发现的。初版本文件只写了"没有缺口"，
因为当时只核对了系数和 `xtreg` 标准误，没有检查输出内部是否自洽。

### 3.1 现象

同一次 `FEOLS` 拟合，`e(rmse)` 与 `e(df_r)` 来自两套不同的约定：

```
westata FEOLS (cluster):
    e(rmse) = 0.2608791     ← 与 Stata areg 相同（分母 N−152 = 1400）
    e(df_r) = 129           ← 与 reghdfe / xtreg 相同
    bse     = 0.0164182     ← 与 xtreg, fe 相同

Stata areg    : rmse = 0.2608791   (分母 1400)
Stata xtreg fe: rmse = 0.2496316   (分母 1529)
```

**`rmse` 跟随 `areg`，而 `df_r` 与标准误跟随 `reghdfe`。** 三者不是同一套约定。

### 3.2 为什么这是缺陷而不是"跟随某个约定"

Stata 的两个命令各自内部自洽：

| 命令 | rmse 分母 | 标准误缩放 | 自洽？ |
|---|---|---|---|
| `areg` | N−152 = 1400 | 同一个分母 | ✅ |
| `xtreg, fe` | N−23 = 1529 | 同一个分母 | ✅ |
| **westata `FEOLS`** | **1400（areg）** | **1529（xtreg）** | ❌ **混用** |

westata 取了 `areg` 的 `rmse` 和 `xtreg` 的方差，两者不是同一个模型的对数似然。
后果：`e(ll)`、`e(r2_a)` 这类从 `rmse` 推导的量，与 `e(bse)` 不在同一个概率模型下。
单看任何一项都"能对上某个 Stata 命令"，合起来却不自洽。

### 3.3 源码位置

- `src/westata/estimators/hdfe.py:599` —— `rmse = sqrt(rss / (n_effective - rank_full))`，走 `areg` 口径
- `src/westata/estimators/hdfe.py:306` —— `vcov` 直接复用 PyFixest 的协方差矩阵，走 `reghdfe` 口径
- 两条路径各自演化，没有共享同一个自由度定义

### 3.4 已有的部分机制

`hdfe.py:251` 有一个 `_absorb_df_drops_constant` 开关，注释明确写着它是用来在
`areg` 与 `reghdfe` 的 `df_a` 约定之间切换的。**但它只影响 `e(df_a)`，没有传导到 `rmse` 与方差缩放。**

这就是修复的抓手：把已有的开关扩展成"完整约定选择"，一处控制三项 ——
`df_a`、`rmse` 分母、方差缩放。

### 3.5 建议的修复

**第一步（该做，与 `areg` 无关）**：消除内部不一致。
让 `rmse` 与方差缩放使用同一个自由度定义。默认跟随 `reghdfe`/`xtreg`（更主流，
且当前标准误已是这个口径），此时 `rmse` 会从 0.2608791 变成 0.2496316。

**第二步（可选）**：把上面的开关提升为公开参数，例如

```python
ws.FEOLS.from_formula(..., convention="areg")      # rmse 与方差都用 1400
ws.FEOLS.from_formula(..., convention="reghdfe")   # 都用 1529（默认）
```

这样才能真正做到"复现用 `areg` 报告的论文"。

**为什么第二步不是必须**：`areg` 与 `xtreg` 都是合法约定，`xtreg`/`reghdfe` 是更主流的一个。
westata 现在跟随它，不算错。但如果**第一步不做**，用户拿到的是一个内部矛盾的输出——
这才是必须修的部分。

### 3.6 分级：P2

| 判据 | 情况 |
|---|---|
| 是否静默出错 | **是**——系数和标准误都对得上某个 Stata 命令，`rmse` 也对得上另一个，单看都正常 |
| 影响面 | 窄：只在"高维固定效应 + 需要 `e(rmse)`/`e(ll)`/`e(r2_a)`"时暴露 |
| 是否有替代 | 有（`PanelOLS` 走 `xtreg` 口径，内部自洽） |

不误导主结果，但输出内部不自洽。**P2。**

---

## 4. 复核方式

```bash
.venv/Scripts/python.exe replication/papers/nq2011/nq2011_probe_gaps.py
```

探针有四种结论：`OK`、`NOTE`（已知约定差异，不计为缺口）、`GAP`、`ERROR`。**本篇当前预期是 `1 gap`（§3）+ `1 note`。**

**测试守护**（`replication/tests/test_nq2011_replication.py`）：

| 测试 | 守住什么 |
|---|---|
| `test_westata_differs_from_areg_by_the_convention_ratio` | 每列 `areg/xtreg` 标准误比值落在 **[1.03, 1.06]** |
| `test_the_gap_probe_reports_the_rmse_inconsistency_and_keeps_the_note` | §3 的缺口仍在、§1 的 NOTE 仍在 |
| `test_the_areg_xtreg_gap_comes_from_the_rmse_denominator` | `sqrt(1529/1400) = 1.0450563895` 这套算术成立 |

三条断言都是**双向**的：

- 比值跑出区间 → 可能是 westata 的真实缺陷，或 Stata 改了约定；
- 缺口探针转为 `OK` → 说明 §3 已修复，本文件需改成"已修复"条目；
- 算术不再成立 → §1.3 的机制解释失效，需重新定位。

---
