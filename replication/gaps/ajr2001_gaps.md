# westata 可优化点清单 · Acemoglu, Cantoni, Johnson & Robinson (2001)

**来源**：AJR (2001)《The Consequences of Radical Reform: The French Revolution》复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/ajr2001/ajr2001_probe_gaps.py`

> 配套报告：[REPORT.md](../papers/ajr2001/REPORT.md)
> 本文只讲 westata 本身**在自身 API 契约下**还可以改进什么。

---

## 0. 结论：1 个缺口

本论文复现过程中发现 **1 个能力缺口**：

| 条目 | 性质 | 状态 |
|---|---|---|
| **`PanelOLS` 不接受权重** | 能力缺失（P2） | **未修复** |
| 系数、聚类标准误、联合检验 | 全部正确 | — |

`xtreg` 是本篇的核心命令，而它在 westata 的对应物 `PanelOLS` **没有 `weights` 参数**，
因此论文 5 列中有 4 列（加权列）无法用它估计。

---

## 1. 缺口：`PanelOLS` 不支持权重（P2，未修复）

### 1.1 现象

论文的命令是 `xtreg urbrate ... [aweight=totalpop1750], fe cluster(id)`。
按文档映射，`xtreg` 对应 `PanelOLS`：

```python
ws.PanelOLS.from_formula("urbrate ~ x1 + x2", df, entity="id", time="year")
```

但 `PanelOLS.from_formula` 的签名里**没有 `weights`**：

```
PanelOLS.from_formula(formula, data, *, entity, time, subset=None)
```

传进去直接报错：

```python
ws.PanelOLS.from_formula(..., weights=df.totalpop1750)
# TypeError: PanelOLS.from_formula() got an unexpected keyword argument 'weights'
```

### 1.2 影响

本篇 5 个复现规格中，**4 个是加权的**（Table 3 列 1、3，Table 4 列 1，以及不含年份
虚拟变量的变体）。用 `PanelOLS` 只能做那 1 个无权重列。

这不是"少一个选项"——`xtreg, fe [aweight=]` 是实证论文里非常常见的组合。

### 1.3 现有的替代路径

`FEOLS` 与 `Areg` 都接受 `weights`/`weight_type`，且都通过吸收实体效应实现同一 within 估计量：

| 估计器 | 接受权重 | 能否复现 `xtreg, fe` |
|---|---|---|
| `PanelOLS` | ✗ | 否（无法加权） |
| `FEOLS` | ✓ | **能**（本篇用它，系数与 SE 均精确吻合） |
| `Areg` | ✓ | 系数能，**聚类 SE 不能**（见 §1.5） |

所以**缺口不影响能否完成任务**，只影响"用文档指定的那个对应物能否完成任务"。

### 1.4 分级：P2

| 判据 | 情况 |
|---|---|
| 是否静默出错 | **否**，`TypeError` 立刻报错 |
| 是否有替代 | **有**，`FEOLS` 精确可用 |
| 影响面 | 中：`PanelOLS` 是文档里 `xtreg` 的对应物，用户会先试它 |

不会算错，且有替代路径，但对应物缺参数会让使用者绕路。**P2。**

### 1.5 顺带记录：`Areg` 的加权聚类标准误

同为加权固定效应，`FEOLS` 与 `Areg` 的**系数相同、标准误不同**：

| 路径 | 1900 年系数 | 聚类标准误 |
|---|---|---|
| Stata `xtreg, fe [aweight]` | 0.633869 | **0.407511** |
| westata `FEOLS` | 0.633869 | **0.407511** |
| westata `Areg` | 0.633869 | 0.452923 |

`Areg` 的标准误偏离 11%。它与 NQ 那篇记录的 `_cons` 问题可能同源（都在 `Areg` 的
方差构造路径上），但**本篇只观察到现象，未定位成因**，因此单列记录，不并入 §1。

---

## 2. 探针结果

```
-- estimator surface --
  [GAP ] PanelOLS weights        PanelOLS.from_formula 不接受 weights=（论文 4/5 列需要它）
  [OK  ] FEOLS weighted FE        coefficient 0.633869, SE 0.407511（与 Stata 一致）
-- agreement --
  [OK  ] coefficients             最大相对差 5.17e-13
  [OK  ] clustered standard errors 最大相对差 3.99e-14
  [OK  ] joint Wald test          F 统计量逐位相同
-- controls --
  [OK  ] public surface           143 public names
```

**1 gap、0 error。**

---

## 3. 复核方式

```bash
.venv/Scripts/python.exe replication/papers/ajr2001/ajr2001_probe_gaps.py
```

探针有四种结论：`OK`、`GAP`、`NOTE`、`ERROR`。**本篇当前预期是 `1 gap`。**

**测试守护**（`replication/tests/test_ajr2001_replication.py`）：

| 测试 | 守住什么 |
|---|---|
| `test_westata_matches_stata_on_every_parameter` | 系数与 SE 对齐到 1e-9 |
| `test_the_joint_test_matches` | 联合 Wald F 统计量一致 |
| `test_panelols_still_rejects_weights` | §1 的缺口仍在（若哪天支持了，本文件需改成"已修复"） |

三条都是双向的：断言失败既可能意味着 westata 退化，也可能意味着它已修复、文档过期。
