# westata 可优化点清单 · Acemoglu, Cantoni, Johnson & Robinson (2001)

**来源**：AJR (2001)《The Consequences of Radical Reform: The French Revolution》复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/ajr2001/ajr2001_probe_gaps.py`

> 配套报告：[REPORT.md](../papers/ajr2001/REPORT.md)
> 本文只讲 westata 本身**在自身 API 契约下**还可以改进什么。

---

## 0. 结论：0 个缺口（1 条已修复）

| 条目 | 性质 | 状态 |
|---|---|---|
| `PanelOLS` 不接受权重 | 能力缺失（P2） | **已修复** |
| 系数、聚类标准误、联合检验 | 全部正确 | — |

`PanelOLS` 现在接受 `weights=`/`weight_type=`，加权拟合与 Stata `xtreg, fe` 逐位相同：

| 量 | westata `PanelOLS`（加权） | Stata `xtreg` |
|---|---|---|
| `fpresence1900` 系数 | 0.633869 | 0.633869 |
| 聚类标准误 | 0.40751063 | 0.40751063 |

`xtreg` 是本篇的核心命令。修复前 westata 的对应物 `PanelOLS` **没有 `weights` 参数**，
论文 5 列中有 4 列（加权列）只能绕道 `FEOLS`；现在 `PanelOLS` 可直接做。

---

## 1. `PanelOLS` 不支持权重（P2）—— 已修复

### 1.1 现象（修复前）

论文的命令是 `xtreg urbrate ... [aweight=totalpop1750], fe cluster(id)`。
按文档映射，`xtreg` 对应 `PanelOLS`，但它原先接受不了权重：

```
PanelOLS.from_formula(formula, data, *, entity, time, subset=None)   # 修复前
PanelOLS.from_formula(formula, data, *, entity, time, subset=None,
                      weights=None, weight_type=None)                 # 现在
```

现在可用：

```python
ws.PanelOLS.from_formula(
    "urbrate ~ " + " + ".join(spec.regressors), frame,
    entity="id", time="year",
    weights=frame["totalpop1750"], weight_type="aweight",
).fit(cov_type="cluster", cov_kwds={"cluster": frame["id"]})
```

**两个实现要点**（都属于"不写出来就会静默出错"的那类）：

1. **权重缺失要删行**。Stata 的 `[aweight=]` 把权重缺失的观测标记为不使用，
   所以样本是**交集**。本篇原始数据 209 行里有 **95 个权重缺失**，忽略这一点会让 N
   算错，然后报「weights contain missing values」——症状看起来像另一个 bug。
2. **权重按位置对齐，不按标签**。调用者传入的 Series 带的是长表 `RangeIndex`，
   而面板索引是 `(entity, time)` MultiIndex，按标签 join 会得到全 NaN。
   "每行一个权重" 说的就是行序。

### 1.2 影响（修复前的处境）

本篇 5 个复现规格中，**4 个是加权的**（Table 3 列 1、3，Table 4 列 1，以及不含年份
虚拟变量的变体）。修复前用 `PanelOLS` 只能做那 1 个无权重列，其余 4 列必须绕道 `FEOLS`。

这不是"少一个选项"——`xtreg, fe [aweight=]` 是实证论文里非常常见的组合。这也是它被
评为 P2（而非可以长期搁置）的原因：影响的是**文档指定的那个对应物能不能直接完成任务**。

### 1.3 现有的替代路径

`FEOLS` 与 `Areg` 都接受 `weights`/`weight_type`，且都通过吸收实体效应实现同一 within
估计量——但**三者的正确对照命令不同**，这一点曾经把本文件引向一个错误结论（§1.5）：

| 估计器 | 接受权重 | 与哪个 Stata 命令对齐 |
|---|---|---|
| `PanelOLS` | ✓（现已支持） | `xtreg, fe` — 系数与聚类 SE 均精确吻合 |
| `FEOLS` | ✓ | `xtreg` / `reghdfe` — 精确吻合（修复前本篇走这条） |
| `Areg` | ✓ | **`areg`** — 系数与 SE 均精确吻合（见 §1.5） |

修复前后"能否完成任务"都不受影响——差别在于是否必须绕道。**这也是当初把它评为 P2、
而不是 P0 的理由：不误导、有替代，只是映射路径不完整。**

### 1.4 分级：P2

| 判据 | 情况 |
|---|---|
| 是否静默出错 | **否**，`TypeError` 立刻报错 |
| 是否有替代 | **有**，`FEOLS` 精确可用 |
| 影响面 | 中：`PanelOLS` 是文档里 `xtreg` 的对应物，用户会先试它 |

不会算错，且有替代路径，但对应物缺参数会让使用者绕路。**P2。**（已修复，分级保留作为
当时的判断依据——它解释了为什么这条该修却不紧急。）

### 1.5 更正：`Areg` 的加权聚类标准误**并非缺陷**（原诊断用错了对照）

初版这里写着「`Areg` 的标准误偏离 11%，成因未定位」。**那个诊断是错的**，错在对照基准。

实测两个 Stata 命令，它们本来就给不同的标准误：

| 命令 | 1900 年系数 | 聚类标准误 |
|---|---|---|
| Stata `areg ... [aweight], absorb(id) vce(cluster id)` | 0.633869 | **0.45292274** |
| Stata `xtreg ... [aweight], fe vce(cluster id)` | 0.633869 | **0.40751063** |
| westata `Areg` | 0.633869 | **0.45292274** ✅ = `areg` |
| westata `FEOLS` | 0.633869 | **0.40751063** ✅ = `xtreg` |

**两个 westata 估计器都对**，各自对齐各自的 Stata 命令。`Areg` 声称替代的是 `areg`，
拿它与 `xtreg` 比就必然差 11%——**差的不是实现，是参照系**。

> 教训：**「数值对不上」时，先确认对的是不是同一个东西**。这条和 §1.4 里「`Areg` 只报
> 斜率」是同一类错误的两个方向——一个是功能缺失被当成能跑，一个是正确结果被当成缺陷。
> 探针原先把它记为 `NOTE`（未定位成因），现在改成 `OK` 并锁死 0.45292274。

---

## 2. 探针结果

```
-- estimator surface --
  [OK   ] PanelOLS weights       weighted x1 coefficient 0.633869, clustered SE 0.407511
  [OK   ] FEOLS weighted FE      coefficient 0.633869, clustered SE 0.407511
  [OK   ] Areg weighted SE       coefficient 0.633869, clustered SE 0.45292274 (= Stata areg)

-- controls --
  [OK   ] public surface         143 public names
```

**4 ok, 0 note, 0 gap, 0 error。**

（修复前是 `2 ok, 1 note, 1 gap`：`PanelOLS weights` 为 `GAP`，`Areg weighted SE` 为
`NOTE`——后者是因为当时拿它与 `xtreg` 对照，见 §1.5。）

---

## 3. 复核方式

```bash
.venv/Scripts/python.exe replication/papers/ajr2001/ajr2001_probe_gaps.py
```

探针有四种结论：`OK`、`GAP`、`NOTE`、`ERROR`。**本篇当前预期是 `0 gap`。**

**测试守护**（`replication/tests/test_ajr2001_replication.py`）：

| 测试 | 守住什么 |
|---|---|
| `test_westata_matches_stata_on_every_parameter` | 系数与 SE 对齐到 1e-9 |
| `test_the_joint_wald_test_matches` | 联合 Wald F 统计量一致 |
| `test_panelols_accepts_the_paper_weight` | `PanelOLS` 有 `weights=`/`weight_type=` 两个参数（方向已随修复反转） |
| `test_panelols_weighted_matches_xtreg` | 加权拟合复现 Stata 的 0.633869 / 0.407511，而不只是"能跑" |
| `test_weights_require_a_weight_type` | 给了权重必须给类型（Stata 从不猜权重类型） |

这些断言都是双向的：失败既可能意味着 westata 退化，也可能意味着行为再次改变、文档过期。

**只断言签名存在是不够的**：一个接了参数但算错的实现会直接通过签名检查——那正是本文件
存在的意义所要防的失效形态，所以数值断言与签名断言并列。
