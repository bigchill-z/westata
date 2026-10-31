# westata 可优化点清单 · 第三篇（Autor & Dorn 1993/2013）

**来源**：Autor & Dorn (2013) 复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/ad2013/probe_gaps.py`

> 配套报告：[REPORT.md](REPORT.md)
> 前两篇：[../adh2013/WESTATA_GAPS.md](../adh2013/WESTATA_GAPS.md)、[../ck1994/WESTATA_GAPS.md](../ck1994/WESTATA_GAPS.md)

---

## 0. 三篇复现，三种不同的发现

| 篇目 | 发现的类别 | 结果 |
|---|---|---|
| ADH 2013 | **能力缺失**：加权 IV 不支持等 | 7 条，**上游已全部修复** |
| CK 1994 | **边界**：无数据操作层 | 0 缺口、1 边界（README 已声明） |
| **AD 2013（本篇）** | **契约缺陷**：签名接受了参数却不按参考实现的方式工作 | **3 条，已全部修复** |

这一篇的价值在于：它暴露的不是"缺功能"，也不是"不做的边界"，而是**一个用户无法预见的失败**——调用看起来完全合理，报错信息却让人以为是自己数据的问题。

修复之后，这一篇记录的是「那个失败长什么样、为什么它是缺陷、怎么修的」，而 §3 的探针反过来充当**回归测试**。

---

## 1. 核心缺陷：权重没有随样本一起收缩 —— 已修复

### 1.1 现象（修复前）

```python
ws.OLS.from_formula(
    "d_rpc ~ l_sh_routine33a + <48 个州虚拟变量>",
    frame,                      # 722 行，其中 d_rpc 有 47 个缺失
    weights=frame["timepwt48"], # 也是 722 个元素，与 frame 对齐
    weight_type="aweight",
).fit()
```

```
WeightError: weights have 722 entries but the data have 675 rows
```

**这个调用是合理的**：权重向量与传入的 `frame` 完全对齐，是 API 签名要求的形式。报错信息读起来像"你的数据有问题"，但数据没有问题。

### 1.2 原因

`OLS.from_formula` 内部：

```python
X, names, kept_index = build_design(formula, data)   # patsy 丢掉 47 个含缺失的行
y = data.loc[kept_index, lhs]                        # y 收缩到 675
model = cls(y, X, weights=weights, ...)              # weights 仍是 722 ← 缺陷所在
```

`weights` 被原样透传，**没有按 `kept_index` 收缩**。

### 1.3 为什么这是缺陷而不是设计选择

两个参照实现都不这样：

| 参照实现 | 行为 |
|---|---|
| Stata 的 `[aw=w]` | 权重随估计样本一起丢弃行，用户无需干预 |
| `statsmodels.formula.api.ols` | `weights` 接受**列名**，在模型框架上统一做 `dropna`，权重天然跟着收缩 |

也就是说，用户的预期不是"异想天开"，而是**两个参照实现的一致行为**。

### 1.4 修复

在 `estimators/linear.py` 的 `from_formula` 里，`weights` 现在跟着估计样本收缩：

```python
X, names, kept_index = build_design(formula, data)
...
if weights is not None:
    weights = _slice_weights_to_sample(weights, kept_index, len(data))
```

`_slice_weights_to_sample` 接受**两种长度**：与 `data` 对齐的全长向量（按 `kept_index` 切），或已经切好的估计样本长度（原样通过，向后兼容）。两种都不是时，报错**同时给出两个数字和原因**：

```
weights have 500 entries; expected 722 (one per row of the data) or 675
(one per estimation-sample row). The formula drops rows with missing values,
which is why the estimation sample can be shorter than the frame.
```

**修复前**的报错是「weights have 722 entries but the data have 675 rows」——描述的是症状，且指向调用者的数据。

实测：修复后该调用给出 **0.6950717**，与作者日志逐位相同（对照项亦为同值）。

---

## 2. 相关的两个次要缺口 —— 均已修复

### 2.1 权重不接受列名 —— 已修复

```python
ws.OLS.from_formula(formula, frame, weights="timepwt48", weight_type="aweight")
# 修复前 → ValueError；修复后可用
```

statsmodels 的 `from_formula` 接受列名，这正是它能把 `dropna` 处理得透明的原因：权重是模型框架的一列，自然与其他变量一起收缩。

现在 `weights` 同时接受数组与列名。列名由 `_resolve_weight_column` 在构建设计矩阵**之前**解析——这样读的就是公式实际求值的那一帧。字符串不是列名时明确报错并列出可用列，而不是猜成一个单元素数组。

### 2.2 文档没提"会丢行" —— 已修复

`OLS.from_formula` 的 docstring 现在写明：

- 公式涉及的任何变量有缺失值的行**会被丢弃**（与 Stata 估计命令、`statsmodels.formula.api` 一致）；
- `weights` 跟着那个样本走，两种形式（列名 / 对齐向量）都正确。

一个只读签名的调用者原本无法知道权重和设计矩阵可能长度不一致——**报错来得突然，是因为文档没预告**。

---

## 3. 探针结果

**修复前：**

```
-- probes --
  [GAP] aligned weights    ...WeightError
  [GAP] weights by name    ...ValueError
  [GAP] documentation      the docstring does not mention row-dropping
-- controls --
  [OK ] pre-aligned call   pre-aligned call reproduces 0.6950717
1 ok, 3 gap, 0 error
```

**修复后：**

```
-- probes --
  [OK ] aligned weights    aligned weights accepted; coefficient 0.6950717
  [OK ] weights by name    weights accepted as a column name
  [OK ] documentation      the docstring documents the row-dropping behaviour
-- controls --
  [OK ] pre-aligned call   pre-aligned call reproduces 0.6950717
4 ok, 0 gap, 0 error
```

对照项 `pre-aligned call` 的意义始终没变：证明**估计本身一直是对的**。修复前后它都是 0.6950717，与作者日志逐位相同。所以问题在契约，不在估计。

---

## 4. 本次复现的绕法：修复后不再是必需，但保留

`run_westata.py` 在建模前先丢弃模型会用到的列中的缺失行：

```python
needed = [spec.depvar, spec.term, *spec.extra_regressors, *BASE_COLUMNS]
frame = frame.dropna(subset=needed)
```

**修复后这段代码不再是为了绕过缺陷**——库现在会自己把权重收缩到估计样本。保留它的理由变成了两个独立的：

- 它还负责 `spec.condition` 的样本筛选，那是这篇论文的样本定义，与库的行为无关；
- 显式丢行让**样本构造这一步在代码里可见**，而这正是第三篇的核心教训（见下面的结论）。

也就是说：它从"绕法"变成了"显式的样本构造"。同一个动作，理由不同了。

---

## 5. 判定标准（沿用前两篇，略作补充）

| 类别 | 含义 | 该怎么做 |
|---|---|---|
| **GAP** | 声称提供、或参照实现都支持，但不能用 | **该修** |
| **BOUNDARY** | 明确声明不做 | 该规划 |
| **OK** | 正常 | — |

本篇新增一条判据的细化：**"参照实现都这么做"是判定 GAP 的关键依据**。§1.3 用 Stata 与 statsmodels 的双重对照说明这一点；没有这个对照，就很容易把契约缺陷说成"设计差异"而放过。

---

## 6. 三篇的累计视图

| | ADH 2013 | CK 1994 | AD 2013 |
|---|---|---|---|
| 估计量 | 加权 2SLS + 聚类 | 无权重 OLS | 加权 OLS + 48 个州虚拟变量 |
| 参数数 | 178 | 26 | 546 |
| 最大相对差 | 1.6e-11 | 2.1e-13 | **2.9e-11** |
| 发现 | 7 条缺口（**已修复**） | 1 条边界 | **3 条契约缺陷（已修复）** |

**累计结论**：

> westata 的**估计精度**在三篇、覆盖 750 个成对参数后仍然成立——最大差异都在 1e-11 以内。
> 真正的问题都在**估计之外**：第一篇是功能缺失（已修），第二篇是数据层边界，**第三篇是 API 契约**——而后者的隐蔽性最强，因为它表现为一个误导性的报错。

**三篇的清单现在全部归零，合计 10 条修复。** 更有价值的是它们各自暴露的**问题类型**不同：

| 类型 | 例 | 能否靠测试发现 |
|---|---|---|
| 功能缺失 | 加权 IV 不支持 | 能——调用直接抛异常 |
| 边界 | 无数据操作层 | 不需要——是声明过的设计 |
| **契约缺陷** | 权重不随样本收缩 | **不能——调用抛异常，但原因指向用户数据** |

第三种最难，因为**报错存在、但方向错了**。判断依据是「参照实现怎么做」：Stata 和 statsmodels 都让权重跟着样本走，所以这不是设计差异，是缺陷。没有这个对照，很容易把契约缺陷说成"我们的设计不同"而放过。
