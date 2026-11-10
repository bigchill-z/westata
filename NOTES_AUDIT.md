# `note:` 与说明性文字的遗漏审计

> 目标：全面检查 westata 的输出是否遗漏了 Stata 会提供的说明文字。
> 环境：StataNow 19.5 活体采集 + westata 工作树 `df607c9`。

## 方法

不先假设「哪里会漏」，而是**先找真实生产者**：对一批可能产生说明的命令各跑一次 Stata，
抓取各类说明文字。十个各产生零提示的用例什么也证明不了。

---

## 第 1 轮：`note:` 类（15 个命令）

只有 2 个产生 `note:`：

| 命令 | Stata 提示 | westata |
|---|---|---|
| `regress price mpg i.foreign##c.mpg` | `note: mpg omitted because of collinearity.` | 不适用（见下）|
| `estat ovtest, rhs`（含重复列）| 两条 `omitted` 提示 | **已实现** ✅ |

其余 13 个（`regress`/`logit`/`probit`/`poisson`/`nbreg`/`summarize`/`tabulate`/
`correlate`/`ivregress`/`vce(robust)`/`if` 子集）**均无提示**。

### `##` 那条属于语法边界，不需要照搬

```stata
regress price mpg i.foreign c.mpg#i.foreign     -> note=0, omitted=0
regress price mpg i.foreign##c.mpg              -> note=1, omitted=2
```

**系数完全相同**（`mpg -329.2551`、`F(3, 70)`、`N=74`）。差别是 `##` 额外展开主效应，
`mpg` 出现两次，第二次必与第一次共线。

westata **不支持 `##`**（`FormulaError`），用户写 `price ~ mpg * C(foreign)`，
而等价显式写法在 Stata 中**本来就不产生 note**。

---

## 第 2 轮：其他说明类别（13 个命令 × 7 类）

| 类别 | 产生者 | 例子 |
|---|---|---|
| **`Iteration k: Log likelihood = ...`** | **5 个** | `logit`/`probit`/`nbreg`/`zip`/`logit(grilic)` |
| `warning:` | 0 | — |
| `(output omitted)` | 0 | — |
| `(N real changes made)` | 0 | — |
| `(omitted)` 行 | 0 | 仅 `##` 下出现 |
| `* p < 0.05` 脚注 | 0 | — |
| `(estimates ... omitted)` | 0 | — |

---

## 第 3 轮：面板与时间序列族（12 个命令 × 10 类）

| 命令 | 说明文字 |
|---|---|
| `xtreg ..., re` / `fe` / `be` | **无** |
| `xtlogit ..., fe` | 拟合失败（munnell 的 `y` 不适合），未采到 |
| `var consumption temp price, lags(1/2)` | `Sample: 3 thru 30`（**表内已有**，非遗漏）|
| `vecrank consumption price, lags(2)` | `Sample: 3 thru 30`（同上）|
| `varsoc ...` | 无 |
| **`prais ..., corc`** | **`Iteration 0: rho = 0.0000` × 7** |
| `ivregress 2sls`（两种设定）| 无 |
| **`heckman`** | **`Iteration k: Log likelihood = ...` × 16，含 `(not concave)` × 12** |

---

## 结论：两类真实缺口，都在**迭代日志**

| 命令族 | Stata 打印 | westata | 影响面 |
|---|---|---|---|
| **ML 离散模型**（`logit`/`probit`/`nbreg`/`zip`/`zinb`）| `Iteration k: Log likelihood = ...` | **不打印** | 5 个命令 |
| **`prais`** | `Iteration k: rho = ...` | **不打印** | 2 个变体 |
| **`heckman`** | 同上，另加 `(not concave)` 标注 | **不打印** | 1 个命令 |

### 精确格式（`logit foreign mpg weight`，74 行数据）

```
(空行)
Iteration 0:  Log likelihood =  -45.03321        <- 每行 43 字符
Iteration 1:  Log likelihood = -29.238536
Iteration 2:  Log likelihood = -27.244139
Iteration 3:  Log likelihood = -27.175277
Iteration 4:  Log likelihood = -27.175156
Iteration 5:  Log likelihood = -27.175156
(空行)
Logistic regression                                     Number of obs =     74
```

### 为什么尚未实现

结果对象上的 `iterations` 只是**计数**（`8.0`），没有逐次的对数似然轨迹。
要打印这条日志，需要先在拟合过程中**收集每步的似然值**——这是对
`discrete.py` / `counts.py` 拟合循环的改动，不是格式调整。

---

## 已经排除的类别（无需再查）

- `note:` 共线提示：`ovtest` 已修；`regress` 的 `##` 属语法边界
- `warning:` / `Warning:`：7 类命令中零生产者
- `(output omitted)` / `(estimates ... omitted)`：零生产者
- `* p < 0.05` 星号脚注：零生产者（Stata 的 `regress` 不打印）
- 面板命令：`xtreg` 三种设定均无说明文字

## 下一步

1. **实现迭代日志**（主要工作）：确认 `statsmodels` 拟合能否返回每步似然；
   若不能，需在 westata 内自行迭代或用 `retvals`。
2. 采 `xtlogit, fe` 的说明文字（需换合适数据）。
3. 检查 `didregress`、`teffects` 等剩余命令族。
