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

---

## 第 4 轮：迭代日志的**算法根因**（本轮）

尝试实现 `prais` 的迭代日志时，从 `prais.ado` 读到了完整机制，
这解释了为什么 westata 的轨迹与 Stata 不同。

### `prais.ado:172-217` 的循环

```stata
scalar rho = 0                 // 起点是 0，不是 OLS 残差算出的值
scalar lastrho = 2
local it_ct = 0
while it_ct < iterate & abs(rho - lastrho) > toleran {
    local lastrho = rho
    qui predict double resid if touse, resid      // 用当前 rho 拟合后的残差
    _crcar1 rho rhotype : resid, k(k) rhotype     // 新 rho
    if it_ct == 0 { di "Iteration 0: rho = " %6.4f 0 }   // Iteration 0 恒打印 0
    prais_e : rho                                  // 用新 rho 重拟合
    local it_ct = it_ct + 1
    di "Iteration `it_ct': rho = " %6.4f rho
}
```

### 两处关键发现

**① rho 的定义**（`prais.ado:402-405`）：

```stata
qui _regress `varlist' l.`varlist' if touse, nocons
scalar rho = _b[L.`varlist']
```

即 **rho = 残差对自身一阶滞后做无截距回归的斜率**：

```
rho = Σ(e_t · e_{t-1}) / Σ(e_{t-1}²)
```

**不是** `Σ(e_t·e_{t-1}) / Σ(e_t²)`。用后者第一步得 `0.3298`，用前者得 **`0.4006`**——
**与 Stata 的 `Iteration 1` 精确一致**。

**② westata 的起点不同**：它用 OLS 残差算出的 rho 起步，Stata 从 **0** 起步。

### 为什么本轮没有落地

用正确的 rho 定义 + 从 0 起步后，第 1 步命中 `0.4006`，但**第 2 步起发散**
（`0.5154, 0.5989…` 对 Stata 的 `0.4008, 0.4009…`）。

说明 `prais_e`（重拟合那一步）的变换与我的 CO 变换不一致。
`prais_e` 不在 `prais.ado` 里，需另找其定义。

**已回退**：改动破坏了 4 个既有 prais 测试（它们断言输出以 banner 开头），
在轨迹未对齐前不应牺牲已正确的输出。46 个 prais 测试恢复通过。

### 结论

迭代日志的**格式**已经量清（43 字符/行、`Iteration 0` 恒为 0、前后空行），
**rho 公式**也已确定。缺的是 `prais_e` 的重拟合细节。

---

## 第 5 轮：`prais` 迭代日志的**算法已完整解出**（未落地）

前一轮卡在"第 2 步发散"。本轮读 `_crcar1.ado` 与 `prais_e.ado` 后，
**完整算法已确定，并用独立脚本 100% 复现**（含最终值）。

### 三个必要条件（缺一不可）

**① rho 的公式** —— `_crcar1.ado:37-42`（默认 `regress` 分支）：

```stata
gen double vals = resids * l.resids
sum vals;  scalar numer = r(sum)
replace vals = l.resids^2 if vals != .
sum vals;  scalar rho = numer / r(sum)
```

即 **`rho = Σ(eₜ·eₜ₋₁) / Σ(eₜ₋₁²)`**（westata 的 `_rho_from_residuals` 已是此式）。

**② corc 的变换丢弃首行** —— `prais_e.ado:28-33`：corc 模式**不做**首观测的
`sqrt(1-ρ²)` 替换，于是首行的滞后是缺失值，`_regress` **丢弃该行**。
（westata 的 `_transform_corc` 已是 `values[1:] - rho*values[:-1]`，正确。）

**③ 迭代的先后顺序** —— `prais.ado:172-217`，这是之前所有失败的根本原因：

```stata
rho = 0 ; lastrho = 2 ; it_ct = 0
while it_ct < iterate & abs(rho - lastrho) > toleran {
    lastrho = rho
    predict resid            // 【上一轮拟合】的残差（首轮是普通 OLS）
    rho = crcar1(resid)      // 由该残差算新 rho
    if it_ct == 0 { 打印 "Iteration 0:  rho = 0" }   // 字面量 0
    prais_e : rho            // 【用新 rho 拟合】
    it_ct++
    打印 "Iteration it_ct:  rho = <rho>"
}
```

**每一轮打印的 rho，既是本轮残差步的结果，也是下一轮拟合采用的 rho。**
把"拟合"和"算 rho"当成一个整体、并用 OLS 估计值重新播种，
就会得到 `0.3895, 0.3975, 0.3999…` 而不是 Stata 的 `0.4008, 0.4009…`。

### 独立脚本验证结果（100% 命中）

```
--- corc ---
  Iteration 0:  rho = 0.0000
  Iteration 1:  rho = 0.4006
  Iteration 2:  rho = 0.4008
  Iteration 3:  rho = 0.4009
  Iteration 4:  rho = 0.4009
  Iteration 5:  rho = 0.4009
  Iteration 6:  rho = 0.4009
  Iteration 7:  rho = 0.4009
  final rho = 0.4009256
```

与 Stata 的 `0.0000 0.4006 0.4008 0.4009 0.4009 0.4009 0.4009 0.4009` **逐值一致**，
最终值 `.4009256` 一致。

### 为什么仍未落地

在 `prais.py` 中改写循环时，我连续三次把「拟合用哪个 rho」与「记录哪个 rho」混在
`current_rho` 一个变量里，导致：迭代次数 11、19（应 7），或最终 rho 漂移
（`0.4009249` 对 `.4009256`）。

**已回退**，46 个 prais 测试恢复通过。

### 下次落地的正确写法

需要**两个独立变量**，不能复用 `current_rho`：

```python
rho   = 0.0          # 拟合所用（首轮 0）
last  = 2.0
it    = 0
trace = [(0, 0.0)]
while it < maxiter and abs(rho - last) > tolerance:
    last = rho
    resid = y - x @ coefficients      # 上一轮拟合的残差
    rho   = _rho_from_residuals(resid)
    coefficients = ols(transform(x, rho), transform(y, rho))
    it += 1
    trace.append((it, rho))
```

注意循环**以 `rho` 为拟合参数**，`coefficients` 随之更新；判据是 `rho` 与 `last` 之差。
