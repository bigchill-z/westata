# westata 可优化点清单 · ADHM (2020)

**来源**：Autor, Dorn, Hanson & Majlesi (2020) 复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复现环境**：StataNow 19.5 (MP) + `ivreg2`（`ssc install ivreg2 ranktest ftools`）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/adhm2020/adhm2020_probe_gaps.py`

> 配套报告：[REPORT.md](REPORT.md)
> 本文只讲 westata 本身**在自身 API 契约下**还可以改进什么。

---

## 0. 结论：2 个缺口，均已修复

本篇测的是**多维聚类**。

| 对照对象 | 修复前 | 现在 |
|---|---|---|
| **`ivreg2` + 双维聚类**（作者实际使用的） | westata 做不到 → **GAP 1** | **已修复**，SE = 9.236939 与 `ivreg2` 逐位相同 |
| **stock Stata `regress` + 双维** | Stata 能做，westata 不能 → **GAP 2** | **已修复** |
| **stock Stata `ivregress` + 双维** | 双方都拒绝 | 仍是 NOTE（这是 Stata 自己的限制，不是 westata 的） |
| `FEOLS` + 双维 | westata 能做 | 现在三个估计器一致 |

修复的判据不是"能跑"，而是**数字对上**：论文报的 9.236939，westata 现在给出同一个值。

---

## 1. 先纠正一个我最初写错的框架

初版本文档把问题写成「westata 不支持双维聚类，而 Stata 支持」——**这个框架不准确**，因为它把 Stata 当成一个整体。实测下来 Stata 自己也不统一：

| Stata 命令 | 双维聚类 | 实测证据 |
|---|---|---|
| `regress ..., vce(cluster a b)` | **支持** | 返回 `se = 5.82559` |
| `ivregress 2sls ..., cluster(a b)` | 拒绝 | `cluster(): too many variables specified` |
| `ivregress 2sls ..., vce(cluster a b)` | 拒绝 | **`multiway cluster VCE not allowed`** |
| `xtreg fe ..., cluster(a b)` | 拒绝 | 语法错误 |
| `areg ..., cluster(a b)` | 拒绝 | `cluster(): too many variables specified` |
| `ivreg2 ..., cluster(a b)`（用户自写） | **支持** | 五列全部复现作者日志 |

**所以正确的说法是**：

- 对 IV 而言，双维聚类在 stock Stata 里**也不存在**，只有用户自写的 `ivreg2` 有；
- 作者能报出双维标准误，是因为他们用了 `ivreg2`，不是内置命令。

这个参照系很重要——它决定了两条缺口的严重性判断完全不同。

---

## 2. GAP 1：IV 的多维聚类（P1）—— 已修复

### 2.1 现象（修复前）

```python
ws.IV2SLS.from_formula(...).fit(
    cov_type="cluster",
    cov_kwds={"clusters": frame[["czone", "congressionaldistrict"]]},
)
```

```
westata.errors.UnsupportedError: two-way clustering is not implemented for IV in P1;
pass a single cluster variable as cov_kwds={'clusters': ...}
```

### 2.2 后果是具体的（修复前）

作者报的标准误 **9.236939**，westata **无法产出**。这不是"少一个选项"，而是论文的核心推断量拿不到。

同一条数据显示，退而用单维会**系统性低估**：

| 列 | 双维（论文） | 单维 czone | 单维 c.dist | 低估幅度 |
|---|---|---|---|---|
| (1) | 9.2369 | 8.3244 | 8.4298 | **9.9%** / 8.7% |
| (2) | 21.2958 | 19.1806 | 19.8054 | **9.9%** / 7.0% |
| (3) | 21.1232 | 20.5386 | 20.4264 | 2.8% / 3.3% |
| (4) | 21.2289 | 20.7755 | 20.6498 | 2.1% / 2.7% |
| (5) | 21.0460 | 20.4907 | 20.7812 | 2.6% / 1.3% |

**两个单维都低于双维**，不是一高一低。所以用单维替代会**必然低估**，方向一致。

这对显著性有实质影响：第 (5) 列 t = 37.232/21.046 = **1.77**（10% 边缘）；换成单维 t = 37.232/20.491 = **1.82**。看起来更显著，但那个更小的标准误是错的。

### 2.3 当时的分级：P1 而非 P0

| 判据 | 情况 |
|---|---|
| 是否静默出错 | **否**，抛 `UnsupportedError` |
| 是否有替代 | 有单维，但**不等价且方向性偏低** |
| 与其他估计器的关系 | `FEOLS` 支持（见 §4） |

不误导，但确实产不出，所以是 **P1**（不是 P0——P0 留给"静默出错"）。

修复后这一条的判据仍然值得保留：它解释了为什么当时**不能**用单维糊弄过去。

---

## 3. GAP 2：OLS 的多维聚类（P2）—— 已修复

### 3.1 现象（修复前）

```python
ws.OLS.from_formula(...).fit(
    cov_type="cluster", cov_kwds={"groups": frame[["czone", "congressionaldistrict"]]}
)
```

```
ValueError: cov_kwds['groups'] must be one-dimensional; got 2 dimensions
```

而 stock Stata 的 `regress` **接受**这个：

```stata
. regress dhs2_tot_cont_2002_2010 d_imp_usch_pd, vce(cluster czone cdist)
      d_imp_usch_pd |   3.645443   5.825585
```

**这是唯一一处 westata 落后于"原版 Stata"（无需任何用户自写命令）的地方。**

### 3.2 当时的分级：P2

会抛异常、不误导，且有 `FEOLS` 作为替代路径。但它构成**内部不一致**（见 §4），所以值得修——P2 不是"可以不做"，是"不紧急但该做"。

---

## 4. 内部不一致：`FEOLS` 能做，`OLS`/`IV2SLS` 不能 —— 已消除

**修复前**，同一个 `cov_kwds=` 写法，三个估计器行为不同：

| 估计器 | 双维聚类 | 报错 |
|---|---|---|
| `FEOLS`（pyfixest 后端） | **支持** | — |
| `OLS` | 拒绝 | `ValueError` |
| `IV2SLS` | 拒绝 | `UnsupportedError` |

```python
# 这个能跑
ws.FEOLS.from_formula("y ~ x | 0", frame).fit(
    cov_type="cluster", cov_kwds={"groups": frame[["a","b"]]})   # OK

# 这个不行，调用形状完全一样
ws.OLS.from_formula("y ~ x", frame).fit(
    cov_type="cluster", cov_kwds={"groups": frame[["a","b"]]})   # ValueError
```

用户无从预判。**包的某个角落已经证明这件事能做**，所以这不是"未实现的统计方法"，而是一致性缺陷——同一份语义没有共享一条实现路径。

**现在三者一致**：`FEOLS`、`OLS`、`IV2SLS` 都接受双列 `groups`/`clusters`，并且共享 `westata.vce.cluster.cov_multiway_cluster` 这一条实现。

---

## 5. 修复路径已验证可行

CGM（Cameron-Gelbach-Miller）公式：

$$\hat V_{AB} = \hat V_A + \hat V_B - \hat V_{A\cap B}$$

**我用 westata 自己的三个一维拟合验证了这个恒等式**：

```
sqrt(VA² + VB² − VAB²) = 9.236939
                          A   = 8.3244
                          B   = 8.4297
                          A∩B = 7.4185
ivreg2 的双维值          = 9.236939     ← 逐位相同
```

**这意味着修复不需要新的统计机制**：三个一维聚类方差 westata 全都算得出来，缺的只是组合步骤。

### 5.1 实际实现

预测的四步都按计划落地，实现在 `westata/vce/cluster.py` 的 `cov_multiway_cluster`：

1. `cov_kwds` 的 `groups` / `clusters` 接受**多列**（DataFrame 或 `(n, k)` 数组）；
2. 每个非空维度子集算一次一维聚类肉，交集用各列标签组成的复合键；
3. 按包含-排除符号求和，**每个子集各自的** $G/(G-1) \cdot (N-1)/(N-K)$（Stata 的约定）；
4. `OLS` 与 `IV2SLS` 共用这一份实现——`IV2SLS._cluster_labels` 不再抛异常。

`e(N_clust)` 跟着报**最小的**每维组数，与 `df_t` 一致（Stata 也是这个约定）。

实现中踩到一个真实的坑：交集键用 `np.asarray(list(zip(...)))` 会得到二维数组，而
`encode_groups` 正确地拒绝二维输入；两列标签 dtype 不同时 `np.unique` 还会抛
`TypeError: '<' not supported between 'str' and 'float'`。两处都已按"显式构造一维对象
数组"修正——**能跑但键构造错了，会得到一个看起来正常的错数字**。

### 5.2 语义：多列即 multiway，不混入 nesting

- **多列** = 相加相减（multiway），与 Stata 的 `vce(cluster a b)` 一致；
- **嵌套**（hierarchical）不在这里表达——嵌套数据只需写最细的那一层，本来就是单列。

两者混为一谈会得到错误的方差，所以这里明确：**只有多列才走 CGM 分解**，单列行为完全不变。

---

## 6. 探针结果

```
-- against the paper --
  [OK  ] multiway cluster (IV)     SE = 9.236939，与 ivreg2 逐位相同
  [NOTE] vs stock Stata (IV)       stock ivregress 也拒绝（'multiway cluster VCE not allowed'）
  [OK  ] one-way cluster (czone)   coefficient 15.83783, SE 8.32436
  [OK  ] multiway cluster (OLS)    two-way OLS clustering works (N=3772)
-- controls --
  [OK  ] FEOLS two-way cluster     works (N=3772)
  [OK  ] CGM decomposition         sqrt(VA²+VB²−VAB²) = 9.236939，与 ivreg2 逐位相同
  [OK  ] public surface            142 public names

6 ok, 1 note, 0 gap, 0 error

**修复前**是 `4 ok, 1 note, 2 gap`。探针现在断言的是**数字**（SE 是否等于 9.236939），
不是"是否抛异常"——能跑但算错比拒绝更糟。
```

---

## 7. `ivreg2` 的安装

```bash
ssc install ftools, replace
ssc install ranktest, replace
ssc install ivreg2, replace
```

`adhm2020_install.py` 会以幂等方式完成这一步，并在缺失时让复现**明确失败**，而不是悄悄退回单维。

> **这里记录一次判断修正**：初版为了不依赖网络，故意没装 `ivreg2`，只用 `ivregress` 单维对比。结果是**论文的标准误从未被复现过**，而这个数字恰恰是整篇复现要检验的对象。装上之后五列全部精确命中。
>
> 教训：**为了避免依赖而放弃验证核心量，是错误取舍**。依赖可以脚本化安装，未被验证的核心结论不能补。

---

