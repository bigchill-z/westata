# westata 可优化点清单 · Nunn & Qian (2011)

**来源**：Nunn & Qian (2011) 复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/nq2011/nq2011_probe_gaps.py`

> 配套报告：[REPORT.md](../papers/nq2011/REPORT.md)

---

## 0. 结论：0 个缺口（2 条均已修复）

westata 新增了 `Areg`（提交 `56304d4`），本文件随之更新。

| 条目 | 性质 | 状态 |
|---|---|---|
| `Areg` 的 `_cons` 与 Stata `areg` 不一致 | 缺口（P1）| **已修复** |
| `Areg` 的聚类标准误遵循 `reghdfe` 而非 `areg` | 缺口 | **已修复**（见 §1.5）|
| 系数（斜率） | 正确 | 6 列全部吻合 |

三条数值现在都与 Stata 的 `areg` **逐位一致**：

| 量 | westata `Areg` | Stata `areg` |
|---|---|---|
| 斜率标准误 | 0.0171579390 | 0.0171579390 |
| `_cons` | 12.64269254 | 12.64269254 |
| `_cons` 标准误 | 0.03383845 | 0.03383845 |

修法见 §1.5（聚类标准误）与 §4（常数项）。**这两条此前被记为"约定差异"和"未修复缺口"，
现在都是已修复项**——它们是同一类问题：`Areg` 继承 `reghdfe` 的方差路径，而 `areg` 的
自由度记账不同。

## 1. 最关键的一点：判定差异的归属需要对照

### 1.1 现象

同一条命令链、同一份数据、同一个系数，标准误不同：

| 路径 | 系数 | 标准误 |
|---|---|---|
| Stata `areg`（作者当年跑的） | 0.1213388 | **0.0171579** |
| Stata `xtreg, fe` | 0.1213388 | **0.0164182** |
| westata `FEOLS` | 0.1213388 | **0.0164182** |
| westata `PanelOLS` | 0.1213388 | **0.0164182** |
| **westata `Areg`（修复后）** | 0.1213388 | **0.0171579** ← 与 `areg` 一致 |

> 修复前 `Areg` 报的是 0.0164182——即 `xtreg`/`reghdfe` 的值。见 §1.5。

如果只对照"论文报告的 0.0171579"，会得出"westata 标准误错 4.5%"的结论。**但这个结论是错的。**

### 1.2 正确的判定方式

关键在于：**Stata 自己就不同意自己**。`areg` 与 `xtreg, fe` 给出同样的系数、不同的标准误。所以"对照论文"这个单一基准不足以判定谁对。

正确的做法是**同时对照两个 Stata 命令**：

```
westata vs xtreg, fe  →  se 相对差 1e-13      （机器精度）
westata vs areg       →  se 相对差 4.3e-02    （4.5%）
```

这就把问题从"westata 是否算错"变成"westata 实现的是哪一种约定"。

`FEOLS`/`PanelOLS` 的答案是 **`xtreg`/`reghdfe`**，这是它们的正确目标。
`Areg` 的答案必须是 **`areg`**——它修好之后报 0.0171579，与 Stata `areg` 逐位相同（§1.5）。

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

### 1.5 聚类标准误的修复：`Areg` 现在与 `areg` 一致

**修复前**：`Areg` 继承 `FEOLS` 的 `reghdfe` 方差路径，报 0.0164182。

**根因**：两个命令用同一个 CR1 公式 `G/(G-1) · (N-1)/(N-K)`，**只有 K 不同**：

| 命令 | K | 值 |
|---|---|---|
| `reghdfe` | `k_reported` = 23 | **0.0164182** |
| `areg` | `k_reported + (k_absorb − 1)` = 23 + 129 = 152 | **0.0171579** |

在本文数据上直接代入两个 K 验证：

```
K = 23  → 0.0164181944   （= reghdfe，westata 修复前的值）
K = 152 → 0.0171579390   （= Stata areg，逐位相同）
```

`k_absorb`（130）数的是**吸收的水平数**，其中常数项本身占一个，所以是 `k_absorb − 1`。
用 `K = 153` 会得到 0.0171640701——差 0.036%，看起来像舍入误差，实际是错的。

**注意这个修正只作用于聚类**：`areg.json` 是 `vce(ols)` 抓取，非聚类路径本来就对，
第一版修正误伤了它（两个元素偏 9%），测试立刻抓到。
---

## 2. 探针结果

```
-- Areg against Stata areg --
  [OK  ] Areg coefficient       coefficient 0.1213388（与作者日志一致）
  [GAP ] Areg _cons             _cons=12.63564782 (se 0.07604164)
                                Stata areg: 12.64269254 (se 0.03383845)
                                点估计差 5.57e-04，标准误差 125%
  [NOTE] Areg clustered SE      SE 0.0164182 = reghdfe 值，非 areg 的 0.0171579
  [OK  ] Areg identity          e(cmd)='areg', e(title)='Linear regression, absorbing indicators'
-- controls --
  [OK  ] public surface         143 public names
```

**3 ok、1 note、1 gap、0 error。**

| 类别 | 含义 |
|---|---|
| `OK` | 正确 |
| `NOTE` | 已归因的预期差异，**不是缺陷** |
| `GAP` | 该修的问题 |

`Areg` 新功能让 `e(cmd)`、`e(title)` 这些**身份**字段对上了，这部分是 `OK`。
但 `_cons` 的**数值**和聚类标准误的**口径**是另外两件事，分别记在 §4 与 §1。

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

## 4. `Areg` 的 `_cons` 与 Stata `areg` 不符 —— 已修复

### 4.1 现象与修复结果

同一份数据、同一个设定（11 个马铃薯交互项 + 11 个年份虚拟变量 + 130 个国家固定效应，
聚类于 `isocode`）：

| | `_cons` 点估计 | `_cons` 标准误 |
|---|---|---|
| Stata `areg` | **12.64269254** | **0.03383845** |
| westata `Areg`（修复前） | 12.63564782 | 0.07604164 |
| **westata `Areg`（修复后）** | **12.64269254** | **0.03383845** |

**6 列的全部斜率参数一直是对的**（与作者日志 1e-6 内吻合），问题只在常数项。

### 4.1.1 根因：等权平均 vs 按组容量加权

`Areg` 原先取「组效应**等权**平均」。正确的规则是**按组容量加权**：

| 规则 | 结果 |
|---|---|
| 等权平均 | 12.63564782 ← 旧实现 |
| **按组容量加权** | **12.64269254** ← Stata |

两者只在**各组容量不等**时才不同。`areg.json` 是 30 个组 × 6 期的平衡面板，
**等权与加权完全相同**，所以那个 fixture 无法区分两种规则——这正是缺陷通过审查的原因。

标准误同理：不能用同方差的 `σ²(X'X)⁻¹`，必须用**聚类稳健**三明治。
本文数据上同方差给出 0.02306936，聚类给出 0.03383845（Stata 的值）。

### 4.2 为什么这算缺口

`Areg` 存在的理由就是提供 Stata `areg` 的语义。提交 `56304d4` 已经做到：

- `e(cmd) = "areg"`
- `e(title) = "Linear regression, absorbing indicators"`
- `e(predict) = "areg_p"`
- 报告 `_cons`（`FEOLS` 不报告）

但**报告一个与 Stata 数值不符的 `_cons`，比不报告更糟**——使用者会以为它可信。
标准误相差 125% 尤其严重：这会让常数项的置信区间宽约 2.2 倍。

### 4.3 可能的成因

提交说明提到，常数项的正确算法是「拟合完整设计（斜率 + 每组一个虚拟变量），再取
`c = (1/G)·[0,0,1,…,1]`」，这样才能同时得到 Stata 的方差与斜率协方差。

本篇的设定比该 fixture 复杂：**除被吸收的 130 个国家效应外，还有 11 个显式的年份虚拟变量**。
推测常数项的构造未正确处理"既有 absorbed 效应、又有显式虚拟变量"的情形。**这是推测，
需要看 `hdfe.py` 的 `_cons` 代码确认。**

### 4.4 修法

`Areg._with_intercept` 现在：

1. 拟合完整设计（斜率 + 每组一个虚拟变量）；
2. 取 `c = (1/Σn)·[0,…,0, n₁,…,n_G]`——**按组容量加权**的组效应均值；
3. 用**聚类稳健**三明治（同一个 `c` 向量）得到方差与协方差。

验证：点估计与标准误都与 Stata 逐位相同。

### 4.5 分级：P1

| 判据 | 情况 |
|---|---|
| 是否静默出错 | **是**——输出看起来正常，`_cons` 有值、有标准误，不报错 |
| 影响面 | 中：常数项通常不是研究关注点，但它是 `predict` 与 `margins` 的输入 |
| 是否有替代 | 有（`FEOLS` 不报常数项，反而不会误导） |

主结果（斜率）不受影响，但会误导需要常数项的使用者。**P1。**
（**已修复**，见 §4.1/§4.4。）

### 4.6 顺带修复：`vce(hc2)` / `vce(hc3)`

`areg.sthlp:47-49` 把 `vce(hc2 [clustvar])` 列为它的 SE/Robust 选项之一。
`Areg` 原先继承 `FEOLS` 的限制而**拒绝** HC2/HC3：

```
ValueError: cov_type 'HC2' is not supported by FEOLS: HC2 needs the leverage of
the full dummy design, which PyFixest does not expose for fixed-effect models
```

**这个理由对 `FEOLS` 成立，对 `Areg` 不成立**——`Areg` 为算 `_cons` 本来就构造了完整
设计矩阵，杠杆值 `h_jj` 就在手里。`areg.sthlp:202` 写得很清楚：

> `vce(hc2)` instead uses `u_j^2/(1-h_jj)` as the observation's variance estimate

现在按此实现：HC2 用 `u²/(1−h)`，HC3 用 `u²/(1−h)²`，两者都在完整设计上重算协方差
（含 `_cons` 那一行/列）。实测四者严格递增（HC0 < HC1 < HC2 < HC3），符合文档预期。

`FEOLS` 仍然拒绝 HC2/HC3——它的理由没有被削弱，测试里也锁住了这一点。

### 4.7 顺带补齐：`noabsorb`（Stata 写作 `noabsorbtest`）

`areg.sthlp:157-161`：

> `noabsorbtest` prevents computation of the F test that all coefficients on indicators
> for absorbed variables are jointly zero. Computation of this test requires an iterative
> search. If the F test is of no interest, you may specify `noabsorbtest` and save
> computational time.

注意它是**省计算**的选项，不是显示开关——所以可观察的后果是
`e(F_absorb)` / `e(p_absorb)` **不存在**（而不是存在但为空）。

`Areg.from_formula(..., noabsorb=True)` 现已实现该行为。实测：

| | 默认 | `noabsorb=True` |
|---|---|---|
| `e(F_absorb)` | 有 | **无** |
| `e(p_absorb)` | 有 | **无** |
| `e(r2_c)` | 有 | 无（它只用于算这个 F） |
| `e(r2)` / `e(rmse)` / `e(df_a)` / 系数 | — | **完全相同** |

`FEOLS` 不获得该参数——`reghdfe` 没有这个选项，测试锁住了这一点。

### 4.8 补齐：`vce(bootstrap)` 与 `vce(jackknife)`

`areg.sthlp:47-49` 把这两者都列为 areg 的 vcetype，且给了一个 bootstrap 的用法示例。
原先 `Bootstrap`/`Jackknife` 只接受 `estimator='regress'`/`'mean'`。

现在两者都支持 `estimator='areg'`（需配 `absorb=`）。实测对照（StataNow 19.5，
210 行面板，30 组 × 7 期）：

| 方法 | westata | Stata | 备注 |
|---|---|---|---|
| **`vce(jackknife)`** | 0.08945882 | **0.08945882** | 确定性，逐位相同 |
| **`vce(bootstrap)`** | 0.08104865 | 0.08104864 | 随机性内（1e-8） |

**实现要点**：`areg` 的估计就是组内去均值后的 OLS，而 `vce(bootstrap)` 用的是
Stata 的 `wvar()` 频数权重形式，所以去均值必须用**加权组均值**，否则与 Stata 差一截。

**踩到一个会静默出错的坑**：设计矩阵里若保留 `Intercept` 列，它与吸收组的哑变量完全
共线，去均值后该列**恰好变成 0**，得到 `b=0`、`se=0` 的结果——不报错、不出 NaN，
只是一列废数据。现在在 `areg` 路径上显式去掉截距列，并加了断言：
**任何标准误为 0 即视为该列仍在**。

> 这与本文件 §4.2 记录的是同一类教训：**看起来正常的数字比报错更危险**。

### 4.8.1 修复：聚类维度与吸收维度的三种配对

这一条是 4.8 之后发现的，独立记录，因为它揭示了一个**「修好一个、弄坏另一个」**的陷阱。

`areg` 的聚类标准误取决于聚类变量与吸收维度的**关系**，三种配对走不同的路径
（StataNow 19.5，210 行面板，30 组 × 7 期，`absorb(a)`）：

| 聚类于 | Stata `x1` | Stata `_cons` | westata（修复后） |
|---|---|---|---|
| `a`（= 吸收维度） | 0.08671082 | 0.00901583 | 逐位相同 |
| `b`（与吸收维度交叉） | 0.10683824 | 0.10021268 | 逐位相同 |
| `a b`（多路） | 0.11086346 | 0.07460234 | 逐位相同 |

**踩到的坑**：我最初把 `areg` 的自由度修正**无条件**施加到所有聚类情形。结果修好了
`a` 与 `a b`，却把 `b` 弄坏了（0.1167 vs 0.1068，差 9%）——因为在 `b` 这种情形下，
westata 的协方差**本来就是对的**。

根因是 `e(rank)` 在这里**没有统一含义**：聚类变量与吸收维度交叉时 `pyfixest` 报 31
（含哑变量），嵌套或同维时只报 2（仅斜率）。判据是「rank 是否已达到
`斜率数 + 吸收水平数 − 1`」——达到就说明哑变量已计入，不需要再修正。

**多路还有一个独立的坑**：即使修正了 K，`a b` 仍差 0.45%。查证发现这不是 K 的问题——
Stata 的 `reghdfe` 在同一次拟合上给 **0.1023861**，而 westata 的 `FEOLS` 给的正是
0.1023861（逐位相同）。也就是说 `FEOLS` 完全正确，它忠实实现了 `reghdfe` 的口径；
但 `areg` 给的是 0.11086346，两者**因子结构不同**，不是某个 K 参数之差（反解隐含 K
得 0.13，不是整数）。

所以 `Areg` 在多路时必须**绕开 pyfixest 的协方差、用完整哑变量设计自算 CGM**。现已如此
实现：手工在完整设计上按 `G/(G−1)·(N−1)/(N−K)`（K = 斜率数 + 吸收水平数 = 31）做包含-排除，
得到 0.11086346，与 Stata 逐位相同。

> 教训与 §4.1.1 同类：**「能跑且数值合理」不等于正确**。0.1024 是个完全合理的标准误，
> 错的只是它属于 `reghdfe` 而不是 `areg`。测试因此同时断言 Areg 得 0.11086346 **且**
> 不等于 0.1023861。

`FEOLS` 未受影响——它仍然报 0.1023861（= Stata `reghdfe`），有测试锁定。

### 4.9 记录：`dfadjust` 尚未实现（附实测对照数据）

`areg` 的 `vce(hc2 [clustvar], dfadjust)` 会按 Imbens & Kolesár (2016) 计算**逐系数**的
调整自由度。它不是显示开关——它直接改变 p 值与置信区间，属于会影响数值输出的选项。

**当前状态：未实现。** 调研过程与结论一并记下，因为下面的实测数据比"没做"更有价值。

**公式来源**：`areg.sthlp` 只给出论文名，公式在 `[R] regress` 的 Methods 章节
（本地 `docs/reference/stata_r_manual.pdf` p.26-27，eq. 15 及聚类变体）。已完整提取：

```
G(l) 的第 g 列 = (I_N − H)_g' (I_Ng − H_gg)^{+1/2} X_g (X'X)^{-1} e_l
K_l = (Σλ_i)² / Σλ_i²        λ_i 是 G'G 的特征值
```

**实测对照数据（可用于将来的验证）**：StataNow 19.5，210 行面板（30 组 × 7 期）：

```stata
areg y x1, absorb(firm) vce(hc2, dfadjust)
* Note: p-values computed using adjusted degrees of freedom.
* r(table) 的 df 行：x1 = 61.711292,  _cons = 170.54664
* se(x1) = .08213816
```

顺带确认：`se(x1) = .08213816` 与 westata 的 HC2 **完全相同**——说明方差本身已经对了，
缺的只是自由度的调整。

**为什么没有实现**：尝试按公式实现后，在同一份数据上算出 19.79 / 1.0，而 Stata 是
61.71，**始终没有收敛**。诊断发现问题在于 `I − H_gg` **奇异**——每组只有 7 个观测，
而该组贡献的哑变量使其 `H_gg` 满秩（实测特征值含 0）。Stata 对这种奇异情形有专门处理，
而 PDF 的 Methods 章节没有展开那一步。

**结论：这需要比现有证据更多的信息，不能靠猜。** 已把未验证的实现从代码中移除，
不留半成品；上表的实测数值与公式一起留档，将来有 Stata 侧细节时可以接上。

> 判据：**能跑但数值可能与 Stata 不符**，比**明确不实现**更糟——这正是本文件 §4.2
> 对 `_cons` 的判断标准，同样的标准也应适用于这里。

### 4.10 明确不做的选项（评估后决定）

`areg.sthlp` 还剩三个选项没有实现。**这是决定，不是遗漏**，理由逐项记录如下，以免后来者
（包括未来的我）反复重新评估同一件事。

| 选项 | 出处 | 作用 | 为什么不做 |
|---|---|---|---|
| `tolerance(#)` | `areg.sthlp:244-246` | APM 迭代的收敛阈值，默认 `1e-8` | 见下方「更正」——**技术上可以接到 `pyfixest` 的 `fixef_tol`**，但对本篇无实际收益 |
| `iterate(#)` | `areg.sthlp:59` | APM 最大迭代次数，默认 50 | 同理，可接到 `fixef_maxiter` |
| `nolog` | `areg.sthlp:61` | 抑制 APM 迭代日志的打印 | westata 的 `pyfixest` 后端本来不打印该日志，接受这个参数等于**假装关掉了一个从未打开的东西** |

三个都属 `Optimization` 组，且被 `areg.sthlp` 标注为 **†**：

> † Ignored if only one absorbed variable is specified.

也就是说，在本篇这种单吸收变量（`absorb(isocode)`）的场景下，Stata 自己都注明这三个选项
**被忽略**。为它们做接口，收益为零。

**⚠️ 更正（写这一节时先记错了）**：初稿写的是「`pyfixest` 的内部阈值不可从外部设定」。
实测检查 `pyfixest.estimation.feols` 的签名，它**确实有** `fixef_tol` 与 `fixef_maxiter`
（还有 `collin_tol`）。所以前两条**不是做不到，是做得到但不值得**——两者的理由强度不同，
不该混为一谈。

**注意区分**：不做 ≠ 无影响。真要改变收敛行为，正确的位置是 `FEOLS`/`pyfixest` 那一层，
而不是在 `Areg` 上开三个转手就丢的空参数。若将来确实需要，应作为 **FEOLS 的能力**统一实现，
`Areg` 再继承。

#### 两项报告选项：`clustertable` 与 `level(` —— 均已实现

评估过程中还确认了两项。它们与上面三项性质不同——**有实际输出效果**，因此当时记为
**待办**而非「不做」，现已实现：

| 选项 | 出处 | 作用 | 状态 |
|---|---|---|---|
| `clustertable` | `areg.sthlp:225-228` | 多路聚类下打印「聚类组合 × 组数」表 | **已实现** |
| `level(#)` | `areg.sthlp:53` | 置信水平，默认 95 | **已实现**（`OLS`/`IV2SLS`/`Areg` 等） |

**`level(#)` 的实测对照**（StataNow 19.5，210 行面板）：

```stata
areg y x1, absorb(firm) level(90)
* _cons 的 [90% conf. interval] = -0.17476631   0.04851883
```

westata 输出 **-0.17476631 / 0.04851883**，逐位相同。摘要表头也随之变为
`[90% conf. interval]`——它本来就是按 `result.level` 生成的，只是此前没有模型能设置它。

**`level` 的坑**：Stata 的 `level()` 收的是**百分数**（`level(90)`）。若把 `level(0.90)`
当成百分数解读，会得到 0.9% 的区间——症状是区间异常窄，看起来像另一个 bug。
实现同时接受两种写法并把区间归一为分数，`level(90)` 与 `level(0.90)` 结果完全相同
（有测试锁定）。

**`clustertable` 的关键点：组合数是实测的，不是相乘的。** 表中 `a#b` 一行是
「不同的 `(a,b)` 组合数」，不是 `30 × 7`。本例是完整交叉所以恰好等于 210，但两个维度
存在缺口时会明显更小——而这正是该表存在的意义。有测试专门用稀疏交叉验证这一点。

列宽按真实捕获定为 31 字符；初版按数据长度计算得到 36，内容全对但两张表对不齐。

**⚠️ 更正二（同样先记错了）**：初稿写的是「`clustertable` 的表头措辞与列宽没有任何 Stata
捕获可对照，实现只能靠猜排版」。**这不成立**——Stata 就在本机，现场跑一次就有了：

```stata
areg y x1, absorb(a) vce(cluster a b) clustertable

Linear regression, absorbing indicators            Number of obs     =     210
Absorbed variable: a                               No. of categories =      30
Clusters per comb.:                                Cluster comb.     =       3
  min =   7                                        F(1, 6)           =    0.02
  avg =  82                                        Prob > F          =    0.8993
  max = 210                                        R-squared         =    0.0945
                                                   Adj R-squared     =    -0.0573
                                                   Root MSE          =    0.9713
-------------------------------
                    |  Clusters
Cluster combination | per comb.
-------------------------------
                  a |        30
                  b |         7
                a#b |       210
-------------------------------

                                  (Std. err. adjusted for multiway clustering)
...
Cluster combinations formed by a and b.
```

所以「没有捕获」是**我没有去取**，不是取不到。保留这段是因为它标出了我自己的一个错误
推理步骤：把「尚未调查」说成「无法调查」，会让一个可做的功能看起来不可做。

> 这两项记为**待办**而非**不做**。上表输出已可直接作为实现与验证的对照基准。

---

## 5. 复核方式

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
