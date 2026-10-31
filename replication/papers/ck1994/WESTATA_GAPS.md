# westata 可优化点清单 · 第二篇（Card & Krueger 1994）

**来源**：Card & Krueger (1994) 复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复核方式**：`../../.venv/Scripts/python.exe replication/papers/ck1994/probe_gaps.py`

> 配套报告：[REPORT.md](REPORT.md) —— 讲这篇论文的复现结果。
> 第一篇的清单见 [../adh2013/WESTATA_GAPS.md](../adh2013/WESTATA_GAPS.md)。

---

## 0. 这份清单为什么和第一篇不一样

第一篇（ADH 的 shift-share IV）的清单列了 **7 条缺口**，从"加权 IV 完全不支持"到"`margins` 不接受序列"。写完之后，上游把 **7 条全部修复并提交了**：

| 原条目 | 修复提交 |
|---|---|
| 加权 IV 不支持 | `01c3ce5` |
| `cov_kwds` 聚类键名三套 | `72a2d12` |
| `FEOLS` 没有 `predict` | `fa5787e` |
| `subset=` 覆盖不全 | `7d11f1e` |
| `margins(at=)` 只接受标量 | `f84a404` |
| `predict(exog=)` 形状问题 | `f84a404` |
| `etable` 无写文件 | `f84a404` |

所以现在跑第一篇的探针，结果是 **0 gap / 9 条对照全部通过**。那份清单已经完成了它的使命。

**那么第二篇该写什么？** 简单复制旧格式没有意义——我实测过，Card & Krueger 需要的估计量（无权重 OLS，357 行、最多 10 个参数）在 westata 上**一次就过，没有任何摩擦**：

```
[OK] plain OLS      coefficient 2.3258, matching the paper's 2.33
[OK] Stata labels   stata_param_names = ['nj', '_cons']
[OK] weighted OLS   weighted OLS runs (N=357)
```

**这次复现暴露的问题不在估计，而在数据。** 因此本文的重点不是"westata 做不到 X"，而是**一篇真实复现中有多大比例的工作 westata 根本没参与**——以及这个空白里最容易出错的地方在哪里。

---

## 1. 判定标准：区分"缺口"与"边界"

第一篇的判据是三条（能力缺失 / 内部不一致 / 契约违约）。这次要加第四条，因为两者的**行动含义完全不同**：

| 类别 | 含义 | 该怎么做 |
|---|---|---|
| **GAP**（缺口） | westata 声称提供、或 statsmodels 风格的调用本应可用，但不能用 | **该修** |
| **BOUNDARY**（边界） | westata 明确声明不做这件事 | **该规划**，不是该修 |

区分二者很重要。把边界当缺口，会催生"让 westata 重写 pandas"这种不合理的需求；把缺口当边界，则会掩盖真正该修的问题。

README 第 259 行明确写着：

> 本库不做数据管理：没有 `use`、`merge`、`reshape`、`egen` 的对应物，输入是 pandas DataFrame。

这是**声明过的边界**，所以下面第 2 节的内容**不是缺口**。但它是这篇复现里最花时间的部分，值得量化记录。

**本次结果：0 个 GAP，1 个 BOUNDARY。**

---

## 2. 边界：数据层完全在库外

### 2.1 量化的结果

这次复现对原始数据做了 **8 个变换**，全部在 pandas 里完成，westata 的 142 个公开接口**一个都没参与**：

| 变换 | Stata 用户会怎么写 | 本次实现 |
|---|---|---|
| 固定宽度解析（46 个字段） | `infile ... using public.dat` | 手写列切片 |
| `.` → 缺失值 | 自动 | `pd.to_numeric(errors="coerce")` |
| 派生 FTE | `gen fte = empft + 0.5*emppt + nmgrs` | 列算术 |
| **永久关闭店 → 就业记 0** | `replace fte2 = 0 if status2==3` | `DataFrame.loc` 掩码 |
| **临时关闭店 → 就业记缺失** | `replace fte2 = . if inlist(status2,2,4,5)` | `DataFrame.loc` 掩码 |
| 派生 GAP | `gen gap = max(5.05-wage_st,0)/wage_st` | `np.where` |
| 样本限制 | `keep if ...` | 布尔掩码 |
| 构造品牌虚拟变量 | `tab chain, gen(chain_)` | `(col == k).astype(float)` |

探针确认这 14 个 Stata 数据动词一个都不存在：

```
[BOUNDARY] data verbs  none of 14 Stata data verbs exist;
                       all 8 transformations in this replication were done in pandas
```

### 2.2 这个边界为什么值得单独写

不是因为"westata 少了功能"，而是因为**这类工作的错误不会报错**。

具体例子（本次实际踩到的）：

> 我把"永久关闭店就业记 0"这条规则**简化**成了"要求两期工资都有值"。
> 结果：样本从 357 家变成 351 家，新泽西系数从 **2.33 变成 2.28**。
> **整个过程没有任何警告。** 回归正常收敛、标准误正常算出、`summary()` 正常打印。

这是数据层特有的风险：估计层的错误通常有信号（秩亏报错、不收敛、`NaN` 传染），而数据层的错误只有一个**看起来合理的数字**。

### 2.3 那么该建议什么？

**不建议**给 westata 加 `egen`/`merge`/`reshape`。理由：

1. pandas 已经做得更好，重写是净损失；
2. README 的定位是对的——一个库做一件事；
3. 加了反而制造两套数据操作 API，增加维护面。

**值得考虑的是更轻的东西**：这个复现真正需要的不是"数据操作功能"，而是**"数据构造的可审计性"**。具体说：

| 建议 | 成本 | 收益 |
|---|---|---|
| 一个 `westata.validate_sample()` 之类的辅助：把"预期 N / 预期均值 / 预期标准差"与数据比对，不符就报错 | 低 | 直接防住 §2.2 那类静默错误。本次复现就是靠论文 Table 4 注释里的 **−0.237 / 8.825** 才发现样本对的 |
| 文档里加一节"如何验证你的样本构造正确" | 低 | 这是本库"证据等级"精神在数据层的自然延伸 |

第二条尤其契合仓库的既有理念：README 说"**每条命令都带证据等级，做不到时会如实说明**"。同样的态度用在数据上，就是"**先证明样本是对的，再谈系数**"。

---

## 3. 估计层：本次没有发现缺口

三个探针全部通过，记录在此以说明"没有缺口"是**测出来的**，不是没测：

```
[OK] plain OLS      coefficient 2.3258, matching the paper's 2.33
[OK] Stata labels   stata_param_names = ['nj', '_cons']
[OK] weighted OLS   weighted OLS runs (N=357)
[OK] public surface 142 public names
```

### 3.1 一处我自己的误判，记下来避免重犯

写 `run_compare.py` 时，我以为 westata 只暴露 patsy 的 `Intercept`，于是手写了一个 `{"Intercept": "_cons"}` 的 rename 映射。

**这是多余的**——库早就提供了正确的访问器：

```python
result.param_names         # ['Intercept', 'nj']      patsy 顺序
result.stata_param_names   # ['nj', '_cons']           Stata 顺序与命名
result.stata_params        # Series, index 为 Stata 名称
```

我在探针里加了 `_stata_labels_available` 来固定这个能力，并在注释里写明"这是我自己绕的远路，不是库的缺失"。

这个误判值得写下来，因为它揭示了一个真实的使用风险：**API 存在不等于使用者找得到**。`param_names` 和 `stata_param_names` 只差几个字符，而 `params`（patsy 标签）和 `stata_params`（Stata 标签）更容易混。如果能在 `summary()` 或文档里更醒目地提示"要 Stata 口径请用 `stata_params`"，这类误判会少很多。

---

## 4. 复核方式与防回归

探针 `probe_gaps.py` 有**四种**结论：

| 结论 | 含义 |
|---|---|
| `OK` | 调用正常 |
| `BOUNDARY` | westata 按设计不提供——记录，不计为缺口 |
| `GAP` | 应该能用却不能用——**该修** |
| `ERROR` | 探针本身异常 |

```bash
.venv/Scripts/python.exe replication/papers/ck1994/probe_gaps.py
# 4 ok, 1 boundary, 0 gap, 0 error
```

**文档中有两个数字是被探针守住的**，改了会失效：

- 若 westata 哪天真的加了数据动词，`_data_surface` 会从 `BOUNDARY` 变成 `BOUNDARY` 但信息改变（提示 README 与本文件都需更新）；
- 若公开接口数量掉到 100 以下，`_control_estimator_surface` 会报 `GAP`。

**测试守护**：`replication/tests/test_ck1994_replication.py` 中的 `test_the_gap_probe_still_reports_what_the_gaps_doc_claims` 会跑本文件的全部探针，并且：

- 任何 `GAP` 或 `ERROR` → 测试失败（§3 声称没有缺口）；
- 三个估计层探针不再 `OK` → 测试失败（那是真实回归）；
- 数据层探针不再报 `BOUNDARY` → 测试失败，并提示 §2 需要重写（westata 若真加了数据动词，本文件的核心论断就过时了）。

写这份文档时我一度打算把"没有守护测试"记为已知缺口——随后意识到**补齐它只要十几行**，记下来反而成了偷懒的借口。所以补上了。

---

## 5. 两篇合起来的结论

| | ADH (2013) | CK (1994) |
|---|---|---|
| 估计量 | 加权 2SLS + 聚类 | 无权重 OLS |
| 参数数 | 178 | 26 |
| westata 与 Stata 最大相对差 | 1.6e-11 | **2.1e-13** |
| 主要难点 | 加权 IV 的实现 | **数据构造** |
| 本次暴露的 westata 问题 | 7 条缺口（**已全部修复**） | 0 条缺口，1 条边界 |

**结论**：

> westata 的**估计层**已经足够好——两篇论文、187 个成对参数，最大差异都在浮点求和顺序量级。
> **限制它的是估计之外的部分**：数据整理（pandas 的活，但缺乏可审计性支持）和绘图（完全空白）。

第一篇的 7 条缺口被上游逐一修完，说明这份清单的机制是有效的：**写清楚 → 有人修 → 探针自动把"缺口"改写成"回归测试"**。第二篇没有产生新缺口，本身就是那个机制在起作用的证据。
