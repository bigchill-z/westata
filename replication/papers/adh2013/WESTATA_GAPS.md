# westata 可优化点清单

**来源**：Autor–Dorn–Hanson (2013) 复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/adh2013/probe_gaps.py`

> 配套报告：[REPORT.md](REPORT.md) —— 讲这篇论文的复现结果。
> 本文只讲 westata 本身**在自身 API 契约下**还可以改进什么。
>
> **状态：7 条全部已修复，清单归零。** 初版是在 `westata` 未作任何改动时测得的；修复过程中本仓库的 `src/westata` 被修改，所以本文现在记录的是「曾经缺什么、怎么修的」，而 §4 的探针反过来充当**回归测试**——任何一条退化都会让 `replication/tests` 失败。按当前 HEAD 实测：`0 gap(s) present, 0 stale report entries`。

## 0. 先说清楚：本文不要求 westata 模仿 Stata

westata 的 API 是 **statsmodels 风格，这是刻意的设计选择**（README 开篇即写"API 沿用 statsmodels 风格"），不是缺陷。

所以本文**不以"像不像 Stata"为判据**。初稿曾按这个错误框架列过 5 条"缺口"（`i.` 语法糖、`lincom` 的 `_b[]`、`predict` 别名、全局 `set seed`、`log using`），它们全部被删除——详见 §5。

本文的判据只有三条（下面括号里的例子都是**初版实测时**的）；

1. **能力缺失**——某类模型根本跑不了（如加权 IV），与语法无关；
2. **同一 API 内部不一致**——同为 statsmodels 风格，A 估计器有这个参数、B 没有；或同一语义在不同估计器上要用不同 kwds 键名；
3. **契约违约**——签名接受了参数却不生效，或报错信息误导。

已排除题目点明的数据操作层（`egen`/`merge`/`reshape`/`collapse` 全缺）与绘图层（`graph`/`twoway` 全缺）——这两块按 README 的定位本来就该交给 pandas 和 matplotlib。

## 1. P0：能力缺失（判据 1）—— 目前没有

本清单初版把「加权 IV 不支持」和「`FEOLS` 没有 `predict`」列为两条能力缺失。**两条都已由上游实现并提交**，见 §6。按当前 HEAD 实测，判据 1（某类模型根本跑不了）**已经没有剩余条目**。

这不是清单失去意义，而是它按预期走完了一轮：写清单 → 上游修复 → 探针把「缺口」改写成「回归测试」。

## 2. P1：同一 API 的内部不一致（判据 2）—— 目前没有

初版记录了两条：`cov_kwds` 聚类键名三套并存、`subset=` 只在部分估计器上存在。**两条都已修复**（§6.1、§6.4）。此后 `margins(at=)` 只接受标量也已修复（§6.5）。按当前 HEAD 实测，判据 2 **已经没有剩余条目**。

## 3. P2：契约违约与工程化（判据 3）—— 目前没有

初版记录了两条：`predict(exog=)` 无形状校验、`etable` 无写文件入口。**两条都已修复**（§6.6、§6.7）。

`etable` 那一条值得单独说：初稿的建议是「这是有意的分层设计，**保留即可**」——即认为它**不该**改。后来还是实现了，理由是：Stata 用户习惯 `esttab`/`outreg2` 直接写文件，"返回数据、自己落盘"在 Python 里成立，但作为**跨语言迁移的便利性**确实缺一环。

实现方式是给 `etable` 加 `file=`，**不动** `to_latex`/`to_markdown`，且 `etable` 仍然返回 `DataFrame`——两种用法都保留。

这是一个「保留即可」判断失误的例子：分层设计本身没错，但**"没错"不等于"够用"**。


### 小结

**7 条全部已修复。** 前 4 条在清单存续早期由上游提交，后 3 条为最后一轮补齐（清单归零）。

| 原优先级 | 事项 | 状态 | 提交 |
|---|---|---|---|
| P0-1 | 加权 IV 不支持 | ✅ 已修复（§6.2）| `01c3ce5` |
| P0-2 | `FEOLS` 无 `predict` | ✅ 已修复（§6.3）| `fa5787e` |
| P1-3 | `cov_kwds` 聚类键名三套 | ✅ 已修复（§6.1）| `72a2d12` |
| P1-4 | `subset=` 部分估计器才有 | ✅ 已修复（§6.4）| `7d11f1e` |
| P1-5 | `margins(at=)` 只接受标量 | ✅ 已修复（§6.5）| `f84a404` |
| P2-6 | `predict(exog=)` 缺校验与重建入口 | ✅ 已修复（§6.6）| `f84a404` |
| P2-7 | `etable` 无写文件 | ✅ 已修复（§6.7）| `f84a404` |

**这份清单现在的用途是回归测试套件**：`probe_gaps.py` 的 9 条对照探针必须全部保持 `FIXED`，任何一条退化都会让 `replication/tests` 失败。所以「曾经修好」是被持续验证的，不是一次性声明。

---

## 4. 复核方式与防回归

探针 `probe_gaps.py` 有**三种**结论，不是两种：

| 结论 | 含义 |
|---|---|
| `GAP` | 缺口确实存在 |
| `FIXED` | westata 已实现，本文该条已过时 |
| `ERROR` | 探针本身异常 |

另有 **3 条对照探针**（两路聚类在 HDFE 上可用、bootstrap 认 `reps`/`seed`、`subset=` 在 OLS 上生效）。对照若失败，说明探针测错了东西，上面所有结论都不可信。

测试 `test_the_capability_gap_list_is_still_accurate` 会在任何一条变 `FIXED` 时失败，强制本文更新——**一条过时的"westata 做不到 X"与谎称"能做到"同样有害**。§6 记录的正是这条机制第一次触发的实例。

```bash
.venv/Scripts/python.exe replication/papers/adh2013/probe_gaps.py
.venv/Scripts/python.exe -m pytest replication/tests -q
```

## 5. 已删除的条目（框架修正记录）

初稿按"与 Stata 的相似度"列出过 5 条，现已全部删除。保留清单是为了让这次框架修正是可审计的，而不只是悄悄改了一版。

| 原条目 | 删除理由 |
|---|---|
| `i.` / `ib#.` / `c.` / `##` / `L.` / `D.` 语法糖 | westata 明确用 patsy 语法。`C(x, base=)` 能表达 `i.` 的全部语义；`L.`/`D.` 属于时间序列预处理，不属于估计器 API |
| `lincom` 认 `_b[x]` | `_b[]` 是 Stata 的显示语法。westata 用参数名（`lincom("x1 + x2")`）已是最自然的 statsmodels 写法 |
| `predict` 认 `resid`/`se` 别名 | 为 Stata 习惯服务，对 statsmodels 用户无益 |
| 全局 `set seed` | RNG 状态属于估计调用而非库的全局状态；`Bootstrap(..., seed=)` 的显式传参更符合 Python 惯例 |
| `log using` 会话记录 | 用户用 `logging` / `contextlib.redirect_stdout` 即可；库不该自建日志体系 |

另外修正过一处**事实性误判**：初稿说"`Bootstrap` 不收 `reps`"，实为看错了签名——我读的是内部基类 `_ResampleBase.__init__`，而公开入口 `Bootstrap.from_formula` 接收 `reps` 与 `seed`，实测同 seed 下 `bse` 逐位可复现。该条已从"缺口"降级并写入对照探针。

## 6. 状态变化的条目

这一节记录**曾经成立、情况已变**的两条。保留它们是因为一条过时的"westata 做不到 X"与谎称"能做到"同样有害，而"最近改了什么、改到什么程度"本身也是有用的信息。

两条的性质不同，必须分开说：

| 条目 | 状态 | 依据 |
|---|---|---|
| 聚类键名三套 | **已修复且已提交**（`72a2d12`） | `tests/test_cluster_key_is_uniform.py` 21 项；探针对照项通过 |
| 加权 IV 不支持 | **已修复且已提交**（`01c3ce5`） | `tests/test_weighted_iv.py` 25 项；探针对照项断言复现 Stata 的 b 与 se |
| `FEOLS` 没有 `predict` | **已修复且已提交**（`fa5787e`） | `tests/test_feols_predict.py` 17 项；探针对照项断言 `xb + residuals == y` |

三条修复的经过值得一提：本清单初版把它们列为 P0 能力缺失，上游随后逐一实现。**清单的作用不只是记录问题，它促成了修复**；而修复之后，同一套探针自动把「缺口」改写成「回归测试」，不需要有人记得回来删条目——`test_the_capability_gap_list_is_still_accurate` 会在任何一条重新变回 `FIXED` 时失败。

### 6.1 `cov_kwds` 聚类键名三套并存 —— 已修复并提交（`72a2d12`）

**原状**（本清单初版 §2 第 3 条）：同样是 `fit(cov_type="cluster", cov_kwds=...)`，四个估计器接受的键名不同（`OLS` 只认 `groups`，`IV2SLS`/`PanelOLS` 认 `groups`+`clusters`，`FEOLS` 认 `groups`+`cluster`）。在 `OLS` 上传 `clusters` 得到的报错是 `cov_kwds['groups'] must be one-dimensional; got 0 dimensions`——读起来像数据有问题，实际是键名用错了。

**现状**：`groups` / `clusters` / `cluster` **三种键名在四个估计器上全部可用**。

```python
for key in ("groups", "clusters", "cluster"):
    for est in (OLS, IV2SLS, FEOLS, PanelOLS):
        ...  # 全部通过
```

**修法**：新增 `westata.vce.dispatch.cluster_labels_from_kwds()` 作为唯一读取入口，替换掉散落在 `estimators/linear.py`、`estimators/discrete.py`、`estimators/hdfe.py`、`estimators/iv.py`、`estimators/panel.py`、`vce/dispatch.py` 六处的各自 `dict.get(...)`。分散读取正是这个不一致能存活下来的原因——每次只修一处，其余照旧。

同时把错误信息改成列出可接受的键名，而不是只报"缺 `groups`"。

**修复过程中被自己的测试抓到一次**：`panel.py` 里我用 `for key in (...)` 循环，覆盖了外层同名变量 `key`（它保存归一化后的 `cov_type`），导致**所有 robust 拟合走进错误分支**。`test_robust_is_statas_clustered_vce_not_hc1` 立即失败。循环变量已改名为 `_label_key`。这条值得记下：修一个一致性问题时，最容易引入的是另一个一致性问题。

**防回归**：`probe_gaps.py` 的对应探针现在报 `FIXED`，`test_the_capability_gap_list_is_still_accurate` 因此失败并强制本文更新——这正是该测试设计的目的，也是它第一次真正触发。

### 6.2 加权 IV —— 已修复并提交（`01c3ce5`）

**原状**：`IV2SLS`/`IVLIML`/`IVGMM` 全部拒绝 `weights=`，抛 `UnsupportedError`。这是本次复现最痛的缺口——论文 166 条估计命令里 153 条是加权 IV。

**一个当时的中间状态值得记录**：修复先出现在工作区（尚未提交），我据此在报告中标注为「本机可用、改动尚未提交」，因为按 `HEAD` 克隆的读者拿到的仍是不支持的版本。这种标注是对的——**但在提交之后它也会过期**，本文后来确实需要回头修正（`fa5787e` 提交后，§6.3 的「未提交」字样已改为「已提交」）。文档里任何描述仓库状态的话，都会随提交而失效。

**现状**：已提交为 `01c3ce5`。实测复现论文 Table 3 列 (6)：

```
native aweighted IV2SLS:
  b  = -0.596360066134     (Stata: -0.596360066134,  相对差 4.8e-14)
  se =  0.0987738830       (Stata:  0.0987738830)
  N  = 1444
```

**保留的防线**：探针的 `weighted IV` 对照项不只断言"不报错"，而是断言**复现出 Stata 的 b 与 se**。加权实现出错时的典型症状是**系数对、标准误错**（掩盖在 `e(V)` 里），只断言"能跑"会放过这种缺陷。

### 6.3 `FEOLS` 的 `predict` —— 已修复并提交（`fa5787e`）

**原状**：`OLS`/`IV2SLS`/`PanelOLS`/`Logit`/`Poisson` 都能 predict，唯独 `reghdfe` 抛 `no prediction types registered`。

**现状**：`kind="xb"`（含吸收效应）、`d`、`residuals`、`dresiduals` 均可用，实现在 `fa5787e` 提交（`estimators/hdfe.py`）。多路吸收（如 `| firm + year`）明确拒绝并说明原因——多路效应只在相差一个归一化常数下可识别。

**保留的防线**：对照项断言 `xb + residuals == y`。固定效应模型很容易返回一个**遗漏吸收项平移**的 `xb`——数值看起来合理，实际是错的。用这条恒等式检查，比断言"不抛异常"能抓住真问题。

### 6.4 `subset=` 覆盖不全 —— 已修复并提交（`7d11f1e`）

**原状**：`OLS`/`Logit`/`Poisson` 接受 `subset=`，而 `IV2SLS`/`IVLIML`/`IVGMM`/`FEOLS`/`PanelOLS` 不接受。"限制估计样本"是跨模型通用操作，却要按估计器换写法。

**现状**：全部 `from_formula` 都接受 `subset=`，布尔掩码与索引标签两种形式均可。

**改动范围**（比清单里写的更大）：除了清单点名的 5 个，实测发现另有 4 个面板估计器也缺 —— `XtGls`、`XtPcse`、`XtIvreg`、`XtUnitroot`，以及 `Didregress`/`Xtdidregress`。清单只列了当时复现用到的那些。

**顺带修正了探针本身的一个缺陷**：它用 `inspect.signature` 判断 `subset=` 是否存在，而 `IV2SLS.from_formula` 走 `**kwargs` 再从里面 `pop("subset")`，签名里永远看不到该参数。于是探针把**已经可用**的 IV 族报成缺失——如果照它的报告去"修"，会去改一条本来正常的路径。

改为**实际调用** `from_formula(..., subset=mask)` 后再判定：

```python
try:
    _probe_subset_call(cls)
except TypeError as error:
    (lacks if "subset" in str(error) else has).append(name)
```

只有「签名拒绝」才算缺失；其他异常说明参数已被接受。这条区分正是探针该做的事——它测的是**能力**，不是**声明**。

**保留的防线**：测试不只断言"能调用"，还断言 `subset=` 的结果与手工 `df[mask].reset_index(drop=True)` 后估计**逐位相同**（系数 `rtol=1e-10`、标准误 `rtol=1e-8`）。一个"接受参数但忽略它"的实现能通过前者、通不过后者——而那种实现的输出看起来完全正常，只是用了全样本。

### 6.5 `margins(at=)` 只接受标量 —— 已修复并提交（`f84a404`）

**原状**：`margins(at={"x1": [0, 1, 2]})` 抛 `TypeError: float() argument must be a string or a real number, not 'list'`。多取值分析只能循环调用。

**现状**：`at=` 接受序列，语义严格按 `margins.sthlp:995-1005`：

| 写法 | 结果 |
|---|---|
| `at(x1=5)` | 1 个场景 |
| `at(x1=(5 10))` | 2 个场景 |
| `at(x1=(5 10) x2=(0 1))` | **4 个场景（笛卡尔积）** |

**为什么是笛卡尔积而不是 zip**：Stata 帮助文件第 3 条规则明写 "all possible combinations of settings are applied in turn"。zip 会在两个列表长度不同时**静默丢弃**场景——而丢弃场景在输出里看不出来，表格只是少了几行，调用者不会发现。

**实现**：`at` 展开成场景列表后逐个递归调用 `margins`，因为导数、弹性缩放和标准误**都是在对齐后的设计矩阵上算的**，跨场景无法复用。每个结果的 `label` 带上 `(x2=0)` 这样的后缀，否则多行表格里读者分不清谁是谁。

**验证**：非线性模型（logit）下三个取值必须给出**不同**的边际效应——线性模型下相同是正常的（边际效应是常数），所以测试用 logit 才能区分「真的按值计算」和「忽略了 at」。

### 6.6 `predict(exog=)` 的形状问题 —— 已修复并提交（`f84a404`）

**原状**：`r.predict(exog=df.head(5))` 在 `np.matmul` 里抛

```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0
```

读起来像是数据本身有问题，实际是「DataFrame 带了模型没用到的列」。

**现状**：DataFrame 按**列名**从 `design_matrix` 里选取，多余列自动忽略；缺列时明确列出缺了哪些。

**一个被测试否掉的过度设计**：我最初也给**数组**路径加了宽度校验（列数必须等于系数个数），结果三个测试立刻失败——有序模型的 `exog` 天然比 `params` 窄，因为切点、`sigma` 是参数但不是设计列。那条"更清晰的报错"挂在一个错误的规则上。已撤回，数组路径保持原样。

> **教训**：加校验时，"报错更清楚"是好事，但**先要确认校验的规则本身是对的**。

### 6.7 `etable` 写文件 —— 已修复并提交（`f84a404`，且推翻了初稿结论）

**原状**：`etable` 返回 `DataFrame`，`to_latex`/`to_markdown` 返回字符串，都无法直接落盘。

**初稿的判断是「保留即可」**，理由是分层设计（返回数据、由用户决定落盘）。这个理由本身没错，但结论错了：Stata 用户习惯 `esttab`/`outreg2` **直接写文件**，"自己落盘"在 Python 里是常识、在跨语言迁移里是缺一环。

**现状**：`etable(..., file=...)` 按扩展名写 `.csv`/`.tsv`/`.tex`/`.md`/`.html`，**同时仍然返回 `DataFrame`**；`to_latex`/`to_markdown` 完全没动。未知扩展名明确拒绝而不是猜。

**探针也跟着改了**：它原本检查 `to_latex`/`to_markdown` 的签名里有没有 `filename=`，但清单自己的建议是「加在 `etable` 上而不是改那两个函数」——于是探针盯着一个**修复永远不会去碰的地方**，会永远报「仍未修复」。现在改成实际调用 `etable(file=...)` 再看文件有没有写出来。

> 这是第二次遇到同一类问题（第一次是 `subset=` 探针读签名而非调用）。**探针要测能力，不要测声明。**
