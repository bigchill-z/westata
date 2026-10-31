# westata 可优化点清单

**来源**：Autor–Dorn–Hanson (2013) 复现过程中实测得到。
**被测版本**：`westata 0.0.3`（本仓库 `src/westata`，未作任何改动）。
**复核方式**：`.venv/Scripts/python.exe replication/papers/adh2013/probe_gaps.py`

> 配套报告：[REPORT.md](REPORT.md) —— 讲这篇论文的复现结果。
> 本文只讲 westata 本身**在自身 API 契约下**还可以改进什么。

## 0. 先说清楚：本文不要求 westata 模仿 Stata

westata 的 API 是 **statsmodels 风格，这是刻意的设计选择**（README 开篇即写"API 沿用 statsmodels 风格"），不是缺陷。

所以本文**不以"像不像 Stata"为判据**。初稿曾按这个错误框架列过 5 条"缺口"（`i.` 语法糖、`lincom` 的 `_b[]`、`predict` 别名、全局 `set seed`、`log using`），它们全部被删除——详见 §5。

本文的判据只有三条：

1. **能力缺失**——某类模型根本跑不了（如加权 IV），与语法无关；
2. **同一 API 内部不一致**——同为 statsmodels 风格，A 估计器有这个参数、B 没有；或同一语义在不同估计器上要用不同 kwds 键名；
3. **契约违约**——签名接受了参数却不生效，或报错信息误导。

已排除题目点明的数据操作层（`egen`/`merge`/`reshape`/`collapse` 全缺）与绘图层（`graph`/`twoway` 全缺）——这两块按 README 的定位本来就该交给 pandas 和 matplotlib。

## 1. P0：能力缺失（判据 1）

**1. 加权 IV 原生不支持——本次复现最痛的缺口**

```python
ws.IV2SLS(..., weights=w, weight_type="aweight")
# → UnsupportedError: IV estimation does not support weights in P1
```

这不是"语法不同"，是**功能没有**：`IV2SLS`/`IVLIML`/`IVGMM` 三个 IV 估计器全部拒绝权重，而 `OLS`/`FEOLS` 都支持。论文 166 条估计命令里 153 条是加权 IV，**92% 无法原生运行**。

目前只能由调用方手工做 √w 缩放，且必须连常数项一起缩放（漏掉会静默产生 4.4% 偏差，见 REPORT.md §5.3 的实测）。**这个绕法本身就是风险**：它要求使用者理解 aweight 的数学等价性，而这恰恰是库本该封装掉的东西。

> 建议：在 `estimators/iv.py` 实现权重（内部 √w 变换，并正确回写 `e(rmse)`/`e(r2)`/`e(df_r)`），补 conformance 证据。这是唯一一条"不做就永远要用户自己兜底"的缺口。

**2. `FEOLS` 没有 `predict`**

```python
ws.FEOLS.from_formula("y ~ x1 | firm", df).fit().predict(kind="xb")
# → UnsupportedError: predict: no prediction types registered for 'reghdfe'
```

`OLS`、`IV2SLS`、`PanelOLS`、`Logit`、`Poisson` **都能 predict**，唯独 `reghdfe` 不能。这是判据 2（内部不一致）与判据 1（能力缺失）叠加：固定效应模型做拟合值/残差是常规需求，而且它在 westata 里并非无解——吸收掉的固定效应可以从 `e()` 里还原。

> 建议：至少支持 `kind="xb"`（含吸收项）与 `kind="residuals"`。

## 2. P1：同一 API 的内部不一致（判据 2）

**3. `subset=` 只在部分估计器上存在**

| 估计器 | `subset=` |
|---|---|
| `OLS` / `Logit` / `Poisson` | ✅ |
| `IV2SLS` / `IVLIML` / `IVGMM` / `FEOLS` / `PanelOLS` | ❌ |

这意味着"限制估计样本"这个**跨模型通用**的操作，在不同估计器上要用不同写法：有的传 `subset=`，有的必须先 `df[mask].reset_index(drop=True)`。后者还有个隐患——手工过滤后 `PanelOLS` 的个体/时间索引可能不再连续。

> 建议：把 `subset=` 提升为所有 `from_formula` 的公共参数（`OLS` 已有实现可直接复用）。

**4. `margins(at=...)` 只接受标量**

```python
r.margins(dydx="x1")            # ✅
r.margins(at={"x1": [0, 1, 2]}) # ❌ TypeError: float() argument must be ... not 'list'
```

`margins` 是 statsmodels 风格 API 的一部分，而"在多个取值上算边际效应/预测值"是这个 API 的常规用法。现在只能每次传一个值、循环调用。

> 建议：`at=` 接受序列，返回按取值展开的表。

## 3. P2：契约违约与工程化（判据 3）

**5. `predict(exog=...)` 的签名承诺了"设计矩阵"，但没有校验，也不接受 DataFrame**

```python
r = ws.OLS.from_formula("y ~ x1 + x2", df).fit()
r.predict(exog=df.head(5))   # ValueError: matmul ... core dimension 0
```

按 statsmodels 惯例，`exog` 是**数组**，所以不接受 DataFrame 本身不算违约。但有两点值得改：

- 报错来自 `np.matmul` 的维数不匹配，**没有说明期望什么形状**；
- 结果对象其实存了 `design_matrix`（`(X, names)` 元组），却没有公开的重建入口——用户要自己拼截距列和列序。

`build_design(formula, new_df)` 是可用的（我实测用它 + `params` 得到了正确的样本外预测），但它不在 `predict` 的文档里。

> 建议：加一个 `predict_on(new_data, formula=...)` 或让 `predict` 接受 DataFrame 并内部调 `build_design`。这是**便利性**改进，不是能力缺失。

**7. `etable` 无写文件入口**

`ws.etable` 返回 `DataFrame`，`to_latex`/`to_markdown` 返回字符串，都没有 `filename=`。这是有意的分层设计（返回数据、由用户决定落盘），保留即可；若要做，加一个 `etable(..., file=)` 比改现有函数更好。

### 小结

| 优先级 | 事项 | 判据 | 影响 |
|---|---|---|---|
| **P0-1** | 加权 IV 不支持 | 能力缺失 | 论文 92% 命令无法原生运行 |
| **P0-2** | `FEOLS` 无 `predict` | 能力缺失 + 不一致 | 固定效应模型无法取拟合值/残差 |
| **P1-3** | `subset=` 部分估计器才有 | 内部不一致 | 样本限制写法因估计器而异 |
| **P1-4** | `margins(at=)` 只接受标量 | 能力缺失（同一 API 内不自洽） | 多取值分析要循环 |
| **P2-5** | `predict(exog=)` 无形状校验/重建入口 | 契约与易用性 | 样本外预测要手工拼矩阵 |
| **P2-6** | `etable` 无写文件 | 可选便利性 | 保留现状即可 |

**如果只做两件事**：P0-1（加权 IV）与 P0-2（`FEOLS` 的 `predict`）。前者是唯一"不做就永远要用户兜底"的能力缺口；后者让固定效应模型与其余估计器一样能取拟合值和残差。原清单里排第二的「统一聚类键名」已于 §6 修复。

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

## 6. 已修复的条目

这一节记录**曾经成立、现已修好**的缺口。保留它们是因为一条过时的"westata 做不到 X"与谎称"能做到"同样有害，而"最近修了什么"本身也是有用的信息。

### 6.1 `cov_kwds` 聚类键名三套并存 —— 已修复

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
