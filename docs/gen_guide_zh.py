"""生成中文版使用指南（docs/GUIDE.zh-CN.md）。

与英文版 GUIDE.md 一样，对照表格**由 docs/comparison.json 渲染**，而不是翻译英文表格里的数字。
这一点不是形式主义：如果中文版把数字抄一遍，那么下次重跑 `collect_comparison.py` 时，只会更新
英文版，中文版会静默过期，而读者无从判断哪一版是对的。

因此这个脚本的结构与 generate_guide.py 平行：读取同一份 JSON，用同一套格式化规则，只是把
叙述部分换成中文。表格内容（命令名、数值、参数名）保持原样——它们是 Stata 与代码的标识符，
翻译它们只会制造歧义。

值得单独说明的几处，都来自实测而非推断：

  * `logit vce(robust)` 与我们的 HC1 差 0.692%，成因是有限样本修正的取法不同：Stata 对似然类
    命令除以 n-1，statsmodels 风格的 HC1 除以 n-k。而 `regress` 上两者完全一致，因为 Stata
    对它用 n/(n-k)。这不是错误，是约定的差别，必须写清楚而不是笼统说"有 0.7% 差异"。

  * `tobit` 的尺度不同：Stata 在 e(b) 里存 var(e.rep78) = 0.807583，我们暴露 /sigma = 0.898656，
    而 0.898656² = 0.807583。直接比会显示 11% 的"误差"，其实是单位问题。

  * 7 条命令的 N 不同，原因分四类（每方程 vs 堆叠行、剔除缺失、条件化/组内平均、全样本 vs
    选择样本）。每个数字单看都像 bug，所以必须逐条说明。
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(r"D:\myCodeProject\statapy")
sys.path.insert(0, str(REPO / "src"))

from westata.conformance.registry import load_registry  # noqa: E402

entries = json.loads((REPO / "docs" / "comparison.json").read_text(encoding="utf-8"))
comparisons = [e for e in entries if "rel_b" in e]
conventions = [e for e in entries if e.get("convention_note")]
comparisons.sort(key=lambda e: max(e["rel_b"], e["rel_se"]))

registry = load_registry(REPO / "tests" / "conformance" / "registry.yml")
counts = {"E3": 0, "E2": 0, "E1": 0}
for entry in registry:
    counts[entry.level] += 1


def fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    if value < 1e-12:
        return f"{value:.1e}"
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def verdict(value: float) -> str:
    if value < 1e-12:
        return "机器精度"
    if value < 1e-9:
        return "浮点噪声"
    if value < 1e-6:
        return "极好"
    if value < 1e-3:
        return "很接近"
    return "有界差异"


machine = sum(1 for e in comparisons if max(e["rel_b"], e["rel_se"]) < 1e-12)
tight = sum(1 for e in comparisons if max(e["rel_b"], e["rel_se"]) < 1e-9)
parameter_rows = sum(len(e["rows"]) for e in comparisons)
worst_value = max(max(e["rel_b"], e["rel_se"]) for e in comparisons)

# ---- 主对照表
rows = []
for entry in comparisons:
    w = max(entry["rel_b"], entry["rel_se"])
    rows.append(
        f"| `{entry['stata_command']}` | {entry['nobs_stata']} | "
        f"{fmt(entry['rel_b'])} | {fmt(entry['rel_se'])} | {verdict(w)} |"
    )
comparison_table = "\n".join(rows)

# ---- 逐参数明细
details = []
for entry in comparisons:
    lines = [
        f"#### `{entry['stata_command']}`",
        "",
        f"数据集 `{entry['dataset']}`，Stata 侧 {entry['nobs_stata']} 个观测。",
        "",
        "| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |",
        "|---|---|---|---|---|",
    ]
    for row in entry["rows"]:
        lines.append(
            f"| `{row['name']}` | {row['stata_b']:.10g} | {row['ours_b']:.10g} | "
            f"{row['stata_se']:.10g} | {row['ours_se']:.10g} |"
        )
    lines.append("")
    if entry.get("note"):
        lines.append(f"> {entry['note']}")
        lines.append("")
    details.append("\n".join(lines))
detail_section = "\n".join(details)

# ---- N 差异表
mismatches = [
    (e["command"], e["nobs_stata"], e["nobs_ours"])
    for e in comparisons
    if e["nobs_stata"] != e["nobs_ours"]
]
n_table = "| 命令 | Stata `e(N)` | westata `nobs` |\n|---|---|---|\n" + "\n".join(
    f"| `{c}` | {s} | {o} |" for c, s, o in mismatches
)

# ---- 约定差异
conv_section = ""
for entry in conventions:
    conv_section += f"""
**`{entry['stata_command']}`**

| | 斜率的标准误 |
|---|---|
| Stata `vce(robust)` | {entry['stata_se_weight']:.10g} |
| westata `cov_type='HC1'` | {entry['ours_hc1_se_weight']:.10g} |
| westata `cov_type='HC0'` | {entry['ours_hc0_se_weight']:.10g} |

`HC1` 与 Stata 差 {entry['rel_hc1']:.2%}，`HC0` 差 {entry['rel_hc0']:.2%}。二者之比为
{entry['observed_ratio']:.10f}，而 `sqrt(n/(n-1))` = {entry['predicted_ratio']:.10f}：

**Stata 对该命令的稳健 VCE 除以 n−1，statsmodels 风格的 HC1 除以 n−k。** 两种取法都有依据，
是有限样本修正的约定差别，不是错误。在 `regress` 上 Stata 用的是 `n/(n-k)`，我们的 HC1 与之
完全一致——分歧只出现在似然类命令上。

"""

DOCUMENT = f"""# westata 使用指南

`westata` 用 Python 重实现了一批 Stata 估计命令，API 沿用 statsmodels 风格，并把每一条命令的
结果与真实 Stata 逐值比对。它回答一个问题：**同一份数据，在 Python 里跑出来的数字，和 Stata
打印出来的是不是一样？**

答案是量出来的，不是声称的。本指南里的每一个对照，都是把**同一个数据数组**分别送进两个程序，
取系数与标准误的最大相对偏差，全双精度——不是比对打印出来的表格。

## 你会得到什么

- **statsmodels 形态的 API。** `Model.from_formula(...).fit()` 返回带 `params`、`bse`、
  `pvalues`、`conf_int()`、`cov_params()`、`summary()` 的对象。
- **Stata 口径的数值。** 小样本修正、权重语义、因子变量编码、自由度与自由度拆分，按对应
  手册条目实现。
- **每条命令一个证据等级**，存在 `tests/conformance/registry.yml`：**E3** 表示在独立复测覆盖的
  每次调用上都全精度一致；**E2** 表示有真实但有界的差异；**E1** 表示某处偏离或未被覆盖。
- **可重跑的验证。** `retest/` 对真实 Stata 重跑对照；`tests/` 重放已入库的捕获结果，
  不需要授权。

当前注册表共 {len(registry)} 条：**{counts['E3']} E3 / {counts['E2']} E2 / {counts['E1']} E1**。

## 安装

`westata` 尚未发布到 PyPI，从仓库安装：

```bash
git clone https://github.com/bigchill-z/westata.git
cd westata
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[test]"
```

Python ≥ 3.10。`numpy`、`pandas`、`scipy`、`statsmodels`、`patsy`、`pyfixest`、`linearmodels`、
`pyyaml` 会自动装好——**后三个不装就无法 import**。

`arch` 可选，只有 GARCH 与 Phillips–Perron 需要：

```bash
.venv/Scripts/python.exe -m pip install -e ".[test,arch]"
```

## API 的形态

```python
import pandas as pd
import westata as ws

df = pd.read_csv("docs/data/auto.csv")

res = ws.OLS.from_formula("price ~ weight + mpg + C(foreign)", df).fit()
print(res.summary(style="stata"))
print(res.params["weight"])
print(res.bse["weight"])
print(res.conf_int().loc["weight"])
```

每个估计器都是三步：`from_formula(...)` 建模型，`.fit()` 估计，结果对象带常规访问器。
Stata 里的命令行选项变成关键字参数：

```python
# Stata:  regress price weight mpg, vce(robust)
ws.OLS.from_formula("price ~ weight + mpg", df).fit(cov_type="HC1")

# Stata:  regress price weight mpg, vce(cluster foreign)
ws.OLS.from_formula("price ~ weight + mpg", df).fit(
    cov_type="cluster", cov_kwds={{"groups": df["foreign"]}},
)

# Stata:  regress price weight mpg [aweight=w]
# weights 与 weight_type 是 from_formula 的参数，不是 fit 的。
weighted = df.assign(w=abs(pd.Series(range(len(df)), dtype=float) % 7 + 1))
ws.OLS.from_formula(
    "price ~ weight + mpg", weighted, weights=weighted["w"], weight_type="aweight",
).fit()
```

## 按族分类的命令

### 线性回归与工具变量

| Stata | westata |
|---|---|
| `regress` | `OLS`（加权用 `weights=`）|
| `ivregress 2sls` / `liml` / `gmm` | `IV2SLS`、`IVLIML`、`IVGMM` |
| `eivreg` | `Eivreg` |
| `reghdfe` | `FEOLS`（后端 `pyfixest`）|
| `areg` | 单因子 absorbed 的 `FEOLS` |
| `nl` | `Nl` |

```python
sim = pd.read_csv("docs/data/sim.csv")
ws.IV2SLS.from_formula("y_iv ~ w + [x_endog ~ z1 + z2]", sim).fit()
```

### 面板

| Stata | westata |
|---|---|
| `xtreg, fe` / `re` / `be` / `fd` | `PanelOLS`、`RandomEffects`、`BetweenOLS`、`FirstDifferenceOLS` |
| `xtlogit, fe` / `re` / `pa` | `XtLogit(method=...)` |
| `xtpoisson` | `XtPoisson` |
| `xtnbreg` | `XtNbreg` |
| `xttobit` | `XtTobit` |
| `xtgls` | `XtGls` |
| `xtpcse` | `XtPcse` |
| `xtabond` | `XtAbond` |
| `xtivreg` | `XtIvreg` |

```python
panel = pd.read_csv("docs/data/panel.csv")
ws.PanelOLS.from_formula("y ~ x1 + x2", panel, entity="firm", time="year").fit()
ws.XtLogit.from_formula("binary ~ x1 + x2", panel, entity="firm", time="year",
                        method="fe").fit()
```

### 离散选择与计数

| Stata | westata |
|---|---|
| `logit` / `probit` | `Logit`、`Probit` |
| `ologit` / `oprobit` | `OrderedLogit`、`OrderedProbit` |
| `poisson` | `Poisson` |
| `nbreg` / `nbreg, dispersion(constant)` | `NegativeBinomial`、`NegativeBinomialNB1` |
| `ztp` / `ztnb` | `ZeroInflatedPoisson`、`ZeroInflatedNegativeBinomial` |
| `tnbreg` | `TruncatedNegativeBinomial` |

### 受限因变量

| Stata | westata |
|---|---|
| `tobit` | `Tobit` |
| `truncreg` | `Truncreg` |
| `intreg` | `Intreg` |
| `heckman` | `Heckman(method="twostep" 或 "ml")` |
| `etregress` | `Etregress(method="control" 或 "ml")` |

```python
ws.Tobit.from_formula("rep78 ~ weight + length", df, lower=0).fit()
ws.Heckman.from_formula("y_sel ~ w + select(observed ~ z1 + z2)", sim).fit(method="ml")
```

### 分位数与稳健回归

| Stata | westata |
|---|---|
| `qreg` | `QuantileRegression` |
| `sqreg` | `Sqreg` |
| `bsqreg` | `Bsqreg` |
| `rreg` | `RobustRegression` |

### 方程组

| Stata | westata |
|---|---|
| `sureg` | `Sureg` |
| `mvreg` | `Mvreg` |
| `manova` | `Manova` |

```python
ws.Sureg.from_formula(["price ~ weight + mpg", "mpg ~ weight + foreign"], df).fit()
ws.Sureg.from_formula(["price ~ weight + mpg", "mpg ~ weight + foreign"], df
                     ).fit(cov_type="robust")
```

### 因果推断

| Stata | westata |
|---|---|
| `teffects nnmatch` / `ra` / `ipw` / `aipw` | `NearestNeighbourMatching`、`RegressionAdjustment`、`IPW`、`AIPW` |
| `didregress` / `xtdidregress` | `Didregress`、`Xtdidregress` |

### 时间序列

| Stata | westata |
|---|---|
| `arima` | `ARIMA` |
| `arch` / `garch` | `GARCH`（需要 `arch`）|
| 单位根检验 | `unitroot(test=...)`、`XtUnitroot` |

### 后估计

| Stata | westata |
|---|---|
| `test` | `res.test(...)` |
| `lincom` | `res.lincom(...)` |
| `margins` | `res.margins(...)` |
| `predict` | `res.predict(...)` |
| `esttab` / `outreg2` | `ws.etable([...])` |

```python
res = ws.OLS.from_formula("mpg ~ weight + C(foreign)", df).fit()
res.test("weight")                       # 单个约束
res.test("weight", "C(foreign)[T.1]")    # 联合检验
res.lincom("weight + 1000*C(foreign)[T.1]")
```

> **命名细节。** 因子水平的系数名沿用公式里的写法，必须原样引用：`C(foreign)[T.1]`，不是
> `foreign`。`test` 与 `lincom` 按名精确匹配，`*` 可作通配符。

## 与 Stata 的同数据对照

下面每条命令都在同一份数据数组上跑过两个程序。数据在 `docs/data/`，数值在
`docs/comparison.json`，本节由该文件生成。

**共 {len(comparisons)} 条命令。{machine} 条达到机器精度（相对偏差低于 1e-12），{tight} 条优于
1e-9。** 表格按最差情况排序，需要先看的在最上面。

| Stata 命令 | N | `b` 最大相对偏差 | `se` 最大相对偏差 | 判定 |
|---|---|---|---|---|
{comparison_table}

### 逐参数明细

{detail_section}

## 已知偏差

上表里的差异分三类，性质完全不同。

### 有限样本修正的约定差别，不是错误

{conv_section}
### 两个 API 的尺度差别

`tobit` 报告误差尺度时用的量不同：Stata 在 `e(b)` 里存 `var(e.rep78)`，westata 暴露 `/sigma`。
换算关系是精确的：

```
Stata  var(e.rep78) = 0.807583
westata       /sigma = 0.898656
0.898656²            = 0.807583
```

所以斜率一致到 1e-15，尺度差一个平方根的关系。直接把两者相比会显示 11% 的"差异"，而它
完全是单位问题。

### 部分命令的观测数不同

有些行的 `N` 两侧不一样。每个都不是错误，而是下面四种情况之一：

{n_table}

1. **`e(N)` 数的是什么。** `sureg` 是方程组：Stata 报**每条方程**的样本量，westata 报堆叠后残差
   向量的长度（方程数 × 行数），因为结果对象是按这个校验 `nobs` 的。两条方程 74 行，就是 74 与 148。
2. **缺失值被剔除。** `tobit` 对 `rep78` 会去掉 5 个缺失观测，估计样本是 69，而 Stata 的 `e(N)`
   仍报 74。我们报实际用到的样本。
3. **条件化与组内平均。** `xtlogit, fe` 与 `xtpoisson, fe` 会条件掉结果从不变化的个体
   （1600 → 1488 与 1592）。`xtreg, be` 塌缩成每个个体一个观测，所以它的 `N` 是**组数**而不是
   行数（1600 → 200）。我们报估计器真正用到的行，Stata 报进入的行。
4. **一个模型两个样本。** `heckman` 全样本 600、被观测样本 354，两个数都对，数的是不同的东西。
   我们报全样本，使 `nobs` 描述模型自身的数据，另一个在 `e.N_selected`。

无论哪种情况，系数与标准误都达到上表的精度，那才是关键。把这些数列出来，是因为单独看
148 与 74 会被当成 bug。

### 等级为 E1 的命令

{counts['E1']} 条注册条目在至少一次调用上实测偏离。完整排查记录见 `docs/GAP_REGISTER.md`，
概要：

| 命令 | 实测最差 | 原因 |
|---|---|---|
| `xtgls` | 3.4e+00 | 仅在跨方程 Sigma 秩亏时（方程数多于期数）。无唯一广义逆，Stata 也报警告。良态系统是机器精度。 |
| `teffects nnmatch` | 7.1e-02 | 仅标准误；点估计精确（3.6e-15）。依赖编译版 Mata。 |
| `frontier` | 4.3e-02 | 仅 `distribution(exponential)`，测试数据上 `sigma_u` 近乎不可识别。在 Stata 估计值处似然完全一致。 |
| `bsqreg` | 3.0e-02 | Bootstrap VCE。系数精确（3.4e-16）；重抽样估计量在有限重复数下不收敛。 |
| `permute` | 1.2e-02 | Stata 的洗牌顺序无法从输出反推。 |
| `arima` | 5.5e-03 | 优化器停止点不同；我们的似然值**更高**。 |
| `rreg` | 1.0e-03 | 最终 biweight 权重。 |
| `xtlogit, pa` | 1.1e-03 | GEE 工作相关矩阵处理。 |
| `heckman` | 1.9e-05 | 两步法标准误。ML 路径是独立的精确估计。 |

有一项能力**缺失而非近似**：`teffects nnmatch, biasadj()` 抛 `UnsupportedError`。

其中几项是**证明过不可达**，不是没做完：秩亏系统没有唯一答案；部分命令调用没有可读源码的编译
字节码；追上 Stata 的停止点意味着报告一个更差的解；bootstrap 在任何有限重复数下都无法逐点一致。

## 复现这里每一个数字

```bash
# 数据集（auto.dta 是 Stata 自带的，另两个由固定种子生成）
.venv/Scripts/python.exe docs/build_doc_data.py

# 重跑全部对照并重写 comparison.json
.venv/Scripts/python.exe docs/collect_comparison.py

# 由 JSON 重新渲染本指南（中英文各一份）
.venv/Scripts/python.exe docs/generate_guide.py
.venv/Scripts/python.exe docs/gen_guide_zh.py
```

前两条需要 Stata 授权，测试套件不需要：

```bash
.venv/Scripts/python.exe -m pytest -q
```

`tests/test_docs_zh.py` 会执行中文文档里的每个代码块，并核对数字与英文版同源；中文版不是
一份无人验证的副本。
"""

out = REPO / "docs" / "GUIDE.zh-CN.md"
out.write_text(DOCUMENT, encoding="utf-8")
print(f"wrote {out}")
print(f"  {len(DOCUMENT.splitlines())} lines, {len(DOCUMENT)} chars")
print(f"  {len(comparisons)} 条对照，{parameter_rows} 个参数行，{len(conventions)} 条约定说明")
print(f"  机器精度 {machine}，优于 1e-9 {tight}，最差 {worst_value:.2e}")
