# westata

Stata 口径的计量经济学实现，API 沿用 statsmodels 风格。

> **在看到具体数字之前，请先读这一节。** `westata` 是版本 0.0.1 的个人研究库。
> 87 条注册命令中 **52 条与 Stata 达到机器精度**，**9 条有明显偏差**，
> 另有 1 项能力缺失。每条偏差都给出实测数值和原因，不做隐藏。
>
> | 命令 | 实测最大相对偏差 | 原因 |
> |---|---|---|
> | `xtgls` | 3.4e+00 | **仅在跨方程 Sigma 秩亏时**（方程数多于期数）。此时广义逆不唯一，Stata 自己也报警告。良态系统是机器精度。 |
> | `teffects nnmatch` | 7.1e-02 | **仅标准误**；点估计精确（3.6e-15）。内部调用编译版 Mata。 |
> | `frontier` | 4.3e-02 | 仅 `distribution(exponential)`，测试数据上 `sigma_u` 近乎不可识别。在 Stata 估计值处似然完全一致。 |
> | `bsqreg` | 3.0e-02 | Bootstrap VCE。系数精确（3.4e-16）；重抽样估计量在任何有限重复数下都未收敛。 |
> | `permute` | 1.2e-02 | Stata 的洗牌顺序无法从输出反推。 |
> | `arima` | 5.5e-03 | 优化器停止点不同；我们的似然值**更高**。 |
> | `rreg` | 1.0e-03 | 最终 biweight 权重。 |
> | `xtlogit, pa` | 1.1e-03 | GEE 工作相关矩阵处理。 |
> | `heckman` | 1.9e-05 | 两步法标准误。后加的 ML 路径是独立的精确估计。 |
>
> 有一项能力**缺失而非近似**：`teffects nnmatch, biasadj()` 抛 `UnsupportedError`。
>
> 以上数值取自独立复测（386 个对照点，对真实 Stata 运行）的最差情况。库的绝大部分远好于这些上界——
> 250 个对照点达到机器精度——但会真正影响你的就是这些，所以放在最前面。

`westata` 回答一个问题：**同一份数据，在 Python 里跑出的数字，和 Stata 打印出来的是不是一样？**

- **API 沿用 statsmodels** —— `Model.from_formula(...).fit()`，返回对象带 `params`、`bse`、
  `pvalues`、`conf_int()`、`cov_params()`。结果对象是鸭子类型，已有的 statsmodels 下游代码不用改。
- **数值沿用 Stata** —— 小样本修正、权重语义、因子变量编码、自由度与自由度拆分，按对应的
  Stata 手册条目实现，而不是"接近即可"。
- **结论可审计** —— 每条命令在 `tests/conformance/registry.yml` 里带一个证据等级，不声明自己
  证明不了的事。

## 状态

| | |
|---|---|
| 测试 | **1448 通过** |
| 覆盖命令 | 87 条注册条目 |
| 证据等级 | **52 × E3**，26 × E2，9 × E1 |
| 独立复测 | 386 个对照点，对真实 Stata 运行（`retest/`） |
| 同数据对照 | 21 条命令、71 个参数，逐值比对（`docs/GUIDE.md`）|

对照基准是 **StataNow 19.5 (MP)**，通过 `pystata` 调用。捕获结果已入库，因此比对可以在没有
授权的情况下重放。

## 安装

`westata` 已发布到 PyPI：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install westata
```

Linux 或 macOS 上把第二行换成 `./.venv/bin/python -m pip install westata`。

要求 Python ≥ 3.10。`numpy`、`pandas`、`scipy`、`statsmodels`、`patsy`、`pyfixest`（HDFE）、
`linearmodels`（IV 与面板）与 `pyyaml` 会自动装好——**后三个不装就无法 import**。

`arch` 是真正可选的，只有 GARCH 与 Phillips–Perron 检验需要：

```bash
.venv/Scripts/python.exe -m pip install "westata[arch]"
```

不装时那些命令会抛 `UnsupportedError` 并指出该装哪个 extra，对应测试跳过而不是失败。

### 想改这个库本身

`pip install westata` 装到的是已发布版本。要改源码，克隆后以可编辑模式安装，并带上测试依赖：

```bash
git clone https://github.com/bigchill-z/westata.git
cd westata
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[test]"
```

之后 `python -m pytest -q` 会跑 1455 个测试。**这一步不需要 Stata 授权**：对照结果已入库，
测试重放它们即可。要重新对真实 Stata 跑对照（`retest/`、`docs/collect_comparison.py`）才需要授权。

## 快速开始

下面每一行都实际执行过，数据是 `docs/data/auto.csv`（Stata 自带的 auto 数据）。

```python
import pandas as pd
import westata as ws

df = pd.read_csv("docs/data/auto.csv")

# 带因子变量与聚类标准误的 OLS（对应 regress）
res = ws.OLS.from_formula("mpg ~ weight + C(foreign)", df).fit(
    cov_type="cluster", cov_kwds={"groups": df["foreign"]},
)
print(res.summary(style="stata"))          # Stata 版式
```

系数名沿用公式里的写法，因子水平是 `C(foreign)[T.1]` 这样的字面名，`test` 与 `lincom` 按名精确
匹配（`*` 可作通配符）：

```python
res.test("weight")                        # 单个约束
res.test("weight", "C(foreign)[T.1]")     # 联合检验
res.lincom("weight + 1000*C(foreign)[T.1]")
```

同一份数据上的其他模型：

```python
# 分位数回归（qreg）—— 中位数
ws.QuantileRegression.from_formula("mpg ~ weight + C(foreign)", df).fit()

# 稳健回归（rreg）
ws.RobustRegression.from_formula("mpg ~ weight", df).fit()

# 方程组（sureg）：每条方程一个列表元素
ws.Sureg.from_formula(["price ~ weight + mpg", "mpg ~ weight + foreign"], df).fit()
ws.Sureg.from_formula(
    ["price ~ weight + mpg", "mpg ~ weight + foreign"], df
).fit(cov_type="robust")                  # 稳健 VCE

# 截断回归（tobit）。`lower` 是 from_formula 的参数，不是 fit 的。
# rep78 有缺失值，会被剔除（74 → 69）。
ws.Tobit.from_formula("rep78 ~ weight + mpg", df, lower=0).fit()
```

出表：

```python
base = ws.OLS.from_formula("mpg ~ weight", df).fit()
full = ws.OLS.from_formula("mpg ~ weight + C(foreign)", df).fit()
print(ws.etable([base, full], model_names=["基础", "完整"]))
```

## 命令覆盖

按族分类。完整清单与逐命令用法见 [docs/GUIDE.md](docs/GUIDE.md)。

| 族 | Stata 命令 |
|---|---|
| 线性与工具变量 | `regress`、`ivregress 2sls/liml/gmm`、`eivreg`、`reghdfe`、`areg`、`nl` |
| 面板 | `xtreg fe/re/be/fd`、`xtlogit`、`xtpoisson`、`xtnbreg`、`xttobit`、`xtgls`、`xtpcse`、`xtabond`、`xtivreg` |
| 离散选择与计数 | `logit`、`probit`、`ologit`、`oprobit`、`poisson`、`nbreg`、`ztp`、`ztnb`、`tnbreg` |
| 受限因变量 | `tobit`、`truncreg`、`intreg`、`heckman`、`etregress` |
| 分位数与稳健 | `qreg`、`sqreg`、`bsqreg`、`rreg` |
| 方程组 | `sureg`、`mvreg`、`manova` |
| 因果推断 | `teffects nnmatch/ra/ipw/aipw`、`didregress`、`xtdidregress` |
| 时间序列 | `arima`、`arch`/`garch`、单位根检验 |
| 后估计 | `test`、`lincom`、`margins`、`predict`、`esttab` 风格的 `etable` |

## 权重语义

Stata 的四种权重在本库里是**分开实现**的，不是都折成一种：

| Stata | `weights_type` | 语义 |
|---|---|---|
| `fweight` | `"fweight"` | 频数权重，等价于按权重展开行。**这是实测的**：`qreg [fweight=w]` 与展开后重估的结果逐位相同，和 `[U] 20.24.1` 的原文一致。 |
| `aweight` | `"aweight"` | 解析权重，方差按 `1/w` 缩放。 |
| `pweight` | `"pweight"` | 抽样权重，隐含稳健 VCE（与 Stata 一样）。 |
| `iweight` | `"iweight"` | 重要性权重，只影响迭代，不影响样本量。 |

有效样本量的定义随权重类型变化，直接影响 HC1 这类自由度修正，所以库内部把"有效 N"
和"行数"分开处理。

## 证据分级

每个命令在注册表里有等级，含义是**这条命令被独立复测到了什么程度**，不是"作者有多自信"。

| 等级 | 含义 | 数量 |
|---|---|---|
| **E3** | 在复测覆盖的每一次调用上，`e(b)` 与 `e(V)` 与真实 Stata 全精度一致 | 52 |
| **E2** | 有真实但可界定的差异；可用，但需知道边界 | 26 |
| **E1** | 至少一次调用上明显偏离，或未被复测覆盖 | 9 |

复测本身也在仓库里：`retest/sweep.py` 会重跑对照，并且**只要注册表的等级高于实测支持的程度
就报错退出**。虚报等级被视为缺陷，不是"文档问题"。

## 与 Stata 的同数据对照

`docs/GUIDE.md` 里有 21 条命令的逐参数对照，全部来自
`docs/comparison.json`——同一份数组分别送进 Stata 和 westata，取全精度 `e(b)`/`e(V)` 比对。

结果概览：

| | |
|---|---|
| 对照命令数 | 21 |
| 参数行数 | 71 |
| 达到机器精度（< 1e-12） | **10** |
| 优于 1e-9 | **16** |
| 最差一例 | **5.80e-05**（`xtpoisson, fe` 的标准误）|

数字由脚本生成，不是手写：`docs/build_doc_data.py` → `docs/collect_comparison.py` →
`docs/comparison.json` → `docs/generate_guide.py`。不在 JSON 里的数字不会出现在表中。

## 值得知道的设计取舍

- **不做静默近似。** 未实现的能力抛 `UnsupportedError`，而不是返回一个"差不多"的结果。
  例如 `vce(jackknife)` 尚未实现时会明确报错，绝不退化成稳健标准误。
- **参数尺度按 Stata 存。** `e(V)` 停留在估计时的尺度上（如 `lnsigma`、`lndelta`），
  派生的 `sigma`/`alpha` 另带 delta 法的标准误。混用两者是这类库最常见的错误。
- **似然类命令停在 Stata 停的地方。** 若继续优化能得到更高的似然，我们不做——那会与
  Stata 的数字不一致，而这不是这个库想要的结果。这类差异在文档里如实标注。
- **两处不可达是证明过的，不是没做完。** 秩亏系统的广义逆不唯一；`teffects nnmatch` 与
  部分 `qreg` 依赖编译字节码，没有可读源码。

## 目录结构

```
src/westata/
  estimators/       各估计器（54 个可 from_formula 的模型）
  vce/              稳健、聚类与小样本修正
  postestimation/   test / lincom / margins / predict
  conformance/      证据注册表的读取与校验
  results/          统一的结果对象
tests/              1448 个测试，不需要 Stata 授权
retest/             对真实 Stata 的复测（需要授权）
docs/               GUIDE.md、对照数据、生成脚本
```

## 文档

| 文件 | 内容 |
|---|---|
| [docs/GUIDE.md](docs/GUIDE.md) | 完整使用指南：命令映射、逐参数对照、已知偏差（英文） |
| [docs/GAP_REGISTER.md](docs/GAP_REGISTER.md) | 每个未解决差异的排查记录与结论 |
| [docs/USING_WITH_STATA.md](docs/USING_WITH_STATA.md) | 哪些量与 Stata 可互换，哪些不可 |
| [CHANGELOG.md](CHANGELOG.md) | 版本变更与已知局限 |

## 已知缺口

9 条 E1 命令见开头的表格，详细排查过程在 `docs/GAP_REGISTER.md`。此外：

- `teffects nnmatch, biasadj()` 未实现（抛 `UnsupportedError`）。
- 部分 Stata 选项未覆盖（如 `vce(jackknife)`、若干 `margins` 子选项）。
- 本库不做数据管理：没有 `use`、`merge`、`reshape`、`egen` 的对应物，输入是 pandas DataFrame。

---

**许可证**：MIT，见 [LICENSE](LICENSE)。
**贡献**：这是个人项目，不设贡献流程；问题请开 issue，附上可复现的 Stata 与本库输出。
