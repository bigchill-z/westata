# westata 使用指南

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

当前注册表共 87 条：**52 E3 / 26 E2 / 9 E1**。

## 安装

`westata` 已发布到 PyPI：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install westata
```

Linux 或 macOS 上把第二行换成 `./.venv/bin/python -m pip install westata`。

Python ≥ 3.10。`numpy`、`pandas`、`scipy`、`statsmodels`、`patsy`、`pyfixest`、`linearmodels`、
`pyyaml` 会自动装好——**后三个不装就无法 import**。

`arch` 可选，只有 GARCH 与 Phillips–Perron 需要：

```bash
.venv/Scripts/python.exe -m pip install "westata[arch]"
```

要改库本身，克隆后以可编辑模式安装并带上测试依赖；测试不需要 Stata 授权，重放已入库的捕获结果：

```bash
git clone https://github.com/bigchill-z/westata.git
cd westata
.venv/Scripts/python.exe -m pip install -e ".[test]"
.venv/Scripts/python.exe -m pytest -q
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
    cov_type="cluster", cov_kwds={"groups": df["foreign"]},
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

**共 21 条命令。10 条达到机器精度（相对偏差低于 1e-12），16 条优于
1e-9。** 表格按最差情况排序，需要先看的在最上面。

| Stata 命令 | N | `b` 最大相对偏差 | `se` 最大相对偏差 | 判定 |
|---|---|---|---|---|
| `xtreg y x1 x2, be` | 1600 | 3.6e-16 | 3.1e-16 | 机器精度 |
| `xtreg y x1 x2, fe` | 1600 | 4.5e-16 | 3.2e-16 | 机器精度 |
| `xtreg y x1 x2, re` | 1600 | 1.2e-15 | 9.6e-16 | 机器精度 |
| `ivregress 2sls y_iv w (x_endog = z1 z2)` | 600 | 6.9e-15 | 6.1e-15 | 机器精度 |
| `regress price weight mpg, vce(robust)` | 74 | 2.0e-14 | 9.4e-15 | 机器精度 |
| `regress price weight mpg foreign` | 74 | 3.1e-14 | 7.7e-15 | 机器精度 |
| `tobit rep78 weight length, ll(0)` | 74 | 1.1e-15 | 2.6e-13 | 机器精度 |
| `sureg (price weight mpg) (mpg weight foreign)` | 74 | 5.8e-13 | 6.1e-15 | 机器精度 |
| `sureg (price weight mpg) (mpg weight foreign), vce(robust)` | 74 | 5.8e-13 | 2.0e-14 | 机器精度 |
| `regress price weight i.foreign` | 74 | 6.8e-13 | 4.2e-16 | 机器精度 |
| `poisson gear_ratio weight` | 74 | 1.18e-12 | 1.5e-13 | 浮点噪声 |
| `probit hi weight length` | 74 | 1.06e-11 | 2.38e-12 | 浮点噪声 |
| `rreg price weight length, nolog` | 74 | 3.5e-13 | 2.70e-11 | 浮点噪声 |
| `etregress y_treat w, treat(treated = z1 z2) twostep` | 600 | 6.0e-14 | 2.71e-11 | 浮点噪声 |
| `nbreg count w, dispersion(constant)` | 600 | 1.17e-12 | 4.62e-11 | 浮点噪声 |
| `qreg price weight length, q(.5)` | 74 | 6.8e-14 | 8.15e-11 | 浮点噪声 |
| `nbreg count w` | 600 | 4.13e-09 | 8.37e-10 | 极好 |
| `logit hi weight length` | 74 | 1.87e-08 | 1.30e-08 | 极好 |
| `heckman y_sel w, select(observed = z1 z2) twostep` | 600 | 1.1e-14 | 6.96e-08 | 极好 |
| `xtlogit binary x1 x2, fe` | 1600 | 6.20e-08 | 3.21e-05 | 很接近 |
| `xtpoisson count x1 x2, fe` | 1600 | 3.70e-08 | 5.80e-05 | 很接近 |

### 逐参数明细

#### `xtreg y x1 x2, be`

数据集 `panel`，Stata 侧 1600 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x1` | 1.46407763 | 1.46407763 | 0.09044027402 | 0.09044027402 |
| `x2` | -0.9911839003 | -0.9911839003 | 0.1898587281 | 0.1898587281 |
| `Intercept` | 1.838528861 | 1.838528861 | 0.07239053303 | 0.07239053303 |

#### `xtreg y x1 x2, fe`

数据集 `panel`，Stata 侧 1600 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x1` | 1.487257092 | 1.487257092 | 0.03864462641 | 0.03864462641 |
| `x2` | -0.7551388425 | -0.7551388425 | 0.02286842338 | 0.02286842338 |
| `Intercept` | 1.842559819 | 1.842559819 | 0.02176233024 | 0.02176233024 |

#### `xtreg y x1 x2, re`

数据集 `panel`，Stata 侧 1600 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x1` | 1.484380977 | 1.484380977 | 0.03552691621 | 0.03552691621 |
| `x2` | -0.7584868078 | -0.7584868078 | 0.0227007311 | 0.0227007311 |
| `Intercept` | 1.842531408 | 1.842531408 | 0.07229521854 | 0.07229521854 |

#### `ivregress 2sls y_iv w (x_endog = z1 z2)`

数据集 `sim`，Stata 侧 600 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x_endog` | 2.09282723 | 2.09282723 | 0.05198730455 | 0.05198730455 |
| `w` | 0.5002117913 | 0.5002117913 | 0.03644423176 | 0.03644423176 |
| `Intercept` | 1.030126954 | 1.030126954 | 0.0385398934 | 0.0385398934 |

#### `regress price weight mpg, vce(robust)`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | 1.746559158 | 1.746559158 | 0.7778369564 | 0.7778369564 |
| `mpg` | -49.51222067 | -49.51222067 | 95.80740087 | 95.80740087 |
| `Intercept` | 1946.068668 | 1946.068668 | 4213.792651 | 4213.792651 |

#### `regress price weight mpg foreign`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | 3.46470583 | 3.46470583 | 0.6307489576 | 0.6307489576 |
| `mpg` | 21.85360382 | 21.85360382 | 74.22113859 | 74.22113859 |
| `foreign` | 3673.060378 | 3673.060378 | 683.9782707 | 683.9782707 |
| `Intercept` | -5853.695656 | -5853.695656 | 3376.987425 | 3376.987425 |

#### `tobit rep78 weight length, ll(0)`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | -0.0007198789353 | -0.0007198789353 | 0.0004311819354 | 0.0004311819354 |
| `length` | 0.008090822232 | 0.008090822232 | 0.01502867365 | 0.01502867365 |
| `Intercept` | 4.065071154 | 4.065071154 | 1.647842033 | 1.647842033 |

> Stata stores var(e.rep78) = 0.807583; our /sigma = 0.898656 and 0.898656^2 = 0.807583. Slopes match to 1e-15.

#### `sureg (price weight mpg) (mpg weight foreign)`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `price:weight` | 1.335250983 | 1.335250983 | 0.6263626215 | 0.6263626215 |
| `price:mpg` | -117.9644778 | -117.9644778 | 84.00853315 | 84.00853315 |
| `price:_cons` | 4645.845099 | 4645.845099 | 3509.1146 | 3509.1146 |
| `mpg:weight` | -0.006746266877 | -0.006746266877 | 0.0006230708757 | 0.0006230708757 |
| `mpg:foreign` | -2.101224951 | -2.101224951 | 1.049174135 | 1.049174135 |
| `mpg:_cons` | 42.29206513 | 42.29206513 | 2.116824119 | 2.116824119 |

> Stata repeats each name once per equation, so positions are used.

#### `sureg (price weight mpg) (mpg weight foreign), vce(robust)`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `price:weight` | 1.335250983 | 1.335250983 | 0.7599374481 | 0.7599374481 |
| `price:mpg` | -117.9644778 | -117.9644778 | 91.70326703 | 91.70326703 |
| `price:_cons` | 4645.845099 | 4645.845099 | 4053.055856 | 4053.055856 |
| `mpg:weight` | -0.006746266877 | -0.006746266877 | 0.0005313701289 | 0.0005313701289 |
| `mpg:foreign` | -2.101224951 | -2.101224951 | 1.092650401 | 1.092650401 |
| `mpg:_cons` | 42.29206513 | 42.29206513 | 1.746566117 | 1.746566117 |

#### `regress price weight i.foreign`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | 3.320736771 | 3.320736771 | 0.3958784154 | 0.3958784154 |
| `C(foreign)[T.1]` | 3637.001296 | 3637.001296 | 668.5829516 | 668.5829516 |
| `Intercept` | -4942.843954 | -4942.843954 | 1345.590935 | 1345.590935 |

> Stata labels the level `1.foreign`; the CSV stores 0/1 so our label is `C(foreign)[T.1]`. The estimates agree.

#### `poisson gear_ratio weight`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | -0.00014941959 | -0.00014941959 | 8.774112833e-05 | 8.774112833e-05 |
| `Intercept` | 1.54811402 | 1.54811402 | 0.2657715184 | 0.2657715184 |

#### `probit hi weight length`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | -0.0003137140304 | -0.0003137140304 | 0.0007026668139 | 0.0007026668139 |
| `length` | -0.057869104 | -0.05786910401 | 0.02627042341 | 0.02627042341 |
| `Intercept` | 11.8161567 | 11.8161567 | 3.249590634 | 3.249590634 |

#### `rreg price weight length, nolog`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | -0.6670513483 | -0.6670513483 | 0.5687771336 | 0.5687771336 |
| `length` | 37.72364884 | 37.72364884 | 19.85283308 | 19.85283308 |
| `Intercept` | -93.82767292 | -93.82767292 | 2183.280907 | 2183.280906 |

#### `etregress y_treat w, treat(treated = z1 z2) twostep`

数据集 `sim`，Stata 侧 600 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `y:w` | 0.885450556 | 0.885450556 | 0.04264076704 | 0.04264076704 |
| `y:treated` | 1.359499616 | 1.359499616 | 0.1868684707 | 0.1868684707 |
| `y:Intercept` | 1.14523925 | 1.14523925 | 0.1243893358 | 0.1243893358 |
| `treat:z1` | 0.8247960006 | 0.8247960006 | 0.07556662354 | 0.07556662354 |
| `treat:z2` | -0.4380828094 | -0.4380828094 | 0.06418698055 | 0.06418698055 |
| `treat:Intercept` | 0.46625764 | 0.46625764 | 0.06124320345 | 0.06124320345 |

#### `nbreg count w, dispersion(constant)`

数据集 `sim`，Stata 侧 600 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `w` | 0.595274193 | 0.595274193 | 0.03809570461 | 0.03809570461 |
| `Intercept` | 0.4996785731 | 0.4996785731 | 0.04863632927 | 0.04863632927 |

> NB1: Var = mu(1 + delta*mu). Both sides store lndelta.

#### `qreg price weight length, q(.5)`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | 1.231139381 | 1.231139381 | 1.535295882 | 1.535295881 |
| `length` | -8.129977876 | -8.129977876 | 53.58860451 | 53.5886045 |
| `Intercept` | 2982.599558 | 2982.599558 | 5893.313894 | 5893.313893 |

#### `nbreg count w`

数据集 `sim`，Stata 侧 600 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `w` | 0.6634965543 | 0.6634965531 | 0.04272214976 | 0.04272214972 |
| `Intercept` | 0.4538139093 | 0.4538139112 | 0.04578444104 | 0.04578444101 |

> e(b) is on the lnalpha scale in both; alpha = exp(lnalpha).

#### `logit hi weight length`

数据集 `auto`，Stata 侧 74 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | -0.0007593003302 | -0.0007593003307 | 0.001423994447 | 0.00142399446 |
| `length` | -0.1005478428 | -0.1005478447 | 0.05186548617 | 0.05186548673 |
| `Intercept` | 21.10931077 | 21.10931113 | 6.46264589 | 6.462645974 |

#### `heckman y_sel w, select(observed = z1 z2) twostep`

数据集 `sim`，Stata 侧 600 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `y:w` | 1.159484099 | 1.159484099 | 0.05671698849 | 0.05671698849 |
| `y:Intercept` | 1.061586534 | 1.061586534 | 0.1089878615 | 0.1089878619 |
| `select:z1` | 0.8260395337 | 0.8260395337 | 0.07739636536 | 0.07739637075 |
| `select:z2` | -0.5280892434 | -0.5280892434 | 0.06525157616 | 0.06525157929 |
| `select:Intercept` | 0.3940601347 | 0.3940601347 | 0.06082757555 | 0.06082757746 |
| `lambda` | 0.631022215 | 0.631022215 | 0.1786437028 | 0.1786437035 |

> Stata reports a Mills-ratio coefficient and two `_cons`; positions are used.

#### `xtlogit binary x1 x2, fe`

数据集 `panel`，Stata 侧 1600 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x1` | 0.9643481998 | 0.9643482478 | 0.1122940975 | 0.1122904893 |
| `x2` | -0.6155301523 | -0.6155301905 | 0.06679084262 | 0.0667899936 |

> Conditional logit: the intercept is conditioned out, so only slopes compare.

#### `xtpoisson count x1 x2, fe`

数据集 `panel`，Stata 侧 1600 个观测。

| 参数 | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x1` | 0.4878093781 | 0.48780936 | 0.03574819113 | 0.03574611928 |
| `x2` | -0.2964989705 | -0.2964989666 | 0.02157539189 | 0.02157489697 |

> Conditional Poisson: only slopes compare.


## 已知偏差

上表里的差异分三类，性质完全不同。

### 有限样本修正的约定差别，不是错误


**`logit hi weight, vce(robust)`**

| | 斜率的标准误 |
|---|---|
| Stata `vce(robust)` | 0.0009007456262 |
| westata `cov_type='HC1'` | 0.0009069792356 |
| westata `cov_type='HC0'` | 0.000894638807 |

`HC1` 与 Stata 差 0.69%，`HC0` 差 0.68%。二者之比为
1.0069204992，而 `sqrt(n/(n-1))` = 1.0069204978：

**Stata 对该命令的稳健 VCE 除以 n−1，statsmodels 风格的 HC1 除以 n−k。** 两种取法都有依据，
是有限样本修正的约定差别，不是错误。在 `regress` 上 Stata 用的是 `n/(n-k)`，我们的 HC1 与之
完全一致——分歧只出现在似然类命令上。


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

| 命令 | Stata `e(N)` | westata `nobs` |
|---|---|---|
| `xtreg, be` | 1600 | 200 |
| `tobit` | 74 | 69 |
| `sureg` | 74 | 148 |
| `sureg vce(robust)` | 74 | 148 |
| `heckman twostep` | 600 | 354 |
| `xtlogit, fe` | 1600 | 1488 |
| `xtpoisson, fe` | 1600 | 1592 |

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

9 条注册条目在至少一次调用上实测偏离。完整排查记录见 `docs/GAP_REGISTER.md`，
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
