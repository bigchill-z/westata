# westata — a guide

`westata` reimplements a set of Stata estimation commands in Python, with a statsmodels-style API, and
checks each one against a real Stata installation. It answers one question: *if I run this in Python,
do I get the numbers Stata would print?*

The answer is measured, not asserted. Every comparison in this guide comes from running the **same
data array** through both programs and taking the maximum relative difference over coefficients and
standard errors, at full double precision — not from comparing printed output.

## What you get

- **statsmodels-shaped API.** `Model.from_formula(...).fit()` returns an object with `params`, `bse`,
  `pvalues`, `conf_int()`, `cov_params()`, `summary()`. Existing downstream code keeps working.
- **Stata numerics.** Small-sample corrections, weight semantics, factor-variable coding, degrees of
  freedom, and degrees-of-freedom splits follow the corresponding Stata manual entry.
- **An evidence grade per command**, stored in `tests/conformance/registry.yml`: **E3** means full
  precision `e(b)`/`e(V)` agreement on every invocation an independent sweep exercised; **E2** means a
  real but bounded difference; **E1** means it deviates somewhere, or is not covered.
- **A reproducible check.** `retest/` re-runs the comparison against live Stata; `tests/` replays the
  committed captures without needing a licence.

Current counts: **52 E3, 26 E2, 9 E1** across 87 registry entries. The nine E1 entries are listed with
their measured numbers in [Known deviations](#known-deviations).

## Install

`westata` is not on PyPI yet. Install from the repository:

```bash
git clone https://github.com/bigchill-z/westata.git
cd westata
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[test]"
```

On Linux or macOS use `./.venv/bin/python` instead.

Python 3.10 or newer. `numpy`, `pandas`, `scipy`, `statsmodels`, `patsy`, `pyfixest`, `linearmodels`
and `pyyaml` install automatically — the package does not import without the last three.

`arch` is optional, needed only for GARCH and the Phillips–Perron test:

```bash
.venv/Scripts/python.exe -m pip install -e ".[test,arch]"
```

## The shape of the API

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

Every estimator follows the same three steps: `from_formula(...)` builds the model, `.fit()` estimates
it, and the result carries the usual accessors. Options that Stata exposes as command syntax become
keyword arguments:

```python
# Stata:  regress price weight mpg, vce(robust)
ws.OLS.from_formula("price ~ weight + mpg", df).fit(cov_type="HC1")

# Stata:  regress price weight mpg, vce(cluster foreign)
ws.OLS.from_formula("price ~ weight + mpg", df).fit(
    cov_type="cluster", cov_kwds={"groups": df["foreign"]},
)

# Stata:  regress price weight mpg [aweight=w]
# weights and weight_type are from_formula arguments, not fit ones.
weighted = df.assign(w=np.abs(np.random.default_rng(0).normal(size=len(df))))
ws.OLS.from_formula(
    "price ~ weight + mpg", weighted, weights=weighted["w"], weight_type="aweight",
).fit()
```

## Commands by family

### Linear regression and instrumental variables

| Stata | westata |
|---|---|
| `regress` | `OLS`, `WLS` via `weights=` |
| `ivregress 2sls` / `liml` / `gmm` | `IV2SLS`, `IVLIML`, `IVGMM` |
| `eivreg` | `Eivreg` |
| `reghdfe` | `FEOLS` (backend: `pyfixest`) |
| `areg` | `FEOLS` with one absorbed factor |
| `nl` | `Nl` |

```python
ws.IV2SLS.from_formula("y_iv ~ w + [x_endog ~ z1 + z2]", sim).fit()
ws.FEOLS.from_formula("y ~ x1 + x2 | firm + year", panel).fit()
```

### Panel data

| Stata | westata |
|---|---|
| `xtreg, fe` / `re` / `be` / `fd` | `PanelOLS`, `RandomEffects`, `BetweenOLS`, `FirstDifferenceOLS` |
| `xtlogit, fe` / `re` / `pa` | `XtLogit(method=...)` |
| `xtpoisson` | `XtPoisson` |
| `xtnbreg` | `XtNbreg` |
| `xttobit` | `XtTobit` |
| `xtgls` | `XtGls` |
| `xtpcse` | `XtPcse` |
| `xtabond` | `XtAbond` |
| `xtivreg` | `XtIvreg` |

```python
ws.PanelOLS.from_formula("y ~ x1 + x2", panel, entity="firm", time="year").fit()
ws.XtLogit.from_formula("binary ~ x1 + x2", panel, entity="firm", time="year",
                        method="fe").fit()
```

### Discrete choice and counts

| Stata | westata |
|---|---|
| `logit` / `probit` | `Logit`, `Probit` |
| `ologit` / `oprobit` | `OrderedLogit`, `OrderedProbit` |
| `poisson` | `Poisson` |
| `nbreg` / `nbreg, dispersion(constant)` | `NegativeBinomial`, `NegativeBinomialNB1` |
| `ztp` / `ztnb` | `ZeroInflatedPoisson`, `ZeroInflatedNegativeBinomial` |
| `tnbreg` | `TruncatedNegativeBinomial` |

### Limited dependent variables

| Stata | westata |
|---|---|
| `tobit` | `Tobit` |
| `truncreg` | `Truncreg` |
| `intreg` | `Intreg` |
| `heckman` | `Heckman(method="twostep" or "ml")` |
| `etregress` | `Etregress(method="control" or "ml")` |

```python
ws.Tobit.from_formula("rep78 ~ weight + length", df, lower=0).fit()
ws.Heckman.from_formula("y_sel ~ w + select(observed ~ z1 + z2)", sim).fit(method="ml")
```

### Quantile and robust regression

| Stata | westata |
|---|---|
| `qreg` | `QuantileRegression` |
| `sqreg` | `Sqreg` |
| `bsqreg` | `Bsqreg` |
| `rreg` | `RobustRegression` |

### Systems of equations

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

### Causal inference

| Stata | westata |
|---|---|
| `teffects nnmatch` / `ra` / `ipw` / `aipw` | `NearestNeighbourMatching`, `RegressionAdjustment`, `IPW`, `AIPW` |
| `didregress` / `xtdidregress` | `Didregress`, `Xtdidregress` |

### Time series

| Stata | westata |
|---|---|
| `arima` | `ARIMA` |
| `arch` / `garch` | `GARCH` (needs `arch`) |
| unit-root tests | `unitroot(test=...)`, `XtUnitroot` |

### Post-estimation

| Stata | westata |
|---|---|
| `test` | `res.test(...)` |
| `lincom` | `res.lincom(...)` |
| `margins` | `res.margins(...)` |
| `predict` | `res.predict(...)` |
| `esttab` / `outreg2` | `ws.etable([...])` |

```python
res = ws.OLS.from_formula("mpg ~ weight + C(foreign)", df).fit()
res.test("weight")                       # one restriction
res.test("weight", "C(foreign)[T.1]")    # joint
res.lincom("weight + 1000*C(foreign)[T.1]")
```

> **Naming detail.** A factor level's coefficient keeps the formula's spelling, so reference it exactly:
> `C(foreign)[T.1]`, not `foreign`. `test` and `lincom` match names literally, with `*` available as a
> wildcard.

## Comparison against Stata

Every command below was run on the same data array through both programs. The data is in
`docs/data/` and the numbers are in `docs/comparison.json`; this section is generated from that file.

**21 commands compared. 10 agree to machine precision (relative difference below 1e-12), 16
to better than 1e-9.** The table is ordered by worst case, so the top rows are where to look first.

| Stata command | N | max rel. diff. in `b` | max rel. diff. in `se` | verdict |
|---|---|---|---|---|
| `xtreg y x1 x2, be` | 1600 | 3.6e-16 | 3.1e-16 | machine precision |
| `xtreg y x1 x2, fe` | 1600 | 4.5e-16 | 3.2e-16 | machine precision |
| `xtreg y x1 x2, re` | 1600 | 1.2e-15 | 9.6e-16 | machine precision |
| `ivregress 2sls y_iv w (x_endog = z1 z2)` | 600 | 6.9e-15 | 6.1e-15 | machine precision |
| `regress price weight mpg, vce(robust)` | 74 | 2.0e-14 | 9.4e-15 | machine precision |
| `regress price weight mpg foreign` | 74 | 3.1e-14 | 7.7e-15 | machine precision |
| `tobit rep78 weight length, ll(0)` | 74 | 1.1e-15 | 2.6e-13 | machine precision |
| `sureg (price weight mpg) (mpg weight foreign)` | 74 | 5.8e-13 | 6.1e-15 | machine precision |
| `sureg (price weight mpg) (mpg weight foreign), vce(robust)` | 74 | 5.8e-13 | 2.0e-14 | machine precision |
| `regress price weight i.foreign` | 74 | 6.8e-13 | 4.2e-16 | machine precision |
| `poisson gear_ratio weight` | 74 | 1.18e-12 | 1.5e-13 | float noise |
| `probit hi weight length` | 74 | 1.06e-11 | 2.38e-12 | float noise |
| `rreg price weight length, nolog` | 74 | 3.5e-13 | 2.70e-11 | float noise |
| `etregress y_treat w, treat(treated = z1 z2) twostep` | 600 | 6.0e-14 | 2.71e-11 | float noise |
| `nbreg count w, dispersion(constant)` | 600 | 1.17e-12 | 4.62e-11 | float noise |
| `qreg price weight length, q(.5)` | 74 | 6.8e-14 | 8.15e-11 | float noise |
| `nbreg count w` | 600 | 4.13e-09 | 8.37e-10 | excellent |
| `logit hi weight length` | 74 | 1.87e-08 | 1.30e-08 | excellent |
| `heckman y_sel w, select(observed = z1 z2) twostep` | 600 | 1.1e-14 | 6.96e-08 | excellent |
| `xtlogit binary x1 x2, fe` | 1600 | 6.20e-08 | 3.21e-05 | very close |
| `xtpoisson count x1 x2, fe` | 1600 | 3.70e-08 | 5.80e-05 | very close |

### Side-by-side detail

#### `xtreg y x1 x2, be`

Dataset: `panel` — 1600 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x1` | 1.46407763 | 1.46407763 | 0.09044027402 | 0.09044027402 |
| `x2` | -0.9911839003 | -0.9911839003 | 0.1898587281 | 0.1898587281 |
| `Intercept` | 1.838528861 | 1.838528861 | 0.07239053303 | 0.07239053303 |

#### `xtreg y x1 x2, fe`

Dataset: `panel` — 1600 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x1` | 1.487257092 | 1.487257092 | 0.03864462641 | 0.03864462641 |
| `x2` | -0.7551388425 | -0.7551388425 | 0.02286842338 | 0.02286842338 |
| `Intercept` | 1.842559819 | 1.842559819 | 0.02176233024 | 0.02176233024 |

#### `xtreg y x1 x2, re`

Dataset: `panel` — 1600 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x1` | 1.484380977 | 1.484380977 | 0.03552691621 | 0.03552691621 |
| `x2` | -0.7584868078 | -0.7584868078 | 0.0227007311 | 0.0227007311 |
| `Intercept` | 1.842531408 | 1.842531408 | 0.07229521854 | 0.07229521854 |

#### `ivregress 2sls y_iv w (x_endog = z1 z2)`

Dataset: `sim` — 600 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x_endog` | 2.09282723 | 2.09282723 | 0.05198730455 | 0.05198730455 |
| `w` | 0.5002117913 | 0.5002117913 | 0.03644423176 | 0.03644423176 |
| `Intercept` | 1.030126954 | 1.030126954 | 0.0385398934 | 0.0385398934 |

#### `regress price weight mpg, vce(robust)`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | 1.746559158 | 1.746559158 | 0.7778369564 | 0.7778369564 |
| `mpg` | -49.51222067 | -49.51222067 | 95.80740087 | 95.80740087 |
| `Intercept` | 1946.068668 | 1946.068668 | 4213.792651 | 4213.792651 |

#### `regress price weight mpg foreign`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | 3.46470583 | 3.46470583 | 0.6307489576 | 0.6307489576 |
| `mpg` | 21.85360382 | 21.85360382 | 74.22113859 | 74.22113859 |
| `foreign` | 3673.060378 | 3673.060378 | 683.9782707 | 683.9782707 |
| `Intercept` | -5853.695656 | -5853.695656 | 3376.987425 | 3376.987425 |

#### `tobit rep78 weight length, ll(0)`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | -0.0007198789353 | -0.0007198789353 | 0.0004311819354 | 0.0004311819354 |
| `length` | 0.008090822232 | 0.008090822232 | 0.01502867365 | 0.01502867365 |
| `Intercept` | 4.065071154 | 4.065071154 | 1.647842033 | 1.647842033 |

> Stata stores var(e.rep78) = 0.807583; our /sigma = 0.898656 and 0.898656^2 = 0.807583. Slopes match to 1e-15.

#### `sureg (price weight mpg) (mpg weight foreign)`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `price:weight` | 1.335250983 | 1.335250983 | 0.6263626215 | 0.6263626215 |
| `price:mpg` | -117.9644778 | -117.9644778 | 84.00853315 | 84.00853315 |
| `price:_cons` | 4645.845099 | 4645.845099 | 3509.1146 | 3509.1146 |
| `mpg:weight` | -0.006746266877 | -0.006746266877 | 0.0006230708757 | 0.0006230708757 |
| `mpg:foreign` | -2.101224951 | -2.101224951 | 1.049174135 | 1.049174135 |
| `mpg:_cons` | 42.29206513 | 42.29206513 | 2.116824119 | 2.116824119 |

> Stata repeats each name once per equation, so positions are used.

#### `sureg (price weight mpg) (mpg weight foreign), vce(robust)`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `price:weight` | 1.335250983 | 1.335250983 | 0.7599374481 | 0.7599374481 |
| `price:mpg` | -117.9644778 | -117.9644778 | 91.70326703 | 91.70326703 |
| `price:_cons` | 4645.845099 | 4645.845099 | 4053.055856 | 4053.055856 |
| `mpg:weight` | -0.006746266877 | -0.006746266877 | 0.0005313701289 | 0.0005313701289 |
| `mpg:foreign` | -2.101224951 | -2.101224951 | 1.092650401 | 1.092650401 |
| `mpg:_cons` | 42.29206513 | 42.29206513 | 1.746566117 | 1.746566117 |

#### `regress price weight i.foreign`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | 3.320736771 | 3.320736771 | 0.3958784154 | 0.3958784154 |
| `C(foreign)[T.1]` | 3637.001296 | 3637.001296 | 668.5829516 | 668.5829516 |
| `Intercept` | -4942.843954 | -4942.843954 | 1345.590935 | 1345.590935 |

> Stata labels the level `1.foreign`; the CSV stores 0/1 so our label is `C(foreign)[T.1]`. The estimates agree.

#### `poisson gear_ratio weight`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | -0.00014941959 | -0.00014941959 | 8.774112833e-05 | 8.774112833e-05 |
| `Intercept` | 1.54811402 | 1.54811402 | 0.2657715184 | 0.2657715184 |

#### `probit hi weight length`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | -0.0003137140304 | -0.0003137140304 | 0.0007026668139 | 0.0007026668139 |
| `length` | -0.057869104 | -0.05786910401 | 0.02627042341 | 0.02627042341 |
| `Intercept` | 11.8161567 | 11.8161567 | 3.249590634 | 3.249590634 |

#### `rreg price weight length, nolog`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | -0.6670513483 | -0.6670513483 | 0.5687771336 | 0.5687771336 |
| `length` | 37.72364884 | 37.72364884 | 19.85283308 | 19.85283308 |
| `Intercept` | -93.82767292 | -93.82767292 | 2183.280907 | 2183.280906 |

#### `etregress y_treat w, treat(treated = z1 z2) twostep`

Dataset: `sim` — 600 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `y:w` | 0.885450556 | 0.885450556 | 0.04264076704 | 0.04264076704 |
| `y:treated` | 1.359499616 | 1.359499616 | 0.1868684707 | 0.1868684707 |
| `y:Intercept` | 1.14523925 | 1.14523925 | 0.1243893358 | 0.1243893358 |
| `treat:z1` | 0.8247960006 | 0.8247960006 | 0.07556662354 | 0.07556662354 |
| `treat:z2` | -0.4380828094 | -0.4380828094 | 0.06418698055 | 0.06418698055 |
| `treat:Intercept` | 0.46625764 | 0.46625764 | 0.06124320345 | 0.06124320345 |

#### `nbreg count w, dispersion(constant)`

Dataset: `sim` — 600 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `w` | 0.595274193 | 0.595274193 | 0.03809570461 | 0.03809570461 |
| `Intercept` | 0.4996785731 | 0.4996785731 | 0.04863632927 | 0.04863632927 |

> NB1: Var = mu(1 + delta*mu). Both sides store lndelta.

#### `qreg price weight length, q(.5)`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | 1.231139381 | 1.231139381 | 1.535295882 | 1.535295881 |
| `length` | -8.129977876 | -8.129977876 | 53.58860451 | 53.5886045 |
| `Intercept` | 2982.599558 | 2982.599558 | 5893.313894 | 5893.313893 |

#### `nbreg count w`

Dataset: `sim` — 600 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `w` | 0.6634965543 | 0.6634965531 | 0.04272214976 | 0.04272214972 |
| `Intercept` | 0.4538139093 | 0.4538139112 | 0.04578444104 | 0.04578444101 |

> e(b) is on the lnalpha scale in both; alpha = exp(lnalpha).

#### `logit hi weight length`

Dataset: `auto` — 74 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `weight` | -0.0007593003302 | -0.0007593003307 | 0.001423994447 | 0.00142399446 |
| `length` | -0.1005478428 | -0.1005478447 | 0.05186548617 | 0.05186548673 |
| `Intercept` | 21.10931077 | 21.10931113 | 6.46264589 | 6.462645974 |

#### `heckman y_sel w, select(observed = z1 z2) twostep`

Dataset: `sim` — 600 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `y:w` | 1.159484099 | 1.159484099 | 0.05671698849 | 0.05671698849 |
| `y:Intercept` | 1.061586534 | 1.061586534 | 0.1089878615 | 0.1089878619 |
| `select:z1` | 0.8260395337 | 0.8260395337 | 0.07739636536 | 0.07739637075 |
| `select:z2` | -0.5280892434 | -0.5280892434 | 0.06525157616 | 0.06525157929 |
| `select:Intercept` | 0.3940601347 | 0.3940601347 | 0.06082757555 | 0.06082757746 |
| `lambda` | 0.631022215 | 0.631022215 | 0.1786437028 | 0.1786437035 |

> Stata reports a Mills-ratio coefficient and two `_cons`; positions are used.

#### `xtlogit binary x1 x2, fe`

Dataset: `panel` — 1600 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x1` | 0.9643481998 | 0.9643482478 | 0.1122940975 | 0.1122904893 |
| `x2` | -0.6155301523 | -0.6155301905 | 0.06679084262 | 0.0667899936 |

> Conditional logit: the intercept is conditioned out, so only slopes compare.

#### `xtpoisson count x1 x2, fe`

Dataset: `panel` — 1600 observations on the Stata side.

| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |
|---|---|---|---|---|
| `x1` | 0.4878093781 | 0.48780936 | 0.03574819113 | 0.03574611928 |
| `x2` | -0.2964989705 | -0.2964989666 | 0.02157539189 | 0.02157489697 |

> Conditional Poisson: only slopes compare.


### Why some observation counts differ

Some rows above show a different `N` on each side. None is a mistake, and each is one of four things.

| Command | Stata `e(N)` | westata `nobs` |
|---|---|---|
| `xtreg, be` | 1600 | 200 |
| `tobit` | 74 | 69 |
| `sureg` | 74 | 148 |
| `sureg vce(robust)` | 74 | 148 |
| `heckman twostep` | 600 | 354 |
| `xtlogit, fe` | 1600 | 1488 |
| `xtpoisson, fe` | 1600 | 1592 |

The four causes:

1. **What `e(N)` counts.** `sureg` is a system: Stata reports the *per-equation* sample size, westata the
   length of the stacked residual vector (equations x rows), which is what its result object validates
   `nobs` against. Two equations on 74 rows give 74 and 148.
2. **Missing values are dropped.** `tobit` on `rep78` loses the 5 observations where `rep78` is missing,
   so the estimation sample is 69, while Stata's `e(N)` still reports 74.
3. **Conditioning and collapsing.** `xtlogit, fe` and `xtpoisson, fe` condition out entities whose outcome
   never varies (1600 -> 1488 and 1592). `xtreg, be` collapses to one observation per entity, so its `N`
   is the number of GROUPS rather than rows (1600 -> 200). We report what the estimator actually used;
   Stata reports what entered.
4. **Two samples in one model.** `heckman` has a full sample of 600 and a selected sample of 354. Both
   numbers are defensible because they count different things. We report the full sample so that `nobs`
   describes the model's own data, and `e.N_selected` carries the other.

In every case the coefficients and standard errors agree to the precision shown in the table above, which
is the check that matters. The counts are listed because a silent 148-versus-74 would otherwise read as a
defect.

## Known deviations

Three kinds of difference exist in the tables above, and they are not the same thing.

### A finite-sample convention, not an error

**`logit hi weight, vce(robust)`**

| | standard error of the slope |
|---|---|
| Stata `vce(robust)` | 0.0009007456262 |
| westata `cov_type='HC1'` | 0.0009069792356 |
| westata `cov_type='HC0'` | 0.000894638807 |

`HC1` differs from Stata by 0.69%, `HC0` by 0.68%. The ratio of our `HC1` to Stata's is 1.0069204992, and `sqrt(n/(n-1))` is 1.0069204978: Stata's robust VCE for this command divides by `n-1`, while statsmodels-style `HC1` divides by `n-k`. Neither is wrong; they are different finite-sample adjustments. For `regress`, Stata uses `n/(n-k)` and our `HC1` matches it exactly — the divergence is specific to the likelihood-based commands.

### A scale difference between the two APIs

`tobit` reports the error scale differently: Stata stores `var(e.rep78)` in `e(b)`, while westata
exposes `/sigma`. The relationship is exact:

```
Stata  var(e.rep78) = 0.807583
westata       /sigma = 0.898656
0.898656**2          = 0.807583
```

So the slopes match to 1e-15 and the scale agrees as its square root. A table that compared the two
directly would show an 11% "difference" that is entirely a units question.

### Commands graded E1

Nine of the 87 registry entries measurably deviate on at least one invocation. The full reasoning is in
`docs/GAP_REGISTER.md`; the summary:

| Command | Worst measured | Why |
|---|---|---|
| `xtgls` | 3.4e+00 | Only when the cross-equation Sigma is rank-deficient (more equations than periods). No unique generalised inverse exists; Stata warns here too. Well-posed systems are at machine precision. |
| `teffects nnmatch` | 7.1e-02 | Standard error only — the point estimate is exact (3.6e-15). Relies on compiled Mata. |
| `frontier` | 4.3e-02 | `distribution(exponential)` only, where `sigma_u` is nearly unidentified on the test data. The likelihood at Stata's estimates matches exactly. |
| `bsqreg` | 3.0e-02 | Bootstrap VCE. Coefficients are exact (3.4e-16); a resampling estimator has not converged at any finite replicate count. |
| `permute` | 1.2e-02 | Stata's shuffle order is not reproducible from its output. |
| `arima` | 5.5e-03 | Optimiser stopping point; our likelihood is *higher* than Stata's. |
| `rreg` | 1.0e-03 | Final biweight case weights. |
| `xtlogit, pa` | 1.1e-03 | GEE working-correlation handling. |
| `heckman` | 1.9e-05 | Two-step standard errors. The ML path is a separate, exact estimate. |

One capability is **absent rather than approximate**: `teffects nnmatch, biasadj()` raises
`UnsupportedError`.

Several of these are provably irreducible rather than unfinished — a rank-deficient system has no
unique answer; some commands call compiled bytecode with no readable source; matching Stata's stopping
point would mean reporting a worse optimum; a bootstrap cannot match pointwise at any finite replicate
count.

## Reproducing every number here

```bash
# The datasets (auto.dta is Stata's own, the other two are generated with recorded seeds)
.venv/Scripts/python.exe docs/build_doc_data.py

# Re-run every comparison against live Stata and rewrite comparison.json
.venv/Scripts/python.exe docs/collect_comparison.py
```

Both scripts need a Stata licence. The test suite does not — it replays committed captures:

```bash
.venv/Scripts/python.exe -m pytest -q
```

`tests/test_readme_examples.py` executes the code blocks in the README; the same discipline applies
here, and the examples above are exercised by `docs/verify_guide.py`.
