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

Current counts: **54 E3, 26 E2, 15 E1** across 95
registry entries. The E1 entries are listed with their measured numbers in
[Known deviations](#known-deviations).

## Install

`westata` is on PyPI:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install westata
```

On Linux or macOS use `./.venv/bin/python` instead.

Python 3.10 or newer. `numpy`, `pandas`, `scipy`, `statsmodels`, `patsy`, `pyfixest`, `linearmodels`
and `pyyaml` install automatically — the package does not import without the last three.

`arch` is optional, needed only for GARCH and the Phillips–Perron test:

```bash
.venv/Scripts/python.exe -m pip install "westata[arch]"
```

To modify the library itself, clone it and install in editable mode with the test extra; the suite
then runs without a Stata licence, replaying the committed captures:

```bash
git clone https://github.com/bigchill-z/westata.git
cd westata
.venv/Scripts/python.exe -m pip install -e ".[test]"
.venv/Scripts/python.exe -m pytest -q
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
| `areg` | `Areg` (its own command; `absorb=` required, reports `_cons`) |
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

**Odds ratios: `logit ..., or`**

Stata's `or` reports exponentiated coefficients. In westata that is the `odds_ratio` keyword on
`summary`:

```python
print(ws.Logit.from_formula("foreign ~ mpg + weight", df).fit().summary(odds_ratio=True))
```

The option changes five things and **not** the inference:

| item | without `or` | with `or` |
|---|---|---|
| column header | `Coefficient` | `Odds ratio` |
| estimate | `b` | `exp(b)` |
| standard error | `se(b)` | `exp(b) * se(b)` |
| confidence interval | `[b ± z·se]` | exponentiated endpoints |
| footer | — | `Note: _cons estimates baseline odds.` |

`z` and `P>|z|` are **identical** either way: they come from the untransformed scale, so `or` is a
presentation change, not a re-test.

The raw values are also available as a property — note it is **not callable**:

```python
fitted = ws.Logit.from_formula("foreign ~ mpg + weight", df).fit()
fitted.odds_ratios          # pd.Series, no parentheses
```

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

### Which `cov_type` matches Stata's `, r`

The robust VCE scale factor **differs by command**, and the letter `r` does not identify it:

| Stata | westata |
|---|---|
| `regress ..., r` | `cov_type="HC1"` |
| `ivregress ..., r` | `cov_type="HC0"` |
| `logit ... , r` / `probit ..., r` | `cov_type="robust"` |

For an ML command, `robust` applies `N/(N-1)` — which is what Stata does — while `HC1` applies
statsmodels' `N/(N-K)` and inflates every standard error. On titanic with `[fweight=freq]`:

| call | `se[child]` |
|---|---|
| Stata `logit ..., r` | `0.2767452` |
| `cov_type="HC1"` | `0.2770602` |
| `cov_type="robust"` | `0.2767452` |

```python
# auto.csv carries `foreign` as an indicator, so this is a runnable ML example.
print(ws.Logit.from_formula("foreign ~ mpg + weight", df).fit(cov_type="robust").summary())
```

To choose rather than guess, compute `Stata_se / HC0_se` and square it: `N/(N-K)` means `HC1`,
`N/(N-1)` means `robust`, `1` means `HC0`.

A robust ML fit also **renames the log-likelihood**: Stata prints `Log pseudolikelihood`, because the
standard errors come from a quasi-ML sandwich rather than the information matrix.

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
| `teffects nnmatch ..., atet` / `caliper()` | `teffects_nnmatch(...)` — the function form adds `caliper` and `tie_tolerance` |
| `teffects psmatch ..., atet` | `teffects_psmatch(...)` |
| `tebalance summarize` | `tebalance_summarize(...)` |
| `didregress` / `xtdidregress` | `Didregress`, `Xtdidregress` |
| `rcm` | `rcm(...)` — pass `selected=` for a result that matches Stata |
| `synth` (default path) | `synth(...)` — six `e()` matrices, evidence grade E2 |

Both matching estimators use Stata's tie rule: every control at the k-th smallest
distance counts as a match, not just the first. That rule is what the numbers hinge
on — on the reference data 47 of 185 treated units have ties. The *standard errors*
are not reproduced (the AI-robust and sandwich forms live in compiled code), so
`se`/`z`/`p`/`ci` come back `NaN` rather than as a plausible guess.

```python
d = df.dropna(subset=["price", "mpg", "weight"]).copy()
d["treat"] = (d["foreign"] > 0).astype(int)
ws.teffects_nnmatch(d["price"], d["treat"], d[["mpg", "weight"]], nneighbor=2)
ws.tebalance_summarize(d["treat"], d[["mpg", "weight"]], names=["mpg", "weight"])
```

### Time series

| Stata | westata |
|---|---|
| `arima` | `ARIMA` |
| `arch` / `garch` | `GARCH` (needs `arch`) |
| unit-root tests | `unitroot(test=...)`, `XtUnitroot` |
| `vecrank` | `vecrank(...)` — `lags(1)` included |
| `varsoc` / `varlmar` / `varstable` / `varwle` | `varsoc`, `varlmar`, `varstable`, `varwle` |
| `veclmar` / `vecstable` | `veclmar`, `vecstable` |
| `vargranger` | `vargranger(fitted)` |
| `prais` | `Prais.from_formula(formula, data)` / `prais(frame, formula)` |
| `newey` | `Newey.from_formula(formula, data, lag=...)` / `newey(formula, data, lag=...)` |
| `synth` | `Synth.from_variables(data, unit=..., time=..., depvar=..., predictors=[...], trunit=..., trperiod=...)` |
| `synth2` | `Synth2.from_variables(...)`, same options |
| `rcm` | `Rcm.from_variables(outcome, data, predictors=[...], unit=..., time=..., treated_unit=..., treatment_period=...)` |
| `rdrobust` | `Rdrobust.from_variables(outcome, running, data, *, bandwidth=...)` |
| `teffects ra` | `TeffectsRa.from_variables(outcome, treatment, data, covariates=[...])` |
| `teffects psmatch` | `TeffectsPsmatch.from_variables(...)` |
| `teffects nnmatch` | `TeffectsNnmatch.from_variables(...)` |

```python
values = df.dropna(subset=["price", "mpg"])[["price", "mpg"]].to_numpy(dtype=float)
ws.vargranger(ws.VAR(values, lags=1, names=["price", "mpg"]).fit())
ws.vecrank(values, lags=1)
```
### `prais` and its `corc` option

`corc=True` is a **different table**, not only a different estimate: it drops the transformed
first observation, so it fits `N-1` rows and its total degrees of freedom are `N-2`.

| item | `prais` | `prais, corc` |
|---|---|---|
| banner | `Prais–Winsten AR(1) regression` | `Cochrane–Orcutt AR(1) regression` |
| `Number of obs` | N | **N-1** |
| `Total` df | N-1 | **N-2** |
| `Adj R-squared` | 0.5799 | **0.6068** |

Both variants list coefficients in the **written formula order** with `_cons` last, exactly as
Stata's `e(b)` does — `consumption ~ temp + price + income` prints `temp, price, income, _cons`.

```python
# The `corc` table was captured on icecream.dta; here we use the bundled
# auto.csv columns so this example runs anywhere.
ws.prais(df, "price ~ mpg + weight")               # Prais-Winsten
ws.prais(df, "price ~ mpg + weight", corc=True)    # Cochrane-Orcutt
```

`prais` and `newey` are estimation commands, so they also have the class form every other
estimator uses (R2.7). The function stays available and returns the same numbers:

```python
ws.Prais.from_formula("price ~ mpg + weight", df).fit()
ws.Newey.from_formula("price ~ mpg + weight", df, lag=2).fit()
```

> **Argument order, and why `fit()` has no `cov_type`.** `Prais(data, formula)` keeps the
> function's positional order, which is the reverse of every other `from_formula` — 52 call
> sites pass `ws.prais(frame, FORMULA)` positionally, and swapping the two would not raise,
> it would try to parse the frame as a formula. Use `Prais.from_formula(formula, data)` for
> the standard order. Neither `fit()` accepts `cov_type`: Stata's `prais` and `newey` have no
> `vce()` option, so the variance is part of the estimator, and offering a `cov_type` would
> promise a choice that cannot be honoured.


### Post-estimation

| Stata | westata |
|---|---|
| `test` | `res.test(...)` |
| `lincom` | `res.lincom(...)` |
| `margins` | `res.margins(...)` |
| `linktest` | `ws.linktest(res)` |
| `suest` | `ws.suest([fit_a, fit_b])` |
| `vargranger` | `ws.vargranger(var_fit)` |
| `summarize` | `ws.summarize(frame, ["y", "x1"])` |
| `ereturn list` | `format_ereturn_list(res)` |
| `predict` | `res.predict(...)` |
| `esttab` / `outreg2` | `ws.etable([...])` |
| `xtserial` | `ws.xtserial(data, "y ~ x1 + x2", entity=..., time=...)` |

### `xtserial`: the Wooldridge panel serial-correlation test

`xtserial` is **not a Stata built-in and is not on SSC**; it is Drukker's package:

```stata
net install xtserial, from("http://www.stata.com/users/ddrukker")
```

Its body is four lines: difference the response **and every regressor** (no constant,
clustered on the panel), regress those residuals on their own lag the same way, then
test whether the lag coefficient equals **-0.5**. The null is -0.5 rather than 0
because differencing an AR(1) error leaves `rho - 1`, which is -0.5 when there is no
serial correlation; testing against zero would reject on any data.

```python
# The numbers above come from `lin_1992.dta` (province x t); the guide's `df` is
# auto.csv, so the example builds a synthetic panel from it. The behaviour is the same.
panel_like = df.assign(
    province=(np.arange(len(df)) % 12) + 1,
    t=(np.arange(len(df)) // 12) + 1,
)
print(ws.xtserial(panel_like, "price ~ mpg + weight", entity="province", time="t"))
```

**The differences run on the (panel, time) axis, not the row axis.** When a period is
missing, the difference is missing on BOTH sides of the gap. Every panel of
`lin_1992.dta` misses `t == 11`, so the sample is 476 - 56 = **420**; pairing rows
instead keeps 474 and moves `r(corr)` from -0.19258 to -0.20871. Measured against
StataNow 19.5: `corr` within 2.7e-10, `F` within 8.4e-09, `p` within 4.5e-12, with the
printed block byte-identical.

Stata's `, output` prints the first-stage differenced regression; that is not
reproduced, since it is an ordinary clustered `ws.OLS` fit on the differenced data.

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

### `estat bgodfrey` and its `nomiss0` option

Breusch-Godfrey differs from `estat dwatson` in one respect worth knowing before reading the
output: the lagged residuals' leading missing values are **replaced by zero by default**, so
the auxiliary regression keeps the full sample. `nomiss0` leaves them missing instead, and the
auxiliary regression then loses `p` rows.

The statistic is `e(N) * e(r2)` where `e(N)` is the **auxiliary** regression's count, so the
option changes the statistic itself, not merely its degrees of freedom:

| lag | default | `nomiss0` |
|---|---|---|
| 1 | 4.237 | 4.704 |
| 2 | 4.487 | 5.111 |
| 3 | 4.775 | 5.053 |
| 4 | 5.099 | 5.679 |

```python
# The table above was captured from `regress consumption temp price income` on
# icecream.dta; here we use the bundled auto.csv so the example runs anywhere.
fit = ws.OLS.from_formula("price ~ mpg + weight", df).fit()

print(ws.estat_bgodfrey(fit, lags=(1, 2)))                 # zero-fill, full sample
print(ws.estat_bgodfrey(fit, lags=(1, 2), nomiss0=True))   # keep them missing
```

`bgodfrey.ado` refuses to run after `regress, robust`, and so does this function.
`vce(cluster)` is `e(vcetype) = Robust` in Stata's eyes too, so it is refused as well.

### The `tsset` equivalent: `time=`

In Stata, `estat bgodfrey` / `estat dwatson` / `wntestq` / `corrgram` all fetch the time
variable with `_ts ..., sort onepanel` and then build their lags with `tsrevar l(...)` or `L.`.
**`L.` is defined on the time axis**, so when a period is missing from the sample that row's
lag is MISSING rather than the previous row -- joining the two sides of a gap produces a number
that looks plausible and is not Stata's.

This API has no `tsset` step: row order IS time order. When the sample has gaps, hand over the
axis with `time=`:

```python
# A period column plus one deliberately missing period: without `time=` the lag pairs the
# rows on either side of the gap, which is NOT what `L.` does.
time_series = df.assign(year=range(1, len(df) + 1)).drop(index=10).reset_index(drop=True)
gapped = ws.OLS.from_formula("price ~ mpg + weight", time_series).fit()

ws.estat_bgodfrey(gapped, lags=1)              # lags by row order
ws.estat_bgodfrey(gapped, lags=1, time="year")  # lags on `year`; missing across the gap
```

`time=` takes a **column name** or an **array of period values the length of the estimation
sample**; `delta=` is `tsset`'s `Delta:` (default 1, must be a positive integer -- `tsset`
itself refuses fractional periods).

Measured on `icecream.dta` with `time == 5` dropped and the model re-fitted (n = 29):

| command | by row order | `time=` (= Stata) |
|---|---|---|
| `estat bgodfrey, lags(1)` | 3.286 | **4.220** |
| `estat dwatson` | 1.0868969 | **.8930395** |
| `wntestq e1, lags(4)` | 4.3302 | **3.9550** |

With gaps Stata announces them, and this package reproduces the wording exactly:

* `estat bgodfrey` / `estat dwatson`: `Number of gaps in sample = 1` (then a blank line)
* `wntestq` / `corrgram`: `(note: time series has 1 gap)` (`gaps` in the plural)

**The default lag order is `max(1, min(int(n/2) - 2, 40))`** (`ac.ado:71`,
`corrgram.ado:18`). It does not depend on how many parameters the previous regression
estimated, and it is **capped at 40**: at n = 758 Stata reports `chi2(40)`, not the 377 an
`(n-k)//2` rule would give.

Panel data is refused outright: `_ts ..., onepanel` answers `sample may not include multiple
panels` (r(459)), and `areg` / `reghdfe` / `newey` / `xtreg` fits answer
`estat bgodfrey not valid` (r(321)).

### `estat ovtest` and its `rhs` option

The default tests powers of the **fitted values**; `rhs=True` tests powers of the **regressors**.
Two details decide the numbers and neither is guessable from the output:

* **Only continuous regressors contribute**, three powers each (`x^2, x^3, x^4` after rescaling to
  `[0, 1]`). Indicator columns are skipped entirely. A model with no continuous regressor at all is
  refused, which is what Stata's "powers of all explanatory variables collinear" means.
* **The powers enter the auxiliary model BEFORE your own regressors**, so a variable of yours that
  duplicates a power is the one omitted. On `grilic.dta` with `expr2 = expr**2`:

  | | F | p |
  |---|---|---|
  | Stata | `F(11, 741) = 1.73` | `0.0626` |
  | without that ordering | `F(10, 741) = 1.27` | `0.2421` |

  The reported `df` counts the surviving **powers**, not the parameters of the auxiliary regression.

```python
# `lnw`/`expr2` belong to grilic.dta; the guide's `df` is auto.csv, so the example uses
# continuous columns from it. The behaviour is the same either way.
fit = ws.OLS.from_formula("price ~ mpg + weight", df).fit()
print(ws.estat_ovtest(fit))            # powers of the fitted values
print(ws.estat_ovtest(fit, rhs=True))  # powers of the regressors
```

### Explanatory text Stata prints around a table

Stata's output is more than the numbers: it prints notes, warnings and iteration logs that explain
what happened to the DATA. The package reproduces the ones that are deterministic, and deliberately
does not fabricate the ones that belong to Stata's own optimiser.

| Where | Text | Reproduced |
|---|---|---|
| `estat ovtest, rhs` | `note: <name> omitted because of collinearity.` | yes, in Stata's order |
| `prais` / `prais, corc` | `Iteration k:  rho = 0.0000` ... (full log) | yes, byte-for-byte |
| `xtlogit, fe` | `note: multiple positive outcomes within groups encountered.` + the wrapped group line | yes |
| `xtpoisson, fe` / `xtnbreg, fe` | `note: N group(s) (N obs) dropped because of all zero outcomes` | yes; the wording is command-specific |
| `poisson` / `nbreg` on a non-integer response | `note: noncount dependent variable encountered; ...` / `note: you are responsible for interpretation of non-count dep. variable` | yes |
| `logit` / `probit` | `Iteration 0:  Log likelihood = -45.03321` | yes, the null-model value, exactly |
| any ML estimator whose optimiser did not converge | `Warning: Convergence not achieved.` (last line) | yes |

**Two deliberate omissions.** The ML *intermediate* log-likelihoods come from Stata's `moptimize`
path and cannot be reproduced by a different optimiser; printing our own would look like Stata while
disagreeing on every line. Only `Iteration 0` and the header's exact `Log likelihood = ...` row are
shown. The `prais` / `xtpcse` note about time gaps requires a `tsset` time variable, which those
two estimators' signatures do not take (the serial-correlation commands above do, through
`time=`), so it is not guessed.

The full audit — 9 rounds, 57+ live captures, a scan of 3962 `.ado` files — is in `NOTES_AUDIT.md`.

### `hausman` and its `sigmamore` / `sigmaless` options

`ws.hausman(consistent, efficient, ...)` reproduces Stata's `hausman`. The two sigma options are
sometimes needed when the estimators' VCEs are not directly comparable:

```python
iv = ws.IV2SLS.from_formula("price ~ weight + [mpg ~ length]", df).fit()
ols = ws.OLS.from_formula("price ~ weight + mpg", df).fit()

print(ws.hausman(iv, ols).statistic)                    # Stata: hausman iv ols
print(ws.hausman(iv, ols, constant=True).statistic)     # hausman iv ols, constant
print(ws.hausman(iv, ols, sigmamore=True).statistic)    # hausman iv ols, sigmamore
```

They rescale one VCE **before** differencing — the consistent side for `sigmamore`, the efficient side
for `sigmaless` — by the squared ratio of the `s2` values, where `s2` is `e(sigma)` falling back to
**`e(rmse)` itself** rather than its square.

**The reported `df` is the rank of the differenced matrix, not the number of coefficients.** A rank
that falls short also adds Stata's five-line `Note:` at the top, and a difference that is not positive
definite ends with `(V_b-V_B is not positive definite)`. Measured on grilic with `reg` plus
`ivregress 2sls`, then `hausman iv ols`:

| options | chi2 | df |
|---|---|---|
| default | 3.422654895 | 6 |
| `constant` | 3.422654895 | 7 |
| `sigmamore` | 3.843794583 | 1 |
| `sigmamore constant` | 3.843794583 | 1 |
| `sigmaless` | 3.437051055 | 1 |

Column headings come from the `estimates store` names in Stata. Set `estimation_label` on each result
to match:

```python
# set on the two results before calling hausman:
iv.estimation_label, ols.estimation_label = "iv", "ols"
```

### `estimates table` — several models side by side

`ws.render_estimates_table` reproduces Stata's `estimates table m1 m2 ..., b se`. It is a **different
layout** from `esttab` (no column numbers, no dependent-variable row, a `Legend:` line at the end) and
it accepts any number of models.

```python
fit_a = ws.OLS.from_formula("price ~ mpg", df).fit(cov_type="HC1")
fit_b = ws.OLS.from_formula("price ~ mpg + weight", df).fit(cov_type="HC1")
print(ws.render_estimates_table([fit_a, fit_b], model_names=["a", "b"], se=True))
```

The geometry is fixed and derived: a 14-character gutter, then 13 columns per model; names are
centred and abbreviated past ten characters; values use eight significant figures at or above 1 and
eight decimals below.

**The robust VCE differs between commands.** `regress ..., r` is HC1 while `ivregress ..., r` is HC0,
so a table mixing OLS and IV models needs `cov_type="HC1"` on the OLS fits and `"HC0"` on the IV ones.
Using HC0 throughout scales every OLS standard error by `sqrt(n/(n-k))`.


#### Adding standard errors, p-values or stars

`se`, `pvalues` and `star` select which rows and marks appear; the coefficient `b` is always printed.

```python
print(ws.render_estimates_table([fit_a, fit_b], model_names=["a", "b"], se=True, pvalues=True))
# Stata: estimates table ..., b se p

print(ws.render_estimates_table([fit_a, fit_b], model_names=["a", "b"], se=False, pvalues=True))
# Stata: estimates table ..., b p   -- identical, because `b` is always printed
```

The p row uses **four decimal places with the leading zero** (`0.0000`, `0.0013`), unlike the
coefficient and standard-error cells, which drop the leading zero and trim. It always follows the
standard error, and the legend names the rows printed: `Legend: b/se`, `b/p` or `b/se/p`.

`pvalues` and `star` are mutually exclusive — Stata rejects `..., b se p star(...)`, and so does this
renderer.

### `estimates table`, `esttab` and their options

`ws.render_estimates_table` serves both Stata layouts from one entry point. They are **different
tables**, not a flag on one:

```python
fits = [
    ws.OLS.from_formula("price ~ mpg", df).fit(cov_type="HC1"),
    ws.OLS.from_formula("price ~ mpg + weight", df).fit(cov_type="HC1"),
    ws.IV2SLS.from_formula("price ~ weight + [mpg ~ length]", df).fit(cov_type="HC0"),
]
names = ["ols_a", "ols_b", "tsls"]

print(ws.render_estimates_table(fits, model_names=names, se=True))
# Stata: estimates table ols_a ols_b tsls, b se

print(ws.render_estimates_table(fits, model_names=names, star=(0.1, 0.05, 0.01)))
# Stata: estimates table ..., star(0.1 0.05 0.01)

print(ws.render_estimates_table(
    fits, model_names=names, se=True, star=(0.1, 0.05, 0.01),
    mtitle=True, stats=["N", "r2"],
))
# Stata: esttab ..., se r2 mtitle star(* 0.1 ** 0.05 *** 0.01)
```

| | `estimates table` | `star(...)` | `mtitle=True` (esttab) |
|---|---|---|---|
| rule | `14 + 13n` | `14 + 16n` | `12 + 16n` |
| label gutter | `    Variable \| ` | same | none |
| standard-error row | yes | no | yes |
| value format | 8 significant / 8 decimals | same | 3 significant (4 at or above 1) |
| footer | `Legend: b/se` | `Legend: * p<.1; ...` | `N` / `R-sq` rows + two notes |

**`star()` changes the column width**, and the stars hang *after* a right-aligned number, so a
starred coefficient reaches further right than the error cell beneath it.

`stats` accepts `N`, `r2`, `r2_a` and `rmse`.

#### `esttab` options for the screen table

Passing `mtitle=True` selects the `esttab` layout, and the rows can be driven by the other esttab
cells. For the full set, build an `EsttabOptions` and call the renderer directly:

```python
from westata.results.esttab_options import EsttabOptions
from westata.results.esttab_render import render_esttab_table

options = EsttabOptions(
    second_row="p",          # one of se / t / p / ci; `b` alone still prints t
    stars=True,
    star_levels=(0.05, 0.01, 0.001),   # esttab's default; NOT estimates table's 0.1/0.05/0.01
    keep=("s", "expr"),
    order=("expr", "s"),
    rename={"s": "Schooling"},
    stats=("r2",),           # N always prints; this ADDS to it
)
print(render_esttab_table(fits, names=names, options=options))
```

Five measured behaviours that surprise people:

* `b` **alone still prints the t statistic** — leaving out `se` swaps the second row, it does not
  remove it.
* `b t p` prints **only `p`**: esttab keeps a single second row and the last option wins. `b se p` is
  an error.
* `star(...)` must pair a symbol with a level — `star(* 0.1 ** 0.05 *** 0.01)` works, a bare
  `star(0.1 0.05 0.01)` exits `r(198)`.
* The **column width is derived**: 16 with stars, 13 without, 26 for `ci`.
* `varlabels` **suppresses the blank lines** between coefficient blocks (21 lines become 18).

`notype`, `nomargin` and `margin` are not screen-table options here; see `ESTTAB_SPEC.md` for why.

### Byte-for-byte table layout

The section above compares coefficient VALUES. A separate property is the printed
table itself: column positions, alignment, significant figures and separators. Those
were verified by capturing live Stata output and comparing it character by character,
at several sample sizes rather than one.

Every table below now matches a live capture exactly, at every size tested:

| Table | Sizes verified |
|---|---|
| `varsoc` | 100, 200, 300, 900, 1500 |
| `varstable` | 60, 200, 900, 1500 |
| `vecstable` | 100, 300, 900 |
| `dfuller` | 4 sizes x 2 lag counts |
| `varlmar`, `varwle`, `veclmar`, `vargranger` | 100, 300, 900 |
| `estat ic` (OLS, absorbed, VAR, zip, zinb) | 2-3 sizes each |
| `estat summarize` | 200, 1500, 3000, and scales 1e-5..1e3 |
| `estat overid`, `estat ovtest`, `estat dwatson` | 2-3 sizes each |
| `logit`, `probit`, `poisson`, `nbreg` headers | 800, 1500 |
| `zip`, `zinb` headers and equation blocks | 300, 1000 |
| `irf_table`, `fevd_table`, `esttab`, `summarize` | 7-39 lines each |

**Testing one sample size is not enough.** Nine of the defects fixed in this release
were invisible on the sample each table was originally built from: a short value in a
right-aligned field, a statistic below one that loses its leading zero, a count that
grows to four digits and gains a separator, a degrees-of-freedom count that widens the
row above it. Each needed a different n or a different scale to surface.

### Remaining known differences

Six differences are numerical rather than presentational and are recorded rather than
papered over:

* `estat vif`'s 1/VIF in the ninth significant figure;
* `varstable`'s eigenvalues at n >= 300, ninth figure, which the modulus column inherits;
* VECM coefficients in the last printed digit;
* `zip`/`zinb` coefficients in the seventh significant figure.

No format string can close a gap that begins in the estimate itself.


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

Some rows above show a different `N` on each side. None is a mistake, and each is one of three
things:

| Command | Stata `e(N)` | westata `nobs` |
|---|---|---|
| `xtreg, be` | 1600 | 200 |
| `tobit` | 74 | 69 |
| `sureg` | 74 | 148 |
| `sureg vce(robust)` | 74 | 148 |
| `heckman twostep` | 600 | 354 |
| `xtlogit, fe` | 1600 | 1488 |
| `xtpoisson, fe` | 1600 | 1592 |

1. **What `e(N)` counts.** `sureg` is a system: Stata reports the *per-equation* sample size,
   westata the length of the stacked residual vector (equations x rows), which is what its result
   object validates `nobs` against. Two equations on 74 rows give 74 and 148.
2. **Missing values are dropped.** `tobit` on `rep78` loses the 5 observations where `rep78` is
   missing, so the estimation sample is 69, while Stata's `e(N)` still reports 74.
3. **Conditioning and collapsing.** `xtlogit, fe` and `xtpoisson, fe` condition out entities whose
   outcome never varies; `xtreg, be` collapses to one observation per entity. We report the rows
   that survived, Stata the rows that entered.

In every case the coefficients and standard errors agree to the precision shown in the table
above, which is the check that matters. The counts are listed because a silent 148-versus-74
would read as a defect.

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

15 of the 93 registry entries measurably deviate on at least one
invocation. The full reasoning is in `docs/GAP_REGISTER.md`; the summary:

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
