# westata

Stata-conformant econometrics with a statsmodels-style API.

> **Read this before relying on a number.** `westata` is a personal research
> library at version 0.0.1. 52 of its 87 commands match Stata to machine
> precision, but **9 do not**, and one feature is missing. The measured
> deviation for each is listed below. They are graded, not hidden, and the
> grading is reproducible from this repository.
>
> | Command | Worst measured relative difference | Why |
> |---|---|---|
> | `xtgls` | 3.4e+00 | **Only when the cross-equation Sigma is rank-deficient** (more equations than periods). No unique generalised inverse exists; Stata warns here too. On well-posed systems this is machine precision. |
> | `teffects nnmatch` | 7.1e-02 | **Standard error only** — the point estimate is exact (3.6e-15). Uses compiled Mata internally. |
> | `frontier` | 4.3e-02 | `distribution(exponential)` only, where `sigma_u` is nearly unidentified on the test data. The likelihood at Stata's estimates matches exactly. |
> | `bsqreg` | 3.0e-02 | Bootstrap VCE. Coefficients are exact (3.4e-16); a resampling estimator has not converged at any finite replicate count. |
> | `permute` | 1.2e-02 | Stata's shuffle order for the permutation draws is not reproducible from its output. |
> | `arima` | 5.5e-03 | Optimiser stopping point; our likelihood is *higher* than Stata's. |
> | `rreg` | 1.0e-03 | Final biweight case weights. |
> | `xtlogit, pa` | 1.1e-03 | GEE working-correlation handling. |
> | `heckman` | 1.9e-05 | Two-step standard errors. The ML path added later is a separate, exact estimate. |
>
> One capability is **absent rather than approximate**: `teffects nnmatch,
> biasadj()` raises `UnsupportedError`.
>
> Each figure is the worst case across the invocations an independent
> 386-point sweep exercised against live Stata. Most of the library is far
> better than these bounds — 250 of 386 comparison points are at machine
> precision — but these are the numbers that would bite you, so they come
> first.

`westata` is a personal research library that answers one question: *if I run
this in Python, do I get the same numbers Stata would print?*

- **API follows statsmodels** — `Model.from_formula(...).fit()`, `params`,
  `bse`, `pvalues`, `conf_int()`, `cov_params()`. Results are duck-typed, so
  existing statsmodels-style downstream code keeps working.
- **Numerics follow Stata** — small-sample corrections, weight semantics,
  factor-variable coding, degrees of freedom, and degrees-of-freedom splits are
  implemented per the corresponding Stata manual entry.
- **Claims are auditable** — every command carries an evidence level in
  `tests/conformance/registry.yml`. Nothing claims more than it can show.

## Status

| | |
|---|---|
| Tests | **1448 passing** |
| Commands covered | 87 registry entries |
| Evidence | **52 × E3**, 26 × E2, 9 × E1 |
| Independent sweep | 386 comparison points against live Stata (`retest/`) |

Verified against **StataNow 19.5 (MP)** through `pystata`; the captures are
committed so the comparison replays without a licence.

## Install

`westata` is on PyPI:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install westata
```

On Linux or macOS the second line is `./.venv/bin/python -m pip install westata`.

Requires Python ≥ 3.10. `numpy`, `pandas`, `scipy`, `statsmodels`, `patsy`,
`pyfixest` (HDFE), `linearmodels` (IV, panel) and `pyyaml` come in
automatically — the package does not import without the last three.

`arch` is optional and only needed for GARCH, EGARCH and the Phillips–Perron
test:

```bash
.venv/Scripts/python.exe -m pip install "westata[arch]"
```

Without it those commands raise an `UnsupportedError` naming the missing extra,
and their tests skip rather than fail.

### Working on the library itself

The released version is what `pip install westata` gives you. To work on the
source, clone and install it in editable mode with the test dependencies:

```bash
git clone https://github.com/bigchill-z/westata.git
cd westata
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[test]"
```

`python -m pytest -q` then runs the 1455-test suite. It needs no Stata licence:
the conformance captures are committed and replayed. Re-running the comparisons
against live Stata (`retest/`, `docs/collect_comparison.py`) does need one.

## Quick start

Every line below is executed as written against `tests/data/auto.dta`.

```python
import pandas as pd
import westata as ws

df = pd.read_stata("tests/data/auto.dta")

# OLS with a factor variable and clustered standard errors (regress)
res = ws.OLS.from_formula("mpg ~ weight + C(foreign)", df).fit(
    cov_type="cluster", cov_kwds={"groups": df["foreign"]},
)
print(res.summary(style="stata"))          # Stata layout
```

Coefficient names keep the formula's spelling, so a factor level appears as
`C(foreign)[T.Foreign]`. Use that exact string:

```python
res.test("weight")                                  # single restriction
res.test("weight", "C(foreign)[T.Foreign]")         # joint test of both
res.test("weight = 0")                              # same, as an equation
res.lincom("weight + 1000*C(foreign)[T.Foreign]")   # linear combination
```

`test` takes one specification per argument; a comma inside a single string is not a separator.

Other models, on the same data:

```python
# Quantile regression (qreg) — median
ws.QuantileRegression.from_formula("mpg ~ weight + C(foreign)", df).fit()

# Robust regression (rreg)
ws.RobustRegression.from_formula("mpg ~ weight", df).fit()

# Systems of equations (sureg): one list entry per equation
ws.Sureg.from_formula(["price ~ weight + mpg", "mpg ~ weight + foreign"], df).fit()
ws.Sureg.from_formula(
    ["price ~ weight + mpg", "mpg ~ weight + foreign"], df
).fit(cov_type="robust")                     # robust VCE

# Censored regression (tobit). `lower` is a from_formula argument, not a fit one.
# `rep78` has missing values, which are dropped.
ws.Tobit.from_formula("rep78 ~ weight + mpg", df, lower=0).fit()
```

A publication-style table:

```python
base = ws.OLS.from_formula("mpg ~ weight", df).fit()
full = ws.OLS.from_formula("mpg ~ weight + C(foreign)", df).fit()
print(ws.etable([base, full], model_names=["Base", "Full"]))
```

## Command coverage

| Family | westata | Stata |
|---|---|---|
| Linear | `OLS` | `regress` (4 weight types, `i.` factors) |
| IV | `IV2SLS` / `IVLIML` / `IVGMM` | `ivregress 2sls/liml/gmm` |
| HDFE | `FEOLS` | `reghdfe` / `areg` |
| Panel | `PanelOLS` / `RandomEffects` / `BetweenOLS` / `FirstDifferenceOLS` | `xtreg fe/re/be/fd` |
| Binary | `Logit` / `Probit` | `logit` / `probit` |
| Count | `Poisson` / `NegativeBinomial` | `poisson` / `nbreg` |
| Ordered | `OrderedLogit` / `OrderedProbit` | `ologit` / `oprobit` |
| Censored | `Tobit` (self-implemented) | `tobit` |
| Time series | `ARIMA` / `VAR` / `VECM` / `GARCH` / `unitroot` | `arima` / `var` / `vec` / `arch` / `dfuller` |
| Postestimation | `test` / `lincom` / `margins` / `predict` | same names |
| Reporting | `etable` | `esttab` |

## Weight semantics

Following Stata `[U] 20.24`:

| Type | Point estimate | Variance | Effective N |
|---|---|---|---|
| `fweight` | literal replication | conventional | `sum(w)` |
| `aweight` | normalized to `sum(w) = N` | conventional | `N` |
| `pweight` | same as aweight | **robust (HC1)** | `N` |
| `iweight` | not normalized | conventional | `sum(w)` |

`weight_type` is mandatory whenever weights are supplied — westata never
guesses, because guessing silently changes the VCE.

## Evidence levels

| Level | Meaning | Count |
|---|---|---|
| E3 | Agrees with full-precision `e(b)`/`e(V)` from real Stata, on every invocation the independent sweep exercised | 52 |
| E2 | Approximate, or exact only on a subset of invocations: real but bounded differences | 26 |
| E1 | Materially differs on at least one measured invocation, or not covered by the sweep | 9 |

**These levels come from the independent 386-point sweep, not from the library's own
tests.** That distinction matters: the library's tests passed at 1247 while the sweep
found 24 defects, because a test written from the same understanding as the code cannot
detect a misunderstanding shared by both.

Measured relative agreement with StataNow 19.5:

| Command | Parameters | Standard errors |
|---|---|---|
| `regress` | 5.1e-15 | 6.8e-16 |
| `regress, vce(robust)` | 5.1e-15 | 4.6e-15 |
| `regress, cluster` | 5.1e-15 | 7.1e-15 |
| `regress, i.factor` | 3.7e-14 | 5.1e-16 |
| `ivregress 2sls` | 7.8e-15 | 7.5e-15 |
| `ivregress liml` | 3.7e-16 | 3.7e-16 |
| `xtreg, fe` | 1.8e-16 | 1.9e-16 |
| `xtreg, re` | 6.8e-16 | 1.0e-15 |
| `xtreg, be` | 3.6e-16 | 1.7e-16 |
| `xtreg, fd` | not comparable (command removed) | — |
| `areg` | 4.6e-16 | 1.8e-16 |
| `reghdfe` (1 FE) | 4.5e-16 | 3.3e-16 |
| `reghdfe` (2 FE) | 4.4e-16 | 1.9e-16 |
| `reghdfe` (2 FE, cluster) | 4.4e-16 | 2.0e-16 |
| `var` | 2.9e-15 | — |
| `poisson` | 3.9e-13 | 5.8e-14 |
| `logit` | ~1e-7 | ~1e-7 |
| `probit` | 1.5e-7 | — |
| `tobit` | 8.5e-8 | 6.4e-7 |
| `margins, dydx` | 1.6e-7 | 5.2e-6 |
| `ologit` / `oprobit` | ~2e-5 | — |
| `nbreg` | 1.7e-5 | 9.1e-7 |
| `arima` | 4.8e-5 | 8.2e-5 |
| `predict` (OLS xb) | 1.6e-15 | — |
| `predict` (logit pr) | 3.1e-11 | — |
| `predict` (tobit e/ystar) | 5.8e-10 | — |
| `etable` vs `esttab` | 0.0 | — |
| `vecrank` (all lag orders) | 0.0 (4 dp) | — |
| `dfuller` | 2.3e-4 | — |
| `ivregress gmm` | 4.9e-16 | 6.9e-18 |
| `arch` | 1.4e-04 | z values not comparable |

The linear, panel-linear, multi-equation and complex-survey families agree to
machine precision (1e-15). **Non-linear maximum-likelihood models are where the
differences live**, and almost always in the second-order quantities rather than the
coefficients: an exact point estimate with a standard error that is 5% off still gives a
wrong z, a wrong p, and a wrong confidence interval.

What was measured wrong, and then fixed:

- `ologit`/`oprobit` `e(V)`: a numpy view alias let `exp()` act on the cut points, and the
  slope-by-cut block was never transformed — `e(V)` off by 20x, all cut-point z and CI wrong.
- `nl` covariance: `pinv` truncation made standard errors wrong by **6–7 orders of
  magnitude**, so every t statistic was inflated.
- `xtreg, fe vce(robust)` was mapped to HC1; Stata defines it as `vce(cluster panelvar)`
  (verified: the two are identical in Stata, `max|V_robust − V_cluster| = 0`). The constant's
  SE was 18x off.
- `tobit` `se(/sigma)` lacked the delta-method Jacobian; `margins` used a fixed numerical
  step; `podiscrete` VCEs silently accepted and ignored `vce(robust)`.

What remains approximate is listed per command in
[`docs/USING_WITH_STATA.md`](docs/USING_WITH_STATA.md), which is organised around the
question a user actually has: **can I quote this number as Stata's?**

Two commands could not be compared at all because StataNow 19 removed them:
`xtreg, fd` (r(198)) and `kpss` (r(199)). Their westata counterparts are
validated by other means.

## Design notes worth knowing

Things that bit during development, documented so they do not bite again:

- **`df_resid` vs `e(df_r)` are different quantities.** Under clustering, t
  tests use `G − 1` while Stata's `e(df_r)` stays `N − K`. Conflating them
  silently gives wrong p-values.
- **Frequency weights must equal the expanded-data fit.** Stata's own stated
  invariant. Getting the robust meat's weight form wrong inflates fweighted
  standard errors by up to 51%.
- **`margins` differentiates the response, not the linear index.** For a logit,
  `mean(p(1-p))·b` — differentiating `xb` returns the coefficient, which was
  4.38× too large in the first implementation.
- **`PanelOLS.from_formula` ignores `entity_effects`.** Passing it through
  `fit()` silently produces pooled OLS. The model must be constructed directly.
- **linearmodels' clustered VCE needs `auto_df=False, group_debias=True`** to
  reproduce Stata's `G/(G-1)·(N-1)/(N-K)`; the default is ~11% off.
- **`ivregress` reports z / Wald chi2**, not t / F. OLS and panel use t.

## Layout

```
src/westata/
  data/            weight normalization, panel/time-series metadata
  formula/         patsy design matrices, C(x, base=)
  vce/             HC0-HC3, CR1, ssc
  estimators/      linear, iv, hdfe, panel, discrete, tobit, tseries
  results/         StataResults, E(RETURN), Stata/statsmodels summaries
  postestimation/  test, lincom, margins, predict
  report/          etable, LaTeX/Markdown renderers
  conformance/     evidence registry, tolerances, invariants
tests/conformance/ registry.yml + published-output fixtures
docs/              design spec and implementation plans
```

## Documentation

- **Comparing westata with Stata**: [`docs/USING_WITH_STATA.md`](docs/USING_WITH_STATA.md) —
  which quantities are interchangeable and which are not, for the four E2 commands.
- Evidence registry: [`tests/conformance/registry.yml`](tests/conformance/registry.yml) —
  per-command level, measured gap, and reasoning.
- Known gaps: [`docs/GAP_REGISTER.md`](docs/GAP_REGISTER.md) — what is deliberately
  not implemented, and why.
- **Independent sweep**: [retest/README.md](retest/README.md) — the 386-point
  check against live Stata, and how to re-run it. This is where the evidence
  levels come from.
- Design spec: [`docs/superpowers/specs/2026-10-22-westata-design.md`](docs/superpowers/specs/2026-10-22-westata-design.md)

## Known gaps

Not implemented, and they raise instead of approximating: Mata, the graphics
system, data management, SEM/GSEM, survival analysis, Bayesian estimation, `mi`,
`xtabond2`, `csdid`, `rdrobust`, `synth`, general `gmm`, `nlsur`, multi-way
clustering (CGM), `vecrank`'s fixed-N exact critical values, and Stata's
arbitrary `r()` statistics outside the `[U] exp_list` grammar.

`docs/GAP_REGISTER.md` lists every one with its reason. Those are boundary
statements, not a to-do list.
