# westata

Stata-conformant econometrics with a statsmodels-style API.

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
| Tests | **1247 passing** |
| Commands covered | 87 registry entries |
| Evidence | **83 × E3**, 4 × E2, **0 × E1** |

Verified against **StataNow 19.5 (MP)** through `pystata`; the captures are
committed so the comparison replays without a licence.

## Install

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Requires Python ≥ 3.10. Optional at runtime: `linearmodels` (IV, panel),
`pyfixest` (HDFE), `arch` (GARCH, Phillips–Perron).

## Quick start

```python
import westata as ws
import pandas as pd

df = pd.read_stata("auto.dta")

# OLS with a factor variable and clustered standard errors (regress)
res = ws.OLS.from_formula("mpg ~ weight + C(foreign)", df).fit(
    cov_type="cluster", cov_kwds={"groups": df["foreign"]},
)
print(res.summary(style="stata"))     # Stata layout
res.test("weight foreign")            # F(2, 71) = 69.75, exactly as Stata
res.lincom("weight + 1000*foreign")   # linear combination

# IV (ivregress 2sls) — z statistics and Wald chi2, like Stata
ws.IV2SLS.from_formula("y ~ x + [endog ~ z1 + z2]", df).fit()

# High-dimensional FE (reghdfe)
ws.FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit()

# Panel (xtreg fe/re/be/fd)
ws.PanelOLS.from_formula("y ~ x1", df, entity="firm", time="year").fit()
ws.RandomEffects.from_formula("y ~ x1", df, entity="firm", time="year").fit()

# Discrete choice and censored regression
ws.Logit.from_formula("y ~ x1", df).fit()
ws.Tobit.from_formula("wage ~ edu", df, lower=0).fit()

# Marginal effects (margins)
ws.Logit.from_formula("y ~ x1 + x2", df).fit().margins(dydx=["x1", "x2"])

# Time series
ws.ARIMA(y, order=(1, 1, 1)).fit()
ws.unitroot(y, test="adf")

# Paper table (esttab)
ws.etable([res1, res2], model_names=["Base", "Full"])
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
| E3 | Compared against full-precision `e(b)`/`e(V)` captured from real Stata | 83 |
| E2 | Measured, understood, and not fully matched -- see the comparison guide | 4 |
| E1 | Implemented from the manual, unit-tested, **not** contrasted with Stata | 0 |

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

The linear and panel models agree to machine precision. Maximum-likelihood
models stop near 1e-7 because Stata's Newton and statsmodels' optimiser reach
the same optimum but halt at slightly different points; that is a real limit,
not slack.

Four commands are **E2**: measured, understood, and not fully matched. Each is
comparable in some respects and not others, which is the part that matters when
you are checking one against the other.

| Command | Coefficients | z / p | Decision |
|---|---|---|---|
| `arch` | to ~4 digits | **not comparable** | agree |
| `teffects nnmatch` | exact | to ~2 digits | agree |
| `xtabond` | to ~3 digits | to ~3 digits | agree |
| `permute` | exact | not replicate-for-replicate | agree (distribution) |

The two that matter most:

- **`arch`** — Stata runs `bhhh bfgs` with `nrtolerance(1e-5)` and stops before
  the likelihood is flat. Measured: westata's maximum is `ll = -1025.4148259782`,
  Stata reports `-1025.4176185349`, **2.8e-03 lower**. So the difference is a
  stopping rule, not an error, and westata's estimates are the ML ones. But the
  z statistics are computed at different points: `z(garch)` is 16.942 here against
  9.755 in Stata, and `p(omega)` 1.5e-03 against 4.97e-02. **Do not quote an
  `arch` z or p value as Stata's.**
- **`teffects nnmatch`** — the ATE is exact to 3.6e-15; the standard error is
  1.59% high because the within-arm variance lives in compiled Mata. The
  significance verdict agreed across every effect size tested.

See [`docs/USING_WITH_STATA.md`](docs/USING_WITH_STATA.md) for the full comparison
guidance, including what to quote and what to verify in Stata.

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
- Design spec: [`docs/superpowers/specs/2026-10-22-westata-design.md`](docs/superpowers/specs/2026-10-22-westata-design.md)

## Known gaps

Not implemented, and they raise instead of approximating: Mata, the graphics
system, data management, SEM/GSEM, survival analysis, Bayesian estimation, `mi`,
`xtabond2`, `csdid`, `rdrobust`, `synth`, general `gmm`, `nlsur`, multi-way
clustering (CGM), `vecrank`'s fixed-N exact critical values, and Stata's
arbitrary `r()` statistics outside the `[U] exp_list` grammar.

`docs/GAP_REGISTER.md` lists every one with its reason. Those are boundary
statements, not a to-do list.
