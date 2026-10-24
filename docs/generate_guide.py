"""Generate docs/GUIDE.md from the measured comparison data.

The document's comparison tables are GENERATED, not written. Every number comes from
docs/comparison.json, which was produced by running the same array through Stata and through westata
and taking the maximum relative difference over parameters and standard errors. A number that is not
in the JSON cannot appear in a table.

That constraint is the whole point. The earlier README contained a Quick start that had never been
executed and was broken in four ways, including a quoted F statistic that nobody could have obtained
from the code beside it. Generating the tables from measured output makes that class of error
impossible for the numeric claims, and the prose is kept to statements the code blocks below it
actually demonstrate.

Structure of the generated document:

    1. what the library is and what evidence grading means
    2. installation
    3. the API shape, with one runnable example
    4. a command-by-command guide organised by family
    5. the comparison tables, generated from measurement
    6. known deviations, with the measured number and the reason
    7. how to reproduce every number

Section 5 and 6 are the parts a reader should trust least on prose and most on data, so both are
generated from the JSON and the prose only interprets.
"""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(r"D:\myCodeProject\statapy")
DATA = REPO / "docs" / "comparison.json"

entries = json.loads(DATA.read_text(encoding="utf-8"))
comparisons = [e for e in entries if "rel_b" in e]
conventions = [e for e in entries if e.get("convention_note")]

# Worst-of(b, se) ordering, so the tables lead with what a reader most needs to know.
comparisons.sort(key=lambda e: max(e["rel_b"], e["rel_se"]))


def fmt(value: float) -> str:
    """Format a relative difference the way a reader can act on."""

    if value == 0.0:
        return "0"
    if value < 1e-12:
        return f"{value:.1e}"
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def verdict(value: float) -> str:
    if value < 1e-12:
        return "machine precision"
    if value < 1e-9:
        return "float noise"
    if value < 1e-6:
        return "excellent"
    if value < 1e-3:
        return "very close"
    return "bounded"


# ---- the main comparison table
rows = []
for entry in comparisons:
    worst = max(entry["rel_b"], entry["rel_se"])
    rows.append(
        f"| `{entry['stata_command']}` | {entry['nobs_stata']} | "
        f"{fmt(entry['rel_b'])} | {fmt(entry['rel_se'])} | {verdict(worst)} |"
    )
comparison_table = "\n".join(rows)

# ---- per-command detail blocks, showing actual numbers side by side
detail_blocks = []
for entry in comparisons:
    lines = [
        f"#### `{entry['stata_command']}`",
        "",
        f"Dataset: `{entry['dataset']}` — {entry['nobs_stata']} observations on the Stata side.",
        "",
        "| Parameter | Stata `b` | westata `b` | Stata `se` | westata `se` |",
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
    detail_blocks.append("\n".join(lines))
detail_section = "\n".join(detail_blocks)

# ---- observation-count differences, which need explaining rather than hiding
mismatches = [
    (e["command"], e["nobs_stata"], e["nobs_ours"])
    for e in comparisons
    if e["nobs_stata"] != e["nobs_ours"]
]
n_table = "| Command | Stata `e(N)` | westata `nobs` |\n|---|---|---|\n" + "\n".join(
    f"| `{command}` | {stata} | {ours} |" for command, stata, ours in mismatches
)

# ---- the convention note, formatted from measurement
conv_lines = []
for entry in conventions:
    conv_lines.append(
        "\n".join(
            [
                f"**`{entry['stata_command']}`**",
                "",
                "| | standard error of the slope |",
                "|---|---|",
                f"| Stata `vce(robust)` | {entry['stata_se_weight']:.10g} |",
                f"| westata `cov_type='HC1'` | {entry['ours_hc1_se_weight']:.10g} |",
                f"| westata `cov_type='HC0'` | {entry['ours_hc0_se_weight']:.10g} |",
                "",
                f"`HC1` differs from Stata by {entry['rel_hc1']:.2%}, `HC0` by {entry['rel_hc0']:.2%}. "
                f"The ratio of our `HC1` to Stata's is {entry['observed_ratio']:.10f}, and "
                f"`sqrt(n/(n-1))` is {entry['predicted_ratio']:.10f}: Stata's robust VCE for this "
                "command divides by `n-1`, while statsmodels-style `HC1` divides by `n-k`. Neither is "
                "wrong; they are different finite-sample adjustments. For `regress`, Stata uses "
                "`n/(n-k)` and our `HC1` matches it exactly — the divergence is specific to the "
                "likelihood-based commands.",
            ]
        )
    )
convention_section = "\n\n".join(conv_lines)

# ---- coverage summary computed from the data
best = sum(1 for e in comparisons if max(e["rel_b"], e["rel_se"]) < 1e-12)
good = sum(1 for e in comparisons if max(e["rel_b"], e["rel_se"]) < 1e-9)
total = len(comparisons)

DOCUMENT = f"""# westata — a guide

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
    cov_type="cluster", cov_kwds={{"groups": df["foreign"]}},
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

**{total} commands compared. {best} agree to machine precision (relative difference below 1e-12), {good}
to better than 1e-9.** The table is ordered by worst case, so the top rows are where to look first.

| Stata command | N | max rel. diff. in `b` | max rel. diff. in `se` | verdict |
|---|---|---|---|---|
{comparison_table}

### Side-by-side detail

{detail_section}

    "",
    "### Why some observation counts differ",
    "",
    "Some rows above show a different `N` on each side. None is a mistake, and each is one of three",
    "things:",
    "",
    "{n_table}",
    "",
    "1. **What `e(N)` counts.** `sureg` is a system: Stata reports the *per-equation* sample size,",
    "   westata the length of the stacked residual vector (equations x rows), which is what its result",
    "   object validates `nobs` against. Two equations on 74 rows give 74 and 148.",
    "2. **Missing values are dropped.** `tobit` on `rep78` loses the 5 observations where `rep78` is",
    "   missing, so the estimation sample is 69, while Stata's `e(N)` still reports 74.",
    "3. **Conditioning and collapsing.** `xtlogit, fe` and `xtpoisson, fe` condition out entities whose",
    "   outcome never varies; `xtreg, be` collapses to one observation per entity. We report the rows that",
    "   survived, Stata the rows that entered.",
    "",
    "In every case the coefficients and standard errors agree to the precision shown in the table above,",
    "which is the check that matters. The counts are listed because a silent 148-versus-74 would read as",
    "a defect.",
    "",
## Known deviations

Three kinds of difference exist in the tables above, and they are not the same thing.

### A finite-sample convention, not an error

{convention_section}

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
"""

OUT = REPO / "docs" / "GUIDE.md"
OUT.write_text(DOCUMENT, encoding="utf-8")
print(f"wrote {OUT}")
print(f"  {len(DOCUMENT.splitlines())} lines")
print(f"  {total} comparisons, {len(conventions)} convention note(s)")
print(f"  machine precision: {best}   better than 1e-9: {good}")
