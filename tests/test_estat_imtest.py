"""Tests for Stata's ``estat imtest`` (Cameron & Trivedi IM-test).

Every expected number comes from ``tests/conformance/stata_fixtures/
imtest_auto.json``, which ``scripts/capture_imtest_fixtures.py`` captured from a
real StataNow 19.5.  The previous version of this file re-derived the formulas
in Python and compared the implementation against that re-derivation -- the
pattern ``docs/TESTING_SOP.md`` warns about, because a re-derivation can carry
the same misreading as the code.

Two behaviours are pinned here because both are silent when wrong:

**Degrees of freedom come from the auxiliary regression's RANK.**  ``rhs2``
contains ``x_i^2``, and an indicator satisfies ``x^2 == x``, so its square is
exactly collinear with the regressor.  Stata's ``_regress`` drops it and reports
the reduced ``e(df_m)``.  On ``regress mpg weight foreign`` White's test is
``df = 4``, not the 5 a column count gives.  The p-value ``chi2tail(4, ...)`` is
smaller than ``chi2tail(5, ...)``, so the wrong df can move a test across a
significance threshold while the statistic itself looks fine.

**Stata's default is FLOAT precision** (``imtest.ado:35``), so its default
``r(chi2)`` differs from an exact computation around the 8th significant digit.
This module computes in double, so it matches ``, double`` exactly and the
default to ~1e-8.  The fixture stores both columns; the tests assert the double
one and separately document the float gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws

FIXTURE = (
    Path(__file__).resolve().parent
    / "conformance"
    / "stata_fixtures"
    / "imtest_auto.json"
)

#: ``model id -> (formula, columns the frame needs)``.
MODELS = {
    "W1": ("mpg ~ weight + foreign", ["mpg", "weight", "foreign"]),
    "W2": (
        "mpg ~ weight + length + displacement",
        ["mpg", "weight", "length", "displacement"],
    ),
    "W3": ("mpg ~ weight", ["mpg", "weight"]),
    "W4": ("price ~ weight + mpg + foreign", ["price", "weight", "mpg", "foreign"]),
    "W5": ("mpg ~ weight + foreign + C(rep78)", ["mpg", "weight", "foreign", "rep78"]),
}


@pytest.fixture(scope="module")
def reference() -> dict:
    assert FIXTURE.exists(), (
        f"{FIXTURE} is missing; regenerate it with "
        "scripts/capture_imtest_fixtures.py"
    )
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def auto() -> pd.DataFrame:
    # ``convert_categoricals=False`` keeps ``foreign`` numeric, which is what
    # makes the squared column collinear -- the df effect under test.
    return pd.read_stata("tests/data/auto.dta", convert_categoricals=False)


def _fit(auto: pd.DataFrame, model: str):
    formula, columns = MODELS[model]
    frame = auto[list(columns)].dropna()
    return ws.OLS.from_formula(formula, frame).fit()


def _double_cases(reference: dict):
    for key, entry in sorted(reference["cases"].items()):
        model, precision, kind = key.split("|")
        if precision != "double" or entry.get("rc") != 0:
            continue
        yield key, model, kind, entry


# ---------------------------------------------------- every captured Stata value
@pytest.mark.parametrize(
    "case_id",
    [
        key
        for key in (
            json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
            if FIXTURE.exists()
            else {}
        )
        if key.split("|")[1] == "double"
    ],
)
def test_matches_stata_in_double_precision(case_id: str, reference, auto) -> None:
    """One captured ``estat imtest`` run per case, compared field by field.

    ``rel=1e-9``: Stata prints 17 significant digits and solves the same
    least-squares problem, so the last bit or two can differ.  1e-9 is still far
    tighter than any formula error produces.
    """

    model, precision, kind = case_id.split("|")
    expected = reference["cases"][case_id]
    rendered = _fit(auto, model).estat_imtest(white=(kind == "white"))

    checks = [
        ("chi2_h", rendered.heteroskedasticity),
        ("df_h", rendered.df_heteroskedasticity),
        ("chi2_s", rendered.skewness),
        ("df_s", rendered.df_skewness),
        ("chi2_k", rendered.kurtosis),
        ("df_k", rendered.df_kurtosis),
        ("chi2_t", rendered.total),
        ("df_t", rendered.df_total),
    ]
    if kind == "white":
        checks.extend([("chi2", rendered.white), ("df", rendered.white_df)])

    for name, got in checks:
        want = expected[name]
        if want is None:
            assert got is None, f"{case_id} {name}: expected None, got {got}"
            continue
        assert got == pytest.approx(want, rel=1e-9), f"{case_id} {name}"


def test_the_fixture_is_not_vacuous(reference) -> None:
    """Guard: an emptied or truncated fixture would make every case pass."""

    assert len(reference["cases"]) >= 20
    assert reference["stata_version"]
    models = {key.split("|")[0] for key in reference["cases"]}
    assert models == {"W1", "W2", "W3", "W4", "W5"}
    kinds = {key.split("|")[2] for key in reference["cases"]}
    assert kinds == {"white", "decomposition"}
    precisions = {key.split("|")[1] for key in reference["cases"]}
    assert precisions == {"float", "double"}


# ------------------------------------- the defect this module was fixed for
def test_dof_is_the_rank_not_the_column_count(reference, auto) -> None:
    """An indicator's squared column is collinear, so Stata DROPS it.

    ``mpg ~ weight + foreign`` builds 1 + 2 + 3 = 6 nominal auxiliary columns:
    ``weight``, ``foreign``, ``weight^2``, ``weight*foreign``, ``foreign^2`` --
    plus the constant.  ``foreign`` is 0/1, so ``foreign^2 == foreign`` and the
    rank is one lower.  Stata reports ``df = 4``; counting columns reports 5.

    This is the regression test for the reported bug: the statistic was right
    while the df, the p-value and therefore the verdict were wrong.
    """

    result = _fit(auto, "W1").estat_imtest(white=True)
    expected_df = reference["cases"]["W1|double|white"]["df"]

    assert expected_df == 4.0
    assert result.white_df == expected_df
    assert result.white_df != 5.0, "column counting is exactly the bug"

    # The p-values differ materially, which is why the df error mattered.
    p_right = float(stats.chi2.sf(result.white, 4))
    p_wrong = float(stats.chi2.sf(result.white, 5))
    assert p_right != pytest.approx(p_wrong, rel=1e-6)


def test_dof_total_uses_the_reduced_rank(reference, auto) -> None:
    """``df_t`` is ``df_h + df_s + df_k`` with every term rank-based."""

    result = _fit(auto, "W1").estat_imtest()
    expected = reference["cases"]["W1|double|decomposition"]

    assert result.df_heteroskedasticity == expected["df_h"]
    assert result.df_skewness == expected["df_s"]
    assert result.df_kurtosis == expected["df_k"]
    assert result.df_total == expected["df_t"] == 7.0


def test_continuous_only_model_keeps_the_nominal_dof(reference, auto) -> None:
    """Control: with three continuous regressors nothing collapses, so df == 9.

    Without this case, a "fix" that always subtracted one would also pass the
    indicator model.
    """

    result = _fit(auto, "W2").estat_imtest(white=True)
    expected = reference["cases"]["W2|double|white"]

    assert expected["df"] == 9.0
    assert result.white_df == 9.0


def test_two_indicators_collapse_two_columns(auto) -> None:
    """``W5`` has two indicator blocks, so more than one column disappears.

    ``weight``, ``foreign`` and the four ``rep78`` dummies give k = 6 slopes, so
    ``rhs2`` has ``k(k+1)/2 = 21`` cross-products and the nominal auxiliary
    design is 1 + 6 + 21 = 28 columns.  The observed rank is 15 (df 14): the six
    squared terms of the five indicator levels (``foreign`` plus four ``rep78``
    dummies) collapse, which is five columns fewer than nominal -- and one more
    than the four a "subtract one per indicator variable" rule would predict.
    Confirms the rule is genuinely rank-driven.
    """

    result = _fit(auto, "W5").estat_imtest(white=True)

    k = 6
    nominal = 1 + k + k * (k + 1) // 2
    assert nominal == 28

    assert result.white_df == 14.0
    assert result.white_df < nominal


# --------------------------------------------------------- precision behaviour
def test_default_float_and_double_captures_differ_as_documented(reference) -> None:
    """Stata's default is float, so its own two captures differ around 1e-8.

    Recorded rather than reproduced: this module computes in double, which is
    the exact answer.  If a future change made the implementation emulate
    float32, this test would notice the fixture no longer explains the gap.
    """

    float_case = reference["cases"]["W1|float|white"]
    double_case = reference["cases"]["W1|double|white"]

    gap = abs(float_case["chi2"] - double_case["chi2"]) / abs(double_case["chi2"])
    assert 1e-9 < gap < 1e-6, gap


def test_we_match_double_not_float(auto, reference) -> None:
    """Our value equals ``, double`` and differs from the float default."""

    result = _fit(auto, "W1").estat_imtest(white=True)
    double_chi2 = reference["cases"]["W1|double|white"]["chi2"]
    float_chi2 = reference["cases"]["W1|float|white"]["chi2"]

    assert result.white == pytest.approx(double_chi2, rel=1e-12)
    assert result.white != pytest.approx(float_chi2, rel=1e-12)


# ------------------------------------------------------------ internal checks
def test_heteroskedasticity_equals_whites_statistic(auto, reference) -> None:
    """White's test and the IM heteroskedasticity component are the same test.

    Stata itself shows this: for every continuous-only model the two captures are
    identical in the fixture.
    """

    for model in ("W2", "W3", "W4"):
        entry = reference["cases"][f"{model}|double|white"]
        assert entry["chi2"] == pytest.approx(entry["chi2_h"], rel=1e-12), model

    for model in MODELS:
        rendered = _fit(auto, model).estat_imtest(white=True)
        assert rendered.heteroskedasticity == pytest.approx(
            rendered.white, rel=1e-9
        ), model


def test_total_is_the_sum_of_the_components(auto) -> None:
    for model in MODELS:
        rendered = _fit(auto, model).estat_imtest()

        assert rendered.total == pytest.approx(
            rendered.heteroskedasticity + rendered.skewness + rendered.kurtosis
        ), model
        assert rendered.df_total == (
            rendered.df_heteroskedasticity
            + rendered.df_skewness
            + rendered.df_kurtosis
        ), model


def test_heteroskedastic_data_is_detected() -> None:
    rng = np.random.default_rng(3)
    n = 600
    x = rng.uniform(1, 10, size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=x * 0.4, size=n)
    frame = pd.DataFrame({"y": y, "x": x})

    rendered = ws.OLS.from_formula("y ~ x", frame).fit().estat_imtest()

    p = float(stats.chi2.sf(rendered.heteroskedasticity, rendered.df_heteroskedasticity))
    assert p < 0.01


def test_homoskedastic_data_is_not_flagged() -> None:
    rng = np.random.default_rng(3)
    n = 600
    x = rng.uniform(1, 10, size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame({"y": y, "x": x})

    rendered = ws.OLS.from_formula("y ~ x", frame).fit().estat_imtest()

    p = float(stats.chi2.sf(rendered.heteroskedasticity, rendered.df_heteroskedasticity))
    assert p > 0.05


# ----------------------------------------------------------------- refusals
def test_logit_is_refused(auto) -> None:
    frame = auto[["mpg", "weight"]].dropna().assign(
        domestic=(auto.loc[auto[["mpg", "weight"]].dropna().index, "foreign"] == 0).astype(int)
    )
    logit = ws.Logit.from_formula("domestic ~ weight", frame).fit()

    with pytest.raises(NotImplementedError, match="after regress"):
        logit.estat_imtest()


def test_robust_vce_is_refused(auto) -> None:
    frame = auto[["mpg", "weight", "length"]].dropna()
    robust = ws.OLS.from_formula("mpg ~ weight + length", frame).fit(cov_type="HC1")

    with pytest.raises(ValueError, match="weighted or clustered"):
        robust.estat_imtest()


# ------------------------------------------------------------- presentation
def test_to_dict_uses_stata_r_names(auto) -> None:
    payload = _fit(auto, "W1").estat_imtest().to_dict()

    assert set(payload) == {
        "chi2_h",
        "df_h",
        "chi2_s",
        "df_s",
        "chi2_k",
        "df_k",
        "chi2_t",
        "df_t",
    }
    with_white = _fit(auto, "W1").estat_imtest(white=True).to_dict()
    assert "chi2" in with_white and "df" in with_white


def test_output_names_the_three_components(auto) -> None:
    text = str(_fit(auto, "W1").estat_imtest())

    assert "Cameron & Trivedi" in text
    for label in ("Heteroskedasticity", "Skewness", "Kurtosis", "Total"):
        assert label in text


def test_output_layout_matches_stata(auto) -> None:
    """The printed table must match Stata line for line.

    Transcribed from a captured StataNow 19.5 log for
    ``regress mpg weight foreign`` / ``estat imtest, white``.  Layout is checked
    separately from the numbers because "the numbers are right" and "the table
    looks like Stata's" fail independently -- a previous round printed the p
    column one space too wide and omitted the opening/closing full rules.
    """

    expected = [
        "",
        "White's test",
        "H0: Homoskedasticity",
        "Ha: Unrestricted heteroskedasticity",
        "",
        "    chi2(4) =   9.79",
        "Prob > chi2 = 0.0440",
        "",
        "Cameron & Trivedi's decomposition of IM-test",
        # CORRECTED in round 193: the decomposition table DOES carry a blank line after its
        # title, three 50-character rules and a 21+'+'+28 separator. Re-captured live from
        # `quietly regress mpg weight fx` / `estat imtest, white`; the transcription this replaces
        # omitted all of them.
        "",
        "-" * 50,
        "              Source |       chi2     df         p",
        "-" * 21 + "+" + "-" * 28,
        "  Heteroskedasticity |       9.79      4    0.0440",
        "            Skewness |       7.31      2    0.0259",
        "            Kurtosis |       1.71      1    0.1912",
        "-" * 21 + "+" + "-" * 28,
        "               Total |      18.81      7    0.0088",
        "-" * 50,
    ]
    rendered = [line.rstrip() for line in str(_fit(auto, "W1").estat_imtest(white=True)).splitlines()]

    assert rendered == expected
