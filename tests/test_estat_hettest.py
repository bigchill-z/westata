"""Tests for Stata's ``estat hettest`` (Breusch-Pagan / Cook-Weisberg).

Every expected number here comes from ``tests/conformance/stata_fixtures/
hettest_auto.json``, which ``scripts/capture_hettest_fixtures.py`` captured from
a real StataNow 19.5.  That distinction is the whole point of this module.

An earlier version of these tests checked the implementation against a second
hand transcription of ``hettest.ado``.  It passed, and the implementation was
still wrong: the transcription had been written from the same misreading, so
test and code agreed with each other and disagreed with Stata.  A transcription
is not an oracle -- a captured ``r()`` is.

The misreading was worth recording because it is the one ``hettest.ado`` invites:
**no varlist means the fitted values**, not "every regressor"
(``hettest.ado:110-115``).  On ``regress mpg weight foreign`` that is
``chi2(1) = 7.89`` rather than ``chi2(2) = 24.17``, and both are plausible
numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.postestimation.hettest import estat_hettest

FIXTURE = (
    Path(__file__).resolve().parent
    / "conformance"
    / "stata_fixtures"
    / "hettest_auto.json"
)

#: ``model id -> (formula, the columns the data pull needs)``.
MODELS = {
    "M1": ("mpg ~ weight + foreign", ["mpg", "weight", "foreign"]),
    "M2": (
        "mpg ~ weight + length + displacement",
        ["mpg", "weight", "length", "displacement"],
    ),
    "M3": ("price ~ weight + mpg + foreign", ["price", "weight", "mpg", "foreign"]),
    "F1": ("mpg ~ weight + C(rep78)", ["mpg", "weight", "rep78"]),
    "F2": ("mpg ~ weight + C(foreign)", ["mpg", "weight", "foreign"]),
}


@pytest.fixture(scope="module")
def reference() -> dict:
    assert FIXTURE.exists(), (
        f"{FIXTURE} is missing; regenerate it with "
        "scripts/capture_hettest_fixtures.py"
    )
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def auto() -> pd.DataFrame:
    return pd.read_stata("tests/data/auto.dta")


def _fit(auto: pd.DataFrame, model: str, weight: dict | None):
    formula, columns = MODELS[model]
    columns = list(columns)
    if weight:
        columns.append(weight["var"])
    frame = auto[list(dict.fromkeys(columns))].dropna()
    kwargs = {}
    if weight:
        kwargs = {"weights": weight["var"], "weight_type": weight["type"]}
    return ws.OLS.from_formula(formula, frame, **kwargs).fit()


def _case(result, kind: str, method: str):
    """Run westata the way the fixture's ``kind`` says Stata was invoked."""

    if kind == "fitted":
        return result.estat_hettest(method=method, variables=[])
    if kind == "rhs":
        return result.estat_hettest(method=method, rhs=True)
    named = kind.split(":", 1)[1].split(",")
    return result.estat_hettest(method=method, variables=named)


def _cases(reference: dict):
    for key, expected in sorted(reference["cases"].items()):
        model, weight_key, kind, method = key.split("|")
        weight = None
        if weight_key != "-":
            weight_type, variable = weight_key.split(":")
            weight = {"type": weight_type, "var": variable}
        yield key, model, weight, kind, method, expected


def _case_ids(reference: dict) -> list[str]:
    return [key for key, *_ in _cases(reference)]


# ------------------------------------------------- every captured Stata value
@pytest.mark.parametrize("case_id", _case_ids(json.loads(FIXTURE.read_text(
    encoding="utf-8"))) if FIXTURE.exists() else [])
def test_matches_stata(case_id: str, reference, auto) -> None:
    """One Stata run per case; the statistic, df, p and df_r must all agree.

    ``rel=1e-9`` rather than an exact match: Stata prints 17 significant digits
    and its ``_regress`` solves the same weighted least-squares problem through a
    different code path, so the last bit or two can differ.  A relative 1e-9 is
    still four orders of magnitude tighter than any wrong formula produces.
    """

    expected_row = reference["cases"][case_id]
    model, weight_key, kind, method = case_id.split("|")
    weight = None
    if weight_key != "-":
        weight_type, variable = weight_key.split(":")
        weight = {"type": weight_type, "var": variable}

    rendered = _case(_fit(auto, model, weight), kind, method)
    statistic, df, pvalue, df_r = expected_row

    assert rendered.statistic == pytest.approx(statistic, rel=1e-9), case_id
    assert rendered.df == df, case_id
    assert rendered.pvalue == pytest.approx(pvalue, rel=1e-9, abs=1e-15), case_id
    if df_r is None:
        assert rendered.df_denom is None, case_id
    else:
        assert rendered.df_denom == df_r, case_id


def test_the_fixture_is_not_vacuous(reference) -> None:
    """Guard: an empty or truncated fixture would make every case above pass."""

    assert len(reference["cases"]) >= 90
    assert reference["stata_version"], "the capture must record its Stata version"
    # Every variant and every weight type is represented.
    kinds = {key.split("|")[2] for key in reference["cases"]}
    assert {"fitted", "rhs"} <= kinds
    assert any(kind.startswith("varlist:") for kind in kinds)
    weights = {key.split("|")[1] for key in reference["cases"]}
    assert {"-", "aweight:rep78", "fweight:rep78", "iweight:rep78"} <= weights
    methods = {key.split("|")[3] for key in reference["cases"]}
    assert methods == {"normal", "iid", "fstat"}


# ------------------------------------------------- the defect this module fixed
def test_default_regresses_on_fitted_values_not_on_every_regressor(auto, reference) -> None:
    """``hettest.ado:110-115``: no varlist -> fitted values.

    This is the regression test for the bug that made the numbers disagree with
    Stata.  ``estat hettest`` and ``estat hettest, rhs`` are DIFFERENT tests on
    the same fit; returning the second when the first was asked for is silent.
    """

    result = _fit(auto, "M1", None)
    fitted = result.estat_hettest()
    every = result.estat_hettest(rhs=True)

    assert fitted.df == 1
    assert every.df == 2
    assert fitted.statistic == pytest.approx(
        reference["cases"]["M1|-|fitted|normal"][0], rel=1e-9
    )
    assert every.statistic == pytest.approx(
        reference["cases"]["M1|-|rhs|normal"][0], rel=1e-9
    )
    assert not np.isclose(fitted.statistic, every.statistic, rtol=1e-6)


def test_the_default_is_not_the_iid_statistic(auto) -> None:
    """Different formulas on one fit, so they must not agree."""

    result = _fit(auto, "M2", None)
    normal = result.estat_hettest(method="normal").statistic
    iid = result.estat_hettest(method="iid").statistic
    fstat = result.estat_hettest(method="fstat").statistic

    assert not np.isclose(normal, iid, rtol=1e-6)
    assert not np.isclose(normal, fstat, rtol=1e-6)


def test_varlist_names_use_source_variables_not_design_columns(auto, reference) -> None:
    """``hettest rep78`` names the raw 1-5 column, not the treatment dummies.

    Stata measures 12.77 (df 1) for the raw column against 24.30 (df 5) for
    ``, rhs``, which uses the ``e(b)`` names.  Accepting only design columns
    would make the raw-variable spelling unreachable.
    """

    result = _fit(auto, "F1", None)
    raw = result.estat_hettest(variables=["rep78"], method="iid")
    every = result.estat_hettest(rhs=True, method="iid")

    assert raw.df == 1
    assert every.df == 5
    assert raw.statistic == pytest.approx(
        reference["cases"]["F1|-|varlist:rep78|iid"][0], rel=1e-9
    )
    assert every.statistic == pytest.approx(
        reference["cases"]["F1|-|rhs|iid"][0], rel=1e-9
    )


def test_a_value_labelled_variable_uses_its_raw_codes(auto, reference) -> None:
    """A labelled 0/1 variable is numeric in Stata, so it is one column."""

    result = _fit(auto, "F2", None)
    raw = result.estat_hettest(variables=["foreign"], method="iid")

    assert raw.df == 1
    assert raw.statistic == pytest.approx(
        reference["cases"]["F2|-|varlist:foreign|iid"][0], rel=1e-9
    )


def test_weights_are_used_by_the_auxiliary_regression(auto, reference) -> None:
    """``hettest.ado:138`` passes ``wgt`` to ``_regress``.

    Ignoring the weights does not merely shift the third digit: on this fit the
    unweighted iid statistic is 2.53 against the weighted 3.52, and the fstat
    degrees of freedom move from 72 to 67.
    """

    weighted = _fit(auto, "M1", {"type": "aweight", "var": "rep78"})
    unweighted = _fit(auto, "M1", None)

    got = weighted.estat_hettest(method="iid")
    assert got.statistic == pytest.approx(
        reference["cases"]["M1|aweight:rep78|fitted|iid"][0], rel=1e-9
    )
    assert got.statistic != pytest.approx(unweighted.estat_hettest(method="iid").statistic)

    fstat = weighted.estat_hettest(method="fstat")
    assert fstat.df_denom == 67


def test_fweight_and_iweight_use_sum_of_weights_as_e_N(auto, reference) -> None:
    """``rep78`` sums to 235, so ``e(N)`` is 235 and ``df_r`` is 233 -- not 69."""

    for weight_type in ("fweight", "iweight"):
        result = _fit(auto, "M1", {"type": weight_type, "var": "rep78"})
        fstat = result.estat_hettest(method="fstat")
        expected = reference["cases"][f"M1|{weight_type}:rep78|fitted|fstat"]
        assert fstat.statistic == pytest.approx(expected[0], rel=1e-9), weight_type
        assert fstat.df_denom == 233, weight_type

        iid = result.estat_hettest(method="iid")
        assert iid.statistic == pytest.approx(
            reference["cases"][f"M1|{weight_type}:rep78|fitted|iid"][0], rel=1e-9
        ), weight_type


# ------------------------------------------------------------------- mtest
def test_mtest_per_variable_statistics_match_stata(auto, reference) -> None:
    for model, entry in reference["mtest"].items():
        table = _fit(auto, model, None).estat_hettest(mtest="noadjust", rhs=True).table
        for variable in entry["variables"]:
            row = table.loc[variable["name"]]
            assert row["chi2"] == pytest.approx(variable["chi2"], rel=1e-9), variable["name"]
            assert row["df"] == variable["df"], variable["name"]
            assert row["p"] == pytest.approx(variable["pvalue"], rel=1e-9), variable["name"]


def test_mtest_reports_the_simultaneous_statistic(auto, reference) -> None:
    """``r(chi2)`` sits alongside the family; it is the SAME test as ``rhs``."""

    result = _fit(auto, "M1", None)
    simultaneous = result.estat_hettest(rhs=True).statistic
    expected = reference["mtest"]["M1"]["simultaneous"]["chi2"]

    assert simultaneous == pytest.approx(expected, rel=1e-9)


def test_mtest_is_dropped_for_a_single_regressor(auto, reference) -> None:
    """``hettest.ado:126-128`` clears ``mtmethod`` when only one regressor is used.

    The no-varlist default IS a single regressor (the fitted values), so
    ``estat hettest, mtest(holm)`` returns the simultaneous test and NO
    ``r(mtest)`` matrix at all -- verified, not inferred.
    """

    result = _fit(auto, "M1", None)
    rendered = result.estat_hettest(mtest="holm")

    expected = reference["mtest_single_regressor"]
    assert type(rendered).__name__ == "HetTestResult"
    assert rendered.statistic == pytest.approx(expected["chi2"], rel=1e-9)
    assert rendered.df == expected["df"]
    assert expected["r_mtest_missing"] is True


def test_mtest_adjusted_values_follow_the_raw_family(auto) -> None:
    from westata.postestimation.mtest import adjust_pvalues

    result = _fit(auto, "M1", None)
    raw_p = result.estat_hettest(mtest="noadjust", rhs=True).table["p"].to_numpy()
    for method in ("bonferroni", "holm", "sidak"):
        table = result.estat_hettest(mtest=method, rhs=True).table
        assert np.allclose(table["adjustp"].to_numpy(), adjust_pvalues(raw_p, method))


def test_mtest_r_matrix_gains_the_adjusted_column(auto) -> None:
    result = _fit(auto, "M1", None)

    assert result.estat_hettest(mtest="noadjust", rhs=True).to_matrix().shape == (2, 3)
    assert result.estat_hettest(mtest="holm", rhs=True).to_matrix().shape == (2, 4)


def test_mtest_display_lists_every_variable_and_the_adjustment(auto) -> None:
    text = str(_fit(auto, "M2", None).estat_hettest(mtest="holm", rhs=True))

    assert "holm" in text
    for name in ("weight", "length", "displacement"):
        assert name in text
    assert "adjustp" in text


# --------------------------------------------------------------------- refusals
def test_robust_vce_is_refused(auto) -> None:
    frame = auto[["mpg", "weight", "foreign"]].dropna()
    robust = ws.OLS.from_formula("mpg ~ weight + foreign", frame).fit(cov_type="HC1")

    with pytest.raises(ValueError, match="robust"):
        robust.estat_hettest()


def test_clustered_vce_is_refused(auto) -> None:
    frame = auto[["mpg", "weight", "foreign"]].dropna()
    groups = (np.arange(len(frame)) % 7).astype(int)
    clustered = ws.OLS.from_formula("mpg ~ weight + foreign", frame).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )

    with pytest.raises(ValueError, match="robust or clustered"):
        clustered.estat_hettest()


def test_noconstant_is_refused(auto, reference) -> None:
    """``estat hettest`` after ``regress, noconstant`` is Stata r(301)."""

    frame = auto[["mpg", "weight", "foreign"]].dropna()
    noconstant = ws.OLS.from_formula("mpg ~ 0 + weight + foreign", frame).fit()

    assert reference["refusals"]["noconstant"] == 301
    with pytest.raises(ValueError, match="noconstant"):
        noconstant.estat_hettest()


def test_iid_with_mtest_is_refused(auto) -> None:
    with pytest.raises(ValueError, match="iid may not be specified"):
        _fit(auto, "M1", None).estat_hettest(method="iid", mtest="holm", rhs=True)


def test_fstat_with_mtest_is_refused(auto) -> None:
    with pytest.raises(ValueError, match="fstat may not be specified"):
        _fit(auto, "M1", None).estat_hettest(method="fstat", mtest="holm", rhs=True)


def test_unknown_method_is_rejected(auto) -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        _fit(auto, "M1", None).estat_hettest(method="bogus")


def test_unknown_mtest_method_is_rejected(auto) -> None:
    with pytest.raises(ValueError, match="not supported"):
        _fit(auto, "M1", None).estat_hettest(mtest="fdr", rhs=True)


def test_unknown_regressor_is_rejected(auto) -> None:
    with pytest.raises(ValueError, match="unknown regressor"):
        _fit(auto, "M1", None).estat_hettest(variables=["nope"])


def test_nonrobust_vce_is_accepted(auto) -> None:
    result = _fit(auto, "M1", None)
    assert result.cov_type == "nonrobust"
    assert np.isfinite(result.estat_hettest().statistic)


# ---------------------------------------------------------------- presentation
def test_to_dict_matches_the_variant(auto) -> None:
    result = _fit(auto, "M1", None)

    assert set(result.estat_hettest().to_dict()) == {"chi2", "df", "p"}
    assert set(result.estat_hettest(method="fstat").to_dict()) == {
        "F",
        "df_m",
        "df_r",
        "p",
    }


def test_output_names_the_auxiliary_regressors(auto) -> None:
    """``hettest.ado:258`` prints the Variable(s): caption.

    The plural ``s`` is set only in the non-fitted branch and only for a
    multi-word varlist (``:117-120``), so the fitted-values line is always
    singular.  Verified against StataNow 19.5:

        estat hettest            -> Variable: Fitted values of mpg
        estat hettest, rhs       -> Variables: All independent variables
        estat hettest weight     -> Variable: weight
    """

    result = _fit(auto, "M1", None)
    single = _fit(auto, "M1", None).estat_hettest(variables=["weight"])

    assert "Variable: Fitted values of mpg" in str(result.estat_hettest())
    assert "Variables: All independent variables" in str(result.estat_hettest(rhs=True))
    assert "Variable: weight" in str(single)
    assert "Normal error terms" in str(result.estat_hettest())
    assert "i.i.d. error terms" in str(result.estat_hettest(method="iid"))
    assert "Constant variance" in str(result.estat_hettest())


def test_pvalue_is_the_upper_tail(auto) -> None:
    rendered = _fit(auto, "M1", None).estat_hettest()

    assert rendered.pvalue == pytest.approx(
        float(stats.chi2.sf(rendered.statistic, rendered.df)), rel=1e-12
    )


def test_module_function_matches_the_method(auto) -> None:
    result = _fit(auto, "M1", None)

    assert estat_hettest(result).to_dict() == result.estat_hettest().to_dict()
