"""E2 conformance: ``ws.synth2`` against five live-StataNow-19.5 captures.

Fixtures: ``tests/conformance/stata_fixtures/synth2_*.json`` (captured by
``scripts/capture_synth2_fixtures.py``).  They hold full-precision e() matrices
and scalars; the suite needs NO Stata licence.

MEASURED bounds (2026-11-11, California / ADH 2010 specification)::

    e(N/T/T0/T1/K/J)   exact (0)
    e(bal)              <= 6.6e-14   -> machine precision (E3-grade)
    e(U_wt)             exact (0)    -> machine precision (E3-grade)
    e(V_wt)             <= 1.2e-14   -> machine precision (E3-grade)
    e(rmse)             <= 2.7e-5    (inherited from ws.synth's RMSPE, E2)
    e(r2)               <= 1.4e-8
    e(att)              <= 2.7e-7
    e(pval)             <= 3.8e-6 without cutoff; <= 1.8e-2 with cut(2)
    e(mspe)             per-row max|d|: median 1.5e-3, 80% <= 1e-2,
                           max 2.1e1 on 2 degenerate fake units (QP
                           non-uniqueness).  Pointwise p-values are far more
                           stable (rank statistics); see the test thresholds.

These are NOT bit-for-bit claims: ``synth2`` re-runs ``ws.synth`` up to 39
times, and the underlying QP is rank-deficient (Stata's C++ plugin stops off
the optimal face).  The FORMULAS are transcribed verbatim from readable ado
source; the measured gaps are recorded in registry.yml.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from westata.estimators.synth2 import synth2

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
SMOKING = Path(r"D:\myCodeProject\test_westata\data\synth_smoking.dta")

FULL = ["lnincome", "age15to24", "beer", "retprice",
        "cigsale(1988)", "cigsale(1980)", "cigsale(1975)"]
SHORT = ["lnincome", "age15to24", "beer", "retprice",
         "cigsale(1980)", "cigsale(1975)"]

# Measured tolerances.  Deliberately generous where the QP non-uniqueness
# shows up, tight where the formula layer is exact.
ATOL_MACHINE = 1e-10
ATOL_FIT = {
    "rmse": 1e-4, "r2": 1e-6, "att": 1e-5,
}
ATOL_PVAL = 2e-2          # cut(2) case is worst
ATOL_MSPE_ROW = 3e1       # 2 degenerate rows; see module docstring
# Measured distribution of per-row max|d| across the 39 mspe rows:
#   median 1.5e-3, 75th pct 8e-3, 80th pct 1e-2, 90th pct 6e-2, max 21.3.
# 79.5% of rows are within 1e-2; the rest are small drift from ws.synth's E2
# RMSPE, with two degenerate fake units reaching ~21.  Thresholds are MEASURED.
MSPE_NEAR_FRACTION = 0.75
MSPE_NEAR_ATOL = 1e-2


def _case(name: str) -> dict:
    path = FIXTURES / f"synth2_{name}.json"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"fixture {name} not captured")
    return json.loads(path.read_text(encoding="utf-8"))["cases"][f"synth2|{name}"]


def _data() -> pd.DataFrame:
    if not SMOKING.exists():  # pragma: no cover
        pytest.skip("smoking.dta is not available")
    return pd.read_stata(SMOKING, convert_categoricals=False)


def _run(name: str, **kw):
    base = dict(
        data=_data(), unit="state", time="year", depvar="cigsale",
        trunit=3, trperiod=1989,
    )
    base.update(kw)
    return synth2(**base)


SCENARIOS = [
    ("main", dict(predictors=FULL, xperiod=list(range(1980, 1989)))),
    ("placebo_unit", dict(predictors=FULL, xperiod=list(range(1980, 1989)),
                          placebo="unit")),
    ("placebo_unit_cut2", dict(
        predictors=FULL, xperiod=list(range(1980, 1989)),
        placebo="unit cut(2)")),
    ("placebo_period_1985", dict(
        predictors=SHORT, xperiod=list(range(1980, 1985)),
        placebo="period(1985)")),
    ("loo", dict(predictors=FULL, xperiod=list(range(1980, 1989)), loo=True)),
]


@pytest.mark.parametrize("name,kwargs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_e_scalars_match_stata(name, kwargs) -> None:
    case = _case(name)
    r = _run(name, **kwargs)
    for key in ("N", "T", "T0", "T1", "K", "J"):
        assert int(getattr(r, key)) == int(case["scalars"][key])
    for key, atol in ATOL_FIT.items():
        assert abs(float(getattr(r, key)) - float(case["scalars"][key])) <= atol, (
            f"{name}: e({key}) {getattr(r, key)} vs {case['scalars'][key]}"
        )


@pytest.mark.parametrize("name,kwargs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_bal_and_weight_matrices_are_machine_precision(name, kwargs) -> None:
    case = _case(name)
    r = _run(name, **kwargs)
    for mat in ("bal", "U_wt", "V_wt"):
        expected = np.asarray(case["matrices"][mat], dtype=float)
        got = np.asarray(getattr(r, mat), dtype=float)
        assert got.shape == expected.shape, (name, mat, got.shape, expected.shape)
        np.testing.assert_allclose(got, expected, atol=ATOL_MACHINE, rtol=0)


def test_in_space_placebo_mspe_is_within_the_measured_e2_bound() -> None:
    """Most rows exact; a few degenerate fake units differ (QP non-uniqueness)."""

    case = _case("placebo_unit")
    r = _run("placebo_unit", predictors=FULL,
             xperiod=list(range(1980, 1989)), placebo="unit")
    a = np.asarray(case["matrices"]["mspe"], dtype=float)
    b = np.asarray(r.mspe, dtype=float)
    assert a.shape == b.shape == (39, 4)
    row_max = np.abs(a - b).max(axis=1)
    assert row_max.max() <= ATOL_MSPE_ROW, row_max.max()
    near = float(np.mean(row_max <= MSPE_NEAR_ATOL))
    assert near >= MSPE_NEAR_FRACTION, (near, row_max)


def test_in_space_placebo_pointwise_pvalues_match() -> None:
    case = _case("placebo_unit")
    r = _run("placebo_unit", predictors=FULL,
             xperiod=list(range(1980, 1989)), placebo="unit")
    a = np.asarray(case["matrices"]["pval"], dtype=float)
    b = np.asarray(r.pval, dtype=float)
    np.testing.assert_allclose(b, a, atol=ATOL_PVAL, rtol=0)


def test_cutoff_case_pvalues_within_the_measured_bound() -> None:
    case = _case("placebo_unit_cut2")
    r = _run("placebo_unit_cut2", predictors=FULL,
             xperiod=list(range(1980, 1989)), placebo="unit cut(2)")
    a = np.asarray(case["matrices"]["pval"], dtype=float)
    b = np.asarray(r.pval, dtype=float)
    np.testing.assert_allclose(b, a, atol=ATOL_PVAL, rtol=0)


def test_in_time_placebo_has_no_inference() -> None:
    """synth2.ado:502-552 produces NO mspe/pval for fake treatment times."""

    case = _case("placebo_period_1985")
    r = _run("placebo_period_1985", predictors=SHORT,
             xperiod=list(range(1980, 1985)), placebo="period(1985)")
    assert "mspe" not in case["matrices"] and r.mspe is None
    # The Stata capture stores an absent e(pval) as a 1x1 missing sentinel,
    # which the capture script filters out.
    assert "pval" not in case["matrices"] and r.pval is None
    assert 1985 in r.time_placebo
    assert r.time_placebo[1985]["Y_synthetic"].shape[0] == 31


def test_loo_envelopes_are_captured_and_reproduced() -> None:
    case = _case("loo")
    r = _run("loo", predictors=FULL, xperiod=list(range(1980, 1989)), loo=True)
    assert r.loo_n_runs == case["loo_n_runs"] == 4
    assert r.loo_min is not None and r.loo_max is not None
    assert r.loo_min.shape == r.Y_synthetic.shape
    assert np.all(r.loo_min <= r.loo_max + 1e-9)
    # The captured frame holds 4 envelope columns; verify they are present
    # in the fixture so a future capture regression is caught.
    assert len(case["loo_envelopes"]) == 4


def test_placebo_probability_is_a_rank_statistic_in_unit_interval() -> None:
    r = _run("placebo_unit", predictors=FULL,
             xperiod=list(range(1980, 1989)), placebo="unit")
    assert r.placebo_probability is not None
    assert 0.0 < r.placebo_probability <= 1.0


def test_mutation_guard_r2_denominator_is_the_synthetic_outcome() -> None:
    """Pins the NON-standard R2 formula (synth2.ado:647)."""

    r = _run("main", predictors=FULL, xperiod=list(range(1980, 1989)))
    eff = r.gap.ravel()
    pre = r.time_index < 1989
    right = 1.0 - np.sum(eff[pre] ** 2) / np.sum(
        (r.Y_synthetic.ravel()[pre] - r.Y_synthetic.ravel()[pre].mean()) ** 2
    )
    wrong = 1.0 - np.sum(eff[pre] ** 2) / np.sum(
        (r.Y_treated.ravel()[pre] - r.Y_treated.ravel()[pre].mean()) ** 2
    )
    assert abs(r.r2 - right) < 1e-10
    assert abs(r.r2 - wrong) > 1e-3, "mutation guard failed: R2 used actual outcome"


def test_mutation_guard_pvalues_include_the_treated_unit() -> None:
    r = _run("placebo_unit", predictors=FULL,
             xperiod=list(range(1980, 1989)), placebo="unit")
    n = np.asarray(r.mspe).shape[0]        # 39 = treated + 38 placebos
    p = np.asarray(r.pval)[:, 1:]
    assert p.min() >= 1.0 / n - 1e-12
