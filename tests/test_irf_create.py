"""irf_create: a numeric .dta export whose five populated columns match Stata exactly.

C13-06 in the report says "irf create / .irf file system missing". Most of the numerical content
is in fact implemented and verified elsewhere (test_var_irf.py, test_var_fevd.py,
test_irf_table_layout.py). What was genuinely missing was writing the results to a FILE.

WHAT ROUND 144 ADDED: `VARResults.irf_create(path, irfname=..., step=...)` writes a Stata .dta
with Stata's exact 23-column schema and 44-row layout for a 2-variable, 11-step run.

VERIFIED against a live `irf create v1, step(10)` on macro_swatson.dta: irf, cirf, oirf, coirf
and fevd agree to 3.4e-13 or better, row for row in the same order.

THE HONEST LIMITATION, pinned so it is not mistaken for a full implementation: `pandas.to_stata`
cannot write dataset-level `_dta[]` characteristics, and Stata's `irf describe` / `irf table`
require ~17 of them (`_dta[irfnames]`, `v1_model`, `v1_tmin`, `v1_timevar`, `v1_endog`, ...).
Adding three by hand made `irf describe` list the name but `irf table` still returned r(498). So
this is a NUMERIC EXPORT readable by pandas, not a drop-in Stata .irf set. Unverified columns
(mse and every std*) are NaN rather than guessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

#: Measured from a live Stata `irf create v1, step(10)` file (first rows).
STATA_STEP0 = {
    ("dinf", "dinf"): {"irf": 1.0, "cirf": 1.0, "oirf": 0.013603, "coirf": 0.013603, "fevd": 0.0},
    ("unem", "dinf"): {"irf": 0.0, "cirf": 0.0, "oirf": -0.000188, "coirf": -0.000188, "fevd": 0.0},
}
STATA_STEP1 = {
    ("dinf", "dinf"): {"irf": -0.470944, "cirf": 0.529056, "oirf": -0.005984, "coirf": 0.007619, "fevd": 1.0},
    ("dinf", "unem"): {"fevd": 0.0},
    ("unem", "dinf"): {"fevd": 0.005822},
}


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    fit = ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()
    path = tmp_path_factory.mktemp("irf") / "ws.irf"
    produced = fit.irf_create(str(path), irfname="ws1", step=10)
    return produced, pd.read_stata(path)


def test_the_file_has_statas_schema_and_row_count(written) -> None:
    produced, reloaded = written
    assert produced.shape == (44, 23)
    assert reloaded.shape == (44, 23)
    assert list(reloaded.columns) == list(produced.columns)
    assert list(reloaded.columns)[:6] == ["irfname", "step", "irf", "cirf", "oirf", "coirf"]
    assert set(reloaded["irfname"]) == {"ws1"}
    assert list(reloaded["step"].unique()) == list(range(11))


def test_row_order_is_step_then_response_then_impulse(written) -> None:
    _, reloaded = written
    first_four = list(zip(reloaded["step"], reloaded["response"], reloaded["impulse"]))[:4]
    assert first_four == [
        (0, "dinf", "dinf"),
        (0, "dinf", "unem"),
        (0, "unem", "dinf"),
        (0, "unem", "unem"),
    ]


@pytest.mark.parametrize("step,expected", [(0, STATA_STEP0), (1, STATA_STEP1)])
def test_the_populated_columns_match_stata(written, step: int, expected: dict) -> None:
    _, reloaded = written
    subset = reloaded[reloaded["step"] == step]
    for (response, impulse), fields in expected.items():
        row = subset[(subset["response"] == response) & (subset["impulse"] == impulse)].iloc[0]
        for column, value in fields.items():
            assert float(row[column]) == pytest.approx(value, abs=5e-6), (
                step, response, impulse, column
            )


def test_cumulative_columns_are_running_sums(written) -> None:
    """The definition, independent of the Stata reference values."""

    _, reloaded = written
    for (response, impulse), group in reloaded.groupby(["response", "impulse"], sort=False):
        ordered = group.sort_values("step")
        assert np.allclose(ordered["cirf"].to_numpy(float), np.cumsum(ordered["irf"].to_numpy(float)))
        assert np.allclose(ordered["coirf"].to_numpy(float), np.cumsum(ordered["oirf"].to_numpy(float)))


def test_unverified_columns_are_nan_not_invented(written) -> None:
    """mse and every std* need standard errors we deliberately do not guess."""

    _, reloaded = written
    for column in ("mse", "stdirf", "stdcirf", "stdoirf", "stdcoirf", "stdfevd", "sirf", "dm", "cdm"):
        assert reloaded[column].isna().all(), column


def test_the_limitation_is_documented(written) -> None:
    """It must say this is a numeric export, not a Stata-readable .irf set.

    Pinned because the temptation is to quietly describe this as "irf create implemented", which
    would be false: Stata's own `irf table` cannot read the file without ~17 _dta[] characteristics.
    """

    import inspect

    fit = _fit_for_doc()
    doc = inspect.getdoc(type(fit).irf_create) or ""
    assert "LIMITATION" in doc
    assert "characteristics" in doc


def _fit_for_doc():
    frame = pd.read_stata(DATA, convert_categoricals=False)
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()
