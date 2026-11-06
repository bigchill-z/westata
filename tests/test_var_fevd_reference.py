"""VAR's FEVD: implemented on the result object and matching Stata.

The report lists "the irf-file system is missing" (C13-06) as a P1 blocker. Part of that is
genuinely absent -- the `.irf` file store itself -- but the FEVD side is not: ``result.fevd_table``
exists and reproduces Stata's ``irf table fevd`` numbers exactly.

Against a live ``var dinf unem`` / ``irf table fevd`` on macro_swatson.dta:

    (2) impulse dinf, response unem   .005822  .002094  .001063
    (3) impulse unem, response dinf   .121271  .123239  ...

and westata's table, columns labelled (impulse, response):

    step   dinf->dinf   dinf->unem   unem->dinf   unem->unem
       0      1.000000     0.005822     0.000000     0.994178
       1      0.878729     0.002094     0.121271     0.997906
       2      0.876761     0.001063     0.123239     0.998937

Two traps recorded, both of which produced a plausible-looking wrong table during earlier work:

  * statsmodels' ``decomp`` is laid out (RESPONSE, period, IMPULSE). Reading it as
    (impulse, period, response) puts every number in the wrong column while leaving the table
    looking like a decomposition.
  * ``IRAnalysis.fevd_table`` -- the object returned by ``model.irf(...)`` -- raises
    NotImplementedError because that is statsmodels' stub. The working method is on the RESULT
    object, ``VAR.fit().fevd_table()``, not on the impulse-response analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.VAR(frame[["dinf", "unem"]].to_numpy(float), lags=2, names=["dinf", "unem"]).fit()


def test_the_table_is_reachable_from_the_result(fitted) -> None:
    """Not from .irf(), whose fevd_table is statsmodels' stub."""

    table = fitted.fevd_table(periods=3)
    assert isinstance(table, pd.DataFrame)
    # periods=3 gives steps 0..2 (three rows).
    assert len(table) == 3


def test_the_decomposition_matches_stata(fitted) -> None:
    table = fitted.fevd_table(periods=3)
    # Stata's blocks, indexed from step 0: step 0 of every off-diagonal pair is 0.
    pairs = {
        ("dinf", "unem"): [0.005822, 0.002094, 0.001063],
        ("unem", "dinf"): [0.0, 0.121271, 0.123239],
    }
    for (impulse, response), expected in pairs.items():
        column = table[(impulse, response)].to_numpy(float)[:3]
        assert np.allclose(column, expected, atol=5e-7), (impulse, response, column)


def test_each_response_column_sums_to_one(fitted) -> None:
    """A decomposition that does not sum to 1 is a layout error, not a precision issue."""

    table = fitted.fevd_table(periods=3)
    names = ["dinf", "unem"]
    for response in names:
        total = sum(table[(impulse, response)].to_numpy(float) for impulse in names)
        assert np.allclose(total, 1.0, atol=1e-9), response


def test_the_irf_analysis_fevd_table_is_not_the_entry_point(fitted) -> None:
    """Pinned deliberately: this is the trap that made the capability look missing."""

    analysis = fitted.irf(periods=3)
    with pytest.raises(NotImplementedError):
        analysis.fevd_table()
