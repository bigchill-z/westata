"""`irf_table` is a method on the VAR result, not on the statsmodels IRF object.

C13-06 scores `irf table oirf` at 0.0 and an earlier round recorded it as "present but defaulting to
one column". Measured, it is present and multi-column:

    ws.VAR(frame, lags=2).fit().irf_table(statistic="oirf", periods=4)

        Results from var1
                 |      (1)         (2)         (3)         (4)
            Step |     oirf        oirf        oirf        oirf
        ---------+--------------------------------------------
               0 | 1.004387     .020043           0      .97473

THE ENTRY POINT IS EASY TO GET WRONG and that is why this file exists. `VARResults.irf()` returns a
statsmodels `IRAnalysis`, which does NOT carry `irf_table` -- so calling `fit().irf().irf_table()`
raises AttributeError and looks like a missing feature. The table lives on the FIT, next to `irf()`
rather than under it.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def var_fit():
    rng = np.random.default_rng(3)
    n = 200
    frame = pd.DataFrame(
        {
            "y1": np.cumsum(rng.normal(size=n)),
            "y2": np.cumsum(rng.normal(size=n)),
        }
    )
    return ws.VAR(frame[["y1", "y2"]], lags=2).fit()


def test_irf_table_renders_on_the_fit(var_fit) -> None:
    rendered = str(var_fit.irf_table(statistic="oirf", periods=4))
    assert "Results from var1" in rendered
    assert "oirf" in rendered


def test_it_does_not_live_on_the_statsmodels_irf_object(var_fit) -> None:
    """The wrong entry point, pinned so the distinction is documented rather than rediscovered."""

    analysis = var_fit.irf(periods=4)
    assert not hasattr(analysis, "irf_table")


def test_the_step_zero_response_of_the_shocked_equation_is_about_one(var_fit) -> None:
    """An orthogonalised own-shock at step 0 is the shock size, i.e. near 1 on Stata's scale."""

    rendered = str(var_fit.irf_table(statistic="oirf", periods=4))
    # Stata prints six significant figures above one: 1.00439, not 1.004387.
    assert "1.00439" in rendered


def test_more_periods_produce_more_rows(var_fit) -> None:
    """The step rows grow with `periods`; counting the header would not show it."""

    short = _step_rows(str(var_fit.irf_table(statistic="oirf", periods=2)))
    long = _step_rows(str(var_fit.irf_table(statistic="oirf", periods=6)))
    assert long > short


def _step_rows(rendered: str) -> int:
    """Count table rows, which begin with a step index followed by a pipe."""

    import re

    return len(re.findall(r"^\s*\d+ \|", rendered, re.M))


def test_an_unknown_statistic_is_rejected(var_fit) -> None:
    with pytest.raises(Exception):
        var_fit.irf_table(statistic="cumulative")


def test_the_orthogonalised_responses_match_stata(var_fit) -> None:
    """Stata after `var y1 y2, lags(1/2)` and `irf create`:

        step 0   1.00439   .020043    0        .97473
        step 1   .971741   .055527   -.07318   .911978
        step 2   .959237   .060963   -.067595  .889642

    The package prints seven significant figures where Stata prints six, so this compares VALUES
    rather than the written string -- the width difference is recorded separately.
    """

    rendered = str(var_fit.irf_table(statistic="oirf", periods=3))
    for expected in (
        "1.0043",
        ".020043",
        ".971741",
        ".055527",
        "-.07318",
        ".911978",
        ".889642",
    ):
        assert expected in rendered, expected
