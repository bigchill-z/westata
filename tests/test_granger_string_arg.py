"""Pin the ``granger_causality`` string-argument bug and ``vargranger``'s numbers.

THE BUG. ``VARResults.granger_causality`` passed ``list(causing)`` to statsmodels, and
``list("unem")`` is ``['u','n','e','m']`` -- so calling it with a plain variable NAME raised

    ValueError: 'u' is not in list

from inside statsmodels' ``util.get_index``. Every single-covariate call failed, which is
exactly the shape ``vargranger`` uses most. It is fixed by only splinter a list when the
argument already IS a list.

PARAMETER SEMANTICS, measured against Stata: ``granger_causality(a, b)`` reports the test in
equation ``a`` excluding ``b``. Stata's ``vargranger`` row "Equation=dinf / Excluded=unem ->
48.123" is reproduced by ``granger_causality("dinf", "unem")``. The parameter is named
``causing`` but holds the EQUATION -- swapping the two silently exchanges 48.123 and 9.009,
both of which look like plausible statistics.
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
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()


def test_a_bare_string_argument_does_not_raise(fitted) -> None:
    """This raised ValueError before the fix; list('unem') splits into characters."""

    outcome = fitted.granger_causality("dinf", "unem")
    assert float(outcome.test_statistic) == pytest.approx(48.123, abs=5e-4)


def test_the_other_direction_matches_stata(fitted) -> None:
    outcome = fitted.granger_causality("unem", "dinf")
    assert float(outcome.test_statistic) == pytest.approx(9.009, abs=5e-4)


def test_a_list_argument_still_works(fitted) -> None:
    """The multi-covariate path must keep working after the fix."""

    outcome = fitted.granger_causality("dinf", ["unem"])
    assert float(outcome.test_statistic) == pytest.approx(48.123, abs=5e-4)


def test_the_two_directions_are_not_symmetric(fitted) -> None:
    """Guards the semantics: 48.123 vs 9.009 must not collapse to the same value."""

    forward = float(fitted.granger_causality("dinf", "unem").test_statistic)
    backward = float(fitted.granger_causality("unem", "dinf").test_statistic)
    assert abs(forward - backward) > 30.0


def test_vargranger_reports_statas_values_and_rows(fitted) -> None:
    """The numbers and the row-to-equation mapping, which is what Stata's table conveys."""

    from westata.postestimation.vargranger import vargranger

    table = vargranger(fitted)
    pairs = {(row.equation, row.excluded): row.chi2 for row in table.rows}
    assert pairs[("dinf", "unem")] == pytest.approx(48.123, abs=5e-4)
    assert pairs[("dinf", "ALL")] == pytest.approx(48.123, abs=5e-4)
    assert pairs[("unem", "dinf")] == pytest.approx(9.009, abs=5e-4)
    assert table.equations == ("dinf", "unem")


def test_vargranger_renders_the_heading_and_all_four_data_rows(fitted) -> None:
    from westata.postestimation.vargranger import vargranger

    rendered = str(vargranger(fitted))
    assert "Granger causality Wald tests" in rendered
    assert rendered.count("ALL") == 2
    assert "48.123" in rendered
    assert "9.009" in rendered
