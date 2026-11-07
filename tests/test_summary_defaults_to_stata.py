"""summary() defaults to Stata's layout.

``style`` used to default to ``"statsmodels"`` on ``StataResults.summary`` and on
``render_summary``. Both now default to ``"stata"``: reproducing Stata's table is what this package
exists to do, so a caller who just prints a fit should get the Stata table, and the
statsmodels-flavoured one is the explicit request.

Three of the five ``summary`` methods in the package already defaulted to ``"stata"`` (the systems
and time-series result classes); only the shared ``StataResults`` path and the module-level
renderer lagged behind, which is why a plain ``regress`` fit printed a statsmodels table while a
``var`` fit printed a Stata one.

The guards below pin both halves: the default is Stata, and the statsmodels table is still
reachable by naming it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws
from westata.results.summary import render_summary

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr", frame).fit()


def test_summary_defaults_to_the_stata_layout(fitted) -> None:
    """The default table carries Stata's header, not statsmodels' labels."""

    default = str(fitted.summary())
    assert "Source |" in default
    assert "Number of obs" in default
    # statsmodels-only spellings must be absent.
    for statsmodels_marker in ("No. Observations:", "Df Residuals:", "coef"):
        assert statsmodels_marker not in default


def test_the_default_equals_the_explicit_stata_style(fitted) -> None:
    assert str(fitted.summary()) == str(fitted.summary(style="stata"))


def test_the_statsmodels_style_is_still_available(fitted) -> None:
    explicit = str(fitted.summary(style="statsmodels"))
    assert "No. Observations:" in explicit or "coef" in explicit
    assert explicit != str(fitted.summary())


def test_the_module_level_renderer_also_defaults_to_stata(fitted) -> None:
    assert render_summary(fitted) == render_summary(fitted, style="stata")


def test_the_other_result_families_were_already_stata() -> None:
    """The systems and time-series summaries always defaulted to stata; this is why the change is
    about the SHARED path rather than about every estimator."""

    import inspect

    from westata.estimators import systems, tseries

    for module in (systems, tseries):
        for name, member in inspect.getmembers(module, inspect.isclass):
            method = member.__dict__.get("summary")
            if method is None:
                continue
            parameters = inspect.signature(method).parameters
            if "style" in parameters:
                assert parameters["style"].default == "stata", name
