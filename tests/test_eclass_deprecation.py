"""R-N5: e-class function wrappers are deprecated for one release.

They remain fully functional (same numbers as the class entry point) but emit a
DeprecationWarning pointing at the class. `prais_winsten` is deliberately NOT deprecated:
it is a low-level numeric solver (ndarray in, tuple out) with no equivalent class API.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _panel():
    return pd.DataFrame(
        {
            "unit": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
            "time": list(range(5)) * 3,
            "y": [1.0, 1.1, 0.9, 1.2, 1.0, 2.0, 2.1, 1.9, 2.2, 2.0, 1.5, 1.6, 1.4, 1.7, 1.5],
            "x": np.arange(15, dtype=float) / 10,
        }
    )


def test_all_five_wrappers_warn_with_class_alternative():
    """Each deprecated wrapper warns and names a concrete class entry point."""

    frame = _panel()
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 60)
    y = 1 + 2 * x + rng.normal(scale=0.1, size=60)

    cases = [
        (
            "newey_west",
            lambda: ws.newey_west("y ~ x", frame, lag=1),
            "Newey.from_formula",
        ),
        ("prais", lambda: ws.prais(frame, "y ~ x"), "Prais.from_formula"),
        (
            "synth",
            lambda: ws.synth(
                frame,
                unit="unit",
                time="time",
                depvar="y",
                predictors=["x"],
                trunit="a",
                trperiod=4,
            ),
            "Synth.from_variables",
        ),
        (
            "synth2",
            lambda: ws.synth2(
                frame,
                unit="unit",
                time="time",
                depvar="y",
                predictors=["x"],
                trunit="a",
                trperiod=4,
            ),
            "Synth2.from_variables",
        ),
        ("rd_robust", lambda: ws.rd_robust(y, x, bandwidth=0.5), "RDRobust.from_variables"),
    ]

    for name, call, alternative in cases:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        dep = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any(name in m for m in dep), f"{name} did not warn: {dep}"
        assert any(alternative in m for m in dep), f"{name} did not name {alternative}: {dep}"


def test_deprecated_wrappers_still_match_the_class_numbers():
    """Deprecation must not change results: wrapper and class share one implementation."""

    frame = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wrapper = ws.prais(frame, "y ~ x")
    klass = ws.Prais.from_formula("y ~ x", frame).fit()
    assert np.allclose(wrapper.params.values, klass.params.values)

    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 60)
    y = 1 + 2 * x + rng.normal(scale=0.1, size=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wrapper = ws.rd_robust(y, x, bandwidth=0.5)
    klass = ws.RDRobust.from_variables("y", "x", pd.DataFrame({"y": y, "x": x}), bandwidth=0.5).fit()
    assert np.isclose(wrapper.tau, klass.tau)


def test_prais_winsten_is_not_deprecated():
    """It is a low-level numeric solver, not an e-class wrapper -- no class equivalent."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rng = np.random.default_rng(1)
        exog = np.column_stack([np.ones(30), rng.normal(size=30)])
        endog = 0.5 * exog[:, 0] + rng.normal(scale=0.3, size=30)
        beta, rho, iterations, trace = ws.prais_winsten(exog, endog)
    dep = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not any("prais_winsten" in m for m in dep)
    assert len(trace) >= 1
