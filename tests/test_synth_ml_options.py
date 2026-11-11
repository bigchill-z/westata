"""RED tests: forward Stata's `ml` optimisation options on the nested path.

``synth.ado:41`` accepts arbitrary ``ml maximize`` options via ``*`` and
``:434`` parses them with ``mlopts``; ``wrapml`` (`:846-895`) passes them to
``ml model d0 ... maximize``.  Live-Stata measurement (2026-11-11) shows
``technique(bfgs)`` and ``iterate(2)`` CHANGE e(RMSPE), while
``nonrtolerance`` / ``difficult`` / ``search(on)`` do not change the
single-start result.

Our Powell-based outer search cannot switch Stata's NR/DFP/BFGS techniques,
so we ACCEPT, VALIDATE and RECORD every option, map the ones with a faithful
semantic counterpart (iterate -> maxiter, nrtolerance -> convergence tol),
and HONESTLY document that technique() is recorded but cannot alter the
numerical result.
"""

from __future__ import annotations

import numpy as np
import pytest

from westata.estimators.synth import SynthResult, synth
from tests.test_synth_extended import _panel


def _base(**kw):
    args = dict(
        data=_panel(), unit="unit", time="year", depvar="y",
        predictors=["x1", "x2"], trunit=1, trperiod=2005,
    )
    args.update(kw)
    return synth(**args)


ML_OPTIONS = [
    {"technique": "bfgs"},
    {"iterate": 5},
    {"nrtolerance": 1e-3},
    {"nonrtolerance": True},
    {"difficult": True},
    {"search": "on"},
    {"trace": True},
]


@pytest.mark.parametrize("opt", ML_OPTIONS, ids=[list(o)[0] for o in ML_OPTIONS])
def test_ml_options_are_accepted_on_nested_path(opt) -> None:
    """Stata accepts these via `*`; westata must not raise UnsupportedError."""

    res = _base(nested=True, **opt)
    assert isinstance(res, SynthResult)
    assert np.isfinite(res.RMSPE)


def test_ml_options_are_recorded_for_audit() -> None:
    res = _base(nested=True, technique="bfgs", iterate=5, nrtolerance=1e-3)
    s = res.optimizer_settings
    assert s["technique"] == "bfgs"
    assert s["iterate"] == 5
    assert s["nrtolerance"] == 1e-3


def test_iterate_maps_to_the_outer_search_budget() -> None:
    """iterate() has a faithful counterpart: the optimiser's iteration cap."""

    a = _base(nested=True, iterate=1)
    b = _base(nested=True, iterate=200)
    # Both must be valid results; a tighter budget must not crash.
    assert np.isfinite(a.RMSPE) and np.isfinite(b.RMSPE)
    assert a.optimizer_settings["iterate"] == 1
    assert b.optimizer_settings["iterate"] == 200


def test_invalid_ml_option_values_are_rejected() -> None:
    for bad in ({"iterate": 0}, {"iterate": -3}, {"nrtolerance": 0},
                {"technique": 123}, {"search": 5}):
        with pytest.raises((ValueError, TypeError)):
            _base(nested=True, **bad)


def test_ml_options_on_non_nested_path_are_rejected() -> None:
    """Stata's `*` options only matter for nested; elsewhere they'd be lies."""

    with pytest.raises(Exception):
        _base(technique="bfgs")


def test_technique_is_documented_as_recorded_but_not_numerically_active() -> None:
    """Guard against silently pretending technique() changes the numbers."""

    a = _base(nested=True, technique="nr")
    b = _base(nested=True, technique="bfgs")
    np.testing.assert_allclose(a.V_matrix, b.V_matrix, atol=1e-12)
    assert a.optimizer_settings["technique"] == "nr"
    assert b.optimizer_settings["technique"] == "bfgs"


def test_numerically_inactive_options_are_flagged_on_the_result() -> None:
    """A caller must be able to see which accepted options do NOT move numbers."""

    r = _base(nested=True, technique="bfgs", sigf=7, margin=0.05)
    assert "technique" in r.numerically_inactive
    assert "sigf" in r.numerically_inactive
    assert "margin" in r.numerically_inactive
    plain = _base(nested=True)
    assert plain.numerically_inactive == ()


def test_qp_plugin_settings_do_not_change_the_result() -> None:
    """sigf/margin/bound are recorded, not faked.  Documented E2 boundary."""

    a = _base(nested=True)
    for kw in ({"sigf": 7}, {"margin": 0.05}, {"bound": 1}):
        b = _base(nested=True, **kw)
        np.testing.assert_allclose(a.V_matrix, b.V_matrix, atol=1e-10)
        np.testing.assert_allclose(a.RMSPE, b.RMSPE, atol=1e-8)
