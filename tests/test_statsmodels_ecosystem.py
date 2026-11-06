"""Statsmodels-ecosystem entry points added in round 18 (v2 report section 6).

The report's audit found 12 of 34 contracts unmet and grouped them into five classes. Three are
closed here, each checked against statsmodels itself rather than against a remembered spec:

    res.t_test('x1=0')          t = 7.1974807336   statsmodels 7.1974807336
    res.f_test('x1=0, x2=0')    F = 30.9297891742  statsmodels 30.9297891742
    res.get_influence()['hat_matrix_diag']         bit-identical to OLSInfluence

ONE THING THIS DOES NOT FIX, and the reason matters: ``anova_lm(res)`` still raises. It reads
``res.model.data.model_spec`` -- patsy's formula terms -- and rebuilds the table from them
(anova.py:78-95), so it wants statsmodels' internal model specification, not its public result
surface. ``res.model`` is provided for consumers that only need endog/exog/exog_names/fit().

Deliberately NOT touched: ``cov_params()`` returns an ndarray because the user ruled it out of
scope, and ``resid``/``fittedvalues`` stay ndarrays because changing them is a compatibility
decision, not a defect fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

sm = pytest.importorskip("statsmodels.api")


@pytest.fixture(scope="module")
def paired():
    rng = np.random.default_rng(7)
    n = 100
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    frame["y"] = 1.0 + 0.8 * frame["x1"] - 0.5 * frame["x2"] + rng.normal(size=n)
    ours = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    design = sm.add_constant(frame[["x1", "x2"]].to_numpy())
    theirs = sm.OLS(frame["y"].to_numpy(), design).fit()
    return ours, theirs


def test_t_test_matches_statsmodels(paired) -> None:
    ours, theirs = paired
    mine = ours.t_test("x1=0")
    reference = theirs.t_test("x1=0")
    assert float(mine.tvalue) == pytest.approx(float(reference.tvalue), rel=1e-12)
    assert float(mine.pvalue) == pytest.approx(float(reference.pvalue), rel=1e-9)


def test_f_test_matches_statsmodels(paired) -> None:
    ours, theirs = paired
    mine = ours.f_test("x1=0, x2=0")
    reference = theirs.f_test("x1=0, x2=0")
    assert float(mine.fvalue) == pytest.approx(float(reference.fvalue), rel=1e-12)
    assert float(mine.pvalue) == pytest.approx(float(reference.pvalue), rel=1e-9)


def test_a_single_restriction_reports_t_and_f_aliases(paired) -> None:
    """Both names must exist: callers read whichever suits their test."""

    ours, _ = paired
    contrast = ours.t_test("x1=0")
    assert float(contrast.fvalue) == pytest.approx(float(contrast.tvalue) ** 2, rel=1e-12)
    assert contrast.q == 1.0


def test_leverage_matches_statsmodels(paired) -> None:
    ours, theirs = paired
    mine = ours.get_influence()
    reference = theirs.get_influence()
    assert np.allclose(
        mine["hat_matrix_diag"], reference.hat_matrix_diag, atol=1e-12
    )


def test_cooks_distance_matches_statsmodels(paired) -> None:
    ours, theirs = paired
    mine = ours.get_influence()["cooks_distance"][0]
    reference = theirs.get_influence().cooks_distance[0]
    assert np.allclose(mine, reference, atol=1e-10)


def test_the_model_attribute_exposes_what_consumers_read(paired) -> None:
    ours, _ = paired
    model = ours.model
    assert model.exog.shape == (100, 3)
    assert list(model.exog_names) == ["Intercept", "x1", "x2"]
    assert model.endog.size == 100
    assert model.fit() is ours


def test_anova_lm_is_still_expected_to_fail(paired) -> None:
    """Pinned so the documented limit cannot silently start passing in a misleading way."""

    ours, _ = paired
    from statsmodels.stats.anova import anova_lm

    with pytest.raises(AttributeError, match="model_spec"):
        anova_lm(ours)


def test_cov_params_is_unchanged_by_design() -> None:
    """The user ruled this out of scope; this test records the decision, not a defect."""

    frame = pd.DataFrame({"y": np.arange(1, 21.0), "x": np.arange(1, 21.0) ** 1.5})
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    assert isinstance(fitted.cov_params(), np.ndarray)
