"""``estat vif``'s model set, and the two exclusions that look like inclusions.

``_LINEAR_MODELS`` decides which estimators accept ``estat vif``. Two nearby families
are EXCLUDED, and both for documented reasons rather than oversight:

* ``ivregress`` -- ``ivregress_estat.ado:20-56`` dispatches only firststage / overid /
  endogenous / weakrobust / sbknown|sbsingle, then falls through to ``estat_default``,
  which has no ``vif``. An IV fit used to return exactly the OLS VIF (measured
  identical), which reads as a feature and is not one (round 102).
* ``newey`` -- ``newey_estat.ado:11-17`` handles only ``durbinalt`` and forwards the
  rest to ``estat_default``. Notably ``newey_p.ado:20`` DOES list ``XB Index STDP
  REsiduals``, so ``predict`` after ``newey`` works while ``estat vif`` does not: two
  different ado files, two different lists.

The four members are the estimators whose postestimation chain does reach ``vif.ado``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.vif import _LINEAR_MODELS


def _frame(n: int = 200, seed: int = 1910):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )


# ------------------------------------------------------------------ the membership
def test_the_set_is_exactly_four() -> None:
    assert _LINEAR_MODELS == frozenset({"regress", "reghdfe", "areg", "eivreg"})


def test_iv_is_not_a_member() -> None:
    for name in ("ivregress 2sls", "ivregress liml", "ivregress gmm"):
        assert name not in _LINEAR_MODELS


def test_newey_is_not_a_member() -> None:
    """``newey_estat.ado:11-17`` forwards everything but ``durbinalt`` to
    ``estat_default``, which has no ``vif``."""

    assert "newey" not in _LINEAR_MODELS


def test_the_exclusions_are_not_accidental_omissions() -> None:
    """Guard: the two names above must be absent from a set that DOES contain
    similarly named linear estimators.

    If the set were empty or degenerate, "not a member" would be trivially true.
    """

    assert len(_LINEAR_MODELS) >= 4
    assert "regress" in _LINEAR_MODELS


# ------------------------------------------------------------------- behaviour
def test_regress_accepts_vif() -> None:
    fitted = ws.OLS.from_formula("y ~ x1 + x2", _frame()).fit()

    rendered = fitted.estat_vif()

    assert len(rendered.names) == 2
    assert rendered.mean_vif > 0.0


def test_eivreg_accepts_vif() -> None:
    """``eivreg`` is a member, and its VIF must be computable, not just permitted."""

    fitted = ws.Eivreg.from_formula(
        "y ~ x1 + x2", _frame(), reliab={"x1": 0.9, "x2": 0.85}
    ).fit()

    rendered = fitted.estat_vif()

    assert set(rendered.names) == {"x1", "x2"}
    assert np.isfinite(rendered.mean_vif)


def test_iv_refuses_vif() -> None:
    rng = np.random.default_rng(1912)
    n = 200
    instrument = rng.normal(size=n)
    endogenous = 0.7 * instrument + rng.normal(size=n) * 0.3
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * endogenous + rng.normal(size=n)
    frame = pd.DataFrame(
        {"y": outcome, "x1": exog, "x2": endogenous, "z": instrument}
    )
    iv = ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z]", frame).fit()

    with pytest.raises(NotImplementedError):
        iv.estat_vif()


def test_newey_refuses_vif() -> None:
    """The other documented exclusion, checked at runtime as well as by membership."""

    # ``westata.newey`` is a FUNCTION that returns the fitted result directly -- there
    # is no separate ``.fit()`` step, unlike the class-based estimators.
    fitted = ws.newey("y ~ x1 + x2", _frame(), lag=2)

    with pytest.raises(NotImplementedError):
        fitted.estat_vif()


def test_newey_predict_still_works() -> None:
    """The contrast that makes the previous test meaningful.

    ``newey_p.ado:20`` lists prediction types, so ``predict`` succeeds while
    ``estat vif`` does not -- the two commands are served by different ado files.
    """

    fitted = ws.newey("y ~ x1 + x2", _frame(), lag=2)

    values = np.asarray(fitted.predict(kind="xb"), dtype=float)

    assert values.shape == (len(_frame()),)
    assert np.isfinite(values).all()


def test_the_vif_numbers_are_the_same_shape_for_both_members() -> None:
    """``regress`` and ``eivreg`` report the same two columns."""

    frame = _frame()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit().estat_vif()
    errors_in_variables = ws.Eivreg.from_formula(
        "y ~ x1 + x2", frame, reliab={"x1": 0.9, "x2": 0.85}
    ).fit().estat_vif()

    assert list(plain.names) == list(errors_in_variables.names)
    assert plain.to_frame().columns.tolist() == errors_in_variables.to_frame().columns.tolist()