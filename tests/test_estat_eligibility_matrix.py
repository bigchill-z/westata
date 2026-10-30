"""Which ``estat`` subcommands each estimator offers -- checked against the dispatch.

An eligibility matrix is easy to get wrong in both directions at once, and rounds 79-84
showed that inferring it from ``e()`` contents errs exactly that way. The authority is
the ``*_estat.ado`` dispatch chain:

* ``logit_estat.ado`` handles ``classification``/``gof``/``auc`` and delegates the rest
  to ``estat_default``;
* ``probit_estat.ado`` is the same shape;
* ``estat_default.ado`` contains no ``vif`` at all.

So ``estat vif`` after a binary or count model must refuse -- not because the numbers
would be hard, but because Stata never dispatches it there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

BASE = Path(r"D:\Program Files\StataNow19\ado\base")


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(727)
    n = 300
    latent = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": latent,
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "yb": (latent > 0).astype(int),
            "ycnt": rng.poisson(3.0, size=n),
        }
    )


UNIVERSAL = ("estat_summarize", "estat_vce", "estat_ic")


@pytest.mark.parametrize(
    "builder",
    [
        lambda f: ws.OLS.from_formula("y ~ x1 + x2", f).fit(),
        lambda f: ws.Logit.from_formula("yb ~ x1", f).fit(),
        lambda f: ws.Poisson.from_formula("ycnt ~ x1", f).fit(),
    ],
    ids=["OLS", "Logit", "Poisson"],
)
@pytest.mark.parametrize("method", UNIVERSAL)
def test_the_universal_commands_work_everywhere(frame, builder, method) -> None:
    """``summarize``/``vce``/``ic`` come from ``estat_default``, so every fit has them."""

    result = builder(frame)

    rendered = getattr(result, method)()

    assert rendered is not None
    assert str(rendered)


@pytest.mark.parametrize(
    "builder",
    [
        lambda f: ws.Logit.from_formula("yb ~ x1", f).fit(),
        lambda f: ws.Poisson.from_formula("ycnt ~ x1", f).fit(),
    ],
    ids=["Logit", "Poisson"],
)
def test_vif_refuses_outside_the_linear_models(frame, builder) -> None:
    """``estat vif`` is not dispatched for these, so a number would be an invention."""

    with pytest.raises(NotImplementedError):
        builder(frame).estat_vif()


def test_vif_works_for_the_linear_models(frame) -> None:
    """The positive side of the same rule."""

    assert ws.OLS.from_formula("y ~ x1 + x2", frame).fit().estat_vif().mean_vif > 0.0


def test_the_dispatch_files_confirm_the_matrix() -> None:
    """Pin the evidence so a future addition has to argue with the ado."""

    logit_estat = (BASE / "l" / "logit_estat.ado").read_text(
        encoding="utf-8", errors="ignore"
    )
    estat_default = (BASE / "e" / "estat_default.ado").read_text(
        encoding="utf-8", errors="ignore"
    )

    # ``logit_estat`` handles three subcommands itself and forwards the rest...
    for handled in ("classification", "gof", "auc"):
        assert handled in logit_estat, handled
    # ...but never dispatches ``vif``.
    assert "vif" not in logit_estat
    # And the fallback program has no ``vif`` either, so nothing reaches vif.ado.
    assert "vif" not in estat_default


def test_iv_is_not_a_linear_model_for_vif(frame) -> None:
    """2SLS reports ``e(r2)`` and fit statistics, yet has no ``estat vif``.

    ``ivregress_estat.ado:20-56`` never dispatches vif, which is why an IV fit used to
    return exactly the OLS VIF -- a plausible number for a model it was not fitted on
    (round 102).
    """

    rng = np.random.default_rng(733)
    n = 300
    instrument = rng.normal(size=n)
    endog = 0.7 * instrument + rng.normal(size=n) * 0.3
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * endog + rng.normal(size=n)
    iv_frame = pd.DataFrame(
        {"y": outcome, "x1": exog, "x2": endog, "z": instrument}
    )

    iv = ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z]", iv_frame).fit()
    ols = ws.OLS.from_formula("y ~ x1 + x2", iv_frame).fit()

    # The OLS model has a VIF; the IV model must not report the same one.
    assert ols.estat_vif().mean_vif > 0.0
    with pytest.raises(NotImplementedError):
        iv.estat_vif()

    ivregress_estat = (BASE / "i" / "ivregress_estat.ado").read_text(
        encoding="utf-8", errors="ignore"
    )
    assert "vif" not in ivregress_estat