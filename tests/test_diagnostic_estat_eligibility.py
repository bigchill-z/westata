"""The three regression-diagnostic ``estat`` commands share one eligibility rule.

``regress_postestimation.sthlp:42-44`` lists ``estat hettest``, ``estat imtest`` and
``estat ovtest`` as REGRESS postestimation commands. They all ask questions about a
linear regression's residuals:

* ``hettest``   -- is the residual variance constant?
* ``imtest``    -- does the information matrix equality hold?
* ``ovtest``    -- does the mean function have the functional form assumed?

None of the three has a meaning after a binary or count model, whose "residuals" are
not the raw ``y - xb`` a linear fit produces. Each is implemented in its own module, so
nothing structurally forces them to agree -- this file checks the dispatch decision
directly, for all three, on the same fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DIAGNOSTICS = ("estat_hettest", "estat_ovtest", "estat_imtest")


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(1900)
    n = 200
    latent = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": 1.0 + 0.5 * latent + rng.normal(size=n),
            "x1": latent,
            "x2": rng.normal(size=n),
            "yb": (latent > 0).astype(int),
            "yc": rng.poisson(3.0, size=n),
        }
    )


@pytest.fixture(scope="module")
def linear(frames):
    return ws.OLS.from_formula("y ~ x1 + x2", frames).fit()


@pytest.fixture(scope="module")
def binary(frames):
    return ws.Logit.from_formula("yb ~ x1", frames).fit()


@pytest.fixture(scope="module")
def count(frames):
    return ws.Poisson.from_formula("yc ~ x1", frames).fit()


@pytest.mark.parametrize("method", DIAGNOSTICS)
def test_all_three_work_after_a_linear_regression(linear, method) -> None:
    """``regress_postestimation.sthlp:42-44`` defines them for ``regress``."""

    rendered = getattr(linear, method)()

    assert rendered is not None
    assert str(rendered)


@pytest.mark.parametrize("method", DIAGNOSTICS)
def test_none_of_them_work_after_logit(binary, method) -> None:
    """A binary fit has no linear residual to diagnose."""

    with pytest.raises((ValueError, NotImplementedError)):
        getattr(binary, method)()


@pytest.mark.parametrize("method", DIAGNOSTICS)
def test_none_of_them_work_after_poisson(count, method) -> None:
    with pytest.raises((ValueError, NotImplementedError)):
        getattr(count, method)()


def test_the_three_agree_on_every_model(linear, binary, count) -> None:
    """Guard: the eligibility must be UNIFORM across the three.

    If one of them accepted a binary fit while the others refused, that model would be
    the odd one out and the matrix below would be wrong. Asserting uniformity is what
    makes the per-method tests above a statement about the family rather than three
    unrelated facts.
    """

    for fitted in (linear, binary, count):
        outcomes = []
        for method in DIAGNOSTICS:
            try:
                getattr(fitted, method)()
            except (ValueError, NotImplementedError):
                outcomes.append(False)
            else:
                outcomes.append(True)
        assert len(set(outcomes)) == 1, (fitted.model_name, outcomes)


def test_the_linear_side_is_genuinely_true(linear) -> None:
    """Guard: if the linear fit refused too, the uniformity test proves nothing."""

    accepted = 0
    for method in DIAGNOSTICS:
        try:
            getattr(linear, method)()
        except (ValueError, NotImplementedError):
            continue
        accepted += 1

    assert accepted == len(DIAGNOSTICS)


def test_the_refusals_name_the_estimator(binary) -> None:
    """A refusal that says only "not available" cannot be acted on."""

    with pytest.raises((ValueError, NotImplementedError)) as caught:
        binary.estat_hettest()

    message = str(caught.value).lower()
    assert "logit" in message or "regress" in message or "linear" in message


def test_hettest_reports_a_statistic_and_a_p_value(linear) -> None:
    """The output has to carry the two numbers a reader needs."""

    rendered = linear.estat_hettest()
    text = str(rendered)

    assert "chi2" in text.lower()
    assert "Prob" in text


def test_ovtest_reports_an_f_statistic(linear) -> None:
    """Ramsey's test is an F test on added powers of the fitted values."""

    text = str(linear.estat_ovtest())

    assert "F(" in text or "Prob > F" in text