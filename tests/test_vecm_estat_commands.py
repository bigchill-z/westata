"""``vec``'s postestimation commands, from ``vec_postestimation.sthlp:44-49``.

    INCLUDE help post_estatic
    INCLUDE help post_estatsum
    INCLUDE help post_estatvce
    INCLUDE help post_estimates
    INCLUDE help post_etable
    INCLUDE help post_forecast

``VECMResults`` is a standalone dataclass, so none of the ``estat`` commands existed --
the same gap ``VARResults`` had before round 191.

Two are implemented; ``estat vce`` is REFUSED with a reason.  A VECM's parameters live in
three blocks of different shapes (``alpha`` k x r, ``beta`` k x r identified only up to a
normalisation, ``gamma`` k x (p-1) x k) and statsmodels exposes no flattened vector
matching ``cov_params_default``'s rows.  Labelling that matrix would require choosing a
normalisation for ``beta``, and a matrix labelled under the wrong normalisation is worse
than no matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws


def _cointegrated(n: int = 300, seed: int = 2400):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(size=n))
    other = common + rng.normal(size=n) * 0.5
    return np.column_stack([other, common])


@pytest.fixture(scope="module")
def vecm_fit():
    return ws.VECM(
        _cointegrated(), lags=2, coint_rank=1, names=["y", "x"]
    ).fit()


# ------------------------------------------------------------------- estat ic
def test_estat_ic_works(vecm_fit) -> None:
    """``vec_postestimation.sthlp:44`` lists it."""

    rendered = vecm_fit.estat_ic()

    assert rendered is not None
    assert str(rendered)


def test_estat_ic_names_the_command(vecm_fit) -> None:
    """The Model column must say ``vec``, not a dot."""

    rendered = str(vecm_fit.estat_ic())

    assert "vec" in rendered


def test_estat_ic_reports_the_criteria(vecm_fit) -> None:
    text = str(vecm_fit.estat_ic())

    for column in ("AIC", "BIC", "AICc", "CAIC"):
        assert column in text, column


def test_the_parameter_count_is_e_k(vecm_fit) -> None:
    """A VECM's penalty count is ``e(k_rank)``, NOT ``e(k)``.

    ``vec.sthlp:339`` calls ``e(k_rank)`` "number of unconstrained parameters"; ``vec``
    stores no ``e(k)`` at all.  The first version of this method passed ``e(k)`` (the
    unrestricted count), which made ``estat ic`` disagree with the ``e(aic)`` macro on
    the SAME fit.
    """

    rendered = vecm_fit.estat_ic()

    assert f"{float(vecm_fit.e['k_rank']):>8.0f}" in str(rendered)


def test_the_df_is_the_k_rank(vecm_fit) -> None:
    assert vecm_fit.estat_ic().df == float(vecm_fit.e["k_rank"])


def test_the_df_is_not_the_unrestricted_count(vecm_fit) -> None:
    """Guard: the two counts differ on this fit, so the previous test is decisive."""

    assert float(vecm_fit.e["k"]) != float(vecm_fit.e["k_rank"])


def test_estat_ic_agrees_with_the_aic_macro(vecm_fit) -> None:
    """``e(aic)`` is per observation; the table reports the total, so they must satisfy
    ``table_aic == e(aic) * N``.

    With the wrong ``df`` the two disagreed, which is what exposed the mistake.
    """

    table = vecm_fit.estat_ic()

    assert float(table.aic) == pytest.approx(
        float(vecm_fit.e["aic"]) * float(vecm_fit.e["N"]), rel=1e-10
    )


def test_estat_ic_agrees_with_the_bic_macro(vecm_fit) -> None:
    table = vecm_fit.estat_ic()

    assert float(table.bic) == pytest.approx(
        float(vecm_fit.e["sbic"]) * float(vecm_fit.e["N"]), rel=1e-10
    )


# -------------------------------------------------------------- estat summarize
def test_estat_summarize_works(vecm_fit) -> None:
    """``vec_postestimation.sthlp:45`` lists it."""

    rendered = vecm_fit.estat_summarize()

    assert rendered is not None
    assert "Number of obs" in str(rendered)


def test_the_summarized_sample_is_the_estimation_sample(vecm_fit) -> None:
    """``vec`` consumes rows to form the differences, so ``N`` is smaller than the
    observed series."""

    rendered = vecm_fit.estat_summarize()

    assert rendered.n_obs == float(vecm_fit.e["N"])
    assert rendered.n_obs < float(np.asarray(vecm_fit.raw.model.endog).shape[0])


def test_the_variables_appear_in_the_estimation_order(vecm_fit) -> None:
    assert list(vecm_fit.estat_summarize().table.index) == list(vecm_fit.names)


# --------------------------------------------------------------- estat vce
def test_estat_vce_is_refused_with_a_reason(vecm_fit) -> None:
    """``vec_postestimation.sthlp:46`` lists it, but it cannot be labelled honestly."""

    with pytest.raises(NotImplementedError) as caught:
        vecm_fit._vce_view()

    message = str(caught.value)
    assert "normalisation" in message
    assert "beta" in message
    # And the message names what DOES work.
    assert "estat ic" in message


def test_the_refusal_is_specific_to_vce(vecm_fit) -> None:
    """Guard: the two implemented commands must not be caught by the same refusal."""

    assert str(vecm_fit.estat_ic())
    assert str(vecm_fit.estat_summarize())


# ------------------------------------------------------- the underlying blocks
def test_the_covariance_blocks_have_different_shapes(vecm_fit) -> None:
    """Why a flat parameter vector cannot be built: three blocks, three shapes.

    Measured on this lags(2) fit: alpha and beta are 2x1 each, while the covariance is
    10x10.  The counts do not reconcile through any single flattening -- the VECM packs
    alpha, the normalised beta and the short-run gamma into one covariance, and the
    deterministic terms add more.
    """

    raw = vecm_fit.raw
    alpha = np.asarray(raw.alpha)
    beta = np.asarray(raw.beta)
    covariance = np.asarray(raw.cov_params_default)

    assert alpha.shape == beta.shape == (2, 1)
    assert covariance.ndim == 2
    assert covariance.shape[0] == covariance.shape[1]
    # The covariance is far larger than the alpha block alone, so it is not simply that.
    assert covariance.shape[0] > alpha.size * 2


def test_beta_is_only_identified_up_to_normalisation(vecm_fit) -> None:
    """``vec.sthlp:49`` lets the user fix the normalisation with constraints.

    The stored ``beta`` happens to be normalised here, but that is a CHOICE the estimator
    made -- not a property of the parameter, which is why labelling its covariance would
    be misleading.
    """

    beta = np.asarray(vecm_fit.beta)

    # The first row is set to one in this normalisation.
    assert beta[0, 0] == pytest.approx(1.0)