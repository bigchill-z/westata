"""``estat auc`` -- the area under the ROC curve.

`logit_estat.ado:19-21` dispatches ``estat auc`` to ``lroc, nograph``, which computes
the area by trapezoid integration of the ROC curve (``lroc.ado:47-49``):

    replace `p' = sum((`spec' - `spec'[_n-1]) * (`sens' + `sens'[_n-1]))
    return scalar area = `p'[_N]/2

That is exactly the Mann-Whitney U statistic divided by the product of the two group
sizes, which is how it is computed here: the rank form is numerically stable and can be
cross-checked against ``scipy.stats.mannwhitneyu`` rather than against itself.

A useful sanity value: AUC = 0.5 means the model ranks no better than chance, and AUC =
1.0 means perfect separation. Anything outside [0, 1] is a structural error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


@pytest.fixture(scope="module")
def separable_frame():
    """A regressor that genuinely predicts the outcome, so AUC is well above 0.5."""

    rng = np.random.default_rng(529)
    n = 400
    latent = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x1": latent + rng.normal(scale=0.5, size=n),
            "yb": (latent > 0).astype(int),
        }
    )


def _hand_auc(fitted, frame) -> float:
    """AUC as the Mann-Whitney statistic, computed from the raw predictor."""

    scores = np.asarray(frame["x1"], dtype=float)
    outcome = np.asarray(frame["yb"], dtype=int)
    positive = scores[outcome == 1]
    negative = scores[outcome == 0]
    statistic = scipy_stats.mannwhitneyu(
        positive, negative, alternative="two-sided"
    ).statistic
    return float(statistic) / (positive.size * negative.size)


def test_estat_auc_exists_for_logit(separable_frame) -> None:
    result = ws.Logit.from_formula("yb ~ x1", separable_frame).fit()

    rendered = result.estat_auc()

    assert rendered.area > 0.5


def test_the_area_matches_the_mann_whitney_statistic(separable_frame) -> None:
    """``lroc.ado:47-49``'s trapezoid integral equals the rank statistic.

    Checking against the rank form rather than re-running the same integral makes this
    an independent comparison.
    """

    result = ws.Logit.from_formula("yb ~ x1", separable_frame).fit()

    assert result.estat_auc().area == pytest.approx(
        _hand_auc(result, separable_frame), rel=1e-12
    )


def test_the_area_is_inside_the_unit_interval(separable_frame) -> None:
    """An AUC outside [0, 1] would mean the curve was integrated the wrong way."""

    area = ws.Logit.from_formula("yb ~ x1", separable_frame).fit().estat_auc().area

    assert 0.0 <= area <= 1.0


def test_a_useless_predictor_gives_about_one_half() -> None:
    """With no relationship the ROC curve is the diagonal, so AUC is 0.5.

    This is the direction check: a systematically inverted implementation reports
    ``1 - AUC``, which for a strong predictor still looks plausible but here would sit
    near 0.5 only by accident.
    """

    rng = np.random.default_rng(547)
    n = 800
    frame = pd.DataFrame(
        {"x1": rng.normal(size=n), "yb": rng.integers(0, 2, size=n)}
    )

    area = ws.Logit.from_formula("yb ~ x1", frame).fit().estat_auc().area

    assert abs(area - 0.5) < 0.1


def test_flipping_the_outcome_leaves_the_area_unchanged() -> None:
    """AUC measures RANKING quality, and refitting inverts the ranking with the labels.

    Swapping the two classes makes the coefficient change sign (measured: -2.4234), so
    the predicted probabilities order the observations the other way round and the
    curve is the mirror image -- the same area.

    My first version of this test asserted ``AUC + AUC_flipped == 1``, which would hold
    only if the model kept its original direction.  It does not: the fit follows the
    labels.  Recording the measured coefficient keeps the reasoning checkable.
    """

    rng = np.random.default_rng(557)
    n = 500
    latent = rng.normal(size=n)
    frame = pd.DataFrame(
        {"x1": latent + rng.normal(scale=0.6, size=n), "yb": (latent > 0).astype(int)}
    )
    flipped = frame.assign(yb=1 - frame["yb"])

    straight_fit = ws.Logit.from_formula("yb ~ x1", frame).fit()
    flipped_fit = ws.Logit.from_formula("yb ~ x1", flipped).fit()

    assert float(straight_fit.params["x1"]) > 0.0
    assert float(flipped_fit.params["x1"]) < 0.0
    assert flipped_fit.estat_auc().area == pytest.approx(
        straight_fit.estat_auc().area, rel=1e-10
    )


def test_a_monotone_rescaling_of_the_predictor_leaves_the_area_unchanged() -> None:
    """AUC is invariant to any monotone transformation of a single predictor.

    Negating ``x1`` makes the fitted coefficient change sign as well, so the predicted
    probabilities keep the same ordering and the curve keeps the same area.

    My first attempts tried to obtain the complement by flipping the labels or negating
    the predictor; both re-fit and both therefore keep the ordering.  AUC = 1 - AUC
    arises only when the ORDERING inverts, which a refitted single-predictor model
    never allows -- so this test records the invariance instead of a false identity.
    """

    rng = np.random.default_rng(563)
    n = 500
    latent = rng.normal(size=n)
    frame = pd.DataFrame(
        {"x1": latent + rng.normal(scale=0.6, size=n), "yb": (latent > 0).astype(int)}
    )

    base = ws.Logit.from_formula("yb ~ x1", frame).fit().estat_auc().area
    for transform in (
        lambda values: -values,
        lambda values: 3.0 * values + 1.0,
        lambda values: np.exp(values),
    ):
        transformed = frame.assign(x1=transform(frame["x1"]))
        other = ws.Logit.from_formula("yb ~ x1", transformed).fit().estat_auc().area
        assert other == pytest.approx(base, rel=1e-10)


def test_the_area_responds_to_a_weaker_relationship() -> None:
    """Adding noise must LOWER the area -- the direction of the effect.

    The invariance test above cannot detect an inverted implementation, so this one
    supplies the sensitivity check: a worse predictor must score lower, and 0.5 is the
    floor for a useless one.
    """

    rng = np.random.default_rng(569)
    n = 600
    latent = rng.normal(size=n)

    strong = pd.DataFrame(
        {"x1": latent + rng.normal(scale=0.3, size=n), "yb": (latent > 0).astype(int)}
    )
    weak = pd.DataFrame(
        {"x1": latent + rng.normal(scale=2.5, size=n), "yb": (latent > 0).astype(int)}
    )

    strong_area = ws.Logit.from_formula("yb ~ x1", strong).fit().estat_auc().area
    weak_area = ws.Logit.from_formula("yb ~ x1", weak).fit().estat_auc().area

    assert strong_area > weak_area
    assert weak_area >= 0.5 - 0.06


def test_probit_exposes_the_same_area(separable_frame) -> None:
    """``probit.ado:259`` routes through the same ``logit_estat`` machinery."""

    logit_area = ws.Logit.from_formula("yb ~ x1", separable_frame).fit().estat_auc().area
    probit_area = (
        ws.Probit.from_formula("yb ~ x1", separable_frame).fit().estat_auc().area
    )

    # Same ranking, same AUC: the link function changes the probabilities but not
    # their ORDER, and the ROC curve depends only on the order.
    assert probit_area == pytest.approx(logit_area, rel=1e-10)


def test_the_rendered_line_names_the_area(separable_frame) -> None:
    """``lroc.ado:52-53`` prints "Area under ROC curve = <value>"."""

    rendered = str(ws.Logit.from_formula("yb ~ x1", separable_frame).fit().estat_auc())

    assert "Area under ROC curve" in rendered


def test_an_ols_fit_refuses(separable_frame) -> None:
    """``estat auc`` is dispatched by ``logit_estat``, so an OLS fit has no such command."""

    ols_frame = separable_frame.assign(y=separable_frame["x1"])
    result = ws.OLS.from_formula("y ~ x1", ols_frame).fit()

    with pytest.raises((AttributeError, NotImplementedError)):
        result.estat_auc()


def test_ologit_refuses_and_the_evidence_is_recorded() -> None:
    """``ologit`` has no ``estat auc``.

    The same reasoning as ``estat classification``: ``ologit.ado`` sets no
    ``e(estat_cmd)``, so ``logit_estat.ado:19-21`` never sees the call, and
    ``ologit_postestimation.sthlp`` contains no ``auc`` section.  A multi-category ROC
    would need a different definition anyway (one curve per cut point).
    """

    from pathlib import Path

    rng = np.random.default_rng(571)
    n = 300
    frame = pd.DataFrame(
        {"x1": rng.normal(size=n), "rep": rng.integers(2, 4, size=n)}
    )
    ordered = ws.OrderedLogit.from_formula("rep ~ x1", frame).fit()

    with pytest.raises(NotImplementedError, match="logit or probit"):
        ordered.estat_auc()

    base = Path(r"D:\Program Files\StataNow19\ado\base")
    ologit_help = (base / "o" / "ologit_postestimation.sthlp").read_text(
        encoding="utf-8", errors="ignore"
    )
    assert "auc" not in ologit_help.lower()


def test_only_one_class_present_is_refused() -> None:
    """The ROC curve is undefined when every observation has the same outcome.

    Returning 0.5 there would look like "no discrimination" when the truth is "there
    is nothing to discriminate", so the function refuses instead.
    """

    from westata.postestimation.auc import estat_auc

    class _OneClass:
        """The minimum surface ``estat_auc`` reads: a label and the response arrays."""

        model_name = "logit"

        def __init__(self, n: int) -> None:
            self.fittedvalues = np.full(n, 0.5)
            self.resid = np.zeros(n)

    with pytest.raises(ValueError, match="both outcomes"):
        estat_auc(_OneClass(50))