"""``suest`` -- seemingly unrelated estimation.

``suest.sthlp:84-88``:

    suest combines the estimation results -- parameter estimates and associated
    (co)variance matrices -- stored under namelist into one parameter vector and
    simultaneous (co)variance matrix of the sandwich/robust type.  This (co)variance
    matrix is appropriate even if the estimates were obtained on the same or on
    overlapping data.

The last clause is the whole point. Two OLS fits on the SAME observations have correlated
scores, so stacking their ``cov_params`` block-diagonally would report zero covariance
between coefficients that are in fact related. The sandwich captures it:

    V = B (Σ_i s_i s_i') B,   B = blockdiag((X_a'X_a)^-1, (X_b'X_b)^-1)

``suest.sthlp:99-102``: the only requirement is that ``predict, score`` works after each
component. In a linear model that is the residual
(``regress_postestimation.sthlp:193``: "score is equivalent to residuals in linear
regression").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 200, seed: int = 3700):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )


@pytest.fixture(scope="module")
def two_models():
    frame = _frame()
    first = ws.OLS.from_formula("y ~ x1", frame).fit()
    second = ws.OLS.from_formula("y ~ x2", frame).fit()
    return frame, first, second


# ------------------------------------------------------------- the prerequisite
def test_regress_predicts_scores() -> None:
    """``suest.sthlp:99-102`` needs ``predict, score``; ``sthlp:193`` says it is the
    residual in a linear model."""

    frame = _frame()
    fitted = ws.OLS.from_formula("y ~ x1", frame).fit()

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)
    residuals = np.asarray(fitted.predict(kind="residuals"), dtype=float)

    assert scores.shape == residuals.shape
    assert np.allclose(scores, residuals)


def test_score_is_registered_for_regress() -> None:
    from westata.postestimation.predict import supported_types

    assert "score" in supported_types("regress")


# --------------------------------------------------------------- the stacking
def test_the_parameters_are_concatenated(two_models) -> None:
    _, first, second = two_models

    stacked = first.suest(second, names=["a", "b"])

    expected = np.concatenate(
        [
            np.asarray(first.params, dtype=float),
            np.asarray(second.params, dtype=float),
        ]
    )
    assert np.allclose(stacked.params, expected)


def test_the_names_carry_the_model_prefix(two_models) -> None:
    _, first, second = two_models

    stacked = first.suest(second, names=["a", "b"])

    assert stacked.param_names[0] == "a:Intercept"
    assert stacked.param_names[2] == "b:Intercept"
    assert len(stacked.param_names) == len(stacked.params)


def test_the_covariance_spans_every_parameter(two_models) -> None:
    _, first, second = two_models

    stacked = first.suest(second)

    size = len(first.params) + len(second.params)
    assert stacked.covariance.shape == (size, size)


def test_the_covariance_is_symmetric(two_models) -> None:
    """A covariance matrix is symmetric by definition; an asymmetric one is a bug."""

    _, first, second = two_models

    covariance = first.suest(second).covariance

    assert np.allclose(covariance, covariance.T)


def test_the_diagonal_is_non_negative(two_models) -> None:
    _, first, second = two_models

    diagonal = np.diag(first.suest(second).covariance)

    assert np.all(diagonal >= 0.0)


# ---------------------------------------------------- the point of the sandwich
def test_the_cross_model_blocks_are_not_zero(two_models) -> None:
    """THE distinguishing test.

    A block-diagonal stack -- the naive way to combine two fits -- reports zero
    covariance between the two models' coefficients. ``suest.sthlp:87-88`` exists to say
    that is wrong when the samples overlap, and the sandwich's off-diagonal blocks are
    exactly what captures it. If they came out zero, the meat would have been built from
    per-model scores separately rather than observation by observation.
    """

    _, first, second = two_models
    stacked = first.suest(second)
    size = len(first.params)

    cross = stacked.covariance[:size, size:]

    assert np.any(np.abs(cross) > 1e-12)


def test_the_meat_is_built_observation_by_observation(two_models) -> None:
    """Rebuild the sandwich from the definition and compare.

    This is the same construction the implementation uses, written independently, so it
    checks that the two blocks' scores are paired by ROW rather than by model.
    """

    frame, first, second = two_models
    design_first = np.asarray(first.design_matrix[0], dtype=float)
    design_second = np.asarray(second.design_matrix[0], dtype=float)
    scores_first = design_first * np.asarray(first.resid, dtype=float).reshape(-1, 1)
    scores_second = design_second * np.asarray(second.resid, dtype=float).reshape(-1, 1)
    scores = np.column_stack([scores_first, scores_second])

    bread_first = np.linalg.inv(design_first.T @ design_first)
    bread_second = np.linalg.inv(design_second.T @ design_second)
    width = bread_first.shape[0] + bread_second.shape[0]
    bread = np.zeros((width, width))
    bread[: bread_first.shape[0], : bread_first.shape[0]] = bread_first
    bread[bread_first.shape[0] :, bread_first.shape[0] :] = bread_second
    expected = bread @ (scores.T @ scores) @ bread

    assert np.allclose(first.suest(second).covariance, expected, rtol=1e-10)


def test_a_single_model_block_resembles_its_own_robust_vce(two_models) -> None:
    """The diagonal block should be the model's HC0 covariance.

    Not equal to ``cov_params()`` -- that is the CLASSICAL VCE, while ``suest`` reports
    the robust one. The magnitudes should be close, which is what makes the comparison
    meaningful without asserting an equality that is not true.
    """

    _, first, _ = two_models
    stacked = first.suest(two_models[2])
    size = len(first.params)
    diagonal_block = stacked.covariance[:size, :size]
    classical = np.asarray(first.cov_params(), dtype=float)

    ratio = np.diag(diagonal_block) / np.diag(classical)
    assert np.all(ratio > 0.5)
    assert np.all(ratio < 2.0)


# ------------------------------------------------------------------ the refusals
def test_one_model_is_refused(two_models) -> None:
    """``suest.sthlp:84-88`` combines results, plural."""

    _, first, _ = two_models

    with pytest.raises(ValueError, match="two or more"):
        first.suest()


def test_different_sample_sizes_are_refused() -> None:
    """The scores must pair by observation, so the samples have to match."""

    frame = _frame(n=200)
    other = _frame(n=150, seed=3702)
    first = ws.OLS.from_formula("y ~ x1", frame).fit()
    second = ws.OLS.from_formula("y ~ x2", other).fit()

    with pytest.raises(ValueError, match="same estimation sample"):
        first.suest(second)


def test_a_name_count_mismatch_is_refused(two_models) -> None:
    _, first, second = two_models

    with pytest.raises(ValueError, match="names for"):
        first.suest(second, names=["only_one"])


# ---------------------------------------------------------------- the r() surface
def test_to_dict_exposes_the_stacked_coefficients(two_models) -> None:
    _, first, second = two_models

    table = first.suest(second, names=["a", "b"]).to_dict()

    assert set(table) == set(
        first.suest(second, names=["a", "b"]).param_names
    )
    for value in table.values():
        assert np.isfinite(value)


def test_to_frame_labels_both_axes(two_models) -> None:
    _, first, second = two_models

    frame = first.suest(second, names=["a", "b"]).to_frame()

    assert list(frame.index) == list(frame.columns)
    assert frame.shape[0] == frame.shape[1] == len(first.params) + len(second.params)


def test_three_models_are_supported() -> None:
    """The block structure must generalise, not just handle a pair."""

    frame = _frame()
    models = [
        ws.OLS.from_formula(f"y ~ {name}", frame).fit()
        for name in ("x1", "x2", "x3")
    ]

    stacked = models[0].suest(*models[1:], names=["a", "b", "c"])

    assert stacked.models == 3
    assert stacked.covariance.shape == (6, 6)
    assert np.allclose(stacked.covariance, stacked.covariance.T)


# ------------------------------------------------- what suest exists FOR
def test_the_result_exposes_cov_params_as_a_call(two_models) -> None:
    """``suest.sthlp:91-94`` names ``test``/``testnl``/``lincom``/``nlcom`` as the typical
    uses, and all four reach for ``result.cov_params()``.

    Exposing the matrix only as an attribute left every one of them unusable on the
    stacked result -- which defeated the command's purpose.
    """

    stacked = two_models[1].suest(two_models[2])

    assert callable(stacked.cov_params)
    assert np.allclose(stacked.cov_params(), stacked.covariance)


def test_a_single_restriction_can_be_tested(two_models) -> None:
    """A Wald test on one stacked coefficient."""

    stacked = two_models[1].suest(two_models[2], names=["m1", "m2"])

    tested = stacked.test("m1:x1 = 0")

    assert float(tested.df) == 1.0
    assert np.isfinite(float(tested.statistic))
    assert 0.0 <= float(tested.pvalue) <= 1.0


def test_the_test_uses_the_smallest_residual_df(two_models) -> None:
    """A joint test needs one denominator; the conservative choice is the smallest."""

    stacked = two_models[1].suest(two_models[2], names=["m1", "m2"])
    smallest = min(
        float(two_models[1].df_resid), float(two_models[2].df_resid)
    )

    assert stacked.df_resid == pytest.approx(smallest)
    assert float(stacked.test("m1:x1 = 0").df_denom) == pytest.approx(smallest)


def test_a_cross_model_equality_can_be_tested(two_models) -> None:
    """The canonical use: is one model's coefficient equal to the other's?

    ``suest.sthlp:91-92`` gives "a generalized Hausman specification test" as an example
    of exactly this shape, and it is only possible because the cross-model covariance
    block is non-zero.
    """

    rng = np.random.default_rng(3802)
    n = 200
    shared = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y1": 1.0 + 0.5 * shared + rng.normal(size=n),
            "y2": 1.0 + 0.5 * shared + rng.normal(size=n),
            "x": shared,
        }
    )
    first = ws.OLS.from_formula("y1 ~ x", frame).fit()
    second = ws.OLS.from_formula("y2 ~ x", frame).fit()

    tested = first.suest(second, names=["a", "b"]).test("a:x = b:x")

    assert float(tested.df) == 1.0
    # The two coefficients were generated equal, so this must not reject.
    assert float(tested.pvalue) > 0.05


def test_the_cross_model_test_is_not_the_same_as_testing_one_side(two_models) -> None:
    """Guard: if the restriction ignored the second model it would equal the first's test.

    The stacked covariance carries the correlation, so the difference's standard error is
    NOT the sum of the two variances -- which is what makes the equality test different
    from two separate tests.
    """

    first, second = two_models[1], two_models[2]
    stacked = first.suest(second, names=["a", "b"])

    difference = stacked.lincom("a:x1 - b:x2")
    alone = first.lincom("x1")

    assert not np.isclose(
        float(difference.std_err), float(alone.std_err), rtol=1e-6
    )


def test_lincom_works_on_the_stacked_vector(two_models) -> None:
    """``sthlp:93-94``: lincom after suest estimates a combination of coefficients."""

    stacked = two_models[1].suest(two_models[2], names=["m1", "m2"])

    combined = stacked.lincom("m1:x1 - m2:x2")

    expected = float(np.asarray(two_models[1].params, dtype=float)[1]) - float(
        np.asarray(two_models[2].params, dtype=float)[1]
    )
    assert float(combined.estimate) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------- vce(cluster clustvar)
def _clustered_fixture():
    rng = np.random.default_rng(3900)
    n = 200
    frame = pd.DataFrame(
        {
            "y1": rng.normal(size=n),
            "y2": rng.normal(size=n),
            "x": rng.normal(size=n),
            "g": np.arange(n) % 20,
        }
    )
    return (
        frame,
        ws.OLS.from_formula("y1 ~ x", frame).fit(),
        ws.OLS.from_formula("y2 ~ x", frame).fit(),
    )


def test_the_cluster_option_is_accepted() -> None:
    """``suest.sthlp:108-109``: "suest returns the robust VCE, allows the
    vce(cluster clustvar) option"."""

    frame, first, second = _clustered_fixture()

    stacked = first.suest(second, names=["a", "b"], cluster=frame["g"])

    assert stacked.covariance.shape == (4, 4)


def test_clustering_changes_the_covariance() -> None:
    """The clustered meat sums scores WITHIN a cluster before squaring.

    If the option were accepted and ignored, the two matrices would coincide -- the
    failure shape this session keeps finding.
    """

    frame, first, second = _clustered_fixture()

    plain = first.suest(second, names=["a", "b"])
    clustered = first.suest(second, names=["a", "b"], cluster=frame["g"])

    assert not np.allclose(plain.covariance, clustered.covariance)


def test_the_clustered_matrix_is_also_symmetric_and_psd_diagonal() -> None:
    frame, first, second = _clustered_fixture()

    covariance = first.suest(
        second, names=["a", "b"], cluster=frame["g"]
    ).covariance

    assert np.allclose(covariance, covariance.T)
    assert np.all(np.diag(covariance) >= 0.0)


def test_clustering_matches_a_hand_built_sandwich() -> None:
    """Rebuild the clustered sandwich from the definition.

    Aggregating each observation's stacked score within its cluster BEFORE the outer
    product is the whole difference from the unclustered meat, so an implementation that
    summed outer products would fail here.
    """

    frame, first, second = _clustered_fixture()
    design_first = np.asarray(first.design_matrix[0], dtype=float)
    design_second = np.asarray(second.design_matrix[0], dtype=float)
    scores = np.column_stack(
        [
            design_first * np.asarray(first.resid, dtype=float).reshape(-1, 1),
            design_second * np.asarray(second.resid, dtype=float).reshape(-1, 1),
        ]
    )
    labels = frame["g"].to_numpy()
    aggregated = np.array(
        [scores[labels == cluster].sum(axis=0) for cluster in pd.unique(labels)]
    )
    size_first = design_first.shape[1]
    width = size_first + design_second.shape[1]
    bread = np.zeros((width, width))
    bread[:size_first, :size_first] = np.linalg.inv(design_first.T @ design_first)
    bread[size_first:, size_first:] = np.linalg.inv(design_second.T @ design_second)
    expected = bread @ (aggregated.T @ aggregated) @ bread

    produced = first.suest(
        second, names=["a", "b"], cluster=frame["g"]
    ).covariance

    assert np.allclose(produced, expected, rtol=1e-10)


def test_clustering_by_every_observation_equals_no_clustering() -> None:
    """Boundary case: singleton clusters add nothing beyond the unclustered meat.

    With one observation per cluster, summing within a cluster is a no-op, so the two
    matrices must coincide. This is the check that the cluster code path is a genuine
    generalisation rather than a different formula.
    """

    frame, first, second = _clustered_fixture()
    singletons = np.arange(len(frame))

    plain = first.suest(second, names=["a", "b"])
    clustered = first.suest(second, names=["a", "b"], cluster=singletons)

    assert np.allclose(plain.covariance, clustered.covariance, rtol=1e-12)


def test_a_cluster_length_mismatch_is_refused() -> None:
    frame, first, second = _clustered_fixture()

    with pytest.raises(ValueError, match="cluster labels"):
        first.suest(second, cluster=np.zeros(10))


def test_the_clustered_result_still_supports_test_and_lincom() -> None:
    """The extra option must not disable the commands suest exists for."""

    frame, first, second = _clustered_fixture()

    stacked = first.suest(second, names=["a", "b"], cluster=frame["g"])

    tested = stacked.test("a:x = 0")
    combined = stacked.lincom("a:x - b:x")

    assert np.isfinite(float(tested.statistic))
    assert np.isfinite(float(combined.estimate))