"""Weights must follow the estimation sample, and may be given by column name.

``replication/papers/ad2013/WESTATA_GAPS.md`` recorded three contract defects in
``OLS.from_formula``:

1. a weight VECTOR aligned with ``frame`` was rejected whenever the formula dropped rows
   for missing values, with ``WeightError: weights have 722 entries but the data have 675
   rows`` -- a message that reads like malformed data when the data are fine;
2. ``weights="colname"`` was unsupported, which is exactly what makes
   ``statsmodels.formula.api``'s ``dropna`` handling transparent;
3. the docstring never said that rows are dropped.

All three are fixed. The distinction worth keeping: the ESTIMATES were never wrong -- the
document's control probe showed a pre-aligned call reproduced 0.6950717 before and after --
so these tests check the contract, and the arithmetic separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 200, n_missing: int = 40, seed: int = 9401) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "w": rng.uniform(0.5, 2.0, size=n),
        }
    )
    # Missing values in a REGRESSOR, so patsy drops those rows and the estimation sample
    # is genuinely shorter than the frame.
    frame.loc[frame.index[:n_missing], "x"] = np.nan
    return frame


@pytest.fixture(scope="module")
def frame():
    return _frame()


# ---------------------------------------------- defect 1: the vector is sliced
def test_a_full_length_weight_vector_is_accepted(frame) -> None:
    """The headline defect: the vector matches ``frame``, which is what the API asks for."""

    fitted = ws.OLS.from_formula(
        "y ~ x", frame, weights=frame["w"], weight_type="aweight"
    ).fit()

    assert fitted is not None


def test_the_estimation_sample_size_is_the_kept_rows(frame) -> None:
    fitted = ws.OLS.from_formula(
        "y ~ x", frame, weights=frame["w"], weight_type="aweight"
    ).fit()

    assert float(fitted.e["N"]) == float(frame["x"].notna().sum())


def test_the_sliced_vector_matches_a_pre_sliced_one(frame) -> None:
    """A caller who pre-sliced by hand must keep working, and agree exactly."""

    aligned = ws.OLS.from_formula(
        "y ~ x", frame, weights=frame["w"], weight_type="aweight"
    ).fit()

    mask = frame["x"].notna().to_numpy()
    pre_sliced = ws.OLS.from_formula(
        "y ~ x", frame, weights=frame["w"].to_numpy()[mask], weight_type="aweight"
    ).fit()

    np.testing.assert_allclose(
        np.asarray(aligned.params, dtype=float),
        np.asarray(pre_sliced.params, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(aligned.bse, dtype=float),
        np.asarray(pre_sliced.bse, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


def test_a_wrong_length_is_refused_with_both_numbers(frame) -> None:
    """Neither length is refused silently, and the message names what was expected."""

    with pytest.raises(ws.WeightError) as caught:
        ws.OLS.from_formula(
            "y ~ x", frame, weights=np.ones(7), weight_type="aweight"
        )

    message = str(caught.value)
    assert "7" in message
    assert str(len(frame)) in message


# ------------------------------------------------ defect 2: weights by name
def test_weights_may_be_a_column_name(frame) -> None:
    fitted = ws.OLS.from_formula("y ~ x", frame, weights="w", weight_type="aweight").fit()

    assert fitted is not None


def test_the_column_name_and_the_vector_agree(frame) -> None:
    """A name is only useful if it resolves to the same numbers."""

    by_name = ws.OLS.from_formula("y ~ x", frame, weights="w", weight_type="aweight").fit()
    by_vector = ws.OLS.from_formula(
        "y ~ x", frame, weights=frame["w"], weight_type="aweight"
    ).fit()

    np.testing.assert_allclose(
        np.asarray(by_name.params, dtype=float),
        np.asarray(by_vector.params, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


def test_an_unknown_column_name_is_refused_clearly(frame) -> None:
    """A string that is not a column must not be read as a one-element array."""

    with pytest.raises(ws.WeightError, match="no such column"):
        ws.OLS.from_formula(
            "y ~ x", frame, weights="not_a_column", weight_type="aweight"
        )


# --------------------------------------------- defect 3: the behaviour is documented
def test_the_docstring_states_that_rows_are_dropped() -> None:
    text = ws.OLS.from_formula.__doc__ or ""

    assert "missing" in text.lower()
    assert "drop" in text.lower()


def test_the_docstring_explains_the_weight_forms() -> None:
    text = ws.OLS.from_formula.__doc__ or ""

    assert "column name" in text.lower()


# ------------------------------------------------------- the arithmetic is intact
def test_the_weighted_estimate_matches_a_hand_computation(frame) -> None:
    """The fix changed WHERE weights are sliced, not what the weighted fit is.

    Recomputed with ``numpy`` on the complete-case rows, so this checks the estimator
    against the definition rather than against another code path here.
    """

    fitted = ws.OLS.from_formula(
        "y ~ x", frame, weights=frame["w"], weight_type="aweight"
    ).fit()

    complete = frame.dropna(subset=["y", "x"])
    design = np.column_stack([np.ones(len(complete)), complete["x"].to_numpy()])
    weight = complete["w"].to_numpy()
    response = complete["y"].to_numpy()

    # aweight: minimises sum(w * (y - Xb)^2)
    root = np.sqrt(weight)
    beta = np.linalg.lstsq(design * root[:, None], response * root, rcond=None)[0]

    np.testing.assert_allclose(
        np.asarray(fitted.params, dtype=float), beta, rtol=1e-10, atol=1e-12
    )


def test_unweighted_fits_are_unaffected(frame) -> None:
    """The change is confined to the weighted path."""

    fitted = ws.OLS.from_formula("y ~ x", frame).fit()

    assert np.all(np.isfinite(np.asarray(fitted.params, dtype=float)))


def test_no_missing_values_means_no_slicing(frame) -> None:
    """A complete frame must take the same path as before."""

    complete = frame.dropna(subset=["x"]).reset_index(drop=True)

    by_vector = ws.OLS.from_formula(
        "y ~ x", complete, weights=complete["w"], weight_type="aweight"
    ).fit()
    by_name = ws.OLS.from_formula(
        "y ~ x", complete, weights="w", weight_type="aweight"
    ).fit()

    np.testing.assert_allclose(
        np.asarray(by_vector.params, dtype=float),
        np.asarray(by_name.params, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )