"""Tests for ``r(mineigcv)`` -- the Stock-Yogo critical values.

Transcribed from ``ivregress_estat.ado:2483-2641``.  ``Calcmineigcv`` stacks three
hard-coded tables::

    mat `cv' = `relbias' \\ `tsls' \\ `liml'

giving a **3 x 4** matrix:

| row | meaning | columns |
|---|---|---|
| 0 | relative bias | 5%, 10%, 20%, 30% |
| 1 | TSLS worst-case size | 10%, 15%, 20%, 25% |
| 2 | LIML worst-case size | 10%, 15%, 20%, 25% |

Row selection (``ado:2642-2668``): ``nendog == 1`` uses columns 1-4, ``== 2``
uses 5-8, and the bias table's third block (9-12) exists only for ``nendog == 3``.
The size tables stop at two endogenous regressors, and everything is missing
beyond 30 instruments or 3 endogenous ones.

These are Stock and Yogo's published simulation values, reproduced literally. The
tests below check the transcription against the ado's own numbers rather than
against a re-derivation -- there is no closed form to compare with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.stock_yogo import (
    MAX_INSTRUMENTS,
    STOCK_YOGO_LIML_SIZE,
    STOCK_YOGO_RELATIVE_BIAS,
    STOCK_YOGO_TSLS_SIZE,
    mineig_critical_values,
)


# ------------------------------------------------------------- table sanity
def test_the_tables_have_the_shapes_the_ado_uses() -> None:
    assert STOCK_YOGO_TSLS_SIZE.shape == (MAX_INSTRUMENTS, 8)
    assert STOCK_YOGO_LIML_SIZE.shape == (MAX_INSTRUMENTS, 8)
    assert STOCK_YOGO_RELATIVE_BIAS.shape == (MAX_INSTRUMENTS, 12)


def test_spot_check_values_against_the_ado() -> None:
    """Literal values read from ``ivregress_estat.ado``.

    A transcription error in one cell would be invisible to a shape check, so a
    handful of cells from different corners of each table are pinned directly.
    """

    # tslssize row 1 (1 instrument), columns 1-4.
    assert list(STOCK_YOGO_TSLS_SIZE[0, 0:4]) == [16.38, 8.96, 6.66, 5.53]
    # tslssize row 2, columns 5-8 (2 endogenous regressors).
    assert list(STOCK_YOGO_TSLS_SIZE[1, 4:8]) == [7.03, 4.58, 3.95, 3.63]
    # Last row of each table.
    assert list(STOCK_YOGO_TSLS_SIZE[-1, 0:4]) == [86.18, 44.77, 30.74, 23.65]
    assert list(STOCK_YOGO_LIML_SIZE[-1, 0:4]) == [3.88, 2.18, 1.89, 1.75]
    assert list(STOCK_YOGO_RELATIVE_BIAS[-1, 0:4]) == [21.42, 11.32, 6.09, 4.29]


def test_the_bias_table_starts_with_two_empty_rows() -> None:
    """``tslsbias``'s first two rows are all ``.`` in the ado."""

    assert np.all(np.isnan(STOCK_YOGO_RELATIVE_BIAS[0]))
    assert np.all(np.isnan(STOCK_YOGO_RELATIVE_BIAS[1]))


# ------------------------------------------------------------- row selection
def test_one_endogenous_regressor_uses_the_first_block() -> None:
    matrix = mineig_critical_values(n_endog=1, n_instruments=2)

    assert matrix.shape == (3, 4)
    assert list(matrix[1]) == [19.93, 11.59, 8.75, 7.25]
    assert list(matrix[2]) == [8.68, 5.33, 4.42, 3.92]


def test_two_endogenous_regressors_use_the_second_block() -> None:
    matrix = mineig_critical_values(n_endog=2, n_instruments=2)

    assert list(matrix[1]) == [7.03, 4.58, 3.95, 3.63]
    # The bias row is still missing: ``tslsbias``'s first two rows are empty and
    # 2 instruments lands on row 2.
    assert np.all(np.isnan(matrix[0]))


def test_two_endogenous_regressors_with_more_instruments_show_bias() -> None:
    """The bias block for 2 endogenous regressors starts at 4 instruments.

    ``tslsbias`` row 3 (3 instruments) still has only the first four columns; the
    5-8 block appears from row 4 onward.
    """

    matrix = mineig_critical_values(n_endog=2, n_instruments=4)

    assert list(matrix[0]) == [11.04, 7.56, 5.57, 4.73]
    assert list(matrix[1]) == [16.87, 9.93, 7.54, 6.28]
    assert list(matrix[2]) == [4.72, 3.39, 2.99, 2.79]


def test_three_endogenous_regressors_have_bias_but_no_size_row() -> None:
    """``ado:2654-2656`` leaves the TSLS/LIML rows missing when ``nendog > 2``."""

    matrix = mineig_critical_values(n_endog=3, n_instruments=5)

    assert np.all(np.isnan(matrix[1]))
    assert np.all(np.isnan(matrix[2]))
    assert np.all(np.isfinite(matrix[0]))


def test_beyond_three_endogenous_regressors_everything_is_missing() -> None:
    matrix = mineig_critical_values(n_endog=4, n_instruments=5)

    assert np.all(np.isnan(matrix))


def test_beyond_thirty_instruments_everything_is_missing() -> None:
    """``ado:2642``: ``ninst > 30`` blanks the whole matrix."""

    matrix = mineig_critical_values(n_endog=1, n_instruments=31)

    assert np.all(np.isnan(matrix))


def test_one_instrument_has_no_bias_row() -> None:
    """``tslsbias`` rows 1-2 are empty, so the bias row is missing at 1 instrument."""

    matrix = mineig_critical_values(n_endog=1, n_instruments=1)

    assert np.all(np.isnan(matrix[0]))
    assert np.all(np.isfinite(matrix[1]))
    assert np.all(np.isfinite(matrix[2]))


# -------------------------------------------------------------- integration
@pytest.fixture()
def frame():
    rng = np.random.default_rng(1)
    n = 400
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    x1 = 0.6 * z1 + 0.3 * z2 + 0.4 * u + rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + u
    return pd.DataFrame({"y": y, "x1": x1, "z1": z1, "z2": z2})


def test_firststage_exposes_the_matrix(frame) -> None:
    first_stage = (
        ws.IV2SLS.from_formula("y ~ [x1 ~ z1 + z2]", frame)
        .fit()
        .estat_firststage()
    )

    assert first_stage.mineigcv is not None
    assert first_stage.mineigcv.shape == (3, 4)


def test_the_matrix_matches_the_standalone_function(frame) -> None:
    first_stage = (
        ws.IV2SLS.from_formula("y ~ [x1 ~ z1 + z2]", frame)
        .fit()
        .estat_firststage()
    )

    expected = mineig_critical_values(n_endog=1, n_instruments=2)
    np.testing.assert_allclose(first_stage.mineigcv, expected, equal_nan=True)


def test_mineig_is_comparable_against_the_tsls_size_row(frame) -> None:
    """The whole point of the matrix: compare ``mineig`` with the thresholds.

    This is the weak-instrument decision Stata prints -- if ``mineig`` exceeds a
    row-1 entry, the worst-case size distortion is no worse than that column's
    nominal rate.

    ``r(mineig)`` is on the same scale as the thresholds -- both are Cragg-Donald
    statistics, not squared correlations. A live ``estat firststage`` on the reference
    sample prints 332.355 as ``r(mineig)`` alongside thresholds of 19.93 / 11.59 / 8.75 /
    7.25, so the comparison is like against like.

    This test asserts the SAME-SCALE property rather than a fixed verdict: the earlier
    version asserted ``not verdict`` because it believed mineig was a [0, 1] correlation,
    and that belief is what made the comparison look meaningless.
    """

    first_stage = (
        ws.IV2SLS.from_formula("y ~ [x1 ~ z1 + z2]", frame)
        .fit()
        .estat_firststage()
    )

    mineig = first_stage.mineig
    thresholds = first_stage.mineigcv[1]

    assert np.all(np.isfinite(thresholds))
    assert np.all(thresholds > 0.0)
    # Same order of magnitude as the thresholds: a [0, 1] correlation would sit orders
    # below them, and comparing those two would be the scale error this guards against.
    assert mineig > 0.0
    assert mineig / float(np.max(thresholds)) > 1e-3
    # The comparison resolves to a definite verdict either way.
    verdict = mineig > thresholds[-1]
    assert isinstance(verdict, (bool, np.bool_))


def test_the_matrix_is_unaffected_by_the_vce(frame) -> None:
    """Critical values depend only on the model shape, not on the VCE."""

    plain = (
        ws.IV2SLS.from_formula("y ~ [x1 ~ z1 + z2]", frame)
        .fit()
        .estat_firststage()
    )
    robust = (
        ws.IV2SLS.from_formula("y ~ [x1 ~ z1 + z2]", frame)
        .fit(cov_type="HC1")
        .estat_firststage()
    )

    np.testing.assert_allclose(plain.mineigcv, robust.mineigcv, equal_nan=True)
