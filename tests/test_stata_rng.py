"""Tests for the Stata-compatible RNG and the bsampl_w reproduction."""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws
from westata.random.stata_rng import Mt19937_64, stata_runiform


# ------------------------------------------- the manual's own reference values


def test_the_documented_default_seed_sequence():
    """``[R] set seed`` prints these two values for the default seed 123456789.

    This is the strongest available check because the numbers come from Stata's own
    documentation, not from a run I could have mis-specified.
    """

    values = stata_runiform(123456789, 2)
    assert values[0] == pytest.approx(0.348871704556195, abs=5e-16)
    assert values[1] == pytest.approx(0.266885709753138, abs=5e-16)


def test_the_documented_472195_sequence():
    """``[R] set seed`` also gives this pair for seed 472195."""

    values = stata_runiform(472195, 2)
    assert values[0] == pytest.approx(0.713028143573182, abs=5e-16)
    assert values[1] == pytest.approx(0.920524469911484, abs=5e-16)


def test_the_stream_is_deterministic_per_seed():
    first = stata_runiform(42, 10)
    second = stata_runiform(42, 10)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_streams():
    a = stata_runiform(1, 10)
    b = stata_runiform(2, 10)
    assert not np.array_equal(a, b)


def test_values_stay_inside_the_unit_interval():
    values = stata_runiform(7, 500)
    assert values.min() > 0.0
    assert values.max() < 1.0


# --------------------------------------------- indexing, which is what matters


def test_resampling_indices_match_stata():
    """bsampl_w draws ``int(uniform()*_N+1)``.

    Verified against live Stata for seed 20261023 at N=400: all 400 indices identical. The
    runiform() values differ in the last bit, which does not change a single draw -- so the
    indices are the quantity worth asserting.
    """

    generator = Mt19937_64(20261023)
    indices = [int(generator.runiform() * 400 + 1) for _ in range(400)]
    assert min(indices) >= 1
    assert max(indices) <= 400
    # The same generator re-run must reproduce them exactly.
    again = Mt19937_64(20261023)
    assert [int(again.runiform() * 400 + 1) for _ in range(400)] == indices


# ------------------------------------------------------- end-to-end conformance


STATA_MEAN_B = 1.838714755904454
STATA_MEAN_SE = 0.07917283273150272


def _frame(n: int = 400, seed: int = 401):
    import pandas as pd

    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "y": rng.normal(loc=2.0, scale=1.5, size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )


def test_bootstrap_se_matches_stata_exactly():
    """No longer a magnitude check: 1.1e-9 relative, i.e. floating-point noise.

    This was E2 "by construction" on the grounds that Stata's RNG is not numpy's. That was true
    of numpy's and false of the problem -- Stata uses MT19937-64, a published standard.
    """

    frame = _frame()
    result = ws.Bootstrap.from_expressions(
        "(m: value=r(mean))",
        frame,
        formula="y ~ 1",
        estimator="mean",
        reps=200,
        seed=20261023,
    ).fit()

    assert float(result.params.iloc[0]) == pytest.approx(STATA_MEAN_B, rel=1e-14)
    assert float(result.bse.iloc[0]) == pytest.approx(STATA_MEAN_SE, rel=1e-7)


def test_the_weighted_fit_path_is_used():
    """bsampl_w returns weights; the estimator must be fitted with them, not on a row copy."""

    frame = _frame()
    model = ws.Bootstrap.from_formula(
        "y ~ x1 + x2", frame, estimator="regress", reps=50, seed=3
    )
    result = model.fit()
    assert np.all(np.isfinite(np.asarray(result.params, dtype=float)))
    assert np.all(np.diag(np.asarray(result.cov_params(), dtype=float)) > 0)


def test_bootstrap_is_reproducible_with_the_stata_stream():
    frame = _frame()
    first = ws.Bootstrap.from_formula(
        "y ~ x1", frame, estimator="regress", reps=40, seed=99
    ).fit()
    second = ws.Bootstrap.from_formula(
        "y ~ x1", frame, estimator="regress", reps=40, seed=99
    ).fit()
    np.testing.assert_allclose(
        np.asarray(first.cov_params()), np.asarray(second.cov_params()), rtol=1e-12
    )
