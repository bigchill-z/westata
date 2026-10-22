"""Frequency weights must reproduce the expanded-data fit exactly.

Stata ``[U] 20.24.1``:

    "the results of running any command with frequency weights should be the
    same as running the command on the unweighted, expanded data."

That principle applies to the *variance* as well, not just the point estimate,
so the ``N`` in the ``N/(N-K)`` and ``(N-1)/(N-K)`` correction factors must be
``sum(w)`` (the expanded sample size) rather than the number of stored rows.
These tests pin that down for every covariance estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.estimators.linear import OLS


def _data(n: int = 30, seed: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 5, size=n).astype(float)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Heteroskedastic, so robust and nonrobust answers differ.
    y = 1.0 + 2.0 * x1 - x2 + rng.normal(scale=0.3 + 0.4 * np.abs(x1), size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2}), counts


def _expanded(df: pd.DataFrame, counts: np.ndarray) -> pd.DataFrame:
    return df.loc[df.index.repeat(counts.astype(int))].reset_index(drop=True)


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def test_fweight_matches_expanded_for_all_hc_variants(cov_type):
    df, counts = _data()
    weighted = OLS.from_formula(
        "y ~ x1 + x2", df, weights=counts, weight_type="fweight"
    ).fit(cov_type=cov_type)

    expanded = OLS.from_formula("y ~ x1 + x2", _expanded(df, counts)).fit(
        cov_type=cov_type
    )

    np.testing.assert_allclose(
        weighted.params.to_numpy(), expanded.params.to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(
        weighted.bse.to_numpy(), expanded.bse.to_numpy(), rtol=1e-12
    )


def test_fweight_matches_expanded_for_cluster():
    df, counts = _data()
    rng = np.random.default_rng(21)
    clusters = rng.integers(0, 4, size=len(df))

    weighted = OLS.from_formula(
        "y ~ x1 + x2", df, weights=counts, weight_type="fweight"
    ).fit(cov_type="cluster", cov_kwds={"groups": clusters})

    expanded_df = _expanded(df, counts)
    expanded_clusters = np.repeat(clusters, counts.astype(int))
    expanded = OLS.from_formula("y ~ x1 + x2", expanded_df).fit(
        cov_type="cluster", cov_kwds={"groups": expanded_clusters}
    )

    np.testing.assert_allclose(
        weighted.bse.to_numpy(), expanded.bse.to_numpy(), rtol=1e-12
    )
    # G is unchanged by replication: the same clusters, just more rows.
    assert weighted.df_resid == expanded.df_resid == len(np.unique(clusters)) - 1


def test_fweight_e_df_r_uses_expanded_sample_size():
    df, counts = _data()
    res = OLS.from_formula(
        "y ~ x1 + x2", df, weights=counts, weight_type="fweight"
    ).fit()
    expanded_n = counts.sum()
    assert res.e["N"] == pytest.approx(expanded_n)
    assert res.e["df_r"] == pytest.approx(expanded_n - 3)
