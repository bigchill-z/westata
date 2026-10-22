"""Panel estimator conformance tests (``xtreg`` family).

Each estimator is checked against an explicitly computed reference rather than
against the backend it wraps, so a wrong mapping cannot pass silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from linearmodels.panel import RandomEffects as LM_RandomEffects

from westata.estimators.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    RandomEffects,
)


def _panel(n_entities: int = 5, n_periods: int = 4, seed: int = 3):
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(scale=1.0, size=n_entities)[entity]
    y = 2.0 + 1.5 * x1 - 0.7 * x2 + effect + rng.normal(size=n)
    df = pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time}
    )
    return df, n_entities, n_periods


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def test_fe_matches_within_transformed_ols():
    df, _, _ = _panel()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()

    # Within transform by hand.
    g = df.groupby("firm")
    yw = df["y"] - g["y"].transform("mean")
    x1w = df["x1"] - g["x1"].transform("mean")
    x2w = df["x2"] - g["x2"].transform("mean")
    beta = _ols(np.column_stack([x1w, x2w]), yw.to_numpy())

    np.testing.assert_allclose(res.params[["x1", "x2"]].to_numpy(), beta, atol=1e-10)


def test_two_way_fe_matches_two_way_demeaned_ols():
    df, n_entities, n_periods = _panel()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit(
        entity_effects=True, time_effects=True
    )

    # Two-way within transform: subtract entity means, time means, add back the
    # grand mean (balanced panel, so this is exact in one pass).
    def demean(series: pd.Series) -> pd.Series:
        return (
            series
            - series.groupby(df["firm"]).transform("mean")
            - series.groupby(df["year"]).transform("mean")
            + series.mean()
        )

    beta = _ols(
        np.column_stack([demean(df["x1"]), demean(df["x2"])]),
        demean(df["y"]).to_numpy(),
    )
    np.testing.assert_allclose(res.params[["x1", "x2"]].to_numpy(), beta, atol=1e-10)


def test_re_matches_linearmodels_swamy_arora():
    df, _, _ = _panel()
    res = RandomEffects.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()

    ref = LM_RandomEffects.from_formula(
        "y ~ 1 + x1 + x2", df.set_index(["firm", "year"])
    ).fit()

    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(
        res.bse.to_numpy(), np.asarray(ref.std_errors, dtype=float), rtol=1e-12
    )


def test_be_matches_group_mean_ols():
    df, n_entities, _ = _panel()
    res = BetweenOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()

    means = df.groupby("firm")[["y", "x1", "x2"]].mean()
    X = np.column_stack([np.ones(len(means)), means["x1"], means["x2"]])
    beta = _ols(X, means["y"].to_numpy())

    np.testing.assert_allclose(res.params.to_numpy(), beta, atol=1e-10)
    # e(N) stays the number of original observations; df_r is computed on the
    # n_entities group means the between estimator actually uses.
    assert res.e["N"] == len(df)
    assert res.e["df_r"] == n_entities - 3


def test_fd_matches_first_differenced_ols():
    df, _, _ = _panel()
    res = FirstDifferenceOLS.from_formula(
        "y ~ x1 + x2", df, entity="firm", time="year"
    ).fit()

    diff = df.sort_values(["firm", "year"]).groupby("firm").diff().dropna()
    X = np.column_stack([diff["x1"], diff["x2"]])
    beta = _ols(X, diff["y"].to_numpy())

    # xtreg, fd fits without a constant, so there is no Intercept parameter.
    np.testing.assert_allclose(res.params[["x1", "x2"]].to_numpy(), beta, atol=1e-10)
    assert "Intercept" not in res.params.index


def test_fe_df_r_follows_stata_formula():
    df, n_entities, n_periods = _panel()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()

    nobs = len(df)
    rank = 3  # Intercept, x1, x2
    # Stata xtreg, fe: df_r = N - K - n_entities + 1
    expected = nobs - rank - n_entities + 1
    assert res.e["df_r"] == expected
    assert res.e["N"] == nobs
    assert res.e["n_entities"] == n_entities
    assert res.e["rank"] == rank


def test_unbalanced_panel_works():
    df, _, _ = _panel(n_entities=6, n_periods=5)
    # Drop one observation from each of three entities.
    drop = df.index[(df["firm"] == 0) & (df["year"] == 0)].tolist()
    drop += df.index[(df["firm"] == 2) & (df["year"] == 1)].tolist()
    drop += df.index[(df["firm"] == 4) & (df["year"] == 2)].tolist()
    unbalanced = df.drop(index=drop)

    res = PanelOLS.from_formula(
        "y ~ x1 + x2", unbalanced, entity="firm", time="year"
    ).fit()
    assert res.e["N"] == len(unbalanced)
    assert np.all(np.isfinite(res.params.to_numpy()))
    assert np.all(np.isfinite(res.bse.to_numpy()))


def test_cluster_uses_g_minus_1_for_inference():
    df, n_entities, _ = _panel()
    clusters = df["firm"].to_numpy()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit(
        cov_type="cluster", cov_kwds={"clusters": clusters}
    )
    assert res.df_resid == n_entities - 1
    # e(df_r) keeps the residual degrees of freedom.
    assert res.e["df_r"] == len(df) - 3 - n_entities + 1


def test_required_e_keys_present():
    df, _, _ = _panel()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()
    for key in ["N", "df_m", "df_r", "r2", "r2_a", "rmse", "F", "ll", "rank"]:
        assert key in res.e, key


def test_summary_stata_renders():
    df, _, _ = _panel()
    text = PanelOLS.from_formula(
        "y ~ x1 + x2", df, entity="firm", time="year"
    ).fit().summary(style="stata")
    for token in ["Coef.", "Std. err.", "P>|t|", "Number of obs"]:
        assert token in text
    assert "xtreg, fe" in text


def test_unknown_cov_type_raises():
    df, _, _ = _panel()
    with pytest.raises(ValueError, match="cov_type"):
        PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit(
            cov_type="bogus"
        )


def test_missing_entity_column_raises():
    df, _, _ = _panel()
    with pytest.raises(Exception, match="firm_id"):
        PanelOLS.from_formula("y ~ x1", df, entity="firm_id", time="year")
