"""Conformance tests for the panel estimators (P1 Task 5).

Every numerical assertion is computed in the test from first principles - the
within / between / first-difference transformations are written out by hand - so
a regression inside :mod:`westata.estimators.panel` cannot hide behind a helper
the implementation also uses.

Two checks encode Stata behaviour directly and are worth calling out:

* ``xtreg, fe`` degrees of freedom, ``df_r = N - K - n_entities + 1``
  (``[R] xtreg``).  For the 5-entity x 4-period panel with two regressors that
  is ``20 - 3 - 5 + 1 = 13``, asserted as that exact integer.
* clustered standard errors use Stata's CR1 factor
  ``G/(G-1) * (N-1)/(N-K)`` where ``K`` counts the absorbed effects as well as
  the reported coefficients, with the sandwich built from the within-transformed
  design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    RandomEffects,
)

N_ENTITIES = 5
N_PERIODS = 4
N_OBS = N_ENTITIES * N_PERIODS  # 20
K_WITH_CONSTANT = 3  # Intercept + x1 + x2
FE_DF_R = 13  # 20 - 3 - 5 + 1, the xtreg, fe formula
REQUIRED_E_KEYS = ("N", "df_m", "df_r", "r2", "r2_a", "rmse", "F", "ll", "rank")


# --------------------------------------------------------------------- fixtures
def make_panel(
    *, n_entities: int = N_ENTITIES, n_periods: int = N_PERIODS, seed: int = 20261022
) -> pd.DataFrame:
    """Balanced long panel with entity effects and modest noise."""

    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    n = entity.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(scale=1.0, size=n_entities)[entity]
    y = 2.0 + 1.5 * x1 - 0.7 * x2 + effect + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


def ols(
    X: np.ndarray, y: np.ndarray, *, df_resid: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS point estimates, residuals and the homoskedastic covariance."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = float(df_resid) if df_resid is not None else float(X.shape[0] - X.shape[1])
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return beta, resid, cov


def demean(df: pd.DataFrame, columns: list[str], by: str) -> np.ndarray:
    """Subtract group means from ``columns``."""

    return (df[columns] - df.groupby(by)[columns].transform("mean")).to_numpy()


def cr1_cov(
    X: np.ndarray, resid: np.ndarray, groups: np.ndarray, *, df_resid: float
) -> np.ndarray:
    """Stata's cluster covariance: ``G/(G-1) * (N-1)/(N-K)`` on the sandwich.

    ``df_resid`` is the **denominator of the second factor**, i.e. ``N - K`` with
    Stata's ``K``.  For ``xtreg, fe`` under ``vce(cluster panelvar)`` Stata's
    ``K`` is the number of *reported* coefficients (3 on the 5x4 test panel), so
    pass ``N - 3`` - not the model's clustered ``e(df_r)``, which is ``G - 1``,
    and not the effects-adjusted ``N - K - n + 1``.  Verified against
    ``xtreg y x1 x2, fe vce(cluster firm)`` on 2026-10-24: it returns
    ``.2024145 / .1204298``, Stata's printed values.
    """

    X = np.asarray(X, dtype=float)
    labels = np.asarray(groups)
    unique = np.unique(labels)
    g = unique.size
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for value in unique:
        mask = labels == value
        score = X[mask].T @ resid[mask]
        meat += np.outer(score, score)
    factor = (g / (g - 1.0)) * (X.shape[0] - 1.0) / df_resid
    return factor * bread @ meat @ bread


def fe_components(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Within-transformed ``(X, y)`` for the entity-effects model."""

    Xd = demean(df, ["x1", "x2"], by="firm")
    yd = (df["y"] - df.groupby("firm")["y"].transform("mean")).to_numpy()
    return Xd, yd


# --------------------------------------------------------------------- FE
def test_fe_matches_within_transformed_ols() -> None:
    df = make_panel()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()

    Xd, yd = fe_components(df)
    beta, _, cov = ols(Xd, yd, df_resid=FE_DF_R)

    np.testing.assert_allclose(res.params[["x1", "x2"]].to_numpy(), beta, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        res.bse[["x1", "x2"]].to_numpy(), np.sqrt(np.diag(cov)), rtol=0.0, atol=1e-10
    )


def test_two_way_fe_matches_two_way_demeaned_ols() -> None:
    df = make_panel()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit(
        entity_effects=True, time_effects=True
    )

    def two_way(series: pd.Series) -> pd.Series:
        return (
            series
            - series.groupby(df["firm"]).transform("mean")
            - series.groupby(df["year"]).transform("mean")
            + series.mean()
        )

    Xd = np.column_stack([two_way(df["x1"]), two_way(df["x2"])])
    yd = two_way(df["y"]).to_numpy()
    df_r = N_OBS - K_WITH_CONSTANT - (N_ENTITIES - 1) - (N_PERIODS - 1)  # 10
    beta, _, cov = ols(Xd, yd, df_resid=df_r)

    np.testing.assert_allclose(res.params[["x1", "x2"]].to_numpy(), beta, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        res.bse[["x1", "x2"]].to_numpy(), np.sqrt(np.diag(cov)), rtol=0.0, atol=1e-10
    )
    assert res.df_resid == df_r


def test_pooled_panel_degenerates_to_ols() -> None:
    df = make_panel()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit(
        entity_effects=False, time_effects=False
    )

    X = np.column_stack([np.ones(N_OBS), df[["x1", "x2"]].to_numpy()])
    beta, _, cov = ols(X, df["y"].to_numpy())

    np.testing.assert_allclose(res.params.to_numpy(), beta, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(res.bse.to_numpy(), np.sqrt(np.diag(cov)), rtol=0.0, atol=1e-10)
    assert res.df_resid == N_OBS - K_WITH_CONSTANT  # 17


# --------------------------------------------------------------------- RE / BE / FD
def test_re_matches_linearmodels_swamy_arora() -> None:
    from linearmodels.panel import RandomEffects as BackendRandomEffects

    df = make_panel()
    res = RandomEffects.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()

    backend = BackendRandomEffects.from_formula(
        "y ~ 1 + x1 + x2", df.set_index(["firm", "year"])
    ).fit()

    np.testing.assert_allclose(res.params.to_numpy(), backend.params.to_numpy(), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        res.bse.to_numpy(), np.asarray(backend.std_errors, dtype=float), rtol=0.0, atol=1e-12
    )
    assert res.df_resid == N_OBS - K_WITH_CONSTANT  # 17: xtreg, re reports N - K


def test_be_matches_group_mean_ols() -> None:
    df = make_panel()
    res = BetweenOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()

    means = df.groupby("firm")[["y", "x1", "x2"]].mean()
    X = np.column_stack([np.ones(N_ENTITIES), means[["x1", "x2"]].to_numpy()])
    df_r = N_ENTITIES - K_WITH_CONSTANT  # 2
    beta, _, cov = ols(X, means["y"].to_numpy(), df_resid=df_r)

    np.testing.assert_allclose(res.params.to_numpy(), beta, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(res.bse.to_numpy(), np.sqrt(np.diag(cov)), rtol=0.0, atol=1e-10)
    assert res.nobs == N_ENTITIES
    assert res.e["N"] == N_ENTITIES
    assert res.df_resid == df_r


def test_fd_matches_first_differenced_ols() -> None:
    df = make_panel()
    res = FirstDifferenceOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()

    diffs = (
        df.set_index(["firm", "year"])[["y", "x1", "x2"]]
        .unstack("year")
        .diff(axis=1)
        .stack(future_stack=True)
        .dropna()
    )
    df_r = len(diffs) - 2  # 15 - 2 = 13
    beta, _, cov = ols(
        diffs[["x1", "x2"]].to_numpy(), diffs["y"].to_numpy(), df_resid=df_r
    )

    np.testing.assert_allclose(res.params.to_numpy(), beta, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(res.bse.to_numpy(), np.sqrt(np.diag(cov)), rtol=0.0, atol=1e-10)
    assert "Intercept" not in res.params.index
    assert res.nobs == len(diffs)
    assert res.df_resid == df_r


# --------------------------------------------------------------------- df_r
def test_fe_df_resid_matches_stata_formula() -> None:
    """``xtreg, fe`` reports ``df_r = N - K - n_entities + 1`` (``[R] xtreg``)."""

    df = make_panel()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()

    n_entities = df["firm"].nunique()
    expected = N_OBS - K_WITH_CONSTANT - n_entities + 1

    assert expected == FE_DF_R == 13
    assert res.df_resid == expected
    assert isinstance(res.df_resid, float)
    assert int(res.df_resid) == 13
    assert res.e["df_r"] == 13
    assert res.e["df_m"] == 2
    assert res.e["rank"] == K_WITH_CONSTANT
    assert res.e["N"] == N_OBS


def test_fe_without_constant_uses_the_no_intercept_df_rule() -> None:
    """``y ~ x - 1`` drops the intercept, so the ``+1`` vanishes too.

    Stata's ``xtreg, fe, noconstant`` reports ``N - K - n_entities``; deriving
    the effect count from whether the fitted model carries an intercept is what
    keeps this case conformant instead of assuming a constant always exists.
    """

    df = make_panel()
    res = PanelOLS.from_formula("y ~ x1 + x2 - 1", df, entity="firm", time="year").fit()

    assert "Intercept" not in res.params.index
    assert res.e["rank"] == 2
    assert res.e["df_m"] == 2  # no constant to exclude
    assert res.df_resid == N_OBS - 2 - N_ENTITIES == 13
    assert res.df_resid == N_OBS - 2 - N_ENTITIES + 1 - 1


def test_backend_df_convention_observed() -> None:
    """Pin the linearmodels behaviour the module docstring documents.

    With a constant present, ``df_resid = N - df_model`` equals Stata's
    ``N - K - n_entities + 1`` because the backend counts the ``n_entities - 1``
    absorbed effect columns.  ``auto_df`` does not move ``df_resid``; it only
    rescales the covariance when an effect is nested in the clusters.
    """

    from linearmodels.panel import PanelOLS as BackendPanelOLS

    df = make_panel()
    frame = df.set_index(["firm", "year"])
    exog = pd.DataFrame(
        {"Intercept": 1.0, "x1": frame["x1"], "x2": frame["x2"]}, index=frame.index
    )

    for auto_df in (True, False):
        backend = BackendPanelOLS(frame["y"], exog, entity_effects=True).fit(auto_df=auto_df)
        assert backend.nobs == N_OBS
        assert backend.df_model == K_WITH_CONSTANT + (N_ENTITIES - 1)  # 7
        assert backend.df_resid == FE_DF_R  # identical for both auto_df settings
        assert backend.df_resid == N_OBS - backend.df_model
        assert backend.df_resid == N_OBS - K_WITH_CONSTANT - N_ENTITIES + 1


def test_auto_df_on_reproduces_stata_clustered_standard_errors() -> None:
    """``auto_df`` must count only the reported coefficients, like Stata.

    Captured with StataNow 19.5 on 2026-10-24 from this exact panel
    (``make_panel()``, 5 firms x 4 years, ``xtset firm year``)::

        xtreg y x1 x2, fe vce(cluster firm)
            e(df_r) = 4   e(df_m) = 1
            se(x1) = .2024145   se(x2) = .1204298   se(_cons) = .03893807

    ``.2024145`` is ``sqrt(G/(G-1) * (N-1)/(N-3))`` applied to the within
    sandwich: Stata's ``K`` is the *reported* coefficient count (3), not that
    count plus the 4 absorbed firm effects (7).

    The previous version of this test asserted the opposite - it required
    ``auto_df=False`` (K = 7, factor 1.8269 instead of 1.3971) and claimed
    ``auto_df=True`` "would drop the effect count for nested clusters".  That is
    backwards: ``auto_df=True`` is the setting that matches Stata, and
    ``auto_df=False`` reproduces Stata's *undocumented* ``dfadj`` option, 14.3%
    higher.  Hence the renamed test and the new reference value.
    """

    from linearmodels.panel import PanelOLS as BackendPanelOLS

    df = make_panel()
    frame = df.set_index(["firm", "year"])
    exog = pd.DataFrame(
        {"Intercept": 1.0, "x1": frame["x1"], "x2": frame["x2"]}, index=frame.index
    )
    clusters = frame.index.get_level_values("firm").to_numpy()
    Xd, yd = fe_components(df)
    _, resid, _ = ols(Xd, yd)

    # ``cr1_cov`` takes the *denominator* of the (N-1)/(N-K) factor, so passing
    # N - K with Stata's K = 3 builds Stata's factor.
    expected = np.sqrt(
        np.diag(cr1_cov(Xd, resid, clusters, df_resid=N_OBS - K_WITH_CONSTANT))
    )
    # Stata's printout, to six decimal places.
    np.testing.assert_allclose(expected, [0.2024145, 0.1204298], rtol=1e-6)

    auto_on = BackendPanelOLS(frame["y"], exog, entity_effects=True).fit(
        cov_type="clustered", clusters=clusters, group_debias=True, auto_df=True
    )
    auto_off = BackendPanelOLS(frame["y"], exog, entity_effects=True).fit(
        cov_type="clustered", clusters=clusters, group_debias=True, auto_df=False
    )

    np.testing.assert_allclose(
        np.asarray(auto_on.std_errors)[1:], expected, rtol=0.0, atol=1e-10
    )
    # ... and the effect-counting setting is a different, larger number.
    assert not np.allclose(np.asarray(auto_off.std_errors)[1:], expected, rtol=1e-6)


# --------------------------------------------------------------------- unbalanced
def test_unbalanced_panel_fe_and_fd() -> None:
    df = make_panel()
    # Remove a scattered set of rows, including one that leaves entity 1 with a
    # gap (times 0, 1, 3) so first differences must not bridge missing periods.
    unbalanced = df.drop(index=[0, 6, 13, 19]).reset_index(drop=True)
    n = len(unbalanced)
    n_entities = unbalanced["firm"].nunique()

    fe = PanelOLS.from_formula("y ~ x1 + x2", unbalanced, entity="firm", time="year").fit()
    Xd, yd = fe_components(unbalanced)
    df_r = n - K_WITH_CONSTANT - n_entities + 1
    beta, _, cov = ols(Xd, yd, df_resid=df_r)

    np.testing.assert_allclose(fe.params[["x1", "x2"]].to_numpy(), beta, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        fe.bse[["x1", "x2"]].to_numpy(), np.sqrt(np.diag(cov)), rtol=0.0, atol=1e-10
    )
    assert fe.nobs == n
    assert fe.df_resid == df_r

    fd = FirstDifferenceOLS.from_formula(
        "y ~ x1 + x2", unbalanced, entity="firm", time="year"
    ).fit()
    # Build the reference differences on the *complete* period grid: a missing
    # period breaks the pair (Stata's L. does not skip a hole), so reindexing is
    # part of the definition, not a convenience.
    frame = unbalanced.set_index(["firm", "year"])
    grid = pd.Index(np.arange(0, N_PERIODS), name="year")
    columns = {}
    for column in ("y", "x1", "x2"):
        wide = frame[column].unstack("year").reindex(columns=grid)
        columns[column] = wide.diff(axis=1).stack(future_stack=True)
    diffs = pd.DataFrame(columns).dropna()
    fd_beta, _, fd_cov = ols(
        diffs[["x1", "x2"]].to_numpy(), diffs["y"].to_numpy(), df_resid=len(diffs) - 2
    )
    np.testing.assert_allclose(fd.params.to_numpy(), fd_beta, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        fd.bse.to_numpy(), np.sqrt(np.diag(fd_cov)), rtol=0.0, atol=1e-10
    )
    assert fd.nobs == len(diffs)
    # The dropped rows create holes, so the difference count is strictly below
    # the balanced-panel total of 15.
    assert fd.nobs < N_OBS - N_ENTITIES
    assert np.all(np.isfinite(fd.params.to_numpy()))


def test_unbalanced_re_and_be() -> None:
    df = make_panel()
    unbalanced = df.drop(index=[0, 7]).reset_index(drop=True)
    n = len(unbalanced)
    n_entities = unbalanced["firm"].nunique()

    re = RandomEffects.from_formula("y ~ x1 + x2", unbalanced, entity="firm", time="year").fit()
    assert re.nobs == n
    assert re.df_resid == n - K_WITH_CONSTANT

    be = BetweenOLS.from_formula("y ~ x1 + x2", unbalanced, entity="firm", time="year").fit()
    assert be.nobs == n_entities
    assert be.df_resid == n_entities - K_WITH_CONSTANT
    means = unbalanced.groupby("firm")[["y", "x1", "x2"]].mean()
    X = np.column_stack([np.ones(n_entities), means[["x1", "x2"]].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, means["y"].to_numpy(), rcond=None)
    np.testing.assert_allclose(be.params.to_numpy(), beta, rtol=0.0, atol=1e-10)


# --------------------------------------------------------------------- e() / summary
@pytest.mark.parametrize(
    "build",
    [
        lambda df: PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year"),
        lambda df: RandomEffects.from_formula("y ~ x1 + x2", df, entity="firm", time="year"),
        lambda df: BetweenOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year"),
        lambda df: FirstDifferenceOLS.from_formula(
            "y ~ x1 + x2", df, entity="firm", time="year"
        ),
    ],
    ids=["fe", "re", "be", "fd"],
)
def test_e_contains_all_required_keys(build) -> None:
    df = make_panel()
    res = build(df).fit()

    missing = [key for key in REQUIRED_E_KEYS if key not in res.e]
    assert not missing, f"missing e() keys: {missing}"
    for key in ("r2", "r2_a", "rmse", "F", "ll"):
        assert np.isfinite(float(res.e[key])), f"e({key}) is not finite"
    assert res.e["N"] > 0
    assert res.e["df_r"] > 0
    assert res.e["rank"] >= 1
    assert res.e["df_m"] == res.e["rank"] - (0 if "Intercept" not in res.params.index else 1)


def test_fe_r2_is_the_within_r_squared() -> None:
    """``xtreg, fe`` reports the within R-squared, not the levels R-squared."""

    df = make_panel()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()

    Xd, yd = fe_components(df)
    beta, resid, _ = ols(Xd, yd, df_resid=FE_DF_R)
    within_r2 = 1.0 - float(resid @ resid) / float(((yd - yd.mean()) ** 2).sum())

    np.testing.assert_allclose(res.e["r2"], within_r2, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(res.e["ll"], res.llf, rtol=0.0, atol=0.0)


def test_summary_stata_style_renders() -> None:
    df = make_panel()
    for model in (
        PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year"),
        RandomEffects.from_formula("y ~ x1 + x2", df, entity="firm", time="year"),
        BetweenOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year"),
        FirstDifferenceOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year"),
    ):
        text = model.fit().summary(style="stata")
        for token in ("Coef.", "Std. err.", "P>|t|", "Number of obs", "R-squared"):
            assert token in text, token


# --------------------------------------------------------------------- cov types
def test_unknown_cov_type_raises_value_error() -> None:
    df = make_panel()
    model = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year")
    with pytest.raises(ValueError, match="cov_type"):
        model.fit(cov_type="definitely-not-a-cov-type")


@pytest.mark.parametrize("cov_type", ["HC0", "HC2", "HC3", "HAC"])
def test_unrepresentable_cov_types_are_rejected(cov_type: str) -> None:
    df = make_panel()
    model = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year")
    with pytest.raises(UnsupportedError):
        model.fit(cov_type=cov_type)


def test_cluster_requires_labels() -> None:
    df = make_panel()
    model = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year")
    with pytest.raises(ValueError, match="clusters"):
        model.fit(cov_type="cluster")


def test_cluster_standard_errors_match_stata_cr1() -> None:
    """Stata's clustered SE uses ``K = reported coefficients``, not ``N - K``.

    Captured from StataNow 19.5 on 2026-10-24, ``xtset firm year`` then
    ``xtreg y x1 x2, fe vce(cluster firm)``::

        e(df_r) = 4        (= G - 1, the reference distribution)
        se(x1) = .2024145  se(x2) = .1204298  se(_cons) = .03893807

    The old expectation used ``df_resid=FE_DF_R`` (13) inside the
    ``(N-1)/(N-K)`` factor, i.e. ``K = 7`` = 3 reported + 4 absorbed effects.
    That is Stata's undocumented ``dfadj`` option and it is 14.3% too large; the
    default is ``K = 3``.  The reported ``e(df_r)`` meanwhile is ``G - 1 = 4``,
    not the model-based 13.
    """

    df = make_panel()
    res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit(
        cov_type="cluster", cov_kwds={"clusters": df["firm"].to_numpy()}
    )

    Xd, yd = fe_components(df)
    _, resid, _ = ols(Xd, yd)
    expected = np.sqrt(
        np.diag(cr1_cov(Xd, resid, df["firm"].to_numpy(), df_resid=N_OBS - K_WITH_CONSTANT))
    )
    np.testing.assert_allclose(expected, [0.2024145, 0.1204298], rtol=1e-6)
    np.testing.assert_allclose(
        res.bse[["x1", "x2"]].to_numpy(), expected, rtol=0.0, atol=1e-10
    )

    # Stata's e(df_r) under a clustered VCE is G - 1 = 4; the model-based value
    # lives in e(df_r_mb) = 13.
    assert res.df_resid == N_ENTITIES - 1  # 4 = G - 1
    assert res.e["df_r"] == N_ENTITIES - 1
    assert res.cov_type == "cluster"
    assert res.e["N_clust"] == N_ENTITIES


def test_clustered_df_m_and_rank_are_documented_deviations() -> None:
    """Pin ``e(df_m)``/``e(rank)``: deliberately Stata's + 1, and staying so.

    Stata's clustered ``xtreg, fe`` redefines both scalars against the *free
    parameters of the clustered problem* rather than the fitted model's columns.
    Captured with StataNow 19.5 on 2026-10-24, 30 entities x 6 periods
    (``xtset id t``), three slope counts::

        command                                  e(df_m) e(df_b) e(rank)  prints
        xtreg y x1,     fe vce(cluster id)          0      1       1      F(1, 29) = 211.29
        xtreg y x1 x2,  fe vce(cluster id)          1      2       2      F(2, 29) = 303.80
        xtreg y x1 x2 z, fe vce(cluster id)         2      3       3      F(3, 29) = 254.47

    i.e. ``e(df_m) = slopes - 1`` and ``e(rank) = slopes``, while the F it prints
    uses ``F(e(df_b), e(df_r))`` - so Stata's own ``e(df_m)`` contradicts the
    degrees of freedom on its own test line.  Stata keeps the unclustered values
    in ``e(rank_mb)`` (3 for two slopes), which is what shows the shift is a
    convention.

    westata stores ``e(df_m)`` = slopes and ``e(rank)`` = ``len(e(b))`` =
    slopes + 1, so both are exactly +1 against Stata's clustered values and the
    printed ``F(2, 29)`` label matches the statistic.  ``e(df_b)`` and ``e(df_r)``
    match Stata.  This test exists so the +1 does not get "fixed" to match Stata
    in isolation: shrinking ``e(rank)`` would silently change the size of
    ``e(b)``/``e(V)`` that the results container and postestimation rely on.
    See the "Known deviations" section of
    :mod:`westata.estimators.panel` and ``retest/fix/panel/FIXES.md``.
    """

    df = make_panel()
    clusters = df["firm"].to_numpy()
    for formula, slopes, stata_df_m, stata_rank in (
        ("y ~ x1", 1, 0, 1),
        ("y ~ x1 + x2", 2, 1, 2),
    ):
        res = PanelOLS.from_formula(formula, df, entity="firm", time="year").fit(
            cov_type="cluster", cov_kwds={"clusters": clusters}
        )
        # westata's own, self-consistent quantities.
        assert res.e["df_m"] == slopes
        assert res.e["rank"] == slopes + 1  # slopes + Intercept
        assert res.e["df_b"] == slopes
        # ... which are exactly Stata's clustered values plus one.
        assert res.e["df_m"] == stata_df_m + 1
        assert res.e["rank"] == stata_rank + 1
        # The two that must match Stata do match it.
        assert res.e["df_b"] == stata_df_m + 1
        assert res.e["df_r"] == N_ENTITIES - 1

    # The unclustered path keeps the parameter-vector reading for both, and
    # there e(rank) equals Stata's (no redefinition happens without clustering).
    plain = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()
    assert plain.e["rank"] == K_WITH_CONSTANT
    assert plain.e["df_m"] == 2  # slopes, not Stata's effects-inclusive 6


def test_robust_is_statas_clustered_vce_not_hc1() -> None:
    """``xtreg, fe vce(robust)`` *is* ``vce(cluster panelvar)`` in Stata.

    ``[XT] xtreg`` says so outright, and the live check agrees: on a 30x6 panel
    ``max|V_robust - V_cluster(id)| = 0.0``, while ``areg ..., absorb(id)
    vce(robust)`` returns a different matrix (constant SE ``.0372797`` vs
    ``.0020300``, 18.4x).  So the HC1 sandwich belongs to ``areg``/``FEOLS`` and
    not to this estimator.

    Captured from StataNow 19.5 on 2026-10-24 on this 5x4 test panel,
    ``xtreg y x1 x2, fe vce(robust)``::

        e(df_r) = 4   e(df_m) = 1
        se(x1) = .2024145  se(x2) = .1204298  se(_cons) = .03893807

    The previous version of this test asserted the HC1 sandwich
    ``(N / df_r) * (X'X)^-1 X'diag(e^2)X (X'X)^-1`` under the name "Stata HC1".
    That number is ``areg``'s, not ``xtreg``'s; it was 1.01x off on x1 but 3.2x
    off on the constant.
    """

    df = make_panel()
    model = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year")
    res = model.fit(cov_type="robust")
    same = model.fit(
        cov_type="cluster", cov_kwds={"clusters": df["firm"].to_numpy()}
    )

    # Stata's vce(robust) and vce(cluster panelvar) are the same matrix.
    np.testing.assert_allclose(res.cov_params(), same.cov_params(), rtol=0.0, atol=0.0)

    # westata's parameter order is [Intercept, x1, x2]; Stata prints the
    # constant last, so its own column order is (x1, x2, _cons).
    expected = np.array([0.03893807, 0.2024145, 0.1204298])
    np.testing.assert_allclose(res.bse.to_numpy(), expected, rtol=1e-6)
    assert res.df_resid == N_ENTITIES - 1  # Stata's e(df_r) = G - 1
    assert res.e["df_r"] == N_ENTITIES - 1
    # ``HC1`` is accepted as an alias for the same spelling, not as HC1.
    alias = model.fit(cov_type="HC1")
    np.testing.assert_allclose(alias.cov_params(), res.cov_params(), rtol=0.0, atol=0.0)
    assert res.cov_type == "robust"


def test_nonrobust_is_the_default() -> None:
    df = make_panel()
    model = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year")
    default = model.fit()
    explicit = model.fit(cov_type="nonrobust")
    assert default.cov_type == "nonrobust"
    np.testing.assert_allclose(default.bse.to_numpy(), explicit.bse.to_numpy(), rtol=0.0, atol=0.0)


def test_ssc_always_applies_stata_defaults() -> None:
    """``ssc`` follows :mod:`westata.vce.ssc`: on by default, switchable off."""

    from westata.protocols import SSC

    df = make_panel()
    model = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year")
    clusters = df["firm"].to_numpy()

    # ``SSC()`` is Stata's default, so it must reproduce the implicit settings.
    np.testing.assert_allclose(
        model.fit(ssc=SSC()).bse.to_numpy(), model.fit().bse.to_numpy(), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        model.fit(cov_type="cluster", cov_kwds={"clusters": clusters}, ssc=SSC()).bse.to_numpy(),
        model.fit(cov_type="cluster", cov_kwds={"clusters": clusters}).bse.to_numpy(),
        rtol=0.0,
        atol=0.0,
    )

    # Turning the correction off shrinks the unadjusted standard errors (no
    # N/(N-K) or effect-counting factor) - and the default CR1 answer is exactly
    # the fully corrected one.
    without = model.fit(ssc=SSC(adj=False)).bse.to_numpy()
    assert np.all(without < model.fit().bse.to_numpy())

    # The default clustered answer is exactly Stata's
    # ``xtreg y x1 x2, fe vce(cluster firm)`` (captured 2026-10-24:
    # se(x1) = .2024145, se(x2) = .1204298), i.e. the ``K = 3`` CR1 factor.
    Xd, yd = fe_components(df)
    _, resid, _ = ols(Xd, yd)
    expected = np.sqrt(
        np.diag(cr1_cov(Xd, resid, clusters, df_resid=N_OBS - K_WITH_CONSTANT))
    )
    np.testing.assert_allclose(
        model.fit(cov_type="cluster", cov_kwds={"clusters": clusters}).bse[["x1", "x2"]].to_numpy(),
        expected,
        rtol=0.0,
        atol=1e-10,
    )


# --------------------------------------------------------------------- input handling
def test_multiindex_input_matches_long_input() -> None:
    df = make_panel()
    indexed = df.set_index(["firm", "year"])
    long_res = PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()
    mi_res = PanelOLS.from_formula("y ~ x1 + x2", indexed, entity="firm", time="year").fit()
    np.testing.assert_allclose(long_res.params.to_numpy(), mi_res.params.to_numpy(), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(long_res.bse.to_numpy(), mi_res.bse.to_numpy(), rtol=0.0, atol=1e-12)


def test_row_order_does_not_change_results() -> None:
    df = make_panel()
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    for build in (
        lambda frame: PanelOLS.from_formula("y ~ x1 + x2", frame, entity="firm", time="year"),
        lambda frame: RandomEffects.from_formula("y ~ x1 + x2", frame, entity="firm", time="year"),
        lambda frame: FirstDifferenceOLS.from_formula(
            "y ~ x1 + x2", frame, entity="firm", time="year"
        ),
    ):
        a = build(df).fit()
        b = build(shuffled).fit()
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(a.bse.to_numpy(), b.bse.to_numpy(), rtol=0.0, atol=1e-12)


def test_duplicate_panel_cells_are_rejected() -> None:
    df = make_panel()
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ConsistencyError, match="repeated"):
        PanelOLS.from_formula("y ~ x1 + x2", duplicated, entity="firm", time="year")


def test_missing_panel_column_is_reported() -> None:
    df = make_panel()
    with pytest.raises(ConsistencyError, match="firm_id"):
        PanelOLS.from_formula("y ~ x1", df, entity="firm_id", time="year")


@pytest.mark.parametrize(
    "build",
    [
        lambda df: RandomEffects.from_formula("y ~ x1 + x2", df, entity="firm", time="year"),
        lambda df: BetweenOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year"),
        lambda df: FirstDifferenceOLS.from_formula(
            "y ~ x1 + x2", df, entity="firm", time="year"
        ),
    ],
    ids=["re", "be", "fd"],
)
def test_fixed_effects_are_rejected_where_stata_has_none(build) -> None:
    df = make_panel()
    with pytest.raises(UnsupportedError):
        build(df).fit(entity_effects=True)
    with pytest.raises(UnsupportedError):
        build(df).fit(time_effects=True)


def test_missing_values_are_dropped_listwise() -> None:
    df = make_panel()
    with_missing = df.copy()
    with_missing.loc[3, "x1"] = np.nan
    with_missing.loc[11, "y"] = np.nan

    res = PanelOLS.from_formula("y ~ x1 + x2", with_missing, entity="firm", time="year").fit()

    assert res.nobs == len(df) - 2
    assert res.nobs == len(res.resid)
    np.testing.assert_allclose(
        res.params.to_numpy(),
        PanelOLS.from_formula(
            "y ~ x1 + x2", df.drop(index=[3, 11]), entity="firm", time="year"
        )
        .fit()
        .params.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_cluster_labels_survive_listwise_deletion() -> None:
    """Labels are given for the *full* frame; they must follow the kept rows.

    This is the alignment trap: if the labels were consumed positionally after
    rows were dropped, every cluster after the first missing observation would
    be shifted by one and the standard errors would be quietly wrong.
    """

    df = make_panel()
    with_missing = df.copy()
    with_missing.loc[3, "x1"] = np.nan
    clean = df.drop(index=[3]).reset_index(drop=True)

    from_fully_observed = PanelOLS.from_formula(
        "y ~ x1 + x2", clean, entity="firm", time="year"
    ).fit(cov_type="cluster", cov_kwds={"clusters": clean["firm"].to_numpy()})
    from_frame_with_na = PanelOLS.from_formula(
        "y ~ x1 + x2", with_missing, entity="firm", time="year"
    ).fit(cov_type="cluster", cov_kwds={"clusters": with_missing["firm"].to_numpy()})

    assert from_frame_with_na.nobs == len(clean)
    np.testing.assert_allclose(
        from_frame_with_na.bse.to_numpy(),
        from_fully_observed.bse.to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )

    # And that shared answer is Stata's CR1 on the retained sample.  Captured
    # 2026-10-24 with ``xtreg y x1 x2, fe vce(cluster firm)`` on the same
    # 19-observation panel (row 3's x1 set to missing): N = 19, e(df_r) = 4,
    # se(x1) = .19931035, se(x2) = .12090296, se(_cons) = .0355033.  Stata's
    # K in the (N-1)/(N-K) factor is again the reported 3.
    Xd, yd = fe_components(clean)
    _, resid, _ = ols(Xd, yd)
    expected = np.sqrt(
        np.diag(
            cr1_cov(
                Xd, resid, clean["firm"].to_numpy(), df_resid=len(clean) - K_WITH_CONSTANT
            )
        )
    )
    np.testing.assert_allclose(expected, [0.19931035, 0.12090296], rtol=1e-6)
    np.testing.assert_allclose(
        from_frame_with_na.bse[["x1", "x2"]].to_numpy(), expected, rtol=0.0, atol=1e-10
    )
    assert from_frame_with_na.e["df_r"] == N_ENTITIES - 1  # 4 = G - 1


def test_panel_indexed_cluster_series_is_aligned_by_index() -> None:
    df = make_panel()
    with_missing = df.copy()
    with_missing.loc[3, "x1"] = np.nan
    labels = pd.Series(
        with_missing["firm"].to_numpy(),
        index=pd.MultiIndex.from_frame(with_missing[["firm", "year"]]),
    )
    positional = PanelOLS.from_formula(
        "y ~ x1 + x2", with_missing, entity="firm", time="year"
    ).fit(cov_type="cluster", cov_kwds={"clusters": with_missing["firm"].to_numpy()})
    by_index = PanelOLS.from_formula(
        "y ~ x1 + x2", with_missing, entity="firm", time="year"
    ).fit(cov_type="cluster", cov_kwds={"clusters": labels})

    np.testing.assert_allclose(by_index.bse.to_numpy(), positional.bse.to_numpy(), rtol=0.0, atol=1e-12)
