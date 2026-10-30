"""``estat gof`` after ``logit``/``probit`` -- ``lfit.ado``.

Round 84 found the gap: ``logit_estat.ado:17-19`` dispatches ``gof`` to ``lfit``,
which is a **different test** from the Poisson ``poisgof``. The refusal at the
time said "gof is only defined after poisson", which was the wrong reason.

``lfit.ado:44-97`` (the default, no ``group()``) collapses observations by
covariate pattern and forms

    chi2 = sum_j (obs_j - exp_j)^2 / (exp_j (1 - exp_j / w_j)),   df = m - k

The ``group(#)`` variant is the Hosmer-Lemeshow test, which needs the
quantile-collapsing loop at ``lfit.ado:252-320``; it is refused explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture()
def discrete_frame():
    """A DISCRETE regressor, so covariate patterns repeat.

    With a continuous regressor every pattern is unique (``m`` equals N) and the
    Pearson statistic degenerates -- the test would still run but would not
    exercise the collapsing. Repeated patterns are what make the check meaningful.
    """

    rng = np.random.default_rng(43)
    n = 400
    x1 = rng.integers(0, 4, size=n).astype(float)
    probability = 1.0 / (1.0 + np.exp(-(-1.0 + 0.8 * x1)))
    outcome = (rng.uniform(size=n) < probability).astype(int)
    return pd.DataFrame({"yb": outcome, "x1": x1})


def _hand_pearson(result, frame, regressor="x1"):
    """Recompute the statistic straight from ``lfit.ado:44-97``."""

    fitted = np.asarray(result.fittedvalues, dtype=float)
    observed = frame["yb"].to_numpy()
    values = frame[regressor].to_numpy()
    chi2 = 0.0
    patterns = 0
    for level in np.unique(values):
        mask = values == level
        weight = float(mask.sum())
        expected = float(fitted[mask].sum())
        successes = float(observed[mask].sum())
        if not 0.0 < expected < weight:
            continue
        chi2 += (successes - expected) ** 2 / (expected * (1.0 - expected / weight))
        patterns += 1
    return chi2, patterns


def test_logit_gof_matches_lfit_pearson(discrete_frame) -> None:
    result = ws.Logit.from_formula("yb ~ x1", discrete_frame).fit()

    rendered = result.estat_gof()
    expected, patterns = _hand_pearson(result, discrete_frame)

    assert rendered.chi2 == pytest.approx(expected, rel=1e-12)
    assert rendered.patterns == patterns


def test_logit_gof_df_is_patterns_minus_parameters(discrete_frame) -> None:
    """``lfit.ado:96``: ``ret scalar df = return(m) - `nvar'``.

    ``nvar`` comes from ``Getivars`` (``lfit.ado:480-504``), which counts the
    COLUMNS of ``e(b)`` INCLUDING ``_cons`` -- it strips ``_cons`` from the
    ``ivars`` variable list but never decreases ``nvar``.  So with one regressor
    plus an intercept the subtraction is 2, not 1.
    """

    result = ws.Logit.from_formula("yb ~ x1", discrete_frame).fit()

    rendered = result.estat_gof()

    assert rendered.patterns == 4  # four discrete levels
    assert rendered.df == 2.0  # 4 - (1 regressor + _cons)
    assert rendered.df == float(rendered.patterns - 2)


def test_probit_gof_uses_the_same_lfit_path(discrete_frame) -> None:
    """``lfit.ado:116-119`` labels the table "logistic" or "probit"; the statistic
    is the same Pearson form."""

    logit = ws.Logit.from_formula("yb ~ x1", discrete_frame).fit()
    probit = ws.Probit.from_formula("yb ~ x1", discrete_frame).fit()

    assert probit.estat_gof().df == logit.estat_gof().df


def test_logit_gof_p_value_is_the_upper_chi2_tail(discrete_frame) -> None:
    from scipy import stats as scipy_stats

    result = ws.Logit.from_formula("yb ~ x1", discrete_frame).fit()
    rendered = result.estat_gof()

    assert rendered.pvalue == pytest.approx(
        float(scipy_stats.chi2.sf(rendered.chi2, rendered.df)), rel=1e-12
    )


def test_logit_gof_reports_the_r_names(discrete_frame) -> None:
    """``lfit.ado:91-98`` returns FIVE names: ``N``, ``m``, ``chi2``, ``df`` and ``p``.

    ``N`` was missing: the docstring quoted the range correctly all along, but
    ``to_dict`` omitted the observation count, so a caller reading Stata's ``r(N)``
    found nothing.
    """

    result = ws.Logit.from_formula("yb ~ x1", discrete_frame).fit()
    table = result.estat_gof().to_dict()

    assert set(table) == {"N", "m", "chi2", "df", "p"}
    assert table["N"] == float(len(discrete_frame))
    assert table["m"] == 4.0


def test_logit_gof_is_not_the_poisson_test(discrete_frame) -> None:
    """The two commands run different statistics.

    ``poisgof`` reports a deviance and a Pearson statistic with
    ``df = N - k - 1``; ``lfit`` reports a single Pearson statistic with
    ``df = m - k``.  If ``logit`` had been wired to the Poisson path the df would
    differ, so comparing the shapes pins the dispatch.
    """

    result = ws.Logit.from_formula("yb ~ x1", discrete_frame).fit()
    rendered = result.estat_gof()

    assert hasattr(rendered, "patterns")  # LfitResult
    assert not hasattr(rendered, "deviance")  # not GofResult


def test_a_continuous_regressor_gives_one_pattern_per_observation() -> None:
    """Each observation is its own pattern, so ``m == N`` and ``df == N - k``.

    That is the degenerate case the statistic collapses to; recording it keeps the
    behaviour from looking like a bug.
    """

    rng = np.random.default_rng(47)
    n = 120
    x1 = rng.normal(size=n)
    outcome = (rng.uniform(size=n) < 1 / (1 + np.exp(-x1))).astype(int)
    frame = pd.DataFrame({"yb": outcome, "x1": x1})

    result = ws.Logit.from_formula("yb ~ x1", frame).fit()
    rendered = result.estat_gof()

    assert rendered.patterns == n
    # ``nvar`` counts the intercept, so the subtraction is 2 (lfit.ado:480-504).
    assert rendered.df == float(n - 2)


def test_hosmer_lemeshow_collapses_into_equal_frequency_classes() -> None:
    """``group(#)`` selects the Hosmer-Lemeshow variant (``lfit.ado:199-250``).

    Observations are sorted by the fitted probability and collapsed into ``#``
    quantile classes, then the Pearson statistic is formed over the classes with
    ``df = # - 2``.

    Checked with a CONTINUOUS regressor, where every class can actually be filled --
    with a discrete regressor the number of distinct fitted probabilities caps the
    class count, and asserting ``#`` classes there would be asserting a fiction.
    """

    rng = np.random.default_rng(587)
    n = 1000
    x1 = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(0.5 + 1.2 * x1)))
    frame = pd.DataFrame(
        {"x1": x1, "yb": (rng.uniform(size=n) < probability).astype(int)}
    )

    result = ws.Logit.from_formula("yb ~ x1", frame).fit()

    for groups in (5, 10, 20):
        rendered = result.estat_gof(group=groups)
        assert rendered.patterns == groups, groups
        assert rendered.df == float(groups - 2), groups
        assert np.isfinite(rendered.chi2)
        assert 0.0 <= rendered.pvalue <= 1.0


def test_the_class_sizes_are_balanced() -> None:
    """The quantile collapse must be near-equal-frequency.

    ``lfit.ado:268`` scales the cumulative count by ``group/obs``, so each class holds
    ``obs/group`` observations.  An off-by-one in the boundary put the whole first
    class at a single observation during development, which still produced a
    plausible-looking chi-square -- so the SIZE distribution is asserted directly.
    """

    rng = np.random.default_rng(593)
    n = 1000
    x1 = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(0.3 + x1)))
    frame = pd.DataFrame(
        {"x1": x1, "yb": (rng.uniform(size=n) < probability).astype(int)}
    )
    result = ws.Logit.from_formula("yb ~ x1", frame).fit()

    fitted = np.asarray(result.fittedvalues, dtype=float)
    scaled = 10 * np.arange(1, n + 1, dtype=float) / n
    classes = np.minimum(np.floor(scaled).astype(int) + 1, 10)
    sizes = np.bincount(classes)

    assert len(sizes) == 11  # index 0 unused
    for size in sizes[1:]:
        assert abs(int(size) - n // 10) <= 2, sizes


def test_hosmer_lemeshow_needs_a_sensible_group_count() -> None:
    """``group(1)`` cannot compare two classes, so it is refused."""

    rng = np.random.default_rng(599)
    n = 300
    x1 = rng.normal(size=n)
    frame = pd.DataFrame(
        {"x1": x1, "yb": (rng.uniform(size=n) < 1 / (1 + np.exp(-x1))).astype(int)}
    )
    result = ws.Logit.from_formula("yb ~ x1", frame).fit()

    with pytest.raises(ValueError, match="at least 2"):
        result.estat_gof(group=1)


def test_the_hl_header_names_the_model_and_the_response() -> None:
    """``lfit.ado:116-120`` prints the same header for both branches.

    The first version of the grouped branch hardcoded "logistic", so a probit fit
    reported "Goodness-of-fit test after logistic model" -- a claim about which model
    was fitted, not a cosmetic difference.
    """

    rng = np.random.default_rng(601)
    n = 800
    x1 = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(0.4 + 1.0 * x1)))
    frame = pd.DataFrame(
        {"x1": x1, "yb": (rng.uniform(size=n) < probability).astype(int)}
    )

    logit_text = str(ws.Logit.from_formula("yb ~ x1", frame).fit().estat_gof(group=10))
    probit_text = str(
        ws.Probit.from_formula("yb ~ x1", frame).fit().estat_gof(group=10)
    )

    assert "logistic model" in logit_text
    assert "probit model" in probit_text
    assert "logistic" not in probit_text
    # The response line belongs to both branches.
    assert "Variable: yb" in logit_text
    assert "Variable: yb" in probit_text


def test_the_hl_statistic_is_not_the_pearson_one() -> None:
    """Collapsing into classes changes the statistic substantially.

    If the grouped call silently returned the uncollapsed Pearson number, both would
    agree -- and the grouped one would be mislabelled rather than wrong-looking.
    """

    rng = np.random.default_rng(607)
    n = 800
    x1 = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(0.4 + 1.0 * x1)))
    frame = pd.DataFrame(
        {"x1": x1, "yb": (rng.uniform(size=n) < probability).astype(int)}
    )
    result = ws.Logit.from_formula("yb ~ x1", frame).fit()

    uncollapsed = result.estat_gof()
    collapsed = result.estat_gof(group=10)

    assert collapsed.chi2 != pytest.approx(uncollapsed.chi2, rel=1e-6)
    assert collapsed.df != uncollapsed.df


def test_other_estimators_still_refuse_gof(discrete_frame) -> None:
    """``nbreg`` has no ``estat gof`` at all (it sets no ``estat_cmd``)."""

    rng = np.random.default_rng(53)
    frame = discrete_frame.assign(ycnt=rng.poisson(3.0, size=len(discrete_frame)))
    result = ws.NegativeBinomial.from_formula("ycnt ~ x1", frame).fit()

    with pytest.raises(NotImplementedError):
        result.estat_gof()


def test_poisson_gof_is_unchanged(discrete_frame) -> None:
    """Regression guard: the Poisson path must keep its deviance test."""

    rng = np.random.default_rng(59)
    frame = discrete_frame.assign(ycnt=rng.poisson(3.0, size=len(discrete_frame)))
    result = ws.Poisson.from_formula("ycnt ~ x1", frame).fit()

    rendered = result.estat_gof()

    assert hasattr(rendered, "deviance")
    assert np.isfinite(rendered.deviance)


def test_boundary_patterns_still_count_towards_df() -> None:
    """A pattern whose expected count leaves ``(0, w)`` is skipped for the
    statistic but STILL counts towards ``m``.

    ``lfit.ado:83-92`` sets ``touse = (w > 0)`` -- the expected count plays no part
    -- and then ``ret scalar m = r(N)``.  Only the chi-square TERM is guarded
    (``lfit.ado:85``).  So a fitted probability reaching 0 or 1 removes a term
    without removing a degree of freedom.

    Counting only contributing patterns would quietly shrink ``df``, and because
    ``df`` feeds the p-value the error would never look wrong -- it would just be
    a smaller number.
    """

    # Perfect separation: every fitted probability is exactly 0 or 1, so no
    # pattern contributes a term, yet all of them still form the group count.
    x = np.repeat([0.0, 100.0], 50)
    outcome = np.tile([0, 1], 50).astype(int)
    frame = pd.DataFrame({"yb": outcome, "x": x})

    result = ws.Logit.from_formula("yb ~ x", frame).fit()
    rendered = result.estat_gof()

    # Two covariate patterns exist; neither contributes a chi-square term.
    assert rendered.patterns == 2
    assert rendered.chi2 == pytest.approx(0.0)
    # df = m - nvar = 2 - 2 (one regressor plus the intercept).
    assert rendered.df == 0.0


def test_lfit_is_not_exposed_as_a_standalone_command() -> None:
    """``lfit.sthlp:6-8`` calls it an out-of-date command superseded by
    ``estat gof``, so it must not appear as a public entry point.

    That is why the implementation lives inside ``estat_gof`` rather than as a
    separate ``ws.lfit``: adding the deprecated name would invent a surface Stata
    has retired.
    """

    assert not hasattr(ws, "lfit")


def test_a_frequency_weighted_fit_refuses_instead_of_crashing() -> None:
    """``lfit_p.ado:123-128`` builds ``w`` from the estimation weight, so the test
    is weight-aware in Stata.

    Here a ``freq_weights`` fit expands the result arrays (``nobs`` becomes
    ``sum(w)``) while the design keeps one row per observation, and the pieces no
    longer line up: the residual vector came back with FIVE entries against a
    300-row design.  The covariate-pattern grouping is therefore not recoverable
    from them, and guessing an alignment would silently group the wrong rows.

    Refusing with an explanation is the honest answer; the alternative was a
    ``ValueError`` about array lengths, which says nothing about why.
    """

    rng = np.random.default_rng(61)
    n = 300
    x1 = rng.integers(0, 4, size=n).astype(float)
    weights = rng.integers(1, 4, size=n)
    probability = 1.0 / (1.0 + np.exp(-(-1.0 + 0.8 * x1)))
    frame = pd.DataFrame(
        {
            "yb": (rng.uniform(size=n) < probability).astype(int),
            "x1": x1,
            "w": weights,
        }
    )

    result = ws.Logit.from_formula("yb ~ x1", frame, freq_weights="w").fit()

    with pytest.raises(ValueError, match="freq_weights"):
        result.estat_gof()


def test_the_unweighted_path_still_works() -> None:
    """Regression guard: the alignment check must not disturb ordinary fits."""

    rng = np.random.default_rng(67)
    n = 200
    x1 = rng.integers(0, 3, size=n).astype(float)
    probability = 1.0 / (1.0 + np.exp(-(-0.5 + 0.7 * x1)))
    frame = pd.DataFrame(
        {"yb": (rng.uniform(size=n) < probability).astype(int), "x1": x1}
    )

    result = ws.Logit.from_formula("yb ~ x1", frame).fit()
    rendered = result.estat_gof()

    assert rendered.patterns == 3
    assert np.isfinite(rendered.chi2)


# ------------------------------------------------- the rendered lfit table
def test_the_table_names_the_model_kind(discrete_frame) -> None:
    """``lfit.ado:116-119`` prints "logistic" for logit and "probit" for probit."""

    logit = ws.Logit.from_formula("yb ~ x1", discrete_frame).fit()
    probit = ws.Probit.from_formula("yb ~ x1", discrete_frame).fit()

    assert "logistic model" in str(logit.estat_gof())
    assert "probit model" in str(probit.estat_gof())


def test_the_table_prints_the_response_name(discrete_frame) -> None:
    """``lfit.ado:120``: ``di as smcl "Variable: " as res "`y'"``.

    The line is easy to miss and the value has to come from somewhere: the result
    did not record ``response_name`` or ``e(depvar)`` at all, so the header would
    have printed an empty variable.  Both are now set by the discrete estimators,
    which ``logit.sthlp:264`` documents anyway ("name of dependent variable").
    """

    result = ws.Logit.from_formula("yb ~ x1", discrete_frame).fit()

    rendered = str(result.estat_gof())

    assert "Variable: yb" in rendered
    assert result.e["depvar"] == "yb"
    assert result.response_name == "yb"


def test_discrete_estimators_record_e_depvar() -> None:
    """``logit.sthlp:264`` and ``poisson.sthlp:244`` both list e(depvar)."""

    rng = np.random.default_rng(71)
    n = 150
    x1 = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "x1": x1,
            "yb": (rng.uniform(size=n) < 1 / (1 + np.exp(-x1))).astype(int),
            "ycnt": rng.poisson(3.0, size=n),
        }
    )

    for estimator, formula, expected in (
        (ws.Logit, "yb ~ x1", "yb"),
        (ws.Probit, "yb ~ x1", "yb"),
        (ws.Poisson, "ycnt ~ x1", "ycnt"),
    ):
        result = estimator.from_formula(formula, frame).fit()
        assert result.e["depvar"] == expected, estimator.__name__
        assert result.response_name == expected, estimator.__name__


def test_tobit_records_e_depvar() -> None:
    """``tobit.sthlp`` lists e(depvar) like every other estimation command.

    ``Tobit`` does not descend from ``_DiscreteBase`` (its MRO is just
    ``(Tobit, object)``), so it needed its own wiring -- the shared discrete fix
    did not reach it.
    """

    rng = np.random.default_rng(73)
    n = 150
    x1 = rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "y": 1.0 + x1 + rng.normal(size=n)})

    result = ws.Tobit.from_formula("y ~ x1", frame).fit()

    assert result.e["depvar"] == "y"
    assert result.response_name == "y"


def test_the_e_property_falls_back_to_response_name() -> None:
    """``e(depvar)`` must appear even when the estimator only set the attribute.

    ``StataResults.e`` falls back so a consumer reading ``e()`` cannot tell
    "not recorded" from "no dependent variable" -- they would see a missing key and
    have no way to know which it was.
    """

    rng = np.random.default_rng(79)
    n = 100
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )
    result = ws.OLS.from_formula("y ~ x1", frame).fit()

    # OLS does not put depvar in its e() dict directly.
    result.response_name = "y_from_attribute"
    result._e.pop("depvar", None)

    assert result.e["depvar"] == "y_from_attribute"


def test_panel_estimators_record_e_depvar() -> None:
    """Regression guard for the panel family, wired in round 81."""

    rng = np.random.default_rng(83)
    entities, periods = 20, 5
    firm = np.repeat(np.arange(entities), periods)
    n = firm.size
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "firm": firm,
            "year": np.tile(np.arange(periods), entities),
        }
    )

    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    assert result.e["depvar"] == "y"


# ------------------------------------- every family records e(depvar)
def _depvar_frame():
    rng = np.random.default_rng(89)
    n = 160
    latent = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": latent,
            "w": np.maximum(latent, 0.5),
            "lo": np.maximum(latent, 0.5),
            "hi": np.maximum(latent, 0.5),
            "x1": rng.normal(size=n),
            "rep": rng.integers(2, 5, size=n),
        }
    )


DEPVAR_CASES = {
    "Tobit": (lambda f: ws.Tobit.from_formula("y ~ x1", f).fit(), "y"),
    "Truncreg": (lambda f: ws.Truncreg.from_formula("w ~ x1", f, lower=0).fit(), "w"),
    "Intreg": (lambda f: ws.Intreg.from_formula("lo hi ~ x1", f).fit(), "lo hi"),
    "OrderedLogit": (
        lambda f: ws.OrderedLogit.from_formula("rep ~ x1", f).fit(),
        "rep",
    ),
    "OrderedProbit": (
        lambda f: ws.OrderedProbit.from_formula("rep ~ x1", f).fit(),
        "rep",
    ),
    "QuantileRegression": (
        lambda f: ws.QuantileRegression.from_formula("y ~ x1", f).fit(),
        "y",
    ),
    "RobustRegression": (
        lambda f: ws.RobustRegression.from_formula("y ~ x1", f).fit(),
        "y",
    ),
}


@pytest.mark.parametrize("label", sorted(DEPVAR_CASES))
def test_each_family_records_e_depvar(label) -> None:
    """Every estimation command documents ``e(depvar)``.

    ``tobit.sthlp``, ``truncreg.sthlp``, ``intreg.sthlp``, ``ologit.sthlp``,
    ``qreg.sthlp`` and ``rreg.sthlp`` all list "name of dependent variable".  Each
    of these estimators builds its own ``StataResults`` -- ``_Ordered.fit`` even
    overrides the shared one -- so every one needed its own wiring.
    """

    factory, expected = DEPVAR_CASES[label]
    result = factory(_depvar_frame())

    assert result.e["depvar"] == expected


def test_intreg_records_both_interval_columns() -> None:
    """``intreg``'s response is an interval given by TWO columns.

    Stata stores both names, space-separated, because neither alone identifies the
    dependent variable.
    """

    frame = _depvar_frame()
    result = ws.Intreg.from_formula("lo hi ~ x1", frame).fit()

    assert result.e["depvar"] == "lo hi"
    assert " " in result.e["depvar"]
