"""Identities that the stored ``e()`` scalars must satisfy.

Rounds 92-94 found two defects by comparing ``e()`` values against each other
rather than against a fixture: a fabricated ``e(df_r)`` on ML models (caught by
``N - df_r == rank``) and a bad suppression rule (caught because ``xtivreg``
legitimately stores it).  Neither would have surfaced from existence checks alone.

The identities here are algebraic, so they need no Stata reference:

* ``r2 == mss / (mss + rss)``;
* ``ll == -N/2 * (log(2*pi) + log(rss/N) + 1)`` for a Gaussian likelihood;
* ``chi2 == 2 * (ll - ll_0)`` for an LR test;
* ``N - df_r == rank`` where ``df_r`` exists and the model has an intercept.

A failure means two stored numbers disagree with each other -- a category of bug
that looks perfectly plausible in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(101)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 0.8 * x1 - 0.3 * x2 + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": latent,
            "x1": x1,
            "x2": x2,
            "yb": (latent > 0).astype(int),
            "ycnt": rng.poisson(3.0, size=n),
        }
    )


def test_r2_decomposes_into_mss_and_rss(frame) -> None:
    """``r2 = mss / (mss + rss)`` -- the two sums of squares must agree with it."""

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    e = result.e

    expected = e["mss"] / (e["mss"] + e["rss"])

    assert e["r2"] == pytest.approx(expected, rel=1e-12)


def test_the_gaussian_loglik_follows_from_rss(frame) -> None:
    """``ll = -N/2 * (log(2*pi) + log(rss/N) + 1)`` for an OLS fit.

    ``e(ll)`` and ``e(rss)`` are stored independently; if either drifted the
    identity would break even though both numbers look reasonable on their own.
    """

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    e = result.e
    n = float(e["N"])

    expected = -n / 2.0 * (np.log(2.0 * np.pi) + np.log(e["rss"] / n) + 1.0)

    assert e["ll"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("label", ["Logit", "Poisson"])
def test_the_lr_statistic_is_twice_the_loglikelihood_gap(frame, label) -> None:
    """``chi2 = 2 * (ll - ll_0)`` -- the LR test's own definition."""

    if label == "Logit":
        result = ws.Logit.from_formula("yb ~ x1 + x2", frame).fit()
    else:
        result = ws.Poisson.from_formula("ycnt ~ x1", frame).fit()
    e = result.e

    assert e["chi2"] == pytest.approx(2.0 * (e["ll"] - e["ll_0"]), rel=1e-10)


def test_the_ll0_of_a_null_model_is_the_constant_only_fit(frame) -> None:
    """``e(ll_0)`` is the log-likelihood of the model with no regressors.

    Checked by fitting that model directly rather than trusting the stored number.
    """

    full = ws.Logit.from_formula("yb ~ x1 + x2", frame).fit()
    null = ws.Logit.from_formula("yb ~ 1", frame).fit()

    assert full.e["ll_0"] == pytest.approx(null.e["ll"], rel=1e-10)


def test_n_minus_df_r_equals_rank_where_df_r_exists(frame) -> None:
    """For a model with an intercept, ``N - df_r == rank``.

    This is the identity that exposed the fabricated ``e(df_r)`` on logit: it gave
    0 where the rank was 2.  It applies only where ``df_r`` is stored at all --
    ``logit``/``poisson``/``nbreg``/``ologit`` document ``e(df_m)`` only, and
    ``tobit`` defines ``df_r = N - df_m`` so its sum is ``N`` by construction.
    """

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    e = result.e

    assert e["N"] - e["df_r"] == e["rank"]


def test_tobit_uses_its_own_df_convention(frame) -> None:
    """``tobit.ado`` defines ``df_m = e(rank) - 1 - hascons`` and ``df_r = N - df_m``.

    So ``df_m + df_r == N`` for tobit, unlike OLS where ``N - df_r == rank``.  The
    two conventions differ and both are correct; an invariant written for one is
    wrong for the other -- which is exactly the false alarm round 94 produced.
    """

    result = ws.Tobit.from_formula("y ~ x1 + x2", frame).fit()
    e = result.e

    assert e["df_m"] + e["df_r"] == e["N"]
    # And the rank includes lnsigma, so it exceeds df_m by more than the intercept.
    assert e["rank"] == e["df_m"] + 2  # intercept + lnsigma


def test_z_models_carry_no_df_r(frame) -> None:
    """The identity cannot be applied where the key is absent, so pin the absence."""

    for result in (
        ws.Logit.from_formula("yb ~ x1", frame).fit(),
        ws.Poisson.from_formula("ycnt ~ x1", frame).fit(),
        ws.NegativeBinomial.from_formula("ycnt ~ x1", frame).fit(),
    ):
        assert "df_r" not in result.e, result.model_name


def test_n_minus_df_r_equals_rank_without_an_intercept_too(frame) -> None:
    """The identity is about ``rank``, not about the intercept.

    With ``- 1`` the rank drops to 2 and ``df_r`` rises to ``N - 2``, so the
    relation holds on both sides of the intercept question.
    """

    with_constant = ws.OLS.from_formula("y ~ x1 + x2", frame).fit().e
    without = ws.OLS.from_formula("y ~ x1 + x2 - 1", frame).fit().e

    assert with_constant["N"] - with_constant["df_r"] == with_constant["rank"] == 3
    assert without["N"] - without["df_r"] == without["rank"] == 2


def test_panel_df_r_counts_the_absorbed_groups(frame) -> None:
    """``xtreg, fe`` uses a DIFFERENT convention: ``df_r`` subtracts the groups.

    For 60 entities and one regressor, ``N - df_r`` came out at 61 (= 60 groups +
    1), not at ``rank`` (2).  Both numbers are right for their own command --
    another case where an invariant written for OLS does not transfer.

    ``xtreg.sthlp:1023`` documents ``e(df_r)`` as "residual degrees of freedom"
    without spelling out the absorbed-group subtraction, so the convention is
    recorded here from the implementation rather than asserted against a reference.
    """

    panel = frame.assign(
        firm=np.repeat(np.arange(60), 5), year=np.tile(np.arange(5), 60)
    )
    result = ws.PanelOLS.from_formula(
        "y ~ x1", panel, entity="firm", time="year"
    ).fit()
    e = result.e

    groups = int(panel["firm"].nunique())
    # ``df_r`` absorbs the entity effects, so N - df_r exceeds the coefficient rank.
    assert e["N"] - e["df_r"] == groups + 1
    assert e["rank"] == 2


def test_the_rank_of_a_tobit_includes_lnsigma(frame) -> None:
    """``tobit``'s ``e(rank)`` covers the slopes, the intercept AND ``lnsigma``."""

    result = ws.Tobit.from_formula("y ~ x1 + x2", frame).fit()
    e = result.e

    # 2 slopes + intercept + lnsigma
    assert e["rank"] == 4
    # but df_m counts only the slopes, and df_r = N - df_m (tobit.ado)
    assert e["df_m"] == 2
    assert e["df_r"] == e["N"] - e["df_m"]


def test_tobit_reports_k_aux(frame) -> None:
    """``tobit.ado:153``: ``ereturn scalar k_aux = 1``.

    The one auxiliary parameter is ``lnsigma``.  ``k_aux`` is what lets Stata print
    the LR test with ``df_m`` degrees of freedom while ``rank`` covers the slopes,
    the intercept and lnsigma -- so the three numbers are related but distinct:

        rank == df_m + 1 (intercept) + k_aux

    ``nbreg`` and ``ologit`` already recorded this key; ``tobit`` is a separate
    module (its MRO is ``(Tobit, object)``), so it needed its own wiring.
    """

    result = ws.Tobit.from_formula("y ~ x1 + x2", frame).fit()
    e = result.e

    assert e["k_aux"] == 1
    assert e["rank"] == e["df_m"] + 1 + e["k_aux"]


def test_k_aux_is_consistent_for_the_other_limited_models(frame) -> None:
    """``nbreg`` has one auxiliary parameter, ``ologit`` as many as its cuts.

    The same ``rank == df_m + 1 + k_aux`` relation must hold, which is what makes
    ``k_aux`` an explanatory quantity rather than a loose label.
    """

    nbreg = ws.NegativeBinomial.from_formula("ycnt ~ x1", frame).fit()
    assert nbreg.e["k_aux"] == 1
    assert nbreg.e["rank"] == nbreg.e["df_m"] + 1 + nbreg.e["k_aux"]

    # Ordered logit needs an ordinal response; derive one from the count.
    ordered_frame = frame.assign(rep=np.clip(frame["ycnt"], 1, 3))
    ologit = ws.OrderedLogit.from_formula("rep ~ x1", ordered_frame).fit()
    # ``K`` categories have ``K - 1`` cut points, so k_aux is 2 for three levels.
    assert ologit.e["k_aux"] == 2
    # NO intercept here: an ordered model identifies location through its cuts, so
    # rank = slopes + cuts (not slopes + intercept + cuts).
    assert ologit.e["rank"] == ologit.e["df_m"] + ologit.e["k_aux"]


def test_truncreg_and_intreg_report_k_aux(frame) -> None:
    """``truncreg.ado:252`` and ``intreg.ado:418``: ``k_aux = 1``.

    Each carries exactly one auxiliary parameter -- ``sigma``.  Both live in
    ``limited.py``, which builds its own e() rather than inheriting the discrete
    one, so the key had to be added there separately.
    """

    censored = frame.assign(w=np.maximum(frame["y"], 0.5))
    bounds = censored.assign(lo=censored["w"], hi=censored["w"])

    truncreg = ws.Truncreg.from_formula("w ~ x1", censored, lower=0).fit()
    intreg = ws.Intreg.from_formula("lo hi ~ x1", bounds).fit()

    for result in (truncreg, intreg):
        assert result.e["k_aux"] == 1, result.model_name
        assert (
            result.e["rank"] == result.e["df_m"] + 1 + result.e["k_aux"]
        ), result.model_name


@pytest.mark.parametrize("label", ["Logit", "Poisson"])
def test_models_without_auxiliary_parameters_omit_k_aux(frame, label) -> None:
    """``logit.ado`` and ``poisson.ado`` contain no ``k_aux`` at all.

    They have no auxiliary parameter, so Stata never sets the macro.  Recording
    the absence keeps a future "add k_aux everywhere" change from inventing a zero
    where Stata has no key.
    """

    if label == "Logit":
        result = ws.Logit.from_formula("yb ~ x1", frame).fit()
    else:
        result = ws.Poisson.from_formula("ycnt ~ x1", frame).fit()

    assert "k_aux" not in result.e


# ------------------- tobit's censoring counts use Stata's own names
def test_tobit_censoring_counts_use_statats_names() -> None:
    """``tobit.sthlp:254-256``: ``e(N_unc)``, ``e(N_lc)``, ``e(N_rc)``.

    Capital ``N`` with the abbreviations ``unc``/``lc``/``rc``.  These were spelled
    ``n_uncensored``/``n_left``/``n_right`` -- descriptive, but not names any Stata
    consumer would look for, so the documented keys read as missing.
    """

    rng = np.random.default_rng(113)
    n = 300
    y = rng.normal(size=n)
    y = np.where(y > 1.0, 1.0, y)  # right-censored at 1
    frame = pd.DataFrame({"y": y, "x1": rng.normal(size=n)})

    result = ws.Tobit.from_formula("y ~ x1", frame, upper=1.0).fit()
    e = result.e

    assert "N_unc" in e and "N_lc" in e and "N_rc" in e
    assert e["N_unc"] + e["N_lc"] + e["N_rc"] == e["N"]
    assert e["N_rc"] == int((y >= 1.0).sum())


def test_tobit_censoring_counts_drop_the_old_spellings() -> None:
    """The descriptive spellings are GONE, not aliased.

    Keeping ``n_uncensored``/``n_left``/``n_right`` alongside ``N_unc``/``N_lc``/
    ``N_rc`` would mean ``e()`` reports keys Stata does not have -- the very thing
    the rename was meant to fix.  A caller who follows ``tobit.sthlp`` uses the
    documented names; a caller who used our invented ones gets a clear KeyError
    instead of a silently different vocabulary.
    """

    rng = np.random.default_rng(127)
    n = 200
    y = rng.normal(size=n)
    y = np.where(y > 1.0, 1.0, y)
    frame = pd.DataFrame({"y": y, "x1": rng.normal(size=n)})

    e = ws.Tobit.from_formula("y ~ x1", frame, upper=1.0).fit().e

    for stale in ("n_uncensored", "n_left", "n_right"):
        assert stale not in e, stale
    assert e["N_unc"] + e["N_lc"] + e["N_rc"] == e["N"]


def test_a_tobit_with_no_censoring_refuses() -> None:
    """With no limit at all, tobit is just OLS and the estimator says so.

    ``tobit.ado:157-159`` has a branch for "no censoring" that sets
    ``N_unc = e(N)`` and the other two to 0, but this package refuses the case
    outright and points at OLS instead -- a defensible choice, recorded here so
    the missing branch is not mistaken for an oversight.
    """

    rng = np.random.default_rng(131)
    n = 150
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )

    with pytest.raises(ws.ConsistencyError, match="censoring"):
        ws.Tobit.from_formula("y ~ x1", frame, lower=None, upper=None)


def test_a_one_sided_tobit_counts_only_the_censored_side() -> None:
    """With only ``upper`` specified, ``N_lc`` must be zero.

    ``tobit``'s DEFAULT is ``lower=0``, so a two-sided fit is what you get unless
    ``lower=None`` is passed explicitly -- my first version of this test forgot
    that and read the 107 left-censored observations as a bug.
    """

    rng = np.random.default_rng(137)
    n = 200
    y = rng.normal(size=n)
    y = np.where(y > 1.0, 1.0, y)
    frame = pd.DataFrame({"y": y, "x1": rng.normal(size=n)})

    e = ws.Tobit.from_formula("y ~ x1", frame, lower=None, upper=1.0).fit().e

    assert e["N_lc"] == 0
    assert e["N_rc"] == int((y >= 1.0).sum())
    assert e["N_unc"] + e["N_rc"] == e["N"]


# ------------------------------------------------------- truncreg / intreg
def test_truncreg_reports_observations_before_truncation() -> None:
    """``truncreg.sthlp:247``: ``e(N_bf)`` is the count BEFORE truncation.

    ``truncreg.ado:232`` sets it from ``N_b``.  ``e(N)`` is the surviving sample, so
    the two differ whenever anything was truncated -- which is always, or tobit's
    cousin would have nothing to do.
    """

    rng = np.random.default_rng(139)
    n = 300
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )

    e = ws.Truncreg.from_formula("y ~ x1", frame, lower=0).fit().e

    assert e["N_bf"] == n
    assert e["N"] < n  # some rows fell below the truncation point


def test_intreg_counts_each_censoring_kind() -> None:
    """``intreg.sthlp:242-245``: N_unc / N_lc / N_rc / N_int.

    The four kinds partition the sample, so they must sum to ``e(N)``.  Each is
    built from the interval bounds: ``-inf`` lower is left-censored, ``+inf`` upper
    right-censored, ``lower == upper`` observed exactly, and a finite non-degenerate
    pair is an interval observation.
    """

    rng = np.random.default_rng(149)
    n = 300
    values = rng.normal(size=n)
    lower = values.copy()
    upper = values.copy()
    lower[0:50] = -np.inf  # left-censored
    upper[50:100] = np.inf  # right-censored
    lower[100:150] = values[100:150] - 0.5  # interval
    upper[100:150] = values[100:150] + 0.5
    frame = pd.DataFrame(
        {"lo": lower, "hi": upper, "x1": rng.normal(size=n)}
    )

    e = ws.Intreg.from_formula("lo hi ~ x1", frame).fit().e

    assert e["N_unc"] == 150
    assert e["N_lc"] == 50
    assert e["N_rc"] == 50
    assert e["N_int"] == 50
    assert e["N_unc"] + e["N_lc"] + e["N_rc"] + e["N_int"] == e["N"]


def test_intreg_se_sigma_is_the_delta_method() -> None:
    """``intreg.ado:416-417``: ``se_sigma = sigma * [lnsigma]_se``.

    ``sigma = exp(lnsigma)``, so the delta method multiplies by the standard error
    of ``lnsigma`` -- NOT by sigma's own, which is what a naive implementation would
    take from the wrong diagonal entry.
    """

    rng = np.random.default_rng(151)
    n = 200
    values = rng.normal(size=n)
    frame = pd.DataFrame(
        {"lo": values, "hi": values, "x1": rng.normal(size=n)}
    )

    result = ws.Intreg.from_formula("lo hi ~ x1", frame).fit()
    e = result.e
    se_lnsigma = float(result.bse[list(result.param_names).index("/lnsigma")])

    assert "se_sigma" in e
    assert e["se_sigma"] == pytest.approx(e["sigma"] * se_lnsigma, rel=1e-10)


def test_truncreg_has_no_se_sigma() -> None:
    """``truncreg`` has no ``/lnsigma`` parameter, so the delta method does not apply.

    ``intreg``'s ``se_sigma`` is defined off that parameter; adding the key to
    ``truncreg`` would mean taking a standard error from a parameter it does not
    carry.
    """

    rng = np.random.default_rng(157)
    n = 200
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )

    assert "se_sigma" not in ws.Truncreg.from_formula(
        "y ~ x1", frame, lower=0
    ).fit().e


def test_tobit_names_the_limits_llopt_and_ulopt() -> None:
    """``tobit.ado:131-132``: ``ereturn local llopt`` / ``ulopt``.

    The censoring limits are stored under the OPTION names, not as
    ``lower``/``upper``.  Our former names were descriptive but meant ``e(llopt)`` --
    the name anyone following the ado would use -- read as missing.
    """

    rng = np.random.default_rng(167)
    n = 200
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )

    e = ws.Tobit.from_formula("y ~ x1", frame, lower=0.0, upper=2.0).fit().e

    assert e["llopt"] == 0.0
    assert e["ulopt"] == 2.0
    assert "lower" not in e
    assert "upper" not in e


def test_tobit_reports_both_limits_when_both_are_given() -> None:
    """A two-sided tobit must carry both bounds."""

    rng = np.random.default_rng(173)
    n = 250
    y = np.clip(rng.normal(size=n), -0.5, 0.5)
    frame = pd.DataFrame({"y": y, "x1": rng.normal(size=n)})

    e = ws.Tobit.from_formula("y ~ x1", frame, lower=-0.5, upper=0.5).fit().e

    assert e["llopt"] == -0.5
    assert e["ulopt"] == 0.5


def test_the_N_cds_block_is_zero_on_homogeneous_covariate_patterns() -> None:
    """``N_cds``/``N_cdf`` count observations in completely determined patterns.

    With a DISCRETE regressor every pattern contains both outcomes, so nothing is
    completely determined and both counts are 0 -- which is what the captured
    ``logit_sim.json`` / ``probit_sim.json`` fixtures report (0.0 each).  A
    hardcoded zero would also pass that, so the companion test below uses a
    continuous regressor where every observation is its own pattern.
    """

    rng = np.random.default_rng(179)
    n = 600
    x1 = rng.integers(0, 3, size=n).astype(float)
    probability = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * x1)))
    frame = pd.DataFrame(
        {"x1": x1, "yb": (rng.uniform(size=n) < probability).astype(int)}
    )

    e = ws.Logit.from_formula("yb ~ x1", frame).fit().e

    assert e["N_cds"] == 0.0
    assert e["N_cdf"] == 0.0


def test_the_N_cds_block_counts_every_singleton_pattern() -> None:
    """With a CONTINUOUS regressor each observation is its own pattern.

    Such a pattern is trivially "completely determined", so the two counts
    partition the sample.  This is the case that proves the value is computed
    rather than assumed to be zero.
    """

    rng = np.random.default_rng(181)
    n = 600
    x1 = rng.normal(size=n)
    frame = pd.DataFrame(
        {"x1": x1, "yb": (rng.uniform(size=n) < 0.5).astype(int)}
    )

    e = ws.Logit.from_formula("yb ~ x1", frame).fit().e

    assert e["N_cds"] + e["N_cdf"] == e["N"]
    assert e["N_cds"] > 0 and e["N_cdf"] > 0


def test_r2_and_ll_are_internally_consistent_for_panel_fits(frame) -> None:
    """``xtreg, fe`` mixes two scales, and the identities follow from that.

    ``e(r2)`` = ``e(r2_w)`` is the WITHIN R-squared while ``e(r2_o)`` is the OVERALL one.
    The sums of squares follow suit -- checked against a live ``xtreg y x1, fe`` on a
    60 x 5 frame:

        e(tss) = 604.64155                     <- RAW total
        e(rss) = 216.16809, e(mss) = 78.087606 <- WITHIN scale
        mss + rss = 294.2557                   <- equals the WITHIN tss, not e(tss)

    This test previously asserted ``mss + rss == tss`` and ``r2_o == mss/tss``. Neither
    holds in Stata, and both held in westata only because ``mss`` was being computed as
    ``tss - rss`` -- a number no Stata run produces. Pinning the real relations instead:
    """

    panel = frame.assign(
        firm=np.repeat(np.arange(60), 5), year=np.tile(np.arange(5), 60)
    )
    result = ws.PanelOLS.from_formula(
        "y ~ x1", panel, entity="firm", time="year"
    ).fit()
    e = result.e

    # The within-scaled sums decompose the WITHIN total, which is a separate key.
    # The within-scaled sums decompose the WITHIN total, which `xtreg, fe` does not store
    # (`display e(tss_within)` prints "." on munnell.dta). Their sum is therefore checked
    # against the raw total, which it must fall below.
    assert e["mss"] + e["rss"] < e["tss"]
    assert e["mss"] > 0.0 and e["rss"] > 0.0
    assert e["mss"] > 0.0 and e["rss"] > 0.0
    # The within R-squared is a different quantity from the overall one.
    assert e["r2"] == pytest.approx(e["r2_w"], rel=1e-12)
    assert not np.isclose(e["r2"], e["r2_o"], rtol=1e-3)
    assert 0.0 <= e["r2_o"] <= 1.0