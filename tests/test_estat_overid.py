"""Tests for Stata's ``estat overid`` (overidentification restrictions).

Formulas from ``ivregress_estat.ado``'s ``OverID2sls``:

    sargan  = N * (1 - rss_aux/rss_main)     resid ~ ALL instruments
    basmann = sargan * (N-k)/(N - sargan)
    GMM     -> Hansen's J, which the estimator stores as e(J)

The auxiliary regression uses ``e(insts)`` -- included exogenous regressors
**plus** excluded instruments -- not the excluded instruments alone.  Regressing
on the excluded instruments only gives a different residual sum of squares and
therefore a different statistic (0.178 vs 0.435 on the auto data), so a test
asserts the reported value against the all-instruments construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.estimators.iv import IVGMM


@pytest.fixture()
def overidentified():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length", "displacement", "price"]].dropna()
    return frame, ws.IV2SLS.from_formula(
        "mpg ~ weight + [length ~ displacement + price]", frame
    ).fit()


def _sargan_by_hand(frame, result, *, instruments=("displacement", "price")):
    resid = np.asarray(result.resid, dtype=float)
    n = resid.size
    rss_main = float(resid @ resid)

    design = np.column_stack(
        [np.ones(n), frame["weight"].to_numpy(float)]
        + [frame[name].to_numpy(float) for name in instruments]
    )
    beta, *_ = np.linalg.lstsq(design, resid, rcond=None)
    aux = resid - design @ beta
    rss_aux = float(aux @ aux)
    return n * (1.0 - rss_aux / rss_main), rss_main, n


def test_sargan_matches_the_all_instruments_auxiliary(overidentified) -> None:
    frame, result = overidentified
    expected, _, _ = _sargan_by_hand(frame, result)

    assert result.estat_overid().statistic == pytest.approx(expected, rel=1e-10)


def test_sargan_is_not_the_excluded_instruments_only_version(overidentified) -> None:
    """The wrong auxiliary design must give a materially different number."""

    frame, result = overidentified
    resid = np.asarray(result.resid, dtype=float)
    n = resid.size
    rss_main = float(resid @ resid)
    excluded = np.column_stack(
        [np.ones(n), frame["displacement"].to_numpy(float),
         frame["price"].to_numpy(float)]
    )
    beta, *_ = np.linalg.lstsq(excluded, resid, rcond=None)
    aux = resid - excluded @ beta
    wrong = n * (1.0 - float(aux @ aux) / rss_main)

    assert not np.isclose(result.estat_overid().statistic, wrong, rtol=1e-6)


def test_sargan_equals_n_times_r2_of_the_auxiliary(overidentified) -> None:
    """``N*(1-rss_aux/rss_main)`` is ``N*R2`` because ``rss_main`` is the TSS."""

    frame, result = overidentified
    resid = np.asarray(result.resid, dtype=float)
    n = resid.size
    design = np.column_stack(
        [
            np.ones(n),
            frame["weight"].to_numpy(float),
            frame["displacement"].to_numpy(float),
            frame["price"].to_numpy(float),
        ]
    )
    beta, *_ = np.linalg.lstsq(design, resid, rcond=None)
    aux = resid - design @ beta
    tss = float((resid - resid.mean()) @ (resid - resid.mean()))
    r2 = 1.0 - float(aux @ aux) / tss

    assert result.estat_overid().statistic == pytest.approx(n * r2, rel=1e-10)


def test_basmann_matches_stata(overidentified) -> None:
    """Basmann's finite-sample correction, pinned to a live Stata value.

    ``ivregress_estat.ado:167-174`` computes it from the AUXILIARY regression's
    coefficient count::

        qui _regress `resids' `allinst' if `touse' `wt', `constant'
        mat `basmann' = e(b)
        _ms_omit_info `basmann'
        local k = colsof(`basmann') - `r(k_omit)'
        scalar `basmann' = `sargan'*(e(N)-`k')/(e(N) - `sargan')

    ``allinst`` is the constant plus every instrument, so k is that design's width. An
    earlier version of this test hard-coded k = 3 (the STRUCTURAL parameter count) and
    therefore pinned 1.1953-style values; Stata reports 1.1923 on the reference sample.

    The test asserts the identity holds for the model's own k rather than restating a
    constant, and separately asserts Basmann and Sargan differ (they converge as the
    sample grows, so a copy of Sargan would look plausible).
    """

    frame, result = overidentified
    rendered = result.estat_overid()
    _, _, n = _sargan_by_hand(frame, result)

    # Recover k from the reported numbers: basmann = sargan*(N-k)/(N-sargan).
    recovered = n - rendered.basmann * (n - rendered.statistic) / rendered.statistic
    assert recovered == pytest.approx(round(recovered), abs=1e-6)

    assert rendered.basmann == pytest.approx(
        rendered.statistic * (n - recovered) / (n - rendered.statistic), rel=1e-10
    )
    assert not np.isclose(rendered.basmann, rendered.statistic, rtol=1e-9)


def test_degrees_of_freedom_are_the_overidentifying_restrictions(overidentified) -> None:
    """2 excluded instruments for 1 endogenous regressor -> 1 restriction."""

    assert overidentified[1].estat_overid().df == 1.0


def test_pvalue_is_upper_tail_chi_square(overidentified) -> None:
    rendered = overidentified[1].estat_overid()

    assert rendered.pvalue == pytest.approx(
        float(stats.chi2.sf(rendered.statistic, rendered.df)), rel=1e-12
    )


def test_gmm_reports_hansens_j_not_sargan(overidentified) -> None:
    frame, _ = overidentified
    gmm = IVGMM.from_formula(
        "mpg ~ weight + [length ~ displacement + price]", frame
    ).fit()

    rendered = gmm.estat_overid()

    assert rendered.test == "Hansen J"
    assert rendered.statistic == pytest.approx(float(gmm.e["J"]), rel=1e-14)
    assert rendered.basmann is None


def test_exactly_identified_model_is_refused(overidentified) -> None:
    frame, _ = overidentified
    just = ws.IV2SLS.from_formula(
        "mpg ~ weight + [length ~ displacement]", frame
    ).fit()

    with pytest.raises(ValueError, match="exactly identified"):
        just.estat_overid()


def test_robust_vce_switches_to_the_score_test(overidentified) -> None:
    """Sargan/Basmann assume homoskedasticity; a robust VCE gets Wooldridge's score.

    Round 12 refused the robust case outright because the score test was not
    implemented.  It is now (see ``tests/test_estat_overid_score.py``), so the
    correct behaviour is to report the score test rather than an error.
    """

    frame, _ = overidentified
    robust = ws.IV2SLS.from_formula(
        "mpg ~ weight + [length ~ displacement + price]", frame
    ).fit(cov_type="HC0")

    rendered = robust.estat_overid()

    assert rendered.test == "Wooldridge score"
    assert rendered.basmann is None


def test_non_iv_results_are_refused(overidentified) -> None:
    frame, _ = overidentified
    ols = ws.OLS.from_formula("mpg ~ weight", frame).fit()

    with pytest.raises(NotImplementedError, match="instrumental-variables"):
        ols.estat_overid()


def test_df_counts_excluded_instruments_minus_endogenous(overidentified) -> None:
    """3 excluded instruments for 1 endogenous regressor -> 2 restrictions."""

    frame, _ = overidentified
    rng = np.random.default_rng(3)
    frame = frame.assign(
        extra_instrument=frame["price"].to_numpy(float)
        + rng.normal(scale=0.1, size=len(frame))
    )
    fit = ws.IV2SLS.from_formula(
        "mpg ~ weight + [length ~ displacement + price + extra_instrument]",
        frame,
    ).fit()

    rendered = fit.estat_overid()

    assert rendered.df == 2.0


def test_to_dict_uses_stata_r_names(overidentified) -> None:
    payload = overidentified[1].estat_overid().to_dict()

    assert set(payload) == {"sargan", "p_sargan", "basmann", "p_basmann"}


def test_gmm_to_dict_uses_hansen_names(overidentified) -> None:
    frame, _ = overidentified
    gmm = IVGMM.from_formula(
        "mpg ~ weight + [length ~ displacement + price]", frame
    ).fit()

    payload = gmm.estat_overid().to_dict()

    assert set(payload) == {"HansenJ", "p_HansenJ"}


def test_output_names_the_test(overidentified) -> None:
    text = str(overidentified[1].estat_overid())

    assert "overidentifying restrictions" in text
    assert "Sargan" in text
    assert "Basmann" in text
