"""Multiway clustering must announce WHERE the capability comes from.

The user's point, in their words: stock ``ivregress`` cannot do two-way clustering either --
only the user-written ``ivreg2`` can. So when westata produces that number, it must say so
at runtime, or a caller will reasonably assume "Stata can do this", try to reproduce it in
vanilla Stata, and fail with no explanation.

The message therefore has to be COMMAND-SPECIFIC, because the underlying fact differs:

* ``regress`` accepts ``vce(cluster a b)`` (``regress.sthlp:196``) -- OLS reproduces stock
  behaviour, and saying "this is not Stata" there would be wrong.
* stock ``ivregress`` refuses it ("multiway cluster VCE not allowed", measured on StataNow
  19.5) -- IV reproduces ``ivreg2``, and NOT saying so is the misleading direction.

Reporting the same note for both would be wrong either way, so both directions are pinned.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 400, seed: int = 12301) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    instrument = rng.normal(size=n)
    endogenous = 0.8 * instrument + rng.normal(size=n) * 0.5
    return pd.DataFrame(
        {
            "y": 1.0 + 0.6 * endogenous + rng.normal(size=n),
            "x": rng.normal(size=n),
            "endog": endogenous,
            "z": instrument,
            "grp_a": np.repeat(np.arange(20), n // 20),
            "grp_b": np.tile(np.arange(10), n // 10),
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


def _messages_for(call) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return [str(item.message) for item in caught]


# ------------------------------------------------------------------- IV note
def test_iv_multiway_warns_that_it_is_not_stock_stata(frame) -> None:
    """THE point: an IV fit must not let a reader assume stock Stata can do this."""

    messages = _messages_for(
        lambda: ws.IV2SLS.from_formula("y ~ 1 + [endog ~ z]", frame).fit(
            cov_type="cluster", cov_kwds={"clusters": frame[["grp_a", "grp_b"]]}
        )
    )
    joined = " ".join(messages)

    assert joined, "no warning was emitted for multiway IV clustering"


def test_the_iv_note_names_ivregress_as_the_refusing_command(frame) -> None:
    messages = _messages_for(
        lambda: ws.IV2SLS.from_formula("y ~ 1 + [endog ~ z]", frame).fit(
            cov_type="cluster", cov_kwds={"clusters": frame[["grp_a", "grp_b"]]}
        )
    )
    joined = " ".join(messages).lower()

    assert "ivregress" in joined
    assert "refus" in joined or "not allowed" in joined or "cannot" in joined


def test_the_iv_note_names_ivreg2_as_the_source(frame) -> None:
    """Provenance means naming the command that actually implements it."""

    messages = _messages_for(
        lambda: ws.IV2SLS.from_formula("y ~ 1 + [endog ~ z]", frame).fit(
            cov_type="cluster", cov_kwds={"clusters": frame[["grp_a", "grp_b"]]}
        )
    )
    joined = " ".join(messages).lower()

    assert "ivreg2" in joined
    # It must be identified as user-written, not bundled with Stata.
    assert "user-written" in joined or "user written" in joined


# ------------------------------------------------------------------ OLS note
def test_ols_multiway_says_it_matches_stock_stata(frame) -> None:
    """The opposite direction: for OLS this IS stock behaviour, and saying otherwise
    would make a built-in feature sound exotic."""

    messages = _messages_for(
        lambda: ws.OLS.from_formula("y ~ x", frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame[["grp_a", "grp_b"]]}
        )
    )
    joined = " ".join(messages).lower()

    assert "regress" in joined


def test_ols_does_not_claim_to_be_non_stock(frame) -> None:
    """Guard against collapsing both notes into one pessimistic message."""

    messages = _messages_for(
        lambda: ws.OLS.from_formula("y ~ x", frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame[["grp_a", "grp_b"]]}
        )
    )
    joined = " ".join(messages).lower()

    assert "not a stock stata feature" not in joined


# ------------------------------------------------- the notes must differ
def test_the_two_notes_are_not_identical(frame) -> None:
    """A single shared note cannot be right for both commands."""

    iv = " ".join(
        _messages_for(
            lambda: ws.IV2SLS.from_formula("y ~ 1 + [endog ~ z]", frame).fit(
                cov_type="cluster", cov_kwds={"clusters": frame[["grp_a", "grp_b"]]}
            )
        )
    )
    ols = " ".join(
        _messages_for(
            lambda: ws.OLS.from_formula("y ~ x", frame).fit(
                cov_type="cluster", cov_kwds={"groups": frame[["grp_a", "grp_b"]]}
            )
        )
    )

    assert iv != ols


# ------------------------------------------------------- one-way stays quiet
def test_one_way_clustering_does_not_warn(frame) -> None:
    """The provenance question only arises for multiway; ordinary clustering is
    documented stock behaviour and should not gain noise."""

    messages = _messages_for(
        lambda: ws.OLS.from_formula("y ~ x", frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame["grp_a"]}
        )
    )
    joined = " ".join(messages).lower()

    assert "multiway" not in joined


def test_no_clustering_does_not_warn(frame) -> None:
    messages = _messages_for(
        lambda: ws.OLS.from_formula("y ~ x", frame).fit()
    )
    joined = " ".join(messages).lower()

    assert "multiway" not in joined


# ------------------------------------------------ the numbers are unaffected
def test_the_warning_does_not_change_the_result(frame) -> None:
    """A note is not a behaviour change; the SE must be identical either way."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quiet = ws.IV2SLS.from_formula("y ~ 1 + [endog ~ z]", frame).fit(
            cov_type="cluster", cov_kwds={"clusters": frame[["grp_a", "grp_b"]]}
        )
    loud = ws.IV2SLS.from_formula("y ~ 1 + [endog ~ z]", frame).fit(
        cov_type="cluster", cov_kwds={"clusters": frame[["grp_a", "grp_b"]]}
    )

    np.testing.assert_allclose(
        np.asarray(quiet.bse, dtype=float), np.asarray(loud.bse, dtype=float),
        rtol=0.0, atol=0.0,
    )