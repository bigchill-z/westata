"""Every postestimation result must carry a ``test`` label naming Stata's own statistic.

Round 207 found ``EndogenousResult`` alone was missing the attribute, so
``getattr(result, "test", None)`` returned ``None`` for one of three sibling types. That
is the failure mode this file closes: a caller who branches on ``test`` (rather than on
the class name) must never fall off a cliff.

The labels are not free text -- they are the names Stata uses in ``r()`` and prints in its
tables, so they are pinned here:

| context | label |
|---|---|
| ``estat overid``, unadjusted VCE | ``Sargan`` |
| ``estat overid``, robust/HAC | ``Wooldridge score`` |
| ``estat endogenous``, unadjusted | ``Durbin-Wu-Hausman`` |
| ``estat endogenous``, robust/HAC | ``robwool`` (Stata's own abbreviation) |
| ``estat endogenous``, clustered | ``regF`` |
| ``estat weakrobust`` | ``Anderson-Rubin`` |
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 250, seed: int = 3400):
    rng = np.random.default_rng(seed)
    first = rng.normal(size=n)
    second = rng.normal(size=n)
    endogenous = 0.8 * first + rng.normal(size=n) * 0.4
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.5 * endogenous + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": outcome,
            "x": exog,
            "e": endogenous,
            "z1": first,
            "z2": second,
            "g": np.arange(n) % 25,
        }
    )


@pytest.fixture(scope="module")
def plain():
    return ws.IV2SLS.from_formula("y ~ x + [e ~ z1 + z2]", _frame()).fit()


@pytest.fixture(scope="module")
def robust():
    return ws.IV2SLS.from_formula("y ~ x + [e ~ z1 + z2]", _frame()).fit(
        cov_type="HC1"
    )


@pytest.fixture(scope="module")
def clustered():
    frame = _frame()
    return ws.IV2SLS.from_formula("y ~ x + [e ~ z1 + z2]", frame).fit(
        cov_type="cluster", cov_kwds={"clusters": frame["g"]}
    )


# ------------------------------------------------------------------ the labels
def test_overid_names_sargan_when_unadjusted(plain) -> None:
    assert plain.estat_overid().test == "Sargan"


def test_overid_names_the_wooldridge_score_under_a_robust_vce(robust) -> None:
    """``ivregress_estat.ado:175-261`` replaces Sargan with this test."""
    assert robust.estat_overid().test == "Wooldridge score"


def test_endogenous_names_durbin_wu_when_unadjusted(plain) -> None:
    assert plain.estat_endogenous().test == "Durbin-Wu-Hausman"


def test_endogenous_uses_stat_as_own_spelling_for_the_robust_score(robust) -> None:
    """``robwool`` is Stata's abbreviation, not an invention of this package.

    Renaming it to "robust score" would be friendlier but would break a caller matching
    on Stata's own label.
    """

    assert robust.estat_endogenous().test == "robwool"


def test_endogenous_names_regF_under_clustering(clustered) -> None:
    """``ivregress_estat.ado:1070-1076``: the augmented-regression F."""
    assert clustered.estat_endogenous().test == "regF"


def test_weakrobust_names_anderson_rubin(plain) -> None:
    assert plain.estat_weakrobust(ar=True).test == "Anderson-Rubin"


# ------------------------------------------------------------- the uniform contract
def test_every_result_has_a_non_empty_test_label(plain, robust, clustered) -> None:
    """The regression test for round 207's defect.

    Every branch of every IV postestimation command is sampled here, so a new branch that
    forgets the attribute fails immediately rather than waiting for a caller to trip on
    ``getattr(result, "test", None) is None``.
    """

    results = [
        plain.estat_overid(),
        robust.estat_overid(),
        plain.estat_endogenous(),
        robust.estat_endogenous(),
        clustered.estat_endogenous(),
        plain.estat_weakrobust(ar=True),
    ]

    for result in results:
        label = getattr(result, "test", None)
        assert label, type(result).__name__
        assert isinstance(label, str), type(result).__name__
        assert label.strip() == label, f"{type(result).__name__}: {label!r}"


def test_the_labels_are_distinct_where_the_tests_are(plain, robust) -> None:
    """Guard: a label shared by two different statistics would be worse than none.

    ``estat overid`` and ``estat endogenous`` answer different questions, and their
    labels must differ under both VCE regimes.
    """

    for fitted in (plain, robust):
        overid = fitted.estat_overid().test
        endogenous = fitted.estat_endogenous().test
        assert overid != endogenous


def test_the_vce_changes_the_label_within_one_command(plain, robust) -> None:
    """The same command reports a DIFFERENT test depending on the VCE.

    This is why a caller cannot hard-code ``estat overid`` == Sargan.
    """

    assert plain.estat_overid().test != robust.estat_overid().test
    assert plain.estat_endogenous().test != robust.estat_endogenous().test


def test_the_label_matches_the_type(plain, robust, clustered) -> None:
    """``test`` and the class name must not contradict each other."""

    mapping = {
        "EndogenousResult": "Durbin-Wu-Hausman",
        "RegFResult": "regF",
        "EndogScoreResult": "robwool",
    }

    for fitted in (plain, robust, clustered):
        result = fitted.estat_endogenous()
        expected = mapping.get(type(result).__name__)
        if expected is not None:
            assert result.test == expected, type(result).__name__