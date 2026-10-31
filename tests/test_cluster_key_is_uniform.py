"""The cluster-label key must be the same word on every estimator.

``replication/papers/adh2013/WESTATA_GAPS.md`` §2 recorded this as a P1 inconsistency: a
caller passing ``cov_kwds={"clusters": g}`` got a working VCE from ``IV2SLS`` but an error
from ``OLS``, and ``{"cluster": g}`` worked on ``FEOLS`` and nowhere else. The message made
it worse -- on ``OLS`` the caller saw

    cov_kwds['groups'] must be one-dimensional; got 0 dimensions

which reads like malformed data rather than a misspelt key.

The root cause was structural, not a single bug: six modules each did their own
``dict.get("groups")``, so every one of them could drift independently and fixing one left
the others behind. The fix is a single accessor,
``westata.vce.dispatch.cluster_labels_from_kwds``.

These tests pin the OUTCOME (all three spellings work everywhere) rather than the
implementation, so the accessor can be refactored as long as the behaviour holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

#: Every spelling that must be accepted. ``groups`` is the OLS documentation, ``clusters``
#: reads naturally from Stata's ``vce(cluster clustvar)``, ``cluster`` is what ``hdfe``
#: used before the unification.
KEY_SPELLINGS = ("groups", "clusters", "cluster")


def _frame(n: int = 200, n_groups: int = 10, seed: int = 9911) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    instrument = rng.normal(size=n)
    endogenous = 0.8 * instrument + rng.normal(size=n) * 0.4
    return pd.DataFrame(
        {
            "y": 1.0 + 0.5 * endogenous + rng.normal(size=n),
            "x1": covariate,
            "endog": endogenous,
            "z": instrument,
            "grp": np.arange(n) % n_groups,
            "firm": np.repeat(np.arange(n // (n // n_groups)), n // n_groups),
            "year": np.tile(
                np.arange(1, (n // n_groups) + 1), n // (n // n_groups)
            ),
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


#: name -> builder that fits with ``cov_kwds={key: labels}``.
BUILDERS = {
    "OLS": lambda frame, key: ws.OLS.from_formula("y ~ x1", frame).fit(
        cov_type="cluster", cov_kwds={key: frame["grp"].to_numpy()}
    ),
    "IV2SLS": lambda frame, key: ws.IV2SLS.from_formula(
        "y ~ x1 + [endog ~ z]", frame
    ).fit(cov_type="cluster", cov_kwds={key: frame["grp"].to_numpy()}),
    "FEOLS": lambda frame, key: ws.FEOLS.from_formula("y ~ x1 | firm", frame).fit(
        cov_type="cluster", cov_kwds={key: frame["grp"].to_numpy()}
    ),
    "PanelOLS": lambda frame, key: ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit(cov_type="cluster", cov_kwds={key: frame["grp"].to_numpy()}),
}


# ------------------------------------------------- every key, every estimator
@pytest.mark.parametrize("key", KEY_SPELLINGS)
@pytest.mark.parametrize("estimator", sorted(BUILDERS))
def test_the_key_is_accepted(frame, estimator: str, key: str) -> None:
    """THE regression test: 4 estimators x 3 spellings, all of which must work."""

    fitted = BUILDERS[estimator](frame, key)

    assert np.all(np.isfinite(np.asarray(fitted.bse, dtype=float)))


# ------------------------------------------- the three spellings agree exactly
@pytest.mark.parametrize("estimator", sorted(BUILDERS))
def test_the_spellings_give_identical_numbers(frame, estimator: str) -> None:
    """An accepted alias that quietly changed the result would be worse than an error.

    Accepting three names is only safe if they are synonyms. Comparing the covariance
    matrices bit-for-bit proves they are, rather than trusting that the lookup returns the
    same array (it could, for instance, be re-coded differently per branch).
    """

    baseline = BUILDERS[estimator](frame, "groups")
    for key in KEY_SPELLINGS[1:]:
        other = BUILDERS[estimator](frame, key)
        np.testing.assert_allclose(
            np.asarray(baseline.cov_params(), dtype=float),
            np.asarray(other.cov_params(), dtype=float),
            rtol=0.0,
            atol=0.0,
            err_msg=f"{estimator}: '{key}' disagrees with 'groups'",
        )


# --------------------------------------------------------- the accessor itself
def test_the_accessor_reads_every_spelling() -> None:
    from westata.vce.dispatch import cluster_labels_from_kwds

    labels = np.arange(5)
    for key in KEY_SPELLINGS:
        assert cluster_labels_from_kwds({key: labels}) is labels


def test_the_accessor_returns_none_when_absent() -> None:
    from westata.vce.dispatch import cluster_labels_from_kwds

    assert cluster_labels_from_kwds({}) is None
    assert cluster_labels_from_kwds(None) is None
    assert cluster_labels_from_kwds({"unrelated": np.arange(3)}) is None


def test_the_accessor_prefers_groups() -> None:
    """With several supplied, the documented name wins -- deterministically, not by dict order."""

    from westata.vce.dispatch import cluster_labels_from_kwds

    first = np.arange(4)
    second = np.arange(4) + 100
    chosen = cluster_labels_from_kwds({"groups": first, "clusters": second})

    assert chosen is first


# ------------------------------------------------------------ the refusal path
def test_a_missing_key_still_raises() -> None:
    """Accepting aliases must not turn a genuine omission into a silent success."""

    frame = _frame()
    with pytest.raises((ValueError, KeyError)):
        ws.OLS.from_formula("y ~ x1", frame).fit(cov_type="cluster", cov_kwds={})


def test_a_genuinely_wrong_key_is_still_rejected() -> None:
    """A typo that is not an alias should fail, not be treated as "no clusters"."""

    frame = _frame()
    with pytest.raises((ValueError, KeyError, TypeError)):
        ws.OLS.from_formula("y ~ x1", frame).fit(
            cov_type="cluster", cov_kwds={"group": frame["grp"].to_numpy()}
        )