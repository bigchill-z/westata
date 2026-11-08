"""What each panel estimator stores in ``e()``, checked against the per-estimator blocks.

``xtreg.sthlp`` documents stored results SEPARATELY for each estimator, and the blocks do
not describe the same set:

| block | lines | mss | rss | tss | sigma family | theta | chi2 |
|---|---|---|---|---|---|---|---|
| `re` | 891-951 | no | yes | no | yes | yes | yes |
| `be` | 1014-1067 | yes | yes | no | no | no | no |
| `fe` | 1068-1144 | yes | yes | yes | yes | no | no |

Two traps live in that table, and both were real:

* ``_stores_wald_chi2`` was inherited by ``be``/``fd`` from ``RandomEffects`` (round 173);
* ``_stores_mss`` does NOT mean "stores mss" -- it means "stores mss WITHOUT tss", which
  is why ``fe`` sets ``_stores_totals`` instead and still emits ``mss``.

So these tests assert the e() CONTENTS rather than the flags, and then pin the flag
semantics separately so neither can drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(990)
    entities, periods = 40, 6
    entity = np.repeat(np.arange(entities), periods)
    n = entity.size
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "firm": entity,
            "year": np.tile(np.arange(periods), entities),
        }
    )


def _e(frame: pd.DataFrame, name: str) -> dict:
    return getattr(ws, name).from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit().e


def _present(macros: dict, key: str) -> bool:
    return macros.get(key) is not None


# ------------------------------------------------------------ the sigma family
@pytest.mark.parametrize("name", ["RandomEffects", "PanelOLS"])
def test_fe_and_re_store_the_sigma_family(frame, name) -> None:
    macros = _e(frame, name)

    for key in ("sigma_u", "sigma_e", "sigma", "rho", "corr"):
        assert _present(macros, key), (name, key)


@pytest.mark.parametrize("name", ["BetweenOLS", "FirstDifferenceOLS"])
def test_be_and_fd_store_no_sigma_family(frame, name) -> None:
    """``be`` runs OLS on entity means and ``fd`` differences the effect away.

    Neither has a panel/epsilon decomposition, so neither has ``sigma_u``/``rho``/etc.
    """

    macros = _e(frame, name)

    for key in ("sigma_u", "sigma_e", "sigma", "rho", "corr"):
        assert not _present(macros, key), (name, key)


# ------------------------------------------------------------------- theta
def test_only_re_stores_theta(frame) -> None:
    """``xtreg_re.ado:265-266``: theta is the quasi-demeaning factor, RE only."""

    assert _present(_e(frame, "RandomEffects"), "theta")
    for name in ("PanelOLS", "BetweenOLS", "FirstDifferenceOLS"):
        assert not _present(_e(frame, name), "theta"), name


# ---------------------------------------------------------------- sums of squares
def test_fe_stores_mss_rss_and_tss(frame) -> None:
    """``xtreg.sthlp:1068-1144`` lists all three."""

    macros = _e(frame, "PanelOLS")

    for key in ("mss", "rss", "tss"):
        assert _present(macros, key), key


def test_be_stores_mss_and_rss_but_not_tss(frame) -> None:
    """``xtreg.sthlp:1014-1067`` has ``mss``/``rss`` and no ``tss``."""

    macros = _e(frame, "BetweenOLS")

    assert _present(macros, "mss")
    assert _present(macros, "rss")
    assert not _present(macros, "tss")


def test_re_stores_no_sums_of_squares_at_all(frame) -> None:
    """`xtreg, re` stores NONE of rss / mss / tss -- measured, not read off the help file.

    An earlier version of this test asserted `rss` was present, on the strength of
    `xtreg.sthlp:891-951` listing it. Probing the live command with `display e(rss)` prints ".",
    so the help text and the implementation disagree and the implementation wins. The same applies
    to e(ll): `estat ic` after `xtreg, re` raises r(321) because there is no likelihood stored.
    """

    macros = _e(frame, "RandomEffects")

    for key in ("rss", "mss", "tss", "ll"):
        assert not _present(macros, key), key
    assert not _present(macros, "mss")
    assert not _present(macros, "tss")


# --------------------------------------------------------------- the flag semantics
def test_stores_mss_means_mss_WITHOUT_tss(frame) -> None:
    """Pin the semantics that the name hides.

    ``_stores_mss`` is read as ``if self._stores_totals or self._stores_mss`` at the
    ``e()`` construction site, so it is the "mss but no tss" case -- ``be``.  ``fe``
    reaches ``mss`` through ``_stores_totals``.  Reading the flag as "stores mss" makes
    ``fe`` look wrong when it is right.
    """

    assert ws.BetweenOLS._stores_mss is True
    assert ws.BetweenOLS._stores_totals is False
    assert ws.PanelOLS._stores_mss is False
    assert ws.PanelOLS._stores_totals is True


def test_both_routes_produce_mss(frame) -> None:
    """Whatever the flag means, both estimators DO store ``mss``."""

    assert _present(_e(frame, "BetweenOLS"), "mss")
    assert _present(_e(frame, "PanelOLS"), "mss")


def test_the_flag_matrix_is_pinned() -> None:
    """A single place where every class-level switch is recorded.

    Round 173's bug was a flag inherited by two subclasses; pinning the matrix makes the
    next such inheritance visible in the diff.
    """

    expected = {
        "PanelOLS": {
            "_stores_totals": True,
            "_stores_absorbed": True,
            "_estimates_on_entity_means": False,
            "_stores_mss": False,
            "_stores_theta": False,
            "_stores_sigma": True,
            "_stores_wald_chi2": False,
        },
        "RandomEffects": {
            "_stores_totals": False,
            "_stores_absorbed": False,
            "_estimates_on_entity_means": False,
            "_stores_mss": False,
            "_stores_theta": True,
            "_stores_sigma": True,
            "_stores_wald_chi2": True,
        },
        "BetweenOLS": {
            "_stores_totals": False,
            "_stores_absorbed": False,
            "_estimates_on_entity_means": True,
            "_stores_mss": True,
            "_stores_theta": False,
            "_stores_sigma": False,
            "_stores_wald_chi2": False,
        },
        "FirstDifferenceOLS": {
            "_stores_totals": False,
            "_stores_absorbed": False,
            "_estimates_on_entity_means": False,
            "_stores_mss": False,
            "_stores_theta": False,
            "_stores_sigma": False,
            "_stores_wald_chi2": False,
        },
    }

    for name, flags in expected.items():
        cls = getattr(ws, name)
        for flag, value in flags.items():
            assert getattr(cls, flag) is value, (name, flag)