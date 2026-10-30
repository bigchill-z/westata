"""``e(chi2)`` belongs to ``xtreg, re`` alone -- not to ``be`` or ``fd``.

``xtreg.sthlp`` documents each estimator's stored results separately:

| block | lines | `e(chi2)` |
|---|---|---|
| `xtreg, re` | 891-951 | present |
| `xtreg, be` | 1014-1067 | **absent** |
| `xtreg, fe` | 1068-1144 | **absent** |

``BetweenOLS`` and ``FirstDifferenceOLS`` both inherit from ``RandomEffects``, which sets
``_stores_wald_chi2 = True``. The flag came along with the inheritance, so the between and
first-difference estimators reported a Wald chi-square that Stata does not store -- and the
value was just ``F * df_m`` under another name, which reads as a legitimate statistic.

A reader who saw ``e(chi2)`` on a ``be`` fit would reasonably conclude the model test was
a Wald test. It is an F test; the chi-square was a copy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

PANEL_ESTIMATORS = ("RandomEffects", "PanelOLS", "BetweenOLS", "FirstDifferenceOLS")


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(980)
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


def _fit(frame: pd.DataFrame, name: str):
    return getattr(ws, name).from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()


@pytest.mark.parametrize("name", PANEL_ESTIMATORS)
def test_every_estimator_reports_an_F(frame, name) -> None:
    """All four report ``e(F)`` in the help file."""

    assert _fit(frame, name).e.get("F") is not None


def test_random_effects_reports_the_wald_chi2(frame) -> None:
    """``xtreg.sthlp:891-951`` has ``e(chi2)`` AND ``e(chi2type)``."""

    e = _fit(frame, "RandomEffects").e

    assert e.get("chi2") is not None
    assert e.get("chi2type") == "Wald"


@pytest.mark.parametrize("name", ["BetweenOLS", "FirstDifferenceOLS", "PanelOLS"])
def test_the_others_do_not_report_a_chi2(frame, name) -> None:
    """``be``/``fd``/``fe`` blocks list only ``e(F)``.

    The first two inherited the flag from ``RandomEffects``; the fix overrides it.
    """

    e = _fit(frame, name).e

    assert e.get("chi2") is None, name
    assert e.get("chi2type") is None, name


def test_the_inheritance_is_the_reason(frame) -> None:
    """Pin the mechanism, so a future refactor does not quietly reintroduce it."""

    assert issubclass(ws.BetweenOLS, ws.RandomEffects)
    assert issubclass(ws.FirstDifferenceOLS, ws.RandomEffects)

    assert ws.RandomEffects._stores_wald_chi2 is True
    assert ws.BetweenOLS._stores_wald_chi2 is False
    assert ws.FirstDifferenceOLS._stores_wald_chi2 is False


def test_the_reporting_flag_matches_the_stored_key(frame) -> None:
    """The flag and the ``e()`` contents must agree, for every estimator."""

    for name in PANEL_ESTIMATORS:
        result = _fit(frame, name)
        flag = getattr(result, "panel_entity", None) is not None
        assert flag, name  # this fixture is a panel for all four
        has_chi2 = result.e.get("chi2") is not None
        expected = name == "RandomEffects"
        assert has_chi2 is expected, name


def test_between_and_fe_agree_on_the_model_test(frame) -> None:
    """Both use an F test, so neither should carry a chi-square variant.

    Guard for the direction of the fix: it removed the key rather than changing
    ``RandomEffects``, whose chi-square is genuinely documented.
    """

    between = _fit(frame, "BetweenOLS").e
    fixed = _fit(frame, "PanelOLS").e

    assert between.get("chi2") is None
    assert fixed.get("chi2") is None
    # And the RE fit keeps its own.
    assert _fit(frame, "RandomEffects").e.get("chi2") is not None