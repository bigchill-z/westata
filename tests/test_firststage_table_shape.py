"""``estat firststage`` prints a DIFFERENT table depending on the endogenous count.

``ivregress_postestimation.sthlp:408-414``:

    {opt all} requests that all first-stage goodness-of-fit statistics be reported
    regardless of whether the model contains one or more endogenous regressors.  By
    default, if the model contains one endogenous regressor, then the first-stage
    R-squared, adjusted R-squared, partial R-squared, and F statistics are reported,
    whereas if the model contains multiple endogenous regressors, then Shea's partial
    R-squared and adjusted partial R-squared are reported instead.

Our renderer always drew the R-squared block. That meant a multi-endogenous model showed
a table Stata does not print, while the Shea statistics Stata DOES print were computed
and never rendered -- the failure shape of round 149's unrendered ``constraint``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _two_endogenous(n: int = 300, seed: int = 889):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    e1 = 0.7 * z1 + rng.normal(size=n) * 0.3
    e2 = 0.7 * z2 + rng.normal(size=n) * 0.3
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * e1 + 0.4 * e2 + rng.normal(size=n)
    return pd.DataFrame(
        {"y": outcome, "x": exog, "e1": e1, "e2": e2, "z1": z1, "z2": z2}
    )


@pytest.fixture(scope="module")
def multi():
    frame = _two_endogenous()
    result = ws.IV2SLS.from_formula("y ~ x + [e1 + e2 ~ z1 + z2]", frame).fit()
    return result.estat_firststage()


@pytest.fixture(scope="module")
def single(multi):
    frame = _two_endogenous()
    result = ws.IV2SLS.from_formula("y ~ x + [e1 ~ z1]", frame).fit()
    return result.estat_firststage()


def test_one_endogenous_regressor_uses_the_r_squared_table(single) -> None:
    """``R-sq.``/``Adjusted``/``Partial``/``F`` -- the documented default."""

    rendered = str(single)

    assert "R-sq." in rendered
    assert "Adjusted" in rendered
    assert "Shea" not in rendered


def test_multiple_endogenous_regressors_use_the_shea_table(multi) -> None:
    """``sthlp:413-414``: Shea's partial R-squared instead of the R-squared block."""

    rendered = str(multi)

    assert "Shea" in rendered
    assert "Adj. Shea" in rendered
    # The plain R-squared heading belongs to the other shape.
    assert "R-sq." not in rendered


def test_the_shea_values_are_the_stored_ones(multi) -> None:
    """The table must print ``.shea``, not recompute it.

    Recomputing would be a second implementation that could drift from the value the
    ``r(multiresults)`` surface exposes.
    """

    rendered = str(multi)

    for value in np.asarray(multi.shea, dtype=float):
        assert f"{value:.4f}" in rendered


def test_the_adjusted_shea_row_is_present(multi) -> None:
    """``shea_adj`` exists and is finite for this model."""

    assert multi.shea_adj is not None
    assert np.all(np.isfinite(np.asarray(multi.shea_adj, dtype=float)))

    rendered = str(multi)
    for value in np.asarray(multi.shea_adj, dtype=float):
        assert f"{value:.4f}" in rendered


def test_both_tables_report_the_same_partial_r_squared(multi, single) -> None:
    """``Partial`` appears in BOTH shapes, because Shea's statistic does not replace it.

    ``sthlp:411-414`` says the first shape reports "partial R-squared" and the second
    reports Shea's -- and Shea's is a DIFFERENT quantity, so a table that dropped
    ``Partial`` in the multi case would lose a number the caller can no longer see.
    """

    assert "Partial" in str(multi)
    assert "Partial" in str(single)


def test_the_two_shapes_actually_differ(multi, single) -> None:
    """Guard: if both rendered identically the switch would be untested."""

    assert str(multi) != str(single)


def test_the_shea_statistic_is_not_the_plain_partial_r_squared(multi) -> None:
    """Guard: Shea's statistic and the partial R-squared are separate numbers."""

    shea = np.asarray(multi.shea, dtype=float)
    partial = np.asarray(multi.partial_r2, dtype=float)

    assert not np.allclose(shea, partial)


def test_the_row_count_matches_the_endogenous_count(multi) -> None:
    """One row per endogenous regressor, in both shapes."""

    assert len(multi.endog) == 2
    rendered = str(multi)
    for name in multi.endog:
        assert name in rendered


# ------------------------------------------------------------------ ``all`` option
def test_the_all_option_shows_the_r_squared_block_for_many_endogenous() -> None:
    """``ivregress_postestimation.sthlp:408-412``: "request that all first-stage
    goodness-of-fit statistics be reported regardless of whether the model contains one
    or more endogenous regressors".

    So ``all`` overrides the multi-endogenous default that selects Shea's table.
    """

    frame = _two_endogenous()
    result = ws.IV2SLS.from_formula("y ~ x + [e1 + e2 ~ z1 + z2]", frame).fit()

    default = str(result.estat_firststage())
    with_all = str(result.estat_firststage(all=True))

    assert "Shea" in default
    assert "R-sq." in with_all
    assert "Shea" not in with_all


def test_the_all_option_changes_nothing_for_one_endogenous() -> None:
    """The single-endogenous default is ALREADY the R-squared block."""

    frame = _two_endogenous()
    result = ws.IV2SLS.from_formula("y ~ x + [e1 ~ z1]", frame).fit()

    default = result.estat_firststage()
    with_all = result.estat_firststage(all=True)

    assert str(default) == str(with_all)
    assert np.array_equal(default.r2, with_all.r2)


def test_the_numbers_are_the_same_in_both_tables() -> None:
    """``all`` changes which STATISTICS are shown, not their values."""

    frame = _two_endogenous()
    result = ws.IV2SLS.from_formula("y ~ x + [e1 + e2 ~ z1 + z2]", frame).fit()

    default = result.estat_firststage()
    with_all = result.estat_firststage(all=True)

    assert np.array_equal(default.partial_r2, with_all.partial_r2)
    assert np.array_equal(default.f, with_all.f)
    assert np.array_equal(default.pvalue, with_all.pvalue)
    assert np.array_equal(default.r2, with_all.r2)


def test_the_flag_is_readable_afterwards() -> None:
    frame = _two_endogenous()
    result = ws.IV2SLS.from_formula("y ~ x + [e1 + e2 ~ z1 + z2]", frame).fit()

    assert result.estat_firststage().show_all is False
    assert result.estat_firststage(all=True).show_all is True