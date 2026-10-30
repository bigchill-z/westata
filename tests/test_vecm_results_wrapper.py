"""``VECM.fit`` must return a Stata-flavoured result, like ``VAR.fit`` does.

``VAR.fit`` returns ``VARResults``, which carries ``e()``, Stata's ``ml_factor`` and the
per-equation accessors. ``VECM.fit`` returned statsmodels' object DIRECTLY, so a ``vec``
fit had no ``e()`` at all -- and any code written generically over ``e()`` worked for
``var`` and raised ``AttributeError`` for ``vec``.

This round adds ``VECMResults``. The keys mirror ``var``'s macro family plus the
cointegration rank, which is what distinguishes the two models:

* ``e(depvar)``/``e(eqnames)``/``e(endog)`` are whitespace-separated LISTS;
* ``e(k_ce)`` is the cointegration rank;
* ``e(k_ar_diff)`` is the number of lagged DIFFERENCES, which is ``lags - 1``.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws


def _cointegrated(n: int = 400, seed: int = 1200):
    """Two series linked by a genuine cointegrating relation."""

    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(size=n))
    other = common + rng.normal(size=n) * 0.5
    return np.column_stack([other, common])


@pytest.fixture(scope="module")
def vecm_fit():
    return ws.VECM(
        _cointegrated(), lags=2, coint_rank=1, names=["y", "x"]
    ).fit()


def test_the_result_is_the_packages_own_wrapper(vecm_fit) -> None:
    """Not a raw statsmodels object: it has to own the Stata surface."""

    assert type(vecm_fit).__name__ == "VECMResults"
    assert hasattr(vecm_fit, "e")


def test_depvar_lists_the_variables(vecm_fit) -> None:
    assert vecm_fit.e["depvar"] == "y x"


def test_eqnames_and_endog_match_depvar(vecm_fit) -> None:
    """One equation per variable, and every variable is endogenous."""

    assert vecm_fit.e["eqnames"] == "y x"
    assert vecm_fit.e["endog"] == "y x"
    assert vecm_fit.e["exog"] == ""


def test_lags_lists_each_lag(vecm_fit) -> None:
    """``lags(2)`` means two VAR lags, so the list is "1 2"."""

    assert vecm_fit.e["lags"] == "1 2"


def test_k_ce_is_the_cointegration_rank(vecm_fit) -> None:
    """The option that makes it a VECM rather than a differenced VAR."""

    assert float(vecm_fit.e["k_ce"]) == 1.0


def test_k_ar_diff_is_one_fewer_than_the_lag_count(vecm_fit) -> None:
    """Stata's ``lags(#)`` counts VAR lags, so the VECM carries ``# - 1`` differences."""

    assert float(vecm_fit.e["k_ar_diff"]) == 1.0
    assert float(vecm_fit.e["k_ar_diff"]) == float(vecm_fit.e["mlag"]) - 1.0


def test_a_three_lag_model_carries_two_differences() -> None:
    """Guard: the relation must track the model, not a constant."""

    fitted = ws.VECM(
        _cointegrated(seed=1202), lags=3, coint_rank=1, names=["y", "x"]
    ).fit()

    assert float(fitted.e["k_ar_diff"]) == 2.0
    assert fitted.e["lags"] == "1 2 3"


def test_the_degrees_of_freedom_partition_the_sample(vecm_fit) -> None:
    """``df_m + df_r == N``, as for ``var``."""

    e = vecm_fit.e

    assert float(e["df_m"]) + float(e["df_r"]) == pytest.approx(
        float(e["N"]), rel=1e-12
    )


def test_the_rank_ceiling_holds() -> None:
    """``k_ce`` cannot exceed the number of variables; the fit must still work at 2."""

    fitted = ws.VECM(
        _cointegrated(seed=1204), lags=1, coint_rank=2, names=["y", "x"]
    ).fit()

    assert float(fitted.e["k_ce"]) == 2.0
    assert float(fitted.e["k_eq"]) == 2.0


def test_the_native_accessors_are_still_reachable(vecm_fit) -> None:
    """The wrapper must not lose what statsmodels exposed.

    ``alpha``/``beta``/``gamma`` are the VECM's own parameter blocks; wrapping the
    result is only acceptable if they survive.
    """

    assert vecm_fit.alpha.shape[0] == 2
    assert vecm_fit.beta.shape[0] == 2
    assert np.isfinite(vecm_fit.gamma).all()
    assert np.isfinite(vecm_fit.llf())


def test_the_raw_object_is_kept(vecm_fit) -> None:
    """Escape hatch for anything the wrapper does not forward."""

    assert vecm_fit.raw is not None
    assert hasattr(vecm_fit.raw, "alpha")


def test_rc_and_converged_are_present(vecm_fit) -> None:
    """``var.sthlp``'s family includes these; ``vec`` shares it."""

    assert float(vecm_fit.e["rc"]) == 0.0
    assert float(vecm_fit.e["converged"]) == 1.0


# ------------------------------------------------------------------ rendering
def test_the_summary_shows_the_cointegrating_table(vecm_fit) -> None:
    """``vec.sthlp:61``: the cointegrating equations are reported by default."""

    rendered = str(vecm_fit)

    assert "Cointegrating equations" in rendered
    for name in vecm_fit.names:
        assert name in rendered


def test_the_summary_shows_the_adjustment_table(vecm_fit) -> None:
    """``vec.sthlp:65``: ``alpha`` appears, in its own block."""

    rendered = str(vecm_fit)

    assert "Adjustment parameters" in rendered


def test_the_dataclass_repr_is_gone(vecm_fit) -> None:
    """The old ``__str__`` printed the wrapper's FIELDS, not a result.

    Seeing ``raw=<statsmodels object>`` in place of a table is not a rendering
    difference -- it is the absence of one.
    """

    rendered = str(vecm_fit)

    assert "raw=" not in rendered
    assert "statsmodels" not in rendered


def test_the_fit_statistics_are_reported(vecm_fit) -> None:
    rendered = str(vecm_fit)

    assert "Number of observations" in rendered
    assert "Log likelihood" in rendered


def test_the_adjustment_values_are_the_stored_alpha(vecm_fit) -> None:
    """The table must print ``alpha``, not recompute it."""

    rendered = str(vecm_fit)
    alpha = vecm_fit.alpha

    for value in alpha.reshape(-1):
        assert f"{float(value):.6f}" in rendered


def test_the_cointegrating_values_are_the_stored_beta(vecm_fit) -> None:
    rendered = str(vecm_fit)
    beta = vecm_fit.beta

    for value in beta.reshape(-1):
        assert f"{float(value):.6f}" in rendered


def test_two_cointegrating_equations_get_two_columns() -> None:
    """``rank(2)`` means two ``ce`` columns, not one column repeated."""

    fitted = ws.VECM(
        _cointegrated(seed=1204), lags=1, coint_rank=2, names=["y", "x"]
    ).fit()

    rendered = str(fitted)

    assert "ce1" in rendered
    assert "ce2" in rendered


def test_a_single_equation_has_no_second_column(vecm_fit) -> None:
    """Guard for the previous test: rank 1 must not show ``ce2``."""

    assert "ce2" not in str(vecm_fit)