"""``ws.Areg`` must be Stata's ``areg``, not ``reghdfe`` under another name.

They are DIFFERENT commands. ``areg`` requires ``absorb()``, prints "Linear regression,
absorbing indicators" with a second line naming the absorbed variable, registers the
``areg_p`` predict program, and forbids a specific set of margins types. ``reghdfe`` has
its own banner (it shows no title line) and registers no predict program at all.

Before this class existed, the package had only the ``reghdfe`` identity: a caller who
wanted ``areg`` got ``e(cmd)="reghdfe"`` and no ``e(title)``/``e(title2)``/``e(predict)``.
That is "areg is not fully implemented", which is what the user reported.

Every expected value below comes from ``tests/conformance/stata_fixtures/areg.json``, a
capture of the real command -- not from prose.

The ESTIMATION is the same model, so the numbers must match ``FEOLS`` exactly; only the
identity differs. Both halves are tested, because getting the metadata right while the
arithmetic drifts would be a worse outcome than not shipping it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURE = Path(__file__).resolve().parent / "conformance" / "stata_fixtures" / "areg.json"


def _fixture() -> dict:
    if not FIXTURE.exists():  # pragma: no cover - committed fixture
        pytest.skip(f"{FIXTURE} not present")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _frame(n_units: int = 30, per_unit: int = 7, seed: int = 14001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_units * per_unit
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "firm": np.repeat(np.arange(n_units), per_unit),
        }
    )


@pytest.fixture(scope="module")
def fitted():
    return ws.Areg.from_formula("y ~ x1 + x2", _frame(), absorb="firm").fit()


# ------------------------------------------------------------- it exists at all
def test_there_is_an_areg_class() -> None:
    """The gap the user reported: no ``areg`` entry point existed."""

    assert hasattr(ws, "Areg")


# ------------------------------------------------ identity, straight from the fixture
def test_cmd_is_areg(fitted) -> None:
    assert fitted.e["cmd"] == _fixture()["macros"]["cmd"] == "areg"


def test_the_banner_matches_stata(fitted) -> None:
    """``areg`` prints a different title from ``reghdfe`` (which prints none)."""

    assert fitted.e["title"] == _fixture()["macros"]["title"]


def test_the_second_title_names_the_absorbed_variable(fitted) -> None:
    """``areg`` prints "Absorbed variable: firm" under the banner."""

    assert fitted.e["title2"] == _fixture()["macros"]["title2"] == "Absorbed variable: firm"


def test_the_predict_program_is_areg_p(fitted) -> None:
    assert fitted.e["predict"] == _fixture()["macros"]["predict"] == "areg_p"


def test_marginsnotok_matches_stata(fitted) -> None:
    """``areg`` refuses these margins types; the macro is part of its contract."""

    assert fitted.e["marginsnotok"] == _fixture()["macros"]["marginsnotok"]


def test_absvar_records_the_absorbed_name(fitted) -> None:
    assert fitted.e["absvar"] == _fixture()["macros"]["absvar"] == "firm"


def test_the_model_name_is_areg_not_reghdfe(fitted) -> None:
    assert fitted.model_name == "areg"


# ------------------------------------------------------ absorb= is required
def test_absorb_is_required() -> None:
    """Stata: "absorb() is required". Without it this is just ``regress``."""

    with pytest.raises(ValueError, match="absorb"):
        ws.Areg.from_formula("y ~ x1", _frame())


def test_an_empty_absorb_is_rejected() -> None:
    with pytest.raises(ValueError):
        ws.Areg.from_formula("y ~ x1", _frame(), absorb=[])


# ------------------------------------------------- the estimation is unchanged
def test_slopes_match_reghdfe(fitted) -> None:
    """Same within model for the SLOPES; only the intercept and identity differ."""

    other = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", _frame()).fit()

    # areg appends _cons; compare the slopes only.
    np.testing.assert_allclose(
        np.asarray(fitted.params, dtype=float)[:-1],
        np.asarray(other.params, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


def test_slope_standard_errors_match_reghdfe(fitted) -> None:
    other = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", _frame()).fit()

    np.testing.assert_allclose(
        np.asarray(fitted.bse, dtype=float)[:-1],
        np.asarray(other.bse, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


def test_the_fit_statistics_are_present(fitted) -> None:
    """``areg`` stores these; a subclass must not lose them."""

    for key in ("N", "r2", "r2_a", "rmse", "F", "df_m", "df_r", "rank", "rss", "mss", "tss"):
        assert key in fitted.e, key


def test_the_absorbed_degrees_of_freedom_are_present(fitted) -> None:
    """``e(df_a)`` and ``e(k_absorb)`` are areg-specific and must survive."""

    assert "df_a" in fitted.e
    assert "k_absorb" in fitted.e


# -------------------------------------------------- predict follows areg_p
def test_predict_offers_the_types_areg_p_has(fitted) -> None:
    """``areg_p.ado`` serves xb (default), d, residuals, dresiduals."""

    assert set(fitted._predict_types) == {"xb", "d", "residuals", "dresiduals"}


def test_predict_xb_includes_the_absorbed_effect(fitted) -> None:
    """Consistent with ``areg_p.ado``: xb is the linear predictor PLUS the effect."""

    np.testing.assert_allclose(
        np.asarray(fitted.predict(kind="xb"), dtype=float),
        np.asarray(fitted.fittedvalues, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


# ------------------------------------------------------ multiple absorb vars
def test_two_absorbed_variables_are_accepted() -> None:
    """Stata's ``absorb(varlist)`` takes more than one."""

    frame = _frame()
    frame["year"] = np.tile(np.arange(7), 30)

    result = ws.Areg.from_formula("y ~ x1", frame, absorb=("firm", "year")).fit()

    assert result.e["absvar"] == "firm year"
    assert np.all(np.isfinite(np.asarray(result.params, dtype=float)))


def test_the_second_title_lists_every_absorbed_variable() -> None:
    frame = _frame()
    frame["year"] = np.tile(np.arange(7), 30)

    result = ws.Areg.from_formula("y ~ x1", frame, absorb=("firm", "year")).fit()

    assert result.e["title2"] == "Absorbed variable: firm year"


# ----------------------------------------------------- it is not reghdfe's twin
def test_reghdfe_still_reports_its_own_identity() -> None:
    """Adding ``Areg`` must not change what ``FEOLS`` says about itself."""

    result = ws.FEOLS.from_formula("y ~ x1 | firm", _frame()).fit()

    assert result.model_name == "reghdfe"
    assert result.e["cmd"] == "reghdfe"
    assert "title" not in result.e
    assert "predict" not in result.e


def test_the_two_commands_disagree_on_identity() -> None:
    """Guard: a shared identity would mean the subclass does nothing."""

    areg = ws.Areg.from_formula("y ~ x1", _frame(), absorb="firm").fit()
    reghdfe = ws.FEOLS.from_formula("y ~ x1 | firm", _frame()).fit()

    assert areg.e["cmd"] != reghdfe.e["cmd"]
    assert areg.model_name != reghdfe.model_name

# -------------------------------------- the intercept, against the real capture
def test_the_reported_names_include_cons(fitted) -> None:
    """``areg`` reports an intercept; ``reghdfe`` does not. areg.json has three names."""

    assert list(fitted.param_names) == _fixture()["names"]
    assert list(fitted.param_names)[-1] == "_cons"


def test_the_coefficients_match_stata_including_the_intercept() -> None:
    """THE decisive check: three values, not two."""

    capture = _fixture()
    result = ws.Areg.from_formula("y ~ x1 + x2", _panel(), absorb="firm").fit()

    np.testing.assert_allclose(
        np.asarray(result.params, dtype=float),
        np.asarray(capture["b"], dtype=float),
        rtol=1e-10,
        atol=0.0,
    )


def test_the_full_covariance_matches_stata() -> None:
    """Including the OFF-DIAGONAL terms.

    areg reports cov(_cons, x1) = 5.80638e-05 on this fixture. A shortcut that widens the
    matrix and leaves those entries at zero reproduces the standard errors but not the
    covariance, so the whole matrix is compared.
    """

    capture = _fixture()
    result = ws.Areg.from_formula("y ~ x1 + x2", _panel(), absorb="firm").fit()

    np.testing.assert_allclose(
        np.asarray(result.cov_params(), dtype=float),
        np.asarray(capture["V"], dtype=float),
        rtol=1e-8,
        atol=1e-12,
    )


def test_the_standard_errors_match_stata() -> None:
    capture = _fixture()
    result = ws.Areg.from_formula("y ~ x1 + x2", _panel(), absorb="firm").fit()
    expected = np.sqrt(np.diag(np.asarray(capture["V"], dtype=float)))

    np.testing.assert_allclose(
        np.asarray(result.bse, dtype=float), expected, rtol=1e-8, atol=1e-12
    )


def test_the_intercept_is_not_zero() -> None:
    """A guard against the bug that was fixed while writing this.

    Deriving the constant as ``mean(y - (fitted - resid))`` gave 0, because pyfixest's
    ``predict`` already includes the absorbed effects -- so that expression is
    ``y - 2*resid``, not the linear part. The value was plausible-looking and wrong.
    """

    result = ws.Areg.from_formula("y ~ x1 + x2", _panel(), absorb="firm").fit()

    assert abs(float(np.asarray(result.params, dtype=float)[-1])) > 1e-6


def test_reghdfe_still_reports_no_intercept() -> None:
    """The subclass must not leak the constant into ``FEOLS``."""

    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", _panel()).fit()

    assert "_cons" not in list(result.param_names)


def _panel() -> pd.DataFrame:
    """The exact panel the areg fixture was captured on (30 firms x 6 periods, seed 21)."""

    rng = np.random.default_rng(21)
    entity = np.repeat(np.arange(30), 6)
    n = 180
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=30)[entity]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity})
