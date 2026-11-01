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


# ------------------------------------------- output parity beyond the numbers
def test_the_summary_prints_the_absorbed_variable_line() -> None:
    """Stata's areg prints a SECOND banner line; a table without it is not "the same output".

    The macro ``e(title2)`` was recorded from the start but the renderer ignored it, so
    ``Areg`` showed the right numbers under a one-line banner. This is the assertion that
    the line actually reaches the table.
    """

    result = ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit()
    rendered = result.summary(style="stata")

    assert "Absorbed variable: firm" in rendered


def test_the_absorbed_variable_line_is_the_second_line() -> None:
    result = ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit()
    lines = result.summary(style="stata").splitlines()

    assert lines[0] == "Linear regression, absorbing indicators"
    assert "Absorbed variable: firm" in lines[1]


def test_reghdfe_prints_no_absorbed_variable_line() -> None:
    """The subtitle is areg-only; leaking it into FEOLS would be a new inconsistency."""

    result = ws.FEOLS.from_formula("y ~ x1 | firm", _panel()).fit()

    assert "Absorbed variable" not in result.summary(style="stata")


@pytest.mark.parametrize(
    "key,expected",
    [
        ("estmethod", "areg"),
        ("marginsprop", "nolinearize"),
        ("vce", "ols"),
    ],
)
def test_the_command_macros_match_stata(key: str, expected: str) -> None:
    """These are in areg.json and were all missing before.

    ``e(vce)`` matters most: it is the macro a Stata user reads to learn which covariance
    was requested, and its absence made Areg look like it had no vce at all.
    """

    capture = _fixture()
    result = ws.Areg.from_formula("y ~ x1 + x2", _panel(), absorb="firm").fit()

    assert result.e[key] == capture["macros"][key] == expected


def test_every_stata_scalar_is_present() -> None:
    """The numeric side of the contract: no areg.json scalar may be missing."""

    capture = _fixture()
    result = ws.Areg.from_formula("y ~ x1 + x2", _panel(), absorb="firm").fit()

    missing = [key for key in capture["scalars"] if key not in result.e]
    assert missing == [], f"areg.json scalars absent from e(): {missing}"


def test_vcetype_is_cluster_under_clustering() -> None:
    """``e(vcetype)`` follows the requested vce, not a fixed string."""

    frame = _panel()
    result = ws.Areg.from_formula("y ~ x1", frame, absorb="firm").fit(
        cov_type="cluster", cov_kwds={"cluster": "firm"}
    )

    assert result.e["vce"] == "cluster"


# --------------------------------------------------- vce(hc2) and vce(hc3)
def test_areg_accepts_hc2() -> None:
    """areg.sthlp lists ``vce(hc2 [clustvar])``; FEOLS refuses it, Areg must not.

    The refusal in FEOLS is honest -- PyFixest does not expose the leverage of the full
    dummy design -- but Areg builds that design itself for ``_cons``, so the restriction
    does not apply.
    """

    result = ws.Areg.from_formula("y ~ x1", _frame(), absorb="firm").fit(cov_type="HC2")

    assert np.all(np.isfinite(np.asarray(result.bse, dtype=float)))


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_the_leverage_vces_get_increasingly_conservative(cov_type: str) -> None:
    """The HC family must actually differ, and in the documented order.

    HC2 divides by ``1-h`` and HC3 by ``(1-h)^2``, so for the same residuals their
    standard errors come out progressively larger. A stub that returned HC1 for all four
    would pass "the call works" and fail this.
    """

    fitted = ws.Areg.from_formula("y ~ x1", _frame(), absorb="firm").fit(cov_type=cov_type)
    standard = float(np.asarray(fitted.bse, dtype=float)[0])

    baseline = ws.Areg.from_formula("y ~ x1", _frame(), absorb="firm").fit(cov_type="HC0")
    hc0 = float(np.asarray(baseline.bse, dtype=float)[0])

    expected_rank = {"HC0": 0, "HC1": 1, "HC2": 2, "HC3": 3}[cov_type]
    others = {"HC0": hc0}
    for other in ("HC1", "HC2", "HC3"):
        result = ws.Areg.from_formula("y ~ x1", _frame(), absorb="firm").fit(cov_type=other)
        others[other] = float(np.asarray(result.bse, dtype=float)[0])

    ordered = sorted(others, key=lambda key: others[key])
    assert ordered.index(cov_type) == expected_rank, (
        f"the HC family is not ordered as documented: {others}"
    )


def test_fit_does_not_change_the_point_estimates() -> None:
    """The VCE never moves the coefficients -- a guard against the rebuild leaking in."""

    baseline = ws.Areg.from_formula("y ~ x1 + x2", _frame(), absorb="firm").fit(
        cov_type="nonrobust"
    )
    robust = ws.Areg.from_formula("y ~ x1 + x2", _frame(), absorb="firm").fit(cov_type="HC2")

    np.testing.assert_allclose(
        np.asarray(baseline.params, dtype=float),
        np.asarray(robust.params, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


def test_reghdfe_still_refuses_hc2() -> None:
    """The refusal belongs to FEOLS and must not have been removed by this change."""

    with pytest.raises(ValueError, match="HC2"):
        ws.FEOLS.from_formula("y ~ x1 | firm", _frame()).fit(cov_type="HC2")


# --------------------------------------------------- noabsorb (noabsorbtest)
def test_absorb_test_is_computed_by_default() -> None:
    """``e(F_absorb)``/``e(p_absorb)`` are part of areg's stored results."""

    result = ws.Areg.from_formula("y ~ x1", _frame(), absorb="firm").fit()

    assert result.e.get("F_absorb") is not None
    assert result.e.get("p_absorb") is not None


def test_noabsorb_omits_the_absorb_test() -> None:
    """Stata's ``noabsorbtest`` "prevents computation of the F test".

    The observable consequence is that the scalars are ABSENT, not present-and-empty --
    so the assertion is on key presence, not on a value.
    """

    result = ws.Areg.from_formula("y ~ x1", _frame(), absorb="firm", noabsorb=True).fit()

    assert "F_absorb" not in result.e
    assert "p_absorb" not in result.e


def test_noabsorb_does_not_change_the_estimate() -> None:
    """It skips a test, not the fit -- every other number must be identical."""

    with_test = ws.Areg.from_formula("y ~ x1 + x2", _frame(), absorb="firm").fit()
    without = ws.Areg.from_formula(
        "y ~ x1 + x2", _frame(), absorb="firm", noabsorb=True
    ).fit()

    for key in ("r2", "rmse", "df_a", "k_absorb", "N"):
        assert without.e[key] == pytest.approx(with_test.e[key]), key

    np.testing.assert_allclose(
        np.asarray(without.params, dtype=float),
        np.asarray(with_test.params, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


def test_noabsorb_is_areg_only() -> None:
    """``reghdfe`` has no such option, so FEOLS must not silently grow one."""

    import inspect

    assert "noabsorb" not in inspect.signature(ws.FEOLS.from_formula).parameters


# ------------------------------------- clustered SE across ABSORB/CLUSTER pairings
#
# Reference values are from StataNow 19.5 on the 210-row panel (30 groups x 7 periods,
# seed 1, one regressor, absorb(a)). The pairing matters: when the cluster variable IS the
# absorbed dimension, when it is crossed with it, and when it is the other dimension each
# take a different path, and an implementation can get one right while breaking another.
def _pair_panel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 210
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "a": np.repeat(np.arange(30), 7),
            "b": np.tile(np.arange(7), 30),
        }
    )


@pytest.mark.parametrize(
    "cluster,expected",
    [
        ("a", (0.08671082, 0.00901583)),
        ("b", (0.10683824, 0.10021268)),
        (["a", "b"], (0.11086346, 0.07460234)),
    ],
    ids=["nested-in-absorb", "crossed-with-absorb", "multiway"],
)
def test_clustered_se_matches_stata_for_every_pairing(cluster, expected) -> None:
    """One reference per pairing, because they exercise different code paths.

    The regression this guards: applying areg's degrees-of-freedom correction
    unconditionally fixed the nested case and broke the crossed one, where westata's
    covariance was already exactly Stata's.
    """

    frame = _pair_panel()
    labels = frame[cluster]
    fitted = ws.Areg.from_formula("y ~ x1", frame, absorb="a").fit(
        cov_type="cluster", cov_kwds={"cluster": labels}
    )

    np.testing.assert_allclose(
        # 1e-6 rather than 1e-7: the reference values are transcribed to 8 decimals,
        # and _cons = 0.00901583 has only 3 significant figures, so its relative
        # tolerance is dominated by the rounding of the reference itself.
        np.asarray(fitted.bse, dtype=float), np.array(expected), rtol=1e-6, atol=0.0
    )


def test_multiway_rebuild_reproduces_stata_not_reghdfe() -> None:
    """The distinction this fix had to preserve.

    On this panel Stata's reghdfe reports 0.1023861 for the multiway slope while areg
    reports 0.11086346. Areg must produce the LATTER. Producing the former is what scaling
    FEOLS' covariance left behind -- and it looked plausible, since 0.1024 is a perfectly
    reasonable standard error.
    """

    frame = _pair_panel()
    fitted = ws.Areg.from_formula("y ~ x1", frame, absorb="a").fit(
        cov_type="cluster", cov_kwds={"cluster": frame[["a", "b"]]}
    )
    slope_se = float(np.asarray(fitted.bse, dtype=float)[0])

    assert slope_se == pytest.approx(0.11086346, rel=1e-7)
    assert abs(slope_se - 0.1023861) > 1e-3, "Areg reported reghdfe's value, not areg's"


def test_reghdfe_keeps_its_own_multiway_value() -> None:
    """Stata's reghdfe says 0.1023861 here; FEOLS must keep saying that.

    The Areg fix rebuilt its covariance locally and must not have touched FEOLS.
    """

    frame = _pair_panel()
    fitted = ws.FEOLS.from_formula("y ~ x1 | a", frame).fit(
        cov_type="cluster", cov_kwds={"cluster": frame[["a", "b"]]}
    )

    assert float(np.asarray(fitted.bse, dtype=float)[0]) == pytest.approx(
        0.1023861, rel=1e-6
    )
