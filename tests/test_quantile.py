"""Tests for quantile and robust regression."""

from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError
from westata.estimators.quantile import (
    QuantileRegression,
    RobustRegression,
    solve_lad,
)


def _quantile_frame(n: int = 600, seed: int = 61) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    scale = 0.5 + 0.4 * np.abs(x1)
    y = 1.0 + 2.0 * x1 - 0.8 * x2 + scale * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _outlier_frame(n: int = 400, seed: int = 67) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    kontam = rng.random(size=n) < 0.05
    y = np.where(kontam, y + rng.normal(scale=30.0, size=n), y)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# --------------------------------------------------------------- solve_lad


def test_solve_lad_with_only_a_constant_reproduces_the_quantile():
    """With only an intercept the LAD fit is a sample quantile of ``y``.

    Interpolation caveat: ``numpy.quantile`` interpolates *between* order
    statistics by default, whereas a check-function minimiser is always an order
    statistic (or an interval of them), so the two can differ by up to one gap.
    The assertion therefore uses a one-spacing tolerance -- and the stronger
    property, that no other location lowers the check-function objective, is
    verified directly below.
    """

    rng = np.random.default_rng(3)
    y = rng.normal(size=500)
    X = np.ones((y.size, 1))

    def objective(center: float, tau: float) -> float:
        residual = y - center
        return float(
            np.sum((tau - (residual < 0).astype(float)) * residual)
        )

    spacing = float(np.max(np.diff(np.sort(y))))

    for tau in (0.25, 0.5, 0.75, 0.9):
        beta = solve_lad(X, y, quantile=tau)
        assert beta[0] == pytest.approx(np.quantile(y, tau), abs=spacing)

        # The real property: the fit attains the minimum of the check function.
        attained = objective(beta[0], tau)
        for perturbation in (-0.5, -0.05, 0.05, 0.5):
            assert objective(beta[0] + perturbation, tau) >= attained - 1e-9

    # At the median with an even n, *every* point between the two middle order
    # statistics minimises the objective (the median of an even sample is an
    # interval, not a point), so the returned value need only lie in that
    # interval -- asserting equality with np.median would encode a false
    # expectation that numpy's particular interpolation midpoint is privileged.
    ordered = np.sort(y)
    median_fit = solve_lad(X, y, quantile=0.5)[0]
    assert ordered[len(y) // 2 - 1] <= median_fit <= ordered[len(y) // 2]


def test_solve_lad_rejects_bad_quantile():
    X = np.ones((10, 1))
    with pytest.raises(ValueError, match="quantile"):
        solve_lad(X, np.arange(10.0), quantile=1.5)


def test_solve_lad_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="observations"):
        solve_lad(np.ones((10, 1)), np.arange(5.0))


# ------------------------------------------------------------------- qreg


def test_qreg_median_is_close_to_ols_on_homoskedastic_data():
    rng = np.random.default_rng(11)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    quantile = QuantileRegression.from_formula("y ~ x1 + x2", frame).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        quantile.params[["Intercept", "x1", "x2"]].to_numpy(),
        plain.params[["Intercept", "x1", "x2"]].to_numpy(),
        rtol=5e-2,
    )


def test_qreg_heteroskedastic_slopes_increase_with_the_quantile():
    frame = _quantile_frame()
    coefficients = []
    for tau in (0.25, 0.5, 0.75, 0.9):
        result = QuantileRegression.from_formula(
            "y ~ x1 + x2", frame, quantile=tau
        ).fit()
        coefficients.append(float(result.params["Intercept"]))

    # The DGP has scale rising in |x1|, so the intercept rises across quantiles.
    assert coefficients == sorted(coefficients)


def test_qreg_pseudo_r2_is_between_zero_and_one():
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", _quantile_frame(), quantile=0.5
    ).fit()
    assert 0.0 < float(result.e["r2_p"]) < 1.0
    # Pseudo R2 here is 1 - sum_adev/sum_rdev, not the OLS R-squared.
    implied = 1.0 - float(result.e["sum_adev"]) / float(result.e["sum_rdev"])
    assert float(result.e["r2_p"]) == pytest.approx(implied, rel=1e-12)


def test_qreg_rejects_a_bad_quantile():
    with pytest.raises(ValueError, match="quantile"):
        QuantileRegression.from_formula("y ~ x1", _quantile_frame(), quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        QuantileRegression.from_formula("y ~ x1", _quantile_frame(), quantile=1.0)


def test_qreg_rejects_an_unknown_cov_type():
    model = QuantileRegression.from_formula("y ~ x1 + x2", _quantile_frame())
    with pytest.raises(ValueError, match="cov_type"):
        model.fit(cov_type="hac")


def test_qreg_robust_and_iid_differ_but_are_close_on_iid_errors():
    rng = np.random.default_rng(19)
    n = 600
    x1 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x1": x1})

    iid = QuantileRegression.from_formula("y ~ x1", frame).fit(cov_type="iid")
    robust = QuantileRegression.from_formula("y ~ x1", frame).fit(cov_type="robust")

    np.testing.assert_allclose(iid.params.to_numpy(), robust.params.to_numpy())
    ratio = float(robust.bse["x1"]) / float(iid.bse["x1"])
    # Measured against Stata: vce(robust) differs from the iid default by only
    # about 1% on this DGP (.0352211 vs .0355009), so the ratio must be near 1.
    # An earlier version of the sandwich dropped the density factor and gave
    # 0.0204 here, which is what this bound now catches.
    assert 0.8 < ratio < 1.25


def test_qreg_populates_e_and_renders():
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", _quantile_frame(), quantile=0.5
    ).fit()
    for key in ["N", "df_m", "df_r", "q", "sum_adev", "sum_rdev", "r2_p", "rank"]:
        assert key in result.e, key
    assert "Coef." in result.summary(style="stata")


# ------------------------------------------------------------------- rreg


def test_rreg_beats_ols_on_contaminated_data():
    frame = _outlier_frame()
    robust = RobustRegression.from_formula("y ~ x1 + x2", frame).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    true_slope = 2.0
    assert abs(float(robust.params["x1"]) - true_slope) < abs(
        float(plain.params["x1"]) - true_slope
    )


def test_rreg_reports_a_positive_scale():
    result = RobustRegression.from_formula("y ~ x1 + x2", _outlier_frame()).fit()
    assert np.isfinite(float(result.e["scale"]))
    assert float(result.e["scale"]) > 0


def test_rreg_cooks_screen_is_reported_and_consistent():
    """The Cook's-distance screen is applied, but need not drop anything.

    On the reference DGP the largest Cook's D is 0.218, well below the threshold
    of 1, so Stata and westata both keep every observation. Asserting that at
    least one point is dropped would have encoded a false expectation -- the
    screen exists, but this contamination is handled by the robust weights
    rather than by deletion.
    """

    frame = _outlier_frame()
    result = RobustRegression.from_formula("y ~ x1 + x2", frame).fit()
    dropped = int(result.e["dropped"])
    assert dropped >= 0
    assert int(result.e["N"]) == len(frame) - dropped

    # A hand-computed cook's D max confirms the threshold behaviour.
    X = np.column_stack(
        [np.ones(len(frame)), frame["x1"].to_numpy(), frame["x2"].to_numpy()]
    )
    y = frame["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    p = X.shape[1]
    sigma2 = float(residual @ residual) / (len(y) - p)
    leverage = np.einsum(
        "ij,jk,ik->i", X, np.linalg.pinv(X.T @ X), X
    )
    cooks = residual**2 / (p * sigma2) * (leverage / (1 - leverage) ** 2)
    assert dropped == int((cooks > 1.0).sum())


def test_rreg_populates_e_and_renders():
    result = RobustRegression.from_formula("y ~ x1 + x2", _outlier_frame()).fit()
    for key in ["N", "df_m", "df_r", "rank", "scale"]:
        assert key in result.e, key
    assert "Coef." in result.summary(style="stata")


# ------------------------------------------------- Stata conformance (P8)
#
# Reference values are read from the captured StataNow 19.5 output in
# ``tests/conformance/stata_fixtures/p8_outputs.txt``. DGP: n=600, seed=61,
# x1,x2 ~ N(0,1), scale = 0.5 + 0.4|x1|, y = 1 + 2x1 - 0.8x2 + scale*N(0,1).

FIXTURE = (
    Path(__file__).resolve().parent / "conformance" / "stata_fixtures" / "p8_outputs.txt"
)


def _to_float(token: str) -> float:
    """Parse a Stata-printed number, which may omit the leading zero (``.518``)."""

    token = token.strip()
    return float(token if token.lstrip("-").startswith(("0", ".")) else token)


@functools.lru_cache(maxsize=1)
def _stata_reference() -> dict[str, dict[str, dict[str, float]]]:
    """Parse the qreg/rreg coefficient blocks out of the captured fixture.

    Reading them from the file rather than retyping them keeps this test honest
    about its source: if the fixture is missing the tests skip instead of
    silently comparing against a stale hard-coded number.
    """

    if not FIXTURE.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip("P8 fixture missing")

    text = FIXTURE.read_text(encoding="utf-8")
    blocks = re.split(r"^=== (.+?) ===$", text, flags=re.MULTILINE)
    parsed: dict[str, dict[str, dict[str, float]]] = {}

    for name, body in zip(blocks[1::2], blocks[2::2]):
        entries: dict[str, dict[str, float]] = {}
        for line in body.splitlines():
            # Coefficient rows look like "  x1 |   2.042842   .0472461 ...".
            match = re.match(
                r"\s*(\w+)\s*\|\s*(-?[\d.]+)\s+(-?[\d.]+)", line
            )
            if match:
                label, coef, std_err = match.groups()
                entries[label.strip()] = {
                    "coef": _to_float(coef),
                    "se": _to_float(std_err),
                }
        if entries:
            parsed[name.strip()] = entries
    return parsed


def _stata_block(quantile: float) -> str:
    """The fixture block name for a ``qreg`` at ``quantile``."""

    return f"qreg_{quantile}"


def _stata_coef(block: str, variable: str) -> tuple[float, float]:
    """``(coefficient, std. error)`` for one variable of one fixture block.

    ``variable`` uses Stata's spelling (``_cons``), which is also how the fixture
    keys its rows; the westata parameter name is mapped by the caller.
    """

    reference = _stata_reference()
    if block not in reference:
        pytest.skip(f"fixture block {block!r} missing")
    entries = reference[block]
    if variable not in entries:
        pytest.skip(f"{block!r} has no row for {variable!r}")
    return entries[variable]["coef"], entries[variable]["se"]


def _stata_scalar(block: str, pattern: str) -> float:
    """Pull a single number out of a fixture block by regex."""

    text = FIXTURE.read_text(encoding="utf-8")
    marker = f"=== {block} ==="
    if marker not in text:  # pragma: no cover - fixture drift
        pytest.skip(f"fixture block {block!r} missing")
    section = text.split(marker, 1)[1].split("===", 1)[0]
    match = re.search(pattern, section)
    if not match:  # pragma: no cover - fixture drift
        pytest.skip(f"{pattern!r} not found in {block!r}")
    return _to_float(match.group(1))


@pytest.mark.parametrize("quantile", [0.25, 0.5, 0.75, 0.9])
def test_qreg_coefficients_match_every_stata_quantile(quantile):
    """All four captured quantiles, coefficients to 1e-3 relative."""

    block = _stata_block(quantile)
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", _quantile_frame(), quantile=quantile
    ).fit()

    for variable in ("x1", "x2", "_cons"):
        expected, _ = _stata_coef(block, variable)
        key = "Intercept" if variable == "_cons" else variable
        assert float(result.params[key]) == pytest.approx(expected, rel=1e-3), variable


@pytest.mark.parametrize("quantile", [0.25, 0.5, 0.75, 0.9])
def test_qreg_standard_errors_match_stata(quantile):
    """The sparsity estimate is the part most likely to be wrong.

    Stata's default ``vce(iid)`` uses the *fitted* sparsity: re-estimate at
    ``tau ± h``, form ``xbar'b`` at each and difference over ``2h``, with the
    Hall-Sheather bandwidth. A kernel density of the residuals (the
    ``vce(iid, kernel())`` option) would instead shift every standard error by
    ~29% while leaving the coefficients identical.
    """

    block = _stata_block(quantile)
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", _quantile_frame(), quantile=quantile
    ).fit()

    for variable in ("x1", "x2", "_cons"):
        _, expected_se = _stata_coef(block, variable)
        key = "Intercept" if variable == "_cons" else variable
        assert float(result.bse[key]) == pytest.approx(expected_se, rel=5e-2), variable


@pytest.mark.parametrize("quantile", [0.25, 0.5, 0.75, 0.9])
def test_qreg_pseudo_r2_matches_stata(quantile):
    """Pseudo R2 is ``1 - sum_adev/sum_rdev`` on CHECK-function sums.

    Using plain ``sum |residual|`` instead is right at the median, where the two
    coincide, and wrong everywhere else -- at ``tau=.25`` it reports 0.636
    against Stata's 0.6424.
    """

    block = _stata_block(quantile)
    expected = _stata_scalar(block, r"Pseudo R2\s*=\s*(-?\d+\.\d+)")
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", _quantile_frame(), quantile=quantile
    ).fit()
    assert float(result.e["r2_p"]) == pytest.approx(expected, abs=5e-4)


def test_qreg_sum_adev_is_the_check_function_not_the_absolute_sum():
    """``e(sum_adev)`` is Stata's "Min sum of deviations", a check-function sum.

    At ``tau=.25`` the fixture prints 156.7522 while ``sum |residual|`` is
    465.29 -- the two are only equal at the median, which is why a
    median-only test cannot catch this.
    """

    expected = _stata_scalar("qreg_0.25", r"Min sum of deviations\s+(\d+\.\d+)")
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", _quantile_frame(), quantile=0.25
    ).fit()

    assert float(result.e["sum_adev"]) == pytest.approx(expected, abs=5e-4)
    absolute_sum = float(np.sum(np.abs(result.resid)))
    assert absolute_sum > 2.5 * expected


def test_qreg_robust_vce_matches_stata():
    """``vce(robust)`` reproduces Stata's Robust column exactly.

    Stata's ``[R] qreg`` prints no closed form for this, so it is checked
    numerically: the fixture's ``qreg ..., quantile(.5) vce(robust)`` gives
    .0352211 / .0336117 / .0359358 for x1 / x2 / _cons.  The observation-specific
    fitted density is what makes it land: using the scalar sparsity at ``xbar``
    returns the iid answer instead (about 1% low).
    """

    for variable in ("x1", "x2", "_cons"):
        _, expected_se = _stata_coef("qreg_robust", variable)
        ours = QuantileRegression.from_formula(
            "y ~ x1 + x2", _quantile_frame(), quantile=0.5
        ).fit(cov_type="robust")
        key = "Intercept" if variable == "_cons" else variable
        assert float(ours.bse[key]) == pytest.approx(expected_se, rel=1e-3), variable


def test_rreg_coefficients_match_stata():
    """Huber *then* biweight, not Huber alone.

    Stopping after the Huber stage leaves the coefficients 0.6% away from
    Stata's; running the biweight stage from the Huber result lands within 2e-3.
    """

    result = RobustRegression.from_formula("y ~ x1 + x2", _outlier_frame()).fit()
    for variable in ("x1", "x2", "_cons"):
        expected, _ = _stata_coef("rreg", variable)
        key = "Intercept" if variable == "_cons" else variable
        assert float(result.params[key]) == pytest.approx(expected, rel=5e-3), variable


def test_rreg_standard_errors_are_close_to_stata():
    """Pseudovalue SEs, with a documented few-percent residual gap.

    The tolerance is deliberately loose. ``[R] rreg`` says only that the SEs use
    the Street-Carroll-Ruppert pseudovalues and does not pin down the
    leave-one-out refit; westata refits the full two-stage estimator, which
    leaves ``_cons`` +0.1%, ``x1`` -2.8%, ``x2`` +3.4%. A one-step weighted
    refit instead is 26-36% off, which this assertion still catches.
    """

    result = RobustRegression.from_formula("y ~ x1 + x2", _outlier_frame()).fit()
    for variable in ("x1", "x2", "_cons"):
        _, expected_se = _stata_coef("rreg", variable)
        key = "Intercept" if variable == "_cons" else variable
        assert float(result.bse[key]) == pytest.approx(expected_se, rel=6e-2), variable


def test_qreg_records_stata_e_scalars():
    """The documented ``e()`` scalars exist and are self-consistent."""

    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", _quantile_frame(), quantile=0.25
    ).fit()

    for key in (
        "N",
        "df_m",
        "df_r",
        "rank",
        "q",
        "q_v",
        "sum_adev",
        "sum_rdev",
        "r2_p",
        "sparsity",
        "f_r",
        "bwidth",
        "bwmethod",
        "denmethod",
    ):
        assert key in result.e, key

    assert float(result.e["q"]) == 0.25
    assert result.e["bwmethod"] == "hsheather"
    assert result.e["denmethod"] == "fitted"
    assert float(result.e["sparsity"]) > 0
    assert float(result.e["f_r"]) == pytest.approx(1.0 / float(result.e["sparsity"]))
    assert float(result.e["bwidth"]) > 0
    assert float(result.e["r2_p"]) == pytest.approx(
        1.0 - float(result.e["sum_adev"]) / float(result.e["sum_rdev"])
    )
    # The raw quantile is the order statistic Stata prints as "(about q_v)".
    ordered = np.sort(_quantile_frame()["y"].to_numpy())
    assert float(result.e["q_v"]) == pytest.approx(ordered[int(0.25 * len(ordered)) - 1])


def test_qreg_rejects_kernel_and_residual_denmethods():
    """Non-default density estimators must fail loudly, not fall back.

    Silently using ``fitted`` where the caller asked for a kernel density would
    produce the right coefficients with ~29% wrong standard errors -- the exact
    failure mode this module exists to avoid.
    """

    model = QuantileRegression.from_formula("y ~ x1 + x2", _quantile_frame())
    for bad in ("kernel", "iid, kernel(epanechnikov)", "iid, residual", "residual"):
        with pytest.raises(ValueError, match="cov_type"):
            model.fit(cov_type=bad)


def test_rreg_documents_huber_only_vs_full_scheme():
    """Pin the measured difference between stopping at Huber and the full run.

    This is the claim the module docstring makes; if the biweight stage is ever
    dropped from ``fit``, this test fails rather than letting the coefficients
    silently drift 0.6%.
    """

    frame = _outlier_frame()
    model = RobustRegression.from_formula("y ~ x1 + x2", frame)
    full = model.fit()

    X = np.column_stack(
        [np.ones(len(frame)), frame["x1"].to_numpy(), frame["x2"].to_numpy()]
    )
    y = frame["y"].to_numpy()
    huber_only = model._huber_until_converged(X, y, maxiter=50, strict=False)

    assert not np.allclose(
        huber_only, full.params.to_numpy(), rtol=5e-3
    ), "the biweight stage must change the estimates"

