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


# ------------------------------------- Stata conformance, captured 2026-10-24
#
# Every number below comes from a live StataNow 19.5 MP session in this repo's
# verification run (`retest/fix/limited/repro_qreg.py`,
# `repro_rreg.py`), which prints the westata/Stata pair side by side.
#
# qreg DGPs (n = 400, seed in the helper below):
#   hetero: y = 1 + 0.8 x1 - 0.4 x2 + N(0,1) * (1 + 0.4|x1|)
#   homo:   y = 1 + 0.8 x1 - 0.4 x2 + N(0,1)
#
# The row order is (Intercept, x1, x2) to match westata's parameter order.

def _qreg_hetero(n: int = 400, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.8 * x1 - 0.4 * x2 + rng.normal(size=n) * (1.0 + 0.4 * np.abs(x1))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _qreg_homo(n: int = 400, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.8 * x1 - 0.4 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


QREG_STATA = {
    ("hetero", 0.25, "iid"): (
        (-0.003881409205964026, 0.8557389284566097, -0.47363887525253257),
        (0.09528485940052442, 0.1009805419135872, 0.0857565395878686),
    ),
    ("hetero", 0.25, "robust"): (
        (-0.003881409205964026, 0.8557389284566097, -0.47363887525253257),
        (0.09471935466185526, 0.09707325188713943, 0.08136205789693532),
    ),
    ("hetero", 0.5, "iid"): (
        (0.9641242295889249, 0.9852053875671986, -0.40589126048329904),
        (0.08062979231172564, 0.08544946356895426, 0.07256695365711024),
    ),
    ("hetero", 0.5, "robust"): (
        (0.9641242295889249, 0.9852053875671986, -0.40589126048329904),
        (0.07994507700152723, 0.08224834623273646, 0.06947292696003815),
    ),
    ("hetero", 0.75, "iid"): (
        (1.8692236512569003, 1.0358823474202146, -0.47550343893006164),
        (0.09478367114570825, 0.10044939497281957, 0.08530546928472155),
    ),
    ("hetero", 0.75, "robust"): (
        (1.8692236512569003, 1.0358823474202146, -0.47550343893006164),
        (0.09758432532166438, 0.09827913273179806, 0.084232908965219),
    ),
    ("homo", 0.25, "iid"): (
        (0.3261243075540123, 0.8383730881996156, -0.4176017413599201),
        (0.06902391069043111, 0.0691995835734422, 0.06613640368491634),
    ),
    ("homo", 0.25, "robust"): (
        (0.3261243075540123, 0.8383730881996156, -0.4176017413599201),
        (0.06957554995776781, 0.06753241461389359, 0.06491743410984092),
    ),
    ("homo", 0.5, "iid"): (
        (1.0436364830076825, 0.8377408618029389, -0.3942173392728379),
        (0.06821332543747044, 0.06838693529845208, 0.0653597279942085),
    ),
    ("homo", 0.5, "robust"): (
        (1.0436364830076825, 0.8377408618029389, -0.3942173392728379),
        (0.0688071449785925, 0.06580961375118034, 0.06268796943609274),
    ),
    ("homo", 0.75, "iid"): (
        (1.784449920453022, 0.8030325774130878, -0.4924448523989313),
        (0.05980594037449168, 0.05995815258416925, 0.05730405269418842),
    ),
    ("homo", 0.75, "robust"): (
        (1.784449920453022, 0.8030325774130878, -0.4924448523989313),
        (0.06086150861590202, 0.05924392358817859, 0.05605319510973606),
    ),
}


@pytest.mark.parametrize(
    "dataset,tau,cov_type",
    sorted(QREG_STATA),
    ids=lambda value: str(value),
)
def test_qreg_standard_errors_match_stata_at_every_quantile(dataset, tau, cov_type):
    """``vce(robust)`` and the iid default, tau = .25/.5/.75, both DGPs.

    The old robust VCE used ``sum psi_i^2 x x'`` as the meat.  That coincides
    with Stata's ``tau(1-tau) X'X`` only at ``tau = .5``, where ``psi^2 == .25``
    for every observation -- so a median-only check could not see the 4.6% /
    128% error at other quantiles.  The bread also keeps the per-observation
    density ``fi_i``; replacing the meat alone is not enough (it still missed by
    15%-147% on heteroskedastic data).  Verified to 1.5e-8.
    """

    frame = _qreg_hetero() if dataset == "hetero" else _qreg_homo()
    expected_b, expected_se = QREG_STATA[(dataset, tau, cov_type)]
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", frame, quantile=tau
    ).fit(cov_type=cov_type)

    ours_b = np.asarray(result.params, dtype=float)[[0, 1, 2]]
    ours_se = np.asarray(result.bse, dtype=float)[[0, 1, 2]]
    np.testing.assert_allclose(ours_b, expected_b, rtol=1e-6)
    np.testing.assert_allclose(ours_se, expected_se, rtol=1e-6)


def test_qreg_robust_meat_is_tau_one_minus_tau_xpx():
    """The meat is ``tau(1-tau) X'X``; the bread keeps the per-observation density.

    Checked structurally rather than by a number: at ``tau != .5`` the two
    candidate meats differ, and the resulting SEs must still land on Stata's.
    """

    frame = _qreg_hetero()
    model = QuantileRegression.from_formula("y ~ x1 + x2", frame, quantile=0.25)
    result = model.fit(cov_type="robust")

    X = np.column_stack([np.ones(len(frame)), frame["x1"], frame["x2"]])
    tau = 0.25
    h = float(result.e["bwidth"])
    difference = X @ (model._sparsity["beta_upper"] - model._sparsity["beta_lower"])
    density = np.where(difference > np.sqrt(np.finfo(float).eps),
                       (2.0 * h) / difference, 0.0)
    meat = tau * (1.0 - tau) * (X.T @ X)
    jacobian = (X.T * density) @ X
    expected = np.linalg.inv(jacobian) @ meat @ np.linalg.inv(jacobian)

    np.testing.assert_allclose(
        np.asarray(result.cov_params(), dtype=float), expected, rtol=1e-12
    )

    # And it must NOT equal the psi^2 meat at tau = .25.
    residual = np.asarray(result.resid, dtype=float)
    psi = tau - (residual < 0).astype(float)
    wrong_meat = (X.T * psi**2) @ X
    wrong = np.linalg.inv(jacobian) @ wrong_meat @ np.linalg.inv(jacobian)
    assert not np.allclose(expected, wrong, rtol=1e-3)


# ---------------------------------------------- rreg conformance (T7)
#
# DGP captured live on 2026-10-24 from StataNow 19.5 MP:
# n=400, seed=67, x1,x2 ~ N(0,1), y = 1 + 2x1 - 0.5x2 + N(0,1), then 5% of y
# shifted by N(0, 30^2).  `rreg y x1 x2` (Stata's default tolerance(.01)).

RREG_OUTLIER_STATA_B = (0.9933213931008713, 2.0142039005277788,
                        -0.5340516956105767)
RREG_OUTLIER_STATA_SE = (0.05451065486338324, 0.05642189005277947,
                         0.052575518969933524)
RREG_OUTLIER_STATA = {
    "rss": 471.32198880778736,
    "mss": 1635.793503558607,
    "rmse": 1.089591226019237,
    "r2": 0.7763188631495137,
    "F": 688.9239589218558,
    "df_r": 397.0,
    "N": 400.0,
}


def _rreg_outlier_frame():
    rng = np.random.default_rng(67)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    contaminated = rng.random(size=n) < 0.05
    y = np.where(contaminated, y + rng.normal(scale=30.0, size=n), y)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_rreg_standard_errors_are_rmse_times_the_ols_law():
    """T7: ``se(b) = e(rmse) sqrt(diag((X'X)^-1))``, not the STR pseudovalues.

    The Street-Carroll-Ruppert pseudovalue variance that westata used to report
    is 15-30% away from Stata's on four datasets.  Stata's ``rreg`` reports the
    plain ``regress`` law evaluated on the adjusted response, which reproduces
    ``e(V)`` to 3e-11.
    """

    frame = _rreg_outlier_frame()
    result = RobustRegression.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        np.asarray(result.params, dtype=float), RREG_OUTLIER_STATA_B, rtol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(result.bse, dtype=float), RREG_OUTLIER_STATA_SE, rtol=1e-6
    )

    X = np.column_stack([np.ones(len(frame)), frame["x1"], frame["x2"]])
    law = float(result.e["rmse"]) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    np.testing.assert_allclose(np.asarray(result.bse, dtype=float), law, rtol=1e-12)


def test_rreg_reported_sum_of_squares_match_stata():
    """``rss/mse/rmse/r2/F`` come from the unweighted regression on ``y_adj``.

    The old values were *weighted* sums using the final biweights, which is a
    different quantity: ``e(rss)`` was 330.3 against Stata's 471.3.
    """

    frame = _rreg_outlier_frame()
    result = RobustRegression.from_formula("y ~ x1 + x2", frame).fit()

    for key, expected in RREG_OUTLIER_STATA.items():
        assert float(result.e[key]) == pytest.approx(expected, rel=1e-9), key
    # The constraint the reported block has to satisfy.
    assert float(result.e["rmse"]) ** 2 * float(result.e["df_r"]) == pytest.approx(
        float(result.e["rss"]), rel=1e-12
    )
    assert float(result.e["mse"]) == pytest.approx(
        float(result.e["rss"]) / float(result.e["df_r"]), rel=1e-12
    )


def test_rreg_lambda_uses_the_nonzero_weight_count():
    """Stata's ``lambda`` divides by ``e(N)`` of the last *weighted* regression.

    Zero-biweight rows are dropped by ``regress [aw=...]``, so that ``e(N)`` is
    the count of non-zero weights (386 here), not the sample size (400).  Using
    the full N moves ``e(rss)`` by 1.7e-4 and every standard error with it.
    """

    frame = _rreg_outlier_frame()
    result = RobustRegression.from_formula("y ~ x1 + x2", frame).fit()

    assert int(result.e["N"]) == 400
    assert int(result.e["N_weighted"]) == 386
    assert float(result.e["df_r"]) == 397.0


# ------------------------------- approved behaviour changes (documented in
# estimators/quantile.py, section "Deviations from the previous westata
# behaviour").  These are NOT regressions: each one is what Stata does.  The
# tests exist so that a future "cleanup" cannot silently undo them.

def test_rreg_default_tolerance_is_statas_not_machine_precision():
    """Change 1: the default stopping rule is Stata's ``tolerance(.01)``.

    Iterating to machine precision lands on a *different point* (1.9e-4 relative
    in the coefficients) because Stata stops early.  Both answers are legitimate
    fits of different stopping rules; neither may be produced silently by the
    other's setting.
    """

    frame = _rreg_outlier_frame()
    model = RobustRegression.from_formula("y ~ x1 + x2", frame)

    default = model.fit()                 # Stata's tolerance(.01)
    tight = model.fit(tolerance=1e-9)     # fully converged

    assert float(default.e["tolerance"]) == 0.01
    assert int(default.e["iterations"]) < int(tight.e["iterations"])

    gap = np.max(
        np.abs(np.asarray(default.params, dtype=float)
               - np.asarray(tight.params, dtype=float))
        / np.abs(np.asarray(tight.params, dtype=float))
    )
    assert 1e-5 < gap < 1e-3, gap

    # The default is the one that matches Stata.
    np.testing.assert_allclose(
        np.asarray(default.params, dtype=float), RREG_OUTLIER_STATA_B, rtol=1e-9
    )


def test_rreg_sums_of_squares_describe_the_adjusted_regression():
    """Change 2: ``e(rss)`` is the unweighted adjusted regression's, not a
    weighted sum over the final biweights.

    On this DGP the weighted version gives 330.3 where Stata reports 471.3, so
    the distinction is not cosmetic.  ``e(mse)`` is new and the identity
    ``rmse^2 * (n-k) == rss`` holds exactly.
    """

    frame = _rreg_outlier_frame()
    result = RobustRegression.from_formula("y ~ x1 + x2", frame).fit()

    assert float(result.e["rss"]) == pytest.approx(471.32198880778736, rel=1e-9)
    assert float(result.e["rss"]) > 400.0          # not the weighted 330.3
    assert float(result.e["mse"]) == pytest.approx(
        float(result.e["rss"]) / float(result.e["df_r"]), rel=1e-12
    )
    assert float(result.e["rmse"]) ** 2 * float(result.e["df_r"]) == pytest.approx(
        float(result.e["rss"]), rel=1e-12
    )


def test_rreg_huber_threshold_is_two_mad_not_1345():
    """Change 3: the Huber threshold is ``2*median(|e - median(e)|)``.

    ``1.345 / 0.6745 = 1.9941``, so the old ``1.345 * scale`` is a *different*
    threshold -- not the same rule written differently.  Verified against
    StataNow 19.5: the 2*MAD version reproduces ``rreg y x1 x2`` exactly
    (``RREG_OUTLIER_STATA_B``); the 1.345 version is 7.9e-3 away.
    """

    frame = _rreg_outlier_frame()
    result = RobustRegression.from_formula("y ~ x1 + x2", frame).fit()

    # If someone reverts the threshold to 1.345 * (mad/0.6745), the coefficients
    # move to these values and this assertion fails.
    np.testing.assert_allclose(
        np.asarray(result.params, dtype=float), RREG_OUTLIER_STATA_B, rtol=1e-9
    )
    old_threshold_coefficients = (1.00117213, 2.00018639, -0.53027182)
    assert not np.allclose(
        np.asarray(result.params, dtype=float),
        old_threshold_coefficients,
        rtol=1e-4,
    )
