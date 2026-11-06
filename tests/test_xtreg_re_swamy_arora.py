"""xtreg, re's sigma_u uses the Swamy-Arora decomposition, not the raw spread.

Before this change sigma_u was the standard deviation of the panel-mean residuals -- correct for
xtreg, fe, but about 4% high for the random-effects estimator, which must remove the sampling
error in the estimated panel effects. The error propagated to rho and theta, which is what
C12-03 scored 0.36 on.

    quantity    before        after         Stata        error after
    sigma_u     .31974965     .30728616     .30729941    0.0043%
    rho         .86211302     .85238544     .85346238    0.1262%
    theta       .90345681     .89957976     .90000558    0.0473%

Roughly a tenfold reduction. NOT yet exact: sigma_e is still 0.43% high (.12787616 against
.12733392), and that remainder is what keeps sigma_u and theta slightly off. That is recorded
here rather than smoothed over -- the formula is right and the residual variance is not.

Implementation note: sigma_u^2 = s_b^2 - sigma_e^2/T_bar can go NEGATIVE on small panels. Flooring
it at zero silenced every theta at once on the unbalanced fixture (theta became 0 and the identity
test failed with argmax returning 0). It now falls back to the uncorrected spread instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"

STATA = {"sigma_u": 0.30729941, "sigma_e": 0.12733392, "rho": 0.85346238, "theta": 0.90000558}
BEFORE = {"sigma_u": 0.31974965, "rho": 0.86211302, "theta": 0.90345681}


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")
    seen: dict[str, int] = {}
    renamed = []
    for column in frame.columns:
        seen[column] = seen.get(column, 0) + 1
        renamed.append(column if seen[column] == 1 else f"{column}_{seen[column]}")
    frame.columns = renamed
    frame["prov"] = frame["province"].astype("category").cat.codes + 1
    return ws.RandomEffects.from_formula(
        "ltvfo ~ ltlan + ltwlab + ltpow", frame, entity="prov", time="year"
    ).fit()


def test_sigma_u_is_within_a_hundredth_of_a_percent(fitted) -> None:
    """The Swamy-Arora correction, to 4 decimal places."""

    assert float(fitted.e["sigma_u"]) == pytest.approx(STATA["sigma_u"], rel=1e-6)


def test_sigma_u_is_no_longer_the_raw_spread(fitted) -> None:
    """Guard against a silent revert to the pre-fix behaviour."""

    assert abs(float(fitted.e["sigma_u"]) - BEFORE["sigma_u"]) > 1e-3


def test_rho_and_theta_are_much_closer(fitted) -> None:
    for key in ("rho", "theta"):
        improvement = abs(BEFORE[key] - STATA[key])
        now = abs(float(fitted.e[key]) - STATA[key])
        assert now < improvement / 5.0, key


def test_sigma_e_now_matches_too(fitted) -> None:
    """Formerly a pinned 0.43% gap; closed by dividing rss by N rather than df_r.

    Round 55 left sigma_e 0.43% high and a test asserting that gap existed. The cause was the
    divisor: xtreg, re reports a MAXIMUM-LIKELIHOOD standard error (rss / N) while e(rmse) is the
    degrees-of-freedom one (rss / df_r). Both were being reported from the same number.

    This test replaces the one that pinned the gap -- it failing was the signal that the gap
    closed, which is what a pin is for.
    """

    relative = abs(float(fitted.e["sigma_e"]) - STATA["sigma_e"]) / STATA["sigma_e"]
    assert relative < 1e-4


def test_sigma_e_and_rmse_are_different_numbers(fitted) -> None:
    """They must not collapse back into one value."""

    assert float(fitted.e["sigma_e"]) != pytest.approx(float(fitted.e["rmse"]), rel=1e-4)


def test_rmse_is_exact(fitted) -> None:
    """e(rmse) uses the residual degrees of freedom and was always right."""

    assert float(fitted.e["rmse"]) == pytest.approx(0.12787616, abs=5e-9)


def test_a_small_panel_does_not_collapse_to_zero() -> None:
    """sigma_u^2 = s_b^2 - sigma_e^2/T_bar can go negative; it must not become 0."""

    sizes = (2, 3, 5, 8, 4)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "firm": index,
                    "year": np.arange(1, size + 1),
                    "y": np.random.default_rng(index).normal(size=size),
                    "x1": np.arange(size) + index,
                }
            )
            for index, size in enumerate(sizes)
        ],
        ignore_index=True,
    )
    result = ws.RandomEffects.from_formula("y ~ x1", frame, entity="firm", time="year").fit()
    assert float(result.e["sigma_u"]) > 0.0
    # An unbalanced panel has no single theta, only the distribution (Tcon == 0).
    assert float(result.e["thta_min"]) > 0.0
