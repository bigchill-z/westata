"""``xtreg ..., mle``: the two e() blocks Stata fills and this package left empty.

RandomEffectsMLE already matched Stata on the coefficients, the standard errors and
the two variance components. What it did NOT expose were the ingredients of the two
lines Stata prints around the table:

    Random effects u_i ~ Gaussian                        Obs per group:
                                                                      min =     17
                                                                      avg =   17.0
                                                                      max =     17
    ...
    LR test of sigma_u=0: chibar2(01) = 464.22             Prob >= chibar2 = 0.0000

Measured with ``ereturn list`` on ``xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mci
ngca, mle`` (lin_1992.dta, ``xtset province t``)::

    e(ll_0)   = -147.6022868749637     constant-only fit (the header's LR chi2)
    e(ll)     =  332.897394354281
    e(ll_c)   =  100.7850378486044     POOLED fit  <- was stored under `ll_pooled`
    e(chi2_c) =  464.2247130113532     2*(e(ll) - e(ll_c))  <- did not exist
    e(g_min)  = 17   e(g_avg) = 17   e(g_max) = 17           <- did not exist

So the table rendered ``min = .`` / ``avg = .`` / ``max = .`` and dropped the
sigma_u=0 line entirely. Both are visible to a reader, and both are cheap to fill
because the numbers were already computed here -- just under private key names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
XS = ["ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mci", "ngca"]
FORMULA = "ltvfo ~ " + " + ".join(XS)

STATA = {
    "ll_0": -147.6022868749637,
    "ll": 332.897394354281,
    "ll_c": 100.7850378486044,
    "chi2_c": 464.2247130113532,
    "sigma_u": 0.2141093601432542,
    "sigma_e": 0.1061020869379882,
    "rho": 0.8028448335959572,
}
STATA_COEF = {
    "ltlan": 0.5643288, "ltwlab": 0.1408974, "ltpow": 0.0717013,
    "ltfer": 0.1772027, "hrs": 0.2153264, "mci": 0.4064832,
    "ngca": 0.7149811, "_cons": 2.490512,
}


@pytest.fixture(scope="module")
def fit():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")
    frame = frame.dropna(subset=["ltvfo", *XS]).reset_index(drop=True)
    return ws.RandomEffectsMLE.from_formula(
        FORMULA, frame, entity="province", time="t"
    ).fit()


def test_coefficients_match_stata(fit) -> None:
    for name, expected in STATA_COEF.items():
        assert float(fit.params[name]) == pytest.approx(expected, abs=5e-6), name


@pytest.mark.parametrize(
    "key, tolerance",
    [
        # ``ll_0`` comes from a separate constant-only Powell fit, so it carries that
        # run's convergence error too (measured 5.2e-07 from Stata).  ``ll_c`` is a
        # closed-form OLS likelihood here, which is why it agrees to 1.4e-14.
        ("ll_0", 1e-6),
        ("ll", 1e-6),
        ("ll_c", 1e-9),
        ("chi2_c", 1e-5),
        # The variance components come out of a Powell run, and two optimisers that
        # agree on the likelihood to 5.7e-09 can still differ by 2e-05 RELATIVE on
        # sigma_u (measured: 0.21410526 against Stata's 0.21410936).  That is the
        # optimiser's convergence path, not a formula error -- pinning it to 1e-6
        # would be asserting that two different optimisers stop at the same place.
        ("sigma_u", 1e-5),
        ("sigma_e", 1e-6),
        ("rho", 1e-5),
    ],
)
def test_the_likelihood_scalars_match_stata(fit, key, tolerance) -> None:
    assert fit.e[key] == pytest.approx(STATA[key], abs=tolerance), key


def test_chi2_c_is_the_ratio_of_the_two_likelihoods(fit) -> None:
    """``e(chi2_c)`` must be 2*(ll - ll_c), not an independent estimate."""

    expected = 2.0 * (fit.e["ll"] - fit.e["ll_c"])
    assert fit.e["chi2_c"] == pytest.approx(expected, rel=1e-12)


def test_the_pooled_likelihood_keeps_its_private_spelling_too(fit) -> None:
    """``ll_pooled`` is read internally; ``ll_c`` is Stata's contract. Both stay."""

    assert fit.e["ll_pooled"] == pytest.approx(fit.e["ll_c"], rel=1e-12)


def test_group_sizes_are_exposed(fit) -> None:
    """``Obs per group: min/avg/max`` was three bare dots without these."""

    assert fit.e["g_min"] == 17.0
    assert fit.e["g_avg"] == pytest.approx(17.0)
    assert fit.e["g_max"] == 17.0
    assert fit.e["g_min"] <= fit.e["g_avg"] <= fit.e["g_max"]


def test_the_table_prints_statas_obs_per_group(fit) -> None:
    rendered = str(fit.summary())
    assert "min =     17" in rendered
    assert "avg =   17.0" in rendered
    assert "max =     17" in rendered
    # ...and NOT the dotted placeholder that stood there before.
    assert "min =      ." not in rendered


def test_the_table_prints_the_sigma_u_lr_test(fit) -> None:
    rendered = str(fit.summary())
    assert "LR test of sigma_u=0" in rendered
    assert "chibar2(01)" in rendered
    assert "464.22" in rendered


def test_unbalanced_panels_report_their_own_min_and_max() -> None:
    """The three cells must follow the DATA, not repeat one number."""

    rng = np.random.default_rng(4)
    rows = []
    for unit in range(6):
        length = 4 + (unit % 3)          # 4, 5, 6, 4, 5, 6
        for period in range(1, length + 1):
            rows.append(
                {"id": unit, "t": period, "y": rng.normal(), "x": rng.normal()}
            )
    frame = pd.DataFrame(rows)
    result = ws.RandomEffectsMLE.from_formula(
        "y ~ x", frame, entity="id", time="t"
    ).fit()
    assert result.e["g_min"] == 4.0
    assert result.e["g_max"] == 6.0
    assert result.e["g_avg"] == pytest.approx(np.mean([4, 5, 6, 4, 5, 6]))
