"""`hausman FE RE, constant sigmamore` after `xtreg, fe` + `xtreg, re`.

The user compared the two-estimator case against Stata and got different numbers. The cause is in
`hausman.ado:103-117`, which picks `s2` per ESTIMATOR, not per command:

    if "`e(cmd)'" == "xtreg" {
        if "`e(model)'" == "fe" || "`e(model)'" == "ml" {
            scalar `s2_`i'' = e(sigma_e)
        }
        else {
            scalar `s2_`i'' = e(rmse)
        }
    }
    else {
        scalar `s2_`i'' = e(sigma)
        if `s2_`i'' == . scalar `s2_`i'' = e(rmse)
    }

So for `xtreg fe` + `xtreg re`, FE contributes `e(sigma_e)` and RE contributes `e(rmse)` -- two
different numbers. westata read `e(sigma)` first for BOTH, and separately stored the whole
"xtreg, fe" string in `e(cmd)` with no `e(model)` at all, so the branch could not be taken.

Measured on lin_1992.dta (28 provinces x 18 years, 504 observations):

    e(sigma_e)   e(rmse)
    FE  .10624809   .10624809     <- fe uses sigma_e
    RE  .10624809   .11137819     <- re uses rmse, 4.8% larger

Result: chi2 went from 68.15 to **48.49**, and every `sqrt(diag(V_b-V_B))` cell now matches Stata.
Verified byte-identical: 23 lines, zero differences.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

LIN = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
FORMULA = "ltvfo ~ ltlan + ltwlab + ltpow + ltfer + hrs + mci + ngca"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(LIN).exists(), reason="lin_1992.dta is not available"
)


@pytest.fixture(scope="module")
def panel():
    frame = pd.read_stata(LIN).copy()
    frame["t"] = frame["t"].astype(int)
    return frame


@pytest.fixture(scope="module")
def pair(panel):
    fe = ws.PanelOLS.from_formula(
        FORMULA, panel, entity="province", time="t"
    ).fit()
    re_fit = ws.RandomEffects.from_formula(
        FORMULA, panel, entity="province", time="t"
    ).fit()
    return fe, re_fit


# ---------------------------------------------------------------- the e() contract
def test_both_fits_record_cmd_as_xtreg(pair) -> None:
    """Stata: `e(cmd)` is "xtreg" for every xtreg estimator; the option goes to e(model)."""

    fe, re_fit = pair
    assert (fe.e or {}).get("cmd") == "xtreg"
    assert (re_fit.e or {}).get("cmd") == "xtreg"


def test_the_estimator_is_recorded_in_e_model(pair) -> None:
    fe, re_fit = pair
    assert (fe.e or {}).get("model") == "fe"
    assert (re_fit.e or {}).get("model") == "re"


# ---------------------------------------------------------------- the sigma sources
def test_fixed_effects_and_random_effects_use_different_scalars(pair) -> None:
    """`fe` takes e(sigma_e); `re` takes e(rmse). They differ by 4.8% on this data."""

    fe, re_fit = pair
    fe_e = fe.e or {}
    re_e = re_fit.e or {}
    assert abs(float(fe_e["sigma_e"]) - float(fe_e["rmse"])) < 1e-12
    assert abs(float(re_e["sigma_e"]) - float(re_e["rmse"])) > 1e-3


# ---------------------------------------------------------------- the statistic
def test_the_statistic_matches_stata(pair) -> None:
    fe, re_fit = pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ws.hausman(fe, re_fit, constant=True, sigmamore=True)
    assert round(result.statistic, 2) == 48.49, result.statistic
    assert int(result.df) == 8


def test_the_standard_errors_match_stata(pair) -> None:
    """The `sqrt(diag(V_b-V_B))` column, measured off a live `hausman FE RE, constant sigmamore`."""

    expected = [
        0.0476709,
        0.0125808,
        0.0081232,
        0.0078425,
        0.0052769,
        0.0583709,
        0.0828671,
        0.2078242,
    ]
    fe, re_fit = pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ws.hausman(fe, re_fit, constant=True, sigmamore=True)
    rendered = str(result)
    for value in expected:
        assert f"{value:.7f}".lstrip("0").lstrip("-")[:8] in rendered.replace("0.", "."), (
            value,
            rendered,
        )


def test_the_coefficients_are_unaffected(pair) -> None:
    """sigmamore rescales a VCE; it must not move (b) or (B)."""

    fe, re_fit = pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = ws.hausman(fe, re_fit, constant=True)
        scaled = ws.hausman(fe, re_fit, constant=True, sigmamore=True)
    assert list(plain.consistent) == pytest.approx(list(scaled.consistent))
    assert list(plain.efficient) == pytest.approx(list(scaled.efficient))


def test_sigmamore_changes_the_statistic_but_not_the_sign(pair) -> None:
    fe, re_fit = pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = ws.hausman(fe, re_fit, constant=True)
        scaled = ws.hausman(fe, re_fit, constant=True, sigmamore=True)
    assert scaled.statistic != plain.statistic
    assert scaled.statistic > 0 and plain.statistic > 0
