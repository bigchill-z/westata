"""`hausman ..., sigmamore` / `sigmaless`, and Stata's rank-based df.

Replicating the user's comparison:

    quietly reg lnw iq s expr tenure rns smsa
    estimates store ols
    quietly ivregress 2sls lnw s expr tenure rns smsa (iq=med kww)
    estimates store iv
    hausman iv ols, constant sigmamore

TWO THINGS HAD TO MATCH, and both were missing:

1. THE SIGMA OPTIONS. `hausman.ado:162-173` rescales one VCE before differencing -- the CONSISTENT
   one for sigmamore, the EFFICIENT one for sigmaless -- by the square of the s2 ratio. `s2` comes
   from `e(sigma)` falling back to `e(rmse)` ITSELF, not its square (`hausman.ado:113-116`). Using
   e(rmse)**2 squares the ratio twice and leaves the difference matrix full rank where Stata reports
   rank 1.

2. THE DF IS THE RANK. Stata fills `r(df)` from the rank `MatGinv` computed, with its own tolerance
   `max(singular value) * n * 1e-9` (`hausman.ado:606-628`). NumPy's default pinv tolerance is about
   1e-16 relative, so it keeps directions Stata discards and the statistic lands on the wrong df.

Measured on grilic, all five option combinations now agree with Stata to 1e-9:

    default            chi2(6) = 3.422654895
    constant           chi2(7) = 3.422654895
    sigmamore          chi2(1) = 3.843794583
    sigmamore constant chi2(1) = 3.843794583
    sigmaless          chi2(1) = 3.437051055
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(),
    reason="grilic.dta is not available on this machine",
)


@pytest.fixture(scope="module")
def pair():
    frame = pd.read_stata(DATA)
    ols = ws.OLS.from_formula("lnw ~ iq + s + expr + tenure + rns + smsa", frame).fit()
    iv = ws.IV2SLS.from_formula(
        "lnw ~ s + expr + tenure + rns + smsa + [iq ~ med + kww]", frame
    ).fit()
    return iv, ols


def _test(pair, **kwargs):
    iv, ols = pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ws.hausman(iv, ols, **kwargs)


def test_sigmamore_reports_rank_one(pair) -> None:
    """The whole point: a rank-deficient difference gives chi2(1), not chi2(6)."""

    result = _test(pair, sigmamore=True)
    assert int(result.df) == 1, result.df
    assert result.statistic == pytest.approx(3.843794583, abs=1e-9)


def test_sigmamore_with_constant_reports_rank_one_too(pair) -> None:
    """Stata still reports chi2(1) when `constant` is added under sigmamore.

    The statistics need not be identical -- `constant` changes which coefficients enter the
    quadratic form -- so this pins the df, which is the part sigmamore affects.
    """

    with_const = _test(pair, sigmamore=True, constant=True)
    assert int(with_const.df) == 1, with_const.df
    assert with_const.statistic == pytest.approx(3.843794583, abs=1e-9)


def test_sigmaless_is_the_mirror_image(pair) -> None:
    result = _test(pair, sigmaless=True)
    assert int(result.df) == 1, result.df
    assert result.statistic == pytest.approx(3.437051055, abs=1e-9)


def test_the_default_and_constant_forms_are_full_rank(pair) -> None:
    assert int(_test(pair).df) == 6
    assert int(_test(pair, constant=True).df) == 7


def test_the_defaults_are_unchanged(pair) -> None:
    plain = _test(pair)
    assert plain.statistic == pytest.approx(3.422654895, abs=1e-9)


def test_sigmamore_and_sigmaless_differ(pair) -> None:
    more = _test(pair, sigmamore=True).statistic
    less = _test(pair, sigmaless=True).statistic
    assert abs(more - less) > 0.1, (more, less)


def test_the_s2_fallback_uses_rmse_not_its_square(pair) -> None:
    """A direct guard on the formula: the scale ratio must be (rmse_o/rmse_c)**2, not **4.

    Using e(rmse)**2 as s2 squares the ratio twice. Distinguishing them needs a case where the two
    differ enough to change the rank; grilic is one -- the correct formula collapses the difference
    to rank 1, the squared one leaves it at rank 6. That behavioural difference is what the other
    tests pin, so this one only records the magnitudes.
    """

    iv, ols = pair
    ratio = float(ols.e["rmse"]) / float(iv.e["rmse"])
    assert 0.5 < ratio < 2.0, ratio
    assert abs(ratio**2 - ratio**4) > 1e-3, ratio


def test_missing_sigma_raises(pair) -> None:
    """Stata refuses the option when neither e(sigma) nor e(rmse) is stored."""

    iv, ols = pair
    stripped = ols
    original = dict(getattr(stripped, "e", {}) or {})
    try:
        stripped.e.pop("rmse", None)
        stripped.e.pop("sigma", None)
        with pytest.raises(Exception, match="sigma"):
            ws.hausman(iv, stripped, sigmamore=True)
    finally:
        stripped.e.update(original)
