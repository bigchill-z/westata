"""``regress`` with factor variables under ``vce(cluster)``: Stata's missing F.

``reg ltvfo ltlan ... i.province, vce(cluster province)`` on lin_1992.dta prints

    Linear regression                               Number of obs     =        476
                                                    F(8, 27)          =          .
                                                    Prob > F          =          .
                                                    R-squared         =     0.9642
                                                    Root MSE          =     .10589

Three separate things are unusual here and each was reproduced only after being
measured, because all three are silent: the F is MISSING, its numerator df is 8
rather than the 36 model columns, and the ``Prob > F`` row still appears.

Measured on StataNow 19.5, 2026-11-14, all with ``vce(cluster province)`` and
``i.province`` (27 dummies):

    non-factor slopes   e(df_m)   e(rank)   printed
    9                   8         9         F(8, 27)
    5                   4         5         F(4, 27)
    2                   1         2         F(1, 27)
    1                   0         1         F(0, 27)

So ``e(df_m)`` = non-factor slopes - 1, ``e(rank)`` = non-factor slopes: the "free
parameters of the clustered problem" convention, not a column count.  The missing F
appears whenever BOTH conditions hold:

    reg ... i.province, vce(cluster prov)   F(8, 27)  = .        both
    reg ... province,   vce(cluster prov)   F(10, 27) = 97.44    no factor
    reg ... i.province                      F(36,439) = 328.23   no cluster

What this package did before: ``e(F)`` was correctly NaN, but the renderer rebuilt an
F from R^2 and printed ``F(36, 27) = 20.19`` -- a number Stata never shows, under a
degree-of-freedom pair that also was not Stata's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
Y = "ltvfo"
XS = ["ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mipric1", "giprice", "mci", "ngca"]

#: Slopes and their clustered SEs from ``reg ... i.province, vce(cluster province)``.
STATA_SLOPES = {
    "ltlan": (0.6370234, 0.1732267),
    "ltwlab": (0.1387786, 0.0643506),
    "ltpow": (0.0577152, 0.0778457),
    "ltfer": (0.1826281, 0.0449126),
    "hrs": (0.2134022, 0.0402952),
    "mipric1": (0.0543577, 0.0608214),
    "giprice": (-0.0151451, 0.0253419),
    "mci": (0.1943697, 0.0793856),
    "ngca": (0.7562031, 0.3937018),
}


def _frame() -> pd.DataFrame:
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")
    return frame.dropna(subset=[Y, *XS]).reset_index(drop=True)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _frame()


def _fit(frame: pd.DataFrame, formula_tail: str, cov: str | None = "cluster"):
    kw = {"groups": frame["province"]} if cov == "cluster" else None
    formula = f"{Y} ~ " + " + ".join(XS) + formula_tail
    return ws.OLS.from_formula(formula, frame).fit(cov_type=cov, cov_kwds=kw)


def test_lsdv_slopes_and_clustered_errors_match_stata(frame) -> None:
    """The point estimates must not move when the dummies are modelled explicitly.

    They are the SAME coefficients as ``xtreg, fe`` here (balanced panel), which is a
    useful cross-check on both paths -- only the standard errors differ.
    """

    fit = _fit(frame, " + C(province)")
    for name, (coef, se) in STATA_SLOPES.items():
        assert float(fit.params[name]) == pytest.approx(coef, abs=5e-7), name
        assert float(fit.bse[name]) == pytest.approx(se, abs=5e-7), name
    assert len(fit.params) == 37          # 9 slopes + 27 dummies + intercept
    assert fit.nobs == 476


def test_the_first_dummy_matches_stata(frame) -> None:
    """Stata prints ``province / beijing`` as the base-vs-level contrast."""

    fit = _fit(frame, " + C(province)")
    assert float(fit.params["C(province)[T.2]"]) == pytest.approx(-0.1816498, abs=5e-7)
    assert float(fit.bse["C(province)[T.2]"]) == pytest.approx(0.1247829, abs=5e-7)


def test_f_is_published_as_missing(frame) -> None:
    """``e(F)`` is ``.`` for a clustered factor model, not a recomputed ratio."""

    fit = _fit(frame, " + C(province)")
    assert "F" in fit.e, "the key must exist and hold NaN, not be absent"
    assert not np.isfinite(fit.e["F"])


def test_the_header_prints_statas_missing_f_and_its_df(frame) -> None:
    rendered = str(_fit(frame, " + C(province)").summary())
    assert "F(8, 27)" in rendered, rendered[:400]
    assert "=          ." in rendered
    assert "Prob > F          =          ." in rendered
    # ...and NOT the reconstructed value the renderer used to invent.
    assert "F(36, 27)" not in rendered
    assert "20.19" not in rendered


def test_a_clustered_model_without_factors_keeps_its_f(frame) -> None:
    """The guard on the fix: no factor columns means ``e(F)`` is real and prints."""

    fit = _fit(frame, "")
    assert np.isfinite(fit.e["F"])
    rendered = str(fit.summary())
    assert "F(9, 27)" in rendered
    assert "81.39" in rendered
    assert "Prob > F          =          ." not in rendered


def test_the_unclustered_factor_model_reports_a_full_f(frame) -> None:
    """Without clustering Stata does test every column: ``F(36, 439) = 328.23``."""

    fit = _fit(frame, " + C(province)", cov="nonrobust")
    assert np.isfinite(fit.e["F"])
    assert int(fit.e["df_m"]) == 36
    assert float(fit.e["F"]) == pytest.approx(328.22853, abs=5e-4)


def test_missing_e_f_is_distinguished_from_absent_e_f() -> None:
    """The renderer's rule depends on the difference, so pin the helper itself."""

    from westata.results.summary import _e_float_is_missing

    class _Result:
        def __init__(self, e):
            self.e = e

    assert _e_float_is_missing(_Result({"F": float("nan")}), "F") is True
    assert _e_float_is_missing(_Result({}), "F") is False
    assert _e_float_is_missing(_Result({"F": 3.0}), "F") is False
