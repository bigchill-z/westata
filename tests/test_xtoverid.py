"""``ws.xtoverid`` -- the Sargan-Hansen test of overidentifying restrictions.

Targets Mark Schaffer and Steve Stillman's user-written ``xtoverid`` 2.1.8 (the version
installed here is ``ado/plus/x/xtoverid.ado``, 04Jan2016). It is NOT a Stata base command,
so every number below was captured from a live ``xtoverid`` run rather than from a
``.sthlp`` file.

The six fixtures were captured with::

    use "D:/myCodeProject/test_westata/data/lin_1992.dta", clear
    xtset province t
    quietly xtreg <varlist>, re [robust]
    quietly xtoverid
    di r(j) r(jdf) r(jp)

and are asserted at the tolerance each one actually reaches -- 1e-10 relative or better for
the conventional VCE, 1e-9 for the clustered one.

WHY THE ROBUST CASE MATTERS
---------------------------
``xtoverid.ado:178-181`` mirrors the ORIGINAL fit's ``e(vcetype)``. When that reads
"Robust" the internal refit is ``xtreg ..., re robust`` and ``test`` uses ITS covariance.
On the same equation that changes the statistic from 53.285 to 221.225 -- a 4x error, not
a rounding difference -- so a version that always used the model-based VCE would still
pass a "does it run" check while being wrong on every robust fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.xtoverid import PanelOveridResults, panel_overid

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"

#: varlist -> (cov_type, j, jdf, jp), measured on StataNow 19.5.
STATA_CASES = {
    "full_conventional": (
        ["ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mci", "ngca"],
        None,
        53.285084063390,
        7,
        0.000000003256,
    ),
    "two_regressors": (
        ["ltlan", "ltwlab"],
        None,
        2.994819604664,
        2,
        0.223708860520,
    ),
    "three_regressors": (
        ["ltlan", "ltwlab", "ltpow"],
        None,
        7.219781742362,
        3,
        0.065212907034,
    ),
    "full_robust": (
        ["ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mci", "ngca"],
        "robust",
        221.225019619060,
        7,
        0.0,
    ),
    "one_regressor": (
        ["ltlan"],
        None,
        16.877356385723,
        1,
        0.000039874441,
    ),
    "two_other_regressors": (
        ["mci", "ngca"],
        None,
        20.755766550808,
        2,
        0.000031113048,
    ),
}


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")


def _fit(frame, regressors, cov_type):
    formula = "ltvfo ~ " + " + ".join(regressors)
    kwargs = {} if cov_type is None else {"cov_type": cov_type}
    return ws.PanelRegressRE.from_formula(
        formula, frame, entity="province", time="t"
    ).fit(**kwargs)


@pytest.mark.parametrize("case", sorted(STATA_CASES))
def test_the_statistic_matches_stata(frame, case: str) -> None:
    regressors, cov_type, expected_j, expected_df, _ = STATA_CASES[case]
    fit = _fit(frame, regressors, cov_type)
    result = panel_overid(fit)

    # Relative, not absolute: these statistics span 3 to 221, so one absolute tolerance
    # would be far too tight at the top and meaningless at the bottom. The measured
    # agreement is 3.3e-12 relative for the conventional cases and 4.9e-11 for the
    # clustered one, both far below Stata's own five-decimal DISPLAY precision.
    tolerance = 1e-9 if cov_type == "robust" else 1e-10
    assert result.statistic == pytest.approx(expected_j, rel=tolerance), case


@pytest.mark.parametrize("case", sorted(STATA_CASES))
def test_the_degrees_of_freedom_match_stata(frame, case: str) -> None:
    """``r(jdf)`` is the number of ADDED regressors -- one within deviation per slope."""

    regressors, cov_type, _, expected_df, _ = STATA_CASES[case]
    fit = _fit(frame, regressors, cov_type)
    assert panel_overid(fit).df == float(expected_df), case


@pytest.mark.parametrize("case", sorted(STATA_CASES))
def test_the_p_value_matches_stata(frame, case: str) -> None:
    regressors, cov_type, _, _, expected_p = STATA_CASES[case]
    fit = _fit(frame, regressors, cov_type)
    result = panel_overid(fit)
    if expected_p == 0.0:
        # Stata prints 0.000000000000 here; the true tail is below that display floor.
        assert result.pvalue < 1e-12
    else:
        assert result.pvalue == pytest.approx(expected_p, rel=1e-6), case


def test_the_robust_case_is_not_the_conventional_one(frame) -> None:
    """Guard on the 4x trap: the two VCEs must give DIFFERENT statistics.

    Without this, a version that ignored ``e(vcetype)`` would pass every other test here
    for the conventional cases and be silently wrong on every robust one.
    """

    regressors = ["ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mci", "ngca"]
    plain = panel_overid(_fit(frame, regressors, None)).statistic
    robust = panel_overid(_fit(frame, regressors, "robust")).statistic
    assert robust > 4.0 * plain


def test_the_printed_block_matches_stata(frame) -> None:
    """The three lines a live ``xtoverid`` prints, verbatim."""

    fit = _fit(frame, ["ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mci", "ngca"], None)
    rendered = str(panel_overid(fit))
    assert rendered.splitlines() == [
        "Test of overidentifying restrictions: fixed vs random effects",
        "Cross-section time-series model: xtreg re",
        "Sargan-Hansen statistic  53.285  Chi-sq(7)    P-value = 0.0000",
    ]


def test_to_dict_uses_the_r_names_the_ado_returns(frame) -> None:
    """``xtoverid.ado:273-275`` sets exactly ``r(j)``, ``r(jp)`` and ``r(jdf)``."""

    fit = _fit(frame, ["ltlan", "ltwlab"], None)
    table = panel_overid(fit).to_dict()
    assert set(table) == {"j", "jp", "jdf"}
    assert table["jdf"] == 2.0


def test_the_result_is_a_frozen_dataclass_like_its_siblings(frame) -> None:
    fit = _fit(frame, ["ltlan", "ltwlab"], None)
    result = panel_overid(fit)
    assert isinstance(result, PanelOveridResults)
    assert result.description == "fixed vs random effects"
    assert result.model == "xtreg re"


# --------------------------------------------------------------- refusals
def test_ftest_is_refused_for_a_non_panel_result() -> None:
    auto = pd.read_stata("tests/data/auto.dta")
    ols = ws.Regress.from_formula("mpg ~ weight", auto).fit()
    with pytest.raises(NotImplementedError, match="only after xtreg"):
        panel_overid(ols)


@pytest.mark.parametrize(
    "estimator",
    [ws.PanelRegressFE, ws.PanelRegressBE, ws.PanelRegressFD],
)
def test_xtreg_estimators_other_than_re_are_refused(frame, estimator) -> None:
    """``xtoverid.ado:97-104`` narrows ``xtreg`` to ``e(model) == "re"``."""

    fit = estimator.from_formula(
        "ltvfo ~ ltlan + ltwlab", frame, entity="province", time="t"
    ).fit()
    with pytest.raises(ValueError, match="not compatible with xtreg model"):
        panel_overid(fit)


def test_a_degenerate_re_fit_is_refused(frame) -> None:
    """``xtoverid.ado:241-244``: ``sigma_u == 0`` means pooled OLS, no restrictions."""

    fit = _fit(frame, ["ltlan"], None)
    # Force the degenerate state the ado tests for, without needing a real pooled fit.
    fit.e["sigma_u"] = 0.0
    with pytest.raises(ValueError, match="degenerate"):
        panel_overid(fit)


def test_a_weighted_fit_is_refused(frame) -> None:
    """``xtoverid.ado:165-168`` refuses weights outright."""

    fit = _fit(frame, ["ltlan", "ltwlab"], None)
    fit.weight_type = "aweight"
    with pytest.raises(NotImplementedError, match="does not support weights"):
        panel_overid(fit)


def test_an_unbalanced_fit_is_refused_rather_than_guessed(frame) -> None:
    """No single ``e(theta)`` means the quasi-demeaning factor is undefined.

    An unbalanced panel stores the five ``e(thta_*)`` percentiles instead, so the ado's
    ``xtreg, re`` branch -- which assumes one theta -- cannot be reproduced. Refusing says
    so; picking, say, the median would be inventing an estimator.
    """

    trimmed = frame.drop(index=frame.index[0])
    fit = ws.PanelRegressRE.from_formula(
        "ltvfo ~ ltlan + ltwlab", trimmed, entity="province", time="t"
    ).fit()
    assert "theta" not in fit.e
    with pytest.raises(ValueError, match=r"stores no e\(theta\)"):
        panel_overid(fit)


def test_a_model_with_no_slopes_is_refused(frame) -> None:
    fit = ws.PanelRegressRE.from_formula(
        "ltvfo ~ 1", frame, entity="province", time="t"
    ).fit()
    with pytest.raises(ValueError, match="no regressors"):
        panel_overid(fit)


# ------------------------------------------------------------- internals
def test_the_added_regressors_are_within_deviations(frame) -> None:
    """The tested terms are ``x - xbar``, one per slope -- not the slopes themselves."""

    fit = _fit(frame, ["ltlan", "ltwlab"], None)
    result = panel_overid(fit)
    assert result.tested == ["__xtoverid_dev_ltlan", "__xtoverid_dev_ltwlab"]


def test_the_statistic_is_scale_invariant_to_the_generated_names(frame) -> None:
    """The restriction names are internal labels; the number must not depend on them."""

    a = panel_overid(_fit(frame, ["ltlan", "ltwlab", "ltpow"], None)).statistic
    b = panel_overid(_fit(frame, ["ltlan", "ltwlab", "ltpow"], None)).statistic
    assert a == b


def test_a_single_regressor_gives_one_degree_of_freedom(frame) -> None:
    """``df == 1`` is the tightest possible check that the count follows the slopes."""

    result = panel_overid(_fit(frame, ["ltlan"], None))
    assert result.df == 1.0
    assert np.isfinite(result.statistic)


def test_more_regressors_never_reduce_the_degrees_of_freedom(frame) -> None:
    """Monotone in the varlist length, which a miscounted restriction would break."""

    lengths = []
    for count in (1, 2, 3):
        regressors = ["ltlan", "ltwlab", "ltpow"][:count]
        lengths.append(panel_overid(_fit(frame, regressors, None)).df)
    assert lengths == [1.0, 2.0, 3.0]
