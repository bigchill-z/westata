"""``PanelOLS`` must report coefficients in the order the formula wrote them.

Stata keeps the written order in ``e(b)`` and appends ``_cons``.  patsy, which builds
the design matrix here, alphabetises instead -- so ``y ~ ltlan + ... + t`` came back as
``[Intercept, t, ltlan, ...]`` and ``stata_param_names`` printed ``t`` FIRST where
Stata prints it LAST.

Measured on lin_1992.dta with ``xtset province t`` and::

    xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca t, fe r

Stata's row order is the nine regressors, then ``t``, then ``_cons``::

           ltlan |   .6517195   .1843858     3.53   0.001
          ltwlab |   .1431791   .0589267     2.43   0.022
           ltpow |   .0366317   .0991178     0.37   0.715
           ltfer |    .180359   .0428995     4.20   0.000
             hrs |   .1916276   .0295596     6.48   0.000
         mipric1 |   .0198772   .0515121     0.39   0.703
         giprice |   -.026268   .0226875    -1.16   0.257
             mci |   .2014685    .078794     2.56   0.016
            ngca |   .6761116    .421738     1.60   0.121
               t |   .0063068   .0106492     0.59   0.559
           _cons |    2.36174   .8262751     2.86   0.008

The cause was narrow: ``OLS`` has recorded ``_formula_rhs_order`` since the beginning,
but the panel estimators never set it, so only they fell back to patsy's order.  A
trend term is the case that exposes it -- every other regressor in this formula happens
to sort after its neighbours by accident, and ``t`` does not.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
XS = ["ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mipric1", "giprice", "mci", "ngca"]
FORMULA = "ltvfo ~ " + " + ".join(XS) + " + t"

#: Stata's coefficient rows, in its own order.
STATA_ORDER = [*XS, "t", "_cons"]
#: The statics from that run, so a reordering cannot silently swap two values.
STATA_VALUES = {
    "ltlan": 0.6517195, "ltwlab": 0.1431791, "ltpow": 0.0366317, "ltfer": 0.1803590,
    "hrs": 0.1916276, "mipric1": 0.0198772, "giprice": -0.0262680, "mci": 0.2014685,
    "ngca": 0.6761116, "t": 0.0063068, "_cons": 2.3617400,
}


@pytest.fixture(scope="module")
def fit():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")
    return ws.PanelOLS.from_formula(
        FORMULA, frame, entity="province", time="t"
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["province"]})


def test_stata_param_names_follow_the_written_order(fit) -> None:
    assert fit.stata_param_names == STATA_ORDER


def test_the_trend_is_last_among_the_slopes(fit) -> None:
    """The specific symptom: ``t`` sat first, because patsy sorts columns."""

    names = fit.stata_param_names
    assert names.index("t") == len(XS), f"t is at {names.index('t')}, expected {len(XS)}"
    assert names[-1] == "_cons"


def test_the_written_order_is_recorded_on_the_result(fit) -> None:
    """``_stata_order`` reads this, and its absence was the whole bug."""

    assert getattr(fit, "_formula_rhs_order", None) == [*XS, "t"]


def test_the_values_travel_with_their_names(fit) -> None:
    """A reordering must not shuffle the numbers -- read them through the label."""

    params = fit.stata_params
    for name, expected in STATA_VALUES.items():
        assert float(params[name]) == pytest.approx(expected, abs=5e-7), name


def test_the_rendered_table_puts_t_after_ngca(fit) -> None:
    lines = [line for line in str(fit.summary()).splitlines() if "|" in line]
    body = [line for line in lines if "0." in line or "-." in line]
    positions = {}
    for line in body:
        label = line.split("|", 1)[0].strip()
        if label in STATA_VALUES:
            positions[label] = len(positions)
    assert list(positions) == [n for n in STATA_ORDER if n in positions], positions


def test_ols_agrees_with_the_panel_estimator() -> None:
    """Both paths must use the same rule; OLS had it right all along."""

    frame = pd.read_stata(DATA, convert_categoricals=False)
    ols = ws.OLS.from_formula(FORMULA, frame).fit()
    assert ols.stata_param_names == STATA_ORDER
