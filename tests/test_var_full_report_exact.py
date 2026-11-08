"""VAR's entire report is byte-identical to Stata's.

C13-01 scored 0.5429. Rounds 83-93 rebuilt the information block, the equation table and the
coefficient table, and all thirty-nine lines now compare equal to a live ar y1 y2, lags(1/2).

THREE RULES EMERGED FROM MEASUREMENT, each after a plausible alternative failed:

  * coefficient digits are limited by the NUMBER's own width (eight characters), not by the rendered
    string, so a negative value prints one digit longer than a positive one of the same size;
  * RMSE uses six SIGNIFICANT digits, not six decimals -- %.6f matched only values below one;
  * the coefficient table's closing rule is a plain line, while its internal separators keep the
    pipe. Using one form for both made the table one row long or one row short depending on which.

TWO LAYOUT FACTS measured off the printed table:

  * data rows end at 25/36/45/53/66/78 with the pipe at 13, header labels at 26/37/44/53/78;
  * the equation table is 64 wide (data) and 65 (rules) with fields ending at 23/35/44/55/64.

"""
STATA_LINES = [
    '',
    'Vector autoregression',
    '',
    'Sample: 3 thru 300                              Number of obs     =        298',
    'Log likelihood =  -833.7156                     AIC               =   5.662521',
    'FPE            =   .9868603                     HQIC              =   5.712183',
    'Det(Sigma_ml)  =   .9227959                     SBIC              =   5.786585',
    '',
    'Equation           Parms      RMSE     R-sq      chi2     P>chi2',
    '----------------------------------------------------------------',
    'y1                    5     1.00254   0.0073   2.190942   0.7007',
    'y2                    5     .977199   0.0101   3.052658   0.5491',
    '----------------------------------------------------------------',
    '',
    '------------------------------------------------------------------------------',
    '             | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]',
    '-------------+----------------------------------------------------------------',
    'y1           |',
    '          y1 |',
    '         L1. |  -.0465397   .0575386    -0.81   0.419    -.1593133    .0662339',
    '         L2. |   .0318462   .0575982     0.55   0.580    -.0810442    .1447367',
    '             |',
    '          y2 |',
    '         L1. |  -.0541565   .0593849    -0.91   0.362    -.1705488    .0622357',
    '         L2. |   .0249058   .0594348     0.42   0.675    -.0915843    .1413959',
    '             |',
    '       _cons |   .0541115    .058077     0.93   0.351    -.0597174    .1679404',
    '-------------+----------------------------------------------------------------',
    'y2           |',
    '          y1 |',
    '         L1. |   .0166631   .0560841     0.30   0.766    -.0932596    .1265858',
    '         L2. |  -.0641368   .0561422    -1.14   0.253    -.1741734    .0458998',
    '             |',
    '          y2 |',
    '         L1. |   .0456425   .0578837     0.79   0.430    -.0678074    .1590924',
    '         L2. |  -.0526983   .0579323    -0.91   0.363    -.1662435     .060847',
    '             |',
    '       _cons |   .0899281   .0566089     1.59   0.112    -.0210233    .2008794',
    '------------------------------------------------------------------------------',
]


import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def rendered():
    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame({"y1": rng.normal(size=n), "y2": rng.normal(size=n)})
    return ws.VAR(frame[["y1", "y2"]], lags=2).fit().summary(style="stata")


def test_every_line_matches_stata(rendered) -> None:
    lines = rendered.splitlines()
    assert len(lines) == len(STATA_LINES)
    problems = [
        (index, theirs, ours)
        for index, (theirs, ours) in enumerate(zip(STATA_LINES, lines))
        if theirs != ours
    ]
    assert not problems, problems[:2]


def test_the_coefficient_digit_limit_is_on_the_number_body() -> None:
    """A negative coefficient keeps one digit more than a positive one of the same size."""

    from westata.estimators.tseries import _format_g

    assert _format_g(-0.0465397, 7) == "-.0465397"
    assert _format_g(0.05753861, 7) == ".0575386"
