"""params vs stata_params: the ordering trap that makes a correct fit look wrong.

C05-02 (estat vce) scores 0.08 in the report. Checking it, every number is right -- but only
after reordering. Comparing ``result.params`` against Stata's ``e(b)`` element by element puts
the intercept (4.1037) opposite ``s`` (0.1026), and the covariance diagonals come out completely
misaligned (7.24e-03 against 3.42e-05).

``stata_params`` and ``stata_cov_params`` already exist for exactly this reason, and this test
pins their correctness against a live ``e(b)`` / ``e(V)``:

    e(b)          s .10264304  expr .03811890  tenure .03561456
                  smsa .13966664  rns -.08407974  _cons 4.10367504
    stata_params  s 0.102643   expr 0.038119   tenure 0.035615
                  smsa 0.139667  rns -0.084080  _cons 4.103675

    e(V) diagonal 3.4208547461e-05 ... 7.2415007901e-03
    relative error of stata_cov_params            1.4e-11

So the 0.08 is a comparison artefact. The same trap applies to every single-equation model, which
is why the accessors exist and why the plain ones are not "wrong" -- they are statsmodels-ordered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

#: Captured from ``regress lnw s expr tenure smsa rns`` on grilic.
STATA_BETA = {
    "s": 0.10264304,
    "expr": 0.03811890,
    "tenure": 0.03561456,
    "smsa": 0.13966664,
    "rns": -0.08407974,
    "_cons": 4.10367504,
}
STATA_DIAGONAL = np.array(
    [
        3.4208547461e-05,
        4.0028347569e-05,
        5.9944303301e-05,
        7.8860679327e-04,
        8.2928351596e-04,
        7.2415007901e-03,
    ]
)


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(
            subset=["lnw", "s", "expr", "tenure", "smsa", "rns"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure + smsa + rns", frame).fit()


def test_stata_params_matches_e_b_in_order(fitted) -> None:
    ordered = fitted.stata_params
    assert list(ordered.index) == list(STATA_BETA)
    for name, value in STATA_BETA.items():
        assert float(ordered[name]) == pytest.approx(value, abs=5e-9), name


def test_stata_cov_params_matches_e_V(fitted) -> None:
    diagonal = np.diag(np.asarray(fitted.stata_cov_params))
    assert np.allclose(diagonal, STATA_DIAGONAL, rtol=1e-9)


def test_the_constant_is_last_in_stata_order(fitted) -> None:
    assert fitted.stata_params.index[-1] == "_cons"
    assert fitted.param_names[0] == "Intercept"


def test_the_plain_params_are_not_wrong_just_differently_ordered(fitted) -> None:
    """The trap: a naive element-wise comparison puts 4.1037 opposite s's 0.1026.

    Both vectors hold the same six numbers; only the order differs. A test that compares them
    positionally would report a catastrophic mismatch on a correct fit.
    """

    plain = np.sort(np.asarray(fitted.params, dtype=float))
    ordered = np.sort(np.asarray(fitted.stata_params, dtype=float))
    assert np.allclose(plain, ordered)


def test_a_positional_comparison_really_would_mislead(fitted) -> None:
    """Make the failure mode explicit rather than describing it in prose."""

    plain_first = float(np.asarray(fitted.params)[0])
    stata_first = float(np.asarray(fitted.stata_params)[0])
    assert abs(plain_first - stata_first) > 1.0
