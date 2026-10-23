"""Tests for ``nl`` and ``frontier``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.nonlinear import Frontier, Nl, compile_expression


def _decay(n: int = 300, seed: int = 103) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.1, 4.0, size=n))
    return pd.DataFrame(
        {"y": 5.0 * np.exp(-0.8 * x) + rng.normal(scale=0.15, size=n), "x": x}
    )


def _frontier(n: int = 400, seed: int = 113) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    inefficiency = np.abs(rng.normal(scale=0.4, size=n))
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * x1 + 0.5 * x2 - inefficiency + rng.normal(scale=0.2, size=n),
            "x1": x1,
            "x2": x2,
        }
    )


# ----------------------------------------------------------------------- nl


def test_nl_matches_stata():
    result = Nl.from_formula("y = {b0} * exp(-{b1} * x)", _decay()).fit()
    # Stata: /b0 4.943526 (se .0381662), /b1 .7912568 (se .0080793)
    assert float(result.params["/b0"]) == pytest.approx(4.943526, rel=1e-3)
    assert float(result.params["/b1"]) == pytest.approx(0.7912568, rel=1e-3)


def test_nl_parameter_names_carry_the_leading_slash():
    result = Nl.from_formula("y = {b0} * exp(-{b1} * x)", _decay()).fit()
    assert list(result.param_names) == ["/b0", "/b1"]


def test_nl_on_a_linear_model_reproduces_ols():
    import westata as ws

    rng = np.random.default_rng(11)
    n = 200
    x = rng.normal(size=n)
    frame = pd.DataFrame({"y": 1.5 + 2.5 * x + rng.normal(size=n), "x": x})

    nonlinear = Nl.from_formula("y = {b0} + {b1} * x", frame).fit()
    plain = ws.OLS.from_formula("y ~ x", frame).fit()

    assert float(nonlinear.params["/b0"]) == pytest.approx(
        float(plain.params["Intercept"]), rel=1e-5
    )
    assert float(nonlinear.params["/b1"]) == pytest.approx(
        float(plain.params["x"]), rel=1e-5
    )


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd')",
        "x.__class__",
        "lambda: 1",
        "[1, 2, 3]",
        "frobnicate(x)",
        "exp(x, 2)",
    ],
)
def test_nl_rejects_dangerous_or_unknown_expressions(expression):
    with pytest.raises((ValueError, ConsistencyError)):
        Nl.from_formula(f"y = {expression}", _decay())


def test_nl_requires_at_least_one_parameter():
    with pytest.raises(ConsistencyError, match="no parameters"):
        Nl.from_formula("y = x", _decay())


def test_compile_expression_rejects_a_bare_unknown_name():
    compile_ = compile_expression("a + b", ["a"], ["b"])
    with pytest.raises(ValueError, match="neither a parameter nor a column"):
        compile_({"a": 1.0}, pd.DataFrame({"x": [1.0, 2.0]}))


# ----------------------------------------------------------------- frontier


def test_frontier_matches_stata():
    result = Frontier.from_formula("y ~ x1 + x2", _frontier()).fit()
    # Stata: x1 .7813064, x2 .517943, _cons 1.047813
    assert float(result.params["x1"]) == pytest.approx(0.7813064, rel=2e-2)
    assert float(result.params["x2"]) == pytest.approx(0.517943, rel=2e-2)
    assert float(result.params["Intercept"]) == pytest.approx(1.047813, rel=2e-2)


def test_frontier_reports_the_variance_components():
    result = Frontier.from_formula("y ~ x1 + x2", _frontier()).fit()
    for key in ("sigma_u", "sigma_v", "lambda", "sigma2", "gamma"):
        assert key in result.e, key
        assert np.isfinite(float(result.e[key]))
    assert float(result.e["sigma_u"]) >= 0
    assert float(result.e["sigma_v"]) > 0


def test_frontier_rejects_an_unsupported_distribution():
    with pytest.raises(UnsupportedError, match="halfnormal"):
        Frontier.from_formula("y ~ x1 + x2", _frontier(), distribution="truncated")


def test_frontier_loglikelihood_is_finite():
    """The tail term must use logcdf; a direct Phi() call underflows to -inf."""

    result = Frontier.from_formula("y ~ x1 + x2", _frontier()).fit()
    assert np.isfinite(float(result.e["ll"]))
    assert float(result.e["ll"]) < 0.0
