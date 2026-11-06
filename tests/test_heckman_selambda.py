"""heckman exposes e(selambda), matching Stata's two-step reference exactly.

Round 64 found the key missing and three attempted fixes all failed. The reason none of them
took: they patched ``_fit_ml``, which plain ``heckman`` NEVER REACHES. The default estimator is
two-step and goes through a different block in ``fit``.

The reference, from ``heckman lnw s expr tenure, select(selected = rns smsa) twostep``:

    e(lambda)   -0.17237340
    e(selambda)  0.09466295

both reproduced here to every printed digit.

The index matters: ``covariance[k_beta, k_beta]`` gave .09366475 -- a different quantity --
because k_beta counts within the outcome equation while ``params`` concatenates beta and gamma.
Locating the slot by NAME gives .09466297.

NOTE ON DEFAULTS, recorded because it surprised me: Stata's plain ``heckman`` reports
e(method) = "ml", not two-step:

    plain heckman     lambda -0.07873069   selambda 0.04644562
    heckman, twostep  lambda -0.17237340   selambda 0.09466295

Our default matches the TWOSTEP numbers. That makes the default-method question a separate open
item -- this test pins the two-step arithmetic, not the default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "smsa"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    frame = frame.copy()
    frame["selected"] = (frame["lnw"] > frame["lnw"].median()).astype(int)
    return ws.Heckman.from_formula(
        "lnw ~ s + expr + tenure + select(selected ~ rns + smsa)", frame
    ).fit()


def test_selambda_is_present(fitted) -> None:
    """It was missing entirely; a caller porting the name read None."""

    assert fitted.e.get("selambda") is not None


def test_selambda_matches_the_two_step_reference(fitted) -> None:
    assert float(fitted.e["selambda"]) == pytest.approx(0.09466295, abs=5e-7)


def test_lambda_matches_the_two_step_reference(fitted) -> None:
    assert float(fitted.e["lambda"]) == pytest.approx(-0.17237340, abs=5e-7)


def test_selambda_equals_the_lambda_standard_error(fitted) -> None:
    """selambda must be sqrt(V[lambda, lambda]), not some neighbouring diagonal."""

    names = list(fitted.param_names)
    position = names.index("lambda")
    assert float(fitted.e["selambda"]) == pytest.approx(float(fitted.bse[position]), abs=5e-7)


def test_selambda_is_not_the_k_beta_entry(fitted) -> None:
    """Pin the wrong index that produced .09366475 -- a plausible but different number."""

    assert abs(float(fitted.e["selambda"]) - 0.09366475) > 5e-4


def test_lambda_equals_rho_times_sigma(fitted) -> None:
    """The two-step definition Stata documents."""

    product = float(fitted.e["rho"]) * float(fitted.e["sigma"])
    assert float(fitted.e["lambda"]) == pytest.approx(product, abs=5e-7)
