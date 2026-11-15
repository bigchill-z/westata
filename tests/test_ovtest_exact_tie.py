"""A polynomial-collinear auxiliary design is flagged, because its df is order-dependent.

On a Mincer-style fit (``lnwage ~ educ exper exper2`` with ``exper2 = exper^2``) the
auxiliary regression's nine powers contain an EXACT tie: after ``{1, exper, exper^2,
exper^3}`` is in the basis, ``exper2^2`` and ``exper^4`` are the same vector -- their
elementwise ratio measures 1.000000000 with zero spread on two independent datasets.

The sweep must omit exactly one of a tied pair, so WHICH one -- and therefore what df
Stata prints -- depends on the order the columns are swept. Stata resolves that with an
internal reordering that StataCorp documents as able to differ run to run
(Statalist 2004-05/msg00057), and two datasets of identical structure were measured to
produce different Stata answers (df 7 on one, df 8 on the other).

So the result says so instead of silently differing: ``ambiguous`` is True and the printed
block carries a warning line. A plain design must NOT be flagged.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _mincer() -> pd.DataFrame:
    rng = np.random.default_rng(77)
    n = 500
    educ = rng.integers(6, 21, size=n).astype(float)
    exper = rng.integers(0, 41, size=n).astype(float)
    return pd.DataFrame(
        {
            "educ": educ,
            "exper": exper,
            "exper2": exper**2,
            "lnwage": 0.09 * educ
            + 0.04 * exper
            - 0.0006 * exper**2
            + 0.01 * rng.normal(size=n),
        }
    )


def _plain() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 400
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    frame["y"] = 1.0 + frame["a"] - 0.5 * frame["b"] + rng.normal(size=n)
    return frame


def test_a_polynomial_collinear_design_is_flagged() -> None:
    fit = ws.OLS.from_formula("lnwage ~ educ + exper + exper2", _mincer()).fit()
    result = ws.estat_ovtest(fit, rhs=True)
    assert result.ambiguous is True
    assert "warning: the auxiliary design contains an exact tie" in str(result)


def test_a_plain_design_is_not_flagged() -> None:
    fit = ws.OLS.from_formula("y ~ a + b + c", _plain()).fit()
    result = ws.estat_ovtest(fit, rhs=True)
    assert result.ambiguous is False
    assert "warning:" not in str(result)


def test_a_duplicated_regressor_also_counts_as_a_tie() -> None:
    """A duplicate is a proportional column, which is numerically the same situation.

    ``x1`` and an exact copy make "which one is omitted" a labelling choice decided by column
    order, just like the ``exper2^2`` / ``exper^4`` pair. The two cases cannot be told apart
    from the design matrix alone -- a copy has an elementwise ratio of exactly 1.0 -- so this
    deliberately reports True rather than pretending the duplicate is benign.
    """

    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "x1_copy": x1})
    frame["y"] = 1.0 + 2.0 * x1 + rng.normal(size=n)

    fit = ws.OLS.from_formula("y ~ x1 + x1_copy", frame).fit()
    result = ws.estat_ovtest(fit, rhs=True)
    assert result.ambiguous is True


def test_the_tie_is_only_among_the_powers_of_the_squared_term() -> None:
    """Dropping ``exper2`` altogether removes the tie, which pins the cause.

    Without the user's own ``exper2`` column the auxiliary design still has powers of
    ``educ`` and ``exper``, but no column is a polynomial of another regressor's square, so
    nothing is tied and the plain output returns.
    """

    frame = _mincer()
    fit = ws.OLS.from_formula("lnwage ~ educ + exper", frame).fit()
    result = ws.estat_ovtest(fit, rhs=True)
    assert result.ambiguous is False, (
        "a model with no squared regressor must not be flagged"
    )
