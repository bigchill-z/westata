"""The stored-result macros every estimation command carries.

Round 91 closed the ``e(depvar)`` gap and learned that a base-class fix is not a
family-wide fix. Round 92 applied the same audit to the other common macros and
found that ``e(cmd)``, ``e(properties)``, ``e(title)``, ``e(vcetype)`` and
``e(cmdline)`` were absent from EVERY estimator, though
``regress.sthlp:435-441`` documents all of them.

Three of the five can be recovered honestly; two cannot, and the difference is the
point of this file:

* ``e(cmd)`` -- ``model_name`` already holds Stata's command name;
* ``e(properties)`` -- documented as the CONSTANT ``b V`` in every help file;
* ``e(title)``/``e(vcetype)``/``e(cmdline)`` -- per-command English strings with no
  derivable source.  Absent, not invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(97)
    n = 150
    latent = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": latent,
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "yb": (latent > 0).astype(int),
            "ycnt": rng.poisson(3.0, size=n),
            "z1": rng.normal(size=n),
            "firm": np.repeat(np.arange(30), 5),
            "year": np.tile(np.arange(5), 30),
        }
    )


CASES = {
    "OLS": (lambda f: ws.OLS.from_formula("y ~ x1", f).fit(), "regress"),
    "Logit": (lambda f: ws.Logit.from_formula("yb ~ x1", f).fit(), "logit"),
    "Poisson": (lambda f: ws.Poisson.from_formula("ycnt ~ x1", f).fit(), "poisson"),
    "Tobit": (lambda f: ws.Tobit.from_formula("y ~ x1", f).fit(), "tobit"),
    "IV2SLS": (
        lambda f: ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z1]", f).fit(),
        "ivregress 2sls",
    ),
    "PanelOLS": (
        lambda f: ws.PanelOLS.from_formula(
            "y ~ x1", f, entity="firm", time="year"
        ).fit(),
        "xtreg, fe",
    ),
    "QuantileRegression": (
        lambda f: ws.QuantileRegression.from_formula("y ~ x1", f).fit(),
        "qreg",
    ),
}


@pytest.mark.parametrize("label", sorted(CASES))
def test_e_cmd_names_the_stata_command(frame, label) -> None:
    """``regress.sthlp:435``: ``e(cmd)`` is the command name.

    ``model_name`` already stores exactly that, so the macro is recovered rather
    than invented.
    """

    factory, expected = CASES[label]

    assert factory(frame).e["cmd"] == expected


@pytest.mark.parametrize("label", sorted(CASES))
def test_e_properties_is_the_documented_constant(frame, label) -> None:
    """``regress.sthlp:439`` and every sibling help file: ``e(properties) = b V``.

    A constant is safe to supply; it carries no per-estimator information.
    """

    factory, _expected = CASES[label]

    assert factory(frame).e["properties"] == "b V"


@pytest.mark.parametrize("label", sorted(CASES))
def test_e_depvar_survives_the_new_fallbacks(frame, label) -> None:
    """Regression guard: adding cmd/properties must not disturb depvar."""

    factory, _expected = CASES[label]
    result = factory(frame)

    assert result.e["depvar"], label
    assert isinstance(result.e["depvar"], str), label


NON_RECOVERABLE = ("cmdline",)


@pytest.mark.parametrize("label", ["OLS", "Logit"])
@pytest.mark.parametrize("macro", NON_RECOVERABLE)
def test_underivable_macros_are_absent_not_invented(frame, label, macro) -> None:
    """``e(cmdline)`` is a per-CALL string with no field to recover it from.

    It records the command AS TYPED, options included, so it cannot be reconstructed from
    an already-fitted object.  Fabricating it would put a plausible-looking string in a
    place a caller trusts -- worse than an absent key, which at least signals "unknown".

    This test exists so a future change that starts inventing one fails.

    Two macros have been REMOVED from this list as their derivations were found, and both
    removals are deliberate rather than concessions:

    * ``title`` -- ``logit.ado:253``/``probit.ado:252`` write literals, and poisson/nbreg
      pass ``title(...)`` to ``ml``; rounds 223/241-242 transcribed them.
    * ``vcetype`` -- rounds 245: ``regress.sthlp:438`` defines it as the DISPLAY title and
      Stata's own code compares against its capitalised form (``arch.ado:1470``:
      ``if "`e(vcetype)'" == "Robust"``), while ``didregress.ado:477`` gives the two-word
      HC spelling ``"Robust HC2"``.  It is derived from ``cov_type``, not invented.
    """

    factory, _expected = CASES[label]

    assert macro not in factory(frame).e


def test_e_cmd_does_not_override_an_explicit_value(frame) -> None:
    """An estimator that sets ``e(cmd)`` itself must win over the fallback."""

    result = ws.OLS.from_formula("y ~ x1", frame).fit()
    result._e["cmd"] = "explicitly_set"

    assert result.e["cmd"] == "explicitly_set"


def test_newey_keeps_its_own_macros(frame) -> None:
    """``newey`` records all five itself (``newey.sthlp:168-180``).

    The fallbacks must not interfere with an estimator that already supplies them,
    including the two that cannot be derived elsewhere.
    """

    result = ws.newey("y ~ x1", frame, lag=1, time=frame.index.to_numpy())._results

    assert result.e["cmd"] == "newey"
    assert result.e["properties"] == "b V"
    assert result.e["title"]
    assert result.e["vcetype"]
    assert result.e["predict"] == "newey_p"


# ------------------------------------------- e(df_r) is not universal
def test_e_df_r_is_absent_for_z_based_models(frame) -> None:
    """``logit``/``poisson``/``nbreg``/``ologit`` do not document ``e(df_r)``.

    Their help files list ``e(df_m)`` only -- the tests are z-tests, so there is no
    residual degrees of freedom.  ``StataResults`` used to inject ``df_r``
    unconditionally, which put a fabricated 200.0 in a logit result's ``e()`` where
    Stata has no such key.

    ``tobit`` DOES document it, and keeps it.
    """

    assert "df_r" not in ws.Logit.from_formula("yb ~ x1", frame).fit().e
    assert "df_r" not in ws.Poisson.from_formula("ycnt ~ x1", frame).fit().e
    assert "df_r" not in ws.NegativeBinomial.from_formula("ycnt ~ x1", frame).fit().e


def test_t_based_models_keep_e_df_r(frame) -> None:
    """``tobit.sthlp`` documents ``e(df_r)``, and OLS certainly needs it."""

    assert "df_r" in ws.OLS.from_formula("y ~ x1", frame).fit().e
    assert "df_r" in ws.Tobit.from_formula("y ~ x1", frame).fit().e


def test_df_r_still_agrees_with_n_minus_rank_where_present(frame) -> None:
    """Where the key exists it must still satisfy ``N - df_r == rank``."""

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    e = result.e

    assert e["N"] - e["df_r"] == e["rank"]


def test_the_internal_df_resid_attribute_survives(frame) -> None:
    """Removing the macro must not remove the statsmodels-facing attribute.

    ``df_resid`` is part of the statsmodels surface and is used by the t and F
    reference distributions; only the Stata ``e()`` key is conditional.
    """

    result = ws.Logit.from_formula("yb ~ x1", frame).fit()

    assert "df_r" not in result.e
    assert np.isfinite(float(result.df_resid))
