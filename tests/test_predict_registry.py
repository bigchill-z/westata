"""Every REGISTERED predict type must actually run.

Round 74 found ``ue`` registered for ``xtreg`` while always raising: a type the
package advertises and then refuses is worse than one it never mentions. A full
audit of the registry (24 models, ~90 types) found four more entries of that kind,
plus two genuine defects it exposed:

* ``xtreg, fd`` -- the retained design had two columns against one coefficient
  (differencing drops the intercept), so every type failed with a matmul error;
* ``nbreg`` -- ``predict, n`` is nbreg's DEFAULT type
  (``nbreg_postestimation.sthlp:91``) but was unreachable because the parameter
  count (3, including ``/lnalpha``) differed from the design columns (2).

The tests here iterate the registry itself rather than a hand-written list, so a
newly registered type is covered automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.predict import SUPPORTED_TYPES, supported_types


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(11)
    n = 150
    groups, periods = 30, 5
    firm = np.repeat(np.arange(groups), periods)
    year = np.tile(np.arange(periods), groups)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.6 * x1 - 0.4 * x2 + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm": firm,
            "year": year,
            "yb": (y > y.mean()).astype(int),
            "ycnt": rng.poisson(3.0, size=n),
            "rep": rng.integers(1, 4, size=n).astype(int),
            "z1": rng.normal(size=n),
            "t": np.arange(n),
        }
    )


def _fit(model_name: str, frame):
    """One fitted model per registered name; ``None`` when it needs setup we skip."""

    builders = {
        "regress": lambda: ws.OLS.from_formula("y ~ x1 + x2", frame).fit(),
        "ivregress 2sls": lambda: ws.IV2SLS.from_formula(
            "y ~ x1 + [x2 ~ z1]", frame
        ).fit(),
        "ivregress liml": lambda: ws.IVLIML.from_formula(
            "y ~ x1 + [x2 ~ z1]", frame
        ).fit(),
        "ivregress gmm": lambda: ws.IVGMM.from_formula(
            "y ~ x1 + [x2 ~ z1]", frame
        ).fit(),
        "logit": lambda: ws.Logit.from_formula("yb ~ x1 + x2", frame).fit(),
        "probit": lambda: ws.Probit.from_formula("yb ~ x1 + x2", frame).fit(),
        "ologit": lambda: ws.OrderedLogit.from_formula("rep ~ x1", frame).fit(),
        "oprobit": lambda: ws.OrderedProbit.from_formula("rep ~ x1", frame).fit(),
        "poisson": lambda: ws.Poisson.from_formula("ycnt ~ x1", frame).fit(),
        "nbreg": lambda: ws.NegativeBinomial.from_formula("ycnt ~ x1", frame).fit(),
        "tobit": lambda: ws.Tobit.from_formula("y ~ x1 + x2", frame).fit(),
        "zip": lambda: ws.ZeroInflatedPoisson.from_formula(
            "ycnt ~ x1", frame, inflate="x2"
        ).fit(),
        "newey": lambda: ws.newey(
            "y ~ x1 + x2", frame, lag=1, time=frame["t"]
        )._results,
        "xtreg, fe": lambda: ws.PanelOLS.from_formula(
            "y ~ x1", frame, entity="firm", time="year"
        ).fit(),
        "xtreg, re": lambda: ws.RandomEffects.from_formula(
            "y ~ x1", frame, entity="firm", time="year"
        ).fit(),
        "xtreg, be": lambda: ws.BetweenOLS.from_formula(
            "y ~ x1", frame, entity="firm", time="year"
        ).fit(),
        "xtreg, fd": lambda: ws.FirstDifferenceOLS.from_formula(
            "y ~ x1", frame, entity="firm", time="year"
        ).fit(),
        # The quantile family.  These four were MISSING from this table, so
        # ``test_every_registered_type_runs`` skipped them with "no builder" -- and that
        # skip is exactly why round 184's bug (the whole qreg family had no registered
        # predict types at all) survived: the one test that would have caught it never
        # ran.  A skip reads as "not applicable", not as "untested".
        "qreg": lambda: ws.QuantileRegression.from_formula("y ~ x1 + x2", frame).fit(),
        "iqreg": lambda: ws.QuantileRegression.from_formula("y ~ x1 + x2", frame).fit(),
        "bsqreg": lambda: ws.Bsqreg.from_formula("y ~ x1 + x2", frame).fit(),
        "sqreg": lambda: ws.Sqreg.from_formula(
            "y ~ x1 + x2", frame, quantiles=(0.25, 0.5)
        ).fit(),
    }
    builder = builders.get(model_name)
    return None if builder is None else builder()


#: Models whose setup needs arguments outside this fixture (interval bounds,
#: selection equations); their registry entries are checked but not executed.
NEEDS_SETUP = {
    "etregress", "heckman", "intreg", "tnbreg", "truncreg", "zinb", "ztp",
}

RUNNABLE = sorted(name for name in SUPPORTED_TYPES if name not in NEEDS_SETUP)


# ------------------------- the models whose setup needs a different fixture
@pytest.fixture(scope="module")
def rich_frame():
    rng = np.random.default_rng(23)
    n = 250
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 0.6 * x1 - 0.4 * x2 + rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "y": latent,
            "w": np.maximum(latent, 0.5),
            "yb": (latent > 0).astype(int),
        }
    )
    # ``intreg`` wants two LHS columns: the lower and upper bounds.
    frame["lo"] = frame["w"]
    frame["hi"] = frame["w"]
    return frame


@pytest.fixture(scope="module")
def truncated_frame():
    """Counts truncated at one: every observation must satisfy ``y >= 1``.

    ``x2`` is carried because ``zinb``'s inflation equation needs a second
    regressor; the truncated models simply ignore it.
    """

    rng = np.random.default_rng(29)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    counts = rng.poisson(np.exp(0.5 + 0.4 * x1))
    frame = pd.DataFrame({"x1": x1, "x2": x2, "yc": counts})
    return frame[frame["yc"] >= 1].reset_index(drop=True)


SELECTION_MODELS = {
    "heckman": lambda f: ws.Heckman.from_formula(
        "w ~ x1 + select(yb ~ x1 + x2)", f
    ).fit(),
    "etregress": lambda f: ws.Etregress.from_formula(
        "w ~ x1 + treat(yb ~ x1 + x2)", f
    ).fit(),
    "intreg": lambda f: ws.Intreg.from_formula("lo hi ~ x1", f).fit(),
    "truncreg": lambda f: ws.Truncreg.from_formula("w ~ x1", f, lower=0).fit(),
}

#: Count models: they need a count column (and ztp/tnbreg need it truncated at 1),
#: not a selection or interval setup.
COUNT_MODELS = {
    "zinb": lambda f: ws.ZeroInflatedNegativeBinomial.from_formula(
        "yc ~ x1", f, inflate="x2"
    ).fit(),
    "ztp": lambda f: ws.TruncatedPoisson.from_formula("yc ~ x1", f).fit(),
    "tnbreg": lambda f: ws.TruncatedNegativeBinomial.from_formula(
        "yc ~ x1", f
    ).fit(),
}


@pytest.mark.parametrize("model_name", sorted(SELECTION_MODELS))
def test_selection_and_inflation_models_run_their_types(
    rich_frame, model_name
) -> None:
    """These were skipped by the audit because the fixture lacked their setup.

    Skipping is not verification: a model the audit cannot build is a model whose
    registry is unexamined.  Each is built here with the arguments it actually
    needs -- ``select()`` for heckman, ``treat()`` for etregress, two LHS columns
    for intreg -- so its entry is really exercised.
    """

    result = SELECTION_MODELS[model_name](rich_frame)

    for kind in supported_types(model_name):
        values = np.asarray(result.predict(kind=kind), dtype=float).reshape(-1)
        assert values.size > 0, (model_name, kind)
        assert np.all(np.isfinite(values)), (model_name, kind)


@pytest.mark.parametrize("model_name", sorted(COUNT_MODELS))
def test_count_models_run_their_types(truncated_frame, model_name) -> None:
    result = COUNT_MODELS[model_name](truncated_frame)

    for kind in supported_types(model_name):
        values = np.asarray(result.predict(kind=kind), dtype=float).reshape(-1)
        assert np.all(np.isfinite(values)), (model_name, kind)


def test_truncated_count_mean_is_the_exponential_link(truncated_frame) -> None:
    """``predict, n`` on a truncated count is still ``exp(xb)``."""

    result = COUNT_MODELS["ztp"](truncated_frame)

    counts = np.asarray(result.predict(kind="n"), dtype=float)
    xb = np.asarray(result.predict(kind="xb"), dtype=float)

    np.testing.assert_allclose(counts, np.exp(xb), rtol=1e-10)


#: Registered types that NEED an argument, so a bare call cannot work.  Stata's
#: ``pr(n)``/``pr(a,b)`` require the count (``poisson_postestimation.sthlp:147-148``).
#: They are legitimately registered -- the test supplies the argument rather than
#: treating the requirement as a defect.
_NEEDS_ARGUMENT = {"poisson": {"pr": 2}, "nbreg": {"pr": 2}}


@pytest.mark.parametrize("model_name", RUNNABLE)
def test_every_registered_type_runs(frame, model_name) -> None:
    """A registered type that raises is a lie in the registry."""

    result = _fit(model_name, frame)
    if result is None:
        # NOT a skip.  A missing builder means this model is silently exempt from the
        # only test that would catch a registered-but-broken type -- which is exactly
        # how round 184's bug survived: qreg/iqreg/bsqreg/sqreg had no builder, so the
        # whole family skipped and nobody noticed its predict surface was absent.
        pytest.fail(
            f"no builder for {model_name!r}: add one to _fit so its registered "
            "prediction types are actually exercised"
        )

    extra = _NEEDS_ARGUMENT.get(model_name, {})
    for kind in supported_types(model_name):
        kwargs = {"n": extra[kind]} if kind in extra else {}
        values = np.asarray(result.predict(kind=kind, **kwargs), dtype=float).reshape(-1)
        assert values.size > 0, (model_name, kind)
        assert np.all(np.isfinite(values)), (model_name, kind)


# ------------------------------------------------- two defects the audit found
def test_xtreg_fd_design_matches_its_coefficient_count(frame) -> None:
    """Differencing drops the intercept, so the design must lose that column too.

    Retaining the two-column design against a one-coefficient model made EVERY
    predict type fail with "matmul: core dimension mismatch" -- and because the
    registry still listed the types, the failure looked like a user error.
    """

    result = ws.FirstDifferenceOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    design, names = result.design_matrix
    assert design.shape[1] == len(names) == len(result.params)
    assert design.shape[1] == 1  # x1 only; no constant after differencing


def test_nbreg_predict_n_is_available(frame) -> None:
    """``nbreg_postestimation.sthlp:91``: ``n`` is "number of events; the default".

    It was unreachable because nbreg carries ``/lnalpha`` beyond the design
    columns, and the parameter/column mismatch guard fired before the count-mean
    branch.  The slice already ignores auxiliary parameters, so the guard was
    simply in the wrong order.
    """

    result = ws.NegativeBinomial.from_formula("ycnt ~ x1", frame).fit()

    counts = np.asarray(result.predict(kind="n"), dtype=float)
    xb = np.asarray(result.predict(kind="xb"), dtype=float)

    np.testing.assert_allclose(counts, np.exp(xb), rtol=1e-10)
    assert np.all(counts > 0)


# --------------------------------------------- entries that had to be withdrawn
def test_newey_does_not_register_index(frame) -> None:
    """``index`` is a synonym for ``xb`` handled by the newey wrapper.

    The base dispatcher cannot serve it, so listing it meant
    ``supported_types("newey")`` claimed a kind that then raised.
    """

    assert "index" not in supported_types("newey")


@pytest.mark.parametrize("model_name", ["zip", "zinb", "ztp", "tnbreg"])
def test_truncated_and_inflated_counts_do_not_register_residuals(model_name) -> None:
    """``zip_postestimation.sthlp:86-93`` and ``tnbreg_p.ado:20`` list no residuals.

    The table used to include it while the models' own hooks refused it.
    """

    assert "residuals" not in supported_types(model_name)


def test_the_registry_and_the_model_hooks_agree(frame) -> None:
    """For models exposing ``_predict_types``, the two must not contradict.

    A hook is what actually answers, so a longer registry entry is a lie.
    """

    for model_name in ("zip",):
        result = _fit(model_name, frame)
        declared = tuple(getattr(result, "_predict_types", ()) or ())
        if not declared:
            continue
        registered = supported_types(model_name)
        assert set(registered) <= set(declared), (model_name, registered, declared)
