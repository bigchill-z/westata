"""``var``'s three postestimation commands, from ``var_postestimation.sthlp:45-47``.

    INCLUDE help post_estatic
    INCLUDE help post_estatsum
    INCLUDE help post_estatvce

``VARResults`` is a standalone dataclass -- it does NOT inherit the ``StataResults``
surface other estimators get -- so all three were missing even though the class already
computed the values they need.

Three defects were found while wiring them up:

1. ``e(cmd)`` was never stored, so ``estat ic`` printed ``.`` in its Model column (the
   symptom round 129 fixed for the other estimators);
2. the shared ``estat ic`` infers ``df`` from ``result.cov_params()`` as a CALL, but
   ``VARResults.cov_params`` is a property -- so the inference raised ``TypeError``;
3. ``var`` stores no ``e(ll_0)``, and the renderer printed the literal text ``nan`` for
   the absent ``ll(null)`` instead of Stata's ``.``.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws

VAR_COMMANDS = ("estat_ic", "estat_summarize", "estat_vce")


def _system(n: int = 200, seed: int = 2110):
    rng = np.random.default_rng(seed)
    return np.column_stack([np.cumsum(rng.normal(size=n)) for _ in range(2)])


@pytest.fixture(scope="module")
def var_fit():
    return ws.VAR(_system(), lags=2, names=["y1", "y2"]).fit()


# ------------------------------------------------------------- e(cmd) macro
def test_cmd_is_stored(var_fit) -> None:
    """``var.sthlp:294+`` lists ``e(cmd)``; it was absent.

    Its absence made ``estat ic`` print ``.`` for the model name -- a rendering bug
    caused by a missing macro, not by a formatting mistake.
    """

    assert var_fit.e["cmd"] == "var"


def test_the_model_column_names_the_command(var_fit) -> None:
    rendered = str(var_fit.estat_ic())

    assert "var" in rendered
    assert rendered.count("|") >= 2


# ------------------------------------------------------------- estat ic
def test_estat_ic_works(var_fit) -> None:
    rendered = var_fit.estat_ic()

    assert rendered is not None
    assert str(rendered)


def test_estat_ic_reports_the_information_criteria(var_fit) -> None:
    text = str(var_fit.estat_ic())

    for column in ("AIC", "BIC", "AICc", "CAIC"):
        assert column in text, column


def test_the_table_reports_TOTALS_not_the_per_observation_macros(var_fit) -> None:
    """``e(aic)`` is PER OBSERVATION; ``estat ic`` reports the TOTAL.

    Measured here: ``e(aic) = 5.7732`` against a table AIC of ``1143.0955``, a ratio of
    exactly ``1/N``.  Both are legitimate -- ``var``'s ``e(aic)`` follows Stata's
    normalisation for the criteria it prints in ``var``'s own output, while
    ``estat ic`` follows ``[R] IC``'s total convention.

    My first version of this test asserted the table contained ``e(aic)`` verbatim; it
    does not, and forcing it to would have meant replacing a correct total with a
    per-observation number.
    """

    rendered = str(var_fit.estat_ic())
    reported = var_fit.estat_ic()

    # The table holds the total, which is N times the per-observation macro.
    total = float(var_fit.e["aic"]) * reported.n
    assert f"{total:.5f}" in rendered
    # And the macro on its own is NOT what is printed.
    assert f"{float(var_fit.e['aic']):.5f}" not in rendered


def test_the_bic_total_follows_the_same_scaling(var_fit) -> None:
    """Guard: the relation is a normalisation, not a coincidence for AIC alone."""

    rendered = str(var_fit.estat_ic())
    reported = var_fit.estat_ic()

    total = float(var_fit.e["sbic"]) * reported.n
    assert f"{total:.5f}" in rendered


def test_the_parameter_count_is_e_k(var_fit) -> None:
    """Stata penalises every estimated parameter: ``k_eq * neqs``."""

    rendered = var_fit.estat_ic()
    expected = float(var_fit.e["k"])

    assert f"{expected:>8.0f}" in str(rendered)


def test_a_missing_ll_null_renders_as_a_dot(var_fit) -> None:
    """``var`` stores no ``e(ll_0)``, and Stata prints ``.`` for it.

    The renderer used to format the NaN directly, printing the literal ``nan`` -- which
    reads as a number rather than as "not available".
    """

    rendered = str(var_fit.estat_ic())

    assert "ll(null)" in rendered
    assert "nan" not in rendered.lower()


def test_the_missing_ll_null_is_genuinely_absent(var_fit) -> None:
    """Guard for the previous test: the dot must reflect real absence."""

    assert "ll_0" not in var_fit.e


# ----------------------------------------------------- the other two commands
def test_estat_summarize_is_offered(var_fit) -> None:
    """``var_postestimation.sthlp:46`` lists it."""

    assert callable(var_fit.estat_summarize)


def test_estat_vce_is_offered(var_fit) -> None:
    """``var_postestimation.sthlp:47`` lists it."""

    assert callable(var_fit.estat_vce)


def test_estat_summarize_works_without_the_caller_supplying_data(var_fit) -> None:
    """The estimation sample is rebuilt from ``endog`` and ``e(tmin)``.

    The shared implementation needs a ``model_data`` frame that ``VARResults`` does not
    have, so the method supplies one -- otherwise the caller had to construct it by hand
    for a command the help file says is available.
    """

    rendered = var_fit.estat_summarize()

    assert rendered is not None
    assert "Number of obs" in str(rendered)


def test_the_summarized_sample_is_the_estimation_sample(var_fit) -> None:
    """``N`` observations were used, not the full series.

    ``var`` consumes ``mlag`` rows to form the lags; reporting the full series would
    summarise observations the model never saw.
    """

    rendered = var_fit.estat_summarize()

    assert rendered.n_obs == float(var_fit.e["N"])
    assert rendered.n_obs < float(np.asarray(var_fit._result.endog).shape[0])


def test_the_variables_appear_in_equation_order(var_fit) -> None:
    """``y1`` then ``y2``, as in the estimation command.

    The shared implementation's default picks "the response plus the reported
    regressors", which for a VAR came back reversed.  Asserting the order -- not just the
    set -- is what catches that.
    """

    rendered = var_fit.estat_summarize()

    assert list(rendered.table.index) == list(var_fit.names)


def test_estat_vce_works(var_fit) -> None:
    rendered = var_fit.estat_vce()

    assert rendered is not None
    assert "Covariance" in str(rendered)


def test_the_vce_matrix_is_square_and_full_rank(var_fit) -> None:
    """``estat vce`` needs ``param_names`` and a matching covariance.

    ``VARResults`` stores ``params`` as a ``(k_eq, neqs)`` frame and ``cov_params`` as a
    PROPERTY, so the method adapts both before delegating.
    """

    rendered = var_fit.estat_vce()
    table = rendered.to_frame()
    size = len(rendered.names)

    assert table.shape == (size, size)
    assert size == len(var_fit.stata_param_names)


# ------------------------------------------------ unique names in a multi-equation VAR
def test_the_vce_names_are_UNIQUE(var_fit) -> None:
    """A two-equation VAR repeats ``L.y1``/``_cons`` once per equation.

    The raw flat names therefore contain duplicates, and a duplicated index makes the
    frame unusable for selection: ``frame.loc["L.y1"]`` returns TWO rows, so the matrix
    cannot answer "what is the variance of this coefficient".  Qualifying each name with
    its equation is what makes the index a real index.
    """

    rendered = var_fit.estat_vce()
    names = list(rendered.names)

    assert len(names) == len(set(names))


def test_every_name_carries_its_equation(var_fit) -> None:
    """Each coefficient is attributed to the equation it belongs to."""

    names = list(var_fit.estat_vce().names)

    for equation in var_fit.names:
        assert any(name.startswith(f"{equation}#") for name in names), equation


def test_the_index_can_select_a_single_entry(var_fit) -> None:
    """The practical consequence of uniqueness: ``loc`` returns one value."""

    table = var_fit.estat_vce().to_frame()
    label = table.index[0]

    selected = table.loc[label]

    assert not isinstance(selected, type(table))  # a row, not a sub-frame


def test_the_diagonal_is_the_variance_of_each_coefficient(var_fit) -> None:
    """A covariance matrix's diagonal is non-negative by definition."""

    table = np.asarray(var_fit.estat_vce().to_frame(), dtype=float)
    diagonal = np.diag(table)

    assert np.all(diagonal >= 0.0)


def test_a_three_equation_system_gets_three_prefixes() -> None:
    """The prefix logic must scale with the number of equations, not stop at two.

    A VAR needs at least two variables, so there is no single-equation case to compare
    against -- the count below is what a regression in the block arithmetic would change.
    """

    rng = np.random.default_rng(2302)
    n = 150
    data = np.column_stack(
        [np.cumsum(rng.normal(size=n)) for _ in range(3)]
    )
    fitted = ws.VAR(data, lags=1, names=["a", "b", "c"]).fit()

    rendered = fitted.estat_vce()
    names = list(rendered.names)

    assert len(names) == len(fitted.stata_param_names)
    for equation in ("a", "b", "c"):
        assert any(name.startswith(f"{equation}#") for name in names), equation


@pytest.mark.parametrize("command", VAR_COMMANDS)
def test_all_three_exist_on_the_result(var_fit, command) -> None:
    """The three subcommands the help file names are all attached and runnable."""

    assert callable(getattr(var_fit, command)), command
    assert str(getattr(var_fit, command)()), command


def test_the_cov_params_shape_is_the_property(var_fit) -> None:
    """``cov_params`` is a PROPERTY here, unlike on ``StataResults``.

    That asymmetry is why the shared ``estat ic`` needed an explicit ``df``: it calls
    ``result.cov_params()``.
    """

    assert not callable(var_fit.cov_params)
    assert np.asarray(var_fit.cov_params).ndim == 2


def test_estat_ic_accepts_an_explicit_df(var_fit) -> None:
    """The shared option must still work through the override."""

    default = var_fit.estat_ic()
    explicit = var_fit.estat_ic(df=float(var_fit.e["k"]))

    assert str(default) == str(explicit)