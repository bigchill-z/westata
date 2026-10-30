"""Tests for Stata's ``estat summarize`` (estimation-sample summary).

Columns per ``estat_summ.ado:706-710``: ``Variable | Obs Mean Std. dev. Min Max``.

Weight handling follows two rules from the ado:

* ``Obs`` changes only under ``fweight`` -- "a, i, and p weights do not change N"
  (``estat_summ.ado:222-233``), because ``fweight`` is literal replication;
* the variance uses the replication form ``sum(w (x-mean)^2)/(sum(w)-1)``, which
  for a *constant* weight ``c`` scales the unweighted ``ddof=1`` variance by
  exactly ``c(N-1)/(cN-1)``.  That identity is asserted to 1e-12, so a wrong
  denominator shows up immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture()
def fit():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length"]].dropna()
    return frame, ws.OLS.from_formula("mpg ~ weight + length", frame).fit()


def test_table_matches_numpy(fit) -> None:
    frame, result = fit
    table = result.estat_summarize().table

    for name in ("mpg", "weight", "length"):
        values = frame[name].to_numpy(float)
        row = table.loc[name]
        assert float(row["Obs"]) == float(values.size)
        assert float(row["Mean"]) == pytest.approx(float(values.mean()), rel=1e-12)
        assert float(row["Std. Dev."]) == pytest.approx(
            float(values.std(ddof=1)), rel=1e-12
        )
        assert float(row["Min"]) == float(values.min())
        assert float(row["Max"]) == float(values.max())


def test_the_response_is_listed_first(fit) -> None:
    """Stata summarizes ``e(depvar)`` first (``estat_summ.ado:86,96``)."""

    frame, result = fit

    assert list(result.estat_summarize().table.index)[0] == "mpg"


def test_n_obs_is_the_estimation_sample_size(fit) -> None:
    _, result = fit

    assert result.estat_summarize().n_obs == float(result.nobs)


def test_fweight_inflates_obs(fit) -> None:
    frame, result = fit
    w = np.full(len(frame), 3.0)

    rendered = result.estat_summarize(weights=w, weight_type="fweight")

    assert rendered.table.loc["mpg", "Obs"] == 3 * len(frame)


def test_aweight_leaves_obs_unchanged(fit) -> None:
    frame, result = fit
    w = np.full(len(frame), 3.0)

    rendered = result.estat_summarize(weights=w, weight_type="aweight")

    assert rendered.table.loc["mpg", "Obs"] == len(frame)


def test_constant_weight_scales_variance_by_the_known_factor(fit) -> None:
    """A ``fweight`` of ``c`` scales the variance by ``c(N-1)/(cN-1)``.

    ``fweight`` is literal replication, so its weights are NOT renormalised and
    this identity holds.  (For ``aweight`` the weights are rescaled to
    ``sum(w) = N`` first, so the ratio is 1 -- see the analogue test below.)
    """

    frame, result = fit
    c = 3.0
    n = len(frame)
    w = np.full(n, c)

    unweighted = float(result.estat_summarize().table.loc["mpg", "Std. Dev."])
    weighted = float(
        result.estat_summarize(weights=w, weight_type="fweight")
        .table.loc["mpg", "Std. Dev."]
    )

    expected = c * (n - 1) / (c * n - 1)
    assert (weighted / unweighted) ** 2 == pytest.approx(expected, rel=1e-12)


def test_weighted_mean_matches_the_hand_computation(fit) -> None:
    frame, result = fit
    values = frame["mpg"].to_numpy(float)
    w = np.where(values > 20, 2.0, 1.0)

    rendered = result.estat_summarize(weights=w, weight_type="aweight")

    expected = float(np.sum(w * values) / np.sum(w))
    assert rendered.table.loc["mpg", "Mean"] == pytest.approx(expected, rel=1e-12)
    assert rendered.table.loc["mpg", "Mean"] != pytest.approx(float(values.mean()))


def test_cluster_block_counts_groups(fit) -> None:
    frame, result = fit
    groups = (np.arange(len(frame)) % 7).astype(int)

    rendered = result.estat_summarize(cluster=groups, cluster_name="grp")

    assert rendered.clusters is not None
    assert rendered.clusters["n"] == 7.0
    _, counts = np.unique(groups, return_counts=True)
    assert rendered.clusters["min"] == float(counts.min())
    assert rendered.clusters["max"] == float(counts.max())
    assert rendered.clusters["avg"] == pytest.approx(float(counts.mean()))


def test_explicit_variable_selection(fit) -> None:
    _, result = fit

    rendered = result.estat_summarize(variables=["weight"])

    assert list(rendered.table.index) == ["weight"]


def test_unknown_variable_is_rejected(fit) -> None:
    _, result = fit

    with pytest.raises(ValueError, match="not in the data"):
        result.estat_summarize(variables=["nope"])


def test_weights_without_a_type_are_rejected(fit) -> None:
    """Stata never guesses the weight type; neither does this."""

    frame, result = fit

    with pytest.raises(ValueError, match="without weight_type"):
        result.estat_summarize(weights=np.ones(len(frame)))


def test_explicit_data_overrides_the_retained_frame(fit) -> None:
    frame, result = fit
    subset = frame.iloc[:10].reset_index(drop=True)

    rendered = result.estat_summarize(data=subset)

    assert rendered.table.loc["mpg", "Obs"] == 10.0


def test_constant_aweight_scales_variance_by_the_known_factor(fit) -> None:
    """An ``aweight`` of ``c`` leaves the variance unchanged once normalised.

    ``aweight``/``pweight``/``iweight`` are rescaled so that ``sum(w) = N`` before
    any variance is formed -- Stata's rule, encoded in
    ``westata.data.weights.normalize_weights`` and followed by every estimator.
    So a CONSTANT aweight carries no information and the ratio to the unweighted
    variance is 1, not ``c(N-1)/(cN-1)``.
    """

    frame, result = fit
    n = len(frame)
    w = np.full(n, 3.0)

    unweighted = float(result.estat_summarize().table.loc["mpg", "Std. Dev."])
    weighted = float(
        result.estat_summarize(weights=w, weight_type="aweight")
        .table.loc["mpg", "Std. Dev."]
    )

    assert weighted == pytest.approx(unweighted, rel=1e-12)


def test_aweight_sd_is_scale_invariant(fit) -> None:
    """Rescaling every aweight must not move the descriptive statistics.

    The estimator path already behaves this way: an aweighted fit's standard
    errors are identical for ``w`` and ``7w``.  The descriptive path used to
    consume the raw weights, so the mean was invariant but the SD was not --
    measured 1.8531 at ``w`` versus 1.8507 at ``7w``.  Same weight type, two
    conventions.
    """

    frame, result = fit
    w = np.linspace(0.5, 3.0, len(frame))

    once = result.estat_summarize(weights=w, weight_type="aweight")
    scaled = result.estat_summarize(weights=w * 7.0, weight_type="aweight")

    assert once.table.loc["mpg", "Mean"] == pytest.approx(
        scaled.table.loc["mpg", "Mean"], rel=1e-12
    )
    assert once.table.loc["mpg", "Std. Dev."] == pytest.approx(
        scaled.table.loc["mpg", "Std. Dev."], rel=1e-12
    )


def test_aweight_does_not_change_obs(fit) -> None:
    """``estat_summ.ado:222-233``: only ``fweight`` changes ``Obs``."""

    frame, result = fit
    w = np.full(len(frame), 3.0)

    rendered = result.estat_summarize(weights=w, weight_type="aweight")

    assert rendered.table.loc["mpg", "Obs"] == len(frame)


def test_pweight_and_iweight_follow_the_aweight_rule(fit) -> None:
    """All three analytic types are normalised to ``sum(w) = N``."""

    frame, result = fit
    w = np.linspace(0.5, 3.0, len(frame))

    base = result.estat_summarize(weights=w, weight_type="aweight")
    for kind in ("pweight", "iweight"):
        other = result.estat_summarize(weights=w, weight_type=kind)
        assert other.table.loc["mpg", "Std. Dev."] == pytest.approx(
            base.table.loc["mpg", "Std. Dev."], rel=1e-12
        ), kind


def test_fweight_still_scales_with_the_constant(fit) -> None:
    """``fweight`` must NOT be normalised: it means literal replication."""

    frame, result = fit
    n = len(frame)
    w = np.full(n, 3.0)

    rendered = result.estat_summarize(weights=w, weight_type="fweight")

    assert rendered.table.loc["mpg", "Obs"] == 3 * n


# ------------------- estimation weights are applied automatically
def _weighted_fit():
    rng = np.random.default_rng(11)
    n = 150
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(size=n)
    w = rng.uniform(0.5, 3.0, size=n)
    frame = pd.DataFrame({"y": y, "x": x})
    return frame, w, ws.OLS.from_formula(
        "y ~ x", frame, weights=w, weight_type="aweight"
    ).fit()


def test_estimation_weights_apply_without_being_passed() -> None:
    """``estat_summ.ado:208-231`` rebuilds the estimation weights automatically.

    So an aweighted fit reports WEIGHTED means even when the caller passes no
    weights. Reporting unweighted statistics there would be a silent difference
    from Stata, and the caller could not tell.
    """

    frame, w, result = _weighted_fit()

    rendered = result.estat_summarize()
    expected = float(np.sum(w * frame["x"]) / np.sum(w))

    assert rendered.table.loc["x", "Mean"] == pytest.approx(expected, rel=1e-12)


def test_an_unweighted_fit_reports_the_plain_mean() -> None:
    """The auto-application must not touch a fit that used no weights."""

    rng = np.random.default_rng(11)
    n = 150
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x": x})
    result = ws.OLS.from_formula("y ~ x", frame).fit()

    rendered = result.estat_summarize()

    assert rendered.table.loc["x", "Mean"] == pytest.approx(
        float(frame["x"].mean()), rel=1e-12
    )


def test_explicit_weights_still_win() -> None:
    """A caller-supplied vector overrides the retained estimation weights."""

    frame, w, result = _weighted_fit()
    other = np.full(len(frame), 2.0)

    rendered = result.estat_summarize(weights=other, weight_type="aweight")

    assert rendered.table.loc["x", "Mean"] == pytest.approx(
        float(frame["x"].mean()), rel=1e-12
    )


def test_the_weight_spec_is_retained_on_the_result() -> None:
    _, w, result = _weighted_fit()

    assert getattr(result, "weight_spec", None) is not None
    assert np.asarray(result.weight_spec.weights).size == len(w)
    assert result.weight_spec.weight_type == "aweight"


def test_a_plain_result_has_no_weight_spec() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {"y": rng.normal(size=50), "x": rng.normal(size=50)}
    )
    result = ws.OLS.from_formula("y ~ x", frame).fit()

    assert getattr(result, "weight_spec", None) is None


def test_summarize_restricts_to_the_estimation_sample() -> None:
    """Rows dropped by listwise deletion must not appear in the table."""

    rng = np.random.default_rng(5)
    n = 100
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x": x})
    frame.loc[0:19, "x"] = np.nan  # 20 dropped

    result = ws.OLS.from_formula("y ~ x", frame).fit()
    rendered = result.estat_summarize()

    assert result.nobs == 80
    assert rendered.table.loc["x", "Obs"] == 80.0


# ------------------- fweight must equal the explicitly expanded data
def test_fweight_reproduces_every_column_of_the_expanded_data() -> None:
    """``fweight`` is literal replication, so ALL five columns must match.

    This is the strongest available check on the whole weighted path: it validates
    the mean, the variance denominator, ``Obs`` and the range at once, against a
    construction that shares no code with the implementation.
    """

    rng = np.random.default_rng(7)
    n = 60
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x": x})
    counts = rng.integers(1, 4, size=n).astype(float)

    result = ws.OLS.from_formula("y ~ x", frame).fit()
    weighted = result.estat_summarize(weights=counts, weight_type="fweight")
    expanded = np.repeat(x, counts.astype(int))

    row = weighted.table.loc["x"]
    assert float(row["Obs"]) == float(expanded.size)
    assert float(row["Mean"]) == pytest.approx(float(expanded.mean()), rel=1e-12)
    assert float(row["Std. Dev."]) == pytest.approx(
        float(expanded.std(ddof=1)), rel=1e-12
    )
    assert float(row["Min"]) == float(expanded.min())
    assert float(row["Max"]) == float(expanded.max())


def test_fweight_on_a_non_integer_multiplier_is_rejected_or_handled() -> None:
    """A non-integer fweight has no expanded-data interpretation.

    Recording the behaviour keeps it from drifting: ``Obs`` is ``sum(w)``, which is
    still well defined, while the variance uses ``sum(w)-1``.
    """

    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]})
    result = ws.OLS.from_formula("y ~ x", frame).fit()

    rendered = result.estat_summarize(
        weights=np.array([0.5, 1.0, 2.5]), weight_type="fweight"
    )

    assert float(rendered.table.loc["x", "Obs"]) == pytest.approx(4.0)


# ------------------------------------------- multi-equation fits (round 79)
def _multi_frame():
    rng = np.random.default_rng(37)
    n = 120
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )


def test_summarize_works_for_mvreg() -> None:
    """Multi-equation fits used to raise a bare ``KeyError``.

    Their coefficient names are ``equation:regressor`` pairs (``y:x2``,
    ``x1:_cons``), which never matched a plain data column, so the variable list
    came out empty and the fixed column selection failed.
    """

    frame = _multi_frame()
    result = ws.Mvreg.from_formula("y x1 = x2", frame).fit()

    rendered = result.estat_summarize()

    assert set(rendered.table.index) == {"y", "x1"}
    assert float(rendered.table.loc["y", "Obs"]) == len(frame)


def test_summarize_works_for_sureg() -> None:
    """``sureg`` stores neither ``e(depvar)`` nor ``e(eqnames)``.

    Its responses are recoverable only from the equation prefixes on the
    coefficient names, so the parse takes the part BEFORE the colon as an equation
    name and the part after as a regressor.
    """

    frame = _multi_frame()
    result = ws.Sureg.from_formula(["y ~ x1", "x1 ~ x2"], frame).fit()

    rendered = result.estat_summarize()

    assert {"y", "x1"} <= set(rendered.table.index)
    assert float(rendered.table.loc["y", "Obs"]) == len(frame)


def test_summarize_still_lists_the_response_first_for_ols() -> None:
    """Regression guard: the multi-equation parse must not reorder plain fits."""

    frame = _multi_frame()
    result = ws.OLS.from_formula("y ~ x1", frame).fit()

    rendered = result.estat_summarize()

    assert list(rendered.table.index) == ["y", "x1"]


def test_summarize_with_unmatched_names_refuses_clearly() -> None:
    """An unmappable name must produce a message naming the column."""

    frame = _multi_frame()
    result = ws.OLS.from_formula("y ~ x1", frame).fit()

    with pytest.raises(ValueError, match="not in the data"):
        result.estat_summarize(variables=["not_a_column"])


# ------------------------------------------ absorbed fits (round 81)
def _absorbed_frame():
    rng = np.random.default_rng(41)
    entities, periods = 30, 5
    firm = np.repeat(np.arange(entities), periods)
    n = firm.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=entities)[firm]
    y = 1.0 + 0.6 * x1 - 0.4 * x2 + effect + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm})


def test_summarize_works_after_reghdfe() -> None:
    """``reghdfe`` supports ``estat summarize``; the data simply was not retained.

    The refusal read like a deliberate limitation ("the estimation-sample data was
    not retained") but was a gap: the estimator held the frame all along.
    """

    frame = _absorbed_frame()
    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", frame).fit()

    rendered = result.estat_summarize()

    assert set(rendered.table.index) == {"y", "x1", "x2"}
    assert float(rendered.table.loc["x1", "Obs"]) == len(frame)


def test_vif_works_after_reghdfe() -> None:
    frame = _absorbed_frame()
    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", frame).fit()

    rendered = result.estat_vif()

    assert set(rendered.names) == {"x1", "x2"}
    assert np.all(np.isfinite(rendered.vif))


def test_fweight_sd_matches_expanded_data(fit) -> None:
    """For fweight the replication form must equal the expanded-data ddof=1 sd."""

    frame, result = fit
    counts = np.array([1, 2, 3] * (len(frame) // 3) + [1] * (len(frame) % 3), dtype=float)

    weighted = result.estat_summarize(weights=counts, weight_type="fweight")

    values = frame["mpg"].to_numpy(float)
    expanded = np.repeat(values, counts.astype(int))
    assert weighted.table.loc["mpg", "Obs"] == float(expanded.size)
    assert weighted.table.loc["mpg", "Mean"] == pytest.approx(
        float(expanded.mean()), rel=1e-12
    )
    assert weighted.table.loc["mpg", "Std. Dev."] == pytest.approx(
        float(expanded.std(ddof=1)), rel=1e-12
    )
    payload = fit[1].estat_summarize().to_dict()

    assert set(payload["mpg"]) == {"obs", "mean", "sd", "min", "max"}
    assert payload["mpg"]["obs"] == float(len(fit[0]))


def test_output_lists_the_variables(fit) -> None:
    text = str(fit[1].estat_summarize())

    assert "Number of obs" in text
    for name in ("mpg", "weight", "length"):
        assert name in text
