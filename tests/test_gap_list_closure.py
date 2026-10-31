"""The last three WESTATA_GAPS.md entries, now closed.

Each was a P1/P2 item on the ADH replication's gap list. They are pinned here so the list
can stay at zero: the probe file asserts them as CONTROLS, and these tests assert the
same behaviour where a reader would look for it.

The three are different in kind, and the difference explains what each test has to do:

* ``margins(at=)`` -- a real capability gap. Tested against Stata's documented rule, which
  is a CARTESIAN PRODUCT for several covariates, not a zip.
* ``predict(exog=)`` -- an error-message gap. Tested by checking the DataFrame path gives
  the same numbers as a hand-built matrix, because "accepts a DataFrame" alone would pass
  even if the column selection were wrong.
* ``etable(file=)`` -- an ergonomics gap. Tested by re-reading the written file, because a
  function that creates an empty file would satisfy "it wrote something".
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _logit_frame(n: int = 400, seed: int = 7301) -> pd.DataFrame:
    """A logistic DGP: the only way to tell "computed at each value" from "ignored at".

    A linear model's marginal effect is a constant, so every ``at`` value would give the
    same answer and the test could not distinguish a working implementation from one that
    drops the sequence.
    """

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(0.3 + 0.8 * x1 - 0.5 * x2)))
    return pd.DataFrame(
        {"yb": (rng.uniform(size=n) < probability).astype(int), "x1": x1, "x2": x2}
    )


def _ols_frame(n: int = 200, seed: int = 7302) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )


# ------------------------------------------------------- margins(at=) sequence
def test_at_accepts_a_sequence() -> None:
    fitted = ws.Logit.from_formula("yb ~ x1 + x2", _logit_frame()).fit()

    produced = fitted.margins(dydx="x1", at={"x2": [-1.0, 0.0, 1.0]})

    assert isinstance(produced, list)
    assert len(produced) == 3


def test_at_sequence_gives_one_result_per_value() -> None:
    """Stata: ``at(x2=(-1 0 1))`` produces separate results for each value."""

    fitted = ws.Logit.from_formula("yb ~ x1 + x2", _logit_frame()).fit()
    produced = fitted.margins(dydx="x1", at={"x2": [0.0, 1.0]})

    assert len(produced) == 2
    for row in produced:
        assert np.isfinite(row.estimate)
        assert np.isfinite(row.std_err)


def test_at_sequence_changes_the_answer_on_a_nonlinear_model() -> None:
    """THE discriminating test.

    On logit the marginal effect depends on where the covariates sit, so three values must
    give three different estimates. If they were identical, the sequence would be accepted
    and ignored -- and the output would look perfectly reasonable.
    """

    fitted = ws.Logit.from_formula("yb ~ x1 + x2", _logit_frame()).fit()
    produced = fitted.margins(dydx="x1", at={"x2": [-1.5, 0.0, 1.5]})

    estimates = [row.estimate for row in produced]
    assert len(set(np.round(estimates, 12))) == 3


def test_at_sequence_labels_identify_the_scenario() -> None:
    """Without a label, three rows of one table are indistinguishable to a reader."""

    fitted = ws.Logit.from_formula("yb ~ x1 + x2", _logit_frame()).fit()
    produced = fitted.margins(dydx="x1", at={"x2": [0.0, 1.0]})

    labels = [row.label for row in produced]
    assert len(set(labels)) == len(labels)
    assert any("x2=0" in label for label in labels)


def test_two_sequences_produce_the_cartesian_product() -> None:
    """``margins.sthlp:1000-1005``: "all possible combinations ... in turn".

    Two values for x1 and two for x2 is FOUR scenarios. A zip would give two, silently
    dropping half of what the caller asked for.
    """

    fitted = ws.Logit.from_formula("yb ~ x1 + x2", _logit_frame()).fit()

    produced = fitted.margins(at={"x1": [0.0, 1.0], "x2": [0.0, 1.0]})

    assert len(produced) == 4


def test_a_scalar_still_returns_a_single_result() -> None:
    """The previous behaviour must not change for the ordinary call."""

    fitted = ws.OLS.from_formula("y ~ x1 + x2", _ols_frame()).fit()

    produced = fitted.margins(dydx="x1", at={"x2": 0.0})

    assert not isinstance(produced, list)
    assert np.isfinite(produced.estimate)


def test_a_string_value_is_treated_as_one_scenario_not_characters() -> None:
    """``str`` is iterable; iterating it would make one scenario per letter."""

    from westata.postestimation.margins import _at_scenarios

    scenarios = _at_scenarios({"g": "treated"})

    assert scenarios == [{"g": "treated"}]


# ------------------------------------------------- predict(exog=DataFrame)
def test_predict_accepts_a_dataframe_with_extra_columns() -> None:
    """The usual case: the fitting frame carries the response and other columns."""

    frame = _ols_frame()
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    predicted = np.asarray(fitted.predict(exog=frame.head(10)), dtype=float)

    assert predicted.shape == (10,)


def test_the_dataframe_path_agrees_with_a_hand_built_matrix() -> None:
    """Selecting the wrong column would still "work" and give a plausible number."""

    frame = _ols_frame()
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    by_frame = np.asarray(fitted.predict(exog=frame), dtype=float)

    design = np.column_stack(
        [np.ones(len(frame)), frame["x1"].to_numpy(), frame["x2"].to_numpy()]
    )
    by_matrix = np.asarray(fitted.predict(exog=design), dtype=float)

    np.testing.assert_allclose(by_frame, by_matrix, rtol=1e-12, atol=0.0)


def test_a_dataframe_missing_a_regressor_is_refused_by_name() -> None:
    frame = _ols_frame()
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    with pytest.raises(ValueError, match="x2"):
        fitted.predict(exog=frame[["x1", "z"]])


def test_the_stored_design_is_used_when_no_exog_is_given() -> None:
    frame = _ols_frame()
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        np.asarray(fitted.predict(), dtype=float),
        np.asarray(fitted.fittedvalues, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


# ------------------------------------------------------------ etable(file=)
@pytest.mark.parametrize("suffix", ["csv", "tsv", "tex", "md", "html"])
def test_etable_writes_the_requested_format(tmp_path, suffix: str) -> None:
    frame = _ols_frame()
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    target = tmp_path / f"table.{suffix}"
    ws.etable([fitted], file=target)

    assert target.exists()
    assert target.stat().st_size > 0


def test_etable_still_returns_the_dataframe(tmp_path) -> None:
    """Writing is a side effect, not a replacement for the return value."""

    frame = _ols_frame()
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    returned = ws.etable([fitted], file=tmp_path / "table.csv")

    assert isinstance(returned, pd.DataFrame)
    assert not returned.empty


def test_the_written_csv_round_trips(tmp_path) -> None:
    """A file that exists but is empty would pass a mere existence check."""

    frame = _ols_frame()
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    target = tmp_path / "table.csv"
    table = ws.etable([fitted], file=target)
    reloaded = pd.read_csv(target, index_col=0)

    assert reloaded.shape == table.shape


def test_an_unknown_extension_is_refused_rather_than_guessed(tmp_path) -> None:
    frame = _ols_frame()
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    with pytest.raises(ValueError, match="extension"):
        ws.etable([fitted], file=tmp_path / "table.xyz")


def test_writing_is_optional() -> None:
    """The original call, with no ``file=``, keeps working unchanged."""

    frame = _ols_frame()
    fitted = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    table = ws.etable([fitted])

    assert isinstance(table, pd.DataFrame)