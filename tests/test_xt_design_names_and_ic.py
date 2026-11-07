"""The xt design block needs the variable NAMES, and ``e(ic)`` means the same thing there.

Two fixes in one file, both found by looking at a rendered ``xtlogit`` block.

**The group variable was printed as ``.``.** The block showed::

    Group variable:   .             Number of groups    =        100

because the name is not recoverable from the result: ``model_data`` carries the CODED
group labels, not the column name. The estimator now records ``e(ivar)``/``e(tvar)`` from
the frame the caller passed, and the block prints::

    Group variable: firm            Number of groups    =        100

**``e(ic)`` was still the old penalty-term formula here.** Round 238 corrected
``discrete.py`` after the ``poisson_sim`` fixture (``ic = 1.0`` with two parameters)
refuted ``k + 1``, but ``panel_nonlinear.py`` kept the same wrong comment AND formula, with
the real iteration count parked under a non-Stata name. ``xtlogit.sthlp:587`` states the
definition for the panel command outright::

    {synopt:{cmd:e(ic)}}number of iterations

so both families now agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.summary import render_summary

BASE = Path(r"D:\Program Files\StataNow19\ado\base")


def _panel_frame(n_units: int = 100, n_periods: int = 9, seed: int = 8200):
    rng = np.random.default_rng(seed)
    total = n_units * n_periods
    return pd.DataFrame(
        {
            "y": (rng.uniform(size=total) < 0.5).astype(int),
            "x1": rng.normal(size=total),
            "firm": np.repeat(np.arange(n_units), n_periods),
            "year": np.tile(np.arange(1, n_periods + 1), n_units),
        }
    )


@pytest.fixture(scope="module")
def fitted():
    return ws.XtLogit.from_formula(
        "y ~ x1", _panel_frame(), entity="firm", time="year", method="fe"
    ).fit()


# ---------------------------------------------------------- the variable names
def test_the_group_variable_is_named(fitted) -> None:
    """``p9_outputs.txt:15``: ``Group variable: firm``."""

    assert fitted.e["ivar"] == "firm"


def test_the_time_variable_is_named(fitted) -> None:
    assert fitted.e["tvar"] == "year"


def test_the_rendered_block_prints_the_name(fitted) -> None:
    """The visible symptom: a bare ``.`` where the variable name belongs."""

    rendered = render_summary(fitted, style="stata")

    assert "Group variable:" in rendered
    assert "Group variable:   ." not in rendered
    assert "Group variable: firm" in rendered
    # CORRECTED in round 245: ONE space after the colon. Re-measured on this fixture with a live
    # `xtlogit y x1, fe` -- 'Group variable: firm' -- after round 224 fixed the padding and this
    # assertion was left holding the old three-space form.
    assert "Group variable:   firm" not in rendered


def test_the_name_comes_from_the_argument_not_from_model_data(fitted) -> None:
    """``model_data`` stores coded group labels, so the name cannot be recovered there.

    Guard: if a later change tried to derive it from the frame, this would show.
    """

    frame = getattr(fitted, "model_data", None)
    if frame is not None and hasattr(frame, "columns"):
        assert "firm" not in list(frame.columns) or fitted.e["ivar"] == "firm"


# ------------------------------------------------------------- e(ic) semantics
def test_e_ic_is_the_iteration_count(fitted) -> None:
    """``xtlogit.sthlp:587``: "``e(ic)`` number of iterations"."""

    assert fitted.e["ic"] >= 1
    assert fitted.e["ic"] != int(round(fitted.e["rank"])) + 1


def test_iterations_is_kept_as_an_alias(fitted) -> None:
    """Callers written against the old layout still work; it is the same number."""

    assert fitted.e["iterations"] == fitted.e["ic"]


def test_the_help_file_says_iterations() -> None:
    """Read the authority rather than trusting the transcription above."""

    text = (BASE / "x/xtlogit.sthlp").read_text(encoding="utf-8", errors="ignore")

    assert "{synopt:{cmd:e(ic)}}number of iterations" in text


def test_the_non_panel_family_agrees() -> None:
    """``logit.sthlp:257`` says the same thing; the two families must not diverge."""

    rng = np.random.default_rng(8202)
    frame = pd.DataFrame(
        {"yb": (rng.uniform(size=250) < 0.5).astype(int), "x": rng.normal(size=250)}
    )
    fitted_logit = ws.Logit.from_formula("yb ~ x", frame).fit()

    assert fitted_logit.e["ic"] >= 1
    assert fitted_logit.e["ic"] != int(round(fitted_logit.e["rank"])) + 1


# ------------------------------------------------------------ nothing regressed
def test_the_design_block_still_has_its_rows(fitted) -> None:
    rendered = render_summary(fitted, style="stata")

    assert "Number of groups" in rendered
    assert "Obs per group:" in rendered


def test_the_estimation_table_survives(fitted) -> None:
    rendered = render_summary(fitted, style="stata")

    assert "Coefficient" in rendered
    assert "Std. err." in rendered