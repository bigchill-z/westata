"""Guards for the two user decisions plus the deterministic count notes.

Option C: ML tables open with Stata's `Iteration 0` line (the null-model likelihood), which matches
exactly; the converged value stays in the header row that already carried it. No fabricated middle
steps.

Convergence warning: an unconverged estimate now ends with Stata's warning instead of looking
identical to a sound one.

Count notes: poisson/nbreg on a non-integer response emit Stata's deterministic data notes.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _auto() -> pd.DataFrame:
    return pd.read_csv(r"docs\data\auto.csv")


def _noncount(seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"y": rng.uniform(1, 5, 120), "x": rng.normal(size=120)}
    )


def _count(seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"y": rng.poisson(3, 120).astype(float), "x": rng.normal(size=120)}
    )


# ---------------------------------------------------------------- option C
def test_logit_opens_with_the_exact_iteration_zero_line() -> None:
    fitted = ws.Logit.from_formula("foreign ~ mpg + weight", _auto()).fit()
    # MEASURED: line 0 is the blank Stata opens with; the trace is line 1. The two spaces before
    # the value come from right-aligning it in a fixed field.
    lines = str(fitted.summary()).splitlines()
    assert lines[0] == "", lines[:3]
    first = lines[1]
    assert first == "Iteration 0:  Log likelihood =  -45.03321  ", repr(first)


def test_a_blank_line_separates_the_log_from_the_banner() -> None:
    lines = str(ws.Logit.from_formula("foreign ~ mpg + weight", _auto()).fit().summary()).splitlines()
    # '' / Iteration 0 / '' / banner -- so the separating blank is at index 2.
    assert lines[2] == "", lines[:4]
    assert lines[3].startswith("Logistic regression"), lines[:4]


def test_the_converged_likelihood_stays_in_the_header() -> None:
    """Option C does not invent a second numbered line for the final value."""

    lines = str(ws.Logit.from_formula("foreign ~ mpg + weight", _auto()).fit().summary()).splitlines()
    assert sum(1 for line in lines if line.startswith("Iteration")) == 1
    assert any(line.startswith("Log likelihood = -27.175156") for line in lines)


def test_no_fake_step_number_is_emitted() -> None:
    rendered = str(ws.Logit.from_formula("foreign ~ mpg + weight", _auto()).fit().summary())
    assert "Iteration ?" not in rendered
    assert "Iteration 5" not in rendered


def test_ols_has_no_iteration_line() -> None:
    rendered = str(ws.OLS.from_formula("price ~ mpg", _auto()).fit().summary())
    assert "Iteration" not in rendered


# ---------------------------------------------------------------- warning
def test_a_converged_fit_has_no_warning() -> None:
    fitted = ws.Logit.from_formula("foreign ~ mpg + weight", _auto()).fit()
    assert not str(fitted.summary()).endswith("Warning: Convergence not achieved.")


def test_an_unconverged_fit_warns_at_the_end() -> None:
    """Stata prints it at line 320 of 321, after the coefficient table."""

    fitted = ws.Logit.from_formula("foreign ~ mpg + weight", _auto()).fit()
    fitted.e["converged"] = 0
    lines = str(fitted.summary()).splitlines()
    assert lines[-1] == "Warning: Convergence not achieved.", lines[-1]


# ---------------------------------------------------------------- count notes
@pytest.mark.parametrize(
    "cls,first",
    [
        ("Poisson", "note: noncount dependent variable encountered; if you are fitting an"),
        ("NegativeBinomial", "note: you are responsible for interpretation of non-count dep. variable"),
    ],
)
def test_a_noncount_response_emits_the_note(cls, first) -> None:
    rendered = str(getattr(ws, cls).from_formula("y ~ x", _noncount()).fit().summary())
    assert rendered.splitlines()[0] == first, rendered.splitlines()[:2]


def test_the_poisson_note_wraps_with_six_spaces() -> None:
    rendered = str(ws.Poisson.from_formula("y ~ x", _noncount()).fit().summary())
    assert rendered.splitlines()[1] == "      exponential-mean model, consider using robust standard errors."


@pytest.mark.parametrize("cls", ["Poisson", "NegativeBinomial"])
def test_an_integer_response_is_silent(cls) -> None:
    rendered = str(getattr(ws, cls).from_formula("y ~ x", _count()).fit().summary())
    assert "note:" not in rendered
