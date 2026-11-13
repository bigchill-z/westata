"""The weighted-logit case the user reported, and the two text rules it exposed.

The user compared

    ws.Logit.from_formula("survive ~ child + female + class1 + class2 + class3", exp).fit(cov_type="HC1")
    # Stata: logit survive child female class1 class2 class3 [fweight=freq], r

and got different numbers. Three separate causes, all measured against a live session:

1. THE ROBUST VCE SCALE. Expanding fweights and fitting `cov_type="HC1"` applies statsmodels'
   N/(N-K) factor, which Stata's `logit, r` does not use here. Stata applies **N/(N-1)** -- the
   renderer already implements that as `cov_type="robust"`. With HC1 every standard error was off
   by a factor of 1.00137; with "robust" all six standard errors match to 1e-7.

2. THE LOG-LIKELIHOOD LABEL. A robust/cluster VCE makes Stata report a **pseudolikelihood**:
   `Log pseudolikelihood = -1105.0306`, in the banner AND in the iteration trace. The nonrobust form
   says `Log likelihood`.

3. THE LEADING BLANK LINE belongs to the ITERATION TRACE only. `logit y x` opens with a blank line
   then the trace; `xtlogit ..., fe` (which prints notes) opens DIRECTLY with the note and has no
   blank at all. The old code stripped every leading blank, so noted tables looked right by accident
   while plain logit was missing its blank.

The iteration value is printed at EIGHT SIGNIFICANT FIGURES, not a fixed decimal count:
`-441.45229` has five decimals and `-1384.7284` has four, and both are eight significant figures.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\titanic.dta"
GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(),
    reason="titanic.dta is not available on this machine",
)


@pytest.fixture(scope="module")
def titanic():
    frame = pd.read_stata(DATA)
    return frame.loc[frame.index.repeat(frame["freq"])].reset_index(drop=True).drop(
        columns=["freq"]
    )


@pytest.fixture(scope="module")
def weighted(titanic):
    return ws.Logit.from_formula(
        "survive ~ child + female + class1 + class2 + class3", titanic
    ).fit(cov_type="robust")


# ---------------------------------------------------------------- the numbers
def test_robust_reproduces_the_fweight_standard_errors(weighted) -> None:
    """The six standard errors Stata prints for `logit ..., [fweight=freq], r`."""

    expected = {
        "child": 0.2767452,
        "female": 0.1363096,
        "class1": 0.1475218,
        "class2": 0.1502193,
        "class3": 0.1621035,
        "Intercept": 0.0798876,
    }
    for name, want in expected.items():
        got = float(weighted.bse[name])
        assert abs(got - want) < 1e-7, (name, got, want)


def test_hc1_is_not_statas_factor(weighted, titanic) -> None:
    """`HC1` scales by N/(N-K); Stata's `logit, r` scales by N/(N-1).

    Using HC1 is what made every standard error slightly too large.
    """

    hc1 = ws.Logit.from_formula(
        "survive ~ child + female + class1 + class2 + class3", titanic
    ).fit(cov_type="HC1")
    robust_se = float(weighted.bse["child"])
    hc1_se = float(hc1.bse["child"])
    n, k = 2201, 6
    ratio = hc1_se / robust_se
    assert abs(ratio - ((n / (n - k)) / (n / (n - 1.0))) ** 0.5) < 1e-6, ratio


def test_the_coefficients_are_unaffected_by_the_vce(weighted) -> None:
    expected = [-1.2338991, 1.0615424, 2.4200603, 0.8576762, -0.1604188, -0.9200861]
    got = [float(weighted.params[name]) for name in weighted.params.index]
    for actual, want in zip(got, expected):
        assert abs(actual - want) < 1e-6, (actual, want)


# ---------------------------------------------------------------- the labels
def test_a_robust_fit_reports_a_pseudolikelihood(weighted) -> None:
    text = str(weighted.summary())
    assert "Log pseudolikelihood = -1105.0306" in text
    assert "Log likelihood = -1105.0306" not in text


def test_a_nonrobust_fit_reports_a_plain_likelihood() -> None:
    frame = pd.read_stata(GRILIC)
    text = str(ws.Logit.from_formula("rns ~ s + expr + tenure", frame).fit().summary())
    assert "Log likelihood = " in text
    assert "Log pseudolikelihood" not in text


def test_the_iteration_line_also_says_pseudolikelihood(weighted) -> None:
    trace = [
        line for line in str(weighted.summary()).splitlines() if line.startswith("Iteration")
    ]
    assert trace and trace[0].startswith("Iteration 0:  Log pseudolikelihood = "), trace


def test_the_iteration_line_matches_stata_character_for_character(weighted) -> None:
    lines = str(weighted.summary()).splitlines()
    assert lines[1] == "Iteration 0:  Log pseudolikelihood = -1384.7284  ", repr(lines[1])


# ---------------------------------------------------------------- the leading blank
def test_a_plain_logit_opens_with_a_blank_line() -> None:
    """Stata: line 0 empty, line 1 the trace, line 2 empty, line 3 the banner."""

    frame = pd.read_stata(GRILIC)
    lines = str(ws.Logit.from_formula("rns ~ s + expr + tenure", frame).fit().summary()).splitlines()
    assert lines[0] == "", lines[:4]
    assert lines[1].startswith("Iteration 0:  Log likelihood = "), lines[:4]
    assert lines[2] == "", lines[:4]
    assert lines[3].startswith("Logistic regression"), lines[:4]


def test_the_iteration_value_uses_eight_significant_figures() -> None:
    """`-441.45229` (five decimals) and `-1384.7284` (four) are both eight significant figures."""

    frame = pd.read_stata(GRILIC)
    lines = str(ws.Logit.from_formula("rns ~ s + expr + tenure", frame).fit().summary()).splitlines()
    assert lines[1] == "Iteration 0:  Log likelihood = -441.45229  ", repr(lines[1])
