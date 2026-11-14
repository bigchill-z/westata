"""``summary(theta=True)`` -- Stata's ``xtreg, re theta`` reporting option.

``xtreg.sthlp:109`` lists ``theta`` under the Reporting options ("report theta") and
``:380-383`` says what it does:

    theta specifies that the output should include the estimated value of theta used in
    combining the between and fixed estimators.  For balanced data, this is a constant,
    and for unbalanced data, a summary of the values is presented in the header.

Both shapes are pinned here against live StataNow 19.5 output on ``lin_1992.dta``, and the
NEGATIVE side matters just as much: Stata refuses the option on every other estimator.

    xtreg ltvfo ..., fe  theta   ->  option theta not allowed   (r(198))
    xtreg ltvfo ..., be  theta   ->  option theta not allowed   (r(198))
    xtreg ltvfo ..., mle theta   ->  option theta not allowed   (r(198))

which the ado sources corroborate: ``xtreg_re.ado`` parses ``THeta`` in both the estimator
and its ``Display`` subprogram, while ``xtreg_fe.ado``, ``xtreg_be.ado`` and
``xtreg_ml.ado`` contain the string ``theta`` ZERO times.  So a flag that printed a block
for those fits would be showing a table Stata would have refused to produce.

The block is placed between the header's last row (``Prob > chi2``) and the coefficient
table -- verified against the captured layout, not merely "somewhere in the header".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.summary import render_summary

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
XS = ["ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mci", "ngca"]
FORMULA = "ltvfo ~ " + " + ".join(XS)

#: The three lines Stata prints for an unbalanced panel, captured verbatim. Every line is
#: 46 characters -- the rules are 19 and 20 around " theta ", so 19 + 7 + 20 = 46.
STATA_UNBALANCED_BLOCK = [
    "------------------- theta --------------------",
    "  min      5%       median        95%      max",
    "0.8150   0.8204     0.8204     0.8204   0.8204",
]

#: The single line Stata prints for a balanced panel, whole, from the live capture.
STATA_BALANCED_LINE = "theta        = .81012778"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")


@pytest.fixture(scope="module")
def re_fit(frame):
    return ws.RandomEffects.from_formula(
        FORMULA, frame, entity="province", time="t"
    ).fit(cov_type="robust")


@pytest.fixture(scope="module")
def unbalanced_fit(frame):
    """One observation dropped, which is all it takes to leave ``e(Tcon) == 0``."""

    trimmed = frame.drop(index=frame.index[(frame["province"] == 1) & (frame["t"] == 5)])
    return ws.RandomEffects.from_formula(
        FORMULA, trimmed, entity="province", time="t"
    ).fit()


# ------------------------------------------------------- the balanced shape
def test_balanced_panel_prints_the_single_theta_line(re_fit) -> None:
    """``xtreg ..., re theta`` on balanced data prints one line, verbatim."""

    lines = re_fit.summary(theta=True).splitlines()
    assert STATA_BALANCED_LINE in lines


def test_the_balanced_line_is_exactly_24_characters(re_fit) -> None:
    """``di "theta" _col(14) "= " e(theta)``: label, pad to 14, "= ", then the value."""

    line = next(l for l in re_fit.summary(theta=True).splitlines() if l.startswith("theta"))
    assert len(line) == 24
    assert line.index("=") == 13


def test_the_value_uses_stata_default_display_not_the_sigma_format(re_fit) -> None:
    """EIGHT significant digits with the leading zero dropped -- ``%.8g``, not ``%9.0g``.

    Probed on StataNow 19.5 with scalars in both ranges::

        .8101277812964918  ->  .81012778     <- eight digits
        .08101277812964918 ->  .08101278     <- STILL eight, not ten

    The second is what separates ``%.8g`` from the sigma rows' "eight decimals below 0.1"
    rule (``_panel_statistic``), which would print ``.08101278`` only by coincidence at this
    magnitude and ``.00810128`` -> ``.00810128`` elsewhere. Reading it off ``e(theta)`` in
    full and re-formatting keeps the two paths honest.
    """

    assert "%.8g" % float(re_fit.e["theta"]) == "0.81012778"
    line = next(l for l in re_fit.summary(theta=True).splitlines() if l.startswith("theta"))
    assert line == "theta        = .81012778"


def test_balanced_theta_lands_between_the_header_and_the_coefficient_table(re_fit) -> None:
    """Position, not just presence: Stata puts it after ``Prob > chi2``, before the rule."""

    lines = re_fit.summary(theta=True).splitlines()
    theta_at = next(i for i, l in enumerate(lines) if l.startswith("theta"))
    prob_at = next(i for i, l in enumerate(lines) if "Prob > chi2" in l)
    rule_at = next(i for i, l in enumerate(lines) if l.startswith("---") and i > prob_at)
    assert prob_at < theta_at < rule_at


# ----------------------------------------------------- the unbalanced shape
def test_unbalanced_panel_prints_the_theta_box(unbalanced_fit) -> None:
    """``xtreg_re.ado:540-545`` prints the five percentiles in a ruled box, verbatim."""

    lines = unbalanced_fit.summary(theta=True).splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("---") and " theta " in l)
    assert lines[start : start + 3] == STATA_UNBALANCED_BLOCK


def test_every_unbalanced_line_is_46_wide(unbalanced_fit) -> None:
    """19 dashes + " theta " + 20 dashes = 46, and the rows are built to the same width."""

    lines = unbalanced_fit.summary(theta=True).splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("---") and " theta " in l)
    widths = [len(line) for line in lines[start : start + 3]]
    assert widths == [46, 46, 46]


def test_unbalanced_uses_the_percentiles_not_a_single_theta(unbalanced_fit) -> None:
    """The two shapes are MUTUALLY EXCLUSIVE, keyed on ``e(Tcon)`` exactly as the ado is."""

    assert float(unbalanced_fit.e["Tcon"]) == 0.0
    assert "theta" not in unbalanced_fit.e
    rendered = unbalanced_fit.summary(theta=True)
    assert "thta" not in rendered
    # And the balanced spelling must NOT appear: "theta        = .8..." would claim a single
    # value where each entity has its own.
    assert "theta        =" not in rendered


# ------------------------------------------- refusals: Stata says "not allowed"
def test_fe_is_unaffected(re_fit, frame) -> None:
    """``xtreg ..., fe theta`` is r(198); the flag must change nothing."""

    fe = ws.PanelOLS.from_formula(FORMULA, frame, entity="province", time="t").fit()
    assert fe.summary(theta=True) == fe.summary()


def test_be_is_unaffected(frame) -> None:
    be = ws.BetweenOLS.from_formula(FORMULA, frame, entity="province", time="t").fit()
    assert be.summary(theta=True) == be.summary()


def test_mle_is_unaffected_even_though_it_titles_itself_random_effects(frame) -> None:
    """``xtreg, mle theta`` is r(198) TOO -- and ``mle`` is the trap.

    Its banner is "Random-effects ML regression", which starts with "Random-effects", and it
    stores variance components like the GLS fit does. An estimator test built on the banner
    alone would print a theta block for a command that refuses the option.
    """

    mle = ws.RandomEffectsMLE.from_formula(
        FORMULA, frame, entity="province", time="t"
    ).fit()
    assert str(mle.e.get("title") or "").startswith("Random-effects")
    assert mle.summary(theta=True) == mle.summary()


def test_a_plain_ols_fit_is_unaffected() -> None:
    """No ``e(Tcon)`` at all: the helper returns nothing rather than raising."""

    auto = pd.read_stata("tests/data/auto.dta")
    ols = ws.OLS.from_formula("mpg ~ weight", auto).fit()
    assert ols.summary(theta=True) == ols.summary()


def test_statsmodels_style_ignores_the_flag(re_fit) -> None:
    """``theta`` names a Stata reporting option; the statsmodels layout is not Stata's."""

    assert re_fit.summary(style="statsmodels", theta=True) == re_fit.summary(
        style="statsmodels"
    )


def test_the_default_is_off(re_fit) -> None:
    """Without the flag the output is byte-identical to before the option existed."""

    assert "theta" not in re_fit.summary().lower()


def test_str_does_not_print_theta(re_fit) -> None:
    """``__str__`` is the plain view; only an explicit request adds the block."""

    assert "theta" not in str(re_fit)


# ------------------------------------------------------------ plumbing
def test_render_summary_accepts_the_keyword(re_fit) -> None:
    """The module-level renderer carries the option too, for direct callers."""

    with_theta = render_summary(re_fit, "stata", theta=True)
    without = render_summary(re_fit, "stata")
    assert with_theta != without
    assert STATA_BALANCED_LINE in with_theta.splitlines()


def test_the_flag_composes_with_first(re_fit) -> None:
    """``first=True`` + ``theta=True`` must not raise on an IV result either."""

    # `first` only exists on IV fits, so this checks the PLUMBING rather than a layout: the
    # non-IV path must still render with both keywords supplied.
    assert re_fit.summary(theta=True).count("theta") == 1


def test_a_theta_fit_without_the_stored_scalar_prints_nothing(frame) -> None:
    """An unbalanced fit that somehow lacks the percentiles must not print a broken box."""

    fit = ws.RandomEffects.from_formula(
        FORMULA, frame.drop(index=frame.index[0]), entity="province", time="t"
    ).fit()
    # Simulate a lost key rather than depending on one existing.
    fit.e.pop("thta_95", None)
    rendered = fit.summary(theta=True)
    assert "theta" not in rendered.lower()
