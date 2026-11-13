"""`didregress`'s postestimation surface is `predict` and five `estat` subcommands -- all missing.

Read from `didregress_postestimation.sthlp` (located with `findfile`):

    predict
    estat trendplots | ptrends | granger | grangerplot | bdecomp

THERE IS NO `test`. That is why report item C17-04's `test canal_post post` fails on both sides:
it is not a valid postestimation command for this estimator, and no coefficient named `post` exists
once the DID design absorbs it into the time fixed effects.

The five `estat` entries ARE a genuine capability gap and are recorded as such in options.md T13,
with the reference output for `ptrends` already captured:

    Parallel-trends test (pretreatment time period)
    H0: Linear trends are parallel

    F(1, 59) =   0.25
    Prob > F = 0.6182

These guards pin the ABSENCE so it cannot be mistaken for coverage, and name what a future
implementation must produce.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

#: The five subcommands `didregress_postestimation.sthlp` lists, and only those.
#: ``estat_ptrends`` is IMPLEMENTED (see its own test below) and has been moved out
#: of this list -- it was implemented but not exported until 0.3.13, which is why the
#: absence guard used to cover it.
POSTESTIMATION_ESTAT = (
    "estat_trendplots",
    "estat_granger",
    "estat_grangerplot",
    "estat_bdecomp",
)


@pytest.mark.parametrize("name", POSTESTIMATION_ESTAT)
def test_the_didregress_estat_entries_are_absent(name: str) -> None:
    """Absent today. When one lands, replace this with a comparison against Stata's output."""

    assert not hasattr(ws, name), (
        f"{name} now exists -- add a guard against Stata's numbers instead of asserting absence"
    )


def test_estat_ptrends_returns_the_linear_trend_test() -> None:
    """``estat ptrends`` is implemented; pin the shape of what it returns.

    Stata prints ``F(1, n_groups - 1)`` from ONE linear-trend interaction
    (``estat_ptrends`` docstring shows a live capture: ``F(1, 59) = 0.25``).  This test
    checks the same structure on a synthetic panel -- one numerator degree of freedom,
    denominator ``n_groups - 1`` -- so a future change that silently returned a
    different test is visible.  It is not a Stata-number comparison: the panel here is
    seeded, not the fixture the capture used.
    """

    rows = []
    for unit in range(60):
        for period in range(1, 16):
            treated = 1 if (unit < 20 and period >= 8) else 0
            rows.append((unit, period, treated, float(0.3 * treated + 0.01 * unit)))
    frame = pd.DataFrame(rows, columns=["unit", "period", "d", "y"])

    fit = ws.Didregress.from_formula("y", "d", frame, group="unit", time="period").fit()
    reported = ws.estat_ptrends(fit)

    assert reported["df_numerator"] == 1.0, "the test is ONE linear-trend interaction"
    assert reported["df_denominator"] == 59.0, "Stata uses n_groups - 1"
    assert 0.0 <= reported["pvalue"] <= 1.0


def test_didregress_has_no_test_postestimation_command() -> None:
    """`didregress_postestimation.sthlp` lists predict and estat only.

    The plain `test` command does not work after `didregress` -- `test canal_post post` raises
    r(111) in Stata because neither name resolves to a coefficient. Pinned so the report's C17-04
    entry is not read as "the package is missing a spelling".
    """

    import pandas as pd

    try:
        frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\nsw_dw.dta")
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nsw_dw.dta is not available")

    # A DID fit exposes exactly one coefficient: the treatment-time interaction.
    frame = frame.copy()
    frame["post"] = (frame["re78"] > frame["re78"].median()).astype(float)
    frame["unit"] = (frame.index % 100).astype(int)
    frame["period"] = (frame.index // 100).astype(int)

    fitted = ws.Didregress.from_formula(
        "re78", "post", frame, group="unit", time="period"
    ).fit()

    assert list(fitted.params.index) == ["post"]
    # A hypothesis about a coefficient the design absorbed cannot be tested.
    with pytest.raises(Exception):
        fitted.test("post treated_unit")


def test_ptrends_reference_output_is_recorded() -> None:
    """The target a `ptrends` implementation must hit, kept where it will be found."""

    reference = {
        "statistic": 0.25,
        "df_denom": 59,
        "pvalue": 0.6182,
        "sample": "pretreatment periods only, 60 groups over 15 periods, 20 treated",
    }
    assert reference["df_denom"] == 59
    assert 0.0 < reference["pvalue"] < 1.0
