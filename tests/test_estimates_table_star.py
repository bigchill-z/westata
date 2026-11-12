"""`estimates table ..., star(...)` -- the layout changes, not just the cells.

Measured on `estimates table ols_no_iq ols_with_iq tsls liml, star(0.1 0.05 0.01)` over grilic:

    ------------------------------------------------------------------------------
        Variable |   ols_no_iq      ols_with_iq        tsls            liml
    -------------+----------------------------------------------------------------
               s |  .10264304***    .09278735***    .06078035***    .06063623***
            expr |   .0381189***    .03934425***    .04332367***    .04334159***
              rns | -.08407974***   -.07453249**    -.04352713      -.04338751
           _cons |   4.103675***    3.8951718***    3.2180433***    3.2149943***
    ------------------------------------------------------------------------------
                                           Legend: * p<.1; ** p<.05; *** p<.01

FOUR things differ from the `b se` layout, and all four were measured rather than assumed:

1. COLUMN WIDTH goes from 13 to 16, so the rule is 14 + 16*n (78 for four models, against 66).
2. THE STANDARD-ERROR ROW DISAPPEARS -- one line per coefficient, not two.
3. THE NUMBER right-aligns to offset 11 inside the 16-wide field, its stars follow immediately, and
   the rest of the field is blank. So `-.04352713` gets five trailing spaces, not two.
4. THE NAME LIMIT rises to 12, which is why `ols_with_iq` prints in full here but is truncated to
   `ols_with~q` in the narrower unstarred layout.

The legend lists the LARGEST threshold first, the reverse of the argument order.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
NAMES = ["ols_no_iq", "ols_with_iq", "tsls", "liml"]
STAR = (0.1, 0.05, 0.01)

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(),
    reason="grilic.dta is not available on this machine",
)


@pytest.fixture(scope="module")
def models():
    frame = pd.read_stata(DATA)
    return [
        ws.OLS.from_formula("lnw ~ s + expr + tenure + rns + smsa", frame).fit(cov_type="HC1"),
        ws.OLS.from_formula("lnw ~ iq + s + expr + tenure + rns + smsa", frame).fit(
            cov_type="HC1"
        ),
        ws.IV2SLS.from_formula(
            "lnw ~ s + expr + tenure + rns + smsa + [iq ~ med + kww]", frame
        ).fit(cov_type="HC0"),
        ws.IVLIML.from_formula(
            "lnw ~ s + expr + tenure + rns + smsa + [iq ~ med + kww]", frame
        ).fit(cov_type="HC0"),
    ]


def _starred(models):
    return ws.render_estimates_table(
        models, model_names=NAMES, star=STAR
    ).splitlines()


def _plain(models):
    return ws.render_estimates_table(models, model_names=NAMES, se=True).splitlines()


# ---------------------------------------------------------------- geometry changes
def test_the_rule_widens_from_sixty_six_to_seventy_eight(models) -> None:
    assert len(_plain(models)[1]) == 66
    assert len(_starred(models)[1]) == 78


def test_the_standard_error_rows_disappear(models) -> None:
    """Six regressors plus a constant means 7 coefficient lines, not 14."""

    starred = _starred(models)
    assert len(starred) == 13, len(starred)
    assert len(_plain(models)) == 20


def test_a_name_that_fits_is_not_abbreviated(models) -> None:
    """The 16-wide column takes eleven characters, so ols_with_iq prints in full."""

    assert "ols_with_iq" in _starred(models)[2]
    assert "ols_with~q" in _plain(models)[2]


# ---------------------------------------------------------------- the stars themselves
def test_three_stars_for_p_below_one_percent(models) -> None:
    row = next(line for line in _starred(models) if line.split("|")[0].strip() == "s")
    assert ".10264304***" in row, row


def test_two_stars_for_p_between_one_and_five_percent(models) -> None:
    """`rns` in the second OLS column has p = .013, so it earns **."""

    row = next(line for line in _starred(models) if line.split("|")[0].strip() == "rns")
    assert "-.07453249**" in row, row


def test_no_stars_above_the_largest_threshold(models) -> None:
    row = next(line for line in _starred(models) if line.split("|")[0].strip() == "rns")
    assert "-.04352713" in row, row
    assert "-.04352713*" not in row, row


def test_the_number_aligns_to_offset_eleven(models) -> None:
    """Every cell: right-aligned 11-character number, then its stars, then blanks."""

    import re

    for line in _starred(models)[4:]:
        if "|" not in line or set(line.strip()) <= {"-", "+"}:
            continue
        for column in range(4):
            segment = line[14 + 16 * column : 14 + 16 * column + 16]
            match = re.match(r"^(\s*)([-\d.]+)(\**)(\s*)$", segment)
            if match is None:
                assert segment.strip() == "", repr(segment)
                continue
            pad, number, _stars_run, _tail = match.groups()
            assert len(pad) + len(number) == 11, (segment, len(pad), len(number))


# ---------------------------------------------------------------- the legend
def test_the_legend_lists_the_largest_threshold_first(models) -> None:
    assert _starred(models)[-1].strip() == "Legend: * p<.1; ** p<.05; *** p<.01"


def test_the_legend_is_right_aligned_to_the_rule(models) -> None:
    last = _starred(models)[-1]
    assert len(last.strip()) < 78
    assert last.endswith("p<.01")


def test_no_star_keeps_the_old_legend(models) -> None:
    assert _plain(models)[-1].strip() == "Legend: b/se"


def test_se_is_ignored_when_stars_are_requested(models) -> None:
    """Stata does not print both; `star()` suppresses the error rows regardless of b/se."""

    both = ws.render_estimates_table(
        models, model_names=NAMES, se=True, star=STAR
    ).splitlines()
    assert both == _starred(models)
