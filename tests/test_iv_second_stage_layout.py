"""`summary(first=True)` reproduces `ivregress ..., first` in full: both tables and the footer.

The user asked for the second-stage table and the `Endogenous:`/`Exogenous:` footer to line up too,
not just the first-stage block. Three things were missing or wrong:

1. `e(vcetype)` was empty for a robust `ivregress 2sls`. A live `display e(vcetype) e(vce)` shows
   "Robust"/"robust" whenever `, r` is given -- the old code believed only `ivregress gmm` set it,
   so a robust 2SLS table printed neither the "Robust" banner nor the lowercase "std. err.".

2. No closing rule before the roles footer: Stata prints the coefficient table's final rule, THEN
   `Endogenous:`.

3. The block separator: Stata opens with one blank line, has TWO blanks between the first-stage
   block and the 2SLS table. Emitting three (because the body carries its own leading blank) put the
   banner one line low.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
FORMULA = "lnw ~ s + expr + tenure + rns + smsa + [iq ~ med + kww]"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(),
    reason="grilic.dta is not available on this machine",
)


def _lines(cov_type: str, first: bool = False) -> list[str]:
    fitted = ws.IV2SLS.from_formula(FORMULA, pd.read_stata(DATA)).fit(cov_type=cov_type)
    return str(fitted.summary(first=first)).splitlines()


# ---------------------------------------------------------------- e(vcetype) follows the VCE
def test_robust_sets_vcetype_and_vce() -> None:
    fitted = ws.IV2SLS.from_formula(FORMULA, pd.read_stata(DATA)).fit(cov_type="HC0")
    assert fitted.e["vcetype"] == "Robust", fitted.e["vcetype"]
    assert fitted.e["vce"] == "robust", fitted.e["vce"]


def test_plain_sets_neither() -> None:
    fitted = ws.IV2SLS.from_formula(FORMULA, pd.read_stata(DATA)).fit(cov_type="nonrobust")
    assert fitted.e["vcetype"] == "", repr(fitted.e.get("vcetype"))
    assert not fitted.e.get("vce"), repr(fitted.e.get("vce"))


# ---------------------------------------------------------------- the second-stage table
def test_robust_second_stage_has_the_banner() -> None:
    lines = _lines("HC0")
    assert "             |               Robust" in lines, lines[6:10]
    assert any("Coefficient  std. err." in line for line in lines), lines


def test_plain_second_stage_has_no_banner() -> None:
    lines = _lines("nonrobust")
    assert not any("Robust" in line for line in lines)
    assert any("Coefficient  Std. err." in line for line in lines)


def test_the_roles_footer_follows_a_closing_rule() -> None:
    lines = _lines("HC0")
    index = next(i for i, line in enumerate(lines) if line.startswith("Endogenous:"))
    assert lines[index - 1].rstrip() == "-" * 78, lines[index - 1]
    assert lines[index + 1] == "Exogenous:  s expr tenure rns smsa med kww", lines[index:]


# ---------------------------------------------------------------- the combined block
def test_the_combined_block_opens_with_one_blank_line() -> None:
    lines = _lines("HC0", first=True)
    assert lines[0] == "", lines[:2]
    assert lines[1] == "First-stage regressions", lines[:2]


def test_exactly_two_blank_lines_separate_the_blocks() -> None:
    lines = _lines("HC0", first=True)
    banner = next(i for i, line in enumerate(lines) if "Instrumental-variables 2SLS" in line)
    assert lines[banner - 1] == "", lines[banner - 2 : banner + 1]
    assert lines[banner - 2] == "", lines[banner - 2 : banner + 1]
    assert lines[banner - 3].rstrip() == "-" * 78, lines[banner - 3 : banner]


def test_the_combined_block_ends_with_the_roles_footer() -> None:
    lines = _lines("HC0", first=True)
    assert lines[-1] == "Exogenous:  s expr tenure rns smsa med kww", lines[-3:]
    assert lines[-2] == "Endogenous: iq", lines[-3:]


def test_the_combined_lengths_differ_only_by_the_robust_banner() -> None:
    """46 robust vs 44 plain: the banner line plus its extra blank."""

    assert len(_lines("HC0", first=True)) == 46
    assert len(_lines("nonrobust", first=True)) == 44
