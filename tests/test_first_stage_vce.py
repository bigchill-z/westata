"""`first_stage()` must follow the main equation's VCE, and `_cons` goes last.

Two defects, both measured against a live `ivregress 2sls ..., r first` on grilic:

1. THE VCE WAS IGNORED. `fit(cov_type="HC0")` printed NON-ROBUST standard errors with no indication
   either way: coefficient `s` showed .216865 where Stata (`, r`) shows .2327755. Stata refits each
   first-stage equation with the SAME vce() as the 2SLS estimator, so the block must too.

2. `_cons` CAME FIRST. Stata lists the regressors in model order and the constant last; the previous
   renderer emitted `_cons` at the top, which is what the user reported.

Also pinned here: Stata's `regress, r` is HC1 (the n/(n-k) factor), NOT HC0 -- HC0 gives .2315439 and
HC1 gives .2327755, matching Stata; under robust the overall F is a ROBUST WALD statistic (47.74 here,
against 47.38 classical); and the "Robust" block sits at a measured position, not a centred guess.
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


def _lines(cov_type: str) -> list[str]:
    fitted = ws.IV2SLS.from_formula(FORMULA, pd.read_stata(DATA)).fit(cov_type=cov_type)
    return str(fitted.first_stage()).splitlines()


def test_the_constant_is_the_last_row() -> None:
    body = [
        line.split("|")[0].strip()
        for line in _lines("HC0")
        if line.startswith(" ") and "|" in line and "Coefficient" not in line
        and "----" not in line and "|" in line
    ]
    rows = [name for name in body if name and name != "Robust"]
    assert rows[0] == "s", rows
    assert rows[-1] == "_cons", rows


def test_robust_prints_the_robust_standard_errors() -> None:
    line = next(x for x in _lines("HC0") if x.strip().startswith("s |"))
    assert ".2327755" in line, line


def test_nonrobust_prints_the_classical_standard_errors() -> None:
    line = next(x for x in _lines("nonrobust") if x.strip().startswith("s |"))
    assert ".216865" in line, line


def test_robust_adds_the_robust_block_and_lowercases_the_se_label() -> None:
    lines = _lines("HC0")
    assert "             |               Robust" in lines, lines[11:14]
    header = next(x for x in lines if "Coefficient" in x)
    assert "Coefficient  std. err." in header, header


def test_nonrobust_has_no_robust_block_and_keeps_the_capitalised_label() -> None:
    lines = _lines("nonrobust")
    assert not any("Robust" in x for x in lines)
    header = next(x for x in lines if "Coefficient" in x)
    assert "Coefficient  Std. err." in header, header


def test_the_robust_overall_f_is_the_wald_statistic() -> None:
    """47.74 robust vs 47.38 classical on this model -- Stata prints the robust one."""

    robust = next(x for x in _lines("HC0") if x.strip().startswith("F("))
    classical = next(x for x in _lines("nonrobust") if x.strip().startswith("F("))
    assert "47.74" in robust, robust
    assert "47.38" in classical, classical


def test_the_stat_block_is_78_wide_and_shares_one_equals_column() -> None:
    """The '=' column MOVES with the widest value (70 here, 69 for a 6-character Root MSE).

    What is invariant is the row width (78) and that every row of a block uses the SAME column.
    """

    block = _lines("HC0")[3:9]
    for line in block:
        assert len(line) == 78, (len(line), line)
    columns = {line.index("=") for line in block}
    assert len(columns) == 1, columns


def test_the_block_opens_with_the_heading_not_a_blank() -> None:
    """`first_stage()` returns just the block; `summary(first=True)` supplies the leading blank."""

    assert _lines("HC0")[0] == "First-stage regressions"
