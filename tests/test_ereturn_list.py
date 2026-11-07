"""ereturn list: implemented in round 249, with its exact column geometry.

C04-03 scored 0.00 layout / 0.41 numeric because the method did not exist at all -- the result had
25 e() macros but no way to print them the way Stata does.

Captured from `quietly regress lnw s expr` then `ereturn list`. FOUR sections and a leading blank
line:

    (blank)
    scalars:
                      e(N) =  758
                     e(r2) =  .2978896544132554
    (blank)
    macros:
                  e(cmdline) : "regress lnw s expr"
    (blank)
    matrices:
                      e(b) :  1 x 3
    (blank)
    functions:
                   e(sample)

THE GEOMETRY, measured rather than guessed:

  * the name is right-aligned into 22 columns;
  * a scalar prints " = " then TWO spaces then the value;
  * a macro prints " : " then the quoted text;
  * a matrix prints " : " then TWO spaces then "r x c";
  * a value drops its leading zero (".2978896544132554") at sixteen significant figures.

WHAT IS EXACT AND WHAT IS NOT. The block is 35 lines either way and every row of the leading
scalars section reproduces byte for byte -- e(N), e(sum_w), e(df_m), e(df_r), e(r2), e(rmse),
e(mss), e(rss), e(r2_a), e(ll), e(ll_0), e(rank). Two known differences remain:

  * e(F) prints one digit longer here. Its STORED value differs by 3.4e-12 (a floating-point
    difference from the order of the mss/rss operations), so this is not a formatting gap and
    tuning the format to hide it would break the scalars that do match;
  * this package's e() carries keys Stata's regress does not define (e(sigma), e(aic), e(bic)), so
    extra rows appear. Choosing which of its own macros a command should publish is a separate
    decision from rendering them.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr", frame).fit()


def test_the_method_exists(fitted) -> None:
    assert callable(getattr(fitted, "ereturn_list", None))


def test_it_starts_with_a_blank_line(fitted) -> None:
    assert fitted.ereturn_list().startswith("\n")


def test_the_four_sections_appear_in_order(fitted) -> None:
    lines = fitted.ereturn_list().splitlines()
    headings = [line for line in lines if line in ("scalars:", "macros:", "matrices:", "functions:")]
    assert headings == ["scalars:", "macros:", "matrices:", "functions:"]


def test_the_name_column_is_twenty_two_wide(fitted) -> None:
    lines = fitted.ereturn_list().splitlines()
    n_line = next(line for line in lines if "e(N) =" in line)
    assert n_line.index("e(N)") + 4 == 22


def test_a_scalar_has_two_spaces_after_the_equals(fitted) -> None:
    lines = fitted.ereturn_list().splitlines()
    n_line = next(line for line in lines if "e(N) =" in line)
    assert " =  758" in n_line


def test_a_scalar_drops_the_leading_zero(fitted) -> None:
    """'.2978896544132554', not '0.2978...'."""

    lines = fitted.ereturn_list().splitlines()
    r2_line = next(line for line in lines if "e(r2) =" in line)
    assert " ." in r2_line
    assert "0." not in r2_line


def test_the_leading_scalars_match_stata_exactly(fitted) -> None:
    """The eleven whose stored values agree; e(F) is excluded for the reason in the docstring."""

    lines = fitted.ereturn_list().splitlines()
    expected = {
        "e(N)": "758",
        "e(sum_w)": "758",
        "e(df_m)": "2",
        "e(df_r)": "755",
        "e(r2)": ".2978896544132554",
        "e(rmse)": ".359901102679567",
        "e(mss)": "41.49190304099152",
        "e(rss)": "97.79424680102603",
        "e(r2_a)": ".2960297594580588",
        "e(ll)": "-299.4325239366634",
        "e(ll_0)": "-433.4714451849196",
        "e(rank)": "3",
    }
    for name, value in expected.items():
        line = next(candidate for candidate in lines if candidate.strip().startswith(name + " "))
        assert line.split("=")[-1].strip() == value, name


def test_the_macros_section_matches_stata(fitted) -> None:
    """Round 250: the ten macros, in Stata's order, with no empty ones.

    e(cmdline) is deliberately absent -- it records the command AS TYPED and this package has no
    such record; inventing one would be a fabricated value (see the _e docstring).
    """

    lines = fitted.ereturn_list().splitlines()
    start = lines.index("macros:")
    end = lines.index("matrices:")
    block = [line for line in lines[start + 1 : end] if line.strip()]
    names = [line.split(":")[0].strip() for line in block]
    assert names == [
        "e(title)", "e(marginsok)", "e(vce)", "e(depvar)", "e(cmd)",
        "e(properties)", "e(predict)", "e(model)", "e(estat_cmd)",
    ]


def test_an_empty_macro_is_not_printed(fitted) -> None:
    """e(vcetype) is "" on an ordinary OLS fit and Stata omits it."""

    assert "e(vcetype)" not in fitted.ereturn_list()


def test_the_recoverable_macros_are_present(fitted) -> None:
    """e(predict) and e(model) are fixed per command, so they can be recovered safely."""

    rendered = fitted.ereturn_list()
    assert 'e(predict) : "regres_p"' in rendered
    assert 'e(model) : "ols"' in rendered


def test_a_matrix_row_shows_its_shape(fitted) -> None:
    lines = fitted.ereturn_list().splitlines()
    b_line = next(line for line in lines if line.strip().startswith("e(b) :"))
    assert "1 x 3" in b_line


def test_the_functions_section_names_the_sample(fitted) -> None:
    lines = fitted.ereturn_list().splitlines()
    assert lines[-1].strip() == "e(sample)"
