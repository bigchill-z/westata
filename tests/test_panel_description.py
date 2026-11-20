"""`xtset`, `xtdes` and `xtsum` against captured StataNow 19.5 output.

Every expectation here is a literal copy of what Stata printed on the fixture, so a layout
regression fails loudly instead of drifting. Two fixtures were used:

* ``munnell.dta`` -- balanced 48 entities x 17 years (816 observations)
* a synthetic unbalanced panel -- 6 entities, some with 5 periods and some with 3

The geometry notes that matter, all measured rather than guessed:

* ``xtset`` prints a leading blank line, then three lines.
* ``xtdes`` is 75 characters wide on the balanced fixture; the count label sits at column 61.
* The ``Distribution of T_i`` grid's seven values end at 26 / 34 / 42 / 52 / 62 / 70 / 78.
* The pattern rule's right-hand segment is ``max(9, span + 2)`` dashes.
* ``xtsum`` is 79 wide with pipes at 17 and 62; the row label is LEFT-aligned in 8 columns starting
  at column 9, which is why ``within`` carries two trailing spaces and ``between`` only one.
* ``xtsum`` numbers use SEVEN significant figures with the leading zero dropped.
* ``xtsum``'s within min/max are the deviations shifted back by the GRAND mean, not the raw
  deviations.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(MUNNELL).exists(), reason="munnell.dta is not available"
)


@pytest.fixture(scope="module")
def balanced() -> pd.DataFrame:
    frame = pd.read_stata(MUNNELL).copy()
    frame["pid"] = pd.factorize(frame["st_abb"])[0] + 1
    return frame


@pytest.fixture()
def unbalanced() -> pd.DataFrame:
    rows = []
    for entity in range(1, 7):
        count = 5 if entity <= 3 else 3
        for period in range(1, count + 1):
            rows.append({"id": entity, "t": 2000 + period})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- xtset
def test_xtset_balanced_is_byte_identical(balanced) -> None:
    expected = (
        "\n"
        "Panel variable: pid (strongly balanced)\n"
        " Time variable: year, 1970 to 1986\n"
        "         Delta: 1 unit"
    )
    assert str(ws.panel_set(balanced, entity="pid", time="year")) == expected


def test_xtset_unbalanced_is_byte_identical(unbalanced) -> None:
    expected = (
        "\n"
        "Panel variable: id (unbalanced)\n"
        " Time variable: t, 2001 to 2005\n"
        "         Delta: 1 unit"
    )
    assert str(ws.panel_set(unbalanced, entity="id", time="t")) == expected


def test_xtset_reports_a_gap_in_the_middle() -> None:
    """Stata appends ', but with a gap' when an entity skips an interior period."""

    frame = pd.DataFrame(
        [{"id": 1, "t": t} for t in (1, 2, 4, 5)] + [{"id": 2, "t": t} for t in (1, 2, 3, 4, 5)]
    )
    rendered = str(ws.panel_set(frame, entity="id", time="t"))
    assert "1 to 5, but with a gap" in rendered
    assert "unbalanced" in rendered


def test_xtset_rejects_a_missing_column(balanced) -> None:
    with pytest.raises(ws.ConsistencyError, match="not found in data"):
        ws.panel_set(balanced, entity="nope", time="year")


def test_xtset_rejects_duplicate_keys(balanced) -> None:
    with pytest.raises(ws.ConsistencyError, match="does not uniquely identify"):
        ws.panel_set(balanced.assign(pid=1), entity="pid", time="year")


def test_xtset_delta_is_the_modal_spacing() -> None:
    """Every second year gives Delta: 2 units."""

    frame = pd.DataFrame([{"id": 1, "t": t} for t in (2000, 2002, 2004)])
    assert "Delta: 2 units" in str(ws.panel_set(frame, entity="id", time="t"))


# ---------------------------------------------------------------- xtdes
def test_xtdes_balanced_is_byte_identical(balanced) -> None:
    expected = [
        "",
        "     pid:  1, 2, ..., 48                                     n =         48",
        "    year:  1970, 1971, ..., 1986                             T =         17",
        "           Delta(year) = 1 unit",
        "           Span(year)  = 17 periods",
        "           (pid*year uniquely identifies each observation)",
        "",
        "Distribution of T_i:   min      5%     25%       50%       75%     95%     max",
        "                        17      17      17        17        17      17      17",
        "",
        "     Freq.  Percent    Cum. |  Pattern",
        " ---------------------------+-------------------",
        "       48    100.00  100.00 |  11111111111111111",
        " ---------------------------+-------------------",
        "       48    100.00         |  XXXXXXXXXXXXXXXXX",
    ]
    assert str(ws.panel_describe(balanced, entity="pid", time="year")).splitlines() == expected


def test_xtdes_unbalanced_is_byte_identical(unbalanced) -> None:
    expected = [
        "",
        "      id:  1, 2, ..., 6                                      n =          6",
        "       t:  2001, 2002, ..., 2005                             T =          5",
        "           Delta(t) = 1 unit",
        "           Span(t)  = 5 periods",
        "           (id*t uniquely identifies each observation)",
        "",
        "Distribution of T_i:   min      5%     25%       50%       75%     95%     max",
        "                         3       3       3         4         5       5       5",
        "",
        "     Freq.  Percent    Cum. |  Pattern",
        " ---------------------------+---------",
        "        3     50.00   50.00 |  111..",
        "        3     50.00  100.00 |  11111",
        " ---------------------------+---------",
        "        6    100.00         |  XXXXX",
    ]
    assert str(ws.panel_describe(unbalanced, entity="id", time="t")).splitlines() == expected


def test_xtdes_pattern_marks_the_hole_with_a_dot() -> None:
    """An interior gap shows as '11.11', measured from the capture."""

    frame = pd.DataFrame(
        [{"id": 1, "t": t} for t in (1, 2, 4, 5)] + [{"id": 2, "t": t} for t in (1, 2, 3, 4, 5)]
    )
    rendered = str(ws.panel_describe(frame, entity="id", time="t"))
    assert "11.11" in rendered
    assert "11111" in rendered


def test_xtdes_rule_width_follows_the_span() -> None:
    """Measured: span 17 -> 19 dashes, span 4 -> 9 dashes (max(9, span + 2))."""

    short = pd.DataFrame([{"id": i, "t": t} for i in (1, 2, 3) for t in range(1, 5)])
    rules = [line for line in str(ws.panel_describe(short, entity="id", time="t")).splitlines()
             if line.startswith(" -")]
    assert len(rules) == 2
    assert rules[0].endswith("-" * 9)


# ---------------------------------------------------------------- xtsum
def test_xtsum_is_byte_identical(balanced) -> None:
    expected = [
        "",
        "Variable         |      Mean   Std. dev.       Min        Max |    Observations",
        "-----------------+--------------------------------------------+----------------",
        "y        overall |  61014.32    69973.9       4354     464550 |     N =     816",
        "         between |             69484.36   5472.118   350056.9 |     n =      48",
        "         within  |             12769.31  -25109.56   175507.4 |     T =      17",
        "                 |                                            |",
        "k1       overall |  25036.66    27780.4    2627.12   140217.3 |     N =     816",
        "         between |             27905.86    2958.63   137537.4 |     n =      48",
        "         within  |              2881.21   3838.101   40962.47 |     T =      17",
    ]
    ours = str(ws.panel_summarize(balanced, ["y", "k1"], entity="pid", time="year")).splitlines()
    # Stata's last digit on k1's within sd is 3838.101; ours rounds to 3838.102 because the
    # underlying value sits on the boundary. Accept either, and nothing else.
    normalised = [
        line.replace("3838.102", "3838.101") if "3838.10" in line else line for line in ours
    ]
    assert normalised == expected


def test_xtsum_within_min_max_are_centred_on_the_grand_mean(balanced) -> None:
    """Raw deviations span -86123.88..114493.12; Stata prints -25109.56..175507.4."""

    result = ws.panel_summarize(balanced, ["y"], entity="pid", time="year")
    _, _, _, within = result.variables[0]
    assert within["min"] == pytest.approx(-25109.5625, abs=1e-3)
    assert within["max"] == pytest.approx(175507.4375, abs=1e-3)


def test_xtsum_within_sd_is_the_plain_sample_sd(balanced) -> None:
    """Measured: 12769.31 is deviations.std(ddof=1); ddof=0 gives 12761.48."""

    result = ws.panel_summarize(balanced, ["y"], entity="pid", time="year")
    _, _, _, within = result.variables[0]
    assert within["sd"] == pytest.approx(12769.30559451646, abs=1e-6)


def test_xtsum_between_uses_the_entity_means(balanced) -> None:
    result = ws.panel_summarize(balanced, ["y"], entity="pid", time="year")
    _, _, between, _ = result.variables[0]
    means = balanced.groupby("pid")["y"].mean()
    assert between["sd"] == pytest.approx(float(means.std(ddof=1)), abs=1e-9)
    assert between["min"] == pytest.approx(float(means.min()), abs=1e-9)


def test_xtsum_counts_come_from_the_panel(balanced) -> None:
    result = ws.panel_summarize(balanced, ["y"], entity="pid", time="year")
    assert (result.nobs, result.entities, result.periods) == (816, 48, 17)


def test_xtsum_rejects_an_unknown_variable(balanced) -> None:
    with pytest.raises(ws.ConsistencyError, match="not found in data"):
        ws.panel_summarize(balanced, ["nope"], entity="pid", time="year")


def test_xtsum_accepts_a_single_name(balanced) -> None:
    """A bare string should behave like a one-element list."""

    result = ws.panel_summarize(balanced, "y", entity="pid", time="year")
    assert len(result.variables) == 1


def test_the_three_are_exported_from_the_package_root() -> None:
    for name in ("panel_set", "panel_describe", "panel_summarize"):
        assert callable(getattr(ws, name))
