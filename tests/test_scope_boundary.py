"""Scope boundary: which missing commands are out of scope, and which have no reference.

Round 133 audited the Stata command names the report mentions but that have no top-level export in
the package, to see whether round 132's dfuller/pperron aliases were an isolated case or the start
of a pattern. They were isolated, and the reason is worth recording because it stops the next audit
from re-litigating these names.

OUT OF SCOPE BY THE REPORT'S OWN STATEMENT. Its scope line reads:

    范围：只覆盖「估计 / 检验 / 后估计 / 结果呈现」。按你的要求，数据管理
    （gen/drop/keep/sort/reshape/merge）与绘图（twoway/scatter/kdensity/coefplot/irf graph）
    不在评估范围内。

so the following, all absent, are not gaps:

    coefplot   kdensity   rdplot   marginsplot   twoway   graph

`rdplot` deserves a note: it is 955 readable lines of ado, and could in principle be ported, but its
body is graph construction -- `twoway (scatter ...)`, `twoway (rcap ...)`, `twoway (rarea ...)`. It
is a plotting command whatever else it computes, and plotting is excluded.

ALSO ABSENT AND NOT PRESENT IN THIS STATA INSTALL:

    rddensity        `r(199)` command unrecognised; no ado under ~/ado or the base path

so there is no reference output to check an implementation against. That is the same state as
`xtserial` (round 112) and it ends the question rather than deferring it.

WHAT IS PRESENT AND DOES HAVE A DIFFERENT NAME, verified this round:

    tebalance summarize  ->  ws.tebalance_summarize(...)     already implemented and documented
    esttab               ->  ws.etable(...) / ws.etable_as_stata(...)

Neither needs an alias urgently; `etable_as_stata` was already the subject of round 98 and the
`tebalance` mapping is recorded in the register's own table.
"""

from __future__ import annotations

import pytest

import westata as ws


def test_the_plotting_family_is_out_of_scope() -> None:
    """Absent, and deliberately so -- the report excludes plotting by name."""

    for name in ("coefplot", "kdensity", "rdplot", "marginsplot"):
        assert not hasattr(ws, name), name


def test_the_data_management_family_is_out_of_scope() -> None:
    for name in ("gen", "drop", "keep", "reshape"):
        assert not hasattr(ws, name), name


def test_the_result_presentation_names_that_do_exist() -> None:
    """esttab's role is filled by etable/etable_as_stata; tebalance by tebalance_summarize."""

    assert hasattr(ws, "etable")
    assert hasattr(ws, "etable_as_stata")
    assert hasattr(ws, "tebalance_summarize")


def test_unit_root_names_are_the_ones_that_were_fixed() -> None:
    """Round 132's aliases, contrasted with the plotting names that stay absent."""

    assert hasattr(ws, "dfuller")
    assert hasattr(ws, "pperron")
    assert hasattr(ws, "unitroot")
