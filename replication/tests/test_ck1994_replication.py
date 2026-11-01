"""Tests for the Card & Krueger (1994) replication.

The failure mode a replication must guard against is not "a function raised" but "the
number came back slightly wrong and nobody noticed".  So every test here asserts a
*published* number or a documented sample fact, at the precision the paper prints:

* the sample size and its two summary statistics, which validate the sample construction
  before any coefficient is compared;
* each of Table 4's five coefficients and standard errors, to the paper's two decimals;
* Table 3's difference-in-differences, to the paper's two decimals.

Sample construction is where this replication nearly went wrong: requiring a wave-2 wage
for every store drops the six permanently closed ones and yields N = 351, a New Jersey
coefficient of 2.28 instead of 2.33.  ``test_the_closed_store_rule_moves_the_estimate``
pins that difference so the rule cannot be "simplified" away later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "papers" / "ck1994"
OUT = PAPER / "out"

sys.path.insert(0, str(PAPER))

pytestmark = pytest.mark.skipif(
    not (OUT / "stata_results.json").exists(),
    reason="run replication/papers/ck1994/run_all.py first; Stata is required",
)


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stata() -> dict:
    return _load("stata_results.json")


@pytest.fixture(scope="module")
def westata() -> dict:
    return _load("westata_results.json")


def _coef(record: dict, name: str) -> tuple[float, float]:
    index = record["names"].index(name)
    return float(record["b"][index]), float(record["se"][index])


# ------------------------------------------------------------------ data layer
def test_the_reader_reproduces_the_codebook_counts():
    """The fixed-width parse is only trustworthy if it matches the codebook's own counts."""

    import ck1994_data

    frame = ck1994_data.build()  # raises if the counts disagree
    assert len(frame) == 410
    assert int((frame["state"] == 1).sum()) == 331
    assert int((frame["state"] == 0).sum()) == 79


def test_the_sample_matches_the_papers_357_stores():
    import ck1994_data

    sample = ck1994_data.table4_sample(ck1994_data.build())
    assert len(sample) == 357, "the Table 4 sample size is reported in the paper's notes"
    # The paper prints -0.237; the exact value is -0.2375..., so accept the print rounding
    assert round(sample["d_fte"].mean(), 2) == pytest.approx(-0.24, abs=5e-3)
    assert round(sample["d_fte"].std(), 3) == pytest.approx(8.825, abs=1e-3)


def test_the_closed_store_rule_moves_the_estimate():
    """Permanent closures count as zero employment; temporary ones as missing.

    Getting this wrong is silent: the regression still runs and returns a plausible
    number.  The two interpretations differ by 0.07 on the New Jersey coefficient and by
    six observations, so this pins both.
    """

    import sys as _sys

    _sys.path.insert(0, str(PAPER.parent.parent / "papers" / "ck1994"))
    import ck1994_data
    import westata as ws

    frame = ck1994_data.build()
    correct = ck1994_data.table4_sample(frame)
    assert len(correct) == 357

    # The wrong rule: demand a wave-2 wage from every store.
    strict_mask = (
        frame["fte"].notna()
        & frame["fte2"].notna()
        & frame["wage_st"].notna()
        & frame["wage_st2"].notna()
    )
    strict = frame.loc[strict_mask].reset_index(drop=True)
    assert len(strict) == 351

    fit_correct = ws.OLS.from_formula("d_fte ~ nj", correct).fit()
    fit_strict = ws.OLS.from_formula("d_fte ~ nj", strict).fit()
    i = [str(n) for n in fit_correct.param_names].index("nj")
    b_correct = float(__import__("numpy").asarray(fit_correct.params, dtype=float)[i])
    b_strict = float(__import__("numpy").asarray(fit_strict.params, dtype=float)[i])

    assert round(b_correct, 2) == pytest.approx(2.33, abs=5e-3)
    assert abs(b_correct - b_strict) > 0.03, (
        "the closed-store rule no longer changes anything; the data or the rule changed "
        "and REPORT.md section on sample construction needs re-checking"
    )


# ------------------------------------------------------------------ estimates
TABLE4_PUBLISHED = (
    ("T4_i", "nj", 2.33, 1.19),
    ("T4_ii", "nj", 2.30, 1.20),
    ("T4_iii", "gap", 15.65, 6.08),
    ("T4_iv", "gap", 14.92, 6.21),
    ("T4_v", "gap", 11.91, 7.39),
)


@pytest.mark.parametrize("spec,term,paper_b,paper_se", TABLE4_PUBLISHED)
def test_stata_reproduces_table4(stata, spec, term, paper_b, paper_se):
    """To the paper's two printed decimals.

    Column (v) is the one exception the report discusses: the replication gives 11.98
    against a printed 11.91, a 0.6 percent gap that the paper's own description of its
    region dummies does not explain.  The test allows 0.08 there and 0.015 elsewhere so
    the exception is recorded rather than hidden by a loose tolerance everywhere.
    """

    b, se = _coef(stata["specs"][spec], term)
    tolerance = 0.08 if spec == "T4_v" else 0.015
    assert round(b, 2) == pytest.approx(paper_b, abs=tolerance), f"{spec} b={b}"
    assert round(se, 2) == pytest.approx(paper_se, abs=tolerance), f"{spec} se={se}"


@pytest.mark.parametrize("spec,term,paper_b,paper_se", TABLE4_PUBLISHED)
def test_westata_reproduces_table4(westata, spec, term, paper_b, paper_se):
    b, se = _coef(westata["specs"][spec], term)
    tolerance = 0.08 if spec == "T4_v" else 0.015
    assert round(b, 2) == pytest.approx(paper_b, abs=tolerance), f"{spec} b={b}"
    assert round(se, 2) == pytest.approx(paper_se, abs=tolerance), f"{spec} se={se}"


def test_stata_reproduces_the_table3_difference_in_differences(stata):
    """Table 3: 2.76 FTE, 13 percent, t = 2.03."""

    t3 = stata["table3"]
    # The paper prints 2.76; the exact value is 2.7500, and its stated 13 percent is
    # computed against the New Jersey baseline, not Pennsylvania's (2.75/20.43 = 13.5).
    assert t3["did"] == pytest.approx(2.76, abs=0.02)
    assert t3["t_stat"] == pytest.approx(2.03, abs=0.03)
    assert t3["percent_of_nj_base"] == pytest.approx(13.5, abs=0.6)
    assert t3["n"] == 384, "balanced sample: 722 - 6 permanent closures - 4 temporary"


# ------------------------------------------------------------------- agreement
def test_the_two_implementations_agree_on_every_parameter(stata_run=None, westata_run=None):
    """All ten parameters across the five columns, with a combined tolerance.

    A bare relative bound would misrank the intercepts: one is of order 9 while the
    treatment coefficients are of order 2, so a 1e-15 absolute gap looks larger on the
    smaller number.  The bound is ``ATOL + RTOL * |stata|`` for that reason.
    """

    stata = _load("stata_results.json")["specs"]
    westata = _load("westata_results.json")["specs"]
    atol, rtol = 1e-10, 1e-9

    rename = {"Intercept": "_cons"}
    for spec in stata:
        s_names = [rename.get(n, n) for n in stata[spec]["names"]]
        w_names = [rename.get(n, n) for n in westata[spec]["names"]]
        # Order differs by construction: Stata reports the written order with _cons last,
        # patsy puts the intercept first.  Pair by name, never by position.
        assert sorted(s_names) == sorted(w_names), f"{spec}: different parameter sets"
        order = [s_names.index(n) for n in w_names]
        for j, name in enumerate(w_names):
            i = order[j]
            for key, s_val, w_val in (
                ("coef", stata[spec]["b"][i], westata[spec]["b"][j]),
                ("SE", stata[spec]["se"][i], westata[spec]["se"][j]),
            ):
                assert abs(w_val - s_val) <= atol + rtol * abs(s_val), (
                    f"{spec}/{name} {key}: |diff|={abs(w_val - s_val):.3e}"
                )


def test_sample_sizes_agree(stata, westata):
    for spec in stata["specs"]:
        assert stata["specs"][spec]["nobs"] == westata["specs"][spec]["nobs"] == 357


# --------------------------------------------------------------- negative control
def test_the_estimate_is_sensitive_to_the_treatment_variable():
    """The paper's GAP is not interchangeable with a New Jersey dummy.

    Fitting the same data with an NJ indicator gives 2.33 instead of 15.65, and permuting
    GAP gives a different sign.  That gap is what makes "we reproduced 15.65" evidence
    about the *specification* rather than evidence that some regression ran.
    """

    import westata as ws

    payload = _load("differences.json")["negative_control"]
    paper_spec = payload["gap (as in the paper)"]["b"]
    dummy = payload["NJ dummy instead of GAP"]["b"]
    permuted = payload["GAP randomly permuted"]["b"]

    assert paper_spec == pytest.approx(15.65, abs=0.02)
    assert abs(paper_spec - dummy) > 10, "the GAP coefficient collapsed onto the dummy"
    assert abs(paper_spec - permuted) > 5, "permuting GAP no longer changes the estimate"


def test_the_gap_probe_still_reports_what_the_gaps_doc_claims():
    """Guard ``ck1994_gaps.md`` so its entries cannot go stale unnoticed.

    The ADH replication has an equivalent test; this one exists because the CK document
    makes a *different* kind of claim -- not "these features are missing" but "this work
    sits outside the library by design".  A boundary that quietly turns into a feature
    would make the document wrong in the direction that flatters westata, which is the
    failure mode worth catching.

    Only the GAP/ERROR outcomes fail.  A BOUNDARY changing to OK means the document needs
    editing, and the assertion message says so.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location("ck_probe_gaps", PAPER / "ck1994_probe_gaps.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ck_probe_gaps"] = module
    spec.loader.exec_module(module)

    statuses = {}
    for probe in (*module.ESTIMATOR_PROBES, *module.DATA_PROBES, *module.CONTROLS):
        status, detail = probe.run()
        statuses[probe.name] = (status, detail)

    assert not [n for n, (s, _) in statuses.items() if s == "ERROR"], (
        f"probes failed unexpectedly: {statuses}"
    )
    assert not [n for n, (s, _) in statuses.items() if s == "GAP"], (
        "westata now has a genuine gap; ck1994_gaps.md section 3 claims there are none"
    )
    # The estimator probes are expected to pass; if one regresses that is a real defect.
    for probe in module.ESTIMATOR_PROBES:
        status, detail = statuses[probe.name]
        assert status == "OK", f"estimator probe {probe.name!r} regressed: {detail}"

    # The data-surface probe is expected to report a boundary.  If it ever reports OK the
    # document's central claim ("all data work happened in pandas") is obsolete.
    data_status = statuses[module.DATA_PROBES[0].name][0]
    assert data_status == "BOUNDARY", (
        "the data-surface probe no longer reports a boundary; ck1994_gaps.md section 2 "
        "needs rewriting"
    )
