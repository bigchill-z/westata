"""End-to-end tests for the ADH (2013) replication, run against the real data.

These are deliberately *not* unit tests of the driver's helpers.  A replication's
failure mode is not "a function raised" -- it is "the number came back slightly wrong and
nobody noticed".  So every test here asserts a published, externally verifiable number:

* the paper's printed coefficient, to its printed precision, from StataNow;
* the paper's printed coefficient, to its printed precision, from westata;
* agreement between the two implementations at a tolerance that is tight enough
  (1e-9 relative) that a genuinely different estimator or weighting could not pass.

The tolerance choice is the point.  ``1e-6`` would pass a replication that used
``N - k`` instead of ``N`` somewhere, or that dropped the weighted constant; the
observed differences are ~1e-15 to ~1e-11, so there is three to six orders of magnitude
of headroom before a real defect hides.  Tightening to 1e-12 would fail on column 6's
1.6e-11, which is float64 summation order rather than a defect -- so the bound is 1e-9
and the reason is written down rather than tuned.

The suite needs Stata and the 3 MB archive; both are skipped with an explicit reason
rather than silently passing when unavailable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "papers" / "adh2013"
OUT = PAPER / "out"

sys.path.insert(0, str(PAPER))

pytestmark = pytest.mark.skipif(
    not (OUT / "stata_results.json").exists(),
    reason="run replication/papers/adh2013/run_all.py first; Stata is required",
)

#: Tolerance for "the same number computed twice", as ``ATOL + RTOL * |stata|``.
#:
#: The absolute floor is not slack for a defect: it exists because one first-stage
#: coefficient in column 4 is 1.6e-06, and two correct float64 evaluations of a
#: regression that small differ in the fifteenth significant digit, which is a large
#: *relative* number and a physically meaningless one.  1e-12 is still six orders of
#: magnitude below the paper's own smallest printed digit (1e-3).
ATOL = 1e-12
RTOL = 1e-9


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stata_run() -> dict:
    return _load("stata_results.json")


@pytest.fixture(scope="module")
def stata(stata_run) -> dict:
    return stata_run["specs"]


@pytest.fixture(scope="module")
def westata_run() -> dict:
    return _load("westata_results.json")


@pytest.fixture(scope="module")
def westata(westata_run) -> dict:
    return westata_run["specs"]


def _coef(record: dict, name: str) -> tuple[float, float]:
    index = record["names"].index(name)
    return float(record["b"][index]), float(record["se"][index])


#: Every column of Table 3's panel I, as the paper prints it, plus Table 2's stacked
#: column.  ``(spec id, coefficient, standard error)``.
TABLE3_PUBLISHED = (
    ("T3_c1", -0.746, 0.068),
    ("T3_c2", -0.610, 0.094),
    ("T3_c3", -0.538, 0.091),
    ("T3_c4", -0.508, 0.081),
    ("T3_c5", -0.562, 0.096),
    ("T3_c6", -0.596, 0.099),
)


@pytest.mark.parametrize("spec,paper_b,paper_se", TABLE3_PUBLISHED)
def test_stata_reproduces_the_published_table3(stata, spec, paper_b, paper_se):
    """StataNow on the delivered data must land on the paper's printed three decimals.

    This is the test that makes the rest meaningful: if the Stata side drifted from the
    paper, agreement between Stata and westata would only prove they are wrong together.
    """

    b, se = _coef(stata[spec], "d_tradeusch_pw")
    assert round(b, 3) == pytest.approx(paper_b, abs=5e-4), f"{spec}: b={b}"
    assert round(se, 3) == pytest.approx(paper_se, abs=5e-4), f"{spec}: se={se}"


@pytest.mark.parametrize("spec,paper_b,paper_se", TABLE3_PUBLISHED)
def test_westata_reproduces_the_published_table3(westata, spec, paper_b, paper_se):
    b, se = _coef(westata[spec], "d_tradeusch_pw")
    assert round(b, 3) == pytest.approx(paper_b, abs=5e-4), f"{spec}: b={b}"
    assert round(se, 3) == pytest.approx(paper_se, abs=5e-4), f"{spec}: se={se}"


def test_stata_reproduces_table2_period_by_period(stata):
    """Table 2's separate-period columns are the only place the sample is not 1,444."""

    expected = {"T2_c1": (-0.89, 0.18, 722), "T2_c2": (-0.72, 0.06, 722),
                "T2_c3": (-0.75, 0.07, 1444)}
    for spec, (paper_b, paper_se, nobs) in expected.items():
        b, se = _coef(stata[spec], "d_tradeusch_pw")
        assert stata[spec]["nobs"] == nobs
        assert round(b, 2 if nobs == 1444 and spec == "T2_c3" else 2) == pytest.approx(
            paper_b, abs=5e-3
        ), f"{spec}: b={b}"
        assert round(se, 2) == pytest.approx(paper_se, abs=5e-3), f"{spec}: se={se}"


def test_stata_reproduces_the_reduced_form_column(stata):
    """Table 10 column 1 is the aweighted-OLS anchor for westata's native weight path."""

    b, se = _coef(stata["OLS_c6"], "d_traderes_pw_lag")
    assert round(b, 3) == pytest.approx(-0.289, abs=5e-4)
    assert round(se, 3) == pytest.approx(0.043, abs=5e-4)


def test_the_two_implementations_agree_on_every_column(stata_run, westata_run):
    """Every paired parameter on both sides, with a combined relative/absolute tolerance.

    A replication that only checks the headline coefficient would miss a wrong control
    set, because the controls move the headline too.  Comparing every parameter -- 89
    structural and 72 first-stage -- means one misaligned regressor fails the test.

    The bound is ``|a - b| <= ATOL + RTOL * |b|`` rather than a bare relative one, and the
    reason is a specific coefficient rather than convenience: in the first stage of
    column 4, ``reg_pacif`` is 1.56e-06 on both sides, i.e. numerically zero.  The two
    runs differ there by 2.9e-15 *in absolute terms* -- a smaller discrepancy than the
    1e-13 seen on coefficients of order one -- but dividing by 1.6e-6 turns it into a
    relative difference of 1.8e-9.  A pure relative test would name the least meaningful
    parameter in the table as the worst agreement, which is precisely backwards.
    """

    for group in ("specs", "first_stage"):
        stata = stata_run[group]
        westata = westata_run[group]
        for spec in stata:
            if spec.endswith("_robust"):
                continue
            s_names = stata[spec]["names"]
            w_names = westata[spec]["names"]
            assert s_names == w_names, f"{group}/{spec}: different parameter sets"
            for index, name in enumerate(s_names):
                s_b, s_se = stata[spec]["b"][index], stata[spec]["se"][index]
                w_b, w_se = westata[spec]["b"][index], westata[spec]["se"][index]
                for key, s_val, w_val in (("coef", s_b, w_b), ("SE", s_se, w_se)):
                    gap = abs(w_val - s_val)
                    bound = ATOL + RTOL * abs(s_val)
                    assert gap <= bound, (
                        f"{group}/{spec}/{name} {key}: |westata-stata|={gap:.3e} "
                        f"exceeds {bound:.3e} (stata={s_val!r}, westata={w_val!r})"
                    )


def test_sample_sizes_agree(stata, westata):
    for spec in stata:
        assert stata[spec]["nobs"] == westata[spec]["nobs"], spec
    assert stata["T3_c6"]["nobs"] == 1444
    assert stata["T2_c1"]["nobs"] == 722


def test_fit_statistics_agree(stata, westata):
    """R-squared and RMSE, on Stata's scale.

    The westata IV path reaches aweighted 2SLS through a square-root rescaling, which
    puts its *native* fit statistics on the transformed scale.  The driver recomputes
    them on the original scale for exactly this comparison; if that recomputation were
    dropped, this test is what would catch it.
    """

    for spec in stata:
        for key in ("r2", "rmse"):
            s_val = stata[spec]["scalars"].get(key)
            w_val = westata[spec]["scalars"].get(key)
            if s_val is None or w_val is None:
                continue
            rel = abs(w_val - s_val) / max(abs(s_val), 1e-12)
            assert rel < 1e-9, f"{spec}: e({key}) rel diff {rel:.3e}"


def test_first_stage_replicates_panel_two(stata_run, westata_run):
    """Panel II: the instrument's coefficient, on both sides and against the paper."""

    published = {"T3_c1": 0.792, "T3_c2": 0.664, "T3_c3": 0.652,
                 "T3_c4": 0.635, "T3_c5": 0.638, "T3_c6": 0.631}
    for spec, paper_b in published.items():
        s_b, s_se = _coef(stata_run["first_stage"][spec], "d_tradeotch_pw_lag")
        w_b, w_se = _coef(westata_run["first_stage"][spec], "d_tradeotch_pw_lag")
        assert round(s_b, 3) == pytest.approx(paper_b, abs=5e-4), spec
        assert abs(w_b - s_b) / abs(s_b) < 1e-9, spec
        assert abs(w_se - s_se) / abs(s_se) < 1e-9, spec


def test_the_published_first_stage_standard_error_is_robust_not_clustered():
    """A finding, pinned as a test so a future Stata version cannot quietly change it.

    Table 3 panel II prints 0.079 / 0.086 / 0.090 / 0.090 / 0.087 / 0.087.  Re-running the
    same regression with ``cluster(statefip)`` gives 0.080 / 0.089 / 0.094 / 0.094 / 0.091 /
    0.091 -- which would change every printed digit.  With ``vce(robust)`` it gives
    0.0787 / 0.0864 / 0.0904 / 0.0896 / 0.0866 / 0.0870, matching the paper.

    The attribution is *not* that the authors mis-transcribed.  Their own archived log
    (``log/czone_analysis_ipw_final.log`` in the delivered archive) records 0.0787565 /
    0.0864398 / 0.0903940 / 0.0896515 / 0.0866208 / 0.0870084 for exactly these six
    columns, i.e. the robust family to ~1e-4 relative, and 2e-2 away from the clustered
    values.  So the numbers the paper prints are what Stata printed to them: the era's
    ``ivregress ..., first`` reported the first-stage panel with a robust covariance even
    when the structural equation was clustered.  StataNow has changed that convention.

    The point of the test is to keep the *attribution* honest and to fail loudly if it
    stops holding -- not to adjudicate the paper.
    """

    first_stage = _load("stata_results.json")["first_stage"]
    clustered_b, clustered_se = _coef(first_stage["T3_c6"], "d_tradeotch_pw_lag")
    robust_b, robust_se = _coef(first_stage["T3_c6_robust"], "d_tradeotch_pw_lag")

    assert clustered_b == pytest.approx(robust_b, rel=1e-9)
    # The published value follows the robust covariance, not the clustered one.
    assert round(robust_se, 3) == pytest.approx(0.087, abs=5e-4)
    assert round(clustered_se, 3) == pytest.approx(0.091, abs=5e-4)
    assert abs(clustered_se - robust_se) / robust_se > 1e-2, (
        "the clustered and robust first-stage SEs have converged; the attribution in "
        "REPORT.md section 5.2 no longer applies and must be re-checked"
    )


def test_ols_counterpart_replicates_the_published_attenuation(westata):
    """Appendix Table 4 panel A column 1: the paper's own OLS benchmark, -0.171.

    This is the only specification in the replication that reaches a *published* number
    through westata's native analytic-weight path (`OLS.from_formula(..., weights=...)`)
    rather than through the square-root rescaling the IV columns need. If the aweight
    handling in westata were wrong, this row and the 2SLS rows would have to disagree in
    opposite directions to both match the paper -- so a single test covers both.
    """

    b, se = _coef(westata["OLS_T4A_c1"], "d_tradeusch_pw")
    assert round(b, 3) == pytest.approx(-0.171, abs=5e-4), f"b={b}"
    assert round(se, 3) == pytest.approx(0.028, abs=5e-4), f"se={se}"


def test_iv_and_ols_do_not_look_alike(westata):
    """The negative control: a replication that fitted the wrong estimator cannot pass.

    2SLS gives -0.596 and OLS gives -0.171 for the *same* regressors on the same data --
    a factor of 3.5 apart. That gap is what makes "we reproduced -0.596" evidence about
    the estimator rather than merely evidence that some regression ran. If a future
    change made ``IV2SLS`` silently fall back to OLS, or made the square-root rescaling
    drop the weighting, ``test_the_two_implementations_agree_on_every_column`` would still
    pass against a Stata capture of the same mistake -- but this assertion would fail,
    because the two published numbers cannot both come out of one estimator.
    """

    iv_b, _ = _coef(westata["T3_c6"], "d_tradeusch_pw")
    ols_b, _ = _coef(westata["OLS_T4A_c1"], "d_tradeusch_pw")
    assert abs(iv_b) / abs(ols_b) == pytest.approx(3.5, abs=0.1), (
        f"2SLS={iv_b}, OLS={ols_b}: the estimator gap has collapsed, so this "
        "replication no longer distinguishes 2SLS from OLS"
    )


def test_the_capability_gap_list_is_still_accurate():
    """The report's section 6.4 claims must not silently outlive their evidence.

    Each entry there is a probe in ``probe_gaps.py``.  If a future westata closes one of
    those gaps, this test fails and forces the report to be corrected -- which is the
    point.  A stale "westata cannot do X" is as damaging to a reader as a false claim
    that it can: it deters work that would now succeed.

    The control probe (two-way clustering on HDFE, which *is* supported) runs too.  If it
    reports a gap, the probe is measuring the wrong thing and every other row is
    suspect -- so a control failure is an error, never a pass.
    """

    import probe_gaps

    statuses = {gap.name: gap.run()[0] for gap in probe_gaps.GAPS}
    stale = [name for name, status in statuses.items() if status == "FIXED"]
    errors = [name for name, status in statuses.items() if status == "ERROR"]

    assert not errors, f"probes failed unexpectedly: {errors}"
    assert not stale, (
        f"westata now supports {stale}; REPORT.md section 6.4 must be updated, because "
        "the report currently tells readers these are missing"
    )

    # Controls assert capabilities the report *does* credit.  One of them guards a
    # mistake an earlier draft actually made: it claimed bootstrap lacked
    # ``reps``/``seed`` after reading the internal base class instead of the public
    # entry point.  A gap list is only trustworthy if what it omits is checked too.
    for control in probe_gaps.CONTROLS:
        status, detail = control.run()
        assert status == "FIXED", (
            f"control {control.name!r} failed ({status}: {detail}); either the capability "
            "regressed, or the probes are measuring the wrong thing"
        )


def test_the_generated_do_file_lists_every_specification():
    """The human-runnable artefact and the machine comparison must cover the same models."""

    from specs import SPECS

    do_file = OUT / "adh2013_replication.do"
    if not do_file.exists():
        pytest.skip("run gen_do.py first")
    text = do_file.read_text(encoding="utf-8")
    for spec in SPECS:
        assert spec.stata_command() in text, f"{spec.id} missing from the generated do-file"


def test_data_provenance_is_pinned():
    """A replication is only worth its data snapshot; the hash is the snapshot's name."""

    provenance = PAPER / "data" / "provenance.json"
    if not provenance.exists():
        pytest.skip("run data.py first")
    record = json.loads(provenance.read_text(encoding="utf-8"))
    assert record["archive"]["sha256"] == "1b56c028a75fba784b14477cdf2e89de6ae90a443b9b1b9648179d852048df12"
    assert record["archive"]["bytes"] == 2_953_516
    assert record["archive"]["url"].startswith("https://www.ddorn.net/")
