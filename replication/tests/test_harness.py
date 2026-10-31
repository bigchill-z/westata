"""Tests for the replication harness.

The point of the harness is that a wrong comparison fails loudly instead of
looking like a pass, so the tests exercise the *failure* paths as much as the
happy one.  They create their own tiny frames and synthetic results: anything
that depended on a downloaded archive would stop protecting the harness the
first time the archive moved.

The tests that start the embedded Stata cost a few seconds and need StataNow
installed; they are the only way to prove that the ``e(b)``/``e(V)`` wiring,
missing-value screening, and the ``sysuse auto`` numbers are real rather than
mocked.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from replication.lib.compare import (  # noqa: E402
    AlignmentError,
    Record,
    Row,
    align_parameters,
    compare_capture,
    rel_diff,
    verdict_for,
)
from replication.lib.report import (  # noqa: E402
    Published,
    difference_table,
    replication_table,
    write_markdown,
)
from replication.lib.stata_session import (  # noqa: E402
    CapturedMissing,
    Capture,
    StataError,
    StataSession,
    screen_missing,
)


# --------------------------------------------------------------- test doubles
@dataclass
class FakeResult:
    """Minimal statsmodels-shaped result, so the comparison can be tested alone."""

    params: np.ndarray
    bse: np.ndarray
    param_names: list[str]
    nobs: int
    e: dict[str, float]


def make_capture(
    names: list[str],
    b: list[float],
    se: list[float],
    *,
    nobs: int = 100,
    scalars: dict[str, float] | None = None,
    command: str = "regress",
) -> Capture:
    """Build a Capture whose V is the diagonal of the requested standard errors."""
    b_arr = np.asarray(b, dtype=float)
    V = np.diag(np.square(np.asarray(se, dtype=float)))
    return Capture(
        b=b_arr,
        V=V,
        names=list(names),
        scalars=dict(scalars or {"N": float(nobs)}),
        macros={},
        nobs=nobs,
        command=command,
        data_nobs=nobs,
    )


def make_result(
    names: list[str],
    params: list[float],
    bse: list[float],
    *,
    nobs: int = 100,
    e: dict[str, float] | None = None,
) -> FakeResult:
    return FakeResult(
        params=np.asarray(params, dtype=float),
        bse=np.asarray(bse, dtype=float),
        param_names=list(names),
        nobs=nobs,
        e=dict(e if e is not None else {"N": float(nobs)}),
    )


# ------------------------------------------------------------ live Stata smoke
@pytest.fixture(scope="module")
def stata() -> StataSession:
    """One Stata for the whole module: ``config.init`` runs once per process."""
    try:
        with StataSession() as session:
            session.run("sysuse auto, clear")
            yield session
    except (RuntimeError, ImportError) as exc:  # pragma: no cover - machine dependent
        pytest.skip(f"embedded Stata unavailable: {exc}")


def test_smoke_auto_regression(stata: StataSession) -> None:
    """The harness must reach Stata and read the documented ``sysuse auto`` fit."""
    stata.run("sysuse auto, clear")
    cap = stata.estimate("regress mpg weight foreign")

    assert cap.nobs == 74, f"e(N) was {cap.nobs}"
    assert cap.data_nobs == 74
    assert cap.names == ["weight", "foreign", "_cons"]
    assert cap.b.shape == (3,)
    assert cap.V is not None and cap.V.shape == (3, 3)

    weight_b = cap.b[cap.names.index("weight")]
    assert weight_b == pytest.approx(-0.0065879, abs=1e-6)
    assert cap.get("r2") == pytest.approx(0.6627029, abs=1e-6)
    assert not np.isnan(cap.se).any()


def test_factor_base_level_is_captured_but_not_counted(stata: StataSession) -> None:
    """``i.foreign`` adds ``0b.foreign`` with a zero coefficient.

    Keeping it in the capture and excluding it at alignment time is what makes
    the exclusion auditable; dropping it during capture would hide it.
    """
    stata.run("sysuse auto, clear")
    cap = stata.estimate("regress mpg weight i.foreign")
    assert "0b.foreign" in cap.names and "1.foreign" in cap.names
    base = cap.names.index("0b.foreign")
    assert cap.b[base] == 0.0

    names, si, wi = align_parameters(cap.names, ["Intercept", "weight", "foreign[T.1]"])
    # Order follows Stata's own output, and the Stata indices point at the
    # non-base parameters.
    assert names == ["weight", "foreign=1", "__const__"]
    assert [cap.names[index] for index in si] == ["weight", "1.foreign", "_cons"]
    assert wi == [1, 2, 0]


def test_run_raises_with_stata_return_code(stata: StataSession) -> None:
    """A Stata error must surface as StataError carrying ``r(###)``."""
    with pytest.raises(StataError) as info:
        stata.run("summarize definitely_not_a_variable")
    assert info.value.rcode == 111
    assert "definitely_not_a_variable" in str(info.value)


def test_capture_without_estimation_raises(stata: StataSession) -> None:
    """No ``e(b)`` is an error, not an empty capture to compare against.

    This also pins the stale-results trap: Stata keeps a previous estimation's
    ``e()`` alive across non-estimating commands, so ``estimate`` must clear it
    rather than hand back the earlier model's coefficients.
    """
    stata.run("sysuse auto, clear")
    stata.run("quietly regress mpg weight")
    stata.run("summarize mpg")
    with pytest.raises(StataError, match="no estimation results in memory"):
        stata.estimate("summarize mpg")


def test_push_pull_round_trips_and_screens_missing(stata: StataSession) -> None:
    """Missing values must arrive and leave as NaN, never as 8.988e307."""
    frame = pd.DataFrame({"y": [1.0, 2.0, 4.0], "x": [0.5, np.nan, 2.5], "g": [1, 1, 2]})
    stata.push(frame)
    assert stata.nobs() == 3

    back = stata.pull()
    assert math.isnan(back.loc[1, "x"])
    assert back.loc[0, "x"] == 0.5

    stata.run("quietly summarize x")
    assert stata.scalar("r(N)") == 2.0


def test_push_rejects_illegal_stata_names(stata: StataSession) -> None:
    """A silently mangled variable name would fail later, with the wrong message."""
    with pytest.raises(ValueError, match="not legal Stata variable names"):
        stata.push(pd.DataFrame({"not a name": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="empty"):
        stata.push(pd.DataFrame())


def test_output_captures_printed_table(stata: StataSession) -> None:
    stata.run("sysuse auto, clear")
    text = stata.output("summarize mpg")
    assert "Variable" in text and "mpg" in text and "Std. dev." in text


def test_load_dta_reports_missing_file(stata: StataSession) -> None:
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        stata.load_dta(_REPO_ROOT / "replication" / "definitely, missing.dta")


# ------------------------------------------------------- comparison semantics
def test_compare_flags_identical_result() -> None:
    cap = make_capture(
        ["d_traderes_pw_lag", "l_shind_manuf_cbp", "t2", "_cons"],
        [-0.289323, -0.085116, -0.6955438, 6.2017],
        [0.0429709, 0.0103684, 0.3202903, 1.737597],
        nobs=1444,
        scalars={"N": 1444.0, "r2": 0.4789},
        command="reg d_sh_empl_mfg d_traderes_pw_lag ... [aw=timepwt48], cluster(statefip)",
    )
    result = make_result(
        ["Intercept", "d_traderes_pw_lag", "l_shind_manuf_cbp", "t2"],
        [6.2017, -0.289323, -0.085116, -0.6955438],
        [1.737597, 0.0429709, 0.0103684, 0.3202903],
        nobs=1444,
        e={"N": 1444.0, "r2": 0.4789},
    )

    record = compare_capture("tab5-col1", cap, result, scalars=("r2",))

    assert record.verdict == "IDENTICAL"
    assert record.max_b_rel <= 1e-9
    assert record.stata_n == record.westata_n == 1444
    assert record.n_params == 4
    assert record.row("__const__").stata_name == "_cons"
    assert record.row("__const__").westata_name == "Intercept"
    assert record.notes == []


def test_compare_flags_perturbed_result() -> None:
    """A 1e-4 change is invisible to the eye and must not read as IDENTICAL."""
    cap = make_capture(["x", "_cons"], [0.25, 1.5], [0.05, 0.2], scalars={"N": 500.0})
    identical = make_result(["Intercept", "x"], [1.5, 0.25], [0.2, 0.05], nobs=500)
    perturbed = make_result(["Intercept", "x"], [1.5, 0.250025], [0.2, 0.05], nobs=500)

    assert compare_capture("same", cap, identical).verdict == "IDENTICAL"

    record = compare_capture("perturbed", cap, perturbed)
    assert record.verdict != "IDENTICAL"
    assert record.verdict == "MINOR_DIFF"
    assert record.max_b_rel == pytest.approx(1e-4, rel=1e-6)


def test_compare_detects_standard_error_drift() -> None:
    """Coefficients can match while the VCE is wrong; the record must say so."""
    cap = make_capture(["x", "_cons"], [0.25, 1.5], [0.05, 0.2])
    result = make_result(["Intercept", "x"], [1.5, 0.25], [0.2, 0.075], nobs=100)

    record = compare_capture("vce", cap, result)
    assert record.verdict == "IDENTICAL"  # coefficients agree
    assert record.max_se_rel == pytest.approx(0.5, rel=1e-9)


def test_length_mismatch_guard_raises() -> None:
    """Two vectors of different length must never be differenced by position."""
    cap = make_capture(["x", "z", "_cons"], [0.25, 0.1, 1.5], [0.05, 0.02, 0.2])
    shorter = make_result(["Intercept", "x"], [1.5, 0.25], [0.2, 0.05])

    with pytest.raises(AlignmentError) as info:
        compare_capture("mismatched", cap, shorter)
    assert info.value.unmatched_stata == ["z"]
    assert info.value.unmatched_westata == []
    assert "only in Stata: z" in str(info.value)

    # A westata result carrying a parameter Stata never estimated.  (A real
    # result cannot list both ``Intercept`` and ``_cons``; they are the same
    # parameter under two names.)
    extra = make_result(
        ["Intercept", "x", "z", "surprise"],
        [1.5, 0.25, 0.1, 1.0],
        [0.2, 0.05, 0.02, 0.1],
    )
    with pytest.raises(AlignmentError) as extra_info:
        compare_capture("extra", cap, extra)
    assert extra_info.value.unmatched_westata == ["surprise"]


def test_align_parameters_raises_on_unmatched_single_parameter() -> None:
    with pytest.raises(AlignmentError):
        align_parameters(["x"], ["y"])


def test_mapping_resolves_a_genuine_label_difference() -> None:
    """``mapping`` is explicit: the harness never guesses a rename.

    Stata stores a log-likelihood scale as ``var(e.y)`` while westata reports
    ``/sigma2``; nothing in either name says so, which is exactly why the
    caller has to state it.
    """
    cap = make_capture(["var(e.y)", "/sigma"], [2.25, 1.5], [0.1, 0.1])
    result = make_result(["/sigma", "/sigma2"], [1.5, 2.25], [0.1, 0.1])

    with pytest.raises(AlignmentError):
        compare_capture("unmapped", cap, result)

    record = compare_capture("mapped", cap, result, mapping={"var(e.y)": "/sigma2"})
    assert record.verdict == "IDENTICAL"
    assert {row.name for row in record.rows} == {"/sigma", "/sigma2"}


def test_missing_coefficient_is_not_a_small_difference() -> None:
    """A NaN coefficient must not be folded into the max as if it were zero."""
    cap = make_capture(["x", "_cons"], [np.nan, 1.5], [0.05, 0.2])
    result = make_result(["Intercept", "x"], [1.5, 0.25], [0.2, 0.05])

    record = compare_capture("missing", cap, result)
    assert math.isnan(record.max_b_rel)
    assert record.verdict == "NOT_COMPARABLE"


def test_screen_missing_converts_sentinels() -> None:
    """A raw 8.988e307 crossing the boundary becomes NaN, not a huge estimate."""
    screened = screen_missing(np.array([1.0, CapturedMissing.SENTINEL, 3.0]))
    assert screened[0] == 1.0 and math.isnan(screened[1]) and screened[2] == 3.0

    frame = screen_missing(pd.DataFrame({"a": [1.0, CapturedMissing.SENTINEL], "g": [1, 2]}))
    assert math.isnan(frame.loc[1, "a"])
    assert str(frame["g"].dtype) == "int64", "integer columns must not be upcast"


def test_sample_size_mismatch_is_recorded() -> None:
    cap = make_capture(["x", "_cons"], [0.25, 1.5], [0.05, 0.2], nobs=1444)
    result = make_result(["Intercept", "x"], [1.5, 0.25], [0.2, 0.05], nobs=722)

    record = compare_capture("n", cap, result)
    assert not record.n_agree
    assert any("sample size differs" in note for note in record.notes)


def test_rel_diff_and_verdict_thresholds() -> None:
    assert rel_diff([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert rel_diff([1.0], [1.0, 2.0]) == float("inf")
    assert rel_diff([1.0 + 5e-13], [1.0]) <= 1e-9
    assert verdict_for(0.0) == "IDENTICAL"
    assert verdict_for(1e-8) == "NUMERICALLY_EQUAL"
    assert verdict_for(1e-4) == "MINOR_DIFF"
    assert verdict_for(1e-1) == "MATERIAL_DIFF"
    assert verdict_for(float("nan")) == "NOT_COMPARABLE"
    assert verdict_for(float("inf")) == "MATERIAL_DIFF"


def test_compare_rejects_a_non_result_object() -> None:
    cap = make_capture(["x"], [1.0], [0.1])
    with pytest.raises(TypeError, match="statsmodels-like"):
        compare_capture("bad", cap, {"params": [1.0]})


# ------------------------------------------------------------- report output
def _report_fixture() -> list[Record]:
    cap = make_capture(
        ["d_traderes_pw_lag", "_cons"],
        [-0.289323, 6.2017],
        [0.0429709, 1.737597],
        nobs=1444,
        scalars={"N": 1444.0, "r2": 0.4789},
    )
    result = make_result(
        ["Intercept", "d_traderes_pw_lag"],
        [6.2017, -0.289323],
        [1.737597, 0.0429709],
        nobs=1444,
        e={"N": 1444.0, "r2": 0.4789012345},
    )
    return [compare_capture("tab5-a1", cap, result, scalars=("r2",))]


def test_replication_table_has_three_sides() -> None:
    records = _report_fixture()
    published = [Published(name="tab5-a1", estimate=-0.2893, se=0.0430, n=1444, r2=0.4789)]
    text = replication_table(records, published)

    assert "paper est." in text and "Stata est." in text and "westata est." in text
    assert "-0.2893" in text  # the paper's own rounding survives
    assert "1444" in text
    assert "tab5-a1" in text


def test_replication_table_reports_r2_from_both_sides() -> None:
    text = replication_table(_report_fixture(), [])
    assert "0.4789" in text  # Stata's e(r2)


def test_published_row_must_exist() -> None:
    records = _report_fixture()
    published = [Published(name="tab5-a1", row="no_such_parameter")]
    with pytest.raises(ValueError, match="not a coefficient of record"):
        replication_table(records, published)


def test_difference_table_flags_non_identical_verdicts() -> None:
    identical = _report_fixture()
    cap = make_capture(["x", "_cons"], [0.25, 1.5], [0.05, 0.2])
    diff = compare_capture(
        "drifted", cap, make_result(["Intercept", "x"], [1.5, 0.3], [0.2, 0.05])
    )

    text = difference_table(identical + [diff])
    assert "**MATERIAL_DIFF**" in text
    assert "2 specification(s)" in text


def test_write_markdown_creates_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.md"
    records = _report_fixture()
    written = write_markdown(
        target,
        records,
        [Published(name="tab5-a1", estimate=-0.2893, se=0.0430, n=1444)],
    )
    assert written.exists()
    body = written.read_text(encoding="utf-8")
    assert body.startswith("# Replication report")
    assert "### Estimates" in body and "### Implementation differences" in body
    with pytest.raises(ValueError, match="no records"):
        write_markdown(target, [])


def test_record_round_trips_to_dict() -> None:
    record = _report_fixture()[0]
    payload = record.to_dict()
    assert payload["verdict"] == "IDENTICAL"
    # Stata's output order, so the record reads like the published table.
    assert [row["name"] for row in payload["rows"]] == ["d_traderes_pw_lag", "__const__"]
    assert payload["scalars"]["r2"]["westata"] == pytest.approx(0.4789012345)
    # ``b_abs`` is derived, so the payload is a superset of the constructor's
    # fields; rebuilding a Row from it must still succeed.
    rebuilt = Row(**{key: value for key, value in payload["rows"][0].items() if key != "b_abs"})
    assert rebuilt.stata_b == pytest.approx(-0.289323)


def test_align_parameters_preserves_stata_output_order() -> None:
    """patsy alphabetises and moves the intercept first; Stata writes ``_cons`` last.

    The record must follow Stata, otherwise a published table cannot be read
    across from the harness output.
    """
    names, si, wi = align_parameters(
        ["b", "a", "_cons"],
        ["Intercept", "a", "b"],
    )
    assert names == ["b", "a", "__const__"]
    assert si == [0, 1, 2]
    assert wi == [2, 1, 0]


def test_allow_unmatched_stata_is_explicit_and_reported() -> None:
    """A coefficient westata does not estimate is excluded only when named."""
    cap = make_capture(["x", "absorbed", "_cons"], [0.25, 0.1, 1.5], [0.05, 0.02, 0.2])
    result = make_result(["Intercept", "x"], [1.5, 0.25], [0.2, 0.05])

    with pytest.raises(AlignmentError):
        compare_capture("undeclared", cap, result)

    record = compare_capture("declared", cap, result, allow_unmatched_stata=["absorbed"])
    assert record.verdict == "IDENTICAL"
    assert record.dropped == ["absorbed"]
    assert any("absorbed" in note for note in record.notes)
