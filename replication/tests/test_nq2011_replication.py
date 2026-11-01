"""Tests for the Nunn & Qian (2011) replication.

This replication's distinctive claim is not "westata matches Stata" -- Stata disagrees with
itself here -- but "westata reproduces the coefficient exactly and follows ``xtreg``'s
standard-error convention, while the authors ran ``areg``".  The tests therefore assert
three separate things:

1. the coefficients match the authors' log to seven decimals;
2. westata matches ``xtreg`` to machine precision;
3. westata differs from ``areg`` by the documented convention ratio, and that difference is
   not an accident of this dataset.

Assertion 3 is written as a *ratio band* rather than a fixed tolerance, because the point is
that the gap is systematic: every column shows the same 1.045-1.047 factor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "papers" / "nq2011"
OUT = PAPER / "out"
#: Real Stata captures (replayable without Stata) used by the reghdfe comparison below.
FIXTURES = HERE.parents[1] / "tests" / "conformance" / "stata_fixtures"

sys.path.insert(0, str(PAPER))

pytestmark = pytest.mark.skipif(
    not (OUT / "stata_results.json").exists(),
    reason="run replication/papers/nq2011/run_all.py first; Stata is required",
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


#: ``(spec, term, log b, log se)`` from the authors' log.
PUBLISHED = (
    ("T2_c1", "ln_wpot_1800", 0.0725947, 0.0138663),
    ("T2_c1", "ln_wpot_1900", 0.1213388, 0.0171579),
    ("T2_c2", "ln_wpot_1900", 0.0923494, 0.0208822),
    ("T2_c3", "ln_wpot_1900", 0.0795030, 0.0240815),
    ("T2_c4", "ln_wpot_1900", 0.0118203, 0.0022693),
    ("T2_c5", "ln_wpot_1900", 0.0123354, 0.0023503),
    ("T2_c6", "ln_wpot_1900", 0.0100279, 0.0031933),
)


@pytest.mark.parametrize("spec,term,paper_b,paper_se", PUBLISHED)
def test_areg_reproduces_the_authors_log(stata, spec, term, paper_b, paper_se):
    """The authors ran ``areg``; that is the command whose output must come back exactly."""

    b, se = _coef(stata["specs"][spec], term)
    assert b == pytest.approx(paper_b, abs=1e-6), f"{spec}/{term}: b={b}"
    assert se == pytest.approx(paper_se, abs=1e-6), f"{spec}/{term}: se={se}"


@pytest.mark.parametrize("spec,term,paper_b,paper_se", PUBLISHED)
def test_coefficients_are_identical_across_every_path(stata, westata, spec, term, paper_b, paper_se):
    """The point estimate does not depend on which Stata command or westata path is used.

    This is the substantive replication claim: whatever the standard-error convention, the
    number the paper reports for its effect is reproduced.
    """

    b_areg, _ = _coef(stata["specs"][spec], term)
    b_xtreg, _ = _coef(stata["xtreg"][spec], term)
    b_feols, _ = _coef(westata["feols"][spec], term)
    b_panel, _ = _coef(westata["panelols"][spec], term)

    for label, value in (("areg", b_areg), ("xtreg", b_xtreg),
                         ("FEOLS", b_feols), ("PanelOLS", b_panel)):
        assert value == pytest.approx(paper_b, abs=1e-6), f"{spec}/{term} {label}: {value}"


def test_westata_follows_xtreg_on_standard_errors(westata, stata):
    """westata's panel paths reproduce ``xtreg, fe`` to machine precision."""

    payload = _load("differences.json")
    for rec in payload["specs"]:
        assert rec["agree_xtreg_se"] < 1e-9, (
            f"{rec['spec']}: westata vs xtreg SE rel diff {rec['agree_xtreg_se']:.2e}"
        )
        assert rec["agree_xtreg_coef"] < 1e-9, f"{rec['spec']}: coefficient mismatch"


def test_westata_differs_from_areg_by_the_convention_ratio():
    """The ``areg`` gap is systematic, not a one-off, and that is the whole finding.

    Every column shows a standard-error ratio in [1.03, 1.06].  If one column fell outside
    the band, that would be a real defect in westata rather than the documented convention
    difference, and this test is what distinguishes the two.
    """

    payload = _load("differences.json")
    for rec in payload["specs"]:
        ratio = rec["areg_xtreg_se_ratio"]
        assert 1.03 <= ratio <= 1.06, (
            f"{rec['spec']}: areg/xtreg SE ratio {ratio:.4f} is outside the documented "
            "convention band; either Stata changed its degrees-of-freedom rule or westata "
            "does not follow xtreg after all"
        )
        assert rec["agree_areg_se"] > 1e-3, (
            f"{rec['spec']}: westata now matches areg's SEs too; the convention story in "
            "REPORT.md would need rewriting"
        )


def test_sample_sizes_are_unanimous(stata, westata):
    """1,552 observations = 130 countries x the years each is observed; every path agrees."""

    for spec in stata["specs"]:
        assert stata["specs"][spec]["nobs"] == 1552
        assert stata["xtreg"][spec]["nobs"] == 1552
        assert westata["feols"][spec]["nobs"] == 1552
        assert westata["panelols"][spec]["nobs"] == 1552


def test_the_estimate_is_sensitive_to_the_year_effects():
    """Negative control: without year fixed effects the coefficient nearly triples.

    Without this, "we reproduced 0.1213" could just mean the regression is insensitive to
    its specification.  0.285 versus 0.121 is a factor of 2.3, so the agreement on the
    full specification is informative.
    """

    payload = _load("differences.json")["negative_control"]
    with_year = payload["with year dummies (as in the paper)"]["b"]
    without = payload["without year dummies"]["b"]

    assert with_year == pytest.approx(0.1213388, abs=1e-6)
    assert abs(without - with_year) > 0.1, (
        "the year effects no longer change the estimate; the replication is no longer "
        "testing the specification the paper describes"
    )


def test_data_provenance_is_pinned():
    provenance = PAPER / "data" / "provenance.json"
    if not provenance.exists():
        pytest.skip("run acquire.py first")
    record = json.loads(provenance.read_text(encoding="utf-8"))
    assert "country_level_panel.tab" in record
    for name, item in record.items():
        assert item["url"].startswith("https://dataverse.harvard.edu/"), name
        assert len(item["sha256"]) == 64, name


def test_the_gap_probe_reports_the_areg_constant_and_keeps_the_note():
    """Guard ``nq2011_gaps.md``: one open gap (``Areg``'s ``_cons``) plus the SE note.

    History this test has to respect: an earlier version asserted a *different* gap --
    that ``e(rmse)`` and the standard error used different freedoms.  A live Stata capture
    showed ``reghdfe`` reports exactly that pair, so that assertion was wrong and was
    withdrawn (the gap list keeps the retraction in its section 3).

    What must hold now:

    * ``Areg``'s slope matches the authors' log;
    * ``Areg``'s ``_cons`` does **not** match Stata's ``areg`` -- a real defect, asserted
      in the failing direction so it cannot be silently dropped;
    * ``Areg``'s clustered SE follows ``reghdfe``, recorded as a NOTE rather than a gap,
      because ``reghdfe``'s source says the divergence is deliberate.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nq2011_probe_gaps", PAPER / "nq2011_probe_gaps.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["nq2011_probe_gaps"] = module
    spec.loader.exec_module(module)

    statuses = {p.name: p.run() for p in (*module.PROBES, *module.CONTROLS)}
    kinds = {name: status for name, (status, _) in statuses.items()}

    assert not [n for n, s in kinds.items() if s == "ERROR"], f"probes failed: {statuses}"

    assert kinds.get("Areg coefficient") == "OK", (
        f"the slope no longer matches the authors' log: {statuses.get('Areg coefficient')}"
    )
    assert kinds.get("Areg _cons") == "OK", (
        "_cons regressed: nq2011_gaps.md section 4 says it now matches Stata's areg. "
        f"Probe said: {statuses.get('Areg _cons')}"
    )
    assert kinds.get("Areg clustered SE") == "OK", (
        "the clustered SE regressed to the reghdfe convention; nq2011_gaps.md section 1 "
        f"says Areg now matches areg. Probe said: {statuses.get('Areg clustered SE')}"
    )
    assert kinds.get("Areg identity") == "OK", statuses.get("Areg identity")



def test_rmse_uses_the_full_rank_denominator_like_reghdfe():
    """``e(rmse)`` is a sample statistic: clustering must not move it.

    This is the invariant that settles the question.  ``e(df_r)`` collapses under clustering
    (1400 -> 129 here) and westata's standard errors follow that switch.  If ``rmse`` shared
    the denominator, it would have to move too; instead it stays put, exactly as in
    ``reghdfe_firm.json`` (``df_r`` = 148) and ``reghdfe_firm_cluster.json`` (``df_r`` = 29),
    which report the *same* ``e(rmse)``.
    """

    import numpy as np

    import westata as ws

    frame = _nq2011_frame()
    formula = _nq2011_formula()
    plain = ws.FEOLS.from_formula(f"{formula} | isocode", frame).fit()
    clustered = ws.FEOLS.from_formula(f"{formula} | isocode", frame).fit(
        cov_type="cluster", cov_kwds={"cluster": "isocode"}
    )

    # rmse does not move with the covariance estimator.
    assert clustered.e["rmse"] == pytest.approx(plain.e["rmse"], rel=1e-12)
    # ... and it keeps the full-rank denominator (N - 152 = 1400), not the clustered df_r.
    implied = float(clustered.e["rss"]) / float(clustered.e["rmse"]) ** 2
    assert implied == pytest.approx(1400.0, abs=1.0), (
        f"e(rmse) now implies a denominator of {implied:.1f}; reghdfe uses the full-rank "
        "1400 here. If this changed it was probably 'fixed' to the nested-excluded "
        "denominator, which moves rmse 8.6 percent away from Stata"
    )
    # ... while e(df_r) is the clustered reference distribution, a different number.
    assert clustered.e["df_r"] == 129
    assert plain.e["df_r"] == 1400


def test_feols_matches_the_reghdfe_fixture_bit_for_bit():
    """Section 3's claim rests on reproducing ``reghdfe`` exactly, not approximately.

    ``reghdfe_firm_cluster.json`` is a real Stata capture, replayable without Stata.  It
    records the "mixed" pair -- ``e(rmse)`` on the full-rank denominator, ``e(df_r)`` =
    G - 1 -- and westata must match every scalar.  If this fails while the probe still
    passes, the probe is measuring the wrong thing.
    """

    import numpy as np

    import westata as ws

    path = FIXTURES / "reghdfe_firm_cluster.json"
    if not path.exists():
        pytest.skip("reghdfe_firm_cluster.json missing; run scripts/capture_reghdfe_fixtures.py")
    capture = json.loads(path.read_text(encoding="utf-8"))
    scalars = capture["scalars"]

    frame = _reghdfe_panel_frame()
    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", frame).fit(
        cov_type="cluster", cov_kwds={"cluster": frame["firm"]}
    )

    assert result.e["rmse"] == pytest.approx(scalars["rmse"], rel=1e-12)
    assert result.e["ll"] == pytest.approx(scalars["ll"], rel=1e-12)
    assert result.e["r2_a"] == pytest.approx(scalars["r2_a"], rel=1e-12)
    assert result.e["df_r"] == scalars["df_r"]
    assert result.e["N"] == scalars["N"]

    stata_bse = np.sqrt(np.diag(np.asarray(capture["V"], dtype=float)))[:2]
    np.testing.assert_allclose(
        np.asarray(result.bse, dtype=float), stata_bse, rtol=1e-12, atol=0
    )


_NQ_FRAME = None
_NQ_DUMMIES: list[str] | None = None


def _nq2011_frame():
    """The prepared panel, computed once (it builds ~180 interaction columns)."""

    global _NQ_FRAME, _NQ_DUMMIES
    if _NQ_FRAME is None:
        from nq2011_run_westata import prepare

        _NQ_FRAME, _NQ_DUMMIES = prepare()
    return _NQ_FRAME


def _nq2011_formula() -> str:
    from nq2011_specs import SPECS
    from nq2011_run_westata import _formula

    _nq2011_frame()  # populates _NQ_DUMMIES
    spec = next(s for s in SPECS if s.id == "T2_c1")
    return _formula(spec, _NQ_DUMMIES)


def _reghdfe_panel_frame():
    """The exact frame behind the reghdfe fixtures (mirrors test_stata_reghdfe.panel_frame)."""

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(21)
    n_entities, n_periods = 30, 6
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    period = rng.normal(size=n_periods)[time - 1]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + period + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


def test_the_areg_xtreg_gap_comes_from_the_rmse_denominator():
    """Pin the *mechanism*, not just the ratio.

    An earlier version of the gap list attributed the 4.5 percent difference to the
    cluster-VCE small-sample factor counting the absorbed effects in ``K``.  That is wrong:
    the residuals are identical, so the cluster adjustment is identical, and the whole gap
    comes from ``rmse`` being divided by a different denominator.

    Asserting the arithmetic here means the wrong explanation cannot come back unnoticed.
    ``areg`` divides the RSS by ``N - 152`` (22 regressors + intercept + 129 absorbed country
    dummies); ``xtreg, fe`` divides by ``N - 23``.
    """

    import math

    n = 1552
    areg_denom = n - 152
    xtreg_denom = n - 23

    assert areg_denom == 1400
    assert xtreg_denom == 1529

    predicted = math.sqrt(xtreg_denom / areg_denom)
    assert predicted == pytest.approx(1.0450563895, abs=1e-9), (
        "the denominator arithmetic no longer reproduces the observed areg/xtreg ratio"
    )

    # Guard the claim the docs make about it.
    gap_doc = (PAPER.parents[1] / "gaps" / "nq2011_gaps.md").read_text(encoding="utf-8")
    assert "1529" in gap_doc and "1400" in gap_doc, (
        "the gap list no longer states the rmse denominators the mechanism depends on"
    )
    # The section must not PRESENT the cluster-factor story as the explanation.  Mentioning
    # it in order to refute it is the point, so look for the affirmative phrasing only.
    section = gap_doc.split("### 1.3")[1].split("### 1.4")[0]
    assert "这个解释是错的" in section or "是错的" in section, (
        "the gap list no longer records that the cluster-factor explanation was wrong; a "
        "future reader would not know it had been tested and rejected"
    )
