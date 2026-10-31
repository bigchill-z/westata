"""Tests for the Autor & Dorn (2013) replication.

Every assertion here is against the **authors' own log** (``log/czone_analysis.log``,
shipped inside the archive), which records seven decimals.  That makes this replication
unusually tightly checkable: agreement is asserted at 1e-6, not against a printed table
where three decimals is the limit.

The sample sizes are asserted too, because they are where a specification error shows up
first -- the four distinct N values (675, 660, 1335, 2166, 722) each correspond to a
different sample restriction, and getting one wrong changes the coefficient silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "papers" / "ad2013"
OUT = PAPER / "out"

sys.path.insert(0, str(PAPER))

pytestmark = pytest.mark.skipif(
    not (OUT / "stata_results.json").exists(),
    reason="run replication/papers/ad2013/run_all.py first; Stata is required",
)


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stata() -> dict:
    return _load("stata_results.json")["specs"]


@pytest.fixture(scope="module")
def westata() -> dict:
    return _load("westata_results.json")["specs"]


def _coef(record: dict, name: str) -> tuple[float, float]:
    index = record["names"].index(name)
    return float(record["b"][index]), float(record["se"][index])


#: ``(spec id, coefficient, standard error, N)`` from the authors' log.
PUBLISHED = (
    ("T3_1", 0.6950717, 0.0614371, 675),
    ("T3_2", 0.4899328, 0.0760501, 660),
    ("T3_3", 0.6185140, 0.0441480, 1335),
    ("T3_4", -0.2540723, 0.0225534, 2166),
    ("T3_5", -0.1526445, 0.0244188, 2166),
    ("T3_6", -0.2945519, 0.0181593, 2166),
    ("T4_1", -0.1333048, 0.0203481, 722),
    ("T4_2", 0.0415499, 0.0315114, 722),
    ("T4_3", 0.0825400, 0.0279456, 722),
    ("T4_4", 0.0843262, 0.0454320, 722),
    ("T4_5", 0.3540265, 0.1100271, 722),
)


@pytest.mark.parametrize("spec,paper_b,paper_se,paper_n", PUBLISHED)
def test_stata_reproduces_the_authors_log(stata, spec, paper_b, paper_se, paper_n):
    """StataNow on the delivered data must land on the authors' own seven decimals."""

    b, se = _coef(stata[spec], "l_sh_routine33a")
    assert stata[spec]["nobs"] == paper_n, f"{spec}: N={stata[spec]['nobs']}, log says {paper_n}"
    assert b == pytest.approx(paper_b, abs=1e-6), f"{spec}: b={b}"
    assert se == pytest.approx(paper_se, abs=1e-6), f"{spec}: se={se}"


@pytest.mark.parametrize("spec,paper_b,paper_se,paper_n", PUBLISHED)
def test_westata_reproduces_the_authors_log(westata, spec, paper_b, paper_se, paper_n):
    b, se = _coef(westata[spec], "l_sh_routine33a")
    assert westata[spec]["nobs"] == paper_n
    assert b == pytest.approx(paper_b, abs=1e-6), f"{spec}: b={b}"
    assert se == pytest.approx(paper_se, abs=1e-6), f"{spec}: se={se}"


def test_the_two_implementations_agree_on_every_parameter():
    """All ~50 parameters per specification, including the 48 state dummies.

    Comparing only the reported coefficient would leave the state fixed effects
    unverified -- and a regression that got the treatment right while dropping a state
    dummy is a different model.  Pairing is by canonical name (``4.statefip``,
    ``_Istatefip_4`` and ``st_4`` are the same parameter).
    """

    import re

    stata = _load("stata_results.json")["specs"]
    westata = _load("westata_results.json")["specs"]
    atol, rtol = 1e-10, 1e-9

    patterns = (
        re.compile(r"^(-?\d+)\.statefip$"),
        re.compile(r"^_Istatefip_(-?\d+)$"),
        re.compile(r"^st_(-?\d+)$"),
    )

    def canon(name: str) -> str:
        if name == "Intercept":
            return "_cons"
        for pattern in patterns:
            match = pattern.match(name)
            if match:
                return f"st_{match.group(1)}"
        return name

    for spec in stata:
        s_names = [canon(n) for n in stata[spec]["names"]]
        w_names = [canon(n) for n in westata[spec]["names"]]
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


def test_sample_sizes_agree_between_the_two_sides(stata, westata):
    for spec in stata:
        assert stata[spec]["nobs"] == westata[spec]["nobs"], spec


def test_the_state_dummies_matter():
    """Negative control: dropping the state fixed effects must move the estimate.

    Without this, matching the paper would only show that *a* regression ran.  The
    control refits the same dependent variable and treatment without the 48 state
    dummies; the coefficient falls from 0.695 to 0.600, i.e. the fixed effects are doing
    real work and the agreement on the full specification is informative.
    """

    payload = _load("differences.json")["negative_control"]
    with_dummies = payload["with state dummies (as in the paper)"]["b"]
    without = payload["without state dummies"]["b"]

    assert with_dummies == pytest.approx(0.6950717, abs=1e-6)
    assert abs(with_dummies - without) > 0.05, (
        "the state dummies no longer change the estimate, so this replication is no "
        "longer testing the specification the paper describes"
    )


def test_the_generated_do_file_lists_every_specification():
    """The reader-runnable artefact and the compared numbers cover the same models."""

    import importlib.util

    spec = importlib.util.spec_from_file_location("ad2013_specs", PAPER / "specs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ad2013_specs"] = module
    spec.loader.exec_module(module)

    do_file = OUT / "ad2013_replication.do"
    if not do_file.exists():
        pytest.skip("run gen_do.py first")
    text = do_file.read_text(encoding="utf-8")
    for item in module.SPECS:
        assert item.stata_command() in text, f"{item.id} missing from the do-file"


def test_data_provenance_is_pinned():
    provenance = PAPER / "data" / "provenance.json"
    if not provenance.exists():
        pytest.skip("run acquire.py first")
    record = json.loads(provenance.read_text(encoding="utf-8"))
    assert record["archive"]["sha256"] == (
        "a4c9b8200a39d357c1141cfe3cb8b54b7ed586dcf9768e3b1c6689d79c45be0b"
    )
    assert record["archive"]["url"].startswith("https://www.ddorn.net/")


def test_the_gap_probe_now_reports_every_defect_as_fixed():
    """Guard ``WESTATA_GAPS.md``: the three contract gaps were fixed, so they must stay fixed.

    This test used to run the other way: the document recorded defects that were NOT fixed,
    so it failed if the probes started passing. They do pass now, and the guard inverts with
    them -- a regression that brings any of the three back should fail here rather than
    quietly contradicting the document.

    The three were:

    * ``aligned weights`` -- a weight vector aligned with ``frame`` was rejected whenever the
      formula dropped rows, with a message that read like malformed data.
    * ``weights by name`` -- ``weights="col"`` was unsupported, which is what makes
      statsmodels' dropna handling transparent.
    * ``documentation`` -- the docstring never said rows are dropped.

    The control still matters, for the same reason as before: it proves the ESTIMATES were
    always right, so what changed is the contract, not the arithmetic.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location("ad2013_probe_gaps", PAPER / "probe_gaps.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ad2013_probe_gaps"] = module
    spec.loader.exec_module(module)

    statuses = {probe.name: probe.run()[0] for probe in module.PROBES}
    assert "ERROR" not in statuses.values(), f"probes failed unexpectedly: {statuses}"
    assert statuses == {"aligned weights": "OK", "weights by name": "OK", "documentation": "OK"}, (
        f"a contract defect came back: {statuses}. WESTATA_GAPS.md §1-2 say all three are "
        "fixed, and section 4's workaround is no longer required by the library."
    )

    controls = {probe.name: probe.run()[0] for probe in module.CONTROLS}
    assert controls == {"pre-aligned call": "OK"}, (
        f"the control regressed: {controls}. The defect and a wrong estimate are no "
        "longer distinguishable, which is what made the gap reportable in the first place"
    )
