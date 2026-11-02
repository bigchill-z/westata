"""Scan every registered command's rendering against live Stata output.

Read-only. It renders each command with westata, runs the Stata equivalent through
pystata on the SAME data, and reports the first structural difference. Nothing is fixed
here -- the point is to turn "39 commands unverified" into a classified list, so the
next rounds can be planned instead of guessed at.

Run it with the venv python and Stata available::

    ./.venv/Scripts/python.exe scripts/scan_all_layouts.py

Each entry reports one of:

  MATCH     every line identical (numbers included, since the data is shared)
  SKELETON  same text skeleton, digits differ (usually a genuine numeric difference)
  DIFFERS   structural difference; the first differing line is shown
  RUNFAIL   westata could not fit the spec, so there is nothing to compare
  STATAFAIL Stata rejected the command (wrong spec for this Stata install)
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "replication"))
sys.path.insert(0, str(ROOT / "scripts"))

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import westata as ws  # noqa: E402
from lib.stata_session import StataSession  # noqa: E402


def _skeleton(line: str) -> str:
    """Collapse digits, signs, dots and runs of spaces to a marker.

    Two lines with the same skeleton have the same COLUMNS and the same wording; only
    the numbers differ. That separates a real numeric discrepancy from a layout one.
    """

    return re.sub(r"[-\s.\d,]+", "#", line)


#: Banner lines westata prints that pystata SUPPRESSES. regress and the ML family
#: emit their title through display when run interactively, but stata.run()
#: captures them without it -- so the banner is present on one side and absent on the
#: other for reasons that have nothing to do with the table. Dropping them from BOTH
#: sides is the only comparison that tests the layout rather than the harness.
_BANNER_PREFIXES = (
    "Linear regression",
    "Logistic regression",
    "Probit regression",
    "Poisson regression",
    "Negative binomial regression",
    "Ordered logistic regression",
    "Ordered probit regression",
    "Tobit regression",
    "Truncated regression",
    "Zero-inflated Poisson regression",
    "Zero-inflated negative binomial regression",
)


def _drop_banners(lines: list[str]) -> list[str]:
    """Remove banner lines AND the blank separator that follows each of them."""

    kept: list[str] = []
    skip_blank = False
    for line in lines:
        if line.strip() and line.strip().startswith(_BANNER_PREFIXES):
            skip_blank = True
            continue
        if skip_blank:
            skip_blank = False
            if not line.strip():
                continue
        kept.append(line)
    # Leading/trailing blanks describe the session, not the table.
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return kept


def _compare(stata_lines: list[str], ours: list[str]) -> tuple[str, str]:
    """Return (verdict, detail) for one command."""

    stata_lines = _drop_banners(stata_lines)
    ours = _drop_banners(ours)
    if [l.rstrip() for l in stata_lines] == [l.rstrip() for l in ours]:
        return "MATCH", ""
    state = [l.rstrip() for l in stata_lines]
    west = [l.rstrip() for l in ours]
    limit = min(len(state), len(west))
    for index in range(limit):
        if state[index] != west[index]:
            if _skeleton(state[index]) == _skeleton(west[index]):
                return "SKELETON", f"line {index}: {state[index][:70]!r} vs {west[index][:70]!r}"
            return "DIFFERS", f"line {index}: {state[index][:70]!r} vs {west[index][:70]!r}"
    return "DIFFERS", f"length {len(state)} vs {len(west)}"


# ---------------------------------------------------------------------------
# Shared data. Small panels keep the scan fast; the shapes matter, not the values.
# ---------------------------------------------------------------------------
def _frames() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(4242)
    n_ent, n_per = 20, 5
    n = n_ent * n_per
    panel = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "ybin": (rng.normal(size=n) > 0).astype(int),
            "ycount": np.abs(np.round(rng.normal(size=n) * 3)).astype(int),
            "yord": np.tile([1, 2, 3, 4, 5], n // 5),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "z": rng.normal(size=n),
            "unit": np.repeat(np.arange(n_ent), n_per),
            "period": np.tile(np.arange(n_per), n_ent),
        }
    )
    panel["treated"] = ((panel["unit"] % 2 == 1) & (panel["period"] >= 2)).astype(int)
    cross = panel.drop(columns=["period"]).copy()
    cross["stratum"] = np.tile(np.arange(5), len(cross) // 5)
    cross["psu"] = np.tile(np.arange(20), len(cross) // 20)
    cross["w"] = np.ones(len(cross))
    cross["t"] = np.arange(1, len(cross) + 1)
    # The survey commands need these in the SAME frame they are run on, so they are
    # duplicated onto the panel frame rather than kept only on cross.
    panel["stratum"] = np.tile(np.arange(5), len(panel) // 5)
    panel["psu"] = np.tile(np.arange(20), len(panel) // 20)
    panel["w"] = np.ones(len(panel))
    panel["t"] = np.arange(1, len(panel) + 1)
    return {"panel": panel, "cross": cross}


FRAMES = _frames()
PANEL = FRAMES["panel"]
CROSS = FRAMES["cross"]

DESIGN = ws.SurveyDesign(psu="psu", strata="stratum", weights="w")


# ---------------------------------------------------------------------------
# The command table. Each entry is (label, stata command, westata callable).
# A callable returning a fitted result, or raising, is the contract.
# ---------------------------------------------------------------------------
def _specs():  # noqa: C901 - a flat table is clearer than a hierarchy here
    from westata.estimators.counts import ZeroInflatedPoisson

    rows: list[tuple[str, str, object]] = [
        ("regress", "regress y x1 x2",
         lambda: ws.OLS.from_formula("y ~ x1 + x2", PANEL).fit()),
        ("logit", "logit ybin x1 x2",
         lambda: ws.Logit.from_formula("ybin ~ x1 + x2", PANEL).fit()),
        ("probit", "probit ybin x1 x2",
         lambda: ws.Probit.from_formula("ybin ~ x1 + x2", PANEL).fit()),
        ("poisson", "poisson ycount x1 x2",
         lambda: ws.Poisson.from_formula("ycount ~ x1 + x2", PANEL).fit()),
        ("nbreg", "nbreg ycount x1 x2",
         lambda: ws.NegativeBinomial.from_formula("ycount ~ x1 + x2", PANEL).fit()),
        ("ologit", "ologit yord x1 x2",
         lambda: ws.OrderedLogit.from_formula("yord ~ x1 + x2", PANEL).fit()),
        ("oprobit", "oprobit yord x1 x2",
         lambda: ws.OrderedProbit.from_formula("yord ~ x1 + x2", PANEL).fit()),
        ("tobit", "tobit y x1 x2, ll(-1)",
         lambda: ws.Tobit.from_formula("y ~ x1 + x2", PANEL, lower=-1.0).fit()),
        ("truncreg", "truncreg y x1 x2, ll(-1)",
         lambda: ws.Truncreg.from_formula("y ~ x1 + x2", PANEL, lower=-1.0).fit()),
        ("zip", "zip ycount x1 x2, inflate(x1)",
         lambda: ZeroInflatedPoisson.from_formula(
             "ycount ~ x1 + x2", PANEL, inflate="x1").fit()),
        ("qreg", "qreg y x1, quantile(.5)",
         lambda: ws.QuantileRegression.from_formula("y ~ x1", PANEL).fit()),
        ("regress_robust", "regress y x1 x2, vce(robust)",
         lambda: ws.OLS.from_formula("y ~ x1 + x2", PANEL).fit(cov_type="HC1")),
        ("regress_cluster", "regress y x1 x2, vce(cluster unit)",
         lambda: ws.OLS.from_formula("y ~ x1 + x2", PANEL).fit(
             cov_type="cluster", cov_kwds={"cluster": PANEL["unit"]})),
        ("logit_robust", "logit ybin x1, vce(robust)",
         lambda: ws.Logit.from_formula("ybin ~ x1", PANEL).fit(cov_type="HC1")),
        ("ivregress_2sls", "ivregress 2sls y x1 (x2 = z)",
         lambda: ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z]", PANEL).fit()),
    ]
    return rows


STATA_SETUP = [
    "xtset unit period",
    "svyset psu [pw=w], strata(stratum) singleunit(certainty)",
]


def _stata_block(command: str) -> list[str]:
    """Run one Stata command and return its printable lines.

    Optimisation chatter ("Iteration 0: ...", "(switching optimization to BFGS)") and
    the svyset/xtset echo are dropped: they describe the session, not the table.
    """

    noise = (
        "Iteration ",
        "(switching ",
        "(setting optimization",
        "(running ",
        "Time variable:",
        "Delta:",
        "Panel variable:",
        "VCE:",
        "Single unit:",
        "Strata ",
        "Sampling unit",
        "FPC 1:",
        "Weight:",
        "Balanced",
        "note:",
        "Fitting ",
        "Grid node ",
        "Refining ",
        "Iteration ",
        "(setting ",
        "(switching ",
    )
    with StataSession() as stata:
        stata.push(PANEL)
        for setup in STATA_SETUP:
            with contextlib.redirect_stdout(io.StringIO()):
                stata.run(setup)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stata.run(command)
        text = buffer.getvalue()
    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if not line.startswith(noise)]
    # "(15 obs truncated)" appears at the START of a footer line and is a note about the
    # sample, not part of the table.
    lines = [line for line in lines if "obs truncated" not in line]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def main(argv: list[str]) -> int:
    only = set(argv[1:])
    verdicts: dict[str, list[str]] = {}
    for label, command, build in _specs():
        if only and label not in only:
            continue
        try:
            ours = str(build().summary(style="stata")).splitlines()
        except Exception as exc:  # noqa: BLE001 - report, do not abort the scan
            verdicts.setdefault("RUNFAIL", []).append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        try:
            state = _stata_block(command)
        except Exception as exc:  # noqa: BLE001
            verdicts.setdefault("STATAFAIL", []).append(f"{label}: {exc}")
            continue
        verdict, detail = _compare(state, ours)
        verdicts.setdefault(verdict, []).append(
            f"{label}" + (f"  [{detail}]" if detail else "")
        )

    for verdict in ("MATCH", "SKELETON", "DIFFERS", "RUNFAIL", "STATAFAIL"):
        entries = verdicts.get(verdict, [])
        if not entries:
            continue
        print(f"\n=== {verdict} ({len(entries)}) ===")
        for entry in entries:
            print(f"  {entry}")

    total = sum(len(v) for v in verdicts.values())
    matched = len(verdicts.get("MATCH", [])) + len(verdicts.get("SKELETON", []))
    print(f"\n{matched}/{total} commands render the same table as Stata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
