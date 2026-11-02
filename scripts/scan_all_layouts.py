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
    "Robust regression",
    "Heckman selection model",
    "Interval regression",
    "Errors-in-variables regression",
    "Simultaneous quantile regression",
    "Median regression",
)


def _drop_banners(lines: list[str]) -> list[str]:
    """Remove banner lines AND the blank separator that follows each of them."""

    kept: list[str] = []
    skip_blank = False
    for line in lines:
        # EXACT match, not startswith: "Linear regression" is a prefix of
        # "Linear regression, correlated panels corrected standard errors (PCSEs)",
        # which is xtpcse's real banner and must survive the filter. Using startswith
        # deleted it and shifted every following row, making xtpcse look broken.
        if line.strip() and line.strip() in _BANNER_PREFIXES:
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
    # Collapse interior runs of blanks to a single one: the chatter left empty lines
    # behind, and every following row then compared against the wrong line.
    collapsed: list[str] = []
    for line in kept:
        if not line.strip() and collapsed and not collapsed[-1].strip():
            continue
        collapsed.append(line)
    return collapsed


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
    # Heckman needs a binary selection indicator that is NOT degenerate; intreg needs a
    # (low, high) interval pair whose low < high everywhere.
    panel["selected"] = (panel["x1"] + panel["x2"] > 0).astype(int)
    # The zero-TRUNCATED models reject observations at the limit, so they need counts that
    # are strictly positive. ycount has zeros; ypos does not.
    panel["ypos"] = panel["ycount"] + 1
    interval = panel.copy()
    interval["lo"] = panel["y"] - 0.5
    interval["hi"] = panel["y"] + 0.5

    tsseries = pd.DataFrame(
        {"y": rng.normal(size=n), "t": np.arange(1, n + 1)}
    )
    return {
        "panel": panel,
        "cross": cross,
        "interval": interval,
        "tsseries": tsseries,
    }


FRAMES = _frames()
PANEL = FRAMES["panel"]
CROSS = FRAMES["cross"]
SELECTED = FRAMES["panel"]
INTERVAL = FRAMES["interval"]
TSSERIES = FRAMES["tsseries"]

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
        # --- second batch: the families not covered by the first scan ---
        ("heckman", "heckman y x1, select(selected = x1 x2)",
         lambda: ws.Heckman.from_formula(
             "y ~ x1 + select(selected ~ x1 + x2)", SELECTED).fit()),
        ("intreg", "intreg lo hi x1",
         lambda: ws.Intreg.from_formula("lo hi ~ x1", INTERVAL).fit()),
        ("etregress", "etregress y x1, treat(treated = x1 x2)",
         lambda: ws.Etregress.from_formula(
             "y ~ x1 + treat(treated ~ x1 + x2)", PANEL).fit()),
        ("sqreg", "sqreg y x1, quantiles(.25 .5 .75)",
         lambda: ws.Sqreg.from_formula("y ~ x1", PANEL).fit()),
        ("bsqreg", "bsqreg y x1, quantile(.5)",
         lambda: ws.Bsqreg.from_formula("y ~ x1", PANEL).fit()),
        ("rreg", "rreg y x1",
         lambda: ws.RobustRegression.from_formula("y ~ x1", PANEL).fit()),
        ("eivreg", "eivreg y x1 x2, reliab(x1 0.7)",
         lambda: ws.Eivreg.from_formula("y ~ x1 + x2", PANEL,
                                        reliab={"x1": 0.7}).fit()),
        ("xtlogit_fe", "xtlogit ybin x1, fe",
         lambda: ws.XtLogit.from_formula("ybin ~ x1", PANEL, entity="unit",
                                         method="fe").fit()),
        ("svy_mean", "svy: mean y",
         lambda: ws.SvyMean.from_frame(["y"], CROSS, DESIGN).fit()),
        ("svy_total", "svy: total y",
         lambda: ws.SvyTotal.from_frame(["y"], CROSS, DESIGN).fit()),
        ("svy_logit", "svy: logit ybin x1",
         lambda: ws.SvyLogit.from_frame(CROSS, "ybin ~ x1", DESIGN).fit()),
        ("xtpcse", "xtpcse y x1",
         lambda: ws.XtPcse.from_formula("y ~ x1", PANEL, entity="unit",
                                        time="period").fit()),
        ("xtgls", "xtgls y x1, panels(heteroskedastic)",
         lambda: ws.XtGls.from_formula("y ~ x1", PANEL, entity="unit",
                                       time="period",
                                       panels="heteroskedastic").fit()),
        ("xtreg_fe", "xtreg y x1, fe",
         lambda: ws.PanelOLS.from_formula("y ~ x1", PANEL, entity="unit",
                                          time="period").fit()),
        ("didregress", "didregress (y x1) (treated), group(unit) time(period)",
         lambda: ws.Didregress.from_formula(
             "y", "treated", PANEL, group="unit", time="period",
             covariates=["x1"]).fit()),
        # --- third batch: everything else the registry knows about ---
        ("arima", "arima y, arima(1,0,0)",
         lambda: ws.ARIMA(TSSERIES["y"].to_numpy(), order=(1, 0, 0)).fit()),
        ("frontier", "frontier y x1",
         lambda: ws.Frontier.from_formula("y ~ x1", PANEL).fit()),
        ("nl", "nl (y = {b0} * exp(-{b1} * x1))",
         lambda: ws.Nl.from_formula(
             "y = {b0} * exp(-{b1} * x1)", PANEL,
             start={"b0": 1.0, "b1": 0.1}).fit()),
        ("reghdfe", "reghdfe y x1, absorb(unit)",
         lambda: ws.FEOLS.from_formula("y ~ x1", PANEL).fit()),
        ("tnbreg", "tnbreg ypos x1, ll(0)",
         lambda: ws.TruncatedNegativeBinomial.from_formula(
             "ypos ~ x1", PANEL, lower=0).fit()),
        ("zinb", "zinb ycount x1, inflate(x1)",
         lambda: ws.ZeroInflatedNegativeBinomial.from_formula(
             "ycount ~ x1", PANEL, inflate="x1").fit()),
        # Stata's ``ztp`` truncates at ZERO, which on the positive-count scale means the
        # lower limit is 1 -- passing lower=0 selects the general ``tpoisson`` instead and
        # the banners differ.
        ("ztp", "ztp ypos x1",
         lambda: ws.TruncatedPoisson.from_formula(
             "ypos ~ x1", PANEL, lower=1.0).fit()),
        ("xtdidregress", "xtdidregress (y x1) (treated), group(unit) time(period)",
         lambda: ws.Xtdidregress.from_formula(
             "y", "treated", PANEL, group="unit", time="period",
             covariates=["x1"]).fit()),
        ("xtivreg_fe", "xtivreg y x1 (x2 = z), fe",
         lambda: ws.XtIvreg.from_formula(
             "y ~ x1 + (x2 = z)", PANEL, entity="unit", time="period",
             method="fe").fit()),
        ("xtlogit_re", "xtlogit ybin x1, re",
         lambda: ws.XtLogit.from_formula(
             "ybin ~ x1", PANEL, entity="unit", method="re").fit()),
        ("xtlogit_pa", "xtlogit ybin x1, pa",
         lambda: ws.XtLogit.from_formula(
             "ybin ~ x1", PANEL, entity="unit", method="pa").fit()),
        ("xtnbreg_fe", "xtnbreg ycount x1, fe",
         lambda: ws.XtNbreg.from_formula(
             "ycount ~ x1", PANEL, entity="unit", method="fe").fit()),
        ("xtpoisson_fe", "xtpoisson ycount x1, fe",
         lambda: ws.XtPoisson.from_formula(
             "ycount ~ x1", PANEL, entity="unit", method="fe").fit()),
        ("xtprobit_re", "xtprobit ybin x1, re",
         lambda: ws.XtProbit.from_formula(
             "ybin ~ x1", PANEL, entity="unit", method="re").fit()),
        ("xttobit", "xttobit y x1, ll(-1)",
         lambda: ws.XtTobit.from_formula(
             "y ~ x1", PANEL, entity="unit", lower=-1.0).fit()),
        ("xtreg_re", "xtreg y x1, re",
         lambda: ws.RandomEffects.from_formula(
             "y ~ x1", PANEL, entity="unit", time="period").fit()),
        ("xtreg_be", "xtreg y x1, be",
         lambda: ws.BetweenOLS.from_formula(
             "y ~ x1", PANEL, entity="unit", time="period").fit()),
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
        # NOTE: "Time variable:", "Panel variable:", "Strata", "Sampling unit", "VCE:",
        # "Single unit:", "FPC 1:", "Weight:" and "Balanced" are NOT filtered here. They
        # look like xtset/svyset echo, but the echo comes out of the SETUP calls (which
        # write to a separate buffer) -- while xtreg, xtpcse and didregress print the same
        # labels as real header content. Filtering them deleted rows from those headers
        # and made correct output look broken.
        "note:",
        "Fitting ",
        "Grid node ",
        "convergence not achieved",
        "Huber iteration",
        "Biweight iteration",
        "(fitting base model)",
        "(bootstrap",
        "(running ",
        "Refining ",
        "Iteration ",
        "(setting ",
        "(switching ",
    )
    with StataSession() as stata:
        # intreg's bound columns only exist on INTERVAL; everything else uses PANEL.
        # A few commands need their own frame: intreg's bounds live on INTERVAL, and
        # arima refuses a panel outright.
        _frame_name = {"intreg": INTERVAL, "arima": TSSERIES}
        stata.push(_frame_name.get(command.split()[0], PANEL))
        # ``xtset`` needs unit/period; the time-series frame for arima has neither, and
        # Stata errors with "variable unit not found" before the command even runs.
        _uses_time_series = command.split()[0] == "arima"
        for setup in STATA_SETUP:
            if _uses_time_series:
                # A pure time-series frame has no unit/period/stratum/psu/weight columns,
                # so BOTH setup lines would fail; run tsset instead.
                if setup.startswith("xtset"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        stata.run("tsset t")
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                stata.run(setup)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            stata.run(command)
        text = buffer.getvalue()
    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if not line.strip().startswith(noise)]
    lines = [line for line in lines if "Bootstrap replications" not in line]
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
