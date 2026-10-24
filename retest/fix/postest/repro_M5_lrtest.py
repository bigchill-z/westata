"""M5 -- ``lrtest`` must refuse ``df = 0`` and must allow a negative LR.

Defects (retest/REPORT.md section 2.7, behaviour differences):

* ``df == 0``.  Two models with the same parameter count.  westata reported
  ``chi2 = 0, df = 0, p = 1.0`` (defending ``chi2(0)`` as a point mass at 0);
  Stata refuses the command outright:
  ``df(unrestricted) = df(restricted) = 2`` / ``r(498)``.
* Non-nested models with *different* parameter counts.  westata raised
  ``ConsistencyError`` because the coefficient names do not nest; Stata checks
  only the counts and reports the resulting likelihood ratio, which is
  **negative** (``LR chi2(1) = -2.6835, Prob > chi2 = 1.0000``).
* Arguments in the "wrong" order.  Stata accepts them and relabels the pair
  ("Assumption: L1 nested within B1"), reporting the positive statistic anyway.

Run from the repository root::

    .venv/Scripts/python.exe retest/fix/postest/repro_M5_lrtest.py

Stata version: ``c(stata_version)`` = 19.5 (MP).  Captured 2026-10-24.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "retest" / "lib"))
sys.path.insert(0, str(ROOT / "src"))

from stata_probe import Stata  # noqa: E402

import westata as ws  # noqa: E402
from westata.errors import ConsistencyError  # noqa: E402

AUTO = ROOT / "tests" / "data" / "auto.dta"


def rel(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1e-12)


def stata_outcome(st: Stata, command: str) -> tuple[bool, dict[str, float], str]:
    try:
        st.quiet(command)
    except Exception as exc:  # noqa: BLE001 - Stata signals errors as exceptions
        return False, {}, str(exc).strip()
    return (
        True,
        {
            "chi2": st.scalar("r(chi2)"),
            "df": st.scalar("r(df)"),
            "p": st.scalar("r(p)"),
        },
        "",
    )


def westata_outcome(callable_) -> tuple[bool, dict[str, float], str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outcome = callable_()
    except ConsistencyError as exc:
        return False, {}, str(exc)
    return (
        True,
        {
            "chi2": outcome.statistic,
            "df": outcome.df,
            "p": outcome.pvalue,
        },
        "",
    )


def report(name: str, stata, westata, note: str = "") -> bool:
    accepted_s, values_s, error_s = stata
    accepted_w, values_w, error_w = westata
    agree = accepted_s == accepted_w
    print(f"\n[{name}]")
    print(f"    Stata   : {'accept ' + str(values_s) if accepted_s else 'reject - ' + error_s.splitlines()[0]}")
    print(f"    westata : {'accept ' + str(values_w) if accepted_w else 'reject - ' + error_w[:120]}")
    print(f"    agreement on accept/reject: {agree}")
    if accepted_s and accepted_w:
        for key in ("chi2", "df", "p"):
            if values_s.get(key) is not None and values_w.get(key) is not None:
                print(f"      {key}: rel {rel(values_w[key], values_s[key]):.2e}")
    if note:
        print(f"    note: {note}")
    return agree


def main() -> int:
    auto = pd.read_stata(AUTO, convert_categoricals=False)
    accept_agreements: list[bool] = []

    with Stata() as st:
        print(f"Stata c(stata_version) = {st.version}")

        # ------------------------------------------------------- nested pair
        st.run("sysuse auto, clear")
        st.quiet("regress mpg weight")
        st.quiet("estimates store L1")
        st.quiet("regress mpg weight foreign")
        st.quiet("estimates store B1")
        accept_agreements.append(
            report(
                "nested: lrtest L1 B1 (larger first, the documented order)",
                stata_outcome(st, "lrtest B1 L1"),
                westata_outcome(lambda: ws.lrtest(
                    ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit(),
                    ws.OLS.from_formula("mpg ~ weight", auto).fit(),
                )),
            )
        )

        # ------------------------------------------- reversed argument order
        print("\n    reversed order (Stata: 'Assumption: L1 nested within B1')")
        print(st.output("lrtest L1 B1"))
        accept_agreements.append(
            report(
                "reversed: lrtest L1 B1",
                stata_outcome(st, "lrtest L1 B1"),
                westata_outcome(lambda: ws.lrtest(
                    ws.OLS.from_formula("mpg ~ weight", auto).fit(),
                    ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit(),
                )),
            )
        )

        # ----------------------------------------------------------- df = 0
        st.quiet("regress mpg weight")
        st.quiet("estimates store E2")
        accept_agreements.append(
            report(
                "df = 0: same model twice",
                stata_outcome(st, "lrtest L1 E2"),
                westata_outcome(lambda: ws.lrtest(
                    ws.OLS.from_formula("mpg ~ weight", auto).fit(),
                    ws.OLS.from_formula("mpg ~ weight", auto).fit(),
                )),
                note="Stata r(498); westata must raise ConsistencyError, not report p=1.0",
            )
        )

        # ------------------------------------------ equal count, other vars
        st.quiet("regress mpg foreign")
        st.quiet("estimates store N3")
        accept_agreements.append(
            report(
                "df = 0 by count only: different regressors, same k",
                stata_outcome(st, "lrtest N3 L1"),
                westata_outcome(lambda: ws.lrtest(
                    ws.OLS.from_formula("mpg ~ foreign", auto).fit(),
                    ws.OLS.from_formula("mpg ~ weight", auto).fit(),
                )),
                note="Stata checks the parameter count before the names",
            )
        )

        # ------------------------------------------------------- non-nested
        st.quiet("regress mpg foreign length")
        st.quiet("estimates store N2")
        non_nested = report(
            "non-nested, different k: Stata reports a NEGATIVE LR",
            stata_outcome(st, "lrtest N2 L1"),
            westata_outcome(lambda: ws.lrtest(
                ws.OLS.from_formula("mpg ~ foreign + length", auto).fit(),
                ws.OLS.from_formula("mpg ~ weight", auto).fit(),
            )),
            note="westata warns that the pair is not nested but still reports the number",
        )
        accept_agreements.append(non_nested)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            negative = ws.lrtest(
                ws.OLS.from_formula("mpg ~ foreign + length", auto).fit(),
                ws.OLS.from_formula("mpg ~ weight", auto).fit(),
            )
        print(f"    westata statistic = {negative.statistic:.15g} (must be < 0)")
        print(f"    westata p         = {negative.pvalue:.15g} (chi2.sf of a negative)")
        print(f"    warnings issued   = {[str(w.message)[:80] for w in caught]}")

    print("\n=== summary ===")
    print(f"accept/reject agreements: {accept_agreements}")
    print("all agree" if all(accept_agreements) else "MISMATCH remains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
