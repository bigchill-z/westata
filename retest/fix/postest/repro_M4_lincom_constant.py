"""M4 -- ``lincom`` must accept a bare constant, exactly as Stata does.

Defect (retest/REPORT.md section 2.7): ``api.py::_split_terms`` treated a bare
number as an error ("put constants on the right-hand side of '='"), so
``lincom 2*weight - 3*foreign + 10`` -- a line copied straight out of a Stata
do-file -- raised instead of returning Stata's 14.9369115.

Stata moves the constant to the right-hand side of the printed constraint and
*never* lets it touch the variance:

    . lincom 2*weight - 3*foreign + 10
     ( 1)  2*weight - 3*foreign = -10
    ------------------------------------------------------------------------
             mpg | Coefficient  Std. err.      t    P>|t|     [95% conf. int.]
    -------------+----------------------------------------------------------
             (1) |   14.93691   3.227227     4.63   0.000     8.502003  21.37182

The same constant handling is needed on the right-hand side of ``test``:
``test 2*weight - 3*foreign + 10 = 0`` is the *same* hypothesis as
``test 2*weight - 3*foreign = -10`` (F = 21.422125424189847 on both sides).

Run from the repository root::

    .venv/Scripts/python.exe retest/fix/postest/repro_M4_lincom_constant.py

Stata version: ``c(stata_version)`` = 19.5 (MP).  Captured 2026-10-24.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "retest" / "lib"))
sys.path.insert(0, str(ROOT / "src"))

from stata_probe import Stata  # noqa: E402

import westata as ws  # noqa: E402

AUTO = ROOT / "tests" / "data" / "auto.dta"

#: ``lincom`` expressions with the reference Stata numbers captured live below.
LINCOM = (
    "2*weight - 3*foreign + 10",
    "2*weight - 3*foreign",
    "10",
    "-10",
    "2*weight - 3*foreign + 1e3",
    "weight + 1",
)

#: ``test`` specifications; equivalents are checked against each other too.
TEST = (
    "2*weight - 3*foreign + 10 = 0",
    "2*weight - 3*foreign = -10",
    "weight = 10",
    "weight + 10 = 0",
)


def rel(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1e-12)


def old_lincom(result, expression: str) -> str:
    """The pre-repair parser's behaviour, reproduced on the same expression."""

    from westata.postestimation.api import _split_terms

    try:
        _split_terms(expression, list(result.param_names))
    except ValueError as exc:
        return f"ValueError: {exc}"
    return "accepted"


def main() -> int:
    auto = pd.read_stata(AUTO, convert_categoricals=False)
    worst = 0.0

    with Stata() as st:
        print(f"Stata c(stata_version) = {st.version}")
        st.run("sysuse auto, clear")
        st.quiet("regress mpg weight foreign")

        print("\n=== lincom ===")
        print(
            f"    {'expression':<30}{'Stata b':>18}{'Stata se':>16}"
            f"{'westata b':>18}{'westata se':>16}{'rel b':>10}{'rel se':>10}"
        )
        ols = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
        for expression in LINCOM:
            st.quiet(f"lincom {expression}")
            stata_b = float(st.scalar("r(estimate)"))
            stata_se = float(st.scalar("r(se)"))
            stata_t = st.scalar("r(t)")
            before = old_lincom(ols, expression)
            try:
                outcome = ols.lincom(expression)
                after_b, after_se = outcome.estimate, outcome.std_err
                rel_b, rel_se = rel(after_b, stata_b), rel(after_se, stata_se)
                worst = max(worst, rel_b, rel_se)
                print(
                    f"    {expression:<30}{stata_b:>18.10g}{stata_se:>16.10g}"
                    f"{after_b:>18.10g}{after_se:>16.10g}"
                    f"{rel_b:>10.2e}{rel_se:>10.2e}"
                )
                if stata_t is not None and outcome.statistic == outcome.statistic:
                    print(
                        f"        t: Stata {stata_t:.10g} vs westata"
                        f" {outcome.statistic:.10g} (rel"
                        f" {rel(outcome.statistic, stata_t):.2e})"
                        f"   printed constraint: {outcome.expression}"
                    )
            except Exception as exc:  # noqa: BLE001 - before the repair it raised
                print(
                    f"    {expression:<30}{stata_b:>18.10g}{stata_se:>16.10g}"
                    f"{'FAILED':>18}{'':>16}  -> {type(exc).__name__}: {exc}"
                )
            print(f"        before the repair: {before}")

        print("\n=== test (a constant on either side of '=') ===")
        print(f"    {'specification':<34}{'Stata F':>16}{'westata F':>16}{'rel':>10}")
        for specification in TEST:
            st.quiet(f"test {specification}")
            stata_f = float(st.scalar("r(F)"))
            outcome = ols.test(specification)
            print(
                f"    {specification:<34}{stata_f:>16.10g}"
                f"{outcome.statistic:>16.10g}"
                f"{rel(outcome.statistic, stata_f):>10.2e}"
            )
            worst = max(worst, rel(outcome.statistic, stata_f))
            print(f"        printed constraint: {outcome.constraint.strip()}")

        print("\n    equivalence: '+10 = 0' must equal '= -10'")
        left = ols.test("2*weight - 3*foreign + 10 = 0")
        right = ols.test("2*weight - 3*foreign = -10")
        print(
            f"      F {left.statistic:.15g} vs {right.statistic:.15g}"
            f"  rel {rel(left.statistic, right.statistic):.2e}"
        )

        print("\n    a constant-only lincom ('lincom 10' -> se 0 on both sides)")
        constant_only = ols.lincom("10")
        print(
            f"      westata b={constant_only.estimate:.10g}"
            f" se={constant_only.std_err:.10g} (Stata: b=10, se=0)"
        )

    print(f"\nworst relative difference across M4: {worst:.3e}")
    tier = "A" if worst <= 1e-9 else ("B" if worst <= 1e-6 else "C or worse")
    print(f"tier: {tier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
