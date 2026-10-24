"""M1 -- ``margins`` average-marginal-effect standard errors vs StataNow 19.5.

Defect (retest/REPORT.md section 2.7 / P0-7): ``margins.py::_statistic_and_gradient``
differentiated the covariate with a *fixed* central-difference step
``h = 1e-6``.  Because an average marginal effect here is ~3.5e-4 while the
predicted probability is ~0.1, the difference quotient loses almost all its
significant digits; the error *grows* as h shrinks.

This script starts a real Stata session, captures Stata's ``r(table)`` and
``r(Jacobian)``, and prints three columns per quantity:

    before  -- the old fixed-step numeric gradient (reproduced verbatim here)
    after   -- the repaired implementation (analytic family derivatives)
    Stata   -- the live capture

plus the relative differences ``|before-Stata|/|Stata|`` and
``|after-Stata|/|Stata|``.

Run from the repository root::

    .venv\\Scripts\\python.exe retest/fix/postest/repro_M1_margins.py

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
ROWS = ("b", "se", "stat", "p", "ll", "ul")


def rel(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1e-12)


def table(st: Stata) -> list[float]:
    """Stata's one-row ``r(table)`` in the printed columns we compare."""
    matrix = st.matrix("r(table)")
    return [float(matrix[index, 0]) for index, _ in enumerate(ROWS)]


def old_statistic_and_gradient(result, exog, kind, *, position, h=1e-6):
    """The **pre-repair** implementation, kept verbatim as the "before" column.

    Fixed ``h = 1e-6`` for the covariate derivative and ``1e-6 * max(1, |b|)``
    for the parameter gradient -- exactly what
    ``margins.py::_statistic_and_gradient`` did before this repair.
    """

    params = np.asarray(result.params, dtype=float).copy()
    k = params.size

    def evaluate(parameter_vector: np.ndarray) -> np.ndarray:
        original = result._params
        try:
            result._params = pd.Series(parameter_vector, index=original.index)
            shifted = exog.copy()
            shifted[:, position] += h
            forward = result.predict(exog=shifted, kind=kind)
            shifted = exog.copy()
            shifted[:, position] -= h
            backward = result.predict(exog=shifted, kind=kind)
            return (forward - backward) / (2.0 * h)
        finally:
            result._params = original

    value = float(np.mean(evaluate(params)))
    gradient = np.zeros(k, dtype=float)
    for index in range(k):
        step = 1e-6 * max(1.0, abs(params[index]))
        forward = params.copy()
        forward[index] += step
        backward = params.copy()
        backward[index] -= step
        gradient[index] = (
            float(np.mean(evaluate(forward))) - float(np.mean(evaluate(backward)))
        ) / (2.0 * step)
    return value, gradient


def kind_for(result) -> str:
    name = str(getattr(result, "model_name", ""))
    if name in ("logit", "probit"):
        return "pr"
    if name in ("poisson", "nbreg"):
        return "n"
    return "xb"


def old_se(result, exog, position) -> float:
    """Standard error the pre-repair code reported for one marginal effect."""

    _, gradient = old_statistic_and_gradient(
        result, exog, kind_for(result), position=position
    )
    cov = result.cov_params()
    return float(np.sqrt(max(float(gradient @ cov @ gradient), 0.0)))


def old_predictive_margin(result, exog) -> float:
    """Predictive-margin SE the pre-repair code reported (linear model)."""

    params = np.asarray(result.params, dtype=float)
    gradient = np.zeros(params.size)
    for index in range(params.size):
        step = 1e-6 * max(1.0, abs(params[index]))
        forward = params.copy()
        forward[index] += step
        backward = params.copy()
        backward[index] -= step
        original = result._params
        try:
            result._params = pd.Series(forward, index=original.index)
            high = float(np.mean(result.predict(exog=exog, kind="xb")))
            result._params = pd.Series(backward, index=original.index)
            low = float(np.mean(result.predict(exog=exog, kind="xb")))
        finally:
            result._params = original
        gradient[index] = (high - low) / (2.0 * step)
    cov = result.cov_params()
    return float(np.sqrt(max(float(gradient @ cov @ gradient), 0.0)))


def stage_sweep(result, exog, kind, position, reference) -> None:
    """Reproduce the step sweep that identifies cancellation as the cause."""

    print("    step sweep on dE[y]/dx (the reported AME), relative to Stata:")
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        estimate, _ = old_statistic_and_gradient(
            result, exog, kind, position=position, h=h
        )
        print(f"      h={h:<8.0e} AME={estimate: .15e}  rel={rel(estimate, reference):.2e}")


def main() -> int:
    auto = pd.read_stata(AUTO, convert_categoricals=False)
    worst = 0.0
    verdicts: list[str] = []

    with Stata() as st:
        st.run("sysuse auto, clear")
        print(f"Stata c(stata_version) = {st.version}")

        # ---------------------------------------------------------- logit
        print("\n[1] logit foreign weight mpg -- margins, dydx(weight)")
        st.quiet("logit foreign weight mpg")
        st.quiet("margins, dydx(weight)")
        stata = table(st)
        stata_jacobian = np.asarray(st.matrix("r(Jacobian)")).ravel()

        logit = ws.Logit.from_formula("foreign ~ weight + mpg", auto).fit()
        exog = np.asarray(logit.design_matrix[0], dtype=float)
        names = list(logit.design_matrix[1])
        position = names.index("weight")

        before_b, _ = old_statistic_and_gradient(logit, exog, "pr", position=position)
        before_se = old_se(logit, exog, position)
        after = logit.margins(dydx="weight")

        print(f"    before  b={before_b: .15e} se={before_se: .15e}")
        print(f"    after   b={after.estimate: .15e} se={after.std_err: .15e}")
        print(f"    Stata   b={stata[0]: .15e} se={stata[1]: .15e}")
        print(
            f"    rel     before b {rel(before_b, stata[0]):.2e}"
            f" se {rel(before_se, stata[1]):.2e}"
        )
        print(
            f"            after  b {rel(after.estimate, stata[0]):.2e}"
            f" se {rel(after.std_err, stata[1]):.2e}"
        )
        print(f"    Stata r(Jacobian) = {np.array2string(stata_jacobian, precision=12)}")
        worst = max(worst, rel(after.estimate, stata[0]), rel(after.std_err, stata[1]))
        verdicts.append(
            f"M1a logit AME dydx(weight): before se {rel(before_se, stata[1]):.2e}"
            f" -> after se {rel(after.std_err, stata[1]):.2e}"
        )
        stage_sweep(logit, exog, "pr", position, stata[0])

        # ---------------------------------------------------------- at()
        print("\n[2] logit foreign weight mpg -- margins, dydx(weight) at(mpg=20)")
        st.quiet("margins, dydx(weight) at(mpg=20)")
        stata_at = table(st)
        exog_at = exog.copy()
        exog_at[:, names.index("mpg")] = 20.0
        before_b_at, _ = old_statistic_and_gradient(
            logit, exog_at, "pr", position=position
        )
        before_se_at = old_se(logit, exog_at, position)
        after_at = logit.margins(dydx="weight", at={"mpg": 20})
        print(f"    before  b={before_b_at: .15e} se={before_se_at: .15e}")
        print(f"    after   b={after_at.estimate: .15e} se={after_at.std_err: .15e}")
        print(f"    Stata   b={stata_at[0]: .15e} se={stata_at[1]: .15e}")
        print(
            f"    rel     before b {rel(before_b_at, stata_at[0]):.2e}"
            f" se {rel(before_se_at, stata_at[1]):.2e}"
        )
        print(
            f"            after  b {rel(after_at.estimate, stata_at[0]):.2e}"
            f" se {rel(after_at.std_err, stata_at[1]):.2e}"
        )
        worst = max(
            worst,
            rel(after_at.estimate, stata_at[0]),
            rel(after_at.std_err, stata_at[1]),
        )
        verdicts.append(
            f"M1b logit AME at(mpg=20): before se {rel(before_se_at, stata_at[1]):.2e}"
            f" -> after se {rel(after_at.std_err, stata_at[1]):.2e}"
        )

        # ---------------------------------------------------------- probit
        print("\n[3] probit foreign weight mpg -- margins, dydx(weight)")
        st.run("sysuse auto, clear")
        st.quiet("probit foreign weight mpg")
        st.quiet("margins, dydx(weight)")
        stata_probit = table(st)
        probit = ws.Probit.from_formula("foreign ~ weight + mpg", auto).fit()
        pexog = np.asarray(probit.design_matrix[0], dtype=float)
        pnames = list(probit.design_matrix[1])
        before_probit_se = old_se(probit, pexog, pnames.index("weight"))
        after_probit = probit.margins(dydx="weight")
        print(
            f"    before se={before_probit_se: .15e}"
            f"   after se={after_probit.std_err: .15e}"
        )
        print(f"    Stata  se={stata_probit[1]: .15e}")
        print(
            f"    rel    before {rel(before_probit_se, stata_probit[1]):.2e}"
            f"   after {rel(after_probit.std_err, stata_probit[1]):.2e}"
        )
        worst = max(worst, rel(after_probit.std_err, stata_probit[1]))
        verdicts.append(
            f"M1c probit AME: before se {rel(before_probit_se, stata_probit[1]):.2e}"
            f" -> after {rel(after_probit.std_err, stata_probit[1]):.2e}"
        )

        # ---------------------------------------------------------- poisson
        print("\n[4] poisson rep78 mpg weight -- margins, dydx(weight)")
        st.run("sysuse auto, clear")
        st.quiet("poisson rep78 mpg weight")
        st.quiet("margins, dydx(weight)")
        stata_poisson = table(st)
        poisson_frame = auto.dropna(subset=["rep78"])
        poisson = ws.Poisson.from_formula("rep78 ~ mpg + weight", poisson_frame).fit()
        poexog = np.asarray(poisson.design_matrix[0], dtype=float)
        ponames = list(poisson.design_matrix[1])
        before_poisson_se = old_se(poisson, poexog, ponames.index("weight"))
        after_poisson = poisson.margins(dydx="weight")
        print(
            f"    before se={before_poisson_se: .15e}"
            f"   after se={after_poisson.std_err: .15e}"
        )
        print(f"    Stata  se={stata_poisson[1]: .15e}")
        print(
            f"    rel    before {rel(before_poisson_se, stata_poisson[1]):.2e}"
            f"   after {rel(after_poisson.std_err, stata_poisson[1]):.2e}"
        )
        worst = max(worst, rel(after_poisson.std_err, stata_poisson[1]))
        verdicts.append(
            f"M1d poisson AME: before se {rel(before_poisson_se, stata_poisson[1]):.2e}"
            f" -> after {rel(after_poisson.std_err, stata_poisson[1]):.2e}"
        )

        # --------------------------------------------------- regress over()
        print("\n[5] regress mpg weight i.foreign -- margins, over(foreign)")
        st.run("sysuse auto, clear")
        st.quiet("regress mpg weight i.foreign")
        st.quiet("margins, over(foreign)")
        stata_over = np.asarray(st.matrix("r(table)")).T.tolist()
        ols = ws.OLS.from_formula("mpg ~ weight + C(foreign)", auto).fit()
        ols_exog = np.asarray(ols.design_matrix[0], dtype=float)
        foreign = auto["foreign"].to_numpy()
        before_rows = [
            old_predictive_margin(ols, ols_exog[foreign == group]) for group in (0, 1)
        ]
        after_rows = ols.margins(over="foreign")
        for index, row in enumerate(after_rows):
            print(
                f"    group {row.label}: before se={before_rows[index]: .15e}"
                f"  after se={row.std_err: .15e}"
                f"  Stata se={stata_over[index][1]: .15e}"
            )
            print(
                f"        rel before {rel(before_rows[index], stata_over[index][1]):.2e}"
                f"   after {rel(row.std_err, stata_over[index][1]):.2e}"
            )
            worst = max(worst, rel(row.std_err, stata_over[index][1]))
        verdicts.append(
            "M1e regress over(foreign) SE after: "
            + ", ".join(
                f"{rel(row.std_err, stata_over[i][1]):.2e}"
                for i, row in enumerate(after_rows)
            )
        )

        # ------------------------------------------------ dydx(foreign) factor
        print("\n[6] regress mpg weight i.foreign -- margins, dydx(foreign)")
        st.quiet("margins, dydx(foreign)")
        stata_factor = np.asarray(st.matrix("r(table)"))
        factor_row = ols.margins(dydx="C(foreign)[T.1]", factor_vars=["C(foreign)[T.1]"])
        factor_row = factor_row[0] if isinstance(factor_row, list) else factor_row
        # Stata prints the base-level row first (b = 0), so the non-base row is 1.
        print(f"    after   b={factor_row.estimate: .15e} se={factor_row.std_err: .15e}")
        print(f"    Stata   b={stata_factor[0, 1]: .15e} se={stata_factor[1, 1]: .15e}")
        print(
            f"    rel     after b {rel(factor_row.estimate, stata_factor[0, 1]):.2e}"
            f" se {rel(factor_row.std_err, stata_factor[1, 1]):.2e}"
        )
        worst = max(
            worst,
            rel(factor_row.estimate, stata_factor[0, 1]),
            rel(factor_row.std_err, stata_factor[1, 1]),
        )
        verdicts.append(
            f"M1f factor discrete change dydx(foreign): after se"
            f" {rel(factor_row.std_err, stata_factor[1, 1]):.2e} (== bse)"
        )

    print("\n=== summary ===")
    for line in verdicts:
        print(" ", line)
    print(f"worst 'after' relative difference across M1: {worst:.3e}")
    tier = "A" if worst <= 1e-9 else ("B" if worst <= 1e-6 else "C or worse")
    print(f"tier: {tier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
