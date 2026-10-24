"""M7 (new finding, outside the original defect list) -- ``tobit``'s ``margins``
default prediction type is ``xb``, not ``e(a,b)``.

This was not in ``retest/REPORT.md`` section 3 and not in the M1-M6 task list:
the retest compared tobit's ``margins`` only through ``predict(e(...))``, so the
default was never pinned.  ``margins.py::DEFAULT_PREDICT`` said ``"tobit": "e"``,
which is wrong, and the failure mode is the worst kind -- ``margins`` succeeds
and reports **a different quantity** than Stata:

    . tobit mpg weight foreign, ll(17)
    . margins, dydx(weight)                       b = -0.00779786618189768
    . margins, dydx(weight) predict(xb)           b = -0.00779786618189768
    . margins, dydx(weight) predict(e(17,.))      b = -0.00477824532202177
    . margins, dydx(weight) predict(ystar(17,.))  b = -0.00548991091113322
    . margins                                     b = 20.6918844403444
    . margins, predict(e(17,.))                   b = 22.7510610608599

Stata tells you which one is the default through ``e(marginsok)``, which for
tobit reads ``default XB Pr(passthru) E(passthru) YStar(passthru)``: ``xb`` is
flagged ``default`` and the other three are ``passthru`` (only reachable through
``predict()``).

Run from the repository root::

    .venv/Scripts/python.exe retest/fix/postest/repro_M7_tobit_margins_default.py

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

import westata as ws  # noqa: E402, F401 - imported for the reader's benefit

AUTO = ROOT / "tests" / "data" / "auto.dta"
ROWS = ("b", "se")


def rel(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1e-12)


def capture(st: Stata, command: str) -> dict[str, float]:
    st.quiet(command)
    matrix = st.matrix("r(table)")
    return {
        "b": float(matrix[0, 0]),
        "se": float(matrix[1, 0]),
        "stat": float(matrix[2, 0]),
        "p": float(matrix[3, 0]),
    }


def main() -> int:
    auto = pd.read_stata(AUTO, convert_categoricals=False)
    worst_after = 0.0
    worst_old = 0.0

    with Stata() as st:
        print(f"Stata c(stata_version) = {st.version}")
        st.run("sysuse auto, clear")
        st.quiet("tobit mpg weight foreign, ll(17)")
        marginsok = st.macro("e(marginsok)")
        print(f"\nStata e(marginsok) = {marginsok!r}")
        print("  -> 'default XB ...' means `margins` differentiates xb;")
        print("     E()/YStar()/Pr() are passthru, i.e. margins, predict(...) only.")

        # ---------------------------------------------------- dydx(weight)
        print("\n=== margins, dydx(weight) ===")
        default = capture(st, "margins, dydx(weight)")
        via_xb = capture(st, "margins, dydx(weight) predict(xb)")
        via_e = capture(st, "margins, dydx(weight) predict(e(17,.))")
        via_ystar = capture(st, "margins, dydx(weight) predict(ystar(17,.))")
        for label, values in (
            ("margins, dydx(weight) [default]", default),
            ("margins, dydx(weight) predict(xb)", via_xb),
            ("margins, dydx(weight) predict(e(17,.))", via_e),
            ("margins, dydx(weight) predict(ystar(17,.))", via_ystar),
        ):
            print(f"    {label:<44} b={values['b']: .15e}  se={values['se']: .15e}")
        print(
            "    default == predict(xb): "
            f"{'yes' if default['b'] == via_xb['b'] else 'NO'}"
            f"   (rel {rel(default['b'], via_xb['b']):.2e})"
        )
        print(
            "    default vs predict(e(17,.)) differs by "
            f"{rel(default['b'], via_e['b']):.2e} in b and "
            f"{rel(default['se'], via_e['se']):.2e} in se"
            "   <- the old westata default answered *this* one"
        )

        # the tobit coefficient, which `margins, dydx()` must reproduce
        b_weight = float(st.scalar("_b[weight]")) if st.scalar("_b[weight]") else None
        st.quiet("matrix b = e(b)")
        coefficient = float(st.matrix("e(b)").ravel()[0])
        print(
            f"    _b[weight] = {coefficient:.15e}"
            f"   (default AME rel diff {rel(default['b'], coefficient):.2e})"
        )

        # ------------------------------------------------------ margins (no dydx)
        print("\n=== margins (predictive margin) ===")
        margin_default = capture(st, "margins")
        margin_xb = capture(st, "margins, predict(xb)")
        margin_e = capture(st, "margins, predict(e(17,.))")
        for label, values in (
            ("margins [default]", margin_default),
            ("margins, predict(xb)", margin_xb),
            ("margins, predict(e(17,.))", margin_e),
        ):
            print(f"    {label:<30} b={values['b']: .15e}  se={values['se']: .15e}")
        print(f"    mean(xb) from the estimation sample = {float(np.mean(margin_xb['b'])):.15e}")
        print(
            "    default vs predict(e(17,.)) differs by "
            f"{rel(margin_default['b'], margin_e['b']):.2e} in b and "
            f"{rel(margin_default['se'], margin_e['se']):.2e} in se"
        )

        # ------------------------------------- what westata now reports
        print("\n=== westata (after the fix: DEFAULT_PREDICT['tobit'] = 'xb') ===")
        tobit = ws.Tobit.from_formula("mpg ~ weight + foreign", auto, lower=17).fit()

        now_default = tobit.margins(dydx="weight")
        now_e = tobit.margins(dydx="weight", predict_kind="e")
        now_margin = tobit.margins()
        now_margin_e = tobit.margins(predict_kind="e")

        print(
            f"    margins, dydx(weight)                 b={now_default.estimate: .15e}"
            f"  se={now_default.std_err: .15e}"
            f"   rel b {rel(now_default.estimate, default['b']):.2e}"
            f"  se {rel(now_default.std_err, default['se']):.2e}"
        )
        print(
            f"    ... predict_kind='e'                  b={now_e.estimate: .15e}"
            f"  se={now_e.std_err: .15e}"
            f"   rel b {rel(now_e.estimate, via_e['b']):.2e}"
            f"  se {rel(now_e.std_err, via_e['se']):.2e}"
        )
        print(
            f"    margins                               b={now_margin.estimate: .15e}"
            f"  se={now_margin.std_err: .15e}"
            f"   rel b {rel(now_margin.estimate, margin_default['b']):.2e}"
            f"  se {rel(now_margin.std_err, margin_default['se']):.2e}"
        )
        print(
            f"    ... predict_kind='e'                  b={now_margin_e.estimate: .15e}"
            f"  se={now_margin_e.std_err: .15e}"
            f"   rel b {rel(now_margin_e.estimate, margin_e['b']):.2e}"
            f"  se {rel(now_margin_e.std_err, margin_e['se']):.2e}"
        )

        worst_after = max(
            rel(now_default.estimate, default["b"]),
            rel(now_default.std_err, default["se"]),
            rel(now_margin.estimate, margin_default["b"]),
            rel(now_margin.std_err, margin_default["se"]),
        )

        # --------------------- what the OLD default ('e') would have reported
        print("\n=== the old default (DEFAULT_PREDICT['tobit'] = 'e') in numbers ===")
        old_default_dydx_b = now_e.estimate
        old_default_dydx_se = now_e.std_err
        old_margin_b = now_margin_e.estimate
        old_margin_se = now_margin_e.std_err
        print(
            "    margins, dydx(weight) would report "
            f"b={old_default_dydx_b:.15e} se={old_default_dydx_se:.15e}"
        )
        print(
            "      Stata reports                "
            f"b={default['b']:.15e} se={default['se']:.15e}"
            f"   -> rel b {rel(old_default_dydx_b, default['b']):.2e}"
            f"  se {rel(old_default_dydx_se, default['se']):.2e}"
        )
        print(
            "    margins would report          "
            f"b={old_margin_b:.15e} se={old_margin_se:.15e}"
        )
        print(
            "      Stata reports                "
            f"b={margin_default['b']:.15e} se={margin_default['se']:.15e}"
            f"   -> rel b {rel(old_margin_b, margin_default['b']):.2e}"
            f"  se {rel(old_margin_se, margin_default['se']):.2e}"
        )
        worst_old = max(
            rel(old_default_dydx_b, default["b"]),
            rel(old_default_dydx_se, default["se"]),
            rel(old_margin_b, margin_default["b"]),
            rel(old_margin_se, margin_default["se"]),
        )

    print("\n=== summary ===")
    print(
        "Stata's default predict type for tobit margins is xb "
        "(e(marginsok) flags it 'default')."
    )
    print(f"old default 'e' worst relative difference : {worst_old:.3e}")
    print(f"fixed default 'xb' worst relative difference: {worst_after:.3e}")
    tier = "A" if worst_after <= 1e-9 else ("B" if worst_after <= 1e-6 else "C+")
    print(f"tier after fix: {tier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
