"""S3 -- ``teffects aipw, atet``: the ATET potential-outcome means and their SEs.

Run:  .venv\\Scripts\\python.exe retest/fix/systems/repro_S3.py

The ATE branch is exact (1e-16) and is asserted here so that the ATET repair
cannot silently regress it.

Stata's ATET reported quantities (``teffects aipw (y x1 x2) (t x1 x2), atet``):

    r1vs0.t   = PO(1) - PO(0)          (e(b) row 1)
    0.t       = PO(0)                  (e(b) row 2)

Stata stores only ``POmean(0)`` in ``e(b)``; ``POmean(1)`` is
``r1vs0.t + 0.t``, so it is reconstructed from the two stored rows rather than
compared against a number Stata never published.
"""

from __future__ import annotations

import numpy as np

from _common import Stata, banner, header, rel_diff, row, treatment_frame

import westata as ws


def _ws_columns(result) -> dict[str, tuple[float, float]]:
    return {
        "effect": (result.estimate, result.std_err),
        "PO(1)": (result.potential_outcomes["1"][0], result.potential_outcomes["1"][1]),
        "PO(0)": (result.potential_outcomes["0"][0], result.potential_outcomes["0"][1]),
    }


def pick(cap):
    """Stata ``(effect, se, PO(0), se_PO(0), PO(1)=effect+PO0)``."""

    i0 = cap.names.index("r1vs0.t")
    i1 = cap.names.index("0.t")
    se = np.sqrt(np.diag(cap.V))
    effect, po0 = float(cap.b[i0]), float(cap.b[i1])
    # PO(1) = effect + PO(0); its SE needs the covariance between the two rows.
    po1 = effect + po0
    var = cap.V[i0, i0] + cap.V[i1, i1] + 2.0 * cap.V[i0, i1]
    return {
        "effect": (effect, float(se[i0])),
        "PO(1)": (po1, float(np.sqrt(var))),
        "PO(0)": (po0, float(se[i1])),
    }


LABELS = [("effect", "effect"), ("PO(1)", "PO(1)"), ("PO(0)", "PO(0)")]


def compare_block(title: str, stata: dict, mine: dict) -> tuple[float, float]:
    banner(title)
    header()
    sb, ss = [], []
    for key, label in LABELS:
        s_val, s_se = stata[key]
        w_val, w_se = mine[key]
        row(label, w_val, w_val, s_val)
        row(f"se({label})", w_se, w_se, s_se)
        sb.append(s_val)
        ss.append(s_se)
    wb = np.array([mine[k][0] for k, _ in LABELS])
    wse = np.array([mine[k][1] for k, _ in LABELS])
    param = rel_diff(wb, sb)
    se_rel = rel_diff(wse, ss)
    print(f"  parameter rel diff {param:.3e}   SE rel diff {se_rel:.3e}")
    print(f"  Stata  : {dict(zip([k for k,_ in LABELS], sb))}")
    print(f"  westata: {dict(zip([k for k,_ in LABELS], wb.tolist()))}")
    return param, se_rel


def main() -> int:
    df = treatment_frame()

    with Stata() as st:
        st.send(df)
        ate_cap = st.estimate("teffects aipw (y x1 x2) (t x1 x2)")
        atet_cap = st.estimate("teffects aipw (y x1 x2) (t x1 x2), atet")

    # ---- ATE: must stay exact
    ate = ws.AIPW.from_formula("y", "t", df, covariates=["x1", "x2"], estimand="ate").fit()
    ate_param, ate_se_rel = compare_block(
        "S3a  teffects aipw  ATE  (the branch that must NOT change)",
        pick(ate_cap), _ws_columns(ate),
    )

    # ---- ATET: the repair
    atet = ws.AIPW.from_formula("y", "t", df, covariates=["x1", "x2"], estimand="atet").fit()
    atet_param, atet_se_rel = compare_block(
        "S3b  teffects aipw  ATET  (the branch that was wrong)",
        pick(atet_cap), _ws_columns(atet),
    )

    print()
    print("  NOTE: the regression-adjustment / IPW branches are untouched;")
    print("        run the pytest suite for their coverage.")

    ok = ate_param <= 1e-9 and ate_se_rel <= 1e-9 and atet_param <= 1e-9 and atet_se_rel <= 1e-9
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
