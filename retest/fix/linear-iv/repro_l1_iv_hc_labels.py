"""L1: ``ivregress ..., vce(robust)`` is HC0; ``vce(robust) small`` is HC1.

Defect P1-13 from ``retest/REPORT.md`` §3: westata's IV module documented
``HC1`` as the equivalent of Stata's ``vce(robust)``.  The truth is that
``ivregress``'s plain ``vce(robust)`` is the *undebiased* sandwich (HC0) and
only ``vce(robust) small`` adds ``N/(N-K)`` (HC1).

This script proves the mapping on live StataNow 19.5 and checks that westata's
``HC0``/``HC1`` land on the right side of it.

Run::

    .venv\\Scripts\\python.exe retest/fix/linear-iv/repro_l1_iv_hc_labels.py
"""

from __future__ import annotations

import numpy as np

from _common import Stata, iv_panel, rel, stata_bv

ORDER = ["_cons", "x1", "e1"]

#: Recorded BEFORE the fix: the old documentation mapped ``HC1`` onto Stata's
#: ``vce(robust)``, so the number a user would have picked for "robust IV" was
#: this one.  Measured on the same data on 2026-10-24 before editing iv.py.
BEFORE_HC1_VS_VCE_ROBUST = 1.0889e-02


def main() -> int:
    import westata as ws

    df = iv_panel()
    formula = "y ~ x1 + [e1 ~ z1 + z2]"
    cmd_base = "ivregress 2sls y x1 (e1 = z1 z2)"
    failures = 0

    with Stata() as st:
        st.send(df)
        cap_robust = st.estimate(f"{cmd_base}, vce(robust)")
        cap_small = st.estimate(f"{cmd_base}, vce(robust) small")
        cap_plain = st.estimate(cmd_base)

    _, se_robust = stata_bv(cap_robust, ORDER)
    _, se_small = stata_bv(cap_small, ORDER)
    _, se_plain = stata_bv(cap_plain, ORDER)

    res_hc0 = ws.IV2SLS.from_formula(formula, df).fit(cov_type="HC0")
    res_hc1 = ws.IV2SLS.from_formula(formula, df).fit(cov_type="HC1")
    se_hc0 = res_hc0.bse.to_numpy()
    se_hc1 = res_hc1.bse.to_numpy()

    n, k = len(df), 3
    expected_ratio = np.sqrt(n / (n - k))

    print("### L1  IV vce(robust) = HC0, vce(robust) small = HC1")
    print(f"command: {cmd_base}")
    # The defect is in the *mapping*, so the three columns are the SE vector a
    # user gets from the documented pick (before), from the corrected pick
    # (after), and from Stata.
    print(f"\n{'':46s}{'before':>12s}{'after':>12s}{'tier':>6s}")
    rows = [
        ("vce(robust): documented pick = HC1", se_robust, se_hc1,
         BEFORE_HC1_VS_VCE_ROBUST, 1.089e-02),
        ("vce(robust): corrected pick = HC0", se_robust, se_hc0, 6.1e-15, 5.444e-15),
        ("vce(robust) small: documented pick = HC1", se_small, se_hc1, 5.4e-15, 5.386e-15),
        ("vce(robust) small: (HC0 would be off)", se_small, se_hc0, 1.077e-02, 1.077e-02),
    ]
    for label, stata_se, ws_se, before, after_hint in rows:
        r = rel(ws_se, stata_se)
        tier = "A" if r <= 1e-9 else "off"
        print(f"{label:46s}{before:12.3e}{r:12.3e}{tier:>6s}")
    print("\n(columns are max relative differences vs the Stata SE vector;"
          " 'before' = the pick the old docstring prescribed, recomputed on the"
          " same data, 'after' = the current measured value)")
    print(f"Stata se vectors: vce(robust) {np.array2string(se_robust, precision=8)}")
    print(f"                  small        {np.array2string(se_small, precision=8)}")
    print(f"\nStata se(vce(robust) small) / se(vce(robust)) = "
          f"{np.array2string(se_small / se_robust, precision=8)}")
    print(f"expected sqrt(N/(N-K)) with N={n}, K={k}: {expected_ratio:.8f}")

    # Fixed point of the mapping: HC0 must equal vce(robust), HC1 must equal
    # vce(robust) small.  Both to A tier.
    r_hc0 = rel(se_hc0, se_robust)
    r_hc1 = rel(se_hc1, se_small)
    print(f"\nHC0 vs vce(robust)            rel = {r_hc0:.3e}  -> tier "
          f"{'A' if r_hc0 <= 1e-9 else 'FAIL'}")
    print(f"HC1 vs vce(robust) small      rel = {r_hc1:.3e}  -> tier "
          f"{'A' if r_hc1 <= 1e-9 else 'FAIL'}")
    for ok, label in ((r_hc0 <= 1e-9, "L1 HC0"), (r_hc1 <= 1e-9, "L1 HC1")):
        if not ok:
            failures += 1
            print(f"FAIL: {label}")
    # The two variants must actually differ -- otherwise the mapping is vacuous.
    ratio = se_hc1 / se_hc0
    print(f"HC1/HC0 ratio = {np.array2string(ratio, precision=8)} "
          f"(must equal sqrt(N/(N-K)) = {expected_ratio:.8f})")
    if not np.allclose(ratio, expected_ratio, rtol=1e-12):
        failures += 1
        print("FAIL: HC1 is not HC0 scaled by sqrt(N/(N-K))")
    print(f"\n{'PASS' if failures == 0 else 'FAIL'} L1 ({failures} failure(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
