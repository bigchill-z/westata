"""L3: ``IVGMM`` + ``vce(cluster)`` must cluster the GMM weight matrix too.

Defect P1-17: Stata's ``ivregress gmm ..., vce(cluster g)`` sets the GMM weight
matrix ``W`` from the *clustered* moment covariance, not the heteroskedastic
one, so the **point estimates** move (3.58e-2 on the retest's 140x14 panel).
westata used linearmodels' default (heteroskedastic) ``W`` regardless of
``cov_type``.

Fix: ``IVGMM._backend_kwargs`` passes ``weight_type="clustered"`` (plus the
labels) whenever ``cov_type="cluster"``.  This script measures coefficients,
standard errors and ``e(V)`` for the default and the clustered variants, plus
``e(J)`` and the exhaustively-checked ``e(V)`` elements.

Run::

    .venv\\Scripts\\python.exe retest/fix/linear-iv/repro_l3_gmm_cluster.py
"""

from __future__ import annotations

import numpy as np

from _common import Stata, iv_panel, rel, retest_iv_panel, stata_bv

ORDER = ["_cons", "x1", "e1"]


def check_retest_panel() -> int:
    """Re-run the retest's own 12d data (same generator) and report 改前/改后.

    The retest report quotes a 3.58e-2 coefficient gap for this case; the
    "before" column below recomputes it with the pre-fix estimator
    (heteroskedastic ``W``), which is exactly what westata returned then.
    """

    import westata as ws

    df = retest_iv_panel()
    formula = "yiv ~ x1 + [e1 ~ z1 + z2]"
    cmd = "ivregress gmm yiv x1 (e1 = z1 z2), vce(cluster gid)"
    with Stata() as st:
        st.send(df)
        cap = st.estimate(cmd)
    b_s, se_s = stata_bv(cap, ORDER)
    before = ws.IVGMM.from_formula(formula, df).fit()  # heteroskedastic W = pre-fix
    after = ws.IVGMM.from_formula(formula, df).fit(
        cov_type="cluster", cov_kwds={"clusters": df["gid"]}
    )
    r_before = rel(before.params.to_numpy(), b_s)
    r_after = rel(after.params.to_numpy(), b_s)
    se_before = rel(before.bse.to_numpy(), se_s)
    se_after = rel(after.bse.to_numpy(), se_s)
    print("\n### L3 on the retest's own 12d data (retest_iv_panel, 140x14)")
    print(f"command: {cmd}")
    print(f"   {'':10s}{'coefficients':>16s}{'standard errors':>18s}")
    print(f"   {'BEFORE':10s}{r_before:16.3e}{se_before:18.3e}")
    print(f"   {'AFTER':10s}{r_after:16.3e}{se_after:18.3e}")
    print("   (retest/REPORT.md quotes 3.58e-2 for the coefficient gap)")
    failures = 0
    if r_after > 1e-9 or se_after > 1e-9:
        failures += 1
        print("   FAIL: the retest panel is not reproduced to A tier")
    if r_before < 1e-2:
        failures += 1
        print("   FAIL: the pre-fix estimator does not reproduce the reported gap")
    return failures


def main() -> int:
    import westata as ws

    failures = 0
    cases = [
        ("140x14 2instr", iv_panel(), 2),
        ("140x14 3instr", iv_panel(seed=11, n=140, g=14, n_instr=3), 3),
        ("240x24 2instr", iv_panel(seed=7, n=240, g=24, n_instr=2), 2),
    ]
    print("### L3  IVGMM vce(cluster) clusters the weight matrix W")
    for label, df, n_instr in cases:
        instrs = " ".join(f"z{j}" for j in range(1, n_instr + 1))
        formula = f"y ~ x1 + [e1 ~ {' + '.join(f'z{j}' for j in range(1, n_instr + 1))}]"
        cmd = f"ivregress gmm y x1 (e1 = {instrs}), vce(cluster gid)"
        with Stata() as st:
            st.send(df)
            cap = st.estimate(cmd)
            cap_plain = st.estimate(f"ivregress gmm y x1 (e1 = {instrs})")
        b_s, se_s = stata_bv(cap, ORDER)
        idx = [cap.names.index(name) for name in ORDER]
        V_s = np.asarray(cap.V)[np.ix_(idx, idx)]
        res = ws.IVGMM.from_formula(formula, df).fit(
            cov_type="cluster", cov_kwds={"clusters": df["gid"]}
        )
        b_w = res.params.to_numpy()
        V_w = res.cov
        print(f"\n-- {label}: {cmd}")
        print(f"   Stata   b {np.array2string(b_s, precision=12)}")
        print(f"   westata b {np.array2string(b_w, precision=12)}")
        print(f"   coef rel = {rel(b_w, b_s):.3e}")
        print(f"   Stata   se {np.array2string(se_s, precision=12)}")
        print(f"   westata se {np.array2string(res.bse.to_numpy(), precision=12)}")
        print(f"   se  rel  = {rel(res.bse.to_numpy(), se_s):.3e}")
        print(f"   e(V) full-matrix rel = {rel(V_w.ravel(), V_s.ravel()):.3e}")

        # The heteroskedastic-W estimator (the pre-fix behaviour) must be the
        # one that disagrees; otherwise this test would pass for the wrong
        # reason and prove nothing about W.
        res_plain = ws.IVGMM.from_formula(formula, df).fit()
        b_plain = res_plain.params.to_numpy()
        print(f"   [control] hetero-W (pre-fix estimator) coef rel vs this Stata "
              f"fit = {rel(b_plain, b_s):.3e}")

        checks = [
            ("coefficients", rel(b_w, b_s)),
            ("standard errors", rel(res.bse.to_numpy(), se_s)),
            ("e(V) all elements", rel(V_w.ravel(), V_s.ravel())),
            ("e(J)", rel([res.e.get("J")], [cap.scalars.get("J")])),
            ("e(N_clust)", rel([res.e.get("N_clust")], [cap.scalars.get("N_clust")])),
        ]
        for name, value in checks:
            ok = value <= 1e-9
            print(f"   {'ok  ' if ok else 'FAIL'} {name}: {value:.3e}")
            if not ok:
                failures += 1
        if rel(b_plain, b_s) < 1e-9:
            failures += 1
            print("   FAIL: heteroskedastic-W fit also matches -- the test is vacuous")

        # Default (non-cluster) GMM must be untouched by the fix.
        b_p, se_p = stata_bv(cap_plain, ORDER)
        r_plain = rel(res_plain.params.to_numpy(), b_p)
        print(f"   [control] default gmm coef rel = {r_plain:.3e}, "
              f"se rel = {rel(res_plain.bse.to_numpy(), se_p):.3e}")
        if r_plain > 1e-9 or rel(res_plain.bse.to_numpy(), se_p) > 1e-9:
            failures += 1
            print("   FAIL: the non-clustered GMM regression")

    failures += check_retest_panel()
    print(f"\n{'PASS' if failures == 0 else 'FAIL'} L3 ({failures} failure(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
