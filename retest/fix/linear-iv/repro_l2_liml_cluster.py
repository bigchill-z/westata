"""L2: ``IVLIML``'s ``vce(cluster)`` must default to CR0, like 2SLS and Stata.

Defect P1-14: ``IVLIML`` went through the backend covariance with
``debiased=True`` for every ``cov_type``, so its clustered VCE always carried
the CR1 factor ``G/(G-1) * (N-1)/(N-K)``.  Stata's ``ivregress ...,
vce(cluster g)`` is CR0 and only adds those factors with ``small``.  The
measured SE ratio was 1.045296408534764 -- exactly ``sqrt(CR1/CR0)``.

The fix routes LIML through the same ``SSC(adj=False)`` default as 2SLS.

Run::

    .venv\\Scripts\\python.exe retest/fix/linear-iv/repro_l2_liml_cluster.py
"""

from __future__ import annotations

import numpy as np

from _common import Stata, iv_panel, rel, stata_bv

ORDER = ["_cons", "x1", "e1"]


def main() -> int:
    import westata as ws
    from westata.protocols import SSC

    failures = 0
    cases = [
        ("140x14 2instr", iv_panel(), 2),
        ("140x14 3instr", iv_panel(seed=11, n=140, g=14, n_instr=3), 3),
        ("240x24 2instr", iv_panel(seed=7, n=240, g=24, n_instr=2), 2),
    ]
    print("### L2  IVLIML vce(cluster) is CR0 by default")
    for label, df, n_instr in cases:
        instrs = " ".join(f"z{j}" for j in range(1, n_instr + 1))
        formula = f"y ~ x1 + [e1 ~ {' + '.join(f'z{j}' for j in range(1, n_instr + 1))}]"
        cmd = f"ivregress liml y x1 (e1 = {instrs}), vce(cluster gid)"
        with Stata() as st:
            st.send(df)
            cap = st.estimate(cmd)
            cap_small = st.estimate(f"{cmd} small")
        _, se_stata = stata_bv(cap, ORDER)
        _, se_small = stata_bv(cap_small, ORDER)
        res = ws.IVLIML.from_formula(formula, df).fit(
            cov_type="cluster", cov_kwds={"clusters": df["gid"]}
        )
        res_small = ws.IVLIML.from_formula(formula, df).fit(
            cov_type="cluster", cov_kwds={"clusters": df["gid"]}, ssc=SSC(adj=True)
        )
        r_default = rel(res.bse.to_numpy(), se_stata)
        r_small = rel(res_small.bse.to_numpy(), se_small)
        n, k = len(df), 3
        n_groups = int(df["gid"].nunique())
        cr1_over_cr0 = float(
            np.sqrt(n_groups / (n_groups - 1.0) * (n - 1) / (n - k))
        )
        print(f"\n-- {label}: {cmd}")
        print(f"   Stata default se   {np.array2string(se_stata, precision=10)}")
        print(f"   westata default se {np.array2string(res.bse.to_numpy(), precision=10)}"
              f"   rel = {r_default:.3e}")
        print(f"   Stata `small`  se  {np.array2string(se_small, precision=10)}")
        print(f"   westata adj=True   {np.array2string(res_small.bse.to_numpy(), precision=10)}"
              f"   rel = {r_small:.3e}")
        print(f"   CR1/CR0 = {cr1_over_cr0:.12f}; "
              f"Stata small/default ratio measured "
              f"{np.array2string(se_small / se_stata, precision=12)}")
        if r_default > 1e-9:
            failures += 1
            print("   FAIL: LIML cluster default is not CR0")
        if r_small > 1e-9:
            failures += 1
            print("   FAIL: LIML cluster with SSC(adj=True) is not Stata's `small`")

        # 2SLS must keep behaving the same way (shared VCE path).
        cmd2 = f"ivregress 2sls y x1 (e1 = {instrs}), vce(cluster gid)"
        with Stata() as st:
            st.send(df)
            cap2 = st.estimate(cmd2)
        _, se2 = stata_bv(cap2, ORDER)
        res2 = ws.IV2SLS.from_formula(formula, df).fit(
            cov_type="cluster", cov_kwds={"clusters": df["gid"]}
        )
        r2 = rel(res2.bse.to_numpy(), se2)
        print(f"   2SLS cluster default rel = {r2:.3e}  (must also be A)")
        if r2 > 1e-9:
            failures += 1
            print("   FAIL: 2SLS cluster regressed")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'} L2 ({failures} failure(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
