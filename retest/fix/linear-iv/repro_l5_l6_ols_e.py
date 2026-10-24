"""L5 + L6: ``e(df_r)`` under clustering and the missing ``e()`` scalars.

L5 (defect P1-16): westata stored ``e(df_r) = N - K`` under ``vce(cluster)``
while Stata stores ``G - 1``.  ``results.df_resid`` was already right, so the
two quantities had silently diverged.

L6 (defect 15a/15b/15c of the retest): ``e(rss)``, ``e(mss)``, ``e(sum_w)`` and
``e(N_clust)`` were not stored at all.  ``sum_w`` is the *raw* weight sum, which
is why ``aweight``/``pweight`` (whose estimation weights are renormalised to
sum to N) need the pre-normalisation value.

Run::

    .venv\\Scripts\\python.exe retest/fix/linear-iv/repro_l5_l6_ols_e.py
"""

from __future__ import annotations

import numpy as np

from _common import Stata, auto_frame, rel

#: Stata's e() keys for ``regress`` without a vce() option (``ereturn list``).
#: ``ll_0`` is checked separately: Stata does not store it for ``noconstant``.
SCALARS = ["N", "sum_w", "df_m", "df_r", "F", "r2", "r2_a", "rmse", "rss", "mss",
           "rank"]

CASES = [
    ("nonrobust", "regress mpg weight foreign", dict(), None),
    ("vce(robust)", "regress mpg weight foreign, vce(robust)", dict(cov_type="HC1"), None),
    ("vce(cluster gid)", "regress mpg weight foreign, vce(cluster gid)",
     dict(cov_type="cluster"), None),
    ("pweight", "regress mpg weight foreign [pweight=w]", dict(), "pweight"),
    ("aweight", "regress mpg weight foreign [aweight=w]", dict(), "aweight"),
    ("fweight", "regress mpg weight foreign [fweight=w]", dict(), "fweight"),
    ("iweight", "regress mpg weight foreign [iweight=w]", dict(), "iweight"),
    ("noconstant", "regress mpg weight foreign, noconstant", dict(), None),
]


def main() -> int:
    import westata as ws

    df = auto_frame()
    failures = 0
    print("### L5/L6  OLS e(df_r) under clustering + e(rss)/e(mss)/e(sum_w)/e(N_clust)")
    with Stata() as st:
        st.send(df)
        for label, cmd, fit_kw, weight_type in CASES:
            fit_kw = dict(fit_kw)
            if label == "vce(cluster gid)":
                fit_kw["cov_kwds"] = {"groups": df["gid"]}
            formula = "mpg ~ 0 + weight + foreign" if label == "noconstant" else "mpg ~ weight + foreign"
            if weight_type is None:
                res = ws.OLS.from_formula(formula, df).fit(**fit_kw)
            else:
                res = ws.OLS.from_formula(
                    formula, df, weights=df["w"], weight_type=weight_type
                ).fit(**fit_kw)
            cap = st.estimate(cmd)
            print(f"\n-- {label}: {cmd}")
            for key in SCALARS:
                s_val = cap.scalars.get(key)
                w_val = res.e.get(key)
                if key == "N_clust":
                    continue
                r = rel([w_val], [s_val])
                flag = "ok  " if r <= 1e-9 else "FAIL"
                print(f"   {flag} e({key:6s}) stata={s_val!r:22s} westata={w_val!r}"
                      f"  rel={r:.3e}")
                if r > 1e-9:
                    failures += 1
            # L5: df_r must be G - 1 under clustering.
            if "cluster" in cmd:
                s_nclust = cap.scalars.get("N_clust")
                print(f"   Stata e(N_clust) = {s_nclust!r}")
                for key, expected in (
                    ("N_clust", s_nclust),
                    ("df_r", s_nclust - 1 if s_nclust else None),
                ):
                    w_val = res.e.get(key)
                    r = rel([w_val], [expected])
                    flag = "ok  " if r <= 1e-12 else "FAIL"
                    print(f"   {flag} e({key:8s}) expected={expected!r} westata={w_val!r}")
                    if r > 1e-12:
                        failures += 1
                # The two df concepts must stay in step with results.df_resid.
                if res.df_resid != res.e["df_r"]:
                    failures += 1
                    print(f"   FAIL: results.df_resid={res.df_resid} != e(df_r)={res.e['df_r']}")
                if not np.isclose(res.e["rmse"], np.sqrt(res.e["rss"] / (res.e["N"] - res.e["rank"]))):
                    failures += 1
                    print("   FAIL: e(rmse) no longer uses the N-K denominator")
            # Consistency identity Stata guarantees.
            print(f"   info e(rss)+e(mss) = {res.e['rss'] + res.e['mss']:.15g}")
            # Stata stores no e(ll_0) for noconstant; elsewhere it is the
            # intercept-only likelihood, which does not move with the VCE.
            s_ll0 = cap.scalars.get("ll_0")
            w_ll0 = res.e.get("ll_0")
            if s_ll0 is None:
                ok = w_ll0 is None
                print(f"   {'ok  ' if ok else 'FAIL'} e(ll_0) absent on both sides"
                      f" (Stata={s_ll0!r}, westata={w_ll0!r})")
                if not ok:
                    failures += 1
            elif weight_type == "aweight":
                # Known, pre-existing weight-convention offset (see FIXES.md):
                # report it explicitly rather than pretending it is fixed.
                print(f"   info e(ll_0) aweight: Stata={s_ll0!r} westata={w_ll0!r}"
                      f"  rel={rel([w_ll0], [s_ll0]):.3e} (known offset)")
            else:
                r = rel([w_ll0], [s_ll0])
                flag = "ok  " if r <= 1e-9 else "FAIL"
                print(f"   {flag} e(ll_0 ) stata={s_ll0!r:22s} westata={w_ll0!r}"
                      f"  rel={r:.3e}")
                if r > 1e-9:
                    failures += 1

        # G - 1 vs N - K must be visibly different for this data (G = 12, N = 74).
        cap = st.estimate("regress mpg weight foreign, vce(cluster gid)")
        res = ws.OLS.from_formula("mpg ~ weight + foreign", df).fit(
            cov_type="cluster", cov_kwds={"groups": df["gid"]}
        )
        print(f"\n-- separation check: Stata e(df_r)={cap.scalars['df_r']!r} "
              f"(G-1), N-K={len(df) - 3} -> westata e(df_r)={res.e['df_r']!r}")
        if res.e["df_r"] == len(df) - 3:
            failures += 1
            print("   FAIL: e(df_r) is still N-K under clustering")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'} L5/L6 ({failures} failure(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
