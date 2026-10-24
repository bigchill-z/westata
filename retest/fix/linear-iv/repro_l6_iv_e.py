"""L6 (IV part): fill the ``e()`` gaps for ``ivregress 2sls`` / ``liml`` / ``gmm``.

The retest recorded two coverage failures (items 15d and 15e):

* Stata stores ``e(rss)``, ``e(mss)``, ``e(kappa)``, ``e(k_endog)``,
  ``e(iterations)`` (and ``e(tss)`` for 2SLS) that westata did not store at all;
* Stata stores ``e(J)`` for ``ivregress gmm`` and no ``kappa`` there.

This script checks that every scalar Stata stores is now present *and* correct,
and that westata's extra convenience scalars are documented rather than
silently different.

Run::

    .venv\\Scripts\\python.exe retest/fix/linear-iv/repro_l6_iv_e.py
"""

from __future__ import annotations

import numpy as np

from _common import Stata, iv_panel, rel

#: Stata's own ``e()`` keys for each estimator (``ereturn list``), recorded live.
STATA_KEYS = {
    "2sls": ["N", "df_m", "df_r", "r2", "r2_a", "rmse", "rss", "mss", "tss", "rank",
             "kappa", "k_endog", "iterations", "chi2"],
    "liml": ["N", "df_m", "df_r", "r2", "r2_a", "rmse", "rss", "mss", "rank",
             "kappa", "k_endog", "iterations", "chi2"],
    "gmm": ["N", "df_m", "df_r", "r2", "r2_a", "rmse", "rss", "mss", "rank",
            "k_endog", "iterations", "J", "chi2"],
}

#: Scalars whose value must match exactly (tier A).
NUMERIC = ["N", "rss", "mss", "kappa", "k_endog", "iterations", "J", "df_m", "r2",
           "r2_a", "rmse"]


def main() -> int:
    import westata as ws

    df = iv_panel()
    formula = "y ~ x1 + [e1 ~ z1 + z2]"
    classes = {"2sls": ws.IV2SLS, "liml": ws.IVLIML, "gmm": ws.IVGMM}
    failures = 0
    print("### L6 (IV)  e() coverage and values for the ivregress family")
    with Stata() as st:
        st.send(df)
        for name, cls in classes.items():
            for suffix, fit_kw, stata_suffix in (
                ("", {}, ""),
                (" vce(cluster gid)", dict(cov_type="cluster"), ", vce(cluster gid)"),
                (" vce(robust)", dict(cov_type="HC0"), ", vce(robust)"),
            ):
                cmd = f"ivregress {name} y x1 (e1 = z1 z2){stata_suffix}"
                fit_kw = dict(fit_kw)
                if "cluster" in stata_suffix:
                    fit_kw["cov_kwds"] = {"clusters": df["gid"]}
                cap = st.estimate(cmd)
                res = cls.from_formula(formula, df).fit(**fit_kw)
                stata_keys = {k for k, v in cap.scalars.items() if v is not None}
                missing = sorted(stata_keys - set(res.e.keys()))
                extra = sorted(set(res.e.keys()) - stata_keys)
                print(f"\n-- {name}{suffix}: {cmd}")
                print(f"   Stata keys  {sorted(stata_keys)}")
                print(f"   westata extra (documented conveniences): {extra}")
                if missing:
                    failures += 1
                    print(f"   FAIL missing in westata: {missing}")
                else:
                    print("   ok   no Stata e() scalar is missing")
                for key in NUMERIC:
                    s_val = cap.scalars.get(key)
                    if s_val is None:
                        continue
                    w_val = res.e.get(key)
                    r = rel([w_val], [s_val])
                    flag = "ok  " if r <= 1e-9 else "FAIL"
                    print(f"   {flag} e({key:10s}) stata={s_val!r:22s} westata={w_val!r}"
                          f"  rel={r:.3e}")
                    if r > 1e-9:
                        failures += 1

        # e(N_clust) for the clustered variants.
        cap = st.estimate("ivregress 2sls y x1 (e1 = z1 z2), vce(cluster gid)")
        res = ws.IV2SLS.from_formula(formula, df).fit(
            cov_type="cluster", cov_kwds={"clusters": df["gid"]}
        )
        print(f"\n-- e(N_clust): stata={cap.scalars.get('N_clust')!r} "
              f"westata={res.e.get('N_clust')!r}")
        if rel([res.e.get("N_clust")], [cap.scalars.get("N_clust")]) > 0:
            failures += 1
            print("   FAIL: N_clust mismatch")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'} L6/IV ({failures} failure(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
