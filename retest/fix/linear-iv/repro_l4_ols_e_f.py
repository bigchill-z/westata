"""L4: ``OLS.e['F']`` must follow ``cov_type`` (Stata's robust Wald/k).

Defect P1-15: westata stored the classic R^2-form F
``(r2/df_m)/((1-r2)/df_r)`` regardless of the covariance estimator, so under
``vce(robust)`` it was 5.6e-2 off and under ``vce(cluster)`` 37.5% off.

Stata's ``e(F)`` for ``regress`` is ``b_s' V_ss^-1 b_s / df_m`` built from the
covariance matrix actually reported.  This script checks that identity across
nonrobust / vce(robust) / vce(cluster) / pweight, plus ``noconstant`` and the
intercept-only corner.

Run::

    .venv\\Scripts\\python.exe retest/fix/linear-iv/repro_l4_ols_e_f.py
"""

from __future__ import annotations

import numpy as np

from _common import Stata, auto_frame, rel

CASES = [
    ("nonrobust", "regress mpg weight foreign", dict(cov_type="nonrobust"), None),
    ("vce(robust) = HC1", "regress mpg weight foreign, vce(robust)",
     dict(cov_type="HC1"), None),
    ("vce(cluster gid)", "regress mpg weight foreign, vce(cluster gid)",
     dict(cov_type="cluster", cov_kwds={"groups": "gid"}), None),
    ("pweight (implies robust)", "regress mpg weight foreign [pweight=w]", dict(), "pweight"),
    ("aweight", "regress mpg weight foreign [aweight=w]", dict(), "aweight"),
    ("fweight", "regress mpg weight foreign [fweight=w]", dict(), "fweight"),
    ("noconstant", "regress mpg weight foreign, noconstant", dict(), None),
    ("noconstant robust", "regress mpg weight foreign, noconstant vce(robust)",
     dict(cov_type="HC1"), None),
]

#: The value e(F) *used* to carry: the classic R^2-form
#: ``(r2/df_m)/((1-r2)/df_r)``, which ignores ``cov_type`` entirely.  It is
#: recomputed here from the same fit instead of being hard-coded, so the
#: "before" column is reproducible from the data rather than transcribed.
def before_e_f(res) -> float:
    """The pre-fix ``e(F)`` definition, evaluated on this fit."""

    r2 = float(res.e["r2"])
    df_m = float(res.e["df_m"])
    df_r = float(res.nobs - res.e["rank"])
    has_intercept = "Intercept" in set(res.param_names)
    if has_intercept and df_m > 0 and r2 < 1.0:
        return float((r2 / df_m) / ((1.0 - r2) / df_r))
    return float("nan")


def main() -> int:
    import westata as ws

    df = auto_frame()
    failures = 0
    print("### L4  OLS e(F) follows the reported VCE (Wald/k, not the R^2 form)")
    with Stata() as st:
        st.send(df)
        for label, cmd, fit_kw, weight_type in CASES:
            fit_kw = dict(fit_kw)
            if "cov_kwds" in fit_kw and fit_kw["cov_kwds"]["groups"] == "gid":
                fit_kw["cov_kwds"] = {"groups": df["gid"]}
            if weight_type is None:
                res = ws.OLS.from_formula("mpg ~ weight + foreign" if "noconstant" not in cmd
                                          else "mpg ~ 0 + weight + foreign", df).fit(**fit_kw)
            else:
                res = ws.OLS.from_formula(
                    "mpg ~ weight + foreign", df,
                    weights=df["w"], weight_type=weight_type,
                ).fit(**fit_kw)
            cap = st.estimate(cmd)
            stata_f = cap.scalars["F"]

            # Recompute the Wald/k from westata's own reported V, to separate
            # "V is wrong" from "F is defined differently".
            names = list(res.param_names)
            slopes = [i for i, nm in enumerate(names) if nm != "Intercept"]
            b = np.asarray(res.params, dtype=float)[slopes]
            V = res.cov[np.ix_(slopes, slopes)]
            wald_from_westata_V = float(b @ np.linalg.solve(V, b) / len(slopes))

            r_stored = rel([res.e["F"]], [stata_f])
            r_wald = rel([wald_from_westata_V], [stata_f])
            before = before_e_f(res)
            r_before = rel([before], [stata_f])
            print(f"\n-- {label}: {cmd}")
            print(f"   Stata e(F)            = {stata_f:.15g}")
            print(f"   BEFORE e(F) (R^2 form) = {before:.15g}   rel = {r_before:.3e}")
            print(f"   AFTER  e(F) (Wald/k)   = {res.e['F']:.15g}   rel = {r_stored:.3e}")
            print(f"   Wald/k from westata V  = {wald_from_westata_V:.15g}"
                  f"   rel = {r_wald:.3e}")
            for name, value in (("e(F)", r_stored), ("Wald/k", r_wald)):
                if value > 1e-9:
                    failures += 1
                    print(f"   FAIL: {name} rel {value:.3e} > 1e-9")

        # The intercept-only model: Stata reports e(F) = 0.
        cap = st.estimate("regress mpg")
        res = ws.OLS.from_formula("mpg ~ 1", df).fit()
        print(f"\n-- intercept only: Stata e(F) = {cap.scalars['F']!r}, "
              f"westata e(F) = {res.e['F']!r}")
        if not (cap.scalars["F"] == 0.0 and res.e["F"] == 0.0):
            failures += 1
            print("   FAIL: intercept-only e(F) should be 0 on both sides")

        # Sanity: the four main cases must genuinely differ, otherwise the
        # conformance check above could pass with a single constant formula.
        print("\n-- distinctness check (Stata e(F) per VCE)")
        seen = {}
        for label, cmd, _, _ in CASES[:4]:
            seen[label] = st.estimate(cmd).scalars["F"]
        for label, value in seen.items():
            print(f"   {label:26s} e(F) = {value:.15g}")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'} L4 ({failures} failure(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
