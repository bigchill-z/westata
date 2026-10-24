"""S6/S7 -- jackknife residual degrees of freedom and bootstrap's reference distribution.

Run:  .venv\\Scripts\\python.exe retest/fix/systems/repro_S6_S7.py

S6  ``jackknife: regress mpg weight`` stores ``e(df_r) = 73 = N - 1``; westata
    reports ``N - k = 72``, so every t test / confidence interval uses the wrong
    critical value (t(72)=1.99346 vs t(73)=1.99300).
S7  ``bootstrap, reps(100): regress mpg weight`` reports ``z`` / ``P>|z|`` and
    stores no ``e(df_r)``; westata sets ``dist="t"`` with ``df_resid = N - k``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from _common import auto_frame, banner, header, rel_diff, row, Stata

import westata as ws


def main() -> int:
    auto = auto_frame()
    n = len(auto)

    with Stata() as st:
        st.run("sysuse auto, clear")
        st.run("set seed 20241024")

        jk = st.estimate("jackknife: regress mpg weight")
        jk_df_r = jk.scalars.get("df_r")
        jk_b = jk.b
        jk_se = np.sqrt(np.diag(jk.V))
        jk_out = st.output("jackknife: regress mpg weight")

        jk_mse = st.estimate("jackknife, mse: regress mpg weight")
        jk_mse_df_r = jk_mse.scalars.get("df_r")

        jk_mean = st.estimate("jackknife: mean mpg")
        jk_mean_df_r = jk_mean.scalars.get("df_r")

        bs = st.estimate("bootstrap, reps(100) seed(20241024) nodots: regress mpg weight")
        bs_df_r = bs.scalars.get("df_r")
        bs_macros = {k: v for k, v in bs.macros.items()
                     if k in ("cmd", "dist", "vcetype", "prefix")}
        bs_out = st.output(
            "bootstrap, reps(100) seed(20241024) nodots: regress mpg weight")

        # does jackknife also differ for a 3-parameter model?
        jk3 = st.estimate("jackknife: regress mpg weight foreign")
        jk3_df_r = jk3.scalars.get("df_r")
        jk3_se = np.sqrt(np.diag(jk3.V))

        # and a "mean" estimator, where k is not a design-matrix width
        st.run("sysuse auto, clear")

    banner("S6  jackknife: e(df_r) and the resulting t reference distribution")
    print(f"  N = {n}")
    print(f"  Stata e(df_r): jackknife regress       = {jk_df_r!r}   (= N-1: {n - 1})")
    print(f"  Stata e(df_r): jackknife, mse regress  = {jk_mse_df_r!r}")
    print(f"  Stata e(df_r): jackknife: mean         = {jk_mean_df_r!r}")
    print(f"  Stata e(df_r): jackknife: regress(3 b) = {jk3_df_r!r}")
    print()
    print("  jackknife output header (the 'df' count is what e(df_r) must encode):")
    for line in jk_out.splitlines():
        if any(tok in line for tok in ("observations", "replications", "df_r",
                                       "Residual", "Total")):
            print("    " + line.strip())

    before_jk = ws.Jackknife.from_formula("mpg ~ weight", auto).fit()
    before_jk3 = ws.Jackknife.from_formula("mpg ~ weight + foreign", auto).fit()
    before_jk_mse = ws.Jackknife.from_formula("mpg ~ weight", auto).fit(mse=True)
    print()
    print(f"  westata df_resid: regress       = {before_jk.df_resid!r}"
          f"   e(df_r) = {before_jk.e['df_r']!r}")
    print(f"  westata df_resid: mse regress   = {before_jk_mse.df_resid!r}")
    print(f"  westata df_resid: regress(3 b)  = {before_jk3.df_resid!r}")

    t72 = float(stats.t.ppf(0.975, 72))
    t73 = float(stats.t.ppf(0.975, 73))
    print()
    print(f"  critical value t(0.975, 72) = {t72:.10f}   t(0.975, 73) = {t73:.10f}"
          f"   rel gap {abs(t72 - t73) / t73:.3e}")
    ci_before = before_jk.conf_int()
    ci_stata = np.column_stack([jk_b - t73 * jk_se, jk_b + t73 * jk_se])
    header()
    for i, name in enumerate(ci_before.index):
        row(f"b({name})", before_jk.params.iloc[i], before_jk.params.iloc[i], jk_b[i])
    print("  CI (_cons) BEFORE:", ci_before.loc["Intercept"].tolist())
    print("  CI (_cons) STATA :", ci_stata[-1].tolist())

    banner("S7  bootstrap: reference distribution")
    print(f"  Stata e(df_r) after bootstrap = {bs_df_r!r}   (absent -> z)")
    print(f"  Stata macros: {bs_macros}")
    print("  bootstrap output table header:")
    for line in bs_out.splitlines():
        if any(tok in line for tok in ("|", "Std. err.", "Coef.")) and "Observed" in line:
            print("    " + line.strip())
    before_bs = ws.Bootstrap.from_formula(
        "mpg ~ weight", auto, reps=100, seed=20241024).fit()
    print(f"  westata dist = {before_bs.dist!r}  df_resid = {before_bs.df_resid!r}")
    print(f"  westata pvalues = {before_bs.pvalues.to_numpy()}")
    print(f"  westata CI(_cons) = {before_bs.conf_int().loc['Intercept'].tolist()}")
    z = float(stats.norm.ppf(0.975))
    bs_se = np.sqrt(np.diag(bs.V))
    print(f"  Stata  se = {bs_se}  z = {bs.b / bs_se}")
    print(f"  Stata  p (normal) = {2 * stats.norm.sf(np.abs(bs.b / bs_se))}")
    print(f"  Stata  CI = {np.column_stack([bs.b - z * bs_se, bs.b + z * bs_se]).tolist()}")
    print(f"  westata CI computed with z  = "
          f"{np.column_stack([before_bs.params.to_numpy() - z * before_bs.bse.to_numpy(), before_bs.params.to_numpy() + z * before_bs.bse.to_numpy()]).tolist()}")

    banner("S6/S7 verification")
    ok_jk = (
        float(before_jk3.df_resid) == float(jk3_df_r)
        and float(before_jk_mse.df_resid) == float(jk_mse_df_r)
    )
    ok_bs = before_bs.dist == "z"
    print(f"  3-parameter jackknife: westata df_resid {before_jk3.df_resid:.0f} vs "
          f"Stata {jk3_df_r:.0f}  -> {'OK' if float(before_jk3.df_resid) == float(jk3_df_r) else 'MISMATCH'}")
    print(f"  mse jackknife         : westata df_resid {before_jk_mse.df_resid:.0f} vs "
          f"Stata {jk_mse_df_r:.0f}  -> {'OK' if float(before_jk_mse.df_resid) == float(jk_mse_df_r) else 'MISMATCH'}")
    print(f"  bootstrap dist        : {before_bs.dist!r} (Stata: 'z')")
    print("  RESULT:", "PASS" if (ok_jk and ok_bs) else "FAIL")
    return 0 if (ok_jk and ok_bs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
