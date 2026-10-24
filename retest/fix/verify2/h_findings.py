"""verify2 audit H: the two findings, reproduced in one standalone script.

H1  nl's rank guard is scale-dependent: it rejects an identified quadratic and
    the verdict flips when the regressor is multiplied by a constant.
H2  panel e(df_m) / e(rank) under a clustered VCE.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/fix/verify2")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import westata as ws
from mybriage import S, fmt, rel, tier

AUTO = "D:/Program Files/StataNow19/auto.dta"


def h1():
    print("=" * 100)
    print("H1. nl rank guard: scale dependence (a capability regression)")
    print("=" * 100)
    s = S().start()
    rng = np.random.default_rng(4242)
    n = 200
    w = rng.uniform(2000, 5000, n)
    y = 50.0 - 0.014 * w + 1.3e-6 * w**2 + rng.normal(scale=2.0, size=n)
    s.send(pd.DataFrame({"weight": w, "y": y}))
    E = s.est("nl (y = {b0} + {b1}*weight + {b2}*weight^2)")
    print(f"Stata estimates it:  b = {E.b}")
    print(f"                     SE = {E.se()}")

    print("\n westata, identical model, only the units of `weight` change:")
    print(f" {'weight x':>9s} {'cond(J)':>10s} {'smin/smax':>11s} "
          f"{'normalised cond':>16s}  verdict")
    for sc in (1.0, 1e-1, 1e-2, 1e-3, 1e-6):
        qq = pd.DataFrame({"weight": w * sc, "y": y})
        J = np.column_stack([np.ones(n), w * sc, (w * sc) ** 2])
        sv = np.linalg.svd(J, compute_uv=False)
        Jn = J / np.linalg.norm(J, axis=0)
        try:
            m = ws.Nl.from_formula("y = {b0} + {b1}*weight + {b2}*weight**2",
                                   qq).fit()
            back = np.array([1.0, sc, sc**2])
            bb, ss = m.params.to_numpy() * back, m.bse.to_numpy() * back
            rb = np.max(np.abs(bb - E.b) / np.abs(E.b))
            rs = np.max(np.abs(ss - E.se()) / np.abs(E.se()))
            verdict = f"OK  b rel={fmt(rb)} se rel={fmt(rs)} ({tier(rs)})"
        except Exception as exc:
            verdict = f"REFUSED ({type(exc).__name__})"
        print(f" {sc:9.0e} {np.linalg.cond(J):10.3e} {sv[-1]/sv[0]:11.3e} "
              f"{np.linalg.cond(Jn):16.3e}  {verdict}")

    print("\n the guard still catches a genuinely unidentified model:")
    rng2 = np.random.default_rng(5)
    z = rng2.normal(size=120)
    rd = pd.DataFrame({"z": z, "x": z.copy(),
                       "y": 1.0 + 2.0 * z + rng2.normal(size=120)})
    try:
        ws.Nl.from_formula("y = {a}*x + {c}*z", rd).fit()
        print("  NOT CAUGHT (unexpected)")
    except Exception as exc:
        print(f"  caught: {str(exc)[:110]}")
    ok = ws.Nl.from_formula("y = {a}*x", rd).fit()
    print(f"  the estimable submodel still fits: bse={ok.bse.to_numpy()}")

    print("\n auto.dta passes, but only just (this is why the BRIEF's own check is green):")
    au = pd.read_stata(AUTO)
    wau = au.weight.to_numpy(dtype=float)   # auto.weight is int16: square it as float
    Ja = np.column_stack([np.ones(len(au)), wau, wau**2])
    sva = np.linalg.svd(Ja, compute_uv=False)
    print(f"  auto.dta  cond(J) = {np.linalg.cond(Ja):.3e} ; "
          f"smin/smax = {sva[-1]/sva[0]:.3e} ; "
          f"guard cut-off = {200*np.finfo(float).eps/1e-5:.3e} "
          f"-> margin = {(sva[-1]/sva[0])/(200*np.finfo(float).eps/1e-5):.1f}x")


def h2():
    print("\n" + "=" * 100)
    print("H2. panel e(df_m) / e(rank) under a clustered VCE")
    print("=" * 100)
    s = S().start()
    rng = np.random.default_rng(5150)
    rows = []
    for i in range(30):
        a = rng.normal()
        for k in range(6):
            x1, x2 = rng.normal(), rng.normal()
            rows.append((i + 1, k + 1, x1, x2,
                         a + 0.6 * x1 - 0.3 * x2 + rng.normal()))
    df = pd.DataFrame(rows, columns=["id", "t", "x1", "x2", "y"])
    s.send(df)
    s.q("xtset id t")
    for k in (2, 1):
        spec = "xtreg y x1 x2, fe" if k == 2 else "xtreg y x1, fe"
        s.q(spec + " vce(cluster id)")
        sd = {key: s.sc(f"e({key})") for key in ("df_m", "df_b", "rank", "df_r", "F")}
        form = "y ~ x1 + x2" if k == 2 else "y ~ x1"
        w = ws.PanelOLS.from_formula(form, df, entity="id", time="t").fit(
            cov_type="cluster", cov_kwds={"clusters": df["id"].to_numpy()})
        wd = {key: w.e.get(key) for key in ("df_m", "df_b", "rank", "df_r", "F")}
        print(f"  {spec} vce(cluster id)  ({k} slopes)")
        print(f"     Stata   {sd}")
        print(f"     westata {wd}")
    print("\n  (the F statistic itself is correct to 6.5e-16 -- see b_panel.py B1)")


if __name__ == "__main__":
    h1()
    h2()
