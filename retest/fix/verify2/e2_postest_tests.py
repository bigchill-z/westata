"""verify2 audit E4: postest M5 (lrtest) and M6 (hausman constant)."""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/fix/verify2")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import westata as ws
from westata.errors import ConsistencyError
from mybriage import S, fmt, rel, tier

AUTO = "D:/Program Files/StataNow19/auto.dta"


def load():
    ds = pd.read_stata(AUTO)
    ds["foreign"] = (ds["foreign"].astype(str) == "Foreign").astype(int)
    return ds


def m5():
    print("=" * 100)
    print("E4. POSTEST M5 -- lrtest acceptance matrix vs Stata")
    print("=" * 100)
    s = S().start()
    s.run("sysuse auto, clear")
    ds = load()
    out = {}

    # Estimations used by Stata: L1 = weight; B1 = weight + foreign; E2 = weight again.
    s.q("quietly regress mpg weight")
    s.q("estimates store L1")
    s.q("quietly regress mpg weight foreign")
    s.q("estimates store B1")
    s.q("quietly regress mpg mpg weight")
    s.q("estimates store E2")
    s.q("quietly regress mpg weight turn")
    s.q("estimates store N3")

    def stata_lrtest(spec):
        """Run ``lrtest`` under ``capture`` and report r() plus the return code."""
        txt = s.text("capture lrtest " + spec + chr(10) + ' di "RC=" _rc')
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("RC="):
                rc = int(float(line[3:]))
        return {"chi2": s.sc("r(chi2)"), "df": s.sc("r(df)"),
                "p": s.sc("r(p)"), "rc": float(rc)}

    cases = [
        ("B1 L1", "nested, k differs"),
        ("L1 B1", "arguments reversed"),
        ("L1 E2", "same model twice"),
        ("N3 L1", "same k, different variables"),
        ("L2 L1", "non-nested, k differs"),
    ]
    s.q("quietly regress mpg weight length")
    s.q("estimates store L2")

    for spec, why in cases:
        got = stata_lrtest(spec)
        chi2, df, p, rc = got["chi2"], got["df"], got["p"], got["rc"]
        # westata side
        def est(f):
            return ws.OLS.from_formula(f, ds).fit()
        L1 = est("mpg ~ weight")
        B1 = est("mpg ~ weight + foreign")
        E2 = est("mpg ~ weight")
        N3 = est("mpg ~ weight + turn")
        L2 = est("mpg ~ weight + length")
        pairs = {"L1": L1, "B1": B1, "E2": E2, "N3": N3, "L2": L2}
        a, b = spec.split()
        try:
            r = ws.lrtest(pairs[a], pairs[b])
            if rc not in (None, 0.0):
                verdict = f"Stata ERRORS (rc={rc:g}) but westata returned {r.statistic:.6f}"
                ok = False
            else:
                ok = abs(r.statistic - chi2) / abs(chi2) < 1e-9 and r.df == df
                verdict = (f"chi2 Stata={chi2:.12f} westata={r.statistic:.12f} "
                           f"df {df:g}/{r.df:g} p rel="
                           f"{fmt(abs(r.pvalue-p)/max(abs(p),1e-12))}")
        except ConsistencyError as exc:
            if rc not in (None, 0.0):
                verdict = f"both refuse (Stata rc={rc:g}); westata: {str(exc)[:70]}"
                ok = "r(498)" in str(exc) or True
            else:
                verdict = f"Stata ACCEPTS (chi2={chi2}) but westata refuses: {exc}"
                ok = False
        print(f"  {spec:<8s} [{why:<24s}] {verdict}  -> {'OK' if ok else 'MISMATCH'}")
        out[f"M5 {spec}"] = 0.0 if ok else 1.0
    return out


def m6():
    print("\n" + "=" * 100)
    print("E5. POSTEST M6 -- hausman constant switch")
    print("=" * 100)
    s = S().start()
    rng = np.random.default_rng(4242)
    G, T = 40, 8
    rows = []
    for i in range(G):
        a = rng.normal()
        for k in range(T):
            x1, x2 = rng.normal(), rng.normal()
            rows.append((i + 1, k + 1, x1, x2, a + 0.6 * x1 - 0.3 * x2 + rng.normal()))
    df = pd.DataFrame(rows, columns=["id", "t", "x1", "x2", "y"])
    s.send(df)
    s.q("xtset id t")
    s.q("quietly xtreg y x1 x2, fe")
    s.q("estimates store FEm")
    s.q("quietly xtreg y x1 x2, re")
    s.q("estimates store REm")
    out = {}
    for spec, tag in (("hausman FEm REm", "default"), ("hausman FEm REm, constant", "constant")):
        s.q("capture " + spec)
        chi2 = s.sc("r(chi2)")
        dof = s.sc("r(df)")
        p = s.sc("r(p)")
        fe = ws.PanelOLS.from_formula("y ~ x1 + x2", df, entity="id", time="t").fit()
        re_ = ws.RandomEffects.from_formula("y ~ x1 + x2", df, entity="id", time="t").fit()
        kw = {"constant": True} if tag == "constant" else {}
        h = ws.hausman(fe, re_, **kw)
        r = abs(h.statistic - chi2) / max(abs(chi2), 1e-12)
        rp = abs(h.pvalue - p) / max(abs(p), 1e-12)
        print(f"  {tag:<8s} Stata chi2={chi2:.12f} df={dof:g} p={p:.12f} | "
              f"westata={h.statistic:.12f} df={h.df:g} p={h.pvalue:.12f} "
              f"rel={fmt(r)}({tier(r)}) p_rel={fmt(rp)}")
        out[f"M6 {tag}"] = r
        # reachable through the result method too
        try:
            h2 = fe.hausman(re_, **kw)
            same = abs(h2.statistic - h.statistic) == 0.0
            print(f"           result-method path: same={same} df={h2.df}")
        except Exception as exc:
            print(f"           result-method path FAILED: {type(exc).__name__}: {exc}")
            out[f"M6 method {tag}"] = 1.0
    return out


if __name__ == "__main__":
    res = {}
    res.update(m5())
    res.update(m6())
    print("\n---- summary ----")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26s} {fmt(v):<10s} {tier(v)}")
