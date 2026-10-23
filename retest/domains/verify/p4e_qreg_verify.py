"""AUDIT probe 4e — qreg: is the tau=.5 robust VCE really verified?

Upstream claims `qreg, vce(robust)` matches Stata to 5.8e-9 at tau=.5 and is
3.5-4.6% "too high" at tau=.25. We test both on independent data, and probe
where the sparsity / bread differences come from.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

def run_case(tag, x1, x2, y, taus):
    df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))
    n = len(df)
    X = np.column_stack([x1, x2, np.ones(n)])
    print("#" * 76)
    print(f"## {tag}  n={n}")
    for tau in taus:
        q = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
        ri = q.fit(cov_type="iid"); rr = q.fit(cov_type="robust")
        pn = list(rr.param_names)
        order = [pn.index(k) for k in ("x1", "x2", "Intercept")]
        with S() as s:
            s.send(df)
            b, Vr, nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
            _b, Vi, _ = s.est(f"qreg y x1 x2, quantile({tau})")
            sp_s, bw_s = s.e("sparsity"), s.e("bwidth")
        ii = [nm.index(k) for k in ("x1", "x2", "_cons")]
        Vr = Vr[np.ix_(ii, ii)]; Vi = Vi[np.ix_(ii, ii)]
        sa, sr = np.sqrt(np.diag(Vi)), np.sqrt(np.diag(Vr))
        wa, wr = np.asarray(ri.bse)[order], np.asarray(rr.bse)[order]
        print(f"\n  tau={tau}")
        print(f"    iid    : Stata {np.round(sa,10)}  westata {np.round(wa,10)}"
              f"  rel={rel(wa, sa):.3e}")
        print(f"    robust : Stata {np.round(sr,10)}  westata {np.round(wr,10)}"
              f"  rel={rel(wr, sr):.3e}")
        print(f"    sparsity: Stata {sp_s!r} westata {q._sparsity['sparsity']!r}"
              f"  bwidth: Stata {bw_s!r} westata {q._sparsity['bandwidth']!r}")
        print(f"    Stata robust/iid ratio   = {np.round(sr/sa, 8)}")
        print(f"    westata robust/iid ratio = {np.round(wr/wa, 8)}")


rng = np.random.default_rng(4242)
n = 400
a1 = rng.normal(size=n); a2 = rng.normal(size=n)
sig = np.exp(0.2 + 0.4 * a1)
run_case("heteroskedastic DGP", a1, a2, 1.0 + 0.75 * a1 - 0.45 * a2 + sig * rng.normal(size=n),
         (0.25, 0.5, 0.75))

rng = np.random.default_rng(99)
b1 = rng.normal(size=n); b2 = rng.normal(size=n)
run_case("homoskedastic DGP", b1, b2, 1.0 + 0.75 * b1 - 0.45 * b2 + rng.normal(size=n),
         (0.25, 0.5, 0.75))

# Stata's own auto.dta as a third, real dataset
auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
run_case("auto.dta price~weight",
         auto["weight"].to_numpy(float),
         auto["length"].to_numpy(float),
         auto["price"].to_numpy(float), (0.25, 0.5, 0.75))
