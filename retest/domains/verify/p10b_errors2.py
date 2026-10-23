"""AUDIT probe 10b — the remaining `error` claims, called correctly this time.

My earlier probe fed the wrong formulas; here every API call is built from the
real signature (inspected first) so a raised exception genuinely means
"westata does not implement this".
"""

from __future__ import annotations

import inspect
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

import westata as ws

print("  signatures:")
for cls, meth in (("Heckman", "from_formula"), ("Etregress", "from_formula"),
                  ("Frontier", "from_formula"), ("XtGls", "from_formula"),
                  ("XtAbond", "__init__"), ("XtPcse", "from_formula")):
    obj = getattr(ws, cls)
    f = getattr(obj, meth)
    print(f"    {cls}.{meth}{inspect.signature(f)}")

rng = np.random.default_rng(2024)
m = 400
z1 = rng.normal(size=m); x1 = rng.normal(size=m)
s = (z1 + rng.normal(size=m) > 0).astype(float)
y = 1.0 + 0.8 * x1 + 0.5 * (z1 + rng.normal(size=m))
Dh = pd.DataFrame(dict(y=np.where(s == 1, y, np.nan), s=s, x1=x1, z1=z1))

t = (z1 + rng.normal(size=m) > 0).astype(float)
Dt = pd.DataFrame(dict(y=1.0 + 0.6 * x1 + 1.2 * t + rng.normal(size=m), t=t, x1=x1, z1=z1))

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
A = pd.DataFrame({c: auto[c].to_numpy(float) for c in ("mpg", "weight", "length", "price", "foreign")})

def probe(tag, fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn()
        print(f"  {tag:56s} RAN -> {str(out)[:80]}")
    except Exception as exc:
        print(f"  {tag:56s} {type(exc).__name__}: {str(exc)[:110]}")

print("\n### heckman / etregress ML")
probe("Heckman('y ~ x1 + select(s ~ z1)').fit(method='ml')",
      lambda: ws.Heckman.from_formula("y ~ x1 + select(s ~ z1)", Dh).fit(method="ml"))
probe("Etregress('y ~ x1 + treat(t ~ z1)').fit(method='ml')",
      lambda: ws.Etregress.from_formula("y ~ x1 + treat(t ~ z1)", Dt).fit(method="ml"))
probe("Etregress.fit() default (control)",
      lambda: ws.Etregress.from_formula("y ~ x1 + treat(t ~ z1)", Dt).fit().params)
probe("Heckman.fit() default (twostep)",
      lambda: ws.Heckman.from_formula("y ~ x1 + select(s ~ z1)", Dh).fit().params)

print("\n### frontier exponential")
probe("Frontier halfnormal",
      lambda: ws.Frontier.from_formula("price ~ weight + length", A, distribution="halfnormal").fit().params)
probe("Frontier exponential",
      lambda: ws.Frontier.from_formula("price ~ weight + length", A, distribution="exponential").fit().params)

print("\n### sureg robust")
probe("Sureg default", lambda: ws.Sureg.from_formula(
    ["price ~ weight + mpg", "mpg ~ weight + foreign"], A.dropna()).fit().params)
probe("Sureg robust", lambda: ws.Sureg.from_formula(
    ["price ~ weight + mpg", "mpg ~ weight + foreign"], A.dropna()).fit(cov_type="robust"))

print("\n### xtgls corr(ar1)")
rng2 = np.random.default_rng(11)
G, T = 10, 8
rows = []
for g in range(G):
    ui = rng2.normal()
    for tt in range(T):
        xx = rng2.normal()
        rows.append((g, tt, 1.0 + 0.7 * xx + ui + rng2.normal(0, .5), xx))
Dp = pd.DataFrame(rows, columns=["id", "t", "y", "x1"])
probe("XtGls panels=heteroskedastic, corr=independent",
      lambda: ws.XtGls.from_formula("y ~ x1", Dp, entity="id", time="t",
                                    panels="heteroskedastic", corr="independent").fit().params)
probe("XtGls panels=heteroskedastic, corr=ar1",
      lambda: ws.XtGls.from_formula("y ~ x1", Dp, entity="id", time="t",
                                    panels="heteroskedastic", corr="ar1").fit().params)

print("\n### XtAbond lags=2 parameter names (naming offset defect)")
probe("XtAbond lags=1 param_names",
      lambda: ws.XtAbond(Dp.assign(y=np.asarray(Dp.y)), endog="y", lags=1,
                         regressors=["x1"], entity="id", time="t").fit().param_names)
probe("XtAbond lags=2 param_names",
      lambda: ws.XtAbond(Dp.assign(y=np.asarray(Dp.y)), endog="y", lags=2,
                         regressors=["x1"], entity="id", time="t").fit().param_names)

print("\n### nnmatch biasadj / svy over")
probe("NearestNeighbourMatching.fit(bias_adjust=...)",
      lambda: ws.NearestNeighbourMatching.from_formula(
          "y", "t", Dt, covariates=["x1"]).fit(bias_adjust=["x1"]))
probe("SvyMean over=",
      lambda: ws.SvyMean.from_frame(Dp, "y", over="id"))
