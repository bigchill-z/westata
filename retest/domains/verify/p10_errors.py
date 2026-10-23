"""AUDIT probe 10 — all `error`-status items: are they genuine capability gaps,
or did upstream call the API wrongly / with unaligned parameters?

This matters: an "unsupported" that is really a wrong call is a false accusation.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

import westata as ws

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
A = pd.DataFrame({c: auto[c].to_numpy(float) for c in
                  ("mpg", "weight", "price", "foreign", "length", "rep78")})

rng = np.random.default_rng(20261024)
n = 400
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
yb = (x1 - 0.5 * x2 + rng.logistic(size=n) > 0).astype(int)
D = pd.DataFrame(dict(x1=x1, x2=x2, yb=yb,
                      w=rng.integers(1, 4, n).astype(float)))

def probe(tag, fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn()
        print(f"  {tag:62s} RAN  -> {str(out)[:90]}")
    except Exception as exc:
        print(f"  {tag:62s} {type(exc).__name__}: {str(exc)[:110]}")


print("=" * 92)
print("### discrete errors")
probe("Logit.fit(cov_type='HC0')", lambda: ws.Logit.from_formula("yb ~ x1 + x2", D).fit(cov_type="HC0").bse)
probe("Logit.fit(cov_type='HC1')", lambda: ws.Logit.from_formula("yb ~ x1 + x2", D).fit(cov_type="HC1").bse)
probe("Probit.fit(cov_type='HC0')", lambda: ws.Probit.from_formula("yb ~ x1 + x2", D).fit(cov_type="HC0").bse)
probe("NegativeBinomial.fit(cov_type='HC0')",
      lambda: ws.NegativeBinomial.from_formula("yb ~ x1 + x2", D).fit(cov_type="HC0").bse)
probe("OrderedLogit.fit(cov_type='HC0')",
      lambda: ws.OrderedLogit.from_formula("yb ~ x1 + x2", D).fit(cov_type="HC0").bse)
probe("Logit weights=[fweight]", lambda: ws.Logit.from_formula("yb ~ x1 + x2", D, weights=D.w).fit())
probe("Poisson cov_type='HC1' vs 'HC0' identical?",
      lambda: np.array_equal(np.asarray(ws.Poisson.from_formula("yb ~ x1 + x2", D).fit(cov_type="HC0").cov_params()),
                            np.asarray(ws.Poisson.from_formula("yb ~ x1 + x2", D).fit(cov_type="HC1").cov_params())))
print("  NB1/NB2 switch: does NegativeBinomial accept a dispersion argument?")
try:
    ws.NegativeBinomial.from_formula("yb ~ x1 + x2", D).fit(dispersion="constant")
except TypeError as exc:
    print(f"      TypeError: {exc}")
except Exception as exc:
    print(f"      {type(exc).__name__}: {exc}")

print()
print("=" * 92)
print("### limited / system / panel errors")
probe("Heckman.fit(method='ml')", lambda: ws.Heckman.from_formula("y ~ x1 + select(s ~ z1)", D).fit(method="ml"))
probe("Frontier(distribution='exponential')",
      lambda: ws.Frontier.from_formula("mpg ~ weight + length", A, distribution="exponential").fit())
probe("Sureg.fit(cov_type='robust')",
      lambda: ws.Sureg.from_formula(["price ~ weight + mpg", "mpg ~ weight + foreign"], A.dropna()).fit(cov_type="robust"))
probe("XtGls(corr='ar1')",
      lambda: ws.XtGls.from_formula("mpg ~ weight", A.assign(id=np.arange(len(A)) // 6 % 10,
                                                            t=np.arange(len(A)) % 6),
                                    panels="heteroskedastic", corr="ar1").fit())
probe("XtAbond lags=2 param_names",
      lambda: ws.XtAbond(D.assign(id=np.arange(len(D)) // 6, t=np.arange(len(D)) % 6),
                         endog="yb", lags=2, regressors=["x1", "x2"], entity="id", time="t").fit().param_names)
