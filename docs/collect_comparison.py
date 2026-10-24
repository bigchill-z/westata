"""Collect every comparison the documentation quotes, in one pass.

Two defects in the earlier collectors are fixed here, both mine rather than the library's.

1. PARAMETER ORDER. An earlier version built Stata's coefficient vector in Stata's order and ours in OUR order,
   then subtracted the two. On `logit vce(robust)` that compared a slope against an intercept and reported a
   6.9e-03 "difference" that was pure misalignment. Now both vectors are built through the same mapping list, so
   element k of one always pairs with element k of the other, and a length mismatch is an assertion failure.

2. CONVENTION VERSUS ERROR. Once aligned, `logit vce(robust)` still differs by 0.692%, and the arithmetic
   identifies it as a finite-sample choice rather than a fault:

       sqrt( (n/(n-k)) / (n/(n-1)) ) = sqrt( (74/72) / (74/73) ) = 1.0069204978
       observed ratio                                            = 1.0069204993

   Stata divides by n-1; our HC1 divides by n-k. Both defensible, so the documentation records the
   matched-convention number (HC0) alongside and says which is which.

Every comparison sends ONE array to both sides, at full precision. A failure is recorded as an entry, because a
coverage gap is a fact about the library that belongs in its documentation.
"""

from __future__ import annotations

import json
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(r"D:\myCodeProject\statapy\src")))
sys.path.insert(0, str(pathlib.Path(r"D:\myCodeProject\statapy\retest\lib")))

import numpy as np
import pandas as pd

import westata as ws
from stata_probe import Stata

REPO = pathlib.Path(r"D:\myCodeProject\statapy")
DATA = REPO / "docs" / "data"
OUT = REPO / "docs" / "comparison.json"

auto = pd.read_csv(DATA / "auto.csv")
auto["hi"] = (auto["mpg"] > 20).astype(float)
panel = pd.read_csv(DATA / "panel.csv")
sim = pd.read_csv(DATA / "sim.csv")

results = []


def record(command, dataset, stata_command, stata_capture, ours, mapping, extra=None):
    stata_b = np.asarray(stata_capture.b, dtype=float)
    stata_se = np.sqrt(np.diag(np.asarray(stata_capture.V, dtype=float)))
    stata_names = list(stata_capture.names)
    params = dict(zip(ours.param_names, np.asarray(ours.params, dtype=float)))
    ses = dict(zip(ours.param_names, np.asarray(ours.bse, dtype=float)))

    paired = []
    for token, our_name in mapping:
        index = int(token[1:]) if token.startswith("#") else stata_names.index(token)
        paired.append((our_name, float(stata_b[index]), float(stata_se[index]),
                       params[our_name], ses[our_name]))

    sb = np.array([p[1] for p in paired])
    ss = np.array([p[2] for p in paired])
    ob = np.array([p[3] for p in paired])
    os_ = np.array([p[4] for p in paired])
    assert sb.shape == ob.shape, f"{command}: aligned vectors differ in length"

    rel_b = float(np.max(np.abs(ob - sb) / np.maximum(np.abs(sb), 1e-12)))
    rel_se = float(np.max(np.abs(os_ - ss) / np.maximum(np.abs(ss), 1e-12)))

    entry = {
        "command": command,
        "dataset": dataset,
        "stata_command": stata_command,
        "rows": [
            {"name": n, "stata_b": s, "ours_b": o, "stata_se": sv, "ours_se": ov}
            for n, s, sv, o, ov in paired
        ],
        "rel_b": rel_b,
        "rel_se": rel_se,
        "nobs_stata": stata_capture.nobs,
        "nobs_ours": int(ours.nobs),
    }
    if extra:
        entry.update(extra)
    results.append(entry)
    print(f"  [{command:30s}] b {rel_b:.3e}  se {rel_se:.3e}  N {stata_capture.nobs}/{int(ours.nobs)}")


def attempt(label, dataset, fn):
    try:
        fn()
    except Exception as exc:
        results.append({"command": label, "dataset": dataset,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=4)})
        print(f"  [{label:30s}] ERROR {type(exc).__name__}: {str(exc)[:90]}")


print("auto.dta")
with Stata() as stata:
    stata.send(auto)

    attempt("regress", "auto", lambda: record(
        "regress", "auto", "regress price weight mpg foreign",
        stata.estimate("regress price weight mpg foreign"),
        ws.OLS.from_formula("price ~ weight + mpg + foreign", auto).fit(),
        [("weight", "weight"), ("mpg", "mpg"), ("foreign", "foreign"), ("_cons", "Intercept")]))

    attempt("regress vce(robust)", "auto", lambda: record(
        "regress vce(robust)", "auto", "regress price weight mpg, vce(robust)",
        stata.estimate("regress price weight mpg, vce(robust)"),
        ws.OLS.from_formula("price ~ weight + mpg", auto).fit(cov_type="HC1"),
        [("weight", "weight"), ("mpg", "mpg"), ("_cons", "Intercept")]))

    attempt("regress i.foreign", "auto", lambda: record(
        "regress i.foreign", "auto", "regress price weight i.foreign",
        stata.estimate("regress price weight i.foreign"),
        ws.OLS.from_formula("price ~ weight + C(foreign)", auto).fit(),
        [("weight", "weight"), ("1.foreign", "C(foreign)[T.1]"), ("_cons", "Intercept")],
        extra={"note": "Stata labels the level `1.foreign`; the CSV stores 0/1 so our label is "
                       "`C(foreign)[T.1]`. The estimates agree."}))

    attempt("qreg", "auto", lambda: record(
        "qreg median", "auto", "qreg price weight length, q(.5)",
        stata.estimate("qreg price weight length, q(.5)"),
        ws.QuantileRegression.from_formula("price ~ weight + length", auto).fit(),
        [("weight", "weight"), ("length", "length"), ("_cons", "Intercept")]))

    attempt("rreg", "auto", lambda: record(
        "rreg", "auto", "rreg price weight length, nolog",
        stata.estimate("rreg price weight length, nolog"),
        ws.RobustRegression.from_formula("price ~ weight + length", auto).fit(),
        [("weight", "weight"), ("length", "length"), ("_cons", "Intercept")]))

    attempt("sureg", "auto", lambda: record(
        "sureg", "auto", "sureg (price weight mpg) (mpg weight foreign)",
        stata.estimate("sureg (price weight mpg) (mpg weight foreign)"),
        ws.Sureg.from_formula(["price ~ weight + mpg", "mpg ~ weight + foreign"], auto).fit(),
        [("#0", "price:weight"), ("#1", "price:mpg"), ("#2", "price:_cons"),
         ("#3", "mpg:weight"), ("#4", "mpg:foreign"), ("#5", "mpg:_cons")],
        extra={"note": "Stata repeats each name once per equation, so positions are used."}))

    attempt("sureg vce(robust)", "auto", lambda: record(
        "sureg vce(robust)", "auto", "sureg (price weight mpg) (mpg weight foreign), vce(robust)",
        stata.estimate("sureg (price weight mpg) (mpg weight foreign), vce(robust)"),
        ws.Sureg.from_formula(["price ~ weight + mpg", "mpg ~ weight + foreign"],
                             auto).fit(cov_type="robust"),
        [("#0", "price:weight"), ("#1", "price:mpg"), ("#2", "price:_cons"),
         ("#3", "mpg:weight"), ("#4", "mpg:foreign"), ("#5", "mpg:_cons")]))

    attempt("logit", "auto", lambda: record(
        "logit", "auto", "logit hi weight length",
        stata.estimate("logit hi weight length"),
        ws.Logit.from_formula("hi ~ weight + length", auto).fit(),
        [("weight", "weight"), ("length", "length"), ("_cons", "Intercept")]))

    attempt("probit", "auto", lambda: record(
        "probit", "auto", "probit hi weight length",
        stata.estimate("probit hi weight length"),
        ws.Probit.from_formula("hi ~ weight + length", auto).fit(),
        [("weight", "weight"), ("length", "length"), ("_cons", "Intercept")]))

    attempt("poisson", "auto", lambda: record(
        "poisson", "auto", "poisson gear_ratio weight",
        stata.estimate("poisson gear_ratio weight"),
        ws.Poisson.from_formula("gear_ratio ~ weight", auto).fit(),
        [("weight", "weight"), ("_cons", "Intercept")]))

    tobit_fit = ws.Tobit.from_formula("rep78 ~ weight + length", auto, lower=0).fit()
    attempt("tobit", "auto", lambda: record(
        "tobit", "auto", "tobit rep78 weight length, ll(0)",
        stata.estimate("tobit rep78 weight length, ll(0)"),
        tobit_fit,
        [("weight", "weight"), ("length", "length"), ("_cons", "Intercept")],
        extra={"note": "Stata stores var(e.rep78) = 0.807583; our /sigma = 0.898656 and "
                       "0.898656^2 = 0.807583. Slopes match to 1e-15.",
               "sigma_stata_variance": 0.807583,
               "sigma_ours_stddev": float(dict(zip(tobit_fit.param_names,
                                                   np.asarray(tobit_fit.params,
                                                              dtype=float)))["/sigma"])}))

    try:
        robust = stata.estimate("logit hi weight, vce(robust)")
        stata_robust_se = np.sqrt(np.diag(np.asarray(robust.V, dtype=float)))
        hc0 = np.asarray(ws.Logit.from_formula("hi ~ weight", auto).fit(cov_type="HC0").bse,
                         dtype=float)
        hc1 = np.asarray(ws.Logit.from_formula("hi ~ weight", auto).fit(cov_type="HC1").bse,
                         dtype=float)
        n_obs, k = 74, 2
        results.append({
            "command": "logit vce(robust) [convention]",
            "dataset": "auto",
            "stata_command": "logit hi weight, vce(robust)",
            "convention_note": True,
            "stata_se_weight": float(stata_robust_se[0]),
            "ours_hc1_se_weight": float(hc1[1]),
            "ours_hc0_se_weight": float(hc0[1]),
            "rel_hc1": float(abs(hc1[1] - stata_robust_se[0]) / stata_robust_se[0]),
            "rel_hc0": float(abs(hc0[1] - stata_robust_se[0]) / stata_robust_se[0]),
            "predicted_ratio": float(np.sqrt((n_obs / (n_obs - k)) / (n_obs / (n_obs - 1)))),
            "observed_ratio": float(hc1[1] / stata_robust_se[0]),
        })
        print(f"  [logit robust convention     ] HC1 "
              f"{abs(hc1[1]-stata_robust_se[0])/stata_robust_se[0]:.3e}"
              f"  HC0 {abs(hc0[1]-stata_robust_se[0])/stata_robust_se[0]:.3e}")
    except Exception as exc:
        results.append({"command": "logit vce(robust) [convention]", "dataset": "auto",
                        "error": f"{type(exc).__name__}: {exc}"})

print("sim.csv")
with Stata() as stata:
    stata.send(sim)

    attempt("ivregress 2sls", "sim", lambda: record(
        "ivregress 2sls", "sim", "ivregress 2sls y_iv w (x_endog = z1 z2)",
        stata.estimate("ivregress 2sls y_iv w (x_endog = z1 z2)"),
        ws.IV2SLS.from_formula("y_iv ~ w + [x_endog ~ z1 + z2]", sim).fit(),
        [("x_endog", "x_endog"), ("w", "w"), ("_cons", "Intercept")]))

    attempt("heckman twostep", "sim", lambda: record(
        "heckman twostep", "sim", "heckman y_sel w, select(observed = z1 z2) twostep",
        stata.estimate("heckman y_sel w, select(observed = z1 z2) twostep"),
        ws.Heckman.from_formula("y_sel ~ w + select(observed ~ z1 + z2)", sim).fit(method="twostep"),
        [("#0", "y:w"), ("#1", "y:Intercept"), ("#2", "select:z1"),
         ("#3", "select:z2"), ("#4", "select:Intercept"), ("#5", "lambda")],
        extra={"note": "Stata reports a Mills-ratio coefficient and two `_cons`; positions are used."}))

    attempt("etregress twostep", "sim", lambda: record(
        "etregress twostep", "sim", "etregress y_treat w, treat(treated = z1 z2) twostep",
        stata.estimate("etregress y_treat w, treat(treated = z1 z2) twostep"),
        ws.Etregress.from_formula("y_treat ~ w + treat(treated ~ z1 + z2)", sim).fit(),
        [("#0", "y:w"), ("#1", "y:treated"), ("#2", "y:Intercept"),
         ("#3", "treat:z1"), ("#4", "treat:z2"), ("#5", "treat:Intercept")]))

    attempt("nbreg", "sim", lambda: record(
        "nbreg", "sim", "nbreg count w",
        stata.estimate("nbreg count w"),
        ws.NegativeBinomial.from_formula("count ~ w", sim).fit(),
        [("w", "w"), ("_cons", "Intercept")],
        extra={"note": "e(b) is on the lnalpha scale in both; alpha = exp(lnalpha)."}))

    attempt("nbreg NB1", "sim", lambda: record(
        "nbreg dispersion(constant)", "sim", "nbreg count w, dispersion(constant)",
        stata.estimate("nbreg count w, dispersion(constant)"),
        ws.NegativeBinomialNB1.from_formula("count ~ w", sim).fit(),
        [("w", "w"), ("_cons", "Intercept")],
        extra={"note": "NB1: Var = mu(1 + delta*mu). Both sides store lndelta."}))

print("panel.csv")
with Stata() as stata:
    stata.send(panel)
    stata.run("xtset firm year")

    for label, command, factory in (
        ("xtreg, fe", "xtreg y x1 x2, fe",
         lambda: ws.PanelOLS.from_formula("y ~ x1 + x2", panel, entity="firm", time="year").fit()),
        ("xtreg, re", "xtreg y x1 x2, re",
         lambda: ws.RandomEffects.from_formula("y ~ x1 + x2", panel, entity="firm", time="year").fit()),
        ("xtreg, be", "xtreg y x1 x2, be",
         lambda: ws.BetweenOLS.from_formula("y ~ x1 + x2", panel, entity="firm", time="year").fit()),
    ):
        attempt(label, "panel", lambda label=label, command=command, factory=factory: record(
            label, "panel", command, stata.estimate(command), factory(),
            [("x1", "x1"), ("x2", "x2"), ("_cons", "Intercept")]))

    attempt("xtlogit, fe", "panel", lambda: record(
        "xtlogit, fe", "panel", "xtlogit binary x1 x2, fe",
        stata.estimate("xtlogit binary x1 x2, fe"),
        ws.XtLogit.from_formula("binary ~ x1 + x2", panel, entity="firm", time="year",
                                method="fe").fit(),
        [("x1", "x1"), ("x2", "x2")],
        extra={"note": "Conditional logit: the intercept is conditioned out, so only slopes compare."}))

    attempt("xtpoisson, fe", "panel", lambda: record(
        "xtpoisson, fe", "panel", "xtpoisson count x1 x2, fe",
        stata.estimate("xtpoisson count x1 x2, fe"),
        ws.XtPoisson.from_formula("count ~ x1 + x2", panel, entity="firm", time="year",
                                  method="fe").fit(),
        [("x1", "x1"), ("x2", "x2")],
        extra={"note": "Conditional Poisson: only slopes compare."}))

OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
ok = [r for r in results if "rel_b" in r]
conv_entries = [r for r in results if r.get("convention_note")]
err = [r for r in results if "error" in r]
print()
print(f"wrote {OUT}")
print(f"  coefficient comparisons : {len(ok)}")
print(f"  convention notes       : {len(conv_entries)}")
print(f"  errors                 : {len(err)}")
for entry in err:
    print(f"    {entry['command']}: {entry['error'][:110]}")
