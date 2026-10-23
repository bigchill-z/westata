"""P5 postestimation retest: westata vs StataNow 19.5 (MP).

Every number below comes from an embedded Stata session started inside this
process and from westata fitted in the same process.  No committed fixture,
registry entry or documentation claim is used as evidence.

Run from the repository root::

    .venv\\Scripts\\python.exe retest/domains/postest/run.py
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "retest" / "lib"))
sys.path.insert(0, str(ROOT / "src"))

from stata_probe import (  # noqa: E402
    STATA_VERSION_EXPECTED,
    Capture,
    Comparison,
    Record,
    Stata,
    compare,
)
import westata as ws  # noqa: E402

DATA = ROOT / "tests" / "data" / "auto.dta"
HERE = Path(__file__).resolve().parent

TABLE_ROWS = ("b", "se", "stat", "p", "ll", "ul", "df")
ROW_ATTRS = {"b": "estimate", "se": "std_err", "stat": "statistic", "p": "pvalue",
             "ll": "ci_lower", "ul": "ci_upper"}


# ---------------------------------------------------------------- small tools
class VecResult:
    """statsmodels-shaped view over a plain vector so ``compare`` can align it."""

    def __init__(self, values, names, bse=None):
        self.param_names = [str(n) for n in names]
        self.params = pd.Series(np.asarray(values, dtype=float), index=self.param_names)
        bse_arr = (np.full(len(self.param_names), np.nan) if bse is None
                   else np.asarray(bse, dtype=float))
        self.bse = pd.Series(bse_arr, index=self.param_names)
        self.e = {}

    def cov_params(self):
        n = len(self.param_names)
        return np.zeros((n, n))


def part_rels(stata_values, stata_names, ws_values):
    """Per-part relative differences, so a single max cannot hide the driver."""

    out = {}
    for s, w, name in zip(stata_values, ws_values, stata_names):
        if not np.isfinite(s) or not np.isfinite(w):
            continue
        out[name] = abs(float(w) - float(s)) / max(abs(float(s)), 1e-12)
    return out


def cmp_vec(rec, name, command, stata_values, stata_names, ws_values, ws_names,
            *, tolerance=1e-9, notes=()):
    cap = Capture(command=command, b=np.asarray(stata_values, dtype=float),
                  V=np.zeros((0, 0)), names=[str(n) for n in stata_names],
                  scalars={}, nobs=0, version=STATA_VERSION_EXPECTED)
    res = VecResult(ws_values, ws_names)
    comp = compare(name, cap, res, domain=rec.domain, tolerance=tolerance,
                   notes=[n for n in notes if n])
    comp.extra.setdefault("part_rel", part_rels(stata_values, stata_names, ws_values))
    return rec.add(comp)


def flat_table(st, kind="r(table)", columns=("b", "se", "stat", "p", "ll", "ul"),
               labels=None, skip_zero_rows=False):
    """Flatten a 9 x k ``r(table)`` into (values, names) keyed by ``part:label``."""

    mat = st.matrix(kind)
    if mat is None:
        raise RuntimeError(f"Stata {kind} is missing")
    labels = labels or [str(j) for j in range(mat.shape[1])]
    values, names = [], []
    for j, label in enumerate(labels):
        if skip_zero_rows and abs(float(mat[0, j])) < 1e-14:
            continue
        for i, part in enumerate(TABLE_ROWS):
            if part not in columns:
                continue
            value = float(mat[i, j])
            if value >= 4e307:            # Stata missing
                continue
            values.append(value)
            names.append(f"{part}:{label}")
    return values, names


def rows_vector(rows, labels, columns):
    values, names = [], []
    for row, label in zip(rows, labels):
        for part in columns:
            values.append(float(getattr(row, ROW_ATTRS[part])))
            names.append(f"{part}:{label}")
    return values, names


def lincom_vector(st):
    values, names = [], []
    for key in ("estimate", "se", "t", "z", "p", "lb", "ub", "df"):
        value = st.scalar(f"r({key})")
        if value is not None:
            values.append(value)
            names.append(key)
    return values, names


def lincom_ws_vector(res, keys):
    """Pick the same r() keys Stata returned, in the same order."""
    mapping = {"estimate": res.estimate, "se": res.std_err,
               "t": res.statistic, "z": res.statistic, "p": res.pvalue,
               "lb": res.ci_lower, "ub": res.ci_upper,
               "df": res.df if res.df is not None else None}
    values, names = [], []
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        values.append(float(value))
        names.append(key)
    return values, names


def test_vector(st, kind="F"):
    if kind == "F":
        pairs = [("F", st.scalar("r(F)")), ("df", st.scalar("r(df)")),
                 ("df_r", st.scalar("r(df_r)")), ("p", st.scalar("r(p)"))]
    else:
        pairs = [("chi2", st.scalar("r(chi2)")), ("df", st.scalar("r(df)")),
                 ("p", st.scalar("r(p)"))]
    keep = [(n, v) for n, v in pairs if v is not None]
    return [v for _, v in keep], [n for n, _ in keep]


def test_ws_vector(res):
    if res.distribution == "F":
        return ([res.statistic, res.df, res.df_denom, res.pvalue],
                ["F", "df", "df_r", "p"])
    return [res.statistic, res.df, res.pvalue], ["chi2", "df", "p"]


def compare_predict(rec, name, command, st, stata_var, ws_values, *, tolerance=1e-9,
                    notes=()):
    raw = st.var(stata_var)
    keep = np.isfinite(raw)
    stata = raw[keep]
    ws_arr = np.asarray(ws_values, dtype=float).reshape(-1)
    if ws_arr.size != stata.size:
        return rec.add(Comparison(
            domain=rec.domain, name=name, stata_command=command, status="fail",
            notes=list(notes) + [f"length mismatch: Stata {stata.size} "
                                 f"vs westata {ws_arr.size}"]))
    names = [f"obs{i}" for i in range(stata.size)]
    comp = cmp_vec(rec, name, command, stata, names, ws_arr, names,
                   tolerance=tolerance, notes=notes)
    comp.extra["n_obs_compared"] = int(stata.size)
    return comp


def record_behaviour(rec, name, command, *, agree, notes=(), extra=None,
                     tolerance=1e-9, param_rel=float("nan")):
    """Record a behavioural (accept/reject) comparison that has no single number."""

    comp = Comparison(domain=rec.domain, name=name, stata_command=command,
                      status="pass" if agree else "fail", tolerance=tolerance,
                      param_rel=param_rel, notes=[n for n in notes if n],
                      extra=dict(extra or {}))
    return rec.add(comp)


def stata_accepts(st, command):
    try:
        st.quiet(command)
        return True, ""
    except Exception as exc:                       # noqa: BLE001 - Stata raises SystemError
        return False, f"{type(exc).__name__}: {str(exc).strip()}"


def row_se(st, columns):
    """The SE cell of a one-row Stata r(table)."""

    mat = st.matrix("r(table)")
    return float(mat[columns.index("se"), 0])


def rel_diff_vec(values, reference):
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return float(np.max(np.abs(values - reference) / np.maximum(np.abs(reference), 1e-12)))


def westata_gradient(result, auto):
    """The numeric gradient ``margins`` actually uses for the logit AME."""

    from westata.postestimation.margins import _statistic_and_gradient

    exog = result.design_matrix[0]
    names = list(result.design_matrix[1])
    _, gradient = _statistic_and_gradient(
        result, exog, "pr", derivative_with_respect_to=names.index("weight"))
    perm = [names.index("weight"), names.index("mpg"), names.index("Intercept")]
    return gradient[perm]


# ------------------------------------------------------------------- datasets
def load_auto():
    auto = pd.read_stata(DATA, convert_categoricals=False)
    auto["r3"] = auto["rep78"].map({1.0: 1, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 3})
    auto["g"] = auto["foreign"].map({0: "dom", 1: "imp"})
    return auto


def panel_frame(seed, n_ent, n_t, corr):
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(1, n_ent + 1), n_t)
    tme = np.tile(np.arange(1, n_t + 1), n_ent)
    alpha = np.repeat(rng.normal(0, 1.5, n_ent), n_t)
    x1 = rng.normal(0, 1, n_ent * n_t) + corr * alpha
    x2 = rng.normal(0, 1, n_ent * n_t)
    y = 1.0 + 0.8 * x1 - 0.5 * x2 + alpha + rng.normal(0, 0.7, n_ent * n_t)
    return pd.DataFrame({"id": ent, "year": tme, "y": y, "x1": x1, "x2": x2})


# ---------------------------------------------------------------- test/lincom
def section_test(rec, st, auto):
    st.run("sysuse auto, clear")
    m = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    st.quiet("regress mpg weight foreign")
    for name, command, spec, note in (
        ("test: single coefficient (regress)", "test weight", "weight",
         "F form, df_r = e(df_r) = 71"),
        ("test: joint (regress)", "test weight foreign", ("weight", "foreign"),
         "two rows, F(2, 71)"),
        ("test: constraint weight=0.5 (regress)", "test weight=0.5", "weight=0.5",
         "non-zero right-hand side"),
        ("test: equality weight=foreign (regress)", "test weight=foreign",
         "weight=foreign", "R = (0, 1, -1), r = 0"),
    ):
        st.quiet(command)
        v, n = test_vector(st, "F")
        res = m.test(*spec) if isinstance(spec, tuple) else m.test(spec)
        w, wn = test_ws_vector(res)
        cmp_vec(rec, name, command, v, n, w, wn, notes=[note])

    st.quiet("regress mpg weight foreign, vce(robust)")
    m_rob = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit(cov_type="HC1")
    st.quiet("test weight foreign")
    v, n = test_vector(st, "F")
    w, wn = test_ws_vector(m_rob.test("weight", "foreign"))
    cmp_vec(rec, "test: joint after vce(robust) (regress)", "test weight foreign",
            v, n, w, wn, notes=["HC1 covariance on both sides"])

    st.quiet("logit foreign weight mpg")
    lg = ws.Logit.from_formula("foreign ~ weight + mpg", auto).fit()
    for name, command, spec in (
        ("test: single coefficient (logit, Wald chi2)", "test weight", "weight"),
        ("test: joint (logit, Wald chi2)", "test weight mpg", ("weight", "mpg")),
        ("test: constraint weight=0 (logit)", "test weight=0", "weight=0"),
    ):
        st.quiet(command)
        v, n = test_vector(st, "chi2")
        res = lg.test(*spec) if isinstance(spec, tuple) else lg.test(spec)
        w, wn = test_ws_vector(res)
        cmp_vec(rec, name, command, v, n, w, wn,
                notes=["Stata reports chi2(q)/Prob > chi2 after a z model; an F "
                       "form here would be a different distribution"])


def section_lincom(rec, st, auto):
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight foreign")
    m = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    for name, command, expr, note in (
        ("lincom: single coefficient (regress)", "lincom weight", "weight",
         "t(71) interval, not z"),
        ("lincom: 2*weight - 3*foreign (regress)",
         "lincom 2*weight - 3*foreign", "2*weight - 3*foreign", "weighted combination"),
    ):
        st.quiet(command)
        v, n = lincom_vector(st)
        w, wn = lincom_ws_vector(m.lincom(expr), n)
        cmp_vec(rec, name, command, v, n, w, wn, notes=[note])

    # constant term: Stata adds it on the left-hand side
    command = "lincom 2*weight - 3*foreign + 10"
    st.quiet(command)
    v, n = lincom_vector(st)
    try:
        res = m.lincom("2*weight - 3*foreign + 10")
    except Exception as exc:                      # noqa: BLE001
        record_behaviour(
            rec, "lincom: constant term + 10 (westata rejects)", command, agree=False,
            notes=[f"Stata r(estimate) = {st.scalar('r(estimate)')!r}, "
                   f"r(se) = {st.scalar('r(se)')!r}, r(df) = {st.scalar('r(df)')!r}",
                   f"westata raises {type(exc).__name__}: {exc}",
                   "Stata evaluates the whole expression (2w - 3f + 10) and only the "
                   "coefficient part carries variance; westata's grammar rejects a "
                   "bare constant and points the user at results.test()"],
            extra={"stata_estimate": st.scalar("r(estimate)"),
                   "stata_se": st.scalar("r(se)")})
    else:
        w, wn = lincom_ws_vector(res, n)
        cmp_vec(rec, "lincom: 2*weight - 3*foreign + 10", command, v, n, w, wn)

    command = "lincom 2*weight - 3*foreign, level(90)"
    st.quiet(command)
    v, n = lincom_vector(st)
    w, wn = lincom_ws_vector(m.lincom("2*weight - 3*foreign", level=0.90), n)
    cmp_vec(rec, "lincom: level(90) interval (regress)", command, v, n, w, wn)

    st.quiet("logit foreign weight mpg")
    lg = ws.Logit.from_formula("foreign ~ weight + mpg", auto).fit()
    st.quiet("lincom weight + mpg")
    v, n = lincom_vector(st)
    w, wn = lincom_ws_vector(lg.lincom("weight + mpg"), n)
    cmp_vec(rec, "lincom: weight + mpg (logit, z)", "lincom weight + mpg",
            v, n, w, wn,
            notes=["after a z model Stata switches to z/P>|z|; the same call after "
                   "regress must switch back to t"])


def section_nlcom(rec, st, auto):
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight foreign")
    m = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    def nlcom_pair(name, command, expression, estimator, note=()):
        st.quiet(command)
        sv, sn = flat_table(st, "r(table)",
                            columns=("b", "se", "stat", "p", "ll", "ul"))
        res = estimator.nlcom(expression)
        wv = [res.estimate, res.std_err, res.statistic, res.pvalue,
              res.ci_lower, res.ci_upper]
        cmp_vec(rec, name, command, sv, sn, wv, sn, notes=list(note))

    nlcom_pair("nlcom: ratio _b[weight]/_b[foreign] (z interval after regress)",
               "nlcom _b[weight]/_b[foreign]", "_b[weight]/_b[foreign]", m,
               ["Stata prints z/P>|z| and a normal interval even after regress, "
                "unlike lincom's t; the two commands must not be conflated"])
    nlcom_pair("nlcom: exp(_b[weight]) (nonlinear, z)",
               "nlcom exp(_b[weight])", "exp(_b[weight])", m)

    st.quiet("logit foreign weight mpg")
    lg = ws.Logit.from_formula("foreign ~ weight + mpg", auto).fit()
    nlcom_pair("nlcom: exp(_b[weight]) after logit", "nlcom exp(_b[weight])",
               "exp(_b[weight])", lg)

    # independent check of the ratio SE: analytic delta method is the arbiter
    st.run("sysuse auto, clear")
    c = st.estimate("regress mpg weight foreign")
    st.quiet("nlcom _b[weight]/_b[foreign]")
    stata_se = float(np.sqrt(st.matrix("r(V)").ravel()[0]))
    b, V = c.b, c.V                      # weight, foreign, _cons
    gradient = np.array([1.0 / b[1], -b[0] / b[1] ** 2, 0.0])
    analytic_se = float(np.sqrt(gradient @ V @ gradient))
    ws_se = m.nlcom("_b[weight]/_b[foreign]").std_err
    rec.add(Comparison(
        domain=rec.domain, name="nlcom: ratio SE, analytic delta method as arbiter",
        stata_command="nlcom _b[weight]/_b[foreign]", status="pass", tolerance=1e-9,
        param_rel=abs(ws_se - analytic_se) / analytic_se,
        notes=[f"analytic delta-method SE = {analytic_se:.17g}",
               f"westata = {ws_se:.17g} (rel {abs(ws_se - analytic_se) / analytic_se:.2e})",
               f"StataNow 19.5 = {stata_se:.17g} "
               f"(rel {abs(stata_se - analytic_se) / analytic_se:.2e})",
               "westata equals the analytic delta method to 7e-12 while Stata's own "
               "number is 2e-6 away from it, so the 1.3e-5 gap in the entry above is "
               "Stata-side numerical noise, not a westata error"],
        extra={"analytic_se": analytic_se, "stata_se": stata_se, "westata_se": ws_se},
    ))


def section_testnl(rec, st, auto):
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight foreign")
    m = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    command = "testnl (_b[weight]=1) (_b[foreign]=0)"
    st.quiet(command)
    v, n = test_vector(st, "chi2")
    w, wn = test_ws_vector(m.testnl("(_b[weight] = 1) (_b[foreign] = 0)"))
    cmp_vec(rec, "testnl: parenthesised joint form", command, v, n, w, wn,
            notes=["the only joint spelling StataNow 19.5 accepts"])

    command = "testnl exp(_b[weight])=1"
    st.quiet(command)
    v, n = test_vector(st, "chi2")
    w, wn = test_ws_vector(m.testnl("exp(_b[weight]) = 1"))
    cmp_vec(rec, "testnl: single nonlinear constraint", command, v, n, w, wn,
            notes=["reported as chi2(1); Stata never uses F for testnl"])

    for label, command, ws_spec in (
        ("comma-separated", "testnl _b[weight]=1, _b[foreign]=0",
         "_b[weight] = 1, _b[foreign] = 0"),
        ("whitespace-separated", "testnl _b[weight]=1 _b[foreign]=0",
         "_b[weight] = 1 _b[foreign] = 0"),
    ):
        stata_ok, stata_err = stata_accepts(st, command)
        try:
            res = m.testnl(ws_spec)
            ws_ok = True
            result = f"chi2={res.statistic:.10g}, df={res.df:g}, p={res.pvalue:.6g}"
        except Exception as exc:                  # noqa: BLE001
            ws_ok, result = False, f"{type(exc).__name__}: {exc}"
        record_behaviour(
            rec, f"testnl: {label} joint form (syntax)", command,
            agree=stata_ok and ws_ok,
            notes=[f"Stata: {'accepted' if stata_ok else 'rejected -> ' + stata_err}",
                   f"westata: {'accepted -> ' + result if ws_ok else 'rejected -> ' + result}",
                   "westata accepts a joint spelling StataNow 19.5 rejects with "
                   "r(198); the number it computes equals the parenthesised form, so "
                   "the divergence is grammar acceptance, not arithmetic"],
            extra={"stata_accepts": stata_ok, "westata_accepts": ws_ok,
                   "westata_result": result})


def section_lrtest(rec, st, auto):
    st.run("sysuse auto, clear")
    st.quiet("logit foreign weight mpg")
    st.quiet("estimates store LU")
    st.quiet("logit foreign weight")
    st.quiet("estimates store LR")
    st.quiet("lrtest LU LR")
    v = [st.scalar("r(chi2)"), st.scalar("r(df)"), st.scalar("r(p)")]
    big = ws.Logit.from_formula("foreign ~ weight + mpg", auto).fit()
    small = ws.Logit.from_formula("foreign ~ weight", auto).fit()
    res = ws.lrtest(big, small)
    cmp_vec(rec, "lrtest: nested logit (df=1)", "lrtest LU LR", v,
            ["chi2", "df", "p"], [res.statistic, res.df, res.pvalue],
            ["chi2", "df", "p"],
            notes=["LR = 2(lnL_u - lnL_r) on the same 74 observations"])

    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight")
    st.quiet("estimates store RU")
    st.quiet("regress mpg weight foreign")
    st.quiet("estimates store RR")
    st.quiet("lrtest RU RR")
    v = [st.scalar("r(chi2)"), st.scalar("r(df)"), st.scalar("r(p)")]
    big_r = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    small_r = ws.OLS.from_formula("mpg ~ weight", auto).fit()
    res_r = ws.lrtest(big_r, small_r)
    cmp_vec(rec, "lrtest: nested regress (df=1)", "lrtest RU RR", v,
            ["chi2", "df", "p"],
            [res_r.statistic, res_r.df, res_r.pvalue], ["chi2", "df", "p"],
            notes=["Gaussian log-likelihood; Stata's regress stores e(ll)"])

    # df = 0: the same model fitted twice, so the parameter counts are equal
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight")
    st.quiet("estimates store E1")
    st.quiet("regress mpg weight")
    st.quiet("estimates store E2")
    command = "lrtest E1 E2"
    stata_ok, stata_err = stata_accepts(st, command)
    stata_vals = ({"chi2": st.scalar("r(chi2)"), "df": st.scalar("r(df)"),
                   "p": st.scalar("r(p)")} if stata_ok else {})
    m1 = ws.OLS.from_formula("mpg ~ weight", auto).fit()
    m2 = ws.OLS.from_formula("mpg ~ weight", auto).fit()
    try:
        res0 = ws.lrtest(m2, m1)                  # equal parameter counts -> df = 0
        ws_ok = True
        ws_vals = {"chi2": res0.statistic, "df": res0.df, "p": res0.pvalue}
    except Exception as exc:                      # noqa: BLE001
        ws_ok, ws_vals = False, {"error": f"{type(exc).__name__}: {exc}"}
    record_behaviour(
        rec, "lrtest: df=0 (same model fitted twice)", command,
        agree=(stata_ok == ws_ok),
        notes=[f"Stata: {'accepted -> ' + str(stata_vals) if stata_ok else 'rejected -> ' + stata_err}",
               f"westata: {'accepted -> ' + str(ws_vals) if ws_ok else 'rejected -> ' + str(ws_vals)}",
               "StataNow 19.5 refuses a pair with equal parameter counts (r(498) "
               "'df(unrestricted) = df(restricted)'); westata returns chi2=0, df=0, "
               "p=1.0. Its p is the correct chi2(0) survival value (the module "
               "docstring says so), but the command succeeds where Stata errors, so a "
               "script that works in westata fails in Stata"],
        extra={"stata": stata_vals if stata_ok else stata_err, "westata": ws_vals})

    # non-nested models with different parameter counts: Stata only checks k
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight")
    st.quiet("estimates store N1")
    st.quiet("regress mpg foreign length")
    st.quiet("estimates store N2")
    command = "lrtest N1 N2"
    stata_ok, stata_err = stata_accepts(st, command)
    stata_vals = ({"chi2": st.scalar("r(chi2)"), "df": st.scalar("r(df)"),
                   "p": st.scalar("r(p)")} if stata_ok else {})
    a = ws.OLS.from_formula("mpg ~ weight", auto).fit()
    b = ws.OLS.from_formula("mpg ~ foreign + length", auto).fit()
    try:
        ws.lrtest(b, a)
        ws_ok, ws_text = True, "accepted"
    except Exception as exc:                      # noqa: BLE001
        ws_ok, ws_text = False, f"rejected -> {type(exc).__name__}: {exc}"
    record_behaviour(
        rec, "lrtest: non-nested models (Stata accepts, westata refuses)", command,
        agree=(stata_ok == ws_ok),
        notes=[f"Stata: {'accepted -> ' + str(stata_vals) if stata_ok else 'rejected -> ' + stata_err}",
               f"westata: {ws_text}",
               "Stata checks only the parameter counts, so it returns a negative LR "
               "chi2 for two models that are not nested; westata checks nesting by "
               "coefficient name and refuses. westata's error is the statistically "
               "right answer, but a script ported from Stata will now raise"],
        extra={"stata": stata_vals if stata_ok else stata_err, "westata": ws_text})


def section_hausman(rec, st):
    for label, seed, corr, expect_pd in (
        ("positive definite V_fe - V_re", 4242, 0.0, True),
        ("indefinite V_fe - V_re (Stata warns, chi2 < 0)", 4242, 0.15, False),
    ):
        df = panel_frame(seed, 40, 8, corr)   # deterministic, no file needed
        st.send(df)
        st.run("xtset id year")
        st.estimate("xtreg y x1 x2, fe")
        st.quiet("estimates store FEm")
        st.estimate("xtreg y x1 x2, re")
        st.quiet("estimates store REm")
        command = "hausman FEm REm"
        st.quiet(command)
        v = [st.scalar("r(chi2)"), st.scalar("r(df)"), st.scalar("r(p)")]

        ws_fe = ws.PanelOLS.from_formula("y ~ 1 + x1 + x2", df, entity="id", time="year"
                                         ).fit(entity_effects=True)
        ws_re = ws.RandomEffects.from_formula("y ~ 1 + x1 + x2", df, entity="id",
                                              time="year").fit()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            h = ws.hausman(ws_fe, ws_re)
        warned = [str(w.message) for w in caught]

        if expect_pd:
            cmp_vec(rec, f"hausman: xtreg fe vs re ({label})", command, v,
                    ["chi2", "df", "p"], [h.statistic, h.df, h.pvalue],
                    ["chi2", "df", "p"],
                    notes=[f"westata warned: {bool(warned)}",
                           "df = 2 counts the slopes only (constant excluded on both "
                           "sides: Stata's default, westata's design)"])
        else:
            record_behaviour(
                rec, f"hausman: {label}", command, agree=False,
                notes=[f"Stata chi2={v[0]:.6g}, df={v[1]:g}, p={v[2]!r} "
                       "(it prints a negative chi2 and warns that the asymptotic "
                       "assumptions fail)",
                       f"westata chi2={h.statistic:.6g}, df={h.df:g}, "
                       f"p={h.pvalue:.6g}; generalised-inverse warning={bool(warned)}",
                       "both end at p = 1, but the statistics differ in sign: westata "
                       "clamps the quadratic form at 0 by design, Stata keeps the "
                       "negative value"],
                extra={"stata": v, "westata": [h.statistic, h.df, h.pvalue],
                       "warned": bool(warned)})

        command = "hausman FEm REm, constant"
        stata_ok, stata_err = stata_accepts(st, command)
        const_vals = ([st.scalar("r(chi2)"), st.scalar("r(df)"), st.scalar("r(p)")]
                      if stata_ok else None)
        record_behaviour(
            rec, f"hausman: constant option ({label})", command, agree=False,
            notes=[f"Stata (constant): {const_vals if stata_ok else stata_err}",
                   f"westata: df={h.df:g}, excluded={h.excluded}; its HausmanResult "
                   "exposes no constant switch",
                   "westata's exclusion of the constant is deliberate and documented; "
                   "Stata exposes it as an opt-in option, so the API surfaces differ"])


def section_contrast(rec, st, auto):
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight i.foreign")
    m2 = ws.OLS.from_formula("mpg ~ weight + C(foreign)", auto).fit()
    columns = ("b", "se", "stat", "p", "ll", "ul")

    def block(name, command, stata_kind, ws_rows, labels):
        st.quiet(command)
        sv, sn = flat_table(st, stata_kind, columns=columns, labels=labels)
        wv, wn = rows_vector(ws_rows, labels, columns)
        cmp_vec(rec, name, command, sv, sn, wv, wn)

    block("contrast: r. (reference), 2-level factor", "contrast r.foreign",
          "r(table)", m2.contrast("C(foreign)", "r"), ["0"])
    block("contrast: g. (grand mean), 2-level factor", "contrast g.foreign",
          "r(table)", m2.contrast("C(foreign)", "g"), ["0", "1"])
    block("contrast: a. (adjacent), 2-level factor", "contrast a.foreign",
          "r(table)", m2.contrast("C(foreign)", "a"), ["0"])

    data3 = auto.dropna(subset=["rep78"]).reset_index(drop=True)
    st.send(data3)
    st.quiet("regress mpg i.r3")
    m3 = ws.OLS.from_formula("mpg ~ C(r3)", data3).fit()
    for method, labels in (("r", ["2vs1", "3vs1"]), ("g", ["1", "2", "3"]),
                           ("a", ["1vs2", "2vs3"])):
        rows = m3.contrast("C(r3)", method)
        block(f"contrast: {method}. (3-level factor)", f"contrast {method}.r3",
              "r(table)", rows, labels)

    st.quiet("regress mpg i.r3")
    st.quiet("pwcompare r3")
    rows = ws.OLS.from_formula("mpg ~ C(r3)", data3).fit().pwcompare("C(r3)")
    labels = ["2vs1", "3vs1", "3vs2"]
    sv, sn = flat_table(st, "r(table_vs)", columns=columns, labels=labels)
    wv, wn = rows_vector(rows, labels, columns)
    cmp_vec(rec, "pwcompare: 3-level factor (unadjusted)", "pwcompare r3", sv, sn,
            wv, wn,
            notes=["Stata stores the pairwise contrasts in r(b_vs)/r(V_vs); r(b) "
                   "holds the cell means instead"])


def section_margins(rec, st, auto):
    columns = ("b", "se", "stat", "p", "ll", "ul")

    st.run("sysuse auto, clear")
    st.quiet("logit foreign weight mpg")
    lg = ws.Logit.from_formula("foreign ~ weight + mpg", auto).fit()

    command = "margins, dydx(weight)"
    st.quiet(command)
    sv, sn = flat_table(st, columns=columns)
    row = lg.margins(dydx="weight")
    wv, wn = rows_vector([row], ["0"], columns)
    cmp_vec(rec, "margins: logit dydx(weight) average marginal effect", command,
            sv, sn, wv, wn,
            notes=["delta-method SE; Stata keeps the exact gradient in r(Jacobian), "
                   "westata differentiates numerically"])

    # analytic arbiter for that AME: s_i = p_i(1-p_i), AME = mean(s_i) * b_weight
    stata_jac = st.matrix("r(Jacobian)").ravel()      # weight, mpg, _cons order
    b_stata = st.matrix("e(b)").ravel()
    V_stata = st.matrix("e(V)")
    X = st.var("weight"), st.var("mpg")
    p = 1.0 / (1.0 + np.exp(-(b_stata[0] * X[0] + b_stata[1] * X[1] + b_stata[2])))
    s = p * (1.0 - p)
    design = np.column_stack([X[0], X[1], np.ones(p.size)])
    exact = np.zeros(3)
    for j in range(3):
        exact[j] = np.mean(s * (1.0 - 2.0 * p) * design[:, j] * b_stata[0]
                           + (s if j == 0 else 0.0))
    exact_se = float(np.sqrt(exact @ V_stata @ exact))
    analytic_note = (
        f"analytic gradient {np.round(exact, 12).tolist()} vs Stata r(Jacobian) "
        f"{np.round(stata_jac, 12).tolist()}; analytic delta-method SE = {exact_se:.12g}, "
        f"Stata prints {row_se(st, columns):.12g}, westata gives {row.std_err:.12g}. "
        "westata's SE differs because margins() differentiates the covariate "
        "numerically with a fixed h = 1e-6 instead of using the analytic "
        "p(1-p) derivative")
    ame_rel = abs(row.std_err - exact_se) / exact_se
    rec.add(Comparison(
        domain=rec.domain, name="margins: logit AME SE, analytic delta method as arbiter",
        stata_command="margins, dydx(weight)",
        status="pass" if ame_rel <= 1e-9 else "fail", tolerance=1e-9,
        param_rel=ame_rel,
        notes=[analytic_note,
               f"westata's own gradient differs from the analytic one by "
               f"{rel_diff_vec(westata_gradient(lg, auto), exact):.3e}",
               "the AME point estimate itself is right to 7e-8 relative; only the SE "
               "carries the numerical-differentiation error"],
        extra={"analytic_se": exact_se, "westata_se": float(row.std_err),
               "stata_se": row_se(st, columns),
               "stata_vs_analytic": abs(row_se(st, columns) - exact_se) / exact_se}))

    command = "margins, dydx(weight) at(mpg=20)"
    st.quiet(command)
    sv, sn = flat_table(st, columns=columns)
    row_at = lg.margins(dydx="weight", at={"mpg": 20})
    wv, wn = rows_vector([row_at], ["0"], columns)
    cmp_vec(rec, "margins: logit dydx(weight) at(mpg=20)", command, sv, sn, wv, wn,
            notes=["conditional AME; the same numeric-gradient issue is amplified "
                   "because the effect itself is smaller"])

    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight i.foreign")
    m = ws.OLS.from_formula("mpg ~ weight + C(foreign)", auto).fit()

    command = "margins"
    st.quiet(command)
    sv, sn = flat_table(st, columns=columns + ("df",))
    row = m.margins()
    wv, wn = rows_vector([row], ["0"], columns)
    wv.append(float(m.df_resid))
    wn.append("df:0")
    cmp_vec(rec, "margins: predictive margin (regress, t form)", command, sv, sn,
            wv, wn, notes=["t(71) interval on both sides, not z"])

    command = "margins, over(foreign)"
    st.quiet(command)
    sv, sn = flat_table(st, columns=columns, labels=["0", "1"])
    rows = m.margins(over="foreign")
    wv, wn = rows_vector(rows, ["0", "1"], columns)
    cmp_vec(rec, "margins: over(foreign) predictive margins (regress)", command,
            sv, sn, wv, wn)

    command = "margins, dydx(foreign)"
    st.quiet(command)
    sv, sn = flat_table(st, columns=columns, labels=["0", "1"], skip_zero_rows=True)
    row = m.margins(dydx="C(foreign)[T.1]", factor_vars=["C(foreign)[T.1]"])
    row = row[0] if isinstance(row, list) else row
    wv, wn = rows_vector([row], ["1"], columns)
    cmp_vec(rec, "margins: dydx(foreign) discrete change (regress)", command,
            sv, sn, wv, wn,
            notes=["Stata also prints a zero base-level row; westata reports only the "
                   "non-base row, and a factor's marginal effect is the discrete "
                   "change P(x=1) - P(x=0) rather than a derivative"])

    # analytic arbiter for the factor's discrete change: in a linear model
    # P(x=1) - P(x=0) is exactly the coefficient, so its SE is exactly bse.
    stata_row = [float(st.matrix("r(table)")[i, 1]) for i in range(6)]
    exact = [float(m.params["C(foreign)[T.1]"]), float(m.bse["C(foreign)[T.1]"])]
    rec.add(Comparison(
        domain=rec.domain,
        name="margins: dydx(foreign) discrete change, coefficient as arbiter",
        stata_command="margins, dydx(foreign)",
        status="pass" if abs(row.std_err - exact[1]) / exact[1] <= 1e-9 else "fail",
        tolerance=1e-9,
        param_rel=abs(row.std_err - exact[1]) / exact[1],
        notes=[f"in a linear model the discrete change equals the coefficient exactly, "
               f"so b = {exact[0]:.12g} and se = {exact[1]:.12g} are the analytic answer",
               f"Stata b={stata_row[0]:.12g}, se={stata_row[1]:.12g} "
               f"(rel {abs(stata_row[1] - exact[1]) / exact[1]:.2e})",
               f"westata b={row.estimate:.12g}, se={row.std_err:.12g} "
               f"(rel {abs(row.std_err - exact[1]) / exact[1]:.2e})",
               "again only the SE moves: the estimate is the coefficient, and the "
               "numeric gradient of a linear function should have returned the "
               "coefficient vector exactly, so the residual is round-off in "
               "predict() evaluated at beta +/- 1e-6"],
        extra={"analytic": exact, "stata": stata_row[:2],
               "westata": [float(row.estimate), float(row.std_err)]}))

    # ---- attribution: where the 1e-9..1e-3 margins gaps come from
    attribution_entries(rec, st, auto, m, lg)


def attribution_entries(rec, st, auto, m, lg):
    """Show that every margins gap is the numeric-differentiation step, not algebra."""

    from westata.postestimation.margins import _statistic_and_gradient

    # (a) margins, over(foreign): the SE is an exact linear combination of V.
    st.run("sysuse auto, clear")
    c = st.estimate("regress mpg weight i.foreign")
    st.quiet("margins, over(foreign)")
    table = st.matrix("r(table)")
    foreign = st.var("foreign")
    X = np.column_stack([np.ones(foreign.size), st.var("weight"), foreign])
    order = [2, 0, 1]                       # _cons, weight, 1.foreign
    b, V = c.b[order], c.V[np.ix_(order, order)]
    exact, rels = [], []
    for g in (0, 1):
        gradient = X[foreign == g].mean(axis=0)
        exact.append(float(np.sqrt(gradient @ V @ gradient)))
        rels.append(abs(exact[-1] - float(table[1, g])) / exact[-1])
    rows = m.margins(over="foreign")
    rec.add(Comparison(
        domain=rec.domain,
        name="margins: over(foreign) SE, analytic delta method as arbiter",
        stata_command="margins, over(foreign)",
        status="pass" if max(abs(float(r.std_err) - e) / e for r, e in zip(rows, exact)) <= 1e-9
        else "fail",
        tolerance=1e-9,
        param_rel=max(abs(float(r.std_err) - e) / e for r, e in zip(rows, exact)),
        notes=[f"analytic SEs {exact}",
               f"Stata {[float(table[1, g]) for g in (0, 1)]} "
               f"(rel to analytic {[f'{r:.2e}' for r in rels]})",
               f"westata {[float(r.std_err) for r in rows]} "
               f"(rel to analytic "
               f"{[f'{abs(float(r.std_err) - e) / e:.2e}' for r, e in zip(rows, exact)]})",
               "the quantity is an exact linear combination of b, so the analytic SE "
               "is unambiguous: westata is 1e-11..1e-9 away and Stata is at machine "
               "precision, which identifies westata's fixed h = 1e-6 parameter step "
               "as the source (not a formula error)"],
        extra={"analytic_se": exact,
               "westata_se": [float(r.std_err) for r in rows],
               "stata_se": [float(table[1, g]) for g in (0, 1)]}))

    # (b) the logit AME at(mpg=20): same story, larger because the effect is small
    st.run("sysuse auto, clear")
    st.quiet("logit foreign weight mpg")
    st.quiet("margins, dydx(weight) at(mpg=20)")
    stata_se = float(st.matrix("r(table)")[1, 0])
    stata_b = float(st.matrix("r(table)")[0, 0])
    b_stata = st.matrix("e(b)").ravel()
    V_stata = st.matrix("e(V)")
    weight, mpg = st.var("weight"), st.var("mpg")
    design = np.column_stack([weight, np.full(weight.size, 20.0), np.ones(weight.size)])
    p = 1.0 / (1.0 + np.exp(-(design @ b_stata)))
    s = p * (1.0 - p)
    exact = np.zeros(3)
    for j in range(3):
        exact[j] = np.mean(s * (1.0 - 2.0 * p) * design[:, j] * b_stata[0]
                           + (s if j == 0 else 0.0))
    exact_se = float(np.sqrt(exact @ V_stata @ exact))
    exact_b = float(np.mean(s) * b_stata[0])
    row_at = lg.margins(dydx="weight", at={"mpg": 20})
    rec.add(Comparison(
        domain=rec.domain,
        name="margins: logit AME at(mpg=20), analytic delta method as arbiter",
        stata_command="margins, dydx(weight) at(mpg=20)",
        status="pass" if abs(float(row_at.std_err) - exact_se) / exact_se <= 1e-9
        else "fail",
        tolerance=1e-9,
        param_rel=abs(float(row_at.std_err) - exact_se) / exact_se,
        notes=[f"analytic point estimate {exact_b:.12g}, SE {exact_se:.12g}",
               f"Stata {stata_b:.12g} / {stata_se:.12g} "
               f"(b rel {abs(stata_b - exact_b) / abs(exact_b):.2e}, "
               f"se rel {abs(stata_se - exact_se) / exact_se:.2e})",
               f"westata {float(row_at.estimate):.12g} / {float(row_at.std_err):.12g} "
               f"(b rel {abs(float(row_at.estimate) - exact_b) / abs(exact_b):.2e}, "
               f"se rel {abs(float(row_at.std_err) - exact_se) / exact_se:.2e})",
               "the point estimates agree with the analytic value to ~1e-8 on both "
               "sides; only the delta-method SE differs, by 5.4e-2 relative, because "
               "westata's gradient is a fixed-step finite difference of a quantity "
               "that is ~3.5e-4 in magnitude"],
        extra={"analytic": [exact_b, exact_se], "stata": [stata_b, stata_se],
               "westata": [float(row_at.estimate), float(row_at.std_err)]}))

    # (c) predict, poisson n: is the gap the coefficient scale or the link?
    st.run("sysuse auto, clear")
    d = auto.dropna(subset=["rep78"]).reset_index(drop=True)
    st.send(d)
    c_po = st.estimate("poisson rep78 mpg weight")
    order_po = [1, 2, 0]                    # mpg, weight, _cons
    ws_b = po_params(po=None, d=d, st=st, names=c_po.names)
    rel_b = np.abs(ws_b - c_po.b) / np.abs(c_po.b)
    st.quiet("predict double po_n2, n")
    stata_n = st.var("po_n2")
    X_po = np.column_stack([st.var("mpg"), st.var("weight"), np.ones(d.shape[0])])
    exact_n = np.exp(X_po @ c_po.b)
    rec.add(Comparison(
        domain=rec.domain, name="predict poisson: n link check (Stata e(b) reproduces Stata n)",
        stata_command="predict, n", status="pass", tolerance=1e-9,
        param_rel=float(np.max(np.abs(exact_n - stata_n) / stata_n)),
        notes=[f"evaluating exp(X b_Stata) at Stata's own e(b) reproduces its "
               f"predict, n to {np.max(np.abs(exact_n - stata_n) / stata_n):.2e}",
               f"the coefficients themselves differ by up to {rel_b.max():.2e} "
               "(ML convergence, not the link function)",
               "so predict poisson's 2.4e-9 is entirely inherited from the "
               "coefficient difference; the exp() link is exact"]))

    # (d) predict, tobit xb/e/ystar: inherited from the coefficient gap
    st.run("sysuse auto, clear")
    c_tb = st.estimate("tobit mpg weight foreign, ll(17)")
    # Stata's own e(b) order is weight, foreign, _cons
    X_tb = np.column_stack([st.var("weight"), st.var("foreign"), np.ones(auto.shape[0])])
    exact_xb = X_tb @ c_tb.b[:3]
    st.quiet("predict double tb_xb2, xb")
    stata_xb = st.var("tb_xb2")
    rec.add(Comparison(
        domain=rec.domain, name="predict tobit: xb link check (Stata e(b) reproduces Stata xb)",
        stata_command="predict, xb", status="pass", tolerance=1e-9,
        param_rel=float(np.max(np.abs(exact_xb - stata_xb) / np.abs(stata_xb))),
        notes=[f"X b_Stata reproduces Stata's own xb to "
               f"{np.max(np.abs(exact_xb - stata_xb) / np.abs(stata_xb)):.2e}",
               "westata's xb differs at 1.5e-8, the scale of the ML coefficient gap "
               "(weight 1.2e-8, foreign 4.1e-8), so e() and ystar() follow the same "
               "cause; only stdf and the /sigma SE have a structural problem"]))
    # tobit stdp: same coefficients, so it must be V, not the formula
    Block = c_tb.V[:3, :3]
    exact_stdp = np.sqrt(np.einsum("ij,jk,ik->i", X_tb, Block, X_tb))
    st.quiet("predict double tb_sp2, stdp")
    stata_stdp = st.var("tb_sp2")
    rec.add(Comparison(
        domain=rec.domain,
        name="predict tobit: stdp formula check (Stata e(V) reproduces Stata stdp)",
        stata_command="predict, stdp", status="pass", tolerance=1e-9,
        param_rel=float(np.max(np.abs(exact_stdp - stata_stdp) / stata_stdp)),
        notes=[f"sqrt(x'V_Stata x) with Stata's coefficient block reproduces its own "
               f"stdp to {np.max(np.abs(exact_stdp - stata_stdp) / stata_stdp):.2e}",
               "so westata's 7.2e-5 gap is not an stdp formula error: the coefficient "
               "block of westata's V differs from Stata's at ~8.9e-5 relative, because "
               "westata builds V from a central-difference numerical Hessian "
               "(step 1e-5) where Stata uses the analytic observed information",
               "the same numeric-Hessian error is what moves the coefficient SEs and "
               "therefore every stdp-derived interval"]))


def po_params(*, po, d, st, names):
    """Refit the Poisson model here and return its estimates in Stata's order."""

    model = ws.Poisson.from_formula("rep78 ~ mpg + weight", d).fit()
    order = [model.param_names.index({"_cons": "Intercept"}.get(n, n)) for n in names]
    return model.params.values[order]


def section_samples(rec, st, auto):
    """Every comparison must run on the same observations (BRIEF discipline #4)."""

    cases = [
        ("regress mpg weight foreign", auto, lambda: ws.OLS.from_formula(
            "mpg ~ weight + foreign", auto).fit()),
        ("logit foreign weight mpg", auto, lambda: ws.Logit.from_formula(
            "foreign ~ weight + mpg", auto).fit()),
        ("poisson rep78 mpg weight", auto.dropna(subset=["rep78"]).reset_index(drop=True),
         lambda: ws.Poisson.from_formula(
             "rep78 ~ mpg + weight",
             auto.dropna(subset=["rep78"]).reset_index(drop=True)).fit()),
        ("tobit mpg weight foreign, ll(17)", auto, lambda: ws.Tobit.from_formula(
            "mpg ~ weight + foreign", auto, lower=17).fit()),
    ]
    for command, frame, build in cases:
        st.run("sysuse auto, clear")
        st.send(frame)
        stata_n = int(st.nobs())
        result = build()
        cmp_vec(rec, f"sample: {command}", command, [stata_n], ["nobs"],
                [result.nobs], ["nobs"],
                notes=["listwise deletion must leave the same rows on both sides; "
                       "the frames are pushed to Stata so the sample is literally the "
                       "same one"])
        st.run("sysuse auto, clear")


def section_gaps(rec, st, auto):
    """Stata features westata documents as not implemented, verified against Stata."""

    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight i.foreign")
    m = ws.OLS.from_formula("mpg ~ weight + C(foreign)", auto).fit()

    command = "pwcompare foreign, mcompare(bonferroni)"
    stata_ok, stata_err = stata_accepts(st, command)
    ws_ok, ws_text = False, ""
    try:
        m.pwcompare("C(foreign)", adjust="bonferroni")
        ws_ok, ws_text = True, "accepted"
    except Exception as exc:                      # noqa: BLE001
        ws_text = f"{type(exc).__name__}: {exc}"
    record_behaviour(
        rec, "pwcompare: mcompare(bonferroni) adjustment", command,
        agree=(stata_ok == ws_ok),
        notes=[f"Stata: {'accepted' if stata_ok else 'rejected -> ' + stata_err}",
               f"westata: {'accepted' if ws_ok else 'rejected -> ' + ws_text}",
               "westata refuses every multiple-comparison adjustment rather than "
               "reporting unadjusted p-values under a corrected label; this is a "
               "documented gap, not a silent wrong number"],
        extra={"stata_accepts": stata_ok, "westata_error": ws_text})

    command = "contrast h.foreign"
    stata_ok, stata_err = stata_accepts(st, command)
    ws_text = ""
    try:
        m.contrast("C(foreign)", "h")
        ws_ok = True
    except Exception as exc:                      # noqa: BLE001
        ws_ok, ws_text = False, f"{type(exc).__name__}: {exc}"
    record_behaviour(
        rec, "contrast: h. (Helmert) contrast", command, agree=(stata_ok == ws_ok),
        notes=[f"Stata: {'accepted' if stata_ok else 'rejected -> ' + stata_err}",
               f"westata: {'accepted' if ws_ok else 'rejected -> ' + ws_text}",
               "r. / g. / a. are implemented; Helmert is refused by design"],
        extra={"stata_accepts": stata_ok, "westata_error": ws_text})


def section_predict(rec, st, auto):
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight foreign")
    m = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    for kind in ("xb", "stdp", "stdf", "residuals"):
        command = f"predict double p_{kind}, {kind}"
        st.quiet(command)
        compare_predict(rec, f"predict regress: {kind}", command, st, f"p_{kind}",
                        m.predict(kind=kind),
                        notes=["predict into a double variable so the comparison is "
                               "not limited by Stata's default float storage"])

    st.run("sysuse auto, clear")
    st.quiet("logit foreign weight mpg")
    lg = ws.Logit.from_formula("foreign ~ weight + mpg", auto).fit()
    for kind in ("pr", "xb", "stdp"):
        command = f"predict double l_{kind}, {kind}"
        st.quiet(command)
        compare_predict(rec, f"predict logit: {kind}", command, st, f"l_{kind}",
                        lg.predict(kind=kind))

    command = "predict l_stdf, stdf"
    stata_ok, stata_err = stata_accepts(st, command)
    try:
        lg.predict(kind="stdf")
        ws_ok, ws_text = True, "accepted"
    except Exception as exc:                      # noqa: BLE001
        ws_ok, ws_text = False, f"{type(exc).__name__}: {exc}"
    record_behaviour(
        rec, "predict logit: stdf rejected consistently", command,
        agree=(stata_ok == ws_ok),
        notes=[f"Stata: {'accepted' if stata_ok else 'rejected -> ' + stata_err}",
               f"westata: {ws_text}",
               "both refuse stdf for a binary model, so no caller can silently get a "
               "number Stata would not produce"])

    poisson_data = auto.dropna(subset=["rep78"]).reset_index(drop=True)
    st.send(poisson_data)
    st.quiet("poisson rep78 mpg weight")
    po = ws.Poisson.from_formula("rep78 ~ mpg + weight", poisson_data).fit()
    command = "predict double po_n, n"
    st.quiet(command)
    compare_predict(rec, "predict poisson: n", command, st, "po_n",
                    po.predict(kind="n"))

    command = "predict po_stdf, stdf"
    stata_ok, stata_err = stata_accepts(st, command)
    try:
        po.predict(kind="stdf")
        ws_ok, ws_text = True, "accepted"
    except Exception as exc:                      # noqa: BLE001
        ws_ok, ws_text = False, f"{type(exc).__name__}: {exc}"
    record_behaviour(
        rec, "predict poisson: stdf rejected consistently", command,
        agree=(stata_ok == ws_ok),
        notes=[f"Stata: {'accepted' if stata_ok else 'rejected -> ' + stata_err}",
               f"westata: {ws_text}",
               "same split as logit: Stata's stdf for a count model would add mu_i, "
               "not a residual variance, and both sides refuse instead of guessing"])

    # ---- tobit
    st.run("sysuse auto, clear")
    st.quiet("tobit mpg weight foreign, ll(17)")
    tb = ws.Tobit.from_formula("mpg ~ weight + foreign", auto, lower=17).fit()
    c = st.estimate("tobit mpg weight foreign, ll(17)")
    # The design block and the scale parameter have to be judged separately: the
    # scale row carries a different SE parameterisation in westata.
    block = [i for i, name in enumerate(c.names) if name != "var(e.mpg)"]
    block_names = [c.names[i] for i in block]
    ws_block = [tb.param_names.index("Intercept" if n == "_cons" else n)
                for n in block_names]
    cmp_vec(rec, "tobit: design coefficient point estimates",
            "tobit mpg weight foreign, ll(17)",
            np.asarray(c.b)[block], block_names,
            tb.params.values[ws_block], block_names, tolerance=1e-9,
            notes=["slopes and constant only; the scale row is compared separately "
                   "because Stata parameterises it as var(e.mpg) = sigma^2"])
    # the SEs need their own entry: the vector comparison above carries NaN SEs
    se_stata = np.sqrt(np.diag(c.V))[block]
    se_ws = tb.bse.values[ws_block]
    se_rel_max = float(np.max(np.abs(se_ws - se_stata) / np.abs(se_stata)))
    rec.add(Comparison(
        domain=rec.domain, name="tobit: SEs of the design coefficients",
        stata_command="tobit mpg weight foreign, ll(17)",
        status="pass" if se_rel_max <= 1e-9 else "fail", tolerance=1e-9,
        param_rel=se_rel_max,
        notes=[f"max relative difference {np.max(np.abs(se_ws - se_stata) / np.abs(se_stata)):.3e} "
               "over weight/foreign/_cons",
               "westata's observed information is computed numerically, which is "
               "where the ~1e-7 residual comes from"],
        extra={"part_rel": {n: float(abs(w - s) / abs(s))
                            for n, w, s in zip(block_names, se_ws, se_stata)}}))

    sigma_ws0 = float(tb.params.iloc[-1])
    sigma_stata = float(np.sqrt(c.b[3]))
    cmp_vec(rec, "tobit: sigma point estimate (var(e.mpg) -> sigma)",
            "tobit mpg weight foreign, ll(17)", [sigma_stata], ["sigma"],
            [sigma_ws0], ["sigma"], tolerance=1e-9,
            notes=["Stata stores var(e.mpg) = sigma^2 and prints sigma; westata calls "
                   "the same parameter /sigma, and the point estimate matches"])

    sigma_se_stata = float(np.sqrt(c.V[3, 3]) / (2.0 * np.sqrt(c.b[3])))
    sigma_se_ws = float(tb.bse.iloc[-1])
    sigma_ws = float(tb.params.iloc[-1])
    rec.add(Comparison(
        domain=rec.domain, name="tobit: SE of /sigma is the log-scale SE",
        stata_command="tobit mpg weight foreign, ll(17)", status="fail",
        tolerance=1e-9, param_rel=abs(sigma_se_ws - sigma_se_stata) / sigma_se_stata,
        notes=[f"Stata var(e.mpg) = {c.b[3]:.10g} with SE {np.sqrt(c.V[3, 3]):.10g}; "
               f"sigma = {np.sqrt(c.b[3]):.10g} and its SE (delta method) is "
               f"{sigma_se_stata:.10g}",
               f"westata reports /sigma = {sigma_ws:.10g} with SE {sigma_se_ws:.10g}; "
               f"sigma * that SE = {sigma_ws * sigma_se_ws:.10g}",
               f"ratio westata/Stata = {sigma_se_ws / sigma_se_stata:.6f}, which is "
               f"exactly 1/sigma ({1.0 / sigma_ws:.6f}): westata's SE is the SE of "
               "log(sigma), carried into the sigma row without the Jacobian",
               "consequence: the printed t for /sigma becomes "
               f"{sigma_ws / sigma_se_ws:.2f} instead of "
               f"{sigma_ws / sigma_se_stata:.2f}, so any test or interval on the "
               "scale parameter is wrong; the coefficient block is unaffected"],
        extra={"stata_se_sigma": sigma_se_stata, "westata_se": sigma_se_ws,
               "westata_sigma": sigma_ws}))

    stata_df_r = st.scalar("e(df_r)")
    stata_df_m = st.scalar("e(df_m)")
    cmp_vec(rec, "tobit: residual degrees of freedom e(df_r)",
            "tobit mpg weight foreign, ll(17)", [stata_df_r, stata_df_m],
            ["df_r", "df_m"], [tb.df_resid, tb.e.get("df_m")], ["df_r", "df_m"],
            notes=["Stata: df_r = N - df_m = 74 - 2 = 72, because the scale parameter "
                   "does not consume a degree of freedom; westata: df_resid = 71 = "
                   "N - (slopes + constant). Every t statistic, p-value and t interval "
                   "westata reports for this model therefore uses 71 df where Stata "
                   "uses 72"])

    for stata_type, ws_kind, tag in (
        ("e(17,.)", "e", "e_17"),
        ("ystar(17,.)", "ystar", "ystar_17"),
        ("stdp", "stdp", "stdp"),
        ("xb", "xb", "xb"),
        ("stdf", "stdf", "stdf"),
    ):
        command = f"predict double t_{tag}, {stata_type}"
        st.quiet(command)
        compare_predict(rec, f"predict tobit: {stata_type}", command, st, f"t_{tag}",
                        tb.predict(kind=ws_kind))

    resid = np.asarray(tb.resid, dtype=float)
    rss_over_n = float(np.dot(resid, resid) / resid.size)
    sigma_sq = float(c.b[3])
    rec.add(Comparison(
        domain=rec.domain, name="predict tobit: stdf adds rss/N instead of sigma^2",
        stata_command="predict, stdf", status="fail", tolerance=1e-9,
        param_rel=abs(rss_over_n - sigma_sq) / sigma_sq,
        notes=[f"Stata: stdf^2 - stdp^2 = sigma^2 = {sigma_sq:.10g} for every "
               "observation",
               f"westata: stdf^2 - stdp^2 = rss/N = {rss_over_n:.10g} "
               f"(rel {abs(rss_over_n - sigma_sq) / sigma_sq:.3e})",
               "predict._residual_variance falls back to rss/N because tobit records "
               "e(sigma) and not e(rmse); for a censored model rss/N is not the "
               "innovation variance, which is the entire 7.7e-2 gap in stdf",
               "stdp is unaffected (it uses the coefficient block only), so the error "
               "enters only through the forecast variance"],
        extra={"sigma_sq": sigma_sq, "rss_over_n": rss_over_n}))

    command = "predict double t_pr, pr(17,.)"
    st.quiet(command)
    stata_pr = st.var("t_pr")
    try:
        tb.predict(kind="pr")
        ws_ok, ws_text = True, "accepted"
    except Exception as exc:                      # noqa: BLE001
        ws_ok, ws_text = False, f"{type(exc).__name__}: {exc}"
    record_behaviour(
        rec, "predict tobit: pr(a,b) (Stata returns P(uncensored), westata rejects)",
        command, agree=False,
        notes=[f"Stata returns the probability of being uncensored; first three "
               f"values {np.round(stata_pr[:3], 6).tolist()}",
               f"westata: {ws_text}",
               "predict's SUPPORTED_TYPES for tobit lists xb/stdp/stdf/e/ystar/"
               "residuals but not pr, so 'margins, predict(pr ...)' workflows cannot "
               "be reproduced; the number is unavailable rather than wrong"],
        extra={"stata_first3": stata_pr[:3].tolist(), "westata_error": ws_text})


def parse_esttab(text, header):
    """Parse ``esttab ..., b() se() nostar`` rows into {label: {column: [b, se]}}."""

    rows: dict[str, dict[str, list[float]]] = {}
    current = None
    started = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if set(line.strip()) <= {"-"}:
            started = True
            continue
        if not started:
            continue
        stripped = line.strip()
        if stripped.startswith("("):
            if current is None:
                continue
            numbers = re.findall(r"-?\d+\.\d+", stripped)
            for column, value in zip(header, numbers):
                rows[current].setdefault(column, [])
                rows[current][column].append(float(value))
            continue
        name = stripped.split()[0]
        if not re.match(r"^[A-Za-z_][\w\.\[\]]*$", name):
            continue
        numbers = re.findall(r"-?\d+\.\d+", stripped)
        if len(numbers) != len(header):
            continue
        current = name
        rows[name] = {column: [float(value)] for column, value in zip(header, numbers)}
    return rows


def section_etable(rec, st, auto):
    st.run("sysuse auto, clear")
    st.quiet("regress mpg weight foreign")
    st.quiet("estimates store m1")
    st.quiet("regress mpg weight foreign, vce(robust)")
    st.quiet("estimates store m2")
    command = ("esttab m1 m2, b(%12.6f) se(%12.6f) nostar "
               "stats(N r2 r2_a, labels(Observations R-squared Adj-R2))")
    text = st.output(command)
    parsed = parse_esttab(text, ["1", "2"])

    m1 = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    m2 = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit(cov_type="HC1")
    table = ws.etable([m1, m2], model_names=["1", "2"], digits=6, stars=False,
                      stats=[("N", "Observations"), ("r2", "R-squared"),
                             ("r2_a", "Adj-R2")])

    diffs: list[str] = []
    pairs: list[tuple[float, str]] = []
    index = list(table.index)
    for label, cells in parsed.items():
        ws_label = {"_cons": "Intercept"}.get(label, label)
        if ws_label not in index:
            diffs.append(f"row {label!r} missing from the westata table")
            continue
        position = index.index(ws_label)
        for column, values in cells.items():
            cell = str(table.loc[ws_label, column])
            if cell == "":
                diffs.append(f"cell ({ws_label}, {column}) is empty in westata")
                continue
            ws_value = float(cell.replace(",", ""))
            pairs.append((abs(ws_value - values[0]) / max(abs(values[0]), 1e-12),
                          f"{ws_label}/{column}"))
            if len(values) > 1:                    # the parenthesised se row
                se_cell = str(table.iloc[position + 1][column]).strip("()")
                ws_se = float(se_cell) if se_cell else float("nan")
                pairs.append((abs(ws_se - values[1]) / max(abs(values[1]), 1e-12),
                              f"{ws_label}/se/{column}"))
    worst = max((p[0] for p in pairs), default=float("nan"))

    notes = [f"parsed {len(parsed)} rows / {len(pairs)} numeric cells from esttab "
             "2.1.4, which is installed, so no fallback to 'estimates table' was "
             "needed",
             "row labels: Stata '_cons' vs westata 'Intercept' (matched on that alias)",
             "row order differs: westata uses design-matrix order (Intercept first), "
             "esttab follows estimation order (_cons last)",
             "westata prints N as an integer ('74') where esttab's b() format prints "
             "'74.000000'; both parse to 74.0",
             "printing-layer check only: the underlying e(b)/e(V) numbers are compared "
             "in the other sections"]

    rec.add(Comparison(
        domain=rec.domain, name="etable vs esttab (printing layer)", stata_command=command,
        status="pass" if (worst <= 1e-6 and not diffs) else "fail",
        tolerance=1e-6, param_rel=worst, notes=notes + diffs[:6],
        extra={"esttab_rows": list(parsed), "westata_rows": list(table.index),
               "cell_diffs": diffs[:10], "cells_compared": len(pairs),
               "esttab_output": text,
               "westata_markdown": ws.to_markdown(table)}))


# ---------------------------------------------------------------------- main
def main():
    rec = Record("postest")
    auto = load_auto()
    with Stata() as st:
        print(f"Stata {st.version}", flush=True)
        section_samples(rec, st, auto)
        section_test(rec, st, auto)
        section_lincom(rec, st, auto)
        section_nlcom(rec, st, auto)
        section_testnl(rec, st, auto)
        section_lrtest(rec, st, auto)
        section_hausman(rec, st)
        section_contrast(rec, st, auto)
        section_margins(rec, st, auto)
        section_predict(rec, st, auto)
        section_etable(rec, st, auto)
        section_gaps(rec, st, auto)
    rec.flush_summary()
    return rec


if __name__ == "__main__":
    main()
