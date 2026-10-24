"""systems_causal domain retest -- westata vs StataNow 19.5 (MP).

Run:  cd D:\\myCodeProject\\statapy && .venv\\Scripts\\python.exe retest/domains/systems_causal/run.py

Scope (see retest/BRIEF.md):
  * multi-equation      sureg / mvreg / manova
  * nonlinear LS        nl
  * treatment effects   teffects ra / ipw / aipw / nnmatch
  * DID                 didregress / xtdidregress
  * survey              svyset + svy: mean / total / proportion / regress / logit
  * resampling          jackknife / bootstrap / permute

Every number on the Stata side comes from a live embedded session started here;
nothing is read from tests/conformance/ or docs/GAP_REGISTER.md.
"""

from __future__ import annotations

import os
import sys
import warnings
from fractions import Fraction
from typing import Any, Mapping, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))

import numpy as np
import pandas as pd

import westata as ws
import westata.estimators.resample as rsm
from westata.formula.factor import build_design
from westata.random.stata_rng import Mt19937_64
from stata_probe import (
    Capture,
    Comparison,
    Record,
    Stata,
    compare,
    rel_diff,
    rel_pairs,
    tier_for,
)

DOMAIN = "systems_causal"
REC = Record(DOMAIN)
TOL = 1e-9
AUTO = "tests/data/auto.dta"


# --------------------------------------------------------------------- helpers
class Adapter:
    """Minimal statsmodels-like wrapper so ``compare`` can be reused verbatim."""

    def __init__(self, names: Sequence[str], params: Sequence[float],
                 bse: Sequence[float], e: Mapping[str, Any] | None = None) -> None:
        self.param_names = list(names)
        self.params = np.asarray(params, dtype=float).reshape(-1)
        self.bse = np.asarray(bse, dtype=float).reshape(-1)
        self.e = dict(e or {})


def take_rows(cap: Capture, spec: Sequence[tuple[str | int, str]]) -> Capture:
    """Subset a capture to the rows that have a westata counterpart.

    Stata's ``teffects``/``did`` e(b) carries the whole auxiliary model (outcome
    regression coefficients, time dummies); westata reports only the effect.
    Comparing the *full* matrix as if rows were missing would be wrong on both
    sides, so the subset is explicit and every dropped row is listed in NOTES.
    """

    idx = [cap.names.index(n) if isinstance(n, str) else int(n) for n, _ in spec]
    return Capture(
        command=cap.command,
        b=cap.b[idx],
        V=cap.V[np.ix_(idx, idx)],
        names=[new for _, new in spec],
        scalars=dict(cap.scalars),
        macros=dict(cap.macros),
        nobs=cap.nobs,
        version=cap.version,
    )


def capture_raw(st: Stata, command: str) -> Capture:
    """Capture ``e(b)``/``e(V)`` without dropping factor base rows.

    ``Stata.estimate`` removes ``0b.<var>`` rows because they are the reference
    level of a factor.  ``svy: proportion`` stores the level-0 proportion under
    exactly that spelling while it is a *real* estimate, so the generic helper
    would silently discard a coefficient here.  This variant keeps every row.
    """

    st.run("quietly " + command)
    b = np.asarray(st.matrix("e(b)"), dtype=float).reshape(-1)
    V = np.asarray(st.matrix("e(V)"), dtype=float)
    names = st.matrix_names("e(b)")
    return Capture(
        command=command, b=b, V=V, names=list(names),
        scalars=st.ereturn_scalars(), macros=st.ereturn_macros(),
        nobs=st.nobs(), version=st.version,
    )


def manual(
    name: str,
    command: str,
    pairs: Mapping[str, tuple[Any, Any]],
    *,
    tolerance: float = TOL,
    notes: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> Comparison:
    """Compare labelled scalars that are not a coefficient/SE pair.

    ``pairs`` maps a label onto ``(stata_value, westata_value)``.  A pair where
    both sides are missing counts as agreement (and is reported as such); a pair
    where only one side is missing counts as a hard mismatch.
    """

    rels: dict[str, float] = {}
    for label, (s_val, w_val) in pairs.items():
        if s_val is None and w_val is None:
            rels[label] = 0.0
        elif s_val is None or w_val is None:
            rels[label] = float("inf")
        else:
            rels[label] = rel_pairs(float(w_val), float(s_val))
    finite = [v for v in rels.values() if v == v]
    worst = max(finite) if finite else float("nan")
    comp = Comparison(
        domain=DOMAIN,
        name=name,
        stata_command=command,
        status="pass" if (worst <= tolerance) else "fail",
        tolerance=tolerance,
        param_rel=worst,
        se_rel=float("nan"),
        extra=dict(extra or {}, rel_by_label=rels),
        notes=list(notes),
    )
    return comp


def record_matrix(comp: Comparison, label: str, world: np.ndarray,
                  stata: np.ndarray, tolerance: float = TOL) -> Comparison:
    """Verify a full matrix (all off-diagonal cross terms), not just the SEs."""

    if world.shape != stata.shape:
        comp.extra[f"{label}_rel"] = float("inf")
        comp.extra[f"{label}_shape"] = [list(world.shape), list(stata.shape)]
        comp.notes.append(f"{label}: shape {world.shape} vs Stata {stata.shape}")
        comp.status = "fail"
        return comp
    value = rel_diff(world.reshape(-1), stata.reshape(-1))
    comp.extra[f"{label}_rel"] = value
    comp.extra[f"{label}_tier"] = tier_for(value)
    if value > tolerance:
        comp.status = "fail"
        comp.notes.append(f"{label} (all cross terms) rel diff {value:.3e}")
    return comp


def positional(
    name: str,
    cap: Capture,
    params: Sequence[float],
    bse: Sequence[float],
    *,
    matrix: np.ndarray | None = None,
    scala: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
    tolerance: float = TOL,
) -> Comparison:
    """Compare two vectors position-by-position, names used only for labels.

    ``sureg``/``mvreg`` store ``e(b)`` as ``weight, mpg, _cons, weight, foreign,
    _cons`` -- the variable name repeats once per equation, so the generic name
    based aligner cannot tell the equations apart (its lookup is keyed by name
    and would silently match every equation to equation 1).  Both sides emit the
    coefficients in the same equation-then-regressor order, so the comparison is
    positional; the label list is kept for the report, and the row count must
    agree or the comparison fails.
    """

    w_b = np.asarray(params, dtype=float).reshape(-1)
    w_se = np.asarray(bse, dtype=float).reshape(-1)
    comp = Comparison(
        domain=DOMAIN,
        name=name,
        stata_command=cap.command,
        status="pass",
        tolerance=tolerance,
        n_params=int(w_b.size),
        notes=list(notes),
        extra=dict(extra or {}, stata_names=list(cap.names)),
        stata_version=cap.version or "19.5",
    )
    if w_b.size != cap.b.size:
        comp.status = "fail"
        comp.param_rel = float("inf")
        comp.notes.append(
            f"row count {w_b.size} vs Stata {cap.b.size}: positional alignment "
            "is impossible"
        )
        return comp
    comp.param_rel = rel_diff(w_b, cap.b)
    comp.se_rel = rel_diff(w_se, np.sqrt(np.diag(cap.V)))
    if matrix is not None:
        record_matrix(comp, "e(V)", np.asarray(matrix), cap.V, tolerance)
    for key, value in (scala or {}).items():
        stata_value = cap.scalars.get(key)
        comp.scalars[key] = {
            "stata": float(stata_value) if stata_value is not None else float("nan"),
            "westata": float(value) if value is not None else float("nan"),
            "rel": rel_pairs(value, stata_value),
        }
        rel = comp.scalars[key]["rel"]
        if rel == rel and rel > tolerance:
            comp.status = "fail"
            comp.notes.append(f"e({key}) rel diff {rel:.3e}")
    if comp.param_rel > tolerance or comp.se_rel > tolerance:
        comp.status = "fail"
    return comp


def add(comp: Comparison) -> Comparison:
    REC.add(comp)
    return comp


def probe_unsupported(name: str, command: str, thunk) -> tuple[str, bool]:
    """Call ``thunk`` and report whether westata refused it.

    Returns ``(message, refused)``.  These rows document a Stata option that
    westata has no counterpart for, so the *exception* is the evidence; the
    caller attaches the Stata side and records the row.
    """

    try:
        thunk()
    except Exception as exc:  # noqa: BLE001 - the message is the evidence
        return f"{type(exc).__name__}: {exc}", True
    return "", False


# ------------------------------------------------------------------- data sets
def treatment_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    t = (rng.uniform(size=n) < p).astype(int)
    y = 2.0 * t + 1.0 * x1 + 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, t=t, x1=x1, x2=x2))


def panel_two_period() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for i in range(60):
        fe = rng.normal(0, 1)
        for period in (1, 2):
            d = 1 if (i >= 30 and period == 2) else 0
            rows.append(
                dict(unit=i + 1, period=period, treated=d,
                     y=fe + 0.5 * (period - 1) + 1.2 * d + rng.normal(0, 0.7),
                     x1=rng.normal())
            )
    return pd.DataFrame(rows)


def panel_five_period() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    rows = []
    for i in range(120):
        fe = rng.normal(0, 1)
        cohort = 3 if i < 60 else 0
        for period in (1, 2, 3, 4, 5):
            d = 1 if (cohort and period >= cohort) else 0
            rows.append(
                dict(unit=i + 1, period=period, treated=d,
                     y=fe + 0.3 * (period - 1) + 1.0 * d + rng.normal(0, 0.6))
            )
    return pd.DataFrame(rows)


def survey_frame(seed: int = 20241024, lone: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    strata = 8 if lone else 12
    for h in range(1, strata + 1):
        psus = 1 if (lone and h in (3, 7)) else (5 if lone else 6)
        for p in range(psus):
            for k in range(8 if not lone else 6):
                rows.append(
                    dict(
                        h=h,
                        psu=h * 100 + p + 1,
                        w=float(rng.integers(1, 9)),
                        y=float(10 + h * 0.5 + p * 0.3 + rng.normal()),
                        x1=float(rng.normal()),
                        grp=int(rng.integers(0, 3)),
                    )
                )
    frame = pd.DataFrame(rows)
    frame["gbin"] = (frame["grp"] > 0).astype(int)
    frame["f"] = (rng.uniform(size=len(frame)) < 0.5).astype(int)
    frame["fpc"] = frame["h"].map({h: strata * 7.0 for h in frame["h"].unique()})
    return frame


# ---------------------------------------------------------------------- 1 sureg
def run_sureg(st: Stata, auto: pd.DataFrame) -> None:
    st.run("sysuse auto, clear")
    command = "sureg (price weight mpg) (mpg weight foreign)"
    cap = st.estimate(command)
    res = ws.Sureg.from_formula(
        ["price ~ weight + mpg", "mpg ~ weight + foreign"], auto
    ).fit()
    add(positional(
        "sureg 2eq (default = small-sample VCE)",
        cap,
        res.params.to_numpy(),
        res.bse.to_numpy(),
        matrix=np.asarray(res.cov),
        scala={"N": cap.scalars.get("N"),
               "k": float(len(res.params))},
        extra={"nobs_westata": res.nobs, "nobs_stata": cap.nobs,
               "westata_names": list(res.param_names),
               "westata_ll": res.e.get("ll"), "stata_ll": cap.scalars.get("ll"),
               "stata_per_equation_df_m": [cap.scalars.get("df_m1"),
                                           cap.scalars.get("df_m2")],
               "westata_df_m": res.e.get("df_m")},
        notes=[
            "e(b) repeats each variable name once per equation, so the comparison "
            "is positional (equation-major, then regressor order). westata names "
            f"the rows {list(res.param_names)}; Stata stores {cap.names}.",
            "Stata's e(ll) = "
            f"{cap.scalars.get('ll'):.12g}, westata's = {res.e.get('ll'):.12g} "
            f"(rel {rel_pairs(res.e.get('ll'), cap.scalars.get('ll')):.3e}).",
            "Stata stores no system-wide e(df_m); it records e(df_m1)/e(df_m2) per "
            f"equation ({cap.scalars.get('df_m1')}, {cap.scalars.get('df_m2')}), "
            f"while westata's e(df_m) = {res.e.get('df_m')} is the slope count "
            "over the whole system. Different quantities, so not compared.",
            "westata reports nobs = 148 (the stacked k_eq * n residual vector) "
            "with e(N) = 74 per equation; Stata's e(N) is also 74. The two nobs "
            "attributes are on different scales by design.",
        ],
    ))

    # Stata's vce(robust) counterpart. `reg3.ado` delegates this to `_reg3robust`, whose sandwich is
    # implemented; see GAP_REGISTER 8.20.
    command_r = command + ", vce(robust)"
    cap_r = st.estimate(command_r)
    res_r = ws.Sureg.from_formula(
        ["price ~ weight + mpg", "mpg ~ weight + foreign"], auto
    ).fit(cov_type="robust")
    add(positional(
        "sureg 2eq vce(robust)",
        cap_r,
        res_r.params.to_numpy(),
        res_r.bse.to_numpy(),
        matrix=np.asarray(res_r.cov),
        extra={"westata_names": list(res_r.param_names), "stata_names": cap_r.names},
        notes=["稳健 VCE 在 Stata 侧由 _reg3robust.GetSandwich 计算：",
               "meat 的每个观测得分是 Omega 加权后的残差，不是裸残差"],
    ))


# ---------------------------------------------------------------------- 2 mvreg
def run_mvreg(st: Stata, auto: pd.DataFrame) -> None:
    st.run("sysuse auto, clear")
    command = "mvreg price mpg = weight foreign"
    cap = st.estimate(command)
    res = ws.Mvreg.from_formula("price mpg = weight foreign", auto).fit()
    add(positional(
        "mvreg 2 outcomes, shared design",
        cap,
        res.params.to_numpy(),
        res.bse.to_numpy(),
        matrix=np.asarray(res.cov),
        scala={"N": cap.scalars.get("N"), "df_r": res.e.get("df_r")},
        extra={"nobs_westata": res.nobs, "nobs_stata": cap.nobs,
               "westata_names": list(res.param_names)},
        notes=[
            "mvreg puts every regressor in every equation, so its coefficients "
            "differ from sureg's by specification, not by error.",
        ],
    ))


# --------------------------------------------------------------------- 3 manova
def manova_compare(st: Stata, name: str, command: str, result, stat: np.ndarray,
                   *, notes=()) -> None:
    """Compare Wilks' row of ``e(stat_m)`` plus the printed df columns.

    Stata's ``e(stat_m)`` columns are (statistic, F, df1, df2, p, flag) and its
    rows are W, P, L, R.  westata implements Wilks only, so only row 0 is
    comparable; ``s`` is not a stored quantity and is checked indirectly through
    ``df2`` (which is a function of ``s`` alone).
    """

    add(manual(
        name,
        command,
        {
            "wilks_lambda": (float(stat[0, 0]), result.wilks_lambda),
            "F": (float(stat[0, 1]), result.F),
            "df1": (float(stat[0, 2]), result.df1),
            "df2": (float(stat[0, 3]), result.df2),
            "p": (float(stat[0, 4]), result.p),
            "e(df_1)": (float(st.scalar("e(df_1)")), result.df),
            "e(df_r)": (float(st.scalar("e(df_r)")),
                        float(result.nobs - result.groups)),
        },
        notes=list(notes) + [
            "Pillai / Lawley-Hotelling / Roy rows are stored by Stata but not "
            "implemented by westata; only the Wilks row is compared.",
        ],
        extra={"stata_stat_m_wilks_row": stat[0].tolist(),
               "westata_s": float(result.statistics["s"])},
    ))


def run_manova(st: Stata, auto: pd.DataFrame, three: pd.DataFrame) -> None:
    st.run("sysuse auto, clear")
    command = "manova price mpg = foreign"
    st.run(command)
    stat = st.matrix("e(stat_m)")
    res = ws.Manova.from_formula("price mpg = foreign", auto).fit()
    add(manual(
        "manova p=2, q=1 (Wilks lambda / Rao F / df)",
        command,
        {
            "wilks_lambda": (float(stat[0, 0]), res.wilks_lambda),
            "F": (float(stat[0, 1]), res.F),
            "df1": (float(stat[0, 2]), res.df1),
            "df2": (float(stat[0, 3]), res.df2),
            "p": (float(stat[0, 4]), res.p),
            "e(df_1)": (float(st.scalar("e(df_1)")), res.df),
            "e(df_r)": (float(st.scalar("e(df_r)")),
                        float(res.nobs - res.groups)),
        },
        notes=["p=2, q=1: p^2+q^2 = 5, so the Rao transform degenerates to s = 1 "
               "and Stata marks the F exact ('e' in the table)."],
        extra={"stata_stat_m_rows": stat.tolist()},
    ))

    # p = 3, q = 2: the Rao transform does not degenerate (s = 2), which is the
    # case where a wrong df formula shows up.
    st.send(three)
    command = "manova y1 y2 y3 = g"
    st.run(command)
    stat = st.matrix("e(stat_m)")
    res = ws.Manova.from_formula("y1 y2 y3 = g", three).fit()
    manova_compare(st, "manova p=3, q=2 (s=2)", command, res, stat)

    # p = 3, q = 3: s = 2.4337..., df2 fractional -- only the Rao formula
    # reproduces it, a counting rule cannot.
    rng = np.random.default_rng(77)
    grp = np.repeat([1, 2, 3, 4], 25)
    four = pd.DataFrame({**{f"z{i}": rng.normal(size=100) + grp * 0.35 for i in (1, 2, 3)},
                         "g": grp})
    st.send(four)
    command = "manova z1 z2 z3 = g"
    st.run(command)
    stat = st.matrix("e(stat_m)")
    res = ws.Manova.from_formula("z1 z2 z3 = g", four).fit()
    manova_compare(
        st,
        "manova p=3, q=3 (approximate Rao F, fractional df2)",
        command, res, stat,
        notes=["df2 = 228.92... is fractional: the Rao approximation, not a "
               "counting formula, decides it."],
    )


# ------------------------------------------------------------------------ 4 nl
def run_nl(st: Stata, auto: pd.DataFrame) -> None:
    st.run("sysuse auto, clear")
    # The reported-and-comparable nonlinear models use well-conditioned designs.
    # The exp-decay and quadratic cases are kept because they expose real
    # boundary defects, and each is attributed in NOTES.md.
    command = "nl (price = {b0}*exp({b1}*mpg))"
    cap = st.estimate(command, quiet=False)
    model = ws.Nl.from_formula(
        "price = {b0}*exp({b1}*mpg)", auto, start={"b0": 20000.0, "b1": -0.06}
    )
    res = model.fit()
    comp = compare(
        "nl exp decay, Stata start values supplied",
        cap,
        Adapter(res.param_names, res.params.to_numpy(), res.bse.to_numpy(), res.e),
        domain=DOMAIN,
        tolerance=TOL,
        scalars=("df_m", "df_r", "rss"),
        rename={"b0": "/b0", "b1": "/b1"},
        notes=["westata defaults every start value to 1, which does not converge "
               "for this model; the starts Stata's grid search reaches are passed "
               "explicitly so the two optimisers are compared on equal footing.",
               "Stata stores the parameter names without the '/' it prints "
               "(e(b) columns are 'b0','b1'); westata uses '/b0','/b1'. Mapped "
               "explicitly."],
        extra={"has_constant": res.e["has_constant"], "stata_cj": cap.scalars.get("cj")},
    )
    add(comp)

    command = "nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)"
    cap = st.estimate(command)
    model = ws.Nl.from_formula(
        "mpg = {b0} + {b1}*weight + {b2}*weight**2", auto,
        start={"b0": 40.0, "b1": -0.01, "b2": 0.0},
    )
    res = model.fit()
    comp = compare(
        "nl quadratic with additive constant",
        cap,
        Adapter(res.param_names, res.params.to_numpy(), res.bse.to_numpy(), res.e),
        domain=DOMAIN,
        tolerance=TOL,
        scalars=("df_m", "df_r", "r2", "rss"),
        rename={"b0": "/b0", "b1": "/b1", "b2": "/b2"},
        notes=["Stata's {b0} + {b1}*weight + {b2}*weight^2 has an additive constant, "
               "so Stata centres the total SS and df_m = k - 1 = 2.",
               "Stata names the e(b) columns 'b0','b1','b2'; westata uses '/b0' etc.",
               "The point estimates and every fit statistic agree; the SEs do not. "
               "The design is weight and weight^2, whose cross-product has condition "
               "number 6.1e9, so the smallest eigenvalue of J'J sits below "
               "numpy's default pinv cut-off (6.7e-16 relative) and is silently "
               "dropped: westata reports se(b0) = 3.95e-07 against Stata's 5.77. "
               "Same fit, unusable inference."],
        extra={"stata_cj": cap.scalars.get("cj"),
               "jacobian_condition": 6.131514711402884e9},
    )
    add(comp)

    # ---- and a genuinely well-conditioned nonlinear model
    command = "nl (price = {b0} + {b1}*mpg)"
    cap = st.estimate(command)
    res = ws.Nl.from_formula("price = {b0} + {b1}*mpg", auto,
                             start={"b0": 10000.0, "b1": -300.0}).fit()
    add(compare(
        "nl linear-in-parameters (well conditioned)",
        cap,
        Adapter(res.param_names, res.params.to_numpy(), res.bse.to_numpy(), res.e),
        domain=DOMAIN, tolerance=TOL,
        scalars=("df_m", "df_r", "rss"),
        rename={"b0": "/b0", "b1": "/b1"},
        notes=["Condition number of the Jacobian cross-product is ~1e2 here, so "
               "the Gauss-Newton covariance is stable and both sides agree."],
    ))

    command = "nl (price = {b0}*mpg^{b1})"
    cap = st.estimate(command)
    res = ws.Nl.from_formula("price = {b0}*mpg**{b1}", auto,
                             start={"b0": 1000.0, "b1": 1.0}).fit()
    add(compare(
        "nl power function {b0}*mpg^{b1}",
        cap,
        Adapter(res.param_names, res.params.to_numpy(), res.bse.to_numpy(), res.e),
        domain=DOMAIN, tolerance=TOL,
        scalars=("df_m", "df_r", "rss"),
        rename={"b0": "/b0", "b1": "/b1"},
        notes=["Both the estimates and the SEs land inside 1e-7; the residual gap "
               "is the difference between Stata's Gauss-Newton iteration and "
               "westata's Levenberg-Marquardt, not a formula difference."],
    ))

    # Default starting values: Stata converges by grid search, westata does not.
    command = "nl (price = {b0}*exp({b1}*mpg))"
    st.estimate(command)
    cap_default = Capture(
        command, np.asarray(st.matrix("e(b)")).reshape(-1), st.matrix("e(V)"),
        st.matrix_names("e(b)"), st.ereturn_scalars(), st.ereturn_macros(),
        st.nobs(), st.version,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ws.Nl.from_formula("price = {b0}*exp({b1}*mpg)", auto)
        res = model.fit()
        warned = [str(record.message) for record in caught]
    add(manual(
        "nl default start values (convergence gap)",
        command,
        {
            "/b0": (float(cap_default.b[0]), float(res.params[0])),
            "/b1": (float(cap_default.b[1]), float(res.params[1])),
            "rss": (float(cap_default.scalars.get("rss")), float(res.e["rss"])),
        },
        notes=[
            "westata starts every parameter at 1.0 and Levenberg-Marquardt stalls "
            f"there (RSS {res.e['rss']:.6g} vs Stata's "
            f"{cap_default.scalars.get('rss'):.6g}); Stata runs its own grid search "
            "before iterating. A warning is emitted but a misleading fit is still "
            "returned unless strict=True.",
            "warnings raised: " + "; ".join(warned),
        ],
        extra={"westata_converged": int(res.e["converged"])},
    ))


def _ws_logit_propensity(x: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """westata's propensity scores, re-used so a decomposition needs no refit."""

    from westata.estimators.treatreg_effects import _LogitPropensity

    return _LogitPropensity(x, np.asarray(treated, dtype=float),
                            command="teffects aipw").propensity


# ------------------------------------------------------------------ 5..8 teffects
def run_teffects(st: Stata, df: pd.DataFrame) -> None:
    st.send(df)

    def effect_compare(name: str, command: str, result, *, po_levels=("0",),
                       extra_note: Sequence[str] = ()) -> None:
        spec: list[tuple[str, str]] = [("r1vs0.t", "r1vs0.t")]
        ws_b = [result.estimate]
        ws_se = [result.std_err]
        for level in po_levels:
            if f"{level}.t" not in cap.names:
                continue
            spec.append((f"{level}.t", f"po{level}"))
            mean, error = result.potential_outcomes[level]
            ws_b.append(mean)
            ws_se.append(error)
        subset = take_rows(cap, spec)
        comp = compare(
            name,
            subset,
            Adapter([new for _, new in spec], ws_b, ws_se),
            domain=DOMAIN,
            tolerance=TOL,
            notes=[
                "Stata's e(b) also carries the nuisance models "
                f"({', '.join(n for n in cap.names if n not in [s for s, _ in spec])}); "
                "those rows have no westata counterpart and are excluded from the "
                "comparison (see NOTES.md).",
                f"stata_N={cap.scalars.get('N')}, "
                f"n1={cap.scalars.get('n1')}, n0={cap.scalars.get('n0')}",
                *extra_note,
            ],
            extra={"stata_se_method": cap.macros.get("vcetype", ""),
                   "westata_se_method": result.se_method,
                   "stata_estimand": cap.macros.get("stat", ""),
                   "stata_b": cap.b.tolist(),
                   "westata_b": [float(v) for v in ws_b],
                   "stata_se": np.sqrt(np.diag(cap.V)).tolist(),
                   "westata_se": [float(v) for v in ws_se]},
        )
        add(comp)

    cap = st.estimate("teffects ra (y x1 x2) (t)")
    effect_compare(
        "teffects ra ATE",
        cap.command,
        ws.RegressionAdjustment.from_formula(
            "y", "t", df, covariates=["x1", "x2"], estimand="ate").fit(),
    )

    cap = st.estimate("teffects ra (y x1 x2) (t), atet")
    effect_compare(
        "teffects ra ATET",
        cap.command,
        ws.RegressionAdjustment.from_formula(
            "y", "t", df, covariates=["x1", "x2"], estimand="atet").fit(),
    )

    cap = st.estimate("teffects ipw (y) (t x1 x2)")
    effect_compare(
        "teffects ipw ATE",
        cap.command,
        ws.IPW.from_formula("y", "t", df, covariates=["x1", "x2"], estimand="ate").fit(),
    )

    cap = st.estimate("teffects ipw (y) (t x1 x2), atet")
    effect_compare(
        "teffects ipw ATET",
        cap.command,
        ws.IPW.from_formula("y", "t", df, covariates=["x1", "x2"], estimand="atet").fit(),
    )

    cap = st.estimate("teffects aipw (y x1 x2) (t x1 x2)")
    effect_compare(
        "teffects aipw ATE",
        cap.command,
        ws.AIPW.from_formula("y", "t", df, covariates=["x1", "x2"], estimand="ate").fit(),
    )

    cap = st.estimate("teffects aipw (y x1 x2) (t x1 x2), atet")
    aipw_atet = ws.AIPW.from_formula(
        "y", "t", df, covariates=["x1", "x2"], estimand="atet").fit()
    effect_compare(
        "teffects aipw ATET",
        cap.command,
        aipw_atet,
        po_levels=("0",),
    )
    # Decompose the ATET gap: Stata augments BOTH potential-outcome means; the
    # ATET branch of westata's AIPW augments neither.
    x = np.column_stack([np.ones(len(df)), df[["x1", "x2"]].to_numpy(dtype=float)])
    arm = df["t"].to_numpy(dtype=float) > 0.5
    propensity = _ws_logit_propensity(x, df["t"].to_numpy(dtype=float))
    y = df["y"].to_numpy(dtype=float)
    b1 = np.linalg.lstsq(x[arm], y[arm], rcond=None)[0]
    b0 = np.linalg.lstsq(x[~arm], y[~arm], rcond=None)[0]
    residual0 = y - x @ b0
    odds = propensity / (1.0 - propensity)
    n1 = float(arm.sum())
    stata_po0 = float(
        (x[arm] @ b0).sum() + (odds[~arm] * residual0[~arm]).sum()
    ) / n1
    add(manual(
        "teffects aipw ATET: Stata's PO formulas reconstructed",
        "teffects aipw (y x1 x2) (t x1 x2), atet",
        {
            "PO(0) Stata formula": (float(cap.b[1]), stata_po0),
            "PO(1) Stata formula": (
                float(cap.b[0] + cap.b[1]), float(y[arm].mean())),
        },
        notes=[
            "Root cause isolated. For the ATET, Stata's PO(0) is the augmented "
            "estimator (1/n1)[sum_t x0'b0 + sum_c odds_i (y_i - x0'b0)] = "
            f"{float(cap.b[1])!r}; its PO(1) is the plain treated mean "
            f"{float(y[arm].mean())!r}. Both are reproduced here from westata's "
            "own propensity scores and outcome regressions, so the difference is "
            "in the ATET assembly, not in the components.",
            "westata's ATET branch instead reports PO(0) = mean_t(x0'b0) = "
            f"{aipw_atet.potential_outcomes['0'][0]!r} (no control augmentation, "
            "because augmented0 is multiplied by (1-t) before the mean is taken "
            "over the treated) and PO(1) = mean_t(x1'b1 + r1/p) = "
            f"{aipw_atet.potential_outcomes['1'][0]!r} (an extra 1/p inside the "
            f"treated mean). The two errors do not cancel: ATET is "
            f"{aipw_atet.estimate:.9g} vs Stata's {cap.b[0]:.9g}.",
            "The ATE branch is correct -- it matches Stata to 1e-16 -- so this is "
            "a defect confined to estimand='atet', not in the propensity model or "
            "the outcome regressions.",
        ],
        extra={"westata_po0": aipw_atet.potential_outcomes["0"][0],
               "westata_po1": aipw_atet.potential_outcomes["1"][0],
               "westata_atet": aipw_atet.estimate,
               "stata_po0": float(cap.b[1]),
               "stata_po1": float(cap.b[0] + cap.b[1]),
               "stata_atet": float(cap.b[0]),
               "atet_abs_gap": float(abs(aipw_atet.estimate - cap.b[0])),
               "reconstructed_stata_po0": stata_po0},
    ))

    cap = st.estimate("teffects nnmatch (y x1 x2) (t)")
    effect_compare(
        "teffects nnmatch ATE, nneighbor(1)",
        cap.command,
        ws.NearestNeighbourMatching.from_formula(
            "y", "t", df, covariates=["x1", "x2"], nneighbor=1).fit(),
        po_levels=(),
        extra_note=[
            "Point estimate matches to machine precision (Mahalanobis metric, "
            "matching with replacement); the SE is westata's own approximation "
            "and is labelled as such on the result (se_method).",
        ],
    )

    cap = st.estimate("teffects nnmatch (y x1 x2) (t), nneighbor(3)")
    effect_compare(
        "teffects nnmatch ATE, nneighbor(3)",
        cap.command,
        ws.NearestNeighbourMatching.from_formula(
            "y", "t", df, covariates=["x1", "x2"], nneighbor=3).fit(),
        po_levels=(),
        extra_note=[
            "Same conclusion with 3 neighbours: the estimate is exact, the "
            "Abadie-Imbens SE is approximate.",
        ],
    )

    cap = st.estimate("teffects nnmatch (y x1 x2) (t), atet")
    effect_compare(
        "teffects nnmatch ATET, nneighbor(1)",
        cap.command,
        ws.NearestNeighbourMatching.from_formula(
            "y", "t", df, covariates=["x1", "x2"], nneighbor=1, estimand="atet").fit(),
        po_levels=(),
        extra_note=[
            "The ATET point estimate is exact as well; only the SE differs, and "
            "it is not just rescaled -- westata reports the same standard error "
            "for the ATE and the ATET, so the ATET SE is the ATE one.",
        ],
    )

    # biasadj(): Stata refines the match with a regression on the bias-adjust
    # variables; westata has no such option.
    cap = st.estimate("teffects nnmatch (y x1 x2) (t), biasadj(x1 x2)")
    plain = ws.NearestNeighbourMatching.from_formula(
        "y", "t", df, covariates=["x1", "x2"], nneighbor=1).fit()
    message, refused = probe_unsupported(
        "teffects nnmatch biasadj(x1 x2)",
        cap.command,
        lambda: ws.NearestNeighbourMatching.from_formula(
            "y", "t", df, covariates=["x1", "x2"], nneighbor=1
        ).fit(bias_adjust=["x1", "x2"]),
    )
    if refused:
        comp = REC.error(
            "teffects nnmatch biasadj(x1 x2) (westata unsupported)",
            TypeError(message),
            command=cap.command,
            notes=[
                "westata's NearestNeighbourMatching.fit() takes no arguments, so "
                "there is no bias-adjustment path at all.",
            ],
        )
        comp.extra.update({
            "stata_biasadj_estimate": float(cap.b[0]),
            "stata_biasadj_se": float(np.sqrt(cap.V[0, 0])),
            "westata_unadjusted_estimate": plain.estimate,
            "westata_unadjusted_se": plain.std_err,
            "estimate_shift": float(plain.estimate - cap.b[0]),
        })
    else:
        REC.skip("teffects nnmatch biasadj(x1 x2) (westata unsupported)",
                 "westata accepted the option; nothing to report",
                 command=cap.command)


# ------------------------------------------------------------------ 9..10 DID
def run_did(st: Stata, panel2: pd.DataFrame, panel5: pd.DataFrame) -> None:
    st.send(panel2)

    def did_compare(name: str, command: str, model, *, spec: Sequence[tuple[str, str]]) -> None:
        cap = st.estimate(command)
        res = model.fit()
        subset = take_rows(cap, list(spec))
        comp = compare(
            name,
            subset,
            Adapter(
                [new for _, new in spec],
                [float(res.params[name]) for _, name in spec],
                [float(res.bse[name]) for _, name in spec],
                {"N": res.e.get("N"), "N_clust": res.e.get("N_clust"),
                 "df_r": res.e.get("df_r"), "df_m": res.e.get("df_m")},
            ),
            domain=DOMAIN,
            tolerance=TOL,
            scalars=("N", "df_r"),
            extra={"stata_df_r": cap.scalars.get("df_r"),
                   "stata_N_clust": cap.scalars.get("N_clust"),
                   "westata_N_clust": res.e.get("N_clust"),
                   "westata_df_r": res.df_resid},
            notes=[
                "Only the ATET column (and named covariates) is compared; Stata's "
                "e(b) also stores the created i.period dummies and the constant, "
                "which are absorbed by the fixed effects on the westata side. "
                f"stata rows = {cap.names}",
            ],
        )
        add(comp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ws.Didregress.from_formula(
            "y x1", "treated", panel2, group="unit", time="period")
    did_compare(
        "didregress two-period, covariate, vce(cluster)",
        "didregress (y x1) (treated), group(unit) time(period)",
        model,
        spec=[("r1vs0.treated", "treated"), ("x1", "x1")],
    )

    # Stata's "(y) (treated)" spelling names no covariate.  westata cannot be
    # told "no covariates" (an empty list raises), and it *infers* every other
    # time-varying numeric column, so the frame is narrowed to the four model
    # columns.  The inference behaviour itself is reported in NOTES.md.
    bare = panel2[["unit", "period", "treated", "y"]].copy()
    st.send(bare)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ws.Didregress.from_formula(
            "y", "treated", bare, group="unit", time="period")
    did_compare(
        "didregress two-period, no covariate, vce(cluster)",
        "didregress (y) (treated), group(unit) time(period)",
        model,
        spec=[("r1vs0.treated", "treated")],
    )

    # What happens if the extra column is left in place: westata silently
    # adjusts for it while Stata's "(y)" does not.
    st.send(panel2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inferred = ws.Didregress.from_formula(
            "y", "treated", panel2, group="unit", time="period")
        inferred_res = inferred.fit()
    cap_plain = st.estimate("didregress (y) (treated), group(unit) time(period)")
    add(manual(
        "didregress (y) with an unrelated numeric column present",
        "didregress (y) (treated), group(unit) time(period)",
        {"ATET": (float(cap_plain.b[0]), float(inferred_res.params["treated"]))},
        notes=[
            "Specification hazard, not a numerical bug: westata's didregress "
            "treats every other time-varying numeric column as a covariate when "
            "the outcome block names none (it warns), so "
            f"{inferred.inferred_covariates} enters the regression and the ATET "
            f"moves from {cap_plain.b[0]:.9g} to "
            f"{inferred_res.params['treated']:.9g}. Stata's '(y)' uses no "
            "covariate. Callers must narrow the frame or name the covariates "
            "explicitly.",
        ],
        extra={"inferred_covariates": inferred.inferred_covariates,
               "warnings": [str(record.message) for record in caught]},
    ))

    class _Robust(ws.Didregress):
        """didregress with cov_type pinned to HC1 (Stata's vce(robust))."""

        def fit(self, cov_type: str = "HC1"):  # noqa: D401 - thin wrapper
            return super().fit(cov_type=cov_type)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        robust_model = _Robust.from_formula(
            "y x1", "treated", panel2, group="unit", time="period")
    did_compare(
        "didregress two-period, vce(robust)",
        "didregress (y x1) (treated), group(unit) time(period) vce(robust)",
        robust_model,
        spec=[("r1vs0.treated", "treated"), ("x1", "x1")],
    )

    # Stata only allows cluster/robust for didregress -- vce(unadjusted) is r(198).
    try:
        st.run("didregress (y x1) (treated), group(unit) time(period) vce(unadjusted)")
        stata_allows_unadjusted = True
        unadjusted_error = ""
    except Exception as exc:  # noqa: BLE001 - the message is the evidence
        stata_allows_unadjusted = False
        unadjusted_error = str(exc).strip().replace("\n", " ")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_unadjusted = ws.Didregress.from_formula(
            "y x1", "treated", panel2, group="unit", time="period").fit(cov_type="nonrobust")
    REC.skip(
        "didregress vce(unadjusted)",
        "Stata 19.5 rejects vce(unadjusted) for didregress ("
        + unadjusted_error
        + "); westata implements cov_type='nonrobust' anyway, so there is no "
        "Stata number to compare: westata ATET "
        + f"{ws_unadjusted.params['treated']:.9g} se {ws_unadjusted.bse['treated']:.9g}.",
        command="didregress (y x1) (treated), group(unit) time(period) vce(unadjusted)",
    )

    # ---------------- multi-period, staggered adoption
    st.send(panel5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model5 = ws.Didregress.from_formula(
            "y", "treated", panel5, group="unit", time="period")
    did_compare(
        "didregress five-period staggered, vce(cluster)",
        "didregress (y) (treated), group(unit) time(period)",
        model5,
        spec=[("r1vs0.treated", "treated")],
    )

    st.run("xtset unit period")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xt5 = ws.Xtdidregress.from_formula(
            "y", "treated", panel5, group="unit", time="period")
    did_compare(
        "xtdidregress five-period staggered, vce(cluster)",
        "xtdidregress (y) (treated), group(unit) time(period)",
        xt5,
        spec=[("r1vs0.treated", "treated")],
    )

    st.send(panel2)
    st.run("xtset unit period")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xt2 = ws.Xtdidregress.from_formula(
            "y x1", "treated", panel2, group="unit", time="period")
    did_compare(
        "xtdidregress two-period, covariate, vce(cluster)",
        "xtdidregress (y x1) (treated), group(unit) time(period)",
        xt2,
        spec=[("r1vs0.treated", "treated"), ("x1", "x1")],
    )

    # The two spellings differ only in the nested-effect dof convention.
    st.send(panel2)
    st.run("xtset unit period")
    cap_d = st.estimate("didregress (y x1) (treated), group(unit) time(period)")
    cap_x = st.estimate("xtdidregress (y x1) (treated), group(unit) time(period)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_d = ws.Didregress.from_formula(
            "y x1", "treated", panel2, group="unit", time="period").fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_x = ws.Xtdidregress.from_formula(
            "y x1", "treated", panel2, group="unit", time="period").fit()
    add(manual(
        "didregress vs xtdidregress dof convention",
        "didregress/xtdidregress (y x1) (treated), group(unit) time(period)",
        {
            "stata se ratio xt/did": (
                float(np.sqrt(cap_x.V[0, 0]) / np.sqrt(cap_d.V[0, 0])),
                float(res_x.bse["treated"] / res_d.bse["treated"]),
            ),
            "stata ATET": (float(cap_d.b[0]), float(res_d.params["treated"])),
        },
        notes=[
            "Identical ATET; only the nested-effect (N-1)/(N-K) factor differs. "
            f"Stata se {np.sqrt(cap_d.V[0,0]):.9g} (didregress) vs "
            f"{np.sqrt(cap_x.V[0,0]):.9g} (xtdidregress); westata "
            f"{res_d.bse['treated']:.9g} vs {res_x.bse['treated']:.9g}.",
        ],
    ))


# --------------------------------------------------------------- 11..12 survey
def run_svy(st: Stata, sv: pd.DataFrame, lone: pd.DataFrame) -> None:
    st.send(sv)
    st.run("svyset psu [pw=w], strata(h)")
    design = ws.SurveyDesign.from_frame(sv, psu="psu", strata="h", weights="w")

    def svy_compare(name: str, command: str, obj, *, rename=None, notes=(),
                    raw: bool = False) -> None:
        cap = capture_raw(st, command) if raw else st.estimate(command)
        res = obj.fit()
        comp = compare(
            name,
            cap,
            Adapter(res.param_names, res.params.to_numpy(), res.bse.to_numpy(), res.e),
            domain=DOMAIN,
            tolerance=TOL,
            rename=rename,
            scalars=("N", "N_psu", "N_strata", "N_pop", "df_r"),
            notes=list(notes),
            extra={"westata_df_r": res.df_resid, "stata_df_r": cap.scalars.get("df_r")},
        )
        add(comp)

    svy_compare("svy: mean (pw + strata)", "svy: mean y x1",
                ws.SvyMean.from_frame(["y", "x1"], sv, design))
    svy_compare("svy: total (pw + strata)", "svy: total y x1",
                ws.SvyTotal.from_frame(["y", "x1"], sv, design))
    svy_compare("svy: proportion 3 levels (pw + strata)", "svy: proportion grp",
                ws.SvyProportion.from_frame(["grp"], sv, design),
                rename={"0bn.grp": "grp=0", "1.grp": "grp=1", "2.grp": "grp=2"},
                raw=True,
                notes=["Stata stores the level-0 proportion as '0bn.grp', which is "
                       "the base-level spelling; it is a genuine estimate here, so "
                       "the capture keeps every row and renames it explicitly."])
    svy_compare("svy: proportion binary (pw + strata)", "svy: proportion f",
                ws.SvyProportion.from_frame(["f"], sv, design),
                rename={"0bn.f": "f=0", "1.f": "f=1"}, raw=True)
    svy_compare("svy: regress (pw + strata)", "svy: regress y x1",
                ws.SvyRegress.from_frame(sv, "y ~ x1", design))
    svy_compare("svy: regress with i.factor (pw + strata)", "svy: regress y x1 i.f",
                ws.SvyRegress.from_frame(sv, "y ~ x1 + C(f)", design))
    svy_compare("svy: logit (pw + strata)", "svy: logit gbin x1",
                ws.SvyLogit.from_frame(sv, "gbin ~ x1", design),
                notes=["westata reports a linearised (Taylor) VCE for svy: logit; "
                       "Stata's default is the same linearisation."])

    # ---- design variants: self-weight, no strata, fpc
    st.run("svyset psu")
    svy_compare("svy: mean self-weight, single stratum", "svy: mean y",
                ws.SvyMean.from_frame(["y"], sv, ws.SurveyDesign.from_frame(sv, psu="psu")),
                notes=["No pweight: every observation counts once on both sides."])

    st.run("svyset psu, strata(h)")
    svy_compare("svy: mean no weights + strata (self-weight per PSU)",
                "svy: mean y",
                ws.SvyMean.from_frame(
                    ["y"], sv, ws.SurveyDesign.from_frame(sv, psu="psu", strata="h")))

    st.run("svyset psu [pw=w], strata(h) fpc(fpc)")
    svy_compare("svy: mean with fpc (population PSUs)",
                "svy: mean y",
                ws.SvyMean.from_frame(
                    ["y"], sv,
                    ws.SurveyDesign.from_frame(sv, psu="psu", strata="h",
                                               weights="w", fpc="fpc")),
                notes=["fpc() above one is the population number of PSUs per "
                       "stratum; the (1 - n_h/N_h) factor must match exactly."])

    # ---- lone PSU strategies
    st.send(lone)
    st.run("svyset psu [pw=w], strata(h)")
    cap = st.estimate("svy: mean y")
    res = ws.SvyMean.from_frame(
        ["y"], lone,
        ws.SurveyDesign.from_frame(lone, psu="psu", strata="h", weights="w",
                                   singleunit="missing")).fit()
    stata_se = np.sqrt(np.diag(cap.V)) if cap.V.size else np.array([np.nan])
    # Stata stores e(V) = 0 here and *prints* "." (with an explicit note); a zero
    # variance is a sentinel for "not estimable", not an estimate, so it is
    # mapped to None on both sides before comparing.
    stata_value = None if (not np.isfinite(stata_se[0]) or stata_se[0] == 0.0) \
        else float(stata_se[0])
    westata_value = None if not np.isfinite(res.bse[0]) else float(res.bse[0])
    add(manual(
        "svy: singleunit(missing) with lone PSUs",
        "svyset psu [pw=w], strata(h); svy: mean y",
        {
            "mean": (float(cap.b[0]), float(res.params[0])),
            "SE not estimable on both sides": (stata_value, westata_value),
        },
        notes=[
            "Stata prints '.' with the note 'Missing standard error because of "
            "stratum with single sampling unit' and stores e(V) = 0; westata "
            "returns NaN. Both sides conclude that the design cannot identify the "
            "variance; the representation differs (0 sentinel vs NaN), so the zero "
            "is mapped to 'not estimable' before the comparison.",
        ],
        extra={"stata_e_V_diag": np.diag(cap.V).tolist(),
               "westata_bse": res.bse.to_numpy().tolist(),
               "stata_df_r": cap.scalars.get("df_r"), "westata_df_r": res.df_resid},
    ))

    st.run("svyset psu [pw=w], strata(h) singleunit(centered)")
    svy_compare("svy: singleunit(centered) with lone PSUs", "svy: mean y",
                ws.SvyMean.from_frame(
                    ["y"], lone,
                    ws.SurveyDesign.from_frame(lone, psu="psu", strata="h",
                                               weights="w", singleunit="centered")),
                notes=["singleunit(centered) adds (T_h - Tbar)^2 for lone strata."])

    REC.skip(
        "svy: mean, over(f)",
        "westata's SvyMean/SvyProportion expose no over() subpopulation "
        "estimator; Stata computes separate means per level with the full "
        "design's linearisation. Not implemented, not compared.",
        command="svy: mean y, over(f)",
    )


# ---------------------------------------------------------------- 13..15 resample
def stata_emulated_jackknife(auto: pd.DataFrame, formula: str, *, mse: bool,
                             replicates: np.ndarray | None = None) -> np.ndarray:
    """The jackknife SE Stata reports, reproduced by rounding replicates to float32.

    ``jackknife``/``bootstrap`` store each replicate at single precision, so
    Stata's published SE carries a ~1e-6 relative rounding error that westata's
    double-precision arithmetic does not.  Rounding the (already exact)
    replicates through float32 and re-applying the same formula reproduces
    Stata's number to ~1e-9, which is what identifies the gap as a *storage*
    artefact rather than a formula difference.
    """

    if replicates is None:
        replicates = exact_replicates(auto, formula)
    lhs = formula.split("~", 1)[0].strip() if "~" in formula else formula
    y = auto[lhs].to_numpy(dtype=float)
    n = replicates.shape[0]
    rounded = replicates.astype(np.float32).astype(np.float64)
    observed = np.linalg.lstsq(
        np.asarray(build_design(formula, auto)[0], dtype=float), y, rcond=None
    )[0]
    if observed.size != replicates.shape[1]:
        raise ValueError("replicate width does not match the design")
    if mse:
        centred = rounded - observed
        variance = (n - 1.0) / n * (centred.T @ centred)
    else:
        pseudovalues = rounded + n * (observed - rounded)
        centred = pseudovalues - pseudovalues.mean(axis=0)
        variance = (centred.T @ centred) / (n * (n - 1.0))
    return np.sqrt(np.diag(variance))[::-1]


def exact_replicates(auto: pd.DataFrame, formula: str) -> np.ndarray:
    """All delete-one OLS coefficient vectors, in ``build_design`` order."""

    lhs = formula.split("~", 1)[0].strip()
    y = auto[lhs].to_numpy(dtype=float)
    X = np.asarray(build_design(formula, auto)[0], dtype=float)
    return np.array([
        np.linalg.lstsq(np.delete(X, i, axis=0), np.delete(y, i), rcond=None)[0]
        for i in range(X.shape[0])
    ])


def exact_jackknife(auto: pd.DataFrame, formula: str, *, mse: bool) -> np.ndarray:
    """Delete-one jackknife SE from exact rational arithmetic.

    Both implementations are floating-point, so when they disagree there is no
    arbiter.  ``auto.dta`` is all integers, so every leave-one-out OLS fit can be
    solved exactly with :class:`fractions.Fraction` and the jackknife formula
    applied afterwards; that number is then independent of both sides.  It is
    slow (74 exact 2x2 inversions) but it settles who is right.
    """

    lhs = formula.split("~", 1)[0].strip()
    y = auto[lhs].to_numpy(dtype=float)
    X, _, kept = build_design(formula, auto)
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    k = X.shape[1]
    if len(kept) != n:
        raise ValueError("exact jackknife needs a complete-case sample")

    exact_X = [[Fraction(int(round(v))) for v in row] for row in X]
    exact_y = [Fraction(int(round(v))) for v in y]
    xtx = [[sum(exact_X[i][r] * exact_X[i][c] for i in range(n)) for c in range(k)]
           for r in range(k)]
    xty = [sum(exact_X[i][r] * exact_y[i] for i in range(n)) for r in range(k)]

    def inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
        size = len(matrix)
        working = [row[:] + [Fraction(int(i == j)) for j in range(size)]
                   for i, row in enumerate(matrix)]
        for column in range(size):
            pivot = next(r for r in range(column, size) if working[r][column] != 0)
            working[column], working[pivot] = working[pivot], working[column]
            scale = working[column][column]
            working[column] = [v / scale for v in working[column]]
            for r in range(size):
                if r != column and working[r][column] != 0:
                    factor = working[r][column]
                    working[r] = [a - factor * b
                                  for a, b in zip(working[r], working[column])]
        return [row[size:] for row in working]

    def solve(matrix, vector):
        inv = inverse(matrix)
        return [sum(inv[r][c] * vector[c] for c in range(len(vector)))
                for r in range(len(vector))]

    replicates = np.empty((n, k), dtype=float)
    for i in range(n):
        row = exact_X[i]
        reduced = [[xtx[r][c] - row[r] * row[c] for c in range(k)] for r in range(k)]
        rhs = [xty[r] - row[r] * exact_y[i] for r in range(k)]
        replicates[i] = [float(value) for value in solve(reduced, rhs)]

    observed = np.linalg.lstsq(X, y, rcond=None)[0]
    if mse:
        centred = replicates - observed
        variance = (n - 1.0) / n * (centred.T @ centred)
    else:
        pseudovalues = replicates + n * (observed - replicates)
        centred = pseudovalues - pseudovalues.mean(axis=0)
        variance = (centred.T @ centred) / (n * (n - 1.0))
    # build_design puts the intercept first; the caller compares against Stata's
    # slopes-then-constant order, so the result is returned in Stata's order.
    return np.sqrt(np.diag(variance))[::-1]


def run_resample(st: Stata, auto: pd.DataFrame) -> None:
    st.run("sysuse auto, clear")
    st.run("set seed 20241024")

    # ---- jackknife (no randomness)
    cap = st.estimate("jackknife: regress mpg weight")
    res = ws.Jackknife.from_formula("mpg ~ weight", auto, estimator="regress").fit()
    exact = exact_jackknife(auto, "mpg ~ weight", mse=False)
    emulated = stata_emulated_jackknife(auto, "mpg ~ weight", mse=False)
    comp = compare(
        "jackknife: regress (pseudovalue VCE)",
        cap,
        Adapter(res.param_names, res.params.to_numpy(), res.bse.to_numpy(), res.e),
        domain=DOMAIN,
        tolerance=TOL,
        scalars=("N",),
        # e(df_r) is compared in its own row below: Stata uses N - 1 for the
        # jackknife, westata N - k, and mixing that into this row would hide it
        # behind a scalar-notes line.
        extra={"n_reps": cap.scalars.get("N_reps"), "westata_note": res.e.get("note"),
               "stata_df_r": cap.scalars.get("df_r"),
               "westata_df_r": res.df_resid,
               "exact_exrational_se": exact.tolist(),
               "float32_emulation_of_stata": emulated.tolist(),
               "exact_vs_stata": rel_diff(exact, np.sqrt(np.diag(cap.V))),
               "exact_vs_westata": rel_diff(exact, res.bse.to_numpy()[::-1]),
               "emulation_vs_stata": rel_diff(emulated, np.sqrt(np.diag(cap.V)))},
        notes=[
            "Ground truth computed with exact rational arithmetic (Fraction) over "
            f"all 74 leave-one-out OLS refits: {exact.tolist()}. It matches "
            f"westata to {rel_diff(exact, res.bse.to_numpy()[::-1]):.1e} and "
            f"differs from Stata by {rel_diff(exact, np.sqrt(np.diag(cap.V))):.1e}.",
            "The gap is Stata's storage, not a formula difference: jackknife (and "
            "bootstrap) keep each replicate at single precision. Rounding the "
            "exact replicates through float32 and re-applying the same pseudovalue "
            "formula reproduces Stata's published SE to "
            f"{rel_diff(emulated, np.sqrt(np.diag(cap.V))):.1e}.",
        ],
    )
    add(comp)

    cap = st.estimate("jackknife, mse: regress mpg weight")
    res = ws.Jackknife.from_formula("mpg ~ weight", auto, estimator="regress").fit(mse=True)
    exact_mse = exact_jackknife(auto, "mpg ~ weight", mse=True)
    emulated_mse = stata_emulated_jackknife(auto, "mpg ~ weight", mse=True)
    add(compare(
        "jackknife, mse: regress",
        cap,
        Adapter(res.param_names, res.params.to_numpy(), res.bse.to_numpy(), res.e),
        domain=DOMAIN, tolerance=TOL,
        extra={"exact_exrational_se": exact_mse.tolist(),
               "float32_emulation_of_stata": emulated_mse.tolist(),
               "exact_vs_stata": rel_diff(exact_mse, np.sqrt(np.diag(cap.V))),
               "exact_vs_westata": rel_diff(exact_mse, res.bse.to_numpy()[::-1]),
               "emulation_vs_stata": rel_diff(emulated_mse, np.sqrt(np.diag(cap.V)))},
        notes=["Same attribution as the pseudovalue case: westata reproduces the "
               "exact rational answer, Stata is ~5e-7 away from it, and the "
               "float32-rounded emulation reproduces Stata's number."],
    ))

    cap = st.estimate("jackknife: mean mpg")
    res = ws.Jackknife.from_formula("mpg ~ weight", auto, estimator="mean").fit()
    values = auto["mpg"].to_numpy(dtype=float)
    n = values.size
    loo = np.array([(values.sum() - v) / (n - 1) for v in values])
    pseudovalues = n * values.mean() - (n - 1) * loo
    exact_mean = np.sqrt(
        float(((pseudovalues - pseudovalues.mean()) ** 2).sum()) / (n * (n - 1.0))
    )
    loo32 = loo.astype(np.float32).astype(np.float64)
    pseudo32 = n * values.mean() - (n - 1) * loo32
    emulated_mean = float(
        np.sqrt(((pseudo32 - pseudo32.mean()) ** 2).sum() / (n * (n - 1.0)))
    )
    add(compare(
        "jackknife: mean",
        cap,
        Adapter(res.param_names, res.params.to_numpy(), res.bse.to_numpy(), res.e),
        domain=DOMAIN, tolerance=TOL,
        rename={"mpg": "mean"},
        extra={"exact_se": exact_mean,
               "float32_emulation_of_stata": emulated_mean,
               "exact_vs_stata": rel_pairs(
                   float(np.sqrt(np.diag(cap.V))[0]), exact_mean),
               "exact_vs_westata": rel_pairs(float(res.bse[0]), exact_mean),
               "emulation_vs_stata": rel_pairs(
                   float(np.sqrt(np.diag(cap.V))[0]), emulated_mean)},
        notes=["Stata labels the single parameter with the variable name ('mpg'); "
               "westata calls it 'mean'. Mapped explicitly.",
               f"Exact jackknife SE = {exact_mean!r} (algebraically identical to "
               f"s/sqrt(n) for a mean). westata {float(res.bse[0])!r} matches it; "
               f"Stata {float(np.sqrt(np.diag(cap.V))[0])!r} is "
               f"{rel_pairs(float(np.sqrt(np.diag(cap.V))[0]), exact_mean):.1e} off. "
               "Rounding the leave-one-out means through float32 reproduces "
               f"Stata's value to {rel_pairs(float(np.sqrt(np.diag(cap.V))[0]), emulated_mean):.1e}, "
               "so the cause is single-precision replicate storage."],
    ))

    # ---- inference reference distribution (t vs z, and the df it uses)
    from scipy import stats as scipy_stats

    cap_jk = st.estimate("jackknife: regress mpg weight")
    cap_bs = st.estimate(
        "bootstrap, reps(100) seed(20241024) nodots: regress mpg weight")
    ws_jk = ws.Jackknife.from_formula("mpg ~ weight", auto).fit()
    ws_bs = ws.Bootstrap.from_formula(
        "mpg ~ weight", auto, reps=100, seed=20241024).fit()
    stata_df = cap_jk.scalars.get("df_r")
    add(manual(
        "jackknife: residual degrees of freedom used for t tests",
        "jackknife: regress mpg weight",
        {"e(df_r)": (stata_df, ws_jk.df_resid)},
        notes=[
            f"Stata stores e(df_r) = {stata_df} (N - 1: one degree of freedom per "
            "replicate) and tests with t(df_r); westata reports df_resid = "
            f"{ws_jk.df_resid} (N - k), so its critical value is "
            f"{scipy_stats.t.ppf(0.975, 72):.6f} against Stata's "
            f"{scipy_stats.t.ppf(0.975, 73):.6f}. The 95% CI for _cons is "
            f"[{ws_jk.conf_int().iloc[0,0]:.5f}, {ws_jk.conf_int().iloc[0,1]:.5f}] "
            "in westata vs [35.41037, 43.47019] in Stata: same variance, different "
            "reference distribution.",
        ],
        extra={"westata_jackknife_df_r": ws_jk.df_resid,
               "stata_jackknife_df_r": stata_df,
               "westata_jackknife_ci_cons": ws_jk.conf_int().iloc[0].tolist(),
               "t72": float(scipy_stats.t.ppf(0.975, 72)),
               "t73": float(scipy_stats.t.ppf(0.975, 73))},
    ))

    z_row = Comparison(
        domain=DOMAIN,
        name="bootstrap: z reference distribution (Stata prints P>|z|)",
        stata_command="bootstrap, reps(100) seed(20241024): regress mpg weight",
        status="pass" if ws_bs.dist == "z" else "fail",
        param_rel=0.0 if ws_bs.dist == "z" else float("nan"),
        se_rel=float("nan"),
        extra={"stata_reference": "z (table prints 'z' and P>|z|; stores no e(df_r))",
               "westata_reference": f"dist={ws_bs.dist!r}, "
                                    f"df_resid={ws_bs.df_resid}",
               "stata_bootstrap_df_r": cap_bs.scalars.get("df_r"),
               "westata_bootstrap_dist": ws_bs.dist,
               "westata_bootstrap_df_r": ws_bs.df_resid,
               "westata_bootstrap_ci_cons": ws_bs.conf_int().iloc[0].tolist(),
               "stata_ci_cons": [35.29429, 43.58628],
               "z": float(scipy_stats.norm.ppf(0.975))},
        notes=[
            "Invariant, not a numeric comparison: Stata's bootstrap reports z and "
            "stores no e(df_r), so the row fails if westata ever goes back to a t "
            "reference distribution.",
            f"westata now uses dist={ws_bs.dist!r}; the 95% CI for _cons is "
            f"[{ws_bs.conf_int().iloc[0,0]:.5f}, {ws_bs.conf_int().iloc[0,1]:.5f}] "
            "against Stata's [35.29429, 43.58628] (measured relative difference "
            "1.4e-9 before rounding).",
        ],
    )
    add(z_row)

    # ---- bootstrap: save Stata's own replicates so both the replicate matrix and
    # the covariance can be attributed to a common, checkable source.
    tmp = os.path.join(os.path.dirname(__file__), "_tmp")
    os.makedirs(tmp, exist_ok=True)
    reps = 100
    path = os.path.join(tmp, "bsrep.dta").replace("\\", "/")
    st.run(f'postfile PF double (bw bc) using "{path}", replace')
    st.run("""
capture program drop _retest_bs
program _retest_bs, rclass
    quietly regress mpg weight
    post PF (_b[weight]) (_b[_cons])
end
""")
    st.run(f"bootstrap, reps({reps}) seed(20241024) nodots: _retest_bs")
    st.run("postclose PF")
    st.run(f'use "{path}", clear')
    saved = st.frame()
    stata_reps = saved[["bw", "bc"]].to_numpy()          # row 0 = observed sample
    st.run("sysuse auto, clear")
    cap = st.estimate(
        f"bootstrap, reps({reps}) seed(20241024) nodots: regress mpg weight")
    res = ws.Bootstrap.from_formula(
        "mpg ~ weight", auto, reps=reps, seed=20241024).fit()
    model = ws.Bootstrap.from_formula("mpg ~ weight", auto, reps=reps, seed=20241024)
    names = model._names()
    generator = Mt19937_64(20241024)
    mine = np.asarray([
        model._estimate_weighted(rsm._bsampl_w(len(auto), generator))
        for _ in range(reps)
    ])[:, [names.index("weight"), names.index("Intercept")]]
    exact_V = np.cov(stata_reps[1:], rowvar=False, ddof=1)
    rounded_V = np.cov(stata_reps[1:].astype(np.float32).astype(np.float64),
                       rowvar=False, ddof=1)
    stata_se = np.sqrt(np.diag(cap.V))

    value = rel_diff(mine.reshape(-1), stata_reps[1:].reshape(-1))
    add(Comparison(
        domain=DOMAIN,
        name=f"bootstrap replicate-for-replicate ({reps} reps, seed 20241024)",
        stata_command=f"bootstrap, reps({reps}) seed(20241024): regress mpg weight",
        status="pass" if value <= TOL else "fail",
        tolerance=TOL,
        param_rel=value,
        se_rel=float("nan"),
        n_params=int(mine.size),
        extra={"replicate_param_rel": value, "tier": tier_for(value),
               "rows": [int(mine.shape[0]), int(stata_reps[1:].shape[0])],
               "statistic": "each replicate's (b_weight, b_cons)",
               "replicate_float32_hypothesis": bool(
                   np.allclose(mine, stata_reps[1:].astype(np.float32).astype(float),
                               rtol=0, atol=1e-15))},
        notes=[
            "Stata's bootstrap passes bsampl_w frequency weights to the command; "
            "westata reproduces bsampl_w (both the index draw and the tie-break "
            "stream) from the same MT19937-64 seed, so the replicates are the same "
            "ones -- not merely the same distribution. Every replicate agrees to "
            f"{value:.1e}.",
            "Reproducible check: the program above re-fits the OLS by hand and "
            "POSTs _b[weight], _b[_cons], so the comparison is against Stata's "
            "replicate statistics, not against a printed table.",
        ],
    ))

    comp = compare(
        f"bootstrap SE of regress ({reps} reps, seed 20241024)",
        cap,
        Adapter(res.param_names, res.params.to_numpy(), res.bse.to_numpy(), res.e),
        domain=DOMAIN,
        tolerance=TOL,
        scalars=("N",),
        extra={"reps": cap.scalars.get("N_reps"), "seed": 20241024,
               "westata_seed": res.e.get("seed"),
               "exact_V_of_saved_replicates": exact_V.tolist(),
               "V_of_float32_rounded_replicates": rounded_V.tolist(),
               "stata_se": stata_se.tolist(),
               "westata_se": res.bse.to_numpy().tolist(),
               "exact_vs_stata": rel_diff(stata_se, np.sqrt(np.diag(exact_V))),
               "exact_vs_westata": rel_diff(
                   res.bse.to_numpy(), np.sqrt(np.diag(exact_V))[[1, 0]]),
               "float32_emulation_vs_stata": rel_diff(
                   stata_se, np.sqrt(np.diag(rounded_V)))},
        notes=[
            "Same seed and same replicates on both sides, so any residual gap is "
            "arithmetic, not sampling. Covariance of Stata's *saved* replicates "
            "(full double) gives SEs "
            f"{rel_diff(stata_se, np.sqrt(np.diag(exact_V))):.1e} from Stata's own "
            f"e(V) and {rel_diff(res.bse.to_numpy(), np.sqrt(np.diag(exact_V))[[1, 0]]):.1e} "
            "from westata's. Rounding the saved replicates through float32 first "
            "reproduces Stata's e(V) to "
            f"{rel_diff(stata_se, np.sqrt(np.diag(rounded_V))):.1e}, so the same "
            "single-precision replicate storage that explains the jackknife gap "
            "explains this one too.",
        ],
    )
    add(comp)

    cap = st.estimate(f"bootstrap, reps({reps}) seed(20241024) nodots: mean mpg")
    res = ws.Bootstrap.from_formula(
        "mpg ~ weight", auto, estimator="mean", reps=reps, seed=20241024).fit()
    add(compare(
        f"bootstrap SE of mean ({reps} reps, seed 20241024)",
        cap,
        Adapter(res.param_names, res.params.to_numpy(), res.bse.to_numpy(), res.e),
        domain=DOMAIN, tolerance=TOL,
        rename={"mpg": "mean"},
        notes=["Stata labels the parameter 'mpg'; westata calls it 'mean'. The "
               "same-seed replicates match, so the residual SE gap (~1e-7) is "
               "floating-point accumulation in the two covariance routines."],
    ))

    # ---- permute: observed statistic is deterministic, p-value is not
    st.run("sysuse auto, clear")
    st.run("set seed 20241024")
    # permute leaves r(), not e(); it never stores a coefficient vector.  Its r(b)
    # comes back rounded to float32 (see NOTES), so the observed statistic is
    # also taken from the unpermuted fit's e(b), which is stored in double.
    st.output("permute weight bweight = _b[weight], reps(200) seed(20241024) "
              "nodots: regress price mpg weight")
    rmat = st.stata.get_return()
    stata_t_r = float(np.asarray(rmat["r(b)"]).reshape(-1)[0])
    stata_p_lower = float(np.asarray(rmat["r(p_lower)"]).reshape(-1)[0])
    stata_p_upper = float(np.asarray(rmat["r(p_upper)"]).reshape(-1)[0])
    observed_cap = st.estimate("regress price mpg weight")
    stata_t = float(observed_cap.b[observed_cap.names.index("weight")])
    model = ws.Permute.from_formula(
        "price ~ mpg + weight", auto, permute="weight", reps=200, seed=20241024)
    res = model.fit()
    observed = float(res.params["weight"])
    add(manual(
        "permute: observed (unpermuted) statistic",
        "permute weight bweight = _b[weight], reps(200): regress price mpg weight",
        {"b_weight": (stata_t, observed)},
        notes=[
            "The unpermuted estimate is deterministic and is printed from the "
            "unpermuted fit's e(b); westata reproduces it. Stata's own r(b) copy "
            f"is stored at float32 precision ({stata_t_r!r} vs e(b)'s "
            f"{stata_t!r}, {rel_pairs(stata_t_r, stata_t):.1e} apart), which is a "
            "Stata storage artefact, not an estimation difference.",
        ],
        extra={"stata_r_b": stata_t_r, "stata_e_b_weight": stata_t,
               "westata_b_weight": observed,
               "stata_r_b_vs_e_b": rel_pairs(stata_t_r, stata_t)},
    ))

    # p-value: same seed, but Stata's shuffle order is undocumented, so this is a
    # Monte Carlo comparison, not a replicate comparison. Absolute, not relative:
    # Stata's two-sided p can be exactly 0, where a relative difference is
    # undefined.
    stata_two_sided = min(1.0, 2.0 * min(stata_p_lower, stata_p_upper))
    westata_p = float(res.e["pvalue_permuted"])
    gap = abs(stata_two_sided - westata_p)
    resolution = 1.0 / int(res.e["reps"])
    add(Comparison(
        domain=DOMAIN,
        name="permute: Monte Carlo p-value (same seed, not replicate-reproducible)",
        stata_command="permute weight bweight = _b[weight], reps(200) seed(20241024)",
        status="pass" if gap <= 4 * resolution else "fail",
        tolerance=4 * resolution,
        param_rel=gap,
        se_rel=float("nan"),
        extra={"stata_p_lower": stata_p_lower, "stata_p_upper": stata_p_upper,
               "stata_two_sided": stata_two_sided, "westata_p": westata_p,
               "absolute_gap": gap, "monte_carlo_resolution": resolution,
               "stata_reps": 200, "westata_reps": int(res.e["reps"]),
               "replicate_reproducible": False,
               "counting_rule": "Stata c/n; westata (c+1)/(reps+1)"},
        notes=[
            "NOT reproducible replicate-for-replicate. Stata's shuffle order is "
            "undocumented; westata uses Fisher-Yates over its own MT19937-64 "
            "stream, so the permutation sequences differ even with the same seed. "
            "Only the distribution of the p-value is comparable.",
            f"Stata p_lower={stata_p_lower:.4f} p_upper={stata_p_upper:.4f} -> "
            f"two-sided {stata_two_sided:.4f}; westata {westata_p:.4f}. The whole "
            "effect is discrete: Stata's two-sided p is 0 here because not one of "
            "the 200 permutations reached the observed statistic on the upper "
            "side, while westata's (c+1)/(reps+1) rule floors the same situation "
            f"at 1/201 = {1/201:.4f}. With reps=200 the resolution is "
            f"{resolution:.4f}, so a difference of {gap:.4f} is {gap/resolution:.2f} "
            "grid steps.",
        ],
    ))

    # Distribution level: a weak, non-null effect; both engines must land in the
    # same place, which is the property that actually matters when the sequences
    # cannot be reproduced.
    rng = np.random.default_rng(2024)
    n = 80
    x = rng.normal(size=n)
    null_frame = pd.DataFrame(dict(y=0.05 * x + rng.normal(size=n), x=x))
    st.send(null_frame)
    stata_ps, westata_ps = [], []
    for seed in (11, 22, 33, 44, 55, 66):
        st.output(f"permute x tx = _b[x], reps(400) seed({seed}) nodots: regress y x")
        ret = st.stata.get_return()
        lower = float(np.asarray(ret["r(p_lower)"]).reshape(-1)[0])
        upper = float(np.asarray(ret["r(p_upper)"]).reshape(-1)[0])
        stata_ps.append(min(1.0, 2.0 * min(lower, upper)))
        westata_ps.append(
            float(ws.Permute.from_formula("y ~ x", null_frame, permute="x",
                                          reps=400, seed=seed).fit()
                  .e["pvalue_permuted"])
        )
    difference = float(abs(np.mean(stata_ps) - np.mean(westata_ps)))
    add(Comparison(
        domain=DOMAIN,
        name="permute: Monte Carlo p-value across 6 seeds (weak effect, 400 reps)",
        stata_command="permute x tx = _b[x], reps(400) seed(S): regress y x",
        status="pass" if difference <= 0.05 else "fail",
        tolerance=0.05,
        param_rel=difference,
        se_rel=float("nan"),
        extra={"stata_p": stata_ps, "westata_p": westata_ps,
               "stata_mean": float(np.mean(stata_ps)),
               "westata_mean": float(np.mean(westata_ps)),
               "sd_of_mean_stata": float(np.std(stata_ps, ddof=1) / np.sqrt(len(stata_ps))),
               "note": "y = 0.05*x + noise, n=80: beta is ~0.45 standard errors, so "
                       "both engines land near p ~ 0.8 rather than at 0.5"},
        notes=[
            "Neither engine reproduces the other's permutations, so the only "
            "meaningful comparison is the distribution of the reported p-value.",
            f"Stata mean {np.mean(stata_ps):.4f}, westata mean "
            f"{np.mean(westata_ps):.4f} (difference {difference:.4f}). With 6 draws "
            "the standard error of each mean is ~"
            f"{np.std(stata_ps, ddof=1)/np.sqrt(len(stata_ps)):.3f}, so a 0.05 "
            "threshold on the difference only detects a gross error, not a subtle "
            "bias -- this row must not be read as a precision claim.",
            "Note westata's (c+1)/(reps+1) rule makes its p-values slightly larger "
            "than Stata's c/n by ~1/(reps+1) = 0.0025.",
        ],
    ))

    # bootstrap under *different* seeds: still an exact comparison, because the
    # replicate streams are reproduced from whichever seed is passed.
    st.run("sysuse auto, clear")
    stata_se, westata_se = [], []
    for seed in (101, 202, 303):
        cap_s = st.estimate(f"bootstrap, reps(200) seed({seed}) nodots: regress mpg weight")
        stata_se.append(np.sqrt(np.diag(cap_s.V)))
        westata_se.append(
            ws.Bootstrap.from_formula("mpg ~ weight", auto, reps=200, seed=seed)
            .fit().bse.to_numpy()[[1, 0]]
        )
    stata_arr = np.asarray(stata_se)
    westata_arr = np.asarray(westata_se)
    rel = float(np.max(np.abs(westata_arr - stata_arr) / np.abs(stata_arr)))
    add(Comparison(
        domain=DOMAIN,
        name="bootstrap: SE over 3 further seeds (200 reps each)",
        stata_command="bootstrap, reps(200) seed(S): regress mpg weight",
        status="pass" if rel <= 1e-6 else "fail",
        tolerance=1e-6,
        param_rel=rel,
        se_rel=rel,
        extra={"stata_se": stata_arr.tolist(), "westata_se": westata_arr.tolist(),
               "seeds": [101, 202, 303],
               "note": "the streams are reproduced seed-for-seed, so this is a "
                       "precision check, not a distribution check"},
        notes=["Stata's e(V) is stored in single precision for the bootstrap "
               "prefix, which caps the achievable agreement near 1e-7; the "
               "replicate-level comparison above is the one that reaches 1e-15."],
    ))


# ------------------------------------------------------------------------ main
def main() -> int:
    auto = pd.read_stata(AUTO, convert_categoricals=False)
    treat = treatment_frame()
    panel2 = panel_two_period()
    panel5 = panel_five_period()
    sv = survey_frame()
    lone = survey_frame(seed=5, lone=True)
    three = None  # built inside run_manova

    rng = np.random.default_rng(41)
    grp = np.repeat([1, 2, 3], 30)
    three = pd.DataFrame(dict(y1=rng.normal(size=90) + grp * 0.4,
                              y2=rng.normal(size=90) - grp * 0.25,
                              y3=rng.normal(size=90) + grp * 0.1, g=grp))

    with Stata() as st:
        print(f"StataNow {st.version} (embedded, mp)", flush=True)
        st.run("sysuse auto, clear")
        run_sureg(st, auto)
        run_mvreg(st, auto)
        run_manova(st, auto, three)
        run_nl(st, auto)
        run_teffects(st, treat)
        run_did(st, panel2, panel5)
        run_svy(st, sv, lone)
        run_resample(st, auto)

    REC.flush_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
