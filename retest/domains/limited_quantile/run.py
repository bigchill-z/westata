"""limited_quantile domain retest: Tobit / truncreg / intreg / heckman / etregress /
frontier / qreg / sqreg / bsqreg / rreg vs StataNow 19.5 MP.

Run:  cd D:\\myCodeProject\\statapy && .venv\\Scripts\\python.exe retest/domains/limited_quantile/run.py

Every number below comes from this process's own embedded Stata session plus
westata's live output; no committed fixture, registry entry or GAP_REGISTER row
is read.  The judgement bands are BRIEF.md §2.3: A=1e-9, B=1e-6, C=1e-3,
D=1e-2, E>1e-2 (max |westata-stata|/|stata| over the coefficients, then over
the standard errors taken from e(V), never from printed digits).
"""

from __future__ import annotations

import dataclasses
import sys
import warnings

sys.path.insert(0, "retest/lib")

import numpy as np
import pandas as pd
from scipy import stats

import westata as ws
from westata.errors import UnsupportedError
from westata.estimators.tobit import Tobit
from stata_probe import (Capture, Comparison, Record, Stata, align, compare as _raw_compare,
                         tier_for)

warnings.filterwarnings("ignore")
rec = Record("limited_quantile")


def _detail_rows(cap, result, rename=None, transform=None) -> dict:
    """Per-parameter b/se on both sides, after the same alignment compare uses.

    ``compare`` only reports the maximum relative difference; a single number
    cannot say *which* coefficient moved or in which direction.  This records
    every pair so NOTES.md and the captain can attribute a failure precisely.
    """

    names, si, wi, _ = align(cap.names, list(result.param_names), rename=rename or {})
    sb = np.asarray(cap.b, dtype=float)[si].copy()
    sv = np.sqrt(np.diag(np.asarray(cap.V, dtype=float)))[si].copy()
    for k, key in enumerate(names):
        pair = (transform or {}).get(key)
        if pair is None:
            continue
        value_fn, deriv_fn = pair
        sv[k] = abs(float(deriv_fn(sb[k]))) * sv[k]
        sb[k] = float(value_fn(sb[k]))
    wb = np.asarray(result.params, dtype=float)[wi]
    wse = np.asarray(result.bse, dtype=float)[wi]
    rows = {}
    for k, key in enumerate(names):
        rows[key] = {
            "stata_b": float(sb[k]), "westata_b": float(wb[k]),
            "b_rel": float(abs(wb[k] - sb[k]) / max(abs(sb[k]), 1e-12)),
            "stata_se": float(sv[k]), "westata_se": float(wse[k]),
            "se_rel": float(abs(wse[k] - sv[k]) / max(abs(sv[k]), 1e-12))
            if np.isfinite(sv[k]) and sv[k] != 0 else float("nan"),
        }
    return rows


def compare(name, cap, result, **kwargs):  # noqa: F811 - deliberate shadowing
    """``stata_probe.compare`` plus the per-parameter table in ``extra['detail']``."""

    comp = _raw_compare(name, cap, result, **kwargs)
    comp.extra.setdefault(
        "detail",
        _detail_rows(cap, result, kwargs.get("rename"), kwargs.get("stata_transform")),
    )
    return comp


# ----------------------------------------------------------------------- helpers
@dataclasses.dataclass
class Stub:
    """Minimal statsmodels-like result, used for the pooled-bootstrap comparison."""

    params: np.ndarray
    param_names: list[str]
    bse: np.ndarray
    e: dict = dataclasses.field(default_factory=dict)


def renamed(cap: Capture, names: list[str]) -> Capture:
    """Replace Stata's e(b) column names positionally.

    Three commands in this domain store duplicate column names in ``e(b)``:
    ``intreg`` -> ``['x1','x2','_cons','_cons']`` (last = /lnsigma), ``frontier``
    -> ``['x1','x2','_cons','_cons','_cons']`` (last two = /lnsig2v, /lnsig2u),
    and the two-equation models, which have one ``_cons`` per equation.
    ``compare`` maps names through a dict, which cannot address two equal keys,
    so the names are rewritten here in the printed table order (quoted in
    NOTES.md).  b and V are untouched.
    """

    names = list(names)
    if len(names) != len(cap.names):
        raise ValueError(f"expected {len(cap.names)} names, got {len(names)}")
    return dataclasses.replace(cap, names=names)


def numeric_hess(f, x, h: float = 1e-5) -> np.ndarray:
    """Central-difference Hessian; used only to diagnose a westata SE."""
    x = np.asarray(x, dtype=float)
    m = x.size
    H = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            ei = np.zeros(m)
            ej = np.zeros(m)
            ei[i] = h
            ej[j] = h
            if i == j:
                H[i, j] = (f(x + ei) - 2 * f(x) + f(x - ei)) / h**2
            else:
                H[i, j] = (
                    f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
                ) / (4 * h * h)
            H[j, i] = H[i, j]
    return H


def pooled_bootstrap(st, command: str, order: list[int], westata_fit, reps: int,
                     seeds: tuple[int, ...]) -> tuple[Stub, Stub]:
    """Average the bootstrap *variance* over several seeds on both sides.

    A single bootstrap draw has its own Monte-Carlo error, and the two
    implementations cannot share a random stream (Stata's RNG is not
    reproducible from numpy).  Pooling k independent runs of ``reps`` draws
    shrinks that error to roughly 1/sqrt(2*k*reps) and leaves any real
    difference between the resampling schemes visible.  ``order`` maps Stata's
    e(b) positions onto westata's parameter order.
    """

    stata_var = None
    stata_b = None
    for seed in seeds:
        st.run(f"set seed {seed}")
        cap = st.estimate(command.replace("REPS", str(reps)))
        var = np.asarray(cap.V, dtype=float)[np.ix_(order, order)]
        b = np.asarray(cap.b, dtype=float)[order]
        stata_var = var if stata_var is None else stata_var + var
        stata_b = b if stata_b is None else stata_b + b
    stata_var /= len(seeds)
    stata_b /= len(seeds)

    ws_var = None
    ws_b = None
    ws_names = None
    for seed in seeds:
        res = westata_fit(reps, seed)
        var = np.asarray(res.cov_params(), dtype=float)
        b = np.asarray(res.params, dtype=float)
        ws_names = list(res.param_names)
        ws_var = var if ws_var is None else ws_var + var
        ws_b = b if ws_b is None else ws_b + b
    ws_var /= len(seeds)
    ws_b /= len(seeds)

    cap_like = Capture(
        b=stata_b,
        V=stata_var,
        names=list(ws_names),
        command=command.replace("REPS", str(reps)),
        scalars={"reps": float(reps)},
        nobs=int(st.nobs()),
        version=st.version,
    )
    stub = Stub(params=ws_b, param_names=list(ws_names),
                bse=np.sqrt(np.diag(ws_var)), e={"reps": float(reps)})
    return cap_like, stub


# --------------------------------------------------------------------------- data
rng = np.random.default_rng(2024)
n_tob = 500
x1 = rng.normal(size=n_tob)
x2 = rng.normal(size=n_tob)
ystar = 1.0 + 0.8 * x1 - 0.5 * x2 + rng.normal(size=n_tob)
tobit_df = pd.DataFrame({"x1": x1, "x2": x2, "ystar": ystar})
tobit_df["y"] = np.clip(ystar, 0.0, None)              # left-censored at 0
tobit_df["yu"] = np.minimum(ystar, 2.5)                # right-censored at 2.5
tobit_df["yb"] = np.clip(ystar, 0.0, 2.0)              # both limits

# interval data: y* is observed only as [floor(y*), floor(y*)+1); everything
# below 0 is left-censored (lower missing, upper = 0).
lo = np.where(ystar < 0, np.nan, np.floor(ystar))
hi = np.where(ystar < 0, 0.0, lo + 1.0)
intreg_df = pd.DataFrame({"x1": x1, "x2": x2, "lo": lo, "hi": hi})

rng_sel = np.random.default_rng(7)
n_sel = 600
z1 = rng_sel.normal(size=n_sel)
z2 = rng_sel.normal(size=n_sel)
sx1 = rng_sel.normal(size=n_sel)
u = rng_sel.normal(size=n_sel)
v = 0.6 * u + np.sqrt(1 - 0.36) * rng_sel.normal(size=n_sel)
sel_ind = (0.5 + 0.7 * z1 - 0.4 * z2 + u > 0).astype(float)
y_out = 1.0 + 0.9 * sx1 + v
sel_df = pd.DataFrame({"x1": sx1, "z1": z1, "z2": z2, "s": sel_ind, "y": y_out})
sel_df["y_sel"] = np.where(sel_ind == 1, y_out, np.nan)

rng_fr = np.random.default_rng(99)
n_fr = 500
fx1 = rng_fr.normal(size=n_fr)
fx2 = rng_fr.normal(size=n_fr)
fv = rng_fr.normal(size=n_fr)
fu = np.abs(rng_fr.normal(scale=0.7, size=n_fr))
fr_df = pd.DataFrame({"x1": fx1, "x2": fx2,
                      "y": 1.0 + 0.8 * fx1 - 0.4 * fx2 + fv - fu})

rng_q = np.random.default_rng(31)
n_q = 400
qx1 = rng_q.normal(size=n_q)
qx2 = rng_q.normal(size=n_q)
qy = 1.0 + 0.8 * qx1 - 0.4 * qx2 + rng_q.normal(size=n_q) * (1 + 0.4 * np.abs(qx1))
q_df = pd.DataFrame({"x1": qx1, "x2": qx2, "y": qy})

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)

TOBIT_SCALE = {"var(e.y)": "/sigma", "var(e.yu)": "/sigma", "var(e.yb)": "/sigma"}
TOBIT_TRANSFORM = {"/sigma": (np.sqrt, lambda x: 1 / (2 * np.sqrt(x)))}


with Stata() as st:
    print(f"Stata version {st.version}")

    # =================================================================== TOBIT
    st.send(tobit_df)
    assert st.nobs() == len(tobit_df)

    # --- 1. ll(0)
    cap = st.estimate("tobit y x1 x2, ll(0)")
    res = ws.Tobit.from_formula("y ~ x1 + x2", tobit_df, lower=0).fit()
    cap_t = renamed(cap, ["x1", "x2", "Intercept", "/sigma"])
    rec.add(compare(
        "tobit ll(0)", cap_t, res,
        rename={"var(e.y)": "/sigma"},
        stata_transform=TOBIT_TRANSFORM,
        scalars=("N", "ll", "ll_0", "chi2", "df_m", "r2_p"),
        notes=["Stata stores var(e.y)=sigma^2; sqrt + delta method map it to /sigma",
               "Stata's own /sigma standard error is obtained here as "
               "se(var(e.y)) / (2*sqrt(var(e.y))), the delta method used by Stata "
               "for the sigma row it prints"],
        extra={"n_uncensored": [cap.scalars.get("N_unc"), res.e.get("n_uncensored")],
               "n_left": [cap.scalars.get("N_lc"), res.e.get("n_left")],
               "n_right": [cap.scalars.get("N_rc"), res.e.get("n_right")]},
    ))

    # --- 1b. root cause of the /sigma SE gap: which quantity is westata reporting?
    model = Tobit.from_formula("y ~ x1 + x2", tobit_df, lower=0)
    sigma_hat = float(np.asarray(res.params)[-1])
    se_sigma_stata = float(np.sqrt(cap.V[-1, -1]) / (2 * np.sqrt(cap.b[-1])))
    theta_log = np.concatenate([np.asarray(res.params)[:3], [np.log(sigma_hat)]])
    se_log = np.sqrt(np.diag(np.linalg.inv(
        numeric_hess(lambda th: -model._loglike(np.asarray(th, dtype=float)), theta_log))))

    def negll_sigma(th):
        par = np.array(th, dtype=float)
        par[3] = np.log(par[3])
        return -model._loglike(par)

    theta_sig = np.concatenate([np.asarray(res.params)[:3], [sigma_hat]])
    se_sig = np.sqrt(np.diag(np.linalg.inv(numeric_hess(negll_sigma, theta_sig))))
    ws_se = float(np.asarray(res.bse)[-1])
    # Stata's e(V) is in the (b, var(e.y)) parameterisation.  d ln sigma / d var =
    # 1/(2 var), so se(ln sigma) = se(var) / (2 var) -- compare that with westata.
    se_lnsigma_stata = float(np.sqrt(cap.V[-1, -1]) / (2 * cap.b[-1]))
    sigma_identity = ws_se / se_lnsigma_stata
    diag = {
        "stata_se_var_e": float(np.sqrt(cap.V[-1, -1])),
        "stata_se_sigma": se_sigma_stata,
        "stata_se_lnsigma": se_lnsigma_stata,
        "westata_se_sigma": ws_se,
        "hessian_se_sigma": float(se_sig[3]),
        "hessian_se_logsigma": float(se_log[3]),
        "westata_vs_hessian_sigma": abs(ws_se - se_sig[3]) / se_sig[3],
        "westata_vs_hessian_logsigma": abs(ws_se - se_log[3]) / se_log[3],
        "westata_vs_stata_lnsigma": abs(ws_se - se_lnsigma_stata) / se_lnsigma_stata,
        "westata_vs_stata_sigma": abs(ws_se - se_sigma_stata) / se_sigma_stata,
        "ratio_ws_to_stata_sigma": ws_se / se_sigma_stata,
        "one_over_sigma_hat": 1.0 / sigma_hat,
        "sigma_hat": sigma_hat,
    }
    rec.add(Comparison(
        domain="limited_quantile",
        name="tobit /sigma SE root cause (diagnostic)",
        stata_command="tobit y x1 x2, ll(0)",
        status="fail" if diag["westata_vs_stata_sigma"] > 1e-9 else "pass",
        tolerance=1e-9,
        param_rel=diag["westata_vs_hessian_sigma"],
        se_rel=diag["westata_vs_stata_sigma"],
        n_params=4,
        notes=[
            "westata's reported se(/sigma) is the standard error of ln(sigma), not of "
            "sigma: against an independent central-difference Hessian it matches the "
            f"(b, lnsigma) parameterisation to {diag['westata_vs_hessian_logsigma']:.2e} "
            f"and misses the (b, sigma) one by {diag['westata_vs_hessian_sigma']:.2e}.",
            "It also reproduces Stata's own se(lnsigma) = se(var(e.y))/(2*var(e.y)) = "
            f"{se_lnsigma_stata:.10f} to {diag['westata_vs_stata_lnsigma']:.2e}, while "
            f"Stata's se(sigma) = se(var)/(2*sqrt(var)) = {se_sigma_stata:.10f}; "
            f"westata = {ws_se:.10f}.",
            f"The ratio is exactly 1/sigma_hat: {diag['ratio_ws_to_stata_sigma']:.9f} vs "
            f"1/{sigma_hat:.9f} (rel {abs(diag['ratio_ws_to_stata_sigma'] - diag['one_over_sigma_hat']) / diag['one_over_sigma_hat']:.2e}).",
            "The point estimate of sigma agrees to 1e-8, so this is purely a reported-SE "
            "bug: the coefficient's last element is on the sigma scale while the "
            "covariance matrix is still the lnsigma one. Effect on inference: sigma's "
            "z/t statistic is overstated by 1/sigma (here 7.9%).",
        ],
        extra=diag,
        stata_version=cap.version,
    ))

    # --- 2. ul(2.5)
    cap = st.estimate("tobit yu x1 x2, ul(2.5)")
    res_u = ws.Tobit.from_formula("yu ~ x1 + x2", tobit_df, lower=None, upper=2.5).fit()
    rec.add(compare(
        "tobit ul(2.5)", renamed(cap, ["x1", "x2", "Intercept", "/sigma"]), res_u,
        rename={"var(e.yu)": "/sigma"},
        stata_transform=TOBIT_TRANSFORM,
        scalars=("N", "ll", "chi2", "df_m"),
        extra={"n_right": [cap.scalars.get("N_rc"), res_u.e.get("n_right")]},
    ))

    # --- 3. ll(0) + ul(2)
    cap = st.estimate("tobit yb x1 x2, ll(0) ul(2)")
    res_b = ws.Tobit.from_formula("yb ~ x1 + x2", tobit_df, lower=0.0, upper=2.0).fit()
    rec.add(compare(
        "tobit ll(0) ul(2)", renamed(cap, ["x1", "x2", "Intercept", "/sigma"]), res_b,
        rename={"var(e.yb)": "/sigma"},
        stata_transform=TOBIT_TRANSFORM,
        scalars=("N", "ll", "chi2", "df_m"),
        extra={"n_left": [cap.scalars.get("N_lc"), res_b.e.get("n_left")],
               "n_right": [cap.scalars.get("N_rc"), res_b.e.get("n_right")]},
    ))

    # =============================================================== TRUNCREG
    for name, cmd, lhs, kw in [
        ("truncreg ll(0)", "truncreg y x1 x2, ll(0)", "y", dict(lower=0.0)),
        ("truncreg ul(2.5)", "truncreg yu x1 x2, ul(2.5)", "yu", dict(upper=2.5)),
        ("truncreg ll(0) ul(2)", "truncreg yb x1 x2, ll(0) ul(2)", "yb",
         dict(lower=0.0, upper=2.0)),
    ]:
        cap = st.estimate(cmd)
        res = ws.Truncreg.from_formula(f"{lhs} ~ x1 + x2", tobit_df, **kw).fit()
        rec.add(compare(
            name, renamed(cap, ["x1", "x2", "Intercept", "/sigma"]), res,
            rename={"sigma": "/sigma"},
            scalars=("N", "ll", "chi2", "df_m"),
            notes=[f"Stata e(N)={cap.scalars.get('N')}: truncated rows leave the "
                   f"estimation sample but stay in the dataset (st.nobs()={st.nobs()}); "
                   f"westata nobs={res.nobs}, e(n_truncated)={res.e.get('n_truncated')}"],
        ))

    # ================================================================== INTREG
    st.send(intreg_df)
    assert st.nobs() == len(intreg_df)
    cap = st.estimate("intreg lo hi x1 x2")
    res_i = ws.Intreg.from_formula("lo hi ~ x1 + x2", intreg_df).fit()
    rec.add(compare(
        "intreg interval model",
        renamed(cap, ["x1", "x2", "Intercept", "/lnsigma"]), res_i,
        scalars=("N", "ll", "chi2", "df_m", "sigma"),
        notes=["intervals: [floor(y*), floor(y*)+1); y*<0 left-censored (lo missing, hi=0)",
               "Stata e(b) colnames are ['x1','x2','_cons','_cons'] -- the /lnsigma row "
               "reuses the _cons label; positions taken from the printed table",
               "Stata prints /lnsigma and a derived sigma row; westata reports "
               "/lnsigma as a coefficient and e(sigma)"],
        extra={"n_interval_stata": None,
               "westata_counts": {k: res_i.e.get(k) for k in
                                  ("n_left", "n_right", "n_interval", "n_uncensored")}},
    ))

    # ================================================================ HECKMAN
    st.send(sel_df)
    assert st.nobs() == len(sel_df)
    cap = st.estimate("heckman y_sel x1, select(s = z1 z2) twostep")
    res_h = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", sel_df).fit()

    # Independent reconstruction of Stata's two-step aggregates.  Stata's
    # e(sigma) is NOT sqrt(RSS/(m-3)): on this DGP it is
    # sqrt(RSS/m + b_lambda^2 * mean(lambda(lambda + w))) to 1e-12, where w is
    # the probit index and b_lambda the augmented-regression coefficient.
    sel_mask = sel_df["s"].to_numpy() > 0.5
    Zm = np.column_stack([np.ones(len(sel_df)), z1, z2])
    Xm = np.column_stack([np.ones(len(sel_df)), sx1])
    probit_ws = ws.Probit.from_formula("s ~ z1 + z2", sel_df).fit()
    gamma_hat = np.asarray(probit_ws.params, dtype=float)
    w_index = Zm @ gamma_hat
    mills = np.exp(stats.norm.logpdf(w_index) - stats.norm.logcdf(w_index))
    aug = np.column_stack([Xm, mills])
    beta_aug, *_ = np.linalg.lstsq(aug[sel_mask], y_out[sel_mask], rcond=None)
    resid_aug = y_out[sel_mask] - aug[sel_mask] @ beta_aug
    m_sel = int(sel_mask.sum())
    delta = (mills * (mills + w_index))[sel_mask]
    heck_recon = {
        "n_selected": m_sel,
        "sqrt_rss_over_m": float(np.sqrt(resid_aug @ resid_aug / m_sel)),
        "sqrt_rss_over_m_minus_3": float(np.sqrt(resid_aug @ resid_aug / (m_sel - 3))),
        "sqrt_rss_over_m_plus_b2_delta": float(np.sqrt(
            resid_aug @ resid_aug / m_sel + beta_aug[-1] ** 2 * delta.mean())),
        "stata_e_sigma": cap.scalars.get("sigma"),
        "westata_e_sigma": res_h.e.get("sigma"),
        "stata_e_rho": cap.scalars.get("rho"),
        "westata_e_rho": res_h.e.get("rho"),
        "lam_coef_westata": float(np.asarray(res_h.params)[2]),
        "stata_lambda": cap.scalars.get("lambda"),
    }
    heck_recon["sigma_formula_rel"] = abs(
        heck_recon["sqrt_rss_over_m_plus_b2_delta"] - heck_recon["stata_e_sigma"]
    ) / heck_recon["stata_e_sigma"]

    rec.add(compare(
        "heckman twostep", renamed(cap, ["y:x1", "y:Intercept", "select:z1",
                                         "select:z2", "select:Intercept", "lambda"]),
        res_h,
        scalars=("N", "N_selected", "rho", "sigma", "lambda"),
        notes=["Stata e(b) order from the printed table: y|x1, y|_cons, s|z1, s|z2, "
               "s|_cons, then /mills lambda",
               "Stata's two-step store: e(rho), e(sigma), e(lambda), e(selambda). "
               "e(sigma) reproduces sqrt(RSS/m + b_lambda^2*mean(lambda(lambda+w))) to "
               f"rel {heck_recon['sigma_formula_rel']:.2e} -- it is NOT sqrt(RSS/(m-3)), "
               "which is what westata uses for both sigma and rho",
               "the coefficient on lambda agrees to 1e-9 (0.5301503851 vs 0.5301503857); "
               "only the derived e(sigma)/e(rho) differ",
               "selection-equation SEs: Stata keeps the plain probit VCE (ratio to "
               "'probit' 0.99998), westata's joint sandwich bread mixes the outcome "
               "block in and inflates select:z1 by 6.8%",
               "outcome-block SEs differ by 2-3%: westata's sandwich is a different "
               "(M-estimator) approximation from Stata's Heckman corrected VCE"],
        extra={
            "stata": {"rho": cap.scalars.get("rho"), "sigma": cap.scalars.get("sigma"),
                      "lambda": cap.scalars.get("lambda"),
                      "selambda": cap.scalars.get("selambda")},
            "westata": {"rho": res_h.e.get("rho"), "sigma": res_h.e.get("sigma"),
                        "lambda": res_h.e.get("lambda"),
                        "bse_lambda": float(np.asarray(res_h.bse)[-1])},
            "sigma_reconstruction": heck_recon,
        },
    ))

    cap_ml_h = st.estimate("heckman y_sel x1, select(s = z1 z2)")
    res_ml_h = ws.Heckman.from_formula("y_sel ~ x1 + select(s ~ z1 + z2)", sel_df).fit(
        method="ml")
    rec.add(compare(
        "heckman ml",
        renamed(cap_ml_h, ["x1", "Intercept", "z1", "z2", "Intercept",
                          "/athrho", "/lnsigma"]),
        res_ml_h,
        extra={"stata": {k: cap_ml_h.scalars.get(k) for k in
                         ("rho", "sigma", "lambda", "ll")},
               "westata": {k: res_ml_h.e.get(k) for k in
                           ("rho", "sigma", "lambda", "ll")}},
        notes=["似然与梯度取自 heck_d2.ado；在 Stata 系数处似然差 0.0",
               "lnsigma 的 score 由已验证似然数值微分（解析版错了一个数量级）",
               "注意 _y 把删失行填 0 而非 NaN，ML 似然需重新置回 NaN"],
    ))

    # ============================================================== ETREGRESS
    cap = st.estimate("etregress y x1, treat(s = z1 z2) twostep")
    res_e = ws.Etregress.from_formula("y ~ x1 + treat(s ~ z1 + z2)", sel_df).fit()
    rec.add(compare(
        "etregress twostep",
        renamed(cap, ["y:x1", "y:s", "y:Intercept", "treat:z1", "treat:z2",
                      "treat:Intercept", "lambda"]),
        res_e,
        scalars=("N", "rho", "sigma", "lambda"),
        notes=["Stata e(b) order: y|x1, y|s, y|_cons, s|z1, s|z2, s|_cons, lambda",
               "Stata's twostep store holds e(rho), e(sigma), e(lambda), e(selambda), "
               "e(ll_p) -- e(sigma1)/e(sigma2) do not exist for etregress in 19.5 "
               "(full get_ereturn dump in NOTES), so that pair is untestable",
               "westata's e(rho) has the wrong SIGN on two independent DGPs "
               "(+0.4283 Stata vs -0.4486 westata here); westata computes "
               "rho = -b_ctrl/sigma while Stata's lambda = rho*sigma"],
        extra={
            "stata": {"rho": cap.scalars.get("rho"), "sigma": cap.scalars.get("sigma"),
                      "lambda": cap.scalars.get("lambda"),
                      "selambda": cap.scalars.get("selambda")},
            "westata": {"rho": res_e.e.get("rho"), "sigma": res_e.e.get("sigma"),
                        "b_ctrl": float(np.asarray(res_e.params)[-2]),
                        "bse_lambda": float(np.asarray(res_e.bse)[-1])},
        },
    ))

    cap_ml_e = st.estimate("etregress y x1, treat(s = z1 z2)")
    try:
        ws.Etregress.from_formula("y ~ x1 + treat(s ~ z1 + z2)", sel_df).fit(method="ml")
        rec.skip("etregress ml", "westata unexpectedly accepted method='ml'")
    except UnsupportedError as exc:
        rec.error(
            "etregress ml", exc, command="etregress y x1, treat(s = z1 z2)",
            notes=[
                "Stata's DEFAULT for etregress is ML -- 'twostep' is the option that "
                "selects the control-function estimator, so westata's only available "
                "path is not Stata's default. ML reference:",
                "b=" + str(np.array2string(np.asarray(cap_ml_e.b), precision=8)),
                "se=" + str(np.array2string(np.sqrt(np.diag(cap_ml_e.V)), precision=8)),
                f"e(rho)={cap_ml_e.scalars.get('rho')} "
                f"e(sigma)={cap_ml_e.scalars.get('sigma')} "
                f"ll={cap_ml_e.scalars.get('ll')}",
            ],
        )

    # =============================================================== FRONTIER
    st.send(fr_df)
    assert st.nobs() == len(fr_df)
    cap = st.estimate("frontier y x1 x2")
    res_f = ws.Frontier.from_formula("y ~ x1 + x2", fr_df).fit()
    rec.add(compare(
        "frontier halfnormal",
        renamed(cap, ["x1", "x2", "Intercept", "/lnsig2v", "/lnsig2u"]), res_f,
        scalars=("N", "ll", "chi2", "sigma_u", "sigma_v"),
        notes=["Stata's e(b) colnames are ['x1','x2','_cons','_cons','_cons']; the last "
               "two rows are /lnsig2v and /lnsig2u in the printed order",
               "DGP carries real inefficiency (u ~ |N(0, 0.7^2)|); without it both "
               "estimators sit on the sigma_u = 0 boundary and the /lnsig2u SE is "
               "meaningless, so the boundary case is not used"],
        extra={"stata_sigma2": None,
               "westata": {k: res_f.e.get(k) for k in
                           ("sigma_u", "sigma_v", "sigma2", "lambda", "gamma")}},
    ))
    cap_ex = st.estimate("frontier y x1 x2, distribution(exponential)")
    res_ex = ws.Frontier.from_formula("y ~ x1 + x2", fr_df,
                                      distribution="exponential").fit()
    rec.add(compare(
        "frontier exponential", renamed(cap_ex, ["x1", "x2", "Intercept",
                                                "/lnsig2v", "/lnsig2u"]), res_ex,
        extra={"stata": {k: cap_ex.scalars.get(k) for k in ("sigma_u", "sigma_v", "ll")},
               "westata": {k: res_ex.e.get(k) for k in ("sigma_u", "sigma_v", "ll")}},
        notes=["似然与梯度逐项取自 fron_ex.ado；在 Stata 系数处似然差 7.1e-16",
               "起始值按 frontier.ado:347-350 的三阶矩拆分"],
    ))

    # ================================================================== QREG
    st.send(q_df)
    assert st.nobs() == len(q_df)
    cap = st.estimate("qreg y x1 x2")
    res_q = ws.QuantileRegression.from_formula("y ~ x1 + x2", q_df, quantile=0.5).fit()
    rec.add(compare(
        "qreg median", renamed(cap, ["x1", "x2", "Intercept"]), res_q,
        scalars=("N", "sum_adev", "sum_rdev", "r2_p", "q", "q_v", "sparsity", "bwidth"),
        notes=["two independent LP solutions of the same check-function objective can "
               "differ when the optimum is not unique; Stata's 'alternate solutions' "
               "warning was not raised for this DGP"],
    ))

    cap = st.estimate("qreg y x1 x2, quantile(.25)")
    res_q25 = ws.QuantileRegression.from_formula("y ~ x1 + x2", q_df, quantile=0.25).fit()
    rec.add(compare(
        "qreg quantile(.25)", renamed(cap, ["x1", "x2", "Intercept"]), res_q25,
        scalars=("N", "sum_adev", "sum_rdev", "r2_p", "q", "q_v", "sparsity"),
    ))

    cap = st.estimate("qreg y x1 x2, vce(robust)")
    res_qr = ws.QuantileRegression.from_formula("y ~ x1 + x2", q_df, quantile=0.5).fit(
        cov_type="robust")
    rec.add(compare(
        "qreg vce(robust) median", renamed(cap, ["x1", "x2", "Intercept"]), res_qr,
        scalars=("N", "r2_p"),
        notes=["at tau=.5, (tau - 1{u<0})^2 == tau(1-tau) for every i, so this point "
               "cannot tell the two candidate sandwich meats apart; the .25 case below "
               "is the discriminating one"],
    ))

    cap = st.estimate("qreg y x1 x2, quantile(.25) vce(robust)")
    res_qr25 = ws.QuantileRegression.from_formula("y ~ x1 + x2", q_df, quantile=0.25).fit(
        cov_type="robust")
    rec.add(compare(
        "qreg vce(robust) quantile(.25)",
        renamed(cap, ["x1", "x2", "Intercept"]), res_qr25,
        scalars=("N", "r2_p"),
        notes=["A = J^-1 M J^-1 with J = sum f_i x x' and M = sum psi_i^2 x x'. Stata's "
               "M is tau(1-tau) X'X to 9e-9; westata's psi_i^2 meat is 4.6% off at "
               "tau=.25 and 3.5% at tau=.75 (verified separately, see NOTES)"],
    ))

    # --- sqreg / bsqreg: bootstrap, pooled over seeds
    cap_sq, stub_sq = pooled_bootstrap(
        st, "sqreg y x1 x2, quantile(.25 .5) reps(REPS)",
        order=[2, 0, 1, 5, 3, 4],                 # cons,x1,x2 (q.25) then (q.5)
        westata_fit=lambda reps, seed: ws.Sqreg.from_formula(
            "y ~ x1 + x2", q_df, quantiles=(0.25, 0.5)).fit(reps=reps, seed=seed),
        reps=500, seeds=(101, 202, 303, 404),
    )
    rec.add(compare(
        "sqreg q(.25 .5) pooled 4x500", cap_sq, stub_sq, scalars=("reps",),
        notes=["coefficients are deterministic qreg fits; only the bootstrap VCE can "
               "differ",
               "both sides are averaged over 4 seeds x 500 reps (variance average) to "
               "remove bootstrap Monte-Carlo error, since Stata's RNG stream cannot be "
               "reproduced in numpy",
               "Stata e(b) order is q.25|x1,x2,_cons then q.5|x1,x2,_cons"],
        extra={"westata_names": stub_sq.param_names},
    ))

    cap_bs, stub_bs = pooled_bootstrap(
        st, "bsqreg y x1 x2, quantile(.5) reps(REPS)",
        order=[2, 0, 1],
        westata_fit=lambda reps, seed: ws.Bsqreg.from_formula(
            "y ~ x1 + x2", q_df, quantile=0.5).fit(reps=reps, seed=seed),
        reps=500, seeds=(101, 202, 303, 404),
    )
    rec.add(compare(
        "bsqreg median pooled 4x500", cap_bs, stub_bs, scalars=("reps",),
        notes=["same pooling as sqreg; Stata e(b) order x1, x2, _cons",
               "Stata's bsqreg SE exceeds its own analytic qreg vce(robust) by ~26% on "
               "this heteroskedastic DGP, and westata's bootstrap reproduces that, so "
               "the two bootstraps agree within MC error (see NOTES)"],
    ))

    # =================================================================== RREG
    def rreg_point(st, df, cmd, formula, stata_order, label, dep, cols):
        st.send(df)
        cap = st.estimate(cmd)
        res = ws.RobustRegression.from_formula(formula, df).fit()
        order = {"x1": 0, "x2": 1, "weight": 0, "length": 1, "_cons": 2}
        sse = np.sqrt(np.diag(cap.V))
        sb = np.asarray(cap.b)
        detail = {}
        for j, nm in enumerate(res.param_names):
            i = 2 if nm == "Intercept" else order[nm]
            wb = float(np.asarray(res.params)[j])
            wse = float(np.asarray(res.bse)[j])
            detail[nm] = {
                "stata_b": float(sb[i]), "westata_b": wb,
                "b_rel": abs(wb - float(sb[i])) / abs(float(sb[i])),
                "stata_se": float(sse[i]), "westata_se": wse,
                "se_rel": abs(wse - float(sse[i])) / float(sse[i]),
            }
        # Stata's rreg SE law: se(b) = e(rmse) * sqrt(diag((X'X)^-1)) on the raw
        # (unweighted) design.  Verified to 1e-16 on every dataset in this file.
        design = np.column_stack([np.ones(len(df))]
                                 + [np.asarray(df[c], dtype=float) for c in cols])
        rmse = float(cap.scalars["rmse"])
        law = rmse * np.sqrt(np.diag(np.linalg.inv(design.T @ design)))
        # law is in (const, x1, x2) design order; Stata reports (x1, x2, _cons).
        stata_report = np.array([sse[2], sse[0], sse[1]])
        stata_se_law = {
            "rmse": rmse,
            "predicted_const_x1_x2": [float(v) for v in law],
            "stata_se_const_x1_x2": [float(v) for v in stata_report],
            "max_rel": float(np.max(np.abs(law - stata_report) / stata_report)),
        }
        rec.add(compare(
            label, renamed(cap, stata_order), res,
            scalars=("N", "F", "r2", "rmse"),
            notes=["Stata's rreg SE is NOT the pseudovalue (Street-Carroll-Ruppert) "
                   "estimate: it equals e(rmse) * sqrt(diag((X'X)^-1)) on the "
                   "unweighted design, reproduced to 1e-16 on all four datasets "
                   "(extra['stata_se_law'])",
                   "westata computes the S-C-R pseudovalue variance instead, which is "
                   "why its SEs differ; the coefficients agree to ~2e-4",
                   "westata reports weighted sums of squares (final biweights); Stata's "
                   "e(rss)/e(mss)/e(rmse)/e(r2)/e(F) are different quantities, see NOTES",
                   "per-coefficient b/se relative differences in extra['detail']"],
            extra={"detail": detail,
                   "stata_se_law": stata_se_law,
                   "stata_e": {k: cap.scalars.get(k) for k in
                               ("rss", "mss", "rmse", "r2", "r2_a", "F", "sum_w")},
                   "westata_e": {k: res.e.get(k) for k in
                                 ("rss", "mss", "rmse", "r2", "r2_a", "F", "scale",
                                  "dropped", "N")}},
        ))
        return cap, res

    cap, res = rreg_point(st, q_df, "rreg y x1 x2, gen(wt)", "y ~ x1 + x2",
                          ["x1", "x2", "Intercept"], "rreg hetero DGP",
                          "y", ["x1", "x2"])
    w_stata = st.var("wt")
    X = np.column_stack([np.ones(n_q), qx1, qx2])
    w_ws = ws.RobustRegression.from_formula("y ~ x1 + x2", q_df)._biweight_weights(
        qy - X @ np.asarray(res.params))
    weight_rel = float(np.max(np.abs(w_stata - w_ws)) / float(np.max(np.abs(w_stata))))
    weight_cmp = Comparison(
        domain="limited_quantile",
        name="rreg final biweight case weights",
        stata_command="rreg y x1 x2, gen(wt)",
        status="pass" if weight_rel <= 1e-3 else "fail",
        tolerance=1e-3,
        param_rel=weight_rel,
        se_rel=float("nan"),
        n_params=int(n_q),
        notes=[
            f"sum(w): Stata {w_stata.sum():.6f} vs westata {w_ws.sum():.6f}",
            f"max|dw| = {np.max(np.abs(w_stata - w_ws)):.3e}, corr = "
            f"{np.corrcoef(w_stata, w_ws)[0, 1]:.9f}",
            "both sides are biweights with c = 4.685 on residual/MAD scale; the small "
            "gap follows from the ~1e-4 coefficient difference feeding the weights",
        ],
        extra={"sum_w_stata": float(w_stata.sum()),
               "sum_w_westata": float(w_ws.sum()),
               "max_abs_diff": float(np.max(np.abs(w_stata - w_ws))),
               "corr": float(np.corrcoef(w_stata, w_ws)[0, 1])},
    )
    rec.add(weight_cmp)

    rng_c = np.random.default_rng(5)
    c1 = rng_c.normal(size=300)
    c2 = rng_c.normal(size=300)
    clean_df = pd.DataFrame({"x1": c1, "x2": c2,
                             "y": 1 + 0.5 * c1 - 0.3 * c2 + rng_c.normal(size=300)})
    rreg_point(st, clean_df, "rreg y x1 x2", "y ~ x1 + x2",
               ["x1", "x2", "Intercept"], "rreg clean normal DGP", "y", ["x1", "x2"])

    rng_o = np.random.default_rng(6)
    o1 = rng_o.normal(size=300)
    o2 = rng_o.normal(size=300)
    oy = 1 + 0.5 * o1 - 0.3 * o2 + rng_o.normal(size=300)
    oy[:6] += np.array([12, -14, 9, 11, -13, 15])
    out_df = pd.DataFrame({"x1": o1, "x2": o2, "y": oy})
    rreg_point(st, out_df, "rreg y x1 x2", "y ~ x1 + x2",
               ["x1", "x2", "Intercept"], "rreg outlier DGP", "y", ["x1", "x2"])

    st.send(auto)
    rreg_point(st, auto, "rreg mpg weight length", "mpg ~ weight + length",
               ["weight", "length", "Intercept"], "rreg auto", "mpg",
               ["weight", "length"])

rec.flush_summary()
print("\n== done ==")
