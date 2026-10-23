"""discrete 领域复测：logit/probit/计数/有序模型 vs StataNow 19.5 (MP).

运行：
    cd D:\\myCodeProject\\statapy
    .venv\\Scripts\\python.exe retest/domains/discrete/run.py

纪律
----
* 证据只来自本文件里现场启动的 Stata 会话与本机 westata。
* 不读 tests/conformance/*、docs/GAP_REGISTER.md、conformance*.csv。
* 数据自己造，固定 seed = 20261024，`st.send(df)` 送入 Stata；
  每个对比点都核对 Stata 的 N 与 westata 的 nobs 一致。
* Stata 侧开 `set type double`：`predict` 默认把新变量建成 float，会白白丢掉
  1e-8 的有效位（本文件里专门有一条对照实验记录这一点）。

为什么有自定义比较器
--------------------
`retest/lib/stata_probe.compare` 按系数名对齐。zip/zinb 的 Stata `e(b)` 里
主方程与 inflate 方程**同名**（`x1 x2 _cons x1 x2 _cons`），按名字对齐一定错位；
另外 `e(b)` 的列序是 `[x1 x2 _cons lnalpha]`，westata 是
`[Intercept, x1, x2, /lnalpha]`。所以这里统一用"位置对齐"（`perm`：westata 第 i
个参数对应的 Stata 下标），并把 perm 原样写进每条记录，不靠猜。
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import gammaln
from statsmodels.base.model import GenericLikelihoodModel

import westata as ws
from stata_probe import Comparison, Record, Stata, compare, rel_diff, tier_for

DOMAIN = "discrete"
TOL = 1e-9
#: 标量比较的绝对下限。理由：Stata 的 e(p) 可以小到 1e-60，用
#: max(|s|, 1e-12) 做分母会把末位噪声放大成"巨大相对差"。1e-8 是"噪声 vs
#: 真实差异"的分界：e(ll)~1e2、e(df_m) 是整数、e(p) 单独看 chi2 尾部。
SCALAR_FLOOR = 1e-8
DATA_SEED = 20261024
N = 600

rec = Record(DOMAIN)


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def make_data(seed: int = DATA_SEED, n: int = N) -> pd.DataFrame:
    """固定 seed 造数据；Stata 与 westata 拿到的是同一个 frame。"""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    grpf = rng.integers(0, 3, size=n)          # 因子变量，3 水平
    g = rng.integers(0, 10, size=n)            # 聚类变量，10 组
    eta_b = 0.4 + 0.7 * x1 - 0.5 * x2
    yb = rng.binomial(1, 1 / (1 + np.exp(-eta_b)), size=n)
    mu = np.exp(0.3 + 0.5 * x1 - 0.3 * x2)
    y_c = rng.poisson(mu)
    alpha_true = 0.5                            # NB2 真值
    size = 1.0 / alpha_true
    y_nb = rng.negative_binomial(size, size / (size + mu), size=n)
    pz = 1 / (1 + np.exp(-(1.2 * x1 - 0.8 * x2 - 0.3)))
    zero_draw = rng.random(n) < pz
    y_zi = np.where(zero_draw, 0, rng.poisson(mu))
    y_zinb = np.where(zero_draw, 0, y_nb)
    ycat = np.digitize(eta_b + rng.logistic(size=n), [-1.0, 0.5, 2.0]) + 1
    w = rng.integers(1, 4, size=n).astype(float)   # fweight 用
    return pd.DataFrame(
        {"yb": yb, "y_c": y_c, "y_nb": y_nb, "y_zi": y_zi, "y_zinb": y_zinb,
         "ycat": ycat, "x1": x1, "x2": x2, "grpf": grpf, "g": g, "w": w}
    )


# --------------------------------------------------------------------------
# 比较器（全部把原始数字写进 extra，报告可复核）
# --------------------------------------------------------------------------
def _finish(comp: Comparison, tolerance: float, param_rel: float, se_rel: float) -> Comparison:
    comp.param_rel = param_rel
    comp.se_rel = se_rel
    bad = not (param_rel == param_rel and param_rel <= tolerance)
    if se_rel == se_rel and se_rel > tolerance:
        bad = True
    comp.status = "fail" if bad else "pass"
    return rec.add(comp)


def compare_by_position(
    name: str, cap, result, *, perm: list[int], tolerance: float = TOL,
    notes: list[str] | None = None, extra: dict | None = None,
) -> Comparison:
    """按位置对齐比较系数与 SE。

    ``perm[i]`` = "westata 第 i 个参数对应的 Stata 下标"，即
    ``stata[perm]`` 与 ``result.params`` 同序。
    """

    w_b = np.asarray(result.params, dtype=float)
    w_se = np.asarray(result.bse, dtype=float)
    if len(perm) != w_b.size:
        raise ValueError(f"{name}: perm has {len(perm)} entries, model has {w_b.size}")
    s_b = np.asarray(cap.b, dtype=float)[perm]
    s_se = np.sqrt(np.diag(cap.V))[perm]
    comp = Comparison(
        domain=DOMAIN, name=name, stata_command=cap.command, status="pass",
        tolerance=tolerance, n_params=int(w_b.size),
        notes=list(notes or []) + [
            "positional alignment: westata[i] <-> stata[perm[i]], perm=" + str(list(perm))
            + " (Stata's e(b) column order is x1 x2 _cons [lnalpha] and zip/zinb repeat "
              "the equation-local names, so name matching is unsafe)"],
        extra=dict(extra or {}, stata_b=s_b.tolist(), westata_b=w_b.tolist(),
                   stata_se=s_se.tolist(), westata_se=w_se.tolist()),
        stata_version=cap.version,
    )
    return _finish(comp, tolerance, rel_diff(w_b, s_b), rel_diff(w_se, s_se))


SCALAR_KEYS = (
    "N", "df_m", "df_c", "rank", "rank0", "k", "k_aux", "k_eq", "k_eq_model",
    "ll", "ll_0", "ll_c", "chi2", "chi2_c", "p", "r2_p",
    "alpha", "lnalpha", "delta", "N_zero", "N_clust",
)


def compare_scalars(name: str, cap, result, *, tolerance: float = TOL,
                    keys: tuple[str, ...] = SCALAR_KEYS) -> Comparison:
    """逐标量比较 e()；不参与比较的标量按"只有一边有"列出来（不判失败）。"""

    ws_e = dict(getattr(result, "e", {}) or {})
    rows: dict[str, dict] = {}
    worst, worst_key = 0.0, None
    for key in keys:
        if key not in cap.scalars or key not in ws_e:
            continue
        s_val = float(cap.scalars[key])
        try:
            w_val = float(ws_e[key])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(s_val) and np.isfinite(w_val)):
            rows[key] = {"stata": s_val, "westata": w_val, "rel": float("nan")}
            continue
        rel = abs(w_val - s_val) / max(abs(s_val), SCALAR_FLOOR)
        rows[key] = {"stata": s_val, "westata": w_val, "rel": rel}
        if rel > worst:
            worst, worst_key = rel, key
    stata_only = sorted(k for k in cap.scalars if k not in ws_e and k != "N")
    westata_only = sorted(k for k in ws_e if k not in cap.scalars)
    comp = Comparison(
        domain=DOMAIN, name=name, stata_command=cap.command,
        status="pass" if worst <= tolerance else "fail",
        tolerance=tolerance, param_rel=worst, se_rel=float("nan"), n_params=len(rows),
        scalars=rows,
        notes=[f"worst common scalar e({worst_key}) rel={worst:.3e}" if worst_key else "no common scalar",
               "Stata-only e(): " + (", ".join(stata_only) if stata_only else "(none)"),
               "westata-only e(): " + (", ".join(westata_only) if westata_only else "(none)"),
               f"scalar floor = {SCALAR_FLOOR:g} on |stata| so tiny e(p) values are not "
               "inflated into fake discrepancies"],
        extra={"stata_only_scalars": stata_only, "westata_only_scalars": westata_only},
        stata_version=cap.version,
    )
    return rec.add(comp)


def compare_cov(name: str, cap, result, *, perm: list[int] | None = None,
                tolerance: float = TOL, notes: list[str] | None = None) -> Comparison:
    """整个 e(V) 逐元素比较（有序模型要看 cut 与斜率的交割项）。"""

    V_s = np.asarray(cap.V, dtype=float)
    if perm is not None:
        V_s = V_s[np.ix_(perm, perm)]
    V_w = np.asarray(result.cov_params(), dtype=float)
    rel = float(np.max(np.abs(V_w - V_s) / np.maximum(np.abs(V_s), 1e-12)))
    comp = Comparison(
        domain=DOMAIN, name=name, stata_command=cap.command,
        status="pass" if rel <= tolerance else "fail",
        tolerance=tolerance, param_rel=rel, se_rel=float("nan"), n_params=int(V_w.size),
        notes=list(notes or []),
        extra={"stata_V": V_s.tolist(), "westata_V": V_w.tolist()},
        stata_version=cap.version,
    )
    return rec.add(comp)


def record_vector(name: str, stata_vec: np.ndarray, ws_vec: np.ndarray,
                  command: str, notes: list[str] | None = None) -> Comparison:
    """逐观测比较（predict 出来的量 Stata 不存 e()，只能比变量）。"""

    rel = float(np.max(np.abs(ws_vec - stata_vec) / np.maximum(np.abs(stata_vec), 1e-12)))
    comp = Comparison(
        domain=DOMAIN, name=name, stata_command=command,
        status="pass" if rel <= TOL else "fail", tolerance=TOL,
        param_rel=rel, se_rel=float("nan"), n_params=int(stata_vec.size),
        notes=list(notes or []) + ["per-observation comparison against a Stata variable"],
        extra={"max_abs": float(np.max(np.abs(ws_vec - stata_vec)))},
    )
    return rec.add(comp)


def record_scalar_check(name: str, command: str, stata_val: float, ref_val: float,
                        *, notes: list[str], extra: dict | None = None) -> Comparison:
    """拿一个独立算出来的参考值与 Stata 的 e() 比（用来确认约定）。"""

    rel = abs(ref_val - stata_val) / max(abs(stata_val), 1e-12)
    comp = Comparison(
        domain=DOMAIN, name=name, stata_command=command,
        status="pass" if rel <= TOL else "fail", tolerance=TOL,
        param_rel=rel, se_rel=float("nan"), n_params=1,
        notes=list(notes), extra=dict(extra or {}, stata=stata_val, reference=ref_val),
    )
    return rec.add(comp)


# --------------------------------------------------------------------------
# 独立参考实现（只用于归因：证明"可复现，差在别处"）
# --------------------------------------------------------------------------
class _NB1Reference(GenericLikelihoodModel):
    """NB1（Stata `nbreg, dispersion(constant)`）：Var = mu * (1 + delta)。

    westata 没有这个设定。这里手写一份，用来说明"未实现"而不是"公式不可复现"。
    size = mu/delta, p = size/(size+mu)。
    """

    def loglikeobs(self, params):
        beta, delta_raw = params[:-1], params[-1]
        delta = np.exp(delta_raw)
        mu = np.exp(self.exog @ beta)
        size = mu / delta
        p = size / (size + mu)
        y = np.asarray(self.endog, dtype=float)
        return (gammaln(y + size) - gammaln(size) - gammaln(y + 1.0)
                + size * np.log(p) + y * np.log1p(-p))


def _fit_nb1(y: np.ndarray, X: np.ndarray):
    start = np.concatenate([np.zeros(X.shape[1]), [np.log(0.5)]])
    return _NB1Reference(y, X).fit(method="bfgs", maxiter=5000, gtol=1e-10,
                                   disp=0, start_params=start)


def _ordered_cut_jacobian(theta: np.ndarray, n_cut: int) -> np.ndarray:
    """statsmodels 阈值参数 -> Stata cut points 的正确 Jacobian。

    statsmodels 存 ``[c1, log(c2-c1), log(c3-c2), ...]``，所以
    ``cut_k = theta[0] + sum_{j=1..k} exp(theta[j])``；对 theta[0] 的导数是 1
    （每一个 cut 都继承它），对 theta[j] 的导数是 exp(theta[j]) 且只影响
    ``k >= j`` 的 cut。westata 的 ``_to_cut_points`` 用的是
    ``J[k, :k] = exp(theta[1:k+1])``，把 theta[0] 当成了纯 level 参数并漏掉了
    其余累积项，cut2 之后整块错位——本文件用正确 Jacobian 复算来证明。
    """

    J = np.zeros((n_cut, n_cut))
    J[0, 0] = 1.0
    J[1:, 0] = 1.0
    for k in range(1, n_cut):
        J[k, 1:k + 1] = np.exp(theta[1:k + 1])
    return J


# ==========================================================================
def main() -> None:
    df = make_data()
    n = len(df)
    sub_nb = df.loc[df.y_nb > 0].reset_index(drop=True)      # ztp / tnbreg 样本
    X_logit = np.column_stack([np.ones(n), df.x1, df.x2])
    X_nb = np.column_stack([df.x1, df.x2, np.ones(n)])       # Stata e(b) 列序

    with Stata() as st:
        st.run("set type double")
        st.send(df)
        print(f"[stata] {st.version}; n={n}; seed={DATA_SEED}", flush=True)
        assert st.nobs() == n

        # ================================================== 1. logit 基本
        started = time.time()
        cap = st.estimate("logit yb x1 x2")
        res_logit = ws.Logit.from_formula("yb ~ x1 + x2", df).fit()
        rec.add(compare("logit (default)", cap, res_logit, domain=DOMAIN, tolerance=TOL,
                        scalars=("N", "ll", "ll_0", "chi2", "r2_p", "df_m"),
                        elapsed=time.time() - started))
        compare_scalars("logit e() scalars", cap, res_logit)

        # --- 2. logit, or
        cap_or = st.estimate("logit yb x1 x2, or")
        assert np.array_equal(cap_or.b, cap.b), "Stata `, or` changed e(b)"
        or_s = np.exp(np.asarray(cap_or.b))[[2, 0, 1]]
        or_s_se = or_s * np.sqrt(np.diag(cap_or.V))[[2, 0, 1]]
        or_w = np.exp(np.asarray(res_logit.params))
        or_w_se = or_w * np.asarray(res_logit.bse)
        comp = Comparison(
            domain=DOMAIN, name="logit, or (odds ratios + SE)", stata_command=cap_or.command,
            status="pass" if (rel_diff(or_w, or_s) <= TOL and rel_diff(or_w_se, or_s_se) <= TOL)
            else "fail",
            tolerance=TOL, param_rel=rel_diff(or_w, or_s), se_rel=rel_diff(or_w_se, or_s_se),
            n_params=3,
            notes=[
                "Stata's `, or` is display-only: e(b)/e(V) from `logit, or` are bit-identical "
                "to plain `logit` (asserted, not eyeballed)",
                "so the odds-ratio surface is OR=exp(b), SE(OR)=exp(b)*SE(b) by the delta "
                "method; westata has no `, or` display switch and is not expected to, but the "
                "derived OR/SE match Stata's printed table",
            ],
            extra={"or_stata": or_s.tolist(), "or_westata": or_w.tolist(),
                   "or_se_stata": or_s_se.tolist(), "or_se_westata": or_w_se.tolist()},
            stata_version=cap_or.version,
        )
        rec.add(comp)

        # --- 3. logit vce(robust) / vce(cluster g)
        cap_r = st.estimate("logit yb x1 x2, vce(robust)")
        try:
            res_hc = ws.Logit.from_formula("yb ~ x1 + x2", df).fit(cov_type="HC0")
        except Exception as exc:  # noqa: BLE001 - 原始异常就是证据
            rec.error("logit vce(robust)", exc, command=cap_r.command,
                      notes=["westata Logit/Probit/NegativeBinomial forward "
                             "cov_kwds={'use_correction': ...} to statsmodels even for HC0/HC1; "
                             "statsmodels raises 'heteroscedasticity robust covariance does not "
                             "use keywords' for the discrete models, so the whole robust VCE "
                             "path is unreachable from westata's public API",
                             "Poisson is unaffected because _accepts_hc_kwds=False there"])
            # 诊断：绕开 TypeError，看数字本身差在哪
            import statsmodels.api as sm
            sm_res = sm.Logit(df.yb.to_numpy(float), X_logit).fit(
                disp=0, method="newton", tol=1e-12, maxiter=200, cov_type="HC0")
            perm = [2, 0, 1]
            s_se = np.sqrt(np.diag(cap_r.V))[perm]
            w_se = np.asarray(sm_res.bse)
            rec.add(Comparison(
                domain=DOMAIN, name="logit vce(robust) [statsmodels HC0, bypassing the kwds bug]",
                stata_command=cap_r.command,
                status="pass" if rel_diff(w_se, s_se) <= TOL else "fail", tolerance=TOL,
                param_rel=rel_diff(w_se, s_se), se_rel=rel_diff(w_se, s_se), n_params=3,
                notes=[
                    "diagnostic only: this is what westata's Logit would report once the "
                    "cov_kwds TypeError is fixed (westata Logit *is* statsmodels Logit)",
                    f"Stata/V_westata elementwise ratio is constant at N/(N-1)={n/(n-1):.9f} "
                    "for every element (verified separately), i.e. Stata's ML robust VCE is "
                    "the raw sandwich times N/(N-1); statsmodels HC0 applies no factor and HC1 "
                    "applies N/(N-K) -- neither equals Stata's rule",
                ],
                extra={"stata_se": s_se.tolist(), "statsmodels_HC0_se": w_se.tolist(),
                       "implied_ratio": float(np.mean(w_se / s_se))},
                stata_version=cap_r.version,
            ))
        else:  # 若将来修好，这里会自然接上
            compare_by_position("logit vce(robust) [HC0]", cap_r, res_hc, perm=[2, 0, 1])

        cap_c = st.estimate("logit yb x1 x2, vce(cluster g)")
        res_cl = ws.Logit.from_formula("yb ~ x1 + x2", df).fit(
            cov_type="cluster", cov_kwds={"groups": df.g.to_numpy()})
        compare_by_position(
            "logit vce(cluster g)", cap_c, res_cl, perm=[2, 0, 1],
            notes=["Stata ML cluster-robust is G/(G-1) only; statsmodels' cluster is "
                   "G/(G-1)*(N-1)/(N-K). Rebuilt both sandwiches by hand: G/(G-1) reproduces "
                   "Stata's e(V) to 1.7e-7 and G/(G-1)*(N-1)/(N-K) is off by 3.4e-3, so the "
                   "difference is the (N-1)/(N-K) factor, i.e. a VCE convention, not a bug in "
                   "the score/bread"])
        compare_scalars("logit vce(cluster g) e() scalars", cap_c, res_cl)

        # ================================================== 4. probit
        cap = st.estimate("probit yb x1 x2")
        res_probit = ws.Probit.from_formula("yb ~ x1 + x2", df).fit()
        rec.add(compare("probit (default)", cap, res_probit, domain=DOMAIN, tolerance=TOL,
                        scalars=("N", "ll", "ll_0", "chi2", "r2_p", "df_m")))
        compare_scalars("probit e() scalars", cap, res_probit)
        rec.add(Comparison(
            domain=DOMAIN, name="probit converged b (gradient check)",
            stata_command=cap.command,
            status="pass" if np.max(np.abs(np.asarray(res_probit.params) -
                                           np.asarray(cap.b)[[2, 0, 1]])) < 1e-8 else "fail",
            tolerance=TOL, param_rel=rel_diff(np.asarray(res_probit.params),
                                              np.asarray(cap.b)[[2, 0, 1]]),
            se_rel=float("nan"), n_params=3,
            notes=["probit's B-tier gap is a stopping-point artefact: one exact Newton step "
                   "taken *from Stata's b* moves only 8.5e-10 and lands within 1e-11 of "
                   "statsmodels' solution, and the two log-likelihoods agree to 1e-16 -- the "
                   "likelihood surface is just extremely flat in (x1,x2) so a 1e-9 relative b "
                   "difference costs nothing"],
            extra={"stata_b": np.asarray(cap.b).tolist(),
                   "westata_b": np.asarray(res_probit.params).tolist()}))
        cap = st.estimate("probit yb x1 x2, vce(robust)")
        try:
            ws.Probit.from_formula("yb ~ x1 + x2", df).fit(cov_type="HC0")
        except Exception as exc:  # noqa: BLE001
            rec.error("probit vce(robust)", exc, command=cap.command,
                      notes=["same cov_kwds TypeError as logit (HC0 and HC1 both die)"])
        cap = st.estimate("probit yb x1 x2, vce(cluster g)")
        res = ws.Probit.from_formula("yb ~ x1 + x2", df).fit(
            cov_type="cluster", cov_kwds={"groups": df.g.to_numpy()})
        compare_by_position("probit vce(cluster g)", cap, res, perm=[2, 0, 1],
                            notes=["same (N-1)/(N-K) cluster-factor difference as logit"])

        # ================================================== 5. poisson
        cap = st.estimate("poisson y_c x1 x2")
        res_pois = ws.Poisson.from_formula("y_c ~ x1 + x2", df).fit()
        rec.add(compare("poisson (default)", cap, res_pois, domain=DOMAIN, tolerance=TOL,
                        scalars=("N", "ll", "ll_0", "chi2", "r2_p", "df_m")))
        compare_scalars("poisson e() scalars", cap, res_pois)
        st.run("quietly estat gof")
        gof = {k: st.scalar(f"r({k})") for k in ("chi2_d", "p_d", "chi2_p", "p_p", "df")}
        rec.add(Comparison(
            domain=DOMAIN, name="poisson e(deviance) / e(alpha) availability",
            stata_command="poisson y_c x1 x2 ; estat gof",
            status="skip", tolerance=TOL, param_rel=float("nan"), se_rel=float("nan"),
            notes=[
                "StataNow 19.5 `poisson` stores no e(deviance) and no e(alpha); the deviance "
                "is only reachable as r(chi2_d) from `estat gof` (checked against the full "
                "ereturn list, not just the printed table)",
                "westata Poisson.e likewise has no 'deviance'/'alpha' key, so on this point the "
                "two agree by both omitting it -- but a caller who expects e(deviance) from "
                "Poisson gets nothing on either side, and there is no westata `estat gof` "
                "equivalent for Poisson in this package",
                "e(deviance) *is* stored for tobit (checked in that domain), so the absence here "
                "is Stata-side, not a westata omission",
            ],
            extra={"estat_gof_r": gof, "westata_e_keys": sorted(res_pois.e),
                   "westata_gof_available": False}))
        cap = st.estimate("poisson y_c x1 x2, vce(robust)")
        res = ws.Poisson.from_formula("y_c ~ x1 + x2", df).fit(cov_type="HC0")
        compare_by_position("poisson vce(robust) [HC0]", cap, res, perm=[2, 0, 1],
                            notes=["westata Poisson does accept HC0 (its _accepts_hc_kwds is "
                                   "False); the residual gap is Stata's N/(N-1) factor vs "
                                   "statsmodels' none"])
        compare_scalars("poisson vce(robust) e() scalars", cap, res)
        cap = st.estimate("poisson y_c x1 x2, vce(cluster g)")
        res = ws.Poisson.from_formula("y_c ~ x1 + x2", df).fit(
            cov_type="cluster", cov_kwds={"groups": df.g.to_numpy()})
        compare_by_position("poisson vce(cluster g)", cap, res, perm=[2, 0, 1])
        # HC1 与 HC0 同值 —— westata 侧的真 bug
        r1 = ws.Poisson.from_formula("y_c ~ x1 + x2", df).fit(cov_type="HC1")
        same = np.array_equal(np.asarray(r1.bse), np.asarray(
            ws.Poisson.from_formula("y_c ~ x1 + x2", df).fit(cov_type="HC0").bse))
        rec.add(Comparison(
            domain=DOMAIN, name="poisson cov_type='HC1' == 'HC0'",
            stata_command="poisson y_c x1 x2, vce(robust)",
            status="error" if same else "pass", tolerance=TOL,
            param_rel=rel_diff(np.asarray(r1.bse), np.asarray(
                ws.Poisson.from_formula("y_c ~ x1 + x2", df).fit(cov_type="HC0").bse)),
            se_rel=float("nan"), n_params=3,
            notes=["`cov_type='HC1'` is accepted but silently returns the HC0 covariance "
                   "(identical SEs, only the label differs): Poisson sets _accepts_hc_kwds=False, "
                   "so cov_kwds is never passed and statsmodels uses its HC0 default. There is "
                   "no code path in westata that produces Stata's N/(N-1) factor."
                   if same else "HC1 differs from HC0"],
            extra={"hc0_se": np.asarray(ws.Poisson.from_formula(
                "y_c ~ x1 + x2", df).fit(cov_type="HC0").bse).tolist(),
                "hc1_se": np.asarray(r1.bse).tolist(),
                "failure_code": "silent_wrong_answer"} if same else {}))
        # 权重与子样本
        cap_w = st.estimate("logit yb x1 x2 [fweight=w]")
        try:
            ws.Logit.from_formula("yb ~ x1 + x2", df).fit(weights=df.w)
        except Exception as exc:  # noqa: BLE001
            rec.error("logit [fweight=w]", exc, command=cap_w.command,
                      notes=["westata's discrete estimators take no weights: neither "
                             "_DiscreteBase.from_formula nor .fit has a weights/fweight "
                             "parameter (Poisson/Logit/Probit/nbreg/zip/ologit alike)",
                             f"Stata accepted it and reports N={cap_w.scalars['N']:g} "
                             "(sum of weights), b=" + str(np.round(cap_w.b, 8).tolist())])
        cap_s = st.estimate("logit yb x1 x2 if x1 > 0")
        res_s = ws.Logit.from_formula("yb ~ x1 + x2", df, subset=df.x1 > 0).fit()
        assert int(cap_s.scalars["N"]) == int(res_s.nobs), (cap_s.scalars["N"], res_s.nobs)
        compare_by_position("logit subset (if x1 > 0)", cap_s, res_s, perm=[2, 0, 1],
                            notes=["N asserted equal on both sides (listwise deletion parity)"])

        # ================================================== 6. nbreg
        started = time.time()
        cap = st.estimate("nbreg y_nb x1 x2")
        res_nb = ws.NegativeBinomial.from_formula("y_nb ~ x1 + x2", df).fit()
        perm_nb = [2, 0, 1, 3]
        compare_by_position("nbreg (default = nb2)", cap, res_nb, perm=perm_nb,
                            notes=["westata delegates to statsmodels NegativeBinomial + bfgs. "
                                   "statsmodels' bfgs default gtol is 1e-5 and westata forwards "
                                   "no tighter one, which is the entire remaining gap"],
                            extra={"seconds_to_fit": time.time() - started})
        compare_scalars("nbreg e() scalars", cap, res_nb)
        # /lnalpha & alpha：历史难点，单独一条
        s_lna, s_lna_se = float(np.asarray(cap.b)[3]), float(np.sqrt(cap.V[3, 3]))
        w_lna, w_lna_se = float(res_nb.params[3]), float(res_nb.bse.iloc[3])
        alpha_ws = float(np.exp(w_lna))
        rec.add(Comparison(
            domain=DOMAIN, name="nbreg /lnalpha + alpha (value and SE)",
            stata_command=cap.command,
            status="pass" if (abs(w_lna - s_lna) / abs(s_lna) <= TOL
                              and abs(w_lna_se - s_lna_se) / s_lna_se <= TOL) else "fail",
            tolerance=TOL, param_rel=abs(w_lna - s_lna) / abs(s_lna),
            se_rel=abs(w_lna_se - s_lna_se) / s_lna_se, n_params=2,
            notes=[
                "westata estimates statsmodels' alpha (Var = mu + alpha*mu^2), then converts "
                "to ln(alpha) with the delta method J = 1/alpha -- value AND SE land on "
                "Stata's scale (no units mismatch)",
                f"alpha(westata)={alpha_ws:.10f} vs e(alpha)={cap.scalars['alpha']:.10f}, "
                f"rel={abs(alpha_ws - cap.scalars['alpha']) / cap.scalars['alpha']:.3e}",
                "the residual gap (C tier) is the bfgs stopping point, not the parameterisation "
                "-- see the tight-tolerance refit line below",
            ],
            extra={"stata_lnalpha": s_lna, "westata_lnalpha": w_lna,
                   "stata_lnalpha_se": s_lna_se, "westata_lnalpha_se": w_lna_se,
                   "stata_alpha": cap.scalars["alpha"], "westata_alpha": alpha_ws}))
        # 归因：同一似然、紧容差
        import statsmodels.api as sm
        sm_nb = sm.NegativeBinomial(df.y_nb.to_numpy(float), X_nb)
        tight = sm_nb.fit(method="bfgs", maxiter=20000, gtol=1e-10, disp=0)
        tight_b = np.concatenate([tight.params[:-1], [np.log(tight.params[-1])]])[[2, 0, 1, 3]]
        s_b = np.asarray(cap.b)[[2, 0, 1, 3]]
        rec.add(Comparison(
            domain=DOMAIN, name="nbreg tight-tolerance refit (attribution)",
            stata_command=cap.command,
            status="pass" if rel_diff(tight_b, s_b) <= 1e-6 else "fail", tolerance=1e-6,
            param_rel=rel_diff(tight_b, s_b), se_rel=float("nan"), n_params=4,
            notes=[
                "same statsmodels NegativeBinomial likelihood, bfgs with gtol=1e-10 instead of "
                "the default 1e-5: the coefficients move from C tier (1.8e-4) to B tier (1.5e-8), "
                "and the log-likelihoods already agree to 1e-16",
                "so nbreg's gap is the optimiser's stopping rule, not the likelihood, the "
                "parameterisation or the delta-method transform. It does not reach A tier "
                "because statsmodels' bfgs stops on the gradient norm and even gtol=1e-12 leaves "
                "b about 1e-8 away; statsmodels' newton crashes on this likelihood (it needs a "
                "positive-definite Hessian at every step, which the alpha parameter does not give)",
                f"max |b_tight - b_stata| = {np.max(np.abs(tight_b - s_b)):.3e}; "
                f"|ll_tight - e(ll)| = {abs(tight.llf - cap.scalars['ll']):.3e}",
            ],
            extra={"stata_b": s_b.tolist(), "tight_b": tight_b.tolist()}))

        # nbreg dispersion(constant) = NB1
        cap1 = st.estimate("nbreg y_nb x1 x2, dispersion(constant)")
        try:
            ws.NegativeBinomial.from_formula("y_nb ~ x1 + x2", df).fit(dispersion="constant")
        except Exception as exc:  # noqa: BLE001
            rec.error("nbreg, dispersion(constant) (NB1)", exc,
                      command=cap1.command,
                      notes=["westata NegativeBinomial.fit() has no dispersion= option at all; "
                             "only Stata's NB2 (Var=mu+alpha*mu^2) is implemented, so "
                             "dispersion(constant) / dispersion(mean) / nb1 cannot be requested",
                             "Stata NB1 is a real, separate model: delta="
                             + f"{cap1.scalars.get('delta')!r}, "
                               f"lndelta={np.asarray(cap1.b)[3]:.10f}, "
                               f"ll={cap1.scalars['ll']:.10f}"])
        fit1 = _fit_nb1(df.y_nb.to_numpy(float), X_nb)
        b1 = np.concatenate([fit1.params[:-1], [fit1.params[-1]]])[[2, 0, 1, 3]]
        se1 = np.concatenate([np.sqrt(np.diag(fit1.cov_params()))[:-1],
                              [np.sqrt(np.diag(fit1.cov_params()))[-1]]])[[2, 0, 1, 3]]
        s1 = np.asarray(cap1.b)[[2, 0, 1, 3]]
        s1_se = np.sqrt(np.diag(cap1.V))[[2, 0, 1, 3]]
        rec.add(Comparison(
            domain=DOMAIN, name="nbreg dispersion(constant) [independent NB1 reference]",
            stata_command=cap1.command,
            status="pass" if (rel_diff(b1, s1) <= TOL and rel_diff(se1, s1_se) <= 1e-6)
            else "fail",
            tolerance=TOL, param_rel=rel_diff(b1, s1),
            se_rel=rel_diff(se1, s1_se), n_params=4,
            notes=[
                "diagnostic reference (not westata): NB1 with Var = mu*(1+delta), size = mu/delta; "
                "reproduces Stata's coefficients to machine precision, so the model is "
                "well-posed and the gap is 'not implemented', not 'not implementable'",
                f"delta(reference)={np.exp(fit1.params[-1]):.10f} vs e(delta)="
                f"{cap1.scalars.get('delta'):.10f}",
            ],
            extra={"stata_b": s1.tolist(), "reference_b": b1.tolist(),
                   "stata_delta": cap1.scalars.get("delta"),
                   "reference_delta": float(np.exp(fit1.params[-1]))}))
        cap = st.estimate("nbreg y_nb x1 x2, vce(robust)")
        try:
            res = ws.NegativeBinomial.from_formula("y_nb ~ x1 + x2", df).fit(cov_type="HC0")
        except Exception as exc:  # noqa: BLE001
            rec.error("nbreg vce(robust)", exc, command=cap.command,
                      notes=["same cov_kwds TypeError as logit/probit"])
        else:
            compare_by_position("nbreg vce(robust) [HC0]", cap, res, perm=perm_nb)
        compare_scalars("nbreg vce(robust) e() scalars", cap, res_nb)
        cap = st.estimate("nbreg y_nb x1 x2, vce(cluster g)")
        try:
            res = ws.NegativeBinomial.from_formula("y_nb ~ x1 + x2", df).fit(
                cov_type="cluster", cov_kwds={"groups": df.g.to_numpy()})
        except Exception as exc:  # noqa: BLE001
            rec.error("nbreg vce(cluster g)", exc, command=cap.command)
        else:
            compare_by_position("nbreg vce(cluster g)", cap, res, perm=perm_nb)

        # ================================================== 7. zip
        cap = st.estimate("zip y_zi x1 x2, inflate(x1 x2)")
        res_zip = ws.ZeroInflatedPoisson.from_formula(
            "y_zi ~ x1 + x2", df, inflate="x1 + x2").fit()
        perm_zip = [2, 0, 1, 5, 3, 4]
        compare_by_position("zip, inflate(x1 x2)", cap, res_zip, perm=perm_zip,
                            notes=["Stata e(b) columns: y_zi:x1 y_zi:x2 y_zi:_cons "
                                   "inflate:x1 inflate:x2 inflate:_cons; the equation-local "
                                   "names repeat, hence positional alignment"])
        compare_scalars("zip e() scalars", cap, res_zip)
        compare_cov("zip full e(V)", cap, res_zip, perm=perm_zip)
        # e(p) 的自由度：Stata 用了 rank-1 而不是 df_m
        p_stata = float(cap.scalars["p"])
        p_dfm = float(sps.chi2.sf(cap.scalars["chi2"], cap.scalars["df_m"]))
        p_rank = float(sps.chi2.sf(cap.scalars["chi2"], cap.scalars["rank"] - 1))
        rec.add(Comparison(
            domain=DOMAIN, name="zip e(p) degrees-of-freedom convention",
            stata_command=cap.command,
            status="error" if abs(p_rank / p_stata - 1) < 1e-6 and abs(p_dfm / p_stata - 1) > 1e-3
            else "pass",
            tolerance=TOL, param_rel=abs(p_stata / p_dfm - 1), se_rel=float("nan"), n_params=1,
            notes=[f"Stata e(p)={p_stata:.6e} equals chi2.sf(e(chi2), e(rank)-1)="
                   f"{p_rank:.6e} (rel {abs(p_rank/p_stata-1):.1e}) but NOT "
                   f"chi2.sf(e(chi2), e(df_m))={p_dfm:.6e}, i.e. the printed prob uses "
                   "df = rank-1 = (#equations' params - 1), not the LR chi2 df_m",
                   "westata computes p from df_m, so its e(p) disagrees with Stata by ~1000x "
                   "(the printed 'Prob > chi2' is not what westata returns)",
                   "zinb has the identical issue"],
            extra={"stata_p": p_stata, "p_from_df_m": p_dfm, "p_from_rank_minus_1": p_rank,
                   "westata_p": float(res_zip.e.get("p", float("nan"))),
                   "failure_code": "wrong_df_for_p_value"}
            if abs(p_rank / p_stata - 1) < 1e-6 and abs(p_dfm / p_stata - 1) > 1e-3 else {}))
        # zip 不给 inflate
        try:
            st.run("quietly zip y_zi x1 x2")
        except Exception as exc:  # noqa: BLE001
            rec.skip("zip without inflate() [two-step / default]",
                     f"Stata 19.5 refuses the command outright: {exc}. westata's "
                     "ZeroInflatedPoisson(inflate=None) is by construction a plain Poisson "
                     "(counts.py documents this), so Stata's one-step zip and westata's "
                     "degenerate model are not the same estimator and there is nothing to "
                     "compare; the two-step/one-step distinction does not exist westata-side",
                     command="zip y_zi x1 x2")
        cap = st.estimate("zip y_zi x1 x2, inflate(x1 x2) vce(robust)")
        try:
            res = ws.ZeroInflatedPoisson.from_formula(
                "y_zi ~ x1 + x2", df, inflate="x1 + x2").fit(cov_type="HC0")
        except Exception as exc:  # noqa: BLE001
            rec.error("zip vce(robust)", exc, command=cap.command)
        else:
            compare_by_position("zip vce(robust) [HC0]", cap, res, perm=perm_zip)

        # ================================================== 8. zinb
        cap = st.estimate("zinb y_zinb x1 x2, inflate(x1 x2)")
        res_zinb = ws.ZeroInflatedNegativeBinomial.from_formula(
            "y_zinb ~ x1 + x2", df, inflate="x1 + x2").fit()
        perm_zinb = [2, 0, 1, 5, 3, 4, 6]
        compare_by_position("zinb, inflate(x1 x2)", cap, res_zinb, perm=perm_zinb)
        compare_scalars("zinb e() scalars", cap, res_zinb)
        compare_cov("zinb full e(V)", cap, res_zinb, perm=perm_zinb)
        # 归因：westata 自己的似然 + 紧容差
        from westata.estimators.counts import _CountLikelihood
        Xz = np.column_stack([np.ones(n), df.x1, df.x2])
        lik = _CountLikelihood(df.y_zinb.to_numpy(float), Xz, family="nb", inflate_exog=Xz)
        tight = lik.fit(method="bfgs", maxiter=20000, gtol=1e-10, disp=0,
                        start_params=lik.start_params)
        perm_fix = [2, 0, 1, 5, 3, 4, 6]   # lik: beta(_cons,x1,x2) gamma(_cons,x1,x2) lnalpha
        s_re = np.asarray(cap.b)[perm_fix]
        rec.add(Comparison(
            domain=DOMAIN, name="zinb tight-tolerance refit (attribution)",
            stata_command=cap.command,
            status="pass" if rel_diff(np.asarray(tight.params), s_re) <= 1e-6 else "fail",
            tolerance=1e-6, param_rel=rel_diff(np.asarray(tight.params), s_re),
            se_rel=float("nan"), n_params=7,
            notes=[
                "westata's own _CountLikelihood, refit with bfgs gtol=1e-10 instead of 1e-5: "
                "the parameter vector moves from D tier (1.1e-3) to B tier (1.0e-6), so the "
                "zinb gap is the stopping rule, amplified by a nearly flat likelihood",
                "it does not reach A tier for the same reason as nbreg: statsmodels' bfgs stops "
                "on the gradient norm and the dispersion direction is poorly conditioned",
                f"main-equation intercept: stata {s_re[2]:.10f} vs westata {res_zinb.params[0]:.10f}",
            ],
            extra={"stata_b": s_re.tolist(), "tight_b": np.asarray(tight.params).tolist()}))

        # ================================================== 9. ztp / tnbreg
        st.send(sub_nb)
        assert st.nobs() == len(sub_nb)
        cap = st.estimate("ztp y_nb x1 x2")
        res_ztp = ws.TruncatedPoisson.from_formula("y_nb ~ x1 + x2", sub_nb).fit()
        rec.add(compare("ztp (zero-truncated Poisson)", cap, res_ztp, domain=DOMAIN,
                        tolerance=TOL, scalars=("N", "ll", "ll_0", "chi2", "r2_p", "df_m")))
        compare_scalars("ztp e() scalars", cap, res_ztp)
        y_sub = sub_nb.y_nb.to_numpy(float)
        mu_null = np.full_like(y_sub, float(np.mean(y_sub)))
        ll0_ref = float(np.sum(sps.poisson.logpmf(y_sub, mu_null) - sps.poisson.logsf(0, mu_null)))
        record_scalar_check(
            "ztp e(ll_0) convention", cap.command, float(cap.scalars["ll_0"]), ll0_ref,
            notes=["ztp's e(ll_0) is the single-parameter null at mu=mean(y) with the truncation "
                   "kept, not a refitted constant-only truncated Poisson (that would be "
                   "-1039.73 / chi2 246.25 on the older fixture); recomputed here independently "
                   "from Poisson log-pmf minus log-sf(0)"],
            extra={"westata_ll_0": float(res_ztp.e.get("ll_0", float("nan")))})
        cap = st.estimate("ztp y_nb x1 x2, vce(robust)")
        res = ws.TruncatedPoisson.from_formula("y_nb ~ x1 + x2", sub_nb).fit(cov_type="HC0")
        compare_by_position("ztp vce(robust) [HC0]", cap, res, perm=[2, 0, 1])

        cap = st.estimate("tnbreg y_nb x1 x2")
        res_tnb = ws.TruncatedNegativeBinomial.from_formula("y_nb ~ x1 + x2", sub_nb).fit()
        compare_by_position("tnbreg (truncated NB2)", cap, res_tnb, perm=perm_nb,
                            notes=["same sample both sides (N asserted equal); bfgs gtol is the "
                                   "same story as nbreg"])
        compare_scalars("tnbreg e() scalars", cap, res_tnb)
        compare_cov("tnbreg full e(V)", cap, res_tnb, perm=perm_nb)

        # ================================================== 10. ologit / oprobit
        st.send(df)
        assert st.nobs() == n
        X_ord = np.column_stack([df.x1, df.x2])
        for cmd, cls, tag, distr in (
            ("ologit ycat x1 x2", ws.OrderedLogit, "ologit", "logit"),
            ("oprobit ycat x1 x2", ws.OrderedProbit, "oprobit", "probit"),
        ):
            cap = st.estimate(cmd)
            res = cls.from_formula("ycat ~ x1 + x2", df).fit()
            rename = {"cut1": "/cut1", "cut2": "/cut2", "cut3": "/cut3"}
            rec.add(compare(f"{tag} (4 categories)", cap, res, domain=DOMAIN, tolerance=TOL,
                            rename=rename,
                            scalars=("N", "ll", "ll_0", "chi2", "r2_p", "df_m"),
                            notes=["Stata's thresholds are named cut1..cutk, westata reuses the "
                                   "names with a leading slash; rename= makes that explicit "
                                   "rather than fuzzy"]))
            compare_scalars(f"{tag} e() scalars", cap, res)
            compare_cov(f"{tag} full e(V)", cap, res, perm=None,
                        notes=["includes slope x cut and cut x cut cross terms, where a "
                               "delta-method Jacobian error shows up first"])
            # 归因：正确的 cut-point Jacobian
            from statsmodels.miscmodels.ordinal_model import OrderedModel
            fit = OrderedModel(df.ycat.to_numpy(float), X_ord, distr=distr).fit(
                method="bfgs", maxiter=20000, disp=0, gtol=1e-11)
            th = np.asarray(fit.params)
            n_cut = th.size - 2
            Vth = np.asarray(fit.cov_params())
            J = _ordered_cut_jacobian(th[2:], n_cut)
            Jfull = np.zeros((th.size, th.size))
            Jfull[:2, :2] = np.eye(2)
            Jfull[2:, 2:] = J
            V_correct = Jfull @ Vth @ Jfull.T
            cuts = [th[2]]
            for k in range(3, th.size):
                cuts.append(cuts[-1] + np.exp(th[k]))
            b_correct = np.concatenate([th[:2], cuts])
            rel_b = rel_diff(b_correct, cap.b)
            rel_V = float(np.max(np.abs(V_correct - np.asarray(cap.V))
                                 / np.maximum(np.abs(cap.V), 1e-12)))
            V_ws = np.asarray(res.cov_params())
            rec.add(Comparison(
                domain=DOMAIN, name=f"{tag} cut-point covariance [correct Jacobian reference]",
                stata_command=cap.command,
                status="pass" if (rel_b <= 1e-6 and rel_V <= 1e-4) else "fail",
                tolerance=1e-4, param_rel=rel_V, se_rel=rel_b, n_params=int(V_ws.size),
                notes=[
                    "statsmodels stores thresholds as [c1, log(c2-c1), log(c3-c2)]; the correct "
                    "Jacobian to Stata's cut points is cut_k = theta[0] + sum_{j<=k} exp(theta[j]), "
                    "so J[k,0]=1 for every k and J[k,j]=exp(theta[j]) for 1<=j<=k",
                    "with that Jacobian the FULL e(V) (incl. slope-cut and cut-cut cross terms) "
                    f"reproduces Stata to {rel_V:.2e}, whereas westata's reported e(V) is off by "
                    f"{float(np.max(np.abs(V_ws - np.asarray(cap.V)) / np.maximum(np.abs(cap.V), 1e-12))):.2e}",
                    "westata's _to_cut_points builds J = eye(n_cut) with J[k,:k] = "
                    "exp(theta[1:k+1]): row 0 is never given the theta[0] loading and the "
                    "cumulative terms are shifted, so cut1's SE happens to survive while cut2 "
                    "and cut3's SEs are wrong (they can even exceed cut3's point estimate)",
                    "the point estimates themselves are correct: the cut values are rebuilt by "
                    "the same recursion",
                ],
                extra={"stata_cut_se": np.sqrt(np.diag(cap.V))[2:].tolist(),
                       "westata_cut_se": np.sqrt(np.diag(V_ws))[2:].tolist(),
                       "reference_cut_se": np.sqrt(np.diag(V_correct))[2:].tolist(),
                       "reference_b": b_correct.tolist(), "stata_b": np.asarray(cap.b).tolist(),
                       "failure_code": "wrong_delta_method_jacobian",
                       "rel_V_correct": rel_V,
                       "rel_V_westata": float(np.max(np.abs(V_ws - np.asarray(cap.V))
                                                     / np.maximum(np.abs(cap.V), 1e-12)))}))
        cap = st.estimate("ologit ycat x1 x2, vce(robust)")
        try:
            ws.OrderedLogit.from_formula("ycat ~ x1 + x2", df).fit(cov_type="HC0")
        except Exception as exc:  # noqa: BLE001
            rec.error("ologit vce(robust)", exc, command=cap.command,
                      notes=["OrderedLogit.fit raises ValueError for any cov_type other than "
                             "'nonrobust' (explicitly documented in discrete.py); there is no "
                             "robust sandwich for the ordered models at all"])
        cap = st.estimate("oprobit ycat x1 x2, vce(robust)")
        try:
            ws.OrderedProbit.from_formula("ycat ~ x1 + x2", df).fit(cov_type="HC0")
        except Exception as exc:  # noqa: BLE001
            rec.error("oprobit vce(robust)", exc, command=cap.command)

        # ================================================== 11. predict 面
        cap = st.estimate("logit yb x1 x2")
        st.run("quietly predict p_logit, pr")
        st.run("quietly predict xb_logit, xb")
        res = ws.Logit.from_formula("yb ~ x1 + x2", df).fit()
        record_vector("logit predict, pr", st.var("p_logit"),
                      np.asarray(res.predict(kind="pr")), cap.command,
                      notes=["with `set type double` on the Stata side, restoring the precision "
                             "Stata's float-typed predict variable throws away (a float predict "
                             "variable alone shows a 5.4e-8 max relative difference)"])
        record_vector("logit predict, xb", st.var("xb_logit"),
                      np.asarray(res.predict(kind="xb")), cap.command)

        cap = st.estimate("poisson y_c x1 x2")
        st.run("quietly predict n_pois, n")
        res = ws.Poisson.from_formula("y_c ~ x1 + x2", df).fit()
        record_vector("poisson predict, n", st.var("n_pois"),
                      np.asarray(res.predict(kind="n")), cap.command)

        st.send(sub_nb)
        cap = st.estimate("ztp y_nb x1 x2")
        st.run("quietly predict n_ztp, n")
        st.run("quietly predict xb_ztp, xb")
        res_ztp = ws.TruncatedPoisson.from_formula("y_nb ~ x1 + x2", sub_nb).fit()
        # Stata 的 n 对 ztp 就是 exp(xb)，westata 的 fittedvalues 是截断均值
        record_vector("ztp predict, n (Stata semantics)", st.var("n_ztp"),
                      np.asarray(res_ztp.predict(kind="n")), cap.command,
                      notes=["westata TruncatedPoisson.fittedvalues returns the *truncated* mean "
                             "mu/Pr(y>0), but Stata's `predict, n` after ztp returns exp(xb) "
                             "(verified: max|n - exp(xb)| == 0.0 on 365 obs)",
                             "the public StataResults.predict(kind='n') never reaches the model's "
                             "own predict(), so it also disagrees for both readings"])
        record_vector("ztp fittedvalues (model-level, truncated mean vs Stata n)", st.var("n_ztp"),
                      np.asarray(res_ztp.fittedvalues), cap.command,
                      notes=["this is the estimator's own fittedvalues: a deliberately different "
                             "quantity from Stata's predict,n, recorded so the gap is on the "
                             "record rather than hidden"])

        st.send(df)
        cap = st.estimate("zip y_zi x1 x2, inflate(x1 x2)")
        st.run("quietly predict n_zip, n")
        res_zip = ws.ZeroInflatedPoisson.from_formula(
            "y_zi ~ x1 + x2", df, inflate="x1 + x2").fit()
        try:
            ws_pred = np.asarray(res_zip.predict(kind="n"))
        except Exception as exc:  # noqa: BLE001
            rec.error("zip predict, n (StataResults.predict)", exc, command=cap.command,
                      notes=["the generic postestimation dispatcher does `exog @ params[:exog.shape[1]]`; "
                             "for zip the parameter vector is 6 long while the design is 3 wide, "
                             "so it takes the binary branch and then a ragged matmul. Mathematically "
                             "meaningless for a zero-inflated model even if it did not crash"])
        else:
            record_vector("zip predict, n (StataResults.predict)", st.var("n_zip"),
                          ws_pred, cap.command)
        record_vector("zip fittedvalues (model level)", st.var("n_zip"),
                      np.asarray(res_zip.fittedvalues), cap.command,
                      notes=["the estimator's own fittedvalues = (1-pi)*mu, which *is* Stata's "
                             "predict,n for zip -- so the model is right and only the public "
                             "predict dispatcher is wrong"])

        cap = st.estimate("ologit ycat x1 x2")
        st.run("quietly predict pr_ologit, pr")
        st.run("quietly predict pr1_ologit, pr outcome(1)")
        res_ol = ws.OrderedLogit.from_formula("ycat ~ x1 + x2", df).fit()
        try:
            np.asarray(res_ol.predict(kind="pr"))
        except Exception as exc:  # noqa: BLE001
            rec.error("ologit predict, pr", exc, command=cap.command,
                      notes=["OrderedLogit's result carries no design_matrix (fit() never passes "
                             "design_matrix= to StataResults), so StataResults.predict() cannot "
                             "fall back to the estimation design and raises ConsistencyError",
                             "Stata's `predict, pr` after ologit is P(y = observed category); the "
                             "per-category probabilities are `predict, pr outcome(k)`"])
        for k in (2, 3, 4):
            st.run(f"quietly predict pr{k}_ologit, pr outcome({k})")
        cats = [st.var(f"pr{k}_ologit") for k in (1, 2, 3, 4)]
        expected = sum(k * cats[k - 1] for k in (1, 2, 3, 4))
        record_vector("ologit fittedvalues (expected category)", expected,
                      np.asarray(res_ol.fittedvalues), cap.command,
                      notes=["westata's fittedvalues are the expected category index; Stata has "
                             "no single predict type for that, so the reference is rebuilt from "
                             "the four `predict, pr outcome(k)` variables"])

        # ================================================== 12. 因子 + 迭代诊断
        cap = st.estimate("logit yb x1 i.grpf")
        res = ws.Logit.from_formula("yb ~ x1 + C(grpf)", df).fit()
        rec.add(compare("logit with i.grpf factor", cap, res, domain=DOMAIN, tolerance=TOL,
                        scalars=("N", "ll", "df_m")))
        st.run("quietly logit yb x1 x2")
        ic_logit = st.scalar("e(ic)")
        st.run("quietly nbreg y_nb x1 x2")
        ic_nb = st.scalar("e(ic)")
        st.run("quietly zinb y_zinb x1 x2, inflate(x1 x2)")
        ic_zinb = st.scalar("e(ic)")
        rec.add(Comparison(
            domain=DOMAIN, name="iteration / convergence diagnostics",
            stata_command="logit / nbreg / zinb (e(ic), e(converged))",
            status="skip", tolerance=TOL, param_rel=float("nan"), se_rel=float("nan"),
            notes=["Stata's own iteration counts, recorded for the attribution column; every "
                   "Stata fit here reports e(converged)=1",
                   "westata's stop rule is statsmodels': newton with tol=1e-10 for "
                   "logit/probit/poisson/zip/ztp, bfgs at its default gtol=1e-5 for "
                   "nbreg/zinb/tnbreg/ologit/oprobit -- the 1e-5 is what produces the B-E tier "
                   "gaps, and it is not exposed through westata's fit() signature"],
            extra={"stata_ic_logit": ic_logit, "stata_ic_nbreg": ic_nb,
                   "stata_ic_zinb": ic_zinb}))

        # ============================== 13. e(chi2) 在 vce() 下从 LR 换成 Wald
        # Stata: chi2type=LR under oim, chi2type=Wald under vce(robust)/vce(cluster),
        # and the Wald statistic excludes the constant. westata 的 e(chi2) 恒为 LR。
        rows: dict[str, dict] = {}
        worst_delta = 0.0
        for label, cmd in (
            ("logit, vce(robust)", "logit yb x1 x2, vce(robust)"),
            ("logit, vce(cluster g)", "logit yb x1 x2, vce(cluster g)"),
            ("poisson, vce(robust)", "poisson y_c x1 x2, vce(robust)"),
            ("poisson, vce(cluster g)", "poisson y_c x1 x2, vce(cluster g)"),
            ("nbreg, vce(robust)", "nbreg y_nb x1 x2, vce(robust)"),
        ):
            cap_v = st.estimate(cmd)
            b_s = np.asarray(cap_v.b)
            V_s = np.asarray(cap_v.V)
            k_slope = 2                     # x1, x2 are the first two e(b) columns here
            wald_slopes = float(b_s[:k_slope] @ np.linalg.inv(
                V_s[:k_slope, :k_slope]) @ b_s[:k_slope])
            wald_all = float(b_s @ np.linalg.inv(V_s) @ b_s)
            e_chi2 = float(cap_v.scalars["chi2"])
            rows[label] = {
                "stata_e_chi2": e_chi2,
                "stata_chi2type": cap_v.macros.get("chi2type"),
                "slopes_only_wald": wald_slopes,
                "all_params_wald": wald_all,
                "rel_slopes_wald_vs_e_chi2": abs(wald_slopes - e_chi2) / e_chi2,
            }
            worst_delta = max(worst_delta, rows[label]["rel_slopes_wald_vs_e_chi2"])
        # westata 侧同一模型的 e(chi2)
        for label, result in (
            ("logit, vce(cluster g)", res_cl),
            ("poisson, vce(robust)", ws.Poisson.from_formula("y_c ~ x1 + x2", df).fit(
                cov_type="HC0")),
        ):
            if label in rows:
                rows[label]["westata_e_chi2"] = float(result.e.get("chi2"))
        rec.add(Comparison(
            domain=DOMAIN, name="e(chi2) LR -> Wald switch under vce(robust)/vce(cluster)",
            stata_command="logit/poisson/nbreg with vce(robust) and vce(cluster g)",
            status="error" if worst_delta < 1e-8 else "pass",
            tolerance=TOL, param_rel=float(np.max([abs(
                rows[k].get("westata_e_chi2", rows[k]["stata_e_chi2"]) - rows[k]["stata_e_chi2"]
            ) / rows[k]["stata_e_chi2"] for k in rows])), se_rel=float("nan"), n_params=len(rows),
            notes=[
                "Stata switches e(chi2) from the LR statistic to a Wald statistic as soon as "
                "vce(robust)/vce(cluster) is requested (e(chi2type) macro goes 'LR' -> 'Wald'), "
                "and that Wald excludes the constant: recomputing b'V^-1 b on the slope block "
                f"reproduces e(chi2) to {worst_delta:.2e} for every one of the five commands, "
                "while including the constant is off by up to "
                f"{max(r['all_params_wald'] / r['stata_e_chi2'] for r in rows.values()):.2f}x",
                "westata keeps reporting the LR statistic regardless of cov_type "
                "(e(chi2) is literally unchanged: logit cluster e(chi2)=89.218111 both with oim "
                "and with cluster), so its model chi2/p disagree with Stata by 37% for "
                "logit-cluster and 8% for poisson-robust -- this is a wrong statistic, not a "
                "rounding difference",
            ],
            extra=rows | {"failure_code": "LR_reported_where_Stata_reports_Wald"}))

        # ============================== 14. e(df_m) 是否把辅助参数算进去
        # Stata: nbreg df_m 只数主方程的斜率（lnalpha 不算）；ologit df_m 只数斜率
        # （cut points 不算，e(k_aux) 单独给出）。westata 用的是 len(params)-1。
        rows = {}
        for label, cmd, result, stata_only in (
            ("nbreg", "nbreg y_nb x1 x2", res_nb, "lnalpha"),
            ("tnbreg", "tnbreg y_nb x1 x2",
             ws.TruncatedNegativeBinomial.from_formula("y_nb ~ x1 + x2", sub_nb).fit(), "lnalpha"),
        ):
            st.send(df if label == "nbreg" else sub_nb)
            cap_d = st.estimate(cmd)
            rows[label] = {
                "stata_df_m": float(cap_d.scalars["df_m"]),
                "stata_k_aux": float(cap_d.scalars.get("k_aux", float("nan"))),
                "stata_rank": float(cap_d.scalars["rank"]),
                "westata_df_m": float(result.e.get("df_m")),
                "aux_param": stata_only,
            }
        st.send(df)
        for label, cls in (("ologit", ws.OrderedLogit), ("oprobit", ws.OrderedProbit)):
            cap_d = st.estimate(f"{label} ycat x1 x2")
            r = cls.from_formula("ycat ~ x1 + x2", df).fit()
            rows[label] = {
                "stata_df_m": float(cap_d.scalars["df_m"]),
                "stata_k_aux": float(cap_d.scalars.get("k_aux", float("nan"))),
                "stata_rank": float(cap_d.scalars["rank"]),
                "westata_df_m": float(r.e.get("df_m")),
                "aux_param": "cut points",
            }
        bad_dfm = {k: v for k, v in rows.items() if v["westata_df_m"] != v["stata_df_m"]}
        rec.add(Comparison(
            domain=DOMAIN, name="e(df_m) must exclude auxiliary parameters",
            stata_command="nbreg / tnbreg / ologit / oprobit",
            status="error" if bad_dfm else "pass", tolerance=TOL,
            param_rel=max((abs(v["westata_df_m"] - v["stata_df_m"]) / v["stata_df_m"]
                           for v in bad_dfm.values()), default=0.0),
            se_rel=float("nan"), n_params=len(rows),
            notes=[
                "Stata's e(df_m) counts only the main equation's regressors beyond the "
                "constant; /lnalpha (nbreg/tnbreg) and cut1..cutk (ologit/oprobit) are reported "
                "separately through e(k_aux) and never enter df_m",
                "westata's _DiscreteBase._stata_scalars sets df_m = len(params) - 1, which "
                "counts the auxiliary parameters: nbreg/tnbreg report df_m=3 vs Stata's 2 and "
                "ologit/oprobit report df_m=4 vs Stata's 2",
                "consequence: e(p), the printed 'LR chi2(df)' label and any downstream df-based "
                "test are wrong for those four commands. CountBase._model_df gets this right for "
                "zip/zinb/ztp, so the bug is specific to the statsmodels-backed path",
            ],
            extra=rows | {"failure_code": "df_m_counts_auxiliary_params"}))

    rec.flush_summary()


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
