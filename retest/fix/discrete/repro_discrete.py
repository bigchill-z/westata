"""discrete 组 D1–D5 + 语义统一：复现、修复前/后与 Stata 的三列对比。

运行::

    cd D:\\myCodeProject\\statapy
    .venv\\Scripts\\python.exe retest/fix/discrete/repro_discrete.py

设计
----
* 数据与 `retest/domains/discrete/run.py` 逐位相同（seed 20261024, n=600），
  所以本文件的数字可以和领域报告那张 70 条表直接对。
* 每一节打印 **改前 / 改后 / Stata 三列**。"改前"取自本轮**现场用 `git stash`
  回到旧源码后重跑**的同一脚本（原始日志见 `before_after.txt`），不是凭报告抄的；
  D1 的旧算法（numpy 视图 + 未变换交叉块）另外在 `_legacy_to_cut_points` 里
  原样重实现，任何人在当前代码上也能复算这一列。
* 判据按 BRIEF §2.2：A ≤1e-9、B ≤1e-6、C ≤1e-3、E 更大。
* `rel()` 的分母带 1e-12 下限（与领域报告一致），否则 `e(p)` 这类 1e-19 的量
  会因末位噪声报出"巨大相对差"。

结论摘要见同目录 `FIXES.md`；本文件只负责把表里的每个数字重算一遍。
"""

from __future__ import annotations

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))

import numpy as np
import pandas as pd

import westata as ws
from _data import align_stata, make_data, rel, tier
from stata_probe import Stata

warnings.simplefilter("ignore")

#: (item, quantity, before, after, tier) -- filled by ``report``, printed last.
ROWS: list[tuple[str, str, str, float, str]] = []
FAILURES: list[str] = []


def report(item: str, quantity: str, before: str | float, after: float) -> float:
    """Print one ``before / after / Stata`` line and remember it for the summary."""

    band = tier(after)
    ROWS.append((item, quantity, _fmt(before), after, band))
    print(f"    {quantity:<32s} before={_fmt(before):<18s} after={after:<11.3e} {band}")
    return after


def _fmt(value: str | float) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.3e}"


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)
        print(f"    !! {message}")


# --------------------------------------------------------------- legacy copy
def _legacy_to_cut_points(params, cov, n_cuts):
    """The pre-fix ``_to_cut_points``, verbatim, for the "before" column.

    Two bugs are deliberately preserved: ``theta`` aliases ``params`` (so the
    assignment overwrites it and ``exp`` lands on cut points), and the
    slope-by-cut cross-block is never touched.
    """

    params = np.asarray(params, dtype=float).copy()
    cov = np.asarray(cov, dtype=float).copy()
    n_slopes = params.size - n_cuts
    theta = params[n_slopes:]  # a VIEW -- this is bug (1)
    cuts = np.empty(n_cuts, dtype=float)
    cuts[0] = theta[0]
    for k in range(1, n_cuts):
        cuts[k] = cuts[k - 1] + np.exp(theta[k])
    params[n_slopes:] = cuts
    jacobian = np.eye(n_cuts)
    for k in range(1, n_cuts):
        jacobian[k, :k] = np.exp(theta[1 : k + 1])
    cov[n_slopes:, n_slopes:] = jacobian @ cov[n_slopes:, n_slopes:] @ jacobian.T
    return params, cov


# --------------------------------------------------------------------- D1
def section_d1(st, df) -> None:
    """D1: ologit/oprobit e(V) -- both defects, on the verifier's reference data."""

    print("\n=== D1  ologit / oprobit 的 e(V)（verifier 参考数据，cut SE 应为 0.0874/0.0962）")
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    # Exactly the verifier's DGP (retest/domains/verify/p1L_ologit_final.py), so
    # the reference cut SEs quoted in BRIEF §4.3 are reproduced bit for bit.
    rng = np.random.default_rng(20261024)
    n = 600
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    eta = 0.5 + 0.8 * x1 - 0.4 * x2 + rng.logistic(size=n)
    ref_df = pd.DataFrame(
        {"x1": x1, "x2": x2, "ycat": np.digitize(eta, [-1.0, 0.3, 1.6]).astype(float)}
    )
    st.send(ref_df)
    y = ref_df["ycat"].to_numpy(float)
    X = np.column_stack([ref_df["x1"], ref_df["x2"]])

    for command, distr, cls in (
        ("ologit ycat x1 x2", "logit", ws.OrderedLogit),
        ("oprobit ycat x1 x2", "probit", ws.OrderedProbit),
    ):
        cap = st.estimate(command)
        backend = OrderedModel(y, X, distr=distr).fit(
            method="bfgs", maxiter=20000, disp=0, gtol=1e-10
        )
        theta = np.asarray(backend.params, dtype=float)
        backend_cov = np.asarray(backend.cov_params(), dtype=float)
        _, legacy_cov = _legacy_to_cut_points(theta, backend_cov, 3)

        result = cls.from_formula("ycat ~ x1 + x2", ref_df).fit()
        cross = np.asarray(result.cov_params())[:2, 2:]

        # Bug (2) is visible without any Stata number: the old code leaves the
        # slope-by-cut block bit-identical to the backend's, i.e. untransformed.
        print(
            f"  [{command}] 缺陷②证据: 旧代码的交叉块与输入逐位相同 = "
            f"{np.array_equal(legacy_cov[:2, 2:], backend_cov[:2, 2:])}"
        )
        report(command, "cut SE（旧算法）", "目标 0.0874/0.0962",
               rel(np.sqrt(np.diag(legacy_cov))[2:], np.sqrt(np.diag(cap.V))[2:]))
        report(command, "cut SE（修复后）", "—",
               rel(np.sqrt(np.diag(result.cov_params()))[2:], np.sqrt(np.diag(cap.V))[2:]))
        report(command, "slope×cut 交叉块", "未变换(=源矩阵)", rel(cross, cap.V[:2, 2:]))
        full = report(command, "e(V) 全矩阵", "2.09e+01 / 1.05e+01",
                      rel(result.cov_params(), cap.V))
        report(command, "e(V) 全矩阵（旧算法复算）", "—", rel(legacy_cov, cap.V))
        report(command, "系数", "3.55e-05 / 3.80e-06", rel(result.params, cap.b))
        print(
            f"      cut SE westata={np.round(np.sqrt(np.diag(result.cov_params()))[2:], 8).tolist()}\n"
            f"      cut SE Stata  ={np.round(np.sqrt(np.diag(cap.V))[2:], 8).tolist()}"
        )
        # ologit reaches A.  oprobit is B, and not because of our Hessian:
        # evaluated *at Stata's own b* it is 2.5e-14, so the residual is Stata's
        # optimiser stopping ~1e-8 short of the gradient root on this surface
        # (|g| = 9e-06 at Stata's b vs 2e-14 at ours).  Recorded, not hidden.
        limit = 1e-9 if cls is ws.OrderedLogit else 1e-6
        check(full <= limit, f"{command}: e(V) 超预算（{full:.3e} > {limit:g}）")

    # The result must not be tuned to one dataset.
    print("  多数据集复核（另一 DGP——4 类有序、probit 生成）：")
    rng = np.random.default_rng(909)
    other = pd.DataFrame({
        "y": (pd.cut(0.6 * rng.normal(size=450) - 0.4 * rng.normal(size=450)
                     + rng.normal(size=450),
                     [-np.inf, -0.8, 0.2, 1.0, np.inf], labels=False) + 1).astype(float),
        "x1": rng.normal(size=450), "x2": rng.normal(size=450),
    })
    st.send(other)
    for command, cls in (("ologit y x1 x2", ws.OrderedLogit),
                         ("oprobit y x1 x2", ws.OrderedProbit)):
        cap = st.estimate(command)
        result = cls.from_formula("y ~ x1 + x2", other).fit()
        value = rel(result.cov_params(), cap.V)
        print(f"    {command:22s} e(V) rel={value:.3e} ({tier(value)})")
        check(value <= 1e-6, f"{command} (第二 DGP): e(V) {value:.3e}")
    st.send(df)


# ----------------------------------------------------------------- D2 / D3
def section_d2_d3(st, df) -> None:
    """D2: vce(robust)/vce(cluster) usable.  D3: HC1 really differs from HC0."""

    print("\n=== D2/D3  稳健 VCE 可用性与 HC0/HC1 的区分")
    g = df["g"].to_numpy()
    specs = [
        ("logit yb x1 x2", ws.Logit, "yb", [2, 0, 1]),
        ("probit yb x1 x2", ws.Probit, "yb", [2, 0, 1]),
        ("poisson y_c x1 x2", ws.Poisson, "y_c", [2, 0, 1]),
        ("nbreg y_nb x1 x2", ws.NegativeBinomial, "y_nb", [2, 0, 1, 3]),
    ]
    for label, cls, lhs, perm in specs:
        print(f"  --- {label}")
        for suffix, cov_type, kw, before in (
            ("", "nonrobust", {}, "1.31e-07"),
            (", vce(robust)", "robust", {}, "ValueError（不可用）"),
            (", vce(cluster g)", "cluster", {"groups": g}, "3.35e-03"),
        ):
            cap = st.estimate(label + suffix)
            _, stata_V = align_stata(perm, cap.b, cap.V)
            fit = cls.from_formula(f"{lhs} ~ x1 + x2", df).fit(
                cov_type=cov_type, cov_kwds=kw
            )
            value = rel(fit.cov_params(), stata_V)
            se_value = rel(fit.bse, np.sqrt(np.diag(stata_V)))
            report(f"{label}{suffix}", "e(V) 全矩阵", before, value)
            report(f"{label}{suffix}", "SE（e(V) 对角开方）", before, se_value)
            check(se_value <= 1e-6, f"{label}{suffix}: SE {se_value:.3e}")
            # The full matrix is only demanded at 1e-6 where the estimator has an
            # analytic Hessian.  nbreg goes through statsmodels' analytic
            # NegativeBinomial Hessian in alpha space plus a delta transform, and
            # its /lnalpha block's off-diagonals are the limit (~1.2e-05): the
            # SEs (what Stata prints and tests use) are 1.8e-08.
            matrix_limit = 1e-6 if cls is not ws.NegativeBinomial else 1e-4
            check(value <= matrix_limit, f"{label}{suffix}: e(V) {value:.3e}")

        # D3: HC1 must not silently equal HC0.
        hc0 = cls.from_formula(f"{lhs} ~ x1 + x2", df).fit(cov_type="HC0")
        hc1 = cls.from_formula(f"{lhs} ~ x1 + x2", df).fit(cov_type="HC1")
        identical = np.array_equal(hc0.cov_params(), hc1.cov_params())
        ratio = len(df) / (len(df) - len(hc0.params))
        value = rel(hc1.cov_params(), np.asarray(hc0.cov_params()) * ratio)
        print(
            f"    HC0 与 HC1 逐位相同? 改前=True（静默等价） 改后={identical} "
            f"| HC1 / (N/(N-K)) vs HC0 = {value:.3e} ({tier(value)})"
        )
        ROWS.append((label, "HC0 == HC1（静默）", "True", 0.0 if identical else 1.0,
                     "FAIL" if identical else "pass"))
        ROWS.append((label, "HC1 == N/(N-K) * HC0", "无此因子", value, tier(value)))
        check(not identical, f"{label}: HC1 仍静默等于 HC0")
        check(value <= 1e-12, f"{label}: HC1 不是 N/(N-K) 倍 HC0（{value:.3e}）")

        # Semantics: HC0 must NOT be Stata's vce(robust); robust must be.
        cap = st.estimate(label + ", vce(robust)")
        _, stata_V = align_stata(perm, cap.b, cap.V)
        hc0_gap = rel(hc0.cov_params(), stata_V)
        print(
            f"    HC0 vs Stata vce(robust): {hc0_gap:.2e}"
            "（HC0 不带 N/(N-1)，故与 Stata 的 ml robust 不同）"
        )


# --------------------------------------------------------------------- D4
def section_d4(st, df) -> None:
    """D4: e(chi2) switches LR -> slope-block Wald under robust/cluster."""

    print("\n=== D4  e(chi2)/e(chi2type) 随 VCE 从 LR 切到斜率块 Wald")
    g = df["g"].to_numpy()
    specs = [
        ("logit yb x1 x2", ws.Logit, "yb"),
        ("probit yb x1 x2", ws.Probit, "yb"),
        ("poisson y_c x1 x2", ws.Poisson, "y_c"),
        ("nbreg y_nb x1 x2", ws.NegativeBinomial, "y_nb"),
        ("ologit ycat x1 x2", ws.OrderedLogit, "ycat"),
        ("oprobit ycat x1 x2", ws.OrderedProbit, "ycat"),
    ]
    for label, cls, lhs in specs:
        print(f"  --- {label}")
        for suffix, cov_type, kw, before in (
            ("", "nonrobust", {}, "—"),
            (", vce(robust)", "robust", {}, "3.72e-01（未切换）"),
            (", vce(cluster g)", "cluster", {"groups": g}, "3.72e-01（未切换）"),
        ):
            cap = st.estimate(label + suffix)
            fit = cls.from_formula(f"{lhs} ~ x1 + x2", df).fit(
                cov_type=cov_type, cov_kwds=kw
            )
            value = rel(fit.e["chi2"], cap.scalars["chi2"])
            type_ok = fit.e["chi2type"] == cap.macros.get("chi2type")
            report(f"{label}{suffix}", f"e(chi2) [type={fit.e['chi2type']}]", before, value)
            check(type_ok, f"{label}{suffix}: chi2type {fit.e['chi2type']!r} != "
                           f"{cap.macros.get('chi2type')!r}")
            check(value <= 1e-6, f"{label}{suffix}: e(chi2) {value:.3e}")


# --------------------------------------------------------------------- D5
def section_d5(st, df) -> None:
    """D5: e(df_m) excludes auxiliary parameters; e(k_aux) reports them."""

    print("\n=== D5  e(df_m) / e(k_aux) / e(rank)")
    before = {
        "logit yb x1 x2": 2.0,
        "probit yb x1 x2": 2.0,
        "poisson y_c x1 x2": 2.0,
        "nbreg y_nb x1 x2": 3.0,
        "ologit ycat x1 x2": 4.0,
        "oprobit ycat x1 x2": 4.0,
    }
    for label, cls, lhs in (
        ("logit yb x1 x2", ws.Logit, "yb"),
        ("probit yb x1 x2", ws.Probit, "yb"),
        ("poisson y_c x1 x2", ws.Poisson, "y_c"),
        ("nbreg y_nb x1 x2", ws.NegativeBinomial, "y_nb"),
        ("ologit ycat x1 x2", ws.OrderedLogit, "ycat"),
        ("oprobit ycat x1 x2", ws.OrderedProbit, "ycat"),
    ):
        cap = st.estimate(label)
        fit = cls.from_formula(f"{lhs} ~ x1 + x2", df).fit()
        stata = {k: cap.scalars.get(k) for k in ("df_m", "rank", "k_aux")}
        ours = {k: fit.e.get(k) for k in ("df_m", "rank", "k_aux")}
        print(
            f"    {label:<22s} df_m 改前={before[label]:.0f} 改后={ours['df_m']} "
            f"Stata={stata['df_m']:.0f} | rank {ours['rank']}/{stata['rank']:.0f} "
            f"| k_aux {ours['k_aux']}/{stata['k_aux']}"
        )
        ROWS.append((label, "e(df_m) vs Stata", f"{before[label]:.0f}",
                     float(ours["df_m"]), "pass" if ours["df_m"] == stata["df_m"] else "FAIL"))
        check(ours["df_m"] == stata["df_m"], f"{label}: df_m {ours['df_m']} != {stata['df_m']}")
        check(ours["rank"] == stata["rank"], f"{label}: rank 不一致")
        check(ours["k_aux"] == stata["k_aux"], f"{label}: k_aux 不一致")


# ---------------------------------------------------------------- predict
def section_predict(st, df) -> None:
    """zip/zinb/ztp/tnbreg predict n/pr/xb, through the public results API."""

    print("\n=== predict  n / pr / xb（走 StataResults.predict，即 postest 的委托面）")
    st.quiet("zip y_zi x1 x2, inflate(x1 x2)")
    zip_fit = ws.ZeroInflatedPoisson.from_formula(
        "y_zi ~ x1 + x2", df, inflate="x1 + x2"
    ).fit()
    for kind in ("n", "pr", "xb", "stdp"):
        st.run(f"cap drop p_{kind}")
        st.run(f"predict double p_{kind}, {kind}")
        value = rel(zip_fit.predict(kind=kind), st.var(f"p_{kind}"))
        report("zip", f"predict, {kind}", "崩溃 (matmul 维数)", value)
        # ``xb`` is a plain linear combination, so its relative error is the
        # coefficient error amplified by cancellation near zero; it is bounded
        # by the coefficients (2.4e-07), not by anything predict-specific.
        check(value <= 1e-4, f"zip predict,{kind}: {value:.3e}")
    print(
        "      注: zip 的 pr 是零膨胀概率 pi（不是 P(y=0)），n 是 (1-pi)*mu；"
        "两者的 Stata 对照已在上面两行给出"
    )

    st.quiet("zinb y_zi x1 x2, inflate(x1 x2)")
    zinb_fit = ws.ZeroInflatedNegativeBinomial.from_formula(
        "y_zi ~ x1 + x2", df, inflate="x1 + x2"
    ).fit()
    for kind in ("n", "pr", "xb"):
        st.run(f"cap drop q_{kind}")
        st.run(f"predict double q_{kind}, {kind}")
        value = rel(zinb_fit.predict(kind=kind), st.var(f"q_{kind}"))
        report("zinb", f"predict, {kind}", "崩溃 (matmul 维数)", value)
        check(value <= 1e-3, f"zinb predict,{kind}: {value:.3e}")

    pos = df[df["y_nb"] > 0].reset_index(drop=True)
    st.send(pos)
    for command, cls, model in (
        ("ztp y_nb x1 x2", ws.TruncatedPoisson, "ztp"),
        ("tnbreg y_nb x1 x2, ll(0)", ws.TruncatedNegativeBinomial, "tnbreg"),
    ):
        st.quiet(command)
        fit = cls.from_formula("y_nb ~ x1 + x2", pos, lower=1).fit()
        for kind in ("n", "xb", "stdp"):
            st.run(f"cap drop r_{kind}")
            st.run(f"predict double r_{kind}, {kind}")
            value = rel(fit.predict(kind=kind), st.var(f"r_{kind}"))
            report(model, f"predict, {kind}", "语义错 (截断均值)", value)
            check(value <= 1e-3, f"{model} predict,{kind}: {value:.3e}")
        try:
            fit.predict(kind="pr")
            check(False, f"{model}: predict, pr 应拒绝（Stata r(198)）")
        except Exception as exc:  # noqa: BLE001
            print(f"    {model:<8s} predict, pr 正确拒绝：{type(exc).__name__}")
    st.send(df)


# ---------------------------------------------------------------- e() 标量
def section_scalars(st, df) -> None:
    """e(ll)/e(ll_0)/e(r2_p)/e(N_clust) under the robust VCEs."""

    print("\n=== e() 派生标量（robust / cluster 下）")
    g = df["g"].to_numpy()
    for label, cls, lhs in (
        ("logit yb x1 x2", ws.Logit, "yb"),
        ("nbreg y_nb x1 x2", ws.NegativeBinomial, "y_nb"),
        ("ologit ycat x1 x2", ws.OrderedLogit, "ycat"),
    ):
        for suffix, cov_type, kw in (
            (", vce(robust)", "robust", {}),
            (", vce(cluster g)", "cluster", {"groups": g}),
        ):
            cap = st.estimate(label + suffix)
            fit = cls.from_formula(f"{lhs} ~ x1 + x2", df).fit(
                cov_type=cov_type, cov_kwds=kw
            )
            worst = max(
                abs(fit.e[key] - cap.scalars[key]) / max(abs(cap.scalars[key]), 1e-12)
                for key in ("ll", "ll_0", "r2_p")
            )
            n_clust_ok = fit.e.get("N_clust") == cap.scalars.get("N_clust")
            report(f"{label}{suffix}", "e(ll)/e(ll_0)/e(r2_p)", "—", worst)
            print(
                f"      N_clust westata={fit.e.get('N_clust')} "
                f"Stata={cap.scalars.get('N_clust')} {'OK' if n_clust_ok else 'MISMATCH'}"
            )
            check(n_clust_ok, f"{label}{suffix}: N_clust 不一致")
            check(worst <= 1e-9, f"{label}{suffix}: e() 标量 {worst:.3e}")


def main() -> int:
    df = make_data()
    with Stata() as st:
        st.run("set type double")
        st.send(df)
        print(f"StataNow {st.version} / westata {ws.__version__}")
        section_d1(st, df)
        section_d2_d3(st, df)
        section_d4(st, df)
        section_d5(st, df)
        section_predict(st, df)
        section_scalars(st, df)

    print("\n=== 汇总表（after=相对差；A ≤1e-9、B ≤1e-6、C ≤1e-3）===")
    print(f"{'项':<38s} {'量':<32s} {'改前':<22s} {'改后':<12s} 档")
    for item, quantity, before, after, band in ROWS:
        print(f"{item:<38s} {quantity:<32s} {before:<22s} {after:<12.3e} {band}")

    if FAILURES:
        print(f"\n未通过 {len(FAILURES)} 项：")
        for message in FAILURES:
            print("  -", message)
        return 1
    print("\n全部断言通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
