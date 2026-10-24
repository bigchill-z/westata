"""panel 组 P1-P5 的可重跑复现 + 验证脚本（真实 StataNow 19.5）。

    cd D:\\myCodeProject\\statapy
    .venv\\Scripts\\python.exe retest/fix/panel/repro_panel.py

每一项打印 改前 / 改后 / Stata 三列与相对差。改前的数取自修复前的实现
（FIXES.md 里写清了每项根因），脚本以常量记录以便对照。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import westata as ws  # noqa: E402
from stata_probe import Stata  # noqa: E402

CAPTURED = "2026-10-24"


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def balanced(seed: int = 2026, n: int = 30, T: int = 6) -> pd.DataFrame:
    """y_it = 1 + .8 x1 - .5 x2 + u_i + e_it；yc 为泊松计数（xtnbreg 用）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        u = rng.normal()
        for t in range(T):
            x1, x2, z = rng.normal(), rng.normal(), rng.normal()
            y = 1.0 + 0.8 * x1 - 0.5 * x2 + u + rng.normal(scale=0.5)
            yc = int(rng.poisson(np.exp(0.5 + 0.4 * x1 - 0.2 * x2 + 0.3 * u)))
            rows.append((i, t, y, x1, x2, z, yc))
    return pd.DataFrame(rows, columns=["id", "t", "y", "x1", "x2", "z", "yc"])


def unbalanced(seed: int = 777, n: int = 20, T: int = 8) -> pd.DataFrame:
    """t>=3 之后逐期 35% 概率缺失；t=0,1,2 全观测（casewise 共同期非空）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        u = rng.normal()
        for t in range(T):
            if t >= 3 and rng.random() < 0.35:
                continue
            x1, x2 = rng.normal(), rng.normal()
            y = 1.0 + 0.8 * x1 - 0.5 * x2 + u + rng.normal(scale=0.5)
            rows.append((i, t, y, x1, x2))
    return pd.DataFrame(rows, columns=["id", "t", "y", "x1", "x2"])


def dynamic(seed: int = 5150, n: int = 60, T: int = 8, a: float = 0.5) -> pd.DataFrame:
    """y_it = a y_i,t-1 + .8 x1 - .4 x2 + u_i + e_it，y_i0 = 0。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        u = rng.normal()
        y = 0.0
        for t in range(T):
            x1, x2 = rng.normal(), rng.normal()
            y = a * y + 0.8 * x1 - 0.4 * x2 + u + rng.normal(scale=0.5)
            rows.append((i, t, y, x1, x2))
    return pd.DataFrame(rows, columns=["id", "t", "y", "x1", "x2"])


def report(tag: str, before, after, stata, note: str = "") -> None:
    print(
        f"  {tag:32s} 改前/Stata {rel(before, stata):.3e}   改后/Stata {rel(after, stata):.3e}"
        + (f"   {note}" if note else "")
    )


# --------------------------------------------------------------------- P1 / P2
def defect_p1_p2() -> None:
    print("\n=== P1/P2: xtreg, fe 的 vce(robust) / vce(cluster) 语义与小样本因子")
    df = balanced()
    with Stata() as st:
        st.send(df)
        st.run("xtset id t")
        robust = st.estimate("xtreg y x1 x2, fe vce(robust)")
        cluster = st.estimate("xtreg y x1 x2, fe vce(cluster id)")
        dfadj = st.estimate("xtreg y x1 x2, fe vce(cluster id) dfadj")
        areg = st.estimate("areg y x1 x2, absorb(id) vce(robust)")
    order = ["x1", "x2", "Intercept"]  # westata 的常数项名
    stata_se = np.array(
        [
            np.sqrt(np.diag(robust.V))[robust.names.index("_cons" if n == "Intercept" else n)]
            for n in order
        ]
    )
    print(
        f"  Stata max|V_robust - V_cluster(id)| = {np.max(np.abs(robust.V - cluster.V)):.1f}"
        "   (严格 0：vce(robust) 就是 cluster(panelvar))"
    )
    print(
        f"  Stata max|V_cluster - V_cluster dfadj| = {np.max(np.abs(cluster.V - dfadj.V)):.3e}"
        "   (dfadj 才是把被吸收效应计入 K 的口径)"
    )

    model = ws.PanelOLS.from_formula("y ~ x1 + x2", df, entity="id", time="t")
    now = model.fit(cov_type="robust")
    now_se = np.array([now.bse[n] for n in order])
    # 改前：cov_type="robust" 映射成 HC1，等价于 Stata 的 areg + vce(robust)。
    before_se = np.array(
        [
            np.sqrt(np.diag(areg.V))[areg.names.index("_cons" if n == "Intercept" else n)]
            for n in order
        ]
    )
    report("se(vce(robust))", before_se, now_se, stata_se, "改前=areg 的 HC1（常数项差 18.4 倍）")
    print(f"    改前 se={np.round(before_se, 7)}")
    print(f"    改后 se={np.round(now_se, 7)}")
    print(f"    Stata  se={np.round(stata_se, 7)}")
    print(f"    e(df_r) 改后 {now.df_resid} vs Stata {robust.scalars.get('df_r')}")

    clustered = model.fit(cov_type="cluster", cov_kwds={"clusters": df["id"].to_numpy()})
    cl_se = np.array([clustered.bse[n] for n in order])
    report("se(vce(cluster id))", before_se, cl_se, stata_se)
    print(f"    e(df_r) 改后 {clustered.df_resid} vs Stata {cluster.scalars.get('df_r')}")
    print(
        f"    e(rmse) 改后 {clustered.e['rmse']:.10f} vs Stata {cluster.scalars.get('rmse'):.10f}"
    )


# -------------------------------------------------------------------------- P3
def defect_p3() -> None:
    print("\n=== P3: xtpcse 非平衡（casewise / pairwise）")
    df = unbalanced()
    with Stata() as st:
        st.send(df)
        st.run("xtset id t")
        cw = st.estimate("xtpcse y x1 x2")
        pw = st.estimate("xtpcse y x1 x2, pairwise")
    names = ["Intercept", "x1", "x2"]
    # 旧实现（_aligned_slices 按行号截断、pairwise 未实现）：两种 selection 同值。
    before_se = np.array([0.0304356, 0.0409808, 0.0342562])
    for label, cap, selection in (("casewise", cw, "casewise"), ("pairwise", pw, "pairwise")):
        res = ws.XtPcse.from_formula(
            "y ~ x1 + x2", df, entity="id", time="t", selection=selection
        ).fit()
        stata_se = np.array(
            [
                np.sqrt(np.diag(cap.V))[cap.names.index("_cons" if n == "Intercept" else n)]
                for n in names
            ]
        )
        report(
            f"se({label})",
            before_se,
            np.asarray(res.bse),
            stata_se,
            "改前 pairwise 与 casewise 逐位相同",
        )
        print(
            f"    e(n_sigma) 改后 {res.e.get('n_sigma')} vs Stata {cap.scalars.get('n_sigma')}"
        )
    # 平衡面板不能改坏。
    bal = balanced()
    with Stata() as st:
        st.send(bal)
        st.run("xtset id t")
        cap = st.estimate("xtpcse y x1 x2")
    res = ws.XtPcse.from_formula("y ~ x1 + x2", bal, entity="id", time="t").fit()
    stata_se = np.array(
        [
            np.sqrt(np.diag(cap.V))[cap.names.index("_cons" if n == "Intercept" else n)]
            for n in names
        ]
    )
    print(
        f"  平衡面板回归检查：se 相对差 {rel(np.asarray(res.bse), stata_se):.3e}"
        f"   e(V) 全矩阵 max|差| {np.max(np.abs(res.cov[np.ix_([1, 2, 0], [1, 2, 0])] - cap.V)):.3e}"
    )


# -------------------------------------------------------------------------- P4
def defect_p4() -> None:
    print("\n=== P4: xtabond 的 e(sig2) 与 lags(2) 参数名")
    df = dynamic()
    with Stata() as st:
        st.send(df)
        st.run("xtset id t")
        one = st.estimate("xtabond y x1 x2, lags(1)")
        one_r = st.estimate("xtabond y x1 x2, lags(1) vce(robust)")
        two = st.estimate("xtabond y x1 x2, lags(2)")
    for lags, cap in ((1, one), (2, two)):
        res = ws.XtAbond.from_formula("y ~ x1 + x2", df, entity="id", time="t", lags=lags).fit()
        print(f"  lags({lags}) param_names = {list(res.param_names)}")
        stata_names = ["L.y"] + (["L2.y"] if lags == 2 else []) + ["x1", "x2"]
        idx = [cap.names.index(n) for n in stata_names]
        print(
            f"    系数相对差 {rel(np.asarray(res.params), cap.b[idx]):.3e}"
            f"   SE(gmm) 相对差 {rel(np.asarray(res.bse), np.sqrt(np.diag(cap.V))[idx]):.3e}"
        )
        print(
            f"    e(sig2) 改后 {res.e['sig2']:.12f} vs Stata {cap.scalars.get('sig2'):.12f}"
            f"   相对差 {abs(res.e['sig2'] - cap.scalars['sig2']) / cap.scalars['sig2']:.3e}"
        )
        print(f"    e(rss)  改后 {res.e['rss']:.10f} vs Stata {cap.scalars.get('rss'):.10f}")
        print(
            f"    改前 e(sig2)（未除 2）= {2 * res.e['sig2']:.12f}"
            f"   相对差 {abs(2 * res.e['sig2'] - cap.scalars['sig2']) / cap.scalars['sig2']:.3f}"
            "   <- 旧缺陷 1.993 倍"
        )
    resr = ws.XtAbond.from_formula("y ~ x1 + x2", df, entity="id", time="t", lags=1).fit(
        cov_type="robust"
    )
    idx = [one_r.names.index(n) for n in ("L.y", "x1", "x2")]
    print(
        f"  lags(1) vce(robust) SE 相对差 "
        f"{rel(np.asarray(resr.bse), np.sqrt(np.diag(one_r.V))[idx]):.3e}"
    )
    print(
        "  lags(2) 旧实现的 param_names = ['L2.y', 'x1', 'x2', 'extra_0']，"
        "真值 ['L1.y','L2.y','x1','x2'] —— 按名取系数会静默取错变量"
    )


# -------------------------------------------------------------------------- P5
def defect_p5() -> None:
    print("\n=== P5: xtnbreg, fe 的 HHG 拟似然")
    df = balanced()
    with Stata() as st:
        st.send(df)
        st.run("xtset id t")
        cap = st.estimate("xtnbreg yc x1 x2, fe")
        tight = st.estimate("xtnbreg yc x1 x2, fe ltolerance(1e-8)")
        poisson = st.estimate("xtpoisson yc x1 x2, fe")
    res = ws.XtNbreg.from_formula("yc ~ x1 + x2", df, entity="id", method="fe").fit()
    names = ["x1", "x2"]
    stata_b = np.array([cap.b[cap.names.index(n)] for n in names])
    stata_se = np.array([np.sqrt(np.diag(cap.V))[cap.names.index(n)] for n in names])
    tight_b = np.array([tight.b[tight.names.index(n)] for n in names])
    ours_b = np.array([res.params[n] for n in names])
    ours_se = np.array([res.bse[n] for n in names])
    # 改前：条件 Poisson（与 xtpoisson, fe 逐位相同）
    before_b = np.array([poisson.b[poisson.names.index(n)] for n in names])
    before_se = np.array([np.sqrt(np.diag(poisson.V))[poisson.names.index(n)] for n in names])
    report(
        "系数 vs Stata 默认",
        before_b,
        ours_b,
        stata_b,
        f"改后 vs Stata ltolerance(1e-8): {rel(ours_b, tight_b):.3e}",
    )
    report("SE   vs Stata 默认", before_se, ours_se, stata_se)
    print(f"    改前 b={np.round(before_b, 8)}（= xtpoisson, fe）")
    print(f"    改后 b={np.round(ours_b, 8)}")
    print(f"    Stata  b={np.round(stata_b, 8)}   tight b={np.round(tight_b, 8)}")
    print(f"    改前 se={np.round(before_se, 8)}")
    print(f"    改后 se={np.round(ours_se, 8)}")
    print(f"    Stata  se={np.round(stata_se, 8)}")
    print(
        f"    e(ll) 改后 {res.e['ll']:.10f} vs Stata {cap.scalars.get('ll'):.10f}"
        f"   相对差 {abs(res.e['ll'] - cap.scalars['ll']) / abs(cap.scalars['ll']):.3e}"
    )
    print(f"    estimator = {res.e['estimator']}，_cons 保留 = {'_cons' in res.param_names}")


def main() -> None:
    print(f"westata panel 修复复核（Stata 真值捕获日 {CAPTURED}）")
    defect_p1_p2()
    defect_p3()
    defect_p4()
    defect_p5()
    print("\n说明：改前数值取自修复前的实现（根因见 FIXES.md）；改后数值与 Stata 列")
    print("      都由本脚本现场重算 / 现场启动的 StataNow 19.5 给出。")


if __name__ == "__main__":
    main()
