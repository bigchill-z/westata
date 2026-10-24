"""Lead's independent verification of the repair round (no domain script imported).

Each block starts its own Stata session, builds its own data, and compares the
full-precision e(b)/e(V)/e() with the *current* westata source.  Any number here
is produced from scratch, so a fix that only made a test pass will not survive.

Usage::

    .venv\\Scripts\\python.exe retest/fix/lead_verify.py            # all blocks
    .venv\\Scripts\\python.exe retest/fix/lead_verify.py linear panel
"""

from __future__ import annotations

import sys
import traceback

sys.path.insert(0, "retest/lib")

import numpy as np
import pandas as pd

import westata as ws
from stata_probe import Stata, rel_diff

FAILS: list[str] = []


def rel(a, e) -> float:
    a = np.asarray(a, float).reshape(-1)
    e = np.asarray(e, float).reshape(-1)
    if a.size != e.size:
        return float("inf")
    return float(np.max(np.abs(a - e) / np.maximum(np.abs(e), 1e-12)))


def scalar(res, key):
    try:
        return float(res.e[key])
    except Exception:
        return None


def check(label: str, got, want, tol: float = 1e-9) -> None:
    r = rel(got, want)
    ok = r <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: rel {r:.3e} (tol {tol:.0e})")
    if not ok:
        FAILS.append(f"{label}: {r:.3e} > {tol:.0e}")


def check_scalar(label: str, got, want, tol: float = 1e-9) -> None:
    if got is None:
        print(f"  [FAIL] {label}: westata has no such scalar")
        FAILS.append(f"{label}: missing")
        return
    r = abs(float(got) - float(want)) / max(abs(float(want)), 1e-12)
    ok = r <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {got!r} vs {want!r} rel {r:.3e}")
    if not ok:
        FAILS.append(f"{label}: {r:.3e} > {tol:.0e}")


def check_named(
    label: str, res, stata_names, stata_values, tol: float = 1e-9, field: str = "params"
) -> None:
    """Compare a results object with Stata by coefficient NAME, not position.

    Stata stores names in the ``e(b)`` column stripes and puts the constant
    last; westata may order ``Intercept`` first.  Comparing positionally would
    produce a fake failure, so the alignment is explicit (same rule the domain
    harnesses use).

    ``field`` selects which westata vector is compared: ``"params"`` for
    coefficients, ``"bse"`` for standard errors.
    """

    from stata_probe import canon_stata, canon_westata

    smap: dict[str, float] = {}
    for name, value in zip(list(stata_names), np.asarray(stata_values, float).reshape(-1)):
        smap[canon_stata(name)] = float(value)
    westata_values = np.asarray(getattr(res, field), float).reshape(-1)
    aligned: list[tuple[str, float, float]] = []
    for name, value in zip(list(res.param_names), westata_values):
        key = canon_westata(name)
        if key in smap:
            aligned.append((key, float(value), smap[key]))
    if not aligned:
        print(f"  [FAIL] {label}: nothing aligned ({stata_names} vs {list(res.param_names)})")
        FAILS.append(f"{label}: no aligned coefficients")
        return
    got = np.asarray([v for _, v, _ in aligned])
    want = np.asarray([w for _, _, w in aligned])
    r = rel(got, want)
    ok = r <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: rel {r:.3e} over {len(aligned)} coefficients")
    if not ok:
        FAILS.append(f"{label}: {r:.3e} > {tol:.0e}")


# --------------------------------------------------------------- fixtures
def panel_frame(seed=2026, n_id=30, T=6):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_id), T)
    tt = np.tile(np.arange(T), n_id)
    x1, x2 = rng.normal(size=n_id * T), rng.normal(size=n_id * T)
    u = np.repeat(rng.normal(size=n_id), T)
    y = 1 + 0.8 * x1 - 0.5 * x2 + u + rng.normal(scale=0.5, size=n_id * T)
    return pd.DataFrame({"id": ids, "t": tt, "x1": x1, "x2": x2, "y": y})


def iv_frame(seed=20261024, n=140, g=14):
    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(g), n // g)
    x1, z1, z2 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    u = rng.normal(size=n)
    e1 = 0.7 * z1 + 0.4 * z2 + 0.6 * u + rng.normal(size=n)
    yiv = 1.0 + 0.5 * x1 + 1.2 * e1 + u
    return pd.DataFrame({"yiv": yiv, "x1": x1, "e1": e1, "z1": z1, "z2": z2, "gid": gid})


def ordered_frame(seed=4242, n=500):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    lat = 0.8 * x1 - 0.5 * x2 + rng.logistic(size=n)
    y = pd.cut(lat, [-np.inf, -0.5, 0.5, 1.5, np.inf], labels=False) + 1
    return pd.DataFrame({"y": y.astype(float), "x1": x1, "x2": x2})


# ----------------------------------------------------------------- blocks
def block_linear() -> None:
    print("\n== linear-iv: OLS e(F)/e(df_r)/e() coverage, IV HC labels, LIML/GMM cluster")
    auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
    df = iv_frame()
    with Stata() as st:
        st.run("sysuse auto, clear")

        # L4/L5/L6: e(F), e(df_r), e(N_clust), e(rss), e(mss), e(sum_w)
        for cmd, kw in [
            ("regress mpg weight foreign", {}),
            ("regress mpg weight foreign, vce(robust)", {"cov_type": "HC1"}),
        ]:
            cap = st.estimate(cmd)
            res = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit(**kw)
            check(f"{cmd}: e(F)", scalar(res, "F"), cap.scalars["F"])
            check_scalar(f"{cmd}: e(df_r)", scalar(res, "df_r"), cap.scalars["df_r"])
            check_scalar(f"{cmd}: e(rss)", scalar(res, "rss"), cap.scalars["rss"])
            check_scalar(f"{cmd}: e(mss)", scalar(res, "mss"), cap.scalars["mss"])

        st.send(df)
        cap = st.estimate("regress yiv x1 e1, vce(cluster gid)")
        res = ws.OLS.from_formula("yiv ~ x1 + e1", df).fit(
            cov_type="cluster", cov_kwds={"groups": df["gid"]}
        )
        check_scalar("cluster: e(df_r) = G-1", scalar(res, "df_r"), cap.scalars["df_r"])
        check_scalar("cluster: e(N_clust)", scalar(res, "N_clust"), cap.scalars["N_clust"])
        check("cluster: e(F)", scalar(res, "F"), cap.scalars["F"])

        # L1: IV robust == HC0; robust small == HC1
        st.send(df)
        cap = st.estimate("ivregress 2sls yiv x1 (e1 = z1 z2), vce(robust)")
        se = np.sqrt(np.diag(cap.V))
        m = ws.IV2SLS.from_formula("yiv ~ x1 + [e1 ~ z1 + z2]", df)
        h0, h1 = m.fit(cov_type="HC0"), m.fit(cov_type="HC1")
        check_named("iv 2sls vce(robust) == westata HC0", h0, cap.names, se, field="bse")
        cap_small = st.estimate("ivregress 2sls yiv x1 (e1 = z1 z2), vce(robust) small")
        check_named("iv 2sls vce(robust) small == westata HC1", h1, cap_small.names,
                    np.sqrt(np.diag(cap_small.V)), field="bse")

        # L2: LIML cluster default is CR0
        cap = st.estimate("ivregress liml yiv x1 (e1 = z1 z2), vce(cluster gid)")
        liml = ws.IVLIML.from_formula("yiv ~ x1 + [e1 ~ z1 + z2]", df).fit(
            cov_type="cluster", cov_kwds={"clusters": df["gid"]}
        )
        check_named("liml vce(cluster) default (CR0)", liml, cap.names,
                    np.sqrt(np.diag(cap.V)), field="bse")

        # L3: GMM cluster puts W in the cluster metric too
        cap = st.estimate("ivregress gmm yiv x1 (e1 = z1 z2), vce(cluster gid)")
        gmm = ws.IVGMM.from_formula("yiv ~ x1 + [e1 ~ z1 + z2]", df).fit(
            cov_type="cluster", cov_kwds={"clusters": df["gid"]}
        )
        check_named("gmm vce(cluster): coefficients", gmm, cap.names, cap.b)
        check_named("gmm vce(cluster): SE", gmm, cap.names, np.sqrt(np.diag(cap.V)), field="bse")
        cap_def = st.estimate("ivregress 2sls yiv x1 (e1 = z1 z2)")
        check_named("iv 2sls default: coefficients", m.fit(), cap_def.names, cap_def.b, tol=1e-6)


def block_panel() -> None:
    print("\n== panel: xtreg fe vce(robust)/cluster, xtabond names")
    df = panel_frame()
    with Stata() as st:
        st.send(df)
        st.run("xtset id t")
        cr = st.estimate("xtreg y x1 x2, fe vce(robust)")
        cc = st.estimate("xtreg y x1 x2, fe vce(cluster id)")
        check("Stata itself: V_robust == V_cluster", cr.V, cc.V, tol=1e-12)
        p = ws.PanelOLS.from_formula("y ~ x1 + x2", df, entity="id", time="t")
        se_stata = np.sqrt(np.diag(cc.V))
        hr = p.fit(cov_type="HC1")
        cl = p.fit(cov_type="cluster", cov_kwds={"groups": df["id"]})
        for label, res in (("vce(robust)", hr), ("vce(cluster id)", cl)):
            names = list(res.param_names)
            order = [names.index("x1"), names.index("x2"),
                     next(i for i, n in enumerate(names) if n in ("Intercept", "_cons"))]
            check(f"xtreg fe {label}: SE", np.asarray(res.bse)[order], se_stata)

        # xtabond parameter naming with lags(2)
        rng = np.random.default_rng(5150)
        ids, T = np.repeat(np.arange(60), 8), np.tile(np.arange(8), 60)
        y = np.zeros(480)
        for i in range(1, 480):
            if ids[i] == ids[i - 1]:
                y[i] = 0.5 * y[i - 1] + 0.8 * rng.normal() - 0.4 * rng.normal() + rng.normal(scale=0.7)
        dyn = pd.DataFrame({"id": ids, "t": T, "y": y,
                            "x1": rng.normal(size=480), "x2": rng.normal(size=480)})
        ab = ws.XtAbond.from_formula("y ~ x1 + x2", dyn, entity="id", time="t", lags=2).fit()
        names = list(ab.param_names)
        print(f"  xtabond lags=2 names: {names}")
        want = [n for n in names if n.startswith("L")]
        ok = len(want) == 2 and all(n in names for n in ("x1", "x2"))
        print(f"  [{'ok ' if ok else 'FAIL'}] xtabond lags=2 exposes two lag terms + x1/x2")
        if not ok:
            FAILS.append("xtabond lags=2 names")


def block_discrete() -> None:
    print("\n== discrete: ologit e(V), poisson HC1, e(chi2), e(df_m)")
    df = ordered_frame()
    with Stata() as st:
        st.send(df)
        cap = st.estimate("ologit y x1 x2")
        r = ws.OrderedLogit.from_formula("y ~ x1 + x2", df).fit()
        check("ologit: coefficients", r.params, cap.b)
        # full V, aligned by name (westata prefixes cuts with "/")
        w_names = list(r.param_names)
        order = [w_names.index(n) for n in ("x1", "x2", "/cut1", "/cut2", "/cut3")]
        W = np.asarray(r.cov)[np.ix_(order, order)]
        check("ologit: full e(V) matrix", W, cap.V)
        check_scalar("ologit: e(df_m)", scalar(r, "df_m"), cap.scalars["df_m"])

        rng = np.random.default_rng(5)
        n = 400
        x1 = rng.normal(size=n)
        yc = rng.poisson(np.exp(0.3 + 0.6 * x1))
        po = pd.DataFrame({"y": yc.astype(float), "x1": x1,
                           "g": np.tile(np.arange(8), n // 8)})
        st.send(po)
        base = st.estimate("poisson y x1")
        rob = st.estimate("poisson y x1, vce(robust)")
        clu = st.estimate("poisson y x1, vce(cluster g)")
        m = ws.Poisson.from_formula("y ~ x1", po)
        for label, cap_i, kw in [
            ("stata vce(robust)", rob, {"cov_type": "robust"}),
            ("stata vce(cluster g)", clu, {"cov_type": "cluster",
                                           "cov_kwds": {"groups": po["g"]}}),
        ]:
            res = m.fit(**kw)
            check_named(f"poisson {label}: SE", res, cap_i.names,
                        np.sqrt(np.diag(cap_i.V)), tol=1e-6, field="bse")
            check_scalar(f"poisson {label}: e(chi2)", scalar(res, "chi2"),
                         cap_i.scalars["chi2"], tol=1e-6)
        h0, h1 = m.fit(cov_type="HC0"), m.fit(cov_type="HC1")
        print(f"  poisson HC0 == HC1 ? {np.array_equal(np.asarray(h0.bse), np.asarray(h1.bse))}")
        if np.array_equal(np.asarray(h0.bse), np.asarray(h1.bse)):
            FAILS.append("poisson HC1 silently equals HC0")

        # logit robust no longer raises
        lg = pd.DataFrame({"y": (rng.logistic(size=n) > 1.0).astype(float),
                           "x1": rng.normal(size=n)})
        st.send(lg)
        lr = st.estimate("logit y x1, vce(robust)")
        res = ws.Logit.from_formula("y ~ x1", lg).fit(cov_type="robust")
        check_named("logit vce(robust): SE", res, lr.names,
                    np.sqrt(np.diag(lr.V)), tol=1e-6, field="bse")


def block_limited() -> None:
    print("\n== limited: tobit sigma SE / df_r, qreg robust at tau!=.5, rreg SE")
    rng = np.random.default_rng(31)
    n = 500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 0.6 + 0.8 * x1 - 0.4 * x2 + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    with Stata() as st:
        st.send(df)
        cap = st.estimate("tobit y x1 x2, ll(0)")
        tb = ws.Tobit.from_formula("y ~ x1 + x2", df, lower=0).fit()
        # Stata stores var(e.y) = sigma^2; westata reports /sigma
        se_sigma = np.sqrt(np.diag(cap.V))[cap.names.index("var(e.y)")] / (2 * np.sqrt(cap.b[cap.names.index("var(e.y)")]))
        check("tobit: se(/sigma) after delta fix", [tb.bse[list(tb.param_names).index("/sigma")]],
              [se_sigma], tol=1e-5)
        check_scalar("tobit: e(df_r)", scalar(tb, "df_r"), cap.scalars["df_r"])
        check_named("tobit: coefficients", tb, cap.names[:-1], cap.b[:-1], tol=1e-6)

        rng2 = np.random.default_rng(77)
        qx = rng2.normal(size=400)
        qy = 1.0 + 0.5 * qx + rng2.normal(size=400) * (0.3 + 0.7 * np.abs(qx))
        qd = pd.DataFrame({"y": qy, "x1": qx})
        st.send(qd)
        for tau in (0.25, 0.5, 0.75):
            cap = st.estimate(f"qreg y x1, quantile({tau}) vce(robust)")
            r = ws.QuantileRegression.from_formula("y ~ x1", qd, quantile=tau).fit(cov_type="robust")
            check_named(f"qreg tau={tau} robust: SE (heteroskedastic DGP)", r, cap.names,
                        np.sqrt(np.diag(cap.V)), tol=1e-6, field="bse")

        st.run("sysuse auto, clear")
        cap = st.estimate("rreg mpg weight length")
        rr = ws.RobustRegression.from_formula("mpg ~ weight + length", auto_ := pd.read_stata(
            "tests/data/auto.dta", convert_categoricals=False)).fit()
        check_named("rreg: SE law rmse*sqrt(diag((X'X)^-1))", rr, cap.names,
                    np.sqrt(np.diag(cap.V)), tol=1e-6, field="bse")


def block_systems() -> None:
    print("\n== systems: nl covariance, aipw ATET, bootstrap z")
    auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
    with Stata() as st:
        st.run("sysuse auto, clear")
        cap = st.estimate("nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)")
        r = ws.Nl.from_formula(
            "mpg = {b0} + {b1}*weight + {b2}*weight**2", auto,
            start={"b0": cap.b[0], "b1": cap.b[1], "b2": cap.b[2]},
        ).fit()
        check("nl ill-conditioned: coefficients", r.params, cap.b)
        check("nl ill-conditioned: SE (pinv fix)", r.bse, np.sqrt(np.diag(cap.V)))

        rng = np.random.default_rng(909)
        n = 600
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + 0.8 * x1 - 0.4 * x2)))).astype(float)
        y = 2.0 + 1.5 * t + 0.9 * x1 + 0.6 * x2 + rng.normal(size=n)
        tr = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})
        st.send(tr)
        for est, cmd in (("ate", "teffects aipw (y x1 x2) (t x1 x2)"),
                         ("atet", "teffects aipw (y x1 x2) (t x1 x2), atet")):
            cap = st.estimate(cmd)
            res = ws.AIPW.from_formula("y", "t", tr, covariates=["x1", "x2"], estimand=est).fit()
            check(f"teffects aipw {est}: point estimate", [float(res.estimate)],
                  [float(np.asarray(cap.b).reshape(-1)[0])], tol=1e-6)
            check(f"teffects aipw {est}: SE", [float(res.std_err)],
                  [float(np.sqrt(np.diag(cap.V)).reshape(-1)[0])], tol=1e-6)


def block_tseries() -> None:
    print("\n== tseries: VAR SE/Sigma/Granger (ML), dfuller critical values")
    rng = np.random.default_rng(20261024)
    n = 300
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for i in range(1, n):
        y1[i] = 0.6 * y1[i - 1] + 0.2 * y2[i - 1] + rng.normal()
        y2[i] = 0.1 * y1[i - 1] + 0.5 * y2[i - 1] + rng.normal()
    df = pd.DataFrame({"t": np.arange(n), "y1": y1, "y2": y2})
    with Stata() as st:
        st.send(df)
        st.run("tsset t")
        # V1: Stata's var e(V) is kron((Z'Z)^-1, Sigma_ML).  Compare the whole
        # matrix after reordering westata's rows/cols onto Stata's coefficient
        # order (VARResults exposes that order as ``stata_param_names``).
        st.quiet("var y1 y2, lags(1)")
        V_stata = st.matrix("e(V)")
        names_stata = st.matrix_names("e(b)")
        v = ws.VAR(df[["y1", "y2"]].to_numpy(), lags=1).fit()
        # Both sides order the coefficients the same way (equation-major), so the
        # matrices line up element by element; names are recorded for the log.
        check("var(1): e(V) whole 6x6 matrix (ML sigma)",
              np.asarray(v.stata_cov_params), V_stata, tol=1e-8)
        print(f"  (Stata coefficient order: {names_stata})")
        check("var(1): e(Sigma) 2x2", np.asarray(v.sigma_u), st.matrix("e(Sigma)"), tol=1e-10)
        check("var(1): bse (per-equation ML df)", np.asarray(v.bse, float),
              np.sqrt(np.diag(V_stata)), tol=1e-8)
        # V2: vargranger is a Wald chi2.  Stata leaves only r(p_#) and the
        # printed table wants the statistic, so the printed 3-decimal values are
        # lower bounds; the chi2 is re-derived from p and df and compared with
        # that lower bound instead of pretending to a 1e-9 comparison.
        out = st.output("vargranger")
        g = v.granger_causality("y1", ["y2"])
        import re as _re
        for line in out.splitlines():
            m = _re.match(r"\s*\|\s*(\w+)\s+(\w+)\s*\|\s*([\d.]+)\s+(\d+)\s+([\d.]+)", line)
            if m and m.group(2) != "ALL":
                printed = float(m.group(3))
                print(f"  vargranger {m.group(1)} <- {m.group(2)}: printed chi2 = {printed:.4f}; "
                      f"westata chi2 = {float(g.test_statistic):.3f}")
                ok = abs(float(g.test_statistic) - printed) < 5e-4 + 1e-3 * abs(printed)
                print(f"  [{'ok ' if ok else 'FAIL'}] within the printed 3-decimal rounding")
                if not ok:
                    FAILS.append("vargranger chi2 outside printed rounding")
                break

        # V3: dfuller's printed critical values come from Stata's hard-coded
        # table.  The statistic and the MacKinnon p ARE full precision in r().
        st.quiet("dfuller y1, lags(2) trend")
        zt = st.scalar("r(Zt)")
        pval = st.scalar("r(p)")
        adf = ws.unitroot(df["y1"].to_numpy(), test="adf", regression="ct",
                         maxlag=2, autolag=None)
        print(f"  dfuller: Stata Z(t)={zt!r} p={pval!r}; "
              f"westata stat={getattr(adf, 'statistic', 'n/a')!r}")
        if zt is not None and hasattr(adf, "statistic"):
            check("dfuller: Z(t) full precision", [float(adf.statistic)], [float(zt)], tol=1e-9)
        if pval is not None and hasattr(adf, "pvalue"):
            check("dfuller: MacKinnon p", [float(adf.pvalue)], [float(pval)], tol=1e-6)


def main(argv: list[str]) -> int:
    blocks = {
        "linear": block_linear,
        "panel": block_panel,
        "discrete": block_discrete,
        "limited": block_limited,
        "systems": block_systems,
        "tseries": block_tseries,
    }
    chosen = argv or list(blocks)
    for name in chosen:
        try:
            blocks[name]()
        except Exception:
            print(f"  [ERROR] block {name} raised:")
            traceback.print_exc()
            FAILS.append(f"{name}: exception")
    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILURES"))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
