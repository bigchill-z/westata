"""verify2 audit C: limited group (T1-T7) -- independent re-measurement."""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/fix/verify2")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import westata as ws
from mybriage import S, fmt, rel, tier
from vlib import cmp_b, cmp_mat, cmp_se, report


def tobit_frame(seed=2024, n=500):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    lat = 0.6 + 0.8 * x1 - 0.4 * x2 + rng.normal(size=n)
    return pd.DataFrame({
        "x1": x1, "x2": x2,
        "yl": np.maximum(lat, 0.0),          # lower censored at 0
        "yu": np.minimum(lat, 2.5),          # upper censored at 2.5
        "yb": np.clip(lat, 0.0, 2.0),        # both
    })


def qreg_frame(seed=31, n=400, hetero=False):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    scale = (1.0 + 0.9 * np.abs(x1)) if hetero else 1.0
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + scale * rng.standard_t(df=5, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def heckman_frame(seed=7, n=1200):
    rng = np.random.default_rng(seed)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    x1 = rng.normal(size=n)
    u = rng.normal(size=n)
    v = 0.6 * u + np.sqrt(1 - 0.36) * rng.normal(size=n)
    s = (0.3 + 0.8 * z1 - 0.5 * z2 + u > 0).astype(float)
    y = 1.2 + 0.7 * x1 + v
    return pd.DataFrame({"x1": x1, "z1": z1, "z2": z2, "s": s, "y": y})


def etreg_frame(seed=707, n=1200):
    rng = np.random.default_rng(seed)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    x1 = rng.normal(size=n)
    u = rng.normal(size=n)
    e = 0.7 * u + np.sqrt(1 - 0.49) * rng.normal(size=n)
    t = (0.2 + 0.7 * z1 + 0.4 * z2 + u > 0).astype(float)
    y = 0.5 + 0.9 * x1 + 1.5 * t + e
    return pd.DataFrame({"x1": x1, "z1": z1, "z2": z2, "t": t, "y": y})


def rreg_frame(seed=67, n=400, contam=0.05):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    bad = rng.choice(n, int(contam * n), replace=False)
    y[bad] += 12.0
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def main():
    print("=" * 100)
    print("C. LIMITED")
    print("=" * 100)
    s = S().start()
    print(f"stata {s.version}")
    out = {}

    # ------------------------------------------------------------- T1/T2/T3
    print("\n[C1] T1 tobit /sigma SE (ll / ul / both) + T2 e(df_r) + T3 coef SE & stdp")
    fr = tobit_frame()
    s.send(fr)
    for cmd, col, lower, upper, tag in (
        ("tobit yl x1 x2, ll(0)", "yl", 0.0, None, "ll(0)"),
        ("tobit yu x1 x2, ul(2.5)", "yu", None, 2.5, "ul(2.5)"),
        ("tobit yb x1 x2, ll(0) ul(2)", "yb", 0.0, 2.0, "ll(0) ul(2)"),
    ):
        E = s.est(cmd)
        res = ws.Tobit.from_formula(f"{col} ~ x1 + x2", fr, lower=lower, upper=upper).fit()
        pairs = [("x1", "x1"), ("x2", "x2"), ("_cons", "Intercept")]
        b, _ = cmp_b(E, res, pairs)
        se, _ = cmp_se(E, res, pairs)
        report(f"tobit {tag} coefficients", b, se, tier(b), tier(se))
        # sigma: Stata stores var(e.y) = sigma^2, so se(sigma) = se(var)/(2 sigma)
        s_idx = E.idx("var(e.%s)" % col) if E.idx("var(e.%s)" % col) is not None else None
        if s_idx is not None:
            sv = E.b[s_idx]
            sse = np.sqrt(E.V[s_idx, s_idx])
            sig = np.sqrt(sv)
            stata_se_sigma = sse / (2 * sig)
        else:
            sig = sse = stata_se_sigma = float("nan")
        wse_sigma = float(res.bse["/sigma"])
        r = abs(wse_sigma - stata_se_sigma) / stata_se_sigma
        print(f"       /sigma  Stata sigma={sig:.10f} se={stata_se_sigma:.10f} | "
              f"westata se={wse_sigma:.10f} rel={fmt(r)}({tier(r)})")
        out[f"tobit_sigma_{tag}"] = r
        dfr = abs(res.e["df_r"] - E.e["df_r"]) / E.e["df_r"]
        dfm = abs(res.e["df_m"] - E.e["df_m"]) / max(E.e["df_m"], 1e-12)
        print(f"       e(df_r) Stata={E.e['df_r']} westata={res.e['df_r']} | "
              f"e(df_m) Stata={E.e['df_m']} westata={res.e['df_m']}")
        out[f"tobit_df_{tag}"] = max(dfr, dfm)
        # stdp / stdf
        s.q("cap drop _sp _sf")
        s.q("predict double _sp, stdp")
        s.q("predict double _sf, stdf")
        sp, sf = s.getvar("_sp"), s.getvar("_sf")
        wsp = np.asarray(res.predict(kind="stdp"), dtype=float).reshape(-1)
        wsf = np.asarray(res.predict(kind="stdf"), dtype=float).reshape(-1)
        rp, rf = rel(wsp, sp), rel(wsf, sf)
        print(f"       predict stdp rel={fmt(rp)}({tier(rp)})  stdf rel={fmt(rf)}({tier(rf)})")
        out[f"tobit_stdp_{tag}"] = rp
        out[f"tobit_stdf_{tag}"] = rf

    # ---------------------------------------------------------------- T4
    print("\n[C2] T4 qreg, vce(robust) -- tau .25/.5/.75 x homoskedastic/heteroskedastic")
    for het in (False, True):
        fq = qreg_frame(seed=77 if het else 31, hetero=het)
        s.send(fq)
        for tau in (0.25, 0.5, 0.75):
            E = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
            res = ws.QuantileRegression.from_formula("y ~ x1 + x2", fq,
                                                    quantile=tau).fit(cov_type="robust")
            pairs = [("x1", "x1"), ("x2", "x2"), ("_cons", "Intercept")]
            b, _ = cmp_b(E, res, pairs)
            se, _ = cmp_se(E, res, pairs)
            tag = f"{'hetero' if het else 'homo'} tau={tau}"
            report(f"qreg robust {tag}", b, se, tier(b), tier(se))
            out[f"qreg_{tag}"] = max(b, se)
            # iid too, since the fix must not break the default VCE
            E2 = s.est(f"qreg y x1 x2, quantile({tau})")
            res2 = ws.QuantileRegression.from_formula("y ~ x1 + x2", fq,
                                                     quantile=tau).fit(cov_type="iid")
            se2, _ = cmp_se(E2, res2, pairs)
            print(f"       iid se rel={fmt(se2)} ({tier(se2)})")
            out[f"qreg_iid_{tag}"] = se2

    # ---------------------------------------------------------------- T5/T6
    print("\n[C3] T5 heckman e(sigma)/e(rho) ; T6 etregress e(rho) sign")
    fh = heckman_frame()
    s.send(fh)
    E = s.est("heckman y x1, select(s = z1 z2) twostep")
    hk = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", fh).fit(method="twostep")
    for key in ("sigma", "rho"):
        sv, wv = E.e.get(key), hk.e.get(key)
        r = abs(wv - sv) / abs(sv)
        print(f"       heckman e({key:<5s}) Stata={sv:.12f} westata={wv:.12f} "
              f"rel={fmt(r)}({tier(r)})")
        out[f"heckman_{key}"] = r
    fe = etreg_frame()
    s.send(fe)
    E = s.est("etregress y x1, treat(t = z1 z2) twostep")
    et = ws.Etregress.from_formula("y ~ x1 + treat(t ~ z1 + z2)", fe).fit(method="twostep")
    for key in ("sigma", "rho", "lambda"):
        sv, wv = E.e.get(key), et.e.get(key)
        if sv is None or wv is None:
            print(f"       etregress e({key}): Stata={sv} westata={wv}")
            continue
        r = abs(wv - sv) / abs(sv)
        print(f"       etregress e({key:<6s}) Stata={sv:.12f} westata={wv:.12f} "
              f"rel={fmt(r)}({tier(r)}) sign_ok={np.sign(sv) == np.sign(wv)}")
        out[f"etregress_{key}"] = r

    # ---------------------------------------------------------------- T7
    print("\n[C4] T7 rreg SE + rmse^2*(n-k)=rss")
    frr = rreg_frame()
    s.send(frr)
    E = s.est("rreg y x1 x2")
    rr = ws.RobustRegression.from_formula("y ~ x1 + x2", frr).fit()
    pairs = [("x1", "x1"), ("x2", "x2"), ("_cons", "Intercept")]
    b, _ = cmp_b(E, rr, pairs)
    se, _ = cmp_se(E, rr, pairs)
    report("rreg b/se vs Stata default tolerance", b, se, tier(b), tier(se))
    out["rreg_se"] = se
    n, k = len(frr), 3
    rss = float(rr.e["rss"])
    rmse = float(rr.e["rmse"])
    ident = abs(rmse**2 * (n - k) - rss) / max(abs(rss), 1e-12)
    print(f"       self-consistency rmse^2*(n-k) vs rss: rel={fmt(ident)}({tier(ident)}) "
          f"rss={rss:.10f} rmse={rmse:.10f}")
    out["rreg_identity"] = ident
    for key in ("rss", "mse", "rmse", "r2", "F", "df_r", "df_m", "N"):
        sv, wv = E.e.get(key), rr.e.get(key)
        if sv is None or wv is None:
            print(f"       e({key}): Stata={sv} westata={wv}")
            continue
        r = abs(wv - sv) / max(abs(sv), 1e-12)
        print(f"       e({key:<5s}) Stata={sv!r:<24s} westata={wv!r:<24s} rel={fmt(r)}({tier(r)})")
        out[f"rreg_{key}"] = r

    print("\n---- summary ----")
    for kk, vv in sorted(out.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)):
        print(f"  {kk:<32s} {fmt(vv):<10s} {tier(vv)}")


if __name__ == "__main__":
    main()
