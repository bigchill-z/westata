"""verify2 audit F: linear-iv group (L1-L6) -- independent re-measurement."""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/fix/verify2")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import westata as ws
from westata.protocols import SSC
from mybriage import S, fmt, rel, tier
from vlib import cmp_b, cmp_mat, cmp_se, report

AUTO = "D:/Program Files/StataNow19/auto.dta"


def load():
    ds = pd.read_stata(AUTO)
    ds["foreign"] = (ds["foreign"].astype(str) == "Foreign").astype(int)
    ds["gid"] = (np.arange(len(ds)) % 12) + 1
    ds["w"] = (np.arange(len(ds)) % 4) + 1
    return ds


def iv_frame(seed=20261024, g=140, t=14):
    """The retest's own IV panel shape: 140 groups x 14 periods."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(g):
        for k in range(t):
            z1, z2 = rng.normal(), rng.normal()
            x1 = rng.normal()
            u = rng.normal()
            e1 = 0.8 * z1 + 0.3 * z2 + 0.6 * u + rng.normal()
            y = 1.0 + 2.0 * x1 - 0.5 * e1 + u
            rows.append((i + 1, k + 1, y, x1, e1, z1, z2))
    return pd.DataFrame(rows, columns=["gid", "tt", "yiv", "x1", "e1", "z1", "z2"])


def main():
    print("=" * 100)
    print("F. LINEAR-IV")
    print("=" * 100)
    s = S().start()
    print(f"stata {s.version}")
    ds = load()
    out = {}

    # ------------------------------------------------------------------ L1
    print("\n[F1] L1 IV vce(robust) == HC0 ; vce(robust) small == HC1")
    iv = iv_frame()
    s.send(iv)
    E = s.est("ivregress 2sls yiv x1 (e1 = z1 z2), vce(robust)")
    Es = s.est("ivregress 2sls yiv x1 (e1 = z1 z2), vce(robust) small")
    model = ws.IV2SLS.from_formula("yiv ~ 1 + x1 + [e1 ~ z1 + z2]", iv)
    hc0 = model.fit(cov_type="HC0")
    hc1 = model.fit(cov_type="HC1")
    pairs = [("x1", "x1"), ("e1", "e1"), ("_cons", "Intercept")]
    r0, _ = cmp_se(E, hc0, pairs)
    r1, _ = cmp_se(Es, hc1, pairs)
    print(f"       HC0 vs vce(robust)        se rel={fmt(r0)} ({tier(r0)})")
    print(f"       HC1 vs vce(robust) small  se rel={fmt(r1)} ({tier(r1)})")
    ratio = np.sqrt(len(iv) / (len(iv) - 3))
    print(f"       Stata small/default ratio = {Es.se()[0]/E.se()[0]:.12f} ; "
          f"sqrt(N/(N-K)) = {ratio:.12f}")
    out["L1 HC0"] = r0
    out["L1 HC1 small"] = r1

    # ------------------------------------------------------------------ L2
    print("\n[F2] L2 IVLIML vce(cluster) is CR0 (Stata default), not CR1")
    E = s.est("ivregress liml yiv x1 (e1 = z1 z2), vce(cluster gid)")
    Es = s.est("ivregress liml yiv x1 (e1 = z1 z2), vce(cluster gid) small")
    liml = ws.IVLIML.from_formula("yiv ~ 1 + x1 + [e1 ~ z1 + z2]", iv)
    cr0 = liml.fit(cov_type="cluster", cov_kwds={"clusters": iv["gid"].to_numpy()})
    cr1 = liml.fit(cov_type="cluster", cov_kwds={"clusters": iv["gid"].to_numpy()},
                   ssc=SSC(adj=True))
    r0, _ = cmp_se(E, cr0, pairs)
    r1, _ = cmp_se(Es, cr1, pairs)
    print(f"       default (CR0) se rel={fmt(r0)} ({tier(r0)})")
    print(f"       small   (CR1) se rel={fmt(r1)} ({tier(r1)})")
    g = iv["gid"].nunique()
    n = len(iv)
    print(f"       Stata small/default = {Es.se()[0]/E.se()[0]:.12f} ; "
          f"sqrt(G/(G-1)*(N-1)/(N-K)) = "
          f"{np.sqrt(g/(g-1)*(n-1)/(n-3)):.12f}")
    out["L2 CR0"] = r0
    out["L2 CR1"] = r1
    # 2SLS must be unchanged (it always took the compute_cov path)
    E2 = s.est("ivregress 2sls yiv x1 (e1 = z1 z2), vce(cluster gid)")
    two = ws.IV2SLS.from_formula("yiv ~ 1 + x1 + [e1 ~ z1 + z2]", iv).fit(
        cov_type="cluster", cov_kwds={"clusters": iv["gid"].to_numpy()})
    r2, _ = cmp_se(E2, two, pairs)
    print(f"       2SLS cluster (regression guard) se rel={fmt(r2)} ({tier(r2)})")
    out["L2 2sls cluster"] = r2

    # ------------------------------------------------------------------ L3
    print("\n[F3] L3 IVGMM vce(cluster): Stata clusters W too (point ESTIMATE moves)")
    E = s.est("ivregress gmm yiv x1 (e1 = z1 z2), vce(cluster gid)")
    gmm = ws.IVGMM.from_formula("yiv ~ 1 + x1 + [e1 ~ z1 + z2]", iv).fit(
        cov_type="cluster", cov_kwds={"clusters": iv["gid"].to_numpy()})
    b, _ = cmp_b(E, gmm, pairs)
    se, _ = cmp_se(E, gmm, pairs)
    V, _ = cmp_mat(E, gmm, pairs)
    report("GMM cluster b/se", b, se, tier(b), tier(se))
    print(f"       e(V) full rel={fmt(V)} ({tier(V)})")
    out["L3 gmm cluster b"] = b
    out["L3 gmm cluster se"] = se
    out["L3 gmm cluster V"] = V
    # non-clustered GMM must not regress
    E2 = s.est("ivregress gmm yiv x1 (e1 = z1 z2)")
    gmm2 = ws.IVGMM.from_formula("yiv ~ 1 + x1 + [e1 ~ z1 + z2]", iv).fit()
    b2, _ = cmp_b(E2, gmm2, pairs)
    se2, _ = cmp_se(E2, gmm2, pairs)
    print(f"       GMM default (guard) b rel={fmt(b2)} se rel={fmt(se2)}")
    out["L3 gmm default"] = max(b2, se2)

    # ------------------------------------------------------------------ L4
    print("\n[F4] L4 OLS e(F) follows the VCE; L5 e(df_r)=G-1; L6 e() coverage")
    s.run("sysuse auto, clear")
    s.send(ds)
    cases = [
        ("regress mpg weight foreign", {}, ws.OLS, "mpg ~ weight + foreign"),
        ("regress mpg weight foreign, vce(robust)", {"cov_type": "HC1"},
         ws.OLS, "mpg ~ weight + foreign"),
        ("regress mpg weight foreign, vce(cluster gid)",
         {"cov_type": "cluster", "cov_kwds": {"groups": ds["gid"].to_numpy()}},
         ws.OLS, "mpg ~ weight + foreign"),
        ("regress mpg weight foreign [pweight=w]",
         {"cov_type": "HC1"}, ws.OLS, "mpg ~ weight + foreign"),
        ("regress mpg weight foreign [aweight=w]", {}, ws.OLS,
         "mpg ~ weight + foreign"),
        ("regress mpg weight foreign [fweight=w]", {}, ws.OLS,
         "mpg ~ weight + foreign"),
        ("regress mpg weight foreign, noconstant", {}, ws.OLS,
         "mpg ~ weight + foreign - 1"),
        ("regress mpg", {}, ws.OLS, "mpg ~ 1"),
    ]
    for cmd, kw, cls, form in cases:
        # weights are passed via the model's weight= keyword below
        weight = None
        for token in ("pweight", "aweight", "fweight"):
            if token in cmd:
                weight = token
        if weight:
            res = cls.from_formula(form, ds, weights=ds["w"].to_numpy(),
                                   weight_type=weight).fit(**kw)
        else:
            res = cls.from_formula(form, ds).fit(**kw)
        E = s.est(cmd)
        f_s, f_w = E.e.get("F"), res.e.get("F")
        if f_s is None or f_w is None:
            print(f"       {cmd:<48s} e(F) Stata={f_s} westata={f_w}")
            continue
        rf = abs(f_w - f_s) / max(abs(f_s), 1e-12)
        d_s, d_w = E.e.get("df_r"), res.e.get("df_r")
        rd = 0.0 if d_s == d_w else 1.0
        print(f"       {cmd:<48s} e(F) rel={fmt(rf)}({tier(rf)}) "
              f"e(df_r) {d_s!r}/{d_w!r} match={rd == 0}")
        out[f"L4 {cmd}"] = rf
        out[f"L5 {cmd}"] = rd
        # L6 coverage: every scalar Stata stores must be present and equal
        missing, wrong = [], []
        for key in ("rss", "mss", "sum_w", "N_clust", "ll", "ll_0", "r2", "r2_a",
                    "rmse", "df_m", "N", "tss"):
            sv = E.e.get(key)
            if sv is None:
                continue
            wv = res.e.get(key)
            if wv is None:
                missing.append(key)
                continue
            if abs(float(wv) - sv) / max(abs(sv), 1e-12) > 1e-9:
                wrong.append(f"{key}:{float(wv):.6g}/{sv:.6g}")
        print(f"           e() coverage: missing={missing} wrong={wrong}")
        out[f"L6 {cmd}"] = float(len(missing) + len(wrong))

    print("\n---- summary ----")
    for k, v in sorted(out.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)):
        print(f"  {k:<48s} {fmt(v):<10s} {tier(v)}")


if __name__ == "__main__":
    main()
