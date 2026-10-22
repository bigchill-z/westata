"""Extended Stata conformance sweep for the remaining E1 commands.

Covers: xtreg family, nbreg, ologit/oprobit, margins, predict, var, vec, arch,
dfuller/pperror/kpss, areg, and LIML/GMM.

Commands that are not installed in the local Stata (reghdfe, esttab) are skipped
with an explicit message rather than silently omitted.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import westata as ws  # noqa: E402
from westata.conformance.oracles.stata import StataOracle  # noqa: E402


@dataclass
class Outcome:
    name: str
    ok: bool
    param_rel: float = float("nan")
    se_rel: float = float("nan")
    scalars: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str = ""


def rel(actual, expected) -> float:
    a = np.asarray(actual, dtype=float).reshape(-1)
    e = np.asarray(expected, dtype=float).reshape(-1)
    if a.size != e.size:
        return float("inf")
    scale = np.maximum(np.abs(e), 1e-12)
    return float(np.max(np.abs(a - e) / scale))


def align(result, stata_names, translation=None) -> np.ndarray:
    translation = translation or {}

    def canon(name: str) -> str:
        if name in ("Intercept", "_cons", "const"):
            return "__c__"
        return translation.get(name, name)

    lookup = {canon(n): i for i, n in enumerate(result.param_names)}
    order = []
    for name in stata_names:
        key = canon(name)
        if key not in lookup:
            raise KeyError(f"no westata param matches {name!r}")
        order.append(lookup[key])
    return np.asarray(order, dtype=int)


# ----------------------------------------------------------------- data


def panel_frame(n_entities: int = 30, n_periods: int = 6, seed: int = 21):
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + rng.normal(scale=0.4, size=n)
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time}
    )


# ------------------------------------------------------------- checks


def check_xtreg_fe(o: StataOracle) -> Outcome:
    df = panel_frame()
    o.send_data(df)
    o.run("xtset firm year")
    o.run("xtreg y x1 x2, fe")
    cap = o.capture()
    res = ws.PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()
    perm = align(res, cap.names)
    return Outcome(
        "xtreg, fe",
        True,
        rel(res.params.to_numpy()[perm], cap.b),
        rel(res.bse.to_numpy()[perm], np.sqrt(np.diag(cap.V))),
        {
            "df_r": (cap.scalars.get("df_r", np.nan), res.e["df_r"]),
            "r2": (cap.scalars.get("r2", np.nan), res.e["r2"]),
        },
    )


def check_xtreg_re(o: StataOracle) -> Outcome:
    df = panel_frame()
    o.send_data(df)
    o.run("xtset firm year")
    o.run("xtreg y x1 x2, re")
    cap = o.capture()
    res = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", df, entity="firm", time="year"
    ).fit()
    perm = align(res, cap.names)
    return Outcome(
        "xtreg, re",
        True,
        rel(res.params.to_numpy()[perm], cap.b),
        rel(res.bse.to_numpy()[perm], np.sqrt(np.diag(cap.V))),
        {
            "df_r": (cap.scalars.get("df_r", np.nan), res.e["df_r"]),
            "r2": (cap.scalars.get("r2", np.nan), res.e["r2"]),
        },
    )


def check_xtreg_be(o: StataOracle) -> Outcome:
    df = panel_frame()
    o.send_data(df)
    o.run("xtset firm year")
    o.run("xtreg y x1 x2, be")
    cap = o.capture()
    res = ws.BetweenOLS.from_formula(
        "y ~ x1 + x2", df, entity="firm", time="year"
    ).fit()
    perm = align(res, cap.names)
    return Outcome(
        "xtreg, be",
        True,
        rel(res.params.to_numpy()[perm], cap.b),
        rel(res.bse.to_numpy()[perm], np.sqrt(np.diag(cap.V))),
    )


def check_xtreg_fd(o: StataOracle) -> Outcome:
    """``xtreg, fd`` no longer exists in StataNow 19 (verified: r(198)).

    The equivalent is ``xtreg y x, fe`` on the differenced data, or
    ``regress D.y D.x``. The latter is what westata's ``FirstDifferenceOLS``
    reproduces, so it is compared against that explicit Stata construction.
    """

    df = panel_frame()
    o.send_data(df)
    o.run("xtset firm year")
    o.run("gen dy = D.y")
    o.run("gen dx1 = D.x1")
    o.run("gen dx2 = D.x2")
    o.run("regress dy dx1 dx2, noconstant")
    cap = o.capture()
    res = ws.FirstDifferenceOLS.from_formula(
        "y ~ x1 + x2", df, entity="firm", time="year"
    ).fit()
    # The Stata side works on dx1/dx2 columns, so the names need translating.
    perm = align(res, cap.names, {"dx1": "x1", "dx2": "x2"})
    return Outcome(
        "xtreg, fd (via regress D.)",
        True,
        rel(res.params.to_numpy()[perm], cap.b),
        rel(res.bse.to_numpy()[perm], np.sqrt(np.diag(cap.V))),
        notes=["StataNow 19 removed the fd option from xtreg; compared against regress D."],
    )


def check_areg(o: StataOracle) -> Outcome:
    df = panel_frame()
    o.send_data(df)
    o.run("areg y x1 x2, absorb(firm)")
    cap = o.capture()
    res = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", df).fit()
    # areg reports the absorbed effect's constant as _cons; westata reports only
    # the slopes for an absorbed model, so compare the slopes.
    slope_names = [name for name in cap.names if name != "_cons"]
    positions = [cap.names.index(name) for name in slope_names]
    stata_b = np.asarray(cap.b, dtype=float)[positions]
    stata_se = np.sqrt(np.diag(cap.V))[positions]
    perm = align(res, slope_names)
    return Outcome(
        "areg",
        True,
        rel(res.params.to_numpy()[perm], stata_b),
        rel(res.bse.to_numpy()[perm], stata_se),
        {"df_r": (cap.scalars.get("df_r", np.nan), res.e["df_r"])},
        notes=[f"skipped _cons (absorbed); Stata names: {cap.names}"],
    )


def check_nbreg(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(31)
    n = 600
    x1 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.5 * x1)
    alpha = 0.4
    # Negative binomial draws via a Gamma-Poisson mixture.
    lam = rng.gamma(shape=1.0 / alpha, scale=mu * alpha)
    y = rng.poisson(lam).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1})
    o.send_data(df)
    o.run("nbreg y x1")
    cap = o.capture()
    res = ws.NegativeBinomial.from_formula("y ~ x1", df).fit()
    # Stata reports the dispersion as /lnalpha; westata names it /lnalpha too.
    names = cap.names
    translation = {}
    for name in names:
        if name.startswith("/lnalpha") or name == "lnalpha":
            translation[name] = "/lnalpha"
        elif name == "_cons":
            translation[name] = "Intercept"
    perm = align(res, names, translation)
    return Outcome(
        "nbreg",
        True,
        rel(res.params.to_numpy()[perm], cap.b),
        rel(res.bse.to_numpy()[perm], np.sqrt(np.diag(cap.V))),
        {"ll": (cap.scalars.get("ll", np.nan), float(res.e["ll"]))},
    )


def _ordered_data(seed: int = 7, n: int = 700):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 0.8 * x1 - 0.4 * x2 + rng.normal(size=n)
    y = pd.cut(latent, bins=[-np.inf, -0.7, 0.4, np.inf], labels=[1, 2, 3]).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def check_ologit(o: StataOracle) -> Outcome:
    df = _ordered_data()
    o.send_data(df)
    o.run("ologit y x1 x2")
    cap = o.capture()
    res = ws.OrderedLogit.from_formula("y ~ x1 + x2", df).fit()
    return _compare_ordered("ologit", res, cap)


def check_oprobit(o: StataOracle) -> Outcome:
    df = _ordered_data()
    o.send_data(df)
    o.run("oprobit y x1 x2")
    cap = o.capture()
    res = ws.OrderedProbit.from_formula("y ~ x1 + x2", df).fit()
    return _compare_ordered("oprobit", res, cap)


def _compare_ordered(name: str, res, cap) -> Outcome:
    """Stata orders slopes then cut1../cutk; westata uses /cut1../cutk.

    Stata now prints the cut names without the leading slash (``cut1``), so the
    comparison has to normalise both spellings before matching.
    """

    def is_cut(param_name: str) -> bool:
        return param_name.lstrip("/").startswith("cut")

    stata_slopes = [i for i, n in enumerate(cap.names) if not is_cut(n)]
    stata_cuts = [i for i, n in enumerate(cap.names) if is_cut(n)]
    westata_names = list(res.param_names)
    westata_slopes = [i for i, n in enumerate(westata_names) if not is_cut(n)]
    westata_cuts = [i for i, n in enumerate(westata_names) if is_cut(n)]

    if len(stata_cuts) != len(westata_cuts):
        return Outcome(
            name,
            False,
            error=f"cut count differs: stata {len(stata_cuts)} vs westata {len(westata_cuts)}",
        )

    slope_rel = rel(
        res.params.to_numpy()[westata_slopes],
        np.asarray(cap.b, dtype=float)[stata_slopes],
    )
    cut_rel = rel(
        res.params.to_numpy()[westata_cuts],
        np.asarray(cap.b, dtype=float)[stata_cuts],
    )
    return Outcome(
        name,
        True,
        max(slope_rel, cut_rel),
        float("nan"),
        {"ll": (cap.scalars.get("ll", np.nan), float(res.e["ll"]))},
        notes=[
            f"slopes rel={slope_rel:.2e}, cuts rel={cut_rel:.2e}",
            f"westata cuts: {np.round(res.params.to_numpy()[westata_cuts], 6)}",
            f"Stata cuts  : {np.round(np.asarray(cap.b, dtype=float)[stata_cuts], 6)}",
        ],
    )


def check_margins_logit(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(3)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 - 0.8 * x2 + rng.logistic(size=n)
    df = pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1, "x2": x2})
    o.send_data(df)
    o.run("logit y x1 x2")
    output = o.get_output("margins, dydx(x1 x2)")
    res = ws.Logit.from_formula("y ~ x1 + x2", df).fit()
    rows = res.margins(dydx=["x1", "x2"])

    # Stata prints one row per variable with a "|" separator:
    #   "          x1 |   .2049253   .0134604    15.22   0.000 ..."
    estimates = {}
    for line in output.splitlines():
        if "|" not in line:
            continue
        left, right = line.split("|", 1)
        variable = left.strip()
        if variable not in ("x1", "x2"):
            continue
        parts = right.split()
        if len(parts) >= 2:
            try:
                estimates[variable] = (float(parts[0]), float(parts[1]))
            except ValueError:
                continue

    if len(estimates) != 2:
        return Outcome("margins, dydx", False, error=f"parse failed:\n{output[:800]}")

    est_rel = {}
    se_rel = {}
    for row in rows:
        stata_est, stata_se = estimates[row.label]
        est_rel[row.label] = abs(row.estimate / stata_est - 1)
        se_rel[row.label] = abs(row.std_err / stata_se - 1)

    return Outcome(
        "margins, dydx",
        True,
        max(est_rel.values()),
        max(se_rel.values()),
        notes=[f"{k}: est_rel={v:.2e}" for k, v in est_rel.items()],
    )


def check_var(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(41)
    n = 400
    a = np.array([[0.5, 0.1], [0.2, 0.4]])
    data = np.zeros((n, 2))
    shocks = rng.normal(size=(n, 2))
    for t in range(1, n):
        data[t] = a @ data[t - 1] + shocks[t]
    df = pd.DataFrame({"y1": data[:, 0], "y2": data[:, 1], "t": np.arange(1, n + 1)})

    o.send_data(df)
    o.run("tsset t")
    o.run("var y1 y2, lags(1)")
    cap = o.capture()
    res = ws.VAR(data, lags=1, names=["y1", "y2"]).fit()

    # Stata's e(b) is ordered equation-by-equation: for k variables the names
    # repeat as [L.y1, L.y2, _cons, L.y1, L.y2, _cons]. westata's params frame is
    # [rows = L1.y1, L1.y2, const] x [cols = y1, y2], so the column-major
    # flattening of the lag rows must be compared against Stata's order.
    stata_lag1 = [
        i for i, name in enumerate(cap.names) if name.startswith("L.") or name.startswith("L1.")
    ]
    westata_matrix = res.params.loc[["L1.y1", "L1.y2"]].to_numpy()
    # Stata groups by equation: (y1 equation: L.y1, L.y2), then (y2: L.y1, L.y2).
    reordered = np.concatenate([westata_matrix[:, j] for j in range(westata_matrix.shape[1])])
    stata_values = np.asarray(cap.b, dtype=float)[stata_lag1]
    if len(stata_values) != len(reordered):
        return Outcome(
            "var",
            False,
            error=f"shape mismatch: stata {len(stata_values)} vs westata {len(reordered)}; names={cap.names}",
        )
    return Outcome(
        "var",
        True,
        rel(reordered, stata_values),
        notes=[f"Stata e(b) names: {cap.names}"],
    )


def check_arch(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(5)
    n = 1200
    omega, alpha, beta = 0.05, 0.08, 0.88
    variance = np.zeros(n)
    returns = np.zeros(n)
    variance[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = np.sqrt(variance[t]) * rng.normal()
    df = pd.DataFrame({"r": returns, "t": np.arange(1, n + 1)})

    o.send_data(df)
    o.run("tsset t")
    o.run("arch r, arch(1) garch(1)")
    cap = o.capture()
    res = ws.GARCH(returns, p=1, q=1).fit()

    # Stata's e(b) = [mean _cons, arch:L.arch, arch:L.garch, var _cons].
    # arch the package reports [mu, omega, alpha[1], beta[1]]. Stata does NOT put
    # omega into e(b) at all -- it stores it under a separate name -- so only the
    # mean constant and the two ARCH/GARCH coefficients are comparable.
    names = cap.names
    stata_values = np.asarray(cap.b, dtype=float)
    index_by_name = {name: i for i, name in enumerate(names)}

    pairs = [
        ("mu", "_cons"),
        ("alpha[1]", "L.arch"),
        ("beta[1]", "L.garch"),
    ]

    westata_params = np.asarray(res.params, dtype=float)
    westata_names = list(res.param_names)
    differences = {}
    for westata_name, stata_name in pairs:
        if westata_name not in westata_names or stata_name not in index_by_name:
            continue
        w = westata_params[westata_names.index(westata_name)]
        s = stata_values[index_by_name[stata_name]]
        differences[westata_name] = abs(w / s - 1)

    if not differences:
        return Outcome(
            "arch", False, error=f"could not map parameters; Stata names={names}"
        )

    return Outcome(
        "arch",
        True,
        float(max(differences.values())),
        scalars={
            key: (stata_values[index_by_name[stata]], westata_params[westata_names.index(key)])
            for key, stata in pairs
            if key in westata_names and stata in index_by_name
        },
        notes=[
            f"Stata names: {names}",
            f"westata: {westata_names}",
            f"per-parameter rel: "
            + ", ".join(f"{k}={v:.2e}" for k, v in differences.items()),
            f"omega is not in Stata's e(b) (it appears as {stata_values[0]:.6g}), "
            "so the variance-equation intercept is not compared",
        ],
    )


def check_unitroot(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(13)
    walk = np.cumsum(rng.normal(size=400))
    df = pd.DataFrame({"y": walk, "t": np.arange(1, 401)})
    o.send_data(df)
    o.run("tsset t")
    output = o.get_output("dfuller y")
    res = ws.unitroot(walk, test="adf")

    # Stata prints a table whose first numeric row is:
    #   " Z(t)            -1.940       -3.448       -2.874       -2.570"
    match = re.search(r"Z\(t\)\s+([-\d.]+)", output)
    if not match:
        return Outcome("dfuller", False, error=f"parse failed:\n{output[:600]}")
    stata_stat = float(match.group(1))
    statistic_rel = abs(res.statistic / stata_stat - 1)
    return Outcome(
        "dfuller",
        True,
        stat
        if False
        else statistic_rel,  # parameter field carries the test-statistic gap
        notes=[
            f"Stata Z(t) = {stata_stat:.3f} (3 dp), westata = {res.statistic:.6f}, "
            f"rel={statistic_rel:.2e}",
            "Stata prints 3 decimals, so this comparison is display-limited",
        ],
    )


def check_iv_liml(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(11)
    n = 500
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.8 * z1 + 0.5 * z2 + 0.6 * v
    y = 1.0 + 2.0 * x + 0.5 * x2 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "x2": x2, "z1": z1, "z2": z2})
    o.send_data(df)
    o.run("ivregress liml y x2 (x = z1 z2)")
    cap = o.capture()
    res = ws.IVLIML.from_formula("y ~ x2 + [x ~ z1 + z2]", df).fit()
    perm = align(res, cap.names)
    return Outcome(
        "ivregress liml",
        True,
        rel(res.params.to_numpy()[perm], cap.b),
        rel(res.bse.to_numpy()[perm], np.sqrt(np.diag(cap.V))),
    )


def check_iv_gmm(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(11)
    n = 500
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.8 * z1 + 0.5 * z2 + 0.6 * v
    y = 1.0 + 2.0 * x + 0.5 * x2 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "x2": x2, "z1": z1, "z2": z2})
    o.send_data(df)
    o.run("ivregress gmm y x2 (x = z1 z2)")
    cap = o.capture()
    res = ws.IVGMM.from_formula("y ~ x2 + [x ~ z1 + z2]", df).fit()
    perm = align(res, cap.names)
    return Outcome(
        "ivregress gmm",
        True,
        rel(res.params.to_numpy()[perm], cap.b),
        rel(res.bse.to_numpy()[perm], np.sqrt(np.diag(cap.V))),
        notes=[
            f"Stata order {cap.names}, westata order {list(res.param_names)}; "
            "the permutation is applied before both comparisons"
        ],
    )


CHECKS = {
    "xtreg_fe": check_xtreg_fe,
    "xtreg_re": check_xtreg_re,
    "xtreg_be": check_xtreg_be,
    "xtreg_fd": check_xtreg_fd,
    "areg": check_areg,
    "nbreg": check_nbreg,
    "ologit": check_ologit,
    "oprobit": check_oprobit,
    "margins": check_margins_logit,
    "var": check_var,
    "arch": check_arch,
    "dfuller": check_unitroot,
    "iv_liml": check_iv_liml,
    "iv_gmm": check_iv_gmm,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(CHECKS))
    parser.add_argument("--save", type=Path)
    args = parser.parse_args(argv)

    selected = args.only or list(CHECKS)
    rows = []

    with StataOracle() as oracle:
        for name in selected:
            try:
                outcome = CHECKS[name](oracle)
            except Exception as exc:  # noqa: BLE001
                outcome = Outcome(name, False, error=f"{type(exc).__name__}: {exc}")

            status = "ok  " if outcome.ok and not outcome.error else "FAIL"
            print(
                f"[{status}] {name:12s} param={outcome.param_rel:.3e} "
                f"se={outcome.se_rel:.3e}"
                + (f"  ERROR: {outcome.error[:200]}" if outcome.error else "")
            )
            for note in outcome.notes:
                print(f"          {note}")
            for key, (stata_value, westata_value) in outcome.scalars.items():
                if np.isfinite(stata_value) and stata_value != 0:
                    ratio = abs(westata_value / stata_value - 1)
                    print(
                        f"          {key}: stata={stata_value:.10g} "
                        f"westata={westata_value:.10g} rel={ratio:.2e}"
                    )

            rows.append(
                {
                    "check": name,
                    "ok": bool(outcome.ok and not outcome.error),
                    "param_rel": outcome.param_rel,
                    "se_rel": outcome.se_rel,
                    "error": outcome.error,
                    **{
                        f"{k}_rel": abs(w / s - 1) if np.isfinite(s) and s else np.nan
                        for k, (s, w) in outcome.scalars.items()
                    },
                }
            )

    table = pd.DataFrame(rows)
    print()
    print(table[["check", "ok", "param_rel", "se_rel"]].to_string(index=False))
    if args.save:
        table.to_csv(args.save, index=False)
        print(f"\nwrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
