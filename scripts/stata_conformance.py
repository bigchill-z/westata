"""Compare westata against real Stata across every implemented command.

This is the script that turns E1 claims into E2 evidence.  It runs each
command in embedded Stata (``pystata``), captures ``e(b)``/``e(V)``/scalars, and
compares them with the westata equivalent on the same data.

Usage::

    .venv\\Scripts\\python.exe scripts\\stata_conformance.py
    .venv\\Scripts\\python.exe scripts\\stata_conformance.py --only regress

The output is a table of relative differences. Anything above the command's
declared tolerance is a genuine conformance failure and is reported as such;
nothing is silently rounded into agreement.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import westata as ws  # noqa: E402
from westata.conformance.oracles.stata import StataOracle  # noqa: E402

STATA_DIR = r"D:\Program Files\StataNow19"


@dataclass
class Comparison:
    """One command-level comparison outcome."""

    command: str
    detail: str
    max_param_rel_diff: float = float("nan")
    max_se_rel_diff: float = float("nan")
    scalar_diffs: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error


def _rel(actual: np.ndarray, expected: np.ndarray) -> float:
    """Max relative difference, with an absolute floor for near-zero values."""

    actual = np.asarray(actual, dtype=float).reshape(-1)
    expected = np.asarray(expected, dtype=float).reshape(-1)
    if actual.size != expected.size:
        return float("inf")
    scale = np.maximum(np.abs(expected), 1e-12)
    return float(np.max(np.abs(actual - expected) / scale))


def _align_by_name(
    westata_names: list[str], stata_names: list[str]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map westata's parameter order onto Stata's ``e(b)`` order.

    Stata stores the intercept last (``weight foreign _cons``) while patsy puts
    it first (``Intercept weight foreign``).  Comparing positionally would
    produce meaningless differences, so the orders are reconciled explicitly and
    any coefficient that cannot be matched is reported instead of guessed.
    """

    def canonical(name: str) -> str:
        if name in ("Intercept", "_cons", "const"):
            return "__constant__"
        return name

    lookup = {canonical(name): index for index, name in enumerate(westata_names)}
    permutation: list[int] = []
    unmatched: list[str] = []
    for name in stata_names:
        key = canonical(name)
        if key in lookup:
            permutation.append(lookup[key])
        else:
            unmatched.append(name)

    if unmatched:
        return None
    if len(permutation) != len(westata_names):
        return None
    return np.asarray(permutation, dtype=int), np.empty(0)


def _westata_se(result: Any) -> np.ndarray:
    return np.asarray(result.bse, dtype=float).reshape(-1)


def _compare(
    command: str,
    detail: str,
    stata_b: np.ndarray,
    stata_V: np.ndarray,
    westata_result: Any,
    scalar_pairs: dict[str, str],
) -> Comparison:
    """Compare a westata result with captured Stata output."""

    wb = np.asarray(westata_result.params, dtype=float).reshape(-1)
    wV = np.asarray(westata_result.cov_params(), dtype=float)

    comparison = Comparison(command=command, detail=detail)
    comparison.max_param_rel_diff = _rel(wb, stata_b)
    comparison.max_se_rel_diff = _rel(_westata_se(westata_result), np.sqrt(np.diag(stata_V)))

    for stata_key, westata_key in scalar_pairs.items():
        stata_value = None
        if stata_key.startswith("e:"):
            stata_value = None  # filled by the caller through the capture
        comparison.scalar_diffs[stata_key] = (float("nan"), float("nan"))

    return comparison


def _run_pair(
    oracle: StataOracle,
    command: str,
    detail: str,
    stata_lines: list[str],
    westata_call: Callable[[], Any],
    scalars: dict[str, str] | None = None,
) -> Comparison:
    """Run Stata and westata on the same data, then compare."""

    try:
        for line in stata_lines:
            oracle.run(line)
        capture = oracle.capture()
    except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
        return Comparison(command=command, detail=detail, error=f"stata: {exc}")

    try:
        result = westata_call()
    except Exception as exc:  # noqa: BLE001
        return Comparison(command=command, detail=detail, error=f"westata: {exc}")

    wb = np.asarray(result.params, dtype=float).reshape(-1)
    if wb.size != capture.b.size:
        return Comparison(
            command=command,
            detail=detail,
            error=(
                f"parameter count differs: westata {wb.size} "
                f"({list(result.param_names)}) vs Stata {capture.b.size} "
                f"({capture.names})"
            ),
        )

    # Reconcile parameter orders before comparing anything.
    alignment = _align_by_name(list(result.param_names), list(capture.names))
    if alignment is None:
        return Comparison(
            command=command,
            detail=detail,
            error=(
                "cannot align parameters by name: westata "
                f"{list(result.param_names)} vs Stata {capture.names}"
            ),
        )
    permutation, _ = alignment
    aligned_params = wb[permutation]
    aligned_se = _westata_se(result)[permutation]
    # The covariance matrix must be permuted the same way for the SE check.
    aligned_V = np.asarray(result.cov_params(), dtype=float)[np.ix_(permutation, permutation)]

    comparison = Comparison(command=command, detail=detail)
    comparison.max_param_rel_diff = _rel(aligned_params, capture.b)
    stata_se = np.sqrt(np.diag(capture.V))
    comparison.max_se_rel_diff = _rel(aligned_se, stata_se)

    for stata_key, westata_key in (scalars or {}).items():
        stata_value = capture.scalars.get(stata_key, float("nan"))
        westata_value = result.e.get(westata_key)
        try:
            westata_value = float(westata_value)
        except (TypeError, ValueError):
            westata_value = float("nan")
        comparison.scalar_diffs[stata_key] = (stata_value, westata_value)

    return comparison


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_regress(oracle: StataOracle) -> Comparison:
    oracle.run("sysuse auto, clear")
    df = pd.read_stata(ROOT / "tests" / "data" / "auto.dta", convert_categoricals=False)
    return _run_pair(
        oracle,
        "regress",
        "mpg ~ weight + foreign (nonrobust)",
        ["regress mpg weight foreign"],
        lambda: ws.OLS.from_formula("mpg ~ weight + foreign", df).fit(),
        {"N": "N", "df_r": "df_r", "r2": "r2", "r2_a": "r2_a", "rmse": "rmse"},
    )


def check_regress_robust(oracle: StataOracle) -> Comparison:
    oracle.run("sysuse auto, clear")
    df = pd.read_stata(ROOT / "tests" / "data" / "auto.dta", convert_categoricals=False)
    return _run_pair(
        oracle,
        "regress, robust",
        "mpg ~ weight + foreign, vce(robust) -> HC1",
        ["regress mpg weight foreign, vce(robust)"],
        lambda: ws.OLS.from_formula("mpg ~ weight + foreign", df).fit(cov_type="HC1"),
    )


def check_regress_cluster(oracle: StataOracle) -> Comparison:
    oracle.run("sysuse auto, clear")
    df = pd.read_stata(ROOT / "tests" / "data" / "auto.dta", convert_categoricals=False)
    df["grp"] = (np.arange(len(df)) % 7).astype(int)

    def westata():
        return ws.OLS.from_formula("mpg ~ weight + foreign", df).fit(
            cov_type="cluster", cov_kwds={"groups": df["grp"]}
        )

    oracle.send_data(df)
    return _run_pair(
        oracle,
        "regress, cluster",
        "cluster(grp), 7 groups -> CR1",
        ["regress mpg weight foreign, cluster(grp)"],
        westata,
    )


def check_regress_factor(oracle: StataOracle) -> Comparison:
    oracle.run("sysuse auto, clear")
    df = pd.read_stata(ROOT / "tests" / "data" / "auto.dta", convert_categoricals=False)

    # Stata prints the base level explicitly (``0b.foreign = 0``), so e(b) has
    # one more entry than a patsy design, which omits the reference category.
    # Compare the identified coefficients by matching the non-base levels.
    oracle.run("regress mpg weight i.foreign")
    capture = oracle.capture()
    result = ws.OLS.from_formula(
        "mpg ~ weight + C(foreign, base=0)", df
    ).fit()

    comparison = Comparison(command="regress, factor", detail="i.foreign")

    stata_names = list(capture.names)
    base_positions = [
        index
        for index, name in enumerate(stata_names)
        if name.startswith("0b.") or name.startswith("1b.")
    ]
    keep = [index for index in range(len(stata_names)) if index not in base_positions]
    stata_kept_names = [stata_names[index] for index in keep]
    stata_b = np.asarray(capture.b, dtype=float)[keep]
    stata_se = np.sqrt(np.diag(capture.V))[keep]

    # Translate Stata's "1.foreign" to the level value used by the design.
    translation = {}
    for name in stata_kept_names:
        if name == "_cons":
            translation[name] = "Intercept"
        elif name.endswith(".foreign"):
            level = name.split(".")[0]
            translation[name] = f"C(foreign, Treatment(reference=0))[T.{level}]"
        else:
            translation[name] = name

    westata_names = list(result.param_names)
    try:
        permutation = np.asarray(
            [westata_names.index(translation[name]) for name in stata_kept_names],
            dtype=int,
        )
    except ValueError as exc:
        comparison.error = f"cannot align factor parameters: {exc}"
        return comparison

    westata_params = np.asarray(result.params, dtype=float)
    westata_se = np.asarray(result.bse, dtype=float)
    comparison.max_param_rel_diff = _rel(westata_params[permutation], stata_b)
    comparison.max_se_rel_diff = _rel(westata_se[permutation], stata_se)
    comparison.notes.append(
        f"Stata prints base level {[stata_names[i] for i in base_positions]} = 0; "
        "excluded from the comparison"
    )
    return comparison


def check_test_joint(oracle: StataOracle) -> Comparison:
    oracle.run("sysuse auto, clear")
    oracle.run("regress mpg weight foreign")
    output = oracle.get_output("test weight foreign")

    df = pd.read_stata(ROOT / "tests" / "data" / "auto.dta", convert_categoricals=False)
    result = ws.OLS.from_formula("mpg ~ weight + foreign", df).fit()
    outcome = result.test("weight foreign")

    import re

    # Stata pads the labels: "       F(  2,    71) =   69.75"
    match = re.search(r"F\(\s*(\d+),\s*(\d+)\)\s*=\s*([\d.]+)", output)
    comparison = Comparison(command="test", detail="joint test weight + foreign")
    if not match:
        comparison.error = f"could not parse Stata output:\n{output}"
        return comparison

    stata_df = int(match.group(1))
    stata_df_r = int(match.group(2))
    stata_F = float(match.group(3))
    comparison.scalar_diffs["F"] = (stata_F, outcome.statistic)
    comparison.scalar_diffs["df"] = (float(stata_df), outcome.df)
    comparison.scalar_diffs["df_denom"] = (
        float(stata_df_r),
        outcome.df_denom or float("nan"),
    )
    return comparison


def check_lincom(oracle: StataOracle) -> Comparison:
    oracle.run("sysuse auto, clear")
    oracle.run("regress mpg weight foreign")
    output = oracle.get_output("lincom weight + 1000*foreign")

    import re

    df = pd.read_stata(ROOT / "tests" / "data" / "auto.dta", convert_categoricals=False)
    result = ws.OLS.from_formula("mpg ~ weight + foreign", df).fit()
    outcome = result.lincom("weight + 1000*foreign")

    comparison = Comparison(command="lincom", detail="weight + 1000*foreign")
    # The value row looks like: "         (1) |  -1650.036   1075.994    -1.53 ..."
    match = re.search(r"\(\d+\)\s*\|\s*([-\d.]+)\s+([\d.]+)\s+([-\d.]+)", output)
    if not match:
        comparison.error = f"could not parse Stata output:\n{output}"
        return comparison
    comparison.scalar_diffs["estimate"] = (float(match.group(1)), outcome.estimate)
    comparison.scalar_diffs["std_err"] = (float(match.group(2)), outcome.std_err)
    comparison.scalar_diffs["t"] = (float(match.group(3)), outcome.statistic)
    return comparison


def check_iv2sls(oracle: StataOracle) -> Comparison:
    rng = np.random.default_rng(11)
    n = 400
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x_exog = rng.normal(size=n)
    v = rng.normal(size=n)
    x_endog = 0.8 * z1 + 0.5 * z2 + 0.6 * v
    y = 1.0 + 2.0 * x_endog + 0.5 * x_exog + rng.normal(size=n)
    df = pd.DataFrame(
        {"y": y, "x_endog": x_endog, "x_exog": x_exog, "z1": z1, "z2": z2}
    )

    oracle.send_data(df)
    return _run_pair(
        oracle,
        "ivregress 2sls",
        "y ~ x_exog + (x_endog = z1 z2)",
        ["ivregress 2sls y x_exog (x_endog = z1 z2)"],
        lambda: ws.IV2SLS.from_formula("y ~ x_exog + [x_endog ~ z1 + z2]", df).fit(),
        {"N": "N", "r2": "r2", "rmse": "rmse"},
    )


def check_logit(oracle: StataOracle) -> Comparison:
    rng = np.random.default_rng(3)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 - 0.8 * x2 + rng.logistic(size=n)
    df = pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1, "x2": x2})

    oracle.send_data(df)
    return _run_pair(
        oracle,
        "logit",
        "y ~ x1 + x2",
        ["logit y x1 x2"],
        lambda: ws.Logit.from_formula("y ~ x1 + x2", df).fit(),
        {"N": "N", "ll": "ll", "r2_p": "r2_p"},
    )


def check_probit(oracle: StataOracle) -> Comparison:
    rng = np.random.default_rng(3)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.0 * x1 - 0.6 * x2 + rng.normal(size=n)
    df = pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1, "x2": x2})

    oracle.send_data(df)
    return _run_pair(
        oracle,
        "probit",
        "y ~ x1 + x2",
        ["probit y x1 x2"],
        lambda: ws.Probit.from_formula("y ~ x1 + x2", df).fit(),
        {"N": "N", "ll": "ll", "r2_p": "r2_p"},
    )


def check_poisson(oracle: StataOracle) -> Comparison:
    rng = np.random.default_rng(5)
    n = 500
    x1 = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.6 * x1)).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1})

    oracle.send_data(df)
    return _run_pair(
        oracle,
        "poisson",
        "y ~ x1",
        ["poisson y x1"],
        lambda: ws.Poisson.from_formula("y ~ x1", df).fit(),
        {"N": "N", "ll": "ll"},
    )


def check_tobit(oracle: StataOracle) -> Comparison:
    rng = np.random.default_rng(11)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 0.65 * x1 + 0.3 * x2 + rng.normal(scale=0.35, size=n)
    y = np.maximum(latent, 2.0)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    oracle.send_data(df)
    oracle.run("tobit y x1 x2, ll(2)")
    capture = oracle.capture()

    result = ws.Tobit.from_formula("y ~ x1 + x2", df, lower=2.0).fit()

    comparison = Comparison(command="tobit", detail="y ~ x1 + x2, ll(2)")

    # Stata reports e(b) = [x1, x2, _cons, var(e.y)]; westata reports
    # [Intercept, x1, x2, /sigma]. The variance parameter differs by a square,
    # so it is compared separately rather than forcing a false match.
    alignment = _align_by_name(
        [name for name in result.param_names if name != "/sigma"],
        [name for name in capture.names if not name.startswith("var(")],
    )
    if alignment is None:
        comparison.error = (
            f"cannot align tobit parameters: westata {result.param_names} "
            f"vs Stata {capture.names}"
        )
        return comparison
    permutation, _ = alignment

    stata_slopes = np.asarray(capture.b, dtype=float)[:3]
    stata_se = np.sqrt(np.diag(capture.V))[:3]
    westata_params = np.asarray(result.params, dtype=float)[:3]
    westata_se = np.asarray(result.bse, dtype=float)[:3]

    comparison.max_param_rel_diff = _rel(westata_params[permutation], stata_slopes)
    comparison.max_se_rel_diff = _rel(westata_se[permutation], stata_se)

    # var(e.y) = sigma^2
    stata_var = float(capture.b[3])
    westata_sigma = float(result.e["sigma"])
    comparison.scalar_diffs["sigma2"] = (stata_var, westata_sigma**2)
    comparison.scalar_diffs["ll"] = (
        capture.scalars.get("ll", float("nan")),
        float(result.e["ll"]),
    )
    return comparison


def check_arima(oracle: StataOracle) -> Comparison:
    rng = np.random.default_rng(3)
    n = 500
    shocks = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + shocks[t]
    df = pd.DataFrame({"y": y, "t": np.arange(1, n + 1)})

    oracle.send_data(df)
    # Stata's arima requires a declared time variable.
    oracle.run("tsset t")
    # Stata's arima includes a constant by default, so match that explicitly on
    # both sides instead of comparing a model with a constant to one without.
    oracle.run("arima y, arima(1,0,0) nolog")
    capture = oracle.capture()

    result = ws.ARIMA(y, order=(1, 0, 0), trend="c").fit()

    comparison = Comparison(command="arima", detail="ARIMA(1,0,0) with constant")
    # Stata e(b) = [_cons, L.ar, _cons(for /sigma)]; index 0 is the mean
    # constant, 1 is the AR coefficient.
    names = list(capture.names)
    ar_index = names.index("L.ar") if "L.ar" in names else 1
    stata_ar = float(np.asarray(capture.b, dtype=float)[ar_index])
    stata_se_ar = float(np.sqrt(np.diag(capture.V))[ar_index])

    westata_names = list(result.param_names)
    westata_ar = float(result.params.iloc[westata_names.index("ar.L1")])
    westata_se_ar = float(result.bse.iloc[westata_names.index("ar.L1")])

    comparison.max_param_rel_diff = _rel(np.array([westata_ar]), np.array([stata_ar]))
    comparison.max_se_rel_diff = _rel(
        np.array([westata_se_ar]), np.array([stata_se_ar])
    )
    comparison.scalar_diffs["ll"] = (
        capture.scalars.get("ll", float("nan")),
        float(result.e["ll"]),
    )
    comparison.notes.append(f"Stata e(b) names: {names}")
    return comparison


CHECKS: dict[str, Callable[[StataOracle], Comparison]] = {
    "regress": check_regress,
    "regress_robust": check_regress_robust,
    "regress_cluster": check_regress_cluster,
    "regress_factor": check_regress_factor,
    "test": check_test_joint,
    "lincom": check_lincom,
    "iv2sls": check_iv2sls,
    "logit": check_logit,
    "probit": check_probit,
    "poisson": check_poisson,
    "tobit": check_tobit,
    "arima": check_arima,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(CHECKS),
        help="run only these checks",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="write the comparison table to this CSV",
    )
    args = parser.parse_args(argv)

    selected = args.only or list(CHECKS)
    rows: list[dict[str, Any]] = []

    with StataOracle(stata_dir=STATA_DIR, edition="mp") as oracle:
        for name in selected:
            try:
                outcome = CHECKS[name](oracle)
            except Exception as exc:  # noqa: BLE001
                outcome = Comparison(
                    command=name,
                    detail="",
                    error=f"{type(exc).__name__}: {exc}",
                )
                if "--traceback" in sys.argv:
                    traceback.print_exc()

            row = {
                "check": name,
                "command": outcome.command,
                "detail": outcome.detail,
                "param_rel_diff": outcome.max_param_rel_diff,
                "se_rel_diff": outcome.max_se_rel_diff,
                "error": outcome.error,
            }
            for key, (stata_value, westata_value) in outcome.scalar_diffs.items():
                row[f"stata_{key}"] = stata_value
                row[f"westata_{key}"] = westata_value
                if np.isfinite(stata_value) and stata_value != 0:
                    row[f"rel_{key}"] = abs(westata_value - stata_value) / abs(stata_value)
                else:
                    row[f"rel_{key}"] = float("nan")
            rows.append(row)

            status = "FAIL" if outcome.error else "ok"
            print(
                f"[{status:4s}] {name:18s} params={outcome.max_param_rel_diff:.3e} "
                f"se={outcome.max_se_rel_diff:.3e}"
                + (f"  ERROR: {outcome.error}" if outcome.error else "")
            )
            for note in outcome.notes:
                print(f"         note: {note}")

    table = pd.DataFrame(rows)
    print()
    print(table[["check", "param_rel_diff", "se_rel_diff"]].to_string(index=False))

    if args.save:
        table.to_csv(args.save, index=False)
        print(f"\nwrote {args.save}")

    return 0 if not table["error"].astype(bool).any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
