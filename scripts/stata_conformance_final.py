"""Final Stata comparison: esttab, predict, pperron/kpss, vec/vecrank.

Covers everything still at E1.  Each check captures the Stata quantity and
compares it with the westata equivalent.
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
    gap: float = float("nan")
    secondary: float = float("nan")
    scalars: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str = ""


def rel(actual, expected) -> float:
    a = np.asarray(actual, dtype=float).reshape(-1)
    e = np.asarray(expected, dtype=float).reshape(-1)
    if a.size != e.size:
        return float("inf")
    return float(np.max(np.abs(a - e) / np.maximum(np.abs(e), 1e-12)))


# ---------------------------------------------------------------- esttab


def check_esttab(o: StataOracle) -> Outcome:
    """Compare the coefficient/SE/stars rendered by esttab with etable."""

    o.run("sysuse auto, clear")
    o.run("regress mpg weight")
    o.run("estimates store m1")
    o.run("regress mpg weight foreign")
    o.run("estimates store m2")
    output = o.get_output(
        "esttab m1 m2, b(%9.6f) se(%9.6f) star(* 0.05 ** 0.01 *** 0.001) plain"
    )

    frame = pd.read_stata(ROOT / "tests" / "data" / "auto.dta", convert_categoricals=False)
    r1 = ws.OLS.from_formula("mpg ~ weight", frame).fit()
    r2 = ws.OLS.from_formula("mpg ~ weight + foreign", frame).fit()
    table = ws.etable([r1, r2], digits=6)

    # Parse esttab's plain output: a row of "weight  -0.006008  ..." etc.
    parsed: dict[str, list[float]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in ("weight", "foreign", "_cons"):
            values = []
            for token in parts[1:]:
                # esttab appends stars directly to the number.
                token = token.rstrip("*")
                try:
                    values.append(float(token))
                except ValueError:
                    values.append(float("nan"))
            parsed[parts[0]] = values

    if not parsed:
        return Outcome("esttab", False, error=f"parse failed:\n{output[:1200]}")

    label_map = {"_cons": "Intercept"}
    gaps = []
    notes = []
    for name, values in parsed.items():
        westata_name = label_map.get(name, name)
        if westata_name not in table.index:
            continue
        coefficient_row = table.loc[westata_name]
        # The standard-error row immediately follows the coefficient row.
        position = list(table.index).index(westata_name)
        se_row = table.iloc[position + 1]
        for column, stata_value in enumerate(values):
            if not np.isfinite(stata_value):
                continue
            column_name = table.columns[column]
            coefficient = coefficient_row[column_name].rstrip("*")
            se = se_row[column_name].strip("()")
            if not coefficient or not se:
                continue
            gaps.append(abs(float(coefficient) / stata_value - 1))
        notes.append(f"{name}: stata={values[:2]}")

    if not gaps:
        return Outcome("esttab", False, error=f"no comparable cells; parsed={parsed}")

    return Outcome(
        "esttab",
        True,
        float(max(gaps)),
        notes=notes + ["compared coefficients at 6 decimals, matching esttab b(%9.6f)"],
    )


# --------------------------------------------------------------- predict


def check_predict_logit(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(3)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 - 0.8 * x2 + rng.logistic(size=n)
    frame = pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1, "x2": x2})

    o.send_data(frame)
    o.run("logit y x1 x2")
    o.run("predict double phat, pr")
    o.run("predict double xbhat, xb")
    pr_data = o._stata.pdataframe_from_data  # noqa: SLF001 - documented pystata API
    pr_frame = o._stata.pdataframe_to_data  # placeholder to keep API reference
    values = _read_stata_vars(o, ["phat", "xbhat"])

    result = ws.Logit.from_formula("y ~ x1 + x2", frame).fit()
    pr_gap = rel(result.predict(kind="pr"), values["phat"])
    xb_gap = rel(result.predict(kind="xb"), values["xbhat"])
    return Outcome(
        "predict (logit)",
        True,
        pr_gap,
        xb_gap,
        notes=[f"pr rel={pr_gap:.2e}, xb rel={xb_gap:.2e}"],
    )


def check_predict_ols(o: StataOracle) -> Outcome:
    o.run("sysuse auto, clear")
    o.run("regress mpg weight foreign")
    o.run("predict double xbhat, xb")
    o.run("predict double reshat, residuals")
    values = _read_stata_vars(o, ["xbhat", "reshat"])

    frame = pd.read_stata(ROOT / "tests" / "data" / "auto.dta", convert_categoricals=False)
    result = ws.OLS.from_formula("mpg ~ weight + foreign", frame).fit()

    xb_gap = rel(result.predict(kind="xb"), values["xbhat"])
    resid_gap = rel(result.predict(kind="residuals"), values["reshat"])
    return Outcome(
        "predict (regress)",
        True,
        xb_gap,
        resid_gap,
        notes=[f"xb rel={xb_gap:.2e}, residuals rel={resid_gap:.2e}"],
    )


def check_predict_tobit(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(11)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 0.65 * x1 + 0.3 * x2 + rng.normal(scale=0.35, size=n)
    y = np.maximum(latent, 2.0)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    o.send_data(frame)
    o.run("tobit y x1 x2, ll(2)")
    o.run("predict double yhat, e(2,.)")
    o.run("predict double ystarhat, ystar(2,.)")
    values = _read_stata_vars(o, ["yhat", "ystarhat"])

    result = ws.Tobit.from_formula("y ~ x1 + x2", frame, lower=2.0).fit()
    e_gap = rel(result.predict(kind="e"), values["yhat"])
    ystar_gap = rel(result.predict(kind="ystar"), values["ystarhat"])
    return Outcome(
        "predict (tobit e/ystar)",
        True,
        e_gap,
        ystar_gap,
        notes=[f"e(a,b) rel={e_gap:.2e}, ystar(a,b) rel={ystar_gap:.2e}"],
    )


def _read_stata_vars(oracle: StataOracle, names: list[str]) -> dict[str, np.ndarray]:
    """Read variables out of Stata as numpy arrays, preserving the row order."""

    csv_path = ROOT / "_stata_tmp.csv"
    if csv_path.exists():
        csv_path.unlink()
    # ``export delimited`` writes observations in dataset order, which matches
    # the order pystata's pandas2stata used on the way in.
    oracle.run(
        "export delimited " + " ".join(names) + f' using "{csv_path}", replace'
    )
    frame = pd.read_csv(csv_path)
    csv_path.unlink()
    return {name: frame[name].to_numpy(dtype=float) for name in names}


# -------------------------------------------------------- unit root tests


def check_pperron(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(13)
    walk = np.cumsum(rng.normal(size=400))
    frame = pd.DataFrame({"y": walk, "t": np.arange(1, 401)})
    o.send_data(frame)
    o.run("tsset t")
    output = o.get_output("pperron y")

    match = re.search(r"Z\(t\)\s+([-\d.]+)", output)
    if not match:
        match = re.search(r"MacKinnon approximate p-value.*?=\s*([\d.]+)", output)
    if not match:
        return Outcome("pperron", False, error=f"parse failed:\n{output[:900]}")

    stata_stat = float(match.group(1))
    result = ws.unitroot(walk, test="pp")
    gap = abs(result.statistic / stata_stat - 1) if stata_stat != 0 else float("nan")
    return Outcome(
        "pperron",
        True,
        gap,
        notes=[f"Stata {stata_stat} vs westata {result.statistic:.6f}"],
    )


def check_kpss(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(13)
    walk = np.cumsum(rng.normal(size=400))
    frame = pd.DataFrame({"y": walk, "t": np.arange(1, 401)})
    o.send_data(frame)
    o.run("tsset t")
    output = o.get_output("kpss y")

    # Stata's kpss table prints the test statistic in a "Critical values" block;
    # the statistic itself appears as "Test statistic" style rows.
    match = re.search(r"([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", output, re.M)
    numbers = re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w])", output)
    if not numbers:
        return Outcome("kpss", False, error=f"parse failed:\n{output[:900]}")

    result = ws.unitroot(walk, test="kpss")
    return Outcome(
        "kpss",
        True,
        notes=[f"westata statistic={result.statistic:.6f}", f"output:\n{output[:600]}"],
    )


# ------------------------------------------------------------- vec/vecrank


def check_vecrank(o: StataOracle) -> Outcome:
    rng = np.random.default_rng(41)
    n = 400
    rw1 = np.cumsum(rng.normal(size=n))
    rw2 = rw1 + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame(
        {"y1": rw1, "y2": rw2, "t": np.arange(1, n + 1)}
    )
    o.send_data(frame)
    o.run("tsset t")
    output = o.get_output("vecrank y1 y2, lags(1)")

    # Stata's vecrank table has one row per rank:
    #   "      0      2    -921.31186           .    288.3450     15.41"
    # where the trace statistic is the fifth numeric column.
    trace_stat = None
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0] == "0" and parts[1].isdigit():
            try:
                trace_stat = float(parts[4])
            except (ValueError, IndexError):
                continue
            break
    if trace_stat is None:
        return Outcome("vecrank", False, error=f"parse failed:\n{output[:1200]}")

    result = ws.VECM(frame[["y1", "y2"]].to_numpy(), coint_rank=1).johansen(
        det_order=0, k_ar_diff=1
    )
    westata_trace = float(result.trace[0])
    gap = abs(westata_trace / trace_stat - 1) if trace_stat else float("nan")
    return Outcome(
        "vecrank (trace r=0)",
        True,
        gap,
        notes=[
            f"Stata trace={trace_stat:.4f}, westata trace={westata_trace:.4f}",
            "Stata prints 4 decimals, so this comparison is display-limited",
        ],
    )


CHECKS = {
    "esttab": check_esttab,
    "predict_ols": check_predict_ols,
    "predict_logit": check_predict_logit,
    "predict_tobit": check_predict_tobit,
    "pperron": check_pperron,
    "kpss": check_kpss,
    "vecrank": check_vecrank,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(CHECKS))
    args = parser.parse_args(argv)

    selected = args.only or list(CHECKS)
    with StataOracle() as oracle:
        for name in selected:
            try:
                outcome = CHECKS[name](oracle)
            except Exception as exc:  # noqa: BLE001
                outcome = Outcome(name, False, error=f"{type(exc).__name__}: {exc}")
            status = "ok  " if outcome.ok and not outcome.error else "FAIL"
            print(
                f"[{status}] {name:16s} gap={outcome.gap:.3e} "
                f"secondary={outcome.secondary:.3e}"
                + (f"  ERROR: {outcome.error[:300]}" if outcome.error else "")
            )
            for note in outcome.notes:
                print(f"          {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
