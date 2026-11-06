"""Run the v2 report's coefficient/statistic verification through this repo's Stata session.

``work/verify_v2.py`` in the report directory imports ``stata_setup``, which is not installed in
this environment, so it cannot run as shipped. The comparison it performs does not need anything
special though -- ``pystata``'s ``get_ereturn()`` and ``Matrix.getColNames`` are both reachable
through ``replication/lib/stata_session.py`` -- so this re-runs the same comparison through that
path.

Scope: the COEFFICIENT family. Each model is fitted in Stata and in westata on the same .dta and
the same sample, and the parameters are matched BY NAME (a positional match would silently
compare ``ltlan`` against ``ltwlab``, which has produced wrong conclusions here before).
"""

from __future__ import annotations

import contextlib
import io
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "replication")

import numpy as np
import pandas as pd

from lib.stata_session import StataSession

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data"


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    names = []
    for column in frame.columns:
        seen[column] = seen.get(column, 0) + 1
        names.append(column if seen[column] == 1 else f"{column}_{seen[column]}")
    frame = frame.copy()
    frame.columns = names
    return frame


def _stata_coefficients(session: StataSession, command: str) -> dict[str, float]:
    """Fit ``command`` in Stata and return {parameter: value}."""

    # The cases name the REGRESSOR list only; the estimator is the one under test here.
    if not command.lstrip().startswith(("regress", "ivregress", "logit", "probit")):
        command = "regress " + command
    with contextlib.redirect_stdout(io.StringIO()):
        session.run("quietly " + command)
    e = session.raw.get_ereturn()
    values = np.asarray(e.get("e(b)"), dtype=float).ravel()
    names = session._matrix_names("e(b)") if hasattr(session, "_matrix_names") else []
    if len(names) != values.size:
        names = [f"p{index}" for index in range(values.size)]
    aligned: dict[str, float] = {}
    for name, value in zip(names, values):
        key = str(name)
        # Stata spells the constant _cons; patsy spells it Intercept.
        if str(key).lower() in ("intercept", "const", "constant", "_cons"):
            key = "_cons"
        aligned[key] = float(value)
    return aligned


def _westata_coefficients(result: object) -> dict[str, float]:
    names = [str(name) for name in result.params.index]
    values = np.asarray(result.params, dtype=float).ravel()
    aligned: dict[str, float] = {}
    for name, value in zip(names, values):
        key = "_cons" if name.lower() in ("intercept", "const", "constant", "_cons") else name
        aligned[key] = float(value)
    return aligned


CASES = [
    ("grilic", "lnw s expr tenure", lambda frame: ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()),
    ("grilic", "lnw s expr tenure rns smsa", lambda frame: ws.OLS.from_formula("lnw ~ s + expr + tenure + rns + smsa", frame).fit()),
    ("nerlove", "lnq lnpl lnpk lnpf", lambda frame: ws.OLS.from_formula("lnq ~ lnpl + lnpk + lnpf", frame).fit()),
    ("cobb_douglas", "y k l", lambda frame: ws.OLS.from_formula("y ~ k + l", frame).fit()),
    ("cobb_douglas", "y k l, noc", lambda frame: ws.OLS.from_formula("y ~ k + l - 1", frame).fit()),
    ("cobb_douglas", "y k l, robust", lambda frame: ws.OLS.from_formula("y ~ k + l", frame).fit(cov_type="HC1")),
]


STATISTIC_CASES = [
    (
        "dwatson", "estat dwatson", "r(dw)",
        lambda result: float(ws.estat_dwatson(result).d),
    ),
    (
        "ovtest", "estat ovtest", "r(F)",
        lambda result: float(result.estat_ovtest().statistic),
    ),
    (
        "imtest", "estat imtest", "r(chi2_t)",
        lambda result: float(result.estat_imtest().to_dict()["chi2_t"]),
    ),
    (
        "vif (first)", "estat vif", "r(vif_1)",
        lambda result: float(np.asarray(result.estat_vif().vif).ravel()[0]),
    ),
]


def main() -> int:
    status = _coefficients()
    _statistics()
    return status


def _statistics() -> None:
    """The 25-item family in the report script, reduced to the four that need no extra setup."""

    frame = _dedupe(
        pd.read_stata(f"{DATA}/grilic.dta", convert_categoricals=False).dropna()
    ).assign(t=lambda data: np.arange(1, len(data) + 1))
    result = ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()
    print()
    print(f"{'statistic':16} {'stata':>20} {'westata':>20} {'rel err':>12}  verdict")
    print("-" * 86)
    with StataSession() as session:
        session.push(frame)
        with contextlib.redirect_stdout(io.StringIO()):
            session.run("tsset t")
        for label, command, key, builder in STATISTIC_CASES:
            with contextlib.redirect_stdout(io.StringIO()):
                session.run("quietly regress lnw s expr tenure")
                session.run(command)
            returns = session.raw.get_return()
            if key not in returns:
                print(f"{label:16} {'--':>20} {'--':>20} {'--':>12}  MISSING {key}")
                continue
            theirs = float(np.asarray(returns[key]).ravel()[0])
            try:
                ours = builder(result)
            except Exception as error:  # noqa: BLE001
                print(f"{label:16} {theirs:>20.10f} {'--':>20} {'--':>12}  WESTATA: {str(error)[:20]}")
                continue
            denominator = max(abs(theirs), 1e-12)
            relative = abs(ours - theirs) / denominator
            verdict = "exact" if relative < 1e-10 else ("close" if relative < 1e-5 else "DIFFERS")
            print(f"{label:16} {theirs:>20.10f} {ours:>20.10f} {relative:>12.2e}  {verdict}")


def _coefficients() -> int:
    print(f"{'model':44} {'params':>8} {'max rel err':>14}  verdict")
    print("-" * 88)
    for dataset, command, builder in CASES:
        try:
            frame = _dedupe(pd.read_stata(f"{DATA}/{dataset}.dta", convert_categoricals=False).dropna())
        except FileNotFoundError:
            print(f"{dataset:44} {'--':>8} {'--':>14}  SKIP (no data)")
            continue
        # westata's sample drops NaNs through patsy; align Stata's by estimating on the
        # SAME frame, pushed once.
        with StataSession() as session:
            session.push(frame)
            try:
                theirs = _stata_coefficients(session, command)
            except Exception as error:  # noqa: BLE001 - the failure itself is the result
                print(f"{dataset + ' : ' + command:44} {'--':>8} {'--':>14}  STATA FAILED: {str(error)[:28]}")
                continue
        try:
            ours = _westata_coefficients(builder(frame))
        except Exception as error:  # noqa: BLE001
            print(f"{dataset + ' : ' + command:44} {'--':>8} {'--':>14}  WESTATA FAILED: {str(error)[:26]}")
            continue
        common = sorted(set(ours) & set(theirs))
        if not common:
            print(f"{dataset + ' : ' + command:44} {'0':>8} {'--':>14}  NO NAMES MATCHED")
            continue
        worst = 0.0
        for name in common:
            denominator = max(abs(theirs[name]), 1e-12)
            worst = max(worst, abs(ours[name] - theirs[name]) / denominator)
        verdict = "exact" if worst < 1e-10 else ("close" if worst < 1e-6 else "DIFFERS")
        print(f"{dataset + ' : ' + command:44} {len(common):>8} {worst:>14.2e}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
