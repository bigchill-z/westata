"""M6 -- ``hausman`` needs a ``constant`` switch (Stata's ``hausman ..., constant``).

Defect (retest/REPORT.md section 2.7, API gap): Stata exposes ``constant`` as an
opt-in option; westata's ``HausmanResult`` had no switch and always compared the
slopes only.

Stata's default excludes the constant for a good reason (a within-transformed
intercept and a GLS intercept estimate different population quantities, so
``V_fe - V_re`` is indefinite as soon as the constant enters), and ``constant``
simply extends the *same* quadratic form over all shared parameters:

    . hausman FEm REm              chi2(2) = .9525460961317572   p = .6210939
    . hausman FEm REm, constant    chi2(3) = .952546096131752    p = .8127322

This script reproduces both on the 40-entity x 8-period panel the retest used,
and verifies the manual quadratic form from Stata's own ``e(b)``/``e(V)``.

Run from the repository root::

    .venv/Scripts/python.exe retest/fix/postest/repro_M6_hausman_constant.py

Stata version: ``c(stata_version)`` = 19.5 (MP).  Captured 2026-10-24.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "retest" / "lib"))
sys.path.insert(0, str(ROOT / "src"))

from stata_probe import Stata  # noqa: E402

import westata as ws  # noqa: E402

NC = ROOT / "tests" / "conformance" / "stata_fixtures"


def rel(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1e-12)


def panel_frame(seed: int, n_entities: int, n_periods: int, corr: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(1, n_entities + 1), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    effect = np.repeat(rng.normal(0.0, 1.5, n_entities), n_periods)
    x1 = rng.normal(0.0, 1.0, n_entities * n_periods) + corr * effect
    x2 = rng.normal(0.0, 1.0, n_entities * n_periods)
    y = 1.0 + 0.8 * x1 - 0.5 * x2 + effect + rng.normal(0.0, 0.7, n_entities * n_periods)
    return pd.DataFrame({"id": entity, "year": time, "y": y, "x1": x1, "x2": x2})


def quadratic_form(difference, difference_cov) -> tuple[float, str, float]:
    symmetric = (difference_cov + difference_cov.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.all(eigenvalues > 0.0):
        return float(difference @ np.linalg.solve(symmetric, difference)), "solve", float(eigenvalues[0])
    return (
        float(difference @ (np.linalg.pinv(symmetric) @ difference)),
        "pinv",
        float(eigenvalues[0]),
    )


def main() -> int:
    for seed, corr, label in ((4242, 0.0, "positive definite"), (4242, 0.15, "indefinite")):
        frame = panel_frame(seed, 40, 8, corr)
        print(f"\n=== panel seed={seed}, corr={corr} ({label}) ===")
        with Stata() as st:
            print(f"Stata c(stata_version) = {st.version}")
            st.send(frame)
            st.run("xtset id year")
            st.quiet("xtreg y x1 x2, fe")
            st.quiet("estimates store FEm")
            fe_names = st.matrix_names("e(b)")
            fe_b = st.matrix("e(b)").ravel().copy()
            fe_V = st.matrix("e(V)").copy()
            st.quiet("xtreg y x1 x2, re")
            st.quiet("estimates store REm")
            re_names = st.matrix_names("e(b)")
            re_b = st.matrix("e(b)").ravel().copy()
            re_V = st.matrix("e(V)").copy()

            print(f"    Stata FE e(b) names = {fe_names}")
            print(f"    Stata RE e(b) names = {re_names}")
            for command in ("hausman FEm REm", "hausman FEm REm, constant"):
                st.quiet(command)
                print(
                    f"    {command:<26} chi2={st.scalar('r(chi2)'):.15g}"
                    f"  df={st.scalar('r(df)'):g}  p={st.scalar('r(p)'):.15g}"
                )

            # Independent check from Stata's own matrices: the quadratic form
            # over the shared parameters, with and without the constant.
            common = [name for name in fe_names if name in re_names]
            fe_index = [fe_names.index(name) for name in common]
            re_index = [re_names.index(name) for name in common]
            difference = fe_b[fe_index] - re_b[re_index]
            difference_cov = fe_V[np.ix_(fe_index, fe_index)] - re_V[np.ix_(re_index, re_index)]
            slopes = [i for i, name in enumerate(common) if name != "_cons"]
            print(
                "    manual (Stata's own e(b)/e(V)):"
                f" slopes-only chi2={quadratic_form(difference[slopes], difference_cov[np.ix_(slopes, slopes)])[0]:.15g}"
                f"  with-constant chi2={quadratic_form(difference, difference_cov)[0]:.15g}"
                f" ({quadratic_form(difference, difference_cov)[1]})"
            )

        fe = ws.PanelOLS.from_formula(
            "y ~ 1 + x1 + x2", frame, entity="id", time="year"
        ).fit(entity_effects=True)
        re = ws.RandomEffects.from_formula(
            "y ~ 1 + x1 + x2", frame, entity="id", time="year"
        ).fit()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            default = ws.hausman(fe, re)
            with_constant = ws.hausman(fe, re, constant=True)
        print(f"    westata default          chi2={default.statistic:.15g} df={default.df:g}"
              f" excluded={default.excluded}")
        print(f"    westata constant=True    chi2={with_constant.statistic:.15g}"
              f" df={with_constant.df:g} excluded={with_constant.excluded}")
        print(f"    warnings: {[str(w.message)[:60] for w in caught]}")

        # ``df`` is the only thing that may move when the constant's difference
        # is (near) zero: the statistic itself stays put, as Stata's does.
        print(
            f"    df: default {default.df:g} vs constant {with_constant.df:g}"
            "  (Stata: 2 vs 3)"
        )

    print("\n=== summary ===")
    print("Stata default: chi2(2)=.9525460961317572 p=.6210938706123512")
    print("Stata constant: chi2(3)=.952546096131752  p=.8127322054037279")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
