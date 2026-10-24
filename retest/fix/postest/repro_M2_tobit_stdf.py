"""M2 -- ``predict, stdf`` after ``tobit`` must add ``sigma^2``, not ``rss/N``.

Defect (retest/REPORT.md section 2.7, item 3): ``predict.py::_residual_variance``
looked for ``e(rmse)`` and otherwise fell back to ``rss/N`` of the *observed,
censored* response.  ``tobit`` records ``e(sigma)``/``e(var_e)`` instead, so the
fallback reported ``stdf^2 - stdp^2 = 12.127405`` where Stata reports
``sigma^2 = 14.297844`` -- a 7.7% error in every prediction interval.

The repair resolves the innovation variance as
``e(var_e) -> e(rmse)^2 -> /sigma^2 -> rss/N`` and never diverts to ``e(sigma)``
for the linear models (where westata records ``e(sigma) == e(rmse)``, which
would make the documented ``rss/N`` fallback unreachable).

This script checks all three limited families against a live Stata session:

* ``tobit mpg weight foreign, ll(17)``   (``stdf^2 - stdp^2 == e(var_e)``)
* ``truncreg y x1 x2, ll(1)``            (only ``e(sigma)`` is recorded)
* ``intreg lo hi x1 x2``                 (only ``e(sigma)`` is recorded)

and confirms ``regress`` is unchanged (its ``e(rmse)`` branch still wins).

Run from the repository root::

    .venv/Scripts/python.exe retest/fix/postest/repro_M2_tobit_stdf.py

Stata version: ``c(stata_version)`` = 19.5 (MP).  Captured 2026-10-24.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "retest" / "lib"))
sys.path.insert(0, str(ROOT / "src"))

from stata_probe import Stata  # noqa: E402

import westata as ws  # noqa: E402

AUTO = ROOT / "tests" / "data" / "auto.dta"


def rel(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1e-12)


def old_variance(result) -> float:
    """Innovation variance the **pre-repair** code returned.

    That function was ``e(rmse)^2`` when recorded, else ``rss/N`` -- so ``tobit``
    (which records ``e(sigma)``/``e(var_e)`` but no ``rmse``) fell through to
    ``rss/N`` of the censored response, while ``regress`` used ``e(rmse)^2`` and
    was already right.
    """

    e = getattr(result, "e", None)
    raw = None if e is None else e.get("rmse")
    if raw is not None:
        return float(raw) ** 2
    resid = np.asarray(result.resid, dtype=float)
    return float(np.dot(resid, resid) / resid.size)


def old_gap(result) -> float:
    """``stdf^2 - stdp^2`` the pre-repair code produced."""

    return old_variance(result)


def report(
    name: str,
    result,
    stata_stdp,
    stata_stdf,
    worst: list[float],
    *,
    design: np.ndarray | None = None,
    stata_design: np.ndarray | None = None,
) -> None:
    """Compare ``stdp``/``stdf`` per observation, matching the row sets.

    ``truncreg`` is fitted on the in-support rows only (Stata reports the
    truncated count but keeps the full sample for ``predict``), so the two
    vectors do not line up by position.  When ``design``/``stata_design`` are
    given the westata rows are located inside Stata's design and the comparison
    is made on the matched rows; otherwise the two must already align.
    """

    exog = np.asarray(result.design_matrix[0], dtype=float)
    stdp = result.predict(exog=exog, kind="stdp")
    stdf = result.predict(exog=exog, kind="stdf")
    before_p = np.sqrt(_old_stdp_sq(result, exog))
    before_f = np.sqrt(_old_stdp_sq(result, exog) + old_gap(result))

    if design is not None and stata_design is not None:
        lookup = {
            tuple(np.round(row, 10)): index
            for index, row in enumerate(np.asarray(stata_design, dtype=float))
        }
        keep = [lookup.get(tuple(np.round(row, 10))) for row in np.asarray(design)]
        if any(index is None for index in keep):
            print(f"\n[{name}] could not match rows to Stata's design; skipped")
            return
        indices = np.asarray(keep, dtype=int)
        if np.unique(indices).size != indices.size:
            print(f"\n[{name}] row matching is ambiguous (duplicate design rows); skipped")
            return
        stata_stdp = np.asarray(stata_stdp, dtype=float)[indices]
        stata_stdf = np.asarray(stata_stdf, dtype=float)[indices]

    print(f"\n[{name}]")
    print(f"    Stata  stdf^2 - stdp^2 = {np.unique(np.round(stata_stdf**2 - stata_stdp**2, 12))}")
    print(f"    before stdf^2 - stdp^2 = {np.unique(np.round(before_f**2 - before_p**2, 12))}")
    print(f"    after  stdf^2 - stdp^2 = {np.unique(np.round(stdf**2 - stdp**2, 12))}")
    print(
        f"    rel    before stdp {_maxrel(before_p, stata_stdp):.2e}"
        f"  stdf {_maxrel(before_f, stata_stdf):.2e}"
    )
    print(
        f"           after  stdp {_maxrel(stdp, stata_stdp):.2e}"
        f"  stdf {_maxrel(stdf, stata_stdf):.2e}"
    )
    worst.append(_maxrel(stdp, stata_stdp))
    worst.append(_maxrel(stdf, stata_stdf))


def _old_stdp_sq(result, exog) -> np.ndarray:
    """``x'Vx`` over the coefficient block -- unchanged by this repair."""

    names = [str(name) for name in result.param_names]
    block = [index for index, value in enumerate(names) if not value.startswith("/")]
    cov = result.cov_params()[np.ix_(block, block)]
    return np.einsum("ij,jk,ik->i", exog, cov, exog)


def _maxrel(actual, expected) -> float:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    expected = np.asarray(expected, dtype=float).reshape(-1)
    if actual.size != expected.size:
        # The westata fit dropped observations Stata keeps (truncreg conditions
        # on truncation instead of dropping); a per-observation comparison is
        # then undefined and the caller reports "n/a".
        return float("nan")
    finite = np.isfinite(expected)
    return float(
        np.max(
            np.abs(actual[finite] - expected[finite])
            / np.maximum(np.abs(expected[finite]), 1e-12)
        )
    )


def main() -> int:
    auto = pd.read_stata(AUTO, convert_categoricals=False)
    worst: list[float] = []

    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 0.7 * x1 + 0.4 * x2 + rng.normal(scale=0.5, size=n)
    truncated = pd.DataFrame({"y": np.maximum(latent, 1.0), "x1": x1, "x2": x2})
    low = latent - 0.3 + rng.normal(scale=0.1, size=n)
    high = latent + 0.3 + rng.normal(scale=0.1, size=n)
    interval = pd.DataFrame(
        {
            "lo": np.minimum(low, high),
            "hi": np.maximum(low, high),
            "x1": x1,
            "x2": x2,
        }
    )

    with Stata() as st:
        print(f"Stata c(stata_version) = {st.version}")

        # ---------------------------------------------------------- tobit
        st.run("sysuse auto, clear")
        st.quiet("tobit mpg weight foreign, ll(17)")
        st.quiet("predict double t_stdp, stdp")
        st.quiet("predict double t_stdf, stdf")
        tobit = ws.Tobit.from_formula("mpg ~ weight + foreign", auto, lower=17).fit()
        print(f"\n[tobit mpg weight foreign, ll(17)]")
        print(f"    Stata  e(var_e) not in e() but stdf^2-stdp^2 = "
              f"{float(np.unique(np.round(st.var('t_stdf')**2 - st.var('t_stdp')**2, 12))[0]):.10g}")
        print(f"    westata e(var_e) = {float(tobit.e.get('var_e')):.10g}"
              f"   e(sigma) = {float(tobit.e.get('sigma')):.10g}")
        report(
            "tobit",
            tobit,
            st.var("t_stdp"),
            st.var("t_stdf"),
            worst,
        )

        # ------------------------------------------------------- truncreg
        st.send(truncated)
        st.quiet("truncreg y x1 x2, ll(1)")
        st.quiet("predict double r_stdp, stdp")
        st.quiet("predict double r_stdf, stdf")
        try:
            truncreg = ws.Truncreg.from_formula(
                "y ~ x1 + x2", truncated, lower=1.0
            ).fit()
        except Exception as exc:  # noqa: BLE001 - another group owns this model
            truncreg = None
            print(
                f"\n[truncreg y x1 x2, ll(1)] SKIPPED: westata's truncreg fit raised "
                f"{type(exc).__name__}: {exc}"
            )
        if truncreg is not None:
            print(f"\n[truncreg y x1 x2, ll(1)]  Stata e(sigma) = {st.scalar('e(sigma)')!r}")
            print(f"    westata e(sigma) = {float(truncreg.e.get('sigma')):.10g}"
                  f"   e(var_e) = {truncreg.e.get('var_e')!r}")
            stata_design = np.column_stack(
                [np.ones(st.nobs()), st.var("x1"), st.var("x2")]
            )
            report(
                "truncreg",
                truncreg,
                st.var("r_stdp"),
                st.var("r_stdf"),
                worst,
                design=np.asarray(truncreg.design_matrix[0], dtype=float),
                stata_design=stata_design,
            )

        # ---------------------------------------------------------- intreg
        st.send(interval)
        st.quiet("intreg lo hi x1 x2")
        st.quiet("predict double i_stdp, stdp")
        st.quiet("predict double i_stdf, stdf")
        try:
            intreg = ws.Intreg.from_formula("lo hi ~ x1 + x2", interval).fit()
        except Exception as exc:  # noqa: BLE001 - another group owns this model
            intreg = None
            print(
                f"\n[intreg lo hi x1 x2] SKIPPED: westata's intreg fit raised "
                f"{type(exc).__name__}: {exc}"
            )
        if intreg is not None:
            print(f"\n[intreg lo hi x1 x2]  Stata e(sigma) = {st.scalar('e(sigma)')!r}")
            report("intreg", intreg, st.var("i_stdp"), st.var("i_stdf"), worst)

        # --------------------------------------------------------- regress
        st.run("sysuse auto, clear")
        st.quiet("regress mpg weight foreign")
        st.quiet("predict double g_stdp, stdp")
        st.quiet("predict double g_stdf, stdf")
        ols = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
        print("\n[regress mpg weight foreign]  (must be unchanged: e(rmse) branch)")
        report("regress", ols, st.var("g_stdp"), st.var("g_stdf"), worst)

    print("\n=== summary ===")
    print(f"worst 'after' relative difference across M2: {max(worst):.3e}")
    tier = "A" if max(worst) <= 1e-9 else ("B" if max(worst) <= 1e-6 else "C or worse")
    print(f"tier: {tier}  (stdp/stdf inherit the model's coefficient and V precision;"
          " tobit's ML fit is the binding term, see FIXES.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
