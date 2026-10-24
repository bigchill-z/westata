"""S5 -- ``didregress`` must not infer covariates from the frame.

Run:  .venv\\Scripts\\python.exe retest/fix/systems/repro_S5.py

Stata's ``didregress (y) (treated)`` names **no** covariate.  westata used to
infer every other time-varying numeric column and only warn, so the ATET moved
when an unrelated column was present.  The default is now "no covariate" and the
old behaviour is ``covariates="all"``.

Three rows, all on the retest's two-period panel (60 units x 2 periods, 30 units
treated in period 2):

  1. ``(y x1)``  -> covariates=["x1"]  (unchanged, must stay exact)
  2. ``(y)``     -> no covariates      (the reversal)
  3. ``(y)`` + an unrelated ``noise`` column, still no covariate
"""

from __future__ import annotations

import warnings

import numpy as np

from _common import Stata, banner, header, panel_two_period, rel_diff, row

import westata as ws


def stata_atet(st, command: str) -> tuple[float, float]:
    cap = st.estimate(command)
    index = cap.names.index("r1vs0.treated")
    return float(cap.b[index]), float(np.sqrt(cap.V[index, index]))


def main() -> int:
    panel = panel_two_period()
    with Stata() as st:
        st.send(panel)
        full_b, full_se = stata_atet(
            st, "didregress (y x1) (treated), group(unit) time(period)")
        bare_b, bare_se = stata_atet(
            st, "didregress (y) (treated), group(unit) time(period)")

    banner("S5.1  didregress (y x1): explicit covariate -- must be unchanged")
    model = ws.Didregress.from_formula(
        "y x1", "treated", panel, group="unit", time="period")
    res = model.fit()
    header()
    row("ATET", res.params["treated"], res.params["treated"], full_b)
    row("se(ATET)", res.bse["treated"], res.bse["treated"], full_se)
    row("b(x1)", res.params["x1"], res.params["x1"], None)
    print(f"  covariates = {model.covariates}")
    explicit_ok = (
        rel_diff([res.params["treated"]], [full_b]) <= 1e-9
        and rel_diff([res.bse["treated"]], [full_se]) <= 1e-9
    )

    banner("S5.2  didregress (y): no covariate, even with an unrelated column")
    # SAME frame, unrelated time-varying column present: before the fix this
    # pulled 'noise' in and moved the ATET.
    noisy = panel.copy()
    rng = np.random.default_rng(4242)
    noisy["noise"] = rng.normal(size=len(noisy))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bare_model = ws.Didregress.from_formula(
            "y", "treated", noisy, group="unit", time="period")
        bare_res = bare_model.fit()
            # (the fix must not need a warning at all here)
    print(f"  covariates           = {bare_model.covariates}")
    print(f"  inferred_covariates  = {bare_model.inferred_covariates}")
    print(f"  warnings             = {[str(w.message) for w in caught]}")
    header()
    row("ATET", bare_res.params["treated"], bare_res.params["treated"], bare_b)
    row("se(ATET)", bare_res.bse["treated"], bare_res.bse["treated"], bare_se)
    bare_ok = (
        rel_diff([bare_res.params["treated"]], [bare_b]) <= 1e-9
        and rel_diff([bare_res.bse["treated"]], [bare_se]) <= 1e-9
        and bare_model.covariates == []
    )

    banner("S5.3  covariates='all': the old behaviour is still available, explicitly")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        all_model = ws.Didregress.from_formula(
            "y", "treated", noisy, group="unit", time="period", covariates="all")
    print(f"  covariates          = {all_model.covariates}")
    print(f"  inferred_covariates = {all_model.inferred_covariates}")
    warned = [str(w.message) for w in caught]
    print(f"  warning             = {warned}")
    all_ok = (
        # "all" selects every time-varying numeric column: x1 (real covariate)
        # and noise (the unrelated one) both vary within units here.
        all_model.covariates == ["x1", "noise"]
        and all_model.inferred_covariates == ["x1", "noise"]
        and bool(warned)
    )
    # and it really changes the ATET (that is why the default mattered)
    all_res = all_model.fit()
    print(f"  ATET with covariates='all' = {float(all_res.params['treated'])!r}"
          f"  vs no-covariate {float(bare_res.params['treated'])!r}"
          f"  vs Stata (y) {bare_b!r}")

    print()
    print("  RESULT:", "PASS" if (explicit_ok and bare_ok and all_ok) else "FAIL")
    return 0 if (explicit_ok and bare_ok and all_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
