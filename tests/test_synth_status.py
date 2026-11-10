"""synth: the ado IS readable, which changes the feasibility of implementing it.

Recorded because an earlier round concluded synth was out of reach. That conclusion was about
`rdrobust`'s selector and `dfgls`, not about synth, but the report's section 7.3 lists "synthetic
control, whole chapter blank" alongside them, so the same pessimism had attached to it.

WHAT ROUND 125 ESTABLISHED:

`~/ado/plus/s/synth.ado` exists and is 931 lines of readable Stata. Only the Mata subroutines live
in a binary `.mlib` (`lsynth_mata_subr.mlib`, 27 KB, `file` reports "data"), so the algorithm's
SHELL is available even though the inner optimiser is not.

The shell shows the full pipeline, which is the Abadie-Diamond-Hainmueller procedure:

    mata: normalize("Xtr", "Xco")                 standardise predictors
    mata: regsval("Xtr","Xco","Ztr","Zco")        regression-based V weights (unless customV)
    mata: normweights("sval")                     normalise custom V
    ... lossfunction optimisation over W ...       the part in the .mlib
    optimisation converged

It also shows the data expectations that made a naive attempt fail: predictors are only present in
CERTAIN years (basque.dta carries them on odd years 1961-1969, and one predictor is missing in
1963 for the treated unit), and synth averages over an explicit period rather than requiring a
complete panel. My first attempt dropped rows with any missing predictor, which collapsed the panel
to ONE row per region; the command instead warns "control units: ... missing obs for predictor
sec_agriculture in period 1966 -ignored for averaging" and carries on.

WHAT REMAINS OUT OF REACH: the optimiser in the .mlib. Reimplementing ADH's nested optimisation
(inner W given V, outer V via regression-based weights) is a substantial piece of work, and the
ATET-style acceptance criterion -- reproducing Stata's RMSPE and weights to several digits -- cannot
be checked until it exists.

So: NOT implemented, but the feasibility judgement recorded earlier is revised. This is a
"readable shell + binary core" problem like dfgls, not a "no source at all" problem.

RESOLVED 2026-11-09 (B-grade implementation):
The "optimiser out of reach" conclusion above was itself too pessimistic.  P0 black-box
identification solved BOTH compiled Mata kernels exactly (normalize = sample SD including
the treated unit; regsval = sum of squared per-period cross-sectional OLS coefficients),
and showed e(V_matrix) is directly exportable while regsval is a deterministic pure
function of its four matrix arguments.  See scripts/verify_synth_mata_formulas.py and
docs/superpowers/specs/2026-11-09-synth-b-design.md.

`ws.synth` now exists for the DEFAULT path (evidence E2).  What genuinely remains out
of reach is only Stata's closed-source `ml` nested optimiser and the C++ QP plugin's
exact stopping point: H is rank-deficient, so RMSPE differs by up to 5.6e-4 while the
reported W matches Stata exactly at 3 decimals.  That gap is measured and disclosed,
not hidden (registry.yml).  The guard below was deliberately converted from absence
to presence when the estimator landed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

DATA = r"D:\myCodeProject\test_westata\data\basque.dta"
COVARIATES = [
    "sec_agriculture", "sec_energy", "sec_industry", "sec_construction",
    "school_illit", "school_prim", "school_med", "school_high", "school_post_high",
    "popdens", "invest",
]


def test_the_package_exports_synth() -> None:
    """The former absence guard, deliberately retired when B-grade synth landed."""

    import westata as ws

    assert hasattr(ws, "synth")
    assert callable(ws.synth)
    assert hasattr(ws, "SynthResult")


@pytest.fixture(scope="module")
def basque():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("basque.dta is not available")


def test_the_predictors_are_sparse_by_year(basque) -> None:
    """The reason a naive dropna collapses the panel, recorded as a fact about the data.

    gdpcap is present every year; the socio-economic predictors only on odd years. Dropping rows
    with any missing predictor leaves one row per region, which is not a panel at all.
    """

    assert int(basque["gdpcap"].notna().sum()) > 600
    present_years = sorted(basque.dropna(subset=["sec_agriculture"])["year"].unique())
    assert len(present_years) < len(sorted(basque["year"].unique())) / 3
    # And a full dropna leaves exactly one row per region.
    trimmed = basque.dropna(subset=COVARIATES + ["gdpcap"])
    assert trimmed.groupby("regionno").size().max() == 1


def test_the_treated_unit_and_period_exist(basque) -> None:
    """Basque Country is regionno 17 and 1975 is in range -- the two arguments synth needs."""

    basque_region = basque[basque["regionname"] == "Basque Country (Pais Vasco)"]
    assert set(basque_region["regionno"].unique()) == {17.0}
    assert 1975 in set(basque["year"].unique())
    assert basque["year"].min() == 1955 and basque["year"].max() == 1997
