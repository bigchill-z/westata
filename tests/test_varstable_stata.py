"""varstable: C13-04 scored 0.27 / 0.00 but the table is byte-identical to Stata.

Round 146 observed in passing that `ws.varstable(...)` reproduces Stata's output line for line,
but left no test behind. This module pins it.

Reference, from a live `var dinf unem, lags(1/2)` then `varstable` on macro_swatson.dta: 11 lines,
all matching exactly -- the box rule, the header, four eigenvalues (two real, one complex pair),
the modulus column, and the two conclusion lines.

The two complex eigenvalues print with the SAME real part and opposite imaginary parts, and the
modulus column repeats for the pair. That repetition is Stata's layout, not a rounding artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_TABLE = [
    "   Eigenvalue stability condition",
    "  +----------------------------------------+",
    "  |        Eigenvalue        |   Modulus   |",
    "  |--------------------------+-------------|",
    "  |     .82182               |    .82182   |",
    "  |   .7970223               |   .797022   |",
    "  |  -.2258777 +  .6292178i  |   .668533   |",
    "  |  -.2258777 -  .6292178i  |   .668533   |",
    "  +----------------------------------------+",
    "   All the eigenvalues lie inside the unit circle.",
    "   VAR satisfies stability condition.",
]


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].to_numpy(float)
    return ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()


def test_every_line_matches_stata(fitted) -> None:
    rendered = str(ws.varstable(fitted)).splitlines()
    assert rendered == STATA_TABLE


def test_the_complex_pair_shares_a_modulus(fitted) -> None:
    result = ws.varstable(fitted)
    # The result exposes the eigenvalues; the printed modulus repeats for a conjugate pair.
    moduli = [abs(complex(value)) for value in np.asarray(result.eigenvalues, dtype=complex)]
    assert moduli[2] == pytest.approx(moduli[3], abs=1e-12)
    assert moduli[2] == pytest.approx(0.668533, abs=5e-7)


def test_the_stability_conclusion_is_based_on_the_unit_circle(fitted) -> None:
    result = ws.varstable(fitted)
    assert bool(result.stable) is True
    assert max(abs(complex(v)) for v in np.asarray(result.eigenvalues, dtype=complex)) < 1.0
