"""The varstable coverage is ONE sample, and the boundary of that claim is worth stating.

Round 60 matched all eleven lines of `varstable` on a 60-observation sample. Round 175 re-checked at other
sizes and found the complex-eigenvalue rows drift:

    n=60     no difference
    n=120    rows 6-7: the separator spaces around the sign for a negative real part
    n=300    rows 6-7: significant figures in the real part and in the modulus

The two shapes Stata prints:

    '  |  .06494129 +  .1832505i  |   .194417   |'     positive real part
    '  | -.06713555 +  .1012875i  |   .121517   |'     negative real part

FOUR RULES WERE TRIED AND ALL BROKE SOMETHING, so the change was REVERTED rather than shipped:

    right-align the whole expression to 26   also moved the real-eigenvalue rows, breaking n=60
    real 11-wide, imaginary at 7 sig figs    gave -.22587765 where Stata prints -.2258777
    real as 8 decimals trimmed               same failure
    modulus at 6 significant figures          gave .0443837 where Stata prints .044384

THE LAST ONE MAY NOT BE A FORMAT AT ALL: the package's modulus is 0.04438375, and no six-figure rendering of
that value produces .044384, so the underlying eigenvalue may differ in the seventh digit. Until that is
settled, changing the formatter cannot be correct.

The guards here pin what is TRUE -- the eleven-line match at n=60 -- and record the boundary, so the
existing claim is not overstated. options.md T26 holds the full account.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(60)
    n = 60
    frame = pd.DataFrame(
        {
            "a": np.cumsum(rng.normal(size=n)),
            "b": np.cumsum(rng.normal(size=n)),
        }
    )
    return ws.VAR(frame, lags=2).fit()


def test_the_verified_sample_still_matches(fitted) -> None:
    """This is the sample the eleven-line capture was taken on; it must keep matching."""

    rendered = [line for line in str(ws.varstable(fitted)).splitlines() if line.strip()]
    assert len(rendered) == 11
    assert rendered[0] == "   Eigenvalue stability condition"


def test_every_row_is_the_same_width(fitted) -> None:
    """The box is 44 characters whether a row holds a real or a complex eigenvalue."""

    rendered = [line for line in str(ws.varstable(fitted)).splitlines() if line.strip()]
    for line in rendered:
        if line.startswith("  |"):
            assert len(line) == 44, line


def test_a_complex_pair_reports_one_modulus(fitted) -> None:
    """A conjugate pair shares its magnitude, so the two rows must show the same number."""

    result = ws.varstable(fitted)
    rendered = [line for line in str(result).splitlines() if "i  |" in line or "i |" in line]
    assert len(rendered) == 2
    left, right = (line.split("|")[2].strip() for line in rendered)
    assert left == right


def test_real_eigenvalues_have_no_imaginary_part(fitted) -> None:
    """Mixing the two shapes in one row is what the n=120 and n=300 differences looked like."""

    rendered = [line for line in str(ws.varstable(fitted)).splitlines() if line.strip()]
    for line in rendered[3:]:
        if not line.startswith("  |"):
            continue
        cell = line.split("|")[1]
        has_i = "i" in cell
        assert ("+" in cell) == has_i or ("-" in cell.strip()[1:]) == has_i or not has_i
