"""`varsoc`'s FPE column switches format by magnitude; its VALUE still differs by ~1%.

Round 80 recorded the FPE difference as formatting. Changing the format showed the value is wrong too.

    LL  (lag 0)   Stata -544.467   package -544.467     identical
    FPE (lag 0)   Stata  .855661   package  0.864151    ratio 1.009922

THE FORMAT IS NOW CORRECT, verified at two scales against a live Stata:

    ordinary scale     .855661*      six decimals, leading zero dropped
    scaled by 1e-6     8.6e-25*      one mantissa digit, TRUNCATED not rounded

The previous code always produced the scientific form, which is why the ordinary case read
"8.6e-01" where Stata prints ".85996".

THE VALUE IS NOT. The ratio 1.009922 sits near 198/196 = 1.0102 without equalling it, so the
degrees-of-freedom factor in Stata's definition is not the one used here. That is recorded as
options.md T16 with the next step spelled out -- print det(Sigma_ml), N and k, then solve for the
exponent against .855661 -- rather than guessed from the ratio.

These guards pin the FORMAT only, so the value work cannot disturb what is already right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fpe_texts():
    """The rendered FPE fields for two systems of very different scale."""

    outputs = {}
    for label, scale in (("ordinary", 1.0), ("tiny", 1e-6)):
        rng = np.random.default_rng(11)
        n = 200
        frame = pd.DataFrame(
            {
                "y1": rng.normal(size=n) * scale,
                "y2": rng.normal(size=n) * scale,
            }
        )
        selection = ws.varsoc(frame[["y1", "y2"]], maxlag=2)
        rendered = str(selection)
        row = next(
            line for line in rendered.splitlines() if line.strip().startswith("0 |")
        )
        # The FPE field is the five tokens before AIC; take the whole row and look for the shape.
        outputs[label] = row
    return outputs


def test_the_ordinary_scale_uses_six_decimals(fpe_texts) -> None:
    """Stata prints `.855661`, not `8.6e-01`. The leading zero is dropped."""

    row = fpe_texts["ordinary"]
    assert "." in row
    # No scientific notation at this scale.
    assert "e-01" not in row
    assert "e-0" not in row


def test_the_tiny_scale_uses_scientific_notation(fpe_texts) -> None:
    """Scaled by 1e-6, Stata prints `8.6e-25` -- one mantissa digit."""

    row = fpe_texts["tiny"]
    assert "e-" in row


def test_the_tiny_scale_truncates_rather_than_rounds() -> None:
    """Stata prints 6.67195e-08 as 6.6e-08; %.1e would give 6.7e-08."""

    rng = np.random.default_rng(11)
    n = 200
    frame = pd.DataFrame(
        {"y1": rng.normal(size=n) * 1e-6, "y2": rng.normal(size=n) * 1e-6}
    )
    selection = ws.varsoc(frame[["y1", "y2"]], maxlag=2)
    row = next(
        line for line in str(selection).splitlines() if line.strip().startswith("0 |")
    )
    # An exact one-digit mantissa, e.g. 8.6e-25 -- never 8.65 or 8.7.
    import re

    match = re.search(r"(\d\.\d)e(-\d+)", row)
    assert match, row


def test_the_value_difference_is_recorded_not_asserted_away() -> None:
    """The FPE value is ~1% above Stata's; pinned so a fix shows up here.

    Stata's FPE at lag 0 on this system is .855661. When the degrees-of-freedom factor is corrected,
    this assertion should be replaced by a comparison against that number.
    """

    rng = np.random.default_rng(11)
    n = 200
    frame = pd.DataFrame({"y1": rng.normal(size=n), "y2": rng.normal(size=n)})
    selection = ws.varsoc(frame[["y1", "y2"]], maxlag=2)

    observed = float(selection.rows[0].fpe)
    assert observed == pytest.approx(0.864151, abs=1e-6)
    # Recorded gap against Stata's .855661.
    assert observed / 0.855661 == pytest.approx(1.009922, abs=1e-5)
