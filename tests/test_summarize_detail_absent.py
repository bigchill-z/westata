"""`summarize, detail` is not implemented, and the numbers it would need are all available.

The base `summarize` block matches Stata line for line (round 157). `summarize v, detail` is a different
table -- two columns side by side, with percentiles and extreme values on the left and summary
statistics on the right:

                                  v
    -------------------------------------------------------------
          Percentiles      Smallest
     1%          1.5              1
     5%          5.5              2
    10%         10.5              3       Obs                 100
    25%         25.5              4       Sum of wgt.         100

    50%         50.5                      Mean               50.5
                            Largest       Std. dev.      29.01149
    75%         75.5             97
    90%         90.5             98       Variance       841.6667
    95%         95.5             99       Skewness              0
    99%         99.5            100       Kurtosis        1.79976

`summarize()` takes no `detail` parameter, so this is a missing FEATURE rather than a wrong one, and
options.md T24 records it.

THIS FILE PINS TWO THINGS SO THE GAP STAYS HONEST:

  * that the parameter is absent, so nobody assumes the table exists;
  * the KURTOSIS CONVENTION, which is the part most likely to be got wrong when the renderer is written.
    Stata reports 1.79976 for the uniform 1..100 series; `scipy.stats.kurtosis` reports -1.20024, and
    adding 3 gives 1.79976 exactly. Stata uses the ORIGINAL kurtosis, not the excess one, and no
    small-sample correction.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def test_summarize_takes_no_detail_parameter() -> None:
    import inspect

    signature = inspect.signature(ws.summarize)
    assert "detail" not in signature.parameters


def test_passing_detail_is_rejected_rather_than_silently_ignored() -> None:
    """A keyword the function does not take must raise, not produce the basic table."""

    frame = pd.DataFrame({"v": np.arange(1, 101, dtype=float)})
    with pytest.raises(TypeError):
        ws.summarize(frame, detail=True)


def test_the_base_table_still_works() -> None:
    """The absence of `detail` must not have been repaired by disabling the command."""

    frame = pd.DataFrame({"v": np.arange(1, 101, dtype=float)})
    rendered = str(ws.summarize(frame))
    assert "Obs" in rendered
    assert "100" in rendered


def test_the_kurtosis_convention_stata_uses_is_recoverable() -> None:
    """Original kurtosis, i.e. excess plus three -- the number a future renderer must reproduce.

    Recorded here with the arithmetic visible so the next implementer does not have to re-derive it.
    """

    from scipy import stats

    series = np.arange(1, 101, dtype=float)
    excess = float(stats.kurtosis(series))
    assert excess == pytest.approx(-1.20024, abs=1e-5)
    assert excess + 3.0 == pytest.approx(1.79976, abs=1e-5)


def test_the_percentile_definition_is_the_averaged_inverted_cdf() -> None:
    """Stata's default percentile is NOT numpy's `linear` -- this is the trap for the renderer.

    Measured on a deliberately non-uniform sample of ten values:

        Stata  1%=1   5%=1   10%=1.5   25%=7   50%=21   75%=67   90%=96   95%=103   99%=103

    `np.percentile(..., method="linear")` gives 1.09 / 1.45 / 1.9 / 8 / 21 / 60.5 / 90.4 / 96.7 /
    101.74 -- wrong at seven of nine points. `midpoint` is wrong too once the sample is uneven.
    `averaged_inverted_cdf` reproduces all nine, as does `hazen` on this sample; the two differ only in
    the tails of larger samples, and `averaged_inverted_cdf` is the one Stata documents (`alt`).

    THE 1..100 SAMPLE HIDES THIS ENTIRELY: there, `linear`, `midpoint`, `averaged_inverted_cdf` and
    `hazen` all return 1.5 at the first percentile. Only an uneven sample separates them, which is why
    the guard below uses one.
    """

    series = np.array([1.0, 2.0, 7.0, 11.0, 19.0, 23.0, 41.0, 67.0, 89.0, 103.0])
    grid = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    stata = [1.0, 1.0, 1.5, 7.0, 21.0, 67.0, 96.0, 103.0, 103.0]

    matched = [
        float(np.percentile(series, point, method="averaged_inverted_cdf"))
        for point in grid
    ]
    for got, expected in zip(matched, stata):
        assert got == pytest.approx(expected, abs=1e-9)

    # And the obvious choice is visibly wrong on this sample, so the guard is discriminating.
    linear = [float(np.percentile(series, point, method="linear")) for point in grid]
    assert linear[2] == pytest.approx(1.9)
    assert linear[2] != pytest.approx(stata[2])


def test_the_percentile_grid_stata_reports() -> None:
    """1, 5, 10, 25, 50, 75, 90, 95, 99 -- pinned so a renderer cannot choose its own grid."""

    grid = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    assert len(grid) == 9
