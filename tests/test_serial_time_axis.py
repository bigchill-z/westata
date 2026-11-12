"""The ``tsset`` axis, the 40-lag default, and the refusals Stata makes.

Three defects motivated this file, and each was found by measuring a live StataNow 19.5
rather than by reading the old code:

1. **Lags were taken by ROW POSITION.** ``bgodfrey.ado:59-60`` sorts by a time variable
   (``_ts timevar panelvar, sort onepanel``) and builds its lags with ``tsrevar l(...)``, so
   ``L.`` is undefined across a gap. Dropping ``time == 5`` from icecream.dta and lagging by
   row gives 3.286 where Stata reports 4.220 -- and the same defect ran through
   ``estat dwatson`` (1.0868969 vs .8930395) and ``wntestq``/``corrgram`` (4.3302 vs 3.9550),
   because all three shared one row-order assumption.
2. **The default lag count was ``(n - k)//2``.** Both ``ac.ado:71`` (which ``wntestq``
   inherits) and ``corrgram.ado:18`` say ``max(1, min(int(n/2) - 2, 40))``. The two agree
   only when ``k == 4``, which every fixture in this repository happened to have; at
   ``n = 150`` the old rule said 73 against Stata's 40.
3. **No cap at 40.** ``n = 758``: Stata reports ``chi2(40)``, the old rule 377.

The refusals matter for the same reason: Stata answers with an error, and a package that
quietly returns a number is worse than one that refuses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"

#: ``estat bgodfrey, lags(1/4)`` / ``nomiss0`` on icecream.dta with ``time == 5`` dropped,
#: re-estimated after the drop (n = 29). Captured 2026-11-12.
GAPPED_BG = [(4.220, 0.0400), (4.476, 0.1067), (4.787, 0.1881), (5.048, 0.2824)]
GAPPED_BG_NOMISS0 = [(5.264, 0.0218), (3.535, 0.1708), (3.814, 0.2823), (5.488, 0.2408)]
#: ``estat dwatson`` and ``wntestq e1, lags(4)`` on the same sample.
GAPPED_DWATSON = 0.8930395
GAPPED_Q4 = 3.9550
#: ``corrgram e1, lags(4)``: (AC, PAC, Q) per lag.
GAPPED_CORRGRAM = [
    (0.3322, 0.4143, 3.544),
    (0.0405, -0.0540, 3.5987),
    (0.0135, 0.1329, 3.605),
    (-0.0987, -0.0590, 3.955),
]


def _load(drop: bool = False) -> pd.DataFrame:
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    frame = frame.sort_values("time").reset_index(drop=True)
    return frame[frame.time != 5].reset_index(drop=True) if drop else frame


@pytest.fixture(scope="module")
def full_fit():
    return ws.OLS.from_formula(
        "consumption ~ temp + price + income", _load()
    ).fit()


@pytest.fixture(scope="module")
def gapped_fit():
    return ws.OLS.from_formula(
        "consumption ~ temp + price + income", _load(drop=True)
    ).fit()


# --------------------------------------------------------------------- the time axis


def test_bgodfrey_with_a_gap_matches_stata(gapped_fit) -> None:
    table = ws.estat_bgodfrey(gapped_fit, lags=[1, 2, 3, 4], time="time")
    assert [row.lags for row in table.rows] == [1, 2, 3, 4]
    for row, (chi2, pvalue) in zip(table.rows, GAPPED_BG):
        assert row.chi2 == pytest.approx(chi2, abs=5e-4), f"lag {row.lags}"
        assert row.pvalue == pytest.approx(pvalue, abs=5e-5), f"lag {row.lags}"


def test_bgodfrey_nomiss0_with_a_gap_matches_stata(gapped_fit) -> None:
    table = ws.estat_bgodfrey(
        gapped_fit, lags=[1, 2, 3, 4], time="time", nomiss0=True
    )
    for row, (chi2, pvalue) in zip(table.rows, GAPPED_BG_NOMISS0):
        assert row.chi2 == pytest.approx(chi2, abs=5e-4), f"lag {row.lags}"
        assert row.pvalue == pytest.approx(pvalue, abs=5e-5), f"lag {row.lags}"


def test_the_gap_changes_the_answer(gapped_fit) -> None:
    """With or without ``time=`` the two paths must DISAGREE, or the flag proves nothing."""

    by_row = [row.chi2 for row in ws.estat_bgodfrey(gapped_fit, lags=[1, 2]).rows]
    by_time = [
        row.chi2 for row in ws.estat_bgodfrey(gapped_fit, lags=[1, 2], time="time").rows
    ]
    assert by_row == pytest.approx([3.286, 3.363], abs=5e-4)
    assert by_time == pytest.approx([4.220, 4.476], abs=5e-4)


def test_the_gap_note_is_printed_first(gapped_fit) -> None:
    """``bgodfrey.ado:44`` prints the count BEFORE the blank line opening the table."""

    rendered = str(ws.estat_bgodfrey(gapped_fit, lags=1, time="time"))
    assert rendered.splitlines()[0] == "Number of gaps in sample = 1"
    assert rendered.splitlines()[1] == ""


def test_no_note_without_a_gap(full_fit) -> None:
    rendered = str(ws.estat_bgodfrey(full_fit, lags=1, time="time"))
    assert rendered.startswith("\nBreusch")


def test_two_gaps_are_counted(full_fit) -> None:
    frame = _load()
    trimmed = frame[~frame.time.isin([5, 12])].reset_index(drop=True)
    fit = ws.OLS.from_formula("consumption ~ temp + price + income", trimmed).fit()
    assert ws.estat_bgodfrey(fit, lags=1, time="time").n_gaps == 2
    assert "Number of gaps in sample = 2" in str(ws.estat_dwatson(fit, time="time"))


def test_dwatson_with_a_gap_matches_stata(gapped_fit) -> None:
    result = ws.estat_dwatson(gapped_fit, time="time")
    assert result.d == pytest.approx(GAPPED_DWATSON, abs=5e-8)
    assert result.n_gaps == 1


def test_dwatson_by_row_is_the_old_wrong_number(gapped_fit) -> None:
    """The pair straddling a gap must be skipped, not differenced."""

    assert ws.estat_dwatson(gapped_fit).d == pytest.approx(1.0868969, abs=5e-8)


def test_wntestq_with_a_gap_matches_stata(gapped_fit) -> None:
    assert ws.wntestq(gapped_fit, lags=4, time="time").q == pytest.approx(GAPPED_Q4, abs=5e-5)
    assert str(ws.wntestq(gapped_fit, lags=4, time="time")).startswith("(note: time series has 1 gap)")


def test_corrgram_with_a_gap_matches_stata(gapped_fit) -> None:
    grid = ws.estat_corrgram(gapped_fit, lags=4, time="time")
    for row, (ac, pac, q) in zip(grid.rows, GAPPED_CORRGRAM):
        assert row.ac == pytest.approx(ac, abs=5e-5), f"AC at lag {row.lag}"
        assert row.pac == pytest.approx(pac, abs=5e-5), f"PAC at lag {row.lag}"
        assert row.q == pytest.approx(q, abs=5e-5), f"Q at lag {row.lag}"


def test_corrgram_note_wording_is_singular_then_plural(full_fit) -> None:
    frame = _load()
    one = frame[frame.time != 5].reset_index(drop=True)
    two = frame[~frame.time.isin([5, 12])].reset_index(drop=True)
    fit_one = ws.OLS.from_formula("consumption ~ temp + price + income", one).fit()
    fit_two = ws.OLS.from_formula("consumption ~ temp + price + income", two).fit()
    assert str(ws.estat_corrgram(fit_one, lags=2, time="time")).startswith(
        "(note: time series has 1 gap)"
    )
    assert str(ws.estat_corrgram(fit_two, lags=2, time="time")).startswith(
        "(note: time series has 2 gaps)"
    )


def test_the_contiguous_sample_is_unchanged_by_the_flag(full_fit) -> None:
    """``time=`` on gapless data must not move a number -- otherwise it is a rewrite."""

    plain = ws.estat_bgodfrey(full_fit, lags=[1, 2, 3, 4])
    axis = ws.estat_bgodfrey(full_fit, lags=[1, 2, 3, 4], time="time")
    assert [row.chi2 for row in plain.rows] == [row.chi2 for row in axis.rows]
    assert ws.estat_dwatson(full_fit).d == ws.estat_dwatson(full_fit, time="time").d


def test_time_accepts_the_values_themselves(gapped_fit) -> None:
    values = _load(drop=True)["time"].to_numpy()
    by_name = ws.estat_bgodfrey(gapped_fit, lags=2, time="time")
    by_values = ws.estat_bgodfrey(gapped_fit, lags=2, time=values)
    assert [row.chi2 for row in by_name.rows] == [row.chi2 for row in by_values.rows]


def test_a_delta_of_two_steps_two_periods(full_fit) -> None:
    """``tsset t, delta(2)`` on t = 1,3,5,... pairs t with t-2, not t-1."""

    t = np.arange(1, 31, dtype=float)  # 1..30
    stepped = np.arange(1, 61, 2, dtype=float)[:30]  # 1,3,5,...,59
    one = ws.estat_dwatson(full_fit, time=t, delta=1.0).d
    two = ws.estat_dwatson(full_fit, time=stepped, delta=2.0).d
    # Same ROWS, same neighbour relation -> the same statistic through either spelling.
    assert one == pytest.approx(two, abs=1e-12)


def test_a_fractional_delta_is_refused(full_fit) -> None:
    with pytest.raises(ValueError):
        ws.estat_dwatson(full_fit, time="time", delta=1.5)


def test_repeated_periods_are_refused_as_a_panel(full_fit) -> None:
    repeated = np.tile(np.arange(1, 16, dtype=float), 2)
    with pytest.raises(Exception, match="multiple panels"):
        ws.estat_bgodfrey(full_fit, time=repeated)


def test_a_time_name_that_is_not_a_column_is_named_in_the_error(full_fit) -> None:
    with pytest.raises(Exception, match="not a column"):
        ws.estat_bgodfrey(full_fit, time="nosuchcolumn")


# ------------------------------------------------------------ the default lag order


@pytest.mark.parametrize(
    "nobs, expected",
    [(10, 3), (20, 8), (30, 13), (40, 18), (50, 23), (80, 38), (100, 40), (758, 40)],
)
def test_the_default_lag_count_follows_n_over_two_minus_two_capped(
    nobs: int, expected: int
) -> None:
    """``max(1, min(int(n/2) - 2, 40))``, measured at every one of these sizes."""

    rng = np.random.default_rng(nobs)
    frame = pd.DataFrame({"y": rng.normal(size=nobs)})
    fit = ws.OLS.from_formula("y ~ 1", frame).fit()
    assert ws.wntestq(fit).df == expected
    assert ws.estat_corrgram(fit).rows[-1].lag == expected


def test_the_default_does_not_depend_on_the_parameter_count() -> None:
    """``reg u x1`` (2 parameters) and ``reg u x1..x6`` (7) both default to 13 at n = 30."""

    rng = np.random.default_rng(5)
    frame = pd.DataFrame({f"x{index}": rng.normal(size=30) for index in range(1, 7)})
    frame["u"] = rng.normal(size=30)
    one = ws.OLS.from_formula("u ~ x1", frame).fit()
    six = ws.OLS.from_formula("u ~ x1 + x2 + x3 + x4 + x5 + x6", frame).fit()
    assert ws.wntestq(one).df == ws.wntestq(six).df == 13


def test_the_small_n_floor_is_one() -> None:
    """``n = 5`` gives ``int(5/2) - 2 = 0``, and Stata reports ``chi2(1)``."""

    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"y": rng.normal(size=5)})
    fit = ws.OLS.from_formula("y ~ 1", frame).fit()
    assert ws.wntestq(fit).df == 1


# -------------------------------------------------------------------------- refusals


def test_a_clustered_vce_is_refused(full_fit) -> None:
    """``vce(cluster)`` sets ``e(vcetype) = Robust``, which ``bgodfrey.ado:5-10`` rejects."""

    frame = _load()
    robust = ws.OLS.from_formula("consumption ~ temp + price + income", frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame["time"] % 3}
    )
    with pytest.raises(Exception, match="does not work after regress, robust"):
        ws.estat_bgodfrey(robust)


def test_a_panel_fit_is_refused() -> None:
    """Stata: ``sample may not include multiple panels`` (r(459))."""

    rng = np.random.default_rng(4)
    rows = [
        {"id": unit, "t": period, "x": rng.normal(), "y": rng.normal()}
        for unit in range(4)
        for period in range(1, 11)
    ]
    frame = pd.DataFrame(rows)
    fit = ws.PanelOLS.from_formula("y ~ x", frame, entity="id", time="t").fit()
    for command in (ws.estat_bgodfrey, ws.estat_dwatson, ws.estat_corrgram, ws.wntestq):
        with pytest.raises(Exception, match="multiple panels"):
            command(fit)


def test_an_absorbed_fit_is_not_a_regress_fit(full_fit) -> None:
    """``areg`` answers ``estat bgodfrey not valid`` (r(321)); ``reghdfe`` likewise."""

    frame = _load()
    frame["unit"] = np.arange(len(frame)) % 5
    absorbed = ws.FEOLS.from_formula("consumption ~ temp + price | unit", frame).fit()
    with pytest.raises(Exception, match="not valid"):
        ws.estat_bgodfrey(absorbed)


# --------------------------------------------------------------------------- wiring


def test_every_serial_command_is_reachable_from_the_result(full_fit) -> None:
    """``estat dwatson`` got this wiring in round 93; the other three were left behind."""

    assert full_fit.estat_bgodfrey(lags=1).rows[0].chi2 == pytest.approx(4.237, abs=5e-4)
    assert full_fit.estat_dwatson().d == pytest.approx(1.021169, abs=1e-6)
    assert full_fit.estat_corrgram(lags=2).rows[-1].lag == 2
    assert full_fit.wntestq(lags=3).df == 3


def test_wntestq_accepts_a_bare_series() -> None:
    """``wntestq e1`` takes a VARIABLE in Stata, so a raw array must work here."""

    values = np.random.default_rng(6).normal(size=40)
    by_values = ws.wntestq(values)
    frame = pd.DataFrame({"y": values})
    by_result = ws.wntestq(ws.OLS.from_formula("y ~ 1", frame).fit())
    # The fit's residuals are de-meaned, which barely moves Q here; the point of the test
    # is that the raw-array path exists at all rather than raising ConsistencyError.
    assert by_values.df == by_result.df
    assert by_values.q == pytest.approx(by_result.q, rel=1e-2)
