"""``level(#)`` and ``clustertable`` -- areg's two reporting options that DO show.

Both were recorded as TODO in ``nq2011_gaps.md`` section 4.10. They are different from
``tolerance()``/``iterate()``/``nolog``: they change what a reader actually sees, so
"not implemented" was a real gap rather than a deliberate omission.

Reference values are from StataNow 19.5 (210 rows, 30 groups x 7 periods, seed 1):

    areg y x1, absorb(firm) level(90)
      _cons  [90% conf. interval] = -0.17476631   0.04851883

    areg y x1, absorb(a) vce(cluster a b) clustertable
      Cluster combination |  Clusters
                          a |        30
                          b |         7
                        a#b |       210
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _panel(n_units: int = 30, per_unit: int = 7, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_units * per_unit
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "firm": np.repeat(np.arange(n_units), per_unit),
            "a": np.repeat(np.arange(n_units), per_unit),
            "b": np.tile(np.arange(per_unit), n_units),
        }
    )


# ------------------------------------------------------------------ level()
def test_level_is_recorded_as_a_fraction() -> None:
    result = ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit(level=90)

    assert result.level == pytest.approx(0.90)


def test_level_changes_the_interval() -> None:
    """A 90% interval must be NARROWER than 95%; a no-op would keep them equal."""

    narrow = ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit(level=90)
    wide = ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit()

    narrow_width = np.subtract(*np.asarray(narrow.conf_int().to_numpy()[-1], dtype=float)[::-1])
    wide_width = np.subtract(*np.asarray(wide.conf_int().to_numpy()[-1], dtype=float)[::-1])

    assert narrow_width < wide_width


def test_level_matches_stata() -> None:
    """Stata's level(90) interval for _cons on this panel, to eight decimals."""

    result = ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit(level=90)
    lower, upper = np.asarray(result.conf_int().to_numpy()[-1], dtype=float)

    assert lower == pytest.approx(-0.17476631, abs=1e-6)
    assert upper == pytest.approx(0.04851883, abs=1e-6)


def test_level_accepts_a_fraction_too() -> None:
    """level(0.90) and level(90) must mean the same thing.

    Getting this wrong is quiet: 0.90 read as a percentage gives a 0.9% interval, whose
    symptom -- intervals far too narrow -- looks like a different bug entirely.
    """

    percent = ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit(level=90)
    fraction = ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit(level=0.90)

    np.testing.assert_allclose(
        np.asarray(percent.conf_int().to_numpy(), dtype=float),
        np.asarray(fraction.conf_int().to_numpy(), dtype=float),
        rtol=0.0, atol=0.0,
    )


def test_the_default_level_is_95_percent() -> None:
    result = ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit()

    assert result.level == pytest.approx(0.95)


def test_the_summary_header_names_the_level() -> None:
    """Stata labels the column "[90% conf. interval]", not always 95%."""

    result = ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit(level=90)

    assert "90% conf. interval" in result.summary(style="stata")


def test_level_applies_to_ols_too() -> None:
    """The option is not Areg-specific; it was missing from the whole package."""

    result = ws.OLS.from_formula("y ~ x1", _panel()).fit(level=90)

    assert result.level == pytest.approx(0.90)


def test_an_impossible_level_is_refused() -> None:
    with pytest.raises(ValueError, match="level"):
        ws.Areg.from_formula("y ~ x1", _panel(), absorb="firm").fit(level=150)


# ------------------------------------------------------------- clustertable
def test_clustertable_lists_the_combinations() -> None:
    frame = _panel()
    result = ws.Areg.from_formula("y ~ x1", frame, absorb="a").fit(
        cov_type="cluster", cov_kwds={"cluster": frame[["a", "b"]]}
    )
    rendered = result.summary(style="stata")

    assert "Cluster combination" in rendered
    assert "a#b" in rendered


def test_clustertable_reports_measured_counts() -> None:
    """The joint row is the number of DISTINCT (a, b) pairs, not 30 * 7.

    Here the panel is complete so the two coincide; the guard is that the numbers come
    from the labels rather than from multiplying, which the sparse case below checks.
    """

    frame = _panel()
    result = ws.Areg.from_formula("y ~ x1", frame, absorb="a").fit(
        cov_type="cluster", cov_kwds={"cluster": frame[["a", "b"]]}
    )

    assert result.e["_cluster_counts"] == [30, 7]
    assert result.e["_cluster_combination_counts"] == {"a#b": 210}


def test_clustertable_uses_the_callers_names() -> None:
    """Scratch column names (``__westata_cluster_0__``) must never reach the table."""

    frame = _panel()
    result = ws.Areg.from_formula("y ~ x1", frame, absorb="a").fit(
        cov_type="cluster", cov_kwds={"cluster": frame[["a", "b"]]}
    )

    assert result.e["_cluster_names"] == ["a", "b"]
    assert "__westata_cluster_" not in result.summary(style="stata")


def test_a_sparse_crossing_reports_fewer_joint_clusters() -> None:
    """THE reason the joint count is measured: two crossed dimensions with gaps.

    If the implementation multiplied 30 by 7 it would print 210 here, which is wrong and
    would hide exactly the sparsity this table exists to expose.
    """

    frame = _panel()
    # Keep only 4 of the 7 b-levels for a subset of groups, so (a, b) pairs are missing.
    keep = ~((frame["firm"] >= 20) & (frame["b"] >= 4))
    sparse = frame.loc[keep].reset_index(drop=True)

    result = ws.Areg.from_formula("y ~ x1", sparse, absorb="a").fit(
        cov_type="cluster", cov_kwds={"cluster": sparse[["a", "b"]]}
    )

    joint = result.e["_cluster_combination_counts"]["a#b"]
    assert joint < 210, f"the joint count looks multiplied rather than measured: {joint}"


def test_one_way_clustering_prints_no_table() -> None:
    """Stata prints ``clustertable`` only for more than one cluster variable."""

    frame = _panel()
    result = ws.Areg.from_formula("y ~ x1", frame, absorb="firm").fit(
        cov_type="cluster", cov_kwds={"cluster": frame["a"]}
    )

    assert "Cluster combination" not in result.summary(style="stata")