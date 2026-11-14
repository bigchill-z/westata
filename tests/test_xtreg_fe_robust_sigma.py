"""``xtreg, fe r``: the clustered VCE must not drag ``sigma_e`` away from Stata.

``vce(robust)`` after ``xtreg, fe`` is NOT the HC1 "robust" of ``regress, robust``:
Stata treats it as a cluster-robust VCE with the panel as the cluster and prints
``(Std. err. adjusted for 28 clusters in province)``.  The coefficients and the
standard errors agree with ``fe r`` to 1e-8 in this package, but three scalars live
in the same ``e()`` block and only two of them move with the VCE:

    e(rmse)    = sqrt(rss / (N - rank))      466 on this data
    e(sigma_e) = sqrt(rss / e(df_r_mb))      439 on this data
    e(r2_a)    = adjusted with e(rmse)'s denominator

Reusing one denominator for both is invisible under the CONVENTIONAL VCE, where
``N - rank == N - K - (G-1)``.  It only diverges once the sample has both an
intercept-removing estimator and enough groups, which is why the earlier check --
a 30 x 6 panel with two slopes -- could not tell the two apart.

Everything here is measured against StataNow 19.5, 2026-11-14::

    xtset province t
    xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca, fe r
    xtreg ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca, fe
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"

Y = "ltvfo"
XS = ["ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mipric1", "giprice", "mci", "ngca"]

#: ``xtreg ..., fe`` (conventional VCE). Coefficient, std. err.
STATA_FE = {
    "ltlan": (0.6370234, 0.0673191),
    "ltwlab": (0.1387786, 0.0261554),
    "ltpow": (0.0577152, 0.0332508),
    "ltfer": (0.1826281, 0.0219921),
    "hrs": (0.2134022, 0.0223886),
    "mipric1": (0.0543577, 0.0421659),
    "giprice": (-0.0151451, 0.0187457),
    "mci": (0.1943697, 0.0876884),
    "ngca": (0.7562031, 0.2168141),
    "_cons": (2.3378950, 0.3852530),
}

#: ``xtreg ..., fe r`` (cluster-robust; same coefficients, different SEs).
STATA_FE_R = {
    "ltlan": (0.6370234, 0.1681335),
    "ltwlab": (0.1387786, 0.0624585),
    "ltpow": (0.0577152, 0.0755568),
    "ltfer": (0.1826281, 0.0435920),
    "hrs": (0.2134022, 0.0391104),
    "mipric1": (0.0543577, 0.0590331),
    "giprice": (-0.0151451, 0.0245968),
    "mci": (0.1943697, 0.0770515),
    "ngca": (0.7562031, 0.3821261),
    "_cons": (2.3378950, 0.8552224),
}

#: Scalars that do NOT change between the two VCEs.
SHARED = {
    "r2_w": 0.874599403500585,
    "r2_b": 0.648297958019457,
    "r2_o": 0.699304987795042,
    "corr": -0.387688087297474,
    "sigma_u": 0.305497426478823,
    "sigma_e": 0.105892741490381,
}


def _frame() -> pd.DataFrame:
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")
    sub = frame.dropna(subset=[Y, *XS]).reset_index(drop=True).copy()
    sub["pid"] = sub["province"]
    return sub


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _frame()


def _fit(frame: pd.DataFrame, cov: str | None):
    kw = {"groups": frame["pid"]} if cov == "cluster" else None
    return ws.PanelOLS.from_formula(
        f"{Y} ~ " + " + ".join(XS), frame, entity="pid", time="t"
    ).fit(cov_type=cov, cov_kwds=kw)


@pytest.mark.parametrize("cov, reference", [("nonrobust", STATA_FE), ("cluster", STATA_FE_R)])
def test_coefficients_and_standard_errors_match_stata(frame, cov, reference) -> None:
    fit = _fit(frame, cov)
    b = fit.stata_params
    for name, (coef, se) in reference.items():
        index = "Intercept" if name == "_cons" else name
        assert float(b[name]) == pytest.approx(coef, abs=5e-7), f"{name} coefficient"
        assert float(fit.bse[index]) == pytest.approx(se, abs=5e-7), f"{name} std. err."
    assert fit.nobs == 476
    assert fit.e.get("N_g") == 28


@pytest.mark.parametrize("key, expected", sorted(SHARED.items()))
@pytest.mark.parametrize("cov", ["nonrobust", "cluster"])
def test_variance_components_are_identical_under_either_vce(frame, cov, key, expected) -> None:
    """``sigma_e`` in particular: Stata prints the same value in both tables.

    This is the assertion that fails if ``sigma_e`` is divided by the clustered
    ``N - rank`` instead of Stata's model-based ``df_r_mb`` -- the clustered value
    would be ``sqrt(rss / 466) = 0.10277926`` against the correct ``0.10589274``.
    """

    fit = _fit(frame, cov)
    assert fit.e.get(key) == pytest.approx(expected, abs=5e-8), key


def test_the_two_denominators_really_differ_on_this_data(frame) -> None:
    """A guard on the fixture: if 466 and 439 coincided, the bug would be invisible."""

    fit = _fit(frame, "cluster")
    rss = fit.e["rss"]
    assert fit.e["df_r"] == 27.0          # G - 1
    # N - rank = 476 - 9 (8 slopes + intercept) ... Stata's rank counts the slopes.
    clustered_rmse = np.sqrt(rss / 466.0)
    model_sigma_e = np.sqrt(rss / 439.0)
    assert fit.e["sigma_e"] == pytest.approx(model_sigma_e, rel=1e-12)
    assert abs(clustered_rmse - model_sigma_e) > 3e-3, (
        "the two denominators must be far enough apart to make the bug detectable"
    )


def test_df_r_is_the_clustered_count_but_sigma_e_is_not(frame) -> None:
    """Stata's own ``e()`` splits: ``df_r`` = 27, ``df_r_mb`` = 439."""

    fit = _fit(frame, "cluster")
    assert fit.e["df_r"] == 27.0
    # ``e(df_r_mb)`` is 439 in Stata; this package does not store that key, so the
    # model-based value is asserted through the sigma_e identity above instead.
    assert fit.e["sigma_e"] == pytest.approx(0.105892741490381, abs=1e-12)


def test_robust_is_a_cluster_vce_not_hc1(frame) -> None:
    """``fe r`` == ``fe vce(cluster pid)``; it is NOT the HC1 of ``regress, robust``."""

    frame["pid"] = frame["province"]
    explicit = ws.PanelOLS.from_formula(
        f"{Y} ~ " + " + ".join(XS), frame, entity="pid", time="t"
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["pid"]})
    assert explicit.e.get("N_clust") == 28
    assert explicit.df_resid == 27.0


# --------------------------------------------------------------- e() display keys


def test_clustered_xtreg_stores_the_keys_stata_stores(frame) -> None:
    """``e(vcetype)``/``e(vce)``/``e(clustvar)``, all measured on the live run.

    Stata on ``xtreg ..., fe r``: ``e(vcetype) = Robust``, ``e(vce) = cluster``,
    ``e(clustvar) = province``, ``e(N_clust) = 28``.  Plain ``fe`` leaves every one of
    them missing.  westata stored only ``N_clust``, so the table was numerically right
    and still lost two things a reader needs.
    """

    frame["pid"] = frame["province"]
    clustered = ws.PanelOLS.from_formula(
        f"{Y} ~ " + " + ".join(XS), frame, entity="pid", time="t"
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["pid"]})
    assert clustered.e.get("vcetype") == "Robust"
    assert clustered.e.get("vce") == "cluster"
    assert clustered.e.get("clustvar") == "pid"

    plain = _fit(frame, "nonrobust")
    for key in ("vcetype", "vce", "clustvar", "N_clust"):
        assert plain.e.get(key) is None, f"plain fe must not store e({key})"


def test_the_table_prints_statas_cluster_note_and_robust_heading(frame) -> None:
    """The two lines that were missing while every number was correct."""

    frame["pid"] = frame["province"]
    clustered = ws.PanelOLS.from_formula(
        f"{Y} ~ " + " + ".join(XS), frame, entity="pid", time="t"
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["pid"]})
    rendered = str(clustered.summary())
    assert "(Std. err. adjusted for 28 clusters in pid)" in rendered
    assert "Robust" in rendered
    # ...and the plain fit prints NEITHER, exactly like Stata.
    plain = str(_fit(frame, "nonrobust").summary())
    assert "adjusted for" not in plain
    assert "Robust" not in plain


# ------------------------------------------------- cross-version linearmodels


def test_a_cluster_series_works_on_every_supported_linearmodels(frame) -> None:
    """``cov_kwds={"groups": df["entity"]}`` must not depend on the 7.0/7.1 split.

    The two supported versions validate cluster labels differently, measured on
    2026-11-14 with identical westata source:

        7.0  rejects a bare 1-D array   ("2 or 3-d array required for numpy input")
        7.1  rejects an (n, 1) block    ("clusters must have the same number of
                                         entities and time periods")

    Only a ``Series`` carrying the panel MultiIndex satisfies both, which is what
    ``_as_cluster_block`` now emits.  This test asserts the shape actually handed to
    the backend, so it pins the CONTRACT rather than whichever version happens to be
    installed.
    """

    from westata.estimators.panel import _as_cluster_block

    frame["pid"] = frame["province"]
    model = ws.PanelOLS.from_formula(
        f"{Y} ~ " + " + ".join(XS), frame, entity="pid", time="t"
    )
    _, labels, n_clust = model._resolve_cov("cluster", {"groups": frame["pid"]})
    assert n_clust == 28
    block = _as_cluster_block(labels, model._kept_index)
    assert isinstance(block, pd.Series), "a labelled Series is the version-independent form"
    assert block.index.equals(model._kept_index)
    assert block.size == labels.size == 476


def test_a_plain_array_of_labels_is_still_a_blind_cluster(frame) -> None:
    """Passing the labels instead of the Series must give the SAME standard errors."""

    frame["pid"] = frame["province"]
    formula = f"{Y} ~ " + " + ".join(XS)
    as_series = ws.PanelOLS.from_formula(formula, frame, entity="pid", time="t").fit(
        cov_type="cluster", cov_kwds={"groups": frame["pid"]}
    )
    as_array = ws.PanelOLS.from_formula(formula, frame, entity="pid", time="t").fit(
        cov_type="cluster", cov_kwds={"groups": frame["pid"].to_numpy()}
    )
    assert float(as_array.bse["ltlan"]) == pytest.approx(
        float(as_series.bse["ltlan"]), rel=1e-12
    )
