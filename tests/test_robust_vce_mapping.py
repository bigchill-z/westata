"""Stata's `robust` maps to DIFFERENT small-sample factors per command family.

This is the trap the API documentation now leads with, and it is worth a test because the two
flavours produce standard errors that differ only in the third digit -- plausible either way.

MEASURED on grilic.dta, standard error of `s`:

    reg lnw s expr, vce(robust)                          0.00638056877997
        westata cov_type="HC1"                           0.00638056877997   MATCH
        westata cov_type="HC0"                           0.00636792980662

    ivregress 2sls lnw expr (s = rns mrt), vce(robust)   0.04061066342746
        westata cov_type="HC0"                           0.04061066342742   MATCH
        westata cov_type="HC1"                           0.04069126687443

`regress` applies n/(n-k); `ivregress` does not unless `small` is given. So the same word picks
OPPOSITE flavours, and the package reproduces each one under its own name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

GIRLIC = r"D:\myCodeProject\test_westata\data\grilic.dta"

#: Captured from a live Stata at %22.14f.
STATA_REGRESS_ROBUST_SE_S = 0.00638056877997
STATA_IV_ROBUST_SE_S = 0.04061066342746


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(GIRLIC).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "mrt"]
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")


def test_regress_robust_is_hc1(frame) -> None:
    """`reg vce(robust)` applies n/(n-k), which is HC1."""

    fitted = ws.OLS.from_formula("lnw ~ s + expr", frame).fit(cov_type="HC1")
    assert float(fitted.bse["s"]) == pytest.approx(STATA_REGRESS_ROBUST_SE_S, abs=1e-12)


def test_regress_hc0_does_not_match_the_stata_robust(frame) -> None:
    """The wrong flavour is close enough to look right -- 0.2% off, in the third digit."""

    fitted = ws.OLS.from_formula("lnw ~ s + expr", frame).fit(cov_type="HC0")
    wrong = float(fitted.bse["s"])
    assert wrong != pytest.approx(STATA_REGRESS_ROBUST_SE_S, abs=1e-9)
    # ... but within 1%, which is exactly why the mix-up goes unnoticed.
    assert abs(wrong - STATA_REGRESS_ROBUST_SE_S) / STATA_REGRESS_ROBUST_SE_S < 0.01


def test_iv_robust_is_hc0(frame) -> None:
    """`ivregress vce(robust)` does NOT apply the factor, so it is HC0."""

    fitted = ws.IV2SLS.from_formula(
        "lnw ~ expr + [s ~ rns + mrt]", frame
    ).fit(cov_type="HC0")
    assert float(fitted.bse["s"]) == pytest.approx(STATA_IV_ROBUST_SE_S, abs=1e-12)


def test_iv_hc1_does_not_match_the_stata_robust(frame) -> None:
    fitted = ws.IV2SLS.from_formula(
        "lnw ~ expr + [s ~ rns + mrt]", frame
    ).fit(cov_type="HC1")
    wrong = float(fitted.bse["s"])
    assert wrong != pytest.approx(STATA_IV_ROBUST_SE_S, abs=1e-9)
    assert abs(wrong - STATA_IV_ROBUST_SE_S) / STATA_IV_ROBUST_SE_S < 0.01


def test_the_two_families_pick_opposite_flavours(frame) -> None:
    """The whole point: `cov_type="robust"` resolves differently per family."""

    ols = ws.OLS.from_formula("lnw ~ s + expr", frame).fit(cov_type="robust")
    iv = ws.IV2SLS.from_formula("lnw ~ expr + [s ~ rns + mrt]", frame).fit(
        cov_type="robust"
    )
    assert ols.cov_type == "HC1"
    assert iv.cov_type == "HC0"


def test_the_canonical_name_is_stored_not_the_alias(frame) -> None:
    """Echoing "robust" back would reproduce the confusion the docs exist to prevent."""

    fitted = ws.OLS.from_formula("lnw ~ s + expr", frame).fit(cov_type="robust")
    assert fitted.cov_type != "robust"


def test_hc2_and_hc3_are_refused_for_iv(frame) -> None:
    """Stata has no leverage-corrected robust VCE for ivregress; labelling HC0 as HC2 lies."""

    for cov in ("HC2", "HC3"):
        with pytest.raises(Exception):
            ws.IV2SLS.from_formula(
                "lnw ~ expr + [s ~ rns + mrt]", frame
            ).fit(cov_type=cov)


def test_the_documentation_states_the_mapping() -> None:
    """Both language files must carry the table; a reader should not have to infer it."""

    import io

    for path in ("docs/API.md", "docs/API.zh-CN.md"):
        text = io.open(path, encoding="utf-8").read()
        assert "HC1" in text and "HC0" in text
        # The regress/HC1 and ivregress/HC0 pairing is the fact being documented.
        assert "regress" in text and "ivregress" in text


def test_panel_robust_equals_clustering_on_the_entity() -> None:
    """A THIRD pattern: for `xtreg ..., fe`, Stata's vce(robust) and vce(cluster id) coincide.

    Measured on a 40x10 balanced panel:

        xtreg y x1, fe                      0.05440432377516
        xtreg y x1, fe vce(robust)          0.05229223550829
        xtreg y x1, fe vce(cluster sid)     0.05229223550829

    The fixed effects are differenced out, so the robust VCE is clustered by construction.
    Reading a panel result's cov_type="robust" as "HC1" would be wrong: here it names the same
    estimator as clustering, not a separate small-sample correction.
    """

    rng = np.random.default_rng(3)
    n_g, n_t = 40, 10
    sid = np.repeat(np.arange(n_g), n_t)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n_g * n_t),
            "x1": rng.normal(size=n_g * n_t),
            "sid": sid,
            "t": np.tile(np.arange(n_t), n_g),
        }
    )

    classical = ws.PanelOLS.from_formula("y ~ x1", frame, entity="sid", time="t").fit()
    robust = ws.PanelOLS.from_formula("y ~ x1", frame, entity="sid", time="t").fit(
        cov_type="robust"
    )
    clustered = ws.PanelOLS.from_formula("y ~ x1", frame, entity="sid", time="t").fit(
        cov_type="cluster", cov_kwds={"groups": frame["sid"]}
    )

    assert float(classical.bse["x1"]) == pytest.approx(0.05440432377516, abs=1e-12)
    assert float(robust.bse["x1"]) == pytest.approx(0.05229223550829, abs=1e-12)
    # The whole point: robust and cluster are the SAME number for an fe fit.
    assert float(robust.bse["x1"]) == pytest.approx(
        float(clustered.bse["x1"]), abs=1e-12
    )


def test_the_docs_carry_the_panel_note() -> None:
    import io as _io

    for path in ("docs/API.md", "docs/API.zh-CN.md"):
        text = _io.open(path, encoding="utf-8").read()
        assert "vce(cluster" in text
        assert "0.05229223550829" in text
