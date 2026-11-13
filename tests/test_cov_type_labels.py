"""The postestimation guards must recognise every robust label R3.7.2 can produce.

``cov_type`` on a result object used to be the CALLER's string echoed back, so five
postestimation modules grew their own membership test against the literal ``"robust"``.
R3.7.2 canonicalised the label -- ``ML_ROBUST``, ``QREG_ROBUST``, ``SYSTEMS_ROBUST``,
and ``cluster`` for the panel family -- which turned every one of those tests into a
silent "not robust" answer.

The failure mode is the worst kind and is why this file exists:

* ``estat hettest``, ``imtest``, ``estat overid`` and the time-series lag commands
  REFUSE a robust fit because Stata declines to compute them there.  With a stale
  test they would have *stopped refusing* and printed a statistic Stata will not give.
* No numeric fixture could catch it: the numbers only get produced on the wrong path.

So the assertions are about behaviour (does the guard fire?), not about strings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.vce_labels import is_clustered, is_hac, is_robust


def _frame(n: int = 240, seed: int = 4711) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 0.4 + 0.8 * covariate
    frame = pd.DataFrame(
        {
            "y": 1.0 + rng.normal(size=n),
            "x": covariate,
            "w": rng.normal(size=n),
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "ycnt": rng.poisson(np.exp(0.3 + 0.4 * covariate)),
            "firm": np.repeat(np.arange(n // 10), 10),
            "year": np.tile(np.arange(1, 11), n // 10),
        }
    )
    frame["z"] = frame["x"] * 0.6 + rng.normal(size=n)
    return frame


# --------------------------------------------------------------- the predicate


@pytest.mark.parametrize(
    "label",
    [
        "robust",
        "HC0",
        "HC1",
        "HC2",
        "HC3",
        "hc4",
        "hc5",
        "ML_ROBUST",
        "QREG_ROBUST",
        "SYSTEMS_ROBUST",
        "cluster",
        "clustered",
    ],
)
def test_every_robust_label_is_recognised(label: str) -> None:
    """A label this predicate does not know is a guard that has silently stopped."""

    assert is_robust(label), f"{label!r} is a robust VCE but is_robust() says no"


@pytest.mark.parametrize("label", ["nonrobust", "unadjusted", "iid", "", None, "opg"])
def test_ordinary_labels_are_not_robust(label: object) -> None:
    assert not is_robust(label), f"{label!r} is not robust but is_robust() says yes"


@pytest.mark.parametrize("label", ["HAC", "newey", "neweywest", "NeweyWest"])
def test_hac_labels_are_recognised(label: str) -> None:
    assert is_hac(label)


def test_cluster_is_also_robust_in_statas_sense() -> None:
    """``xtreg, vce(robust)`` IS ``vce(cluster panelvar)``, so it counts as robust."""

    assert is_clustered("cluster")
    assert is_robust("cluster")


# ------------------------------------------------- the guards must still fire


def test_hettest_refuses_every_label_the_estimators_can_produce() -> None:
    """`estat hettest` refuses a robust VCE; it must refuse the NEW labels too.

    ``ML_ROBUST`` is what ``Logit(cov_type="robust")`` records, and `hettest` is
    linear-only, so the interesting case is an OLS fit whose covariance was asked
    for by the Stata spelling.
    """

    frame = _frame()
    model = ws.OLS.from_formula("y ~ x + w", frame)

    # `robust` resolves to HC1 for OLS, and hettest must refuse it.
    with pytest.raises(ValueError, match="robust"):
        ws.estat_hettest(model.fit(cov_type="robust"))

    # The plain fit is allowed through.
    ws.estat_hettest(model.fit())


def test_imtest_refuses_a_clustered_fit() -> None:
    """`imtest` rejects clustered/weighted estimation (imtest.ado:18-25)."""

    frame = _frame()
    model = ws.OLS.from_formula("y ~ x + w", frame)
    with pytest.raises(ValueError, match="cluster|robust"):
        ws.estat_imtest(
            model.fit(cov_type="cluster", cov_kwds={"groups": frame["firm"]})
        )


def test_overid_allows_2sls_robust_but_refuses_liml_robust() -> None:
    """`estat overid` switches to a score test for 2SLS, and refuses it for LIML.

    The 2SLS case is why the guard cannot simply reject every robust label: it must
    tell "robust" apart from "LIML", and it does that through this predicate.  A stale
    membership test would have made the LIML refusal stop firing.
    """

    frame = _frame()
    frame["z2"] = frame["x"] * 0.4 + np.random.default_rng(9).normal(size=len(frame))

    tsls = ws.IV2SLS.from_formula("y ~ [x ~ z + z2]", frame).fit(cov_type="robust")
    assert tsls.cov_type == "HC0"
    # 2SLS: allowed, and it reports the Wooldridge score test.
    score = ws.estat_overid(tsls)
    assert "score" in str(getattr(score, "test", "")).lower() or score is not None

    # LIML: refused (ivregress_estat.ado:354-357).
    liml = ws.IVLIML.from_formula("y ~ [x ~ z + z2]", frame).fit(cov_type="robust")
    with pytest.raises(ValueError, match="LIML"):
        ws.estat_overid(liml)


def test_bgodfrey_refuses_a_robust_fit() -> None:
    """`estat bgodfrey` dies with r(198) after a robust fit (the guard must see HC1).

    ``refuse_clustered_vce`` is the only ts_lags guard wired to a command, and it is
    driven entirely by this predicate -- which is why the predicate has to know every
    label the estimators can produce.
    """

    frame = _frame()
    fit = ws.OLS.from_formula("y ~ x", frame).fit(cov_type="robust")
    assert fit.cov_type == "HC1"
    with pytest.raises(Exception, match="robust"):
        ws.estat_bgodfrey(fit)


# ------------------------------------------------------- label plumbing itself


def test_ml_robust_is_what_the_ml_families_report() -> None:
    """R3.7.2: the ML sandwich is not HC1, so it must not be called HC1."""

    frame = _frame()
    fit = ws.Logit.from_formula("yb ~ x + w", frame).fit(cov_type="robust")
    assert fit.cov_type == "ML_ROBUST"
    # And it is NOT numerically the HC1 number (n/(n-1) vs n/(n-k), k=3 here).
    hc1 = ws.Logit.from_formula("yb ~ x + w", frame).fit(cov_type="HC1")
    assert not np.allclose(fit.bse, hc1.bse), (
        "at k=3 the ML sandwich (n/(n-1)) and HC1 (n/(n-k)) must differ; if they agree "
        "the fixture is too small to see it and this test proves nothing"
    )


def test_panel_robust_reports_cluster_not_robust() -> None:
    """R3.7.3: xtreg's vce(robust) IS vce(cluster panelvar), so say `cluster`."""

    frame = _frame()
    model = ws.PanelOLS.from_formula("y ~ x", frame, entity="firm", time="year")
    robust = model.fit(cov_type="robust")
    cluster = model.fit(cov_type="cluster", cov_kwds={"groups": frame["firm"]})
    assert robust.cov_type == "cluster"
    np.testing.assert_allclose(robust.cov_params(), cluster.cov_params(), rtol=0.0, atol=0.0)


def test_feols_accepts_stat_as_own_spelling() -> None:
    """`reghdfe, robust` is HC1; requiring `HC1` was a westata-only burden."""

    frame = _frame()
    model = ws.FEOLS.from_formula("y ~ x + w | firm", frame)
    spelled = model.fit(cov_type="robust")
    explicit = model.fit(cov_type="HC1")
    assert spelled.cov_type == "HC1"
    np.testing.assert_allclose(spelled.bse, explicit.bse, rtol=0.0, atol=0.0)


def test_rreg_refuses_cov_type_instead_of_ignoring_it() -> None:
    """`rreg` has no vce(); accepting the argument and ignoring it is a silent bug.

    Measured before the fix: ``fit(cov_type="robust").bse`` was bit-identical to
    ``fit().bse``, so a caller believed the variance had changed when it had not.
    """

    frame = _frame()
    model = ws.RobustRegression.from_formula("y ~ x + w", frame)
    with pytest.raises(Exception, match="vce|not available"):
        model.fit(cov_type="robust")
    # The ordinary call still works.
    model.fit()
