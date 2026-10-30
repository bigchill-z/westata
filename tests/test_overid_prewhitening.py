"""Tests for the HAC prewhitening step of the Wooldridge score test.

**The end-to-end HAC path cannot be exercised.** westata's IV estimators accept
``cov_type`` in (nonrobust, HC0..HC3, cluster) only -- ``vce/dispatch.py``'s
``SUPPORTED_COV_TYPES`` contains no HAC entry. So ``estat_overid`` never receives
a HAC result, and the prewhitening branch of ``_wooldridge_score`` is unreachable
through the public API.

The branch is kept because it is part of ``ivregress_estat.ado:250-261``, and its
arithmetic is tested here directly at the function level. **No comparison against
Stata's HAC output has been made**, and the tests say so rather than implying
otherwise.

The sample handling is the substantive point. ``mvreg khat = L(1/lags).khat,
nocons`` drops the first ``lags`` rows. An earlier version of this code padded
them with zeros instead. That is wrong in general -- but *exactly* neutral when
there is a single overidentifying restriction, because step 3's fit then reduces
to ``(sum x)^2 / sum x^2`` and zero rows cancel from both sums. A test pins both
sides of that boundary, since Q = 1 is the common case and would hide the bug.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws
from westata.postestimation.overid import _wooldridge_score


def _prewhiten(khat: np.ndarray, lags: int, mode: str) -> np.ndarray:
    n = khat.shape[0]
    if mode == "drop":
        target = khat[lags:]
        lagged = np.column_stack(
            [khat[lags - 1 - lag : n - 1 - lag] for lag in range(lags)]
        )
        out = np.empty_like(target)
        for c in range(target.shape[1]):
            beta, *_ = np.linalg.lstsq(lagged, target[:, c], rcond=None)
            out[:, c] = target[:, c] - lagged @ beta
        return out
    out = np.empty_like(khat)
    for c in range(khat.shape[1]):
        series = khat[:, c]
        lagged = np.column_stack(
            [series[lags - 1 - lag : n - 1 - lag] for lag in range(lags)]
        )
        beta, *_ = np.linalg.lstsq(lagged, series[lags:], rcond=None)
        padded = np.zeros(n)
        padded[lags:] = series[lags:] - lagged @ beta
        out[:, c] = padded
    return out


def _score(ehat: np.ndarray) -> float:
    ones = np.ones(ehat.shape[0])
    beta, *_ = np.linalg.lstsq(ehat, ones, rcond=None)
    residual = ones - ehat @ beta
    return float(ehat.shape[0] - residual @ residual)


@pytest.fixture()
def iv_pieces():
    rng = np.random.default_rng(31)
    n = 400
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    scale = 0.4 + 1.6 * np.abs(z1)
    u = rng.normal(size=n) * scale
    x = 0.7 * z1 + 0.3 * z2 + 0.6 * u + rng.normal(size=n) * scale
    y = 1.0 + 2.0 * x + u
    import pandas as pd

    frame = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})
    fit = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")
    return fit.iv_structure, np.asarray(fit.resid, dtype=float)


def _khat(structure, resid):
    instruments = np.asarray(structure["instruments"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    exog = np.asarray(structure["exog"], dtype=float)
    names = [str(v) for v in structure["exog_names"]]
    n = resid.size

    constant_names = {"_cons", "intercept", "const"}
    has_const = any(v.strip().lower() in constant_names for v in names)
    included = [j for j, v in enumerate(names) if v.strip().lower() not in constant_names]

    parts = [np.ones(n)] if has_const else []
    if included:
        parts.append(exog[:, included])
    parts.append(instruments)
    all_inst = np.column_stack(parts)

    endoghat = np.empty((n, endog.shape[1]))
    for c in range(endog.shape[1]):
        beta, *_ = np.linalg.lstsq(all_inst, endog[:, c], rcond=None)
        endoghat[:, c] = all_inst @ beta

    cond_parts = [endoghat]
    if has_const:
        cond_parts.insert(0, np.ones(n))
    if included:
        cond_parts.append(exog[:, included])
    cond = np.column_stack(cond_parts)

    restrictions = instruments.shape[1] - endog.shape[1]
    columns = []
    for c in range(instruments.shape[1]):
        if len(columns) >= restrictions:
            break
        beta, *_ = np.linalg.lstsq(cond, instruments[:, c], rcond=None)
        columns.append((instruments[:, c] - cond @ beta) * resid)
    return np.column_stack(columns)


# ------------------------------------------------- reachability is documented
def test_the_prewhitening_branch_is_now_reachable() -> None:
    """Round 29 made HAC available; round 30 wired it into the IV estimators.

    Round 28 recorded this branch as unreachable.  That is no longer true, and
    this test now pins the opposite fact so the documentation has to follow.
    """

    from westata.vce.dispatch import SUPPORTED_COV_TYPES
    from westata.estimators.iv import _IV_COV_TYPES

    assert "HAC" in SUPPORTED_COV_TYPES
    assert "HAC" in _IV_COV_TYPES


def test_lags_is_refused_without_a_hac_vce(iv_pieces) -> None:
    """``ivregress_estat.ado:80-83``: lags() only valid with HAC VCEs."""

    structure, resid = iv_pieces
    rng = np.random.default_rng(1)
    n = resid.size
    frame = _frame(n)
    fit = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(cov_type="HC1")

    with pytest.raises(ValueError, match="only valid with models with HAC"):
        fit.estat_overid(lags=1)


def _frame(n):
    import pandas as pd

    rng = np.random.default_rng(31)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    scale = 0.4 + 1.6 * np.abs(z1)
    u = rng.normal(size=n) * scale
    x = 0.7 * z1 + 0.3 * z2 + 0.6 * u + rng.normal(size=n) * scale
    y = 1.0 + 2.0 * x + u
    return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})


# ------------------------------------------------------------- arithmetic level
def test_lags_zero_is_the_plain_statistic(iv_pieces) -> None:
    structure, resid = iv_pieces
    khat = _khat(structure, resid)

    assert _wooldridge_score(structure, resid, lags=0) == pytest.approx(
        _score(khat), rel=1e-10
    )


@pytest.mark.parametrize("lags", [1, 2, 3])
def test_prewhitening_matches_the_drop_route(iv_pieces, lags) -> None:
    structure, resid = iv_pieces
    khat = _khat(structure, resid)

    expected = _score(_prewhiten(khat, lags, "drop"))
    assert _wooldridge_score(structure, resid, lags=lags) == pytest.approx(
        expected, rel=1e-10
    )


def test_prewhitening_uses_n_minus_lags_rows(iv_pieces) -> None:
    """The statistic's ``N`` must be the step-3 sample, not the full sample."""

    structure, resid = iv_pieces
    khat = _khat(structure, resid)
    lags = 2

    dropped = _score(_prewhiten(khat, lags, "drop"))
    padded = _score(_prewhiten(khat, lags, "pad"))

    got = _wooldridge_score(structure, resid, lags=lags)
    assert got == pytest.approx(dropped, rel=1e-10)
    # At Q = 1 the two routes coincide, so this test cannot distinguish them;
    # ``test_padding_is_neutral_only_at_one_restriction`` covers Q >= 2.
    assert np.isclose(dropped, padded, rtol=1e-10)


def test_padding_is_neutral_only_at_one_restriction() -> None:
    """Pin the boundary: Q=1 hides the bug, Q>=2 exposes it.

    Step 3 at Q = 1 is ``(sum x)^2 / sum x^2``, where zero rows cancel.  At
    Q >= 2 they do not, so zero-padding changes the answer.  Recording both
    sides stops a future refactor from "simplifying" back to padding and passing
    every Q=1 test.
    """

    rng = np.random.default_rng(5)
    n = 300

    # Q = 1: identical.
    khat1 = rng.normal(size=(n, 1))
    assert _score(_prewhiten(khat1, 2, "drop")) == pytest.approx(
        _score(_prewhiten(khat1, 2, "pad")), rel=1e-12
    )

    # Q = 2 and above: genuinely different.
    for Q in (2, 3, 5):
        khat = rng.normal(size=(n, Q))
        dropped = _score(_prewhiten(khat, 2, "drop"))
        padded = _score(_prewhiten(khat, 2, "pad"))
        assert not np.isclose(dropped, padded, rtol=1e-6), Q


def test_prewhitening_changes_the_statistic(iv_pieces) -> None:
    """A prewhitened score must differ from the un-prewhitened one."""

    structure, resid = iv_pieces

    plain = _wooldridge_score(structure, resid, lags=0)
    whitened = _wooldridge_score(structure, resid, lags=1)

    assert not np.isclose(plain, whitened, rtol=1e-6)


def test_end_to_end_through_a_hac_vce() -> None:
    """The public path: a HAC fit reaches the prewhitening branch.

    ``ivregress_estat.ado:87-95`` defaults ``lags`` to 1 for a HAC VCE, so the
    default statistic must equal ``lags=1`` and differ from ``lags=0``.
    """

    import pandas as pd
    import westata as ws

    rng = np.random.default_rng(9)
    n = 400
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + rng.normal()
    x = 0.7 * z1 + 0.3 * z2 + 0.6 * u + rng.normal(size=n)
    y = 1.0 + 2.0 * x + u
    frame = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})

    fit = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )

    default = fit.estat_overid()
    lag0 = fit.estat_overid(lags=0)
    lag1 = fit.estat_overid(lags=1)
    lag3 = fit.estat_overid(lags=3)

    assert default.test == "Wooldridge score"
    assert default.statistic == pytest.approx(lag1.statistic)
    assert not np.isclose(lag0.statistic, lag1.statistic)
    assert not np.isclose(lag1.statistic, lag3.statistic)
