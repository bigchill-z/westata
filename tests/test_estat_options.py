"""Every documented OPTION of every `estat` method must run -- and matter.

The audits in rounds 72-77 checked the (method x estimator) and
(predict type x estimator) grids, i.e. the DEFAULTS. Optional arguments were never
covered, which is the same blind spot that let `Sqreg(seed=None)` crash for 70
rounds.

Every option value here comes from Stata's own source, not from guessing:

* ``mtest()`` methods -- ``_mtest.ado:84``: ``local opt `bonferroni' `holm'
  `sidak' `noadjust'``
* ``hettest`` methods -- ``hettest.ado``: normal / iid / fstat
* ``vif`` options -- ``vif.ado:121-122``: ``nosort`` suppresses ``gsort -vif``
* ``imtest`` -- ``imtest.ado``: ``white``

Two assertions per option: it RUNS, and it CHANGES the output. An option accepted
and then ignored is the quietest possible bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    """Heteroskedastic by construction, so ``hettest`` has something to reject.

    The regressors carry clearly DIFFERENT VIFs: two nearly equal VIFs would make
    the ``nosort`` comparison look identical even when sorting works -- which is
    exactly the false alarm the first audit run produced.
    """

    rng = np.random.default_rng(31)
    n = 300
    x3 = rng.normal(size=n)
    x2 = x3 * 0.85 + rng.normal(size=n) * 0.5  # high VIF with x3
    x1 = rng.normal(size=n)  # low VIF
    y = 1.0 + 0.6 * x1 - 0.4 * x2 + 0.3 * x3 + rng.normal(size=n) * (0.5 + np.abs(x1))
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    frame["yb"] = (y > np.median(y)).astype(int)
    return frame


@pytest.fixture(scope="module")
def linear(frame):
    return ws.OLS.from_formula("y ~ x1 + x2 + x3", frame).fit()


@pytest.fixture(scope="module")
def logit(frame):
    return ws.Logit.from_formula("yb ~ x1 + x2", frame).fit()


# ------------------------------------------------------------------- they run
@pytest.mark.parametrize("method", ["normal", "iid", "fstat"])
def test_hettest_methods_run(linear, method) -> None:
    assert linear.estat_hettest(method=method) is not None


@pytest.mark.parametrize("mtest", ["bonferroni", "holm", "sidak", "noadjust"])
def test_hettest_mtest_methods_run(linear, mtest) -> None:
    """``_mtest.ado:84`` lists exactly these four."""

    assert linear.estat_hettest(mtest=mtest) is not None


@pytest.mark.parametrize(
    ("kwargs", "label"),
    [
        ({}, "default"),
        ({"uncentered": True}, "uncentered"),
        ({"nosort": True}, "nosort"),
        ({"uncentered": True, "nosort": True}, "both"),
    ],
)
def test_vif_option_combinations_run(linear, kwargs, label) -> None:
    assert linear.estat_vif(**kwargs) is not None, label


@pytest.mark.parametrize("kwargs", [{}, {"white": True}])
def test_imtest_options_run(linear, kwargs) -> None:
    assert linear.estat_imtest(**kwargs) is not None


@pytest.mark.parametrize("kwargs", [{}, {"rhs": True}])
def test_ovtest_options_run(linear, kwargs) -> None:
    assert linear.estat_ovtest(**kwargs) is not None


@pytest.mark.parametrize("df", [None, 1, 2])
def test_ic_df_values_run(linear, df) -> None:
    assert linear.estat_ic() if df is None else linear.estat_ic(df=df)


@pytest.mark.parametrize("cutoff", [None, 0.3, 0.5, 0.7])
def test_classification_cutoffs_run(logit, cutoff) -> None:
    assert (
        logit.estat_classification()
        if cutoff is None
        else logit.estat_classification(cutoff=cutoff)
    )


# ---------------------------------------------------- they must CHANGE output
def test_hettest_methods_differ(linear) -> None:
    """``normal`` uses ``e(mss)/2``, ``iid`` uses ``N*R2``, ``fstat`` an F.

    Three different formulas, so the three statistics must not coincide.
    """

    normal = float(linear.estat_hettest(method="normal").statistic)
    iid = float(linear.estat_hettest(method="iid").statistic)
    fstat = float(linear.estat_hettest(method="fstat").statistic)

    assert not np.isclose(normal, iid, rtol=1e-6)
    assert not np.isclose(normal, fstat, rtol=1e-6)


def test_mtest_adjusts_the_p_values(linear) -> None:
    """``_mtest`` multiplies p-values; leaving them untouched means it did nothing."""

    unadjusted = linear.estat_hettest()
    adjusted = linear.estat_hettest(mtest="holm")

    assert str(unadjusted) != str(adjusted)


def test_vif_nosort_changes_the_row_order(linear) -> None:
    """``vif.ado:121-122``: the default does ``gsort -vif``; ``nosort`` skips it.

    The fixture deliberately has clearly different VIFs. With two nearly equal
    VIFs the sorted and unsorted order coincide and the check would pass even if
    ``nosort`` were ignored.
    """

    sorted_result = linear.estat_vif()
    unsorted_result = linear.estat_vif(nosort=True)

    assert list(sorted_result.names) != list(unsorted_result.names)

    # The default really is ordered by VIF, descending.
    assert np.all(np.diff(sorted_result.vif) <= 1e-9)

    # And nosort preserves the model's own order.
    assert list(unsorted_result.names) == ["x1", "x2", "x3"]


def test_vif_uncentered_changes_the_values(linear) -> None:
    """``uncentered`` drops the constant from each auxiliary regression.

    It also reports a row for the constant itself, so the two results do not have
    the same length -- comparing elementwise would be wrong. Compare the overlap
    by NAME instead.
    """

    centred = linear.estat_vif()
    uncentered = linear.estat_vif(uncentered=True)

    centred_map = dict(zip(centred.names, centred.vif))
    uncentered_map = dict(zip(uncentered.names, uncentered.vif))

    shared = sorted(set(centred_map) & set(uncentered_map))
    assert shared, "the two runs share no variables"
    assert any(
        not np.isclose(centred_map[name], uncentered_map[name], rtol=1e-6)
        for name in shared
    ), "uncentered changed nothing on the shared variables"


def test_imtest_white_adds_the_white_component(linear) -> None:
    """``white`` adds a heteroskedasticity term to the Cameron-Trivedi split."""

    plain = linear.estat_imtest()
    with_white = linear.estat_imtest(white=True)

    assert str(plain) != str(with_white)


def test_ic_df_changes_the_criterion(linear) -> None:
    """``estat ic`` is ``-2*ll + 2*df``, so a different ``df`` must move it."""

    first = linear.estat_ic(df=1)
    second = linear.estat_ic(df=2)

    assert str(first) != str(second)


def test_classification_cutoff_changes_the_table(logit) -> None:
    low = logit.estat_classification(cutoff=0.3)
    high = logit.estat_classification(cutoff=0.7)

    assert str(low) != str(high)


def test_vce_correlation_changes_the_render(linear) -> None:
    covariance = linear.estat_vce()
    correlation = linear.estat_vce(correlation=True)

    assert str(covariance) != str(correlation)


# ---------------------------------------------------------------- refusals
@pytest.mark.parametrize("mtest", ["scheffe", "notamethod"])
def test_unknown_mtest_is_refused(linear, mtest) -> None:
    """``_mtest.ado:84`` accepts four methods; anything else must be rejected."""

    with pytest.raises(ValueError, match="mtest"):
        linear.estat_hettest(mtest=mtest)


def test_mtest_with_iid_is_refused(linear) -> None:
    """``hettest.ado:49-61``: ``mtest()`` may not be combined with ``iid``."""

    with pytest.raises(ValueError, match="iid"):
        linear.estat_hettest(method="iid", mtest="holm")


def test_unknown_hettest_method_is_refused(linear) -> None:
    with pytest.raises(ValueError, match="method"):
        linear.estat_hettest(method="notamethod")
