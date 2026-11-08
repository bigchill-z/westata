"""`estat ptrends` exists now, and it is CLOSE but not exact -- stated plainly.

C13-01's sibling, T13: `didregress`'s postestimation surface. `estat ptrends` is the one the report
names, so it is the one implemented here.

AGAINST A LIVE STATA (60 groups, 15 periods, 20 treated, seeded):

    Stata  F(1, 59) = 0.251088   p = 0.618175
    here   F(1, 59) = 0.254200   p = 0.616000

The denominator degrees of freedom match exactly (59 = n_groups - 1), so the group accounting is
right. The statistic is 1.2% high.

WHAT IS ESTABLISHED: the test is a single linear-trend interaction on the PRETREATMENT window, after
a two-way within transformation. Each of those was necessary:

  * without demeaning, the group and period effects stayed in the residual and F came out 0.7654;
  * with demeaning by group AND period it drops to 0.2542;
  * clustering to match `didregress`'s default made it worse (0.3005), so the remaining gap is not
    the variance estimator.

THE GAP IS RECORDED AS UNRESOLVED rather than tuned away. options.md T13 lists the next hypothesis:
Stata may use a different residual-dof convention, or may fit the trend on the untreated-only
comparison group.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.did import estat_ptrends

#: Captured from a live `estat ptrends` on the fixture below.
STATA_STATISTIC = 0.251088
STATA_PVALUE = 0.618175


@pytest.fixture(scope="module")
def did_result():
    rng = np.random.default_rng(7)
    n_groups, n_periods = 60, 15
    group = np.repeat(np.arange(n_groups), n_periods)
    period = np.tile(np.arange(1, n_periods + 1), n_groups)
    treated_units = rng.choice(n_groups, 20, replace=False)
    post = ((np.isin(group, treated_units)) & (period >= 9)).astype(float)
    outcome = 1.0 + 0.5 * post + 0.05 * period + rng.normal(size=n_groups * n_periods)

    frame = pd.DataFrame(
        {"y": outcome, "canal_post": post, "county": group, "year": period}
    )
    return ws.Didregress.from_formula(
        "y", "canal_post", frame, group="county", time="year"
    ).fit()


def test_ptrends_returns_the_population_shape(did_result) -> None:
    """The four quantities Stata's `F(1, 59)` line needs."""

    outcome = estat_ptrends(did_result)
    assert set(outcome) == {"statistic", "df_numerator", "df_denominator", "pvalue"}


def test_the_denominator_dof_is_n_groups_minus_one(did_result) -> None:
    """59 matches Stata exactly, which is what identifies the design as 60 groups."""

    assert estat_ptrends(did_result)["df_denominator"] == 59.0


def test_the_statistic_is_close_but_not_yet_exact(did_result) -> None:
    """1.2% high. Pinned at that tolerance so a real fix lands as a failure HERE.

    When the remaining gap is closed, tighten this to abs=1e-6 against STATA_STATISTIC.
    """

    observed = estat_ptrends(did_result)["statistic"]
    assert observed == pytest.approx(STATA_STATISTIC, rel=0.02)
    # And it is NOT exact yet -- this assertion is the reminder.
    assert observed != pytest.approx(STATA_STATISTIC, abs=1e-4)


def test_it_refuses_a_result_without_a_did_design() -> None:
    """A plain DID-less result must not silently return a number."""

    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"y": rng.normal(size=40), "x": rng.normal(size=40)})
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()

    with pytest.raises(Exception):
        estat_ptrends(fitted)
