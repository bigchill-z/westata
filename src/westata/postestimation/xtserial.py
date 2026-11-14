"""Wooldridge's panel serial-correlation test: the user-written Stata ``xtserial``.

``xtserial`` is not part of Stata -- it is Drukker's ado (``xtserial.ado`` v1.1.0,
22apr2003), distributed outside SSC.  It is short enough to transcribe in full::

    que tsset
    local ivar "`r(panelvar)'"
    local tvar "`r(timevar)'"
    qui regress d.(`varlist') if `touse', nocons cluster(`ivar')
    qui predict double `res' if `touse', residuals
    qui regress `res' l.`res' if `touse', nocons cluster(`ivar')
    qui test l.`res' == -.5
    scalar `corr' = _b[l.`res']
    scalar `df_r' = r(df_r)
    scalar `F'    = r(F)
    scalar `df'   = r(df)
    scalar `p'    = r(p)

So the procedure is:

1. difference the response **and every regressor**, no constant, clustered on the panel;
2. take those residuals and lag them once, again no constant and clustered;
3. test whether the lag coefficient equals **-0.5** (not zero), and report that Wald
   statistic as F with ``df = 1`` and ``df_r = G - 1``.

Reference, StataNow 19.5 with ``xtset province t`` on lin_1992.dta::

    xtserial ltvfo ltlan ltwlab ltpow ltfer hrs mipric1 giprice mci ngca, output

        Wooldridge test for autocorrelation in panel data
        H0: no first order autocorrelation
            F(  1,      27) =     12.511
                   Prob > F =      0.0015

        r(corr) = -0.192578492734578   r(F) = 12.511374216982579
        r(df) = 1   r(df_r) = 27   r(p) = 0.001484513025783

Two traps, both measured rather than guessed:

* **Lags follow (panel, time), not row order.**  Every panel of lin_1992 misses
  ``t == 11``, so ``D.`` is missing at ``t == 11`` *and* ``t == 12``: 476 - 56 = 420.
  Pairing adjacent rows gives 448 and moves ``r(corr)`` from -0.19258 to -0.20871.
* **The null is -0.5, not 0.**  Differencing an AR(1) error gives a lag coefficient of
  ``rho - 1``, which is -0.5 at ``rho = 0.5`` -- the value implied by no serial
  correlation.  Testing against zero would always reject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError
from westata.estimators.linear import OLS

__all__ = ["XtSerialResult", "xtserial"]


@dataclass(frozen=True)
class XtSerialResult:
    """``xtserial``: Wooldridge's test for first-order panel autocorrelation."""

    #: ``r(corr)``: the lag coefficient of the differenced residual (about -0.5 under H0).
    corr: float
    #: ``r(F)``: the Wald statistic for ``corr == -0.5``.
    f: float
    #: ``r(df)``: 1 for this test.
    df: int
    #: ``r(df_r)``: ``number of panels - 1``.
    df_r: int
    #: ``r(p)``: upper-tail p-value of F.
    p: float
    #: Observations in the first-differenced regression (Stata's displayed N).
    nobs: int
    #: Panels contributing to the second-stage regression; ``df_r`` is this minus one.
    n_panels: int
    #: The first-stage regression: ``D.y`` on ``D.x1 ...`` with no constant, clustered
    #: on the panel.  This is exactly what Stata's ``, output`` prints before the
    #: three-line test block, kept as a full fitted result rather than discarded --
    #: the differenced equation is an estimate in its own right (its coefficients are
    #: ``beta`` of the AR(1)-differenced model), and ``, output`` exists precisely
    #: because readers want to see it.  ``None`` only when the caller asked to skip it.
    first_stage: Any = None

    def to_dict(self) -> dict[str, Any]:
        """The five ``r()`` scalars, keyed as Stata names them (R4.1)."""

        return {
            "corr": self.corr,
            "F": self.f,
            "df": self.df,
            "df_r": self.df_r,
            "p": self.p,
            "N": self.nobs,
            "N_g": self.n_panels,
        }

    def first_stage_table(self) -> str:
        """The first-stage regression as Stata prints it under ``, output``.

        Stata labels the response ``D.ltvfo`` and each row ``ltlan / D1.`` -- the
        differenced-variable banner is part of the output, so the fixture comparison
        needs it.  Returns an empty string when the fit was skipped.
        """

        if self.first_stage is None:
            return ""
        body = str(self.first_stage.summary())
        return body.replace("Linear regression", "Linear regression", 1)

    def to_frame(self) -> pd.DataFrame:
        """One row per scalar, which is how the table reads on screen."""

        rows = [
            ("corr", self.corr),
            ("F", self.f),
            ("df", float(self.df)),
            ("df_r", float(self.df_r)),
            ("Prob > F", self.p),
        ]
        return pd.DataFrame(rows, columns=["statistic", "value"]).set_index("statistic")

    def __str__(self) -> str:
        """Stata's block, blank line first, columns measured from the capture.

        ``xtserial.ado`` builds the two statistic lines with explicit ``_col`` offsets::

            di as txt _col(5) "F(" %3.0f `df' "," %8.0f `df_r' ") = " as res %10.3f `F'
            di as txt _col(12) "Prob > F = " as res %11.4f `p'

        so the leading ``F(`` starts at column 5 and ``Prob > F`` at column 12; the
        verbatim capture is ``'    F(  1,      27) =     12.511'`` and
        ``'           Prob > F =      0.0015'``.
        """

        return (
            "\n"
            "Wooldridge test for autocorrelation in panel data\n"
            "H0: no first order autocorrelation\n"
            f"{'':>4}F({self.df:>3.0f},{self.df_r:>8.0f}) = {self.f:>10.3f}\n"
            f"{'Prob > F = ':>22}{self.p:>11.4f}"
        )


def _lag_on_axis(values: np.ndarray, panel: np.ndarray, time: np.ndarray) -> np.ndarray:
    """``L.`` on the (panel, time) axis: NaN wherever ``t - 1`` is not in the panel."""

    lookup = {(p, t): v for p, t, v in zip(panel, time, values)}
    out = np.full(values.size, np.nan)
    for index, (p, t) in enumerate(zip(panel, time)):
        out[index] = lookup.get((p, t - 1.0), np.nan)
    return out


def xtserial(
    data: pd.DataFrame,
    formula: str,
    *,
    entity: str,
    time: str,
    keep_first_stage: bool = True,
) -> XtSerialResult:
    """Wooldridge's test for serial correlation in linear panel-data models.

    ``formula`` is a ``"y ~ x1 + x2"`` spec, ``entity`` the panel variable and ``time``
    the period variable -- what ``xtset`` would declare.  The model is fitted in FIRST
    DIFFERENCES, so ``entity`` and ``time`` must identify consecutive periods.

    Stata's ``xtserial`` takes a variable list, not a formula; this mirrors the rest of
    the package's ``from_formula`` convention.

    **The first-stage regression is returned, not discarded.**  Stata prints it when
    ``, output`` is given; here it is always fitted (the test needs its residuals) and
    kept on ``result.first_stage`` as an ordinary :class:`StataResults`, so the
    differenced equation can be inspected, tabulated or reused like any other fit.
    Print it with ``print(result.first_stage.summary())``.  Pass
    ``keep_first_stage=False`` to drop the reference when only the three-line test
    block is wanted.

    Raises
    ------
    ConsistencyError
        When the panel/time columns are absent, the panel is not usable, or the second
        stage has too few panels for a Wald test.
    """

    response = formula.split("~", 1)[0].strip()
    regressors = [term.strip() for term in formula.split("~", 1)[1].split("+")]
    regressors = [term for term in regressors if term and term != "-1" and term != "1"]
    if not regressors:
        raise ConsistencyError("xtserial: the formula lists no regressors")
    for column in (response, *regressors, entity, time):
        if column not in data.columns:
            raise ConsistencyError(f"xtserial: {column!r} is not a column of the data")

    working = data.dropna(subset=[response, *regressors]).sort_values([entity, time])
    working = working.reset_index(drop=True)
    panel = working[entity].to_numpy()
    periods = np.asarray(working[time], dtype=float)

    columns = [response, *regressors]
    differenced = pd.DataFrame(
        {
            f"d_{c}": working[c].to_numpy(dtype=float)
            - _lag_on_axis(working[c].to_numpy(dtype=float), panel, periods)
            for c in columns
        }
    )
    differenced[entity] = panel
    differenced["_t"] = periods
    differenced = differenced.dropna(subset=[f"d_{c}" for c in columns]).reset_index(drop=True)

    first = OLS.from_formula(
        f"d_{response} ~ " + " + ".join(f"d_{c}" for c in regressors) + " - 1",
        differenced,
    ).fit(cov_type="cluster", cov_kwds={"groups": differenced[entity]})

    resid = np.asarray(first.resid, dtype=float)
    lagged = _lag_on_axis(
        resid,
        differenced[entity].to_numpy(),
        differenced["_t"].to_numpy(),
    )
    keep = np.isfinite(lagged)
    if keep.sum() <= 1:
        raise ConsistencyError(
            "xtserial: no consecutive observations survive the lag; the panel needs "
            "at least two adjacent periods per unit"
        )

    second_data = pd.DataFrame(
        {
            "_res": resid[keep],
            "_lag": lagged[keep],
            entity: differenced[entity].to_numpy()[keep],
        }
    )
    second = OLS.from_formula("_res ~ _lag - 1", second_data).fit(
        cov_type="cluster", cov_kwds={"groups": second_data[entity]}
    )

    corr = float(second.params["_lag"])
    variance = float(second.bse["_lag"]) ** 2
    if not np.isfinite(variance) or variance <= 0.0:
        raise ConsistencyError("xtserial: the second-stage variance is not positive")
    f_value = float(((corr + 0.5) ** 2) / variance)
    n_panels = int(second_data[entity].nunique())
    df_r = n_panels - 1
    if df_r < 1:
        raise ConsistencyError(
            f"xtserial: {n_panels} panel(s) is too few for a clustered Wald test"
        )
    return XtSerialResult(
        corr=corr,
        f=f_value,
        df=1,
        df_r=df_r,
        p=float(stats.f.sf(f_value, 1, df_r)),
        nobs=int(first.nobs),
        n_panels=n_panels,
        first_stage=first if keep_first_stage else None,
    )
