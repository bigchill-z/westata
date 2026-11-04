"""Endogeneity tests: Stata's ``estat endogenous``.

Transcribed from ``ivregress_estat.ado``'s ``Endog2sls`` program (StataNow 19.5).
The Durbin-Wu branch (taken whenever ``e(vce) == "unadjusted"``) is:

.. code-block:: stata

    * consistent model residuals
    predict double `uc' if `touse', residuals

    * efficient model residuals -- OLS when EVERY endogenous regressor is
      being tested; otherwise a restricted 2SLS that treats the tested ones as
      exogenous
    if "`endogeff'" == "" {
        _regress `depvar' `exogeff' `wt' if `touse', `nocons'
    }
    else {
        ivregress 2sls `depvar' `exogeff' (`endogeff' = `addlinst') ...
    }
    predict double `ue' if `touse', residuals
    sca `USSR' = e(rss)

    * projections of the two residual series on the instruments
    _regress `ue' `exogeff' `addlinst' `wt' if `touse', `nocons'
    sca `uePue' = e(mss)
    _regress `uc' `exog' `addlinst' `wt' if `touse', `nocons'
    sca `ucPuc' = e(mss)

    sca `durbin' = (`uePue' - `ucPuc') / (`USSR' / `N')
    local dfFdenom = `N' - `k' - `df'
    sca `wu' = (`uePue' - `ucPuc') / `df' / ///
        ((`USSR' - (`uePue' - `ucPuc')) / `dfFdenom')

so

    Durbin = (mss(ue ~ instruments) - mss(uc ~ instruments)) / (USSR/N)
    Wu     = Durbin-based F with df = #tested endogenous regressors and
             dfFdenom = N - k - df

Note ``nocons`` throughout: ``exogeff``/``exog`` already carry the constant when
the model has one (the ado strips ``_cons`` from the name list but keeps the
variable, hence the explicit ``nocons``).

When every endogenous regressor is tested, the "efficient" model is plain OLS on
``exog + endog``, and ``uePue - ucPuc`` reduces to ``N * R2`` of the first-stage
residuals added to that OLS -- the familiar Durbin-Wu-Hausman statistic.

Under a robust/clustered VCE the ado switches to the regression-based (rob) test
instead of Durbin-Wu; that variant is not implemented here and is refused rather
than reported under the Durbin-Wu name.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["EndogenousResult", "estat_endogenous"]

_CONSTANT_NAMES = frozenset({"intercept", "intercept_", "_cons", "const"})


def _is_constant(name: str) -> bool:
    return str(name).strip().lower() in _CONSTANT_NAMES


def _six_g(value: float) -> str:
    """Stata's ``%9.0g`` column: six significant figures, leading zero dropped."""

    number = float(value)
    text = f"{number:.6g}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


class EndogenousResult:
    """Durbin and Wu endogeneity tests.

    Attributes
    ----------
    durbin, durbin_pvalue:
        Durbin's chi-square statistic and its p-value, with ``df`` equal to the
        number of tested endogenous regressors.
    wu, wu_pvalue, wu_df, wu_df_denom:
        Wu's F statistic, its two degrees of freedom and the p-value.
    tested:
        Names of the endogenous regressors the test covers.
    """

    __slots__ = (
        "durbin",
        "durbin_pvalue",
        "wu",
        "wu_pvalue",
        "wu_df",
        "wu_df_denom",
        "tested",
        "test",
    )

    def __init__(
        self,
        *,
        durbin: float,
        durbin_pvalue: float,
        wu: float,
        wu_pvalue: float,
        wu_df: float,
        wu_df_denom: float,
        tested,
    ) -> None:
        self.durbin = float(durbin)
        self.durbin_pvalue = float(durbin_pvalue)
        self.wu = float(wu)
        self.wu_pvalue = float(wu_pvalue)
        self.wu_df = float(wu_df)
        self.wu_df_denom = float(wu_df_denom)
        self.tested = list(tested)
        #: The sibling result types (``RegFResult``, ``EndogScoreResult``) both carry a
        #: ``test`` label, so a caller can branch on it rather than on the class name.
        #: This one did not, which meant ``getattr(result, "test", None)`` returned
        #: ``None`` for exactly one of the three branches.
        self.test = "Durbin-Wu-Hausman"

    def __repr__(self) -> str:
        return (
            f"<EndogenousResult durbin chi2({self.wu_df:g})={self.durbin:.4f} "
            f"p={self.durbin_pvalue:.4f}>"
        )

    def __str__(self) -> str:
        """Reproduce ``estat endogenous`` exactly::

            Tests of endogeneity
            H0: Variables are exogenous
            Durbin (score) chi2(1)          =  192.536  (p = 0.0000)
            Wu-Hausman F(1,396)             =  367.505  (p = 0.0000)

        Two details: the null line capitalises "Variables", and "=" sits at column 32 (a
        32-wide label field plus a space). Both statistics print at six significant figures,
        so the Durbin value shows 192.536 rather than 192.5357.
        """

        durbin_label = f"Durbin (score) chi2({self.wu_df:g})"
        wu_label = f"Wu-Hausman F({self.wu_df:g},{self.wu_df_denom:g})"
        return chr(10).join(
            [
                "",
                "Tests of endogeneity",
                "H0: Variables are exogenous",
                f"{durbin_label:<31} = {_six_g(self.durbin):>8}"
                f"  (p = {self.durbin_pvalue:.4f})",
                f"{wu_label:<31} = {_six_g(self.wu):>8}"
                f"  (p = {self.wu_pvalue:.4f})",
            ]
        )

    @property
    def test_count(self) -> int:
        """How many endogenous regressors the test covers."""

        return len(self.tested)

    def to_dict(self) -> dict[str, float]:
        """The ``r()`` scalars Stata returns."""

        return {
            "durbin": self.durbin,
            "p_durbin": self.durbin_pvalue,
            "wu": self.wu,
            "p_wu": self.wu_pvalue,
            "wudf_r": self.wu_df_denom,
        }


def _require_iv(result: Any) -> dict[str, Any]:
    structure = getattr(result, "iv_structure", None)
    if structure is None:
        raise NotImplementedError(
            "estat endogenous is only defined after an instrumental-variables "
            "fit: Stata reaches it through ivregress_estat and it needs the "
            "split between included exogenous regressors, endogenous "
            "regressors and excluded instruments; got "
            f"{getattr(result, 'model_name', 'an unnamed model')!r} without that "
            "structure"
        )
    if structure["endog"].shape[1] == 0:
        raise ValueError("estat endogenous: model contains no endogenous variables")
    return structure


def _ols_mss(
    y: np.ndarray, design: np.ndarray
) -> tuple[float, float]:
    """OLS without a constant; returns ``(mss, rss)``.

    ``mss`` is Stata's ``e(mss)``: the explained sum of squares about the mean
    (``tss_centred - rss``), which is what the ado's ``e(mss)`` holds.
    """

    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coefficients
    rss = float(resid @ resid)
    centred = y - y.mean()
    tss = float(centred @ centred)
    return tss - rss, rss


def estat_endogenous(
    result: Any, *, lags: int | None = None
) -> Any:
    """Stata's ``estat endogenous``.

    Which test comes back depends on the VCE
    (``ivregress_estat.ado:1128-1145``):

    ==============  =========================================================
    ``vce``         reported
    ==============  =========================================================
    unadjusted      Durbin-Wu (``durbin``, ``wu``)
    robust          the robust score (``robwool``)
    HAC             the HAC score (``acwool``)
    cluster         refused -- Stata reports only ``regF``, which needs the
                    augmented regression and is not implemented here
    ==============  =========================================================

    Parameters
    ----------
    lags:
        HAC truncation for ``acwool``.  ``ivregress_estat.ado:1055-1063`` defaults
        it to 1 under a HAC VCE and rejects a non-positive value; ``lags`` is only
        valid with HAC.

    Raises
    ------
    NotImplementedError
        For a non-IV result.
    ValueError
        When the model has no endogenous regressor, when ``lags`` is given
        without a HAC VCE, or after a clustered VCE.
    """

    structure = _require_iv(result)

    model_name = str(getattr(result, "model_name", "")).strip().lower()
    if "liml" in model_name:
        # ``ivregress_estat.ado:1002-1019``:
        #     if "`e(estimator)'" == "2sls"        Endog2sls
        #     else if "`e(estimator)'" == "gmm"    Endoggmm
        #     else  di as error "not available after ... estimator" ; exit 321
        # LIML falls into that final ``else``: there is no LIML branch at all.
        # Both tests below are derived for 2SLS (``Endog2sls`` regresses on
        # ``e(exogr)``/``e(insts)``); running them on a k-class fit would report a
        # number with no derivation, which is why Stata refuses outright.
        raise NotImplementedError(
            "estat endogenous is not available after LIML estimation: "
            "ivregress_estat.ado:1002-1019 dispatches only 2SLS (Endog2sls) and "
            "GMM (Endoggmm) and raises error 321 for anything else.  Use IV2SLS "
            "or IVGMM."
        )

    if "gmm" in model_name:
        # ``Endoggmm`` (``ado:1427-1524``) reports the GMM C statistic in EVERY
        # case -- ``C = e(J) - J'`` from a constrained refit whose weight matrix
        # drops the challenged orthogonality conditions.  It never computes a
        # Durbin-Wu pair, so returning one here would mislabel a 2SLS
        # construction as the GMM answer.
        raise NotImplementedError(
            "estat endogenous after GMM reports the GMM C statistic "
            "(C = e(J) - J' from a constrained refit), not the Durbin-Wu pair or "
            "the Wooldridge score; ivregress_estat.ado:1427-1524.  That "
            "constrained refit is not implemented here, so no number is reported "
            "rather than a mislabelled one.  Use IV2SLS for the score test."
        )

    cov_type = str(getattr(result, "cov_type", "nonrobust")).strip().lower()
    is_hac = "hac" in cov_type or "newey" in cov_type
    is_clustered = "cluster" in cov_type
    is_robust = cov_type in ("hc0", "hc1", "hc2", "hc3", "robust")

    if lags is not None and not is_hac:
        raise ValueError(
            "lags() only valid with models with HAC VCEs; this result used "
            f"{cov_type or 'the default nonrobust VCE'}"
        )

    if is_clustered:
        # ``ivregress_estat.ado:1134-1136``: a clustered VCE gets ONLY ``regF``
        # (the score tests are not run there).  ``regF`` is the augmented-
        # regression F test, implemented in ``endog_regf``.
        from westata.postestimation.endog_regf import endog_regF_test

        fitted = np.asarray(result.fittedvalues, dtype=float).reshape(-1)
        residual = np.asarray(result.resid, dtype=float).reshape(-1)
        labels = getattr(result, "cluster_labels", None)
        return endog_regF_test(
            structure,
            fitted + residual,
            cov_type="cluster",
            cluster_labels=labels,
        )

    if is_robust or is_hac:
        from westata.postestimation.endog_score import endog_score_test

        # The estimation-sample response is rebuilt from the fitted values and
        # residuals, which the result always carries.
        fitted = np.asarray(result.fittedvalues, dtype=float).reshape(-1)
        residual = np.asarray(result.resid, dtype=float).reshape(-1)
        y_values = fitted + residual
        if is_hac:
            effective_lags = 1 if lags is None else int(lags)
            if effective_lags < 1:
                raise ValueError(
                    f"lags() must be positive under a HAC VCE; got {effective_lags}"
                )
        else:
            effective_lags = 0
        return endog_score_test(
            structure,
            residual,
            y_values,
            hac=is_hac,
            lags=effective_lags,
        )

    if cov_type not in ("nonrobust", "unadjusted", "ols", ""):
        raise ValueError(
            "estat endogenous: the Durbin-Wu form assumes an unadjusted VCE; "
            f"got {cov_type}"
            f"after a {cov_type} VCE Stata runs a regression-based variant "
            "instead, which westata does not implement"
        )

    exog = np.asarray(structure["exog"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    instruments = np.asarray(structure["instruments"], dtype=float)
    y = np.asarray(structure["response"], dtype=float).reshape(-1)
    exog_names = list(structure["exog_names"])
    endog_names = list(structure["endog_names"])

    n = y.size
    has_constant = any(_is_constant(name) for name in exog_names)
    included = [j for j, name in enumerate(exog_names) if not _is_constant(name)]
    included_matrix = exog[:, included] if included else np.empty((n, 0))

    if has_constant:
        included_with_constant = np.column_stack([np.ones(n), included_matrix])
    else:
        included_with_constant = included_matrix

    # ``addlinst``: the excluded instruments, i.e. inst minus exog.
    addlinst = instruments

    # --- consistent-model residuals: the fitted IV residuals ---------------
    uc = np.asarray(result.resid, dtype=float).reshape(-1)

    # --- efficient-model residuals ----------------------------------------
    # Every endogenous regressor is tested here, so ``endogeff`` is empty and
    # the efficient model is plain OLS of y on exog + endog, with ``nocons``
    # (the constant is already inside included_matrix when present).
    efficient_design = np.column_stack([included_matrix, endog])
    if has_constant:
        efficient_design = np.column_stack([np.ones(n), efficient_design])
    eff_coef, *_ = np.linalg.lstsq(efficient_design, y, rcond=None)
    ue = y - efficient_design @ eff_coef
    ussr = float(ue @ ue)

    # --- projections on the instruments -----------------------------------
    # ue ~ exogeff + addlinst, noconstant.  ``exogeff`` = exog + tested endog, so
    # with every endogenous regressor tested exogeff is the full structural
    # regressor set.
    design_eff = np.column_stack([included_with_constant, endog, addlinst])
    ue_p_ue, _ = _ols_mss(ue, design_eff)

    # uc ~ exog + addlinst, noconstant
    design_cons = np.column_stack([included_with_constant, addlinst])
    uc_p_uc, _ = _ols_mss(uc, design_cons)

    difference = ue_p_ue - uc_p_uc
    durbin = difference / (ussr / n)

    # Wu: F form.
    df = float(endog.shape[1])
    k = exog.shape[1] + endog.shape[1]  # structural coefficients (incl. const)
    df_denom = float(n - k - df)
    if df_denom > 0.0 and (ussr - difference) > 0.0:
        wu = (difference / df) / ((ussr - difference) / df_denom)
        wu_p = float(stats.f.sf(wu, df, df_denom))
    else:  # pragma: no cover - degenerate fits only
        wu = float("nan")
        wu_p = float("nan")

    return EndogenousResult(
        durbin=durbin,
        durbin_pvalue=float(stats.chi2.sf(durbin, df)),
        wu=wu,
        wu_pvalue=wu_p,
        wu_df=df,
        wu_df_denom=df_denom,
        tested=endog_names,
    )
