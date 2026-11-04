"""Overidentification tests: Stata's ``estat overid``.

Transcribed from ``ivregress_estat.ado``'s ``OverID2sls`` program (StataNow
19.5).  The homoskedastic branch is short:

.. code-block:: stata

    sca `eqnrss' = e(rss)                       // main-equation RSS
    qui predict double `resids' if `touse', residual
    qui _regress `resids' `allinst' if `touse' `wt', `constant'
    scalar `sargan'  = e(N) * (1 - e(rss)/`eqnrss')
    scalar `basmann' = `sargan' * (e(N)-`k') / (e(N) - `sargan')

    // Wooldridge's score test: with homoskedastic errors it IS Sargan's
    if "`vcetype'" == "" {
        sca `score' = `sargan'
    }

So the Sargan statistic is ``N * (1 - rss_aux/rss_main)``, where ``rss_aux`` is
the residual sum of squares from regressing the IV residuals on **all**
instruments (included exogenous plus excluded instruments).  Two details decide
whether the number is right:

* the auxiliary regression includes the constant when the model has one;
* the ratio is ``rss_aux/rss_main`` -- dividing by the *auxiliary* RSS instead
  gives a statistic with no distribution.

The degrees of freedom are the number of overidentifying restrictions,
``#excluded instruments - #endogenous regressors``.

``estat overid`` is refused after a robust/clustered VCE: ``OverID2sls`` sets
both statistics missing when ``e(vcetype)`` is Robust or HAC, and the robust
route is the Wooldridge score test, which the ado only implements for non-cluster
VCEs.  Rather than report a statistic of a different kind under the same name,
the westata version raises.

For GMM the analogous test is Hansen's J, which ``ivregress`` already stores as
``e(J)`` (``ivregress_estat.ado:428-436``); :func:`estat_overid` returns it for
that estimator instead of Sargan, because Sargan assumes homoskedasticity and J
does not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["OverIdResult", "estat_overid"]

_CONSTANT_NAMES = frozenset({"intercept", "intercept_", "_cons", "const"})


def _is_constant(name: str) -> bool:
    return str(name).strip().lower() in _CONSTANT_NAMES


class OverIdResult:
    """The overidentification test.

    Attributes
    ----------
    statistic, df, pvalue:
        The reported chi-square statistic, its degrees of freedom (the number of
        overidentifying restrictions) and the upper-tail p-value.
    basmann, basmann_pvalue:
        Basmann's finite-sample variant, ``Sargan*(N-k)/(N-Sargan)``.  ``None``
        for GMM, where Hansen's J is reported instead.
    test:
        ``"Sargan"`` (2SLS/LIML) or ``"Hansen J"`` (GMM).
    """

    __slots__ = (
        "statistic",
        "df",
        "pvalue",
        "basmann",
        "basmann_pvalue",
        "test",
    )

    def __init__(
        self,
        *,
        statistic: float,
        df: float,
        pvalue: float,
        test: str,
        basmann: float | None = None,
        basmann_pvalue: float | None = None,
    ) -> None:
        self.statistic = float(statistic)
        self.df = float(df)
        self.pvalue = float(pvalue)
        self.test = str(test)
        self.basmann = None if basmann is None else float(basmann)
        self.basmann_pvalue = (
            None if basmann_pvalue is None else float(basmann_pvalue)
        )

    def __repr__(self) -> str:
        return (
            f"<OverIdResult {self.test} chi2({self.df:g})={self.statistic:.4f} "
            f"p={self.pvalue:.4f}>"
        )

    def __str__(self) -> str:
        """Reproduce ``estat overid`` exactly::

            Tests of overidentifying restrictions:
            Sargan (score) chi2(1) =  1.20073  (p = 0.2732)
            Basmann chi2(1)        =   1.1923  (p = 0.2749)

        Two formats were wrong before: the statistic and its p-value were on SEPARATE
        lines, where Stata puts "  (p = x.xxxx)" on the same line, and the Sargan row was
        labelled without the "(score)" qualifier Stata prints.
        """

        label = "Sargan (score)" if self.test == "Sargan" else self.test
        sargan_row = (
            f"{label + ' chi2(' + f'{self.df:g}' + ')':<22}"
            f" = {self.statistic:>8.5f}  (p = {self.pvalue:.4f})"
        )
        lines = ["", "Tests of overidentifying restrictions:", sargan_row]
        if self.basmann is not None:
            basmann_row = (
                f"{'Basmann chi2(' + f'{self.df:g}' + ')':<22}"
                f" = {self.basmann:>8.4f}  (p = {self.basmann_pvalue:.4f})"
            )
            lines.append(basmann_row)
        return chr(10).join(lines)

    def to_dict(self) -> dict[str, float]:
        """The ``r()`` scalars Stata returns."""

        if self.test == "Hansen J":
            return {"HansenJ": self.statistic, "p_HansenJ": self.pvalue}
        if self.test == "Wooldridge score":
            # ``OverID2sls`` returns e(score) under a robust/HAC VCE.
            return {"score": self.statistic, "p_score": self.pvalue}
        payload = {"sargan": self.statistic, "p_sargan": self.pvalue}
        if self.basmann is not None:
            payload["basmann"] = self.basmann
            payload["p_basmann"] = float(self.basmann_pvalue)
        return payload


def _require_iv(result: Any) -> dict[str, Any]:
    structure = getattr(result, "iv_structure", None)
    if structure is None:
        raise NotImplementedError(
            "estat overid is only defined after an instrumental-variables fit: "
            "Stata reaches it through ivregress_estat and it needs the split "
            "between included exogenous regressors and excluded instruments; "
            f"got {getattr(result, 'model_name', 'an unnamed model')!r} without "
            "that structure"
        )
    return structure


def _instrument_names(structure: dict[str, Any]) -> list[str]:
    return [
        str(name) for name in structure.get("instrument_names", [])
    ]


def _wooldridge_score(
    structure: dict[str, Any], resid: np.ndarray, lags: int = 0
) -> float:
    """Wooldridge's score statistic for the overidentifying restrictions.

    Transcribed from ``ivregress_estat.ado:184-260``.  With ``resids`` the IV
    residuals:

    1. project each endogenous regressor on **all** instruments (``allinst`` =
       included exogenous + excluded instruments) and keep the fitted values
       ``endoghat``;
    2. for each excluded instrument, regress it on ``endoghat`` plus the
       included exogenous regressors, take the residual ``r_j``, and form
       ``khat_j = r_j * resids``;
    3. regress a vector of ones on ``khat`` **without a constant**; the statistic
       is ``N - rss``.

    With HAC and ``lags > 0`` the ado prewhitens the ``khat`` series by
    regressing them on their own lags (``mvreg khat = L(1/lags).khat, nocons``)
    and keeps the residuals before step 3.  ``lags == 0`` (and every robust
    VCE) skips that.

    The number of ``khat`` columns used is exactly the number of overidentifying
    restrictions; the ado stops after ``noverid`` instruments because "which Q
    excluded exog. vars is irrelevant -- statistic is the same"
    (``ivregress_estat.ado:199-201``).
    """

    instruments = np.asarray(structure["instruments"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    exog = np.asarray(structure["exog"], dtype=float)
    exog_names = list(structure["exog_names"])

    n = resid.size
    has_constant = any(_is_constant(name) for name in exog_names)
    included = [j for j, name in enumerate(exog_names) if not _is_constant(name)]

    def _allinstruments() -> np.ndarray:
        parts = []
        if has_constant:
            parts.append(np.ones(n))
        if included:
            parts.append(exog[:, included])
        parts.append(instruments)
        return np.column_stack(parts)

    all_inst = _allinstruments()

    # 1. endoghat: each endogenous regressor projected on all instruments.
    endoghat = np.empty((n, endog.shape[1]))
    for column in range(endog.shape[1]):
        beta, *_ = np.linalg.lstsq(all_inst, endog[:, column], rcond=None)
        endoghat[:, column] = all_inst @ beta

    # 2. khat_j = residual of (excluded instrument j on endoghat + included
    #    exogenous) times the IV residual.
    restrictions = instruments.shape[1] - endog.shape[1]
    conditioning_parts = [endoghat]
    if has_constant:
        conditioning_parts.insert(0, np.ones(n))
    if included:
        conditioning_parts.append(exog[:, included])
    conditioning = np.column_stack(conditioning_parts)

    khat_columns: list[np.ndarray] = []
    for column in range(instruments.shape[1]):
        if len(khat_columns) >= restrictions:
            break
        beta, *_ = np.linalg.lstsq(
            conditioning, instruments[:, column], rcond=None
        )
        residual = instruments[:, column] - conditioning @ beta
        khat_columns.append(residual * resid)
    khat = np.column_stack(khat_columns)

    # 3. Prewhitening for HAC with lags > 0.
    if lags > 0:
        # ``mvreg khat = L(1/lags).khat, nocons`` drops the first ``lags``
        # observations, because those rows have no lagged values.  The sample
        # used by step 3 is therefore N - lags, not N -- and the earlier version
        # of this code padded with zeros instead, which is structurally wrong:
        # a zero row is not a zero residual, it is a row where no regression
        # happened.  On a 120-row probe that inflated the statistic by 4.0%
        # (0.63700 vs 0.61223) while still looking like a plausible number.
        rows = khat.shape[0] - lags
        if rows <= 0:  # pragma: no cover - too short to prewhiten
            return float("nan")
        lagged = np.column_stack(
            [khat[lags - 1 - lag : khat.shape[0] - 1 - lag] for lag in range(lags)]
        )
        target = khat[lags:]
        prewhitened = np.empty_like(target)
        # ``mvreg`` fits every equation on the same regressor matrix; with
        # ``nocons`` and no cross-equation restriction that is exactly one OLS
        # per column on the SAME rows.
        for column in range(khat.shape[1]):
            beta, *_ = np.linalg.lstsq(lagged, target[:, column], rcond=None)
            prewhitened[:, column] = target[:, column] - lagged @ beta
        khat = prewhitened

    # Regress ones on khat WITHOUT a constant: score = N - rss.
    # ``e(N)`` here is the sample the step-3 regression uses.
    ones = np.ones(khat.shape[0])
    beta, *_ = np.linalg.lstsq(khat, ones, rcond=None)
    residual = ones - khat @ beta
    rss = float(residual @ residual)
    return float(khat.shape[0] - rss)


def estat_overid(
    result: Any, *, lags: int | None = None, forcenonrobust: bool = False
) -> OverIdResult:
    """Stata's ``estat overid``.

    Parameters
    ----------
    lags:
        HAC truncation for the prewhitening step.  Stata's ``OverID2sls``
        (``ivregress_estat.ado:84-95``) defaults this to **1** for a HAC VCE and
        rejects a negative value; ``lags()`` is only meaningful with HAC, which
        the ado enforces with "lags() only valid with models with HAC VCEs".
        ``None`` selects Stata's default: 1 under HAC, 0 otherwise.
    forcenonrobust:
        Stata's ``forcenonrobust`` option
        (``ivregress_postestimation.sthlp:443-451``): perform the Sargan and Basmann
        tests after 2SLS or LIML even though a robust VCE was used.  Both assume
        independent, identically distributed normal errors, which is why a robust VCE
        replaces them with Wooldridge's score test by default.

    Raises
    ------
    NotImplementedError
        For a non-IV result.
    ValueError
        When the model is exactly identified, when the VCE is clustered
        (``ivregress_estat.ado:290-296`` offers no test there), when ``lags`` is
        negative, or when ``lags`` is given for a non-HAC VCE.
    """

    structure = _require_iv(result)
    model_name = str(getattr(result, "model_name", "")).strip().lower()

    instruments = np.asarray(structure["instruments"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    exog = np.asarray(structure["exog"], dtype=float)
    exog_names = list(structure["exog_names"])

    n_instruments = instruments.shape[1]
    n_endog = endog.shape[1]
    restrictions = n_instruments - n_endog

    if restrictions <= 0:
        raise ValueError(
            "estat overid: the model is exactly identified -- "
            f"{n_instruments} excluded instrument(s) for {n_endog} endogenous "
            "regressor(s) leaves no overidentifying restriction to test"
        )

    # GMM reports Hansen's J, which the estimator already stores.
    if "gmm" in model_name:
        try:
            j_stat = float(result.e["J"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "estat overid: this GMM result records no e(J), so Hansen's J "
                "cannot be reported"
            ) from exc
        return OverIdResult(
            statistic=j_stat,
            df=float(restrictions),
            pvalue=float(stats.chi2.sf(j_stat, restrictions)),
            test="Hansen J",
        )

    cov_type = str(getattr(result, "cov_type", "nonrobust")).strip().lower()
    resid = np.asarray(result.resid, dtype=float).reshape(-1)
    is_clustered = "cluster" in cov_type
    is_hac = "hac" in cov_type or "newey" in cov_type
    is_robust = cov_type in ("hc0", "hc1", "hc2", "hc3", "robust")
    is_liml = "liml" in model_name

    if (
        is_liml
        and (is_robust or is_hac or is_clustered)
        and lags is None
        and not forcenonrobust
    ):
        # ``ivregress_estat.ado:345-357`` (``OverIDliml``):
        #     if "`e(vcetype)'" != "" & "`forcenonrobust'" == "" {
        #         di as error "tests not available after LIML estimation with
        #                       robust VCE"
        #         di as error "use {cmd:forcenonrobust} to override"
        #
        # LIML has no robust/HAC overidentification test at all -- unlike 2SLS,
        # which switches to the Wooldridge score.  Reporting that score for LIML
        # would apply a 2SLS bread (``Xhat'Xhat``) to a k-class fit whose bread is
        # ``(1-kappa)X'X + kappa X'P_Z X``.
        raise ValueError(
            "estat overid: Stata does not report an overidentification test "
            f"after LIML estimation with a {cov_type} VCE "
            "(ivregress_estat.ado:354-357: 'tests not available after LIML "
            "estimation with robust VCE').  Refit with an unadjusted VCE for "
            "Sargan/Basmann, or use IV2SLS or IVGMM."
        )

    # ``ivregress_estat.ado:80-83``: "lags() only valid with models with HAC VCEs".
    if lags is not None and not is_hac:
        raise ValueError(
            "lags() only valid with models with HAC VCEs; this result used "
            f"{cov_type or 'the default nonrobust VCE'}"
        )
    if lags is not None and int(lags) < 0:
        raise ValueError("lags() must be nonnegative")

    if is_clustered:
        # ``ivregress_estat.ado:290-296`` refuses the robust test after
        # ``vce(cluster)``: "robust tests of overidentifying restrictions after
        # 2SLS are not available with clustered errors".
        raise ValueError(
            "estat overid: Stata does not report an overidentification test "
            f"after a clustered VCE (this result used {cov_type})"
        )

    if cov_type not in ("nonrobust", "ols", "") and not forcenonrobust:
        # Robust or HAC: Sargan and Basmann are left missing and the Wooldridge
        # score test is reported instead (``ivregress_estat.ado:175-261``).
        # ``ivregress_estat.ado:87-95``: a HAC VCE defaults lags to 1, while a
        # plain robust VCE never prewhitens (the ado's condition is
        # ``vcetype == "Robust"`` OR ``HAC & lags == 0``).
        #
        # ``forcenonrobust`` (``ivregress_postestimation.sthlp:443-451``) skips
        # this branch: the user has asked for Sargan/Basmann anyway, accepting
        # that their critical values assume i.i.d. normal errors.
        if lags is None:
            effective_lags = 1 if is_hac else 0
        else:
            effective_lags = int(lags)
        score = _wooldridge_score(structure, resid, lags=effective_lags)
        return OverIdResult(
            statistic=score,
            df=float(restrictions),
            pvalue=float(stats.chi2.sf(score, restrictions)),
            test="Wooldridge score",
        )
    n = resid.size
    rss_main = float(resid @ resid)
    if rss_main <= 0.0:
        raise ValueError(
            "estat overid: the main equation fits exactly, so the Sargan "
            "statistic is undefined"
        )

    # Auxiliary regression: the IV residuals on **all** instruments, which is
    # ``e(insts)`` in the ado -- included exogenous regressors *plus* excluded
    # instruments, with the constant supplied by ``_regress`` itself (so it is
    # not duplicated when ``exog`` already carries one).
    #
    # This detail is decisive: regressing on the excluded instruments alone
    # gives a different residual sum of squares and therefore a different
    # statistic (0.178 vs 0.435 on the auto data).  Note also why the ado's
    # ``N*(1 - rss_aux/rss_main)`` is legitimate -- ``rss_main`` here *is* the
    # centred total sum of squares of the residual series, so the expression is
    # exactly ``N * R2`` of the auxiliary regression.
    has_constant = any(_is_constant(name) for name in exog_names)
    included = [
        j for j, name in enumerate(exog_names) if not _is_constant(name)
    ]
    design_parts = []
    if has_constant:
        design_parts.append(np.ones(n))
    if included:
        design_parts.append(exog[:, included])
    design_parts.append(instruments)
    design = np.column_stack(design_parts)
    coefficients, *_ = np.linalg.lstsq(design, resid, rcond=None)
    aux_resid = resid - design @ coefficients
    rss_aux = float(aux_resid @ aux_resid)

    sargan = n * (1.0 - rss_aux / rss_main)
    sargan = max(sargan, 0.0)

    # --- Basmann finite-sample variant ----------------------------------
    # k is the number of coefficients in the AUXILIARY regression, not in the structural
    # equation. ``ivregress_estat.ado:169-174``:
    #
    #     qui _regress `resids' `allinst' if `touse' `wt', `constant'
    #     ...
    #     mat `basmann' = e(b)
    #     _ms_omit_info `basmann'
    #     local k = colsof(`basmann') - `r(k_omit)'
    #     scalar `basmann' = `sargan'*(e(N)-`k')/(e(N) - `sargan')
    #
    # ``allinst`` is the constant plus every instrument, so k = design.shape[1]. Counting
    # the structural equation's parameters instead (constant + exogenous + endogenous = 3
    # here) gave 1.1953 against Stata's 1.1923; the auxiliary regression has four columns
    # and reproduces 1.1923 exactly.
    k = int(design.shape[1])
    denominator = n - sargan
    if denominator > 0.0 and n > k:
        basmann = sargan * (n - k) / denominator
        basmann_p = float(stats.chi2.sf(basmann, restrictions))
    else:
        basmann = None
        basmann_p = None

    return OverIdResult(
        statistic=sargan,
        df=float(restrictions),
        pvalue=float(stats.chi2.sf(sargan, restrictions)),
        test="Sargan",
        basmann=basmann,
        basmann_pvalue=basmann_p,
    )
