"""Discrete-choice and count estimators targeting Stata's ML commands.

=======================  ==================
westata                  Stata
=======================  ==================
``Logit``                ``logit``
``Probit``               ``probit``
``Poisson``              ``poisson``
``NegativeBinomial``     ``nbreg``
``OrderedLogit``         ``ologit``
``OrderedProbit``        ``oprobit``
=======================  ==================

Conformance notes
-----------------
* All of these report **z** statistics in Stata (``P>|z|``), unlike ``regress``,
  so ``StataResults`` is built with ``dist="z"`` and the summary prints
  ``Wald chi2`` rather than an F test.
* ``e(ll)`` is the log-likelihood at the estimates.
* ``e(r2_p)`` is McFadden's pseudo R-squared, which is what Stata prints as
  ``Pseudo R2``.
* Standard errors: Stata's default is the observed-information (Hessian)
  variance, which is statsmodels' ``cov_type="nonrobust"`` for ``Logit`` and
  ``Probit``.  For ``Poisson`` Stata's default is also the observed
  information, and that is the default here too.
* Robust VCEs are delegated to statsmodels' own machinery for these models
  rather than to :func:`westata.vce.compute_cov`: the nonlinear models' robust
  sandwich is built from the score function, not from ``X`` and residuals.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError, ConvergenceError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = [
    "Logit",
    "Probit",
    "Poisson",
    "NegativeBinomial",
    "OrderedLogit",
    "OrderedProbit",
]

_SUPPORTED_COV = ("nonrobust", "HC0", "HC1", "cluster")


class _DiscreteBase:
    """Shared plumbing for statsmodels-backed ML estimators."""

    _stata_command = ""
    _model_name = ""
    #: statsmodels optimiser.  ``newton`` matches Stata's ml default and is
    #: fastest for logit/probit, but ``nbreg`` needs a quasi-Newton method
    #: because its alpha parameter makes the Newton step diverge from a
    #: Poisson start.
    _optimizer = "newton"
    #: Whether ``fit`` accepts ``cov_kwds`` for HC0/HC1 (Poisson does not).
    _accepts_hc_kwds = True

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        param_names: list[str] | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        X = np.asarray(exog, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._X = X
        self._param_names = (
            list(param_names)
            if param_names is not None
            else [f"x{i}" for i in range(X.shape[1])]
        )
        if len(self._param_names) != X.shape[1]:
            raise ValueError(
                f"param_names has {len(self._param_names)} entries but exog has "
                f"{X.shape[1]} columns"
            )
        #: Estimation-sample covariates, retained for margins(over=...).
        self._model_data: pd.DataFrame | None = None

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "_DiscreteBase":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"{cls._stata_command}: left-hand side must be a plain column "
                f"name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(y, X, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    # ------------------------------------------------------------------ fit
    def _make_sm_model(self) -> Any:
        raise NotImplementedError

    def _extra_param_names(self, count: int) -> list[str]:
        """Names for backend parameters beyond the regressors.

        Stata ``nbreg`` calls its dispersion parameter ``/lnalpha``; other
        models should override this rather than emit anonymous names.
        """

        return [f"param{i + 1}" for i in range(count)]

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 500,
        tol: float = 1e-10,
        strict: bool = False,
    ) -> StataResults:
        """Estimate by maximum likelihood.

        Parameters
        ----------
        strict:
            Raise :class:`ConvergenceError` instead of only warning when the
            optimiser fails to converge.
        """

        requested = "nonrobust" if cov_type is None else str(cov_type).strip()
        canonical = requested.lower()
        if canonical not in {value.lower() for value in _SUPPORTED_COV}:
            raise ValueError(
                f"{self._stata_command}: unsupported cov_type {cov_type!r}; "
                f"supported: {', '.join(_SUPPORTED_COV)}"
            )

        model = self._make_sm_model()
        if canonical == "nonrobust":
            sm_cov, sm_kwds = "nonrobust", None
        elif canonical in ("hc0", "hc1"):
            sm_cov = "HC0" if canonical == "hc0" else "HC1"
            # statsmodels is inconsistent here: Poisson's covariance rejects
            # ``cov_kwds`` outright with "heteroscedasticity robust covariance
            # does not use keywords", while the other discrete models require
            # ``use_correction`` for HC0. Pass it only where it is accepted.
            sm_kwds = (
                {"use_correction": canonical == "hc1"} if self._accepts_hc_kwds else None
            )
        else:
            sm_cov, sm_kwds = "cluster", dict(cov_kwds or {})

        fit_kwargs: dict[str, Any] = {
            "method": self._optimizer,
            "maxiter": maxiter,
            "disp": 0,
            "cov_type": sm_cov,
            "use_t": False,
        }
        if self._optimizer == "newton":
            # Every other optimiser rejects ``tol`` (statsmodels warns, and the
            # keyword is ignored).
            fit_kwargs["tol"] = tol
        if sm_kwds is not None:
            fit_kwargs["cov_kwds"] = sm_kwds

        result = model.fit(**fit_kwargs)

        converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
        if not converged:
            import warnings

            message = (
                f"{self._stata_command}: optimiser did not converge after "
                f"{getattr(result, 'mle_retvals', {}).get('iterations', '?')} "
                "iterations; estimates may be unreliable"
            )
            if strict:
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        params = np.asarray(result.params, dtype=float)
        try:
            cov = np.asarray(result.cov_params(), dtype=float)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ConvergenceError(
                f"{self._stata_command}: the optimiser did not produce a usable "
                f"covariance matrix ({exc}). This normally means the model is "
                "not identified (e.g. perfect separation or a collinear "
                "regressor); respecify the model or inspect the data."
            ) from None
        # Keep the design-matrix names from ``from_formula``; statsmodels'
        # ``exog_names`` is ``const``/``x1``/``x2`` for array-built models and
        # would break ``results.test("mpg")``.
        names = list(self._param_names)
        if len(names) != params.size:
            # Extra parameters come from the backend, e.g. nbreg's ``alpha`` or
            # a negative-binomial wrapper. Name them the way Stata does where
            # the position is known, and generically otherwise.
            extras = self._extra_param_names(params.size - len(names))
            names = names + extras
        llf = float(result.llf)
        nobs = int(result.nobs)

        e = self._stata_scalars(result, params, llf, nobs)

        return StataResults(
            params=params,
            param_names=names,
            cov=cov,
            resid=np.asarray(result.resid_response, dtype=float).reshape(-1),
            fittedvalues=np.asarray(result.predict(), dtype=float).reshape(-1),
            nobs=nobs,
            df_resid=float(nobs),
            df_model=float(len(params) - 1),
            dist="z",
            model_name=self._stata_command,
            cov_type=requested,
            e=e,
            design_matrix=(self._X, list(names)),
            model_data=self._model_data,
        )

    # ------------------------------------------------------------- e() scalars
    def _stata_scalars(
        self, result: Any, params: np.ndarray, llf: float, nobs: int
    ) -> dict[str, Any]:
        ll0 = getattr(result, "llnull", None)
        ll_null = float(ll0) if ll0 is not None else float("nan")
        if np.isfinite(ll_null) and ll_null != 0:
            pseudo_r2 = 1.0 - llf / ll_null
        else:
            pseudo_r2 = float("nan")
        k = len(params)
        return {
            "N": nobs,
            "df_m": k - 1,
            "rank": k,
            "ll": llf,
            "ll_0": ll_null,
            "chi2": float(getattr(result, "llr", float("nan"))),
            "p": float(getattr(result, "llr_pvalue", float("nan"))),
            "r2_p": pseudo_r2,
            "aic": float(getattr(result, "aic", float("nan"))),
            "bic": float(getattr(result, "bic", float("nan"))),
        }


class Logit(_DiscreteBase):
    """Logistic regression (Stata ``logit``)."""

    _stata_command = "logit"

    def _make_sm_model(self) -> Any:
        import statsmodels.api as sm

        return sm.Logit(self._y, self._X)


class Probit(_DiscreteBase):
    """Probit regression (Stata ``probit``)."""

    _stata_command = "probit"

    def _make_sm_model(self) -> Any:
        import statsmodels.api as sm

        return sm.Probit(self._y, self._X)


class Poisson(_DiscreteBase):
    """Poisson regression (Stata ``poisson``)."""

    _stata_command = "poisson"
    _accepts_hc_kwds = False

    def _make_sm_model(self) -> Any:
        import statsmodels.api as sm

        return sm.Poisson(self._y, self._X)


class NegativeBinomial(_DiscreteBase):
    """Negative binomial regression (Stata ``nbreg``).

    statsmodels parameterises the dispersion as ``alpha`` (``Var = mu +
    alpha*mu^2``), the same as Stata's default ``nbreg`` output.  ``bfgs`` is
    used rather than Newton: the alpha parameter makes the Newton step diverge
    when started from the Poisson fit, which is statsmodels' own
    ``NegativeBinomial.fit`` default for this reason.
    """

    _stata_command = "nbreg"
    _optimizer = "bfgs"

    def _extra_param_names(self, count: int) -> list[str]:
        # Stata reports nbreg's dispersion as /lnalpha.
        if count == 1:
            return ["/lnalpha"]
        return super()._extra_param_names(count)

    def _make_sm_model(self) -> Any:
        import statsmodels.api as sm

        return sm.NegativeBinomial(self._y, self._X)


class _Ordered(_DiscreteBase):
    """Shared implementation for ``ologit`` / ``oprobit``.

    The ordered model identifies location through its cut points, so
    statsmodels refuses a design matrix containing a constant.  Stata behaves
    equivalently: ``ologit`` reports no ``_cons``, only ``/cut1``, ``/cut2``,
    ...
    """

    _distribution = "probit"

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "_Ordered":
        model = super().from_formula(formula, data, subset=subset)
        names = model._param_names
        if "Intercept" in names:
            position = names.index("Intercept")
            model._X = np.delete(model._X, position, axis=1)
            model._param_names = [n for n in names if n != "Intercept"]
        return model

    def _make_sm_model(self) -> Any:
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        return OrderedModel(self._y, self._X, distr=self._distribution)

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 500,
        tol: float = 1e-10,
        strict: bool = False,
    ) -> StataResults:
        if cov_type not in (None, "nonrobust"):
            raise ValueError(
                f"{self._stata_command}: only cov_type='nonrobust' is "
                "implemented; the ordered-model robust sandwich is not yet "
                "wired up"
            )
        model = self._make_sm_model()
        result = model.fit(method="bfgs", maxiter=maxiter, disp=0)

        params = np.asarray(result.params, dtype=float)
        # statsmodels orders the slope parameters first, then the cut points.
        # Stata prints /cut1, /cut2, ... after the coefficients, so the design
        # names line up positionally and the remaining entries are cuts.
        names = list(self._param_names)
        extra = params.size - len(names)
        if extra < 0:
            raise ConsistencyError(
                f"{self._stata_command}: backend returned fewer parameters "
                f"({params.size}) than regressors ({len(names)})"
            )
        names = names + [f"/cut{i + 1}" for i in range(extra)]

        cov = np.asarray(result.cov_params(), dtype=float)
        llf = float(result.llf)
        nobs = int(result.nobs)

        ll0 = getattr(result, "llnull", None)
        ll_null = float(ll0) if ll0 is not None else float("nan")
        # ``OrderedModel.predict()`` returns a (n_obs, n_categories) matrix of
        # category probabilities; the results container needs one value per
        # observation, so report the expected category index.
        predicted = np.asarray(result.predict(), dtype=float)
        if predicted.ndim == 2 and predicted.shape[0] == nobs:
            categories = np.arange(1, predicted.shape[1] + 1, dtype=float)
            predicted = predicted @ categories
        else:
            predicted = predicted.reshape(-1)
        e = {
            "N": nobs,
            "df_m": params.size - 1,
            "rank": params.size,
            "ll": llf,
            "ll_0": ll_null,
            "chi2": float(getattr(result, "llr", float("nan"))),
            "p": float(getattr(result, "llr_pvalue", float("nan"))),
            "r2_p": (
                1.0 - llf / ll_null if np.isfinite(ll_null) and ll_null else float("nan")
            ),
        }
        return StataResults(
            params=params,
            param_names=names,
            cov=cov,
            resid=np.zeros(nobs),
            fittedvalues=predicted,
            nobs=nobs,
            df_resid=float(nobs),
            df_model=float(params.size - 1),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
        )


class OrderedLogit(_Ordered):
    """Ordered logistic regression (Stata ``ologit``)."""

    _stata_command = "ologit"
    _distribution = "logit"


class OrderedProbit(_Ordered):
    """Ordered probit regression (Stata ``oprobit``)."""

    _stata_command = "oprobit"
    _distribution = "probit"
