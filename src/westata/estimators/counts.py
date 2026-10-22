"""Zero-inflated and truncated count models.

==============================  ==================================
westata                         Stata
==============================  ==================================
``ZeroInflatedPoisson``         ``zip``
``ZeroInflatedNegativeBinomial``  ``zinb``
``TruncatedPoisson``            ``ztp`` / ``tpoisson`` / ``tpois``
``TruncatedNegativeBinomial``   ``tnbreg``
==============================  ==================================

Likelihoods
-----------
Zero-inflated (inflation probability ``pi = logistic(w'g)``, ``f`` the Poisson
or negative-binomial mass)::

    P(y = 0) = pi + (1 - pi) f(0; mu)
    P(y = k) = (1 - pi) f(k; mu),   k > 0

With ``inflate is None`` there is no mixture: ``P(y = k) = f(k; mu)`` and the
model *is* the plain Poisson / negative binomial of
:mod:`westata.estimators.discrete`.  That degenerate case is the strongest
available correctness check and is covered by a regression test.

Truncated counts, observed only for integers in ``[lower, upper]``::

    P(y = k | lower <= k <= upper) = f(k; mu) / (F(upper) - F(lower - 1))

Zero truncation (``ztp`` / ``tnbreg``) is the special case ``lower = 1``.

Why not wrap statsmodels?
-------------------------
statsmodels offers ``ZeroInflatedPoisson``/``ZeroInflatedNegativeBinomialP``,
but its inflation equation is written with the opposite sign convention from
Stata's ``inflate()`` and its finite-difference Hessian is not what Stata
reports; it has no truncated-count model at all.  Wrapping would therefore
leave one convention per family and no way to test the likelihood itself, so
the likelihoods here are written out explicitly and verified two ways: against
a hand-written ``sum(log ...)`` in the tests, and against captured StataNow
19.5 output in ``tests/conformance/stata_fixtures/p6p7_outputs.txt``.

What *is* reused is all the surrounding machinery: these classes subclass
:class:`westata.estimators.discrete._DiscreteBase`, so parameter naming,
``cov_type`` dispatch, the convergence policy (``strict=True`` raises
:class:`~westata.errors.ConvergenceError`) and the ``StataResults`` assembly are
shared with ``poisson``/``nbreg``.  Only the likelihood, the start values and
the constant-only baseline are new.

Stata reporting notes
---------------------
* ``e(ll_0)`` is Stata's "constant-only model": the *main* equation reduced to a
  constant, with the inflation equation (and the NB dispersion) left as
  specified.  This matters: ``e(chi2)``, ``e(p)`` and ``e(r2_p)`` are all
  relative to it, and using ``mean(y)`` as the null instead makes ``zip``
  report chi2 657.68 where Stata reports 248.04.
* ``e(df_m)`` counts only the main-equation regressors beyond the constant, so
  ``zip y x1, inflate(x1)`` prints ``LR chi2(1)`` even though it estimates four
  parameters (``e(rank)`` is 4).
* The NB dispersion is parameterised as ``alpha`` with ``Var = mu +
  alpha*mu^2`` (Stata's NB2) and estimated on the log scale, so the reported
  coefficient is ``/lnalpha`` exactly as ``nbreg``/``zinb``/``tnbreg`` print it;
  ``e(alpha)`` is the exponentiated value.
* Robust/cluster VCEs are passed through to statsmodels' sandwich and are *not*
  yet verified against Stata ``vce(robust)``; the default observed-information
  covariance is verified (ztp: .0350626/.04102 vs Stata .0350626/.04102).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from statsmodels.base.model import (
    GenericLikelihoodModel,
    GenericLikelihoodModelResults,
)

from westata.errors import ConsistencyError
from westata.estimators.discrete import _DiscreteBase
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = [
    "TruncatedNegativeBinomial",
    "TruncatedPoisson",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
]

#: Largest linear predictor fed through ``exp``.  Beyond this the density is
#: numerically zero anyway, and clipping keeps ``mu`` finite so the optimiser
#: sees ``-inf`` instead of NaN when it wanders.
_ETA_CLIP = 700.0


# ----------------------------------------------------------------- mass functions


def _poisson_logpmf(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Poisson log mass, evaluated without leaving ``[0, 1]`` probabilities."""

    return y * np.log(mu) - mu - gammaln(y + 1.0)


def _nb_terms(mu: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """``(size, p)`` of the NB2 mass written as ``nbinom(size, p)``.

    Stata's ``alpha`` satisfies ``Var = mu + alpha*mu^2``, i.e.
    ``size = 1/alpha`` and ``p = size / (size + mu)``.
    """

    size = 1.0 / max(alpha, 1e-12)
    return size, size / (size + mu)


def _nb_logpmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    size, p = _nb_terms(mu, alpha)
    return (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * np.log(p)
        + y * np.log(np.maximum(1.0 - p, 1e-300))
    )


def _nb_cdf(k: float, mu: np.ndarray, alpha: float) -> np.ndarray:
    size, p = _nb_terms(mu, alpha)
    return stats.nbinom.cdf(k, size, p)


def _nb_logsf(k: float, mu: np.ndarray, alpha: float) -> np.ndarray:
    size, p = _nb_terms(mu, alpha)
    return stats.nbinom.logsf(k, size, p)


# ---------------------------------------------------------- statsmodels backend


class _CountResults(GenericLikelihoodModelResults):
    """Results object the generic likelihood needs to look like a real fit.

    ``GenericLikelihoodModelResults`` already supplies ``params``, ``llf``,
    ``aic``, ``bic``, ``cov_params`` and ``predict``; the two pieces the
    :class:`._DiscreteBase` plumbing expects and it does not provide are
    ``resid_response`` (response minus fitted) and ``llnull`` (Stata's
    constant-only baseline, which the model knows how to compute).
    """

    @property
    def resid_response(self) -> np.ndarray:
        return np.asarray(self.model.endog, dtype=float) - np.asarray(
            self.predict(), dtype=float
        )

    @property
    def llnull(self) -> float:
        return self.model.null_loglike()


class _CountLikelihood(GenericLikelihoodModel):
    """One explicit count likelihood, optimised by statsmodels' machinery.

    Parameters are ordered ``[beta, gamma, lnalpha]``, where ``gamma`` is
    present only when an inflation equation was given and ``lnalpha`` only for
    a negative binomial.  Keeping the dispersion on the log scale (as ``nbreg``
    reports it) means no delta-method rescaling is needed afterwards, unlike
    :class:`westata.estimators.discrete.NegativeBinomial`.

    The truncation/zero-inflation switch is explicit rather than implied: the
    two are never combined, but the branch order keeps that obvious.
    """

    results_class = _CountResults

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        family: str = "poisson",
        inflate_exog: Any = None,
        lower: float | None = None,
        upper: float | None = None,
        **kwargs: Any,
    ) -> None:
        if family not in ("poisson", "nb"):
            raise ValueError(f"unknown count family {family!r}")
        self.family = family
        self._inflate_exog = (
            None if inflate_exog is None else np.asarray(inflate_exog, dtype=float)
        )
        self.lower = lower
        self.upper = upper
        self._null_loglike: float | None = None

        n_main = np.asarray(exog, dtype=float).shape[1]
        n_inflate = 0 if self._inflate_exog is None else self._inflate_exog.shape[1]
        n_extra = n_inflate + (1 if family == "nb" else 0)

        super().__init__(endog, exog, **kwargs)

        if n_extra:
            # statsmodels sizes its parameter vector from ``exog_names``; the
            # real names are supplied by the westata wrapper, these are only
            # placeholders that keep the generic model consistent.
            self.exog_names.extend([f"param{i + 1}" for i in range(n_extra)])
            self.k_extra = n_extra
            self.nparams = n_main + n_extra

        # Keep statsmodels' own bookkeeping self-consistent so its results
        # constructor does not emit "df_model/df_resid differs from k_params"
        # warnings.  These values are *not* what westata reports: ``_DiscreteBase``
        # supplies ``e(df_m)``/``df_model`` from Stata's convention (main-equation
        # regressors beyond the constant, excluding inflation and dispersion).
        self.df_model = float(n_main - self.k_constant)
        self.df_resid = float(np.asarray(endog).shape[0] - self.nparams)

    # ------------------------------------------------------------ parameters
    @property
    def start_params(self) -> np.ndarray:
        """Start from a plain Poisson / log-OLS fit, as Stata does."""

        log_y = np.log(np.maximum(np.asarray(self.endog, dtype=float), 0.5))
        beta0, *_ = np.linalg.lstsq(self.exog, log_y, rcond=None)

        parts = [np.asarray(beta0, dtype=float)]
        if self._inflate_exog is not None:
            share_zero = float(np.mean(np.asarray(self.endog, dtype=float) == 0.0))
            share_zero = min(max(share_zero, 1e-3), 0.95)
            gamma0 = np.zeros(self._inflate_exog.shape[1])
            gamma0[0] = np.log(share_zero / (1.0 - share_zero))
            parts.append(gamma0)
        if self.family == "nb":
            counts = np.asarray(self.endog, dtype=float)
            mean = max(float(np.mean(counts)), 1e-6)
            variance = float(np.var(counts))
            alpha0 = max((variance - mean) / max(mean**2, 1e-12), 0.1)
            parts.append(np.array([np.log(alpha0)]))
        return np.concatenate(parts)

    def _split(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, float | None]:
        """Unpack ``[beta, gamma, lnalpha]`` into mean, pi-index and alpha."""

        params = np.asarray(params, dtype=float).reshape(-1)
        k = self.exog.shape[1]
        eta = np.clip(self.exog @ params[:k], -_ETA_CLIP, _ETA_CLIP)
        mu = np.exp(eta)

        gamma = None
        if self._inflate_exog is not None:
            m = self._inflate_exog.shape[1]
            gamma = self._inflate_exog @ params[k : k + m]
        alpha = None
        if self.family == "nb":
            alpha = float(np.exp(np.clip(params[-1], -_ETA_CLIP, _ETA_CLIP)))
        return mu, gamma, alpha

    # ------------------------------------------------------------ likelihood
    def _log_mass(self, mu: np.ndarray, alpha: float | None) -> np.ndarray:
        if alpha is None:
            return _poisson_logpmf(np.asarray(self.endog, dtype=float), mu)
        return _nb_logpmf(np.asarray(self.endog, dtype=float), mu, alpha)

    def _log_truncation(self, mu: np.ndarray, alpha: float | None) -> np.ndarray:
        """``log(F(upper) - F(lower - 1))``, the truncation normaliser."""

        assert self.lower is not None
        below = self.lower - 1.0
        if self.upper is None:
            if alpha is None:
                return stats.poisson.logsf(below, mu)
            return _nb_logsf(below, mu, alpha)
        if alpha is None:
            upper_cdf = stats.poisson.cdf(self.upper, mu)
            lower_cdf = stats.poisson.cdf(below, mu)
        else:
            upper_cdf = _nb_cdf(self.upper, mu, alpha)
            lower_cdf = _nb_cdf(below, mu, alpha)
        return np.log(np.maximum(upper_cdf - lower_cdf, 1e-300))

    def loglikeobs(self, params: np.ndarray) -> np.ndarray:
        mu, gamma, alpha = self._split(params)
        log_f = self._log_mass(mu, alpha)

        if self.lower is not None:
            return log_f - self._log_truncation(mu, alpha)

        if gamma is None:
            return log_f

        # Zero-inflated: P(0) = pi + (1-pi) f(0), P(k>0) = (1-pi) f(k).
        # Both logarithms are taken in log-space (log sigmoid / log 1-sigmoid)
        # so a near-certain structural zero cannot underflow to log(0).
        gamma = np.clip(gamma, -_ETA_CLIP, _ETA_CLIP)
        log_pi = -np.logaddexp(0.0, -gamma)
        log_one_minus = -np.logaddexp(0.0, gamma)
        log_zero = np.logaddexp(log_pi, log_one_minus + self._zero_log_mass(mu, alpha))
        y = np.asarray(self.endog, dtype=float)
        return np.where(y == 0.0, log_zero, log_one_minus + log_f)

    def _zero_log_mass(self, mu: np.ndarray, alpha: float | None) -> np.ndarray:
        """``log f(0; mu)`` -the mass the mixture replaces at ``y = 0``."""

        zeros = np.zeros_like(mu)
        if alpha is None:
            return _poisson_logpmf(zeros, mu)
        return _nb_logpmf(zeros, mu, alpha)

    # --------------------------------------------------------------- predict
    def predict(self, params: np.ndarray, exog: Any = None, **kwargs: Any) -> np.ndarray:
        """Conditional mean of the observed count (Stata ``predict, n``)."""

        params = np.asarray(params, dtype=float).reshape(-1)
        if exog is None:
            exog = self.exog
        exog = np.asarray(exog, dtype=float)
        eta = np.clip(exog @ params[: exog.shape[1]], -_ETA_CLIP, _ETA_CLIP)
        mu = np.exp(eta)

        if self.lower is not None:
            return self._truncated_mean(mu, params)

        if self._inflate_exog is None:
            return mu
        m = self._inflate_exog.shape[1]
        k = self.exog.shape[1]
        gamma = self._inflate_exog @ params[k : k + m]
        pi = 1.0 / (1.0 + np.exp(-np.clip(gamma, -_ETA_CLIP, _ETA_CLIP)))
        return (1.0 - pi) * mu

    def _truncated_mean(self, mu: np.ndarray, params: np.ndarray) -> np.ndarray:
        """``E[Y | lower <= Y <= upper]`` using ``E[Y; sel] = mu (F(u-1)-F(l-2))``."""

        assert self.lower is not None
        alpha = None
        if self.family == "nb":
            alpha = float(np.exp(np.clip(params[-1], -_ETA_CLIP, _ETA_CLIP)))

        denominator = np.exp(self._log_truncation(mu, alpha))
        below_two = self.lower - 2.0
        if self.upper is None:
            numerator = mu * (
                stats.poisson.sf(below_two, mu)
                if alpha is None
                else _nb_logsf(below_two, mu, alpha)
            )
        else:
            if alpha is None:
                numerator = mu * (
                    stats.poisson.cdf(self.upper - 1.0, mu)
                    - stats.poisson.cdf(below_two, mu)
                )
            else:
                numerator = mu * (
                    _nb_cdf(self.upper - 1.0, mu, alpha) - _nb_cdf(below_two, mu, alpha)
                )
        return numerator / np.maximum(denominator, 1e-300)

    # ------------------------------------------------------- constant-only ll
    def null_loglike(self) -> float:
        """Log-likelihood of Stata's "constant-only model" for this family.

        Stata's baseline is *not* uniform across these commands, and using one
        rule everywhere silently breaks ``e(chi2)``/``e(r2_p)``:

        * ``zip``/``zinb``/``tnbreg`` print a "Fitting constant-only model" block
          and use its *maximised* log-likelihood, i.e. the model refitted with
          the main equation reduced to a constant but the inflation equation
          kept.  For the ``zip`` fixture that is -1824.7258 (chi2 248.04) and
          for ``tnbreg`` -1407.598 (chi2 171.80).
        * ``ztp`` prints no such block and measures chi2 against the
          ``mean(y)`` single-parameter null *with the truncation applied*:
          -1051.8155 (chi2 270.42).  Refitting a truncated constant-only model
          instead gives -1039.73 and chi2 246.25, which matches no Stata output.

        The evidence for the split is in ``p6p7_outputs.txt``: only ``ztp``
        lacks a constant-only block, and only ``ztp``'s chi2 is reproducible
        from ``mean(y)``.
        """

        if self._null_loglike is not None:
            return self._null_loglike

        y = np.asarray(self.endog, dtype=float)
        n = y.shape[0]
        if self.lower is not None and self.family == "poisson":
            # ztp/tpoisson: single-parameter null at mu = mean(y), truncation kept.
            mu = np.full(n, max(float(np.mean(y)), 1e-6))
            value = float(
                np.sum(_poisson_logpmf(y, mu) - self._log_truncation(mu, None))
            )
        else:
            null = type(self)(
                self.endog,
                np.ones((n, 1)),
                family=self.family,
                inflate_exog=self._inflate_exog,
                lower=self.lower,
                upper=self.upper,
            )
            try:
                fitted = null.fit(
                    method="bfgs", maxiter=2000, disp=0, start_params=null.start_params
                )
                value = float(fitted.llf)
            except Exception:  # noqa: BLE001 - a failed null fit only costs e(ll_0)
                value = float("nan")
        self._null_loglike = value
        return value


# ---------------------------------------------------------------- westata layer


class _CountBase(_DiscreteBase):
    """Shared Stata-facing behaviour for the count extensions."""

    _family = "poisson"
    _accepts_hc_kwds = False

    def _main_count(self) -> int:
        """Regressors in the main equation (including the constant)."""

        return int(self._X.shape[1])

    def _model_df(self) -> int:
        """``e(df_m)``: main-equation regressors *beyond* the constant.

        Stata prints ``LR chi2(df_m)`` for these models and the inflation
        equation does not contribute to it: ``zip y x1, inflate(x1)`` reports
        ``LR chi2(1)`` with ``e(rank) = 4``.
        """

        return max(self._main_count() - 1, 0)

    # -- backend -----------------------------------------------------------
    def _make_sm_model(self) -> Any:
        return _CountLikelihood(
            self._y,
            self._X,
            family=self._family,
            inflate_exog=getattr(self, "_inflate_exog", None),
            lower=getattr(self, "_lower", None),
            upper=getattr(self, "_upper", None),
        )

    # -- helpers used by the tests and by postestimation -------------------
    def _loglike(self, theta: np.ndarray) -> float:
        """Log-likelihood at ``theta`` (kept public-ish for verification)."""

        return float(np.sum(self._make_sm_model().loglikeobs(np.asarray(theta, dtype=float))))

    def start_params(self) -> np.ndarray:
        return self._make_sm_model().start_params

    # -- Stata e() ---------------------------------------------------------
    def _stata_scalars(
        self, result: Any, params: np.ndarray, llf: float, nobs: int
    ) -> dict[str, Any]:
        ll0 = getattr(result, "llnull", None)
        ll_null = float(ll0) if ll0 is not None else float("nan")
        df_m = self._model_df()

        if np.isfinite(ll_null):
            chi2 = 2.0 * (llf - ll_null)
            p_value = float(stats.chi2.sf(chi2, df_m)) if df_m > 0 else float("nan")
            pseudo_r2 = 1.0 - llf / ll_null if ll_null != 0 else float("nan")
        else:
            chi2 = float("nan")
            p_value = float("nan")
            pseudo_r2 = float("nan")

        scalars: dict[str, Any] = {
            "N": nobs,
            "df_m": df_m,
            "rank": int(params.size),
            "ll": float(llf),
            "ll_0": ll_null,
            "chi2": chi2,
            "chi2type": "LR",
            "p": p_value,
            "r2_p": pseudo_r2,
            "aic": float(getattr(result, "aic", float("nan"))),
            "bic": float(getattr(result, "bic", float("nan"))),
        }
        if self._family == "nb":
            lnalpha = float(params[-1])
            scalars["lnalpha"] = lnalpha
            scalars["alpha"] = float(np.exp(np.clip(lnalpha, -_ETA_CLIP, _ETA_CLIP)))
        return scalars

    def _extra_scalars(self) -> dict[str, Any]:
        """Model-specific extra ``e()`` scalars, merged in by subclasses."""

        return {}

    # -- fit ---------------------------------------------------------------
    def _optimizer_tolerance(self, tol: float) -> float:
        """Relax ``tol`` for statsmodels' ``newton``.

        ``_DiscreteBase.fit`` forwards ``tol`` as the Newton convergence
        criterion, but statsmodels' Newton stops on the gradient norm, which is
        not on the log-likelihood scale: 1e-10 is below what finite-difference
        Hessians can resolve, so the line search makes no progress and every
        fit burns all ``maxiter`` iterations while still being flagged
        non-converged.  Measured on ``tpoisson`` (upper=4) that meant 2000
        iterations and a spurious ConvergenceError warning at ``tol=1e-10``
        versus 5 iterations and the identical log-likelihood at 1e-8.  The
        estimates are unchanged, so the looser criterion costs nothing.
        """

        return max(float(tol), 1e-8)

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
        """Maximise the count likelihood.

        ``_DiscreteBase.fit`` reuses every part of the estimation machinery but
        reports ``df_model = len(params) - 1``, which counts inflation and
        dispersion parameters.  Stata's ``e(df_m)`` does not, so the container
        is restamped here with the corrected value rather than reimplementing
        the fit.
        """

        if self._optimizer == "newton":
            tol = self._optimizer_tolerance(tol)
        result = super().fit(
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            ssc=ssc,
            maxiter=maxiter,
            tol=tol,
            strict=strict,
        )
        extras = self._extra_scalars()
        df_m = float(self._model_df())
        if not extras and df_m == result.df_model:
            return result

        e = dict(result.e)
        e.update(extras)
        e["df_m"] = df_m
        return StataResults(
            params=result.params.to_numpy(),
            param_names=result.param_names,
            cov=result.cov_params(),
            resid=result.resid,
            fittedvalues=result.fittedvalues,
            nobs=result.nobs,
            df_resid=result.df_resid,
            df_model=df_m,
            dist=result.dist,
            model_name=result.model_name,
            cov_type=result.cov_type,
            e=e,
            design_matrix=result.design_matrix,
            model_data=result.model_data,
        )


class _ZeroInflatedBase(_CountBase):
    """Shared ``from_formula`` handling for the two zero-inflated models."""

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        inflate_exog: Any = None,
        inflate_names: list[str] | None = None,
        param_names: list[str] | None = None,
    ) -> None:
        super().__init__(endog, exog, param_names=param_names)
        self._inflate_exog: np.ndarray | None = (
            None if inflate_exog is None else np.asarray(inflate_exog, dtype=float)
        )
        self._inflate_names: list[str] = list(inflate_names or [])
        if (
            self._inflate_exog is not None
            and len(self._inflate_names) != self._inflate_exog.shape[1]
        ):
            self._inflate_names = [f"z{i}" for i in range(self._inflate_exog.shape[1])]

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        inflate: str | None = None,
        subset: Any = None,
    ) -> "_ZeroInflatedBase":
        """Build the model from a formula (Stata ``zip y x, inflate(z)``).

        ``inflate=None`` leaves the mixture switched off entirely, which is the
        pure Poisson/NB model.  ``inflate="_cons"`` is Stata's spelling for a
        constant-only inflation equation and is translated to ``~ 1`` so patsy
        does not go looking for a column named ``_cons``.
        """

        model = super().from_formula(formula, data, subset=subset)
        if inflate is not None:
            model._set_inflate(formula, str(inflate).strip(), model._model_data)
        return model

    def _set_inflate(
        self, formula: str, inflate: str, frame: pd.DataFrame | None
    ) -> None:
        if frame is None:  # pragma: no cover - from_formula always sets it
            raise ConsistencyError(
                f"{self._stata_command}: the estimation sample was not retained"
            )
        lhs = formula.split("~", 1)[0].strip()
        inflate_formula = f"{lhs} ~ 1" if inflate in ("_cons", "1", "constant") else f"{lhs} ~ {inflate}"
        Z, names, index = build_design(inflate_formula, frame)
        if len(index) != len(frame):
            raise ConsistencyError(
                f"{self._stata_command}: the inflation equation drops "
                f"{len(frame) - len(index)} observations that the count equation "
                "keeps; both equations must describe the same sample"
            )
        self._inflate_exog = Z
        self._inflate_names = list(names)

    def _extra_param_names(self, count: int) -> list[str]:
        names = [f"inflate:{name}" for name in self._inflate_names]
        if self._family == "nb":
            names.append("/lnalpha")
        if len(names) != count:
            names = names[:count]
            names += [f"param{i + 1}" for i in range(count - len(names))]
        return names

    def _extra_scalars(self) -> dict[str, Any]:
        y = self._y
        scalars: dict[str, Any] = {
            "n_zero": int(np.sum(y == 0.0)),
            "n_nonzero": int(np.sum(y != 0.0)),
        }
        if self._inflate_exog is None:
            scalars["inflate"] = "none"
        return scalars


class ZeroInflatedPoisson(_ZeroInflatedBase):
    """Zero-inflated Poisson (Stata ``zip``).

    With ``inflate is None`` the model is the plain Poisson regression, fit by
    the same likelihood, so its coefficients agree with
    :class:`westata.estimators.discrete.Poisson` to optimiser tolerance.
    """

    _stata_command = "zip"
    _family = "poisson"
    #: Stata's ``ml`` default.  Newton converges in a handful of steps from the
    #: Poisson start and reproduces the Stata coefficients exactly.
    _optimizer = "newton"


class ZeroInflatedNegativeBinomial(_ZeroInflatedBase):
    """Zero-inflated negative binomial, NB2 (Stata ``zinb``).

    ``alpha`` is estimated as ``/lnalpha``.  Quasi-Newton is used because the
    dispersion parameter makes the Newton step diverge from the Poisson start,
    the same reason ``nbreg`` uses ``bfgs`` here.
    """

    _stata_command = "zinb"
    _family = "nb"
    _optimizer = "bfgs"


class _TruncatedBase(_CountBase):
    """Shared truncation bookkeeping for ``ztp``/``tpoisson``/``tnbreg``."""

    #: Stata command name used in error messages before the limits are known.
    _base_command = "tpoisson"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        lower: float | None = 1,
        upper: float | None = None,
        param_names: list[str] | None = None,
    ) -> None:
        super().__init__(endog, exog, param_names=param_names)
        self._set_limits(lower, upper)

    def _set_limits(self, lower: float | None, upper: float | None) -> None:
        if lower is None:
            lower = 1
        if upper is not None and not float(lower) <= float(upper):
            raise ConsistencyError(
                f"{self._base_command}: the lower truncation point ({lower}) must "
                f"not exceed the upper one ({upper})"
            )
        if np.any(self._y < float(lower)):
            raise ConsistencyError(
                f"{self._base_command}: the data violate the truncation: "
                f"{int(np.sum(self._y < float(lower)))} observation(s) fall below "
                f"the lower limit {lower}. Truncation requires every y >= {lower}"
            )
        if upper is not None and np.any(self._y > float(upper)):
            raise ConsistencyError(
                f"{self._base_command}: the data violate the truncation: "
                f"{int(np.sum(self._y > float(upper)))} observation(s) fall above "
                f"the upper limit {upper}"
            )
        if not np.allclose(self._y, np.round(self._y)):
            raise ConsistencyError(
                f"{self._base_command}: truncated count models require integer "
                "outcomes; the data contain non-integer values"
            )
        self._lower = float(lower)
        self._upper = None if upper is None else float(upper)
        # Stata names the zero-truncated case ``ztp`` and the general case
        # ``tpoisson``; ``tnbreg`` already covers both.
        if self._family == "poisson":
            self._stata_command = (
                "ztp"
                if self._lower == 1.0 and self._upper is None
                else "tpoisson"
            )

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        lower: float | None = 1,
        upper: float | None = None,
        subset: Any = None,
    ) -> "_TruncatedBase":
        """Build from a formula (Stata ``ztp y x``, ``tpoisson y x, ll(1)``).

        This mirrors :meth:`westata.estimators.discrete._DiscreteBase.from_formula`
        but passes the truncation limits into ``__init__``, where the data are
        validated: the design construction itself is the shared
        :func:`westata.formula.factor.build_design`.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"truncated count model: left-hand side must be a plain column "
                f"name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(y, X, lower=lower, upper=upper, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    def _extra_param_names(self, count: int) -> list[str]:
        names = ["/lnalpha"] if self._family == "nb" else []
        return (names + [f"param{i + 1}" for i in range(count)])[:count]

    def _extra_scalars(self) -> dict[str, Any]:
        return {
            "truncation_point": self._lower,
            "lower": self._lower,
            "upper": self._upper,
        }


class TruncatedPoisson(_TruncatedBase):
    """Poisson truncated to ``[lower, upper]`` (Stata ``ztp`` / ``tpoisson``).

    ``lower=1`` with no upper limit is the zero-truncated ``ztp``; any other
    limit is Stata's ``tpoisson``, and ``model_name`` follows suit.
    """

    _family = "poisson"
    _base_command = "ztp"
    _stata_command = "ztp"
    _optimizer = "newton"


class TruncatedNegativeBinomial(_TruncatedBase):
    """Negative binomial truncated to ``[lower, upper]`` (Stata ``tnbreg``)."""

    _family = "nb"
    _base_command = "tnbreg"
    _stata_command = "tnbreg"
    _optimizer = "bfgs"
