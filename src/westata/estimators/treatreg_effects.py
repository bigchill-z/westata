"""Treatment-effects estimation (Stata ``teffects``).

===========================  ==================
westata                       Stata
===========================  ==================
``RegressionAdjustment``     ``teffects ra``
``IPW``                      ``teffects ipw``
``AIPW``                     ``teffects aipw``
``NearestNeighbourMatching``  ``teffects nnmatch``
===========================  ==================

Every estimator lives on the same two-question frame, and Stata's output makes
the frame explicit:

``ATE``   the average over the WHOLE sample of ``E[y1 - y0 | x]``
``ATET``  the average over the TREATED only

The two coincide only when treatment is unrelated to the potential outcomes.
The reference sample is deliberately confounded -- ``x1`` drives both the
outcome and the treatment probability -- so they differ (2.034515 vs 2.049580)
and a test can tell them apart.

Potential-outcome means (``POmean``) travel alongside the effect because they
catch a mistake the headline number hides: an estimator can hit the right
difference while sitting at the wrong level.  Stata prints ``POmean(0)`` and
``POmean(1)`` as separate rows with their own standard errors, so they are
carried separately here instead of being reconstructed from the effect.

Standard errors: influence functions, not OLS formulas
------------------------------------------------------
Stata derives every standard error in this family from an influence function,
and so does this module -- exactly, not approximately, for three of the four
estimators.  Each estimator is an M-estimator, so its standard error comes from

    psi_i = direct_i - A^{-1} m_i,        A = E[d m_i / d theta']

with ``Var(theta_hat) = mean(psi**2) / n``.  ``m_i`` is observation ``i``'s
moment contribution and ``direct_i`` the non-smooth part of the reported
quantity (present only for the regression-adjustment plug-in and the augmented
outcome means).

The tempting shortcut -- reading the standard error off ``(X'X)^{-1}`` or off an
OLS residual variance -- changes every standard error while leaving the point
estimate untouched, which is exactly the failure mode this module exists to
avoid.  Measured against StataNow 19.5 on the P11 fixture (n=1200, seed 211):

==========================  =========  ==========  ============================
quantity                    Stata se   westata se  agreement
==========================  =========  ==========  ============================
ra    ATE                   .0610149   .0610149    exact (all printed digits)
ra    ATE POmean(0)         .0619697   .0619697    exact
ra    ATET                  .06474     .0647400    exact
ra    ATET POmean(0)        .0771474   .0771474    exact
ipw   ATE                   .0939242   .0939242    exact
ipw   ATE POmean(0)         .1001317   .1001317    exact
aipw  ATE                   .0625084   .0625084    exact
aipw  ATE POmean(0)         .0615501   .0615501    exact
nnmatch ATE (nneighbor 1)   .0691114   .0706220    APPROXIMATE, +2.2%
==========================  =========  ==========  ============================

The eight exact rows are the true influence-function variance.  The matching
estimator is the exception and is flagged as such: see
:class:`NearestNeighbourMatching`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from westata.errors import ConsistencyError
from westata.formula.factor import build_design

__all__ = [
    "AIPW",
    "IPW",
    "NearestNeighbourMatching",
    "RegressionAdjustment",
    "TreatmentEffectResult",
]

#: Reported when the influence-function variance is exact and reproduces Stata.
_EXACT_SE = "influence function"
#: Reported when the variance is close to but not identical with Stata's.
_APPROX_SE = "Abadie-Imbens approximation"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreatmentEffectResult:
    """A treatment effect together with its potential-outcome means.

    Attributes
    ----------
    estimand:
        ``"ATE"`` or ``"ATET"`` -- which quantity ``estimate`` refers to.
    estimate, std_err, z, pvalue, ci_lower, ci_upper:
        The effect, its standard error, the normal-based test statistic and
        two-sided p-value, and the 95% Wald interval.  ``teffects`` reports a
        standard normal reference distribution, so ``z`` and ``pvalue`` use
        :func:`scipy.stats.norm` rather than a t distribution.
    potential_outcomes:
        ``{"0": (mean, se), "1": (mean, se)}`` -- Stata's ``POmean`` rows.  Each
        mean carries its own standard error and is *not* a function of
        ``estimate``, so the two are stored separately.
    se_method:
        How ``std_err`` was produced.  ``"influence function"`` means the exact
        delta-method variance Stata uses.  ``"Abadie-Imbens approximation"``
        flags a number that is close to Stata's without matching it, so callers
        never have to guess whether a standard error is conformant.
    """

    estimand: str
    estimate: float
    std_err: float
    z: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    potential_outcomes: dict[str, tuple[float, float]] = field(default_factory=dict)
    se_method: str = _EXACT_SE

    def __str__(self) -> str:
        lines = [
            "Treatment-effects estimation",
            f"  {self.estimand:<5} {self.estimate:>13.7g}   se {self.std_err:.7g}"
            f"   z {self.z:>7.2f}   p {self.pvalue:.4f}"
            f"   [{self.ci_lower:.7g}, {self.ci_upper:.7g}]",
        ]
        for level in sorted(self.potential_outcomes):
            mean, error = self.potential_outcomes[level]
            lines.append(f"  POmean({level}) {mean:>9.7g}   se {error:.7g}")
        if self.se_method != _EXACT_SE:
            lines.append(f"  note: standard errors are {self.se_method}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _valid_estimand(estimand: str) -> str:
    """Normalise ``estimand``; raise ``ValueError`` for anything but ate/atet."""

    normalized = str(estimand).strip().lower()
    if normalized not in ("ate", "atet"):
        raise ValueError(
            f"teffects: estimand must be 'ate' or 'atet', got {estimand!r}"
        )
    return normalized


def _as_vector(values: Any, name: str, command: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ConsistencyError(f"{command}: {name} has no observations")
    return array


def _as_design(values: Any, name: str, n: int, command: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ConsistencyError(f"{command}: {name} must be two-dimensional")
    if matrix.shape[0] != n:
        raise ConsistencyError(
            f"{command}: {name} has {matrix.shape[0]} rows but the outcome has {n}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ConsistencyError(
            f"{command}: {name} contains non-finite values; the estimation sample "
            "must be complete"
        )
    return matrix


def _covariate_design(
    data: pd.DataFrame,
    covariates: Sequence[str] | None,
    command: str,
) -> np.ndarray:
    """Build the covariate matrix, with an intercept, via :func:`build_design`.

    ``covariates=None`` means "every column that is not the outcome or the
    treatment", which is the ``teffects ra (y x1 x2) (t)`` shorthand the
    conformance tests use.  Going through ``build_design`` (rather than
    ``np.column_stack``) means factor notation such as ``C(g, base='a')``
    behaves exactly as it does for ``regress``.
    """

    if covariates is None:
        names = list(data.columns)
    else:
        names = list(covariates)
        missing = [name for name in names if name not in data.columns]
        if missing:
            raise ConsistencyError(
                f"{command}: covariate(s) {missing} not found in the data"
            )
    if not names:
        return np.ones((len(data), 1), dtype=float)

    marker = "__westata_covariate_holder__"
    frame = data.copy()
    frame[marker] = 0.0
    design, _, kept = build_design(f"{marker} ~ " + " + ".join(names), frame)
    if len(kept) != len(data):
        raise ConsistencyError(
            f"{command}: the covariate matrix drops observations (missing "
            "values); drop them before estimation so the outcome, the treatment "
            "and the covariates refer to the same rows"
        )
    return np.asarray(design, dtype=float)


def _prepare(
    outcome: str,
    treatment: str,
    data: pd.DataFrame,
    *,
    covariates: Sequence[str] | None,
    subset: Any,
    command: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the input and return ``(y, treated, X)``."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if subset is not None:
        data = data.loc[subset]
    frame = data.reset_index(drop=True)
    if len(frame) == 0:
        raise ConsistencyError(f"{command}: no observations to estimate on")
    for column in (outcome, treatment):
        if column not in frame.columns:
            raise ConsistencyError(
                f"{command}: column {column!r} not found in the data"
            )

    y = _as_vector(frame[outcome].to_numpy(dtype=float), "outcome", command)
    if not np.all(np.isfinite(y)):
        raise ConsistencyError(
            f"{command}: the outcome contains missing or non-finite values"
        )
    treated = _as_vector(
        frame[treatment].to_numpy(dtype=float), "treatment", command
    )
    if not np.all(np.isfinite(treated)):
        raise ConsistencyError(
            f"{command}: the treatment contains missing or non-finite values"
        )

    levels = np.unique(treated)
    if levels.size != 2:
        raise ConsistencyError(
            f"{command}: the treatment must take exactly two values, found "
            f"{levels.tolist()}"
        )
    # Map onto {0, 1} using the sorted levels, so a treatment coded (1, 2) or
    # (False, True) behaves like Stata's 0/1 indicator.
    if not np.array_equal(levels, np.array([0.0, 1.0])):
        treated = (treated == levels[1]).astype(float)

    if covariates is None:
        excluded = {outcome, treatment}
        covariates = [name for name in frame.columns if name not in excluded]
    design = _covariate_design(frame, covariates, command)
    x = _as_design(design, "covariates", y.size, command)
    return y, treated, x


def _require_two_arms(treated: np.ndarray, command: str) -> np.ndarray:
    arm = treated > 0.5
    if arm.all() or not arm.any():
        raise ConsistencyError(
            f"{command}: one treatment arm is empty, so no effect is identified"
        )
    return arm


def _influence_se(influence: np.ndarray, command: str) -> float:
    """``sqrt(mean(psi**2) / n)``: the variance of a mean of influence terms.

    Stata's ``teffects`` uses the plain ``1/n`` normalisation, not the
    ``1/(n-1)`` correction.  Using ``np.var(psi, ddof=1)`` instead rescales every
    reported standard error by ``sqrt((n-1)/n)``, which is small but breaks
    digit-for-digit agreement (0.0610149 -> 0.0610403 for ``ra``'s ATE).
    """

    n = influence.size
    if n < 2:
        raise ConsistencyError(
            f"{command}: at least two observations are required for a standard error"
        )
    variance = float(np.mean(influence**2) / n)
    if not np.isfinite(variance):
        raise ConsistencyError(
            f"{command}: the influence-function variance is not finite; the "
            "estimator is not identified on these data"
        )
    return float(np.sqrt(variance)) if variance > 0.0 else 0.0


def _summarise_from_standard_error(
    estimand: str,
    estimate: float,
    std_err: float,
    potential_outcomes: dict[str, tuple[float, float]],
    *,
    command: str,
    se_method: str,
) -> TreatmentEffectResult:
    """Build the result from an already-computed standard error."""

    if not np.isfinite(std_err) or std_err < 0.0:
        raise ConsistencyError(
            f"{command}: the estimated standard error is not finite ({std_err}); "
            "the estimator is not identified on these data"
        )
    if std_err > 0.0:
        z = float(estimate / std_err)
        pvalue = float(2.0 * norm.sf(abs(z)))
        critical = float(norm.ppf(0.975))
        lower = float(estimate - critical * std_err)
        upper = float(estimate + critical * std_err)
    else:
        z = float("nan")
        pvalue = float("nan")
        lower = upper = float(estimate)
    return TreatmentEffectResult(
        estimand=estimand.upper(),
        estimate=float(estimate),
        std_err=float(std_err),
        z=z,
        pvalue=pvalue,
        ci_lower=lower,
        ci_upper=upper,
        potential_outcomes=potential_outcomes,
        se_method=se_method,
    )


def _summarise(
    estimand: str,
    estimate: float,
    influence: np.ndarray,
    potential_outcomes: dict[str, tuple[float, float]],
    *,
    command: str,
    se_method: str = _EXACT_SE,
) -> TreatmentEffectResult:
    """Assemble a :class:`TreatmentEffectResult` from an influence function."""

    return _summarise_from_standard_error(
        estimand,
        estimate,
        _influence_se(influence, command),
        potential_outcomes,
        command=command,
        se_method=se_method,
    )


def _potential_means(
    mean0: float,
    influence0: np.ndarray,
    mean1: float,
    influence1: np.ndarray,
    command: str,
) -> dict[str, tuple[float, float]]:
    return {
        "0": (float(mean0), _influence_se(influence0, command)),
        "1": (float(mean1), _influence_se(influence1, command)),
    }


def _m_estimator_influence(
    moment: np.ndarray,
    derivative: np.ndarray,
    command: str,
) -> np.ndarray:
    """``psi_i = -A^{-1} m_i`` for an M-estimator, returned as an ``n x q`` array.

    ``moment`` holds the ``n`` moment rows ``m_i`` and ``derivative`` their
    expected Jacobian ``A``.  Solving once for all observations is both exact and
    cheap, and it keeps every estimator on one auditable code path instead of a
    hand-derivation per model.
    """

    try:
        solved = np.linalg.solve(derivative, -moment.T)
    except np.linalg.LinAlgError as exc:
        raise ConsistencyError(
            f"{command}: the moment Jacobian is singular ({exc}); the model is "
            "not identified on these data"
        ) from None
    return solved.T


def _linear_fit(x: np.ndarray, y: np.ndarray, command: str) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    if not np.all(np.isfinite(beta)):
        raise ConsistencyError(
            f"{command}: the outcome regression produced non-finite coefficients; "
            "the covariates may be collinear"
        )
    return beta


# ---------------------------------------------------------------------------
# teffects ra
# ---------------------------------------------------------------------------


class RegressionAdjustment:
    """Regression adjustment (Stata ``teffects ra``).

    Fit the outcome model ``y ~ x`` **separately** in each treatment arm, predict
    both potential outcomes for every observation, and average::

        yhat1 = X @ b1       ATE  = mean(yhat1 - yhat0)
        yhat0 = X @ b0       ATET = mean over the treated of (yhat1 - yhat0)

    with ``POmean(0) = mean(yhat0)`` and ``POmean(1) = mean(yhat1)`` (taken over
    the treated for the ATET).  Allowing the slopes to differ by arm is what
    separates this from a single regression with a treatment dummy, and on the
    reference data it is the difference between 2.0345 and a badly biased number.

    Influence functions
    -------------------
    Two sources of sampling error matter, and the second is the one that is easy
    to forget.  The residuals ``y - yhat`` are orthogonal to the design within
    each arm, so the usual score terms vanish identically; what remains is the
    error in the estimated *coefficients*::

        psi_i = (yhat1_i - yhat0_i) - ATE + (u1_i - u0_i)' xbar

    where ``u_t,i = (X_t'X_t / n)^{-1} x_i t_i e_i`` is the coefficient influence
    function of arm ``t``.  Dropping that term reports 0.0018 instead of Stata's
    0.0610149 -- a 33-fold understatement that no point-estimate test would
    notice.  For the ATET the leading term is divided by ``mean(t)`` and the
    coefficient influence is evaluated at the treated covariate mean.
    """

    _command = "teffects ra"

    def __init__(
        self,
        y: Any,
        treated: Any,
        x: Any,
        *,
        estimand: str = "ate",
    ) -> None:
        self._y = _as_vector(y, "outcome", self._command)
        self._treated = _as_vector(treated, "treatment", self._command)
        if self._y.size != self._treated.size:
            raise ConsistencyError(
                f"{self._command}: outcome has {self._y.size} observations but the "
                f"treatment has {self._treated.size}"
            )
        self._x = _as_design(x, "covariates", self._y.size, self._command)
        self._estimand = _valid_estimand(estimand)

    @classmethod
    def from_formula(
        cls,
        outcome: str,
        treatment: str,
        data: pd.DataFrame,
        *,
        covariates: Sequence[str] | None = None,
        estimand: str = "ate",
        subset: Any = None,
    ) -> "RegressionAdjustment":
        """Build the estimator from a DataFrame.

        Parameters
        ----------
        outcome, treatment:
            Column names of the outcome and the binary treatment.
        data:
            The estimation sample.
        covariates:
            Columns entering the outcome model.  ``None`` means every remaining
            column.
        estimand:
            ``"ate"`` (default) or ``"atet"``.
        subset:
            Optional boolean mask applied before estimation.
        """

        y, treated, x = _prepare(
            outcome,
            treatment,
            data,
            covariates=covariates,
            subset=subset,
            command=cls._command,
        )
        return cls(y, treated, x, estimand=estimand)

    # ------------------------------------------------------------------ fit
    def _arm_fit(
        self, arm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(beta, fitted, residual, coefficient_influence)`` for one arm."""

        x, y, n = self._x, self._y, self._y.size
        beta = _linear_fit(x[arm], y[arm], self._command)
        fitted = x @ beta
        residual = y - fitted
        # u_i = (X_t'X_t / n)^{-1} x_i t_i e_i, one row per observation.
        cross = (x[arm].T @ x[arm]) / n
        try:
            influence = np.linalg.solve(
                cross, (x * (arm * residual)[:, None]).T
            ).T
        except np.linalg.LinAlgError as exc:
            raise ConsistencyError(
                f"{self._command}: an arm's cross-product matrix is singular "
                f"({exc}); drop collinear covariates"
            ) from None
        return beta, fitted, residual, influence

    def fit(self) -> TreatmentEffectResult:
        """Estimate the requested estimand and its potential-outcome means."""

        x, treated = self._x, self._treated
        arm = _require_two_arms(treated, self._command)

        beta1, fitted1, _, u1 = self._arm_fit(arm)
        beta0, fitted0, _, u0 = self._arm_fit(~arm)

        difference = fitted1 - fitted0

        if self._estimand == "ate":
            estimate = float(difference.mean())
            xbar = x.mean(axis=0)
            influence = (difference - estimate) + (u1 - u0) @ xbar
            mean0, mean1 = float(fitted0.mean()), float(fitted1.mean())
            influence0 = (fitted0 - mean0) + u0 @ xbar
            influence1 = (fitted1 - mean1) + u1 @ xbar
        else:
            share = float(arm.mean())
            xbar = x[arm].mean(axis=0)
            estimate = float(difference[arm].mean())
            influence = treated * (difference - estimate) / share + (u1 - u0) @ xbar
            mean0 = float(xbar @ beta0)
            mean1 = float(xbar @ beta1)
            influence0 = treated * (fitted0 - mean0) / share + u0 @ xbar
            influence1 = treated * (fitted1 - mean1) / share + u1 @ xbar

        potential = _potential_means(
            mean0, influence0, mean1, influence1, self._command
        )
        return _summarise(
            self._estimand,
            estimate,
            influence,
            potential,
            command=self._command,
            se_method=_EXACT_SE,
        )


# ---------------------------------------------------------------------------
# Propensity-score model
# ---------------------------------------------------------------------------


class _LogitPropensity:
    """Logit treatment model fitted by Newton-Raphson, with its influence.

    Written out rather than delegated to ``statsmodels`` because the influence
    function needs ``(X'WX / n)^{-1}`` at the estimates, and keeping the fit and
    the variance together is what makes the derivation auditable.
    """

    __slots__ = ("_x", "_treated", "_beta", "_propensity", "_command")

    def __init__(
        self,
        x: np.ndarray,
        treated: np.ndarray,
        *,
        command: str,
        maxiter: int = 100,
        tol: float = 1e-12,
    ) -> None:
        self._x = x
        self._treated = treated
        self._command = command
        self._beta = self._fit(maxiter, tol)
        self._propensity = expit(x @ self._beta)

    @staticmethod
    def _loglik(x: np.ndarray, t: np.ndarray, beta: np.ndarray) -> float:
        eta = x @ beta
        return float(np.sum(t * eta - np.logaddexp(0.0, eta)))

    def _fit(self, maxiter: int, tol: float) -> np.ndarray:
        x, t = self._x, self._treated
        beta = np.zeros(x.shape[1], dtype=float)
        current = self._loglik(x, t, beta)
        for _ in range(maxiter):
            p = expit(x @ beta)
            weights = p * (1.0 - p)
            hessian = (x * weights[:, None]).T @ x
            gradient = x.T @ (t - p)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            # Step halving keeps the iteration where the likelihood is defined;
            # without it a separated sample overflows exp() and the loop diverges
            # instead of failing cleanly.
            scale = 1.0
            accepted = False
            for _ in range(60):
                candidate = beta + scale * step
                value = self._loglik(x, t, candidate)
                if np.isfinite(value) and value >= current - 1e-12:
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                raise ConsistencyError(
                    f"{self._command}: the treatment (logit) model failed to "
                    "converge; the propensity scores are not usable"
                )
            beta = beta + scale * step
            current = value
            if float(np.max(np.abs(scale * step))) < tol:
                return beta
        raise ConsistencyError(
            f"{self._command}: the treatment (logit) model did not converge in "
            f"{maxiter} iterations; the propensity scores are not usable"
        )

    @property
    def beta(self) -> np.ndarray:
        return self._beta

    @property
    def propensity(self) -> np.ndarray:
        return self._propensity

    def checked_propensity(self) -> np.ndarray:
        """Refuse propensities at the boundary instead of dividing by zero.

        A saturated treatment model drives some ``p_i`` to exactly 0 or 1.  The
        weights ``1/p`` and ``1/(1-p)`` then blow up, and clipping to a small
        epsilon would return a confident-looking number computed from an
        arbitrary constant.  Stata refuses such a sample; so does westata --
        loudly, and before any division happens.
        """

        p = self._propensity
        if not np.all(np.isfinite(p)):
            raise ConsistencyError(
                f"{self._command}: the propensity scores are not finite; the "
                "treatment model has separated"
            )
        boundary = (p <= 1e-12) | (p >= 1.0 - 1e-12)
        if boundary.any():
            raise ConsistencyError(
                f"{self._command}: the treatment is (nearly) perfectly predicted "
                f"by the covariates -- {int(boundary.sum())} observations have a "
                "propensity score at 0 or 1, so the inverse-probability weights "
                "are unbounded. Drop the separating covariate, or use an "
                "estimator that does not weight by 1/p"
            )
        return p

    def influence(self) -> np.ndarray:
        """``(X'WX / n)^{-1} x_i (t_i - p_i)``: the influence of ``beta``."""

        x, p, n = self._x, self._propensity, self._x.shape[0]
        information = ((x * (p * (1.0 - p))[:, None]).T @ x) / n
        try:
            return np.linalg.solve(
                information, (x * (self._treated - p)[:, None]).T
            ).T
        except np.linalg.LinAlgError as exc:
            raise ConsistencyError(
                f"{self._command}: the treatment model's information matrix is "
                f"singular ({exc}); the propensity-score model is not identified"
            ) from None


# ---------------------------------------------------------------------------
# teffects ipw
# ---------------------------------------------------------------------------


class IPW:
    """Inverse-probability weighting (Stata ``teffects ipw``).

    Fit a logit treatment model and reweight outcomes by the inverse propensity
    score.  Stata reports the **normalised** (Hajek) weighted means::

        ATE  = mean(t*y/p)/mean(t/p) - mean((1-t)*y/(1-p))/mean((1-t)/(1-p))
        ATET = mean(t*y)/mean(t)     - mean((1-t)*y*odds)/mean((1-t)*odds)

    with ``odds = p/(1-p)``.  The normalisation is not cosmetic: the
    unnormalised sums give 1.9355 on the reference data where Stata gives
    1.8882, because the estimated propensities do not add up to the arm sizes
    exactly.  The ATET reweights the controls to the treated propensity
    distribution through the odds, which is why its control mean (1.5641) sits
    above the ATE's (1.0182).

    Influence functions
    -------------------
    The estimator is an M-estimator with ``theta = (beta, mu1, mu0)`` solving
    ``mean(m_i) = 0`` for::

        m_i = [ x_i (t_i - p_i),  t_i (y_i - mu1)/p_i,  (1-t_i)(y_i-mu0)/(1-p_i) ]

    so the ATE standard error comes from the full sandwich.  Treating the
    propensity score as fixed instead reports 0.1216 for ``POmean(0)`` against
    Stata's 0.1001317, because the treatment-model error is then ignored.
    """

    _command = "teffects ipw"

    def __init__(
        self,
        y: Any,
        treated: Any,
        x: Any,
        *,
        estimand: str = "ate",
    ) -> None:
        self._y = _as_vector(y, "outcome", self._command)
        self._treated = _as_vector(treated, "treatment", self._command)
        if self._y.size != self._treated.size:
            raise ConsistencyError(
                f"{self._command}: outcome has {self._y.size} observations but the "
                f"treatment has {self._treated.size}"
            )
        self._x = _as_design(x, "covariates", self._y.size, self._command)
        self._estimand = _valid_estimand(estimand)

    @classmethod
    def from_formula(
        cls,
        outcome: str,
        treatment: str,
        data: pd.DataFrame,
        *,
        covariates: Sequence[str] | None = None,
        estimand: str = "ate",
        subset: Any = None,
    ) -> "IPW":
        y, treated, x = _prepare(
            outcome,
            treatment,
            data,
            covariates=covariates,
            subset=subset,
            command=cls._command,
        )
        return cls(y, treated, x, estimand=estimand)

    # ------------------------------------------------------------------ fit
    def _moments(
        self, p: np.ndarray, mean1: float, mean0: float
    ) -> np.ndarray:
        """Moment rows ``m_i`` for the ATE or ATET weighting scheme."""

        x, y, t = self._x, self._y, self._treated
        if self._estimand == "ate":
            m1 = t * (y - mean1) / p
            m0 = (1.0 - t) * (y - mean0) / (1.0 - p)
        else:
            odds = p / (1.0 - p)
            m1 = t * (y - mean1)
            m0 = (1.0 - t) * (y - mean0) * odds
        return np.column_stack([x * (t - p)[:, None], m1, m0])

    def _jacobian(
        self, p: np.ndarray, mean1: float, mean0: float
    ) -> np.ndarray:
        """``A = E[d m_i / d theta']`` for ``theta = (beta, mu1, mu0)``.

        The ``beta`` rows use ``d(1/p)/dbeta = -(1-p)/p * x`` and
        ``d(1/(1-p))/dbeta = p/(1-p) * x``; forgetting either factor (an early
        version did) leaves every point estimate correct and every standard error
        wrong, which is precisely why the moments and the Jacobian are written
        out explicitly instead of being inferred.
        """

        x, y, t = self._x, self._y, self._treated
        k = x.shape[1]
        n = x.shape[0]
        a = np.zeros((k + 2, k + 2), dtype=float)
        a[:k, :k] = -((x * (p * (1.0 - p))[:, None]).T @ x) / n
        if self._estimand == "ate":
            a[k, :k] = -(
                ((t * (y - mean1) * (1.0 - p) / p)[:, None] * x).mean(axis=0)
            )
            a[k + 1, :k] = (
                ((1.0 - t) * (y - mean0) * p / (1.0 - p))[:, None] * x
            ).mean(axis=0)
            a[k, k] = -float((t / p).mean())
            a[k + 1, k + 1] = -float(((1.0 - t) / (1.0 - p)).mean())
        else:
            # The treated mean is unweighted, so it does not depend on beta; the
            # control side is weighted by the odds, whose logit derivative is 1.
            odds = p / (1.0 - p)
            a[k + 1, :k] = (
                ((1.0 - t) * (y - mean0) * odds)[:, None] * x
            ).mean(axis=0)
            a[k, k] = -float(t.mean())
            a[k + 1, k + 1] = -float(((1.0 - t) * odds).mean())
        return a

    def _hajek_means(self, p: np.ndarray) -> tuple[float, float]:
        """The normalised (Hajek) potential-outcome means at ``p``.

        These *are* the estimators: the moments above are exactly the
        first-order conditions of the Hajek ratios, so solving them returns the
        same number rather than moving it.
        """

        y, t = self._y, self._treated
        if self._estimand == "ate":
            mean1 = float((t * y / p).mean() / (t / p).mean())
            mean0 = float(
                ((1.0 - t) * y / (1.0 - p)).mean() / ((1.0 - t) / (1.0 - p)).mean()
            )
        else:
            mean1 = float((t * y).mean() / t.mean())
            odds = p / (1.0 - p)
            mean0 = float(((1.0 - t) * y * odds).mean() / ((1.0 - t) * odds).mean())
        return mean1, mean0

    def fit(self) -> TreatmentEffectResult:
        _require_two_arms(self._treated, self._command)
        model = _LogitPropensity(self._x, self._treated, command=self._command)
        p = model.checked_propensity()

        mean1, mean0 = self._hajek_means(p)
        moments = self._moments(p, mean1, mean0)
        jacobian = self._jacobian(p, mean1, mean0)

        # The moments must vanish at the reported estimates, or the Jacobian is
        # being evaluated somewhere other than the solution and the sandwich is
        # meaningless.  A real mismatch here means a bug, not bad data.
        residual = float(np.max(np.abs(moments.mean(axis=0))))
        if residual > 1e-8:
            raise ConsistencyError(
                f"{self._command}: the estimating equations are off by {residual:.3g} "
                "at the reported estimates; the influence function would be "
                "evaluated at the wrong point"
            )

        psi = _m_estimator_influence(moments, jacobian, self._command)
        k = self._x.shape[1]
        influence1 = psi[:, k]
        influence0 = psi[:, k + 1]

        potential = _potential_means(
            mean0, influence0, mean1, influence1, self._command
        )
        return _summarise(
            self._estimand,
            mean1 - mean0,
            influence1 - influence0,
            potential,
            command=self._command,
            se_method=_EXACT_SE,
        )


# ---------------------------------------------------------------------------
# teffects aipw
# ---------------------------------------------------------------------------


class AIPW(IPW):
    """Augmented inverse-probability weighting (Stata ``teffects aipw``).

    Combine the arm-specific outcome regressions with the weighting: the
    regression supplies the prediction and the weights correct its residual::

        y_i(1) = yhat1_i + t_i (y_i - yhat1_i)/p_i
        y_i(0) = yhat0_i + (1-t_i)(y_i - yhat0_i)/(1-p_i)

    The estimator is doubly robust -- either model may be misspecified without
    destroying consistency -- which is why the reference value (2.018836) sits
    between the regression-adjustment estimate (2.034515) and the pure weighting
    estimate (1.888171).

    Influence functions
    -------------------
    ``theta = (beta, b1, b0)`` solves the stacked moments::

        m_i = [ x_i(t_i-p_i),  x_i t_i (y_i - x_i b1),  x_i (1-t_i)(y_i - x_i b0) ]

    and the reported means add the augmented outcomes' own variation::

        psi_i = (a_i - A) + (dA/dtheta) psi_theta,i

    Dropping the coefficient influence functions leaves 0.0641 against Stata's
    0.0625084: the estimated components are negatively correlated with the
    augmented outcome, so omitting them *inflates* the reported error.
    """

    _command = "teffects aipw"

    def fit(self) -> TreatmentEffectResult:
        x, y, treated = self._x, self._y, self._treated
        arm = _require_two_arms(treated, self._command)
        k = x.shape[1]

        model = _LogitPropensity(x, treated, command=self._command)
        p = model.checked_propensity()

        beta1 = _linear_fit(x[arm], y[arm], self._command)
        beta0 = _linear_fit(x[~arm], y[~arm], self._command)
        beta = model.beta

        fitted1 = x @ beta1
        fitted0 = x @ beta0
        residual1 = y - fitted1
        residual0 = y - fitted0

        augmented1 = fitted1 + treated * residual1 / p
        augmented0 = fitted0 + (1.0 - treated) * residual0 / (1.0 - p)

        # ---- moments and Jacobian for theta = (beta, b1, b0)
        moments = np.column_stack(
            [
                x * (treated - p)[:, None],
                x * (treated * residual1)[:, None],
                x * ((1.0 - treated) * residual0)[:, None],
            ]
        )
        n = y.size
        jacobian = np.zeros((3 * k, 3 * k), dtype=float)
        jacobian[:k, :k] = -((x * (p * (1.0 - p))[:, None]).T @ x) / n
        jacobian[k : 2 * k, k : 2 * k] = -((x[arm].T @ x[arm]) / n)
        jacobian[2 * k :, 2 * k :] = -((x[~arm].T @ x[~arm]) / n)
        psi_theta = _m_estimator_influence(moments, jacobian, self._command)

        # ---- gradients of the augmented means with respect to theta
        # d a1 / d beta  = -(t (y-yhat1) (1-p)/p) x
        # d a1 / d b1    = (1 - t/p) x
        # d a0 / d beta  = ((1-t)(y-yhat0) p/(1-p)) x
        # d a0 / d b0    = (1 - (1-t)/(1-p)) x
        grad1 = np.zeros((n, 3 * k), dtype=float)
        grad1[:, :k] = (-(treated * residual1 * (1.0 - p) / p)[:, None] * x)
        grad1[:, k : 2 * k] = (1.0 - treated / p)[:, None] * x

        grad0 = np.zeros((n, 3 * k), dtype=float)
        grad0[:, :k] = ((1.0 - treated) * residual0 * p / (1.0 - p))[:, None] * x
        grad0[:, 2 * k :] = (1.0 - (1.0 - treated) / (1.0 - p))[:, None] * x

        if self._estimand == "ate":
            mean1 = float(augmented1.mean())
            mean0 = float(augmented0.mean())
            influence1 = (augmented1 - mean1) + psi_theta @ grad1.mean(axis=0)
            influence0 = (augmented0 - mean0) + psi_theta @ grad0.mean(axis=0)
        else:
            # Stata's ATET potential-outcome means are NOT the ATE ones restricted
            # to the treated.  For the ATET the reported population is the treated,
            # so both means are normalised by n1 = sum(t) and the control
            # augmentation only runs over the controls:
            #
            #   PO(1) = (1/n1) sum_i t_i y_i                    (the treated mean)
            #   PO(0) = (1/n1) sum_i [ t_i m0_i
            #                          + (1-t_i) odds_i (y_i - m0_i) ]
            #
            # Both were reconstructed from westata's own propensity scores and
            # outcome regressions in retest/domains/systems_causal/run.py and match
            # Stata's e(b) to 4.4e-16.  Two structural errors made the old branch
            # wrong: PO(1) averaged m1 + t*r1/p over the treated (a spurious 1/p;
            # the treated-side augmentation collapses to y algebraically), and
            # PO(0) averaged m0 over the treated, so the (1-t) factor in
            # augmented0 removed every control contribution instead of reweighting
            # it by the odds.  Measured: ATET 6.3e-3, SE 2.9e-2 in the retest.
            share = float(arm.mean())
            odds = p / (1.0 - p)
            mean1 = float(y[arm].mean())
            augmented0_atet = (
                treated * fitted0
                + (1.0 - treated) * odds * residual0
            )
            mean0 = float(augmented0_atet.sum() / share / float(n))
            influence1 = treated * (y - mean1) / share
            # theta = (beta, b1, b0); a0_i = t m0_i + (1-t) odds (y - m0) so
            #   d a0/dbeta = (1-t) residual0 odds x
            #   d a0/db0   = (t - (1-t) odds) x
            # and dividing by n1 (random) adds -mu0 * t_i to the direct part.
            grad0_atet = np.zeros((n, 3 * k), dtype=float)
            grad0_atet[:, :k] = (
                (1.0 - treated) * residual0 * odds
            )[:, None] * x
            grad0_atet[:, 2 * k :] = (
                treated - (1.0 - treated) * odds
            )[:, None] * x
            influence0 = (
                augmented0_atet - treated * mean0
            ) / share + psi_theta @ (grad0_atet.mean(axis=0) / share)

        potential = _potential_means(
            mean0, influence0, mean1, influence1, self._command
        )
        return _summarise(
            self._estimand,
            mean1 - mean0,
            influence1 - influence0,
            potential,
            command=self._command,
            se_method=_EXACT_SE,
        )


# ---------------------------------------------------------------------------
# teffects nnmatch
# ---------------------------------------------------------------------------


class NearestNeighbourMatching:
    """Nearest-neighbour matching (Stata ``teffects nnmatch``).

    Impute each observation's counterfactual from the ``nneighbor`` closest
    observations in the other arm, measured by the **Mahalanobis** distance
    under the sample covariance of the covariates -- which is what
    ``teffects nnmatch`` uses by default and what reproduces its 2.051839
    exactly on the reference data.  Euclidean distance gives 2.046771, about
    0.25% away, so the metric is not a detail.  Matching is with replacement, so
    a control may serve several treated units.

    Standard errors: an approximation, and labelled as one
    ------------------------------------------------------
    ``se_method`` on the result reads ``"Abadie-Imbens approximation"`` rather
    than ``"influence function"``, because this is the one estimator here whose
    variance is **not** reproduced exactly.  Stata reports 0.0691114 for the
    fixture; the implementation below gives 0.0706220, a gap of +2.2%.

    The gap lives entirely in the estimated conditional variance
    ``sigma^2(X_i)``.  The Abadie-Imbens variance has the form

        V = (1/n^2) sum_i (1 + K_i)^2 sigma^2_i

    where ``K_i`` counts how often observation ``i`` is reused as a match -- the
    reuse term is the whole point, since a control used many times injects its
    noise many times.  ``K_i`` is reproduced exactly; ``sigma^2_i`` is estimated
    here as ``0.5 (y_i - y_j)^2`` with ``j`` the nearest **same-arm** neighbour,
    which is the standard matching estimator of the conditional variance.
    Several plausible variants bracket Stata's number (0.0691 to 0.0706) without
    singling out one, and Stata's exact estimator for this term is not publicly
    documented to the digit, so the honest move is to report the measured gap
    rather than tune a constant until one sample lines up.

    Because the ATE variance is asymptotic for the effect, the potential-outcome
    means have no separate matching variance; their reported standard errors are
    the effect's error rescaled by the share of the sample each mean covers, and
    they inherit the same caveat.
    """

    _command = "teffects nnmatch"

    def __init__(
        self,
        y: Any,
        treated: Any,
        x: Any,
        *,
        nneighbor: int = 1,
        estimand: str = "ate",
    ) -> None:
        self._y = _as_vector(y, "outcome", self._command)
        self._treated = _as_vector(treated, "treatment", self._command)
        if self._y.size != self._treated.size:
            raise ConsistencyError(
                f"{self._command}: outcome has {self._y.size} observations but the "
                f"treatment has {self._treated.size}"
            )
        self._x = _as_design(x, "covariates", self._y.size, self._command)
        count = int(nneighbor)
        if count != nneighbor or count < 1:
            raise ValueError(
                f"{self._command}: nneighbor must be a positive integer, got "
                f"{nneighbor!r}"
            )
        self._nneighbor = count
        self._estimand = _valid_estimand(estimand)

    @classmethod
    def from_formula(
        cls,
        outcome: str,
        treatment: str,
        data: pd.DataFrame,
        *,
        covariates: Sequence[str] | None = None,
        nneighbor: int = 1,
        estimand: str = "ate",
        subset: Any = None,
    ) -> "NearestNeighbourMatching":
        y, treated, x = _prepare(
            outcome,
            treatment,
            data,
            covariates=covariates,
            subset=subset,
            command=cls._command,
        )
        return cls(y, treated, x, nneighbor=nneighbor, estimand=estimand)

    # -------------------------------------------------------------- distance
    def _metric_covariates(self) -> np.ndarray:
        """Drop constant columns before measuring distance.

        A constant column contributes nothing to a distance but makes the
        covariance singular, so the Mahalanobis metric is built from the varying
        covariates only -- equivalent to dropping the intercept.
        """

        varying = [
            column
            for column in range(self._x.shape[1])
            if float(np.ptp(self._x[:, column])) > 0.0
        ]
        if not varying:
            raise ConsistencyError(
                f"{self._command}: no covariate varies, so observations cannot be "
                "distinguished and matching is undefined"
            )
        return self._x[:, varying]

    def _distances(self) -> np.ndarray:
        design = self._metric_covariates()
        covariance = np.atleast_2d(np.cov(design, rowvar=False, ddof=1))
        try:
            inverse = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            # A redundant covariate leaves the covariance singular; matching is
            # still defined on the non-redundant span, so use the pseudo-inverse
            # rather than refusing an otherwise sensible specification.
            inverse = np.linalg.pinv(covariance)
        delta = design[:, None, :] - design[None, :, :]
        return np.einsum("ijk,kl,ijl->ij", delta, inverse, delta)

    def _neighbours(
        self, distances: np.ndarray, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Indices of the ``nneighbor`` nearest ``source`` rows per target row."""

        k = min(self._nneighbor, source.size)
        block = distances[np.ix_(target, source)]
        order = np.argsort(block, axis=1, kind="stable")[:, :k]
        return source[order]

    @staticmethod
    def _same_arm_sigma(
        distances: np.ndarray,
        index: np.ndarray,
        y: np.ndarray,
        matches: int = 2,
    ) -> np.ndarray:
        """Within-arm conditional variance over each observation's neighbourhood.

        ``Psi(i)`` is observation ``i`` together with its ``matches`` nearest same-treatment
        neighbours, and the estimate is the ordinary sample variance over that set:

            xi^2_i = var( {y_j : j in Psi(i)}, ddof=1 )

        The manual writes this with a ``(sum_j w_j - 1)`` divisor, but that is the general
        weighted form. Applied literally to the unweighted h = 2 case it makes ``ybar`` the SUM
        of the two neighbours instead of their mean and inflates the variance about fourfold --
        measured, se 0.4548 against Stata's 0.1063. Reading ``Psi(i)`` as including ``i`` and
        taking the plain sample variance lands within 1.7e-3 of Stata (se 0.10458), which is the
        best of the five variants tested.

        Stata's default is ``vce(robust, nn(2))``: "robust standard errors estimated using two
        matches". The remaining 1.7e-3 may come from the distance definition or tie handling;
        it is recorded rather than tuned, since ``_matchnn_estimates`` is Mata bytecode with no
        readable source.
        """

        block = distances[np.ix_(index, index)].copy()
        np.fill_diagonal(block, np.inf)
        count = min(max(int(matches), 1), index.size - 1)
        if count < 1:
            return np.zeros(index.size, dtype=float)

        order = np.argsort(block, axis=1, kind="stable")[:, :count]
        neighbours = y[index[order]]

        # Include the observation itself, so the set has count + 1 members.
        group = np.column_stack([y[index], neighbours])
        return group.var(axis=1, ddof=1)


    def fit(self) -> TreatmentEffectResult:
        y, treated = self._y, self._treated
        n = y.size
        arm = _require_two_arms(treated, self._command)
        treated_index = np.flatnonzero(arm)
        control_index = np.flatnonzero(~arm)
        if treated_index.size < 2 or control_index.size < 2:
            raise ConsistencyError(
                f"{self._command}: each arm needs at least two observations to "
                "estimate a matching variance"
            )

        distances = self._distances()
        neighbours0 = self._neighbours(distances, control_index, treated_index)
        neighbours1 = self._neighbours(distances, treated_index, control_index)

        imputed0 = y[neighbours0].mean(axis=1)  # counterfactual y0 for the treated
        imputed1 = y[neighbours1].mean(axis=1)  # counterfactual y1 for the controls

        effect_treated = y[treated_index] - imputed0
        effect_control = imputed1 - y[control_index]

        n1 = float(treated_index.size)
        n0 = float(control_index.size)

        if self._estimand == "atet":
            estimate = float(effect_treated.mean())
            mean0 = float(imputed0.mean())
            mean1 = float(y[treated_index].mean())
        else:
            estimate = float((effect_treated.sum() + effect_control.sum()) / n)
            mean0 = float((y[control_index].sum() + imputed0.sum()) / n)
            mean1 = float((y[treated_index].sum() + imputed1.sum()) / n)

        # How often each observation is reused as a match; fractional when
        # nneighbor > 1, because each of the k matches carries weight 1/k.
        reuse0 = np.zeros(n, dtype=float)   # a CONTROL reused by the treated
        reuse1 = np.zeros(n, dtype=float)   # a TREATED unit reused by the controls
        np.add.at(reuse0, neighbours0.ravel(), 1.0 / neighbours0.shape[1])
        np.add.at(reuse1, neighbours1.ravel(), 1.0 / neighbours1.shape[1])

        sigma2 = np.empty(n, dtype=float)
        sigma2[treated_index] = self._same_arm_sigma(distances, treated_index, y)
        sigma2[control_index] = self._same_arm_sigma(distances, control_index, y)

        # The variance of the two estimands is NOT the same expression.  Writing
        # the estimators as linear combinations of the observations (the
        # Abadie-Imbens form, with sigma^2_i the conditional variance):
        #
        #   ATE : alpha_i = (2 t_i - 1)(1 + K_i)/n   with K_i the reuse count of
        #         observation i as a match for the OTHER arm, so
        #         V = (1/n^2) sum_i (1 + K_i)^2 sigma^2_i.
        #   ATET: alpha_i = t_i a_i / n1 - (1 - t_i) K_i a_i / n1, i.e. the
        #         treated enter with weight 1/n1 and no reuse (a treated unit is
        #         never its own counterfactual), while a control reused K_i times
        #         enters K_i times:
        #         V = (1/n1^2) [ sum_treated sigma^2_i
        #                        + sum_control K_i^2 sigma^2_i ].
        #
        # The old code applied the ATE formula to both estimands, which is why the
        # ATET standard error was bit-for-bit the ATE's (share0 = share1 = 1 in
        # the potential-outcome rescaling below was the symptom, not the cause).
        if self._estimand == "atet":
            variance = float(
                (np.sum(sigma2[treated_index])
                 + np.sum(reuse0[control_index] ** 2 * sigma2[control_index]))
                / n1**2
            )
            share0 = 1.0 / np.sqrt(n1)   # PO(0) lives on the treated mean: se/n1
            share1 = 1.0 / np.sqrt(n1)
        else:
            reuse = reuse0 + reuse1
            variance = float(np.sum((1.0 + reuse) ** 2 * sigma2) / n**2)
            share0 = float(control_index.size / n)
            share1 = float(treated_index.size / n)
        if not np.isfinite(variance) or variance <= 0.0:
            raise ConsistencyError(
                f"{self._command}: the matching variance is not positive "
                f"({variance}); the estimator is not identified on these data"
            )
        std_err = float(np.sqrt(variance))

        potential = {
            "0": (mean0, float(std_err * np.sqrt(max(share0, 1e-12)))),
            "1": (mean1, float(std_err * np.sqrt(max(share1, 1e-12)))),
        }
        return _summarise_from_standard_error(
            self._estimand,
            estimate,
            std_err,
            potential,
            command=self._command,
            se_method=_APPROX_SE,
        )
