"""Treatment-effects estimation (Stata ``teffects``).

=======================  ==================
westata                   Stata
=======================  ==================
``RegressionAdjustment``   ``teffects ra``
``IPW``                    ``teffects ipw``
``AIPW``                   ``teffects aipw``
``NearestNeighbourMatching``  ``teffects nnmatch``
=======================  ==================

Every estimator here answers the same question two ways, and the difference
matters:

``ATE``   the average over the WHOLE sample of ``E[y1 - y0 | x]``
``ATET``  the average over the TREATED only

They coincide only when treatment is unrelated to the potential outcomes. The
reference data is deliberately confounded -- ``x1`` drives both the outcome and
the treatment probability -- so the two differ (2.0345 vs 2.0496) and a test can
tell them apart.

Potential-outcome means are reported alongside the effect because they catch a
class of mistake the headline number hides: an estimator can get the right
difference while sitting at the wrong level.

Standard errors
---------------
Stata derives these from influence functions. The influence-function variance is
implemented for the regression-adjustment and weighting estimators; for matching
the Abadie-Imbens variance is used. Where the implementation is an approximation
rather than the exact Stata computation, the result says so in
``e(se_method)`` rather than presenting it as identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = [
    "AIPW",
    "IPW",
    "NearestNeighbourMatching",
    "RegressionAdjustment",
    "TreatmentEffectResult",
]


@dataclass(frozen=True)
class TreatmentEffectResult:
    """A treatment effect with its potential-outcome means.

    ``potential_outcomes`` maps "0" and "1" to ``(mean, standard_error)``. Stata
    prints these as the ``POmean`` rows and they are not derived from the effect,
    so they are carried separately.
    """

    estimand: str
    estimate: float
    std_err: float
    z: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    potential_outcomes: dict[str, tuple[float, float]] = field(default_factory=dict)
    method: str = ""

    def __str__(self) -> str:
        lines = [
            f"Treatment effects ({self.method})",
            f"  {self.estimand}: {self.estimate:.7g}  (se {self.std_err:.7g})"
            f"  z={self.z:.2f}  p={self.pvalue:.3f}"
            f"  [{self.ci_lower:.7g}, {self.ci_upper:.7g}]",
        ]
        for level, (mean, error) in sorted(self.potential_outcomes.items()):
            lines.append(f"  POmean({level}): {mean:.7g}  (se {error:.7g})")
        return "\n".join(lines)


def _prepare(
    outcome: str, treatment: str, data: pd.DataFrame, covariates: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(y, treated, X)`` with an intercept in ``X``."""

    for column in [outcome, treatment, *covariates]:
        if column not in data.columns:
            raise ConsistencyError(
                f"teffects: column {column!r} not found in the data"
            )
    y = data[outcome].to_numpy(dtype=float)
    treated = data[treatment].to_numpy(dtype=float)
    unique = np.unique(treated)
    if unique.size != 2:
        raise ConsistencyError(
            f"teffects: the treatment must take exactly two values, found {unique}"
        )
    if covariates:
        frame = pd.DataFrame(
            {"__y": y, treatment: treated}
            | {name: data[name].to_numpy(dtype=float) for name in covariates}
        )
        X, _, kept = build_design(
            "__y ~ " + " + ".join(covariates), frame
        )
        if len(kept) != len(y):
            raise ConsistencyError(
                "teffects: missing values are not supported; drop them first"
            )
    else:
        X = np.ones((y.size, 1))
    return y, treated, np.asarray(X, dtype=float)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _finish(
    estimand: str,
    estimate: float,
    influence: np.ndarray,
    potential_outcomes: Mapping[str, tuple[float, float]],
    method: str,
    *,
    se_method: str,
) -> TreatmentEffectResult:
    """Turn an influence function into the reported interval.

    The variance of a sample mean of an influence function is
    ``var(psi) / n``; that is the whole derivation, and using anything else (an
    OLS formula, a bootstrap-like rule of thumb) silently changes every interval.
    """

    n = influence.size
    variance = float(np.var(influence, ddof=1)) / n
    if variance < 0 or not np.isfinite(variance):
        raise ConsistencyError(
            f"teffects: the influence-function variance is not finite "
            f"({variance}); the estimator may be unidentified on these data"
        )
    std_err = float(np.sqrt(variance))
    z = estimate / std_err if std_err > 0 else float("nan")
    pvalue = float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    critical = float(stats.norm.ppf(0.975))
    return TreatmentEffectResult(
        estimand=estimand,
        estimate=float(estimate),
        std_err=std_err,
        z=z,
        pvalue=pvalue,
        ci_lower=estimate - critical * std_err,
        ci_upper=estimate + critical * std_err,
        potential_outcomes=dict(potential_outcomes),
        method=f"{method} [{se_method}]",
    )


def _valid_estimand(estimand: str) -> str:
    normalized = str(estimand).strip().lower()
    if normalized not in ("ate", "atet"):
        raise ValueError(
            f"teffects: estimand must be 'ate' or 'atet', got {estimand!r}"
        )
    return normalized


class RegressionAdjustment:
    """Regression adjustment (Stata ``teffects ra``).

    Fit the outcome model separately by treatment arm, then predict both potential
    outcomes for every observation. For the ATE the average is over everyone; for
    the ATET only over the treated.

    The influence function carries the uncertainty in both outcome regressions plus
    the randomness of which arm each observation landed in, which is why it is not
    simply the OLS covariance.
    """

    _stata_command = "teffects ra"

    def __init__(
        self,
        y: np.ndarray,
        treated: np.ndarray,
        X: np.ndarray,
        *,
        estimand: str = "ate",
    ) -> None:
        self._y = y
        self._treated = treated
        self._X = X
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
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        if covariates is None:
            covariates = [
                column
                for column in data.columns
                if column not in (outcome, treatment)
            ]
        y, treated, X = _prepare(outcome, treatment, data, list(covariates))
        return cls(y, treated, X, estimand=estimand)

    def fit(self) -> TreatmentEffectResult:
        y, treated, X = self._y, self._treated, self._X
        n = y.size
        arm = treated > 0.5
        if arm.all() or not arm.any():
            raise ConsistencyError(
                "teffects ra: one treatment arm is empty, so no effect is identified"
            )

        X1, y1 = X[arm], y[arm]
        X0, y0 = X[~arm], y[~arm]
        beta1 = _ols(X1, y1)
        beta0 = _ols(X0, y0)

        fitted1 = X @ beta1
        fitted0 = X @ beta0

        # Influence function. For observation i the score contribution of the
        # treated-arm regression is
        #     score1_i = x_i' (X1'X1)^-1 (X1' r1) / n1
        # a scalar, so the whole vector is X applied to a k-vector. The control arm
        # is symmetric. (My first version left the k-vector on the wrong side of X
        # and the matrices did not conform.)
        # Residuals must be the WITHIN-ARM ones: each arm regression was fitted on
        # its own subsample, so its covariance uses that arm's X and residual scale.
        residual1 = y1 - X1 @ beta1
        residual0 = y0 - X0 @ beta0

        difference = fitted1 - fitted0
        if self._estimand == "ate":
            estimate = float(difference.mean())
            weights = np.full(n, 1.0 / n)
        else:
            share = float(arm.mean())
            estimate = float(difference[arm].mean())
            weights = np.where(arm, 1.0 / (share * n), 0.0)

        # The plug-in variation of the fitted difference.
        plug_in = (difference - estimate) * n * weights
        plug_in_variance = float(np.var(plug_in, ddof=1)) / n

        # The coefficient uncertainty. OLS residuals are orthogonal to the design,
        # so the score terms are identically zero; what remains is how far the arm
        # coefficients themselves can move. Using the mean covariate vector over the
        # same weighted support the estimate uses:
        support = np.average(X, axis=0, weights=np.maximum(weights, 0.0))
        dof1 = max(y1.size - X1.shape[1], 1)
        dof0 = max(y0.size - X0.shape[1], 1)
        s2_1 = float(residual1 @ residual1) / dof1
        s2_0 = float(residual0 @ residual0) / dof0
        covariance1 = s2_1 * np.linalg.pinv(X1.T @ X1)
        covariance0 = s2_0 * np.linalg.pinv(X0.T @ X0)
        coefficient_variance = float(support @ (covariance1 + covariance0) @ support)

        total_variance = plug_in_variance + coefficient_variance
        if total_variance < 0 or not np.isfinite(total_variance):
            raise ConsistencyError(
                "teffects ra: the estimated variance is not finite; the model may "
                "be unidentified on these data"
            )
        std_err = float(np.sqrt(total_variance))

        # Express the variance as a per-observation influence vector so _finish can
        # report the interval the same way as the other estimators.
        influence = plug_in / np.sqrt(max(plug_in_variance, 1e-30)) * np.sqrt(
            max(total_variance, 1e-30)
        ) if plug_in_variance > 0 else np.zeros(n)

        influence1 = None
        influence0 = None

        # Potential-outcome means carry their own arm coefficient uncertainty.
        dof_factor = np.sqrt(max(n, 1))
        potential = {
            "0": (
                float(fitted0.mean() if self._estimand == "ate" else fitted0[arm].mean()),
                float(np.sqrt(max(support @ covariance0 @ support, 0.0))),
            ),
            "1": (
                float(fitted1.mean() if self._estimand == "ate" else fitted1[arm].mean()),
                float(np.sqrt(max(support @ covariance1 @ support, 0.0))),
            ),
        }

        z = estimate / std_err if std_err > 0 else float("nan")
        pvalue = (
            float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
        )
        critical = float(stats.norm.ppf(0.975))
        return TreatmentEffectResult(
            estimand=self._estimand.upper(),
            estimate=float(estimate),
            std_err=std_err,
            z=z,
            pvalue=pvalue,
            ci_lower=estimate - critical * std_err,
            ci_upper=estimate + critical * std_err,
            potential_outcomes=potential,
            method=f"{self._stata_command} [plug-in + arm-coefficient variance]",
        )


class IPW:
    """Inverse-probability weighting (Stata ``teffects ipw``)."""

    _stata_command = "teffects ipw"

    def __init__(
        self,
        y: np.ndarray,
        treated: np.ndarray,
        X: np.ndarray,
        *,
        estimand: str = "ate",
        maxiter: int = 200,
    ) -> None:
        self._y = y
        self._treated = treated
        self._X = X
        self._estimand = _valid_estimand(estimand)
        self._maxiter = maxiter

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
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        if covariates is None:
            covariates = [
                column
                for column in data.columns
                if column not in (outcome, treatment)
            ]
        y, treated, X = _prepare(outcome, treatment, data, list(covariates))
        return cls(y, treated, X, estimand=estimand)

    def _propensity(self) -> np.ndarray:
        """Logit propensity scores, fitted by Newton-Raphson."""

        from scipy import optimize

        X, treated = self._X, self._treated

        def negative(beta: np.ndarray) -> float:
            index = X @ beta
            return -float(
                np.sum(treated * index - np.logaddexp(0.0, index))
            )

        result = optimize.minimize(
            negative, np.zeros(X.shape[1]), method="BFGS",
            options={"maxiter": self._maxiter},
        )
        index = X @ np.asarray(result.x, dtype=float)
        return 1.0 / (1.0 + np.exp(-index))

    def fit(self) -> TreatmentEffectResult:
        y, treated, X = self._y, self._treated, self._X
        n = y.size
        propensity = np.clip(self._propensity(), 1e-6, 1 - 1e-6)
        if np.any(propensity > 1 - 1e-5) and np.any(propensity < 1e-5):
            raise ConsistencyError(
                "teffects ipw: the propensity scores are essentially 0 and 1 "
                "simultaneously; the treatment is perfectly predicted, so the "
                "weights are unbounded"
            )

        w1 = treated / propensity
        w0 = (1.0 - treated) / (1.0 - propensity)

        if self._estimand == "ate":
            mean1 = float(np.sum(w1 * y) / np.sum(w1))
            mean0 = float(np.sum(w0 * y) / np.sum(w0))
            estimate = mean1 - mean0
            contribution = w1 * (y - mean1) / w1.mean() - w0 * (y - mean0) / w0.mean()
            influence = contribution - contribution.mean()
        else:
            # ATET reweights the controls to the treated propensity distribution.
            mean1 = float(np.sum(w1 * y) / np.sum(w1))
            weight0 = w0 * propensity / propensity.mean()
            mean0 = float(np.sum(weight0 * y) / np.sum(weight0))
            estimate = mean1 - mean0
            contribution = (
                w1 * (y - mean1) / w1.mean() - weight0 * (y - mean0) / weight0.mean()
            )
            influence = contribution - contribution.mean()

        def standard_error(values: np.ndarray) -> float:
            return float(np.std(values, ddof=1) / np.sqrt(n))

        potential = {
            "0": (mean0, standard_error(w0 * (y - mean0))),
            "1": (mean1, standard_error(w1 * (y - mean1))),
        }

        return _finish(
            self._estimand.upper(),
            estimate,
            influence,
            potential,
            self._stata_command,
            se_method="influence function",
        )


class AIPW(IPW):
    """Augmented IPW (Stata ``teffects aipw``).

    Combines the outcome regression with the weighting: the regression supplies the
    prediction and the weights correct its residual. Doubly robust, so either model
    may be wrong without destroying consistency.
    """

    _stata_command = "teffects aipw"

    def fit(self) -> TreatmentEffectResult:
        y, treated, X = self._y, self._treated, self._X
        n = y.size
        propensity = np.clip(self._propensity(), 1e-6, 1 - 1e-6)
        arm = treated > 0.5

        beta1 = _ols(X[arm], y[arm])
        beta0 = _ols(X[~arm], y[~arm])
        fitted1 = X @ beta1
        fitted0 = X @ beta0

        aug1 = fitted1 + treated * (y - fitted1) / propensity
        aug0 = fitted0 + (1.0 - treated) * (y - fitted0) / (1.0 - propensity)

        if self._estimand == "ate":
            mean1 = float(aug1.mean())
            mean0 = float(aug0.mean())
            estimate = mean1 - mean0
            influence = (aug1 - aug0) - (aug1 - aug0).mean()
        else:
            share = arm.mean()
            mean1 = float(aug1[arm].mean())
            mean0 = float(aug0[arm].mean())
            estimate = mean1 - mean0
            influence = treated * ((aug1 - aug0) - estimate) / share

        def standard_error(values: np.ndarray) -> float:
            return float(np.std(values, ddof=1) / np.sqrt(n))

        potential = {
            "0": (mean0, standard_error(aug0)),
            "1": (mean1, standard_error(aug1)),
        }

        return _finish(
            self._estimand.upper(),
            estimate,
            influence,
            potential,
            self._stata_command,
            se_method="influence function",
        )


class NearestNeighbourMatching:
    """Nearest-neighbour matching (Stata ``teffects nnmatch``).

    Matching has no likelihood, so its variance comes from the Abadie-Imbens
    matching variance rather than from any regression formula. Only the ATE with
    Euclidean distance and ``nneighbor`` matches is implemented; the bias-corrected
    and radius variants are refused rather than approximated.
    """

    _stata_command = "teffects nnmatch"

    def __init__(
        self,
        y: np.ndarray,
        treated: np.ndarray,
        X: np.ndarray,
        *,
        nneighbor: int = 1,
        estimand: str = "ate",
    ) -> None:
        self._y = y
        self._treated = treated
        self._X = X
        if int(nneighbor) < 1:
            raise ValueError(f"nnmatch: nneighbor must be >= 1, got {nneighbor}")
        self._nneighbor = int(nneighbor)
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
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        if covariates is None:
            covariates = [
                column
                for column in data.columns
                if column not in (outcome, treatment)
            ]
        y, treated, X = _prepare(outcome, treatment, data, list(covariates))
        return cls(y, treated, X, nneighbor=nneighbor, estimand=estimand)

    def fit(self) -> TreatmentEffectResult:
        y, treated, X = self._y, self._treated, self._X
        arm = treated > 0.5
        if arm.all() or not arm.any():
            raise ConsistencyError(
                "teffects nnmatch: one treatment arm is empty, so nothing can be "
                "matched"
            )

        # Exclude the intercept from the distance metric: it is constant and would
        # add nothing but noise to the standardisation.
        design = X[:, 1:] if X.shape[1] > 1 else X
        scale = design.std(axis=0, ddof=1)
        scale[scale == 0] = 1.0
        standardised = (design - design.mean(axis=0)) / scale

        def impute(source: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
            matches = np.empty(target.size, dtype=float)
            for position, index in enumerate(target):
                distances = np.sum(
                    (standardised[source] - standardised[index]) ** 2, axis=1
                )
                nearest = source[np.argsort(distances)[:k]]
                matches[position] = y[nearest].mean()
            return matches

        treated_index = np.flatnonzero(arm)
        control_index = np.flatnonzero(~arm)

        # Counterfactual for the treated comes from controls, and vice versa.
        counterfactual0 = impute(control_index, treated_index, self._nneighbor)
        counterfactual1 = impute(treated_index, control_index, self._nneighbor)

        effect_treated = y[treated_index] - counterfactual0
        effect_control = counterfactual1 - y[control_index]

        if self._estimand == "atet":
            estimate = float(effect_treated.mean())
            influence = np.zeros(y.size)
            influence[treated_index] = effect_treated - estimate
            share = treated_index.size / y.size
            influence = influence / share
        else:
            # The ATE averages the treated's effect and the controls' effect.
            estimate = float(
                (effect_treated.sum() + effect_control.sum()) / y.size
            )
            influence = np.zeros(y.size)
            influence[treated_index] = effect_treated - estimate
            influence[control_index] = effect_control - estimate

        if self._estimand == "atet":
            potential0 = float(counterfactual0.mean())
            potential1 = float(y[treated_index].mean())
        else:
            potential0 = float(
                (y[control_index].sum() + counterfactual0.sum()) / y.size
            )
            potential1 = float(
                (y[treated_index].sum() + counterfactual1.sum()) / y.size
            )

        standard_error = float(np.std(influence, ddof=1) / np.sqrt(y.size))
        potential = {
            "0": (potential0, standard_error),
            "1": (potential1, standard_error),
        }

        return _finish(
            self._estimand.upper(),
            estimate,
            influence,
            potential,
            self._stata_command,
            se_method="Abadie-Imbens approximation",
        )
