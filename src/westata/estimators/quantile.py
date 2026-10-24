"""Quantile and robust regression.

===================  ===============
westata              Stata
===================  ===============
``QuantileRegression``  ``qreg``
``RobustRegression``    ``rreg``
===================  ===============

Neither estimator is maximum likelihood, so neither can reuse
:mod:`westata.estimators.discrete`. ``qreg`` minimises the sum of the check
function; ``rreg`` iteratively reweights least squares.

Quantile regression
-------------------
The check function for quantile ``tau`` is

    rho_tau(u) = (tau - 1{u < 0}) u

and the estimator minimises ``sum rho_tau(y_i - x_i'b)``. Because the objective is
piecewise linear, several parameter vectors can attain the same minimum; Stata
warns "alternate solutions exist" when it notices. westata solves with IRLS,
which converges to one of the minimisers.

Standard errors follow ``[R] qreg`` Methods and formulas:

    cov(b_tau) = tau (1 - tau) / f_Y(xi_tau)^2 * (E[x x'])^-1

The density at the quantile is the *sparsity* ``s_tau = 1/f``.  Stata's default
estimator is ``fitted``, **not** a kernel density of the residuals: it estimates
the regression at ``tau + h`` and ``tau - h``, forms ``xbar'b`` at each, and takes

    s_tau = (xbar'b(tau+h) - xbar'b(tau-h)) / (2h)

with the Hall-Sheather bandwidth. Using a kernel density instead shifts every
standard error by roughly 29% on the reference DGP while leaving the coefficients
untouched -- which is why the three extra regressions below are not optional.

Two reporting details that are easy to get wrong:

* ``e(sum_adev)``/``e(sum_rdev)`` are **check-function** sums
  ``sum (tau - 1{u<0}) u``, not sums of ``|u|``.  The two coincide only at the
  median, so a median-only check cannot tell them apart; at ``tau=.25`` Stata
  prints 156.7522 against an absolute sum of 465.29.
* ``e(q_v)``, the raw unconditional quantile printed as "(about q)", is the
  ``ceil(n*tau)``-th order statistic here, not an interpolated quantile.

``vce(robust)`` is Stata's ``V = tau(1-tau) H (X'X) H`` with the
observation-specific fitted density baked into ``H`` (see
:meth:`QuantileRegression._robust_covariance`).  Both halves matter: the meat is
``tau(1-tau) X'X`` (not ``sum psi_i^2 x x'``, which only coincides with it at
``tau = .5``), **and** the bread keeps ``fi_i``.  Verified to 1.5e-8 at
``tau = .25/.5/.75`` on homoskedastic and heteroskedastic data.

Robust regression
-----------------
``rreg.ado`` (StataNow 19.5), in order:

1. Fit OLS, drop observations with Cook's D > 1, refit.
2. Huber weights until the largest weight change falls below ``5*tolerance``:
   ``w = 1`` if ``|e| <= 2*med(|e - med(e)|)`` else
   ``2*med(|e - med(e)|)/|e|``.
3. Biweight weights from that result, at least one pass, until the largest
   change falls below ``tolerance``: ``w = max(1-(e/(c_b s))^2, 0)^2`` with
   ``s = med(|e - med(e)|)/0.6745`` and ``c_b = 4.685*tune/7``.  ``tolerance``
   defaults to Stata's ``0.01``: reproducing that stopping rule is what puts the
   estimates on Stata's numbers (iterating to machine precision lands 2e-4 away
   in the coefficients, because Stata stops early).
4. ``aa = mean[(1-(e/s)^2/c_b^2)(1-5(e/s)^2/c_b^2)]`` over ``|e/s| <= c_b``,
   ``lambda = 1 + ((df_m+1)/N_w)*(1-aa)/aa`` where ``N_w`` is the count of
   **non-zero** biweights -- ``e(N)`` at that point in the program is the count
   of the last *weighted* regression, which drops zero-weight rows.  Then
   ``y_adj = xb + (lambda*s/aa)*(e/s)*w`` and an **unweighted** regression of
   ``y_adj`` on ``X``.
5. ``se(b) = e(rmse) * sqrt(diag((X'X)^-1))`` on the unweighted design, exactly
   as ``regress`` reports it.

All five steps are implemented, in Stata's statement order.  That order matters:
the ``scale`` used in step 4 is built from the *penultimate* residual (the loop
refreshes it before refitting and never after the last refit), so the adjustment
mixes a residual with the previous pass's scale.  Getting that off by one
iteration is invisible at ``tolerance(1e-9)`` but changes ``aa`` by 1.7e-4 at the
default ``tolerance(0.01)``.

The **pseudovalue** variance (Street-Carroll-Ruppert) is *not* what Stata
reports for ``rreg``: against ``e(rmse)*sqrt(diag((X'X)^-1))`` it is 15-30% away
on four datasets.  westata used to compute it; it now uses Stata's closed form,
which reproduces ``e(V)`` to 3e-11 and ``e(rss)/e(mse)/e(rmse)/e(r2)/e(F)`` to
1e-15 with ``rmse^2 (n-k) = rss`` holding by construction.

Deviations from the previous westata behaviour
----------------------------------------------
Three changes to ``RobustRegression`` alter numbers that existing callers may
have stored.  They are not side effects of the covariance fix; they are
deliberate, and each one is what Stata does:

1. **``tolerance`` now defaults to ``0.01``, Stata's default -- not to machine
   precision.**  The iteration loops stop when the largest change in the
   *weights* drops below ``5*tolerance`` (Huber) or ``tolerance`` (biweight), so
   ``e(iterations)`` is typically 5-10 rather than the hundreds a 1e-8 rule
   needed.  This is not a shortcut: iterating to convergence lands on a
   *different point*, **1.9e-4 relative in the coefficients** on the reference
   DGP, because Stata itself stops early, and matching Stata is the goal (the
   iteration count drops from 23 to 6).  Pass ``tolerance=1e-9`` to recover
   fully-converged estimates; those two answers differ from each other by that
   amount, so they are reported as distinct fits rather than blended into one.

2. **``e(mss)``, ``e(rss)``, ``e(mse)``, ``e(rmse)``, ``e(r2)`` and ``e(F)`` now
   describe the *unweighted regression of the adjusted response*
   ``y_adj = xb + (lambda*s/aa)(e/s)w``**, which is the regression Stata
   actually posts.  They used to be **weighted** sums of squares built from the
   final biweights, which is a different quantity -- on the reference DGP the old
   ``e(rss)`` was 330.3 where Stata reports 471.3.  ``e(mse)`` is new, and
   ``rmse^2 * (n-k) = rss`` now holds exactly.

3. **The Huber constant is ``2*median(|e - median(e)|)``, not ``1.345``.**  The
   two differ by the ``0.6745`` MAD-to-sigma factor (``1.345/0.6745 = 1.9941``,
   a 3e-3 relative difference in the threshold), so they define **different
   estimators** -- this is not a tuning constant that was merely re-expressed.
   ``rreg.ado`` writes ``abs(res) > 2*r(p50)`` with ``r(p50)`` the median of the
   absolute deviations from the median residual.  Holding the stopping rule
   fixed, swapping the threshold moves the coefficients by **7.9e-3 relative**
   on the reference DGP (``x1`` 2.0142 -> 2.0002), which is far larger than the
   covariance fix.

Two smaller surface changes ride along: ``fit`` gained a keyword-only
``tolerance`` parameter, and ``_pseudovalue_covariance`` is kept (``[R] rreg``
cites that estimator) but is no longer called by ``fit``.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, ConvergenceError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = [
    "QuantileRegression",
    "RobustRegression",
    "solve_lad",
]

#: Huber tuning constant, ``[R] rreg`` Methods and formulas.
_HUBER_C = 1.345
#: Biweight tuning constant at ``tune(7)``.
_BIWEIGHT_C = 4.685
#: Consistency factor turning the MAD into a scale estimate for normal errors.
_MAD_TO_SIGMA = 0.6745


def solve_lad(
    X: np.ndarray,
    y: np.ndarray,
    *,
    quantile: float = 0.5,
    maxiter: int = 2000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Minimise ``sum rho_tau(y - X b)`` by iteratively reweighted least squares.

    The weights are ``tau / |u|`` above the line and ``(1 - tau) / |u|`` below it,
    with a small floor so a zero residual does not divide by zero. The floor is a
    function of the current scale rather than a fixed constant, so it shrinks as
    the fit improves instead of pinning the step size.

    With ``X`` a single constant column the minimiser is a sample quantile of
    ``y``.  Which one is **not** unique: the objective is piecewise linear, and
    whenever the solution lands between two order statistics a whole interval of
    locations is optimal, so the returned value can differ from
    ``numpy.quantile`` by up to one order-statistic gap.  That is a property of
    the estimator, not a solver defect -- Stata's linear-programming solution has
    the same freedom and warns "alternate solutions exist" when it matters.

    Interpolation caveat: ``numpy.quantile`` interpolates *between* order
    statistics by default, while a check-function minimiser is always an order
    statistic (or an interval of them).  Tests must therefore compare with a
    tolerance of about one spacing, or use a plain type-1 quantile, rather than
    asserting equality.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(
            f"solve_lad: X has shape {X.shape} but y has {y.size} observations"
        )
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile!r}")

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    for _ in range(maxiter):
        residual = y - X @ beta
        scale = float(np.median(np.abs(residual)))
        floor = max(scale * 1e-8, 1e-12)
        weight = np.where(
            residual >= 0.0,
            quantile / np.maximum(np.abs(residual), floor),
            (1.0 - quantile) / np.maximum(np.abs(residual), floor),
        )
        root = np.sqrt(weight)
        weighted_X = X * root[:, None]
        weighted_y = y * root
        updated, *_ = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)

        change = float(np.max(np.abs(updated - beta)))
        beta = updated
        if change <= tol * max(1.0, float(np.max(np.abs(beta)))):
            break

    return beta


def _hall_sheather_bandwidth(quantile: float, nobs: int, alpha: float) -> float:
    """``[R] qreg`` Methods and formulas, the ``h_s`` formula."""

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    phi_q = float(stats.norm.pdf(stats.norm.ppf(quantile)))
    zq = float(stats.norm.ppf(quantile))
    inner = (1.5 * phi_q**2) / (2.0 * zq**2 + 1.0)
    return float(nobs ** (-1.0 / 3.0) * z ** (2.0 / 3.0) * inner ** (1.0 / 3.0))


def _bofinger_bandwidth(quantile: float, nobs: int) -> float:
    zq = float(stats.norm.ppf(quantile))
    phi_q = float(stats.norm.pdf(zq))
    inner = (4.5 * phi_q**4) / (2.0 * zq**2 + 1.0) ** 2
    return float(nobs ** (-1.0 / 5.0) * inner ** (1.0 / 5.0))


def _chamberlain_bandwidth(quantile: float, nobs: int, alpha: float) -> float:
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(z * np.sqrt(quantile * (1.0 - quantile) / nobs))


class QuantileRegression:
    """Quantile regression (Stata ``qreg``)."""

    _stata_command = "qreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        quantile: float = 0.5,
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
        if not 0.0 < quantile < 1.0:
            raise ValueError(
                f"qreg: quantile must lie strictly between 0 and 1, got {quantile!r}"
            )
        self._quantile = float(quantile)
        #: Estimation-sample covariates, retained for postestimation.
        self._model_data: pd.DataFrame | None = None
        #: Populated by ``fit`` with the fitted-sparsity components.
        self._sparsity: dict[str, Any] = {}

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        quantile: float = 0.5,
        subset: Any = None,
    ) -> "QuantileRegression":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"qreg: the left-hand side must be a plain column name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(y, X, quantile=quantile, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    # ------------------------------------------------------------- fitting
    def fit(
        self,
        cov_type: str | None = "iid",
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        reps: int = 0,
        maxiter: int = 2000,
        strict: bool = False,
    ) -> StataResults:
        normalized = "iid" if cov_type is None else str(cov_type).strip().lower()
        if normalized not in ("iid", "robust"):
            raise ValueError(
                f"qreg: cov_type must be 'iid' or 'robust', got {cov_type!r}"
            )

        n, k = self._X.shape
        if n <= k:
            raise ConsistencyError(
                f"qreg: {n} observations but {k} regressors; the model is not "
                "identified"
            )

        beta = solve_lad(self._X, self._y, quantile=self._quantile, maxiter=maxiter)
        residual = self._y - self._X @ beta

        # Stata's e(sum_adev) and e(sum_rdev) are values of the CHECK function
        # sum (tau - 1{u<0}) u, not sums of absolute deviations. Verified against
        # the fixture: qreg at tau=.25 prints 156.7522 and the check function
        # gives 156.7522 exactly, while the absolute sum gives 465.2928.
        tau = self._quantile

        def check(values: np.ndarray) -> float:
            return float(np.sum((tau - (values < 0).astype(float)) * values))

        sum_adev = check(residual)

        # The constant-only model's minimiser is the tau-th sample quantile;
        # Stata prints it as "Raw sum of deviations ... (about q_v)".
        raw_center = self._raw_quantile(tau)
        sum_rdev = check(self._y - raw_center)
        pseudo_r2 = 1.0 - sum_adev / sum_rdev if sum_rdev != 0 else float("nan")

        covariance = self._covariance(beta, normalized, n, k, residual)
        sparsity = self._sparsity["sparsity"]
        bandwidth = self._sparsity["bandwidth"]

        e: dict[str, Any] = {
            "N": n,
            "df_m": float(k - 1),
            "df_r": float(n - k),
            "q": self._quantile,
            "q_v": raw_center,
            "sum_adev": sum_adev,
            "sum_rdev": sum_rdev,
            "r2_p": pseudo_r2,
            "rank": k,
            "sparsity": sparsity,
            "f_r": 1.0 / sparsity if sparsity > 0 else float("nan"),
            "bwidth": bandwidth,
            "bwmethod": "hsheather",
            "denmethod": "fitted",
            "vce": normalized,
            "convcode": 0,
        }

        return StataResults(
            params=beta,
            param_names=list(self._param_names),
            cov=covariance,
            resid=residual,
            fittedvalues=self._X @ beta,
            nobs=n,
            df_resid=float(n - k),
            df_model=float(k - 1),
            dist="t",
            model_name=self._stata_command,
            cov_type=normalized,
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    # ---------------------------------------------------------- covariance
    def _covariance(
        self,
        beta: np.ndarray,
        cov_type: str,
        n: int,
        k: int,
        residual: np.ndarray,
    ) -> np.ndarray:
        self._sparsity = self._fitted_sparsity(n)
        if cov_type == "robust":
            return self._robust_covariance(beta, n, k)
        return self._iid_covariance(beta, n, k)

    def _fitted_sparsity(self, n: int) -> dict[str, Any]:
        """The ``fitted`` sparsity estimate, ``[R] qreg`` equation (4).

        Requires the two extra quantile regressions at ``tau ± h``; their
        ``xbar'b`` values give the inverse-quantile difference, and the shifted
        coefficient vectors are also what the robust sandwich needs at each
        observation, so they are cached here rather than recomputed.
        """

        tau = self._quantile
        # Two-sided 95%, matching Stata's default level(95).
        alpha = 0.05
        h = _hall_sheather_bandwidth(tau, n, alpha)
        # Keep both shifted quantiles inside (0, 1); past that a quantile
        # regression is undefined and Stata likewise refuses.
        h = min(h, 0.5 * min(tau, 1.0 - tau))
        if h <= 0.0:
            raise ConsistencyError(
                f"qreg: the Hall-Sheather bandwidth is not positive at quantile "
                f"{tau}; the sparsity estimate is undefined this close to 0 or 1"
            )

        beta_upper = solve_lad(self._X, self._y, quantile=float(tau + h))
        beta_lower = solve_lad(self._X, self._y, quantile=float(tau - h))
        xbar = np.mean(self._X, axis=0)
        sparsity = float((xbar @ beta_upper - xbar @ beta_lower) / (2.0 * h))
        if not np.isfinite(sparsity) or sparsity <= 0.0:
            raise ConsistencyError(
                "qreg: the fitted sparsity estimate is not positive; the "
                "quantile regression may not be identified on these data"
            )
        return {
            "sparsity": sparsity,
            "bandwidth": float(h),
            "beta_upper": beta_upper,
            "beta_lower": beta_lower,
        }

    def _iid_covariance(self, beta: np.ndarray, n: int, k: int) -> np.ndarray:
        """``tau(1-tau) s^2 (X'X/n)^-1 / n``, ``[R] qreg`` equation (5)."""

        tau = self._quantile
        sparsity = self._sparsity["sparsity"]
        xtx_inverse = np.linalg.pinv((self._X.T @ self._X) / n)
        return tau * (1.0 - tau) * sparsity**2 * xtx_inverse / n

    def _robust_covariance(self, beta: np.ndarray, n: int, k: int) -> np.ndarray:
        """Stata's ``vce(robust)``: ``tau(1-tau) J^-1 (X'X) J^-1``.

        ``qreg.ado`` builds the robust VCE in ``GetVCE`` from two matrices::

            V = tau*(1-tau) * H * J * H

        where ``J = X'X`` (``mat accum J = indep``) and ``H`` is the inverse of
        the observation-specific density-weighted cross-product that
        ``VCE_fitted`` returns under ``robust``::

            fi_i = (2h) / (xb_i(tau+h) - xb_i(tau-h))   -- zeroed when the
                   difference is not above sqrt(epsdouble), Stata's guard
            H    = inv( sum_i fi_i x_i x_i' )

        Two things this pins down, both verified live against StataNow 19.5 on
        homoskedastic **and** heteroskedastic data at tau = .25/.5/.75
        (``retest/fix/limited/repro_qreg.py``):

        * the **meat is** ``tau(1-tau) X'X``, not ``sum psi_i^2 x x'``.  The
          two coincide only at ``tau = .5``, where ``psi^2 == .25`` for every
          observation -- which is exactly why a median-only check could not see
          the difference (3.5%-128% off at other quantiles).
        * the **bread does** carry the per-observation density ``fi_i``; it does
          not cancel against the meat.  With Stata's own tau+-h fits this
          expression reproduces Stata's ``e(V)`` to ``7e-16`` at every
          quantile on both DGPs.

        The residual gap is the accuracy of our own ``solve_lad`` at
        ``tau +- h`` (our check-function value is 1e-9 above Stata's); it
        propagates at ~1e-8 into the standard errors.
        """

        tau = self._quantile
        h = self._sparsity["bandwidth"]
        difference = self._X @ (self._sparsity["beta_upper"] - self._sparsity["beta_lower"])
        # Stata: `replace fi = cond(fi>sqrt(c(epsdouble)), (2*band)/fi, 0)`
        threshold = float(np.sqrt(np.finfo(float).eps))
        with np.errstate(divide="ignore", invalid="ignore"):
            density_i = np.where(difference > threshold, (2.0 * h) / difference, 0.0)

        if not np.all(np.isfinite(density_i)):
            raise ConsistencyError(
                "qreg: the observation-specific fitted density is not finite, so "
                "the robust covariance is undefined"
            )

        jacobian = (self._X.T * density_i) @ self._X
        meat = tau * (1.0 - tau) * (self._X.T @ self._X)

        try:
            jacobian_inverse = np.linalg.inv(jacobian)
        except np.linalg.LinAlgError:
            jacobian_inverse = np.linalg.pinv(jacobian)
        return jacobian_inverse @ meat @ jacobian_inverse

    def _raw_quantile(self, tau: float) -> float:
        """The raw (unconditional) ``tau`` quantile of ``y``, as Stata prints it.

        Stata reports this as "Raw sum of deviations ... (about q_v)".  For the
        P8 fixture ``n * tau`` is an integer at all four quantiles and the
        printed value is then exactly the ``n*tau``-th order statistic -- not the
        average of the two neighbours that linear interpolation would give.  The
        order statistic at 1-based position ``ceil(n*tau)`` reproduces all four
        printed values.

        Interpolation caveat: every reference case has integer ``n*tau``, so the
        fixture cannot distinguish this rule from a neighbouring convention.  The
        choice is also numerically unimportant, because the check-function sum is
        flat in the centre: ``sum_rdev``, and hence the pseudo-R2, comes out
        identical whether the raw centre is this order statistic, ``numpy``'s
        interpolated quantile, or the LAD constant-only estimate.
        """

        ordered = np.sort(self._y)
        position = int(np.ceil(ordered.size * tau))
        return float(ordered[min(max(position, 1), ordered.size) - 1])


class RobustRegression:
    """Robust regression (Stata ``rreg``)."""

    _stata_command = "rreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        tune: float = 7.0,
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
        if tune <= 0:
            raise ValueError(f"rreg: tune must be positive, got {tune!r}")
        self._tune = float(tune)
        self._model_data: pd.DataFrame | None = None
        self._dropped: np.ndarray = np.zeros(self._y.size, dtype=bool)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        tune: float = 7.0,
        subset: Any = None,
    ) -> "RobustRegression":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"rreg: the left-hand side must be a plain column name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(y, X, tune=tune, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    # ------------------------------------------------------------- fitting
    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 1000,
        tolerance: float = 0.01,
        strict: bool = False,
    ) -> StataResults:
        """Estimate by ``[R] rreg``'s Huber-then-biweight iteration.

        ``tolerance`` mirrors Stata's ``tolerance(#)`` option and defaults to
        Stata's own default of ``0.01``: the loops stop when the largest change
        in the *weights* falls below ``5*tolerance`` (Huber) or ``tolerance``
        (biweight).  Reproducing that stopping rule is what puts the estimates
        on Stata's numbers; iterating to machine precision instead lands on a
        slightly different point (2e-4 away in the coefficients) because Stata
        itself stops early.
        """

        n, k = self._X.shape
        if n <= k:
            raise ConsistencyError(
                f"rreg: {n} observations but {k} regressors; the model is not "
                "identified"
            )
        if tolerance <= 0:
            raise ValueError(f"rreg: tolerance must be positive, got {tolerance!r}")

        keep = self._cooks_distance_screen()
        X = self._X[keep]
        y = self._y[keep]
        self._dropped = ~keep

        state = self._iterate(X, y, tolerance=tolerance, maxiter=maxiter,
                             strict=strict)
        beta = state["beta"]
        residual = state["residual"]
        weights = state["weights"]
        scale = state["scale"]
        aa = state["aa"]
        # Stata's lambda divides by ``e(N)`` **at that point in the program**,
        # which is the observation count of the last *weighted* regression: rows
        # whose biweight is exactly zero are dropped by ``regress [aw=...]``, so
        # this is the number of non-zero weights, not the sample size.  Using
        # the full N instead is a 1e-4 error in ``e(rss)`` and every SE (verified
        # by solving the implied divisor: 3/m = 386.0000001 on n=400).
        effective_n = int(np.count_nonzero(weights))
        df_m = k - 1
        lam = 1.0 + ((df_m + 1) / effective_n) * (1.0 - aa) / aa

        # ``y_adj = xb + (lambda*scale/aa) * (res/scale) * w``.  The final
        # regression is **unweighted** on ``y_adj``; every reported sum of
        # squares, the rmse, R2 and F come from it, and so does the SE law
        # ``se(b) = e(rmse) * sqrt(diag((X'X)^-1))``.
        adjusted = X @ beta + (lam * scale / aa) * (residual / scale) * weights
        beta_final, *_ = np.linalg.lstsq(X, adjusted, rcond=None)
        residual_final = adjusted - X @ beta_final
        rss = float(residual_final @ residual_final)
        n_used = int(keep.sum())
        df_r = float(n_used - k)
        rmse = float(np.sqrt(rss / df_r)) if df_r > 0 else float("nan")
        # Stata's mss is the explained sum of squares of the adjusted regression
        # about the mean of the *adjusted* response.
        centred = adjusted - float(np.mean(adjusted))
        mss = float(centred @ centred - rss)
        total = mss + rss
        r2 = mss / total if total > 0 else float("nan")
        r2_adjusted = (
            1.0 - (1.0 - r2) * (n_used - 1) / df_r if df_r > 0 else float("nan")
        )
        f_stat = float("nan")
        if k > 1 and df_r > 0 and rss > 0:
            f_stat = (mss / (k - 1)) / (rss / df_r)

        # ``se(b) = e(rmse) sqrt(diag((X'X)^-1))`` -- the same law as ``regress``,
        # on the unweighted design and the adjusted response.  Reproduced to
        # 1e-16 on four datasets; the Street-Carroll-Ruppert pseudovalues this
        # used to compute are *not* what Stata reports for rreg (they are 15-30%
        # away).
        covariance = rmse**2 * np.linalg.pinv(X.T @ X)

        e: dict[str, Any] = {
            "N": n_used,
            "N_weighted": effective_n,
            "df_m": float(df_m),
            "df_r": df_r,
            "rank": k,
            "dropped": int(self._dropped.sum()),
            "scale": scale,
            "aa": aa,
            "lambda": lam,
            "tune": self._tune,
            "tolerance": tolerance,
            "mss": mss,
            "rss": rss,
            "mse": float(rss / df_r) if df_r > 0 else float("nan"),
            "r2": r2,
            "r2_a": r2_adjusted,
            "F": f_stat,
            "rmse": rmse,
            "huber_c": _HUBER_C,
            "biweight_c": _BIWEIGHT_C * self._tune / 7.0,
            "iterations": state["iterations"],
        }

        return StataResults(
            params=beta_final,
            param_names=list(self._param_names),
            cov=covariance,
            resid=self._y - self._X @ beta_final,
            fittedvalues=self._X @ beta_final,
            nobs=n_used,
            df_resid=df_r,
            df_model=float(df_m),
            dist="t",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    # --------------------------------------------------------------- steps
    def _iterate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        tolerance: float,
        maxiter: int,
        strict: bool,
    ) -> dict[str, Any]:
        """Huber then biweight, transcribed from ``rreg.ado`` step by step.

        The order of the statements matters and is preserved: each loop computes
        its weights from the residual of the previous pass, refits, and only then
        refreshes the residual and the median absolute deviation.  Consequently
        the ``scale`` used by the final adjustment is the one built from the
        *penultimate* residual -- Stata never recomputes it after the last
        refit.  Recomputing it (the natural-looking choice) moves ``aa`` by 1e-4.
        """

        n, k = X.shape
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ beta
        absdev = np.abs(residual - np.median(residual))
        weights = np.ones(n)
        iterations: list[tuple[str, int, float]] = []

        # -- Huber: w = 1 if |res| <= 2*mad else 2*mad/|res| -----------------
        iteration = 1
        largest = 1.0
        while largest > 5.0 * tolerance and iteration <= maxiter:
            mad = float(np.median(absdev))
            updated = np.where(
                np.abs(residual) > 2.0 * mad, 2.0 * mad / np.abs(residual), 1.0
            )
            largest = float(np.max(np.abs(updated - weights)))
            weights = updated
            root = np.sqrt(weights)
            beta, *_ = np.linalg.lstsq(X * root[:, None], y * root, rcond=None)
            residual = y - X @ beta
            absdev = np.abs(residual - np.median(residual))
            iterations.append(("huber", iteration, largest))
            iteration += 1
        else:
            if largest > 5.0 * tolerance:
                message = "rreg: Huber iterations did not converge"
                if strict:
                    raise ConvergenceError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        # -- Biweight: w = max(1-(res/(tune*scale))^2, 0)^2, at least one pass
        c_b = _BIWEIGHT_C * self._tune / 7.0
        largest = 1.0
        notyet = True
        scale = float(np.median(absdev)) / _MAD_TO_SIGMA
        while (largest > tolerance and iteration <= maxiter) or notyet:
            notyet = False
            # ``scale`` is computed at the TOP of Stata's loop body from the
            # residual of the previous refit, and the loop then overwrites the
            # residual.  On exit the retained ``scale`` therefore describes the
            # *penultimate* residual while ``res`` is the last one, and the
            # adjustment below deliberately mixes the two.  Refreshing the scale
            # one step too eagerly is invisible at tolerance(1e-9) but moves
            # ``aa`` by 1.7e-4 at Stata's default tolerance(0.01).
            scale_used = scale
            scaled = residual / scale_used
            updated = np.maximum(1.0 - (scaled / c_b) ** 2, 0.0) ** 2
            largest = float(np.max(np.abs(updated - weights)))
            weights = updated
            if not np.any(weights > 0):
                raise ConsistencyError(
                    "rreg: every observation received zero biweight; the "
                    "residual scale has collapsed"
                )
            root = np.sqrt(weights)
            beta, *_ = np.linalg.lstsq(X * root[:, None], y * root, rcond=None)
            residual = y - X @ beta
            absdev = np.abs(residual - np.median(residual))
            scale = float(np.median(absdev)) / _MAD_TO_SIGMA
            iterations.append(("biweight", iteration, largest))
            iteration += 1
        else:
            if largest > tolerance:
                message = "rreg: biweight iterations did not converge"
                if strict:
                    raise ConvergenceError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)

        # -- aa over |res/scale| <= tune -------------------------------------
        z = residual / scale_used
        aa_terms = (1.0 - z**2 / c_b**2) * (1.0 - 5.0 * z**2 / c_b**2)
        aa_terms = np.where(np.abs(z) > c_b, 0.0, aa_terms)
        aa = float(np.mean(aa_terms))
        return {
            "beta": beta,
            "residual": residual,
            "weights": weights,
            "scale": scale_used,
            "aa": aa,
            "iterations": len(iterations),
        }


    # --------------------------------------------------------------- steps
    def _cooks_distance_screen(self) -> np.ndarray:
        """Drop observations with Cook's D > 1, per ``[R] rreg``."""

        n, k = self._X.shape
        beta, *_ = np.linalg.lstsq(self._X, self._y, rcond=None)
        residual = self._y - self._X @ beta
        dof = max(n - k, 1)
        sigma2 = float(residual @ residual) / dof
        if sigma2 <= 0:
            return np.ones(n, dtype=bool)

        xtx_inverse = np.linalg.pinv(self._X.T @ self._X)
        leverage = np.einsum("ij,jk,ik->i", self._X, xtx_inverse, self._X)
        # Cook's D for a linear model.
        cooks = (
            residual**2
            / (k * sigma2)
            * (leverage / np.maximum(1.0 - leverage, 1e-12) ** 2)
        )
        keep = cooks <= 1.0
        if keep.sum() <= k:
            # Refuse to drop so much that the model becomes unidentified.
            warnings.warn(
                "rreg: the Cook's-distance screen would leave too few "
                "observations; keeping the full sample",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.ones(n, dtype=bool)
        return keep

    @staticmethod
    def _scale(residual: np.ndarray) -> float:
        """``s = M / 0.6745`` with ``M`` the MAD from the median residual."""

        centered = np.abs(residual - np.median(residual))
        return float(np.median(centered) / _MAD_TO_SIGMA)

    def _biweight_weights(self, residual: np.ndarray) -> np.ndarray:
        """The final biweight case weights at ``residual``.

        Used for the reported weighted sums of squares (``e(mss)``/``e(rss)``),
        which Stata computes from the weights that produced the estimate.
        """

        scale = self._scale(residual)
        if scale <= 0:
            return np.ones_like(residual)
        c_b = _BIWEIGHT_C * self._tune / 7.0
        scaled = residual / scale
        return np.where(np.abs(scaled) <= c_b, (1.0 - (scaled / c_b) ** 2) ** 2, 0.0)

    def _two_stage(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        start: np.ndarray | None = None,
        maxiter: int = 1000,
        strict: bool = False,
        tolerance: float = 0.01,
    ) -> np.ndarray:
        """The whole ``rreg`` estimator: Huber to convergence, then biweight."""

        beta = self._huber_until_converged(
            X, y, maxiter=maxiter, strict=strict, start=start, tolerance=tolerance
        )
        return self._biweight_until_converged(
            X, y, beta, maxiter=maxiter, strict=strict, tolerance=tolerance
        )

    def _huber_until_converged(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        maxiter: int = 1000,
        strict: bool = False,
        start: np.ndarray | None = None,
        tolerance: float = 0.01,
    ) -> np.ndarray:
        """Stata's Huber stage alone (``rreg.ado`` lines 86-120).

        The threshold is ``2*med(|e - med(e)|)`` in the units of the residual, not
        ``1.345 * scale``: ``rreg.ado`` writes ``abs(res) > 2*r(p50)`` where
        ``r(p50)`` is the median of the *absolute deviations from the median
        residual*.  The two differ by the ``0.6745`` consistency factor, so using
        1.345 here would be a different estimator (it was, previously).
        """

        if start is None:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        else:
            beta = np.asarray(start, dtype=float).copy()
        weight = np.ones(y.size, dtype=float)
        tolerance = float(tolerance)

        for iteration in range(maxiter):
            residual = y - X @ beta
            mad = float(np.median(np.abs(residual - np.median(residual))))
            if mad <= 0:
                break
            updated = np.where(
                np.abs(residual) > 2.0 * mad, 2.0 * mad / np.abs(residual), 1.0
            )
            change = float(np.max(np.abs(updated - weight)))
            weight = updated
            root = np.sqrt(weight)
            beta, *_ = np.linalg.lstsq(X * root[:, None], y * root, rcond=None)
            if change < 5.0 * tolerance:
                return beta
        else:
            message = "rreg: Huber iterations did not converge"
            if strict:
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return beta

    def _biweight_until_converged(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta: np.ndarray,
        *,
        maxiter: int = 1000,
        strict: bool = False,
        tolerance: float = 0.01,
    ) -> np.ndarray:
        """Stata's biweight stage alone, from the Huber estimate."""

        c_b = _BIWEIGHT_C * self._tune / 7.0
        weight = np.ones(y.size, dtype=float)
        tolerance = float(tolerance)
        largest = 1.0
        notyet = True
        passed = 0
        residual = y - X @ beta
        scale = self._scale(residual)

        while (largest > tolerance and passed < maxiter) or notyet:
            notyet = False
            passed += 1
            if scale <= 0:
                break
            scaled = residual / scale
            updated = np.where(
                np.abs(scaled) <= c_b, (1.0 - (scaled / c_b) ** 2) ** 2, 0.0
            )
            largest = float(np.max(np.abs(updated - weight)))
            weight = updated
            if not np.any(weight > 0):
                raise ConsistencyError(
                    "rreg: every observation received zero biweight; the "
                    "residual scale has collapsed"
                )
            root = np.sqrt(weight)
            beta, *_ = np.linalg.lstsq(X * root[:, None], y * root, rcond=None)
            residual = y - X @ beta
            scale = self._scale(residual)
        if largest > tolerance and passed >= maxiter:
            message = "rreg: biweight iterations did not converge"
            if strict:
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return beta

    # ---------------------------------------------------------- covariance
    def _pseudovalue_covariance(
        self, X: np.ndarray, y: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Street-Carroll-Ruppert pseudovalues -- **not** Stata's ``rreg`` VCE.

        Kept because ``[R] rreg`` cites this estimator and it is the right
        variance for the *sandwich* reading of the problem, but ``fit`` no longer
        calls it: Stata reports the plain ``regress`` law on the adjusted
        response, ``se(b) = e(rmse) sqrt(diag((X'X)^-1))``, which reproduces its
        ``e(V)`` to 3e-11.  These pseudovalues sit 15-30% away from it on four
        datasets, so using them was the defect, not a rounding difference.

        Each observation's pseudovalue is ``n*b - (n-1)*b_{-i}``, where
        ``b_{-i}`` re-estimates the same two-stage Huber-then-biweight estimator
        without observation ``i``.
        """

        n, k = X.shape
        if n <= k + 1:
            residual = y - X @ beta
            sigma2 = float(residual @ residual) / max(n - k, 1)
            return sigma2 * np.linalg.pinv(X.T @ X)

        pseudovalues = np.empty((n, k), dtype=float)
        for index in range(n):
            mask = np.ones(n, dtype=bool)
            mask[index] = False
            try:
                leave_one_out = self._two_stage(
                    X[mask], y[mask], start=beta, maxiter=200, strict=False
                )
            except (np.linalg.LinAlgError, ConsistencyError):
                leave_one_out = beta
            pseudovalues[index] = n * beta - (n - 1) * leave_one_out

        covariance = np.cov(pseudovalues, rowvar=False) / n
        covariance = np.atleast_2d(covariance)
        if not np.all(np.isfinite(covariance)):
            residual = y - X @ beta
            sigma2 = float(residual @ residual) / max(n - k, 1)
            covariance = sigma2 * np.linalg.pinv(X.T @ X)
        return covariance
