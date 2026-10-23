"""Panel non-linear models: binary, censored and count outcomes (P9a).

=============================  =========================  ==============================
westata                        Stata                      estimator
=============================  =========================  ==============================
``XtLogit``                    ``xtlogit, fe``            conditional (Chamberlain 1980)
``XtLogit``                    ``xtlogit, re``            random effects, GH quadrature
``XtLogit``                    ``xtlogit, pa``            population-averaged GEE
``XtProbit``                   ``xtprobit, re``           random effects, GH quadrature
``XtTobit``                    ``xttobit``                random-effects Tobit
``XtPoisson``                  ``xtpoisson, fe``          conditional (multinomial)
``XtPoisson``                  ``xtpoisson, re``          gamma mixture, closed form
``XtNbreg``                    ``xtnbreg, fe``            HHG quasi-likelihood
=============================  =========================  ==============================

Stata gives one command name to three genuinely different estimators, and
conflating them is the main way a reimplementation goes quietly wrong.  They
are therefore three separate objectives here, not three options on one:

``fe`` - conditional likelihood
    Condition each group on ``k_i = sum_t y_it``.  A group whose outcome never
    varies contributes a factor of one and is dropped, which is exactly why
    ``xtlogit y x1 x2, fe`` reports ``N = 912`` rather than 960 on the P9
    fixture panel (6 groups / 48 observations dropped).  Verified against
    ``p9_outputs.txt`` lines 1-28: this module reproduces Stata's
    ``.3307537`` / ``-.3134259``, standard errors ``.0807304`` / ``.0795823``,
    conditional log likelihood ``-387.92362`` and ``LR chi2(2) = 32.46``.

``re`` - integrated likelihood
    ``L_i = integral prod_t F(y_it, x_it'b + nu_i) phi(nu_i/sigma_u)/sigma_u dnu_i``

    Stata's default integration method is ``mvaghermite`` (adaptive
    Gauss-Hermite with Naylor-Smith posterior centring), not plain
    Gauss-Hermite; :func:`_adaptive_log_integral_batch` implements the adaptive rule
    as printed in ``[XT] xtlogit`` Methods and formulas.  ``intpoints(12)`` is
    Stata's default and this module's default.  Verified: ``xtlogit, re``
    ``.3428887`` / ``-.2896591`` / ``.1685264`` to 3e-6 and ``xtprobit, re``
    ``.2098340`` / ``-.1765590`` / ``.1023312`` to 6e-6.

``pa`` - GEE, *not* a likelihood
    Population-averaged estimating equations with an exchangeable working
    correlation (``xtlogit, pa`` is a convenience wrapper for
    ``xtgee, family(binomial) link(logit) corr(exchangeable)``).  There is no
    likelihood, so ``e(ll)``, ``e(ll_0)``, ``e(r2_p)``, ``e(aic)`` and
    ``e(bic)`` are deliberately **absent** from the result and the model test
    is a Wald statistic.  Reporting a "log-likelihood" for a GEE would be a
    fabricated number.

``xtnbreg, fe`` is the other estimator whose objective is not a likelihood:
Stata maximises the Hausman-Hall-Griliches (1984) *quasi*-likelihood, a
conditional multinomial kernel.  It is labelled ``e(quasi_likelihood) = 1`` and
``e(ll)`` carries the conditional Poisson log-likelihood at the estimates - the
same value Stata prints (``-775.44069`` vs ``xtpoisson, fe``'s ``-775.44068``).
See :class:`XtNbreg`.

Numerical notes
---------------
* Standard errors are the observed information (the negative numerical Hessian
  of the log-likelihood), which is Stata's default ``vce(oim)``.
* The conditional fixed-effects likelihoods have **no intercept**: the group
  effect absorbs it, and the likelihood is exactly flat in a constant.  A patsy
  ``Intercept`` column is therefore removed rather than estimated.  The previous
  version of this module reported it as ``NaN`` with a ``NaN`` covariance row,
  which corrupts any downstream ``test``/``lincom``/``predict`` call; dropping
  the column is the honest option.
* ``xtpoisson, re`` integrates the gamma random effect in closed form (the
  gamma is conjugate to the Poisson), so no quadrature is involved and
  ``intpoints`` is ignored for it.
* ``xttobit``'s published defaults use Stata's own ``intpoints(12)``, but 12
  points do not reach Stata's accuracy on the fixture panel; see
  :class:`XtTobit` for the measured table and pass ``intpoints=50`` for a
  1e-6 match.

Known gaps (measured, not silently approximated)
------------------------------------------------
* **``xtlogit, pa`` coefficients do not match Stata.**  On the fixture panel
  Stata reports ``.3007615`` / ``-.2542736`` / ``.1488734``; this module's GEE
  converges to ``.2981533`` / ``-.2523210`` / ``.1481984`` (gap 2.6e-3 /
  2.0e-3 / 6.8e-4).  Stata's ``pa`` standard errors here are close to the
  GEE *model-based* ones computed at an exchangeable ``alpha`` of about .117,
  which is neither the Pearson nor the deviance estimate this module computes;
  the residual gap is reported rather than tuned away.  The result is
  nevertheless an honest GEE and reports no likelihood.
* **``xtnbreg, fe``'s intercept.**  Stata prints ``_cons = 15.21552`` with a
  standard error of ``641``.  That is not an estimate - a constant is absorbed
  by the fixed effect and the objective is flat in it - so none is reported.
  The slope agrees to ``1.1e-5``.
* Robust/cluster VCEs are not implemented.  ``cov_type="robust"`` is available
  only for the GEE, where the sandwich *is* the estimator's own variance;
  anything else raises :class:`~westata.errors.UnsupportedError` rather than
  returning the observed-information matrix under a robust label.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from westata.errors import ConsistencyError, ConvergenceError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["XtLogit", "XtNbreg", "XtPoisson", "XtProbit", "XtTobit"]

_LOG_2PI = float(np.log(2.0 * np.pi))
#: Largest exponent handed to ``exp``.  Past this the density is numerically
#: zero; clipping keeps the optimiser out of NaN land instead of aborting.
_ETA_CLIP = 700.0
#: Tolerance for "this outcome is at the censoring limit".
_LIMIT_TOL = 1e-8

_CONSTANT_NAMES = frozenset({"intercept", "const", "_cons", "1"})


# --------------------------------------------------------------------- helpers


def _stata_names(names: list[str]) -> list[str]:
    """Rename parameters to Stata's spellings for the panel models.

    ``Intercept`` -> ``_cons``. A leading ``/`` is dropped from auxiliary parameters:
    Stata prints ``sigma_u``, ``sigma_e``, ``lnalpha`` and ``lnsig2u`` bare for
    ``xttobit``, ``xtpoisson`` and ``xtprobit`` -- the slash belongs to ``nl``, not to
    these commands. Both spellings resolve because ``StataResults`` matches on the
    canonical name and the old names remain unique prefixes.
    """

    mapped: list[str] = []
    for name in names:
        if name == "Intercept":
            mapped.append("_cons")
        elif name.startswith("/"):
            mapped.append(name[1:])
        else:
            mapped.append(name)
    return mapped


def _codes(values: Any) -> np.ndarray:
    """Dense integer codes for the panel identifier, in first-appearance order."""

    codes, _ = pd.factorize(pd.Series(np.asarray(values).reshape(-1)))
    return np.asarray(codes, dtype=int)


def _group_slices(codes: np.ndarray) -> list[np.ndarray]:
    """Row positions of every group, sorted within the group."""

    codes = np.asarray(codes, dtype=int).reshape(-1)
    order = np.argsort(codes, kind="stable")
    boundaries = np.flatnonzero(np.diff(codes[order])) + 1
    return [np.sort(block) for block in np.split(order, boundaries) if block.size]


def _constant_position(names: Sequence[str]) -> int | None:
    """Index of a patsy intercept column, or ``None`` when the model has none."""

    for position, name in enumerate(names):
        if str(name).strip().lower() in _CONSTANT_NAMES:
            return position
    return None


def _numerical_hessian(
    objective: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1e-5
) -> np.ndarray:
    """Central-difference Hessian with a step scaled to each coordinate."""

    point = np.asarray(point, dtype=float)
    size = point.size
    hessian = np.zeros((size, size))
    increments = step * np.maximum(np.abs(point), 1.0)
    center = objective(point)

    for i in range(size):
        ei = np.zeros(size)
        ei[i] = increments[i]
        for j in range(i, size):
            ej = np.zeros(size)
            ej[j] = increments[j]
            if i == j:
                value = (
                    objective(point + ei) - 2 * center + objective(point - ei)
                ) / increments[i] ** 2
            else:
                value = (
                    objective(point + ei + ej)
                    - objective(point + ei - ej)
                    - objective(point - ei + ej)
                    + objective(point - ei - ej)
                ) / (4 * increments[i] * increments[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def _covariance(
    negative: Callable[[np.ndarray], float], point: np.ndarray, *, command: str
) -> np.ndarray:
    """Inverse of the negative Hessian of ``negative`` (the negated log-lik)."""

    hessian = _numerical_hessian(negative, point)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
    if not np.all(np.isfinite(covariance)):
        raise ConvergenceError(
            f"{command}: the observed information matrix is not invertible; the "
            "model is likely not identified (perfect prediction, collinearity, "
            "or a degenerate variance component)"
        )
    return covariance


def _gauss_hermite(points: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite abscissas and weights, "physicists' convention".

    ``np.polynomial.hermite.hermgauss`` implements
    ``integral exp(-x^2) h(x) dx ~ sum_m w_m h(a_m)``, which is the convention
    Stata's ``mvaghermite`` formula is written in (``[XT] xtlogit`` Methods and
    formulas), so the weights are used unmodified.
    """

    points = int(points)
    if points < 2:
        raise ConsistencyError(
            f"intpoints must be at least 2 quadrature points; got {points}"
        )
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)


def _log_combinations(size: int, count: int) -> float:
    """``log C(size, count)``."""

    return float(
        special.gammaln(size + 1.0)
        - special.gammaln(count + 1.0)
        - special.gammaln(size - count + 1.0)
    )


def _pad_groups(
    groups: Sequence[np.ndarray], length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ragged group row-positions into ``(G, T_max)`` plus a ``0/1`` mask.

    The panel likelihoods are products over each group's periods, so every group
    can be evaluated in one vectorised call if the shorter groups are padded and
    the padding is masked out.  Without this the fit spends all its time in a
    Python loop over panels, which matters: on the fixture panel a single
    ``xttobit`` finite-difference Hessian evaluates the objective ~40 times over
    100 groups.
    """

    width = max(group.size for group in groups)
    rows = np.zeros((len(groups), width), dtype=int)
    mask = np.zeros((len(groups), width), dtype=float)
    for index, group in enumerate(groups):
        rows[index, : group.size] = group
        mask[index, : group.size] = 1.0
    # Padding repeats each group's last row so ``eta[rows]`` never indexes out of
    # range; the mask zeroes those terms out before any summation.
    for index, group in enumerate(groups):
        if group.size < width:
            rows[index, group.size :] = group[-1]
    return rows, mask


def _adaptive_log_integral_batch(
    conditional_loglik: Callable[[np.ndarray], np.ndarray],
    sigma: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
    *,
    maxiter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """Adaptive Gauss-Hermite integration over many groups at once.

    ``conditional_loglik`` maps a ``(G, M)`` matrix of random-effect draws to the
    ``(G, M)`` matrix of per-group conditional log likelihoods.  All groups share
    one quadrature rule but each keeps its own posterior centre and scale, which
    is exactly Stata's per-panel adaptation.
    """

    prior_const = -np.log(sigma) - 0.5 * _LOG_2PI
    root_two = float(np.sqrt(2.0))
    log_weights = np.log(weights)

    def log_integrand(tau: np.ndarray) -> np.ndarray:
        return (
            conditional_loglik(tau)
            + prior_const
            - 0.5 * (tau / sigma) ** 2
            + nodes[None, :] ** 2
            + log_weights[None, :]
        )

    mu = np.zeros(int(n_groups))
    scale = np.ones(int(n_groups))
    for _ in range(maxiter):
        tau = root_two * scale[:, None] * nodes[None, :] + mu[:, None]
        logw = log_integrand(tau)
        peak = logw.max(axis=1, keepdims=True)
        weighted = np.exp(logw - peak)
        total = weighted.sum(axis=1)
        mu_new = (weighted * tau).sum(axis=1) / total
        variance = (weighted * tau * tau).sum(axis=1) / total - mu_new**2
        scale_new = np.sqrt(np.maximum(variance, 1e-12))
        settled = np.all(
            np.abs(mu_new - mu) <= tol * np.maximum(1.0, np.abs(mu_new))
        ) and np.all(np.abs(scale_new - scale) <= tol * np.maximum(1.0, scale_new))
        mu, scale = mu_new, scale_new
        if settled:
            break

    tau = root_two * scale[:, None] * nodes[None, :] + mu[:, None]
    logw = log_integrand(tau)
    peak = logw.max(axis=1, keepdims=True)
    integral = np.exp(logw - peak).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log(root_two * scale) + peak[:, 0] + np.log(integral)
    values[~np.isfinite(integral) | (integral <= 0.0)] = -np.inf
    return values


# ------------------------------------------------------------- pooled baselines
def _pooled_logit_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pooled logit by Newton, used for start values and ``e(ll_0)``."""

    beta = np.zeros(X.shape[1])
    for _ in range(200):
        eta = X @ beta
        mu = special.expit(eta)
        weight = np.maximum(mu * (1.0 - mu), 1e-10)
        bread = X.T @ (X * weight[:, None])
        score = X.T @ (y - mu)
        try:
            step = np.linalg.solve(bread, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(bread) @ score
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return beta


def _pooled_logit_loglike(X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ _pooled_logit_fit(X, y)
    return float(
        np.sum(np.where(y > 0.5, -np.logaddexp(0.0, -eta), -np.logaddexp(0.0, eta)))
    )


def _constant_only_logit(y: np.ndarray) -> float:
    """Log likelihood of the intercept-only logit on ``y``."""

    share = float(np.clip(np.mean(y), 1e-12, 1.0 - 1e-12))
    return float(np.sum(np.where(y > 0.5, np.log(share), np.log(1.0 - share))))


def _pooled_probit_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pooled probit by Fisher scoring."""

    beta = np.zeros(X.shape[1])
    for _ in range(200):
        eta = X @ beta
        mu = np.clip(special.ndtr(eta), 1e-12, 1.0 - 1e-12)
        pdf = np.maximum(stats.norm.pdf(eta), 1e-12)
        variance = np.maximum(mu * (1.0 - mu), 1e-12)
        score = X.T @ (pdf * (y - mu) / variance)
        weight = np.maximum(pdf**2 / variance, 1e-12)
        bread = X.T @ (X * weight[:, None])
        try:
            step = np.linalg.solve(bread, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(bread) @ score
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return beta


def _pooled_probit_loglike(X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ _pooled_probit_fit(X, y)
    return float(
        np.sum(np.where(y > 0.5, special.log_ndtr(eta), special.log_ndtr(-eta)))
    )


def _poisson_start(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Log-OLS start values, the usual Poisson starting point."""

    beta, *_ = np.linalg.lstsq(X, np.log(np.maximum(y, 0.5)), rcond=None)
    return np.asarray(beta, dtype=float)


def _pooled_poisson_loglike(X: np.ndarray, y: np.ndarray) -> float:
    """Poisson log likelihood at the pooled ML fit (``xtpoisson, re``'s baseline)."""

    beta = _poisson_start(X, y)
    for _ in range(200):
        eta = np.clip(X @ beta, -_ETA_CLIP, _ETA_CLIP)
        mu = np.exp(eta)
        bread = X.T @ (X * mu[:, None])
        score = X.T @ (y - mu)
        try:
            step = np.linalg.solve(bread, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(bread) @ score
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    eta = np.clip(X @ beta, -_ETA_CLIP, _ETA_CLIP)
    mu = np.exp(eta)
    return float(np.sum(y * eta - mu - special.gammaln(y + 1.0)))


def _pooled_tobit_start(
    X: np.ndarray, y: np.ndarray, lower: float | None, upper: float | None
) -> np.ndarray:
    """OLS start for the slopes, which is where Stata starts ``xttobit``."""

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.asarray(beta, dtype=float)


def _tobit_observation_loglike(
    observed: np.ndarray,
    latent_mean: np.ndarray,
    sigma: float,
    lower: float | None,
    upper: float | None,
) -> np.ndarray:
    """Per-observation Tobit log likelihood at arbitrary latent means."""

    z = (observed - latent_mean) / sigma
    terms = -np.log(sigma) - 0.5 * _LOG_2PI - 0.5 * z**2
    if lower is not None:
        at_lower = observed <= lower + _LIMIT_TOL * max(1.0, abs(lower))
        terms = np.where(at_lower, special.log_ndtr((lower - latent_mean) / sigma), terms)
    if upper is not None:
        at_upper = observed >= upper - _LIMIT_TOL * max(1.0, abs(upper))
        terms = np.where(at_upper, special.log_ndtr((latent_mean - upper) / sigma), terms)
    return terms


def _pooled_tobit_loglike(
    X: np.ndarray, y: np.ndarray, lower: float | None, upper: float | None
) -> float:
    """Intercept-only Tobit log likelihood (Stata's ``e(ll_0)`` analogue)."""

    mean0 = float(np.mean(y))
    sigma0 = max(float(np.std(y)), 1e-3)

    def negative(theta: np.ndarray) -> float:
        mu, log_sigma = float(theta[0]), float(theta[1])
        sigma = float(np.exp(np.clip(log_sigma, -_ETA_CLIP, _ETA_CLIP)))
        terms = _tobit_observation_loglike(y, np.full(y.size, mu), sigma, lower, upper)
        return -float(np.sum(terms))

    result = optimize.minimize(
        negative, np.array([mean0, np.log(sigma0)]), method="BFGS",
        options={"maxiter": 2000},
    )
    return -float(result.fun)


# ------------------------------------------------------------------ the base
class _PanelBase:
    """Shared plumbing: sample construction, optimiser, assembly, ``e()``."""

    _stata_command = ""
    #: Stata estimator flags, in the order they should be reported.
    _methods: tuple[str, ...] = ("re",)
    _default_method = "re"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        groups: Any,
        *,
        param_names: Sequence[str],
        method: str,
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        X = np.asarray(exog, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != self._y.size:
            raise ConsistencyError(
                f"{self._stata_command}: the design matrix has {X.shape[0]} rows "
                f"but the response has {self._y.size}"
            )
        self._X = X
        self._groups = _codes(groups)
        if self._groups.size != self._y.size:
            raise ConsistencyError(
                f"{self._stata_command}: the panel identifier has "
                f"{self._groups.size} entries but there are {self._y.size} "
                "observations"
            )
        self._param_names = [str(name) for name in param_names]
        if len(self._param_names) != X.shape[1]:
            raise ValueError(
                f"param_names has {len(self._param_names)} entries but exog has "
                f"{X.shape[1]} columns"
            )
        self._slices = _group_slices(self._groups)
        self._n_groups = len(self._slices)
        self._model_data = model_data
        self._method = str(method).strip().lower()

    # ------------------------------------------------------------- construction
    @classmethod
    def _build(
        cls, formula: str, data: pd.DataFrame, entity: str, subset: Any
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, pd.DataFrame]:
        """Build ``(y, X, names, entity_values, estimation_frame)``."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(formula, str) or "~" not in formula:
            raise ConsistencyError(
                f"{cls._stata_command}: formula must be two-sided (contain '~'), "
                f"got {formula!r}"
            )
        frame = data if subset is None else data.loc[subset]
        frame = frame.reset_index(drop=True)
        if entity not in frame.columns:
            raise ConsistencyError(
                f"{cls._stata_command}: column {entity!r} not found in data; "
                f"available columns: {list(frame.columns)}"
            )
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in frame.columns:
            raise ConsistencyError(
                f"{cls._stata_command}: left-hand side must be a plain column "
                f"name, got {lhs!r}"
            )
        X, names, kept = build_design(formula, frame)
        retained = frame.loc[kept].reset_index(drop=True)
        y = retained[lhs].to_numpy(dtype=float)
        return y, np.asarray(X, dtype=float), names, retained[entity].to_numpy(), retained

    @classmethod
    def _check_method(cls, method: str) -> str:
        normalized = str(method).strip().lower()
        if normalized not in cls._methods:
            supported = ", ".join(repr(name) for name in cls._methods)
            raise UnsupportedError(
                f"{cls._stata_command}: method={method!r} is not available; "
                f"supported: {supported}"
            )
        return normalized

    # ------------------------------------------------------------------- names
    @property
    def _model_name(self) -> str:
        """Stata spelling, e.g. ``"xtlogit, fe"`` or ``"xttobit"``."""

        if self._methods == (self._default_method,) and self._stata_command == "xttobit":
            return "xttobit"
        return f"{self._stata_command}, {self._method}"

    # ----------------------------------------------------------------- helpers
    def _slope_positions(self, names: Sequence[str]) -> list[int]:
        """Positions of all reported coefficients except the intercept."""

        constant = _constant_position(names)
        return [i for i in range(len(names)) if i != constant]

    def _drop_constant(self) -> tuple[np.ndarray, list[str]]:
        """Design matrix and names with a patsy intercept removed."""

        position = _constant_position(self._param_names)
        if position is None:
            return self._X, list(self._param_names)
        keep = [i for i in range(self._X.shape[1]) if i != position]
        return self._X[:, keep], [self._param_names[i] for i in keep]

    def _varying_groups(self) -> list[np.ndarray]:
        """Groups whose outcome is not constant (the ``fe`` conditioning set)."""

        return [g for g in self._slices if 0 < self._y[g].sum() < g.size]

    def _nonempty_count_groups(self) -> list[np.ndarray]:
        """Groups with a positive total (what conditioning on ``S_i`` requires)."""

        return [g for g in self._slices if self._y[g].sum() > 0.0]

    def _group_size_summary(self, groups: Sequence[np.ndarray]) -> dict[str, Any]:
        sizes = np.array([group.size for group in groups], dtype=float)
        return {
            "g_min": int(sizes.min()),
            "g_avg": float(sizes.mean()),
            "g_max": int(sizes.max()),
        }

    def _wald(
        self, params: np.ndarray, covariance: np.ndarray, positions: Sequence[int]
    ) -> tuple[float, float]:
        """Wald statistic over ``positions`` and its chi-squared p-value."""

        index = list(positions)
        if not index:
            return float("nan"), float("nan")
        block = np.asarray(covariance, dtype=float)[np.ix_(index, index)]
        coefficients = np.asarray(params, dtype=float)[index]
        try:
            statistic = float(coefficients @ np.linalg.solve(block, coefficients))
        except np.linalg.LinAlgError:
            statistic = float(coefficients @ np.linalg.pinv(block) @ coefficients)
        if not np.isfinite(statistic) or statistic < 0.0:
            return float("nan"), float("nan")
        return statistic, float(stats.chi2.sf(statistic, len(index)))

    # --------------------------------------------------------------- optimiser
    def _optimize(
        self,
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        *,
        maxiter: int,
        strict: bool,
        label: str,
    ) -> tuple[np.ndarray, bool, int]:
        """BFGS, polished with Nelder-Mead, mirroring the ``tobit`` adapter."""

        result = optimize.minimize(
            objective,
            np.asarray(start, dtype=float),
            method="BFGS",
            options={"maxiter": maxiter, "gtol": 1e-10},
        )
        if not result.success:
            refined = optimize.minimize(
                objective,
                np.asarray(result.x, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-10, "fatol": 1e-10},
            )
            if refined.fun <= result.fun:
                result = refined

        converged = bool(result.success) and np.isfinite(float(result.fun))
        if not converged:
            message = (
                f"{self._model_name}: the {label} optimiser did not converge "
                f"({getattr(result, 'message', 'unknown reason')}); estimates may "
                "be unreliable"
            )
            if strict:
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return (
            np.asarray(result.x, dtype=float),
            converged,
            int(getattr(result, "nit", 0)),
        )

    # --------------------------------------------------------------- assembly
    def _assemble(
        self,
        params: np.ndarray,
        names: Sequence[str],
        covariance: np.ndarray,
        *,
        rows: np.ndarray,
        fitted: np.ndarray,
        llf: float,
        ll0: float,
        df_m: int,
        extra: Mapping[str, Any] | None = None,
        converged: bool = True,
        iterations: int = 0,
        model_name: str | None = None,
        cov_type: str = "nonrobust",
        regressor_names: Sequence[str] | None = None,
        n_groups: int | None = None,
    ) -> StataResults:
        """Wrap one fitted model into a :class:`StataResults`.

        ``llf`` and ``ll0`` are ``NaN`` for the GEE; that is what keeps
        ``e(ll)``/``e(ll_0)``/``e(r2_p)``/``e(aic)``/``e(bic)`` out of the
        container instead of inventing a likelihood.
        """

        params = np.asarray(params, dtype=float).reshape(-1)
        rows = np.asarray(rows, dtype=int).reshape(-1)
        fitted = np.asarray(fitted, dtype=float).reshape(-1)
        nobs = int(rows.size)
        # ``N_g`` is the number of panels the likelihood actually consumes: 114
        # for ``xtlogit, fe`` on the fixture panel, not the 120 in the frame,
        # because the 6 constant-outcome groups are dropped.  A subclass that
        # restricts the sample passes ``n_groups`` for exactly this reason.
        groups_fitted = int(self._n_groups if n_groups is None else n_groups)
        if fitted.size == self._y.size and nobs != self._y.size:
            # The conditional models fit on a subset but evaluate the linear
            # predictor on every row; restrict it to the estimation sample so the
            # residuals line up with ``e(N)``.
            fitted = fitted[rows]
        if fitted.size != nobs:
            raise ConsistencyError(
                f"{self._model_name}: {fitted.size} fitted values for {nobs} "
                "estimation observations"
            )

        e: dict[str, Any] = {
            "N": nobs,
            "N_g": groups_fitted,
            "df_m": int(df_m),
            "rank": int(params.size),
            "converged": int(converged),
            "ic": int(iterations),
        }
        if np.isfinite(llf):
            e["ll"] = float(llf)
        if np.isfinite(llf) and np.isfinite(ll0):
            e["ll_0"] = float(ll0)
            if ll0 != 0.0:
                e["r2_p"] = float(1.0 - llf / ll0)
            # The LR statistic is only the *model* test for the estimators that
            # compare against a restricted fit.  The Wald-based ones (``re``,
            # ``pa``) supply their own ``chi2``/``p`` through ``extra``, and an
            # LR difference would be a different, unlabelled statistic.
            e.setdefault("chi2", float(2.0 * (llf - ll0)))
        if np.isfinite(llf):
            k = params.size
            e["aic"] = float(-2.0 * llf + 2.0 * k)
            e["bic"] = float(-2.0 * llf + np.log(max(nobs, 1)) * k)
        if extra:
            # ``extra`` wins: a model that computed its own Wald statistic must
            # not have it overwritten by an unrelated LR difference.
            for key, value in dict(extra).items():
                e[key] = value
        if "chi2" in e and "p" not in e:
            e["p"] = (
                float(stats.chi2.sf(float(e["chi2"]), int(df_m))) if df_m > 0 else float("nan")
            )

        design_names = list(regressor_names if regressor_names is not None else names)
        # ``rows`` already restricts the design to the estimation sample, so the
        # stored matrix is the one the coefficients refer to (regressors only for
        # the conditional models, which drop the intercept).
        if len(design_names) == self._X.shape[1]:
            design = self._X[rows]
        else:
            design = self._drop_constant()[0][rows]
        return StataResults(
            params=params,
            param_names=_stata_names(list(names)),
            cov=np.asarray(covariance, dtype=float),
            resid=self._y[rows] - fitted,
            fittedvalues=fitted,
            nobs=nobs,
            df_resid=float(nobs),
            df_model=float(df_m),
            dist="z",
            model_name=model_name or self._model_name,
            cov_type=cov_type,
            e=e,
            design_matrix=(design, design_names),
            model_data=self._model_data,
        )


# ------------------------------------------------------------------- xtlogit
class XtLogit(_PanelBase):
    """Panel logit (Stata ``xtlogit``), ``fe`` / ``re`` / ``pa``.

    ``model_name`` is ``"xtlogit, fe"``, ``"xtlogit, re"`` or ``"xtlogit, pa"``
    depending on ``method``; all three report ``dist="z"`.

    ``method="fe"``
        Chamberlain's conditional likelihood.  Groups with a constant outcome are
        dropped, ``e(N_drop)``/``e(N_group_drop)`` record how many observations
        and groups that costs, there is no intercept, and ``e(chi2type)`` is
        ``"LR"``.  ``e(ll_0)`` is the intercept-only logit on the *retained*
        observations, which is what makes ``e(chi2)`` reproduce Stata's
        ``LR chi2(2) = 32.46``.
    ``method="re"``
        Integrated likelihood with a Gaussian random effect, adaptive
        Gauss-Hermite.  ``/lnsig2u`` is a parameter in ``e(b)`` (as Stata stores
        it, and as this module labels it); ``e(sigma_u)`` and ``e(rho)`` hold
        the derived quantities.
    ``method="pa"``
        GEE with an exchangeable working correlation.  **Not a likelihood**: no
        ``e(ll)``, ``e(ll_0)``, ``e(r2_p)``, ``e(aic)``/``e(bic)``, and the model
        test is ``e(chi2type)="Wald"``.  ``e(estimator)="gee"`` marks it.
    """

    _stata_command = "xtlogit"
    _methods = ("fe", "re", "pa")
    _default_method = "re"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        method: str = "re",
        subset: Any = None,
    ) -> "XtLogit":
        """Build the model from a formula (Stata ``xtlogit y x1 x2, fe``).

        ``time`` is accepted for symmetry with ``xtset`` and is not used: none of
        these likelihoods needs the periods to be sorted or evenly spaced.
        """

        normalized = cls._check_method(method)
        y, X, names, groups, model_data = cls._build(formula, data, entity, subset)
        return cls(
            y, X, groups, param_names=_stata_names(list(names)), method=normalized, model_data=model_data
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        intpoints: int = 12,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        """Estimate the model.

        Parameters
        ----------
        cov_type:
            ``None``/``"nonrobust"`` (observed information, Stata's default) or,
            for ``pa`` only, ``"robust"`` for the GEE sandwich.
        intpoints:
            Gauss-Hermite points for ``re`` (Stata default 12).  Ignored by
            ``fe``, which has no integral, and by ``pa``.
        maxiter:
            BFGS iteration cap (for ``pa``, the GEE alternation cap).
        strict:
            Raise :class:`ConvergenceError` instead of warning on non-convergence.
        """

        requested = self._check_cov(cov_type, cov_kwds, ssc)
        if self._method == "fe":
            return self._fit_conditional(maxiter=maxiter, strict=strict)
        if self._method == "pa":
            return self._fit_gee(
                maxiter=maxiter, strict=strict, robust=requested != "nonrobust"
            )
        return self._fit_random(intpoints=intpoints, maxiter=maxiter, strict=strict)

    def _check_cov(
        self, cov_type: str | None, cov_kwds: Mapping[str, Any] | None, ssc: Any
    ) -> str:
        requested = "nonrobust" if cov_type is None else str(cov_type).strip().lower()
        allowed = {"nonrobust"}
        if self._method == "pa":
            allowed |= {"robust", "hc0"}
        if requested not in allowed:
            raise UnsupportedError(
                f"{self._model_name}: cov_type={cov_type!r} is not implemented; "
                f"supported: {', '.join(sorted(allowed))}"
            )
        return requested

    # ------------------------------------------------------------------- fe
    def _fit_conditional(self, *, maxiter: int, strict: bool) -> StataResults:
        """Chamberlain's conditional likelihood, enumerated exactly.

        ``ln L_i = sum_{t: y_it=1} x_it'b - ln sum_{D: |D|=k_i} exp(sum_{t in D} x_it'b)``

        with ``D`` ranging over every ``k_i``-subset of the group's periods.
        ``T = 8`` makes the worst case ``C(8, 4) = 70`` terms, so exhaustive
        enumeration is both exact (no quadrature, no approximation) and cheap.
        """

        X, slope_names = self._drop_constant()
        y = self._y
        groups = self._varying_groups()
        if not groups:
            raise ConsistencyError(
                "xtlogit, fe: no group has variation in the outcome, so the "
                "conditional likelihood carries no information about the slopes"
            )

        dropped_obs = int(y.size - sum(group.size for group in groups))
        dropped_groups = int(self._n_groups - len(groups))

        # One indicator matrix per (group size, successes) pair, cached: the
        # denominator is then a single matrix product instead of a Python loop.
        cache: dict[tuple[int, int], np.ndarray] = {}

        def subset_indicators(size: int, successes: int) -> np.ndarray:
            key = (size, successes)
            if key not in cache:
                members = list(combinations(range(size), successes))
                indicator = np.zeros((len(members), size))
                for row, chosen in enumerate(members):
                    indicator[row, list(chosen)] = 1.0
                cache[key] = indicator
            return cache[key]

        def negative(beta: np.ndarray) -> float:
            eta = X @ beta
            total = 0.0
            for group in groups:
                outcomes = y[group]
                design = eta[group]
                successes = int(outcomes.sum())
                # logsumexp of the denominator, written out so a huge eta cannot
                # overflow before the max is subtracted.
                weights = subset_indicators(group.size, successes) @ design
                peak = float(weights.max())
                total += float(design[outcomes > 0.5].sum()) - float(
                    peak + np.log(np.exp(weights - peak).sum())
                )
            return -total

        beta, converged, iterations = self._optimize(
            negative, np.zeros(X.shape[1]), maxiter=maxiter, strict=strict,
            label="conditional-likelihood",
        )
        covariance = _covariance(negative, beta, command="xtlogit, fe")
        llf = -float(negative(beta))

        rows = np.sort(np.concatenate(groups))
        # Stata's LR chi2 compares against the *conditional* null model, i.e.
        # the same conditioning set fitted with only an intercept.  There the
        # beta terms cancel exactly and what is left is
        # ``-sum_i ln C(T_i, k_i)``, which reproduces Stata's
        # ``LR chi2(2) = 32.46`` from ``ll = -387.92362``.  Using a pooled logit
        # baseline instead would report chi2 = 485 on this panel.
        ll0 = -float(
            np.sum(
                [_log_combinations(group.size, int(y[group].sum())) for group in groups]
            )
        )
        df_m = len(slope_names)
        statistic = 2.0 * (llf - ll0)
        p_value = float(stats.chi2.sf(statistic, df_m)) if df_m > 0 else float("nan")

        extra: dict[str, Any] = {
            "chi2type": "LR",
            "chi2": statistic,
            "p": p_value,
            "N_drop": dropped_obs,
            "N_group_drop": dropped_groups,
            "estimator": "conditional_ml",
        }
        extra.update(self._group_size_summary(groups))
        return self._assemble(
            beta,
            slope_names,
            covariance,
            rows=rows,
            fitted=special.expit(X @ beta),
            llf=llf,
            ll0=ll0,
            df_m=df_m,
            extra=extra,
            converged=converged,
            iterations=iterations,
            regressor_names=slope_names,
            n_groups=len(groups),
        )

    # ------------------------------------------------------------------- re
    def _fit_random(self, *, intpoints: int, maxiter: int, strict: bool) -> StataResults:
        """Gaussian random effect, integrated by adaptive Gauss-Hermite.

        The sum over periods is taken inside each group and the groups are then
        integrated in one vectorised call (:func:`_adaptive_log_integral_batch`),
        which keeps the objective O(1) Python work per evaluation instead of
        O(n_groups).
        """

        nodes, weights = _gauss_hermite(intpoints)
        X, names = self._X, list(self._param_names)
        y = self._y
        k = X.shape[1]
        rows, mask = _pad_groups(self._slices, self._n_groups)
        outcomes = y[rows]
        positive = (outcomes > 0.5).astype(float)

        def negative(theta: np.ndarray) -> float:
            beta = theta[:k]
            lnsig2u = float(np.clip(theta[k], -_ETA_CLIP, _ETA_CLIP))
            sigma_u = float(np.exp(0.5 * lnsig2u))
            eta = (X @ beta)[rows][:, :, None]

            def conditional(tau: np.ndarray) -> np.ndarray:
                # ``eta`` is (G, T, 1) and ``tau`` is (G, M); the sum broadcasts
                # to (G, T, M) and the period axis is contracted below.
                z = eta + np.asarray(tau)[:, None, :]
                # log Lambda(z) and log(1 - Lambda(z)), both evaluated stably.
                terms = np.where(
                    positive[:, :, None] > 0.5,
                    -np.logaddexp(0.0, -z),
                    -np.logaddexp(0.0, z),
                )
                return (terms * mask[:, :, None]).sum(axis=1)

            return -float(
                np.sum(
                    _adaptive_log_integral_batch(
                        conditional, sigma_u, nodes, weights, rows.shape[0]
                    )
                )
            )

        start = np.zeros(k + 1)
        start[:k] = _pooled_logit_fit(X, y)
        theta, converged, iterations = self._optimize(
            negative, start, maxiter=maxiter, strict=strict, label="random-effects"
        )
        covariance = _covariance(negative, theta, command="xtlogit, re")
        llf = -float(negative(theta))

        lnsig2u = float(theta[k])
        sigma_u = float(np.exp(0.5 * lnsig2u))
        # Stata's rho for a logit is sigma_u^2 / (sigma_u^2 + pi^2/3), because
        # the logistic idiosyncratic error has variance pi^2/3 exactly.
        rho = float(sigma_u**2 / (sigma_u**2 + np.pi**2 / 3.0))
        positions = self._slope_positions(names)
        statistic, p_value = self._wald(theta, covariance, positions)

        extra: dict[str, Any] = {
            "chi2type": "Wald",
            "chi2": statistic,
            "p": p_value,
            "sigma_u": sigma_u,
            "rho": rho,
            "intpoints": int(intpoints),
            "intmethod": "mvaghermite",
            "estimator": "random_effects_ml",
        }
        return self._assemble(
            theta,
            names + ["/lnsig2u"],
            covariance,
            rows=np.arange(y.size),
            fitted=special.expit(X @ theta[:k]),
            llf=llf,
            ll0=_pooled_logit_loglike(X, y),
            df_m=len(positions),
            extra=extra,
            converged=converged,
            iterations=iterations,
        )

    # ------------------------------------------------------------------- pa
    def _fit_gee(
        self, *, maxiter: int, strict: bool, robust: bool = False
    ) -> StataResults:
        """Population-averaged GEE with an exchangeable working correlation.

        ``xtlogit, pa`` *is* ``xtgee, family(binomial) link(logit)
        corr(exchangeable)``: the coefficients solve estimating equations rather
        than maximising a likelihood.  Nothing here produces an ``e(ll)``.

        Two variances are available and they differ, so the choice is explicit
        rather than assumed:

        * ``cov_type="nonrobust"`` (the default) reports the **model-based**
          variance ``(sum_i D_i' V_i^-1 D_i)^-1``.  That is what Stata's plain
          ``xtlogit, pa`` prints, verified on the fixture panel:
          ``.0685429`` / ``.0659606`` / ``.0888305`` against Stata's
          ``.0685441`` / ``.0659616`` / ``.0888241``.
        * ``cov_type="robust"`` reports the cluster-robust sandwich
          ``A^-1 B A^-1`` with ``B = sum_i D_i' V_i^-1 (y_i-mu_i)(y_i-mu_i)'
          V_i^-1 D_i``, which is what ``vce(robust)`` asks for
          (``.0703630`` / ``.0623757`` / ``.0886319`` here).
        """

        X, names = self._X, list(self._param_names)
        y = self._y
        slices = self._slices
        k = X.shape[1]

        def correlation(group: np.ndarray, mu: np.ndarray, alpha: float):
            """Working covariance and its ``sqrt(mu(1-mu))`` scaling."""

            size = group.size
            sd = np.sqrt(np.maximum(mu * (1.0 - mu), 1e-12))
            working = np.full((size, size), alpha)
            np.fill_diagonal(working, 1.0)
            return working * np.outer(sd, sd), sd

        def solve_beta(alpha: float, beta0: np.ndarray, rounds: int = 200) -> np.ndarray:
            beta = np.asarray(beta0, dtype=float).copy()
            for _ in range(rounds):
                mu = special.expit(X @ beta)
                bread = np.zeros((k, k))
                score = np.zeros(k)
                for group in slices:
                    mg = mu[group]
                    working, sd = correlation(group, mg, alpha)
                    # dmu/dbeta = mu(1-mu) = sd^2, so the GEE design is sd^2 * X.
                    design = X[group] * (sd**2)[:, None]
                    try:
                        inverse = np.linalg.inv(working)
                    except np.linalg.LinAlgError:
                        inverse = np.linalg.pinv(working)
                    bread += design.T @ inverse @ design
                    score += design.T @ inverse @ (y[group] - mg)
                step = np.linalg.solve(bread, score)
                beta = beta + step
                if np.max(np.abs(step)) < 1e-12:
                    break
            return beta

        def estimate_alpha(beta: np.ndarray) -> float:
            """Method-of-moments exchangeable correlation from Pearson residuals."""

            mu = special.expit(X @ beta)
            sd = np.sqrt(np.maximum(mu * (1.0 - mu), 1e-12))
            residual = (y - mu) / sd
            numerator = 0.0
            denominator = 0.0
            for group in slices:
                rg = residual[group]
                outer = np.outer(rg, rg)
                numerator += float(outer.sum() - np.trace(outer))
                denominator += group.size * (group.size - 1)
            if denominator == 0.0:
                return 0.0
            return float(np.clip(numerator / denominator, -1.0, 1.0 - 1e-6))

        beta = _pooled_logit_fit(X, y)
        alpha = 0.0
        change = float("inf")
        iterations = 0
        for iterations in range(1, int(maxiter) + 1):
            alpha = estimate_alpha(beta)
            updated = solve_beta(alpha, beta)
            change = float(np.max(np.abs(updated - beta)))
            beta = updated
            if change < 1e-10:
                break
        converged = change < 1e-10
        if not converged:
            message = (
                f"{self._model_name}: the GEE alternation did not converge "
                f"(max coefficient change {change!r}); estimates may be unreliable"
            )
            if strict:
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        mu = special.expit(X @ beta)
        model_based = np.zeros((k, k))
        meat = np.zeros((k, k))
        for group in slices:
            mg = mu[group]
            working, sd = correlation(group, mg, alpha)
            design = X[group] * (sd**2)[:, None]
            try:
                inverse = np.linalg.inv(working)
            except np.linalg.LinAlgError:
                inverse = np.linalg.pinv(working)
            model_based += design.T @ inverse @ design
            residual = (y[group] - mg)[:, None]
            meat += design.T @ inverse @ (residual @ residual.T) @ inverse @ design

        bread = np.linalg.pinv(model_based)
        sandwich = bread @ meat @ bread
        chosen = sandwich if robust else bread
        variance_label = "robust" if robust else "model-based"
        statistic, p_value = self._wald(beta, chosen, self._slope_positions(names))

        # No llf/ll0 at all: this is GEE, and a fabricated likelihood would be
        # read as comparable to the fe/re numbers above.
        result = self._assemble(
            beta,
            names,
            chosen,
            rows=np.arange(y.size),
            fitted=mu,
            llf=float("nan"),
            ll0=float("nan"),
            df_m=len(self._slope_positions(names)),
            extra={
                "chi2type": "Wald",
                "chi2": statistic,
                "p": p_value,
                "corr": "exchangeable",
                "alpha": alpha,
                "family": "binomial",
                "link": "logit",
                "scale": 1.0,
                "vce": variance_label,
                "estimator": "gee",
                # Spelled out so no reader mistakes this for a likelihood.
                "quasi_likelihood": 1,
            },
            converged=converged,
            iterations=iterations,
            cov_type="robust" if robust else "nonrobust",
        )
        return result


# ------------------------------------------------------------------ xtprobit
class XtProbit(_PanelBase):
    """Panel probit (Stata ``xtprobit``), random effects only.

    The same integral as ``xtlogit, re`` with ``Phi`` in place of ``Lambda``.
    Stata offers no fixed-effects or population-averaged ``xtprobit`` in this
    shape (there is no sufficient statistic to condition on for the probit), so
    any other ``method`` raises :class:`~westata.errors.UnsupportedError`.
    """

    _stata_command = "xtprobit"
    _methods = ("re",)
    _default_method = "re"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        method: str = "re",
        subset: Any = None,
    ) -> "XtProbit":
        normalized = cls._check_method(method)
        y, X, names, groups, model_data = cls._build(formula, data, entity, subset)
        return cls(
            y, X, groups, param_names=_stata_names(list(names)), method=normalized, model_data=model_data
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        intpoints: int = 12,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        if cov_type is not None and str(cov_type).strip().lower() != "nonrobust":
            raise UnsupportedError(
                f"{self._model_name}: cov_type={cov_type!r} is not implemented; "
                "supported: nonrobust"
            )
        nodes, weights = _gauss_hermite(intpoints)
        X, names = self._X, list(self._param_names)
        y = self._y
        k = X.shape[1]
        rows, mask = _pad_groups(self._slices, self._n_groups)
        positive = (y[rows] > 0.5)[:, :, None].astype(float)

        def negative(theta: np.ndarray) -> float:
            beta = theta[:k]
            lnsig2u = float(np.clip(theta[k], -_ETA_CLIP, _ETA_CLIP))
            sigma_u = float(np.exp(0.5 * lnsig2u))
            eta = (X @ beta)[rows][:, :, None]

            def conditional(tau: np.ndarray) -> np.ndarray:
                z = eta + np.asarray(tau)[:, None, :]
                terms = np.where(
                    positive > 0.5, special.log_ndtr(z), special.log_ndtr(-z)
                )
                return (terms * mask[:, :, None]).sum(axis=1)

            return -float(
                np.sum(
                    _adaptive_log_integral_batch(
                        conditional, sigma_u, nodes, weights, rows.shape[0]
                    )
                )
            )

        start = np.zeros(k + 1)
        start[:k] = _pooled_probit_fit(X, y)
        theta, converged, iterations = self._optimize(
            negative, start, maxiter=maxiter, strict=strict, label="random-effects"
        )
        covariance = _covariance(negative, theta, command="xtprobit, re")
        llf = -float(negative(theta))

        sigma_u = float(np.exp(0.5 * theta[k]))
        # Probits standardise the idiosyncratic error to variance 1, so
        # rho = sigma_u^2 / (sigma_u^2 + 1).
        rho = float(sigma_u**2 / (sigma_u**2 + 1.0))
        positions = self._slope_positions(names)
        statistic, p_value = self._wald(theta, covariance, positions)

        return self._assemble(
            theta,
            names + ["/lnsig2u"],
            covariance,
            rows=np.arange(y.size),
            fitted=special.ndtr(X @ theta[:k]),
            llf=llf,
            ll0=_pooled_probit_loglike(X, y),
            df_m=len(positions),
            extra={
                "chi2type": "Wald",
                "chi2": statistic,
                "p": p_value,
                "sigma_u": sigma_u,
                "rho": rho,
                "intpoints": int(intpoints),
                "intmethod": "mvaghermite",
                "estimator": "random_effects_ml",
            },
            converged=converged,
            iterations=iterations,
        )


# ------------------------------------------------------------------- xttobit
class XtTobit(_PanelBase):
    """Random-effects Tobit (Stata ``xttobit``).

    ``y*_it = x_it'b + u_i + e_it`` with ``u_i ~ N(0, sigma_u^2)`` and
    ``e_it ~ N(0, sigma_e^2)``; the observed outcome is censored at ``lower``
    and/or ``upper``.  Both variance components are estimated and reported as
    ``/sigma_u`` and ``/sigma_e`` (Stata's own labels, and the scale Stata stores
    them on), with ``e(rho) = sigma_u^2/(sigma_u^2 + sigma_e^2)``.

    Quadrature accuracy
    -------------------
    ``intpoints=12`` is Stata's default and this module's default, and with the
    *adaptive* rule it is already sufficient here: on the P9 fixture panel
    (100 firms x 7 years, ``ll(.5)``) the estimates are stable to 7 digits for
    every ``intpoints`` from 8 upwards.

    ===========  ==========  ==========  ==========  ===========
    intpoints    x1          _cons       log L       sigma_u
    ===========  ==========  ==========  ==========  ===========
    8            .9020131    .9253713    -598.85059  .4883512
    10           .9020131    .9253712    -598.85058  .4883516
    12           .9020131    .9253712    -598.85058  .4883516
    30           .9020131    .9253712    -598.85058  .4883516
    50           .9020131    .9253712    -598.85058  .4883516
    Stata        .9020131    .9253712    -598.85058  .4883516
    ===========  ==========  ==========  ==========  ===========

    A *non-adaptive* rule is what needs 30+ points here; the adaptation is what
    buys the accuracy, not the node count.  That is why
    :func:`_adaptive_log_integral_batch` follows Stata's Naylor-Smith centring
    literally instead of reaching for more quadrature points.
    """

    _stata_command = "xttobit"
    _methods = ("re",)
    _default_method = "re"

    def __init__(
        self,
        *args: Any,
        lower: float | None = 0.0,
        upper: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if lower is None and upper is None:
            raise ConsistencyError(
                "xttobit: at least one censoring limit is required; with no "
                "censoring use a linear panel estimator (xtreg)"
            )
        if lower is not None and upper is not None and not float(lower) < float(upper):
            raise ConsistencyError(
                f"xttobit: the lower limit ({lower}) must be below the upper "
                f"limit ({upper})"
            )
        self._lower = None if lower is None else float(lower)
        self._upper = None if upper is None else float(upper)
        _, _, uncensored = self._censoring()
        if not uncensored.any():
            raise ConsistencyError(
                "xttobit: every observation is censored, so the likelihood is "
                "flat and nothing can be estimated"
            )

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        lower: float | None = 0.0,
        upper: float | None = None,
        method: str = "re",
        subset: Any = None,
    ) -> "XtTobit":
        normalized = cls._check_method(method)
        y, X, names, groups, model_data = cls._build(formula, data, entity, subset)
        return cls(
            y,
            X,
            groups,
            param_names=_stata_names(list(names)),
            method=normalized,
            model_data=model_data,
            lower=lower,
            upper=upper,
        )

    def _censoring(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = self._y
        left = np.zeros(y.size, dtype=bool)
        right = np.zeros(y.size, dtype=bool)
        if self._lower is not None:
            left = y <= self._lower + _LIMIT_TOL * max(1.0, abs(self._lower))
        if self._upper is not None:
            right = y >= self._upper - _LIMIT_TOL * max(1.0, abs(self._upper))
        return left, right, ~(left | right)

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        intpoints: int = 12,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        if cov_type is not None and str(cov_type).strip().lower() != "nonrobust":
            raise UnsupportedError(
                f"{self._model_name}: cov_type={cov_type!r} is not implemented; "
                "supported: nonrobust"
            )
        nodes, weights = _gauss_hermite(intpoints)
        X, names = self._X, list(self._param_names)
        y = self._y
        k = X.shape[1]
        lower, upper = self._lower, self._upper
        rows, mask = _pad_groups(self._slices, self._n_groups)
        outcomes = y[rows][:, :, None]

        def negative(theta: np.ndarray) -> float:
            beta = theta[:k]
            sigma_u = float(np.exp(np.clip(theta[k], -_ETA_CLIP, _ETA_CLIP)))
            sigma_e = float(np.exp(np.clip(theta[k + 1], -_ETA_CLIP, _ETA_CLIP)))
            eta = (X @ beta)[rows][:, :, None]

            def conditional(tau: np.ndarray) -> np.ndarray:
                latent = eta + np.asarray(tau)[:, None, :]
                terms = _tobit_observation_loglike(
                    outcomes, latent, sigma_e, lower, upper
                )
                return (terms * mask[:, :, None]).sum(axis=1)

            return -float(
                np.sum(
                    _adaptive_log_integral_batch(
                        conditional, sigma_u, nodes, weights, rows.shape[0]
                    )
                )
            )

        start = np.zeros(k + 2)
        start[:k] = _pooled_tobit_start(X, y, lower, upper)
        residual = y - X @ start[:k]
        start[k] = np.log(0.5)
        start[k + 1] = np.log(max(float(np.std(residual)), 1e-2))

        theta, converged, iterations = self._optimize(
            negative, start, maxiter=maxiter, strict=strict, label="random-effects"
        )
        covariance = _covariance(negative, theta, command="xttobit")
        llf = -float(negative(theta))

        sigma_u = float(np.exp(theta[k]))
        sigma_e = float(np.exp(theta[k + 1]))
        rho = float(sigma_u**2 / (sigma_u**2 + sigma_e**2))
        positions = self._slope_positions(names)
        statistic, p_value = self._wald(theta, covariance, positions)
        left, right, uncensored = self._censoring()

        # ``theta`` carries the variances on the log scale because that is what
        # keeps the optimiser unconstrained, but Stata reports ``/sigma_u`` and
        # ``/sigma_e`` as *standard deviations* (``.4883516`` / ``.6004406`` on
        # the fixture panel), so the reported vector is rescaled with the delta
        # method rather than the log parameters being relabelled.
        reported = np.asarray(theta, dtype=float).copy()
        reported[k] = sigma_u
        reported[k + 1] = sigma_e
        reported_cov = np.asarray(covariance, dtype=float).copy()
        jacobian = np.ones(reported.size)
        jacobian[k] = sigma_u
        jacobian[k + 1] = sigma_e
        reported_cov = reported_cov * np.outer(jacobian, jacobian)

        return self._assemble(
            reported,
            names + ["/sigma_u", "/sigma_e"],
            reported_cov,
            rows=np.arange(y.size),
            fitted=X @ theta[:k],
            llf=llf,
            ll0=_pooled_tobit_loglike(X, y, lower, upper),
            df_m=len(positions),
            extra={
                "chi2type": "Wald",
                "chi2": statistic,
                "p": p_value,
                "sigma_u": sigma_u,
                "sigma_e": sigma_e,
                "rho": rho,
                "lower": lower,
                "upper": upper,
                "n_uncensored": int(uncensored.sum()),
                "n_left": int(left.sum()),
                "n_right": int(right.sum()),
                "intpoints": int(intpoints),
                "intmethod": "mvaghermite",
                "estimator": "random_effects_ml",
            },
            converged=converged,
            iterations=iterations,
        )


# ----------------------------------------------------------------- xtpoisson
class XtPoisson(_PanelBase):
    """Panel Poisson (Stata ``xtpoisson``), ``fe`` / ``re``.

    ``method="fe"``
        Conditioning on the group total ``S_i = sum_t y_it`` removes ``alpha_i``
        exactly (the multinomial reduction), so this is a genuine likelihood:

            ``ln L = sum_i [ sum_t y_it x_it'b - S_i ln sum_t exp(x_it'b)
                      + ln S_i! - sum_t ln y_it! ]``

        Groups with ``S_i = 0`` carry no information and are dropped.  Verified
        against ``p9_outputs.txt`` lines 226-250: ``.5041803`` (se
        ``.0344352``), log likelihood ``-775.44068`` - every printed digit.
    ``method="re"``
        The gamma random effect is conjugate to the Poisson, so the group
        likelihood is available in closed form and no quadrature is involved:

            ``l_i = sum_t [y_it ln mu_it - ln y_it!]
                    + ln Gamma(S_i + 1/a) - ln Gamma(1/a) - (1/a) ln a
                    - (S_i + 1/a) ln(M_i + 1/a)``

        with ``M_i = sum_t mu_it`` and ``a`` the gamma dispersion
        (``Var = a``, Stata's ``alpha``).  Reported as ``/lnalpha``, which is
        how Stata stores it.  Verified: ``.4795509`` / ``.3485993`` /
        ``-2.073955``, log likelihood ``-1076.9058``.
    """

    _stata_command = "xtpoisson"
    _methods = ("fe", "re")
    _default_method = "re"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        method: str = "re",
        subset: Any = None,
    ) -> "XtPoisson":
        normalized = cls._check_method(method)
        y, X, names, groups, model_data = cls._build(formula, data, entity, subset)
        if np.any(y < 0.0):
            raise ConsistencyError(
                "xtpoisson: the dependent variable must be non-negative counts; "
                f"{int(np.sum(y < 0.0))} observation(s) are negative"
            )
        return cls(
            y, X, groups, param_names=_stata_names(list(names)), method=normalized, model_data=model_data
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        intpoints: int = 12,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        if cov_type is not None and str(cov_type).strip().lower() != "nonrobust":
            raise UnsupportedError(
                f"{self._model_name}: cov_type={cov_type!r} is not implemented; "
                "supported: nonrobust"
            )
        if self._method == "fe":
            return self._fit_conditional(maxiter=maxiter, strict=strict)
        return self._fit_random(maxiter=maxiter, strict=strict)

    def _fit_conditional(self, *, maxiter: int, strict: bool) -> StataResults:
        """The multinomial conditional likelihood (exact, no quadrature)."""

        X, slope_names = self._drop_constant()
        y = self._y
        groups = self._nonempty_count_groups()
        if not groups:
            raise ConsistencyError(
                "xtpoisson, fe: every group has a zero total, so the conditional "
                "likelihood carries no information"
            )
        dropped_obs = int(y.size - sum(group.size for group in groups))

        def negative(beta: np.ndarray) -> float:
            eta = X @ beta
            total = 0.0
            for group in groups:
                counts = y[group]
                design = eta[group]
                total_weight = float(counts.sum())
                peak = float(design.max())
                # counts @ design - total * logsumexp(design), max-shifted.
                total += float(counts @ design) - total_weight * float(
                    peak + np.log(np.exp(design - peak).sum())
                )
            return -total

        beta, converged, iterations = self._optimize(
            negative, np.zeros(X.shape[1]), maxiter=maxiter, strict=strict,
            label="conditional-likelihood",
        )
        covariance = _covariance(negative, beta, command="xtpoisson, fe")

        # e(ll) is Stata's scale, which includes the beta-free combinatorial term.
        constant = 0.0
        for group in groups:
            counts = y[group]
            constant += float(
                special.gammaln(counts.sum() + 1.0) - special.gammaln(counts + 1.0).sum()
            )
        llf = -float(negative(beta)) + constant

        rows = np.sort(np.concatenate(groups))
        statistic, p_value = self._wald(beta, covariance, range(len(slope_names)))
        extra: dict[str, Any] = {
            "chi2type": "Wald",
            "chi2": statistic,
            "p": p_value,
            "N_drop": dropped_obs,
            "N_group_drop": int(self._n_groups - len(groups)),
            "estimator": "conditional_ml",
        }
        extra.update(self._group_size_summary(groups))
        return self._assemble(
            beta,
            slope_names,
            covariance,
            rows=rows,
            fitted=np.exp(np.clip(X @ beta, -_ETA_CLIP, _ETA_CLIP)),
            llf=llf,
            ll0=float("nan"),
            df_m=len(slope_names),
            extra=extra,
            converged=converged,
            iterations=iterations,
            regressor_names=slope_names,
            n_groups=len(groups),
        )

    def _fit_random(self, *, maxiter: int, strict: bool) -> StataResults:
        """Gamma random effect, integrated in closed form."""

        X, names = self._X, list(self._param_names)
        y = self._y
        k = X.shape[1]

        def negative(theta: np.ndarray) -> float:
            beta = theta[:k]
            lnalpha = float(np.clip(theta[k], -_ETA_CLIP, _ETA_CLIP))
            alpha = float(np.exp(lnalpha))
            inverse_alpha = 1.0 / alpha
            mu = np.exp(np.clip(X @ beta, -_ETA_CLIP, _ETA_CLIP))
            total = 0.0
            for group in self._slices:
                counts = y[group]
                mg = mu[group]
                total_count = float(counts.sum())
                total_mean = float(mg.sum())
                total += float(
                    (counts * np.log(np.maximum(mg, 1e-300))).sum()
                    - special.gammaln(counts + 1.0).sum()
                    + special.gammaln(total_count + inverse_alpha)
                    - special.gammaln(inverse_alpha)
                    - inverse_alpha * np.log(alpha)
                    - (total_count + inverse_alpha) * np.log(total_mean + inverse_alpha)
                )
            return -total

        start = np.zeros(k + 1)
        start[:k] = _poisson_start(X, y)
        start[k] = -2.0
        theta, converged, iterations = self._optimize(
            negative, start, maxiter=maxiter, strict=strict, label="random-effects"
        )
        covariance = _covariance(negative, theta, command="xtpoisson, re")
        llf = -float(negative(theta))

        lnalpha = float(theta[k])
        alpha = float(np.exp(lnalpha))
        positions = self._slope_positions(names)
        statistic, p_value = self._wald(theta, covariance, positions)
        mu = np.exp(np.clip(X @ theta[:k], -_ETA_CLIP, _ETA_CLIP))

        return self._assemble(
            theta,
            names + ["/lnalpha"],
            covariance,
            rows=np.arange(y.size),
            fitted=mu,
            llf=llf,
            ll0=_pooled_poisson_loglike(X, y),
            df_m=len(positions),
            extra={
                "chi2type": "Wald",
                "chi2": statistic,
                "p": p_value,
                "alpha": alpha,
                "lnalpha": lnalpha,
                "estimator": "random_effects_ml",
                "intmethod": "gamma (closed form)",
            },
            converged=converged,
            iterations=iterations,
        )


# ------------------------------------------------------------------- xtnbreg
class XtNbreg(_PanelBase):
    """Conditional fixed-effects negative binomial (Stata ``xtnbreg, fe``).

    **This objective is a quasi-likelihood, not a likelihood.**  Stata maximises
    the Hausman-Hall-Griliches (1984) conditional kernel

        ``q_i(b) = sum_t y_it x_it'b - S_i ln sum_t exp(x_it'b)``

    over the slopes, which is algebraically the *conditional Poisson* kernel: the
    negative-binomial dispersion cancels from it, so ``xtnbreg, fe`` and
    ``xtpoisson, fe`` estimate the same slopes.  That is why Stata prints almost
    identical coefficients (``.5041909`` vs ``.5041803`` on the fixture panel)
    and an **identical** "Log likelihood" (``-775.44069`` vs ``-775.44068``) --
    the number under that label is the conditional Poisson log likelihood.

    What is recorded in ``e()`` accordingly:

    * ``e(quasi_likelihood) = 1`` and ``e(estimator) = "conditional_quasi_ml"``
      -- the objective is not a log likelihood.
    * ``e(ll)`` is the conditional Poisson log likelihood at the estimates,
      i.e. exactly the value Stata prints, not a separate NB quantity.
    * No intercept.  Stata prints ``_cons = 15.21552`` with a standard error of
      ``641`` on the fixture panel; a constant is absorbed by the fixed effect,
      so that is the optimiser drifting along a flat direction rather than an
      estimate, and none is reported.
    * No ``alpha``: the conditional kernel does not identify the dispersion, and
      Stata does not report one for ``fe`` either.  (The previous version of this
      module estimated ``/lnalpha`` here, which is why its slope was
      ``.494621`` -- 1.9e-2 off Stata instead of 1.1e-5.)
    """

    _stata_command = "xtnbreg"
    _methods = ("fe",)
    _default_method = "fe"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        method: str = "fe",
        subset: Any = None,
    ) -> "XtNbreg":
        normalized = cls._check_method(method)
        y, X, names, groups, model_data = cls._build(formula, data, entity, subset)
        if np.any(y < 0.0):
            raise ConsistencyError(
                "xtnbreg: the dependent variable must be non-negative counts; "
                f"{int(np.sum(y < 0.0))} observation(s) are negative"
            )
        if not np.allclose(y, np.round(y)):
            raise ConsistencyError(
                "xtnbreg: fixed-effects negative binomial requires integer counts; "
                "the data contain non-integer values"
            )
        return cls(
            y, X, groups, param_names=_stata_names(list(names)), method=normalized, model_data=model_data
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        intpoints: int = 12,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        if cov_type is not None and str(cov_type).strip().lower() != "nonrobust":
            raise UnsupportedError(
                f"{self._model_name}: cov_type={cov_type!r} is not implemented; "
                "supported: nonrobust"
            )
        X, slope_names = self._drop_constant()
        y = self._y
        groups = self._nonempty_count_groups()
        if not groups:
            raise ConsistencyError(
                "xtnbreg, fe: every group has a zero total, so the conditional "
                "quasi-likelihood carries no information"
            )
        dropped_obs = int(y.size - sum(group.size for group in groups))

        def negative(beta: np.ndarray) -> float:
            eta = X @ beta
            total = 0.0
            for group in groups:
                counts = y[group]
                design = eta[group]
                total_weight = float(counts.sum())
                peak = float(design.max())
                total += float(counts @ design) - total_weight * float(
                    peak + np.log(np.exp(design - peak).sum())
                )
            return -total

        beta, converged, iterations = self._optimize(
            negative, np.zeros(X.shape[1]), maxiter=maxiter, strict=strict,
            label="quasi-likelihood",
        )
        covariance = _covariance(negative, beta, command="xtnbreg, fe")

        constant = 0.0
        for group in groups:
            counts = y[group]
            constant += float(
                special.gammaln(counts.sum() + 1.0) - special.gammaln(counts + 1.0).sum()
            )
        llf = -float(negative(beta)) + constant

        rows = np.sort(np.concatenate(groups))
        statistic, p_value = self._wald(beta, covariance, range(len(slope_names)))
        extra: dict[str, Any] = {
            "chi2type": "Wald",
            "chi2": statistic,
            "p": p_value,
            "N_drop": dropped_obs,
            "N_group_drop": int(self._n_groups - len(groups)),
            "estimator": "conditional_quasi_ml",
            # Stata's own documentation calls this a quasi-likelihood; recording
            # it here keeps e(ll) from being read as a genuine likelihood value.
            "quasi_likelihood": 1,
            "quasi_likelihood_note": (
                "Hausman-Hall-Griliches conditional kernel; e(ll) is the "
                "conditional Poisson log likelihood, not an NB likelihood"
            ),
        }
        extra.update(self._group_size_summary(groups))
        return self._assemble(
            beta,
            slope_names,
            covariance,
            rows=rows,
            fitted=np.exp(np.clip(X @ beta, -_ETA_CLIP, _ETA_CLIP)),
            llf=llf,
            ll0=float("nan"),
            df_m=len(slope_names),
            extra=extra,
            converged=converged,
            iterations=iterations,
            regressor_names=slope_names,
            n_groups=len(groups),
        )
