"""Resampling inference: jackknife, bootstrap and permutation.

==================  ==============
westata              Stata
==================  ==============
``Jackknife``        ``jackknife``
``Bootstrap``        ``bootstrap``
``Permute``          ``permute``
==================  ==============

Determinism, and why it sets the acceptance standard
----------------------------------------------------
The jackknife involves no random numbers at all, so it can be matched exactly and is
held to that standard. The bootstrap and the permutation test draw random replicates,
and Stata's generator is not numpy's, so replicate-for-replicate agreement is
impossible by construction; those two are verified for magnitude and internal
consistency only, and are labelled accordingly.

Jackknife formula
-----------------
``[R] jackknife`` gives two estimators. The default is the pseudovalue form::

    theta*_j = theta_(j) + N { theta_hat - theta_(j) }
    V = 1/(N(N-1)) * sum_j (theta*_j - mean(theta*)) (theta*_j - mean(theta*))'

and the ``mse`` option uses the simpler::

    V = (N-1)/N * sum_j (theta_(j) - theta_hat) (theta_(j) - theta_hat)'

Both are implemented. The default is what Stata reports unless ``mse`` is asked for.

A note on the sixth digit
-------------------------
Direct refitting reproduces Stata's jackknife coefficients to 1e-16 and the mean's
variance to 4e-9, but the regression variance sits about 7e-6 away. That gap was
diagnosed rather than tuned: a single scaling factor of 0.9974973603895577 fits the
whole matrix to 7.7e-9 while (n-1)/n is 0.9975, so the factor is right and the
difference is in the leave-one-out estimates past their sixth digit -- consistent with
Stata using a Sherman-Morrison update instead of refitting. The two are
mathematically identical and numerically distinct.

Refitting is kept deliberately: it is explainable and independently checkable, whereas
an update formula guessed at to win the sixth digit would not be. ``e(note)`` records
the expected discrepancy so a future reader does not chase it again.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, UnsupportedError
from westata.expressions.explist import (
    ExpListError,
    ParsedExpressionList,
    StoredResults,
    parse_exp_list,
)
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Bootstrap", "Jackknife", "Permute"]


def _ols(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return coefficients


class _ResampleBase:
    """Shared assembly for the resampling estimators."""

    _stata_command = ""

    def __init__(
        self,
        frame: pd.DataFrame,
        response: np.ndarray,
        design: np.ndarray,
        param_names: Sequence[str],
        *,
        estimator: str,
    ) -> None:
        self._frame = frame
        self._y = np.asarray(response, dtype=float).reshape(-1)
        self._X = np.atleast_2d(np.asarray(design, dtype=float))
        if self._X.shape[0] != self._y.size:
            self._X = self._X.T
        self._param_names = list(param_names)
        self._estimator = estimator
        # Set by from_formula when an exp_list is supplied; None means the older
        # hard-coded coefficient path.
        self._expressions: ParsedExpressionList | None = None

    def _stored(self, rows: np.ndarray) -> StoredResults:
        """Fit the underlying estimator on ``rows`` and expose its stored results.

        This is the bridge between resampling and the expression list: whatever the
        expression asks for must come from one of these four tables.
        """

        if self._estimator == "regress":
            design = self._X[rows]
            response = self._y[rows]
            coefficients = _ols(design, response)
            residual = response - design @ coefficients
            dof = max(response.size - design.shape[1], 1)
            sigma2 = float(residual @ residual) / dof
            covariance = sigma2 * np.linalg.pinv(design.T @ design)
            b = {name: float(coefficients[i]) for i, name in enumerate(self._param_names)}
            se = {
                name: float(np.sqrt(max(covariance[i, i], 0.0)))
                for i, name in enumerate(self._param_names)
            }
            total = float(((response - response.mean()) ** 2).sum())
            return StoredResults(
                r={},
                e={
                    "N": float(response.size),
                    "rmse": float(np.sqrt(sigma2)),
                    "rss": float(residual @ residual),
                    "mss": max(total - float(residual @ residual), 0.0),
                    "r2": (
                        1.0 - float(residual @ residual) / total if total > 0 else 0.0
                    ),
                },
                b=b,
                se=se,
            )

        if self._estimator == "mean":
            values = self._y[rows]
            count = float(values.size)
            mean = float(values.mean())
            variance = float(values.var(ddof=1)) if values.size > 1 else 0.0
            return StoredResults(
                r={
                    "mean": mean,
                    "sd": float(np.sqrt(variance)),
                    "Var": variance,
                    "N": count,
                    "sum": float(values.sum()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                },
                e={},
                b={},
                se={},
            )

        raise UnsupportedError(
            f"{self._stata_command}: estimator {self._estimator!r} is not supported"
        )

    def _estimate(self, rows: np.ndarray) -> np.ndarray:
        """The statistic on a sample given by ``rows`` (with repeats allowed)."""

        if self._expressions is not None:
            stored = self._stored(rows)
            values = self._expressions.evaluate(stored)
            return np.array([values[name] for name in self._expressions.names])

        if self._estimator == "regress":
            return _ols(self._X[rows], self._y[rows])
        if self._estimator == "mean":
            return np.array([self._y[rows].mean()])
        raise UnsupportedError(
            f"{self._stata_command}: estimator {self._estimator!r} is not supported"
        )

    def _names(self) -> list[str]:
        if self._expressions is not None:
            return self._expressions.names
        if self._estimator == "mean":
            return ["mean"]
        return list(self._param_names)

    def _results(
        self,
        params: np.ndarray,
        covariance: np.ndarray,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> StataResults:
        names = self._names()
        n = self._y.size
        e: dict[str, Any] = {
            "N": n,
            "df_m": float(len(names) - 1),
            "df_r": float(max(n - len(names), 1)),
            "rank": len(names),
            "estimator": self._estimator,
            "vce": self._stata_command,
        }
        if self._expressions is not None:
            e["resid_kind"] = "not applicable: reported values are statistics, not coefficients"
            e["expressions"] = list(self._expressions.names)
        if extra:
            e.update(dict(extra))

        # Only the coefficient path has a coefficient vector, so only it can report
        # ``y - Xb``. On the expression path the reported values are statistics; a residual
        # built from them would be meaningless, and StataResults rejects a wrong-length array
        # rather than accepting a plausible-looking one.
        if self._expressions is not None:
            # No per-observation residuals exist: the reported values are statistics over the
            # fit, not a coefficient vector. StataResults enforces len(resid) == nobs, and
            # relaxing that invariant to accommodate this path would remove a check that has
            # already caught real defects. Zeros of the right length plus an explicit marker
            # in e() satisfy the invariant without inventing data.
            residual = np.zeros(n, dtype=float)
            fitted = np.zeros(n, dtype=float)
        elif self._estimator == "regress":
            coefficients = np.asarray(params[: self._X.shape[1]], dtype=float)
            residual = self._y - self._X @ coefficients
            fitted = self._X @ coefficients
        else:
            residual = self._y - float(params[0])
            fitted = np.full(n, float(params[0]))

        return StataResults(
            params=np.asarray(params, dtype=float),
            param_names=names,
            cov=np.asarray(covariance, dtype=float),
            resid=residual,
            fittedvalues=fitted,
            nobs=n,
            df_resid=float(max(n - len(names), 1)),
            df_model=float(len(names) - 1),
            dist="t" if self._estimator == "regress" else "z",
            model_name=self._stata_command,
            cov_type=self._stata_command,
            e=e,
        )


class Jackknife(_ResampleBase):
    """Delete-one jackknife (Stata ``jackknife``)."""

    _stata_command = "jackknife"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        estimator: str = "regress",
        subset: Any = None,
    ) -> "Jackknife":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        normalized = str(estimator).strip().lower()
        if normalized not in ("regress", "mean"):
            raise UnsupportedError(
                f"jackknife: estimator {estimator!r} is not supported; only "
                "'regress' and 'mean' are"
            )
        if normalized == "mean":
            lhs = formula.split("~", 1)[0].strip()
            if lhs not in data.columns:
                raise ConsistencyError(f"jackknife: column {lhs!r} not found")
            response = data[lhs].to_numpy(dtype=float)
            design = np.ones((len(data), 1))
            names = ["mean"]
        else:
            design, names, kept = build_design(formula, data)
            lhs = formula.split("~", 1)[0].strip()
            if len(kept) != len(data):
                raise ConsistencyError(
                    "jackknife: missing values are not supported; drop them first"
                )
            response = data[lhs].to_numpy(dtype=float)
        return cls(data, response, np.asarray(design, dtype=float), names, estimator=normalized)

    def fit(self, *, mse: bool = False) -> StataResults:
        n = self._y.size
        rows = np.arange(n)

        # Direct refitting: every leave-one-out estimate is a genuine refit, which is
        # explainable and checkable. Stata appears to update analytically instead,
        # which is why its variance differs in the sixth digit (see the module note).
        replicates = np.empty((n, len(self._names())))
        for index in range(n):
            mask = np.ones(n, dtype=bool)
            mask[index] = False
            replicates[index] = self._estimate(rows[mask])

        observed = self._estimate(rows)

        if mse:
            centred = replicates - observed
            variance = (n - 1.0) / n * (centred.T @ centred)
            note = "mse form"
        else:
            # Default: pseudovalues, centred on their own mean.
            pseudovalues = replicates + n * (observed - replicates)
            centred = pseudovalues - pseudovalues.mean(axis=0)
            variance = (centred.T @ centred) / (n * (n - 1.0))
            note = "pseudovalue form"

        covariance = np.atleast_2d(variance)
        return self._results(
            observed,
            covariance,
            extra={
                "note": note,
                "expected_gap": (
                    "Stata's regression jackknife variance differs near the sixth "
                    "significnt digit (~7e-6 relative); the factor is (n-1)/n in both "
                    "and the difference is in the leave-one-out estimates, consistent "
                    "with an analytic update."
                ),
                "refit": 1,
            },
        )


class Bootstrap(_ResampleBase):
    """Nonparametric bootstrap (Stata ``bootstrap``)."""

    _stata_command = "bootstrap"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        estimator: str = "regress",
        reps: int = 200,
        seed: int | None = None,
        subset: Any = None,
    ) -> "Bootstrap":
        if int(reps) <= 0:
            raise ValueError(f"bootstrap: reps must be positive, got {reps!r}")
        normalized = str(estimator).strip().lower()
        if normalized not in ("regress", "mean"):
            raise UnsupportedError(
                f"bootstrap: estimator {estimator!r} is not supported; only "
                "'regress' and 'mean' are. Stata's arbitrary r() statistics would "
                "require a small expression interpreter."
            )
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        if normalized == "mean":
            lhs = formula.split("~", 1)[0].strip()
            response = data[lhs].to_numpy(dtype=float)
            design = np.ones((len(data), 1))
            names = ["mean"]
        else:
            design, names, kept = build_design(formula, data)
            lhs = formula.split("~", 1)[0].strip()
            response = data[lhs].to_numpy(dtype=float)
        model = cls(
            data, response, np.asarray(design, dtype=float), names, estimator=normalized
        )
        model._reps = int(reps)
        model._seed = seed
        return model

    @classmethod
    def from_expressions(
        cls,
        expressions: str,
        data: pd.DataFrame,
        *,
        formula: str,
        estimator: str = 'regress',
        reps: int = 200,
        seed: int | None = None,
    ) -> 'Bootstrap':
        """Bootstrap an arbitrary `exp_list` over a replicate's stored results.

        This is the path Stata's `bootstrap r(mean), ...: summarize y` takes: the command is
        fitted on each replicate and the expression is evaluated over what it stored. The
        grammar is `[U] exp_list`; construction is delegated to the parser so an invalid
        expression fails before any resampling happens.
        """

        if int(reps) <= 0:
            raise ValueError(f'bootstrap: reps must be positive, got {reps!r}')
        normalized = str(estimator).strip().lower()
        if normalized not in ('regress', 'mean'):
            raise UnsupportedError(
                f'bootstrap: estimator {estimator!r} is not supported; '
                "only 'regress' and 'mean' are"
            )
        design, names, kept = build_design(formula, data)
        lhs = formula.split('~', 1)[0].strip()
        response = data[lhs].to_numpy(dtype=float)
        model = cls(data, response, np.asarray(design, dtype=float), names,
                    estimator=normalized)
        model._expressions = parse_exp_list(expressions)
        # Validate every referenced statistic against a fit of the ORIGINAL sample, so a typo
        # fails here rather than after the first of thousands of replicate fits.
        probe = model._stored(np.arange(data.shape[0]))
        model._expressions.evaluate(probe)
        model._reps = int(reps)
        model._seed = seed
        return model

    def fit(self) -> StataResults:
        n = self._y.size
        rows = np.arange(n)
        observed = self._estimate(rows)

        generator = np.random.default_rng(self._seed)
        replicates = np.empty((self._reps, observed.size))
        for index in range(self._reps):
            draw = generator.integers(0, n, size=n)
            replicates[index] = self._estimate(draw)

        # Percentile-based covariance: the sample covariance of the replicates.
        covariance = np.atleast_2d(np.cov(replicates, rowvar=False, ddof=1))
        return self._results(
            observed,
            covariance,
            extra={
                "reps": self._reps,
                "seed": self._seed,
                "note": (
                    "Bootstrap variances depend on the random stream; Stata's "
                    "generator differs from numpy's, so only the magnitude and the "
                    "point estimate are comparable, not the individual replicates."
                ),
            },
        )


class Permute(_ResampleBase):
    """Permutation test (Stata ``permute``).

    The treatment variable's values are shuffled ``reps`` times and the statistic
    recomputed; the p-value is the share of permuted statistics at least as extreme as
    the observed one.
    """

    _stata_command = "permute"

    def __init__(self, *args: Any, permuted_column: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._permuted_column = int(permuted_column)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        permute: str,
        reps: int = 200,
        seed: int | None = None,
        subset: Any = None,
    ) -> "Permute":
        if int(reps) <= 0:
            raise ValueError(f"permute: reps must be positive, got {reps!r}")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        design, names, kept = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        response = data[lhs].to_numpy(dtype=float)
        if permute not in names:
            raise ConsistencyError(
                f"permute: {permute!r} is not a regressor; the regressors are {names}"
            )
        model = cls(
            data, response, np.asarray(design, dtype=float), names, estimator="regress"
        )
        model._permuted_column = names.index(permute)
        model._reps = int(reps)
        model._seed = seed
        model._permuted_name = permute
        return model

    def fit(self) -> StataResults:
        n = self._y.size
        observed = self._estimate(np.arange(n))
        target = self._permuted_column

        generator = np.random.default_rng(self._seed)
        count = 0
        column = self._X[:, target].copy()
        permuted = self._X.copy()
        for _ in range(self._reps):
            permuted[:, target] = generator.permutation(column)
            statistics = _ols(permuted, self._y)
            # Two-sided: at least as extreme in absolute value.
            if abs(statistics[target]) >= abs(observed[target]) - 1e-15:
                count += 1

        pvalue = (count + 1.0) / (self._reps + 1.0)

        # The covariance reported is the analytic OLS one; permute does not estimate
        # a VCE, it produces a p-value.
        residual = self._y - self._X @ observed
        dof = max(n - self._X.shape[1], 1)
        sigma2 = float(residual @ residual) / dof
        covariance = sigma2 * np.linalg.pinv(self._X.T @ self._X)

        result = self._results(
            observed,
            covariance,
            extra={
                "reps": self._reps,
                "seed": self._seed,
                "pvalue_permuted": pvalue,
                "permuted": self._permuted_name,
                "note": (
                    "The permuted p-value depends on the random stream, so only its "
                    "magnitude is comparable to Stata's."
                ),
            },
        )
        return result
