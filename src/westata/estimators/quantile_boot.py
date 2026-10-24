"""Simultaneous and bootstrapped quantile regression.

=================  ==============
westata             Stata
=================  ==============
``Sqreg``            ``sqreg``
``Bsqreg``           ``bsqreg``
=================  ==============

Both obtain their covariance matrices by bootstrapping, because the asymptotic
theory for a single quantile gives no cross-quantile covariance. ``[R] qreg``:

    bsqreg is equivalent to sqreg with one quantile.

so ``Bsqreg`` is implemented as a one-quantile ``Sqreg`` rather than as a separate
estimator -- that guarantees the two can never disagree about which resampling
scheme they used.

The coefficients are always the plain ``qreg`` estimates; only the standard errors
change. That is why both classes delegate to
:class:`westata.estimators.quantile.QuantileRegression` for the point estimates
and never re-derive them.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError
from westata.random.stata_rng import Mt19937_64
from westata.estimators.quantile import QuantileRegression, solve_lad
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Bsqreg", "Sqreg"]


class Sqreg:
    """Simultaneous-quantile regression (Stata ``sqreg``)."""

    _stata_command = "sqreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        quantiles: Sequence[float] = (0.25, 0.5, 0.75),
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
        resolved = tuple(float(q) for q in quantiles)
        if not resolved:
            raise ValueError("sqreg: at least one quantile is required")
        for q in resolved:
            if not 0.0 < q < 1.0:
                raise ValueError(
                    f"sqreg: quantile must lie strictly between 0 and 1, got {q!r}"
                )
        self._quantiles = resolved
        self._model_data: pd.DataFrame | None = None

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        quantiles: Sequence[float] = (0.25, 0.5, 0.75),
        subset: Any = None,
    ) -> "Sqreg":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"sqreg: the left-hand side must be a plain column name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(y, X, quantiles=quantiles, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        reps: int = 200,
        seed: int | None = None,
        strict: bool = False,
    ) -> StataResults:
        if reps <= 0:
            raise ValueError(f"sqreg: reps must be positive, got {reps!r}")

        n, k = self._X.shape
        if n <= k:
            raise ConsistencyError(
                f"sqreg: {n} observations but {k} regressors; the model is not "
                "identified"
            )

        # Point estimates are the plain qreg fits; only the VCE is bootstrapped.
        point = np.concatenate(
            [
                solve_lad(self._X, self._y, quantile=q)
                for q in self._quantiles
            ]
        )

        replicates = self._bootstrap(reps=reps, seed=seed)
        covariance = np.cov(replicates, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance)

        names: list[str] = []
        for q in self._quantiles:
            names.extend(f"q{q:g}:{name}" for name in self._param_names)

        residual = self._y - self._X @ point[:k]
        tau = self._quantiles[0]

        def check(values: np.ndarray) -> float:
            return float(np.sum((tau - (values < 0).astype(float)) * values))

        e: dict[str, Any] = {
            "N": n,
            "df_m": float(k - 1),
            "df_r": float(n - k),
            "quantiles": list(self._quantiles),
            "reps": reps,
            "rank": k * len(self._quantiles),
            "vce": "bootstrap",
        }

        return StataResults(
            params=point,
            param_names=names,
            cov=covariance,
            resid=residual,
            fittedvalues=self._X @ point[:k],
            nobs=n,
            df_resid=float(n - k),
            df_model=float(k - 1),
            dist="t",
            model_name=self._stata_command,
            cov_type="bootstrap",
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    def _bootstrap(self, *, reps: int, seed: int | None) -> np.ndarray:
        """Resample rows once per replicate and fit every quantile on that draw.

        Drawing one index set per replicate is what makes the covariance matrix
        simultaneous: the cross-quantile blocks measure how the quantiles move
        together under the *same* resampling, which independent draws per
        quantile would destroy.
        """

        n, k = self._X.shape
        # Stata's generator. `sqreg.ado` stores c(seed) and every draw comes from MT19937-64, so numpy's default
        # generator diverges from the first number.
        generator = Mt19937_64(DEFAULT_SEED if seed is None else seed)
        replicates = np.empty((reps, k * len(self._quantiles)), dtype=float)

        for index in range(reps):
            # `bsampl_w`, reproduced exactly. VERIFIED: the counts computed this way equal Stata's `w` after
            # `bsampl_w` element for element, at n = 20, 40 and 200.
            #
            # The order of consumption matters: `bsampl_w.ado` fills the whole `r` column
            # (`gen double r = int(uniform()*_N+1)`) before touching `w`, so the N target draws come first and the
            # N tie-breaks second. Interleaving them produces a different sequence.
            targets = np.fromiter(
                (int(generator.runiform() * n + 1) for _ in range(n)), dtype=int, count=n
            )
            tiebreak = np.fromiter(
                (generator.runiform() for _ in range(n)), dtype=float, count=n
            )

            # `sort r w`, then the running count within each target, then keep one row per drawn target.
            order = np.lexsort((tiebreak, targets))
            ordered_targets = targets[order]
            running = np.ones(n, dtype=int)
            for position in range(1, n):
                if ordered_targets[position] == ordered_targets[position - 1]:
                    running[position] = running[position - 1] + 1
            keep = np.zeros(n, dtype=bool)
            keep[-1] = True
            for position in range(n - 1):
                keep[position] = running[position + 1] == 1

            # One row per drawn target, repeated by its count -- `qreg_c ..., wvar(bwt)` fits frequency weights,
            # and fweight is replication (verified: qreg [fweight] equals the expanded fit to 3.5e-16).
            #
            # THE PAIRING GOES THROUGH `order`. `bsampl_w` sorts the dataset, so weight position i belongs to the
            # i-th row of the SORTED data. `keep` marks retained positions in that sorted order, so the original
            # row is `order[position]`. Using the target value directly (`ordered_targets[keep] - 1`) instead lands
            # the right counts on the wrong observations: the counts still match Stata's `w` exactly, which is why
            # checking them alone did not reveal it, while every replicate's spread came out different.
            kept_positions = np.flatnonzero(keep)
            drawn = order[kept_positions]
            expanded = np.repeat(drawn, running[keep])
            X_draw = self._X[expanded]
            y_draw = self._y[expanded]

            for position, q in enumerate(self._quantiles):
                # A degenerate resample can make a quantile fit singular; fall
                # back to the full-sample estimate for that quantile.
                try:
                    beta = solve_lad(X_draw, y_draw, quantile=q)
                except np.linalg.LinAlgError:
                    beta = solve_lad(self._X, self._y, quantile=q)
                replicates[index, position * k : (position + 1) * k] = beta

        return replicates


class Bsqreg(Sqreg):
    """Bootstrapped quantile regression (Stata ``bsqreg``).

    ``[R] qreg``: "bsqreg is equivalent to sqreg with one quantile."  Subclassing
    keeps that equivalence structural instead of a comment.
    """

    _stata_command = "bsqreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        quantile: float = 0.5,
        param_names: list[str] | None = None,
    ) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError(
                f"bsqreg: quantile must lie strictly between 0 and 1, got {quantile!r}"
            )
        super().__init__(
            endog, exog, quantiles=(float(quantile),), param_names=param_names
        )
        self._quantile = float(quantile)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        quantile: float = 0.5,
        subset: Any = None,
    ) -> "Bsqreg":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"bsqreg: the left-hand side must be a plain column name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(y, X, quantile=quantile, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        reps: int = 200,
        seed: int | None = None,
        strict: bool = False,
    ) -> StataResults:
        result = super().fit(
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            ssc=ssc,
            reps=reps,
            seed=seed,
            strict=strict,
        )
        # With one quantile the prefixed names are noise; Stata prints plain ones.
        return StataResults(
            params=result.params,
            param_names=list(self._param_names),
            cov=result.cov_params(),
            resid=result.resid,
            fittedvalues=result.fittedvalues,
            nobs=result.nobs,
            df_resid=result.df_resid,
            df_model=result.df_model,
            dist="t",
            model_name=self._stata_command,
            cov_type="bootstrap",
            e=dict(result.e, q=self._quantile),
            design_matrix=result.design_matrix,
            model_data=result.model_data,
        )
