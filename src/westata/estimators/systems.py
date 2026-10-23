"""System estimation: SUR, multivariate regression, errors-in-variables, MANOVA.

=================  ==========
westata             Stata
=================  ==========
``Sureg``            ``sureg``
``Mvreg``            ``mvreg``
``Eivreg``           ``eivreg``
``Manova``           ``manova``
=================  ==========

``sureg`` is the reason this module exists. Its coefficients alone do not
demonstrate correctness: if the cross-equation covariance ``Sigma`` is wrong, every
point estimate stays put and every standard error moves. The algebra is

    Sigma_ij = e_i'e_j / n            (from per-equation OLS residuals, divide by n)
    Omega    = Sigma^-1 (x) I_n
    beta     = (X' Omega X)^-1 X' Omega y
    V        = (X' Omega X)^-1

verified against Stata to the printed digit -- coefficients and standard errors
together -- on the reference data, where the fitted cross-equation correlation is
0.5544.

Per-equation prefixing
----------------------
Both equations carry a constant, so the parameter names must be prefixed with the
equation's dependent variable (``y1:_cons``, ``y2:_cons``). ``StataResults``
rejects duplicate names, and beyond that the prefix is the only thing that makes a
stacked coefficient vector readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Eivreg", "Manova", "ManovaResult", "Mvreg", "Sureg"]


def _design(formula: str, data: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    X, names, kept = build_design(formula, data)
    return np.asarray(X, dtype=float), list(names), np.asarray(kept)


class Sureg:
    """Seemingly unrelated regression (Stata ``sureg``)."""

    _stata_command = "sureg"

    def __init__(
        self,
        designs: Sequence[np.ndarray],
        responses: Sequence[np.ndarray],
        *,
        equation_names: Sequence[str],
        param_names: Sequence[Sequence[str]],
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._designs = [np.asarray(X, dtype=float) for X in designs]
        self._responses = [np.asarray(y, dtype=float).reshape(-1) for y in responses]
        self._equation_names = list(equation_names)
        self._param_names = [list(names) for names in param_names]
        sizes = {X.shape[0] for X in self._designs}
        if len(sizes) != 1:
            raise ConsistencyError(
                f"sureg: the equations have different sample sizes {sorted(sizes)}; "
                "SUR requires the same observations in every equation"
            )
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls,
        equations: Sequence[str],
        data: pd.DataFrame,
        *,
        names: Sequence[str] | None = None,
        subset: Any = None,
    ) -> "Sureg":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        if not equations:
            raise ConsistencyError("sureg: at least one equation is required")

        designs, responses, labels, names_per_equation = [], [], [], []
        common_index: np.ndarray | None = None
        for formula in equations:
            X, param_names, kept = _design(formula, data)
            lhs = formula.split("~", 1)[0].strip()
            if lhs not in data.columns:
                raise ConsistencyError(
                    f"sureg: the left-hand side must be a plain column name, got {lhs!r}"
                )
            if common_index is None:
                common_index = kept
            elif not np.array_equal(common_index, kept):
                raise ConsistencyError(
                    "sureg: the equations must use the same observations; one of "
                    "them drops rows the others keep"
                )
            designs.append(X)
            responses.append(data.loc[kept, lhs].to_numpy(dtype=float))
            labels.append(lhs)
            names_per_equation.append(param_names)

        if names is not None:
            if len(names) != len(equations):
                raise ValueError(
                    f"sureg: {len(names)} equation names given for {len(equations)} "
                    "equations"
                )
            labels = list(names)

        return cls(
            designs,
            responses,
            equation_names=labels,
            param_names=names_per_equation,
            model_data=data,
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        iterate: bool = False,
    ) -> StataResults:
        if iterate:
            raise UnsupportedError(
                "sureg: iterate=True (Stata's isure) is not implemented. Iterated "
                "FGLS needs a convergence criterion Stata does not document "
                "precisely, and a wrong rule would change every standard error on "
                "the final iteration."
            )

        equations = len(self._designs)
        n = self._designs[0].shape[0]
        sizes = [X.shape[1] for X in self._designs]
        total = sum(sizes)

        # Step 1: per-equation OLS, for the cross-equation residual covariance.
        coefficients = []
        residuals = []
        for X, y in zip(self._designs, self._responses):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            coefficients.append(beta)
            residuals.append(y - X @ beta)

        sigma = np.empty((equations, equations), dtype=float)
        for i in range(equations):
            for j in range(i, equations):
                # Divide by n, not n - k: that is the convention that reproduces
                # Stata's standard errors, verified to the printed digit.
                value = float(residuals[i] @ residuals[j]) / n
                sigma[i, j] = value
                sigma[j, i] = value

        # Step 2: feasible GLS on the stacked system.
        stacked = np.zeros((equations * n, total), dtype=float)
        target = np.zeros(equations * n, dtype=float)
        offset = 0
        column = 0
        for X, y, size in zip(self._designs, self._responses, sizes):
            stacked[offset : offset + n, column : column + size] = X
            target[offset : offset + n] = y
            offset += n
            column += size

        precision = np.linalg.pinv(sigma)
        omega = np.kron(precision, np.eye(n))
        information = stacked.T @ omega @ stacked
        beta = np.linalg.lstsq(information, stacked.T @ omega @ target, rcond=None)[0]
        covariance = np.linalg.pinv(information)

        names: list[str] = []
        for label, param_names in zip(self._equation_names, self._param_names):
            names.extend(f"{label}:{name}" for name in param_names)

        residual = target - stacked @ beta
        e: dict[str, Any] = {
            "N": n,
            "n_equations": equations,
            "df_m": float(total - equations),
            "rank": total,
            "equations": list(self._equation_names),
            "sigma": sigma.tolist(),
            "corr": (
                sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
            ).tolist(),
            "sureg": 1,
            "estimated": "feasible_gls",
        }
        for position, label in enumerate(self._equation_names):
            equation_residual = residual[position * n : (position + 1) * n]
            e[f"r2_{label}"] = float(
                1.0
                - (equation_residual @ equation_residual)
                / float(
                    (
                        (self._responses[position] - self._responses[position].mean())
                        ** 2
                    ).sum()
                )
            )
            e[f"rmse_{label}"] = float(
                np.sqrt(equation_residual @ equation_residual / n)
            )

        return StataResults(
            params=beta,
            param_names=names,
            cov=covariance,
            # The stacked residual has one entry per equation-observation, so nobs
            # must match it; the per-equation sample size is e(N).
            resid=residual,
            fittedvalues=stacked @ beta,
            nobs=int(equations * n),
            df_resid=float(equations * n - total),
            df_model=float(total - equations),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            model_data=self._model_data,
        )


class Mvreg:
    """Multivariate regression (Stata ``mvreg``).

    Every equation shares one design matrix, so the coefficients are per-equation
    OLS; only the VCE uses the cross-equation residual covariance. Note that this
    makes ``mvreg`` differ from ``sureg`` on identical data -- a specification
    difference, not an inconsistency.
    """

    _stata_command = "mvreg"

    def __init__(
        self,
        designs: Sequence[np.ndarray],
        responses: Sequence[np.ndarray],
        *,
        outcome_names: Sequence[str],
        param_names: Sequence[str],
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._designs = [np.asarray(X, dtype=float) for X in designs]
        self._responses = [np.asarray(y, dtype=float).reshape(-1) for y in responses]
        self._outcome_names = list(outcome_names)
        self._param_names = list(param_names)
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "Mvreg":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        if "=" not in formula:
            raise ConsistencyError(
                "mvreg: the formula must separate outcomes from regressors with "
                "'=', as in 'y1 y2 = x1 x2 x3'"
            )
        lhs, rhs = formula.split("=", 1)
        outcomes = [name for name in lhs.replace("+", " ").split() if name]
        if not outcomes:
            raise ConsistencyError("mvreg: no outcome variables were given")

        reference = outcomes[0]
        for column in outcomes:
            if column not in data.columns:
                raise ConsistencyError(f"mvreg: column {column!r} not found in the data")

        # Stata writes the regressor list space-separated ("x1 x2 x3") but patsy
        # needs explicit "+". Missing that join makes every mvreg formula a
        # SyntaxError rather than a model.
        regressor_terms = rhs.strip().replace("=", "+").split()
        X, param_names, kept = _design(
            f"{reference} ~ " + " + ".join(regressor_terms), data
        )

        return cls(
            [X for _ in outcomes],
            [data.loc[kept, name].to_numpy(dtype=float) for name in outcomes],
            outcome_names=outcomes,
            param_names=param_names,
            model_data=data,
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
    ) -> StataResults:
        equations = len(self._designs)
        X = self._designs[0]
        n, k = X.shape
        total = equations * k

        coefficients = []
        residuals = []
        for y in self._responses:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            coefficients.append(beta)
            residuals.append(y - X @ beta)

        sigma = np.empty((equations, equations), dtype=float)
        for i in range(equations):
            for j in range(i, equations):
                value = float(residuals[i] @ residuals[j]) / max(n - k, 1)
                sigma[i, j] = value
                sigma[j, i] = value

        covariance = np.kron(sigma, np.linalg.pinv(X.T @ X))
        beta = np.concatenate([np.asarray(b, dtype=float) for b in coefficients])

        names: list[str] = []
        for outcome in self._outcome_names:
            names.extend(f"{outcome}:{name}" for name in self._param_names)

        e: dict[str, Any] = {
            "N": n,
            "n_equations": equations,
            "df_m": float(k - 1),
            "rank": total,
            "outcomes": list(self._outcome_names),
            "sigma": sigma.tolist(),
        }

        return StataResults(
            params=beta,
            param_names=names,
            cov=covariance,
            resid=np.concatenate(residuals),
            fittedvalues=np.concatenate([X @ b for b in coefficients]),
            nobs=int(equations * n),
            df_resid=float(n - k),
            df_model=float(k - 1),
            dist="t",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            model_data=self._model_data,
        )


class Eivreg:
    """Errors-in-variables regression (Stata ``eivreg``).

    ``[R] eivreg`` Methods and formulas::

        b = A^-1 X' y,   A = X'X - S,   S = diag((1 - r_j) * s_j^2)

    where ``r_j`` is the reliability and ``s_j^2`` the sample variance of variable
    j. The measurement-error variance is SUBTRACTED from the cross-product, which
    inflates the coefficient relative to OLS by exactly the attenuation. It is not
    a reweighting: two earlier versions of this code reweighted by 1/lambda and
    then by lambda, giving x1 = 1.214 and 1.735 against Stata's 2.073.

    The variance is ``A^-1 H'H A^-1`` with

        h_i = e_i x_i + (x_i - xbar)^2 * (1 - r_j) * b_j

    and the RMSE is ``(y'y - b'Ab)/(n - p)``.
    """

    _stata_command = "eivreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        reliab: Mapping[str, float],
        *,
        param_names: Sequence[str],
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        self._X = np.atleast_2d(np.asarray(exog, dtype=float))
        self._param_names = list(param_names)
        self._reliab = dict(reliab)
        for name, value in self._reliab.items():
            if not 0.0 < float(value) <= 1.0:
                raise ConsistencyError(
                    f"eivreg: the reliability of {name!r} must lie in (0, 1], got {value}"
                )
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        reliab: Mapping[str, float] | None = None,
        subset: Any = None,
    ) -> "Eivreg":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if reliab is None:
            raise ConsistencyError(
                "eivreg: reliab is required; without it the attenuation cannot be "
                "corrected and the result would silently be OLS"
            )
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, param_names, kept = _design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        unknown = [name for name in reliab if name not in param_names]
        if unknown:
            raise ConsistencyError(
                f"eivreg: reliab refers to {unknown}, which are not regressors; "
                f"the regressors are {param_names}"
            )
        return cls(
            data.loc[kept, lhs].to_numpy(dtype=float),
            X,
            reliab,
            param_names=param_names,
            model_data=data,
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
    ) -> StataResults:
        n, k = self._X.shape

        reliability = np.ones(k, dtype=float)
        for position, name in enumerate(self._param_names):
            if name in self._reliab:
                reliability[position] = float(self._reliab[name])

        variance = np.var(self._X, axis=0, ddof=1)
        s_matrix = np.diag((1.0 - reliability) * variance)

        # A = X'X/n - S, not X'X - S: S is built from sample variances, which are
        # per-observation quantities, so the cross-product has to be normalised for
        # the subtraction to have compatible units. Getting this wrong leaves the
        # correction ~200x too small -- x1 came out 1.452 instead of 2.073.
        cross = (self._X.T @ self._X) / n - s_matrix
        beta = np.linalg.solve(cross, (self._X.T @ self._y) / n)
        residual = self._y - self._X @ beta

        centred_squared = (self._X - self._X.mean(axis=0)) ** 2
        correction = centred_squared * ((1.0 - reliability) * beta)
        H = residual[:, None] * self._X + correction
        # Scale the sandwich to match the normalised A: V = A^-1 (H'H/n) A^-1 / n.
        A_inverse = np.linalg.inv(cross)
        covariance = A_inverse @ (H.T @ H / n) @ A_inverse / n

        dof = max(n - k, 1)
        sigma2 = float(self._y @ self._y - beta @ (cross * n) @ beta) / dof
        r2 = float(
            1.0
            - (residual @ residual)
            / float(((self._y - self._y.mean()) ** 2).sum())
        )
        e: dict[str, Any] = {
            "N": n,
            "df_m": float(k - 1),
            "df_r": float(dof),
            "rank": k,
            "r2": r2,
            "rmse": float(np.sqrt(max(sigma2, 0.0))),
            "reliab": dict(self._reliab),
        }

        return StataResults(
            params=beta,
            param_names=list(self._param_names),
            cov=covariance,
            resid=residual,
            fittedvalues=self._X @ beta,
            nobs=n,
            df_resid=float(dof),
            df_model=float(k - 1),
            dist="t",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )


@dataclass(frozen=True)
class ManovaResult:
    """MANOVA statistics for one factor."""

    wilks_lambda: float
    df_hypothesis: float
    df_error: float
    f_statistic: float
    pvalue: float
    term: str

    def __str__(self) -> str:
        return (
            f"MANOVA {self.term}\n"
            f"  Wilks' lambda = {self.wilks_lambda:.4f}\n"
            f"  F({self.df_hypothesis:g}, {self.df_error:g}) = {self.f_statistic:.2f}"
            f"   p = {self.pvalue:.4f}"
        )


class Manova:
    """One-way MANOVA (Stata ``manova``), Wilks' lambda only."""

    _stata_command = "manova"

    def __init__(
        self,
        responses: np.ndarray,
        groups: np.ndarray,
        *,
        term: str = "group",
    ) -> None:
        Y = np.atleast_2d(np.asarray(responses, dtype=float))
        if Y.shape[0] == 1:
            Y = Y.T
        self._Y = Y
        codes, labels = pd.factorize(pd.Series(np.asarray(groups).reshape(-1)))
        self._codes = np.asarray(codes, dtype=int)
        self._term = term
        self._labels = list(labels)

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "Manova":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        if "=" not in formula:
            raise ConsistencyError(
                "manova: the formula must be 'outcomes = factor', as in 'y1 y2 = g'"
            )
        lhs, rhs = formula.split("=", 1)
        outcomes = [name for name in lhs.replace("+", " ").split() if name]
        factors = [name for name in rhs.replace("+", " ").split() if name]
        if len(factors) != 1:
            raise UnsupportedError(
                f"manova: exactly one factor is supported, got {factors}"
            )
        factor = factors[0]
        for column in outcomes + [factor]:
            if column not in data.columns:
                raise ConsistencyError(f"manova: column {column!r} not found")

        Y = data[outcomes].to_numpy(dtype=float)
        return cls(Y, data[factor].to_numpy(), term=factor)

    def fit(self) -> ManovaResult:
        Y = self._Y
        n, p = Y.shape
        groups = np.unique(self._codes)
        g = groups.size
        if g < 2:
            raise ConsistencyError(
                "manova: the factor has a single level, so there is nothing to test"
            )

        grand_mean = Y.mean(axis=0)
        within = np.zeros((p, p))
        between = np.zeros((p, p))
        for code in groups:
            block = Y[self._codes == code]
            block_mean = block.mean(axis=0)
            centred = block - block_mean
            within += centred.T @ centred
            difference = (block_mean - grand_mean).reshape(-1, 1)
            between += block.shape[0] * (difference @ difference.T)

        # Wilks' lambda = |W| / |W + B|
        numerator = float(np.linalg.det(within))
        denominator = float(np.linalg.det(within + between))
        if denominator == 0:
            raise ConsistencyError(
                "manova: the total sum of squares is singular; a response is "
                "collinear with the group indicator"
            )
        wilks = numerator / denominator

        df_hypothesis = float(g - 1)
        df_error = float(n - g)
        # Rao's F approximation for Wilks' lambda.
        if p == 1:
            f_statistic = (1.0 - wilks) / wilks * df_error / df_hypothesis
            df1, df2 = df_hypothesis, df_error
        else:
            s = min(p, int(df_hypothesis))
            m = (abs(p - df_hypothesis) - 1.0) / 2.0
            r = (df_error - p - 1.0) / 2.0
            if s == 1:
                df1 = p
                df2 = df_error - p + 1.0
                f_statistic = (1.0 - wilks) / wilks * df2 / df1
            else:
                t = np.sqrt(
                    (p**2 * df_hypothesis**2 - 4.0)
                    / max(p**2 + df_hypothesis**2 - 5.0, 1e-12)
                )
                df1 = p * df_hypothesis
                df2 = r * t - (p * df_hypothesis - 2.0) / 2.0
                exponent = 1.0 / t
                f_statistic = (
                    (1.0 - wilks**exponent)
                    / (wilks**exponent)
                    * (df2 / df1)
                )
        pvalue = float(stats.f.sf(f_statistic, df1, df2))

        return ManovaResult(
            wilks_lambda=float(wilks),
            df_hypothesis=df_hypothesis,
            df_error=df_error,
            f_statistic=float(f_statistic),
            pvalue=pvalue,
            term=self._term,
        )
