"""``suest`` -- seemingly unrelated estimation: stack several fits into one.

``suest.sthlp:84-88``:

    suest combines the estimation results -- parameter estimates and associated
    (co)variance matrices -- stored under namelist into one parameter vector and
    simultaneous (co)variance matrix of the sandwich/robust type.  This (co)variance
    matrix is appropriate even if the estimates were obtained on the same or on
    overlapping data.

That last sentence is the whole point. Stacking two OLS fits and taking their
``cov_params`` block-diagonally would assume the two samples' scores are uncorrelated --
false when the same observations feed both models. The SANDWICH form captures that
correlation through the outer product of each observation's stacked score:

    V = B (sum_i s_i s_i') B,     B = blockdiag((X_a'X_a)^-1, (X_b'X_b)^-1)
    s_i = (r_ai * x_ai, r_bi * x_bi)

with the two blocks of ``s_i`` evaluated on the SAME observation. That is what makes the
off-diagonal blocks informative rather than zero.

``suest.sthlp:99-102`` gives the one hard requirement: each model must produce
equation-level scores via ``predict, score``. In a linear model ``score`` is the residual
(``regress_postestimimation.sthlp:193``: "score is equivalent to residuals in linear
regression"), so the score contribution of observation ``i`` is ``residual_i * x_i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class SucestResult:
    """The stacked parameter vector and its simultaneous covariance matrix.

    Attributes
    ----------
    params:
        All models' coefficients, concatenated in the order the models were given.
    covariance:
        The sandwich covariance, spanning every coefficient.
    param_names:
        Model-prefixed names, so a coefficient can be traced back to its equation.
    models:
        How many fits were combined.
    n_observations:
        Size of the union of the estimation samples.
    """

    params: np.ndarray
    covariance: np.ndarray
    param_names: list[str]
    models: int
    n_observations: int
    #: Residual degrees of freedom carried from the components, used by :meth:`test`.
    _df_resid: float = 1.0

    def __len__(self) -> int:
        return int(self.params.size)

    def cov_params(self) -> np.ndarray:
        """The simultaneous covariance matrix, as a CALL.

        ``suest.sthlp:91-94`` names ``test``, ``testnl``, ``lincom`` and ``nlcom`` as the
        typical things to run after ``suest`` -- and they all reach for
        ``result.cov_params()``.  Exposing the matrix only as an attribute left all four
        unusable on the stacked result, which defeated the command's purpose.
        """

        return np.asarray(self.covariance, dtype=float)

    @property
    def df_resid(self) -> float:
        """Residual degrees of freedom of the SMALLEST component.

        A Wald test on the stacked vector needs one denominator; with components of
        different sizes the conservative choice is the fewest residual degrees of
        freedom, which is also what makes an F test no less powerful than it should be.
        """

        return float(self._df_resid)

    def test(self, *specs: Any, **kwargs: Any) -> Any:
        """Wald test across the stacked coefficients (``suest.sthlp:91-94``).

        Hypotheses name coefficients with their model prefix, e.g.
        ``test "m1:x1 = m2:x2"``.
        """

        from westata.postestimation.api import test_as_stata

        return test_as_stata(self, *specs, **kwargs)

    def lincom(self, expression: str, **kwargs: Any) -> Any:
        """Linear combination of the stacked coefficients (``sthlp:93-94``)."""

        from westata.postestimation.api import lincom_as_stata

        return lincom_as_stata(self, expression, **kwargs)

    def to_frame(self) -> pd.DataFrame:
        """The covariance matrix as a labelled DataFrame."""

        return pd.DataFrame(
            self.covariance, index=self.param_names, columns=self.param_names
        )

    def __str__(self) -> str:
        return (
            "\nSimultaneous covariance matrix\n"
            f"  {self.models} models, {len(self)} parameters, "
            f"{self.n_observations} observations\n"
            f"  parameters: {', '.join(self.param_names)}"
        )

    def to_dict(self) -> dict[str, float]:
        """The stacked coefficients under their model-prefixed names."""

        return {
            name: float(value)
            for name, value in zip(self.param_names, self.params)
        }


def _design(result: Any) -> np.ndarray:
    """The model matrix of a fit, as a plain 2-D array."""

    stored = getattr(result, "design_matrix", None)
    if stored is None:
        raise NotImplementedError(
            "suest needs each model's design matrix; this result did not retain one.  "
            "Stata's requirement is that predict, score works after every estimation "
            "command (suest.sthlp:99-102), and building the score needs the regressors."
        )
    return np.asarray(stored[0], dtype=float)


def _scores(result: Any, count: int) -> np.ndarray:
    """Observation-level score contributions, one row per observation.

    For a linear model the score is the residual times the regressor
    (``regress_postestimation.sthlp:193``), which is what ``predict, score`` returns.
    """

    residuals = np.asarray(result.resid, dtype=float).reshape(-1)
    design = _design(result)
    if residuals.size != design.shape[0]:
        raise ValueError(
            "suest: the residual vector has "
            f"{residuals.size} entries for {design.shape[0]} design rows"
        )
    if design.shape[0] != count:
        raise ValueError(
            "suest: this model has "
            f"{design.shape[0]} observations but the shared frame has {count}; align "
            "them on the same estimation sample first"
        )
    return design * residuals.reshape(-1, 1)


def _sandwich(scores: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """``B (s's) B`` with ``B`` block diagonal."""

    meat = scores.T @ scores
    return bread @ meat @ bread


def _clustered_meat(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``Σ_c (Σ_{i in c} s_i)(Σ_{i in c} s_i)'`` -- the clustered meat.

    ``suest.sthlp:108-109``: "suest returns the robust VCE, allows the vce(cluster
    clustvar) option".  Clustering aggregates each observation's stacked score WITHIN its
    cluster before taking the outer product, which is what lets the covariance reflect
    correlation among observations that share a cluster.

    Summing first and squaring once -- rather than summing the outer products -- is the
    whole difference: it lets the within-cluster scores cancel, which is exactly the
    dependence the unclustered meat ignores.
    """

    unique = pd.unique(labels)
    aggregated = np.empty((len(unique), scores.shape[1]), dtype=float)
    for position, cluster in enumerate(unique):
        aggregated[position] = scores[labels == cluster].sum(axis=0)
    return aggregated.T @ aggregated


def suest(
    models: Sequence[Any],
    *,
    names: Sequence[str] | None = None,
    cluster: Any = None,
) -> SucestResult:
    """Stata's ``suest`` -- stack several fits into one simultaneous covariance.

    Parameters
    ----------
    models:
        Two or more fitted linear results, all on the SAME observations.
    names:
        Labels for each model; defaults to ``model1``, ``model2``, ...
    cluster:
        Cluster labels, giving Stata's ``vce(cluster clustvar)`` form
        (``suest.sthlp:108-109``).  Each observation's stacked score is summed within its
        cluster before the outer product, so the covariance reflects within-cluster
        dependence.

    Raises
    ------
    NotImplementedError
        When a model cannot produce scores -- a non-linear estimator whose score is not
        the residual, or one without a design matrix.
    ValueError
        When fewer than two models are given, when their sample sizes differ, when a
        block is singular, or when the cluster labels do not match the sample.

    Notes
    -----
    ``suest.sthlp:105-107`` advises estimating the components WITHOUT
    ``vce(robust)``/``vce(cluster)``: ``suest`` supplies the robust VCE itself, so a
    robust component would be doubly corrected. The sandwich built here uses only the
    residuals and the design, so it is robust regardless of what the components used --
    which is exactly why the advice exists.
    """

    if len(models) < 2:
        raise ValueError(
            "suest combines two or more estimation results; got "
            f"{len(models)} (suest.sthlp:84-88)"
        )
    labels = list(names) if names is not None else [
        f"model{index + 1}" for index in range(len(models))
    ]
    if len(labels) != len(models):
        raise ValueError(
            f"suest: {len(labels)} names for {len(models)} models"
        )

    sample_sizes = {int(np.asarray(model.resid).size) for model in models}
    if len(sample_sizes) != 1:
        raise ValueError(
            "suest: every model must use the same estimation sample; got sizes "
            f"{sorted(sample_sizes)}"
        )
    count = sample_sizes.pop()

    cluster_labels = None
    if cluster is not None:
        cluster_labels = np.asarray(cluster).reshape(-1)
        if cluster_labels.size != count:
            raise ValueError(
                f"suest: {cluster_labels.size} cluster labels for {count} observations"
            )

    # Stack the scores observation by observation, so the cross-model blocks of the meat
    # carry the correlation between the two fits' contributions from the SAME row.
    stacked_scores = np.column_stack(
        [_scores(model, count) for model in models]
    )

    params: list[np.ndarray] = []
    bread_blocks: list[np.ndarray] = []
    param_names: list[str] = []
    for label, model in zip(labels, models):
        design = _design(model)
        gram = design.T @ design
        try:
            bread = np.linalg.inv(gram)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"suest: the design matrix of {label!r} is singular, so its parameter "
                "block cannot be inverted"
            ) from error
        bread_blocks.append(bread)
        coefficients = np.asarray(model.params, dtype=float).reshape(-1)
        if coefficients.size != design.shape[1]:
            raise ValueError(
                f"suest: {label!r} has {coefficients.size} coefficients for "
                f"{design.shape[1]} design columns"
            )
        params.append(coefficients)
        for name in model.param_names:
            param_names.append(f"{label}:{name}")

    bread = np.zeros((stacked_scores.shape[1], stacked_scores.shape[1]))
    cursor = 0
    for block in bread_blocks:
        size = block.shape[0]
        bread[cursor : cursor + size, cursor : cursor + size] = block
        cursor += size

    if cluster_labels is None:
        meat = stacked_scores.T @ stacked_scores
    else:
        meat = _clustered_meat(stacked_scores, cluster_labels)
    covariance = bread @ meat @ bread
    # The denominator for a joint Wald test: the smallest component's residual df.
    residual_df = min(
        float(getattr(model, "df_resid", count - 1)) for model in models
    )
    return SucestResult(
        params=np.concatenate(params),
        covariance=covariance,
        param_names=param_names,
        models=len(models),
        n_observations=count,
        _df_resid=residual_df,
    )
