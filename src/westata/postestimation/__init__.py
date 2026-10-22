"""Postestimation: hypothesis tests, linear combinations, marginal effects.

Targets Stata's ``test``, ``lincom`` and ``margins`` commands.
"""

from __future__ import annotations

from westata.postestimation.api import (
    lincom_as_stata,
    statsmodels_wald_test,
    test_as_stata,
)
from westata.postestimation.lincom import LincomResult, run_lincom
from westata.postestimation.tests import (
    WaldTest,
    build_contrast,
    describe_constraint,
    resolve_names,
    run_wald_test,
)

__all__ = [
    "WaldTest",
    "LincomResult",
    "build_contrast",
    "describe_constraint",
    "lincom_as_stata",
    "resolve_names",
    "run_lincom",
    "run_wald_test",
    "statsmodels_wald_test",
    "test_as_stata",
]
