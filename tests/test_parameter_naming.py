"""One mapping, one implementation.

Rounds 123-124 each found a different postestimation table disagreeing with the
coefficient table about how a parameter is spelled: ``estat vce`` used patsy's ``:``
for interactions, then ``estat vif`` did too. Each module carried its OWN
``_display_name``, and the copies drifted.

The mapping now lives in :mod:`westata.postestimation.names`. These tests pin both the
mapping itself and the fact that the tables agree with each other -- the second is what
would have caught the original bug immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.names import is_constant_name, stata_display_name


# ------------------------------------------------------------ the mapping itself
@pytest.mark.parametrize(
    "stored, printed",
    [
        ("Intercept", "_cons"),
        ("intercept", "_cons"),
        ("_cons", "_cons"),
        ("const", "_cons"),
        ("x1", "x1"),
        ("x1:x2", "x1#x2"),
        ("x1:x2:x3", "x1#x2#x3"),
        ("C(grp)[T.b]", "C(grp)[T.b]"),
        ("cut1", "cut1"),
        ("/cut1", "cut1"),
        ("/lnsigma", "lnsigma"),
        ("lnalpha", "lnalpha"),
    ],
)
def test_the_display_mapping(stored, printed) -> None:
    assert stata_display_name(stored) == printed


def test_the_constant_mapping_is_case_insensitive() -> None:
    """``Intercept``/``intercept``/``const`` all mean the same parameter."""

    assert is_constant_name("Intercept")
    assert is_constant_name("intercept")
    assert is_constant_name("_cons")
    assert is_constant_name("const")
    assert not is_constant_name("x1")


def test_only_the_separator_changes_not_the_order() -> None:
    """Stata renames ``a:b`` to ``a#b``; it does not sort the parts."""

    assert stata_display_name("x2:x1") == "x2#x1"


def test_the_slash_is_dropped_not_kept() -> None:
    """``e(b)`` and the matrix commands print auxiliary names WITHOUT the slash.

    The coefficient TABLE adds it back for display, which is a separate decision --
    conflating the two is what made ``nl``'s ``/b0`` vanish in round 122.
    """

    assert stata_display_name("/lnsigma") == "lnsigma"
    assert stata_display_name("/cut1") == "cut1"


def test_both_postestimation_modules_delegate_to_the_shared_mapping() -> None:
    """The shims must not carry logic of their own any more."""

    from westata.postestimation import vce, vif

    for probe in ("Intercept", "x1:x2", "/cut1", "plain"):
        assert vce._display_name(probe) == stata_display_name(probe)
        assert vif._display_name(probe) == stata_display_name(probe)


# --------------------------------------------------- the tables must agree
@pytest.fixture(scope="module")
def interaction_result():
    rng = np.random.default_rng(439)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.8 * x1 - 0.3 * x2 + 0.5 * x1 * x2 + rng.normal(size=n),
            "x1": x1,
            "x2": x2,
        }
    )
    return ws.OLS.from_formula("y ~ x1*x2", frame).fit()


def test_no_table_prints_patsy_separators(interaction_result) -> None:
    """The cheapest possible guard, applied to every table at once."""

    result = interaction_result
    tables = {
        "summary": result.summary(style="stata"),
        "vce": str(result.estat_vce()),
        "vif": str(result.estat_vif()),
    }

    for label, text in tables.items():
        assert "x1:x2" not in text, label
        assert "x1#x2" in text, label


def test_the_three_tables_name_the_same_regressors(interaction_result) -> None:
    """Set equality across the tables, which is what drifted before."""

    result = interaction_result
    vce_names = {name for name in result.estat_vce().names if name != "_cons"}
    vif_names = set(result.estat_vif().names)

    assert vce_names == vif_names == {"x1", "x2", "x1#x2"}


def test_a_matrix_command_never_prints_intercept(interaction_result) -> None:
    """patsy's spelling must not survive anywhere in the rendered tables."""

    assert "Intercept" not in str(interaction_result.estat_vce())
    assert "Intercept" not in str(interaction_result.estat_vif())


def test_the_matrix_row_count_is_unchanged(interaction_result) -> None:
    """The rename must not merge or drop a parameter."""

    vce = interaction_result.estat_vce()

    assert len(vce.names) == len(interaction_result.param_names)
    assert vce.to_frame().shape == (
        len(interaction_result.param_names),
        len(interaction_result.param_names),
    )