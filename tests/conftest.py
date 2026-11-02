"""Pytest configuration: make ``src/`` and the test helpers importable.

Two things happen here.

1. ``src/`` goes on ``sys.path`` so the package is importable without being installed,
   and the test directory goes on it too so ``stata_helpers`` can be imported by name.

2. Tests that read Stata's own ``.ado``/``.sthlp`` files are SKIPPED when no installation
   is present. Those paths used to be hard-coded to ``D:\\Program Files\\StataNow19\\...``,
   which held on the development machine and nowhere else: every one of the 20 files
   raised ``FileNotFoundError`` on the Linux CI runners and took all three jobs down.

   The skip is decided by the FILE, not by the individual test, so a new test added to one
   of these modules cannot be forgotten. The marker is applied here rather than by editing
   20 files, which is also why it cannot be undone piecemeal by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for entry in (str(SRC), str(TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import stata_helpers  # noqa: E402  (needs the sys.path line above)

#: Modules that read the Stata installation's own files as their evidence.
#: Listed by filename so the guard is visible in one place.
_STATA_FILE_MODULES = frozenset(
    {
        "test_completely_determined_counts.py",
        "test_e_aic_is_not_stata.py",
        "test_estat_auc.py",
        "test_estat_classification.py",
        "test_estat_cmd_per_command.py",
        "test_estat_dispatch_macros.py",
        "test_estat_eligibility_matrix.py",
        "test_ic_is_iterations_everywhere.py",
        "test_ml_titles_all_commands.py",
        "test_ml_titles_indirect_route.py",
        "test_predict_pr_count.py",
        "test_predict_program_macros.py",
        "test_predict_scores_shape.py",
        "test_regress_e_key_inventory.py",
        "test_title_macros_per_command.py",
        "test_tobit_has_no_aic.py",
        "test_vce_macros_ml_families.py",
        "test_vce_type_macros.py",
        "test_xt_design_names_and_ic.py",
        "test_xtreg_estat_surface.py",
    }
)


def pytest_collection_modifyitems(config, items):
    """Skip the Stata-file checks when there is no Stata installation to read."""

    if stata_helpers.available():
        return

    marker = pytest.mark.skip(reason=stata_helpers.reason())
    for item in items:
        if Path(str(item.fspath)).name in _STATA_FILE_MODULES:
            item.add_marker(marker)
