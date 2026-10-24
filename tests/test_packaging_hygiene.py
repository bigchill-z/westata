"""Guard against UTF-8 BOM in the packaging files.

A BOM has now broken the build twice in this project:

  1. `pyproject.toml` -- `Set-Content -Encoding utf8` in PowerShell 5 prepends `EF BB BF`, and TOML does not allow
     it: `ERROR Failed to parse pyproject.toml: Invalid statement (at line 1, column 1)`.

  2. `MANIFEST.in` -- same cause, but the symptom was silent rather than loud. The file still parsed as a manifest,
     the `include CHANGELOG.md` directive was simply never recognised because the first line read
     `\\ufeffinclude`, so the changelog was quietly left out of the sdist. Built successfully, checked out fine,
     and the file was missing.

The second kind is what makes this worth a test: a loud failure is self-correcting, a quiet one ships.

The check is on the FILE BYTES rather than on decoded text, because `read_text(encoding="utf-8")` leaves the BOM
in the string while `utf-8-sig` silently strips it -- and the tools that consume these files do the former.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Files that are parsed by tools which do NOT strip a BOM.
PACKAGING_FILES = [
    "pyproject.toml",
    "MANIFEST.in",
    "setup.cfg",
    "setup.py",
    ".gitignore",
    ".gitattributes",
]


@pytest.mark.parametrize("relative", PACKAGING_FILES)
def test_packaging_file_has_no_bom(relative: str) -> None:
    path = REPO / relative
    if not path.exists():
        pytest.skip(f"{relative} not present")
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), (
        f"{relative} starts with a UTF-8 BOM; tools that parse it (setuptools, tomllib, git) do not strip it. "
        f"Rewrite the file without the BOM."
    )


def test_manifest_lists_the_changelog_and_licence() -> None:
    """The sdist must carry both, and the manifest is the only thing that puts them there.

    This asserts the manifest CONTENT rather than rebuilding, because building in a test is slow and the failure
    mode being guarded against is a directive that silently does nothing -- which is visible in the text.
    """

    manifest = (REPO / "MANIFEST.in").read_text(encoding="utf-8-sig")
    lines = {line.strip() for line in manifest.splitlines() if line.strip()}
    assert "include CHANGELOG.md" in lines
    assert "include LICENSE" in lines
    # And the files must actually exist, or the include is a no-op.
    assert (REPO / "CHANGELOG.md").exists()
    assert (REPO / "LICENSE").exists()


def test_pyproject_declares_the_required_runtime_dependencies() -> None:
    """A regression guard for the defect that made the wheel unimportable.

    `pyfixest`, `linearmodels` and `pyyaml` are imported at module level by the package. They were once declared
    nowhere, so `pip install` succeeded and `import westata` failed. Nothing in the test suite could see it,
    because the suite runs from the source tree where the dependencies happen to be installed.

    This pins the declaration, which is the only part a test can reach cheaply. Whether the declaration is
    SUFFICIENT is checked by the CI step that builds and imports the wheel.
    """

    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    for dependency in ("pyfixest", "linearmodels", "pyyaml"):
        assert dependency in text, (
            f"{dependency} is imported at module level by westata but is not declared in pyproject.toml; "
            f"a wheel built from this file would install cleanly and fail on import"
        )
    # `arch` must stay in an EXTRA rather than the required list: forcing it on users who never run GARCH is the
    # situation the optional split exists to avoid.
    # The extra is written rch = ["arch>=6.0"], so the key is bare and the package name is the value.
    assert 'arch = [' in text, "arch should remain declared as an optional extra"
    # And it must NOT appear in the required list: forcing a large dependency on users who never run GARCH is
    # exactly the situation the optional split avoids.
    dependencies_block = text.split('dependencies = [', 1)[1].split(']', 1)[0]
    assert 'arch>=' not in dependencies_block, "arch must not be a required dependency"
