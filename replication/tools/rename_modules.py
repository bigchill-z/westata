"""Give every replication package a unique module namespace so imports cannot collide.

The problem
-----------
Each replication under ``replication/papers/<paper>/`` is a flat set of modules called
``acquire``, ``specs``, ``data``, ``run_stata``, ``run_westata``, ``run_compare``,
``run_all``, ``probe_gaps``, ``gen_do``, ``make_dags``.  Ten names, two to four copies each.

That works when a driver is run as a script (``python run_all.py`` puts its own directory
first on ``sys.path``).  It breaks the moment two packages are imported in **one process**,
which is exactly what ``pytest`` does: whichever test inserted its directory last wins, so
``from acquire import panel`` inside the NQ probe resolved to the *AD 2013* ``acquire`` and
raised ``ImportError: cannot import name 'panel'``.

Three separate bugs in this session had that single cause (``specs``, ``probe_gaps``,
``acquire``).  Patching each test individually fixes the symptom while leaving the next
one to be discovered the same way, so this script fixes the cause instead.

The fix
-------
Rename each package's shared module names to ``<slug>_<name>.py`` (e.g.
``nq2011_acquire.py``) and rewrite the imports accordingly.  A package-local module then
cannot shadow another package's, no matter what ``sys.path`` looks like.

What is NOT renamed
--------------------
``run_stata``/``run_westata``/``run_compare``/``run_all``/``gen_do``/``make_dags`` keep
their names: they are entry points, invoked as scripts, never imported across packages.
Renaming them would make the documented commands longer for no benefit.  Only modules that
are *imported by other modules or by tests* get the prefix.

Idempotent: running it twice is a no-op.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: ``replication/`` -- the tool lives in ``tools/``, its subjects in ``papers/``.
REPLICATION = HERE.parent
PAPERS = REPLICATION / "papers"

#: Modules imported across package boundaries -- these get the slug prefix.
#: ``run_westata`` is included because ``run_compare`` imports its ``prepare`` helper.
SHARED = ("acquire", "specs", "data", "probe_gaps", "run_westata")


def _slug(paper_dir: Path) -> str:
    return paper_dir.name


def rewrite(paper_dir: Path) -> list[str]:
    """Rename ``SHARED`` modules in one package and rewrite imports that reference them.

    Returns the list of changes so the caller can print a summary.
    """

    slug = _slug(paper_dir)
    changes: list[str] = []

    mapping = {}
    for name in SHARED:
        source = paper_dir / f"{name}.py"
        target = paper_dir / f"{slug}_{name}.py"
        if source.exists():
            source.rename(target)
            mapping[name] = f"{slug}_{name}"
            changes.append(f"{source.name} -> {target.name}")

    if not mapping:
        return changes

    # Rewrite every python file in the package (including the ones just renamed) and the
    # tests, which import these modules too.
    files = list(paper_dir.glob("*.py")) + list((REPLICATION / "tests").glob("test_*.py"))
    pattern = re.compile(
        r"(^|\n)(\s*)from (" + "|".join(map(re.escape, mapping)) + r") import "
    )
    plain = re.compile(
        r"(^|\n)(\s*)import (" + "|".join(map(re.escape, mapping)) + r")(\s|$|as )"
    )

    for path in files:
        if path.parent == REPLICATION / "tests" and not path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        original = text

        def sub_from(match: re.Match) -> str:
            return f"{match.group(1)}{match.group(2)}from {mapping[match.group(3)]} import "

        def sub_plain(match: re.Match) -> str:
            return f"{match.group(1)}{match.group(2)}import {mapping[match.group(3)]}{match.group(4)}"

        text = pattern.sub(sub_from, text)
        text = plain.sub(sub_plain, text)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changes.append(f"rewrote imports in {path.relative_to(REPLICATION)}")

    return changes


def main() -> int:
    total = 0
    for paper_dir in sorted(PAPERS.iterdir()):
        if not paper_dir.is_dir():
            continue
        changes = rewrite(paper_dir)
        if changes:
            print(f"\n{paper_dir.name}:")
            for change in changes:
                print(f"  {change}")
            total += len(changes)
    print(f"\n{total} change(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
