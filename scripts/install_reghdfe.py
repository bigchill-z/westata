"""Install reghdfe and ftools into Stata from the authors' GitHub sources.

SSC (``fmwww.bc.edu``) is unreachable from this network, but
``raw.githubusercontent.com`` is not, and both packages' sources live in the
authors' repositories.  This script mirrors what ``ssc install`` would do:
download every ``.ado``/``.mata``/``.sthlp`` file into ``PLUS/`` and compile the
Mata library inside the embedded Stata session.

Only files the packages actually ship are downloaded; the lists below are the
``src/`` directories of the two repositories.

Usage::

    .venv\\Scripts\\python.exe scripts\\install_reghdfe.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

FTOOLS_FILES = [
    "fcollapse.ado", "fcollapse.sthlp", "fcollapse_functions.mata",
    "fcollapse_main.mata", "fegen.ado", "fegen.sthlp", "fegen_group.ado",
    "fisid.ado", "fisid.sthlp", "flevelsof.ado", "flevelsof.sthlp",
    "fmerge.ado", "fmerge.sthlp", "freshape.ado", "fsort.ado", "fsort.sthlp",
    "ftab.ado", "ftools.ado", "ftools.mata", "ftools.sthlp",
    "ftools_common.mata", "ftools_experimental.mata", "ftools_hash1.mata",
    "ftools_main.mata", "ftools_plugin.mata", "ftools_type_aliases.mata",
    "join.ado", "join.sthlp", "local_inlist.ado", "local_inlist.sthlp",
    "ms_add_comma.ado", "ms_compile_mata.ado", "ms_expand_varlist.ado",
    "ms_fvstrip.ado", "ms_fvstrip.sthlp", "ms_fvunab.ado", "ms_get_version.ado",
    "ms_parse_absvars.ado", "ms_parse_depvar.ado", "ms_parse_varlist.ado",
    "ms_parse_vce.ado", "parallel_map.ado", "parallel_map.sthlp",
]

REGHDFE_FILES = [
    "estfe.ado",
    "reghdfe.ado", "reghdfe.mata", "reghdfe.sthlp",
    "reghdfe_estat.ado", "reghdfe_footnote.ado", "reghdfe_header.ado",
    "reghdfe_p.ado", "reghdfe_programming.sthlp",
]

FTOOLS_BASE = "https://raw.githubusercontent.com/sergiocorreia/ftools/master/src/"
REGHDFE_BASE = "https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/src/"

# reghdfe checks its dependencies through the ``require`` package, which is what
# raises "reghdfe requires uses the require package" when it is absent.
REQUIRE_BASE = "https://raw.githubusercontent.com/sergiocorreia/stata-require/master/src/"
REQUIRE_FILES = ["require.ado", "require.sthlp"]


def plus_dir() -> Path:
    """Locate Stata's PLUS ado directory (where ``ssc install`` writes).

    The path is taken from Stata itself rather than guessed: ``sysdir`` on this
    machine reports ``C:\\Users\\<user>\\ado\\plus``, which is not the
    ``<stata>/ado/plus`` layout older installations use.
    """

    with StataOracle() as oracle:
        output = oracle.get_output("sysdir")
    for line in output.splitlines():
        if line.strip().startswith("PLUS:"):
            path = line.split(":", 1)[1].strip()
            # Stata prints forward slashes with a trailing separator.
            return Path(path.rstrip("/\\"))
    raise SystemExit(f"could not parse PLUS from sysdir:\n{output}")


def download(base: str, name: str, target: Path) -> tuple[str, int]:
    url = base + name
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, OSError) as exc:
        return f"FAILED {name}: {exc}", 0
    destination = target / name
    destination.write_bytes(payload)
    return f"{name:32s} {len(payload):>8d} bytes", len(payload)


def main() -> int:
    target = plus_dir()
    # The author's files live under a letter directory in Stata's layout, but
    # putting them directly in PLUS works because PLUS is on the adopath when
    # PERSONAL is not.
    target.mkdir(parents=True, exist_ok=True)
    print(f"installing into {target}")

    failures = 0
    total = 0
    for base, files, label in [
        (REQUIRE_BASE, REQUIRE_FILES, "require"),
        (FTOOLS_BASE, FTOOLS_FILES, "ftools"),
        (REGHDFE_BASE, REGHDFE_FILES, "reghdfe"),
    ]:
        print(f"\n--- {label} ---")
        for name in files:
            message, size = download(base, name, target)
            total += size
            if size == 0:
                failures += 1
            print(f"  {message}")

    print(f"\ndownloaded {total} bytes, {failures} failure(s)")

    print("\n--- compiling Mata (ftools) inside Stata ---")
    with StataOracle() as oracle:
        for command in [
            "discard",
            "mata: mata clear",
            "mata: mata mlib index",
            "which ftools",
            "which require",
            "which reghdfe",
        ]:
            try:
                print(f"$ {command}")
                output = oracle.get_output(command)
                if output.strip():
                    print("  " + output.strip().splitlines()[0][:160])
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED: {str(exc)[:200]}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
