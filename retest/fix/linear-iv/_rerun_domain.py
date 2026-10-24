"""Run ``retest/domains/linear_iv/run.py`` with its recorder redirected.

The domain script writes ``retest/domains/linear_iv/{results.jsonl,summary.json}``,
which the Lead owns.  This wrapper patches ``Record`` so the output lands in
``retest/fix/linear-iv/_rerun/`` instead, and prints only the failing items --
that is the evidence for "which run.py entries change status after the fix".
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "retest" / "lib"))
sys.path.insert(0, str(REPO / "src"))

import stata_probe  # noqa: E402

OUT = HERE.parent / "_rerun"


class RedirectedRecord(stata_probe.Record):
    def __init__(self, domain: str, outdir=None):
        OUT.mkdir(parents=True, exist_ok=True)
        super().__init__(domain, outdir=OUT)


stata_probe.Record = RedirectedRecord

# The domain module binds the symbol at import time.
import importlib  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "linear_iv_run", REPO / "retest" / "domains" / "linear_iv" / "run.py"
)
module = importlib.util.module_from_spec(spec)
module.__dict__["Record"] = RedirectedRecord
warnings.simplefilter("ignore")
spec.loader.exec_module(module)

raise SystemExit(module.main())
