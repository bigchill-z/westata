# The independent conformance sweep

A check that gets skipped is worth nothing. This sweep exists because the library's own
test suite passed at **1247 tests** while an independent comparison against live Stata found **24
defects** — including `ologit`'s `e(V)` being off by 20× and `nl`'s standard errors by seven orders
of magnitude.

The reason is structural, not carelessness: **a test written from the same understanding as the
code cannot detect a misunderstanding shared by both.** The sweep avoids that by comparing against
Stata itself, on data built for the comparison, at full `e(b)`/`e(V)` precision.

## Running it

```bash
.venv\Scripts\python.exe -m retest.sweep                 # everything, then compare
.venv\Scripts\python.exe -m retest.sweep --domains tseries panel
.venv\Scripts\python.exe -m retest.sweep --no-run        # compare saved results only
```

Requires StataNow with `pystata` available. Full run is roughly 25 minutes; the timeout varies by
domain (`limited_quantile` alone takes about 4 minutes because of the bootstrap entries).

Exit codes:

| Code | Meaning |
|---|---|
| 0 | every registry level is supported by the measurement |
| 1 | **a registry level is better than measured** — a false claim |
| 2 | the sweep could not run (no Stata, a domain died, no results) |

## What it checks

Three things, in order:

1. **Runs each domain** and confirms it produced results. A domain exits non-zero when it *finds
   defects*, so completion is judged by whether `results.jsonl` was rewritten — not by exit code.
   Reading the exit code as a crash would have hidden every finding.
2. **Regenerates `retest/aggregate.json`** — the machine-readable table of all comparison points.
3. **Diffs measured tiers against `tests/conformance/registry.yml`** and fails when a registry
   level is better than the measurement supports.

Only the dangerous direction fails the run. A level that is too *low* is conservative; a level that
is too *high* is a false claim, and that is what this exists to catch.

## How a tier is assigned

Each comparison point gets a coefficient tier and a standard-error tier, from A (within 1e-9) to E
(materially wrong). A row's tier is the **worse of the two**, and an entry's tier is the worst among
the invocations that actually ran.

That rule is not decoration. `teffects nnmatch` has exact coefficients and standard errors 5–16%
away; grading on coefficients alone would have promoted it, which is precisely the error the sweep
was built to prevent.

## Row filtering — the part that produced the most wrong answers

`retest/lib/tiering.py` holds three rules, each learned by getting it wrong first:

| Rule | What it excludes | What went wrong without it |
|---|---|---|
| Drop **PROBE** rows | the sweep's own diagnostics, e.g. `var: VAR(1) standard errors: internal DOF factor` | `var` was demoted to E1 while its actual output measured at 1e-14 |
| Drop **ALTERNATIVE** rows | comparisons against a mapping the repair round *corrected*, e.g. `ivregress 2sls vce(robust) vs westata HC1` | `ivregress` and `areg` were penalised **for having been fixed** |
| Use an **explicit** entry→row map | keyword matching | `jackknife: regress` was charged to `regress`; `teffects ra/ipw/aipw` absorbed `nnmatch`'s failures |

Attribution is written out explicitly rather than inferred. Five successive heuristics each fixed
one case and broke another, because the registry and the sweep were written independently and their
vocabularies collide: `regress, vce(robust)` is one entry but reads like an option; `teffects ra`
and `teffects nnmatch` share a word; `xtreg, fe` is one entry whose comma is not a separator.

## Files

```
retest/
  sweep.py              entry point: run, aggregate, compare
  aggregate.py          assemble per-domain results into aggregate.json
  aggregate.json        every comparison point with its tiers
  REPORT.md             the baseline sweep: 391 points, 24 defects, and what was wrong
  FIX_REPORT.md         what was repaired, what was left, and the 8 capability gaps
  lib/
    stata_probe.py      Stata session, full-precision capture, naming alignment
    tiering.py          row filtering, tier rules, entry→row mapping
  domains/
    {linear_iv,panel,discrete,limited_quantile,tseries,systems_causal,postest}/
      run.py            one domain's comparisons
      results.jsonl     its output
  fix/                  per-domain FIXES.md and reproduction scripts
```

## When the sweep fails

The message names the entry, the declared level, and the worst tier measured. Two responses are
legitimate:

- **fix the code** so the measurement improves, then regrade upward;
- **fix the entry** if the level was set from an incomplete comparison.

There is a third response that is not legitimate: widening the tolerance or lowering the level
until the check passes. That converts the check into decoration. If a difference is real and
irreducible, record it as the level it actually is and say why in the entry's notes — which is what
`arch`, `teffects nnmatch` and `permute` do.

## Tests for the check itself

`tests/test_retest_sweep.py` pins the rules above and asserts the end-to-end property — that no
registry entry claims a level better than the measurement. Those tests need no Stata session, so
they run in the normal suite; they exist because a conformance check that silently rots produces
confidence without evidence.
