# Using westata alongside Stata

This page is for one situation: you have a result from westata and a result from
Stata, and you need to know whether they can be compared, quoted interchangeably,
or used to check one another.

Most commands are interchangeable. Four are not, and every one of them is
interchangeable in *some* respects but not others. That asymmetry is what makes
them worth a page: a flat "differs by 0.2%" would tell you neither what is safe
nor what to watch for.

**Short version.**

| Command | Can compare coefficients? | Can compare z / p? | Can compare the decision? |
|---|---|---|---|
| `arch` | ⚠️ to ~4 digits | ❌ **no** | ✅ yes |
| `teffects nnmatch` | ✅ exact | ⚠️ to ~2 digits | ✅ yes |
| `xtabond` | ⚠️ to ~3 digits | ⚠️ | ✅ yes |
| `permute` | ✅ exact | ❌ not replicate-for-replicate | ✅ yes (distribution) |

"Can compare the decision" means: does the 5% verdict, the sign, or the selected
rank come out the same. For all four, it does.

---

## `arch` — do not quote the z statistics

**Status: converges to a different point than Stata, in westata's favour.**

The numbers, on a GARCH(1,1) fit to 800 observations:

| parameter | westata | Stata | gap |
|---|---|---|---|
| mean constant | −0.040276 | −0.040746 | 4.7e-04 |
| `arch` | 0.115746 | 0.116922 | 1.2e-03 |
| `garch` | 0.782847 | 0.780018 | 2.8e-03 |
| `omega` | 0.080299 | 0.081835 | 1.5e-03 |

Those are the fourth-digit differences you would expect from two optimisers. The
**standard errors** are a different matter:

| parameter | westata z | Stata z | westata p | Stata p |
|---|---|---|---|---|
| mean constant | −1.377 | −1.389 | 0.169 | 0.165 |
| `arch` | 4.144 | 3.101 | 3.4e-05 | 1.9e-03 |
| `garch` | **16.942** | **9.755** | 2.2e-64 | 1.8e-22 |
| `omega` | 3.166 | 1.962 | **1.5e-03** | **4.97e-02** |

Look at `omega`: Stata puts it at p = 0.0497, marginally significant at 5%.
westata puts it at p = 0.0015, decisively significant. Both reject at 5%, so the
verdict agrees — but a reader who saw only "significant" would miss that one is
on the boundary and the other is nowhere near it.

### Why

Stata reports `e(technique) = bhhh bfgs` and `e(vcetype) = OPG`, with
`nrtolerance(1e-5)`. It runs BHHH to build an OPG-based Hessian approximation,
polishes with BFGS, and stops when the scaled gradient `g'H⁻¹g < 1e-5`.

That criterion is met *before* the likelihood is flat. Measured directly:

```
westata's likelihood maximum         ll = -1025.4148259782
Stata's reported likelihood          ll = -1025.4176185349
```

**westata's likelihood is higher by 2.8e-03.** Reached identically from three
starting points, with BFGS tolerances from 1e-6 to 1e-12, and after 2 to 20 BHHH
iterations — the maximum is stable and does not move.

An optimiser that stops early lands *below* the maximum, and that is exactly what
Stata's number is. So Stata is not maximising a different function; it is stopping
short of the same function's peak.

### What to do

- **Do** compare coefficients to about four significant figures.
- **Do** compare the significance verdict.
- **Do not** quote a z statistic, a Wald statistic, or a p-value from westata as
  if it were Stata's. They are computed at different points.
- If you need to reproduce a published Stata table exactly, use Stata.

westata does not chase Stata's stopping point, because doing so would require
reproducing an undocumented internal Hessian path *and* would move the estimates
away from the maximum-likelihood answer. The larger likelihood is the more
defensible estimate; the difference is a stopping rule, not an error.

---

## `teffects nnmatch` — coefficients exact, standard errors approximate

**Status: point estimates exact; standard errors 1.59% high.**

| | westata | Stata | gap |
|---|---|---|---|
| ATE | 2.495899509193757 | 2.4958995091937606 | 3.6e-15 |
| standard error | 0.10458249615030689 | 0.10627231325063335 | **1.59%** |

The point estimate is exact to machine precision. Only the standard error
differs, and it is uniformly slightly *small*.

### Why

Stata's default is `vce(robust, nn(2))` — "robust standard errors estimated using
two matches". The variance involves a within-treatment matching step whose exact
form lives in `_matchnn_estimates`, a compiled Mata function. Its symbol table is
readable (`_matchnn_biasAdjustedMeans`, `__matchnn_lsfit`, `_matchnn_indices`,
`_matchnn_neighbors`, `_matchnn_scaleVariables`, `_matchnn_estimates`) but its
arithmetic is not.

Five candidate forms of the within-treatment variance were measured against Stata
across `nn` from 2 to 8. The best tracks Stata's standard error with a ratio
stable at 0.983–0.990 — a near-constant factor, not a structural difference.

### What to do

- **Do** use the ATE. It is exact.
- **Do** treat the standard error as approximate to about 1.5%.
- **Do** trust the significance verdict: tested across true effects of 0, 0.05,
  0.10, 0.15 and 0.20, the verdict agreed in every case. The largest movement seen
  was p = 0.0953 → 0.1048, which crosses no threshold.
- **Do not** quote the standard error to three significant figures as Stata's.

Because westata's standard error is slightly *small*, its confidence intervals are
slightly *narrow*. If a conclusion depends on a narrow miss — p between about 0.04
and 0.06 — check it in Stata.

---

## `xtabond` — coefficients agree to three digits

**Status: coefficients 0.2–0.6% apart; every decision agrees.**

| panel | westata | Stata | relative gap |
|---|---|---|---|
| φ = 0.5 | 0.46379405 | 0.46102130 | 6.0e-03 |
| φ = 0.3 | 0.56504555 | 0.56358038 | 2.6e-03 |

Both fits give p values below 1e-25, so no significance threshold is anywhere
near. The sign agrees.

### Why

`xtabond` dispatches to `xtdpd`, which is compiled. One-step GMM is implemented
here from `[XT] xtdpd` and the instrument count matches Stata exactly — 30
instruments for `lags(1)` on a T=9 panel — but Stata's moment-weighting details
are not documented in enough detail to close the last 0.2%.

### What to do

- **Do** use westata for exploration and for the sign / significance.
- **Do** treat the coefficient as good to about three significant figures.
- **Do not** report the coefficient to five digits in a table that claims to
  reproduce Stata.
- Note that Sargan / Hansen / AR(1) / AR(2) tests are **deliberately not
  reported**: the manual does not fully specify their construction, and an
  approximate statistic under Stata's name would look more authoritative than its
  absence. Use Stata for those.

---

## `permute` — right distribution, different draws

**Status: the generator matches Stata; the shuffle order does not.**

The random-number generator is Stata's: MT19937-64, verified against the two
reference values printed in `[R] set seed` (seed 123456789 → 0.348871704556195,
0.266885709753138) and against resampling indices, 400 out of 400 identical.

What is not reproducible is the **shuffle order**. Stata's permutation routine
does not document its call sequence, so the same seed does not produce the same
permutation.

### What to do

- **Do** use the p-value. It is computed from the correct generator and is
  statistically meaningful.
- **Do not** expect a specific seed to produce Stata's exact permutation, or the
  same p-value to the last digit.
- The same applies to `bootstrap` *only* in the sense that its replicate draws are
  exact (verified: `b` gap 0, `se` gap 1.1e-9) — `bootstrap` **is**
  replicate-for-replicate comparable, `permute` is not.

---

## A general caution

For all four commands, the gap is in a quantity computed **at** the estimate
rather than in the estimate itself: a standard error, a test statistic, a p-value.
Point estimates are consistently better than the quantities that depend on
curvature or on a matching neighbourhood.

That gives a practical rule:

> Compare point estimates freely. Before comparing a standard error, a z
> statistic, or a p-value across implementations, check this page.

## Where the details live

- `tests/conformance/registry.yml` — per-command evidence level, the measured gap,
  and the reasoning for the level.
- `docs/GAP_REGISTER.md` — what is deliberately not implemented, and why.
- Every E2 entry names the exact measurement that keeps it from E3.
