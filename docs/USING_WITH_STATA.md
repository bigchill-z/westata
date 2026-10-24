# Using westata alongside Stata

You have a number from westata and a number from Stata. This page answers one question:
**can the two be compared?**

It is organised by risk rather than by command family, because the failures are not
distributed evenly. In the linear models the answer is "yes, to machine precision". In the
non-linear models the point estimate is usually exact and the **standard error** usually is
not — which still corrupts the z statistic, the p-value and the confidence interval.

Current grading follows an independent 386-point sweep against a live StataNow 19.5 (MP)
session, recorded in `retest/REPORT.md` and `retest/FIX_REPORT.md`:

| Level | Entries | Meaning |
|---|---|---|
| E3 | **51** | Agrees with full-precision `e(b)`/`e(V)` on every invocation the sweep exercised |
| E2 | **26** | Bounded, measured difference — usable, with the caveat below |
| E1 | **10** | Materially differs on at least one measured invocation, or not covered |

---

## Safe to quote directly

These families measured at machine precision across every invocation the sweep tried:

`regress` (all four weight types, `i.` factors, `vce(robust)`, `vce(cluster)`) ·
`areg` · `reghdfe` (one- and two-way) · `ivregress 2sls/liml/gmm` · `xtreg fe/re/be/fd` ·
`xtivreg fe/re` · `sureg` · `mvreg` · `eivreg` · `test` · `lincom` · `contrast` ·
`svy:` mean/total/proportion/regress · `xtabond` (after repair)

For these, coefficients **and** standard errors agree at 1e-15. Report them as Stata's.

---

## Safe with a stated caveat

### Second-order quantities in non-linear ML

Applies to: `logit`, `probit`, `poisson`, `nbreg`, `ologit`, `oprobit`, `zip`, `zinb`,
`ztp`, `tnbreg`, `tobit`, `truncreg`, `intreg`, `heckman`, `etregress`, `qreg`, `arima`.

**The coefficient is exact; the standard error may differ in the third or fourth digit.**
The cause is always the same shape of problem: a Jacobian or delta-method transform applied
on the wrong scale, or an optimiser stopping at a different point of a flat likelihood.

Practical rule:

- quoting a coefficient: fine
- quoting a standard error: round to 3 significant figures and say which library produced it
- quoting a z, p or confidence interval at a decision-relevant boundary: **verify in Stata**

Where this genuinely bites: `ologit`/`oprobit` cut-point standard errors were off by 20x
before the repair, and `tobit`'s `se(/sigma)` by 66%. Both are fixed, both are now inside
E2 rather than E3 because the gradient is not exactly zero at Stata's own stopping point.

### `arch` — do not quote z or p

Coefficients agree to about 4 digits. **The z statistics do not agree, and the reason is
that Stata stops early.**

| | westata | Stata |
|---|---|---|
| likelihood at the estimate | −1025.4148259782 | −1025.4176185349 |
| z(`garch`) | 16.942 | 9.755 |
| p(`omega`) | 1.5e-03 | 4.97e-02 |

Stata runs `bhhh bfgs` with `nrtolerance(1e-5)` and halts before the likelihood is flat;
westata converges further and finds a **higher** likelihood. So westata's estimates are the
maximum-likelihood ones and Stata's are its own stopping point — but the test statistics are
computed at different points and are not interchangeable.

Read `p(omega)` in that table: Stata puts it at 0.0497, marginally significant; westata at
0.0015, decisively so. Both reject at 5%, so a "significant / not significant" summary hides
the difference. **Use Stata for `arch` inference.**

### `teffects nnmatch` — coefficient exact, SE approximate

The ATE is exact to 3.6e-15. The standard error is 5–16% away, depending on the matching
configuration, because the within-treatment variance is computed inside a compiled Mata
function (`_matchnn_estimates`) whose arithmetic is not documented. Its symbol table is
readable and confirms the structure; the arithmetic is not.

Practically: **the significance verdict agreed in every scenario tested**, and the largest p
movement observed was 0.0953 → 0.1048, crossing no threshold. Use it, but do not quote the
standard error to three significant figures.

### `permute` — right distribution, different draws

The random-number generator is Stata's own MT19937-64, verified against the two reference
values in `[R] set seed` and against resampling indices 400-for-400. What is not
reproducible is Stata's **shuffle order**, which is undocumented, so a given seed does not
produce the same permutation.

`bootstrap` **is** replicate-for-replicate comparable (after repair: `b` gap 0, `se` gap
1.1e-9). `permute` gives a statistically meaningful p-value but not the same draw sequence.

---

## Do not compare

These measured materially wrong on at least one invocation and are graded E1:

- `arima`
- `heckman`
- `etregress`
- `rreg`
- `bsqreg`
- `xtlogit, pa`
- `xtgls`
- `frontier`
- `teffects nnmatch`
- `permute`

Either the difference is structural (a different estimator, a different distributional
assumption) or the invocation is unsupported and raises. `docs/GAP_REGISTER.md` records the
reason for each.

---

## The general rule

> **Compare point estimates freely. Before comparing a standard error, a z statistic or a
> p-value across implementations, check this page.**

The pattern is consistent: coefficients are estimated by solving a well-conditioned problem, and
two implementations land on the same answer. Second-order quantities come from a Hessian, a
Jacobian, a sandwich, or a stopping rule — and every one of those is a place where two
libraries can differ while both are "right".

## Where the evidence lives

- `retest/REPORT.md` — the 386-point sweep, its per-domain findings, and the 24 defects it found
- `retest/FIX_REPORT.md` — what was repaired, what was deliberately left, and the 8 remaining capability gaps
- `tests/conformance/registry.yml` — per-entry level with the measured tier and row count
- `docs/GAP_REGISTER.md` — what is deliberately not implemented, and why
