# Missing auxiliary statistics and display features — linear / IV / panel / time-series

**Audit target**: `westata` at `D:\myCodeProject\statapy` (Stata-conformance econometrics library)
**Scope**: `regress` (OLS, robust, cluster), `areg`, `reghdfe`, `ivregress 2sls/liml/gmm`, `etregress`, `didregress`, `xtdidregress`, `xtreg fe/be/re/fd`, `xtivreg`, `xtgls`, `xtpcse`, `xtabond`, `arima`, `var`, `vec`, `vecrank`, `arch`, `dfuller`, `pperron`, `xtunitroot`
**Method**: source reading of every `e` dict constructor + StataNow 19.5 fixture comparison + live fits with `./.venv/Scripts/python.exe` dumping `sorted(result.e.keys())`
**Rule**: every row cites `path:line`. "Stata stores it" and "Stata prints it" are tracked separately.

---

## 0. Executive summary

The user's report — *"Stata prints RSS/TSS and westata's result does not contain them, and printing the covariance matrix of regression coefficients is missing"* — is **half right and half wrong**, and the two halves need different fixes:

| Claim | Verdict | Evidence |
|---|---|---|
| "RSS/TSS missing" | **Wrong for `regress`/`ivregress`** — `e(rss)`/`e(mss)` are present and Stata's `regress` never stores `e(tss)`. **Right for `xtreg`, `areg`/`reghdfe`, `xtgls`** — those *do* have a Stata `e(tss)`/`e(tss_within)` with no westata counterpart. | `src/westata/estimators/linear.py:322-323`, `tests/conformance/stata_fixtures/regress_auto.json:32-33`, `tests/conformance/stata_fixtures/xtreg_fe.json:51`, `src/westata/estimators/panel.py:741-754` |
| "print covariance matrix of coefficients missing" | **Correct, and it is a pure display gap.** `cov_params()` already returns the full `k×k` matrix for every estimator; nothing renders it as a table anywhere in the package. | `src/westata/results/base.py:203-206`, no `estat`/`vce` renderer exists (§3, Q2) |

The three cheapest and highest-value fixes are: (1) `e(tss)`/`e(tss_within)` for `xtreg`/`areg`/`reghdfe`/`xtgls` — the numbers are **already computed internally** and thrown away; (2) the `xtreg` variance-component block (`r2_b`, `r2_o`, `sigma_u`, `rho`, `theta`, `N_g`, `Tbar`, `g_*`) — same story; (3) a `estat vce`-style covariance/correlation matrix renderer — new display code, zero new math.

---

## 1. Per-estimator `e()` key tables

Notation: **MISSING** = in the Stata fixture, absent from westata. **PRESENT** = in both. **fixture-only** = in westata but not that fixture (not a gap). **westata-only** = westata addition (documented, not a conformance gap).

### 1.1 `regress` (OLS) — nonrobust / `vce(robust)` / `vce(cluster)`

Fixture: `tests/conformance/stata_fixtures/regress_auto.json:24-38` (and `regress_auto_robust.json:24-38`, `regress_auto_factor.json:34-48`)
westata `e` built at `src/westata/estimators/linear.py:311-332`.

| Stata key | In fixture? | In westata? | Verdict |
|---|---|---|---|
| `N` | yes | yes | PRESENT |
| `sum_w` | yes | yes | PRESENT |
| `df_m` | yes | yes | PRESENT |
| `df_r` | yes | yes | PRESENT |
| `F` | yes | yes | PRESENT |
| `p` | **no** | no | n/a — `regress` stores no `e(p)`; the summary recomputes `Prob > F` (`src/westata/results/summary.py:153-155`) |
| `r2` | yes | yes | PRESENT |
| `r2_a` | yes | yes | PRESENT |
| `rmse` | yes | yes | PRESENT |
| `mss` | yes | yes | PRESENT |
| `rss` | yes | yes | PRESENT |
| `tss` | **no** | no | **fixture-only — and this is Q1's answer, see §2** |
| `ll` | yes | yes | PRESENT |
| `ll_0` | yes | yes | PRESENT |
| `rank` | yes | yes | PRESENT |
| `N_clust` | (cluster fixtures) | yes | PRESENT (`linear.py:332`) |
| `df_r_mb` | no | no | westata does not store the model-based df separately; see §2 note |
| `aic`, `bic`, `sigma` | no | yes | westata-only additions (`linear.py:319`, `:325-326`); `estat ic` equivalent, documented |

Live confirmation (`auto.dta`, `mpg ~ weight + dom`):
```
['F', 'N', 'aic', 'bic', 'df_m', 'df_r', 'll', 'll_0', 'mss', 'r2', 'r2_a', 'rank', 'rmse', 'rss', 'sigma', 'sum_w']
+ 'N_clust' under cov_type='cluster'
```

### 1.2 `areg` / `reghdfe` (FEOLS)

Fixtures: `areg.json:24-47`, `reghdfe_firm.json:24-54`, `reghdfe_firm_year.json:24-54`, `reghdfe_firm_cluster.json:24-57`, `reghdfe_firm_robust.json:24-54`, `reghdfe_firm_year_cluster.json:24-57`
westata `e` built at `src/westata/estimators/hdfe.py:538-550`.

| Stata key | In fixture? | In westata? | Verdict |
|---|---|---|---|
| `N`, `rank`, `df_r`, `df_m`, `rmse`, `F`, `ll`, `r2`, `r2_a`, `rss` | yes | yes | PRESENT |
| `mss` | yes | **no** | **MISSING** |
| `tss` | yes | **no** | **MISSING** |
| `tss_within` | yes | **no** | **MISSING** |
| `r2_within` | yes | **no** | **MISSING** |
| `r2_a_within` | yes | **no** | **MISSING** |
| `df_a` | yes | **no** | **MISSING** (absorbed degrees of freedom) |
| `df_a_nested`, `df_a_redundant`, `df_a_initial` | yes | no | MISSING (reghdfe-internal dof bookkeeping) |
| `ll_0` | yes | **no** | **MISSING** |
| `sumweights` | yes | no | MISSING (reghdfe spelling; westata has no `sum_w` here either) |
| `N_clust` | cluster fixtures | yes | PRESENT (`hdfe.py:550`) |
| `N_clust1`, `N_clustervars` | yes | no | MISSING (reghdfe multi-cluster bookkeeping) |
| `ic`, `converged`, `report_constant`, `N_full`, `num_singletons`, `drop_singletons`, `N_hdfe`, `N_hdfe_extended` | yes | no | reghdfe-internal provenance; arguably **not applicable** to a PyFixest-backed adapter (no singleton policy, no reghdfe solver) |

Live confirmation (5×4 panel, `y ~ x1 + x2 | firm`):
```
['F', 'N', 'df_m', 'df_r', 'll', 'r2', 'r2_a', 'rank', 'rmse']      (+ 'N_clust' when clustered)
```
Note `tss` **is** computed at `src/westata/estimators/hdfe.py:522` and simply never stored — see §4(i).

### 1.3 `ivregress 2sls / liml / gmm`

Fixtures: `iv_liml.json:24-38`, `iv_gmm.json:24-38` (2sls coverage measured live in `retest/domains/linear_iv/results.jsonl:55`)
westata `e` built at `src/westata/estimators/iv.py:809-839`.

| Stata key | In fixture? | In westata? | Verdict |
|---|---|---|---|
| `N`, `df_m`, `r2`, `r2_a`, `rmse`, `mss`, `rss`, `tss` | yes | yes | PRESENT |
| `rank` | yes | yes | PRESENT |
| `chi2` | yes | yes | PRESENT |
| `k_endog` | yes | yes | PRESENT |
| `iterations` | yes | yes | PRESENT |
| `kappa` | liml: yes | yes | PRESENT |
| `J` | gmm: yes | yes | PRESENT (`iv.py:842-854`) |
| `df_r` | yes (= 8.99e307 sentinel) | yes (= `N`) | **PRESENT but semantically different** — documented deviation, `iv.py:111-123` |
| `F`, `F_first`, `ll` | no | yes | westata-only (`iv.py:128-134`) |
| `rank_mb` | yes under robust/cluster | yes | PRESENT (`iv.py:834-839`) |
| `N_clust` | yes under cluster | yes | PRESENT (`iv.py:833`) |

This family is essentially complete. Live keys (2sls over-identified):
```
['F', 'F_first', 'N', 'chi2', 'df_m', 'df_r', 'iterations', 'k_endog', 'kappa', 'll', 'mss', 'r2', 'r2_a', 'rank', 'rmse', 'rss', 'tss']
+ 'rank_mb' (robust/cluster), 'N_clust' (cluster), 'J' (gmm)
```

### 1.4 `etregress`

Fixture: `e_upgrade_etregress.txt:1-25` (b/V only — no scalar capture), `e_upgrade_outputs.txt:57-68`
westata `e` built at `src/westata/estimators/limited.py:2107-2123` (ML) and `:2235-2251` (control-function).

| Stata key | In fixture? | In westata? | Verdict |
|---|---|---|---|
| `N`, `ll`, `athrho`, `lnsigma`, `rho`, `sigma`, `lambda` | not captured | yes (ML) | PRESENT-in-source |
| `se_rho`, `se_sigma`, `se_lambda` | not captured | yes | PRESENT-in-source |
| `N_treated` | not captured | yes | westata extra |
| `chi2`, `chi2type`, `p`, `r2_p`, `ll_0` | not captured | yes (control) | PRESENT-in-source |
| `df_m`, `df_r` | not captured | yes | westata convention |

**Caveat**: no scalar fixture exists for `etregress`, so this table is a source-level inventory, not a verified diff. Live keys: ML `['N','N_treated','athrho','converged','df_m','df_r','lambda','ll','lnsigma','method','rho','se_lambda','se_rho','se_sigma','sigma']`; control `['N','N_treated','chi2','chi2type','df_m','df_r','estimator','ll','ll_0','p','r2_p','rank','rho','se_corrected','sigma','vce']`.

### 1.5 `didregress` / `xtdidregress`

Fixture: `p11_did.txt:1-70` (printed table only — no scalar capture)
westata `e` assembled at `src/westata/estimators/did.py:778-799`.

| Stata key | In fixture? | In westata? | Verdict |
|---|---|---|---|
| `N`, `df_r`, `N_clust` | printed | yes | PRESENT |
| `r2`, `r2_a`, `rmse`, `F`, `ll`, `df_m` | **not printed by Stata** | yes | **westata-only, documented at `did.py:81-89`** |

Live keys: `['F','N','N_clust','data_type','df_m','df_r','estimand','ll','r2','r2_a','rank','rmse']`.
Stata's `didregress` prints only the ATET row plus the treatment/time table; westata carries the whole FE regression block. This is **not a gap** — it is extra diagnostics. The genuine display gap is that westata never prints Stata's *"Treatment and time information"* table (`p11_did.txt:5-19`) nor the *"Data type:"* line nor the *"Note: ATET estimate adjusted for covariates, group effects, and time effects."* footnote.

### 1.6 `xtreg fe / be / re / fd`

Fixtures: `xtreg_fe.json:24-62`, `xtreg_be.json:24-48`, `xtreg_re.json:24-45`
westata `e` built at `src/westata/estimators/panel.py:741-757`.

| Stata key | `fe` fixture | `be` fixture | `re` fixture | In westata? | Verdict |
|---|---|---|---|---|---|
| `N`, `df_m`, `df_r`, `r2`, `r2_a`, `rmse`, `F`, `ll`, `rank` | yes | yes | yes | yes | PRESENT |
| `df_b` | yes | no | no | yes | PRESENT |
| `rss` | yes | yes | no | yes | PRESENT |
| `tss` | **yes** | **no** | no | **no** | **MISSING (fe)** |
| `mss` | **yes** | **yes** | no | **no** | **MISSING (fe, be)** |
| `tss_within` | not in fixture (Stata stores it) | no | no | **no** | **MISSING (fe)** |
| `r2_w` | yes | yes | yes | **no** (only `r2`, which equals `r2_w`) | **MISSING — name-level** |
| `r2_b` | yes | yes | yes | **no** | **MISSING** |
| `r2_o` | yes | yes | yes | **no** | **MISSING** |
| `r2_c` | yes | no | no | **no** | **MISSING (fe)** |
| `sigma_e` | yes | no | yes | **no** (only `rmse`, which equals `sigma_e`) | **MISSING — name-level** |
| `sigma_u` | yes | no | yes | **no** | **MISSING** |
| `sigma` | yes | no | yes | **no** | **MISSING (fe, re)** |
| `rho` | yes | no | yes | **no** | **MISSING** |
| `theta` | no | no | **yes** | **no** | **MISSING (re)** |
| `corr` | yes | no | no | **no** | **MISSING (fe)** |
| `N_g` | yes | yes | yes | **no** (westata stores `n_entities`) | **MISSING — name-level** |
| `Tbar` | yes | yes | yes | **no** | **MISSING** |
| `Tcon` | yes | yes | yes | **no** | **MISSING** |
| `g_min`, `g_avg`, `g_max` | yes | yes | yes | **no** | **MISSING** |
| `k_absorb` | yes | no | no | **no** | **MISSING (fe)** |
| `df_a` | yes | no | no | **no** | **MISSING (fe)** |
| `F_f` | **yes** | no | no | **no** | **MISSING (fe)** |
| `p_f` | **yes** | no | no | **no** | **MISSING (fe)** |
| `ui` | **yes** | no | no | **no** | **MISSING (fe)** |
| `F_absorb`, `p_absorb` | no (areg has them) | no | no | **no** | **MISSING (fe)** |
| `p` | yes | yes | yes | **no** | **MISSING — trivial** |
| `rc` | yes | no | no | **no** | Stata return code; arguably not applicable |
| `chi2`, `chi2type` | no | no | yes | **no** | **MISSING (re)** — Stata's `xtreg, re` reports Wald `chi2`, not `F` |
| `thta_min/5/50/95/max` | n/a | n/a | unbalanced only | no | not applicable to balanced fixtures |

Live confirmation (5×4 panel, `y ~ x1 + x2`), **identical for all four estimators**:
```
['F', 'N', 'df_b', 'df_m', 'df_r', 'll', 'n_entities', 'r2', 'r2_a', 'rank', 'rmse', 'rss']
```
This is the single largest inventory gap in the audit: **all four `xtreg` estimators return the same twelve keys**, so `xtreg, be` loses `mss` (which its fixture has), `xtreg, re` loses `chi2`/`sigma_u`/`sigma_e`/`sigma`/`rho`/`theta`/`r2_w`/`r2_b`/`r2_o`/`N_g`/`Tbar`/`g_*`, and `xtreg, fe` loses the whole variance-component block.

Two documented semantic mismatches (not missing keys, but a reader trap):
* `e(df_m)`: Stata `fe` = 31 (includes 29 absorbed effects), westata = 2 (slopes only). Recorded at `retest/domains/panel/results.jsonl:4`; westata's `e(F)` still matches Stata, so the correct df is known internally but not stored under Stata's name.
* `e(r2)` in westata **is** Stata's `e(r2_w)` and `e(rmse)` **is** Stata's `e(sigma_e)` (verified 1e-16 in `retest/domains/panel/NOTES.md:314`) — but under Stata's names, so `e["r2_w"]` returns `None`.

### 1.7 `xtivreg`

No JSON scalar fixture. Display reference: `p9_outputs.txt:330-345`.
westata `e` built at `src/westata/estimators/panel_iv.py:864-903`.

Live keys: `fe` → `['N','N_g','chi2','chi2type','corr','df_m','df_r','g_avg','g_max','g_min','ll','method','p','r2','r2_b','r2_o','r2_w','rank','rho','rmse','rss','sigma_e','sigma_u']`; `re` adds `theta`.

`xtivreg` is the **richest** estimator in this audit — it stores everything `xtreg` does not (`r2_w/r2_b/r2_o`, `sigma_u`, `sigma_e`, `rho`, `corr`, `N_g`, `g_min/g_avg/g_max`, `theta`, `chi2type`, `p`). Missing relative to Stata's `xtivreg`: `mss` (fe), `tss`, `df_b`, `Tbar`, `Tcon`, `k_absorb`, `df_a`, `F_f`, `ui`, `p_f`. This contrast is the strongest internal argument that the `xtreg` gap is cheap to close: the same quantities are already produced by a sibling class in the same file.

### 1.8 `xtgls`

Display reference: `p9_outputs.txt:468-489` (Stata prints "Estimated covariances / autocorrelations / coefficients", `Wald chi2`, `Time periods`).
westata `e` built at `src/westata/estimators/panel_iv.py:1059-1084`.

Live keys: `['F','N','N_g','N_t','chi2','chi2type','corr','df','df_m','df_pear','df_r','g_avg','g_max','g_min','n_cf','n_cr','n_cv','p','panels','r2','r2_a','rank','rc','rmse']`.

| Stata key (printed/stored) | In westata? | Verdict |
|---|---|---|
| `n_cf`, `n_cv`, `n_cr`, `df`, `df_pear`, `chi2`, `panels`, `corr`, `N_g`, `N_t`, `g_*` | yes | PRESENT |
| `rss`, `mss`, `tss` | **no** (`r2`/`r2_a`/`rmse` are stored as NaN) | **MISSING** |
| `ll` | **no** | **MISSING** |
| `Wald chi2` / `Prob > chi2` | `chi2`/`p` stored | PRESENT as data; **not rendered** (see §3) |

`xtgls` deliberately stores `r2`/`r2_a`/`rmse`/`F` as `float("nan")` (`panel_iv.py:1074-1077`). Stata likewise prints no R-squared for `xtgls` (`p9_outputs.txt:472-482`), so NaN is the honest choice — but `rss`/`mss`/`tss` are computable and absent.

### 1.9 `xtpcse`

Display reference: `p9_outputs.txt:492-514`.
westata `e` built at `src/westata/estimators/panel_iv.py:1538-1574`.

Live keys: `['N','N_g','chi2','chi2type','corr','df','df_m','df_r','g_avg','g_max','g_min','missmeth','mss','n_cf','n_cr','n_cv','n_sigma','p','panels','r2','r2_a','rank','rc','rmse','rss','selection','sum_w']`.

Nearly complete: `rss`/`mss` present, `n_cf`/`n_cv`/`n_cr`/`n_sigma`/`missmeth`/`selection` all present. Missing: `tss`, `ll`, `df_b`, `N_t`, `Tbar`. Stata prints `R-squared` for `xtpcse` (`p9_outputs.txt:504`) and westata stores it — but §3 shows the renderer never shows it for `dist="z"` models.

### 1.10 `xtabond`

Display reference: `p9_outputs.txt:346-406` (Stata prints "Number of obs / Number of groups / Obs per group min-avg-max / Number of instruments / Wald chi2 / One-step results", plus the instrument block).
westata `e` built at `src/westata/estimators/panel_dynamic.py:427-458`.

Live keys: `['N','N_g','chi2','chi2type','df_m','df_r','instruments','instruments_difference','lags','method','p','rank','rss','sig2','sigma2','vce','zrank']`.

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `N_g`, `instruments`/`zrank`, `chi2`, `df_m`, `df_r` | yes | PRESENT |
| `g_min`, `g_avg`, `g_max` | **no** | **MISSING** (Stata prints min/avg/max obs per group, `p9_outputs.txt:353-356`) |
| `sargan`, `hansen`, `ar1`, `ar2`, `artests` | **no** | **MISSING** — deliberately, documented at `panel_dynamic.py:455-457` ("Deliberately absent: Sargan, Hansen, AR(1), AR(2)") |
| `sig2`/`sigma2` | yes | PRESENT |
| `rss` | yes | PRESENT |
| `idtype`, `eqnames`, the "Instruments for differenced/level equation" block | **no** | **MISSING as display** (`p9_outputs.txt:370-374`) |

The absent overidentification/serial-correlation tests are a documented, principled omission (approximating them under Stata's names would be worse than omitting). The group-size block is pure data availability.

### 1.11 `arima`

No JSON fixture; authoritative in-repo evidence: `docs/GAP_REGISTER.md:655-684` (§8.14) and `retest/domains/tseries/NOTES.md:248,264`.
westata `e` built at `src/westata/estimators/tseries.py:188-216`.

Live keys: `['N','aic','bic','chi2','chi2type','converged','df_m','df_r','ll','order','p','rank','seasonal_order','sigma','sigma2','trend']`.

| Stata key | In westata? | Verdict |
|---|---|---|
| `N` (estimation sample, post-differencing) | yes | PRESENT (`tseries.py:189`, convention fixed in `retest/domains/tseries/summary.json:66`) |
| `ll` | yes | PRESENT |
| `sigma` | yes | PRESENT — the §8.14 fix (`tseries.py:202-206`) |
| `sigma2` | yes | westata-only (Stata does not store it; both are provided deliberately, `tseries.py:196-201`) |
| `aic`, `bic` | yes | **PRESENT but only because westata computes them itself.** Stata's `arima` does **not** store `e(aic)`/`e(bic)`; they exist only via `estat ic` → `r(S)` (`retest/domains/tseries/run.py:262-273`, `NOTES.md:248`). westata's values agree, but they are westata-computed, not Stata-stored. |
| `chi2`, `p`, `df_m`, `chi2type` | yes | westata convention (Wald chi2 on all ARMA coefficients) |
| `converged` | yes | PRESENT |
| `rank` | yes | PRESENT |
| `order`, `seasonal_order`, `trend` | yes | westata extras |
| `df_r` | yes (= `N`) | westata convention |

Answer to Q4 in one line: **Stata stores `e(ll)`, `e(sigma)`, `e(N)`, `e(chi2)`/`e(F)`-family model test, `e(rank)`, `e(converged)`; it does *not* store `e(aic)`/`e(bic)` (only `estat ic`'s `r(S)`). westata stores all of those plus `e(sigma2)`, `e(aic)`, `e(bic)`, `e(order)`, `e(trend)`.**

### 1.12 `var`

Fixture: `var.json:60-97`.
westata returns a **`VARResults` dataclass, not `StataResults`** — it has **no `.e` at all** (`tseries.py:343`, confirmed live: `hasattr(vres, "e") == False`). Stata-facing values are exposed as properties/methods: `sigma_u` (`:386-389`), `cov_params`/`stata_cov_params` (`:391-408`), `bse` (`:410-418`), `stata_params()` (`:467-475`), `aic`/`bic` (`:477-483`), `granger_causality()` (`:507-563`), `irf()` (`:565-568`).

| Stata key | In fixture? | In westata? | Verdict |
|---|---|---|---|
| `b`, `V` | yes | yes (`stata_params()`, `cov_params`) | PRESENT |
| `neqs`, `k`, `k_dv`, `k_eq`, `tparms`, `rank` | yes | partial (`neqs`, `k_ar`, `k_trend`, `lags`, `names`) | **MISSING as `e()`** |
| `N`, `T`, `tmax`, `tmin`, `N_gaps` | yes | `nobs` only | **MISSING** (`T`, `tmax`, `tmin`, `N_gaps`) |
| `ll`, `ll_1`, `ll_2`, `df_eq`, `obs_1/2`, `k_1/2`, `df_m1/2` | yes | `raw_result.llf` reachable but not surfaced | **MISSING** |
| `rmse_1/2`, `r2_1/2` | yes | no | **MISSING** |
| `chi2_1/2`, `p_1/2` | yes | `granger_causality()` only | **MISSING as `e()`** |
| `aic`, `bic`, `hqic`, `fpe`, `sbic` | yes | `aic`, `bic` properties; `select_order()` DataFrame | **MISSING `hqic`, `fpe` as scalars** (present via `select_order`) |
| `mlag`, `detsig`, `detsig_ml` | yes | no | **MISSING** |

This is a **structural** gap: because `VAR.fit()` returns a dataclass rather than `StataResults`, every `var` scalar is inaccessible through the uniform `result.e[...]` surface the rest of the package provides. That is a design consequence, not an oversight — but it is the reason a user reading `e(r2_1)` or `e(ll)` after `var` gets an `AttributeError` rather than `None`.

### 1.13 `vec` / `vecrank`

Fixtures/display: `e_upgrade_outputs.txt:70-78`; `vecrank` printed-table evidence in `retest/domains/tseries/run.py:279-294` ("`[TS] vecrank` stores only `r(trendnumber)`/`r(ttext)`; the trace and max-eigenvalue statistics exist nowhere else").
westata: `VECM.fit()` returns statsmodels' `VECMRESults` (`tseries.py:935-968`); `JohansenResult` is a dataclass (`tseries.py:971-1060`) with `trace`, `maxeig`, `trace_critical`, `maxeig_critical`, `eigenvectors`, `to_frame()`, `to_stata_table()`, `select_rank()`.

Neither has an `.e`. Stata's `vec` stores `e(beta)`, `e(alpha)`, `e(Sigma)`, `e(lr)`, `e(ll)`, `e(N)`, `e(rank)`, `e(tmax/min)`, `e(k_eq)`; `vecrank` stores almost nothing (`r(trendnumber)`, `r(ttext)`). westata's `JohansenResult.to_stata_table()` already reproduces `vecrank`'s **printed** table (`tseries.py:1022-1032`), which is the more important surface for this command. The missing items are `e()`-shaped: `e(lr)`, `e(ll)`, `e(Sigma)`, `e(rank)`, `e(N)`.

### 1.14 `arch`

Display reference: `e_upgrade_outputs.txt:80-88`.
westata `e` built at `src/westata/estimators/tseries.py:1281-1299`.

Live keys: `['N','aic','alpha','archi','beta','bic','chi2type','converged','df_m','df_r','gamma','ll','omega','rank','vcetype']`.

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `ll`, `rank`, `converged`, `vcetype` | yes | PRESENT |
| `archi` (presample variance) | yes | PRESENT (`tseries.py:1292`) |
| `omega`, `alpha`, `beta`, `gamma` | yes | PRESENT (as tuples) |
| `aic`, `bic` | yes | **Same caveat as `arima`**: Stata does not store these; only `estat ic` does (`NOTES.md:248`) |
| `chi2`, `p`, `df_m`, `chi2type` | `chi2type` only | **MISSING** — `arch` stores no `e(chi2)`/`e(p)`; `chi2type="Wald"` is stored with no statistic to accompany it, so the summary renderer (§3) prints `Wald chi2(.) = .` |

### 1.15 `dfuller` / `pperron` / `xtunitroot`

These are **r-class tests in Stata, not estimations** — no `e()` at all.
westata returns `UnitRootResult` (`tseries.py:1374-1396`) and `PanelUnitRootResult` (`panel_iv.py:1823-1856`), explicitly documented as r-class containers (`panel_iv.py:1827-1830`: "Stata's `xtunitroot` stores these in `r()`, not `e()`").

Live `UnitRootResult` attrs: `['critical','lags','null','observations','pvalue','statistic','test']` — Stata's `r(Zt)`, `r(p)`, `r(lags)`, `r(N)`, `r(cv_1)`/`r(cv_5)`/`r(cv_10)` are all reachable.
Live `PanelUnitRootResult` attrs: `['details','name','pvalue','statistic','test']` with `details` carrying `t-bar`, `t-tilde-bar`, `Z-t-bar`, `N_t`, `lags` for IPS (`panel_iv.py:2105-2113`).

**Gaps here are display, not data**: neither result class has a `__str__` that reproduces Stata's multi-row table. `xtunitroot ips` prints three statistic rows (`t-bar`, `t-tilde-bar`, `Z-t-tilde-bar`) with **fixed-N exact critical values** in three columns (`p9_outputs.txt:531-538`); westata's `PanelUnitRootResult.__str__` prints one line (`panel_iv.py:1854-1856`). The critical-value tables are **not** exposed at all for IPS/LLC — `_critical_values` raises `UnsupportedError` (`xtunitroot_ips.py:145-146`: "the fixed-N exact critical values (IPS Table 2) are not yet implemented"). The Z statistics, `t-bar` and `t-tilde-bar` are implemented and match Stata.

---

## 2. Answers to Q1–Q5

### Q1. Does Stata's `regress` output display `e(tss)`?

**No — and `regress` does not even store it.** The evidence is the fixture asymmetry the question points at, and it is real:

* `tests/conformance/stata_fixtures/regress_auto.json:24-38` stores `mss` and `rss` (lines 32–33) but **no `tss`**. Same for `regress_auto_robust.json:24-38` and `regress_auto_factor.json:34-48`.
* `tests/conformance/stata_fixtures/areg.json:35` **does** store `tss` (= 515.5338737486169), alongside `mss` (line 39) and `rss` (line 38) — and `rss + mss = tss` holds (24.137434709274864 + 491.396439039342 = 515.5338737486169).
* `tests/conformance/stata_fixtures/xtreg_fe.json:51` stores `tss` too (and `mss` at line 55, `rss` at line 54).
* `reghdfe_*.json` all store `tss` **and** `tss_within` (e.g. `reghdfe_firm.json:29-30`).

**Implication.** The absence of `tss` from `regress_auto*.json` is not a capture omission: it reflects that `regress`'s `e()` set is `N sum_w df_m df_r F r2 r2_a rmse mss rss ll ll_0 rank` (+ `N_clust` clustered), and Stata's `regress` **prints no RSS/TSS line either** — its header block is `Number of obs / F( ) / Prob > F / R-squared / Adj R-squared / Root MSE`. So westata's `regress` is **complete**, and adding `e(tss)` there would be storing a number Stata does not have.

The absorption-based estimators are different: `areg`, `reghdfe`, `xtreg, fe` all store `tss` (and `areg`/`reghdfe` additionally `tss_within`), because their R² decompositions are stated against a within/absorbed total. westata **computes** the centred total at `src/westata/estimators/hdfe.py:522` and the uncentred/centred total at `src/westata/estimators/linear.py:249,255` — the latter is used only to derive `mss` (`linear.py:256`) and then discarded. So the `tss` gap for `areg`/`reghdfe`/`xtreg, fe` is a **one-line store**, not a computation.

Note also that westata's `ivregress` **does** store `tss` (`iv.py:826`) even though Stata's `ivregress 2sls` stores it only for 2sls per `iv.py:821-823` — so westata is slightly *more* generous than Stata there, which is harmless but worth knowing when diffing key sets.

### Q2. Is the "print covariance matrix of regression coefficients" feature (`estat vce`, `vce, corr`/`vce, covariance`) implemented anywhere?

**No. There is no display/render path anywhere in the package.** Evidence, in order of increasing specificity:

1. **No `estat` entry point exists.** A full-tree search for `estat` across `src/` returns only doc-comment mentions of `estat_cmd`/`estat firststage`/`estat ic` in prose (`src/westata/estimators/iv.py:131-133`, `src/westata/estimators/limited.py:13`-area docstrings) and the `GAP_REGISTER.md:81` line that states it outright: *"`estat vce`、`summarize`、`firststage`、`classification` 未实现"* ("`estat vce`, `summarize`, `firststage`, `classification` not implemented"). No module, class or function named `estat*` exists.
2. **`cov_params()` is sufficient as data but is never rendered.** `src/westata/results/base.py:203-206` returns the full `k×k` matrix; live check confirms `cov_params().shape == (3, 3)` for a 3-parameter OLS. The complete public surface of `StataResults` is `['conf_int','contrast','cov_params','hausman','lincom','lrtest','margins','nlcom','predict','pwcompare','summary','test','testnl','wald_test']` — no `vce`, no `corr`, no `cov_matrix`, no `estat`.
3. **The only renderer is `summary()`, and it never touches `cov`.** `src/westata/results/summary.py:132-245` (`stata` style) and `:248-323` (`statsmodels` style) both read `params`, `bse`, `tvalues`/`zvalues`, `pvalues`, `conf_int()`, `df_resid`, `df_model`, and a handful of `e()` scalars. `result.cov` appears nowhere in either function.
4. **`etable` does not render matrices either.** `src/westata/report/etable.py:52-164` builds a coefficient table with `se=True` underneath; the bottom `stats` block reads `res.e.get(key)` (`etable.py:147`) and can only emit scalars. There is no covariance/correlation block, and no `vce`/`corr` option in the signature (`etable.py:52-64`).
5. **`corr` in westata means something else entirely.** Searching for correlation-matrix display finds only Stata *option* names being stored as strings: `"corr": "correlated"` (`panel_iv.py:1254`, `xtgls`), `"corr": "independent"` (`panel_iv.py:1564`, `xtpcse`), `"corr": "exchangeable"` (`panel_nonlinear.py:1199`, `xtgee`), `"corr": _corr(fitted, y)` (`panel_iv.py:894`, `xtivreg`'s *y–ŷ* correlation, a scalar), and cross-equation residual correlation lists for `sureg`/`mvreg` (`systems.py:548`, `:750`). **None of these is a coefficient covariance or correlation matrix.**

**Conclusion for Q2.** `cov_params()` is *enough to implement* `estat vce`/`vce, corr` with zero new mathematics — the matrix, the parameter names and the `dist` are all already on the result object. What is missing is a renderer (a `Stata-style` matrix table) and an entry point. This is a **display-only** gap and belongs in priority class (iii), but with an unusually low cost because no estimation code changes.

### Q3. For `xtreg, fe`, which of `tss`, `tss_within`, `r2_w`, `r2_b`, `r2_o`, `sigma_e`, `sigma_u`, `theta`, `corr`, `df_a`, `k_absorb`, `N_g`, `Tbar`, `g_min/g_avg/g_max`, `F_f`, `ui` does westata set?

**None of the sixteen.** Live `sorted(result.e.keys())` for `PanelOLS.from_formula("y ~ x1 + x2", panel, entity="firm", time="year").fit()` is exactly:

```
['F', 'N', 'df_b', 'df_m', 'df_r', 'll', 'n_entities', 'r2', 'r2_a', 'rank', 'rmse', 'rss']
```

built at `src/westata/estimators/panel.py:741-757`. None of the sixteen keys appears.

Three of them exist under different names with the *same value* (verified to 1e-16 in `retest/domains/panel/NOTES.md:314`):
* `r2_w` ≡ `e["r2"]` (within R²)
* `sigma_e` ≡ `e["rmse"]`
* `N_g` ≡ `e["n_entities"]` (`panel.py:752`)

The remaining thirteen (`tss`, `tss_within`, `r2_b`, `r2_o`, `sigma_u`, `theta`, `corr`, `df_a`, `k_absorb`, `Tbar`, `g_min`, `g_avg`, `g_max`, `F_f`, `p_f`, `ui`) have **no westata counterpart at all**, against `xtreg_fe.json:25-62` which stores every one of them. The same twelve-key set is returned by `RandomEffects`, `BetweenOLS` and `FirstDifferenceOLS` (`panel.py:793,874,915`), so `xtreg, re` also loses `chi2`, `theta`, `sigma_u`, `sigma_e`, `sigma`, `rho`, `r2_w`, `r2_b`, `r2_o`, `N_g`, `Tbar`, `g_*` that `xtreg_re.json:24-45` stores.

Note that `theta` **is** computed for `re` — `panel_iv.py:902` stores it for `xtivreg, re` from the same Swamy–Arora machinery — and `sigma_u`/`rho`/`r2_b`/`r2_o`/`corr`/`g_min`/`g_avg`/`g_max` are all computed for `xtivreg` at `panel_iv.py:844-898`. The `xtreg` classes simply never call those code paths.

### Q4. For `arima`, what `e()` scalars does Stata store, and which does westata set?

Stata's `arima` `e()` set (from the in-repo evidence): `e(N)`, `e(ll)`, `e(sigma)`, `e(rank)`, `e(converged)`, the model-test family (`e(chi2)`/`e(F)`/`e(p)`/`e(df_m)`), `e(cmd)`, `e(depvar)`, `e(predict)`. Authoritative citations: `docs/GAP_REGISTER.md:655-668` (§8.14, which records that Stata reports `e(sigma)` and **not** `e(sigma2)`) and `retest/domains/tseries/NOTES.md:248` ("Stata's `arch` does not store `e(aic)/e(bic)` (`arima` either) — reading `e(aic)` directly gives missing; you must use `estat ic` to take full precision from `r(S)`").

**Critically, Stata's `arima` does NOT store `e(aic)`/`e(bic)`** — `retest/domains/tseries/run.py:262-273` implements `stata_ic()` precisely because the values only exist in `r(S)` after `estat ic`, with column order `[N, ll0, ll, df, AIC, BIC]`.

westata's `arima` `e()` (built at `src/westata/estimators/tseries.py:188-216`, confirmed live):
```
['N', 'aic', 'bic', 'chi2', 'chi2type', 'converged', 'df_m', 'df_r', 'll', 'order', 'p', 'rank', 'seasonal_order', 'sigma', 'sigma2', 'trend']
```

| Stata `e()` key | westata | Note |
|---|---|---|
| `N` | yes | post-differencing convention, fixed and verified (`tseries.py:168-169`, `retest/domains/tseries/summary.json:66`) |
| `ll` | yes | PRESENT |
| `sigma` | yes | PRESENT — the §8.14 fix (`tseries.py:202-206`) |
| `rank`, `converged` | yes | PRESENT |
| model test | yes (`chi2`, `p`, `df_m`, `chi2type`) | westata convention: Wald chi2 over all ARMA coefficients (`tseries.py:173-186`) |
| `aic`/`bic` | yes | **westata-computed, not Stata-stored.** Stata only exposes them via `estat ic` |
| `sigma2` | yes | westata-only by design (`tseries.py:196-201`) |
| `order`, `seasonal_order`, `trend`, `df_r` | yes | westata extras |

So for `arima` the `e()` surface is **complete relative to what Stata stores**, and strictly larger (which is documented and harmless). The genuine gap is §3's: `summary(style="stata")` prints `R-squared = .` for `arima` (no R² exists) but never prints the AIC/BIC line Stata's `estat ic` produces.

### Q5. Does westata expose `estat vce` / `estat ic` / `estat summarize` / `estat firststage` equivalents?

| Stata command | westata equivalent | Evidence |
|---|---|---|
| `estat vce` / `vce, corr` / `vce, covariance` | **None** | `GAP_REGISTER.md:81`; no `estat` symbol in `src/`; `cov_params()` exists but is never rendered (Q2) |
| `estat ic` | **Partial, data-only** | `e(aic)`/`e(bic)` are readable on `StataResults` (`results/base.py:315-324`) for estimators that store them — but there is **no `estat ic`-style table renderer**, and `var`/`vec`/`vecrank` results have no `.e` at all so `e(aic)` raises `AttributeError` there |
| `estat summarize` | **None** | Not implemented anywhere; `GAP_REGISTER.md:81` |
| `estat firststage` | **Partial, scalar-only** | `e(F_first)` is stored for `ivregress` (`iv.py:807,817`) and a `UserWarning` fires below F=10 (`iv.py:869-876`) — but Stata's full first-stage table (per-endogenous-variable F, R², partial R², Shea's) is not produced. `iv.py:131-133` calls `e(F_first)` an "`estat firststage` analogue" |
| `estat gof` / `estat hetest` | **None** | `GAP_REGISTER.md:84` |
| `estat classification` | **None** | `GAP_REGISTER.md:81` |

The full postestimation surface actually exported is `test`, `lincom`, `lrtest`, `hausman`, `contrast`, `pwcompare`, `nlcom`, `testnl`, `margins`, `predict`, `wald_test` — see `src/westata/results/base.py:329-442` and `src/westata/__init__.py:94-107`. Nothing named `estat`, and nothing that renders a matrix.

---

## 3. Display surface gaps — what Stata prints but westata renders nowhere

The only two renderers are `src/westata/results/summary.py` (`stata` style at `:132-245`, `statsmodels` style at `:248-323`) and `src/westata/report/etable.py`. Everything below is absent from both.

### 3.1 Things the `stata` renderer *does* print (so they are not gaps)

`Number of obs` (`summary.py:180`), `F(df_m, df_r)` + `Prob > F` (`:210-214`), `Wald chi2(df)` + `Prob > chi2` (`:191-195`, `:205-208`), `R-squared`/`Adj R-squared`/`Root MSE` (`:222-224`), `Log likelihood` + `Pseudo R2` for likelihood models (`:216-220`), and the coefficient table with `Coef./Std. err./t or z/P>|t| or P>|z|/[95% conf. interval]` (`:227-244`).

### 3.2 Gaps, by estimator family

| Stata-displayed item | Command(s) | Where it should come from | Status |
|---|---|---|---|
| **Covariance matrix of coefficients** (`estat vce`) | all | `result.cov_params()` — data already present | **MISSING — no renderer** (Q2) |
| **Correlation matrix of coefficients** (`vce, corr`) | all | derived from `cov_params()` | **MISSING — no renderer** |
| **`e(tss)`/`e(rss)`/`e(mss)` line** | `areg`, `reghdfe`, `xtreg fe/be` | computed at `hdfe.py:522`, `panel.py:737`; not stored, not printed | **MISSING** (data + display) |
| **Variance-components block** `sigma_u`, `sigma_e`, `rho` (`xtreg fe/re`) | `xtreg` | not stored for `xtreg`; stored for `xtivreg` (`panel_iv.py:891-894`) | **MISSING** |
| **`rho`/`corr(u_i, Xb)` line** | `xtreg fe` | not stored | **MISSING** |
| **`theta` line** | `xtreg re` | not stored for `xtreg`; stored for `xtivreg re` (`panel_iv.py:902`) | **MISSING** |
| **`F_f`/`p_f` (test that absorbed effects are all zero)** | `xtreg fe`, `areg` | `xtreg_fe.json:26-27`; not stored | **MISSING** |
| **`R-square within/between/overall`** | `xtreg fe/be/re` | `r2_w`/`r2_b`/`r2_o` in all three fixtures | **MISSING** |
| **Group header block**: `Number of groups`, `Obs per group min/avg/max`, `Tbar`, `g_min/g_avg/g_max` | `xtreg`, `xtivreg`, `xtabond`, `xtgls`, `xtpcse` | `panel_iv.py:896-898,1080-1082,1560-1562` store it for three classes; `xtreg` stores nothing; no renderer for any | **MISSING** |
| **`Number of instruments` + instrument blocks** | `xtabond` | `e["instruments"]` stored (`panel_dynamic.py:440`); the "Instruments for differenced/level equation" text (`p9_outputs.txt:370-374`) is not reproduced | **MISSING (display)** |
| **Sargan/Hansen/AR(1)/AR(2) test block** | `xtabond` | deliberately omitted (`panel_dynamic.py:455-457`) | **Documented omission, not a display bug** |
| **`Wald chi2`/`Prob > chi2` for `dist="z"` panel estimators** | `xtgls`, `xtpcse`, `xtivreg` | `e["chi2"]`/`e["p"]` are stored (`panel_iv.py:1069-1073`, `1552-1556`) but the renderer only reaches the chi2 branch when `e["chi2type"]` is set **and** the branch at `summary.py:185-195` runs — it does, so this one **works**; verify by rendering | PRESENT (data + display) |
| **`Estimated covariances / autocorrelations / coefficients` header block** | `xtgls`, `xtpcse` | `n_cf`/`n_cv`/`n_cr` stored (`panel_iv.py:1063-1065`, `1558-1559`) | **MISSING (display)** |
| **`Panels:`/`Correlation:`/`Autocorrelation:` header lines** | `xtgls`, `xtpcse` | `e["panels"]`, `e["corr"]`, `e["selection"]` stored | **MISSING (display)** |
| **ANOVA decomposition table** (`Source | SS df MS`, `Model`/`Residual`/`Total` rows) | `regress` with a single regressor | `rss`/`mss` stored (`linear.py:322-323`); `tss` derivable (`rss+mss`); the **renderer has no ANOVA code at all** — searching `summary.py` for `SS`/`Source`/`Model`/`Residual`/`Total` finds only `df_model`/`df_resid` labels (`:283`) | **MISSING — entirely absent renderer** |
| **`e(F)`/`e(p)` line for `xtreg, re`** | `xtreg re` | Stata reports Wald `chi2`, westata stores `F` (`panel.py:749`) | **MISSING `chi2`/`p` (data)** |
| **`estat ic` table** (N, ll, df, AIC, BIC/HQIC) | `arima`, `arch`, and any ML estimator | `e(aic)`/`e(bic)` present on `StataResults`; **no renderer**; and `var`/`vec`/`vecrank` have no `.e` | **MISSING (display)** |
| **Treatment-and-time information table** + `Data type:` + the ATET footnote | `didregress`, `xtdidregress` | `e["data_type"]` stored (`did.py:786`); the table itself (`p11_did.txt:5-19`) is not reproduced | **MISSING (display)** |
| **"Absorbed variable: firm" / "Absorbing 1 HDFE group" header** | `areg`, `reghdfe` | `results.absorbed` is set (`hdfe.py:567`) but the renderer never prints it | **MISSING (display)** |
| **"Statistics robust to heteroskedasticity" / cluster footnote** | `reghdfe` clustered | `e["N_clust"]` stored (`hdfe.py:550`) | **MISSING (display)** |
| **Fixed-N exact critical-value columns** | `xtunitroot ips/llc` | `_critical_values` raises `UnsupportedError` (`xtunitroot_ips.py:145-146`) | **MISSING (capability)** |
| **`vecrank` printed table** | `vecrank` | `JohansenResult.to_stata_table()` exists (`tseries.py:1022-1032`) | **PRESENT** — the one matrix-shaped display in the package |
| **`Root MSE` for `xtgls`** | `xtgls` | stored as NaN by design (`panel_iv.py:1076`); Stata also omits it (`p9_outputs.txt:472-482`) | Correctly omitted |

### 3.3 A specific renderer defect worth flagging

`src/westata/results/summary.py:216-224` prints `R-squared` / `Adj R-squared` / `Root MSE` for every non-likelihood model. For `arima`, `arch` and `xtgls` those `e()` keys are absent or NaN, so the live render shows:

```
R-squared             =          .
Adj R-squared         =          .
Root MSE              =     1.0120
```

Stata prints **no** R-squared block for these commands at all. Printing a `.` line is not wrong, but it is noise the Stata layout does not have, and a scraper keying on `R-squared` will read `.` as a missing value rather than "not applicable". This is a display-fidelity issue distinct from the missing-statistics inventory.

---

## 4. Priority ranking

### (i) Pure data-availability — the number is already computed internally and just not stored

Cheapest first. All of these are a store, not a computation.

| # | Item | Command(s) | Already computed at | Effort |
|---|---|---|---|---|
| 1 | `tss` | `areg`, `reghdfe`, `xtreg fe`, `xtgls`, `xtpcse` | `hdfe.py:522` (discarded), `linear.py:249/255` (discarded), `panel_iv.py` `tss` local | one line per class |
| 2 | `tss_within`, `r2_within`, `r2_a_within` | `areg`, `reghdfe` | `hdfe.py:522` (within total is the same expression when the FE is one-way) | one line |
| 3 | `mss` | `areg`, `reghdfe`, `xtreg fe/be`, `xtgls` | `tss - rss`, both available | one line per class |
| 4 | `r2_b`, `r2_o` | `xtreg fe/be/re` | already implemented for `xtivreg` at `panel_iv.py:832-846` | port |
| 5 | `sigma_u`, `rho` | `xtreg fe/re` | already implemented for `xtivreg` at `panel_iv.py:844-846` | port |
| 6 | `theta` | `xtreg re` | already implemented for `xtivreg re` at `panel_iv.py:902` | port |
| 7 | `N_g`, `Tbar`, `g_min`, `g_avg`, `g_max` | `xtreg` (all four) | `n_entities` already stored (`panel.py:752`); group sizes are on `self._data.index` | one line |
| 8 | `p` (model p-value) | `xtreg`, `xtivreg` | `e["F"]`/`e["chi2"]` present | one line |
| 9 | `g_min/g_avg/g_max` | `xtabond` | `rows_per_unit` already computed (`panel_dynamic.py:354`) | one line |
| 10 | `ll_0` | `areg`, `reghdfe` | `llf` computed at `hdfe.py:532`; `ll_0` formula identical to `linear.py:280-284` | port |
| 11 | `p_f`, `F_f` | `xtreg fe`, `areg` | **requires a new F test of the absorbed effects** — not currently computed for `xtreg`. `xtivreg` has no `F_f` either. | small new computation |
| 12 | `ui`, `corr`, `k_absorb`, `df_a` | `xtreg fe` | `df_a` ≈ `n_entities - 1` (already available); `ui`/`corr` need the variance decomposition | small new computation |

Items 1–10 are the recommended first batch: they close the user-visible `tss`/`rss` complaint for `areg`/`reghdfe`/`xtreg` and the whole `xtreg` variance-component block, with no new numerics and therefore no conformance risk.

### (ii) Require a new computation

| # | Item | Command(s) | What is new |
|---|---|---|---|
| 13 | `F_f` / `p_f` (joint test of absorbed effects) | `xtreg fe`, `areg` | A Wald F on the absorbed-effect space; not computed anywhere today |
| 14 | `ui` (`sigma_u` on the within scale) and `corr(u_i, Xb)` | `xtreg fe` | Needs the entity-effect estimates themselves, which linearmodels does not return by default |
| 15 | `thta_min/5/50/95/max` | `xtreg re`, unbalanced | Per-panel theta distribution |
| 16 | Sargan / Hansen / AR(1) / AR(2) tests | `xtabond` | Deliberately omitted (`panel_dynamic.py:455-457`); reinstating requires a defensible implementation, not an approximation |
| 17 | `e(Sigma)`, `e(lr)`, `e(ll)`, `e(rank)`, `e(N)`, `e(tmax/min)`, `e(k_eq)`, `e(neqs)`, `e(mlag)`, `e(detsig)` | `var`, `vec` | Requires wrapping `VARResults`/statsmodels `VECMRESults` into (or alongside) a `StataResults`-shaped `e()` |
| 18 | `r2_1/2`, `rmse_1/2`, `chi2_1/2`, `p_1/2`, `ll_1/2`, `df_eq` | `var` | Per-equation scalars; `granger_causality()` covers only `chi2_*` |
| 19 | IPS/LLC fixed-N exact critical values | `xtunitroot` | Requires transcribing IPS Table 2 / LLC Table 2 (`xtunitroot_ips.py:145-146`) |
| 20 | `N_drop`, `N_group_drop` | `xtreg` (unbalanced) | Dropped-observation accounting (verified present for other estimators, `retest/domains/panel/NOTES.md:200-204`) |

### (iii) Require a new display/render feature

| # | Item | Where | Note |
|---|---|---|---|
| 21 | `estat vce` — covariance matrix table | new renderer + entry point | Data is already there (`cov_params()`); this is the user's "print the covariance matrix" request |
| 22 | `vce, corr` — correlation matrix table | same renderer | `D = diag(sqrt(diag(V)))`, `R = D⁻¹ V D⁻¹` |
| 23 | **ANOVA decomposition table** for single-regressor `regress` | `summary.py` | Stata prints `Source / SS / df / MS` with `Model`, `Residual`, `Total` rows. westata has `rss`, `mss`, `df_m`, `df_r` and can derive `tss = rss + mss` — **every number is already on the result object**; only the table code is absent |
| 24 | `estat ic` table (N, ll, df, AIC, BIC, HQIC) | new renderer | `e(aic)`/`e(bic)` exist on `StataResults`; `var`/`vec` need #17 first |
| 25 | `xtreg` variance-components + group header block | `summary.py` | Depends on (i) items 5–8 |
| 26 | `xtabond` instrument block + "Obs per group" header | `summary.py` | `e["instruments"]` stored; instrument *names* are not retained |
| 27 | `xtgls`/`xtpcse` header block (`Panels:`, `Correlation:`, `Estimated covariances/autocorrelations/coefficients`) | `summary.py` | All scalars stored |
| 28 | `didregress` treatment-and-time table + `Data type:` + footnote | `summary.py` | `e["data_type"]` stored; the table needs group/period counts |
| 29 | `areg`/`reghdfe` "Absorbed variable" / "Absorbing N HDFE groups" header | `summary.py` | `results.absorbed` already set (`hdfe.py:567`) |
| 30 | `xtunitroot` multi-row table (t-bar / t-tilde-bar / Z, with critical columns) | `PanelUnitRootResult.__str__` | Data present for the statistics; critical values blocked by #19 |
| 31 | Suppress `R-squared`/`Root MSE` lines where Stata prints none (`arima`, `arch`, `xtgls`) | `summary.py:216-224` | Fidelity fix; currently prints `.` |
| 32 | Cluster/robust footnote ("Statistics robust to heteroskedasticity", "Std. err. adjusted for N clusters") | `summary.py` | `e["N_clust"]` stored |

---

## 5. Items that are **not** gaps (explicitly not applicable to westata's design)

Stated so they are not re-litigated as gaps:

* **`e(tss)` for `regress`** — Stata's `regress` does not store it (Q1). westata matches.
* **`e(aic)`/`e(bic)` for `arima`/`arch`** — Stata does not store these either; they exist only through `estat ic`'s `r(S)` (`retest/domains/tseries/run.py:262-273`). westata storing them is a superset, not a gap.
* **`e(sigma2)` for `arima`** — Stata stores `e(sigma)` only; westata stores both deliberately, with the 3.7%-error rationale documented at `tseries.py:196-201` and `GAP_REGISTER.md:655-668`.
* **`e(rc)`** — Stata's return-code scalar; not a statistical quantity.
* **reghdfe-internal provenance** (`ic`, `converged`, `report_constant`, `N_full`, `num_singletons`, `drop_singletons`, `N_hdfe`, `N_hdfe_extended`, `df_a_nested`, `df_a_redundant`, `df_a_initial`) — these describe reghdfe's own solver and singleton policy. westata is PyFixest-backed (`hdfe.py:62-65`, `:571-592`), has no singleton policy, and reports `e(rank)` via PyFixest's `_df_k`. Storing a `converged=1` that describes a different solver would be inventing a number.
* **`e(J)` for `ivregress gmm`** — present in westata (`iv.py:842-854`); the historical gap recorded in `retest/domains/linear_iv/results.jsonl:56` is closed.
* **`xtabond` Sargan/Hansen/AR tests** — deliberately absent with a written rationale (`panel_dynamic.py:455-457`), consistent with the package's rule of never reporting an approximate number under a Stata name.
* **`didregress` `e(r2)`/`e(r2_a)`/`e(rmse)`/`e(F)`/`e(ll)`** — westata additions, not Stata values; documented at `did.py:81-89`.

---

## 6. Reproducing this audit

Every claim above comes from either reading the cited line or from a live fit. The live fits used `./.venv/Scripts/python.exe` on `tests/data/auto.dta` (with `df["dom"] = (df["foreign"]=="Foreign").astype(int)` for the string column) and the `tests/test_panel.py:42-55` `make_panel()` pattern for panel data, dumping `sorted(result.e.keys())` per estimator. The probe script was removed after the audit; the key sets quoted in §1.1–§1.15 are verbatim from those runs.
