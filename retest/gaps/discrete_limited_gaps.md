# Discrete / Limited / Panel-nonlinear: missing auxiliary statistics & display features

**Audit target** — `westata` at `D:\myCodeProject\statapy`
**Scope** — three families only:

1. Binary & count: `logit`, `probit`, `poisson`, `nbreg` (NB2), `nbreg, nbre1` (NB1), `zip`, `zinb`, `ztp`, `tnbreg`, `tpoisson`, `ologit`, `oprobit`, `svy: logit`
2. Limited DV: `tobit`, `truncreg`, `intreg`, `heckman`, `etregress`
3. Panel nonlinear: `xtlogit fe/pa/re`, `xtprobit re`, `xttobit`, `xtpoisson fe/re`, `xtnbreg fe`

**Method** — read the source, read the Stata fixtures, then **run every fit** with `./.venv/Scripts/python.exe` and print `sorted(result.e.keys())`. Every row below is backed either by a `path:line` citation or by the runtime key list reproduced in §1–§3.

**Output vocabulary** — `Stata stores it` = it exists in Stata's `e()`/`r()`; `Stata prints it` = it appears in the command's on-screen output. These are separate columns because a large part of this audit's finding set is "Stata stores it, and also prints it; westata does neither".

---

## 0. How the `e()` dict is built (the chain that matters)

Every estimator in scope funnels into one of four assembly sites. Knowing which one a model uses tells you exactly which keys it can possibly carry.

| Site | File:line | Used by | Keys set |
|---|---|---|---|
| `_DiscreteBase._stata_scalars` | `src/westata/estimators/discrete.py:586-623` | `logit`, `probit`, `poisson`, `nbreg`(NB2), `zip`, `zinb`, `ztp`, `tnbreg`, `ologit`, `oprobit` (ordered overrides below) | `N`, `df_m`, `rank`, `ll`, `ll_0`, `chi2`, `chi2type`, `p`, `r2_p`, `aic`, `bic` (+ `k_aux`, `N_clust`) |
| `_CountBase._stata_scalars` + `_extra_scalars` | `src/westata/estimators/counts.py:521-582`, `:584-587` | the count family | the above + `lnalpha`, `alpha`, `k_aux`, and per-subclass extras |
| `_MLLimitedBase._stata_scalars` + `_extra_scalars` | `src/westata/estimators/limited.py:495-537`, `:492-493` | `truncreg`, `intreg` | `N`, `df_m`, `rank`, `ll`, `ll_0`, `chi2`, `chi2type`, `p`, `r2_p`, `sigma`, `var_e`, `converged`, `aic`, `bic` + extras |
| bespoke `e: dict` literals | `discrete.py:944-959` (NB1), `discrete.py:1562-1583` (ordered), `tobit.py:383-402`, `limited.py:1675-1691` (heckman ML), `limited.py:1827-1849` (heckman 2-step), `limited.py:2107-2123` (etreg ML), `limited.py:2235-2251` (etreg control), `panel_nonlinear.py:719-750` + per-model `extra` | NB1, ordered, tobit, heckman, etregress, all panel nonlinear | see tables |

**`_extra_scalars()` chains followed** (the audit brief asked for this explicitly):

| Class | `_extra_scalars` definition | Extra keys it injects |
|---|---|---|
| `_CountBase` (counts.py) | `src/westata/estimators/counts.py:584-587` | `{}` (base) |
| `_ZeroInflatedBase` | `src/westata/estimators/counts.py:816-824` | `n_zero`, `n_nonzero`, `inflate="none"` |
| `_TruncatedBase` | `src/westata/estimators/counts.py:991-996` | `truncation_point`, `lower`, `upper` |
| `Truncreg` | `src/westata/estimators/limited.py:752-758` | `lower`, `upper`, `n_truncated`, `n_uncensored` |
| `Intreg` | `src/westata/estimators/limited.py:947-955` | `lnsigma`, `n_left`, `n_right`, `n_interval`, `n_uncensored` |
| `_MLLimitedBase` | `src/westata/estimators/limited.py:492-493` | `{}` (base) |
| `_Ordered` (ologit/oprobit) | — (no `_extra_scalars`; inline at `discrete.py:1562-1583`) | `k_aux` inline |

**Automatic keys added by the results container**, not by the estimator: `StataResults.__init__` merges `N`, `df_m`, `df_r` as computed defaults, and an explicit `e()` value wins (`src/westata/results/base.py:66-67`, `:139-140`). That is why every fitted result shows `df_r` even though no estimator in this scope writes it. Confirmed at runtime: `logit` → `e keys: ['N', 'aic', 'bic', 'chi2', 'chi2type', 'df_m', 'df_r', 'll', 'll_0', 'p', 'r2_p', 'rank']`.

---

## 1. Family 1 — Binary & count models

### 1.1 `logit` / `probit`

Reference: `tests/conformance/stata_fixtures/logit_sim.json`, `probit_sim.json`.

| Stata key | In Stata fixture? | In westata? | Verdict |
|---|---|---|---|
| `b`, `V` | yes (`b`, `V`) | yes (`params`, `cov_params()`) | match |
| `N` | yes | yes | match |
| `ll` | yes | yes | match |
| `ll_0` | yes | yes | match |
| `df_m` | yes | yes | match |
| `chi2` | yes | yes | match |
| `p` | yes | yes | match |
| `r2_p` | yes | yes | match |
| `rank` | yes | yes | match |
| `converged` | yes (1.0) | **no** | **missing** |
| `rc` | yes (0.0) | **no** | **missing** |
| `ic` | yes | **no** — westata stores `aic`/`bic` instead | **missing + naming gap** (see Q5) |
| `k` | yes (3.0) | **no** | **missing** |
| `k_dv` | yes (1.0) | **no** | **missing** |
| `k_eq` | yes (1.0) | **no** | **missing** |
| `k_eq_model` | yes (1.0) | **no** | **missing** |
| `k_autoCns` | yes (0.0) | **no** | **missing** |
| `consonly` | yes (0.0) | **no** | **missing** |
| `noconstant` | yes (0.0) | **no** | **missing** |
| `N_cdf`, `N_cds` | yes (0.0/0.0) | **no** | **missing** |
| `aic` / `bic` | **no** (Stata stores only `ic`) | **yes** | westata addition |
| `df_r` | **no** | **yes** | westata addition |
| macros `chi2type`, `crittype`, `ml_method`, `technique`, `opt`, `estat_cmd`, `marginsok`, `predict`, `properties`, `diparm1`, `dispers`, `which`, `user`, `singularHmethod`, `deriv_useminbound` | yes (full macro block, logit_sim.json:46-67) | **none stored**; only `chi2type` is stored as a scalar-shaped string | **missing (all macros)** |

Runtime confirmation (`./.venv/Scripts/python.exe`, `dom ~ weight + mpg` on `tests/data/auto.dta`):

```
logit   e keys: ['N','aic','bic','chi2','chi2type','df_m','df_r','ll','ll_0','p','r2_p','rank']
probit  e keys: ['N','aic','bic','chi2','chi2type','df_m','df_r','ll','ll_0','p','r2_p','rank']
```

### 1.2 `poisson`

Reference: `tests/conformance/stata_fixtures/poisson_sim.json`.

Same table as `logit`, with two additions:

| Stata key | In Stata fixture? | In westata? | Verdict |
|---|---|---|---|
| `ic` | yes | no (stores `aic`/`bic`) | missing + naming gap |
| `k`, `k_dv`, `k_eq`, `k_eq_model`, `k_autoCns`, `consonly`, `noconstant`, `converged`, `rc` | yes | **no** | missing |
| macro `gof = "poiss_g"` | **yes** (poisson_sim.json:41) | **no** | **missing — see Q2** |
| `N_cdf`/`N_cds` | **no** in this fixture | **no** | n/a |

Runtime: `poisson e keys: ['N','aic','bic','chi2','chi2type','df_m','df_r','ll','ll_0','p','r2_p','rank']`.

### 1.3 `nbreg` (NB2)

Reference: `tests/conformance/stata_fixtures/nbreg.json`.

| Stata key | In fixture? | In westata? | Verdict |
|---|---|---|---|
| `N`, `ll`, `ll_0`, `df_m`, `chi2`, `p`, `r2_p`, `rank`, `k_aux` | yes | yes | match |
| `alpha` | yes (0.42028…) | **no** — NB2 path stores only `/lnalpha` as a *parameter*, no `e(alpha)` | **missing** |
| `ll_c` | yes (-1025.016) | **no** | **missing** |
| `chi2_c` | yes (96.450) | **no** | **missing** |
| `rank0` | yes (2.0) | **no** | **missing** |
| `ic` | yes | no (`aic`/`bic`) | missing + naming gap |
| `k`, `k_dv`, `k_eq`(=2), `k_eq_model`, `k_autoCns`, `consonly`, `noconstant`, `converged`, `rc` | yes | no | missing |
| macros `dispers="mean"`, `diparm1="lnalpha, exp label(\"alpha\")"`, `diparm_opt2="noprob"`, `chi2_ct="LR"` | yes (nbreg.json:54-62) | no | **missing** |

Runtime (this dataset's NB2 is degenerate — `alpha → 0`), and it exposes a **separate, bigger display defect**:

```
nbreg  e keys: ['N','aic','bic','chi2','chi2type','df_m','df_r','k_aux','ll','ll_0','p','r2_p','rank']
nbreg  e values: ll_0 = nan, r2_p = nan, chi2 = nan, p = nan
```

Because `e(ll_0)` is `NaN` for `nbreg`, `summary(style="stata")` falls into the **OLS branch** and prints `R-squared / Adj R-squared / Root MSE` instead of `Log likelihood / Pseudo R2` (see §4, Q1). Stata's `nbreg` fixture *does* carry `ll_0 = -1040.145594343475` (nbreg.json:38), so the statistic exists on Stata's side.

### 1.4 `nbreg, nbre1` (NB1) — `westata.NegativeBinomialNB1`

No NB1 fixture exists in `tests/conformance/stata_fixtures/`. Stata's `nbreg, dispersion(constant)` stores `e(dispersion)` (a *macro* = `"constant"`), `e(delta)`, `e(sedelta)`, `e(lndelta)`, `e(ll_c)`, `e(chi2_c)`, `e(rank0)`, plus the whole `k*`/`converged`/`rc` block.

| Stata key | In westata? | Verdict |
|---|---|---|
| `lndelta`, `delta`, `sedelta` | **yes** — `src/westata/estimators/discrete.py:954-957` | match (exceeds Stata's surface) |
| `dispersion` | yes, as a **string** (`discrete.py:958`) — Stata's is a macro | match in substance |
| `ll_c`, `chi2_c`, `rank0` | **no** | missing |
| `ic` | no | missing |
| `k`, `k_eq`, `k_dv`, `k_eq_model`, `k_autoCns`, `consonly`, `noconstant`, `converged`, `rc` | **no** | missing |
| `alpha` | no (correct: NB1 has no `alpha`) | n/a |

Runtime: `NB1 e keys: ['N','chi2','chi2type','converged','delta','df_m','df_r','dispersion','ll','ll_0','lndelta','p','rank','sedelta']`. Note NB1 *does* carry `converged` (`discrete.py:953`) while NB2 does not.

### 1.5 `zip` / `zinb`

No dedicated JSON fixture; the Stata capture is `tests/conformance/stata_fixtures/p6p7_outputs.txt:118-131`.

| Stata key / printed line | In Stata capture? | In westata? | Verdict |
|---|---|---|---|
| `Number of obs` | yes | yes | match |
| `Nonzero obs = 660` / `Zero obs = 540` header pair | **yes, printed** (p6p7_outputs.txt:121-122) | stored as `n_zero`/`n_nonzero` (counts.py:819-820) but **never printed** | **display gap** |
| `Inflation model: logit` | **yes, printed** (p6p7_outputs.txt:121) | **no** | **missing (both)** |
| `LR chi2(1)` / `Prob > chi2` | yes | yes (`chi2`, `p`) | match |
| `Log likelihood` | yes | yes (`ll`) | match |
| `e(ic)` | Stata stores | no (`aic`/`bic` only) | missing + naming gap |
| `e(k)`, `e(k_eq)`, `e(k_eq_model)`, `e(k_aux)`, `e(converged)`, `e(rc)` | Stata stores | **no** | missing |
| `e(alpha)` for `zinb` | Stata stores | **yes** — `counts.py:580` | match |
| iteration log | printed by Stata | no | **display gap** |

Runtime:
```
zip   e keys: ['N','aic','bic','chi2','chi2type','df_m','df_r','ll','ll_0','n_nonzero','n_zero','p','r2_p','rank']
zinb  e keys: ['N','aic','alpha','bic','chi2','chi2type','df_m','df_r','k_aux','ll','ll_0','lnalpha','n_nonzero','n_zero','p','r2_p','rank']
```

### 1.6 `ztp` / `tpoisson` / `tnbreg`

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `ll`, `ll_0`, `df_m`, `chi2`, `p`, `r2_p`, `rank` | yes | match |
| `lower`, `upper`, `truncation_point` | yes — `counts.py:991-996` | westata addition (Stata stores `e(ll)`, `e(ul)` only as options, not as `e()` scalars) |
| `lnalpha`, `alpha`, `k_aux` (tnbreg) | yes — `counts.py:577-581` | match |
| `ic` | no | missing + naming gap |
| `k`, `k_eq`, `k_dv`, `k_eq_model`, `k_autoCns`, `consonly`, `noconstant`, `converged`, `rc` | **no** | missing |

Runtime:
```
ztp    e keys: [... 'lower','p','r2_p','rank','truncation_point','upper']
tnbreg e keys: [... 'alpha','k_aux','lnalpha','lower','p','r2_p','rank','truncation_point','upper']
```

### 1.7 `ologit` / `oprobit` — see Q3

Reference: `tests/conformance/stata_fixtures/ologit.json`, `oprobit.json`.

| Stata key | In fixture? | In westata? | Verdict |
|---|---|---|---|
| `b`, `V` | yes | yes | match |
| `N`, `ll`, `ll_0`, `df_m`, `chi2`, `p`, `r2_p`, `rank` | yes | yes | match |
| `k_aux` | yes (2.0) | **yes** — `discrete.py:1565` | match |
| `k_cat` | yes (3.0) | **no** | **missing** |
| `N_cd` | yes (0.0) | **no** | **missing** |
| `ic` | yes | no | missing + naming gap |
| `k`, `k_eq`(=3), `k_dv`, `k_eq_model`, `k_autoCns`, `consonly`, `noconstant`(=1), `converged`, `rc` | yes | **no** | missing |
| cut points as named params | yes — `names: ["x1","x2","cut1","cut2"]` | **yes, but with a westata-only spelling** — runtime `param names: ['weight','mpg','/cut1','/cut2','/cut3','/cut4']` | **naming gap** (see Q3) |
| macros `depvar_outcomes`, `marginsdefault`, `marginsderiv`, `marginsprop` | yes | **no** | missing |
| iteration log / convergence note | printed by Stata | **no**, and `e(converged)` is absent too | **missing (both)** |

Runtime: `ologit e keys: ['N','chi2','chi2type','df_m','df_r','k_aux','ll','ll_0','p','r2_p','rank']`; `converged in e? False`.

### 1.8 `svy: logit`

Assembly: `src/westata/estimators/survey.py:634-644` (base) merged with `src/westata/estimators/survey.py:1290-1295` (logit extras).

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `N_strata`, `N_psu`, `N_pop`, `df_m`, `df_r`, `rank`, `singleunit`, `vce` | yes — `survey.py:635-643` | match (a genuinely strong surface) |
| `ll` | yes — `survey.py:1291` | match |
| `F`, `p` | yes — `survey.py:1292-1293` | match (Stata `svy: logit` reports `F`, not `chi2`) |
| `e(ic)` | **no** | missing |
| `e(converged)`, `e(rc)` | **no** | missing |
| `e(r2_p)` | **no** | missing |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)` | **no** | missing |
| printed `R-squared / Adj R-squared / Root MSE` block | **rendered by westata, but as `.`** (see §4) | **display defect** |

Runtime: `svy: logit e keys: ['F','N','N_pop','N_psu','N_strata','df_m','df_r','iterations','ll','p','rank','singleunit','vce']`.

---

## 2. Family 2 — Limited dependent variables

### 2.1 `tobit`

Assembly: `src/westata/estimators/tobit.py:383-402`.

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `ll`, `ll_0`, `df_m`, `df_r`, `rank`, `chi2`, `p`, `r2_p`, `sigma`, `converged` | yes — `tobit.py:384-399` | match |
| `n_uncensored`, `n_left`, `n_right` | yes — `tobit.py:385-387` | match (these are exactly Stata's `e(n_uncensored)`/`e(n_left)`/`e(n_right)`) |
| `var_e` | yes — `tobit.py:398` | match |
| `lower`, `upper` | yes — `tobit.py:400-401` | match |
| `e(ic)` | **no** | missing |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |
| macros `chi2type`, `crittype`, `ml_method`, `technique`, `opt`, `estat_cmd`, `predict`, `properties`, `which`, `user`, `singularHmethod`, `deriv_useminbound` | **no** | missing |
| iteration log | printed by Stata | **no** | display gap |

Runtime: `tobit e keys: ['N','chi2','chi2type','converged','df_m','df_r','ll','ll_0','lower','n_left','n_right','n_uncensored','p','r2_p','rank','sigma','upper','var_e']`.

### 2.2 `truncreg`

Assembly: `src/westata/estimators/limited.py:518-537` + `_extra_scalars` at `:752-758`.

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `ll`, `ll_0`, `df_m`, `chi2`, `p`, `r2_p`, `rank`, `sigma`, `var_e`, `converged`, `aic`, `bic` | yes — `limited.py:519-535` | match |
| `lower`, `upper`, `n_truncated`, `n_uncensored` | yes — `limited.py:754-757` | match |
| `e(ic)` | **no** | missing + naming gap |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |
| all macros | **no** | missing |
| `Limit: Lower = 0 / Upper = +inf` printed pair | **yes in Stata** (p6p7_outputs.txt:13-14) | **not rendered** | **display gap** |
| iteration log | yes in Stata | no | display gap |

Runtime: `truncreg e keys: ['N','aic','bic','chi2','chi2type','converged','df_m','df_r','ll','ll_0','lower','n_truncated','n_uncensored','p','r2_p','rank','sigma','upper','var_e']`.

### 2.3 `intreg`

Assembly: `src/westata/estimators/limited.py:518-537` + `_extra_scalars` at `:947-955`.

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `ll`, `ll_0`, `df_m`, `chi2`, `p`, `r2_p`, `rank`, `sigma`, `var_e`, `converged`, `aic`, `bic` | yes | match |
| `lnsigma`, `n_left`, `n_right`, `n_interval`, `n_uncensored` | yes — `limited.py:950-954` | match |
| `e(ic)` | no | missing + naming gap |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |
| all macros | **no** | missing |
| iteration log | yes in Stata (p6p7_outputs.txt:32-36) | no | display gap |

Runtime: `intreg e keys: ['N','aic','bic','chi2','chi2type','converged','df_m','df_r','ll','ll_0','lnsigma','n_interval','n_left','n_right','n_uncensored','p','r2_p','rank','sigma','var_e']`.

### 2.4 `heckman` — see Q4

Two assembly sites: two-step at `src/westata/estimators/limited.py:1827-1849`, ML at `src/westata/estimators/limited.py:1675-1691`.

| Stata key | In westata 2-step? | In westata ML? | Verdict |
|---|---|---|---|
| `e(lambda)` | yes — `limited.py:1844` | yes — `limited.py:1683` | match |
| `e(rho)` | yes — `limited.py:1840` | yes — `limited.py:1681` | match |
| `e(sigma)` | yes — `limited.py:1839` | yes — `limited.py:1682` | match |
| `e(sel_ll)` | **no** | **no** | **missing (both)** |
| `e(N_selected)` | yes — `limited.py:1829` | yes — `limited.py:1677` | match |
| `e(N_nonselected)` | yes — `limited.py:1830` | **no** | **missing in ML only** |
| `e(ll)` | yes — `limited.py:1833` | yes — `limited.py:1678` | match |
| `chi2` (Wald) + `p` | yes — `limited.py:1835-1837` | **no** | **missing in ML** |
| `e(ic)` | **no** | **no** | missing |
| `e(k)`, `e(k_eq)`, `e(k_eq_model)`, `e(k_dv)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | **no** | missing |
| `e(athrho)`, `e(lnsigma)` as `e()` scalars | **no** (2-step has no such params) | yes — `limited.py:1679-1680` | match |
| `e(se_rho)`, `e(se_sigma)`, `e(se_lambda)` | **no** | yes — `limited.py:1684-1688` | westata addition |
| `e(method)`, `e(estimator)`, `e(vce)`, `e(se_corrected)`, `e(lam_coef)` | yes (as strings/ints) | yes | westata addition |
| printed `(regression model with sample selection)` + `Selected`/`Nonselected` header | **yes in Stata** (p6p7_outputs.txt:73-75) | **not rendered** | **display gap** |
| printed `rho / sigma / lambda` delta-method block | **yes in Stata** (p6p7_outputs.txt:93-95) | **not rendered** | **display gap** |
| printed `LR test of indep. eqns. (rho = 0): chi2(1) = 8.42  Prob > chi2 = 0.0037` | **yes in Stata** (p6p7_outputs.txt:97) | **no surface anywhere** | **missing (both stored and printed)** |
| iteration log | yes in Stata | no | display gap |

Runtime:
```
heckman (twostep) e keys: ['N','N_nonselected','N_selected','chi2','chi2type','df_m','df_r','estimator','lam_coef','lambda','ll','ll_0','p','r2_p','rank','rho','se_corrected','sigma','vce']
heckman (ml)     e keys: ['N','N_selected','athrho','converged','df_m','df_r','lambda','ll','lnsigma','method','rho','se_lambda','se_rho','se_sigma','sigma']
```

### 2.5 `etregress`

Two assembly sites: control function at `src/westata/estimators/limited.py:2235-2251`, ML at `src/westata/estimators/limited.py:2107-2123`.

| Stata key | In westata control? | In westata ML? | Verdict |
|---|---|---|---|
| `N`, `ll`, `ll_0`, `df_m`, `rank`, `chi2`, `p`, `r2_p`, `sigma`, `rho` | yes — `limited.py:2236-2247` | partially — `limited.py:2108-2114` has `N`, `ll`, `rho`, `sigma` but **no `chi2`/`p`/`r2_p`/`df_m`** | control: match; ML: **missing** |
| `e(N_treated)` | yes — `limited.py:2237` | yes — `limited.py:2109` | match |
| `e(ic)` | **no** | **no** | missing |
| `e(k)`, `e(k_eq)`, `e(k_eq_model)`, `e(k_dv)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | **no** | missing |
| `e(athrho)`, `e(lnsigma)` | **no** (control has none) | yes — `limited.py:2111-2112` | match |
| `e(se_rho)`, `e(se_sigma)`, `e(se_lambda)` | **no** | yes — `limited.py:2116-2120` | westata addition |
| `e(estimator)`, `e(vce)`, `e(se_corrected)`, `e(converged)`, `e(method)` | yes | yes | westata addition |
| printed `lambda` row and `rho`/`sigma`/`lambda` block | yes in Stata | **not rendered** | **display gap** |

Runtime:
```
etregress (control) e keys: ['N','N_treated','chi2','chi2type','df_m','df_r','estimator','ll','ll_0','p','r2_p','rank','rho','se_corrected','sigma','vce']
```

---

## 3. Family 3 — Panel nonlinear

Assembly: `src/westata/estimators/panel_nonlinear.py:719-750` (`_assemble`) plus per-model `extra` dicts. **Note the `e(ic)` collision** — `panel_nonlinear.py:725` sets `"ic": int(iterations)`, i.e. westata's `e(ic)` in the panel family is the **iteration count**, not Stata's AIC/BIC. See Q5.

### 3.1 `xtlogit fe` — `extra` at `panel_nonlinear.py:955-962`

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `ll`, `chi2`, `p`, `df_m`, `rank`, `converged` | yes | match |
| `N_g` | yes — `panel_nonlinear.py:721` | match |
| `N_drop`, `N_group_drop` | yes — `panel_nonlinear.py:959-960` | match (Stata's `e(N_drop)`) |
| `g_avg`, `g_max`, `g_min` | yes — `panel_nonlinear.py:963` | match |
| `estimator`, `chi2type` | yes | match |
| `e(ic)` | yes **but = iteration count** (`panel_nonlinear.py:725`) | **semantic mismatch** (Q5) |
| `e(ll_0)` | yes (conditional null) | match |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |
| all macros (`chi2type` is stored; `crittype`, `ml_method`, `technique`, `opt`, `estat_cmd`, `predict`, `properties`, `which`, `user`, `singularHmethod`, `deriv_useminbound`, `intmethod` as macro) | mostly no; `intmethod`/`estimator` stored as strings | partial |
| iteration log | printed by Stata | no | display gap |

Runtime: `xtlogit fe e keys: ['N','N_drop','N_g','N_group_drop','aic','bic','chi2','chi2type','converged','df_m','df_r','estimator','g_avg','g_max','g_min','ic','ll','ll_0','p','r2_p','rank']` (`ic = 7`).

### 3.2 `xtlogit re` — `extra` at `panel_nonlinear.py:1040-1049`

| Stata key | In westata? | Verdict |
|---|---|---|
| base block (`N`, `N_g`, `ll`, `ll_0`, `chi2`, `p`, `df_m`, `rank`, `converged`, `aic`, `bic`, `ic`) | yes | match |
| `sigma_u`, `rho` | yes — `panel_nonlinear.py:1044-1045` | match |
| `intpoints`, `intmethod` | yes — `panel_nonlinear.py:1046-1047` | match |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |
| `e(ic)` semantics | iteration count | mismatch (Q5) |

Runtime: `xtlogit re e keys: ['N','N_g','aic','bic','chi2','chi2type','converged','df_m','df_r','estimator','ic','intmethod','intpoints','ll','ll_0','p','r2_p','rank','rho','sigma_u']` (`ic = 156`).

### 3.3 `xtlogit pa` — `extra` at `panel_nonlinear.py:1195-1208`

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `N_g`, `chi2`, `p`, `df_m`, `rank`, `converged` | yes | match |
| `corr`, `alpha`, `family`, `link`, `scale`, `vce`, `estimator`, `quasi_likelihood` | yes — `panel_nonlinear.py:1199-1207` | match |
| `e(ll)`, `e(ll_0)`, `e(r2_p)`, `e(aic)`, `e(bic)` | **deliberately absent** (`llf=nan`, `panel_nonlinear.py:1192-1193`; GEE has no likelihood) | correct absence, documented |
| `e(ic)` | iteration count | mismatch (Q5) |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |

Runtime: `xtlogit pa e keys: ['N','N_g','alpha','chi2','chi2type','converged','corr','df_m','df_r','estimator','family','ic','link','p','quasi_likelihood','rank','scale','vce']` — **no `ll`**, which is why the Stata-style summary renders an `F`-test + OLS R² block (see §4).

### 3.4 `xtprobit re`

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `N_g`, `ll`, `ll_0`, `chi2`, `p`, `df_m`, `rank`, `converged`, `aic`, `bic`, `ic` | yes | match |
| `sigma_u`, `rho`, `intpoints`, `intmethod`, `estimator` | yes | match |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |

Runtime: `xtprobit re e keys: ['N','N_g','aic','bic','chi2','chi2type','converged','df_m','df_r','estimator','ic','intmethod','intpoints','ll','ll_0','p','r2_p','rank','rho','sigma_u']`.

### 3.5 `xttobit` — `extra` at `panel_nonlinear.py:1516-1531`

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `N_g`, `ll`, `ll_0`, `chi2`, `p`, `df_m`, `rank`, `converged`, `aic`, `bic`, `ic` | yes | match |
| `sigma_u`, `sigma_e`, `rho` | yes — `panel_nonlinear.py:1520-1522` | match |
| `lower`, `upper`, `n_uncensored`, `n_left`, `n_right` | yes — `panel_nonlinear.py:1523-1527` | match |
| `intpoints`, `intmethod`, `estimator` | yes | match |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |

Runtime: `xttobit e keys: ['N','N_g','aic','bic','chi2','chi2type','converged','df_m','df_r','estimator','ic','intmethod','intpoints','ll','ll_0','lower','n_left','n_right','n_uncensored','p','r2_p','rank','rho','sigma_e','sigma_u','upper']`.

### 3.6 `xtpoisson fe` — `extra` at `panel_nonlinear.py:1660-1667`

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `N_g`, `ll`, `chi2`, `p`, `df_m`, `rank`, `converged`, `aic`, `bic`, `ic` | yes | match |
| `N_drop`, `N_group_drop`, `g_avg/g_max/g_min`, `estimator` | yes — `panel_nonlinear.py:1664-1668` | match |
| `e(ll_0)` | **absent** (`ll0=nan`, `panel_nonlinear.py:1676`) | match — Stata also reports no `ll_0` for `xtpoisson, fe` |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |

Runtime: `xtpoisson fe e keys: ['N','N_drop','N_g','N_group_drop','aic','bic','chi2','chi2type','converged','df_m','df_r','estimator','g_avg','g_max','g_min','ic','ll','p','rank']`.

### 3.7 `xtpoisson re` — `extra` at `panel_nonlinear.py:1738-1746`

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `N_g`, `ll`, `ll_0`, `chi2`, `p`, `df_m`, `rank`, `converged`, `aic`, `bic`, `ic` | yes | match |
| `alpha`, `lnalpha` | yes — `panel_nonlinear.py:1742-1743` | match |
| `estimator`, `intmethod` | yes — `panel_nonlinear.py:1744-1745` | match |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |

Runtime: `xtpoisson re e keys: ['N','N_g','aic','bic','alpha','chi2','chi2type','converged','df_m','df_r','estimator','ic','intmethod','ll','ll_0','lnalpha','p','r2_p','rank']`.

### 3.8 `xtnbreg fe` — `extra` at `panel_nonlinear.py:1989-2006`

| Stata key | In westata? | Verdict |
|---|---|---|
| `N`, `N_g`, `ll`, `chi2`, `p`, `df_m`, `rank`, `converged`, `aic`, `bic`, `ic` | yes | match |
| `N_drop`, `N_group_drop`, `g_avg/g_max/g_min`, `estimator` | yes — `panel_nonlinear.py:1993-1995` | match |
| `quasi_likelihood`, `quasi_likelihood_note` | yes — `panel_nonlinear.py:2000-2005` | westata addition (honest labelling) |
| `e(ll_0)` | absent (`ll0=nan`, `panel_nonlinear.py:2015`) | match |
| `e(k)`, `e(k_eq)`, `e(k_dv)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(rc)` | **no** | missing |

Runtime: `xtnbreg fe e keys: ['N','N_drop','N_g','N_group_drop','aic','bic','chi2','chi2type','converged','df_m','df_r','estimator','g_avg','g_max','g_min','ic','ll','p','quasi_likelihood','quasi_likelihood_note','rank']`.

---

## 4. Display surface gaps — what Stata prints but westata renders nowhere

Verified by running `result.summary(style="stata")` for each model. The renderer is `src/westata/results/summary.py`; `_stata_summary` is `:132-245`.

### 4.1 Q1 — `logit`'s four post-table lines

**Question:** does westata render all four of `Log likelihood`, `Pseudo R2`, `LR chi2(2)`, `Prob > chi2`?

**Answer: yes — all four, for `logit`.** Runtime output:

```
Number of obs         =         74
LR chi2(2)            =      35.72
Prob > chi2           =     0.0000
Log likelihood        =   -27.1752
Pseudo R2             =     0.3966
```

Evidence chain:

* `Log likelihood` — `src/westata/results/summary.py:219` (`lines.append(f"{'Log likelihood':<21} = ...")`), gated on `r2_p` being finite (`:218`).
* `Pseudo R2` — `src/westata/results/summary.py:220`.
* `LR chi2(2)` — `src/westata/results/summary.py:191-194`; the label comes from `e(chi2type)` (`:185-186`), which `logit` sets to `"LR"` at `src/westata/estimators/discrete.py:613`.
* `Prob > chi2` — `src/westata/results/summary.py:195`.

**But the four-line guarantee is conditional, not structural.** The `Log likelihood` / `Pseudo R2` pair lives inside `if np.isfinite(pseudo_r2):` (`summary.py:218`). Whenever `e(r2_p)` is `NaN`, the renderer silently falls through to the **OLS branch** at `summary.py:221-224` and prints `R-squared / Adj R-squared / Root MSE` — three lines that are *meaningless* for a maximum-likelihood model. Two live failures in scope:

1. **`nbreg`** — `e(ll_0)` is `NaN` on this dataset (`e(ll_0)` comes from statsmodels' `result.llnull`, which `NegativeBinomial` does not provide; the code path is `src/westata/estimators/discrete.py:597-602`). Result:
   ```
   LR chi2(2)            =          .
   Prob > chi2           =          .
   R-squared             =          .
   Adj R-squared         =          .
   Root MSE              =     0.2951
   ```
   Stata's `nbreg` fixture *does* carry `ll_0 = -1040.145594343475` and `r2_p = 0.060909064` (nbreg.json:38, 47). So this is a **data-availability gap** (see §5), not a renderer gap.
2. **`xtlogit, pa`** — no likelihood by construction (`panel_nonlinear.py:1192-1193`), so `r2_p` is absent and the renderer prints `F(2, 21)` + an all-`.` R² block. Stata's `xtlogit, pa` also has no `e(ll)`, but it *does* print `Wald chi2` rather than an F test, and it does not print an R-squared block at all.

`svy: logit` has `e(ll)` but no `e(r2_p)`, so it also renders the OLS block (`R-squared . / Adj R-squared . / Root MSE .6629`). Stata's `svy: logit` prints neither.

**Verdict on Q1:** the four lines are all implemented and correct for `logit`/`probit`/`poisson`/`ologit`/`oprobit`/`tobit`/`truncreg`/`intreg`/`heckman`/`etregress`/`xtlogit fe|re`/`xtprobit re`/`xttobit`/`xtpoisson re`. The gap is that the **fallback is an OLS block rather than a likelihood-appropriate block**, which surfaces for `nbreg` (a real data gap) and `xtlogit, pa` (a real semantic gap).

### 4.2 Q2 — `estat gof` (deviance / Pearson goodness of fit)

**Question:** is there any goodness-of-fit surface in westata for `poisson`/`nbreg`?

**Answer: no — none anywhere in `src/`.**

Evidence:

* `grep -rn "gof" src/westata` → **0 hits** outside a docstring-free code path; the only `gof` token in the whole repo's fixtures is Stata's own macro `"gof": "poiss_g"` at `tests/conformance/stata_fixtures/poisson_sim.json:41`. westata stores no such macro.
* `grep -rn "deviance" src/westata` → exactly two hits, both in the **`nl`** estimator, out of scope: `src/westata/estimators/nonlinear.py:86` (docstring, "``e(deviance)`` is Stata's 'Res. dev.'") and `src/westata/estimators/nonlinear.py:1340` (`"deviance": -2.0 * log_likelihood`). No count model sets it.
* `grep -rn "pearson" src/westata` → one hit, and it is a *comment about a different estimator*: `src/westata/estimators/panel_nonlinear.py:81` ("which is neither the Pearson nor the deviance estimate this module computes"). No Pearson residual statistic is exposed.
* There is no `estat` command surface at all. `src/westata/postestimation/__init__.py:1-31` exports exactly `test`, `lincom`, `nlcom`, `testnl`, `contrast`, `pwcompare`, `hausman`, `lrtest`, `margins`, `predict`. No `estat ic`, no `estat gof`, no `estat summarize`, no `estat ic`→`r()` return.

So `poisson`/`nbreg`/`zip`/`zinb`/`ztp`/`tnbreg`/`xtpoisson`/`xtnbreg` in westata expose **no deviance, no Pearson chi2, no dispersion statistic, no `estat gof` output block**. This is a whole missing postestimation command, not a missing key.

### 4.3 Q3 — `ologit`/`oprobit` cutpoints and convergence

**Question:** does westata expose cutpoints as named params, and record convergence?

**Cutpoints: yes, but under a westata-only spelling.** Runtime `param names: ['weight','mpg','/cut1','/cut2','/cut3','/cut4']`. The `/cut` prefix is assigned at `src/westata/estimators/discrete.py:1541` (`names = names + [f"/cut{i + 1}" for i in range(extra)]`). Stata's fixture uses bare `cut1`, `cut2` (ologit.json:80-85). So the values, order, and covariance are right, but a script doing `r.e.bnames` / `r.params["cut1"]` against Stata's names will miss. (The same convention is deliberate and documented for `/lnalpha` at `discrete.py:1016-1019`, where it matches Stata exactly — but for cutpoints Stata does *not* use the slash.)

**Convergence: no.** `ologit`/`oprobit` do not set `e(converged)`. Runtime: `converged in e? False`. The ordered assembly site `discrete.py:1562-1583` writes `N`, `df_m`, `k_aux`, `rank`, `ll`, `ll_0`, `r2_p`, then `chi2`/`chi2type`/`p`/`df_m` and `N_clust` — no `converged`, no `rc`, no iteration count. Compare `tobit` (`tobit.py:399`), `truncreg`/`intreg` (`limited.py:530-532`), NB1 (`discrete.py:953`), all panel models (`panel_nonlinear.py:724`), which *do* record it. The ordered models are the gap.

**`N_cd` / `k_cat` / `k_aux` vs westata:**

| Key | ologit fixture | oprobit fixture | westata ologit/oprobit |
|---|---|---|---|
| `k_aux` | 2.0 (ologit.json:54) | 2.0 (oprobit.json:54) | **present**, but = the *number of cut points actually estimated*, so `4` on this 5-category `rep78` data vs Stata's `2` on its 3-category fixture — same definition, different sample |
| `k_cat` | 3.0 (ologit.json:53) | 3.0 (oprobit.json:53) | **missing** |
| `N_cd` | 0.0 (ologit.json:52) | 0.0 (oprobit.json:52) | **missing** |
| `noconstant` | 1.0 (ologit.json:40) | 1.0 (oprobit.json:40) | **missing** |
| `k_eq` | 3.0 (ologit.json:39) | 3.0 (oprobit.json:39) | **missing** |
| `ic` | 4.0 / 3.0 | | **missing** (stores `aic`/`bic`? **no** — ordered models set neither; see §5) |
| `version` | 3.0 (ologit.json:55) | 3.0 (oprobit.json:55) | **missing** |
| macros `depvar_outcomes` (`"1 2 3"`), `marginsdefault`, `marginsderiv`, `marginsprop` | yes | yes | **missing** |

**Also missing for ordered models:** `e(aic)`/`e(bic)` are **absent** from `ologit`/`oprobit` entirely (runtime confirms: no `aic`/`bic` keys), even though every other ML model in scope has them. And Stata's printed `Iteration log` has no westata counterpart.

### 4.4 Q4 — `heckman`'s `e()` set

**Question:** which of `e(lambda)`, `e(rho)`, `e(sigma)`, `e(sel_ll)`, `e(N_selected)`, `e(N_nonselected)`, Wald chi2, and the selection note does westata set?

| Item | 2-step | ML | Citation |
|---|---|---|---|
| `e(lambda)` | **yes** | **yes** | `limited.py:1844` / `limited.py:1683` |
| `e(rho)` | **yes** | **yes** | `limited.py:1840` / `limited.py:1681` |
| `e(sigma)` | **yes** | **yes** | `limited.py:1839` / `limited.py:1682` |
| `e(N_selected)` | **yes** | **yes** | `limited.py:1829` / `limited.py:1677` |
| `e(N_nonselected)` | **yes** | **no** | `limited.py:1830`; ML assembly `limited.py:1675-1691` has no such key |
| `e(sel_ll)` | **no** | **no** | neither assembly site sets it |
| Wald chi2 + `p` | **yes** | **no** | `limited.py:1835-1837`; ML site has no `chi2`/`p` |
| `e(ic)` | **no** | **no** | |
| `e(ll)` | yes | yes | `limited.py:1833` / `limited.py:1678` |
| printed `(regression model with sample selection)` + `Selected`/`Nonselected` | **no** | **no** | Stata prints it (p6p7_outputs.txt:73-75); no renderer path exists |
| printed `rho / sigma / lambda` delta-method block | **no** | **no** | Stata prints it (p6p7_outputs.txt:93-95); westata stores the three values but never renders them together |
| printed `LR test of indep. eqns. (rho = 0)` | **no** | **no** | Stata prints it (p6p7_outputs.txt:97); not stored, not computed, not printed |
| iteration log | **no** | **no** | Stata prints it (p6p7_outputs.txt:69-71) |

westata also adds `e(se_rho)`, `e(se_sigma)`, `e(se_lambda)` for the ML path (`limited.py:1684-1688`) and `e(lam_coef)`, `e(estimator)`, `e(vce)`, `e(se_corrected)` for 2-step (`limited.py:1845-1848`) — genuine additions beyond Stata's surface.

### 4.5 Other display gaps (all families)

| Stata prints | Stata evidence | westata |
|---|---|---|
| `Iteration log` (`Iteration 0: Log likelihood = …`) | p6p7_outputs.txt:6-10 (truncreg), :32-36 (intreg), :69-71 (heckman), :105-118 (zip), :143-151 (zinb) | **never rendered**; the iteration *count* is stored only in the panel family (`panel_nonlinear.py:725`, as `e(ic)`) and in `svy: logit` (`survey.py:1294`) |
| `Inflation model: logit` + `Nonzero obs` / `Zero obs` header pair for `zip`/`zinb` | p6p7_outputs.txt:121-122 | values stored (`counts.py:819-820`) but not rendered |
| `Limit: Lower = 0 / Upper = +inf` pair for `truncreg` | p6p7_outputs.txt:13-14 | values stored (`limited.py:754-755`) but not rendered |
| `(regression model with sample selection)` note | p6p7_outputs.txt:74 | not rendered |
| `rho / sigma / lambda` delta-method block | p6p7_outputs.txt:93-95 | not rendered |
| `LR test of indep. eqns. (rho = 0)` | p6p7_outputs.txt:97 | not rendered |
| AIC/BIC line (`estat ic`) | Stata's `estat ic` postestimation | **no `estat ic` surface at all** (see Q5) |
| `estat gof` (deviance/Pearson) | Stata's `poisson`/`nbreg` postestimation | **no surface at all** (see Q2) |
| `Number of obs` for the *selected* subsample in `heckman` | p6p7_outputs.txt:73 (`Number of obs = 1,500`) | westata's `e(N)` is 74 (full) and `nobs` is 22 (selected) — the printed header uses `nobs`, so the label and the count disagree with Stata's framing |

---

## 5. Q5 — AIC/BIC and the `e(ic)` compatibility gap

**Question:** does westata expose AIC/BIC for every ML model? Stata stores `e(ic)`; the logit fixture has `ic` but no `aic`/`bic`. What does westata store — `aic`/`bic` or `ic`? Is that a compatibility gap for `estat ic`?

**Findings.**

1. **Stata stores only `e(ic)`** for these commands — confirmed in every fixture: `logit_sim.json:27` (`"ic": 4.0`), `probit_sim.json:27`, `poisson_sim.json:19`, `nbreg.json:27`, `ologit.json:37`, `oprobit.json:37`. None of the six fixtures has an `aic` or `bic` key. Stata's `estat ic` *computes* AIC and BIC from `e(ll)` and `e(ic)` on demand and returns them in `r(S)`, `r(S_aic)`, `r(S_bic)` — it does not pre-store them.

2. **westata stores `aic` and `bic` directly, and stores no `e(ic)`** — for most of the scope:
   * `logit`/`probit`/`poisson`/NB2: `src/westata/estimators/discrete.py:616-617` (`"aic": float(getattr(result, "aic", ...))`, `"bic": ...`). Runtime: `has 'ic'? False  has aic? True  has bic? True`.
   * `zip`/`zinb`/`ztp`/`tnbreg`: `src/westata/estimators/counts.py:572-573`.
   * `truncreg`/`intreg`: `src/westata/estimators/limited.py:533-534`.
   * `xtlogit fe/re`, `xtprobit re`, `xttobit`, `xtpoisson fe/re`, `xtnbreg fe`: `src/westata/estimators/panel_nonlinear.py:739-741`.

3. **Three exceptions where westata stores neither `ic` nor `aic`/`bic`:**
   * **`ologit` / `oprobit`** — the ordered assembly at `discrete.py:1562-1583` sets no AIC/BIC at all. Runtime confirms no `aic`/`bic`/`ic` key. This is the clearest "ML model with no information criterion" gap in the whole scope.
   * **`xtlogit, pa`** — GEE, no likelihood, so no criterion. Runtime: no `aic`/`bic`/`ll`.
   * **`heckman` (both methods) and `etregress` (both methods)** — no `ic`, `aic`, or `bic` at any of the four assembly sites (`limited.py:1675-1691`, `:1827-1849`, `:2107-2123`, `:2235-2251`). Runtime confirms.

4. **The `e(ic)` name is actively dangerous in the panel family.** `src/westata/estimators/panel_nonlinear.py:725` sets `"ic": int(iterations)`. So for `xtlogit`, `xtprobit`, `xttobit`, `xtpoisson`, `xtnbreg`, `e(ic)` is the **optimiser iteration count**, not Stata's "number of parameters + 1" penalty term. Runtime: `xtlogit fe ic = 7`, `xtlogit re ic = 156`, `xttobit ic = 162`, `xtpoisson re ic = 116`. A reader porting `estat ic` logic from Stata would compute `-2*ll + 2*e(ic)` and get nonsense. (The docstring at `panel_nonlinear.py:695-696` says this key is intended to keep `e(ll)`/`e(r2_p)`/`e(aic)`/`e(bic)` out of the container — it does not explain the `ic` collision.)

**Verdict — yes, this is a real compatibility gap, in three distinct ways:**

* **Naming gap (all of scope except panel):** a Stata-porting caller reads `e(ic)`; westata answers `KeyError`. `e.aic`/`e.bic` are westata additions that happen to hold the same *numbers* Stata's `estat ic` would print, but under different names, so `estat ic` cannot be implemented as a thin `e(ic)` lookup.
* **Semantic collision (panel family only):** `e(ic)` *exists* but means the iteration count. This is worse than a missing key, because it silently returns a plausible-looking integer.
* **Coverage gap (`ologit`/`oprobit`/`heckman`/`etregress`):** no AIC/BIC under any name. For `ologit`, the inputs are all present (`e(ll)`, `e(rank)`, `e(N)`), so this is the cheapest fix in the entire audit.

---

## 6. Priority ranking

### 6.1 (i) Pure data-availability — already computed internally, just not stored. Cheap.

| # | Item | Where the value already exists | Cost |
|---|---|---|---|
| 1 | `e(converged)` for `ologit`/`oprobit` | the fit loop knows it; every sibling stores it (`tobit.py:399`, `limited.py:530`, `discrete.py:953`, `panel_nonlinear.py:724`) | one line at `discrete.py:1562-1583` |
| 2 | `e(converged)`/`e(rc)` for `logit`/`probit`/`poisson`/NB2/`zip`/`zinb`/`ztp`/`tnbreg` | `result.mle_retvals["converged"]` is already read at `discrete.py:462` for the strict-mode check | one line in `_stata_scalars` |
| 3 | `e(aic)`/`e(bic)` for `ologit`/`oprobit` | `ll`, `rank`, `N` all in scope at `discrete.py:1562-1574`; formula is the one used at `discrete.py:616-617` | two lines |
| 4 | `e(aic)`/`e(bic)` for `heckman`/`etregress` | `ll` and parameter count are in scope at all four assembly sites | two lines each |
| 5 | `e(k)`, `e(k_dv)`, `e(k_eq)`, `e(k_eq_model)`, `e(k_aux)`, `e(consonly)`, `e(noconstant)`, `e(k_autoCns)`, `e(rc)` for the whole scope | pure constants derived from the design already in hand (`k = rank`, `k_dv = 1`, `noconstant` known from `_param_names`) | one shared helper, applied at the four assembly sites |
| 6 | `e(alpha)` for NB2 (`nbreg`) | `/lnalpha` is already the reported parameter (`discrete.py:1045`) — `exp()` it, exactly as `counts.py:580` already does for `zinb`/`tnbreg` | one line |
| 7 | `e(ll_c)` / `e(chi2_c)` / `e(rank0)` for `nbreg`/`nbreg, nbre1` | the constant-only fit is already computed for `e(ll_0)`; `ll_c` is the *dispersion-only* variant, a second small fit | one extra fit |
| 8 | `e(k_cat)`, `e(N_cd)`, `e(version)` for `ologit`/`oprobit` | category count is `self._n_cuts + 1`, already computed at `discrete.py:1535-1540` | one line |
| 9 | `e(sel_ll)`, `e(N_nonselected)` (heckman ML) | `n_selected` is in scope at `limited.py:1677`; `N_nonselected = N - N_selected` | two lines |
| 10 | `e(chi2)`/`e(p)` for heckman ML and etregress ML | `covariance` and `params` are in scope at `limited.py:1675-1691` / `:2107-2123`; `_wald_statistic` is already imported | two lines |
| 11 | Rename `e(ic)` → iteration count under a non-colliding name (`e(iterations)` or `e(ic_iters)`) in the panel family | the value is already computed (`panel_nonlinear.py:669`, `:725`) | one line + a test update |
| 12 | `e(ll_0)` for NB2 | statsmodels' `NegativeBinomial` returns no `llnull` (confirmed at runtime: `e(ll_0) = nan`), but the constant-only Poisson-scale fit is already computed for the count family (`counts.py:541`) | reuse the NB1 constant-only construction at `discrete.py:933-937` |

Items 1–4 and 11 are the highest value-per-line in the whole audit: they close the `nbreg` R²-block defect (Q1), the ordered-model AIC gap (Q3), and the `e(ic)` collision (Q5) simultaneously.

### 6.2 (ii) Requires new computation

| # | Item | What has to be computed |
|---|---|---|
| 1 | `estat gof` for `poisson`/`nbreg`/`zip`/`zinb`/`ztp`/`tnbreg`/`xtpoisson`/`xtnbreg`: deviance `D = 2*sum[y*ln(y/mu) - (y-mu)]`, Pearson `X2 = sum[(y-mu)^2 / V(mu)]`, dispersion, and the GoF table with df and p | new function per family; the NB dispersion `V(mu)` differs by model (Poisson `V=mu`, NB2 `V=mu+alpha*mu^2`, NB1 `V=mu(1+delta)`), and the truncated/inflated variants need their own normalisation |
| 2 | `heckman`'s `LR test of indep. eqns. (rho = 0)` | a constrained fit with `rho = 0`, then `2*(ll_full - ll_constrained)` on 1 df; Stata prints it (p6p7_outputs.txt:97) |
| 3 | `heckman`'s `e(sel_ll)` | a separate selection-equation log-likelihood; the two-step path currently keeps only the probit coefficients, not its ll |
| 4 | `heckman` ML's Wald chi2 + `p` | `b_s' V_ss^-1 b_s` over slopes — `_wald_statistic` exists, needs wiring into `limited.py:1675-1691` |
| 5 | `etregress` ML's `chi2`/`p`/`r2_p`/`df_m` | same Wald wiring plus a null-model fit; currently the ML assembly (`limited.py:2107-2123`) writes none of them |
| 6 | NB1's `e(ll_c)`/`e(chi2_c)`/`e(rank0)` | the dispersion-only constant fit; `_nb1_loglike` is already in hand (`discrete.py:662-689`) |
| 7 | `N_cdf`/`N_cds` (logit/probit) and `N_cd` (ordered) | counting conventions; trivial once the semantics are pinned down, but they need a definition, not just a value |
| 8 | The macro block (`chi2type`, `crittype`, `ml_method`, `technique`, `opt`, `estat_cmd`, `marginsok`, `predict`, `properties`, `which`, `user`, `singularHmethod`, `deriv_useminbound`, `dispers`, `diparm1`, `depvar_outcomes`, `marginsdefault`, `marginsderiv`, `marginsprop`, `gof`) | each is a fixed string per command; no computation, but a decision about whether westata's `e()` should hold macros at all (today it holds `chi2type`, `dispersion`, `estimator`, `vce`, `method`, `corr`, `family`, `link`, `intmethod`, `quasi_likelihood_note` as plain strings, so the container is ready) |

### 6.3 (iii) Requires new display / render

| # | Item | Renderer change |
|---|---|---|
| 1 | **`estat gof` command surface** | a brand-new postestimation entry point, not a key. Nothing in `src/westata/postestimation/` corresponds to it (`__init__.py:1-31`) |
| 2 | **`estat ic` command surface** returning AIC/BIC | new entry point; must resolve the `e(ic)` collision first (§6.1 #11) or it will be actively wrong |
| 3 | Likelihood-appropriate fallback block for the `summary.py:221-224` OLS branch | when `r2_p` is `NaN` **and** `e(ll)` is present, print `Log likelihood` without `Pseudo R2` rather than `R-squared/Adj R-squared/Root MSE`. This is the fix for `nbreg`'s current output and for `svy: logit` |
| 4 | `zip`/`zinb` header block: `Inflation model: logit`, `Nonzero obs`, `Zero obs` | renderer addition; values already stored (`counts.py:819-820`) |
| 5 | `truncreg` header block: `Limit: Lower` / `Upper` | renderer addition; values already stored (`limited.py:754-755`) |
| 6 | `heckman` header note `(regression model with sample selection)` + `Selected`/`Nonselected` | renderer addition; values already stored (`limited.py:1829-1830`) |
| 7 | `heckman`/`etregress` delta-method block `rho / sigma / lambda` with SEs and CIs | renderer addition; values already stored (`limited.py:1681-1688`, `:1839-1845`) |
| 8 | `heckman`'s `LR test of indep. eqns. (rho = 0)` printed line | depends on §6.2 #2 |
| 9 | `ologit`/`oprobit` cutpoints under Stata's bare `cut1`/`cut2` spelling | a naming decision at `discrete.py:1541`; affects `params` index and any `test`/`lincom` written against Stata names |
| 10 | Iteration log | new renderer path; Stata prints it by default (`p6p7_outputs.txt:6-10`). Note westata stores the count only in the panel family, so this needs §6.1 #2 first |

---

## 7. Audit hygiene

* Temporary probes `_audit_probe.py`, `_audit_probe2.py`, `_audit_probe3.py` were created in the repo root and **deleted** after the run. No other file outside `retest/gaps/` was created or modified.
* All fits were run with `./.venv/Scripts/python.exe` from `D:\myCodeProject\statapy`.
* Every `e()` key list in §1–§3 is verbatim runtime output, not inference.
* No claim in this document rests on a line that was not read.
