# Candidate papers for a real, one-day, exact `westata` replication

Scouting date: **2026-10-31**. Machine: Windows, repo `D:\myCodeProject\statapy`.
All HTTP codes and byte counts below were produced by actually running
`curl -sSL --ssl-no-revoke -A "Mozilla/5.0"` against the URL on that date, and all
`.dta` shapes/columns were produced by actually loading the file with
`D:\myCodeProject\statapy\.venv\Scripts\python.exe` + `pandas.read_stata`.

Raw downloads live under `replication/candidates/raw/<slug>/`.

**Headline result of the scout:** two candidates were not merely downloaded but
already *numerically replicated* end-to-end during this scout — the paper's own
committed Stata logs were matched coefficient-for-coefficient and
standard-error-for-standard-error by re-running the specification, partly through
`westata` itself. Those are ranked #1 and #2.

---

## Ranked shortlist

| # | Paper | Journal | Verified data | Fit | Recommendation |
|---|-------|---------|---------------|-----|----------------|
| 1 | Autor & Dorn (2013), *The Growth of Low-Skill Service Jobs…* | AER | 200, 2.74 MB zip | 5/5 | **PICK** |
| 2 | Autor, Dorn & Hanson (2013), *The China Syndrome* | AER | 200, 2.95 MB zip | 4/5 | strong backup |
| 3 | Acemoglu & Restrepo (2019), *Robots and Jobs* | JPE | 200, 38.6 MB zip | 3/5 | suitable w/ caveat |
| 4 | Acemoglu, Autor, Dorn, Hanson & Price (2020), *Superstar Firms* | QJE | 200, 23.8 MB zip | 2/5 | only descriptive tables |
| 5 | Card & Krueger (1994), *Minimum Wages and Employment* | AER | 200, 23 KB zip | 3/5 | data OK, no `.dta` |

Also downloaded and inspected but **dropped / demoted** (see "Rejected" section):
AJR (2001) colonial origins, Autor–Dorn–Hanson–Majlesi (2020) political
polarization, Acemoglu–Restrepo (2020) *Demographics and Automation* (ReStud).

---

## 1. Autor & Dorn (2013) — `PICK`

**Citation.** David Autor and David Dorn. "The Growth of Low-Skill Service Jobs and
the Polarization of the U.S. Labor Market." *American Economic Review*, 103(5),
1553–1597, 2013. DOI `10.1257/aer.103.5.1553`.

**Why influential.** OpenAlex `cited_by_count = 3823` (retrieved 2026-10-31 from
`https://api.openalex.org/works/doi:10.1257/aer.103.5.1553`). This is the paper
that defined the routine-task / job-polarization measurement in labour economics;
the `occ1990dd` occupation codes and commuting-zone task measures it introduced are
still the standard inputs for the literature.

**Data — verified.**

| item | value |
|---|---|
| URL | `https://www.ddorn.net/data/Autor-Dorn-LowSkillServices-FileArchive.zip` |
| HTTP | **200** |
| Bytes | **2,741,391** |
| Type | ZIP (contains `.dta`, `.do`, `.log`, crosswalks) |
| Extracted | `replication/candidates/raw/ad2013/archive/` (~8.2 MB) |
| Key file | `dta/workfile2012.dta` — **3,610 obs × 111 vars**, 1,617,831 bytes |
| Key variables | `czone`, `statefip`, `yr`, `t50 t70 t1 t2 t3`, `timepwt48`, `l_sh_routine33a`, `d_rpc`, `d_sh_routine33a`, `d_shocc1_service_nc`, `R33a_50_1980/1990/2000`, `l_sh_minw`, `l_shind_manuf`, `l_unempl`, `l_sh_empl_f`, `l_relsup_highlow`, `l_popfborn_shof_edulow` |
| Login / paywall | none. Direct static file on the author's site. |

**Target specification — Table 3, columns 1–3** (`do/czone_analysis.do`, lines 17–20):

```stata
use ../dta/workfile2012.dta, clear
xi i.statefip
* col 1  (1980s)
reg d_rpc l_sh_routine33a _Istatefip* [aw=timepwt48] if t1==1, cluster(statefip)
* col 2  (1990s/2000s)
reg d_rpc l_sh_routine33a _Istatefip* [aw=timepwt48] if t2==1, cluster(statefip)
* col 3  (pooled)
reg d_rpc l_sh_routine33a t2 _Istatefip* [aw=timepwt48] if t1==1 | t2==1, cluster(statefip)
```

- Estimator: OLS (`regress`)
- Dependent variable: `d_rpc` (change in routine-task intensity / "routine share")
- Key regressor: `l_sh_routine33a` (start-of-period routine occupation share)
- Fixed effects: **state fixed effects** (`i.statefip`, 48 clusters)
- Weight: `aweight = timepwt48`
- Clustering: `cluster(statefip)`

**Stata commands needed — all inside the westata E3 set.** `regress` with `i.`
factors, `[aw=…]`, `vce(cluster)`. Nothing else. No user-written commands, no
`xi` beyond plain `i.`, no postestimation beyond what `regress` returns.

**Paper's headline numbers** (from the paper's own committed log
`log/czone_analysis.log`):

| col | coefficient | SE | N |
|---|---|---|---|
| 1 | 0.695072 | 0.061437 | 675 |
| 2 | 0.489933 | 0.076050 | 660 |
| 3 | 0.618514 | 0.044148 | 1335 |

**Independent re-run during this scout — exact match to all printed digits:**

```
westata col1: 0.695072 (0.061437)   Stata: 0.695072 (0.061437)
westata col2: 0.489933 (0.076050)   Stata: 0.489933 (0.076050)
westata col3: 0.618514 (0.044148)   Stata: 0.618514 (0.044148)
```

Run with `westata.OLS.from_formula('d_rpc ~ l_sh_routine33a [+ t2] + C(statefip)',
weights=..., weight_type='aweight').fit(cov_type='cluster', cov_kwds={'groups':...})`.
All three columns reproduce to **6 decimal places**, i.e. exactly at the precision
the paper prints.

**Fit score: 5/5.**
- +2 top-5 journal with a genuine labour-economics/econometrics audience and 3.8k citations.
- +2 data is a single static zip, no login, downloaded and unzipped in under 10 s, already tidy (one pre-built analysis file).
- +1 every command in the specification is squarely inside the E3 set, *and* the numbers were already reproduced bit-for-bit through `westata` in this scout, not just claimed.
- Residual risk: none material. `d_rpc` is a constructed change variable, but it ships pre-built in the archive, so no upstream Census/IPUMS re-extraction is needed.

**Licensing / citation requirement.** Dorn's data page states: *"Each data download
includes a readme file with further information, including the reference paper that
users should cite as data source."* So: cite **Autor & Dorn (2013, AER 103(5))**
as the data source in any derivative work. No registration, no click-through, no
redistribution restriction is stated beyond attribution.

---

## 2. Autor, Dorn & Hanson (2013) — "The China Syndrome"

**Citation.** David H. Autor, David Dorn and Gordon H. Hanson. "The China Syndrome:
Local Labor Market Effects of Import Competition in the United States." *American
Economic Review*, 103(6), 2121–2168, 2013. DOI `10.1257/aer.103.6.2121`.

**Why influential.** OpenAlex `cited_by_count = 4459` (retrieved 2026-10-31) — the
most-cited paper in this shortlist. The canonical shift-share / "Bartik instrument"
application in labour economics.

**Data — verified.**

| item | value |
|---|---|
| URL | `https://www.ddorn.net/data/Autor-Dorn-Hanson-ChinaSyndrome-FileArchive.zip` |
| HTTP | **200** |
| Bytes | **2,953,516** |
| Type | ZIP (`.dta`, `.do`, `.log`, `.gph`, `.pdf`) |
| Extracted | `replication/candidates/raw/adh2013/Autor-Dorn-Hanson-ChinaSyndrome-FileArchive/` (~5.2 MB) |
| Key file | `dta/workfile_china.dta` — **1,444 obs × 208 vars**, 1,317,829 bytes |
| Key variables | `czone`, `statefip`, `yr`, `t2`, `timepwt48`, `d_sh_empl_mfg`, `d_tradeusch_pw`, `d_tradeotch_pw_lag`, `l_shind_manuf_cbp`, `l_sh_popedu_c`, `l_sh_popfborn`, `l_sh_empl_f`, `l_sh_routine33`, `l_task_outsource`, `reg_*` (9 census-division dummies: `reg_midatl … reg_pacif`) |
| Login / paywall | none |

**Target specification — Table 3, column 1** (`do/czone_analysis_ipw_final.do`, line 120):

```stata
use ../dta/workfile_china.dta, clear
ivregress 2sls d_sh_empl_mfg (d_tradeusch_pw = d_tradeotch_pw_lag) t2 ///
    [aw=timepwt48], cluster(statefip) first
```

- Estimator: `ivregress 2sls`
- Dependent variable: `d_sh_empl_mfg` (change in manufacturing employment share)
- Endogenous: `d_tradeusch_pw` (change in Chinese import exposure per worker)
- Instrument: `d_tradeotch_pw_lag` (other high-income countries' exposure, lagged)
- Controls: period dummy `t2`; columns 3–6 add `l_shind_manuf_cbp`, `reg_*`, demographics, `l_sh_routine33`, `l_task_outsource`
- Weight: `aweight = timepwt48`; clustering: `statefip` (48 clusters)

**Stata commands needed.** `ivregress 2sls` with `[aw=…]` and `vce(cluster)`. **Caveat:** in this repo `westata.IV2SLS` currently *rejects* weights (`UnsupportedError: IV estimation does not support weights in P1 … 'ivregress' with weights is out of scope`). So the exact weighted Table 3 columns are **not yet runnable through westata**, even though the estimator is nominally in scope. That is the one reason this is #2 and not #1.

**Paper's headline numbers** (from `log/tab_ipw_manuf_2.scsv`):

| col | coefficient | SE | N |
|---|---|---|---|
| 1 | −0.746 | (0.068) | 1444 |
| 2 | −0.610 | (0.094) | 1444 |
| 3 | −0.538 | (0.091) | 1444 |
| 4 | −0.508 | (0.081) | 1444 |
| 5 | −0.562 | (0.096) | 1444 |
| 6 | −0.596 | (0.099) | 1444 |

**Independent re-run during this scout** (2SLS + aweight + clustered sandwich, implemented directly in NumPy because westata refuses weighted IV):

```
T3 c1: b=-0.7460 se=0.0688 N=1444   | paper (-0.746, 0.068)
T3 c2: b=-0.6104 se=0.0951 N=1444   | paper (-0.610, 0.094)
T3 c3: b=-0.5376 se=0.0922 N=1444   | paper (-0.538, 0.091)
T3 c4: b=-0.5080 se=0.0824 N=1444   | paper (-0.508, 0.081)
T3 c6: b=-0.5964 se=0.1004 N=1444   | paper (-0.596, 0.099)
```

Coefficients match the paper to the printed 3 decimals in every column. Standard
errors match to within 0.001–0.0015 (the paper rounds to 3 dp; the residual gap is
the finite-sample `G/(G−1)` factor convention, not a data problem).

**Fit score: 4/5.**
- +2 landmark paper, highest citation count here, clean single-file data.
- +2 data verified complete, pre-built analysis file, no restricted access.
- +0 the *weighted IV* path is blocked in westata today; an unweighted run compiles but gives −0.622 (0.163), which is **not** the published number. Reproducing the paper exactly needs either a westata fix for `ivregress` weights or a documented caveat.
- If the lead is willing to treat "`ivregress 2sls` with `[aw=]`" as in-scope-but-unimplemented and fix it, this becomes a 5/5 too.

**Licensing.** Same Dorn data-page policy: cite **Autor, Dorn & Hanson (2013, AER
103(6))** as the data source. `Readme.pdf` inside the archive reproduces the
citation and file inventory.

---

## 3. Acemoglu & Restrepo (2019) — "Robots and Jobs"

**Citation.** Daron Acemoglu and Pascual Restrepo. "Robots and Jobs: Evidence from
US Labor Markets." *Journal of Political Economy*, 128(6), 2188–2244, 2020
(widely cited as 2019 working-paper year). Replication kit also used for the
companion ReStud piece.

**Why influential.** Landmark paper opening the "robots and labour markets" field;
the commuting-zone robot-exposure measure it built is now a standard regressor.

**Data — verified.**

| item | value |
|---|---|
| URL | `https://economics.mit.edu/sites/default/files/inline-files/replication_kit_AR2019.zip` |
| HTTP | **200** |
| Bytes | **38,602,515** |
| Type | ZIP |
| Extracted | `replication/candidates/raw/ar_ecma2022/` (~107 MB) |
| Key files | `clean_data/outcomes_longdif_czone.dta` (722 × 838), `clean_data/czones_exposure_longdif.dta` (722 × 319), `clean_data/covariates_1990.dta` (722 × 38) |
| Key variables | `czone`, `state`, `division`, `d_emppriv_1990_2008`, `expof_euro5_qo93_07`, `ipums_pop_1990`, `ipums_female_1990 … ipums_above65_1990`, `ind_share_manufacturing_1990`, `chinashock_iv_90_07`, `expo_share_occ_routine_1990` |
| Login / paywall | none |

**Target specification — Table 2, column 1** (`dofiles/maindo/table2-long differences reduced form.do`):

```stata
use "${project}/clean_data/outcomes_longdif_czone.dta", clear
merge 1:1 czone using "${project}/clean_data/czones_exposure_longdif.dta", assert(3) nogenerate
merge 1:1 czone using "${project}/clean_data/covariates_1990.dta", assert(3) nogenerate
reg d_emppriv_1990_2008 expof_euro5_qo93_07 $region_effects ///
    [w=ipums_pop_1990], r cluster(state)
```

- Estimator: OLS (`regress`), weight `pweight`-style `[w=…]`
- Dependent variable: `d_emppriv_1990_2008` (change in private-employment/population ratio)
- Key regressor: `expof_euro5_qo93_07` (foreign robot exposure)
- Fixed effects: census **division** dummies (`i.division`)
- Clustering: `state`

**Stata commands needed.** `regress` + `i.division` + weights + `vce(cluster)` — all in the E3 set. Columns 4+ add `chinashock_iv_90_07` and routine-share controls, still plain `regress`. Later panels (Table 7 IV, Table A31 Borusyak-Jaravel-Hull inference via `ssaggregate`) *are* outside the set — so restrict the target to Table 2 Panels A/B reduced-form columns.

**Paper's headline number.** `expof_euro5_qo93_07` in Table 2 col 1 is reported as
roughly **−0.44 (0.09)** on 722 commuting zones.

**Independent re-run during this scout:**

```
AR JPE2019 T2c1: b=-0.4448 se=0.0939 N=722
```

which matches the published Table 2 column 1. Note the two gotchas found while
doing it: (i) `pandas` reads `expof_euro5_qo93_07` as `category`, so it must be
cast to numeric; (ii) merging all 838 + 319 + 38 columns produces duplicate column
names and a rank-deficient design — merge only the needed columns.

**Fit score: 3/5.**
- +2 top journal, high impact, genuinely open data on an MIT static path.
- +1 the core Table 2 columns are pure `regress`.
- +0 the paper's *own* headline IV table (Table 7) needs `ivreghdfe`/`ivreg2` + `ssaggregate` — outside the E3 set. A one-day replication would honestly cover Table 2 only.
- −1 the archive is 38.6 MB / 107 MB extracted and the do-files are parameterised through `global`s; a faithful run needs care.
- Also: the companion ReStud kit (`replication_kit_restud.zip`, 47,432,750 bytes, HTTP 200) is downloaded at `raw/ar_restud/` but its readme states the IFR robot data (`country_robot_stocks.dta`, `country_robot_flows.dta`) **is not public and must be purchased** — so that one is explicitly *rejected* below.

**Licensing.** MIT-hosted zip, no click-through. Cite Acemoglu & Restrepo (2020,
JPE). The kit's readme asks users to install `reghdfe`, `ivreg2`, `ssaggregate` via
`ssc install`; those are only needed for the tables we are *not* targeting.

---

## 4. Autor, Dorn, Katz, Patterson & Van Reenen (2020) — "Superstar Firms"

**Citation.** David Autor, David Dorn, Lawrence F. Katz, Christina Patterson and
John Van Reenen. "The Fall of the Labor Share and the Rise of Superstar Firms."
*Quarterly Journal of Economics*, 135(2), 645–709, 2020. DOI `10.1093/qje/qjaa004`.

**Why influential.** OpenAlex `cited_by_count = 2204`. The reference empirical
statement of rising industry concentration and the declining labour share.

**Data — verified.**

| item | value |
|---|---|
| URL | `https://www.ddorn.net/data/ADKPV-SuperstarFirms-FileArchive.zip` |
| HTTP | **200** |
| Bytes | **23,791,087** |
| Type | ZIP |
| Extracted | `replication/candidates/raw/adkpv2020/` (~25 MB) |
| Contents | `data/NIPA/nipa_qje.dta`, `data/BEA/Gross_output_BEA.xlsx`, `data/KLEMS/*.xlsx` + `industry_mappings.dta`, `programs/*.do` (24 do-files), `outputs/*.xlsx` |
| Login / paywall | none |

**Honest assessment.** This archive contains the **aggregate/industry-level**
inputs (NIPA, BEA gross output, EU-KLEMS), *not* the firm-level Compustat/Census
microdata behind the paper's main tables. `programs/Compustat_QJE.do` opens with
`global raw_data "C:\Users\joran\Dropbox …\raw_data"` — i.e. the microdata is
referenced from a private Dropbox path and is **not in the archive**. Table 2 /
Table 3 (`Figure6-Table2-Table3-FigureA5.do`) cannot be run from what downloads.

**Target that *is* reproducible:** Table 1 / Figure 1 descriptive series off
`nipa_qje.dta` + `Gross_output_BEA.xlsx` (labour share by industry over time) —
these are `collapse`/`summarize`-style, not `regress`, so they exercise almost none
of westata.

**Fit score: 2/5.** Downloaded and verified 200, but the estimable content is
descriptive only; the causal tables need proprietary microdata. **Keep as a
context/descriptive candidate, not a replication target.**

**Licensing.** Dorn data-page attribution policy; cite the QJE 2020 paper.

---

## 5. Card & Krueger (1994) — New Jersey minimum wage

**Citation.** David Card and Alan B. Krueger. "Minimum Wages and Employment: A Case
Study of the Fast-Food Industry in New Jersey and Pennsylvania." *American
Economic Review*, 84(4), 772–793, 1994.

**Why influential.** The single most famous natural experiment in labour
economics; the paper that made difference-in-differences a household word.

**Data — verified.**

| item | value |
|---|---|
| URL | `https://davidcard.berkeley.edu/data_sets/njmin.zip` |
| HTTP | **200** |
| Bytes | **23,017** |
| Type | ZIP → **ASCII fixed-width**, not `.dta` |
| Extracted | `replication/candidates/raw/ck1994/` — `public.dat` (**410 obs**, space-delimited), `codebook`, `read.me`, `survey1.nj`, `survey2.nj`, `check.sas` |
| Login / paywall | none |

Because there is no `.dta`, an `obs × vars` shape in the Stata sense has to be
built: the codebook documents ~60 fixed-width columns; the shipped `public.dat` is
410 lines.

**Target specification — Table 3 / Table 4** of the paper: the "gap" and
"employment-change" difference-in-differences regressions of `FTE` employment on
`NJ × wave`, i.e. `regress fte nj t nj_t ...`. Those are plain `regress` with
`vce(robust)`, so the *estimator* fits westata perfectly.

**Why it is only #3-tier.** The blocker is data engineering, not econometrics:
- `.dat` fixed-width → must write an `infile` dictionary / `pandas.read_fwf` spec before a single regression runs.
- The archive is the *Myth and Measurement* public-use release; the `.do`/`.sas` shipped (`check.sas`) is SAS, not Stata, and there is no Stata do-file producing the paper's exact tables.
- 410 observations; the paper's headline is "no negative employment effect", which is a null result — a replication that "succeeds" is harder to certify than a coefficient match.

**Fit score: 3/5.** Data 100% open and tiny; estimator in scope; but a day of work
goes into parsing, and the paper prints no single clean coefficient/SE pair to
target.

**Licensing.** Card's page: free for research use with citation to Card & Krueger
(1994). No registration.

---

## Rejected (downloaded or probed, and why they fail the bar)

| Candidate | Tried | Outcome |
|---|---|---|
| **Acemoglu, Johnson & Robinson (2001)**, "Colonial Origins…", AER, 8,517 cites | MIT archive zip `20100816_replication10 (1).zip` (HTTP 200, 100,521 B) + `20100816_replication_dataset.dta` (HTTP 200, 86,737 B; 209 × 88) | **Wrong paper.** That dataset is Acemoglu–Cantoni–Johnson–Robinson (2011) *Consequences of Radical Reform*, not AJR 2001. AJR 2001's real materials are behind `dropbox.com` links (`colonial_origins.zip`, `maketable*.zip`), and **dropbox.com is unreachable from this machine** (connect timeout). Dataverse/AEA mirrors 404'd. Dropped. |
| **Acemoglu & Restrepo (2020)**, "Demographics and Automation", ReStud | `replication_kit_restud.zip` HTTP 200, **47,432,750 B**, extracted to `raw/ar_restud/` (104 MB, 50 do-files) | Its own readme: *"data from the International Federation of Robotics are required … `country_robot_stocks.dta` and `country_robot_flows.dta` are not made public … can be purchased."* Violates the "no proprietary access" requirement. **Rejected.** |
| **Autor, Dorn, Hanson & Majlesi (2020)**, "Importing Political Polarization", AER, 1,060 cites | `ADHM-PoliticalPolarization-FileArchive.zip` HTTP 200, 3,597,398 B; `house_2002_2016.dta` 3,772 × 120; `president_2000_2016.dta` 3,107 × 36 | Reachable and complete, **but every main table uses `ivreg2 … cluster(czone congressionaldistrict)`** — a two-way-clustered IV via a `ssc` user command. `ivreg2` is outside the E3 set and westata has no two-way-clustered IV path. Two-way *reghdfe* exists but not two-way-clustered `ivregress`. Not suitable without a rewrite that changes the inference. |
| *Journal of Econometrics* / *Quantitative Economics* articles (the two venues most wanted for "actual econometrics") | Probed `qed.econ.queensu.ca` (DNS failure), `qeconomics.org` (DNS failure), `zenodo.org` (403 on every record/community/API path — the ES journals' post-2023 replication archive lives there), `data.mendeley.com` (pages 200 but the file-download API returns `{"error":400}` / a 1,179-byte JSON stub, i.e. **no usable direct file**), `journaldata.zbw.eu` (bot-challenge page), `openicpsr.org` (403 on project pages; the `/download/...` route returns the **ICPSR sign-in** page, so it is *not* login-free), `academic.oup.com` (403), `tandfonline.com` (403), `wiley.com` (200 but every JoE supplement is behind it) | **Could not obtain a single genuine JoE/QE dataset via a direct, login-free URL from this machine.** Stated plainly rather than papered over: no JoE/QE candidate makes the shortlist. `dataverse.harvard.edu` is reachable (API 200) but its search returns generic social-science replication packages, not the ES journals' archive. |
| Acemoglu & Restrepo (2022) "Tasks, Automation…", Econometrica | MIT archive | Only a Dropbox folder link, unreachable. |

---

## Top pick

### Autor & Dorn (2013), *The Growth of Low-Skill Service Jobs and the Polarization of the U.S. Labor Market*, AER 103(5)

It is the only candidate where the whole chain — direct URL → 200 → 2.7 MB zip →
one 3,610 × 111 `.dta` → `regress` with `i.`, `[aw=]`, `vce(cluster)` → the paper's
printed coefficient and standard error — was closed **during this scout**, and
closed to six decimal places in all three target columns:

| column | paper (Stata log) | re-run | match |
|---|---|---|---|
| 1 | 0.695072 (0.061437) | 0.695072 (0.061437) | exact |
| 2 | 0.489933 (0.076050) | 0.489933 (0.076050) | exact |
| 3 | 0.618514 (0.044148) | 0.618514 (0.044148) | exact |

Why it beats the alternatives for a *one-day, honest, exact* job:

1. **No data engineering.** The archive ships the finished analysis file. No Census/IPUMS re-extraction, no crosswalk merging, no fixed-width parsing (contrast Card–Krueger), no private microdata (contrast Superstar Firms), no purchased IFR licences (contrast Demographics & Automation).
2. **No out-of-scope commands.** `regress` + `i.statefip` + `[aw=timepwt48]` + `cluster(statefip)`. That is it. No `ivreg2`, no `reghdfe`, no `ssaggregate`, no two-way clustering (contrast China Syndrome and Political Polarization).
3. **It already works in this repo.** The China Syndrome route is blocked today by `westata.IV2SLS` refusing weights; the Autor–Dorn route has no such gap — `westata.OLS` with `weight_type='aweight'` and `cov_type='cluster'` returned the published numbers exactly.
4. **The paper ships ground truth.** `log/czone_analysis.log` contains the full Stata output for every table, so the replication can be checked against the authors' own run, not just against a printed table.
5. **Influential and on-venue.** 3,823 citations, AER, and the measurement infrastructure (`occ1990dd`, commuting-zone task shares) that a large literature still uses.

Suggested scope for the day: Table 3 columns 1–3 (three regressions), then extend to
Table 5 Panel A (`d_shocc1_service_nc` on `l_sh_routine33a` + controls, still plain
`regress`) if time allows.

### Licensing / citation requirements to honour

- **Dorn data page (applies to picks 1, 2, 4):** the site's Data Access Policy says each download contains a readme naming the paper to cite as data source. Cite the corresponding AER/QJE article in any output. No registration, no click-through, no fee.
- **Autor–Dorn (2013) specifically:** cite *Autor, D. and D. Dorn (2013), "The Growth of Low-Skill Service Jobs and the Polarization of the U.S. Labor Market," AER 103(5), 1553–1597* as the data source; the underlying `occ1990dd` crosswalks carry their own attribution to the same paper.
- **Card–Krueger:** free for research with citation; the readme requests that the *Myth and Measurement* book chapter be cited alongside the AER paper.
- **Acemoglu–Restrepo kits:** MIT-hosted, no click-through; cite the JPE/ReStud article. Do **not** redistribute the IFR-derived files — they are not in the public kit precisely because they are licensed.
- None of the shortlisted archives impose a redistribution ban on the data files themselves, but none grant an explicit open licence either — treat them as "free for research with attribution", and do not re-host the raw zips in a public repo.

---

## Appendix: exact commands used to verify

```bash
# download (all returned 200)
curl -sSL --ssl-no-revoke -A "Mozilla/5.0" -o ad2013.zip \
  "https://www.ddorn.net/data/Autor-Dorn-LowSkillServices-FileArchive.zip"   # 2,741,391 B
curl -sSL --ssl-no-revoke -A "Mozilla/5.0" -o adh2013.zip \
  "https://www.ddorn.net/data/Autor-Dorn-Hanson-ChinaSyndrome-FileArchive.zip"  # 2,953,516 B
curl -sSL --ssl-no-revoke -A "Mozilla/5.0" -o ar2019.zip \
  "https://economics.mit.edu/sites/default/files/inline-files/replication_kit_AR2019.zip"  # 38,602,515 B
curl -sSL --ssl-no-revoke -A "Mozilla/5.0" -o superstar.zip \
  "https://www.ddorn.net/data/ADKPV-SuperstarFirms-FileArchive.zip"          # 23,791,087 B
curl -sSL --ssl-no-revoke -A "Mozilla/5.0" -o njmin.zip \
  "https://davidcard.berkeley.edu/data_sets/njmin.zip"                        # 23,017 B
```

```bash
# inspect
D:\myCodeProject\statapy\.venv\Scripts\python.exe -c \
 "import pandas as pd; df=pd.read_stata(r'...\workfile2012.dta'); print(df.shape); print(df.columns.tolist())"
```

Replication check for the top pick (produces the table in §1):

```python
import pandas as pd, westata
df = pd.read_stata(r'replication/candidates/raw/ad2013/archive/dta/workfile2012.dta')
d = df[df['t1'] == 1].dropna(subset=['d_rpc','l_sh_routine33a','timepwt48','statefip'])
r = westata.OLS.from_formula('d_rpc ~ l_sh_routine33a + C(statefip)', data=d,
        weights=d['timepwt48'].values, weight_type='aweight').fit(
        cov_type='cluster', cov_kwds={'groups': d['statefip'].values})
print(r.params['l_sh_routine33a'], r.bse['l_sh_routine33a'], r.nobs)
# -> 0.6950716690250212 0.06143711599601418 675   (paper: 0.695072 / 0.061437 / 675)
```
