| 1 | `regress mpg weight foreign` | ws.OLS.from_formula('mpg ~ weight + foreign').fit() | auto.dta | A | 5.1e-15 | 6.8e-16 |
| 2 | `regress mpg weight i.foreign` | OLS 'mpg ~ weight + C(foreign)' | auto.dta | A | 3.7e-14 | 5.1e-16 |
| 3 | `regress mpg weight i.rep78` | OLS 'mpg ~ weight + C(rep78)' | auto.dta | A | 1.2e-13 | 2.8e-15 |
| 4 | `regress ..., vce(robust)` | OLS(...).fit(cov_type='HC1') | auto.dta | A | 5.1e-15 | 4.6e-15 |
| 5 | `同上` | 同上，仅比 e(F) | auto.dta | A* | 5.1e-15 | 4.6e-15 |
| 6 | `regress y x1 x2, vce(cluster gid)` | OLS(...).fit(cov_type='cluster', cov_kwds={'groups': gid}) | 自造 140x14 组 | A | 3.5e-16 | 1.1e-15 |
| 7 | `同上` | 同上，仅比 e(df_r) | 自造 140x14 组 | A* | 3.5e-16 | 1.1e-15 |
| 8 | `同上 + test x1` | 同上，比 t 检验 p | 自造 140x14 组 | A | 3.5e-16 | 1.1e-15 |
| 9 | `同上` | 同上，仅比 e(F) | 自造 140x14 组 | A* | 3.5e-16 | 1.1e-15 |
| 10 | `regress y x1 x2 [fweight=w]` | OLS(weights=w, weight_type='fweight') | 自造 140x14 组 | A | 4.9e-16 | 3.0e-16 |
| 11 | `regress y x1 x2 [aweight=w]` | OLS(weights=w, weight_type='aweight') | 自造 140x14 组 | A | 1.1e-15 | 5.6e-16 |
| 12 | `regress y x1 x2 [pweight=w]` | OLS(weights=w, weight_type='pweight') | 自造 140x14 组 | A | 1.1e-15 | 8.9e-16 |
| 13 | `regress y x1 x2 [iweight=w]` | OLS(weights=w, weight_type='iweight') | 自造 140x14 组 | A | 4.9e-16 | 3.0e-16 |
| 14 | `同上` | 同上，仅比 e(F) | 自造 140x14 组 | A* | 1.1e-15 | 8.9e-16 |
| 15 | `regress ... [fweight=w], vce(robust)` | fweight + cov_type='HC1' | 自造 140x14 组 | A | 4.9e-16 | 1.6e-16 |
| 16 | `regress ... [aweight=w], vce(robust)` | aweight + cov_type='HC1' | 自造 140x14 组 | A | 1.1e-15 | 8.9e-16 |
| 17 | `regress mpg weight foreign if mpg>20` | OLS(..., subset=mpg>20) | auto.dta | A | 3.0e-14 | 4.2e-16 |
| 18 | `regress ... if weight<4000 & mpg>18` | OLS(..., subset=两个条件) | auto.dta | A | 3.0e-15 | 6.2e-16 |
| 19 | `areg y x1 x2, absorb(gid)` | ws.FEOLS('y ~ x1 + x2 \| gid') | 自造 140x14 组 | A | 1.1e-15 | 1.8e-16 |
| 20 | `areg ..., absorb(gid) vce(robust)` | FEOLS(...).fit(cov_type='HC1') | 自造 140x14 组 | A | 1.1e-15 | 8.3e-16 |
| 21 | `areg ..., absorb(gid) vce(cluster gid)` | FEOLS(...).fit(cov_type='cluster', {'cluster': 'gid'}) | 自造 140x14 组 | E | 1.1e-15 | 4.9e-02 |
| 22 | `reghdfe ..., absorb(gid) vce(cluster gid)` | 同 7c，但改比 reghdfe | 自造 140x14 组 | A | 9.4e-16 | 7.9e-16 |
| 23 | `（手算，无 Stata 命令）` | 手算 within CR0 × 两个候选 CR1 因子 | 自造 140x14 组 | 证据 | n/a | n/a |
| 24 | `areg ..., absorb(gid) vce(cluster t)` | FEOLS(...).fit(cov_type='cluster', {'cluster': 't'}) | 自造 140x14 组 | A | 7.8e-16 | 1.3e-15 |
| 25 | `areg y x1 x2, absorb(cont)` | FEOLS('y ~ x1 + x2 \| cont') | 自造 140x14 组 | A | 5.9e-16 | 3.3e-16 |
| 26 | `reghdfe y x1 x2, absorb(gid)` | FEOLS('y ~ x1 + x2 \| gid') | 自造 140x14 组 | A | 9.4e-16 | 1.9e-16 |
| 27 | `reghdfe ..., absorb(gid t)` | FEOLS('y ~ x1 + x2 \| gid + t') | 自造 140x14 组 | A | 1.1e-15 | 2.1e-16 |
| 28 | `reghdfe ..., absorb(gid t) vce(robust)` | FEOLS(gid+t).fit(cov_type='HC1') | 自造 140x14 组 | A | 1.1e-15 | 3.4e-16 |
| 29 | `reghdfe ..., absorb(gid t) vce(cluster gid)` | FEOLS(gid+t).fit(cov_type='cluster', {'cluster': 'gid'}) | 自造 140x14 组 | A | 1.1e-15 | 2.1e-16 |
| 30 | `areg y x1 x2 i.t, absorb(gid) vce(robust)` | FEOLS(gid+t).fit(cov_type='HC1') | 自造 140x14 组 | A | 9.4e-16 | 7.8e-16 |
| 31 | `ivregress 2sls yiv x1 (e1=z1)` | ws.IV2SLS.from_formula('yiv ~ x1 + [e1 ~ z1]') | 自造 140x14 组 | A | 2.6e-15 | 2.8e-15 |
| 32 | `ivregress 2sls yiv x1 (e1=z1 z2)` | IV2SLS('yiv ~ x1 + [e1 ~ z1 + z2]') | 自造 140x14 组 | A | 2.6e-15 | 3.2e-15 |
| 33 | `ivregress 2sls ..., vce(robust)` | IV2SLS(...).fit(cov_type='HC0') | 自造 140x14 组 | A | 2.6e-15 | 6.1e-15 |
| 34 | `同上` | IV2SLS(...).fit(cov_type='HC1') | 自造 140x14 组 | E | 2.6e-15 | 1.1e-02 |
| 35 | `ivregress 2sls ..., vce(robust) small` | IV2SLS(...).fit(cov_type='HC1') | 自造 140x14 组 | A | 2.6e-15 | 6.0e-15 |
| 36 | `ivregress 2sls ..., vce(cluster gid)` | IV2SLS(...).fit(cov_type='cluster', {'clusters': gid}) | 自造 140x14 组 | A | 2.6e-15 | 6.1e-15 |
| 37 | `ivregress liml yiv x1 (e1=z1)` | ws.IVLIML.from_formula('yiv ~ x1 + [e1 ~ z1]') | 自造 140x14 组 | A | 2.4e-15 | 2.0e-15 |
| 38 | `ivregress liml yiv x1 (e1=z1 z2)` | IVLIML('yiv ~ x1 + [e1 ~ z1 + z2]') | 自造 140x14 组 | A | 1.5e-15 | 7.2e-16 |
| 39 | `ivregress liml ..., vce(robust)` | IVLIML(...).fit(cov_type='HC0') | 自造 140x14 组 | A | 1.5e-15 | 7.3e-16 |
| 40 | `同上` | IVLIML(...).fit(cov_type='HC1') | 自造 140x14 组 | E | 1.5e-15 | 1.1e-02 |
| 41 | `ivregress liml ..., vce(cluster gid)` | IVLIML(...).fit(cov_type='cluster', {'clusters': gid}) | 自造 140x14 组 | E | 1.5e-15 | 4.5e-02 |
| 42 | `ivregress liml ..., vce(cluster gid) small` | 同上（Stata 侧加 small） | 自造 140x14 组 | A | 1.5e-15 | 8.7e-16 |
| 43 | `ivregress gmm yiv x1 (e1=z1 z2)` | ws.IVGMM.from_formula('yiv ~ x1 + [e1 ~ z1 + z2]') | 自造 140x14 组 | A | 2.2e-16 | 3.7e-16 |
| 44 | `同上` | 同上，比 V 全矩阵 | 自造 140x14 组 | A | 2.2e-16 | 3.7e-16 |
| 45 | `ivregress gmm ..., vce(robust)` | IVGMM(...).fit(cov_type='HC0') | 自造 140x14 组 | A | 2.2e-16 | 8.1e-16 |
| 46 | `同上` | IVGMM(...).fit(cov_type='HC1') | 自造 140x14 组 | E | 2.2e-16 | 1.1e-02 |
| 47 | `ivregress gmm ..., vce(cluster gid)` | IVGMM(...).fit(cov_type='cluster', {'clusters': gid}) | 自造 140x14 组 | E | 3.6e-02 | 1.2e-01 |
| 48 | `eivreg y x1 x2, reliab(x1 .9 x2 .95)` | ws.Eivreg.from_formula(reliab={'x1': .9, 'x2': .95}) | 自造 140x14 组 | A | 4.5e-16 | 6.4e-16 |
| 49 | `regress mpg weight foreign` | OLS(...).fit() | auto.dta | A | 5.1e-15 | 6.8e-16 |
| 50 | `同上` | resid / fitted 反算 | auto.dta | A | 5.1e-15 | 6.8e-16 |
| 51 | `regress y x1 x2 [fweight=w]` | OLS(weights=w, weight_type='fweight') | 自造 140x14 组 | A | 4.9e-16 | 3.0e-16 |
| 52 | `regress mpg weight foreign` | 比较 e() 键集合 | auto.dta | 覆盖度 fail | n/a | n/a |
| 53 | `regress y x1 x2 [fweight=w]` | 比较 e() 键集合 | 自造 140x14 组 | 覆盖度 fail | n/a | n/a |
| 54 | `regress y x1 x2, vce(cluster gid)` | 比较 e() 键集合 | 自造 140x14 组 | 覆盖度 fail | n/a | n/a |
| 55 | `ivregress 2sls yiv x1 (e1=z1 z2)` | 比较 e() 键集合 | 自造 140x14 组 | 覆盖度 fail | n/a | n/a |
| 56 | `ivregress gmm yiv x1 (e1=z1 z2)` | 比较 e() 键集合 | 自造 140x14 组 | 覆盖度 fail | n/a | n/a |
| 57 | `eivreg y x1 x2, reliab(...)` | 比较 e() 键集合 | 自造 140x14 组 | 覆盖度 pass | n/a | n/a |
| 58 | `areg y x1 x2, absorb(uid)` | （不比对） | 自造 140x14 组 | skip | n/a | n/a |
| 59 | `eivreg ..., reliab(x1 .5)` | （不比对） | 自造 140x14 组 | skip | n/a | n/a |
| 60 | `ivregress 2sls ... [pw=w]` | （不比对） | 自造 140x14 组 | skip | n/a | n/a |
| 61 | `which reghdfe` | which reghdfe | — | skip | n/a | n/a |
| 62 | `regress y x1 x2` | 把 westata 的 e(V) 代入 b'V^-1b/k | 自造 140x14 组 | 证据 | n/a | n/a |
| 63 | `regress y x1 x2, vce(robust)` | 同上 | 自造 140x14 组 | 证据 | n/a | n/a |
| 64 | `regress y x1 x2, vce(cluster gid)` | 同上 | 自造 140x14 组 | 证据 | n/a | n/a |
| 65 | `regress y x1 x2 [pweight=w]` | 同上 | 自造 140x14 组 | 证据 | n/a | n/a |