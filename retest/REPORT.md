# westata vs StataNow 19.5 (MP) 独立复测报告

**复测日期**：2026-10-24
**被测对象**：本仓库 `westata 0.0.1`（`D:\myCodeProject\statapy\src\westata`，editable 安装）
**基准**：本机 `D:\Program Files\StataNow19`，内嵌 `pystata`，实测 `c(stata_version) = 19.5`，edition `mp`
**结论口径**：全部数字取自 Stata 的 `e(b)/e(V)/e()` 与 `r()` 全精度矩阵，**不使用**打印值的四舍五入。

---

## 0. 这次复测和以前不一样的地方

按你的要求，**旧测试全部作废、从零开始**。具体做法：

1. **不读旧证据**。`tests/conformance/stata_fixtures/`、`tests/conformance/registry.yml`、
   `conformance*.csv`、`docs/GAP_REGISTER.md` 一律只读、**不作为结论依据**——它们本身是被检验对象。
   每个领域组都被明确要求：不得引用旧结论，必须自己起 Stata 会话重跑。
2. **重写对比工具**。新建 `retest/lib/stata_probe.py`（588 行），不依赖仓库里任何既有 conformance 代码。
   它负责：起内嵌 Stata、捕获 `e(b)/e(V)/e()/r()/matrix`、把缺失值 `8.988e+307` 转 `NaN`、
   显式做因子变量与常数项的名字对齐、按档位判定。
3. **七个领域并行、互斥写作用域**。每个子智能体只写 `retest/domains/<领域>/`，
   `src/`、`tests/`、`scripts/`、`retest/lib/` 零改动（已用 `git status` 核实：只有 `?? retest/` 和 `?? .agent-teams/`）。
4. **两层独立复核**。第一层：Lead 另写 6 个脚本，
   独立复核了 ologit 的 cut 协方差、`xtreg,fe vce(robust)` 与 `vce(cluster id)` 的关系、
   IV 的 HC0/HC1 映射、Poisson 的 HC1 与 `e(chi2)`、`nl` 的 `pinv` 截断、tobit 的 σ 尺度。
   全部**复现了领域组的结论**。第二层：一个专职 verifier **不复用共享工具、不 import 领域脚本**，
   自建 Stata 桥 + 51 个探针独立重跑，**找出 3 处归因错误和 1 个漏掉的独立缺陷**，
   这些已在报告里被改写（见 §5.1）。

判定档位（全报告统一）：

| 档位 | 相对差上限 | 含义 |
|---|---|---|
| **A** | ≤ 1e-9 | 机器精度，可以声称"与 Stata 完全一致" |
| **B** | ≤ 1e-6 | 数值实现差异（收敛、算法），系数第 7 位起不同 |
| **C** | ≤ 1e-3 | 可见差异，第 4 位有效数字起不同 |
| **D** | ≤ 1e-2 | 明显不同，需要解释 |
| **E** | > 1e-2 | 结论可能不同，必须归因 |

相对差 = `max(|westata − stata| / max(|stata|, 1e-12))`，逐系数取最大。
下面说的"系数"“标准误"都指这个量。

---

## 1. 总体结果

| 领域 | 对比点 | pass | fail | error | skip | **A 档（整条通过）** | 最大系数相对差 |
|---|---|---|---|---|---|---|---|
| linear_iv | 65 | 46 | 15 | 0 | 4 | **40** | 3.58e-2 |
| panel | 42 | 22 | 18 | 1 | 1 | **22** | 2.12e-1 |
| discrete | 70 | 15 | 39 | 13 | 3 | **10** | 2.09e+1 |
| limited_quantile | 25 | 0 | 22 | 3 | 0 | **0** | 5.43e-2 |
| tseries | 56 | 35 | 21 | 0 | 0 | **20** | 1.71e-2 |
| systems_causal | 56 | 36 | 16 | 2 | 2 | **33** | 1.42e+1 |
| postest | 77 | 45 | 32 | 0 | 0 | **43** | 7.36e-1 |
| **合计** | **391** | **199** | **163** | **19** | **10** | **168** | — |

**A 档 168 条 = 43.0%**（分母 391）。这里的"A 档"是**严格口径**：整条 pass **且** 系数与 SE 都 ≤1e-9。

> **两个容易混淆的计数，先讲清楚**：
> - 按**档位**统计，有 **194 条**的系数/SE 落在 A 档；但其中 26 条因为**专项比对的 `e()` 标量或语义**
>   被判 fail（例如 `regress vce(robust)` 的系数/SE 是 5.1e-15，但 `e(F)` 语义错 → 整条 fail）。
>   这 26 条列在本报告 §2.1 与 §3 的"语义/标签"清单里，**它们不是数值不一致，是含义不一致**。
> - "pass 199 / fail 163"这个比例会误导：很多 fail 是**同一个缺陷被多个变体重复计入**
>   （`ologit` 的 `e(V)` 一个转换错误，连带 cut 的 SE、z、CI、`e(df_m)`、`e(chi2)` 全部 fail）。
> 真正的**独立缺陷**数量见 §3，共 **24 个**。

> **证据可信度分级**（verifier 复核后）：linear_iv / panel / systems_causal / postest = **高**（可直接引用）；discrete / tseries = **中高**（数值可复现，但 ologit 与 vecrank 的**归因/口径已按复核结果改写**）；limited_quantile = **中**（tobit/heckman/etregress/rreg 成立，qreg 的归因已改写）。

**一句话结论**：westata 在**线性、面板线性、多方程、复杂抽样、单位根**这些闭式可解的部分
做到了真正的机器精度（1e-15 量级），这部分可以作为 Stata 的可信替代；
但在**非线性 ML 的二阶量（协方差）、某些 VCE 的语义映射、以及派生 `e()` 标量**上，
存在 24 个可复现的独立缺陷，其中 8 个会直接导致"显著性判断不同"。

---

## 2. 各领域结论

### 2.1 linear_iv（`regress` / `areg` / `reghdfe` / IV）— 最扎实的一块

**整条 A 档 40 条**（另有 9 条系数/SE 也是 A 档、但`e()` 语义 fail，见下）。可以声称完全一致的：

- `regress` 基础、`i.` 因子（含缺失值 listwise 删除行为）、`vce(robust)`、`vce(cluster)`
  （含 `t(G-1)` 的 p 值）、四种权重（`fweight`/`aweight`/`pweight`/`iweight`，含加权 meat）、
  `if` 子样本 —— 全部 ≤ 1.1e-15。
- `areg`（分类 FE、数值型的"连续"FE、robust、嵌套/非嵌套 cluster）、
  `reghdfe`（单维、双向、robust、cluster）、`FEOLS` 双向 —— ≤ 1.1e-15。
  **本机 `reghdfe` 6.14.1 已装**，所以这一块是与真 `reghdfe` 对拍，不是用 `areg` 替代。
- `ivregress 2sls`（恰好识别 / 过度识别 / robust / cluster）、
  `ivregress liml`、`ivregress gmm`（**两步 GMM 的 `e(V)` 全 9 元素 4.4e-15**）、
  `eivreg` —— ≤ 6.1e-15。

**不够 A 的，只有 1 条是真缺陷**：

| 缺陷 | 量级 | 归因 |
|---|---|---|
| `ivregress gmm, vce(cluster)` **点估计**差 | 3.6e-2（SE 1.2e-1） | Stata 在该 VCE 下把 GMM 权重矩阵 `W` 也聚类，westata 固定用 heteroskedastic `W`。用 `W=cluster` 手算两步 GMM 可复现 Stata 到 1.5e-15 → **估计量选择缺失，不是数值错误** |

另有 3 条**语义/标签**问题（数值对、含义错，最容易误导下游）：

- `ivregress ... vce(robust)` 的 HC 映射：Stata 默认是 **HC0**（westata `HC0` 差 6.1e-15），
  加 `small` 才是 HC1（westata `HC1` 差 6.0e-15）。但 `iv.py:586/624` 把 HC1 写成 `vce(robust)` 的对应项
  → 按文档选 HC1 会系统性偏大 `sqrt(N/(N-K))` = 1.0109。
- `IVLIML` 的 `vce(cluster)` **恒加 CR1**，Stata 默认 CR0 → SE 差 4.53e-2（比值 1.045296408534764 与公式逐位吻合）。
- `OLS.e['F']` **不随 `cov_type` 变**（恒为经典 R² 型 F），robust 差 5.6e-2、cluster 差 **37.5%**；
  `OLS.e['df_r']` 在 cluster 下存 `N-K=137`，而 Stata 存 `G-1=13`（**但 `results.df_resid` 和 p 值是对的**）。

> **防误判提示**：`areg ... vce(cluster gid)` 与 `reghdfe`/`FEOLS` 同规格差 4.86e-2。
> 这**不是 westata 的错**——areg 把吸收的 13 个虚拟变量计入 `K`（K=16），reghdfe 按
> `dofadjustments(nested)` 记 K=3。闭式复算三方都对得上（1e-15）。谁拿 areg 的 SE 当基准
> 去测 FEOLS，都会误判出一个 5% 的假 bug。

### 2.2 panel（`xtreg` 族 / xt 非线性 / 单位根）— 线性部分极准，vce 与非线性二阶量有问题

**整条 A 档 22 条**：`xtreg, fe`（平衡/非平衡，`e(V)` 全矩阵 1.07e-15）、`xtreg, re`
（含 theta→0 边界，全矩阵 2.36e-15）、`xtreg, be`、`xtreg, fd` 的等价口径、
`xtgls` heteroskedastic 与 correlated（满秩）、`xtpcse`（平衡）、`xtivreg fe/re`、
`xtunitroot ips/llc`（有/无单位根两组，p 值判决一致）。

其中两个细节值得单说：`xtreg, re` 的 `e(V)` 全矩阵 2.36e-15 说明 westata 的 Swamy-Arora
与 Stata 的 GLS theta 是**同一个东西**；`xtivreg, fe` 的常数项 8e-15 说明它正确复现了
Stata 的 `w − w̄_i + w̄` 变换（常被写错成"within 变换会消灭常数"）。

**4 个真缺陷**：

| 缺陷 | 量级 | 归因（已验证） |
|---|---|---|
| `xtreg, fe vce(cluster id)` | SE **9.36e-2** | 强制 `linearmodels auto_df=False`，把小样本因子的 `K` 算成 32（含 29 个被吸收效应）；Stata 用 K=3。改 `auto_df=True` 或手工 `G/(G-1)·(N-1)/(N-K), K=3` 即逐位相同（9.4e-15）。**源码注释说 auto_df=True 会漏效应，实测相反** |
| `xtreg, fe vce(robust)` | 常数项 SE **18.36 倍** | Stata 的 `xtreg,fe vce(robust)` **等价于 `vce(cluster panelvar)`**——Lead 独立验证 `e(V)` max abs diff = **0**。westata 映射成 HC1。HC1 实现本身没错（与 `areg absorb(id) vce(robust)` 逐位相同），错在语义映射 |
| `xtpcse` 非平衡（casewise / pairwise） | SE **0.281 / 0.419** | `_aligned_slices` 按行号取每个面板前 k 行，Stata 按"所有面板共同期"估 Σ、meat 用全部观测；**pairwise 分支根本未实现**（与 casewise 返回逐位相同）——静默的同名不同量 |
| `xtabond` | `e(sig2)` **差 1.993 倍**；`lags(2)` 参数名**整体错位一位** | 差分误差方差是 2σ²，未除 2；`lags(2)` 时 `param_names=['L2.y','x1','x2','extra_0']`，真值是 `[L1.y,L2.y,x1,x2]`——**按名字取系数会静默取错变量** |

`xtnbreg, fe` 是另一类问题：westata 与它自己的 `xtpoisson, fe` **逐位相同**，说明它只实现了
条件 Poisson OIM，没做 HHG 负二项（系数差 1.36e-3、SE 差 6.79e-2）。

### 2.3 discrete（logit/probit/计数/有序）— 问题最集中

**整条 A 档只有 10 条**，且集中在最基础的地方：`logit` 默认（1.09e-13/3.56e-14）、`logit, or`
（OR 与 SE）、`logit/probit/poisson/ztp` 的 `e()` 标量、`logit` 子样本、`logit predict,pr/xb`。

**7 个真缺陷**：

| 缺陷 | 量级 | 归因 |
|---|---|---|
| `ologit`/`oprobit` 的 **`e(V)` 全矩阵** | **2.09e+01 / 1.05e+01** | **两个独立缺陷**（verifier 插桩确认，见 §5）：① `discrete.py:386-398` 的 **numpy 视图别名**——`theta = params[n_slopes:]` 是视图，随后 `params[n_slopes:] = cuts` 把 `theta` 也改成 cut 点，于是 `exp()` 作用在 **cut 点**上而不是 log-gap 上；② **slope×cut 交叉块 `cov[:ns, ns:]` 从未变换**（`cov_in` 与 `cov_out` 逐位相同）。`cut1` 的 SE 碰巧对，`cut2` +20.5%，**`cut3` +176%**；点估计本身是对的。只修 Jacobian 而不修交叉块，`e(V)` 只从 3.944 降到 3.269e-05 的「一半」——两块都补才到 Stata 的 0.0874/0.0962 |
| `vce(robust)` 对 **Logit/Probit/NegativeBinomial 全崩** | 异常 | `discrete.py:178` 对 HC0 **和** HC1 都传 `cov_kwds`，statsmodels 直接报 `ValueError`。只有 Poisson 幸免 |
| `Poisson.fit(cov_type='HC1')` **静默等于 HC0** | — | `array_equal=True`，只有标签变。无任何路径产 Stata 的 `N/(N-1)` 因子 |
| `e(chi2)` 在 `vce(robust)/vce(cluster)` 下**没从 LR 换成 Wald** | 37.2% | Stata 的 `chi2type` 从 LR 变 Wald，且 Wald 只含斜率块（实测五个命令 `b'V⁻¹b` 斜率块全部复现 `e(chi2)` 到 1e-16）。westata 的 `e(chi2)` 对 `cov_type` 完全不敏感 |
| `e(df_m)` 把辅助参数算进去 | 100% | nbreg 报 3（Stata 2）、ologit/oprobit 报 4（Stata 2）。**tnbreg 反而对**（走另一条路径），同一似然两条路径两个答案 |
| `zip/zinb` 的 `e(p)` 自由度 | 1.15e+2 | Stata 用 `e(rank)-1`，westata 用 `e(df_m)`。logit/probit/poisson/nbreg/ologit 用 df_m 是对的，**这是多方程专有约定** |
| `zip predict,n` 崩溃 / `ologit predict,pr` 崩溃 / `ztp predict,n` 语义错 | — | zip 的分发器走二值分支导致 `matmul` 维度不匹配（模型层 `fittedvalues` 其实对到 1.1e-9）；ologit 因 `fit()` 从不传 `design_matrix` 抛 `ConsistencyError`；ztp 返回**截断均值** `mu/Pr(y>0)`，Stata 返回 `mu` |

`nbreg` 的 `/lnalpha` 是历史难点，**本次实测是对的**：alpha 与 lnalpha 同级相对差（1.82e-4），
SE 只差 4.59e-6，**不存在单位错配**。它的 B/C 档差异来自 `bfgs` 默认 `gtol=1e-5`（收到 1e-10 后
nbreg 1.82e-4→1.50e-8）。

### 2.4 limited_quantile（tobit/truncreg/intreg/heckman/etregress/frontier/qreg/rreg）— 没有一条到 A 档

这是唯一**全场 0 条 A 档**的领域。原因是失败的几乎全在 SE 与派生标量上，而系数多在 B 档以内。

（严格说 `etregress twostep` 的**系数**在 A 档 3.84e-11，但 SE 在 E 档 6.77e-2，整条仍判 fail。）

**5 个真缺陷**：

| 缺陷 | 量级 | 归因（三条独立证据） |
|---|---|---|
| `tobit` 的 **`/sigma` 的标准误** | 5.4%（ll）、1.6%（ul）、6.6%（双边） | westata 报的是 **se(lnσ)**：`cov` 用 log 尺度参数化求 Hessian，换回 σ 时**没做 Jacobian 修正**。证据：① 对独立中心差分 Hessian，与 `(b,lnσ)` 参数化差 **0.00e+00**，与 `(b,σ)` 差 5.43e-2；② 与 Stata 的 `se(lnsigma)` 比差 2.1e-6；③ `westata/stata_se(σ) = 0.945719091 ≈ 1/σ̂ = 0.945721083`。后果：σ 的 z 被放大 1/σ |
| `etregress` 的 **`e(rho)` 符号错** | 反号 | westata 用 `rho = −b_ctrl/σ`，但 Stata 的 `e(lambda)=b_ctrl=ρσ`，正确是 `+b_ctrl/σ`。两个独立 DGP 都是互为相反数 |
| `heckman` 的 **`e(sigma)`/`e(rho)` 定义错** | 6.7% / 7.2% | Stata 的 `e(sigma)=sqrt(RSS/m + b_λ²·mean(λ(λ+w)))`，复现到 1.8e-10；westata 用 `sqrt(RSS/(m−3))`。**λ 系数本身两边一致到 1e-9**，只是派生量算错 |
| `qreg, vce(robust)` 在 **τ≠.5** | 3.5–4.6%（同方差 DGP）/ **最高 128%**（异方差 DGP） | westata 的 meat 用 `Σψ_i²xx'`，Stata 用 **`τ(1−τ)X'X`**；在领域组的数据上换 meat 后 τ=.25 复现 9.0e-9、τ=.75 1.4e-8。**但 verifier 在自己的异方差 DGP 上复测，换 meat 后仍差 15%–147%**——说明 **bread（逐点密度 `f_i`）也参与**，这不是一条「换个 meat 就修好」的公式。同时「偏高 3.5–4.6%」低估了量级（异方差 τ=.75 实测偏高 **128%**） |
| `rreg` 的 **SE** | 15–30% | Stata **不用 pseudovalue**，而是 `se(b)=e(rmse)·sqrt(diag((X'X)⁻¹))`，四个数据集复现到 1.9e-16。westata 的 STR 伪值实现本身没错（独立重实现复现了它的三个数），**错的是方差定义** |

`frontier` halfnormal 是 D 档，归因是数值 Hessian 步长（`1e-5` 太小，改 `1e-3` 后差降到 0.2%）；
`truncreg`/`intreg` 在 C 档，**旧结论站得住**。

**3 个 error**：`heckman, ml`、`etregress, ml`、`frontier, distribution(exponential)` 都抛
`UnsupportedError`。注意 **Stata 的 `etregress` 默认就是 ML**，westata 唯一可用的 control-function
路径**不是** Stata 的默认——这一点文档必须写明。

### 2.5 tseries（ARIMA / VAR / VECM / 单位根 / GARCH）

**整条 A 档 20 条**：VAR(1)/VAR(2) 系数、`vec` 三组（alpha/beta/ll）、`dfuller` 五组（统计量与 MacKinnon p）、
`pperron` 四组。

**最值得上报的一条是"结论级"的风险：`vecrank` 的临界值表与 Stata 不同。** Stata 用 Johansen(1995) 表
（r=1 时临界值 **3.76**），statsmodels 用 Osterwald-Lenum（**3.8415**）。领域组造出一组
`trace(r=1)=3.8369` 的边界数据（正好卡在两者之间），据此判断**选阶会翻（Stata 选 rank 2，westata 选 rank 1）**。
verifier 独立确认了**两侧的临界值表确实不同**（Stata 打印 3.76 vs `VECM.johansen()` 返回 3.8415），
但**没能复现出落进 (3.76, 3.8415) 区间的数据**，所以翻转本身属于"机制清楚、未独立复现"。
报告的措辞按后者：**临界值表不同，边界处会翻转，但不是每条数据都翻**。

其余不达 A 的，归因都做得很干净：

- **ARIMA**（C/D，系数 2e-5 ~ 5.5e-3）：但 `ll/aic/bic` 全在 1e-10 ~ 1.5e-9。用 statsmodels 自己的
  loglike 在 **Stata 参数点**求值 = Stata 的 `e(ll)` 逐位相同 → **同一似然面上的停机点差异**，
  最大差恰在最平坦的常数/漂移项。不影响显著性。
- **VAR 的 SE / `e(Sigma)` / Granger**（E 档）：全部是**每方程**自由度因子 `(T−k_eq)/T`
  （k_eq 是每方程回归元数，不是全系统 k——用全 k 会得到 0.9~9.6 的假差异）。归一后 2e-15。
  Granger 另有一层：`granger_causality()` 默认返回 statsmodels `kind="f"` 的 F 统计量，
  Stata `vargranger` 是 **Wald chi2**。
- **`dfuller` 打印临界值**（D）：统计量与 p 值全 A 档；临界值差是 Stata 硬编码 6 点表线性插值，
  照抄后可复算到 1.28e-16。不影响结论。
- **`arch`/GARCH**（E，参数 1.07e-2、SE 2.78e-1）：两个独立成因都量化了——
  ① SE：Stata 是 OPG，westata 继承 arch 包的默认 robust（实测 westata 的 SE 与 arch `cov_type="robust"` **逐位相同**）；
  ② **presample h₀ 不同导致目标函数不同**（交叉验证：Stata 的 h₀ 下 Stata 参数点更优，arch backcast 下 westata 参数点更优
  ——各自在各自目标下更优 ⇒ 不是停机点差异）。

**三个"陷阱"提醒**（对任何后续对比都有用）：

1. **Stata 的 `var, lags(2)` 不是 VAR(2)**——`lags()` 收 numlist，`lags(2)` 只用第 2 阶滞后，
   `e(b)` 只有 6 个系数 vs westata 10 个。**参数个数都不一样**。必须写 `lags(1/2)`。
2. `arima`/`arch` **不存 `e(aic)/e(bic)`**，须 `estat ic` → `r(S)`；`e(N)` 在 `d=1` 时 Stata=299 / westata=300。
3. **`kpss` 不在 StataNow 19.5 里**——本领域 `ssc install` 了第三方包 v1.2.3 (Baum)，
   报告里必须标明**不是官方命令**，且其 H0 与 dfuller/pperron 相反。

### 2.6 systems_causal（多方程 / 处理效应 / DID / svy / 重抽样）

**整条 A 档 33 条**（比例 59%，七个领域最高）：

- `sureg` 2 方程：系数 5.75e-13、SE 6.09e-15、**`e(V)` 全 6×6 矩阵 1.76e-12**（跨方程协方差块全对）
- `mvreg`：7.18e-15 / 9.12e-14 / `e(V)` 3.61e-12
- `manova` 三种 (p,q)：≤7.6e-14，含 `df2=228.9219` 的分数自由度（只有正确 Rao 变换才能得到）
- `teffects ra/ipw/aipw` 的 **ATE**、`didregress`/`xtdidregress`（含 5 期交错、两种拼写自由度比值 1.81e-16）
- **`svy` 全系列 9 条全 A 档**：mean/total/proportion/regress/logit/self-weight/strata/fpc，
  多数 SE 逐位相同，`e(N)/N_psu/N_strata/N_pop/df_r` 全等
- **`bootstrap` 同种子逐次复现到 1.41e-15**（100 reps × 2 系数 = 200 个数）

**4 个真缺陷**：

| 缺陷 | 量级 | 归因 |
|---|---|---|
| `nl` 二次式的 **SE** | **相对差 = 1.0**（差 7 个数量级） | `sigma2 * pinv(J'J)` 的默认截断：`J'J` 条件数 6.13e9，最小/最大奇异值 = 3.91e-17 < `max(shape)*eps = 6.66e-16`，最小特征方向被静默丢弃。`inv` 和 `pinv(rcond=0)` 都给 Stata 的数。**系数/R²/RSS 全对**（Lead 已独立复现） |
| `nl` 默认初值 | 1.42e+01 | 初值全 1，LM 停在离解很远的点（RSS 3.42e9 vs 4.54e8），**只发 RuntimeWarning 仍返回结果** |
| `teffects aipw ..., atet` | ATET 6.3e-3、SE 2.93e-2 | ATET 分支的增广项结构错（`augmented0` 带 (1−t) 使控制组增广项消失，`augmented1` 多带 1/p）。**ATE 分支精确，缺陷只在 `estimand="atet"`** |
| `teffects nnmatch` | SE 5.09% / 4.53% / 16.3% | 自标 Abadie-Imbens 近似；**新发现独立缺陷：ATET 的 SE 与 ATE 的 SE 逐位相同**（`share0=share1=1`，ATET 方差未单独估计） |

另有 3 条**语义问题**：`didregress` 的 `covariates=None` 会**自动把其他时变数值列当协变量**
（Stata 的 `(y)` 不会，且 westata 无法显式关掉——`covariates=[]` 抛错）；
`jackknife` 用 `df_r=72` 而非 Stata 的 73；`bootstrap` 用 **t(72) 而 Stata 用 z**（区间系统性偏宽 1.7%）。

**一条推翻旧归因的发现**：`jackknife` 的 SE 与 Stata 差 ~5e-7，源码注释归因于
"Stata 用 Sherman-Morrison 解析更新"。用 `Fraction` 精确有理数求真值后：
**westata 1.15e-15 命中真值，Stata 偏 5.11e-7**；把 replicates 过 float32 能复现 Stata 到 2.2e-16
→ 真因是 **Stata 的单精度存储**，westata 是**更准**的那个。

### 2.7 postest（test / lincom / nlcom / lrtest / hausman / contrast / margins / predict / etable）

**整条 A 档 43 条**：`test` 8 个变体（含 F 形式与 logit 的 Wald chi2 形式）、`lincom` 4 个、
`nlcom` 3 个、`testnl` 2 个、`lrtest` 2 个、`hausman`、`contrast` 6 个、`pwcompare`、
regress 的 `predict xb/stdp/stdf/residuals`、logit 的 `pr/xb/stdp`、`etable` vs `esttab` 20 格全 0
（`test`/`margins` 这类都同时比了**统计量、df、p 值**三项）。

**5 个真缺陷**（其中 4 个与 tobit 相关）：

| 缺陷 | 量级 | 归因 |
|---|---|---|
| `margins` 后 logit 的 **AME 标准误** | 7.1e-3（全样本）/ **5.4e-2**（`at(mpg=20)`） | `margins.py::_statistic_and_gradient` 用**固定 h=1e-6 中心差分**而非解析 p(1−p)。步长扫描：h=1e-3→4.9e-6、1e-6→7.1e-3、1e-8→**0.70**——误差随 h 减小反而变大，典型相消 |
| `tobit` 的 `/sigma` SE | 7.36e-1 | 与 §2.4 同一个根因（log 尺度 SE），这里量级更大因为分母接近 0 |
| `tobit` 的 `predict, stdf` | 7.7e-2 | 用 `rss/N` 代替 `sigma²`（`stdf²−stdp²=12.127` vs `sigma²=14.298`）——tobit 只记 `e(sigma)` 不记 `e(rmse)`，fallback 取到删失模型的 rss/N |
| `tobit` 的 `e(df_r)` | 1.4e-2 | 71 vs Stata 72：Stata 的 df_m 只数斜率、σ 不耗自由度；westata 多扣常数项 |
| `tobit` 设计系数 SE | 8.9e-5 | 数值 Hessian(step=1e-5) vs Stata 解析 observed information |

**行为/能力差异**（不是算错，但会打断移植）：`lincom` 拒绝裸常数项（Stata 返回 14.94）；
`testnl` 的逗号/空格联合写法 **westata 接受而 StataNow 19.5 报 r(198)**；
`lrtest` 的 `df=0` westata 返回 p=1.0 而 Stata 硬报错 r(498)，非嵌套 westata 拒绝而 Stata 接受并返回**负 LR**；
`hausman` 非正定差时 westata 夹到 0 而 Stata 报 −2.35；tobit 无 `predict, pr(a,b)`。

**一条已验证的好消息**：logit/poisson 的 `predict, stdf` —— Stata `r(198)`、westata `UnsupportedError`，
**双方一致拒绝**。这个"拒绝"是对齐的。

---

## 3. 必须修的缺陷清单（按严重度）

### P0 — 会给出错误结论，必须先修

| # | 缺陷 | 位置 | 量级 | 影响 |
|---|---|---|---|---|
| 1 | `nl` 协方差用默认 `pinv` 截断 | `nonlinear.py` | SE 差 7 个数量级 | t 值虚高，显著性判断完全错 |
| 2 | `ologit/oprobit`：① `theta` 视图别名使 `exp()` 作用在 cut 点上；② slope×cut 交叉块未变换 | `discrete.py:386-398` `_to_cut_points` | `e(V)` 差 20 倍（只修 ① 仍差 3.3e-05 量级的换位错误） | 所有 cut 的 z/CI、以及任何混合斜率与 cut 的 `lincom/test` 全错 |
| 3 | `tobit` 的 `/sigma` SE 未做 delta 变换 | `tobit.py` | 5.4–66% | σ 的 z 被放大 1/σ |
| 4 | `xtreg, fe vce(robust)` 映射成 HC1 | `panel.py` | 常数项 18.36 倍 | Stata 语义是 `vce(cluster panelvar)` |
| 5 | `xtreg, fe vce(cluster)` 的 `K` 计数 | `panel.py::_extra_fit_kwargs` | 9.36% | 聚类 SE 系统性偏大 |
| 6 | `xtpcse` 非平衡的 Σ/meat 截断 + pairwise 未实现 | `panel.py::_aligned_slices` | 28% / 42% | 静默返回同名不同量 |
| 7 | `margins` 用固定步长数值微分 | `margins.py::_statistic_and_gradient` | 5.4e-2 | AME 的 SE/z/p 全错 |
| 8 | `teffects aipw` 的 ATET 分支 | `treatreg_effects.py` | SE 2.9e-2 | ATE 对、ATET 错 |
| 9 | `discrete` 家族 `vce(robust)` 全崩 | `discrete.py:178` | 异常 | Logit/Probit/NB 的稳健推断完全不可用 |
| 10 | `Poisson` 的 `HC1` 静默等于 `HC0` | `discrete.py` | — | 接受参数但忽略，比抛异常更危险 |
| 11 | `e(chi2)` 不随 `vce()` 切换 LR→Wald | `counts.py`/`discrete.py` | 37.2% | 报错了统计量 |
| 12 | `xtabond` 的 `e(sig2)` 未除 2 + `lags≥2` 参数名错位 | `panel_dynamic.py` | 2 倍 / 错位 | 按名字取系数会静默取错变量 |

### P1 — 语义/标签错，不会算错数但会误导

| # | 缺陷 | 量级 |
|---|---|---|
| 13 | IV 的 `HC1` 标签与 Stata 默认（HC0）不符 | 1.09e-2 |
| 14 | `IVLIML` 的 cluster 恒加 CR1（Stata 默认 CR0） | 4.53e-2 |
| 15 | `OLS.e['F']` 不随 `cov_type` 变 | 37.5%（cluster） |
| 16 | `OLS.e['df_r']` 在 cluster 下存 N−K 而非 G−1 | — |
| 17 | `gmm, vce(cluster)` 的 W 未聚类 | 点估计 3.6e-2 |
| 18 | `e(df_m)` 把辅助参数算进去（nbreg/ologit/oprobit） | 100% |
| 19 | `zip/zinb` 的 `e(p)` 用 df_m 而非 `e(rank)-1` | 1.15e+2 |
| 20 | `heckman`/`etregress` 的 `e(sigma)`/`e(rho)` 定义错、rho 反号 | 6–7% / 反号 |
| 21 | `qreg vce(robust)` 在 τ≠.5：meat 用 ψ² 而非 τ(1−τ)X'X，且 bread 的逐点密度也偏 | 3.5–128%（随异方差程度） |
| 22 | `rreg` 的方差定义与 Stata 不同 | 15–30% |
| 23 | `bootstrap` 用 t 而非 z；`jackknife` 的 df_r | 1.7% |
| 24 | `xtnbreg, fe` 只做了条件 Poisson | SE 6.79e-2 |

### 能力缺口（`error` 19 条）

`heckman, ml`、`etregress, ml`、`frontier, distribution(exponential)`、`ologit/oprobit vce(robust)`、
`nbreg` 的 NB1 / 权重、`xtgls, corr(ar1)`、`sureg vce(robust)`、`nnmatch biasadj()`、
`zip predict,n` / `ologit predict,pr`（崩溃）、`testnl` 语法超集造成的反向不兼容。

---

## 4. 与旧结论的出入（这是本次最有价值的部分）

旧的 87 条 registry 条目里，**大部分的核心数值经独立复核成立**（`regress` 全家、`xtreg fe/re/be`、
`ivregress` 非稳健、`vec`、`dfuller`、`pperron`、`sureg`、`svy`、`test`、`contrast`、`etable` 等）。
但下面这些**被推翻或需要降级**：

| 旧结论 | 本次实测 | 判定 |
|---|---|---|
| `ivregress gmm` 稳健 SE 差 **2.9e-2**，"GMM 权重矩阵构造与 Stata 不同"（GAP_REGISTER §8） | 按 `vce(robust)` ↔ **HC0** 映射差 **8.15e-16**；GMM 两步权重矩阵构造**是对的**（`e(V)` 全矩阵 4.4e-15） | **推翻**：2.9e-2 是把 HC1 当成 Stata 默认 robust 的产物 |
| `reghdfe` 装不上、多路 HDFE 仍 E1 | 本机 **reghdfe 6.14.1 已装**；单维/双向/robust/cluster 四项全 A（≤1.1e-15） | **过期**：多路 HDFE 实测通过 |
| `xtpcse` E3 | 平衡面板 A 档；**非平衡 casewise/pairwise 是 E 档，pairwise 未实现** | **降级**：旧结论高估 |
| `xtlogit, pa` E3，"gaps 9.1e-6" | 实测 **3.19e-4**，且 westata 源码文档自己承认不匹配 | **降级**：旧结论与源码自述矛盾 |
| `xtnbreg, fe` E3，"gap 2.1e-5" | 系数 1.36e-3、**SE 6.79e-2**；westata 与自己的 `xtpoisson, fe` 逐位相同 | **降级**：旧结论只看系数 |
| `xtabond` E2，"系数约 3 位、结论一致" | 系数 2.0e-3 但 **SE 差 0.91**、`e(sig2)` 差 2 倍、`lags(2)` 参数名错位 | **降级**：漏掉最危险的两件事 |
| `tobit` E3，"SE 6.4e-7" | 那 6.4e-7 是所有系数 SE 的最大值，**恰好漏掉了 `/sigma` 的 SE**（大 5 个数量级） | **漏检** |
| `heckman` E3，"rho 是 plug-in 值，排除比较" | 排除 rho 是对的，但 **`e(sigma)` 同样错**（6.7%）；且拿 two-step SE 比 ML SE 是跨估计量比较 | **漏检** |
| `etregress`，"matches all six to 1e-9" | 只对**系数**成立（3.8e-11）；SE 6.8e-2、**`e(rho)` 反号** | **漏检** |
| `qreg, vce(robust)` E3（CAVEAT 自认只验 τ=.5） | τ≠.5 **确实是错的**（meat 用 ψ² 而非 τ(1−τ)X'X） | **确认并降级** |
| `rreg` E3，"SE 差 −2.8%/+3.4%，文档没写清留一构造" | Stata 有确定闭式 `e(rmse)·sqrt(diag((X'X)⁻¹))`，复现到 1e-16；实测差 **15–30%** | **误判**：不是文档模糊，是方差定义选错 |
| `margins` E3，"SE 5.2e-6" | logit AME 的 SE 差 **5.4e-2** | **推翻** |
| `testnl`"逗号/空格/括号三种都被接受" | 只在 westata 侧成立；StataNow 19.5 对逗号/空格报 **r(198)** | **推翻**（超集，西→东会失败） |
| `lrtest`"df=0 返回 p=1.0" | Stata 侧是**硬报错 r(498)** | **推翻** |
| `predict`"e(a,b)/ystar(a,b) 5.8e-10" | 实测 **5.78e-09**（差一个数量级），且 `tobit stdf` 缺陷未被记录 | **部分推翻** |
| `nlcom`"westata identical" | westata **精确等于解析 delta method，Stata 自己偏 2e-6** | **表述反了** |
| `jackknife`"Stata 用 Sherman-Morrison 解析更新" | 真因是 **Stata 单精度存储**；westata 更准 | **归因错** |
| `nnmatch`"SE 差 +2.2%" | 实测 5.09%/4.53%/16.3%，且 **ATET 与 ATE 共用 SE** | **低估 + 新缺陷** |
| `sqreg/bsqreg`"2e-2 是 seed 波动" | 4×500 池化后仍差 2.2%，且 bootstrap/robust 比值两边系统性不同 | **掩盖了真实方案差异** |
| `frontier` E3，"1e-4" | 系数 1e-3、SE 4e-3 | **偏乐观** |
| GARCH 归因"优化器/rescaling 不同" | rescale 已被显式关掉；真因是 cov_type + **presample h₀ → 目标函数不同** | **归因需修正** |
| `_johansen_statistics`"匹配打印位数" ⇒ 可互换 | 统计量确实匹配，但**临界值表不同会让选阶翻转** | **推论不成立** |

---

## 5. Lead 的独立抽查（防止两类错误）

子智能体的结论可能错在两头：**工具对齐 bug 被当成 westata 缺陷**，或**westata 缺陷被对齐 bug 掩盖**。
Lead 另写脚本独立复核了 6 处（都不 import 领域脚本）：

| 抽查项 | 方法 | 结果 |
|---|---|---|
| `ologit` 的 cut SE | 自己造有序数据，比 `e(b)/e(V)` | **复现**：Stata `[0.10103, 0.09685, 0.12406]` vs westata `[0.10103, 0.12763, 0.38926]`——cut2 +31.8%，cut3 **+213%**。根因由 verifier 插桩定为「视图别名 + 交叉块未变换」，我按它复算 `e(V)` 从 3.944 降到 Stata 的 0.0874/0.0962 |
| `xtreg, fe vce(robust)` vs `vce(cluster id)` | 直接比两个 `e(V)` | **复现**：`max abs diff = 0.0`——Stata 侧两者**完全相同**，westata 用 HC1 是错的 |
| IV 的 HC0/HC1 映射 | 比 `vce(robust)` 的 SE 与 westata 的 HC0/HC1 | **复现**：Stata SE `[0.12286, 0.068594, 0.088897]` = westata HC0 **逐位相同**；HC1 大 `sqrt(140/137)` |
| Poisson `HC1` 与 `e(chi2)` | `array_equal` + 读 `e('chi2')` | **复现**：`HC0==HC1` 为 `True`；Stata `chi2` 从 183.206 变 173.907，westata 恒 183.206；cluster 下 Stata 227.039 vs westata 183.206 |
| `nl` 的 `pinv` 截断 | 用 Stata 的参数点重跑 | **复现**：系数完全一致（`[51.183, -0.014158, 1.3244e-06]`），SE `[3.95e-7, 5.94e-4, 1.71e-7]` vs Stata `[5.768, 3.88e-3, 6.26e-7]`——差 6–7 个数量级 |
| `tobit` 的 σ 尺度 | 比两组参数化下的 Hessian | **复现**：westata 的 `se(/sigma)=0.06358` 与 Stata 的 `se(var)=0.12809`（→ `se(σ)=0.0805`）不符，且与领域组独立算出的 `se(lnσ)` 一致 |

抽查全部**支持**领域结论，没有发现"工具 bug 冒充缺陷"的情况。领域组也主动报告了两处**反向**发现
（`areg` 的 5.11% 不是 westata 缺陷、`jackknife` 是 westata 更准），说明他们没有一味"找茬"。

### 5.1 verifier 的独立复核（第二层）

在上述 Lead 抽查之外，还有一个专职的 verifier 角色：**不复用** `retest/lib/stata_probe.py`、
**不 import** 任何领域 `run.py`，自建 62 行的 Stata 桥 + 51 个独立探针，数据全部自己重造
（`retest/domains/verify/AUDIT.md`，444 行）。它的产出不是"再确认一遍"，而是找出了**3 处归因错误**
和**1 个被漏掉的独立缺陷**——这部分已经**改写进本报告**：

**① `ologit/oprobit` 的根因不是"Jacobian 转置"（已改写 §2.3、§3）。**
verifier 用插桩把 `_to_cut_points` 的真实输入输出抓出来，逐步复算：
`V_out == J_alias·V_θ·J_alias'` **精确成立（max|diff| = 0.000e+00）**，而
`J_code` **既不等于** `J_correct` **也不等于**它的转置（`max|J_code − J_correct^T| = 1.25`）。
真实机制是 **numpy 视图别名**：`theta = params[n_slopes:]` 是视图，随后
`params[n_slopes:] = cuts` 把 `theta` 一起改成 cut 点，于是 `exp()` 作用在 **cut 点**上而不是 log-gap 上。
**按"转置"这个描述去修会改错地方**——这是本次复核最有价值的一条。

**② 上游漏掉的第 2 个独立缺陷：slope×cut 交叉块 `cov[:ns, ns:]` 从未变换。**
`cov_in[:2,2:]` 与 `cov_out[:2,2:]` **逐位相同**。所以**光修 Jacobian 修不好**：
任何混合斜率与 cut 的 `lincom`/`test` 都会算错。两块都补之后，`e(V)` 从 **3.944 → 3.269e-05**
（oprobit 5.384e-05），cut2/cut3 的 SE 从 `0.0714/0.1773` 回到 `0.0874/0.0962`（Stata 同）。
这也证明了另一件事：**后端协方差本身是对的，错的只有转换**。

**③ `qreg vce(robust)` 的归因被过度推广（已改写 §2.4、§3）。**
在领域组自己的同方差 DGP 上，"把 meat 换成 `τ(1−τ)X'X`"确实把 τ=.25/.75 复现到 1e-8；
但 verifier 在**自己的异方差 DGP** 上换 meat 后**仍差 15%–147%**，说明 **bread（逐点密度 `f_i`）也参与**。
同时"偏高 3.5–4.6%"**低估了量级**：异方差 τ=.75 实测偏高 **128%**。
→ 不能说"换成 τ(1−τ)X'X 就好了"。

**④ 一个工具层面的陷阱，会影响任何后续复测。**
`fweight` 下**不能用 `nobs` 比样本量**：Stata 侧 `st.nobs()=140` 但 `e(N)=364`；
westata 侧 `res.nobs=140` 而 `res.e['N']=364`。本报告 BRIEF §3.4 原本写的
`assert nobs == nobs` 会对 fweight **假报"样本不同"**，必须比 `e(N)`。
（本次所有领域的加权条目都另核了 `e(N)`，所以结论不受影响。）

**⑤ 两条"上游对、但需要限定条件"的补充。**
- `xtpcse` 在**平衡面板**上其实是 A 档（1.44e-15）；"pairwise 静默等于 casewise"**只在非平衡面板**才是缺陷。
- 按位置对齐会造假失败：`xtpcse` 的 westata 列序把 `Intercept` 放前面，按位置得 9.18e-01、
  按名字得 1.44e-15——**这条正好说明为什么必须按名字对齐**。

**verifier 的 A 档抽查（12 条）全部通过，无假阳性**：`regress` / `i.factor` / cluster / fweight /
pweight / 2SLS / GMM（含全 `e(V)` 5.53e-15）/ `eivreg` / `xtreg fe` / `xtreg re` / `areg` / `logit`，
每条都核了 `e(N)`、用了 `e(V)` 全矩阵、按名对齐。

**verifier 还确认了两条"westata 没问题"的结论**：
- panel 的 `vce(robust)` ≡ `vce(cluster)` 在 **Stata 侧 `max|V_robust − V_cluster| = 0.0`**（严格 0，
  不是"接近"）——所以 westata 用 HC1 确实是错的；
- `jackknife` 的"westata 更准"成立：`Fraction` 精确真值下 westata 差 **6.4e-15**、Stata 差 5.11e-7。

**未复核项（如实记录，没当作已验证）**：`sqreg`/`bsqreg` 的 2–3%、`gmm` cluster 的点估计、
`xtnbreg`/`xtabond` 的具体数值、`lrtest`/`pwcompare` 的行为差异、`nlcom` 比值、ARIMA/GARCH 的归因。
这些在 `AUDIT.md` §6 逐条列了原因。

---

## 6. 使用建议

**可以放心用（整条 A 档，1e-15 量级）**：

- `regress` 全家（含四种权重、`i.` 因子、robust、cluster、子样本）、`areg`、`reghdfe`/`FEOLS`
- `ivregress 2sls/liml/gmm`（**除 `gmm + vce(cluster)`**）
- `xtreg fe/re/be/fd`、`xtivreg fe/re`、`xtgls`（hetero / 满秩 correlated）、`xtpcse`（平衡）
- `xtunitroot ips/llc`
- `sureg`、`mvreg`、`manova`、`svy:` 全系列、`didregress`/`xtdidregress`、`teffects ra/ipw` 的 ATE
- `test`、`lincom`、`nlcom`、`contrast`、`pwcompare`、`etable`
- VAR/VECM/dfuller/pperron、bootstrap（同种子可逐次复现）

**可以用但不要引用超过 4 位有效数字的 SE**（B/C 档）：

`logit`/`probit`/`poisson`/`ztp` 的系数、`nbreg`/`zinb`（`gtol` 收紧后可到 1e-8）、
`xtlogit fe/re`、`xtprobit re`、`xttobit`、`xtpoisson`、`qreg`、`truncreg`、`intreg`、ARIMA、VAR 的 SE

**不要用 / 必须避开**：

| 你想做的事 | 不要用 | 改用 |
|---|---|---|
| 病态设计的非线性回归 SE | `Nl` 的默认 `pinv` | 自己算 `inv(J'J)`，或等修复 |
| 有序模型的 cut 推断 | `ologit/oprobit` 的 `bse` | 只用点估计；cut 的 SE 要用正确 Jacobian **并补 slope×cut 交叉块**重算 |
| Tobit 的 σ 检验 | `Tobit` 的 `bse['/sigma']` | 除以 σ 手工修正 |
| Logit/Probit/NB 的稳健推断 | `fit(cov_type='HC0'/'HC1')` | 会抛异常；改用 cluster（也要注意 G/(G-1) 因子） |
| 面板 FE 的稳健/聚类 SE | `PanelOLS.fit(cov_type=...)` | 与 Stata 差 9–1800% |
| 非平衡面板的 `xtpcse` | `XtPcse` | **只在平衡面板用**（平衡时 A 档 1.4e-15；非平衡 SE 差 28%–42%，pairwise 未实现） |
| `nl` 不给初值 | `Nl.from_formula(...).fit()` | 显式给 `start=` |
| GARCH 的 z 统计量 | `GARCH` 的 SE | 与 Stata OPG 差 28% |
| 协整选阶 | `vecrank` | 临界值表不同（3.76 vs 3.8415），**边界样本会翻转**；靠边界的结论要用 Stata 复核 |
| `rreg` 的 SE | `RobustRegression` | 用 Stata 的闭式 |
| `teffects aipw, atet` | `AIPW(estimand='atet')` | 只用 ATE |

---

## 7. 未覆盖与局限（如实说明）

**方法学局限**（会影响结论的适用范围）：

1. **每个领域只用了 1–2 个数据集**（auto.dta + 自造固定 seed 数据）。A 档结论是"在这一组数据上逐位一致"，
   不等于"在任意数据上逐位一致"——但 A 档是**闭式线性代数**（`sureg` 的 `e(V)` 全矩阵、`xtreg re` 的 GMM theta），
   这类结果的普适性远高于迭代估计。
2. **"最大相对差"对接近 0 的参数会放大**。例如 `qreg(.25)` 的截距真值约 0.0039，
   1e-9 的绝对差就成了 5.7e-7 的相对差。报告里保留了相对差口径（BRIEF 统一定义），
   但个别 B 档条目实际上是"绝对误差在 1e-9"。
3. **`e(b)` 的按名字对齐在某些模型上不成立**：`arima`/`arch` 有两行同名 `_cons`；
   `var`/`vec` 按方程重名；`intreg`/`frontier` 有多个 `_cons`。这些领域的组改用显式声明配对，
   并在 `results.jsonl` 里存了逐参数相对差可复核。
4. **`kpss` 用的是第三方包**（Stata 19.5 未内置），其 p 值不可直接引用。
5. **随机方法**（`bootstrap`/`permute`/`sqreg`/`bsqreg`）的"逐次复现"只在 Stata 的 RNG 可复现时成立。
   `bootstrap` 同种子**可以**逐位复现（1.41e-15，因为 westata 实现了 Stata 的 MT19937-64）；
   `permute` **不能**逐次复现（Stata 的 shuffle 顺序未公开），只能比分布。

**复核不充分、不能当作已验证的**（verifier 明确列出）：`sqreg`/`bsqreg` 的 2–3% 差、
`gmm + vce(cluster)` 的点估计、`xtnbreg`/`xtabond` 的具体数值、`lrtest`/`pwcompare` 的行为差异、
`nlcom` 比值、ARIMA/GARCH 的归因——这些是**单个领域组报告过、但没有第二个人独立复算过**的。

**明确没测的**（各领域 NOTES 的"未覆盖"章节逐条列了原因，共 60+ 项），主要几类：

- 面板：`xtreg cre/mle/pa`、`xtregar`、`xtgls igls`、`corr(psar1)`、`xtnbreg re`、有洞面板的 fd
- 时间序列：EGARCH/GJR/t 分布、`archm`、`vec` 的 `rank(2)`、`vecrank` 的其它 trend、`dfgls`、3+ 方程 VAR
- 离散：`tpoisson` 一般截断、权重、`estat gof/lroc/linktest`、`mlogit`
- 受限：`tobit` 的 `margins, predict(e())`、`rreg` 的 `e(rss)` 精确公式（试了 12 种加权形式未统一）
- 后估计：`margins, expression()`/`vce(unconditional)`、`suest`、`pwcompare` 的 Holm/Šidák/Tukey
- 系统：`svy: mean, over()`（子总体估计）、`didregress vce(unadjusted)`（Stata 19.5 自己拒绝）

**这些问题无法在本次判断**，需要下一轮补测。

---

## 8. 复现方式

```bash
cd D:\myCodeProject\statapy

# 单个领域（每个脚本独立可跑，会实测 Stata）
.venv\Scripts\python.exe retest/domains/linear_iv/run.py
.venv\Scripts\python.exe retest/domains/panel/run.py
.venv\Scripts\python.exe retest/domains/discrete/run.py
.venv\Scripts\python.exe retest/domains/limited_quantile/run.py
.venv\Scripts\python.exe retest/domains/tseries/run.py
.venv\Scripts\python.exe retest/domains/systems_causal/run.py
.venv\Scripts\python.exe retest/domains/postest/run.py

# 汇总与表格
.venv\Scripts\python.exe retest/aggregate.py       # -> retest/aggregate.json
.venv\Scripts\python.exe retest/report_tables.py   # -> retest/_tables.md
.venv\Scripts\python.exe retest/failures.py --tier E   # 只看最差的
```

**前置条件**：

1. `D:\Program Files\StataNow19` 可访问且 licence 有效。
2. `.venv` 里已装 westata（editable）。
3. **`pystata` 的两个运行前提**（本次实测才发现的，不满足会直接报错）：
   - Stata 的 `utilities` 目录必须在 `sys.path` 上（`retest/lib/stata_probe.py` 已处理）；
   - **venv 的 `Scripts/` 下要有 `python310.dll` 与 `python3.dll`**。venv 默认不带这两个文件，
     Stata 的 Python 接口会因为找不到共享库而报 `Unable to find the shared library`
     （随后是 `ModuleNotFoundError: No module named 'sfi'`）。
     本次是把系统 Python 目录下的 `python310.dll` / `python3.dll` 拷到 `.venv/Scripts/` 解决的
     （已拷贝，无需重做；**换机器或重建 venv 时要重做这一步**）。

**产物结构**：

```
retest/
├── BRIEF.md                      # 七个领域共用的任务书（判定口径、纪律）
├── lib/stata_probe.py            # 唯一的对比工具（588 行，七个领域共用）
├── aggregate.py / report_tables.py / failures.py / list_a.py   # 汇总与取数
├── lead_check1..6.py             # Lead 自己的 6 个独立抽查
├── _tables.md                    # 由 report_tables.py 生成的全部 A/E 档明细
├── aggregate.json                # 全部 391 条的结构化汇总
├── domains/
│   ├── linear_iv/       {run.py, results.jsonl, summary.json, NOTES.md}
│   ├── panel/           {...}
│   ├── discrete/        {...}
│   ├── limited_quantile/{...}
│   ├── tseries/         {...}
│   ├── systems_causal/  {...}
│   ├── postest/         {...}
│   └── verify/          {AUDIT.md, my_stata.py, 51 个独立探针}
└── REPORT.md                     # 本文件
```

每个 `results.jsonl` 一行一条，含两侧全精度数字、档位、归因备注；
每个 `NOTES.md` 是人工可读的逐条说明（覆盖矩阵 + 归因 + 与旧结论出入 + 未覆盖）。

---

## 9. 一句话总结

westata 在**闭式可解的线性代数部分**（线性回归、面板线性、IV、多方程、复杂抽样、单位根）
达到了与 StataNow 19.5 **机器精度一致**，这部分可以替代 Stata；
但在**非线性 ML 的二阶量**（协方差/标准误）、**VCE 的 Stata 语义映射**、
**派生 `e()` 标量**这三类地方，存在 24 个独立缺陷，其中 12 个会直接改变显著性判断。

另外有 3 处**归因**在独立复核后被改写——不是数值错，是"为什么错"错了，
**按原归因去修会修错地方**：`ologit` 是 numpy 视图别名（不是转置）且还有一块交叉项从未变换；
`qreg robust` 换 meat 并不能普适修好（bread 也参与，异方差下偏差可达 128%）；
`vecrank` 是临界值表不同（3.76 vs 3.8415），边界会翻，但不是每条数据都翻。

旧 registry 的 87 条 E3 里，核心数值大部分经独立复核成立，
但**至少 8 条属于漏检或高估**（tobit 的 σ SE、heckman 的 σ、etregress 的 rho 符号与 SE、
qreg robust 在 τ≠.5、rreg 的 SE 定义、margins 的 AME SE、xtpcse 非平衡、xtabond 的 sig2），
**2 条的归因需要改写**（GMM 稳健 SE、jackknife 的 Sherman-Morrison），
**2 条已被环境变化推翻**（reghdfe 装不上、多路 HDFE 是 E1）。
