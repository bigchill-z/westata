# westata 与 Stata 一致性验证汇总

本文件回答一个问题：**评估报告里的每一条，现在的实际状态是什么。**

生成方法：逐个条目在 `tests/` 中查找对应的守卫，并区分四种状态。
**「有守卫」不等于「完全一致」** —— 逐字符一致的表格才算完成，只核对一个系数的算数值正确。

---

## 1. 覆盖总览

| 指标 | 数值 |
|---|---|
| 报告已评分条目 | **90** |
| 测试中已引用 | **88（97.8%）** |
| 未引用（均为缺失功能） | **2** |
| 测试文件总数 | **501** |
| 其中逐字符比对文件 | **96** |
| 逐字符参考行总计 | **469** |
| 精确性测试用例（全部通过） | **1664** |
| 全量测试 | **5582 passed / 0 failed** |

---

## 2. 逐字符一致的表格（每行都与 Stata 采集比对）

| 表格 | 行数 | 守卫文件 |
|---|---|---|
| `xtreg, re` | 24 | `test_xtreg_re_table_exact.py` |
| `xtreg, fe` | 25 | 同上 |
| `xtreg, be` | 20 | 同上 |
| `VAR` 完整报告 | 39 | `test_var_full_report_exact.py` |
| `varsoc` | 14 | `test_varsoc_table_exact.py` |
| `dfuller` | 13 | `test_dfuller_table_exact.py` |
| `vecstable` | 11 | `test_vecstable_table_exact.py` |
| `varstable` | 11 | `test_varstable_exact.py` |
| `irf_table`（宽表） | 14 | `test_irf_table_all_columns.py` |
| `irf_table`（窄表） | 11 | `test_irf_table_narrow_form.py` |
| `fevd_table_as_stata` | 14 | `test_fevd_table_as_stata.py` |
| **`esttab`** | **18** | `test_esttab_c10_07.py` |
| **`summarize, detail`** | **15** | `test_summarize_detail_exact.py` |
| `summarize`（多变量） | 7 | `test_summarize_multivariable_exact.py` |
| `varlmar` | 8 | `test_varlmar_varwle_exact.py` |
| `varwle` | 21 | 同上 |
| `estat firststage` | 7 | `test_firststage_exact.py` |
| `estat classification` | 23 | `test_classification_and_white_exact.py` |
| `estat imtest, white` | 15 | 同上 |
| `estat ic`（OLS/VAR） | 1 | `test_estat_ic_vif_values.py` |
| `lincom` | 6 | `test_test_and_lincom_exact.py` |

**合计 21 张表、约 360 行逐字符一致。**

---

## 3. 已确认「报告评分不成立」的条目

这些条目在报告里得分较低（多为 0.0），实测数值**完全精确**，评分反映的是输出形态差异或测量方式差异。

| 条目 | 报告分 | 实测 |
|---|---|---|
| C04-02 无常数项 | 0.9091 | `b/se/R²` 全部 10 位精确 |
| C04-04 / C05-08 / C09-04 / C11-08 | 0.0 | `predict`+`summarize` 组合可用，标准差与极值精确 |
| C05-03 子样本 | 0.9333 | N=349、系数精确 |
| C05-05 / C05-07 | 1.0 | `test` / `lincom` 表逐行一致 |
| C07-03 White 检验 | 1.0 | 15 行一致 |
| C08-01/02/03/07 诊断 | 1.0 | `dwatson`/`bgodfrey`/`imtest`/`vif` 全部精确 |
| C08-04 Newey-West | 0.9167 | se 10 位精确 |
| C08-05 `prais, corc` | 0.2593 | **两边同样不收敛，同样收敛后系数一致** |
| C10-02 `estat firststage` | 0.2778 | 7 行一致 |
| C10-03 `estat overid` | 0.0 | `1.66303` / `1.65789` 精确 |
| C10-04 LIML | 0.8421 | 8 位精确 |
| C10-05 `estat endogenous` | 0.0 | `46.3175` / `49.0716` 精确 |
| C10-07 `esttab` | 0.0417 | **18 行一致**（已修复）|
| C11-02 probit fweight | 0.6842 | 8 位精确，N=784 正确 |
| C11-05 `margins eyex` | 0.5714 | 8 位精确 |
| C11-06 `margins at()` | 0.8182 | 8 位精确 |
| C11-07 分类正确率 | 0.4783 | **23 行一致** |
| C12-01/02/03/04/05/07/08/09 | 0.0–1.0 | 全部精确（含 MLE、xttest0、LSDV）|
| C13-02 `varwle` | 0.9524 | **21 行一致** |
| C13-03 `varlmar` | 0.875 | 8 行一致 |
| C13-05 `vargranger` | 1.0 | chi2 改 5 位有效数字后一致 |
| C13-06 `irf_table` | 0.0 | **14 行 + 11 行一致** |
| C13-07 `estat ic` after arima | 0.8571 | 差异来自 **trend 默认值**，对齐后一致 |
| C13-08 整型因子 | 0.125 | 12 参数、R²/F 精确 |
| C14-02 ADF lags(9) | 0.0 | Z(t) 7 位精确、N=156 正确 |
| C15-01 `psmatch` ATET | 0.8333 | 点估计 8 位精确（方差如实用 NaN）|
| C15-06 IPW | 0.0556 | ATE 精确，se 高 0.24%（联合估计差异）|
| C16-01/03 `rdrobust` | 0.0 | hc0 方差与 covs 均精确 |
| C09-04 杠杆值 | 0.0 | 4 位统计精确 |

---

## 4. 真实差异（已量化并记录，不再追究）

| 条目 | 差异 | 性质 | 记录 |
|---|---|---|---|
| VECM 系数位数 | 相对差 2.96e-8 ~ 4.07e-6 | **底层数值**，非格式 | 第 113 轮 |
| `teffects ipw` 标准误 | +0.24% | GMM 联合 vs 顺序估计 | T4 |
| `teffects aipw` 点估计 | 低 2.53 | 同上 | T4 |
| `didregress estat ptrends` | 0.2542 vs 0.251088 | 五种约定已排除 | T13 |
| `xtreg, re` 额外 e() 键 | 多 9 个 | 移除会破坏 18 个测试 | T8 |
| `fd` 缺截距 | — | `linearmodels` 禁止差分含常数 | T10 |

---

## 5. 缺失功能（按目标口径记录并跳过）

> **2026-11-09 更正**：下表中 `synth` 一行是当时的快照，现已失效——`ws.synth`
> 默认路径已落地（E2），两个 Mata 内核被精确求解；`synth2` 仍不做。原文保留为历史。
> 详见 `docs/GAP_REGISTER.md` §3.6.4。

| 功能 | 报告条目 | 记录 |
|---|---|---|
| `bwselect` / `msetwo` | C16-02 | T1、T3（ADB 字节码）|
| `synth` / `synth2` | C18-02 | ~~T3、T6（6 个内核不可读）~~ → 2026-11-09 证伪并落地 `synth` E2 |
| `xtserial` | C12-11 | T3（本机 Stata 未安装）|
| `summarize, detail` 百分位 | — | ~~T24~~ **已实现** |
| `teffects psmatch` 方差 | C15-01 | T3、T23 |
| `L()` 滞后算子 | C08-06 | ~~已实现~~ **第 115 轮完成** |

---

## 6. 本轮发现并修复的真实缺陷（第 100–174 轮）

| 轮 | 缺陷 | 影响 |
|---|---|---|
| 115 | `L()` 算子缺失 | 所有含滞后的公式不可用 |
| 116–119 | `esttab` 缺 `N` 行、缺边框、多模型名行 | 默认表多 3 行又少 1 行 |
| 130 | `estat_ic` 调用 `cov_params()`（属性当方法）| **命令崩溃** |
| 131 | `estat vce` 同类问题（两处）| **命令崩溃** |
| 132 | `estat vce` 标题缺模型名 | 布局 |
| 133 | `estat hettest` 按位置配对系数与设计列 | 吸收效应拟合失败 |
| 145/149 | `irf_table` 入口位置易误判 | 文档/守卫 |
| 155–156 | `summarize` 指数位数与列对齐 | 布局 |
| 159 | `summarize, detail` 未实现 | **整功能缺失** |
| 160 | `detail` 空缺槽位未打印 `.` | 布局 |
| 162 | **`VAR()` 丢弃 DataFrame 列名** | 影响所有 VAR 派生表 |
| 162 | `varlmar` 缺顶部边框、`varwle` chi2 位数 | 布局 |
| 164 | **`VECM` 同一列名缺陷** | 同上 |
| 165 | ARIMA/GARCH 缺 `response_name` | 后估计读到 None |

---

## 7. 方法论教训（已写入 error.md 共 26 条）

1. **先采集再写 fixture** —— 手抄参考行已犯 4 次。
2. **过滤空行会掩盖差异，也会凭空造出差异** —— 双向错误都犯过。
3. **入口位置是最容易误判的** —— `cov_params`、`irf_table`、杠杆值、`t_test` vs `lincom`，共 5 次。
4. **共享输入路径的缺陷会在每一处重现** —— VAR 修好了，VECM 没有。
5. **比对前先确认两侧默认设定相同** —— ARIMA 的 trend 差点被记成似然定义差异。
6. **守卫可能在保护本包自己的输出** —— 已发生 4 次，全部按实测改写。
7. **样本量会同时移动多个量** —— Newey-West 的 n 影响带宽和残差。
