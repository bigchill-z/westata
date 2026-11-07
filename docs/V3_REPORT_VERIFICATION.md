# v3 报告核实清单

本文件记录对 `westata_vs_Stata_替代评估报告_v3.md` 各条目的实测核实结果。
**报告中的部分结论与当前代码不符**，逐条核实后记录如下。

## 一、报告成立并已修复

| 条目 | 报告内容 | 核实 | 处理 |
|---|---|---|---|
| **N14** | `prais, corc` 标准误 4.7% 偏差 | ✅ 成立：系数与 rho 精确，`se(price)` `.8491466` vs `.8108501` | **已修复**：最终方差用了错误的变换（PW 而非 corc），且 corc 的残差自由度应少 1 |
| **P0-2** | `rdrobust` 缺自动带宽 | ✅ 成立：`bandwidth=None` 抛 `UnsupportedError` | **部分**：三种组合规则（mserd/msesum/msetwo）已从 `rdbwselect.ado` 精确转写；`C[1..3]` 在 `lrdrobust.mlib` 字节码中，不可得 |

## 二、报告不成立（当前代码已正确）

| 条目 | 报告内容 | 实测结果 |
|---|---|---|
| **P0-1** | `PanelOLS.fit(cov_type=)` 在 linearmodels 7.x 下全部 `ValueError` | ❌ **不成立**。linearmodels **7.1** 下 `robust` / `HC1` / `cluster` 三种写法**全部成功**，且数值不同（`0.065784` vs `0.116169`），证明确实在计算 |
| **C13-08** | `reg airpassengers i.m` 整型因子未展开，`F(0,143)=0.00`，只有常数项 | ❌ **不成立**。`C(m)` 正常展开为 12 个参数，`F=1.4245`（Stata 1.42）、`R²=0.1061`（Stata 0.1061），全部精确 |

## 三、已记录但仍未完成

| 条目 | 状态 | 原因 |
|---|---|---|
| `teffects psmatch` 推断量 | ATET 精确（`2639.8646` vs `2639.865`），`se`/`z`/`p`/CI 为 NaN | 需要完整的 Abadie-Imbens 影响函数。模块 docstring 已排除两种简化路线：直接对处理效应取样本方差会小 1.2948 倍；该比值既非 `n/n_t` 也非 `(n-1)/(n_t-1)`，说明携带一阶推断函数修正 |
| `rdrobust` 自动带宽 | 组合规则已完成 | `C[1..3]` 在 Mata 字节码中 |
| `synth` / `synth2` | 未实现 | 核心在 Mata 字节码（A 类：信息不可得）|

## 四、`nnmatch` 标准误的诚实标注

`NearestNeighbourMatching` 的 `std_err` 使用的是 **Abadie-Imbens approximation**，
不是精确影响函数，且与 Stata 有 0.78% 偏差（`777.4784` vs `771.4713`）。

**该近似已在输出中明确标注**：

```
note: standard errors are Abadie-Imbens approximation
```

这与其他条目「宁可标 NaN 也不给看似合理的数」的处理一致。

## 五、核实方法

每条都遵循同一原则：**拿实时 Stata 对照，不以报告描述为准**。

- P0-1：构造真实面板（30 组 × 8 期），逐个 `cov_type` 实测
- C13-08：用报告的原始数据（`airpassengers`）+ 从 `time` 提取月份构造 `m`，复现全部三个统计量
- N14：用报告的原始配置（`icecream` + `consumption ~ temp + price + income, corc`）逐项比对
- `psmatch`：先确认点估计精确，再确认识别 NaN 的确切字段

## 六、最终验证

```
5137 passed, 3 skipped, 0 failed
```
