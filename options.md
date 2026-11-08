# 目标模式：任务过程中跳过的、需要用户拍板的决策

> 本文件记录需用户决定的选项；若后续不再需要，及时更新。

## 一、2026-11-07 安装《教材必装 Stata 程序清单》时的决策

### 已关闭（2026-11-07 23:47 用户答复：「都不用」）

| 决策 | 结果 |
|---|---|
| O1 是否补装 `xtserial` | **保持现状，不补装**（本机未装，文档中为"建议装"非"必装"）|
| O2 是否补装"不必装"的 `rd`/`psmatch2`/`xtbalance` | **维持不装** |
| O3 示例数据集安放位置 | **维持现状** |
| O4 可复现产物 `_stata_pkgs\` | **保留** |
| 是否整理 SOP / 可复用 skill | **用户明确拒绝，不再询问、不创建** |

---

## 二、v3 报告修复过程中记录的技术判断（非用户决策，仅登记）

以下不是"等待用户拍板"的事项，而是**已做出并验证的技术判断**，登记在此以便追溯。

### T1. `rdrobust` 自动带宽：停在"组合规则已实现"

- **现状**：三种组合规则（`mserd`/`msesum`/`msetwo`）已从 `rdbwselect.ado:499/527/537`
  精确转写，7 个守卫覆盖算术与符号方向。
- **阻塞点**：`C[1..3]` 由 `rdrobust_bw()` 在 `l/lrdrobust.mlib` 中计算。
  `mata which` 确认位置；该文件 321 KB、**可打印字节仅 1.9%**，函数名是孤立字符串，
  前后皆二进制 —— 是编译字节码。注意这与 `synth` **不是同一类**：`synth` 的顶层算法
  是可读的 `synth.ado`，只有数值子过程在库里（见 T6）；而 `rdrobust` 的带宽选择器
  **整体**在库里，ado 一侧只有参数设置。
- **已采取**：`rdrobust` 仍拒绝自动选带宽（`UnsupportedError`），而非接入一个
  会返回 `h=0.0757`（Stata 是 `0.182`）的选择器。
- **若需推进**：需要 `rdrobust` 官方源码或论文附录的完整 `C[1..3]` 定义。

### T2. `nbreg` 的多起点选择阈值

- **现状**：阈值 `1e-3`、起点下限 `1e-9`，均已由实测确定。
- **遗留**：`alpha` 落在 `1e-6`，Stata 在 `1.03e-09`。全部起点的似然一致到 `4e-4`，
  **无法凭似然区分**"真边界"与数值噪声（另一份数据上噪声增益 `5.3e-4`，量级相同、
  方向相反）。
- **影响**：`ll` 差 `4e-4`、系数末位不同；`/lnalpha` 与 `alpha` 的 SE 已按边界规则显示 `.`。
- **若需推进**：需要 Stata 的停止准则（梯度容差 / 边界投影方式），而非继续调起点。

### T3. 未实现的条目（已确认为能力边界）

| 条目 | 状态 | 依据 |
|---|---|---|
| `synth` / `synth2` | 未实现 | **顶层算法可读**（`synth.ado` 931 行，含两步优化、V 矩阵、nested 结构）；六个数值子过程在 `lsynth_mata_subr` 库中。见 T6 |
| `xtserial` | 未实现 | 本机 Stata 未装，`r(111)` |
| `arimasoc` | 未实现 | — |
| `rdrobust` 自动带宽 | 部分 | 见 T1 |
| `teffects` AI-robust 方差 | 未实现 | 六种影响函数形式试过，均差 2.9–3.4%，见 `teffects_ra.py` 文档 |
| `teffects psmatch` 推断量 | 未实现 | 需完整 Abadie-Imbens 影响函数，docstring 已排除两种简化 |

### T4. `teffects aipw` 点估计低 2.53（阻塞点已定位到可读源码）

**重要进展**：`_teffects_gmm_aipw.ado`（1500 行）**是可读的**，核心矩条件已找到。

```stata
* L469  ipw = 1/pij             只看处理组
* L475  resj = cond(tr, ipw*(depvar-muj), 0) + muj - pop[1,j]
* L630  CMomentj, ofform=linear:  resj = cond(tr, depvar - muj, 0)
```

- **PO 矩条件**：`t*(1/p)*(y - m) + m - mu` —— 与本包的 `fitted + t*(y-fitted)/p` 形式一致；
- **条件均值矩条件**：`t*(y - m)` —— 按处理组拟合，也与本包一致。

**真正的差异在求解方式，不在公式**：

```stata
* L753  dcm = cond(tr, 1-ipw, 1)
```

`muj` 的导数含 `(1 - 1/p)`，说明 **`muj` 与 `pop` 是联合 GMM 估计的**，
而不是「先 OLS 拟合、再对 augmented 取均值」。

本包走的是后者，因此**落在另一个点**。修正项 `-6.58` / `-4.05` 不是 bug，
而是两个估计点之间的差。

- **若需推进**：把 `(pop, cm, ps)` 作为整体做 GMM，用上面三条矩条件联立求解。
  这是估计器级改动，不是修补。
- **参考**：`_teffects_gmm_aipw.ado` 在 `ado\base\_\` 下，`findfile` 可定位。

### T5. `estimate` 之外还差什么，已知清单

| 命令 | 现状 | 与 Stata 的差 |
|---|---|---|
| `teffects ra`（类） | **精确** | — |
| `teffects ipw` | 点估计精确 | SE 高 0.24% |
| `teffects aipw` | 点估计低 2.53 | 见 T4 |
| `teffects psmatch` | 点估计精确 | `se`/`z`/`p`/CI 为 NaN（需 Abadie-Imbens 影响函数）|
| `teffects nnmatch`（类） | 点估计精确 | SE 高 0.78%，输出已标注为近似 |

### T6. `synth` / `synth2`：可读与不可读的边界，已用 `findfile` 与 `mata which` 界定

前几轮把 `synth` 一句「核心在 Mata 字节码」概括为 A 类「信息不可得」。**这个判断过粗。**

**可读的部分** — `synth.ado`（`C:\Users\zjl\ado\plus\s\synth.ado`，931 行）：

```
L367  /* now go to optimization */
L371  di as res "Second Step: Run Optimization"
L386  /* Set up V matrix */
L427  /* now if the user wantes the full nested method, go and get Vstar via nested method */
L39   nested
```

即**两步优化流程、V 矩阵构造、`nested` 方法**都在可读代码里。

**不可读的部分** — 六个子过程全部位于 `lsynth_mata_subr` 库，`mata which` 逐个确认：

| 函数 | 所在 |
|---|---|
| `normalize()` | `lsynth_mata_subr` |
| `regsval()` | `lsynth_mata_subr` |
| `normweights()` | `lsynth_mata_subr` |
| `getabs()` | `lsynth_mata_subr` |
| `roottaker()` | `lsynth_mata_subr` |
| `roundmat()` | `lsynth_mata_subr` |

**这六个是数值内核**（标准化、回归得 V、权重归一、取绝对值、求根、矩阵舍入），
而损失函数的**结构**由 ado 一侧的 `loss` 变量与 `$bd`/`$marg`/`$maxit`/`$sig` 参数控制。

- **结论**：`synth` 并非「完全不可得」。重建需要按 `synth.ado` 的调用顺序实现那六个内核，
  再套上可读的两步优化骨架。工作量是**中等到大**，不是**不可能**。
- **此前记录有误**，以此条为准。
