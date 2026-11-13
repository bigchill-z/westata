# esttab 屏幕输出类选项：实测规格与实现状态

> 依据 `esttab.ado`（estout 2.1.4, Ben Jann）第 1138–1206 行的 syntax，
> 以及对本仓库 grilic.dta 的活体实测。

## 实现入口

```python
ws.render_estimates_table(
    fits,
    model_names=names,
    se=True,                          # b se
    star=(0.05, 0.01, 0.001),         # star(...)
    mtitle=True,                      # 切到 esttab 布局
    stats=["N", "r2"],                # N / R-sq / Adj R-sq / RMSE
    pvalues=True,                     # b p
)
```

更细的选项通过 `EsttabOptions` 直接传给 `esttab_render.render_esttab_table`。

## 已实测确认（13 组逐字符零差异）

| 选项 | 状态 | 关键实测事实 |
|---|---|---|
| `se` | ✅ | 第二行 `(0.00638)` |
| `t` | ✅ | 第二行 `(16.74)` |
| `p` | ✅ | 第二行 `(0.000)`，三位小数 |
| `ci` | ✅ | 列宽 26，注释换成 `95% confidence intervals in brackets` |
| `b`（单独） | ✅ | **仍打印 t 值**，不缩短表格 |
| `se nostar` | ✅ | 列宽 13（非 16） |
| `se brac` | ✅ | 方括号 + 注释改 `brackets` |
| `keep(...)` | ✅ | 行数 12 |
| `drop(...)` | ✅ | 行数 18 |
| `order(...)` | ✅ | 重排生效 |
| `rename(old new)` | ✅ | 标签改写 |
| `varlabels(x "L")` | ✅ | 标签改写，且**移除块间空行**（21→18）|
| `nocons` | ✅ | 去掉常数行（21→18）|

## 关键规则（易错点）

### 1. cells 只出两行，"最后一个胜出"

`esttab m1 m2, b t p` → 只有 p。`b se p` → 报错 `only one allowed of ...`。

### 2. 星号默认 5%/1%/0.1%

`esttab` 默认 `* p<0.05, ** p<0.01, *** p<0.001`，
与 `estimates table` 的 `star(0.1 0.05 0.01)`（10%/5%/1%）**不同**。

### 3. `star(...)` 必须符号与阈值配对

```stata
esttab m1 m2, star(0.1 0.05 0.01)                  → r(198) unmatched list
esttab m1 m2, star(* 0.1 ** 0.05 *** 0.01)         → 正确
```

### 4. 列宽是推导的，不是常数

| 情形 | 列宽 | 规则线（2 模型）|
|---|---|---|
| 带星号 | 16 | 44 |
| 无星号 | 13 | 38 |
| `ci` | 26 | 64 |

### 5. `N` 默认打印

`stats=["r2"]` 是**追加**，不是替换；显式 `stats=["N","r2"]` 不会打印两次 N。

### 6. `varlabels` 抑制块间空行

- `se` → 21 行，空行在 0, 7, 10, 13
- `se varlabels(s "Years")` → 18 行，空行只在 0

## 明确不属于屏幕输出的选项

| 选项 | 原因 |
|---|---|
| `notype` | `esttab.ado:409` 只在写 `using` 文件时内部添加；屏幕输出下调用报 **r(7)** |
| `nomargin` | 实测与 `margin` 输出相同，对屏幕表格无影响 |
| `margin` | 属 marginal-effects 语义，会去掉 N 行并换注释，不是单元格格式 |
| `using` / `style` / `append` / `replace` | 文件导出，不在屏幕布局范围 |
| `title` / `prehead` / `postfoot` / `smcl*` | 文件或标记语言输出 |

## 未实现（后续可做）

`label`（变量标签列加宽）、`unstack`、`mgroups`、`indicate`、`interaction`、
`equations`、`substitute`、`fmt`、`starlevels` 的自定义符号形式、`nolz`。
