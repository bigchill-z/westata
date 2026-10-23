"""生成 NOTES.md 的覆盖矩阵行（避免手抄数字出错）。

用法：.venv\\Scripts\\python.exe retest/domains/linear_iv/_mktable.py
输出：retest/domains/linear_iv/_matrix.md
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

M = "自造 140x14 组"

# item name -> (westata 调用, 数据)
META = {
    "1 regress basic: mpg weight foreign": ("ws.OLS.from_formula('mpg ~ weight + foreign').fit()", "auto.dta"),
    "2a regress i.foreign": ("OLS 'mpg ~ weight + C(foreign)'", "auto.dta"),
    "2b regress i.rep78（5 个缺失两侧都删）": ("OLS 'mpg ~ weight + C(rep78)'", "auto.dta"),
    "3 regress vce(robust) [westata HC1]": ("OLS(...).fit(cov_type='HC1')", "auto.dta"),
    "3b regress vce(robust)：e(F) 语义": ("同上，仅比 e(F)", "auto.dta"),
    "4a regress cluster(gid) 14 组：系数/SE/常规标量": ("OLS(...).fit(cov_type='cluster', cov_kwds={'groups': gid})", M),
    "4b regress cluster：存下来的 e(df_r) 语义": ("同上，仅比 e(df_r)", M),
    "4c regress cluster：p 值（两边都用 t(G-1)）": ("同上，比 t 检验 p", M),
    "4d regress cluster：e(F) 语义": ("同上，仅比 e(F)", M),
    "5 regress [fweight=w]：系数/SE/df": ("OLS(weights=w, weight_type='fweight')", M),
    "5 regress [aweight=w]：系数/SE/df": ("OLS(weights=w, weight_type='aweight')", M),
    "5 regress [pweight=w]：系数/SE/df": ("OLS(weights=w, weight_type='pweight')", M),
    "5 regress [iweight=w]：系数/SE/df": ("OLS(weights=w, weight_type='iweight')", M),
    "5e regress [pweight=w]：e(F) 语义（pw 隐含 robust）": ("同上，仅比 e(F)", M),
    "5g regress [fweight=w] vce(robust)：复制型 meat": ("fweight + cov_type='HC1'", M),
    "5h regress [aweight=w] vce(robust)：精度型 meat": ("aweight + cov_type='HC1'", M),
    "6a regress if mpg>20": ("OLS(..., subset=mpg>20)", "auto.dta"),
    "6b regress if weight<4000 & mpg>18": ("OLS(..., subset=两个条件)", "auto.dta"),
    "7a areg absorb(gid) [分类 FE] vs FEOLS": ("ws.FEOLS('y ~ x1 + x2 | gid')", M),
    "7b areg absorb(gid) vce(robust)": ("FEOLS(...).fit(cov_type='HC1')", M),
    "7c areg absorb(gid) vce(cluster gid) vs FEOLS": ("FEOLS(...).fit(cov_type='cluster', {'cluster': 'gid'})", M),
    "7c-bis areg cluster(gid) vs reghdfe 同规格（归因用）": ("同 7c，但改比 reghdfe", M),
    "7d areg absorb(gid) vce(cluster t) [非嵌套]": ("FEOLS(...).fit(cov_type='cluster', {'cluster': 't'})", M),
    "7e areg absorb(cont) [数值存储、8 个取值]": ("FEOLS('y ~ x1 + x2 | cont')", M),
    "8a reghdfe absorb(gid) vs FEOLS": ("FEOLS('y ~ x1 + x2 | gid')", M),
    "8b reghdfe absorb(gid t) 双向 vs FEOLS 双向": ("FEOLS('y ~ x1 + x2 | gid + t')", M),
    "8c reghdfe absorb(gid t) vce(robust)": ("FEOLS(gid+t).fit(cov_type='HC1')", M),
    "8d reghdfe absorb(gid t) vce(cluster gid)": ("FEOLS(gid+t).fit(cov_type='cluster', {'cluster': 'gid'})", M),
    "9 FEOLS(gid+t, HC1) vs areg absorb(gid)+i.t": ("FEOLS(gid+t).fit(cov_type='HC1')", M),
    "10a ivregress 2sls 恰好识别 (e1=z1)": ("ws.IV2SLS.from_formula('yiv ~ x1 + [e1 ~ z1]')", M),
    "10b ivregress 2sls 过度识别 (e1=z1 z2)": ("IV2SLS('yiv ~ x1 + [e1 ~ z1 + z2]')", M),
    "10c ivregress 2sls vce(robust) vs westata HC0（实测正确映射）": ("IV2SLS(...).fit(cov_type='HC0')", M),
    "10d ivregress 2sls vce(robust) vs westata HC1（文档映射）": ("IV2SLS(...).fit(cov_type='HC1')", M),
    "10e ivregress 2sls vce(robust) small vs westata HC1": ("IV2SLS(...).fit(cov_type='HC1')", M),
    "10f ivregress 2sls vce(cluster gid)": ("IV2SLS(...).fit(cov_type='cluster', {'clusters': gid})", M),
    "11a ivregress liml 恰好识别 (e1=z1)": ("ws.IVLIML.from_formula('yiv ~ x1 + [e1 ~ z1]')", M),
    "11b ivregress liml 过度识别 (e1=z1 z2)": ("IVLIML('yiv ~ x1 + [e1 ~ z1 + z2]')", M),
    "11c ivregress liml vce(robust) vs westata HC0（实测正确映射）": ("IVLIML(...).fit(cov_type='HC0')", M),
    "11d ivregress liml vce(robust) vs westata HC1（文档映射）": ("IVLIML(...).fit(cov_type='HC1')", M),
    "11e ivregress liml vce(cluster gid) vs westata cluster": ("IVLIML(...).fit(cov_type='cluster', {'clusters': gid})", M),
    "11f ivregress liml vce(cluster gid) small vs westata cluster": ("同上（Stata 侧加 small）", M),
    "12a ivregress gmm 过度识别（两步 GMM，核 e(V)）": ("ws.IVGMM.from_formula('yiv ~ x1 + [e1 ~ z1 + z2]')", M),
    "12a-bis gmm：e(V) 全矩阵逐元素（12a 的对角已含于 SE）": ("同上，比 V 全矩阵", M),
    "12b ivregress gmm vce(robust) vs westata HC0（实测正确映射）": ("IVGMM(...).fit(cov_type='HC0')", M),
    "12c ivregress gmm vce(robust) vs westata HC1（文档映射）": ("IVGMM(...).fit(cov_type='HC1')", M),
    "12d ivregress gmm vce(cluster gid)": ("IVGMM(...).fit(cov_type='cluster', {'clusters': gid})", M),
    "13 eivreg reliab(x1 .9 x2 .95)": ("ws.Eivreg.from_formula(reliab={'x1': .9, 'x2': .95})", M),
    "14a OLS e(): N/df_r/df_m/r2/r2_a/rmse/F/ll": ("OLS(...).fit()", "auto.dta"),
    "14b OLS e(rss)/e(mss)：westata 未存，靠 resid 反算": ("resid / fitted 反算", "auto.dta"),
    "14c 加权 OLS e(N)/e(df_r)/e(r2)：fweight 的 N=sum(w)": ("OLS(weights=w, weight_type='fweight')", M),
    "15a OLS e() 覆盖度（rss/mss/sum_w/rank/tss）": ("比较 e() 键集合", "auto.dta"),
    "15b 加权 OLS e() 覆盖度（sum_w）": ("比较 e() 键集合", M),
    "15c cluster OLS e() 覆盖度（N_clust）": ("比较 e() 键集合", M),
    "15d 2SLS e() 覆盖度（rss/mss/kappa/k_endog/iterations）": ("比较 e() 键集合", M),
    "15e GMM e() 覆盖度（Stata 的 e(J) 无对应）": ("比较 e() 键集合", M),
    "15f eivreg e() 覆盖度": ("比较 e() 键集合", M),
    "7c-ter 闭式复算：within CR0 × K 候选因子（7c 归因的独立证据）": ("手算 within CR0 × 两个候选 CR1 因子", M),
    "17 非稳健：Stata e(F) 与「westata V 的 Wald/k」": ("把 westata 的 e(V) 代入 b'V^-1b/k", M),
    "17 robust：Stata e(F) 与「westata V 的 Wald/k」": ("同上", M),
    "17 cluster：Stata e(F) 与「westata V 的 Wald/k」": ("同上", M),
    "17 pweight：Stata e(F) 与「westata V 的 Wald/k」": ("同上", M),
    "16a areg absorb(真连续变量)：Stata 侧饱和，不构成可比的估计": ("（不比对）", M),
    "16b eivreg reliab 取小值（0.5）：Stata 直接报错，无可比结果": ("（不比对）", M),
    "16c ivregress + 权重（westata P1 不支持）": ("（不比对）", M),
    "16d 环境检查：reghdfe 安装状态": ("which reghdfe", "—"),
}

CMDS = {
    "1 regress basic: mpg weight foreign": "regress mpg weight foreign",
    "2a regress i.foreign": "regress mpg weight i.foreign",
    "2b regress i.rep78（5 个缺失两侧都删）": "regress mpg weight i.rep78",
    "3 regress vce(robust) [westata HC1]": "regress ..., vce(robust)",
    "3b regress vce(robust)：e(F) 语义": "同上",
    "4a regress cluster(gid) 14 组：系数/SE/常规标量": "regress y x1 x2, vce(cluster gid)",
    "4b regress cluster：存下来的 e(df_r) 语义": "同上",
    "4c regress cluster：p 值（两边都用 t(G-1)）": "同上 + test x1",
    "4d regress cluster：e(F) 语义": "同上",
    "5 regress [fweight=w]：系数/SE/df": "regress y x1 x2 [fweight=w]",
    "5 regress [aweight=w]：系数/SE/df": "regress y x1 x2 [aweight=w]",
    "5 regress [pweight=w]：系数/SE/df": "regress y x1 x2 [pweight=w]",
    "5 regress [iweight=w]：系数/SE/df": "regress y x1 x2 [iweight=w]",
    "5e regress [pweight=w]：e(F) 语义（pw 隐含 robust）": "同上",
    "5g regress [fweight=w] vce(robust)：复制型 meat": "regress ... [fweight=w], vce(robust)",
    "5h regress [aweight=w] vce(robust)：精度型 meat": "regress ... [aweight=w], vce(robust)",
    "6a regress if mpg>20": "regress mpg weight foreign if mpg>20",
    "6b regress if weight<4000 & mpg>18": "regress ... if weight<4000 & mpg>18",
    "7a areg absorb(gid) [分类 FE] vs FEOLS": "areg y x1 x2, absorb(gid)",
    "7b areg absorb(gid) vce(robust)": "areg ..., absorb(gid) vce(robust)",
    "7c areg absorb(gid) vce(cluster gid) vs FEOLS": "areg ..., absorb(gid) vce(cluster gid)",
    "7c-bis areg cluster(gid) vs reghdfe 同规格（归因用）": "reghdfe ..., absorb(gid) vce(cluster gid)",
    "7d areg absorb(gid) vce(cluster t) [非嵌套]": "areg ..., absorb(gid) vce(cluster t)",
    "7e areg absorb(cont) [数值存储、8 个取值]": "areg y x1 x2, absorb(cont)",
    "8a reghdfe absorb(gid) vs FEOLS": "reghdfe y x1 x2, absorb(gid)",
    "8b reghdfe absorb(gid t) 双向 vs FEOLS 双向": "reghdfe ..., absorb(gid t)",
    "8c reghdfe absorb(gid t) vce(robust)": "reghdfe ..., absorb(gid t) vce(robust)",
    "8d reghdfe absorb(gid t) vce(cluster gid)": "reghdfe ..., absorb(gid t) vce(cluster gid)",
    "9 FEOLS(gid+t, HC1) vs areg absorb(gid)+i.t": "areg y x1 x2 i.t, absorb(gid) vce(robust)",
    "10a ivregress 2sls 恰好识别 (e1=z1)": "ivregress 2sls yiv x1 (e1=z1)",
    "10b ivregress 2sls 过度识别 (e1=z1 z2)": "ivregress 2sls yiv x1 (e1=z1 z2)",
    "10c ivregress 2sls vce(robust) vs westata HC0（实测正确映射）": "ivregress 2sls ..., vce(robust)",
    "10d ivregress 2sls vce(robust) vs westata HC1（文档映射）": "同上",
    "10e ivregress 2sls vce(robust) small vs westata HC1": "ivregress 2sls ..., vce(robust) small",
    "10f ivregress 2sls vce(cluster gid)": "ivregress 2sls ..., vce(cluster gid)",
    "11a ivregress liml 恰好识别 (e1=z1)": "ivregress liml yiv x1 (e1=z1)",
    "11b ivregress liml 过度识别 (e1=z1 z2)": "ivregress liml yiv x1 (e1=z1 z2)",
    "11c ivregress liml vce(robust) vs westata HC0（实测正确映射）": "ivregress liml ..., vce(robust)",
    "11d ivregress liml vce(robust) vs westata HC1（文档映射）": "同上",
    "11e ivregress liml vce(cluster gid) vs westata cluster": "ivregress liml ..., vce(cluster gid)",
    "11f ivregress liml vce(cluster gid) small vs westata cluster": "ivregress liml ..., vce(cluster gid) small",
    "12a ivregress gmm 过度识别（两步 GMM，核 e(V)）": "ivregress gmm yiv x1 (e1=z1 z2)",
    "12a-bis gmm：e(V) 全矩阵逐元素（12a 的对角已含于 SE）": "同上",
    "12b ivregress gmm vce(robust) vs westata HC0（实测正确映射）": "ivregress gmm ..., vce(robust)",
    "12c ivregress gmm vce(robust) vs westata HC1（文档映射）": "同上",
    "12d ivregress gmm vce(cluster gid)": "ivregress gmm ..., vce(cluster gid)",
    "13 eivreg reliab(x1 .9 x2 .95)": "eivreg y x1 x2, reliab(x1 .9 x2 .95)",
    "14a OLS e(): N/df_r/df_m/r2/r2_a/rmse/F/ll": "regress mpg weight foreign",
    "14b OLS e(rss)/e(mss)：westata 未存，靠 resid 反算": "同上",
    "14c 加权 OLS e(N)/e(df_r)/e(r2)：fweight 的 N=sum(w)": "regress y x1 x2 [fweight=w]",
    "15a OLS e() 覆盖度（rss/mss/sum_w/rank/tss）": "regress mpg weight foreign",
    "15b 加权 OLS e() 覆盖度（sum_w）": "regress y x1 x2 [fweight=w]",
    "15c cluster OLS e() 覆盖度（N_clust）": "regress y x1 x2, vce(cluster gid)",
    "15d 2SLS e() 覆盖度（rss/mss/kappa/k_endog/iterations）": "ivregress 2sls yiv x1 (e1=z1 z2)",
    "15e GMM e() 覆盖度（Stata 的 e(J) 无对应）": "ivregress gmm yiv x1 (e1=z1 z2)",
    "15f eivreg e() 覆盖度": "eivreg y x1 x2, reliab(...)",
    "7c-ter 闭式复算：within CR0 × K 候选因子（7c 归因的独立证据）": "（手算，无 Stata 命令）",
    "17 非稳健：Stata e(F) 与「westata V 的 Wald/k」": "regress y x1 x2",
    "17 robust：Stata e(F) 与「westata V 的 Wald/k」": "regress y x1 x2, vce(robust)",
    "17 cluster：Stata e(F) 与「westata V 的 Wald/k」": "regress y x1 x2, vce(cluster gid)",
    "17 pweight：Stata e(F) 与「westata V 的 Wald/k」": "regress y x1 x2 [pweight=w]",
    "16a areg absorb(真连续变量)：Stata 侧饱和，不构成可比的估计": "areg y x1 x2, absorb(uid)",
    "16b eivreg reliab 取小值（0.5）：Stata 直接报错，无可比结果": "eivreg ..., reliab(x1 .5)",
    "16c ivregress + 权重（westata P1 不支持）": "ivregress 2sls ... [pw=w]",
    "16d 环境检查：reghdfe 安装状态": "which reghdfe",
}

ORDER = ["A", "B", "C", "D", "E"]


def fmt(value) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{value:.1e}"


def main() -> int:
    rows = []
    idx = 0
    for line in (HERE / "results.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        idx += 1
        name = d["name"]
        westata_call, data = META.get(name, ("?", "?"))
        cmd = CMDS.get(name, "?")
        pt = d.get("param_tier") or "-"
        st = d.get("se_tier") or "-"
        if d["status"] == "skip":
            tier = "skip"
        elif d["status"] == "error":
            tier = "error"
        elif d.get("param_rel") is None or d.get("param_rel") != d.get("param_rel"):
            # 没有系数比较的两类：e() 覆盖度 / 归因用证据条目
            if name.startswith("15"):
                tier = "覆盖度 pass" if d["status"] == "pass" else "覆盖度 fail"
            else:
                tier = "证据"
        else:
            a = pt if pt in ORDER else "E"
            b = st if st in ORDER else "E"
            tier = a if ORDER.index(a) >= ORDER.index(b) else b
            if d["status"] == "fail" and tier == "A":
                tier = "A*"
        # 表格内的 | 会把单元格切开，统一换成 \| 转义
        cmd_cell = cmd.replace("|", "\|")
        call_cell = westata_call.replace("|", "\|")
        rows.append(
            f"| {idx} | `{cmd_cell}` | {call_cell} | {data} | {tier} | "
            f"{fmt(d.get('param_rel'))} | {fmt(d.get('se_rel'))} |"
        )
    out = HERE / "_matrix.md"
    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
