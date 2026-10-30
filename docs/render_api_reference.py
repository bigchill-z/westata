"""Render docs/API.md and docs/API.zh-CN.md from the collected data.

Nothing in either document is typed by hand. Each entry combines:

* the signature, from ``inspect`` on the live object (``api_data.json``);
* the example's OUTPUT, from actually running it (``api_data.json``);
* the prose -- Stata command, description, caveats -- from ``prose_*.json``.

That split is deliberate. A manual this size written by hand drifts from the code, and
this repository has already paid for that lesson: README counts that lagged the suite, a
Quick start that had never run, and `e(aic)` presented as a Stata macro when `regress`
stores nothing of the sort. Generating everything mechanical leaves only the prose to get
wrong, and prose is reviewable.

An export with no prose is emitted with a visible ::: warning block rather than being
dropped, so an incomplete manual looks incomplete instead of looking finished.

Usage::

    python docs/render_api_reference.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "api_source"

#: Grouping used for the table of contents. Order matters: it is the reading order.
FAMILIES: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "Linear models and IV",
        "线性模型与工具变量",
        (
            "OLS",
            "FEOLS",
            "PanelOLS",
            "RandomEffects",
            "BetweenOLS",
            "FirstDifferenceOLS",
            "IV2SLS",
            "IVLIML",
            "IVGMM",
            "XtIvreg",
            "XtGls",
            "XtPcse",
            "XtAbond",
            "Eivreg",
            "Sureg",
            "Mvreg",
            "RobustRegression",
            "QuantileRegression",
            "Bsqreg",
            "Sqreg",
            "NeweyWestResult",
            "Nl",
        ),
    ),
    (
        "Discrete choice, counts and limited responses",
        "离散选择、计数与受限因变量",
        (
            "Logit",
            "Probit",
            "OrderedLogit",
            "OrderedProbit",
            "Poisson",
            "NegativeBinomial",
            "NegativeBinomialNB1",
            "ZeroInflatedPoisson",
            "ZeroInflatedNegativeBinomial",
            "TruncatedPoisson",
            "TruncatedNegativeBinomial",
            "Tobit",
            "Truncreg",
            "Intreg",
            "Heckman",
            "Etregress",
            "Frontier",
            "XtLogit",
            "XtProbit",
            "XtPoisson",
            "XtNbreg",
            "XtTobit",
        ),
    ),
    (
        "Time series, multivariate and panel roots",
        "时间序列、多元与面板单位根",
        (
            "ARIMA",
            "GARCH",
            "VAR",
            "VECM",
            "XtUnitroot",
            "IpsResult",
            "LlcResult",
            "UnitRootResult",
            "Manova",
            "ManovaResult",
        ),
    ),
    (
        "Treatment effects and survey data",
        "处理效应与调查数据",
        (
            "Didregress",
            "Xtdidregress",
            "TreatmentEffectResult",
            "IPW",
            "AIPW",
            "NearestNeighbourMatching",
            "RegressionAdjustment",
            "SurveyDesign",
            "SvyMean",
            "SvyTotal",
            "SvyProportion",
            "SvyRegress",
            "SvyLogit",
        ),
    ),
    (
        "Resampling",
        "重抽样",
        ("Bootstrap", "Jackknife", "Permute"),
    ),
    (
        "Postestimation functions",
        "后估计函数",
        (
            "testnl",
            "lrtest",
            "hausman",
            "contrast",
            "pwcompare",
            "nlcom",
            "margins",
            "estat_classification",
            "estat_endogenous",
            "estat_firststage",
            "estat_gof",
            "estat_hettest",
            "estat_ic",
            "estat_imtest",
            "estat_overid",
            "estat_ovtest",
            "estat_summarize",
            "estat_vce",
            "estat_vif",
            "estat_auc",
            "estat_esize",
            "estat_moran",
            "estat_mundlak",
            "estat_szroeter",
            "estat_weakrobust",
            "linktest",
            "suest",
            "newey",
            "unitroot",
            "levin_lin_chu",
            "im_pesaran_shin",
            "check_time_spacing",
            "parse_exp_list",
            "normalize_weights",
        ),
    ),
    (
        "Working with a fitted result",
        "已拟合结果对象",
        (
            "StataResults",
            "params",
            "bse",
            "pvalues",
            "tvalues",
            "zvalues",
            "fvalue",
            "llf",
            "aic",
            "bic",
            "nobs",
            "df_model",
            "df_resid",
            "dist",
            "model_name",
            "cov_type",
            "cov",
            "e",
            "fittedvalues",
            "resid",
            "rsquared",
            "rsquared_adj",
            "model_data",
            "converged",
            "iterations",
            "grad_norm",
            "param_names",
            "test",
            "lincom",
            "predict",
            "summary",
            "wald_test",
            "cov_params",
            "conf_int",
        ),
    ),
    (
        "Errors and support types",
        "错误与支撑类型",
        (
            "WestataError",
            "ConsistencyError",
            "ConvergenceError",
            "FormulaError",
            "UnsupportedError",
            "WeightError",
            "ExpListError",
            "PackageNotFoundError",
            "StoredResults",
            "WeightSpec",
            "WEIGHT_TYPES",
            "SSC",
            "VceMatrix",
            "ParsedExpressionList",
            "WaldTest",
            "LincomResult",
            "IcResult",
            "VifResult",
            "MarginsResult",
            "GofResult",
            "ClassificationResult",
            "HetTestResult",
            "ImTestResult",
            "OvTestResult",
            "EndogenousResult",
            "FirstStageResult",
            "OverIdResult",
            "LRTestResult",
            "HausmanResult",
        ),
    ),
]

TEXT = {
    "en": {
        "title": "westata — complete API reference",
        "intro": (
            "Every public name in `westata`, with the Stata command it corresponds to, a "
            "description, and a runnable example whose output is captured from a real run.\n\n"
            "**How this document is built.** The signatures come from `inspect` on the live "
            "objects and the example outputs come from executing the examples, so neither can "
            "drift from the code. Only the prose is hand-written. A name whose prose is "
            "missing is shown with a warning rather than omitted, so an incomplete section "
            "looks incomplete.\n\n"
            "**Reading the examples.** Each block runs as written against a 200-row frame "
            "reachable as `base`, whose columns are "
            "`y x1 x2 x3 yb ycnt ycens lo hi y1 y2 firm year ypanel treat ytreat sel yselect`; "
            "`ws`, `np` and `pd` are in scope. The text below the code is the actual return "
            "value, not a transcription."
        ),
        "toc": "Contents",
        "stata": "Stata",
        "signature": "Signature",
        "example": "Example",
        "output": "Output",
        "notes": "Notes",
        "missing": (
            "This name has no hand-written description yet. The signature and example are "
            "still generated from the code, so they are accurate; only the prose is pending."
        ),
        "no_example": "No example was written for this name.",
        "methods": "Methods",
        "properties": "Properties",
        "generated": "Generated by",
        "version": "westata version",
    },
    "zh": {
        "title": "westata —— 完整 API 参考",
        "intro": (
            "`westata` 的全部公开名称，逐项给出对应的 Stata 指令、功能说明，以及一段"
            "**真实运行**、输出即所得的可运行示例。\n\n"
            "**本文档如何生成。** 函数签名取自运行时对象的 `inspect`，示例输出由"
            "**实际执行**示例得到，因此二者都不会与代码脱节。只有说明文字是手写的。"
            "若某个名称尚未写说明，会**显式标注**而不是略去——让不完整的章节看起来就是"
            "不完整的。\n\n"
            "**如何阅读示例。** 每个代码块都可照抄运行，作用域内有一个 200 行的数据框 "
            "`base`，列为 "
            "`y x1 x2 x3 yb ycnt ycens lo hi y1 y2 firm year ypanel treat ytreat sel yselect`，"
            "并有 `ws`、`np`、`pd`。代码块下方是**真实返回值**，不是转述。"
        ),
        "toc": "目录",
        "stata": "Stata 指令",
        "signature": "签名",
        "example": "示例",
        "output": "输出",
        "notes": "注意",
        "missing": (
            "该名称尚无手写说明。签名与示例仍由代码自动生成，因此是准确的；"
            "待补的只有说明文字。"
        ),
        "no_example": "该名称未编写示例。",
        "methods": "方法",
        "properties": "属性",
        "generated": "生成脚本",
        "version": "westata 版本",
    },
}


def _load() -> dict[str, Any]:
    path = REFERENCE / "api_data.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing; run docs/collect_api_reference.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _run_examples(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Execute every prose example once and index the outcome by name."""

    import sys

    sys.path.insert(0, str(ROOT / "docs"))
    from collect_api_reference import _Collector  # noqa: E402

    collector = _Collector()
    outcomes: dict[str, dict[str, Any]] = {}
    for name, entry in data.get("prose", {}).items():
        example = entry.get("example")
        if not example:
            continue
        outcomes[name] = collector.run(str(example))
    return outcomes


def _code_block(text: str, language: str) -> str:
    return f"```{language}\n{text.rstrip()}\n```"


def _entry_block(
    name: str,
    entry: dict[str, Any] | None,
    generated: dict[str, Any] | None,
    outcome: dict[str, Any] | None,
    lang: str,
) -> str:
    words = TEXT[lang]
    lines: list[str] = [f"### `{name}`", ""]

    prose = entry or {}
    stata = prose.get("stata")
    if stata:
        lines.append(f"**{words['stata']}:** `{stata}`")
        lines.append("")

    description = prose.get(lang)
    if description:
        lines.append(description)
    else:
        lines.append(f"> **⚠️ {words['missing']}**")
    lines.append("")

    if generated:
        signature = generated.get("signature")
        if signature:
            lines.append(f"**{words['signature']}**")
            lines.append("")
            lines.append(_code_block(f"ws.{name}{signature}", "python"))
            lines.append("")
        for extra, label in (("from_formula", "from_formula"), ("fit", "fit")):
            value = generated.get(extra)
            if value:
                lines.append(f"**`{name}.{label}`**")
                lines.append("")
                lines.append(_code_block(f"{name}.{label}{value}", "python"))
                lines.append("")

    example = prose.get("example")
    if example:
        lines.append(f"**{words['example']}**")
        lines.append("")
        lines.append(_code_block(str(example), "python"))
        lines.append("")
        if outcome is not None:
            if outcome.get("ok"):
                rendered = str(outcome.get("value") or "").rstrip()
                if rendered:
                    lines.append(f"**{words['output']}**")
                    lines.append("")
                    lines.append(_code_block(rendered, "text"))
                    lines.append("")
            else:
                # A failing example is shown, not hidden: it is a defect in the manual.
                lines.append(f"**⚠️ {outcome.get('error')}**")
                lines.append("")
    else:
        lines.append(f"_{words['no_example']}_")
        lines.append("")

    notes = prose.get(f"notes_{lang}")
    if notes:
        lines.append(f"**{words['notes']}:** {notes}")
        lines.append("")

    return "\n".join(lines)


def render(lang: str, data: dict[str, Any]) -> str:
    words = TEXT[lang]
    outcomes = _run_examples(data)
    generated = {entry["name"]: entry for entry in data["exports"]}
    surface = data.get("result_surface") or {}
    for member in surface.get("methods", []):
        generated.setdefault(member["name"], {"signature": member.get("signature")})
    for member in surface.get("properties", []):
        generated.setdefault(member["name"], {})

    prose = data.get("prose", {})
    in_families = {name for _, _, names in FAMILIES for name in names}

    parts: list[str] = [f"# {words['title']}", ""]
    parts.append(
        f"_{words['version']}: {data.get('westata_version', 'unknown')} · "
        f"{words['generated']} `docs/render_api_reference.py`_"
    )
    parts.append("")
    parts.append(words["intro"])
    parts.append("")

    # Table of contents.
    parts.append(f"## {words['toc']}")
    parts.append("")
    for en_title, zh_title, names in FAMILIES:
        title = en_title if lang == "en" else zh_title
        anchors = ", ".join(f"[`{name}`](#{_anchor(name)})" for name in names if name in prose or name in generated)
        if anchors:
            parts.append(f"- **{title}** — {anchors}")
    leftovers = sorted(set(prose) - in_families)
    if leftovers:
        label = "Other names" if lang == "en" else "其他名称"
        parts.append(f"- **{label}** — " + ", ".join(f"[`{n}`](#{_anchor(n)})" for n in leftovers))
    parts.append("")

    for en_title, zh_title, names in FAMILIES:
        title = en_title if lang == "en" else zh_title
        parts.append(f"## {title}")
        parts.append("")
        for name in names:
            block = _entry_block(
                name, prose.get(name), generated.get(name), outcomes.get(name), lang
            )
            parts.append(block)
        parts.append("")

    if leftovers:
        label = "Other names" if lang == "en" else "其他名称"
        parts.append(f"## {label}")
        parts.append("")
        for name in leftovers:
            parts.append(
                _entry_block(name, prose.get(name), generated.get(name), outcomes.get(name), lang)
            )
        parts.append("")

    return "\n".join(parts)


def _anchor(name: str) -> str:
    """GitHub's heading anchor for a ``### `name` `` heading.

    GitHub lower-cases the heading, drops everything that is not alphanumeric/space/hyphen
    (so backticks vanish), and turns spaces into hyphens. Underscores are KEPT, which is
    why ``estat_ic`` anchors as ``estat_ic`` and not ``estatic``.
    """

    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9_ \-]", "", slug)
    return slug.replace(" ", "-")


def main() -> int:
    data = _load()
    for lang, filename in (("en", "API.md"), ("zh", "API.zh-CN.md")):
        document = render(lang, data)
        target = ROOT / "docs" / filename
        target.write_text(document, encoding="utf-8")
        print(f"wrote {target} ({len(document.splitlines())} lines)")

    # Report coverage so an incomplete manual is visible at build time, not discovered by
    # a reader. The numbers are printed, not written into the document: a stale percentage
    # inside the file is exactly the kind of claim this project keeps having to correct.
    prose = set(data.get("prose", {}))
    generated = {entry["name"] for entry in data["exports"]}
    surface = data.get("result_surface") or {}
    generated |= {m["name"] for m in surface.get("methods", [])}
    generated |= {p["name"] for p in surface.get("properties", [])}
    generated.add("StataResults")
    documented = generated & prose
    print(
        f"prose coverage: {len(documented)}/{len(generated)} "
        f"({100.0 * len(documented) / max(len(generated), 1):.1f}%)"
    )
    missing = sorted(generated - prose)
    if missing:
        print("names still without prose:")
        for name in missing:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
