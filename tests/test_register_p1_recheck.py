"""The gap register's P1 recheck section must stay in place.

The report's P1 list (lines 620-628) is a 0.1.0 snapshot. Round 145 verified item by item that
six of its entries have since been fixed -- rdrobust covs(), psmatch caliper, all five vecrank
trend() options, xtreg mle, prais corc, and the IRF/FEVD numerics -- while three remain genuine
open items (automatic bandwidth, synth, xtserial) or honest refusals (AI-robust variance).

Without section 3.6.1 a future reader starts from the stale list and re-investigates work that is
done. This guard is deliberately about the DOCUMENT, not the code: it asserts the recheck exists
and names the items that must not be described as missing any more.
"""

from __future__ import annotations

from pathlib import Path

REGISTER = Path(__file__).resolve().parents[1] / "docs" / "GAP_REGISTER.md"


def test_the_p1_recheck_section_exists() -> None:
    text = REGISTER.read_text(encoding="utf-8")
    assert "3.6.1 报告 P1 清单实测复核" in text


def test_the_recheck_records_what_was_fixed() -> None:
    text = REGISTER.read_text(encoding="utf-8")
    section = text.split("3.6.1 报告 P1 清单实测复核", 1)[1].split("### 3.7", 1)[0]
    for item in (
        "C16-01/02/03",
        "C15-03",
        "C14-05",
        "C12-05",
        "C08-05",
        "C13-06",
        "C18-01/02",
    ):
        assert item in section, item
    # The three that are genuinely still open must be described as such, not as done.
    assert "仍缺" in section
    assert "无参照可验证" in section
    assert "明确拒绝" in section or "仍明确拒绝" in section


def test_the_stale_psmatch_row_was_corrected() -> None:
    text = REGISTER.read_text(encoding="utf-8")
    assert "| `teffects psmatch` / `ipwra` | ⬜" not in text
    assert "variance_matches_stata=False" in text


def test_the_recheck_warns_that_scores_are_not_a_work_order() -> None:
    text = REGISTER.read_text(encoding="utf-8")
    assert "不能当作工单" in text


def test_the_extension_recheck_records_the_honest_gaps() -> None:
    """X05's structural gaps were REAL and are now fixed; the deliberate differences remain."""

    text = REGISTER.read_text(encoding="utf-8")
    assert "3.6.2 扩展项 X01 / X05 实测复核" in text
    section = text.split("3.6.2 扩展项 X01 / X05 实测复核", 1)[1].split("### 3.7", 1)[0]
    assert "e(beta)" in section
    assert "e(sample)" in section
    assert "varstable" in section
    # Round 156 closed the two structural gaps; the entry must say so rather than still
    # describing them as missing.
    assert "第 156 轮补齐结构缺口" in section
    assert "缺 `e(beta)`" not in section
    # The remaining differences are deliberate and must stay described as such.
    assert "有意保留" in section


def test_the_three_terminal_states_are_recorded() -> None:
    """Round 149's classification, corrected in round 152 after dfgls proved it wrong."""

    text = REGISTER.read_text(encoding="utf-8")
    assert "3.6.3 剩余缺口的三种终结状态" in text
    section = text.split("3.6.3 剩余缺口的三种终结状态", 1)[1].split("### 3.7", 1)[0]
    for item in ("二进制 Mata", "本环境无参照", "rddensity", "xtserial", "dfgls", "rcm", "rdrobust"):
        assert item in section, item
    assert "反编译" in section


def test_dfgls_is_no_longer_filed_under_the_mata_core_class() -> None:
    """Round 151 proved the five-round dfgls failure was an inverted DEFAULT, not Mata.

    The correction matters: filing "my reconstruction disagrees with Stata" under the
    binary-core class turns a checkable mistake into an unfalsifiable excuse.
    """

    text = REGISTER.read_text(encoding="utf-8")
    section = text.split("3.6.3 剩余缺口的三种终结状态", 1)[1].split("### 3.7", 1)[0]
    class_a = [line for line in section.splitlines() if line.startswith("| **A.")]
    assert class_a, "class A row missing"
    assert "dfgls" not in class_a[0], "dfgls must not be listed as a binary-Mata item"
    assert "默认选项写反" in section
    assert "不要把" in section and "直接归入 A 类" in section


def test_dfgls_left_the_class_c_list_after_round_154() -> None:
    """The critical-value columns were the last dfgls gap; they are now implemented."""

    text = REGISTER.read_text(encoding="utf-8")
    section = text.split("3.6.3 剩余缺口的三种终结状态", 1)[1].split("### 3.7", 1)[0]
    class_c = [line for line in section.splitlines() if line.startswith("| **C.")]
    assert class_c, "class C row missing"
    assert "dfgls" not in class_c[0], "dfgls critical columns are implemented now"
    assert "已于第 154 轮完全移出本表" in section
    # And the register's own row must not still advertise the columns as missing.
    assert "1%/5%/10% 临界值列未实现" not in text


def test_the_summary_retracts_the_builtin_command_claim() -> None:
    """'dfgls is builtin, so no source exists' was false and steered five rounds wrong."""

    summary = (REGISTER.parent / "STATA_ALIGNMENT_SUMMARY.md").read_text(encoding="utf-8")
    assert "以上两段已被第 151 轮证伪" in summary
    assert "ado/base/d/dfgls.ado" in summary
    assert "从未被单独验证过的" in summary
