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
    """X05 is NOT silently upgraded to 'fixed' -- its structural gaps are real."""

    text = REGISTER.read_text(encoding="utf-8")
    assert "3.6.2 扩展项 X01 / X05 实测复核" in text
    section = text.split("3.6.2 扩展项 X01 / X05 实测复核", 1)[1].split("### 3.7", 1)[0]
    assert "e(beta)" in section
    assert "e(sample)" in section
    assert "varstable" in section
    assert "不刷分" in section


def test_the_three_terminal_states_are_recorded() -> None:
    """Round 149's classification: do not re-run the dfgls-style rebuild on class A."""

    text = REGISTER.read_text(encoding="utf-8")
    assert "3.6.3 剩余缺口的三种终结状态" in text
    section = text.split("3.6.3 剩余缺口的三种终结状态", 1)[1].split("### 3.7", 1)[0]
    for item in ("二进制 Mata", "本环境无参照", "rddensity", "xtserial", "dfgls", "rcm", "rdrobust"):
        assert item in section, item
    assert "反编译" in section
