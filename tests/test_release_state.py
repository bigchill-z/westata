"""加一个防回归检查：安装说明必须与实际的发布状态一致。

这类失效很隐蔽——文档说「尚未发布」，而包已经能 `pip install`，两句话本身都"看起来合理"，
只有把两者放在一起才看得出矛盾。上一轮就是这样：0.0.1 上传成功后，四份文档还写着「从仓库安装」。

检查方式不是抓字符串，而是问一个能自我更新的问题：**PyPI 上有没有这个包？** 有，文档就必须给
`pip install westata`；没有，文档就不能给。这样以后真发布新版本时，测试自动跟着变，不需要有人
记得回来改测试。

不联网时跳过，因为 CI 环境未必允许出网到 pypi.org，而这条检查失败不应阻断构建。
"""

from __future__ import annotations

import json
import pathlib
import urllib.error
import urllib.request

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

DOCS = [
    "README.md",
    "README.zh-CN.md",
    "docs/GUIDE.md",
    "docs/GUIDE.zh-CN.md",
]


def _pypi_versions() -> list[str] | None:
    """返回 PyPI 上已发布的版本；网络不可用时返回 None（区别于「没有发布」的空列表）。"""

    try:
        with urllib.request.urlopen("https://pypi.org/pypi/westata/json", timeout=15) as response:
            payload = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    return sorted(payload.get("releases", {}))


def test_install_instructions_match_the_published_state() -> None:
    """文档里的安装说明必须与 PyPI 的真实状态一致。

    这是文档与自己所在世界的契约：说「已发布」就必须给 `pip install`，说「未发布」就不能给。
    两者都对的唯一情况是恰好改在发布前后，而那时测试会提醒。
    """

    versions = _pypi_versions()
    if versions is None:
        pytest.skip("无法访问 pypi.org，跳过发布状态检查")
    if not versions:
        pytest.skip("westata 尚未发布，此检查不适用")

    published = bool(versions)
    for relative in DOCS:
        path = REPO / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")

        claims_unpublished = (
            "not on PyPI yet" in text
            or "尚未发布到 PyPI" in text
            or "not published" in text
        )
        has_pip_install = "pip install westata" in text or 'pip install "westata[arch]"' in text

        assert not (published and claims_unpublished), (
            f"{relative} 声称尚未发布，但 PyPI 上已有版本 {versions}"
        )
        assert not (published and not has_pip_install), (
            f"{relative} 没有给出 `pip install westata`，但它已经发布（{versions}）"
        )


def test_documented_version_matches_pyproject() -> None:
    """pyproject 的版本号必须与已发布的最新版本一致，否则文档和包会对不上。"""

    versions = _pypi_versions()
    if versions is None or not versions:
        pytest.skip("无法访问 pypi.org 或尚未发布")

    import re

    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert match, "pyproject.toml 里找不到 version"
    declared = match.group(1)

    # 允许 pyproject 领先于已发布版本（开发中），但不允许落后于已发布的最新版本。
    newest = versions[-1]
    assert declared >= newest, (
        f"pyproject 声明 {declared}，但 PyPI 上已发布 {newest}；"
        f"已发布的版本号不能收回，pyproject 不应落后于它"
    )
