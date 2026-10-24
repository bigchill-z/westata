"""确保 `__version__` 与 pyproject.toml 不会再次分叉。

发现过程值得记下来：从 0.0.2 的 sdist 安装后，`pip show westata` 报 **0.0.2**，而
`westata.__version__` 报 **0.0.1**。原因是 `__init__.py` 里硬编码了一份版本号，`pyproject.toml` 里
另有一份，两者各自维护。

这类漂移特别难发现，因为：
  * 两处都是"看起来正确"的字符串；
  * 没有报错，只是拿到旧值；
  * 版本号只在发布时才改，所以漂移窗口很短，但一旦漂移，任何按 `__version__` 做兼容判断或写日志
    的代码都会拿到错的那个。

现在的实现从 `importlib.metadata` 读取，元数据由 pip 在安装时写入，是权威来源。这个测试断言它与
pyproject 声明一致，从而保证两份永远同步。
"""

from __future__ import annotations

import re

import westata

REPO_VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)


def test_runtime_version_matches_pyproject() -> None:
    """`westata.__version__` 必须等于 pyproject.toml 里声明的版本。"""

    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = REPO_VERSION_PATTERN.search(pyproject.read_text(encoding="utf-8"))
    assert match, "pyproject.toml 里找不到 version"
    declared = match.group(1)

    assert westata.__version__ == declared, (
        f"westata.__version__ = {westata.__version__!r}，"
        f"pyproject.toml 声明 {declared!r}。"
        "两者必须一致：版本号以前硬编码在 __init__.py 里，与 pyproject 漂移过。"
    )


def test_version_is_a_release_like_string() -> None:
    """版本号必须是可解析的形式，否则 PyPI 会拒绝，或在比较时出意外。"""

    parts = westata.__version__.split(".")
    assert len(parts) >= 2, f"版本号 {westata.__version__!r} 不像语义化版本"
    assert all(part.isdigit() for part in parts[:2]), (
        f"版本号 {westata.__version__!r} 的前两段应为数字"
    )
