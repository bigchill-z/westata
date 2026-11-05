"""Stata's ``ereturn list``: dump the estimation results as four labelled blocks.

Reference from a live ``regress y k l`` followed by ``ereturn list`` on cobb_douglas.dta::

    scalars:
                      e(N) =  24
                  e(sum_w) =  24
                   e(df_m) =  2
                   e(df_r) =  21
                      e(F) =  171.9277809215706
                     e(r2) =  .9424429769032044
                   e(rmse) =  10.98532816796313
                    e(mss) =  41495.6071992186
                    e(rss) =  2534.226134114726
                   e(r2_a) =  .9369613556558906
                     e(ll) =  -89.9696019953156
                   e(ll_0) =  -124.2293513940736
                   e(rank) =  3

    macros:
                e(cmdline) : "regress y k l"
                  e(title) : "Linear regression"
                  ...

    matrices:
                      e(b) :  1 x 3
                      e(V) :  3 x 3

    functions:

FOUR details that decide whether the output matches:

1. Each block ends with a blank line; the blocks appear in a FIXED order -- scalars, macros,
   matrices, functions -- regardless of the order the estimates were stored in.
2. Scalar lines are ``<name>:>> = <value>`` with the name field right-aligned to 23
   characters and TWO spaces before the ``=``.
3. Scalar values use Stata's ``%18.0g``-style display: ``24`` for e(N) but
   ``.9424429769032044`` for e(r2), i.e. up to 16 significant digits with no leading zero and
   no trailing zeros.
4. Macro lines use a DIFFERENT separator -- ``<name> : "<value>"`` with the name right-aligned
   to 23 and a single space before the colon.

``regress y k l`` stores ``e(sum_w)`` which a Python model may not carry; the block prints
whatever keys the result object actually has, in its own order, rather than inventing keys.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["format_ereturn_list"]

#: Stata prints the blocks in this order no matter how the estimates were stored.
_BLOCK_ORDER = ("scalars", "macros", "matrices", "functions")

_NAME_WIDTH = 23


def _scalar_value(value: Any) -> str:
    """Stata's scalar display: up to 16 significant digits, no leading zero, no padding."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "."
    text = f"{number:.16g}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def format_ereturn_list(result: Any) -> str:
    """Render ``ereturn list`` for a fitted result object.

    ``result`` must expose ``e`` (a mapping). Matrices are discovered through
    ``cov_params()`` (reported as ``e(V)``) and the parameter vector (``e(b)``); a result that
    supplies neither simply gets no matrices block entries.
    """

    e_macros = dict(getattr(result, "e", None) or {})
    # Stata distinguishes scalars from macros by TYPE: strings go under "macros", numbers
    # under "scalars". Splitting on the Python type reproduces that without a hard-coded list
    # of names, which would silently mis-file any key this package adds later.
    scalars: list[tuple[str, Any]] = []
    strings: list[tuple[str, Any]] = []
    for key in e_macros:
        value = e_macros[key]
        if isinstance(value, str):
            strings.append((str(key), value))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            scalars.append((str(key), value))

    matrices: list[tuple[str, str]] = []
    parameters = getattr(result, "params", None)
    if parameters is not None:
        try:
            count = int(np.asarray(parameters).size)
            matrices.append(("b", f"1 x {count}"))
        except (TypeError, ValueError):
            pass
    try:
        covariance = np.asarray(result.cov_params())
    except (AttributeError, TypeError, ValueError):
        covariance = None
    if covariance is not None and covariance.ndim == 2:
        matrices.append(("V", f"{covariance.shape[0]} x {covariance.shape[1]}"))

    blocks: dict[str, list[str]] = {"scalars": [], "macros": [], "matrices": [], "functions": []}
    # Spacing measured from Stata's own output:
    #     '                  e(N) =  24'          -> " = " then TWO spaces before the value
    #     '            e(cmdline) : "regress y k l"'   -> " : " then the quoted value
    #     '                  e(b) :  1 x 3'        -> " : " then TWO spaces, then the shape
    for name, value in scalars:
        blocks["scalars"].append(
            f"{('e(' + name + ')'):>{_NAME_WIDTH}} =  {_scalar_value(value)}"
        )
    for name, value in strings:
        blocks["macros"].append(f"{('e(' + name + ')'):>{_NAME_WIDTH}} : \"{value}\"")
    for name, shape in matrices:
        blocks["matrices"].append(f"{('e(' + name + ')'):>{_NAME_WIDTH}} :  {shape}")

    lines: list[str] = []
    for block in _BLOCK_ORDER:
        lines.append(f"{block}:")
        lines.extend(blocks[block])
        lines.append("")
    return "\n".join(lines).rstrip("\n")
