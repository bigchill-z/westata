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

__all__ = ["format_ereturn_list", "standardized_coefficients"]

#: Stata prints the blocks in this order no matter how the estimates were stored.
_BLOCK_ORDER = ("scalars", "macros", "matrices", "functions")

#: Width of the ``e(name)`` field. Captured from a live ``ereturn list`` after ``regress lnw s``:
#:
#:     '                  e(N) =  758'        29 characters
#:     '                  e(F) =  255.7039662336325'   43
#:
#: Eighteen spaces, then "e(N)" (4), then " =  " (4), then the value. That is a 22-wide field for
#: the name; the previous value of 23 put every scalar line one column past Stata's.
_NAME_WIDTH = 22


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


def _standardized_shape(result: Any) -> str | None:
    """The shape string for ``e(beta)``, or ``None`` when the result has no regressors.

    Stata prints ``1 x k-1`` for an OLS fit with a constant and ``1 x k`` without one. The
    standard errors and the matrix values are available from :func:`standardized_coefficients`;
    this helper only decides WHETHER the row appears and how wide it is, so that a model without
    an intercept is not silently given the wrong shape.
    """

    parameters = getattr(result, "params", None)
    if parameters is None:
        return None
    try:
        names = [str(name) for name in parameters.index]
    except AttributeError:
        return None
    constant_like = {"intercept", "_cons", "const", "constant"}
    regressors = [name for name in names if name.strip().lower() not in constant_like]
    if not regressors:
        return None
    return f"1 x {len(regressors)}"


def standardized_coefficients(result: Any) -> Any:
    """``e(beta)``: coefficients from regressing the standardized response on standardized X.

    Equivalent identities, both verified on grilic.dta: standardising the data and refitting gives
    the same numbers as scaling each coefficient by ``sd(x_k)/sd(y)``, because in a regression with
    an intercept ``b_k * sd(x_k)/sd(y)`` IS the standardized coefficient. The design-matrix route
    is used here because it also works for fits whose sample sds are not at hand.

    An earlier draft of this docstring claimed the scaling formula was "wrong with correlated
    regressors" and quoted .77974961 for ``expr``. That was a mis-computation on my part, not a
    property of the formula: the correct scaling gives .18441586, exactly Stata's value, and the
    two routes agree to 8 decimals on every coefficient.
    """

    import pandas as pd

    parameters = result.params
    names = [str(name) for name in parameters.index]
    constant_like = {"intercept", "_cons", "const", "constant"}
    regressors = [name for name in names if name.strip().lower() not in constant_like]
    if not regressors:
        return pd.Series(dtype=float)

    endog = getattr(result, "model", None)
    model = getattr(result, "model", None)
    if model is None:
        raise ConsistencyError(
            "ereturn: e(beta) needs the fitted design matrix; this result does not expose one"
        )
    design = getattr(model, "exog", None)
    response = getattr(model, "endog", None)
    if design is None or response is None:
        raise ConsistencyError(
            "ereturn: e(beta) needs the fitted design matrix; this result does not expose one"
        )
    design = np.asarray(design, dtype=float)
    response = np.asarray(response, dtype=float).reshape(-1)

    def standardise(values: np.ndarray) -> np.ndarray:
        return (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)

    standard_response = standardise(response)
    columns = [names.index(name) for name in regressors]
    standard_design = standardise(design[:, columns])
    coefficients, *_ = np.linalg.lstsq(standard_design, standard_response, rcond=None)
    return pd.Series(coefficients, index=regressors, name="beta")


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
    # ``e(beta)``: standardized coefficients. Stata lists it for OLS-style fits as a 1 x (k-1)
    # matrix -- one column per REGRESSOR, the constant excluded:
    #
    #     e(beta)[1,3]
    #                 s       expr     tenure
    #     r1  .55469752  .18441586   .1487096
    #
    # Computed as the OLS coefficients after standardising the response AND every regressor
    # (sample sd, ddof=1); verified to 8 decimals on grilic.dta. This is equivalent to
    # b_k * sd(x_k)/sd(y) -- both give .55469752 / .18441586 / .1487096 -- and the design route is
    # used because it does not require the sample sds to be at hand.
    beta_shape = _standardized_shape(result)
    if beta_shape is not None:
        matrices.append(("beta", beta_shape))

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
    # ``functions:`` lists the functions a result exposes; Stata prints the name and NOTHING
    # else -- '             e(sample)   ' is 25 characters with no value after it. Every fitted
    # result here knows its estimation sample, so the row is unconditional.
    if parameters is not None:
        blocks["functions"].append(f"{'e(sample)':>{_NAME_WIDTH}}   ")

    lines: list[str] = []
    for block in _BLOCK_ORDER:
        lines.append(f"{block}:")
        lines.extend(blocks[block])
        lines.append("")
    return "\n".join(lines).rstrip("\n")
