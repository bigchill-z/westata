"""Parser and evaluator for Stata's ``exp_list`` grammar.

``[U] exp_list`` defines the grammar completely, and every production is listable:

    exp_list -> ( name : elist )
    elist    -> newvar = (exp) | (exp)
    exp      -> a standard Stata expression
    eexp     -> specname | [eqno] specname
    specname -> _b | _b[] | _se | _se[] | r(stat) | e(stat)

That is what makes this a bounded task rather than an open-ended interpreter: it is not
"evaluate arbitrary Stata", it is a small specified grammar over stored results.

The manual's own examples define the required subset::

    bootstrap (location: mean=r(mean) median=r(p50))
              (scale: sd=r(sd) iqr=(r(p75)-r(p25)) range=(r(max)-r(min)))
              : summarize price, detail

    bootstrap or=(exp(_b[mpg])): logit foreign mpg weight

    permute price _b rmse=(e(rmse)): regress price trunk

So: numbers, ``r(name)`` and ``e(name)``, ``_b[name]`` and ``_se[name]``, parentheses, the
four arithmetic operators with ``^``, unary minus, and a handful of functions. Constructs
outside that set raise and name themselves, because a silently wrong number is the one
outcome an evidence-graded library cannot afford.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "ExpListError",
    "ParsedExpressionList",
    "StoredResults",
    "evaluate_statistic",
    "parse_exp_list",
]


class ExpListError(ValueError):
    """Raised when text does not conform to Stata's ``exp_list`` grammar."""


@dataclass(frozen=True)
class StoredResults:
    """The stored results one replicate's command leaves behind.

    ``r`` and ``e`` hold scalars; ``b`` and ``se`` map coefficient names. These are exactly
    the four tables ``specname`` can reach.
    """

    r: Mapping[str, float] = field(default_factory=dict)
    e: Mapping[str, float] = field(default_factory=dict)
    b: Mapping[str, float] = field(default_factory=dict)
    se: Mapping[str, float] = field(default_factory=dict)

    def lookup(self, table: str, key: str) -> float:
        available = {"r": self.r, "e": self.e, "b": self.b, "se": self.se}[table]
        if key not in available:
            raise ExpListError(
                f"{table}({key}) was not stored; available: {sorted(available)}"
            )
        return float(available[key])


# ------------------------------------------------------------------ lexer

_TOKEN = re.compile(
    r"""
    \s*(?:
        (?P<number>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)
      | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
      | (?P<op>[-+*/^(),\[\]:=])
    )
    """,
    re.VERBOSE,
)

_STAT_TABLES = {"r", "e"}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if not match:
            raise ExpListError(
                f"unexpected character {text[position]!r} at offset {position} in {text!r}"
            )
        position = match.end()
        tokens.append((match.lastgroup or "", match.group(match.lastgroup or 0)))
    return tokens


# ------------------------------------------------------------- parser core


class _Parser:
    """Recursive descent over the grammar, one method per production.

    Precedence, loosest first: ``+ -``, then ``* /``, then ``^`` (right associative), then
    unary sign, then atoms.
    """

    def __init__(self, tokens: Sequence[tuple[str, str]], text: str) -> None:
        self._tokens = list(tokens)
        self._at = 0
        self._text = text

    # -- cursor ------------------------------------------------------------
    def _peek(self) -> tuple[str, str] | None:
        return self._tokens[self._at] if self._at < len(self._tokens) else None

    def _next(self) -> tuple[str, str]:
        token = self._peek()
        if token is None:
            raise ExpListError(f"unexpected end of {self._text!r}")
        self._at += 1
        return token

    def _accept(self, value: str) -> bool:
        token = self._peek()
        if token is not None and token[1] == value:
            self._at += 1
            return True
        return False

    def _require(self, value: str) -> None:
        token = self._next()
        if token[1] != value:
            raise ExpListError(
                f"expected {value!r} but found {token[1]!r} in {self._text!r}"
            )

    # -- expressions -------------------------------------------------------
    def expression(self) -> Any:
        node = self._additive()
        if self._peek() is not None:
            raise ExpListError(
                f"trailing {self._peek()[1]!r} in {self._text!r}"
            )
        return node

    def _additive(self) -> Any:
        node = self._multiplicative()
        while True:
            token = self._peek()
            if token is None or token[1] not in ("+", "-"):
                return node
            operator = self._next()[1]
            node = ("binary", operator, node, self._multiplicative())

    def _multiplicative(self) -> Any:
        node = self._power()
        while True:
            token = self._peek()
            if token is None or token[1] not in ("*", "/"):
                return node
            operator = self._next()[1]
            node = ("binary", operator, node, self._power())

    def _power(self) -> Any:
        node = self._unary()
        if self._accept("^"):
            node = ("binary", "^", node, self._power())
        return node

    def _unary(self) -> Any:
        token = self._peek()
        if token is not None and token[1] in ("-", "+"):
            operator = self._next()[1]
            operand = self._unary()
            return ("negate", operand) if operator == "-" else operand
        return self._atom()

    def _atom(self) -> Any:
        token = self._next()
        kind, value = token

        if kind == "number":
            return ("number", float(value))

        if value == "(":
            node = self._additive()
            self._require(")")
            return node

        if kind == "name":
            return self._named(value)

        raise ExpListError(f"unexpected {value!r} in {self._text!r}")

    def _named(self, value: str) -> Any:
        lowered = value.lower()

        # A name followed by '(' is a function call.
        if self._peek() is not None and self._peek()[1] == "(":
            return self._function(lowered)

        if lowered in ("_b", "_se"):
            return self._coefficient(lowered)

        raise ExpListError(
            f"{value!r} is not a stored-result reference; expected r(...), e(...), _b[...], "
            f"_se[...], a function, or a number, in {self._text!r}"
        )

    def _coefficient(self, lowered: str) -> Any:
        table = "b" if lowered == "_b" else "se"
        if self._accept("["):
            name = self._next()
            if name[0] != "name":
                raise ExpListError(
                    f"expected a coefficient name inside {lowered}[...] in {self._text!r}"
                )
            self._require("]")
            return ("lookup", table, name[1])
        # Bare _b is legal in Stata but ambiguous here (which coefficient?), so it is
        # rejected with an explanation rather than guessed at.
        raise ExpListError(
            f"bare {lowered} is ambiguous; write {lowered}[name] in {self._text!r}"
        )

    def _function(self, lowered: str) -> Any:
        self._require("(")

        if lowered in _STAT_TABLES:
            # r(...) / e(...) take a bare statistic name, which _atom would reject, so the
            # argument is read here rather than through the expression grammar.
            token = self._next()
            if token[0] != "name":
                raise ExpListError(
                    f"{lowered}() needs a statistic name such as {lowered}(mean), "
                    f"in {self._text!r}"
                )
            name = token[1]
            self._require(")")
            return ("stat", lowered, name)

        argument = self._additive()
        self._require(")")
        return ("call", lowered, argument)


# ------------------------------------------------------------- exp_list


@dataclass(frozen=True)
class Statistic:
    """One reported statistic: its name and the parsed expression producing it."""

    name: str
    expression: Any


@dataclass(frozen=True)
class StatisticGroup:
    """A parenthesised group, i.e. one equation's worth of statistics."""

    label: str | None
    statistics: tuple[Statistic, ...]


@dataclass(frozen=True)
class ParsedExpressionList:
    """A parsed ``exp_list``, ready to evaluate against one replicate's results."""

    groups: tuple[StatisticGroup, ...]

    @property
    def names(self) -> list[str]:
        """Flattened statistic names, labelled by group when a group has a label."""

        output: list[str] = []
        for group in self.groups:
            for statistic in group.statistics:
                output.append(
                    f"{group.label}:{statistic.name}"
                    if group.label is not None
                    else statistic.name
                )
        return output

    def evaluate(self, results: StoredResults) -> dict[str, float]:
        """Evaluate every statistic, returning the flattened name -> value mapping."""

        output: dict[str, float] = {}
        for group in self.groups:
            for statistic in group.statistics:
                key = (
                    f"{group.label}:{statistic.name}"
                    if group.label is not None
                    else statistic.name
                )
                output[key] = evaluate_statistic(statistic.expression, results)
        return output


def evaluate_statistic(node: Any, results: StoredResults) -> float:
    """Evaluate one parsed expression against a replicate's stored results."""

    kind = node[0]

    if kind == "number":
        return float(node[1])

    if kind == "negate":
        return -evaluate_statistic(node[1], results)

    if kind == "lookup":
        return results.lookup(node[1], node[2])

    if kind == "stat":
        return results.lookup(node[1], node[2])

    if kind == "call":
        _, function, argument = node
        value = evaluate_statistic(argument, results)
        if function == "exp":
            return math.exp(value)
        if function == "ln":
            if value <= 0:
                raise ExpListError("ln() of a non-positive value")
            return math.log(value)
        if function == "sqrt":
            if value < 0:
                raise ExpListError("sqrt() of a negative value")
            return math.sqrt(value)
        if function == "abs":
            return abs(value)
        raise ExpListError(
            f"function {function!r} is not implemented; supported: exp, ln, sqrt, abs"
        )

    if kind == "binary":
        _, operator, left, right = node
        a = evaluate_statistic(left, results)
        b = evaluate_statistic(right, results)
        if operator == "+":
            return a + b
        if operator == "-":
            return a - b
        if operator == "*":
            return a * b
        if operator == "/":
            if b == 0.0:
                raise ExpListError("division by zero")
            return a / b
        if operator == "^":
            return a**b
        raise ExpListError(f"operator {operator!r} is not implemented")

    raise ExpListError(f"internal: unknown node {kind!r}")


def parse_exp_list(text: str) -> ParsedExpressionList:
    """Parse an ``exp_list`` per ``[U] exp_list``.

    Accepts either the fully parenthesised form (``(location: mean=r(mean))``) or the bare
    form Stata allows at top level (``or=(exp(_b[mpg]))``, ``_b[x1]``), which is what the
    manual's later examples use.
    """

    source = text.strip()
    if not source:
        raise ExpListError("empty expression list")

    tokens = _tokenize(source)

    # Split at top level on '(' that begin a group. A leading '(' plus a ':' before the
    # matching ')' marks a labelled group; otherwise the whole text is one anonymous group.
    groups: list[StatisticGroup] = []
    position = 0

    while position < len(tokens):
        if tokens[position][1] == "(":
            # Find the matching close paren.
            depth = 0
            for index in range(position, len(tokens)):
                if tokens[index][1] == "(":
                    depth += 1
                elif tokens[index][1] == ")":
                    depth -= 1
                    if depth == 0:
                        break
            else:
                raise ExpListError(f"unbalanced parentheses in {source!r}")

            inner = tokens[position + 1 : index]
            position = index + 1

            # A label is present when the first token is a name followed by ':'.
            label = None
            if len(inner) >= 2 and inner[0][0] == "name" and inner[1][1] == ":":
                label = inner[0][1]
                inner = inner[2:]

            groups.append(
                StatisticGroup(label=label, statistics=_parse_statements(inner, source))
            )
        else:
            # Bare form: treat the rest as one anonymous group, honouring top-level commas
            # only if they appear inside the parenthesised syntax above.
            remaining = tokens[position:]
            groups.append(
                StatisticGroup(
                    label=None, statistics=_parse_statements(remaining, source)
                )
            )
            position = len(tokens)

    if not groups or all(not group.statistics for group in groups):
        raise ExpListError(f"no statistics found in {source!r}")

    return ParsedExpressionList(groups=tuple(groups))


def _parse_statements(tokens: Sequence[tuple[str, str]], text: str) -> tuple[Statistic, ...]:
    """Split a group body into statistics.

    Statements are separated by a top-level comma, or simply by juxtaposition -- Stata's own
    example writes ``(location: mean=r(mean) median=r(p50))`` with no separator at all. A new
    statement starts at a top-level ``name =``, or at a top-level ``(`` once something has
    already been collected.
    """

    statements: list[Statistic] = []
    chunk: list[tuple[str, str]] = []

    def flush() -> None:
        if not chunk:
            return
        statements.append(_parse_statement(list(chunk), text))
        chunk.clear()

    depth = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token[1] in ("(", "["):
            # A '(' at depth 0 after a complete statement begins the next one.
            if depth == 0 and chunk:
                following = tokens[index + 1] if index + 1 < len(tokens) else None
                labelled = (
                    following is not None
                    and following[0] == "name"
                    and index + 2 < len(tokens)
                    and tokens[index + 2][1] == ":"
                )
                if labelled:
                    flush()
            depth += 1
        elif token[1] in (")", "]"):
            depth -= 1

        if depth == 0 and token[1] == ",":
            flush()
            index += 1
            continue

        # A top-level "name =" always starts a new statement.
        if (
            depth == 0
            and token[0] == "name"
            and index + 1 < len(tokens)
            and tokens[index + 1][1] == "="
            and chunk
        ):
            flush()

        chunk.append(token)
        index += 1

    flush()

    if not statements:
        raise ExpListError(f"no statistics found in {text!r}")
    return tuple(statements)


def _parse_statement(tokens: list[tuple[str, str]], text: str) -> Statistic:
    """Parse ``name = (exp)`` or ``(exp)``."""

    if (
        len(tokens) >= 3
        and tokens[0][0] == "name"
        and tokens[1][1] == "="
    ):
        name = tokens[0][1]
        body = tokens[2:]
    else:
        name = ""
        body = tokens

    # Strip one wrapping pair of parentheses when it encloses the whole body.
    if body and body[0][1] == "(" and body[-1][1] == ")":
        depth = 0
        encloses = True
        for index, token in enumerate(body):
            if token[1] == "(":
                depth += 1
            elif token[1] == ")":
                depth -= 1
                if depth == 0 and index != len(body) - 1:
                    encloses = False
                    break
        if encloses:
            body = body[1:-1]

    if not body:
        raise ExpListError(f"empty expression in {text!r}")

    parser = _Parser(body, text)
    expression = parser.expression()

    if not name:
        # Stata names an unnamed statistic after its expression; reproduce the common cases
        # so the result still has a usable label.
        name = _default_name(body)

    return Statistic(name=name, expression=expression)


def _default_name(tokens: Sequence[tuple[str, str]]) -> str:
    """Stata's default label for an unnamed statistic."""

    literal = "".join(token[1] for token in tokens)
    return literal if literal else "stat"
