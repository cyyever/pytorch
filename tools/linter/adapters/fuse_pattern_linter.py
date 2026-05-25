"""Flags fuse-able / more-idiomatic PyTorch patterns via AST.

Pattern list is auto-generated from `PATTERNS` below. See module docstring
at runtime (`fuse_pattern_linter.__doc__`) for the user-facing summary.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


LINTER_NAME = "FUSE_PATTERNS"

# torch.<old> -> (replacement, optional caveat). Caveat is set when the migration
# is not a 1:1 rename (signature changes, rank-dependent semantics, etc.).
DEPRECATED_FUNCS: dict[str, tuple[str, str | None]] = {
    "range": (
        "torch.arange",
        "torch.arange excludes the endpoint; adjust the upper bound",
    ),
    "norm": (
        "torch.linalg.vector_norm / matrix_norm / norm",
        "signature-dependent; not a 1:1 rename - pick by rank and `p` value",
    ),
    "chain_matmul": (
        "torch.linalg.multi_dot",
        "pass the matrices as one sequence: multi_dot([a, b, c])",
    ),
    "lu": ("torch.linalg.lu_factor", None),
    "cholesky": (
        "torch.linalg.cholesky",
        "default `upper=False` matches; if upper=True is passed it stays the same",
    ),
    "lu_solve": (
        "torch.linalg.lu_solve",
        "argument order changes: LU and pivots come first, B goes last",
    ),
    "qr": ("torch.linalg.qr", "`some=True/False` becomes `mode='reduced'/'complete'`"),
    "triangular_solve": (
        "torch.linalg.solve_triangular",
        "argument order changes: B comes first; `unitriangular`/`transpose` are kwargs",
    ),
}


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


@dataclass(frozen=True)
class Pattern:
    """A single fuse-pattern rule.

    `summary` is the one-line entry rendered in the module docstring.
    `name` is the short string put into LintMessage.name (downstream-visible).
    `description` is the long-form advice text.
    `node_type` is the AST type the matcher applies to (ast.Call or ast.BinOp).
    `matches` returns True when `node` is an instance of the pattern.
    """

    summary: str
    name: str
    description: str
    node_type: type[ast.AST]
    # False = no match; True = match (use `description`); str = match with custom description.
    matches: Callable[[ast.AST], bool | str]


# ---- Matchers ---------------------------------------------------------------


def _is_torch_attr(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr == attr
    )


def _is_torch_call(node: ast.AST, attr: str) -> bool:
    return isinstance(node, ast.Call) and _is_torch_attr(node.func, attr)


def _is_torchy_call(node: ast.AST, name: str) -> bool:
    """`<recv>.<name>(...)` where recv is a bare name (torch/F/tensor var) or a
    dotted path rooted at torch (e.g. torch.nn.functional.<name>). Rejects
    unrelated libraries such as scipy.special.<name>.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ):
        return False
    recv = node.func.value
    if isinstance(recv, ast.Name):
        return True
    while isinstance(recv, ast.Attribute):
        recv = recv.value
    return isinstance(recv, ast.Name) and recv.id == "torch"


def _is_one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in (1, 1.0)


def _is_num_literal(node: ast.AST) -> bool:
    """A numeric literal, including a signed one like -1 (UnaryOp over Constant)."""
    if isinstance(node, ast.Constant):
        return True
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
    )


def _has_out_kwarg(node: ast.AST) -> bool:
    """Any call in the subtree writes to an out= buffer. A rewrite that replaces
    the expression would silently drop that write, so such matches are skipped.
    """
    return any(
        isinstance(n, ast.Call) and any(kw.arg == "out" for kw in n.keywords)
        for n in ast.walk(node)
    )


def _comm_binop(
    node: ast.AST,
    op_type: type[ast.operator],
    pred_a: Callable[[ast.AST], bool],
    pred_b: Callable[[ast.AST], bool],
) -> bool:
    """Match `a OP b` for commutative OP; (pred_a, pred_b) applies in either order."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, op_type)):
        return False
    return (pred_a(node.left) and pred_b(node.right)) or (
        pred_a(node.right) and pred_b(node.left)
    )


def _is_zero_arg_method_call(node: ast.AST, attr: str) -> bool:
    """`x.<attr>()` with no args/kwargs."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and not node.args
        and not node.keywords
    )


def _zero_arg_method_chain(node: ast.Call, outer: str, inner: str) -> bool:
    """`x.<inner>().<outer>(...)` - inner must be zero-arg."""
    if not isinstance(node.func, ast.Attribute) or node.func.attr != outer:
        return False
    return _is_zero_arg_method_call(node.func.value, inner)


# ---- Pattern matchers (Call) ------------------------------------------------


def _m_contig_reshape(node: ast.AST) -> bool:
    """x.contiguous().reshape(...)."""
    return isinstance(node, ast.Call) and _zero_arg_method_chain(
        node, "reshape", "contiguous"
    )


def _m_double_contig(node: ast.AST) -> bool:
    """x.contiguous().contiguous()."""
    match node:
        case ast.Call(
            func=ast.Attribute(
                attr="contiguous",
                value=ast.Call(
                    func=ast.Attribute(attr="contiguous"), args=[], keywords=[]
                ),
            ),
            args=[],
            keywords=[],
        ):
            return True
    return False


def _m_deprecated_torch(node: ast.AST) -> bool | str:
    """torch.<deprecated_op>(...) - emits a per-op message naming the replacement."""
    match node:
        case ast.Call(func=ast.Attribute(value=ast.Name(id="torch"), attr=old)) if (
            old in DEPRECATED_FUNCS
        ):
            new, caveat = DEPRECATED_FUNCS[old]
            msg = f"`torch.{old}` is deprecated; use `{new}` instead."
            if caveat:
                msg += f" Note: {caveat}."
            return msg
    return False


def _torch_log_arg(node: ast.AST) -> ast.expr | None:
    """The single positional arg of `torch.log(arg)`, else None. Returns None if
    any call in the subtree writes out= (a log-rewrite would drop that write).
    """
    match node:
        case ast.Call(
            func=ast.Attribute(value=ast.Name(id="torch"), attr="log"), args=[arg]
        ) if not _has_out_kwarg(node):
            return arg
    return None


def _m_softplus(node: ast.AST) -> bool:
    """torch.log(1 + torch.exp(x)) or torch.log(torch.exp(x) + 1)."""
    arg = _torch_log_arg(node)
    return arg is not None and _comm_binop(
        arg, ast.Add, _is_one, lambda e: _is_torch_call(e, "exp")
    )


def _m_log1p(node: ast.AST) -> bool:
    """torch.log(1 + x) or torch.log(x + 1) - softplus shape is a strict subset."""
    arg = _torch_log_arg(node)
    return arg is not None and _comm_binop(arg, ast.Add, _is_one, lambda _: True)


def _m_log1p_neg(node: ast.AST) -> bool:
    """torch.log(1 - x) -> log1p(-x). Not commutative: only `1 - x`."""
    arg = _torch_log_arg(node)
    return (
        isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Sub) and _is_one(arg.left)
    )


def _m_log_softmax(node: ast.AST) -> bool:
    """torch.log(<softmax>(...)) - torch/F/method softmax."""
    arg = _torch_log_arg(node)
    return arg is not None and _is_torchy_call(arg, "softmax")


def _m_logsigmoid(node: ast.AST) -> bool:
    """torch.log(<sigmoid>(...))."""
    arg = _torch_log_arg(node)
    return arg is not None and _is_torchy_call(arg, "sigmoid")


def _m_logsumexp(node: ast.AST) -> bool:
    """torch.log(torch.sum(torch.exp(x), ...)) or torch.log(x.exp().sum(...))."""
    arg = _torch_log_arg(node)
    if not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute)):
        return False
    if arg.func.attr != "sum":
        return False
    summed = (arg.args[:1] if arg.args else []) + [arg.func.value]
    return any(_is_torchy_call(c, "exp") for c in summed)


# ---- Pattern matchers (BinOp) -----------------------------------------------


def _m_sigmoid(node: ast.AST) -> bool:
    """1 / (1 + torch.exp(-x))."""

    def _is_neg_exp(e: ast.AST) -> bool:
        match e:
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id="torch"), attr="exp"),
                args=[ast.UnaryOp(op=ast.USub())],
            ):
                return True
        return False

    match node:
        case ast.BinOp(op=ast.Div(), left=ast.Constant(value=1 | 1.0), right=denom):
            return _comm_binop(
                denom, ast.Add, _is_one, _is_neg_exp
            ) and not _has_out_kwarg(node)
    return False


def _m_rsqrt(node: ast.AST) -> bool:
    """1 / torch.sqrt(x). (x ** -0.5 is not included: pow already special-cases it.)"""
    match node:
        case ast.BinOp(
            op=ast.Div(),
            left=ast.Constant(value=1 | 1.0),
            right=ast.Call(func=ast.Attribute(value=ast.Name(id="torch"), attr="sqrt")),
        ):
            return not _has_out_kwarg(node)
    return False


def _m_silu(node: ast.AST) -> bool:
    """x * torch.sigmoid(x) or torch.sigmoid(x) * x.

    The two operands must be structurally identical AND side-effect-free: the
    rewrite to F.silu(x) evaluates x once, so collapsing a call (RNG, next(),
    pop()) that was evaluated twice would change behavior.
    """
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
        return False
    for maybe_sig, other in ((node.left, node.right), (node.right, node.left)):
        if (
            isinstance(maybe_sig, ast.Call)
            and isinstance(maybe_sig.func, ast.Attribute)
            and maybe_sig.func.attr == "sigmoid"
            and len(maybe_sig.args) == 1
            and not maybe_sig.keywords
            and ast.dump(maybe_sig.args[0]) == ast.dump(other)
            and not any(isinstance(n, ast.Call) for n in ast.walk(other))
        ):
            return True
    return False


def _m_expm1(node: ast.AST) -> bool:
    """torch.exp(x) - 1."""
    match node:
        case ast.BinOp(
            op=ast.Sub(),
            left=ast.Call(func=ast.Attribute(value=ast.Name(id="torch"), attr="exp")),
            right=ast.Constant(value=1 | 1.0),
        ):
            return not _has_out_kwarg(node)
    return False


def _m_matmul_add(node: ast.AST) -> bool:
    """a @ b + c  or  c + a @ b, where c is a non-scalar bias.

    Rank decides addmm (2-D) vs baddbmm (batched) and we can't see it
    statically, so the single advice lists both. A bare literal bias
    (`a @ b + 1`) is an add-scalar, not a GEMM bias, so it is rejected.
    """
    return _comm_binop(
        node,
        ast.Add,
        lambda e: isinstance(e, ast.BinOp) and isinstance(e.op, ast.MatMult),
        lambda e: not _is_num_literal(e),
    )


# ---- The registry -----------------------------------------------------------
# Order = evaluation order; first match per node wins. Softplus must precede
# log1p (its arg shape is a subset).

PATTERNS: list[Pattern] = [
    Pattern(
        summary=".contiguous().reshape(...)           -> drop .contiguous()",
        name="drop redundant .contiguous() before .reshape()",
        description="Drop the leading `.contiguous()`; `.reshape` already handles non-contiguous input.",
        node_type=ast.Call,
        matches=_m_contig_reshape,
    ),
    Pattern(
        summary=".contiguous().contiguous()           -> drop redundant call",
        name="redundant .contiguous().contiguous()",
        description="Second `.contiguous()` is a no-op; drop it.",
        node_type=ast.Call,
        matches=_m_double_contig,
    ),
    Pattern(
        summary="torch.<deprecated>(...)              -> see message for replacement",
        name="deprecated torch.* op",
        description="(see per-match message)",
        node_type=ast.Call,
        matches=_m_deprecated_torch,
    ),
    Pattern(
        summary="torch.log(1 + torch.exp(x))          -> F.softplus(x)",
        name="prefer F.softplus(x) over torch.log(1 + torch.exp(x))",
        description="Use `F.softplus(x)`: numerically stable, avoids overflow for large x.",
        node_type=ast.Call,
        matches=_m_softplus,
    ),
    Pattern(
        summary="torch.log(1 + x)                     -> torch.log1p(x)",
        name="prefer torch.log1p(x) over torch.log(1 + x)",
        description="Use `torch.log1p(x)`: numerically stable for small x.",
        node_type=ast.Call,
        matches=_m_log1p,
    ),
    Pattern(
        summary="torch.log(1 - x)                     -> torch.log1p(-x)",
        name="prefer torch.log1p(-x) over torch.log(1 - x)",
        description="Use `torch.log1p(-x)`: numerically stable for small x.",
        node_type=ast.Call,
        matches=_m_log1p_neg,
    ),
    Pattern(
        summary="torch.log(torch.softmax(x))          -> torch.log_softmax(x)",
        name="prefer torch.log_softmax over log(softmax)",
        description="Use `torch.log_softmax(x, dim)`: numerically stable vs. softmax then log.",
        node_type=ast.Call,
        matches=_m_log_softmax,
    ),
    Pattern(
        summary="torch.log(torch.sigmoid(x))          -> F.logsigmoid(x)",
        name="prefer F.logsigmoid(x) over log(sigmoid(x))",
        description="Use `F.logsigmoid(x)`: numerically stable for large |x|.",
        node_type=ast.Call,
        matches=_m_logsigmoid,
    ),
    Pattern(
        summary="torch.log(torch.sum(torch.exp(x)))   -> torch.logsumexp(x)",
        name="prefer torch.logsumexp over log(sum(exp))",
        description="Use `torch.logsumexp(x, dim)`: numerically stable log-sum-exp, avoids overflow in exp.",
        node_type=ast.Call,
        matches=_m_logsumexp,
    ),
    Pattern(
        summary="1 / (1 + torch.exp(-x))              -> torch.sigmoid(x)",
        name="prefer torch.sigmoid(x) over 1 / (1 + torch.exp(-x))",
        description="Use `torch.sigmoid(x)`: fused kernel, numerically stable for large |x|.",
        node_type=ast.BinOp,
        matches=_m_sigmoid,
    ),
    Pattern(
        summary="1 / torch.sqrt(x)                    -> torch.rsqrt(x)",
        name="prefer torch.rsqrt(x) over 1 / torch.sqrt(x)",
        description="Use `torch.rsqrt(x)`: single fused reciprocal-sqrt, faster on GPU.",
        node_type=ast.BinOp,
        matches=_m_rsqrt,
    ),
    Pattern(
        summary="x * torch.sigmoid(x)                 -> F.silu(x)",
        name="prefer F.silu(x) over x * torch.sigmoid(x)",
        description="Use `F.silu(x)`: fused SiLU/swish kernel.",
        node_type=ast.BinOp,
        matches=_m_silu,
    ),
    Pattern(
        summary="torch.exp(x) - 1                     -> torch.expm1(x)",
        name="prefer torch.expm1(x) over torch.exp(x) - 1",
        description="Use `torch.expm1(x)`: numerically stable for small x where exp(x) rounds to 1.",
        node_type=ast.BinOp,
        matches=_m_expm1,
    ),
    Pattern(
        summary="a @ b + c                            -> torch.addmm / torch.baddbmm",
        name="consider torch.addmm/baddbmm (matmul+bias)",
        description="Fuse matmul+bias into one GEMM: `torch.addmm(c, a, b)` if `a`, `b` are 2-D, or `torch.baddbmm(c, a, b)` if 3-D batched (bias must match `a@b`'s shape).",
        node_type=ast.BinOp,
        matches=_m_matmul_add,
    ),
]


def _render_docstring() -> str:
    lines = ["Flags fuse-able / more-idiomatic PyTorch patterns via AST:", ""]
    lines += [f"  - {p.summary}" for p in PATTERNS]
    return "\n".join(lines) + "\n"


# Replace the placeholder module docstring with the registry-derived one so
# `python -c 'import fuse_pattern_linter; print(fuse_pattern_linter.__doc__)'`
# yields the canonical pattern list.
__doc__ = _render_docstring()


# Pre-bucket patterns by node type so the visitor's per-node loop is small.
_PATTERNS_BY_TYPE: dict[type[ast.AST], list[Pattern]] = {}
for _p in PATTERNS:
    _PATTERNS_BY_TYPE.setdefault(_p.node_type, []).append(_p)


class FusePatternVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.messages: list[LintMessage] = []

    def _emit(self, node: ast.expr, name: str, description: str) -> None:
        try:
            snippet: str | None = ast.unparse(node)
        except Exception:
            snippet = None  # never block linting on unparse hiccups
        self.messages.append(
            LintMessage(
                path=self.path,
                line=node.lineno,
                char=node.col_offset + 1,
                code=LINTER_NAME,
                severity=LintSeverity.ADVICE,
                name=name,
                original=snippet,
                replacement=None,
                description=description,
            )
        )

    def _check_patterns(self, node: ast.expr) -> None:
        for pat in _PATTERNS_BY_TYPE.get(type(node), ()):
            result = pat.matches(node)
            if not result:
                continue
            description = result if isinstance(result, str) else pat.description
            self._emit(node, pat.name, description)
            return  # first match per node wins

    def visit_Call(self, node: ast.Call) -> None:
        self._check_patterns(node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._check_patterns(node)
        self.generic_visit(node)


def lint_file(path: str) -> list[LintMessage]:
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return []

    visitor = FusePatternVisitor(path)
    visitor.visit(tree)
    return visitor.messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag fuse-able PyTorch patterns.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    for path in args.filenames:
        for msg in lint_file(path):
            print(json.dumps(msg._asdict()), flush=True)


if __name__ == "__main__":
    main()
