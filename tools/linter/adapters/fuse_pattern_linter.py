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
CHAIN_THRESHOLD = 8

# torch.<name>(...) that always returns contiguous: safe to .view(), no-op to .contiguous().
CREATOR_FUNCS = frozenset(
    {
        "tensor",
        "as_tensor",
        "scalar_tensor",
        "arange",
        "range",
        "linspace",
        "logspace",
        "zeros",
        "ones",
        "full",
        "empty",
        "rand",
        "randn",
        "randint",
        "randperm",
        "bernoulli",
        "poisson",
        "normal",
        "multinomial",
        "cat",
        "stack",
        "hstack",
        "vstack",
        "dstack",
        "column_stack",
        "row_stack",
        "tile",
        "repeat_interleave",
        "eye",
    }
)

# torch.<old> -> (replacement, optional caveat). Caveat is set when the migration
# is not a 1:1 rename (signature changes, rank-dependent semantics, etc.).
DEPRECATED_FUNCS: dict[str, tuple[str, str | None]] = {
    "range": (
        "torch.arange",
        "torch.arange excludes the endpoint; adjust the upper bound",
    ),
    "norm": (
        "torch.linalg.vector_norm / matrix_norm / norm",
        "signature-dependent; not a 1:1 rename — pick by rank and `p` value",
    ),
    "chain_matmul": ("torch.linalg.multi_dot", None),
    "lu": ("torch.linalg.lu_factor", None),
    "cholesky": (
        "torch.linalg.cholesky",
        "default `upper=False` matches; if upper=True is passed it stays the same",
    ),
    "lu_solve": ("torch.linalg.lu_solve", None),
    "qr": ("torch.linalg.qr", "`some=True/False` becomes `mode='reduced'/'complete'`"),
    "triangular_solve": (
        "torch.linalg.solve_triangular",
        "argument order changes: B comes first; `unitriangular`/`transpose` are kwargs",
    ),
}

# Tensor methods returning int (or SymInt under tracing); int() around them is redundant.
_INT_RETURNING_TENSOR_METHODS = frozenset({"size", "numel", "dim", "ndimension"})


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
    # If True, a match doesn't end the per-node loop — later patterns still get to fire.
    # Use for orthogonal advice on the same AST shape (e.g. addmm vs baddbmm).
    concurrent: bool = False


# ---- Matchers ---------------------------------------------------------------
# Pure predicates on AST nodes; first match in the registry wins per visit.


def _is_torch_attr(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr == attr
    )


def _is_torch_call(node: ast.AST, attr: str) -> bool:
    return isinstance(node, ast.Call) and _is_torch_attr(node.func, attr)


def _is_one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in (1, 1.0)


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


def _is_method_call(node: ast.AST, attr: str) -> bool:
    """`x.<attr>(...)`, args allowed."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    )


def _method_chain_target(node: ast.Call, outer: str, inner: str) -> bool:
    """`x.<inner>(...).<outer>(...)` — outer attr name, inner attr name."""
    if not isinstance(node.func, ast.Attribute) or node.func.attr != outer:
        return False
    return _is_method_call(node.func.value, inner)


def _zero_arg_method_chain(node: ast.Call, outer: str, inner: str) -> bool:
    """`x.<inner>().<outer>(...)` — inner must be zero-arg."""
    if not isinstance(node.func, ast.Attribute) or node.func.attr != outer:
        return False
    return _is_zero_arg_method_call(node.func.value, inner)


# Kwargs that can change a creator's output layout, breaking the "always contiguous" guarantee.
_LAYOUT_AFFECTING_KWARGS = frozenset({"memory_format", "out"})


def _torch_creator_then(node: ast.Call, outer: str) -> bool:
    """`torch.<creator>(...).<outer>(...)` where creator is in CREATOR_FUNCS and has no layout-affecting kwargs."""
    if not isinstance(node.func, ast.Attribute) or node.func.attr != outer:
        return False
    inner = node.func.value
    if not (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and isinstance(inner.func.value, ast.Name)
        and inner.func.value.id == "torch"
        and inner.func.attr in CREATOR_FUNCS
    ):
        return False
    return all(kw.arg not in _LAYOUT_AFFECTING_KWARGS for kw in inner.keywords)


# ---- Pattern matchers (Call) ------------------------------------------------


def _m_contig_view(node: ast.AST) -> bool:
    # x.contiguous().view(...)
    return isinstance(node, ast.Call) and _zero_arg_method_chain(
        node, "view", "contiguous"
    )


def _m_contig_reshape(node: ast.AST) -> bool:
    # x.contiguous().reshape(...)
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


def _m_view_reshape(node: ast.AST) -> bool:
    # x.view(...).reshape(...)
    return isinstance(node, ast.Call) and _method_chain_target(node, "reshape", "view")


def _m_creator_reshape(node: ast.AST) -> bool:
    # torch.<creator>(...).reshape(...)
    return isinstance(node, ast.Call) and _torch_creator_then(node, "reshape")


def _m_creator_contig(node: ast.AST) -> bool:
    # torch.<creator>(...).contiguous()
    return (
        isinstance(node, ast.Call)
        and _is_zero_arg_method_call(node, "contiguous")
        and _torch_creator_then(node, "contiguous")
    )


def _m_deprecated_torch(node: ast.AST) -> bool | str:
    """torch.<deprecated_op>(...) — emits a per-op message naming the replacement."""
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


def _m_redundant_int_cast(node: ast.AST) -> bool:
    """int() around .size()/.shape[i]/.numel()/.dim()/.ndimension()/len(...)."""
    match node:
        case ast.Call(
            func=ast.Name(id="int"),
            args=[ast.Call(func=ast.Attribute(attr=method))],
            keywords=[],
        ) if method in _INT_RETURNING_TENSOR_METHODS:
            return True
        case ast.Call(
            func=ast.Name(id="int"),
            args=[ast.Call(func=ast.Name(id="len"))],
            keywords=[],
        ):
            return True
        case ast.Call(
            func=ast.Name(id="int"),
            args=[ast.Subscript(value=ast.Attribute(attr="shape"))],
            keywords=[],
        ):
            return True
    return False


def _m_size_subscript(node: ast.AST) -> bool:
    """t.size()[i] — prefer t.size(i)."""
    match node:
        case ast.Subscript(
            value=ast.Call(func=ast.Attribute(attr="size"), args=[], keywords=[])
        ):
            return True
    return False


def _m_wrap_shape(node: ast.AST) -> bool:
    """tuple(t.shape) / list(t.shape)."""
    match node:
        case ast.Call(
            func=ast.Name(id="tuple" | "list"),
            args=[ast.Attribute(attr="shape")],
            keywords=[],
        ):
            return True
    return False


def _m_len_shape(node: ast.AST) -> bool:
    """len(t.shape) — use t.ndim instead."""
    match node:
        case ast.Call(
            func=ast.Name(id="len"),
            args=[ast.Attribute(attr="shape")],
            keywords=[],
        ):
            return True
    return False


def _m_softplus(node: ast.AST) -> bool:
    """torch.log(1 + torch.exp(x)) or torch.log(torch.exp(x) + 1)."""
    match node:
        case ast.Call(
            func=ast.Attribute(value=ast.Name(id="torch"), attr="log"),
            args=[arg],
        ):
            return _comm_binop(
                arg, ast.Add, _is_one, lambda e: _is_torch_call(e, "exp")
            )
    return False


def _m_log1p(node: ast.AST) -> bool:
    """torch.log(1 + x) or torch.log(x + 1) — softplus shape is a strict subset."""
    match node:
        case ast.Call(
            func=ast.Attribute(value=ast.Name(id="torch"), attr="log"),
            args=[arg],
        ):
            return _comm_binop(arg, ast.Add, _is_one, lambda _e: True)
    return False


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
            return _comm_binop(denom, ast.Add, _is_one, _is_neg_exp)
    return False


def _m_rsqrt(node: ast.AST) -> bool:
    """1 / torch.sqrt(x)."""
    match node:
        case ast.BinOp(
            op=ast.Div(),
            left=ast.Constant(value=1 | 1.0),
            right=ast.Call(func=ast.Attribute(value=ast.Name(id="torch"), attr="sqrt")),
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
            return True
    return False


def _m_matmul_add(node: ast.AST) -> bool:
    """a @ b + c   or   c + a @ b — shared by the addmm (2-D) and baddbmm (batched) patterns."""
    return _comm_binop(
        node,
        ast.Add,
        lambda e: isinstance(e, ast.BinOp) and isinstance(e.op, ast.MatMult),
        lambda _e: True,
    )


# ---- The registry -----------------------------------------------------------
# Order = evaluation order; first match per node wins. Softplus must precede
# log1p (its arg shape is a subset).

PATTERNS: list[Pattern] = [
    # --- .contiguous() / .view() / .reshape() fusions on Call nodes ---
    Pattern(
        summary=".contiguous().view(...)              -> .reshape(...)",
        name="prefer .reshape() over .contiguous().view()",
        description="Use `.reshape(...)`: copies only when needed vs. always allocating.",
        node_type=ast.Call,
        matches=_m_contig_view,
    ),
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
        summary=".view(...).reshape(...)              -> drop the .view()",
        name="drop intermediate .view() before .reshape()",
        description="Drop the intermediate `.view(...)`; apply `.reshape` directly.",
        node_type=ast.Call,
        matches=_m_view_reshape,
    ),
    Pattern(
        summary="torch.<creator>(...).reshape(...)    -> .view(...)",
        name="prefer .view() over .reshape() on tensor-creator output",
        description="Creator outputs are contiguous; use `.view` to skip `.reshape`'s contiguity check.",
        node_type=ast.Call,
        matches=_m_creator_reshape,
    ),
    Pattern(
        summary="torch.<creator>(...).contiguous()    -> drop .contiguous()",
        name="drop redundant .contiguous() on tensor-creator output",
        description="Creator outputs are already contiguous; drop the `.contiguous()`.",
        node_type=ast.Call,
        matches=_m_creator_contig,
    ),
    # --- SymInt-aware shape idioms ---
    Pattern(
        summary="int(t.size(...))/.shape[i]/.numel()/len(.) -> drop the int()",
        name="redundant int() on tensor size/shape/numel/len",
        description="Drop the `int()`: these already return int, and forcing it on SymInt creates a guard.",
        node_type=ast.Call,
        matches=_m_redundant_int_cast,
    ),
    Pattern(
        summary="t.size()[i]                          -> t.size(i)",
        name="prefer t.size(i) over t.size()[i]",
        description="Use `t.size(i)`: cheaper than `t.size()[i]`, SymInt-safe in compile mode.",
        node_type=ast.Subscript,
        matches=_m_size_subscript,
    ),
    Pattern(
        summary="tuple(t.shape) / list(t.shape)       -> drop the wrapper",
        name="redundant tuple/list around t.shape",
        description="Drop the wrapper: `torch.Size` is already a tuple subclass and SymInt-aware.",
        node_type=ast.Call,
        matches=_m_wrap_shape,
    ),
    Pattern(
        summary="len(t.shape)                         -> t.ndim",
        name="prefer t.ndim over len(t.shape)",
        description="Use `t.ndim`: faster and avoids materializing the size tuple.",
        node_type=ast.Call,
        matches=_m_len_shape,
    ),
    # --- Deprecated torch top-level ops ---
    Pattern(
        summary="torch.<deprecated>(...)              -> see message for replacement",
        name="deprecated torch.* op",
        description="(see per-match message)",
        node_type=ast.Call,
        matches=_m_deprecated_torch,
    ),
    # --- torch.log(...) fusions ---
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
    # --- BinOp fusions ---
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
        summary="torch.exp(x) - 1                     -> torch.expm1(x)",
        name="prefer torch.expm1(x) over torch.exp(x) - 1",
        description="Use `torch.expm1(x)`: numerically stable for small x where exp(x) rounds to 1.",
        node_type=ast.BinOp,
        matches=_m_expm1,
    ),
    Pattern(
        summary="a @ b + c (2-D)                      -> torch.addmm(c, a, b)",
        name="consider torch.addmm (2-D matmul+bias)",
        description="If `a`, `b` are 2-D: use `torch.addmm(c, a, b)` to fuse matmul+bias in one cuBLAS call.",
        node_type=ast.BinOp,
        matches=_m_matmul_add,
        concurrent=True,
    ),
    Pattern(
        summary="a @ b + c (batched)                  -> torch.baddbmm(c, a, b)",
        name="consider torch.baddbmm (batched matmul+bias)",
        description="If `a`, `b` are 3-D batched: use `torch.baddbmm(c, a, b)`. Bias must match `a@b`'s shape (no broadcast).",
        node_type=ast.BinOp,
        matches=_m_matmul_add,
        concurrent=True,
    ),
]


# Cross-cutting check (not a node-shape pattern): warns on long method chains.
# Kept out of PATTERNS because it isn't expressible as a single-node matcher —
# it counts walks across the chain spine and must dedupe sub-call visits.
LONG_CHAIN_NAME = "long method chain"  # actual emitted name appends the count
LONG_CHAIN_DESC = (
    f"Chain of {CHAIN_THRESHOLD}+ method calls; break into intermediate variables."
)
LONG_CHAIN_SUMMARY = f"method chains of length >= {CHAIN_THRESHOLD}"


def _render_docstring() -> str:
    lines = ["Flags fuse-able / more-idiomatic PyTorch patterns via AST:", ""]
    lines += [f"  - {p.summary}" for p in PATTERNS]
    lines.append(f"  - {LONG_CHAIN_SUMMARY}")
    return "\n".join(lines) + "\n"


# Replace the placeholder module docstring with the registry-derived one so
# `python -c 'import fuse_pattern_linter; print(fuse_pattern_linter.__doc__)'`
# yields the canonical pattern list.
__doc__ = _render_docstring()


def _chain_length(node: ast.AST) -> int:
    """Length of a consecutive method-call chain rooted at `node`.

    For `a.b().c().d()` returns 3 when called with the outermost Call.
    """
    n = 0
    cur = node
    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
        n += 1
        cur = cur.func.value
    return n


# Pre-bucket patterns by node type so the visitor's per-node loop is small.
_PATTERNS_BY_TYPE: dict[type[ast.AST], list[Pattern]] = {}
for _p in PATTERNS:
    _PATTERNS_BY_TYPE.setdefault(_p.node_type, []).append(_p)


class FusePatternVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.messages: list[LintMessage] = []
        # ids of Call nodes already accounted for as part of a chain head
        self._chain_counted: set[int] = set()

    def _emit(self, node: ast.AST, name: str, description: str) -> None:
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

    def _check_patterns(self, node: ast.AST) -> None:
        for pat in _PATTERNS_BY_TYPE.get(type(node), ()):
            result = pat.matches(node)
            if not result:
                continue
            description = result if isinstance(result, str) else pat.description
            self._emit(node, pat.name, description)
            if not pat.concurrent:
                return  # first match wins unless the pattern opts to be concurrent

    def visit_Call(self, node: ast.Call) -> None:
        self._check_patterns(node)

        # Cross-cutting: chain-length check. Count only at the chain head and
        # mark all sub-calls so we don't re-count them on visit.
        if id(node) not in self._chain_counted:
            length = _chain_length(node)
            if length >= CHAIN_THRESHOLD:
                self._emit(node, f"long method chain ({length} calls)", LONG_CHAIN_DESC)
            cur: ast.AST = node
            while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
                self._chain_counted.add(id(cur))
                cur = cur.func.value

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._check_patterns(node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
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
