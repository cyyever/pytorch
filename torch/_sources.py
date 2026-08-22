# mypy: allow-untyped-defs
import inspect
from textwrap import dedent
from typing import Any



def get_source_lines_and_file(
    obj: Any,
    error_msg: str | None = None,
) -> tuple[list[str], int, str | None]:
    """
    Wrapper around inspect.getsourcelines and inspect.getsourcefile.

    Returns: (sourcelines, file_lineno, filename)
    """
    filename = None  # in case getsourcefile throws
    try:
        filename = inspect.getsourcefile(obj)
        sourcelines, file_lineno = inspect.getsourcelines(obj)
    except OSError as e:
        msg = (
            f"Can't get source for {obj}. TorchScript requires source access in "
            "order to carry out compilation, make sure original .py files are "
            "available."
        )
        if error_msg:
            msg += "\n" + error_msg
        raise OSError(msg) from e

    return sourcelines, file_lineno, filename


def normalize_source_lines(sourcelines: list[str]) -> list[str]:
    """
    This helper function accepts a list of source lines. It finds the
    indentation level of the function definition (`def`), then it indents
    all lines in the function body to a point at or greater than that
    level. This allows for comments and continued string literals that
    are at a lower indentation than the rest of the code.
    Args:
        sourcelines: function source code, separated into lines by
                        the '\n' character
    Returns:
        A list of source lines that have been correctly aligned
    """

    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix) :]

    # Find the line and line number containing the function definition
    idx = None
    for i, l in enumerate(sourcelines):
        if l.lstrip().startswith("def"):
            idx = i
            break

    # This will happen when the function is a lambda- we won't find "def" anywhere in the source
    # lines in that case. Currently trying to JIT compile a lambda will throw an error up in
    # `parse_def()`, but we might want to handle this case in the future.
    if idx is None:
        return sourcelines

    # Get a string representing the amount of leading whitespace
    fn_def = sourcelines[idx]
    whitespace = fn_def.split("def")[0]

    # Add this leading whitespace to all lines before and after the `def`
    aligned_prefix = [
        whitespace + remove_prefix(s, whitespace) for s in sourcelines[:idx]
    ]
    aligned_suffix = [
        whitespace + remove_prefix(s, whitespace) for s in sourcelines[idx + 1 :]
    ]

    # Put it together again
    aligned_prefix.append(fn_def)
    return aligned_prefix + aligned_suffix
