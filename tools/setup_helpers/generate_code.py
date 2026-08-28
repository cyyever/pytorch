from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# This script is run by path from the build, so sys.path[0] is this file's
# directory rather than the repository root; the tools.* imports below need
# the root on the path.
sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

NATIVE_FUNCTIONS_PATH = "aten/src/ATen/native/native_functions.yaml"
TAGS_PATH = "aten/src/ATen/native/tags.yaml"


def generate_code(
    gen_dir: Path,
    native_functions_path: str | None = None,
    tags_path: str | None = None,
    install_dir: str | None = None,
    subset: str | None = None,
    disable_autograd: bool = False,
) -> None:
    from tools.autograd.gen_annotated_fn_args import gen_annotated
    from tools.autograd.gen_autograd import gen_autograd, gen_autograd_python

    # Build ATen based Variable classes
    if install_dir is None:
        install_dir = os.fspath(gen_dir / "torch/csrc")
        python_install_dir = os.fspath(gen_dir / "torch/testing/_internal/generated")
    else:
        python_install_dir = install_dir
    autograd_gen_dir = os.path.join(install_dir, "autograd", "generated")
    for d in (autograd_gen_dir, python_install_dir):
        os.makedirs(d, exist_ok=True)
    autograd_dir = os.fspath(Path(__file__).parent.parent / "autograd")

    if subset == "pybindings" or not subset:
        gen_autograd_python(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            tags_path or TAGS_PATH,
            autograd_gen_dir,
            autograd_dir,
        )

    if subset == "libtorch" or not subset:
        gen_autograd(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            tags_path or TAGS_PATH,
            autograd_gen_dir,
            autograd_dir,
            disable_autograd=disable_autograd,
        )

    if subset == "python" or not subset:
        gen_annotated(
            native_functions_path or NATIVE_FUNCTIONS_PATH,
            tags_path or TAGS_PATH,
            python_install_dir,
            autograd_dir,
        )




def main() -> None:
    parser = argparse.ArgumentParser(description="Autogenerate code")
    parser.add_argument("--native-functions-path")
    parser.add_argument("--tags-path")
    parser.add_argument(
        "--gen-dir",
        type=Path,
        default=Path("."),
        help="Root directory where to install files. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--install-dir",
        "--install_dir",
        help=(
            "Deprecated. Use --gen-dir instead. The semantics are different, do not change "
            "blindly."
        ),
    )
    parser.add_argument(
        "--subset",
        help='Subset of source files to generate. Can be "libtorch" or "pybindings". Generates both when omitted.',
    )
    parser.add_argument(
        "--disable-autograd",
        default=False,
        action="store_true",
        help="It can skip generating autograd related code when the flag is set",
    )
    options = parser.parse_args()

    # Path: aten/src/ATen
    aten_path = os.path.dirname(
        os.path.dirname(options.native_functions_path or NATIVE_FUNCTIONS_PATH)
    )
    generate_code(
        options.gen_dir,
        options.native_functions_path,
        options.tags_path,
        options.install_dir,
        options.subset,
        options.disable_autograd,
    )

    # Generate the python bindings for functionalization's `ViewMeta` classes.
    from torchgen.gen_functionalization_type import (
        gen_functionalization_view_meta_classes,
    )

    functionalization_templates_dir = os.path.join(aten_path, "templates")
    install_dir = options.install_dir or os.fspath(options.gen_dir / "torch/csrc")
    functionalization_install_dir = os.path.join(
        install_dir, "functionalization", "generated"
    )

    os.makedirs(functionalization_install_dir, exist_ok=True)
    if not os.path.isdir(functionalization_install_dir):
        raise AssertionError(f"Not a directory: {functionalization_install_dir}")
    if not os.path.isdir(functionalization_templates_dir):
        raise AssertionError(f"Not a directory: {functionalization_templates_dir}")

    gen_functionalization_view_meta_classes(
        options.native_functions_path or NATIVE_FUNCTIONS_PATH,
        options.tags_path or TAGS_PATH,
        install_dir=functionalization_install_dir,
        template_dir=functionalization_templates_dir,
    )


if __name__ == "__main__":
    main()
