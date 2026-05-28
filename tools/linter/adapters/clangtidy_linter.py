from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from sysconfig import get_paths as gp
from typing import NamedTuple


# PyTorch directory root
def scm_root() -> str:
    path = os.path.abspath(os.getcwd())
    while True:
        if os.path.exists(os.path.join(path, ".git")):
            return path
        if os.path.isdir(os.path.join(path, ".hg")):
            return path
        n = len(path)
        path = os.path.dirname(path)
        if len(path) == n:
            raise RuntimeError("Unable to find SCM root")


PYTORCH_ROOT = scm_root()


# Returns '/usr/local/include/python<version number>'
def get_python_include_dir() -> str:
    return gp()["include"]


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


# c10/core/DispatchKey.cpp:281:26: error: 'k' used after it was moved [bugprone-use-after-move]
RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?:(?P<column>-?\d+):)?
    \s(?P<severity>\S+?):?
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)


def run_command(
    args: list[str],
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            capture_output=True,
            check=False,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


# Severity is either "error" or "note":
# https://github.com/python/mypy/blob/8b47a032e1317fb8e3f9a818005a6b63e9bf0311/mypy/errors.py#L46-L47
severities = {
    "error": LintSeverity.ERROR,
    "warning": LintSeverity.WARNING,
}


def clang_search_dirs() -> list[str]:
    # Compilers are ordered based on fallback preference
    # We pick the first one that is available on the system
    compilers = ["clang", "gcc", "cpp", "cc"]
    compilers = [c for c in compilers if shutil.which(c) is not None]
    if len(compilers) == 0:
        raise RuntimeError(f"None of {compilers} were found")
    compiler = compilers[0]

    result = subprocess.run(
        [compiler, "-E", "-x", "c++", "-", "-v"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )
    stderr = result.stderr.decode().strip().split("\n")
    search_start = r"#include.*search starts here:"
    search_end = r"End of search list."

    append_path = False
    search_paths = []
    for line in stderr:
        if re.match(search_start, line):
            if append_path:
                continue
            else:
                append_path = True
        elif re.match(search_end, line):
            break
        elif append_path:
            search_paths.append(line.strip())

    return search_paths


include_args = []
include_dir = [
    "/usr/lib/llvm-11/include/openmp",
    get_python_include_dir(),
    os.path.join(PYTORCH_ROOT, "third_party/pybind11/include"),
    # For header-only lints (no compile_commands.json entry) to resolve <ATen/...>.
    os.path.join(PYTORCH_ROOT, "aten/src"),
    PYTORCH_ROOT,
] + clang_search_dirs()
for dir in include_dir:
    include_args += ["--extra-arg", f"-I{dir}"]


def detect_cuda_path() -> str | None:
    for candidate in (os.environ.get("CUDA_HOME"), "/opt/cuda", "/usr/local/cuda"):
        if candidate and os.path.isdir(os.path.join(candidate, "include")):
            return candidate
    return None


def detect_clang_resource_dir() -> str | None:
    # Look for clang's bundled headers, which include __clang_cuda_runtime_wrapper.h.
    # The clang-tidy binary at .lintbin/ does not ship its own resource dir, so we
    # fall back to whatever system clang installation exposes one.
    for clang in ("clang", "clang++"):
        path = shutil.which(clang)
        if path is None:
            continue
        try:
            result = subprocess.run(
                [path, "-print-resource-dir"],
                capture_output=True,
                check=True,
                text=True,
            )
            candidate = result.stdout.strip()
            if candidate and os.path.isfile(
                os.path.join(candidate, "include", "__clang_cuda_runtime_wrapper.h")
            ):
                return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    # Fall back: scan known system locations.
    for base in ("/usr/lib/clang", "/usr/lib64/clang"):
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base), reverse=True):
            candidate = os.path.join(base, entry)
            if os.path.isfile(
                os.path.join(candidate, "include", "__clang_cuda_runtime_wrapper.h")
            ):
                return candidate
    return None


# Checks disabled in CUDA mode (noisy false positives or risky transforms).
CUDA_DISABLED_CHECKS = (
    "cppcoreguidelines-init-variables",  # FP: out-by-ref params, inline-asm outputs
    "readability-redundant-declaration",  # FP: per-kernel `extern __shared__` redecls
    "modernize-loop-convert",  # risky inside __device__ + #pragma unroll
    "cppcoreguidelines-pro-type-member-init",  # forced memset on large prep structs
    "misc-static-assert",  # FP: device-side runtime asserts in CUDA test kernels
)


def build_cuda_extra_args(cuda_path: str, resource_dir: str | None) -> list[str]:
    extras = [
        "-x",
        "cuda",
        "-std=c++20",
        f"--cuda-path={cuda_path}",
        "--no-cuda-version-check",
        "--cuda-host-only",
        # Errors-as-warnings are forced on by .clang-tidy WarningsAsErrors='*';
        # these CUDA-mode quirks are not real issues.
        "-Wno-unknown-cuda-version",
        "-Wno-unused-command-line-argument",
        f"-I{cuda_path}/include",
        f"-I{os.path.join(PYTORCH_ROOT, 'aten/src')}",
    ]
    # CUDA 13 bundles Thrust/CUB/libcudacxx under include/cccl; older toolkits
    # put them directly under include/. Add the cccl path when present.
    cccl_path = os.path.join(cuda_path, "include", "cccl")
    if os.path.isdir(cccl_path):
        extras.append(f"-I{cccl_path}")
    if resource_dir is not None:
        extras.append(f"-resource-dir={resource_dir}")
    # --checks is appended to .clang-tidy Checks (last match wins), so trailing
    # negative entries override the positive globs.
    disabled = ",".join(f"-{c}" for c in CUDA_DISABLED_CHECKS)
    flattened = [f"--checks={disabled}"]
    for arg in extras:
        flattened += ["--extra-arg", arg]
    return flattened


def check_file(
    filename: str,
    binary: str,
    build_dir: Path,
    std: str | None,
    cuda_extras: list[str] | None,
) -> list[LintMessage]:
    # Explicitly pass include path for linters that only check headers.
    # build/aten/src covers generated <ATen/...> headers (Functions.h etc.).
    build_include_args = include_args + [
        "--extra-arg",
        f"-I{build_dir}",
        "--extra-arg",
        f"-I{build_dir}/aten/src",
    ]
    if cuda_extras is not None:
        # In CUDA mode we bypass the compile_commands.json lookup: nvcc commands
        # contain flags clang cannot parse (-Xfatbin, -gencode, etc). Build a
        # minimal clang-driver command ourselves.
        cmd = [
            binary,
            *build_include_args,
            *cuda_extras,
            filename,
        ]
    else:
        cmd = [
            binary,
            f"-p={build_dir}",
            *build_include_args,
            filename,
        ]
    # Only add -- and -std flag if std is explicitly specified
    if std is not None:
        cmd.extend(["--", f"-std={std}"])

    try:
        proc = run_command(cmd)
    except OSError as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="CLANGTIDY",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    lint_messages = []
    try:
        # Change the current working directory to the build directory, since
        # clang-tidy will report files relative to the build directory.
        saved_cwd = os.getcwd()
        os.chdir(build_dir)

        for match in RESULTS_RE.finditer(proc.stdout.decode()):
            # Convert the reported path to an absolute path.
            abs_path = str(Path(match["file"]).resolve())
            if not abs_path.startswith(PYTORCH_ROOT):
                continue
            # Skip CUTLASS template parse failures (not actionable lint).
            if (
                cuda_extras is not None
                and match["code"] == "clang-diagnostic-error"
                and "third_party/cutlass/" in abs_path
            ):
                continue
            message = LintMessage(
                path=abs_path,
                name=match["code"],
                description=match["message"],
                line=int(match["line"]),
                char=int(match["column"])
                if match["column"] is not None and not match["column"].startswith("-")
                else None,
                code="CLANGTIDY",
                severity=severities.get(match["severity"], LintSeverity.ERROR),
                original=None,
                replacement=None,
            )
            lint_messages.append(message)
    finally:
        os.chdir(saved_cwd)

    return lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="clang-tidy wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="clang-tidy binary path",
    )
    parser.add_argument(
        "--build-dir",
        "--build_dir",
        required=True,
        help=(
            "Where the compile_commands.json file is located. "
            "Gets passed to clang-tidy -p"
        ),
    )
    parser.add_argument(
        "--std",
        default=None,
        help=(
            "C++ standard to use for compilation (e.g., c++17, c++20). "
            "If not specified, uses the standard from compile_commands.json."
        ),
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help=(
            "Treat inputs as CUDA sources/headers. Bypasses compile_commands.json "
            "and builds a clang -x cuda command line with --cuda-host-only."
        ),
    )
    parser.add_argument(
        "--cuda-path",
        default=None,
        help="CUDA toolkit path (default: $CUDA_HOME or /opt/cuda or /usr/local/cuda).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    if not os.path.exists(args.binary):
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code="CLANGTIDY",
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=(
                f"Could not find clang-tidy binary at {args.binary},"
                " you may need to run `lintrunner init`."
            ),
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        sys.exit(0)

    abs_build_dir = Path(args.build_dir).resolve()

    # Get the absolute path to clang-tidy and use this instead of the relative
    # path such as .lintbin/clang-tidy. The problem here is that os.chdir is
    # per process, and the linter uses it to move between the current directory
    # and the build folder. And there is no .lintbin directory in the latter.
    # When it happens in a race condition, the linter command will fails with
    # the following no such file or directory error: '.lintbin/clang-tidy'
    binary_path = os.path.abspath(args.binary)

    cuda_extras: list[str] | None = None
    if args.cuda:
        cuda_path = args.cuda_path or detect_cuda_path()
        if cuda_path is None:
            err = LintMessage(
                path="<none>",
                line=None,
                char=None,
                code="CLANGTIDY",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    "CUDA toolkit not found. Set --cuda-path, $CUDA_HOME, or install "
                    "to /opt/cuda or /usr/local/cuda."
                ),
            )
            print(json.dumps(err._asdict()), flush=True)
            sys.exit(0)
        cuda_extras = build_cuda_extra_args(cuda_path, detect_clang_resource_dir())

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(
                check_file,
                filename,
                binary_path,
                abs_build_dir,
                args.std,
                cuda_extras,
            ): filename
            for filename in args.filenames
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
