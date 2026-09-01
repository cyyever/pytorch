# This CPP builder is designed to support both Windows and Linux OS.
# The design document please check this RFC: https://github.com/pytorch/pytorch/issues/124245

import copy
import errno
import functools
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import warnings
from collections.abc import Sequence
from ctypes import cdll
from ctypes.util import find_library
from pathlib import Path

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config, exc
from torch._inductor.cpu_vec_isa import invalid_vec_isa, VecISA
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.torch_version import TorchVersion
from torch.utils._ordered_set import OrderedSet


# Windows need setup a temp dir to store .obj files.
_BUILD_TEMP_DIR = "CxxBuild"
_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
_LINKER_SCRIPT = os.path.join(_TORCH_PATH, "_inductor/script.ld")

# initialize variables for compilation
_IS_LINUX = sys.platform.startswith("linux")
_IS_MACOS = sys.platform.startswith("darwin")

SUBPROCESS_DECODE_ARGS = ()

log = logging.getLogger(__name__)


# =============================== toolchain ===============================
def _split_compiler_command(compiler: str) -> list[str]:
    command = shlex.split(compiler)
    if not command:
        raise ValueError("empty compiler command")
    return command


def _compiler_command(compiler: str, *args: str) -> list[str]:
    return [*_split_compiler_command(compiler), *args]


@functools.cache
def _compiler_version_string(cpp_compiler: str) -> str:
    try:
        return (
            subprocess.check_output(
                _compiler_command(cpp_compiler, "--version"),
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode(*SUBPROCESS_DECODE_ARGS)
        )
    except FileNotFoundError, subprocess.SubprocessError, ValueError:
        return ""


def _compiler_version_first_line(cpp_compiler: str) -> str:
    version_string = _compiler_version_string(cpp_compiler)
    return version_string.splitlines()[0] if version_string else ""


@functools.lru_cache(1)
def cpp_compiler_search(search: Sequence[str | None]) -> str:
    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT

    for cxx in search:
        try:
            if cxx is None:
                # gxx package is only available for Linux
                # according to https://anaconda.org/conda-forge/gxx/
                if sys.platform != "linux":
                    continue
                # Do not install GXX by default
                if not os.getenv("TORCH_INDUCTOR_INSTALL_GXX"):
                    continue
                from torch.utils._filelock import FileLock

                lock_dir = get_lock_dir()
                lock = FileLock(
                    os.path.join(lock_dir, "g++.lock"), timeout=LOCK_TIMEOUT
                )
                with lock:
                    cxx = install_gcc_via_conda()
            subprocess.check_output(_compiler_command(cxx, "--version"))
            return cxx
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            ImportError,
            ValueError,
        ):
            continue
    raise exc.InvalidCxxCompiler


def install_gcc_via_conda() -> str:
    """On older systems, this is a quick way to get a modern compiler"""
    prefix = os.path.join(cache_dir(), "gcc")
    cxx_path = os.path.join(prefix, "bin", "g++")
    if not os.path.exists(cxx_path):
        log.info("Downloading GCC via conda")
        conda = os.environ.get("CONDA_EXE", "conda")
        if conda is None:
            conda = shutil.which("conda")
        if conda is not None:
            subprocess.check_call(
                [
                    conda,
                    "create",
                    f"--prefix={prefix}",
                    "--channel=conda-forge",
                    "--quiet",
                    "-y",
                    "python=3.8",
                    "gxx",
                ],
                stdout=subprocess.PIPE,
            )
    return cxx_path


def get_cpp_compiler() -> str:
    if isinstance(config.cpp.cxx, (list, tuple)):
        search = tuple(config.cpp.cxx)
    else:
        search = (config.cpp.cxx,)
    compiler = cpp_compiler_search(search)
    return compiler


def get_ld_and_objcopy(use_relative_path: bool) -> tuple[str, str]:
    ld = "ld"
    objcopy = "objcopy"
    return ld, objcopy


def convert_cubin_to_obj(
    cubin_file: str,
    kernel_name: str,
    ld: str,
    objcopy: str,
) -> str:
    obj_file = cubin_file + ".o"
    # Convert .cubin to .o
    cmd = f"{ld} -r -b binary -z noexecstack -o {obj_file} {cubin_file}"
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    # Rename .data to .rodata
    cmd = f"{objcopy} --rename-section .data=.rodata,alloc,load,readonly,data,contents {obj_file}"
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    # By default objcopy will create *_start, *_size, *_end symbols using the full path
    # Rename to use the unique kernel name
    file_name = re.sub(r"[\W]", "_", cubin_file)
    cmd = (
        objcopy
        + f" --redefine-sym _binary_{file_name}_start=__{kernel_name}_start "
        + f"--redefine-sym _binary_{file_name}_size=__{kernel_name}_size "
        + f"--redefine-sym _binary_{file_name}_end=__{kernel_name}_end "
        + obj_file
    )
    subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
    return obj_file


def batch_convert_cubins_to_obj(
    cubins: list[tuple[str, str]],
    output_dir: str,
    cpp_compiler: str = "gcc",
) -> str:
    """Convert multiple cubin files to a single .o using batched .incbin assembly.

    Instead of spawning 3 subprocesses per cubin (ld + 2x objcopy), generates
    a single .S file with .incbin directives for all cubins and compiles it
    with one compiler invocation. Produces bit-identical rodata and symbols
    as the per-cubin convert_cubin_to_obj approach.

    Args:
        cubins: list of (cubin_file_path, kernel_name) tuples.
        output_dir: directory for the generated .S and .o files.
        cpp_compiler: C compiler to use for assembling (default: gcc).

    Returns:
        Path to the combined .o file.
    """
    asm_path = os.path.join(output_dir, "cubins_combined.S")
    obj_path = os.path.join(output_dir, "cubins_combined.o")

    with open(asm_path, "w") as f:
        f.write(".section .rodata\n")
        for cubin_file, kernel_name in cubins:
            # Use absolute path to avoid issues with working directory
            abs_cubin = os.path.abspath(cubin_file)
            escaped_path = abs_cubin.replace("\\", "\\\\").replace('"', '\\"')
            f.write(
                f".balign 16\n"
                f".global __{kernel_name}_start\n"
                f".global __{kernel_name}_end\n"
                f"__{kernel_name}_start:\n"
                f'.incbin "{escaped_path}"\n'
                f"__{kernel_name}_end:\n"
                f".global __{kernel_name}_size\n"
                f".set __{kernel_name}_size, "
                f"__{kernel_name}_end - __{kernel_name}_start\n"
            )

    subprocess.run(
        _compiler_command(cpp_compiler, "-c", asm_path, "-o", obj_path),
        capture_output=True,
        text=True,
        check=True,
    )
    return obj_path


@functools.cache
def _is_apple_clang(cpp_compiler: str) -> bool:
    first_line = _compiler_version_first_line(cpp_compiler)
    return bool(re.search(r"\bApple\b.*\bclang\b", first_line))


@functools.cache
def _is_clang(cpp_compiler: str) -> bool:
    if sys.platform == "darwin" and _is_apple_clang(cpp_compiler):
        return True

    first_line = _compiler_version_first_line(cpp_compiler)
    if "Intel" not in first_line and re.search(
        r"(^|[\s/-])clang version\b", first_line
    ):
        return True

    return bool(re.search(r"(^|[/\s-])(clang\+\+|clang)(?=[\s-]|$)", cpp_compiler))


@functools.cache
def _is_gcc(cpp_compiler: str) -> bool:
    # Since "clang++" ends with "g++", the regex match below would validate on it.
    if _is_clang(cpp_compiler):
        return False

    first_line = _compiler_version_first_line(cpp_compiler)
    if re.search(
        r"(^|[\s/-])(gcc|g\+\+|gnu-c\+\+)(?=[\s(-]|$)",
        first_line,
        re.IGNORECASE,
    ):
        return True

    return bool(re.search(r"(^|[/\s-])(gcc|g\+\+|gnu-c\+\+)(?=[\s-]|$)", cpp_compiler))


@functools.cache
def _is_gcc_version_less_than(cpp_compiler: str, major: int) -> bool:
    if not _is_gcc(cpp_compiler):
        return False

    try:
        output_msg = (
            subprocess.check_output(
                _compiler_command(cpp_compiler, "-dumpfullversion", "-dumpversion"),
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode(*SUBPROCESS_DECODE_ARGS)
        )
    except FileNotFoundError, subprocess.SubprocessError:
        return False

    version_search = re.search(r"\d+(?:\.\d+)*", output_msg)
    if version_search is None:
        return False

    return TorchVersion(version_search.group(0)) < TorchVersion(str(major))


@functools.cache
def _is_msvc_cl(cpp_compiler: str) -> bool:
    return False


@functools.cache
def _is_intel_compiler(cpp_compiler: str) -> bool:
    def _check_minimal_version(compiler_version: TorchVersion) -> None:
        """
        On Windows: early version icx has `-print-file-name` issue, and can't preload correctly for inductor.
        """
        min_version = "0.0.0"
        if compiler_version < TorchVersion(min_version):
            raise RuntimeError(
                f"Intel Compiler error: less than minimal version {min_version}."
            )

    try:
        output_msg = _compiler_version_string(cpp_compiler)
        lines = output_msg.splitlines()
        is_intel_compiler = bool(lines) and "Intel" in lines[0]
        if is_intel_compiler:
            # Version check
            icx_ver_search = re.search(r"(\d+[.]\d+[.]\d+[.]\d+)", output_msg)
            if icx_ver_search is not None:
                icx_ver = icx_ver_search.group(1)
                _check_minimal_version(TorchVersion(icx_ver))

        return is_intel_compiler
    except FileNotFoundError:
        return False
    except subprocess.SubprocessError:
        # --version args not support.
        return False

    # pyrefly: ignore [unreachable]
    return False


@functools.cache
def is_gcc() -> bool:
    return _is_gcc(get_cpp_compiler())


@functools.cache
def is_clang() -> bool:
    return _is_clang(get_cpp_compiler())


@functools.cache
def is_intel_compiler() -> bool:
    return _is_intel_compiler(get_cpp_compiler())


@functools.cache
def is_apple_clang() -> bool:
    return _is_apple_clang(get_cpp_compiler())


@functools.cache
def is_msvc_cl() -> bool:
    return _is_msvc_cl(get_cpp_compiler())


@functools.cache
def get_compiler_version_info(compiler: str) -> str:
    env = os.environ.copy()
    env["LC_ALL"] = "C"  # Don't localize output
    try:
        version_string = subprocess.check_output(
            _compiler_command(compiler, "-v"),
            stderr=subprocess.STDOUT,
            env=env,
        ).decode(*SUBPROCESS_DECODE_ARGS)
    except Exception:
        try:
            version_string = subprocess.check_output(
                _compiler_command(compiler, "--version"),
                stderr=subprocess.STDOUT,
                env=env,
            ).decode(*SUBPROCESS_DECODE_ARGS)
        except Exception:
            return ""
    # Multiple lines to one line string.
    version_string = version_string.replace("\r", "_")
    version_string = version_string.replace("\n", "_")
    return version_string


# =============================== cpp builder ===============================
def _append_list(dest_list: list[str], src_list: list[str]) -> None:
    dest_list.extend(copy.deepcopy(item) for item in src_list)


def _remove_duplication_in_list(orig_list: list[str]) -> list[str]:
    new_list: list[str] = []
    for item in orig_list:
        if item not in new_list:
            new_list.append(item)
    return new_list


def _create_if_dir_not_exist(path_dir: str) -> None:
    if not os.path.exists(path_dir):
        try:
            Path(path_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise RuntimeError(f"Fail to create path {path_dir}") from exc


def _remove_dir(path_dir: str) -> None:
    if os.path.exists(path_dir):
        for root, dirs, files in os.walk(path_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                dir_path = os.path.join(root, name)
                os.rmdir(dir_path)
        os.rmdir(path_dir)


def _run_compile_cmd(cmd_line: str, cwd: str) -> None:
    cmd = shlex.split(cmd_line)
    try:
        subprocess.run(
            cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        output = e.stdout.decode(*SUBPROCESS_DECODE_ARGS)
        openmp_problem = "'omp.h' file not found" in output or "libomp" in output
        if openmp_problem and sys.platform == "darwin":
            instruction = (
                "\n\nOpenMP support not found. Please try one of the following solutions:\n"
                "(1) Set the `CXX` environment variable to a compiler other than Apple clang++/g++ "
                "that has builtin OpenMP support;\n"
                "(2) install OpenMP via conda: `conda install llvm-openmp`;\n"
                "(3) install libomp via brew: `brew install libomp`;\n"
                "(4) manually setup OpenMP and set the `OMP_PREFIX` environment variable to point to a path"
                " with `include/omp.h` under it."
            )
            output += instruction
        raise exc.CppCompileError(cmd, output) from e


def run_compile_cmd(cmd_line: str, cwd: str) -> None:
    with dynamo_timed("compile_file"):
        _run_compile_cmd(cmd_line, cwd)


def normalize_path_separator(orig_path: str) -> str:
    return orig_path


class BuildOptionsBase:
    """
    This is the Base class for store cxx build options, as a template.
    Actually, to build a cxx shared library. We just need to select a compiler
    and maintains the suitable args.
    """

    def __init__(
        self,
        compiler: str = "",
        definitions: list[str] | None = None,
        include_dirs: list[str] | None = None,
        cflags: list[str] | None = None,
        ldflags: list[str] | None = None,
        libraries_dirs: list[str] | None = None,
        libraries: list[str] | None = None,
        passthrough_args: list[str] | None = None,
        aot_mode: bool = False,
        use_relative_path: bool = False,
        compile_only: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        self._compiler = compiler
        self._definitions: list[str] = definitions or []
        self._include_dirs: list[str] = include_dirs or []
        self._cflags: list[str] = cflags or []
        self._ldflags: list[str] = ldflags or []
        self._libraries_dirs: list[str] = libraries_dirs or []
        self._libraries: list[str] = libraries or []
        # Some args are hard to abstract to OS compatible, passthrough directly.
        self._passthrough_args: list[str] = passthrough_args or []

        # Optionally, the path to a precompiled header which should be included on the
        # build command line.
        self.precompiled_header: str | None = None

        self._aot_mode: bool = aot_mode
        self._use_relative_path: bool = use_relative_path
        self._compile_only: bool = compile_only
        self._precompiling: bool = precompiling
        self._preprocessing: bool = preprocessing

    def _process_compile_only_options(self) -> None:
        if self._compile_only or self._precompiling or self._preprocessing:
            self._libraries_dirs = []
            self._libraries = []
            self._ldflags = []

    def _remove_duplicate_options(self) -> None:
        self._definitions = _remove_duplication_in_list(self._definitions)
        self._include_dirs = _remove_duplication_in_list(self._include_dirs)
        self._cflags = _remove_duplication_in_list(self._cflags)
        self._ldflags = _remove_duplication_in_list(self._ldflags)
        self._libraries_dirs = _remove_duplication_in_list(self._libraries_dirs)
        self._libraries = _remove_duplication_in_list(self._libraries)
        self._passthrough_args = _remove_duplication_in_list(self._passthrough_args)

    def _finalize_options(self) -> None:
        self._process_compile_only_options()
        self._remove_duplicate_options()

    def get_compiler(self) -> str:
        return self._compiler

    def get_definitions(self) -> list[str]:
        return self._definitions

    def get_include_dirs(self) -> list[str]:
        return self._include_dirs

    def get_cflags(self) -> list[str]:
        return self._cflags

    def get_ldflags(self) -> list[str]:
        return self._ldflags

    def get_libraries_dirs(self) -> list[str]:
        return self._libraries_dirs

    def get_libraries(self) -> list[str]:
        return self._libraries

    def get_passthrough_args(self) -> list[str]:
        return self._passthrough_args

    def get_aot_mode(self) -> bool:
        return self._aot_mode

    def get_use_relative_path(self) -> bool:
        return self._use_relative_path

    def get_compile_only(self) -> bool:
        return self._compile_only

    def get_precompiling(self) -> bool:
        return self._precompiling

    def get_preprocessing(self) -> bool:
        return self._preprocessing

    def save_flags_to_json(self, file: str) -> None:
        attrs = {
            "compiler": self.get_compiler(),
            "definitions": self.get_definitions(),
            "include_dirs": self.get_include_dirs(),
            "cflags": self.get_cflags(),
            "ldflags": self.get_ldflags(),
            "libraries_dirs": self.get_libraries_dirs(),
            "libraries": self.get_libraries(),
            "passthrough_args": self.get_passthrough_args(),
            "aot_mode": self.get_aot_mode(),
            "use_relative_path": self.get_use_relative_path(),
            "compile_only": self.get_compile_only(),
        }

        with open(file, "w") as f:
            json.dump(attrs, f)


def _get_warning_all_cflag(warning_all: bool = True) -> list[str]:
    return ["Wall"] if warning_all else []


def _get_cpp_std_cflag(std_num: str = "c++23") -> list[str]:
    return [f"std={std_num}"]


def _get_os_related_cpp_cflags(cpp_compiler: str) -> list[str]:
    cflags = ["Wno-unused-variable", "Wno-unknown-pragmas"]
    if _is_clang(cpp_compiler):
        ignored_optimization_argument = (
            "Werror=ignored-optimization-argument"
            if config.aot_inductor.raise_error_on_ignored_optimization
            else "Wno-ignored-optimization-argument"
        )
        cflags.append(ignored_optimization_argument)
    if _is_gcc(cpp_compiler):
        # Issue all the warnings demanded by strict ISO C and ISO C++.
        # Ref: https://github.com/pytorch/pytorch/issues/153180#issuecomment-2986676878
        cflags.append("pedantic")
    return cflags


def _get_os_related_cpp_definitions(cpp_compiler: str) -> list[str]:
    os_definitions: list[str] = []
    return os_definitions


def _get_ffast_math_flags() -> list[str]:
    # This starts from the flags implied by -ffast-math, as in
    # https://github.com/gcc-mirror/gcc/blob/4700ad1c78ccd7767f846802fca148b2ea9a1852/gcc/opts.cc#L3458-L3468
    # however gcc<13 sets the FTZ/DAZ flags for runtime on x86 even if we have
    # -ffast-math -fno-unsafe-math-optimizations because the flags for runtime
    # are added by linking in crtfastmath.o. This is done by the spec file which
    # only does globbing for -ffast-math.
    flags = [
        "fno-trapping-math",
        "funsafe-math-optimizations",
        "ffinite-math-only",
        "fno-signed-zeros",
    ]

    flags.append("fno-finite-math-only")
    if not config.cpp.enable_unsafe_math_opt_flag:
        flags.append("fno-unsafe-math-optimizations")
    # Keep errno-preserving libm semantics.  With -fno-math-errno, GCC can
    # inline/transform libm call pairs like sin(atan(x)) in ways that do not
    # preserve NaN values (see https://github.com/pytorch/pytorch/issues/143978).
    flags.append("fmath-errno")
    flags.append(f"ffp-contract={config.cpp.enable_floating_point_contract_flag}")

    if is_gcc():
        flags.append("fexcess-precision=fast")

    return flags


def _get_inductor_debug_symbol_cflags() -> tuple[list[str], list[str]]:
    """
    When we turn on generate debug symbol.
    On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
    On Linux, it should create some debug sections in binary file.
    """
    cflags: list[str] = []
    ldflags: list[str] = []

    cflags.append("g")

    return cflags, ldflags


@functools.cache
def _get_linux_aarch64_cpu_flags() -> OrderedSet[str]:
    flags: OrderedSet[str] = OrderedSet()

    if platform.machine() not in ("aarch64", "arm64"):
        return flags

    if not sys.platform.startswith("linux"):
        return flags

    return flags


@functools.cache
def _get_linux_aarch64_arch_flag(cpp_compiler: str) -> str:
    flags = _get_linux_aarch64_cpu_flags()

    if _is_gcc(cpp_compiler) and _is_gcc_version_less_than(cpp_compiler, 13):
        if OrderedSet(["bf16"]).issubset(flags):
            return "march=armv8.6-a+bf16"

    return "march=native"


def _get_cpu_arch_cflags(cpp_compiler: str) -> list[str]:

    march = config.cpp.march
    if march == "":
        return []

    # -march=native is not recognized on Apple Silicon, so the default macOS
    # behavior is no architecture flag unless the user explicitly configures one.
    if sys.platform == "darwin" and march is None:
        return []

    machine = platform.machine()
    if march is None:
        if machine in ("aarch64", "arm64"):
            return [_get_linux_aarch64_arch_flag(cpp_compiler)]
        return ["march=native"]

    return [f"march={march}"]


def _get_optimization_cflags(
    cpp_compiler: str, min_optimize: bool = False
) -> tuple[list[str], list[str]]:
    cflags: list[str] = []
    ldflags: list[str] = []

    should_use_optimized_flags = not (
        config.aot_inductor.debug_compile
        or os.environ.get("TORCHINDUCTOR_DEBUG_COMPILE", "0") == "1"
    )
    should_add_debug_symbol_flags = (
        config.aot_inductor.debug_compile
        or config.aot_inductor.debug_symbols
        or os.environ.get("TORCHINDUCTOR_DEBUG_COMPILE", "0") == "1"
        or os.environ.get("TORCHINDUCTOR_DEBUG_SYMBOL", "0") == "1"
    )
    if should_use_optimized_flags:
        cflags += [
            config.aot_inductor.compile_wrapper_opt_level if min_optimize else "O3",
            "DNDEBUG",
        ]
    else:
        cflags += ["O0"]

    if should_add_debug_symbol_flags:
        debug_cflags, debug_ldflags = _get_inductor_debug_symbol_cflags()
        cflags += debug_cflags
        ldflags += debug_ldflags

    if config.aot_inductor.enable_frame_pointer:
        cflags.append("fno-omit-frame-pointer")

    if config.aot_inductor.enable_line_tables and not should_add_debug_symbol_flags:
        if _is_clang(cpp_compiler):
            cflags.append("gline-tables-only")
        else:
            cflags.append("g1")

    cflags += _get_ffast_math_flags()

    # on macos, unknown argument: '-fno-tree-loop-vectorize'
    if sys.platform != "darwin" and _is_gcc(cpp_compiler):
        cflags.append("fno-tree-loop-vectorize")
    cflags += _get_cpu_arch_cflags(cpp_compiler)

    if config.aot_inductor.enable_lto and _is_clang(cpp_compiler):
        cflags.append("flto=thin")

    return cflags, ldflags


def _get_shared_cflags(cpp_compiler: str, do_link: bool) -> list[str]:
    if platform.system() == "Darwin" and _is_clang(cpp_compiler):
        # This causes undefined symbols to behave the same as linux
        return ["shared", "fPIC", "undefined dynamic_lookup"]
    flags = []
    if do_link:
        flags.append("shared")

    flags.append("fPIC")
    return flags


def get_cpp_options(
    cpp_compiler: str,
    do_link: bool,
    warning_all: bool = True,
    extra_flags: Sequence[str] = (),
    min_optimize: bool = False,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    definitions: list[str] = []
    include_dirs: list[str] = []
    cflags: list[str] = []
    ldflags: list[str] = []
    libraries_dirs: list[str] = []
    libraries: list[str] = []
    passthrough_args: list[str] = []

    opt_cflags, opt_ldflags = _get_optimization_cflags(cpp_compiler, min_optimize)

    cflags = (
        opt_cflags
        + _get_shared_cflags(cpp_compiler, do_link)
        + _get_warning_all_cflag(warning_all)
        + _get_cpp_std_cflag()
        + _get_os_related_cpp_cflags(cpp_compiler)
    )

    definitions += _get_os_related_cpp_definitions(cpp_compiler)

    if config.aot_inductor.enable_lto and _is_clang(cpp_compiler):
        ldflags.append("fuse-ld=lld")
        ldflags.append("flto=thin")

    passthrough_args.append(" ".join(extra_flags))

    return (
        definitions,
        include_dirs,
        cflags,
        ldflags + opt_ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
    )


class CppOptions(BuildOptionsBase):
    """
    This class is inherited from BuildOptionsBase, and as cxx build options.
    This option need contains basic cxx build option, which contains:
    1. OS related args.
    2. Toolchains related args.
    3. Cxx standard related args.
    Note:
    1. This Options is good for assist modules build, such as x86_isa_help.
    """

    def __init__(
        self,
        compile_only: bool = False,
        warning_all: bool = True,
        extra_flags: Sequence[str] = (),
        use_relative_path: bool = False,
        compiler: str = "",
        min_optimize: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        super().__init__(
            compile_only=compile_only,
            use_relative_path=use_relative_path,
            precompiling=precompiling,
            preprocessing=preprocessing,
        )
        self._compiler = compiler if compiler else get_cpp_compiler()

        (
            definitions,
            include_dirs,
            cflags,
            ldflags,
            libraries_dirs,
            libraries,
            passthrough_args,
        ) = get_cpp_options(
            cpp_compiler=self._compiler,
            do_link=not (compile_only or precompiling or preprocessing),
            extra_flags=extra_flags,
            warning_all=warning_all,
            min_optimize=min_optimize,
        )

        _append_list(self._definitions, definitions)
        _append_list(self._include_dirs, include_dirs)
        _append_list(self._cflags, cflags)
        _append_list(self._ldflags, ldflags)
        _append_list(self._libraries_dirs, libraries_dirs)
        _append_list(self._libraries, libraries)
        _append_list(self._passthrough_args, passthrough_args)
        self._finalize_options()


def _get_torch_cpp_wrapper_definition() -> list[str]:
    defs = ["TORCH_INDUCTOR_CPP_WRAPPER", "STANDALONE_TORCH_HEADER"]
    if config.cpp_cache_precompile_headers:
        defs.append("TORCH_INDUCTOR_PRECOMPILE_HEADERS")
    return defs


def _use_custom_generated_macros() -> list[str]:
    return [" C10_USING_CUSTOM_GENERATED_MACROS"]


def _use_fb_internal_macros() -> list[str]:
    return []


def _setup_standard_sys_libs(
    cpp_compiler: str,
    aot_mode: bool,
    use_relative_path: bool,
    cpp_stdlib: CppStdlib,
) -> tuple[list[str], list[str], list[str], list[str]]:
    cflags: list[str] = []
    include_dirs: list[str] = []
    passthrough_args: list[str] = []
    ldflags: list[str] = []

    return cflags, include_dirs, passthrough_args, ldflags


def _get_build_args_of_chosen_isa(vec_isa: VecISA) -> tuple[list[str], list[str]]:
    macros: list[str] = []
    build_flags: list[str] = []
    if vec_isa != invalid_vec_isa:
        # Add Windows support later.
        macros.extend(copy.deepcopy(x) for x in vec_isa.build_macro())

        build_flags = [vec_isa.build_arch_flags()]

    return macros, build_flags


def _get_torch_related_args(
    include_pytorch: bool, aot_mode: bool
) -> tuple[list[str], list[str], list[str]]:
    from torch.utils.cpp_extension import include_paths, TORCH_LIB_PATH

    libraries = []
    include_dirs = include_paths()

    if config.aot_inductor.link_libtorch:
        libraries_dirs = [TORCH_LIB_PATH]
        if sys.platform != "darwin":
            libraries.extend(["torch", "torch_cpu"])
            if not aot_mode:
                libraries.append("torch_python")
    else:
        libraries_dirs = []

    return include_dirs, libraries_dirs, libraries


def _get_python_include_dirs() -> list[str]:
    include_dir = Path(sysconfig.get_path("include"))
    # On Darwin Python executable from a framework can return
    # non-existing /Library/Python/... include path, in which case
    # one should use Headers folder from the framework
    if not include_dir.exists() and platform.system() == "Darwin":
        std_lib = Path(sysconfig.get_path("stdlib"))
        include_dir = (std_lib.parent.parent / "Headers").absolute()
    if not (include_dir / "Python.h").exists():
        warnings.warn(f"Can't find Python.h in {str(include_dir)}")
    return [str(include_dir)]


def _get_python_related_args() -> tuple[list[str], list[str]]:
    python_include_dirs = _get_python_include_dirs()
    python_include_path = sysconfig.get_path("include", scheme="posix_prefix")
    if python_include_path is not None:
        python_include_dirs.append(python_include_path)

    python_lib_path = [sysconfig.get_config_var("LIBDIR")]

    return python_include_dirs, python_lib_path


@functools.cache
def is_conda_llvm_openmp_installed() -> bool:
    try:
        command = "conda list llvm-openmp --json"
        output = subprocess.check_output(command.split()).decode("utf8")
        return len(json.loads(output)) > 0
    except subprocess.SubprocessError, FileNotFoundError:
        return False


@functools.cache
def homebrew_libomp() -> tuple[bool, str]:
    try:
        # check if `brew` is installed
        if shutil.which("brew") is None:
            return False, ""
        # get the location of `libomp` if it is installed
        # this is the location that `libomp` **would** be installed
        # see https://github.com/Homebrew/brew/issues/10261#issuecomment-756563567 for details
        libomp_path = (
            subprocess.check_output(["brew", "--prefix", "libomp"])
            .decode("utf8")
            .strip()
        )
        # check if `libomp` is installed
        omp_available = os.path.exists(libomp_path)
        return omp_available, libomp_path
    except subprocess.SubprocessError:
        return False, ""


@functools.cache
def perload_clang_libomp_win(cpp_compiler: str, omp_name: str) -> None:
    try:
        output = subprocess.check_output(
            _compiler_command(cpp_compiler, "-print-file-name=bin")
        ).decode("utf8")
        omp_path = os.path.join(output.rstrip(), omp_name)
        if os.path.isfile(omp_path):
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            cdll.LoadLibrary(omp_path)
    except subprocess.SubprocessError:
        pass


@functools.cache
def perload_icx_libomp_win(cpp_compiler: str) -> None:
    def _load_icx_built_in_lib_by_name(cpp_compiler: str, lib_name: str) -> bool:
        try:
            output = subprocess.check_output(
                _compiler_command(cpp_compiler, f"-print-file-name={lib_name}"),
                stderr=subprocess.DEVNULL,
            ).decode(*SUBPROCESS_DECODE_ARGS)
            omp_path = output.rstrip()
            if os.path.isfile(omp_path):
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                cdll.LoadLibrary(omp_path)
                return True
        except subprocess.SubprocessError:
            pass
        return False

    """
    Intel Compiler implemented more math libraries than clang, for performance purposes.
    We need to preload them like openmp library.
    """
    preload_list = [
        "libiomp5md.dll",  # openmp
        "svml_dispmd.dll",  # svml library
        "libmmd.dll",  # libm
    ]

    for lib_name in preload_list:
        _load_icx_built_in_lib_by_name(cpp_compiler, lib_name)


def _get_openmp_args(
    cpp_compiler: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    cflags: list[str] = []
    ldflags: list[str] = []
    include_dir_paths: list[str] = []
    lib_dir_paths: list[str] = []
    libs: list[str] = []
    passthrough_args: list[str] = []

    if _IS_MACOS:
        # Per https://mac.r-project.org/openmp/ right way to pass `openmp` flags to MacOS is via `-Xclang`
        cflags.append("Xclang")
        cflags.append("fopenmp")

        # only Apple builtin compilers (Apple Clang++) require openmp
        omp_available = not _is_apple_clang(cpp_compiler)

        # check the `OMP_PREFIX` environment first
        omp_prefix = os.getenv("OMP_PREFIX")
        if omp_prefix is not None:
            header_path = os.path.join(omp_prefix, "include", "omp.h")
            valid_env = os.path.exists(header_path)
            if valid_env:
                include_dir_paths.append(os.path.join(omp_prefix, "include"))
                lib_dir_paths.append(os.path.join(omp_prefix, "lib"))
            else:
                warnings.warn("environment variable `OMP_PREFIX` is invalid.")
            omp_available = omp_available or valid_env

        if not omp_available:
            libs.append("omp")

        # prefer to use openmp from `conda install llvm-openmp`
        conda_prefix = os.getenv("CONDA_PREFIX")
        if not omp_available and conda_prefix is not None:
            omp_available = is_conda_llvm_openmp_installed()
            if omp_available:
                conda_lib_path = os.path.join(conda_prefix, "lib")
                include_dir_paths.append(os.path.join(conda_prefix, "include"))
                lib_dir_paths.append(conda_lib_path)
                # Prefer Intel OpenMP on x86 machine
                if os.uname().machine == "x86_64" and os.path.exists(
                    os.path.join(conda_lib_path, "libiomp5.dylib")
                ):
                    libs.append("iomp5")

        # next, try to use openmp from `brew install libomp`
        if not omp_available:
            omp_available, libomp_path = homebrew_libomp()
            if omp_available:
                include_dir_paths.append(os.path.join(libomp_path, "include"))
                lib_dir_paths.append(os.path.join(libomp_path, "lib"))

        # if openmp is still not available, we let the compiler to have a try,
        # and raise error together with instructions at compilation error later
    if _is_clang(cpp_compiler):
        # TODO: fix issue, can't find omp.h
        cflags.append("fopenmp")
        libs.append("gomp")
    elif _is_intel_compiler(cpp_compiler):
        cflags.append("fiopenmp")
    else:
        cflags.append("fopenmp")
        libs.append("gomp")

    return cflags, ldflags, include_dir_paths, lib_dir_paths, libs, passthrough_args


def get_mmap_self_macro(
    use_mmap_weights: bool, use_mmap_weights_external: bool
) -> list[str]:
    macros = []

    if use_mmap_weights and use_mmap_weights_external:
        raise RuntimeError(
            "Only one of use_mmap_weights and use_mmap_weights_external should be true"
        )
    if use_mmap_weights:
        macros.append(" USE_MMAP_SELF")
    elif use_mmap_weights_external:
        macros.append(" USE_MMAP_EXTERNAL")
    return macros


def get_caching_allocator_macro() -> list[str]:
    from torch._inductor import config

    macros = []
    if config.aot_inductor.weight_use_caching_allocator:
        macros.append(" AOT_INDUCTOR_USE_CACHING_ALLOCATOR")
    return macros


def get_cpp_torch_options(
    cpp_compiler: str,
    vec_isa: VecISA,
    include_pytorch: bool,
    aot_mode: bool,
    use_relative_path: bool,
    use_mmap_weights: bool,
    use_mmap_weights_external: bool,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    """
    This function is used to get the build args of torch related build options.
    1. Torch include_directories, libraries, libraries_directories.
    2. Python include_directories, libraries, libraries_directories.
    3. OpenMP related.
    4. Torch MACROs.
    5. MISC
    6. Return the build args
    """
    definitions: list[str] = []
    include_dirs: list[str] = []
    cflags: list[str] = []
    ldflags: list[str] = []
    libraries_dirs: list[str] = []
    libraries: list[str] = []
    passthrough_args: list[str] = []

    torch_cpp_wrapper_definitions = _get_torch_cpp_wrapper_definition()
    use_custom_generated_macros_definitions = _use_custom_generated_macros()

    (
        sys_libs_cflags,
        sys_libs_include_dirs,
        sys_libs_passthrough_args,
        sys_libs_ldflags,
    ) = _setup_standard_sys_libs(cpp_compiler, aot_mode, use_relative_path, cpp_stdlib)

    isa_macros, isa_ps_args_build_flags = _get_build_args_of_chosen_isa(vec_isa)

    (
        torch_include_dirs,
        torch_libraries_dirs,
        torch_libraries,
    ) = _get_torch_related_args(include_pytorch=include_pytorch, aot_mode=aot_mode)

    python_include_dirs, python_libraries_dirs = _get_python_related_args()

    (
        omp_cflags,
        omp_ldflags,
        omp_include_dir_paths,
        omp_lib_dir_paths,
        omp_lib,
        omp_passthrough_args,
    ) = _get_openmp_args(cpp_compiler)

    fb_macro_passthrough_args = _use_fb_internal_macros()

    mmap_self_macros = get_mmap_self_macro(use_mmap_weights, use_mmap_weights_external)
    caching_allocator_macros = get_caching_allocator_macro()

    definitions = (
        torch_cpp_wrapper_definitions
        + use_custom_generated_macros_definitions
        + isa_macros
        + fb_macro_passthrough_args
        + mmap_self_macros
        + caching_allocator_macros
    )
    include_dirs = (
        sys_libs_include_dirs
        + python_include_dirs
        + torch_include_dirs
        + omp_include_dir_paths
    )
    cflags = sys_libs_cflags + omp_cflags
    ldflags = sys_libs_ldflags + omp_ldflags
    libraries_dirs = python_libraries_dirs + torch_libraries_dirs + omp_lib_dir_paths
    libraries = torch_libraries + omp_lib
    passthrough_args = (
        sys_libs_passthrough_args + isa_ps_args_build_flags + omp_passthrough_args
    )

    return (
        definitions,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
    )


class CppTorchOptions(CppOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options. And then it will maintains torch related build
    args.
    1. Torch include_directories, libraries, libraries_directories.
    2. Python include_directories, libraries, libraries_directories.
    3. OpenMP related.
    4. Torch MACROs.
    5. MISC
    """

    def __init__(
        self,
        vec_isa: VecISA = invalid_vec_isa,
        include_pytorch: bool = False,
        warning_all: bool = True,
        aot_mode: bool = False,
        compile_only: bool = False,
        use_relative_path: bool = False,
        use_mmap_weights: bool = False,
        use_mmap_weights_external: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
        compiler: str = "",
        min_optimize: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
    ) -> None:
        super().__init__(
            compile_only=compile_only,
            warning_all=warning_all,
            extra_flags=extra_flags,
            use_relative_path=use_relative_path,
            compiler=compiler,
            min_optimize=min_optimize,
            precompiling=precompiling,
            preprocessing=preprocessing,
        )

        self._aot_mode = aot_mode

        (
            torch_definitions,
            torch_include_dirs,
            torch_cflags,
            torch_ldflags,
            torch_libraries_dirs,
            torch_libraries,
            torch_passthrough_args,
        ) = get_cpp_torch_options(
            cpp_compiler=self._compiler,
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            use_relative_path=use_relative_path,
            use_mmap_weights=use_mmap_weights,
            use_mmap_weights_external=use_mmap_weights_external,
        )

        _append_list(self._definitions, torch_definitions)
        _append_list(self._include_dirs, torch_include_dirs)
        _append_list(self._cflags, torch_cflags)
        _append_list(self._ldflags, torch_ldflags)
        _append_list(self._libraries_dirs, torch_libraries_dirs)
        _append_list(self._libraries, torch_libraries)
        _append_list(self._passthrough_args, torch_passthrough_args)
        self._finalize_options()


@functools.lru_cache(8)
def _find_libcudart_static(path: str) -> Path | None:
    lib_dirs = list(Path(path).rglob("libcudart_static.a"))
    if lib_dirs:
        return lib_dirs[0].resolve().parent
    log_msg = f'"libcudart_static.a" not found under {path}'
    log.info(log_msg)
    return None


def _transform_cuda_paths(lpaths: list[str]) -> None:
    # This handles two cases:
    # 1. Cases where libs are in (e.g.) lib/cuda-12 and lib/cuda-12/stubs
    # 2. Linux machines may have CUDA installed under either lib64/ or lib/
    for i, path in enumerate(lpaths):
        if "CUDA_HOME" in os.environ and path.startswith(os.environ["CUDA_HOME"]):
            lib_dir: Path | None = _find_libcudart_static(path)
            if lib_dir is None:
                continue
            lpaths[i] = str(lib_dir)
            stub_dir = lib_dir / "stubs"
            if stub_dir.exists():
                lpaths.append(str(stub_dir))


def get_cpp_torch_device_options(
    device_type: str,
    aot_mode: bool = False,
    compile_only: bool = False,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    """
    This function is used to get the build args of device related build options.
    1. Device include_directories, libraries, libraries_directories.
    2. Device MACROs.
    3. MISC
    4. Return the build args
    """
    try:
        device_options = get_interface_for_device(device_type).get_cpp_device_options(
            aot_mode, compile_only
        )
    except NotImplementedError:
        device_options = None
    if device_options is not None:
        return device_options

    definitions: list[str] = []
    include_dirs: list[str] = []
    cflags: list[str] = []
    ldflags: list[str] = []
    libraries_dirs: list[str] = []
    libraries: list[str] = []
    passthrough_args: list[str] = []

    from torch.utils import cpp_extension

    # cpp_extension resolves CUDA_HOME into a module-level global at import time,
    # so an fbcode process that imported it before the env var above was written
    # caches None; refresh the global here so the just-set CUDA_HOME env actually
    # takes effect for include_paths/library_paths below.
    if cpp_extension.CUDA_HOME is None and os.environ.get("CUDA_HOME"):
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]

    include_dirs = cpp_extension.include_paths(
        device_type, config.aot_inductor.link_libtorch is None
    )
    link_libtorch = config.aot_inductor.link_libtorch
    libraries_dirs = cpp_extension.library_paths(
        device_type,
        torch_include_dirs=link_libtorch,
        cross_target_platform=config.aot_inductor.cross_target_platform,
    )
    if device_type == "cuda":
        definitions.append(" USE_ROCM" if torch.version.hip else " USE_CUDA")

        if torch.version.hip is not None:
            if not link_libtorch:
                libraries += ["amdhip64"]
            else:
                libraries += ["torch_hip"]
            definitions.append(" __HIP_PLATFORM_AMD__")
        else:
            if not link_libtorch:
                libraries += ["cuda"]
            else:
                libraries += ["cuda", "torch_cuda"]
            libraries += ["cudart"]
            _transform_cuda_paths(libraries_dirs)

    if device_type == "xpu":
        definitions.append(" USE_XPU")
        xpu_error_string = (
            "Intel GPU driver is not properly installed, please follow the instruction "
            "in https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support."
        )
        # Suppress multi-line comment warnings in sycl headers
        cflags += ["Wno-comment"]
        if not find_library("ze_loader"):
            raise OSError(xpu_error_string)

        libraries += ["ze_loader", "sycl"]
        if link_libtorch:
            libraries += ["torch_xpu"]

    if device_type == "mps":
        definitions.append(" USE_MPS")

    if config.aot_inductor.custom_op_libs:
        libraries += config.aot_inductor.custom_op_libs

    return (
        definitions,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthrough_args,
    )


class CppTorchDeviceOptions(CppTorchOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options and torch common build options. And then it will
    maintains cuda/xpu device related build args.
    """

    def __init__(
        self,
        vec_isa: VecISA = invalid_vec_isa,
        include_pytorch: bool = False,
        device_type: str = "cuda",
        aot_mode: bool = False,
        compile_only: bool = False,
        use_relative_path: bool = False,
        use_mmap_weights: bool = False,
        use_mmap_weights_external: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
        min_optimize: bool = False,
        precompiling: bool = False,
        preprocessing: bool = False,
        compiler: str = "",
    ) -> None:
        super().__init__(
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_relative_path=use_relative_path,
            use_mmap_weights=use_mmap_weights,
            use_mmap_weights_external=use_mmap_weights_external,
            extra_flags=extra_flags,
            min_optimize=min_optimize,
            precompiling=precompiling,
            preprocessing=preprocessing,
            compiler=compiler,
        )

        device_definitions: list[str] = []
        device_include_dirs: list[str] = []
        device_cflags: list[str] = []
        device_ldflags: list[str] = []
        device_libraries_dirs: list[str] = []
        device_libraries: list[str] = []
        device_passthrough_args: list[str] = []

        (
            device_definitions,
            device_include_dirs,
            device_cflags,
            device_ldflags,
            device_libraries_dirs,
            device_libraries,
            device_passthrough_args,
        ) = get_cpp_torch_device_options(
            device_type=device_type,
            aot_mode=aot_mode,
            compile_only=compile_only,
        )
        _append_list(self._definitions, device_definitions)
        _append_list(self._include_dirs, device_include_dirs)
        _append_list(self._cflags, device_cflags)
        _append_list(self._ldflags, device_ldflags)
        _append_list(self._libraries_dirs, device_libraries_dirs)
        _append_list(self._libraries, device_libraries)
        _append_list(self._passthrough_args, device_passthrough_args)
        self._finalize_options()

    def _finalize_options(self) -> None:
        super()._finalize_options()


def get_name_and_dir_from_output_file_path(
    file_path: str,
) -> tuple[str, str]:
    """
    This function help prepare parameters to new cpp_builder.
    Example:
        input_code: /tmp/tmpof1n5g7t/5c/c5crkkcdvhdxpktrmjxbqkqyq5hmxpqsfza4pxcf3mwk42lphygc.cpp
        name, dir = get_name_and_dir_from_output_file_path(input_code)
    Run result:
        name = c5crkkcdvhdxpktrmjxbqkqyq5hmxpqsfza4pxcf3mwk42lphygc
        dir = /tmp/tmpof1n5g7t/5c/

    put 'name' and 'dir' to CppBuilder's 'name' and 'output_dir'.
    CppBuilder --> get_target_file_path will format output path according OS:
    Linux: /tmp/tmppu87g3mm/zh/czhwiz4z7ca7ep3qkxenxerfjxy42kehw6h5cjk6ven4qu4hql4i.so
    Windows: [Windows temp path]/tmppu87g3mm/zh/czhwiz4z7ca7ep3qkxenxerfjxy42kehw6h5cjk6ven4qu4hql4i.dll
    """
    name_and_ext = os.path.basename(file_path)
    name, _ext = os.path.splitext(name_and_ext)
    dir = os.path.dirname(file_path)

    return name, dir


class CppBuilder:
    """
    CppBuilder is a cpp jit builder, and it supports both Windows, Linux and MacOS.
    Args:
        name:
            1. Build target name, the final target file will append extension type automatically.
            2. Due to the CppBuilder is supports multiple OS, it will maintains ext for OS difference.
        sources:
            Source code file list to be built.
        BuildOption:
            Build options to the builder.
        output_dir:
            1. The output_dir the target file will output to.
            2. The default value is empty string, and then use the current dir as output dir.
            3. Final target file: output_dir/name.ext
    """

    @staticmethod
    def __get_python_module_flags() -> tuple[str, str]:
        extension = ".so"
        output_flags = "-o"
        return extension, output_flags

    @staticmethod
    def __get_object_flags() -> tuple[str, str]:
        extension = ".o"
        output_flags = "-c -o"
        return extension, output_flags

    @staticmethod
    def __get_precompiled_header_flags() -> tuple[str, str]:
        extension = ".gch" if not is_gcc() else ".pch"
        output_flags = "-o"
        return extension, output_flags

    @staticmethod
    def __get_preprocessor_output_flags() -> tuple[str, str]:
        extension = ".i"
        output_flags = "-E -P -o"
        return extension, output_flags

    def __init__(
        self,
        name: str,
        sources: str | list[str],
        BuildOption: BuildOptionsBase,
        output_dir: str = "",
    ) -> None:
        self._compiler = ""
        self._cflags_args = ""
        self._definitions_args = ""
        self._include_dirs_args = ""
        self._ldflags_args = ""
        self._libraries_dirs_args = ""
        self._libraries_args = ""
        self._passthrough_parameters_args = ""

        # When relative path is used, we need to maintain the source dir list.
        self._orig_source_paths = []
        self._output_dir = ""
        self._target_file = ""

        self._use_relative_path: bool = False
        self._aot_mode: bool = False

        self._name = name
        self._target_name = (
            config.aot_inductor.model_name_for_generated_files or "aoti_model"
        )

        # Code start here, initial self internal variables firstly.
        self._build_option = BuildOption
        self._compiler = BuildOption.get_compiler()
        self._use_relative_path = BuildOption.get_use_relative_path()
        self._aot_mode = BuildOption.get_aot_mode()

        self._output_dir = output_dir

        self._compile_only = BuildOption.get_compile_only()
        self._precompiling = BuildOption.get_precompiling()
        self._preprocessing = BuildOption.get_preprocessing()
        # Only one of these options (if any) should be true at any given time.
        if sum((self._compile_only, self._precompiling, self._preprocessing)) > 1:
            raise AssertionError(
                "at most one of compile_only, precompiling, preprocessing may be set"
            )
        self._do_link = not (
            self._compile_only or self._precompiling or self._preprocessing
        )

        if self._compile_only:
            file_ext, output_flags = self.__get_object_flags()
        elif self._precompiling:
            file_ext, output_flags = self.__get_precompiled_header_flags()
        elif self._preprocessing:
            file_ext, output_flags = self.__get_preprocessor_output_flags()
        else:
            file_ext, output_flags = self.__get_python_module_flags()
        self._target_file = os.path.join(self._output_dir, f"{self._name}{file_ext}")

        relative_target_file = (
            os.path.basename(self._target_file)
            if self._use_relative_path
            else self._target_file
        )
        self._output = f"{output_flags} {relative_target_file}"

        if isinstance(sources, str):
            sources = [sources]

        if self._precompiling:
            if len(sources) != 1:
                raise AssertionError(
                    f"expected exactly one source when precompiling, got {len(sources)}"
                )
            # See above; we can currently assume this is not on MSVC.
            self._sources_args = f"-x c++-header {sources[0]}"
            if self._use_relative_path and _is_clang(BuildOption.get_compiler()):
                # Store PCH paths relative to -isysroot so the .pch can
                # be used from a different build directory.  The matching
                self._cflags_args += " -relocatable-pch -Xclang -fno-pch-timestamp "
        else:
            self._sources_args = " ".join(sources)

        for cflag in BuildOption.get_cflags():
            self._cflags_args += f"-{cflag} "

        for definition in BuildOption.get_definitions():
            self._definitions_args += f"-D {definition} "

        if precompiled_header := BuildOption.precompiled_header:
            self._include_dirs_args = f"-include {precompiled_header} "
            if self._use_relative_path and _is_clang(BuildOption.get_compiler()):
                # Skip clang's own PCH validation during consumption.
                # _precompile_header() already handles cache invalidation
                # via content hashing, and -fno-validate-pch allows the
                # PCH to be used even when the original source file is at
                # a different path (e.g. across Remote Execution workers).
                self._cflags_args += " -Xclang -fno-validate-pch "

        for inc_dir in BuildOption.get_include_dirs():
            self._include_dirs_args += f"-I{shlex.quote(inc_dir)} "

        for ldflag in BuildOption.get_ldflags():
            self._ldflags_args += f"-{ldflag} "

        for lib_dir in BuildOption.get_libraries_dirs():
            self._libraries_dirs_args += f"-L{shlex.quote(lib_dir)} "

        for lib in BuildOption.get_libraries():
            self._libraries_args += f"-l{lib} "

        for passthrough_arg in BuildOption.get_passthrough_args():
            self._passthrough_parameters_args += f"{passthrough_arg} "

    def get_command_line(self) -> str:
        def format_build_command(
            compiler: str,
            sources: str,
            include_dirs_args: str,
            definitions_args: str,
            cflags_args: str,
            ldflags_args: str,
            libraries_args: str,
            libraries_dirs_args: str,
            passthrough_args: str,
            output: str,
        ) -> str:
            cmd = (
                f"{compiler} {sources} {definitions_args} {cflags_args} "
                f"{include_dirs_args} {passthrough_args} {output}"
            )
            if self._do_link:
                cmd += f" {ldflags_args} {libraries_args} {libraries_dirs_args}"
            return cmd

        command_line = format_build_command(
            compiler=self._compiler,
            sources=self._sources_args,
            include_dirs_args=self._include_dirs_args,
            definitions_args=self._definitions_args,
            cflags_args=self._cflags_args,
            ldflags_args=self._ldflags_args,
            libraries_args=self._libraries_args,
            libraries_dirs_args=self._libraries_dirs_args,
            passthrough_args=self._passthrough_parameters_args,
            output=self._output,
        )
        return command_line

    def get_target_file_path(self) -> str:
        return normalize_path_separator(self._target_file)

    def build(self) -> None:
        """
        It is must need a temporary directory to store object files in Windows.
        After build completed, delete the temporary directory to save disk space.
        """
        _create_if_dir_not_exist(self._output_dir)
        _build_tmp_dir = os.path.join(
            self._output_dir, f"{self._name}_{_BUILD_TEMP_DIR}"
        )
        _create_if_dir_not_exist(_build_tmp_dir)

        build_cmd = self.get_command_line()
        run_compile_cmd(build_cmd, cwd=_build_tmp_dir)
        _remove_dir(_build_tmp_dir)

    def save_compile_cmd_to_cmake(
        self,
        cmake_path: str,
        device_type: str,
    ) -> None:
        """
        Save global cmake settings here, e.g. compiler options.
        If targeting CUDA, also emit a custom function to embed CUDA kernels.
        """

        definitions = " ".join(self._build_option.get_definitions())
        target_library_type = (
            "STATIC" if not config.aot_inductor.dynamic_linkage else "SHARED"
        )

        contents = textwrap.dedent(
            f"""
            cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
            project({self._target_name} LANGUAGES CXX)
            set(CMAKE_CXX_STANDARD 23)

            # Set a library target
            add_library({self._target_name} {target_library_type})

            """
        )

        if config.aot_inductor.link_libtorch or config.test_configs.use_libtorch:
            # When compile_standalone is True, the generated cpp project should
            # not use Torch. But for unit testing purpose, we need to use Torch here.
            contents += textwrap.dedent(
                """
                # May need to point CMAKE_PREFIX_PATH to the right torch location
                find_package(Torch REQUIRED)

                """
            )
            # flags and macros here are mostly CPU specific. Not emitting them for GPU models
            # will make the generated CMake file more portable and won't really hurt performance.
            # NOTE: standalone focuses on GPU now. For CPU, some of the flags and macros may
            # be still needed.
            contents += textwrap.dedent(
                f"""
                # Add macro definitions
                target_compile_definitions({self._target_name} PRIVATE {definitions})

                # Add compile flags
                target_compile_options({self._target_name} PRIVATE {self._cflags_args})

                # Backend-specific flags
                target_compile_options({self._target_name} PRIVATE {self._passthrough_parameters_args} -c)

                """
            )
        else:
            # When compile_standalone is True, use TorchStandalone instead of Torch
            contents += textwrap.dedent(
                f"""
                find_package(TorchStandalone REQUIRED)
                # Set up include directories to find headers at the correct paths
                target_include_directories({self._target_name} PRIVATE ${{TorchStandalone_INCLUDE_DIRS}})
                target_include_directories({self._target_name} PRIVATE ${{TorchStandalone_INCLUDE_DIRS}}/standalone)

                """
            )

        if device_type == "cuda" and torch.version.hip is None:
            from torch._inductor.codegen.cuda import compile_utils

            cuda_arch = compile_utils._aoti_cuda_target_arch()
            cuda_gencode_flags = "\n                                ".join(
                f"-gencode {option}"
                for option in compile_utils._cuda_multi_arch_gencode_options(cuda_arch)
            )
            contents += textwrap.dedent(
                f"""
                enable_language(CUDA)
                set(CMAKE_CUDA_STANDARD 17)
                find_package(CUDAToolkit REQUIRED)
                target_include_directories({self._target_name} PRIVATE ${{CUDAToolkit_INCLUDE_DIRS}})
                target_compile_definitions({self._target_name} PRIVATE USE_CUDA)
                target_link_libraries({self._target_name} PRIVATE cuda CUDA::cudart_static)

                find_program(OBJCOPY_EXECUTABLE objcopy)
                if(NOT OBJCOPY_EXECUTABLE)
                    message(FATAL_ERROR "objcopy not found. Cannot embed fatbin as object file")
                endif()

                set(KERNEL_TARGETS "")
                set(KERNEL_OBJECT_FILES "")
                # Function to embed a single kernel
                function(embed_gpu_kernel KERNEL_NAME PTX_FILE)
                    set(FATBIN_BASENAME ${{KERNEL_NAME}}.fatbin)
                    set(FATBIN_FILE ${{CMAKE_CURRENT_BINARY_DIR}}/${{FATBIN_BASENAME}})
                    set(OBJECT_BASENAME ${{KERNEL_NAME}}.fatbin.o)
                    set(OBJECT_FILE ${{CMAKE_CURRENT_BINARY_DIR}}/${{OBJECT_BASENAME}})

                    # --- Define UNIQUE C symbol names ---
                    set(SYMBOL_START __${{KERNEL_NAME}}_start)
                    set(SYMBOL_END __${{KERNEL_NAME}}_end)
                    set(SYMBOL_SIZE __${{KERNEL_NAME}}_size)
                    string(REGEX REPLACE "[^a-zA-Z0-9]" "_" MANGLED_BASENAME ${{FATBIN_FILE}})
                    set(OBJCOPY_START_SYM _binary_${{MANGLED_BASENAME}}_start)
                    set(OBJCOPY_END_SYM _binary_${{MANGLED_BASENAME}}_end)
                    set(OBJCOPY_SIZE_SYM _binary_${{MANGLED_BASENAME}}_size)

                    # --- PTX to FATBIN Command & Target ---
                    add_custom_command(
                        OUTPUT ${{FATBIN_FILE}}
                        COMMAND ${{CUDAToolkit_NVCC_EXECUTABLE}} --fatbin ${{PTX_FILE}} -o ${{FATBIN_FILE}} ${{NVCC_GENCODE_FLAGS}}
                                {cuda_gencode_flags}
                        DEPENDS ${{PTX_FILE}}
                    )

                    # --- FATBIN to Object File (.o) Command ---
                    add_custom_command(
                        OUTPUT ${{OBJECT_FILE}}
                        COMMAND ${{CMAKE_LINKER}} -r -b binary -z noexecstack -o ${{OBJECT_FILE}} ${{FATBIN_FILE}}
                        COMMAND ${{OBJCOPY_EXECUTABLE}} --rename-section .data=.rodata,alloc,load,readonly,data,contents
                                ${{OBJECT_FILE}}
                        COMMAND ${{OBJCOPY_EXECUTABLE}}
                                --redefine-sym ${{OBJCOPY_START_SYM}}=${{SYMBOL_START}}
                                --redefine-sym ${{OBJCOPY_END_SYM}}=${{SYMBOL_END}}
                                --redefine-sym ${{OBJCOPY_SIZE_SYM}}=${{SYMBOL_SIZE}}
                                ${{OBJECT_FILE}}
                        DEPENDS ${{FATBIN_FILE}}
                    )
                    add_custom_target(build_kernel_object_${{KERNEL_NAME}} DEPENDS ${{OBJECT_FILE}})

                    # --- Add to a list for linking later ---
                    set(KERNEL_TARGETS ${{KERNEL_TARGETS}} build_kernel_object_${{KERNEL_NAME}} PARENT_SCOPE)
                    set(KERNEL_OBJECT_FILES ${{KERNEL_OBJECT_FILES}} ${{OBJECT_FILE}} PARENT_SCOPE)
                endfunction()

                """
            )
        elif device_type == "xpu":
            contents += textwrap.dedent(
                """
                find_program(OBJCOPY_EXECUTABLE objcopy)
                if(NOT OBJCOPY_EXECUTABLE)
                    message(FATAL_ERROR "objcopy not found. Cannot embed spv as object file")
                endif()

                set(KERNEL_TARGETS "")
                set(KERNEL_OBJECT_FILES "")
                # Function to embed a single kernel
                function(embed_gpu_kernel KERNEL_NAME SPV_FILE)
                    set(OBJECT_BASENAME ${KERNEL_NAME}.spv.o)
                    set(OBJECT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${OBJECT_BASENAME})

                    # --- Define UNIQUE C symbol names ---
                    set(SYMBOL_START __${KERNEL_NAME}_start)
                    set(SYMBOL_END __${KERNEL_NAME}_end)
                    set(SYMBOL_SIZE __${KERNEL_NAME}_size)
                    string(REGEX REPLACE "[^a-zA-Z0-9]" "_" MANGLED_BASENAME ${SPV_FILE})
                    set(OBJCOPY_START_SYM _binary_${MANGLED_BASENAME}_start)
                    set(OBJCOPY_END_SYM _binary_${MANGLED_BASENAME}_end)
                    set(OBJCOPY_SIZE_SYM _binary_${MANGLED_BASENAME}_size)

                    # --- SPV_FILE to Object File (.o) Command ---
                    add_custom_command(
                        OUTPUT ${OBJECT_FILE}
                        COMMAND ${CMAKE_LINKER} -r -b binary -z noexecstack -o ${OBJECT_FILE} ${SPV_FILE}
                        COMMAND ${OBJCOPY_EXECUTABLE} --rename-section .data=.rodata,alloc,load,readonly,data,contents
                                ${OBJECT_FILE}
                        COMMAND ${OBJCOPY_EXECUTABLE}
                                --redefine-sym ${OBJCOPY_START_SYM}=${SYMBOL_START}
                                --redefine-sym ${OBJCOPY_END_SYM}=${SYMBOL_END}
                                --redefine-sym ${OBJCOPY_SIZE_SYM}=${SYMBOL_SIZE}
                                ${OBJECT_FILE}
                        DEPENDS ${SPV_FILE}
                    )
                    add_custom_target(build_kernel_object_${KERNEL_NAME} DEPENDS ${OBJECT_FILE})

                    # --- Add to a list for linking later ---
                    set(KERNEL_TARGETS ${KERNEL_TARGETS} build_kernel_object_${KERNEL_NAME} PARENT_SCOPE)
                    set(KERNEL_OBJECT_FILES ${KERNEL_OBJECT_FILES} ${OBJECT_FILE} PARENT_SCOPE)
                endfunction()

                """
            )

        with open(cmake_path, "w") as f:
            f.write(contents)

    def save_src_to_cmake(self, cmake_path: str, src_path: str) -> None:
        # Remove the directory part of file_path
        src_path = "${CMAKE_CURRENT_SOURCE_DIR}/" + Path(src_path).name
        with open(cmake_path, "a") as f:
            f.write(f"target_sources({self._target_name} PRIVATE {src_path})\n")

    def save_kernel_asm_to_cmake(self, cmake_path: str, asm_files: list[str]) -> None:
        # TODO: make this work beyond CUDA
        with open(cmake_path, "a") as f:
            for asm_file in asm_files:
                kernel_name = Path(asm_file).name.split(".")[0]
                asm_file = f"${{CMAKE_CURRENT_SOURCE_DIR}}/{Path(asm_file).name}"
                contents = textwrap.dedent(
                    f"""
                    embed_gpu_kernel({kernel_name} {asm_file})
                    """
                )
                f.write(contents)
            if asm_files:
                f.write(f"add_dependencies({self._target_name} ${{KERNEL_TARGETS}})\n")
                f.write(
                    f"target_link_libraries({self._target_name} PRIVATE ${{KERNEL_OBJECT_FILES}})\n"
                )

    def save_link_cmd_to_cmake(self, cmake_path: str) -> None:
        lflags = " ".join(self._build_option.get_ldflags())
        libs = " ".join(self._build_option.get_libraries())
        contents = textwrap.dedent(
            f"""
            # Add linker flags
            target_link_options({self._target_name} PRIVATE {lflags})

            # Add libraries
            target_link_libraries({self._target_name} PRIVATE {libs})
         """
        )

        if not os.path.exists(cmake_path):
            raise AssertionError(
                f"save_link_cmd_to_cmakefile expects {cmake_path} to already exist"
            )
        with open(cmake_path, "a") as f:
            f.write(contents)


def run_asm_build_object(src: str, target: str, cwd: str) -> None:
    def get_asm_compiler() -> str:
        ASM_CC = get_cpp_compiler()
        # Intel compiler is not support to compile asm, switch to gcc.
        if _is_intel_compiler(ASM_CC):
            ASM_CC = "gcc"
        return ASM_CC

    def get_command_line(asm_cc: str, src: str, target: str) -> str:
        cmd = f"{asm_cc} -c {src} -o {target}"

        return cmd

    asm_cc = get_asm_compiler()
    cmd = get_command_line(
        asm_cc=asm_cc,
        src=normalize_path_separator(src),
        target=normalize_path_separator(target),
    )
    run_compile_cmd(cmd, cwd=normalize_path_separator(cwd))
