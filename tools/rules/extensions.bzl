"""Module extension for PyTorch's local third-party dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _new_local_repo_impl(repository_ctx):
    """Symlinks a local directory and uses a provided BUILD file."""
    workspace_root = repository_ctx.path(Label("@pytorch//:MODULE.bazel")).dirname
    src = workspace_root.get_child(repository_ctx.attr.path)
    for child in src.readdir():
        base = child.basename
        if base in ("BUILD", "BUILD.bazel", "MODULE.bazel", "WORKSPACE", "WORKSPACE.bazel", "WORKSPACE.bzlmod"):
            continue
        repository_ctx.symlink(child, base)
    build_file = repository_ctx.attr.build_file
    if build_file:
        repository_ctx.symlink(repository_ctx.path(build_file), "BUILD.bazel")
    else:
        repository_ctx.file("BUILD.bazel", repository_ctx.attr.build_file_content)

_new_local_repo = repository_rule(
    implementation = _new_local_repo_impl,
    attrs = {
        "path": attr.string(doc = "Path relative to workspace root"),
        "build_file": attr.label(allow_single_file = True),
        "build_file_content": attr.string(),
    },
)

def _local_repo_impl(repository_ctx):
    """Symlinks a local directory including its own BUILD files."""
    workspace_root = repository_ctx.path(Label("@pytorch//:MODULE.bazel")).dirname
    src = workspace_root.get_child(repository_ctx.attr.path)
    for child in src.readdir():
        repository_ctx.symlink(child, child.basename)

_local_repo = repository_rule(
    implementation = _local_repo_impl,
    attrs = {
        "path": attr.string(doc = "Path relative to workspace root"),
    },
)

def _find_cudnn_impl(repository_ctx):
    cudnn_path = repository_ctx.os.environ.get("CUDNN_PATH", "")

    if cudnn_path:
        repository_ctx.symlink(cudnn_path, "cudnn")
    else:
        python = repository_ctx.which("python3") or repository_ctx.which("python")
        if not python:
            repository_ctx.file("BUILD.bazel", 'cc_library(name = "cudnn", visibility = ["//visibility:public"])')
            return
        result = repository_ctx.execute([
            python, "-c",
            "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])",
        ])
        if result.return_code != 0:
            repository_ctx.file("BUILD.bazel", 'cc_library(name = "cudnn", visibility = ["//visibility:public"])')
            return
        repository_ctx.symlink(result.stdout.strip(), "cudnn")

    lib_dir = "lib"
    if repository_ctx.path("cudnn/lib64").exists:
        lib_dir = "lib64"
    repository_ctx.file("BUILD.bazel", """
cc_library(
    name = "cudnn_headers",
    hdrs = glob(["cudnn/include/cudnn*.h"]),
    includes = ["cudnn/include"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cudnn_lib",
    shared_library = "cudnn/{lib_dir}/libcudnn.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "cudnn",
    visibility = ["//visibility:public"],
    deps = [
        "cudnn_headers",
        "cudnn_lib",
    ],
)
""".format(lib_dir = lib_dir))

_find_cudnn = repository_rule(
    implementation = _find_cudnn_impl,
    environ = ["CUDNN_PATH", "PATH"],
)

_CUDA_STUB_NAMES = [
    "cuda_headers", "cuda_driver", "cuda", "cufft",
    "cublas", "curand", "cusolver", "cusparse", "cufile", "nvrtc",
]

def _find_cuda_impl(repository_ctx):
    cuda_path = repository_ctx.os.environ.get("CUDA_PATH", "")
    if not cuda_path:
        cuda_path = "/usr/local/cuda"
    if not repository_ctx.path(cuda_path).exists:
        lines = []
        for name in _CUDA_STUB_NAMES:
            lines.append('cc_library(name = "{name}", visibility = ["//visibility:public"])'.format(name = name))
        repository_ctx.file("BUILD.bazel", "\n".join(lines))
        return
    repository_ctx.symlink(cuda_path, "cuda")

    if repository_ctx.path("cuda/targets/x86_64-linux/lib").exists:
        lib_dir = "targets/x86_64-linux/lib"
        inc_dirs = ["include", "targets/x86_64-linux/include", "targets/x86_64-linux/include/nvtx3"]
    else:
        lib_dir = "lib64"
        inc_dirs = ["include", "include/nvtx3"]

    inc_globs = "\n".join(['        "cuda/{dir}/**",'.format(dir = d) for d in inc_dirs])
    inc_includes = "\n".join(['        "cuda/{dir}",'.format(dir = d) for d in inc_dirs])

    repository_ctx.file("BUILD.bazel", """
cc_library(
    name = "cuda_headers",
    hdrs = glob([
{inc_globs}
    ]),
    includes = [
{inc_includes}
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_driver",
    srcs = ["cuda/{lib_dir}/stubs/libcuda.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    srcs = ["cuda/{lib_dir}/libcudart.so"],
    visibility = ["//visibility:public"],
    deps = [":cuda_headers"],
)

cc_library(
    name = "cufft",
    srcs = ["cuda/{lib_dir}/libcufft.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas",
    srcs = [
        "cuda/{lib_dir}/libcublasLt.so",
        "cuda/{lib_dir}/libcublas.so",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "curand",
    srcs = ["cuda/{lib_dir}/libcurand.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cusolver",
    srcs = ["cuda/{lib_dir}/libcusolver.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cusparse",
    srcs = ["cuda/{lib_dir}/libcusparse.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cufile",
    srcs = ["cuda/{lib_dir}/libcufile.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvrtc",
    srcs = glob(["cuda/{lib_dir}/libnvrtc*.so"]),
    visibility = ["//visibility:public"],
)
""".format(lib_dir = lib_dir, inc_globs = inc_globs, inc_includes = inc_includes))

_find_cuda = repository_rule(
    implementation = _find_cuda_impl,
    environ = ["CUDA_PATH"],
)

def _pytorch_local_repos_impl(module_ctx):
    # new_local_repository equivalents (custom BUILD files)
    _new_local_repo(name = "gloo", path = "third_party/gloo", build_file = Label("@pytorch//third_party:gloo.BUILD"))
    _new_local_repo(name = "onnx", path = "third_party/onnx", build_file = Label("@pytorch//third_party:onnx.BUILD"))
    _new_local_repo(name = "cutlass", path = "third_party/cutlass", build_file = Label("@pytorch//third_party:cutlass.BUILD"))
    _local_repo(name = "fbgemm", path = "third_party/fbgemm")
    _new_local_repo(name = "ideep", path = "third_party/ideep", build_file = Label("@pytorch//third_party:ideep.BUILD"))
    _new_local_repo(name = "mkl_dnn", path = "third_party/ideep/mkl-dnn", build_file = Label("@pytorch//third_party:mkl-dnn.BUILD"))
    _new_local_repo(name = "asmjit", path = "third_party/fbgemm/external/asmjit", build_file = Label("@pytorch//third_party:fbgemm/external/asmjit.BUILD"))
    _new_local_repo(name = "sleef", path = "third_party/sleef", build_file = Label("@pytorch//third_party:sleef.BUILD"))
    _new_local_repo(name = "fmt", path = "third_party/fmt", build_file = Label("@pytorch//third_party:fmt.BUILD"))
    _new_local_repo(name = "kineto", path = "third_party/kineto", build_file = Label("@pytorch//third_party:kineto.BUILD"))
    _new_local_repo(name = "cpp-httplib", path = "third_party/cpp-httplib", build_file = Label("@pytorch//third_party:cpp-httplib.BUILD"))
    _new_local_repo(name = "moodycamel", path = "third_party/concurrentqueue", build_file = Label("@pytorch//third_party:moodycamel.BUILD"))
    _new_local_repo(name = "tensorpipe", path = "third_party/tensorpipe", build_file = Label("@pytorch//third_party:tensorpipe.BUILD"))
    _new_local_repo(name = "cudnn_frontend", path = "third_party/cudnn_frontend", build_file = Label("@pytorch//third_party:cudnn_frontend.BUILD"))

    # local_repository equivalents (use their own BUILD files)
    _local_repo(name = "pthreadpool", path = "third_party/pthreadpool")
    _local_repo(name = "FXdiv", path = "third_party/FXdiv")
    _local_repo(name = "XNNPACK", path = "third_party/XNNPACK")
    _local_repo(name = "gemmlowp", path = "third_party/gemmlowp/gemmlowp")
    _local_repo(name = "kleidiai", path = "third_party/kleidiai")

    # CUDA/cuDNN detection
    _find_cuda(name = "cuda")
    _find_cudnn(name = "cudnn")

    # MKL (http_archive equivalents)
    http_archive(
        name = "mkl",
        build_file = Label("@pytorch//third_party:mkl.BUILD"),
        sha256 = "59154b30dd74561e90d547f9a3af26c75b6f4546210888f09c9d4db8f4bf9d4c",
        strip_prefix = "lib",
        urls = [
            "https://anaconda.org/anaconda/mkl/2020.0/download/linux-64/mkl-2020.0-166.tar.bz2",
        ],
    )

    http_archive(
        name = "mkl_headers",
        build_file = Label("@pytorch//third_party:mkl_headers.BUILD"),
        sha256 = "2af3494a4bebe5ddccfdc43bacc80fcd78d14c1954b81d2c8e3d73b55527af90",
        urls = [
            "https://anaconda.org/anaconda/mkl-include/2020.0/download/linux-64/mkl-include-2020.0-166.tar.bz2",
        ],
    )

    # Unused repos (hide submodule BUILD files from //... builds)
    _local_repo(name = "unused_tensorpipe_googletest", path = "third_party/tensorpipe/third_party/googletest")
    _local_repo(name = "unused_fbgemm", path = "third_party/fbgemm")
    _local_repo(name = "unused_ftm_bazel", path = "third_party/fmt/support/bazel")
    _local_repo(name = "unused_kineto_fmt_bazel", path = "third_party/kineto/libkineto/third_party/fmt/support/bazel")
    _local_repo(name = "unused_kineto_dynolog_googletest", path = "third_party/kineto/libkineto/third_party/dynolog/third_party/googletest")
    _local_repo(name = "unused_kineto_dynolog_gflags", path = "third_party/kineto/libkineto/third_party/dynolog/third_party/gflags")
    _local_repo(name = "unused_kineto_dynolog_glog", path = "third_party/kineto/libkineto/third_party/dynolog/third_party/glog")
    _local_repo(name = "unused_kineto_googletest", path = "third_party/kineto/libkineto/third_party/googletest")
    _local_repo(name = "unused_onnx_benchmark", path = "third_party/onnx/third_party/benchmark")

pytorch_local_repos = module_extension(
    implementation = _pytorch_local_repos_impl,
)
