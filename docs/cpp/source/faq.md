---
myst:
  html_meta:
    description: Frequently asked questions about the PyTorch C++ API and libtorch.
    keywords: PyTorch, C++, FAQ, libtorch, troubleshooting
---

# FAQ

Listed below are a number of common issues users face with the various parts of
the C++ API.

## C++ Extensions

### Undefined symbol errors from PyTorch/ATen

**Problem**: You import your extension and get an `ImportError` stating that
some C++ symbol from PyTorch or ATen is undefined. For example:

```cpp
>>> import extension
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: /home/user/.pyenv/versions/3.7.1/lib/python3.7/site-packages/extension.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZN2at19UndefinedTensorImpl10_singletonE
```

**Fix**: The fix is to `import torch` before you import your extension. This will make
the symbols from the PyTorch dynamic (shared) library that your extension
depends on available, allowing them to be resolved once you import your extension.

### I created a tensor using a function from `at::` and get errors

**Problem**: You created a tensor using e.g. `at::ones` or `at::randn` or
any other tensor factory from the `at::` namespace and are getting errors.

**Fix**: Replace `at::` with `torch::` for factory function calls. You
should never use factory functions from the `at::` namespace, as they will
create tensors. The corresponding `torch::` functions will create variables,
and you should only ever deal with variables in your code.

## Build and Compilation

### CMake cannot find Torch

**Problem**: When building your project with CMake, you get an error that
`Torch` package cannot be found.

**Fix**: You need to specify the path to the LibTorch installation using
`CMAKE_PREFIX_PATH`:

```cpp
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
```

Alternatively, set `Torch_DIR` to point to the directory containing
`TorchConfig.cmake`:

```cpp
cmake -DTorch_DIR=/path/to/libtorch/share/cmake/Torch ..
```

### Linker errors with undefined references

**Problem**: Your project compiles but you get linker errors with undefined
references to PyTorch symbols.

**Fix**: Ensure you're linking against all required libraries in your
`CMakeLists.txt`:

```cpp
find_package(Torch REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app "${TORCH_LIBRARIES}")
```

Also ensure that the compiler flags are set correctly:

```cpp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
```
