---
myst:
  html_meta:
    description: PyTorch C++ API documentation — ATen tensors, Autograd, the torch namespace, and C++ Extensions.
    keywords: PyTorch, C++, API, libtorch, ATen, Autograd, C++ Extensions
---

# PyTorch C++ API

These pages provide the documentation for the public portions of the PyTorch C++
API.  This API can roughly be divided into four parts:

- **ATen**: The foundational tensor and mathematical operation library on which all else is built.
- **Autograd**: Augments ATen with automatic differentiation.
- **The torch:: namespace**: The device, fft, special and nested interfaces layered on top of them.
- **C++ Extensions**: A means of extending the Python API with custom C++ and CUDA routines.

Combined, these building blocks form a research and production ready C++
library for tensor computation with strong emphasis on GPU acceleration.

```{warning}

At the moment, the C++ API should be considered "beta" stability; we may
make major breaking changes to the backend in order to improve the API,
or in service of providing the Python interface to PyTorch, which is our
most stable and best supported interface.
```

## ATen

ATen is fundamentally a tensor library, on top of which almost all other Python
and C++ interfaces in PyTorch are built. It provides a core `Tensor` class,
on which many hundreds of operations are defined. Most of these operations have
both CPU and GPU implementations, to which the `Tensor` class will
dynamically dispatch based on its type. A small example of using ATen could
look as follows:

```cpp
#include <ATen/ATen.h>

at::Tensor a = at::ones({2, 2}, at::kInt);
at::Tensor b = at::randn({2, 2});
auto c = a + b.to(at::kInt);
```

This `Tensor` class and all other symbols in ATen are found in the `at::`
namespace, documented
[here](https://pytorch.org/cppdocs/api/namespace_at.html#namespace-at).

## Autograd

What we term *autograd* are the portions of PyTorch's C++ API that augment the
ATen `Tensor` class with capabilities concerning automatic differentiation.
The autograd system records operations on tensors to form an *autograd graph*.
Calling `backwards()` on a leaf variable in this graph performs reverse mode
differentiation through the network of functions and tensors spanning the
autograd graph, ultimately yielding gradients. The following example provides
a taste of this interface:

```cpp
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
torch::Tensor b = torch::randn({2, 2});
auto c = a + b;
c.backward(); // a.grad() will now hold the gradient of c w.r.t. a.
```

The `at::Tensor` class in ATen is not differentiable by default. To add the
differentiability of tensors the autograd API provides, you must use tensor
factory functions from the `torch::` namespace instead of the `at::` namespace.
For example, while a tensor created with `at::ones` will not be differentiable,
a tensor created with `torch::ones` will be.

## The torch:: Namespace

Above ATen and autograd, the `torch::` namespace collects what the C++ API
offers directly:

- `torch::autograd`, the differentiation interface described above, including
  custom functions;
- `torch::fft`, `torch::special` and `torch::nested`, mirroring their Python
  counterparts;
- `torch::cuda`, `torch::mps` and `torch::xpu`, for querying and synchronising
  devices;
- `torch::python`, the support code for binding C++ extensions into Python
  with pybind11.

Include `<torch/torch.h>` to get all of it.

## C++ Extensions

*C++ Extensions* offer a simple yet powerful way of accessing all of the above
interfaces for the purpose of extending regular Python use-cases of PyTorch.
C++ extensions are most commonly used to implement custom operators in C++ or
CUDA to accelerate research in vanilla PyTorch setups. The C++ extension API
does not add any new functionality to the PyTorch C++ API. Instead, it
provides integration with Python setuptools as well as JIT compilation
mechanisms that allow access to ATen, the autograd and other C++ APIs from
Python. To learn more about the C++ extension API, go through
[this tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html).

## Contents

```{toctree}
:maxdepth: 2

installing
api/index
faq
```

# Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

## Acknowledgements

This documentation website for the PyTorch C++ universe uses the Sphinx
C++ domain for API documentation.
