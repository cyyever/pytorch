#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <utility>

#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/Context.h>

#include <c10/util/Exception.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/linalg/BatchLinearAlgebraLib.h>

#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_solve_triangular.h>

namespace at::native {
#if defined(BUILD_LAZY_CUDA_LINALG)
// All registrations with PyTorch runtime should be done dynamically
// so if library is lazy loaded it must not export anything, otherwise
// it can result in symbol clashes
namespace lazy_linalg {
#endif

namespace {

void ldl_factor_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  ldl_factor_cusolver(LD, pivots, info, upper, hermitian);
}

void ldl_solve_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
  if (LD.is_complex()) {
    TORCH_CHECK(
        !hermitian,
        "torch.linalg.ldl_solve: complex tensors with hermitian=True flag are not supported on CUDA.");
  }

  ldl_solve_cusolver(LD, pivots, B, upper);
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)
REGISTER_CUDA_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {
// At the time of writing, the unconditional dispatch
// to the native cholesky_solve method in cuSOLVER is slow
// with batched inputs.
template <bool use_dedicated_kernel_unconditionally = false>
inline void _cholesky_solve_helper_cuda_cusolver_algo_selector(
  Tensor& self,
  const Tensor& A,
  bool upper) {
  if constexpr (use_dedicated_kernel_unconditionally) {
    _cholesky_solve_helper_cuda_cusolver(self, A, upper);
  } else {
    // TODO: cusolverDn<T>potrsBatched only supports nrhs == 1 and does not have good performance.
    // TODO: Non-batched potrs is too slow in the batched setting compared to two triangular solves.
    // Non-batched input -> non-batched potrs.
    // Batched input -> two triangular solves.
    if (batchCount(self) == 1) {
      _cholesky_solve_helper_cuda_cusolver(self, A, upper);
    } else {
      const auto L = upper
        ? c10::MaybeOwned<Tensor>::owned(A.mH())
        : c10::MaybeOwned<Tensor>::borrowed(A);
      // NOTE: we tolerate redispatch with at::triangular_solve_triangular
      // because it handles memory layout optimization and conj/neg flags.
      // IMPORTANT NOTE: `self` and `A` are not processed for kernel calls yet!
      // Step 1: Solve for Y: L Y = B or U^H Y = B.
      at::linalg_solve_triangular_out(self, *L, self, /*upper=*/false);
      // Step 2: Solve for X: L^H X = Y or U X = Y.
      at::linalg_solve_triangular_out(self, L->mH(), self, /*upper=*/true);
    }
  }
}

inline void _cholesky_solve_helper_cuda_cusolver_dispatcher(
    Tensor& self,
    const Tensor& A,
    bool upper) {
  // For now, unconditional dispatch to the dedicated cholesky solve
  // kernel in cuSOLVER is slow for batched inputs.
  // TODO: switch once resolved.
  _cholesky_solve_helper_cuda_cusolver_algo_selector<
    /*use_dedicated_kernel_unconditionally=*/false
  >(self, A, upper);
}

} // namespace (anonymous)

Tensor _cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
  at::Tensor self_working_copy = cloneBatchedColumnMajor(self);
  _cholesky_solve_helper_cuda_cusolver_dispatcher(self_working_copy, A, upper);
  return self_working_copy;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static void cholesky_kernel(const Tensor& input, const Tensor& info, bool upper) {
  cholesky_helper_cusolver(input, upper, info);
}

REGISTER_CUDA_DISPATCH(cholesky_stub, &cholesky_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor& cholesky_inverse_kernel_impl(Tensor &result, [[maybe_unused]] Tensor& infos, bool upper) {
  // This function calculates the inverse matrix in-place
  // result should be in column major order and contain matrices to invert
  // the content of result is overwritten
  at::Tensor A = cloneBatchedColumnMajor(result);
  result.fill_(0);
  result.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);
  _cholesky_solve_helper_cuda_cusolver_dispatcher(result, A, upper);
  return result;
}

REGISTER_CUDA_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifdef USE_LINALG_SOLVER
enum class SolverBackend : char {
  CUSOLVER,
  CUBLAS
};
#ifndef USE_ROCM
namespace {

  // Based on benchmarks across H100, A100, L40, RTX5090 with about 3800 points:
  // - with batch dims in the range 2^i, with i in 0-8;
  // - square matrices of dim 2^i and (2^{i+1} + 2^i)/2, with 2^k <= 8192;
  // - square matrices of dim 2^i-/+1;
  // Rule: use cuSOLVER when n*n > threshold, where threshold depends on
  // batch size and dtype.

  // batch <= 2:
  //   threshold = T * batch
  //   float32/complex64: T = 8400
  //   float64/complex128: T = 2200

  // batch > 2:
  //   float32/complex64: threshold = 18600 * batch
  //   float64:           threshold = 16600 * batch * isqrt(batch)
  //   complex128:        threshold = 5200 * batch * isqrt(batch)

  // cuBLAS's strided-batched kernel is nearly O(1) in batch for fixed N -
  // it processes all matrices in one launch. cuSOLVER is O(batch) since it
  // processes matrices independently. So cuBLAS wins at high batch / small N,
  // and the threshold grows with batch.

  // float64 needs super-linear scaling (batch^1.5) because cuBLAS's cost
  // grows more slowly with N for float64 than for other dtypes - its
  // advantage over cuSOLVER persists to larger N at high batch. Empirically
  // N^2/batch at the crossover roughly doubles each time batch doubles.

  // complex128 also uses batch^1.5 scaling empirically, though its crossover
  // ratios are less regular. The lower multiplier (5200 vs 16600) reflects
  // that cuSOLVER overtakes cuBLAS at smaller N for complex128.
  //
  // NOTE: additionally validated on Blackwell CUDA 13.2 with FP64 emulation
  // on/off for cuSOLVER (on by default for cuBLAS).
  // No severe mispredictions observed.
  inline SolverBackend get_lu_factor_solver_backend(int64_t batch, int64_t m, int64_t n, const ScalarType& dtype) {
    // cuBLAS does not support rectangular inputs.
    if (m != n) {
      return SolverBackend::CUSOLVER;
    }

    if (batch == 1) {
      // cuBLAS is optimized for batched inputs.
      return SolverBackend::CUSOLVER;
    } else {
      int64_t threshold = 0;
      if (batch == 2) {
        // batch <= 2:  n * n > T_small * batch
        // At batch=2, cuBLAS has minimal batching advantage - kernel launch overhead
        // dominates. cuSOLVER is competitive at much smaller N, so lower thresholds
        // suffice. Only two groups needed: float32/complex64 vs float64/complex128.
        switch (dtype) {
          case ScalarType::Float:
          case ScalarType::ComplexFloat:
            threshold = 8400 * batch;
            break;
          default:
            // i.e. Double, ComplexDouble
            threshold = 2200 * batch;
        }
      } else {
        // batch > 2:
        // At larger batch, cuBLAS's batching advantage kicks in. For float64/complex128
        // this advantage grows super-linearly (cuBLAS stays flat while cuSOLVER scales
        // linearly), captured by the batch * isqrt(batch) term.
        switch (dtype) {
          case ScalarType::Float:
          case ScalarType::ComplexFloat:
            threshold = 18600 * batch;
            break;
          case ScalarType::Double:
            threshold = 16600 * batch * static_cast<int64_t>(std::sqrt(batch));
            break;
          default:
            // i.e. ComplexDouble
            threshold = 5200 * batch * static_cast<int64_t>(std::sqrt(batch));
        }
      }

      return n * n > threshold ? SolverBackend::CUSOLVER : SolverBackend::CUBLAS;
    }
  }

}
#endif
#endif

static void lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  auto batch_size = batchCount(input);
  auto m = input.size(-2);
  auto n = input.size(-1);

  const auto lu_factor_cusolver = [batch_size, m, n](const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
#ifdef USE_ROCM
    // FIXME: this heuristic is likely incorrect for ROCM.
    if (m != n || (batch_size == 1 || m >= 512)) {
      lu_factor_looped_cusolver(input, pivots, infos, compute_pivots);
    } else {
      lu_factor_batched_cublas(input, pivots, infos, compute_pivots);
    }
#else
    const auto solver_backend = get_lu_factor_solver_backend(batch_size, m, n, input.scalar_type());
    if (solver_backend == SolverBackend::CUSOLVER) {
      lu_factor_looped_cusolver(input, pivots, infos, compute_pivots);
    } else {
      lu_factor_batched_cublas(input, pivots, infos, compute_pivots);
    }
#endif
  };

  lu_factor_cusolver(input, pivots, infos, compute_pivots);

  // We return the trivial permutation of pivots starting with 1 (FORTRAN indexing)
  if (!compute_pivots) {
    auto k = std::min(input.size(-2), input.size(-1));
    auto pivots_tmp = at::arange(1, k + 1, input.options().dtype(at::kInt));
    pivots.copy_(pivots_tmp);
  }
}

REGISTER_CUDA_DISPATCH(lu_factor_stub, &lu_factor)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangular_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  // For batches smaller than 8 and matrix sizes larger than 64x64 cuBLAS forloop is faster than batched version
  if (batchCount(A) <= 8 && A.size(-1) >= 64) {
    triangular_solve_cublas(A, B, left, upper, transpose, unitriangular);
  } else {
    triangular_solve_batched_cublas(A, B, left, upper, transpose, unitriangular);
  }
}

REGISTER_CUDA_DISPATCH(triangular_solve_stub, &triangular_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ orgqr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor& orgqr_kernel_impl(Tensor& result, const Tensor& tau) {
#ifdef USE_LINALG_SOLVER
  return orgqr_helper_cusolver(result, tau); // cusolver
#else
  TORCH_CHECK(false, "Calling torch.orgqr on a CUDA tensor requires compiling ",
    "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#endif
}

REGISTER_CUDA_DISPATCH(orgqr_stub, &orgqr_kernel_impl)

void ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
#ifdef USE_LINALG_SOLVER
  ormqr_cusolver(input, tau, other, left, transpose);
#else
  TORCH_CHECK(false,
      "Calling torch.ormqr on a CUDA tensor requires compiling ",
      "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#endif
}

REGISTER_CUDA_DISPATCH(ormqr_stub, &ormqr_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ qr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void geqrf_kernel(const Tensor& input, const Tensor& tau) {
  auto geqrf_cusolver_backend = [](const Tensor& input, const Tensor& tau) {
      // For the benchmarks see
      // https://github.com/pytorch/pytorch/pull/56253#discussion_r622851107
      // TODO: re-eval
      if (input.size(-2) <= 256 && batchCount(input) >= std::max<int64_t>(2, input.size(-2) / 16)) {
        geqrf_batched_cublas(input, tau);
        return;
      } else {
        geqrf_cusolver(input, tau);
        return;
      }
      geqrf_batched_cublas(input, tau);
      return;
  };
  return geqrf_cusolver_backend(input, tau);
}

REGISTER_CUDA_DISPATCH(geqrf_stub, &geqrf_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eigh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  linalg_eigh_cusolver(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
}

REGISTER_CUDA_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_cuda());
  // This function calculates the non-symmetric eigendecomposition in-place
  // tensors should be in batched column major memory format
  // the content of eigenvalues, eigenvectors and infos is overwritten by
  // 'linalg_eig_cusolver_xgeev', which modifies the provided input matrix in-place,
  // therefore we need a copy
  linalg_eig_cusolver_xgeev(eigenvalues, eigenvectors, input, infos, compute_eigenvectors);
}

REGISTER_CUDA_DISPATCH(linalg_eig_stub, &linalg_eig_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ svd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void svd_kernel(const Tensor& A,
                const bool full_matrices,
                const bool compute_uv,
                const std::optional<std::string_view>& driver,
                const Tensor& U,
                const Tensor& S,
                const Tensor& Vh,
                const Tensor& info) {
  // svd_cusolver computes V rather than Vh, so we pass a view of Vh.mT
  // and then conjugate Vh in-place
  svd_cusolver(A, full_matrices, compute_uv, driver, U, S, compute_uv ? Vh.mT() : Vh, info);
  if (compute_uv && Vh.is_complex()) {
    Vh._set_conj(!Vh.is_conj());
  }
}

REGISTER_CUDA_DISPATCH(svd_stub, &svd_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

c10::MaybeOwned<Tensor> maybe_expand_lu(const Tensor& B, const Tensor& LU) {
  // B and LU have the same number of dimensions
  if (batchCount(B) != batchCount(LU)) {
        auto n = B.dim();
    auto expand_shape = DimVector(B.sizes().slice(0, n - 2));
    expand_shape.append({LU.size(-2), LU.size(-1)});
    return c10::MaybeOwned<Tensor>::owned(
        cloneBatchedColumnMajor(LU.expand(expand_shape)));
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(LU);
  }
}

c10::MaybeOwned<Tensor> maybe_expand_pivots(const Tensor& B, const Tensor& pivots) {
  // B and pivots have the same number of dimensions
  if (batchCount(B) != batchCount(pivots.unsqueeze(-1))) {
    auto expand_shape = DimVector(B.sizes().slice(0, B.dim() - 2));
    expand_shape.push_back(pivots.size(-1));
    return c10::MaybeOwned<Tensor>::owned(pivots.expand(expand_shape).contiguous());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(pivots);
  }
}

static void lu_solve_kernel(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // Trivial case. Remove it once `torch.solve` is removed, as linalg.solve already shortcuts this case
  if (B.numel() == 0) {
    return;
  }

  auto b = batchCount(B);
  auto n = LU.size(-2);
  auto k = B.size(-1);
  // heuristics determined from tests discussed in https://github.com/pytorch/pytorch/pull/72935

  // Computes X = U^{-1}L^{-1}P^T B via triangular solves
  auto lu_solve_triangular = [n](const Tensor& LU, const Tensor& pivots, const Tensor& B, const TransposeType trans) {
    auto LU_ = maybe_expand_lu(B, LU);
    auto pivots_ = maybe_expand_pivots(B, pivots);
    // LAPACK / cublas / etc returns the permutation in an odd format
    // Here we transform it to a vector representing a permutation, i.e. a (batch of) vectors st. P(i) = j
    auto perm = at::arange(n, pivots_->options().dtype(kLong)).expand(pivots_->sizes()).contiguous();
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .declare_static_shape(pivots_->sizes(), /*squash_dim=*/pivots_->dim() - 1)
      .add_output(perm)
      .add_const_input(*pivots_)
      .build();
    unpack_pivots_stub(pivots_->device().type(), iter, n, n);

    if (trans == TransposeType::NoTranspose) {
      // Get the inverse permutation
      // This is an insertion sort, and it's equivalent to
      // perm = at::argsort(perm);
      // but more parallelisable and O(n), exploiting that perm is a permutation
      auto id_perm = at::arange(n, perm.options()).expand(perm.sizes());
      auto inv_perm = perm.scatter(-1, perm, id_perm);
      // B1 = P^T @ B  (must be done out-of-place as B is both source and target)
      auto B1 = B.scatter(-2, inv_perm.unsqueeze(-1).expand_as(B), B);
      // B = L^{-1} @ B1
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), *LU_, B1, /*upper=*/false, /*left=*/true, /*unitriangular=*/true);
      // B = U^{-1} @ B
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), *LU_, B, /*upper=*/true);
    } else {
      auto LU_H = LU_->mH();
      // B = U^{-H} @ B
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), LU_H, B, /*upper=*/false);
      // B = L^{-H} @ B
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), LU_H, B, /*upper=*/true, /*left=*/true, /*unitriangular=*/true);
      // B = P @ B
      B.scatter_(-2, perm.unsqueeze(-1).expand_as(B), B.clone());
    }
  };


#ifdef USE_LINALG_SOLVER
  auto lu_solve_batched_cublas_fn = [](const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
    auto LU_ = maybe_expand_lu(B, LU);
    auto pivots_ = maybe_expand_pivots(B, pivots);
    lu_solve_batched_cublas(*LU_, *pivots_, B, trans);
  };

  // Preferred Backend
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  if (preferred_backend == at::LinalgBackend::Cusolver) {
    // TODO: Re-eval this condition
    if (b <= 2 && n >= 64) {
      lu_solve_looped_cusolver(LU, pivots, B, trans);
    } else {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
    }
    return;
  }

  // TODO: Re-eval this heuristic
  // Heuristic
  //if (n == k) {
  // if (k <= 16) batched_cublas
  // else solve_triag
  //} else {
  //if (n <= 8) {
  // batched_cublas
  //} else if (n <= 32) {
  //  b <= 2 looped_cusolver
  //  k <= 8 batched_cusolver
  //  solve_triag
  //} else if (n <= 64) {
  //  b <= 2 && (k <= 64 || adjoint) looped_cusolver
  //  k <= 8 batched_cusolver
  //  solve_triag
  //} else if (n <= 128) {
  //  if (b <= 2 && k <= 2) looped_cusolver
  //  else if (k <= 2) batched_cusolver
  //  else solve_triag
  //} else { // n > 128
  //  solve_triag
  //}
  //}

  // Particular case when multiplying A^{-1}B where B is square
  // In this case doing two triangular solves is almost always fastest
  if (n == k) {
    if (n <= 16) {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
      return;
    }
    lu_solve_triangular(LU, pivots, B, trans);
    return;
  }

  if (n <= 8) {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
  } else if (n <= 64) {
    if (b <= 2 && (k <= 64 || trans != TransposeType::NoTranspose || n <= 32)) {
      lu_solve_looped_cusolver(LU, pivots, B, trans);
    } else if (k <= 8) {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
    } else {
      lu_solve_triangular(LU, pivots, B, trans);
    }
  } else if (n <= 128) {
    if (b <= 2 && k <= 2)  {
      lu_solve_looped_cusolver(LU, pivots, B, trans);
    } else if (k <= 2)  {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
    } else {
      lu_solve_triangular(LU, pivots, B, trans);
    }
  } else { // n > 128
    lu_solve_triangular(LU, pivots, B, trans);
  }
#else
  // No cublas or cusolver
  // lu_solve_triangular is almost always best
  lu_solve_triangular(LU, pivots, B, trans);
#endif // ifdef USE_LINALG_SOLVER
}

REGISTER_CUDA_DISPATCH(lu_solve_stub, &lu_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lstsq ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void linalg_lstsq_gels(const Tensor& A, const Tensor& B, const Tensor& /*infos*/) {
  // The steps for using the QR decomposition for solving least squares problems
  // are outlined here https://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto mn = std::min(m, n);

  // explicitly broadcast the batch dimensions of A
  // TODO: revisit this later to use batch_iterator_with_broadcasting in triangular_solve
  IntArrayRef A_batch_sizes(A.sizes().data(), A.dim() - 2);
  IntArrayRef B_batch_sizes(B.sizes().data(), B.dim() - 2);
  std::vector<int64_t> expand_batch_portion = at::infer_size(A_batch_sizes, B_batch_sizes);

  auto tau_shape = A.sizes().vec();
  tau_shape.pop_back();
  tau_shape.back() = mn;
  Tensor tau = at::empty(tau_shape, A.options());

  if (m >= n) {
    // Step 1: compute QR factorization using geqrf
    geqrf_kernel(A, tau);

    // explicitly broadcast the batch dimensions of A
    // we do it after geqrf so that we don't do redundant computations for the same input
    auto A_expand_batch = expand_batch_portion;
    A_expand_batch.insert(A_expand_batch.end(), {A.size(-2), A.size(-1)});
    Tensor A_expanded = A.expand({A_expand_batch});
    bool is_fortran_contiguous = A_expanded.mT().is_contiguous();
    Tensor A_broadcasted = is_fortran_contiguous ? A_expanded : cloneBatchedColumnMajor(A_expanded);
    auto tau_expand_batch = expand_batch_portion;
    tau_expand_batch.push_back(tau.size(-1));
    Tensor tau_broadcasted = tau.expand({tau_expand_batch}).contiguous();

    // Step 2: B <- Q^H B
    ormqr_kernel(A_broadcasted, tau_broadcasted, B, /*left=*/true, /*transpose=*/true);

    // Step 3: solve R X = B
    triangular_solve_kernel(
        A_broadcasted,
        B,
        /*left=*/true,
        /*upper=*/true,
        /*transpose=*/TransposeType::NoTranspose,
        /*unitriangular=*/false);
  } else { // underdetermined case
    Tensor Ah = cloneBatchedColumnMajor(A.mH());

    // Step 1: compute QR factorization of conjugate transpose of A using geqrf
    geqrf_kernel(Ah, tau);

    // explicitly broadcast the batch dimensions of A
    // we do it after geqrf so that we don't do redundant computations for the same input
    auto A_expand_batch = expand_batch_portion;
    A_expand_batch.insert(A_expand_batch.end(), {Ah.size(-2), Ah.size(-1)});
    Tensor Ah_expanded = Ah.expand({A_expand_batch});
    bool is_fortran_contiguous = Ah_expanded.mT().is_contiguous();
    Tensor Ah_broadcasted = is_fortran_contiguous ? Ah_expanded : cloneBatchedColumnMajor(Ah_expanded);

    // Step 2: R^H Z = B
    const auto trans = Ah_broadcasted.is_complex() ? TransposeType::ConjTranspose
                                                   : TransposeType::Transpose;
    triangular_solve_kernel(
        Ah_broadcasted,
        B,
        /*left=*/true,
        /*upper=*/true,
        /*transpose=*/trans,
        /*unitriangular=*/false);

    // B matrix has the size max(m, n) x nrhs
    // triangular_solve_kernel writes its output into the first m rows of B leaving the rest untouched
    // we need to set the rest of the rows to zero so that the multiplication from step 3 is correct
    B.narrow(-2, m, n - m).zero_();

    auto tau_expand_batch = std::move(expand_batch_portion);
    tau_expand_batch.push_back(tau.size(-1));
    Tensor tau_broadcasted = tau.expand({tau_expand_batch}).contiguous();

    // Step 3: X <- Q Z
    ormqr_kernel(Ah_broadcasted, tau_broadcasted, B, /*left=*/true, /*transpose=*/false);
  }
}

void lstsq_kernel(const Tensor& a, Tensor& b, Tensor& /*rank*/, Tensor& /*singular_values*/, Tensor& infos, double /*rcond*/, std::string /*driver_name*/)  {
  auto m = a.size(-2);
  auto n = a.size(-1);

  // first handle the underdetermined case (m < n)
  // this case is not supported by cuBLAS
  if (m < n) {
    linalg_lstsq_gels(a, b, infos);
  } else { // m >= n
    // On CUDA platform we use either cuBLAS or cuSOLVER here
    // the batched vs looped dispatch is implemented based on the following performance results
    // https://github.com/pytorch/pytorch/pull/54725#issuecomment-832234456
    if (m <= 256 && batchCount(b) >= std::max<int64_t>(2, m / 16)) {
      gels_batched_cublas(a, b, infos);
    } else {
      linalg_lstsq_gels(a, b, infos);
    }
  }
}

REGISTER_CUDA_DISPATCH(lstsq_stub, &lstsq_kernel)


#if defined(BUILD_LAZY_CUDA_LINALG)
struct DispatchInitializer {
  DispatchInitializer() {
    cuda::detail::LinalgDispatch disp{_cholesky_solve_helper_cuda};
    cuda::detail::registerLinalgDispatch(disp);
  };
} initializer;

}  // namespace lazy_linalg
#endif
}  // namespace at::native
