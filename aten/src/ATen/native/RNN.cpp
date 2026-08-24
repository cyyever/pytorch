#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/RNN.h>

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/TensorOperators.h>
#include <ATen/mps/MPSDevice.h>
#include <c10/core/GradMode.h>
#include <c10/macros/Macros.h>

#include <array>
#include <atomic>
#include <c10/util/irange.h>
#include <torch/library.h>
#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/Utils.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_lstm_mps.h>
#include <ATen/ops/_thnn_differentiable_gru_cell_backward_native.h>
#include <ATen/ops/_thnn_differentiable_lstm_cell_backward_native.h>
#include <ATen/ops/_thnn_fused_gru_cell.h>
#include <ATen/ops/_thnn_fused_lstm_cell.h>
#include <ATen/ops/_thnn_fused_lstm_cell_backward.h>
#include <ATen/ops/_thnn_fused_lstm_cell_backward_impl.h>
#include <ATen/ops/_thnn_fused_lstm_cell_backward_native.h>
#include <ATen/ops/_use_cudnn_rnn_flatten_weight_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cudnn_is_acceptable.h>
#include <ATen/ops/dropout.h>
#include <ATen/ops/gru_cell_native.h>
#include <ATen/ops/gru_native.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/lstm_cell_native.h>
#include <ATen/ops/lstm_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/rnn_relu_cell_native.h>
#include <ATen/ops/rnn_relu_native.h>
#include <ATen/ops/rnn_tanh_cell_native.h>
#include <ATen/ops/rnn_tanh_native.h>
#include <ATen/ops/sigmoid_backward.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/tanh_backward.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros_like_ops.h>
#include <utility>
#endif

namespace at::native {

namespace {

// MIOpen RNN kernels are JIT-compiled at first use. A broken runtime compile
// environment (e.g. device-libs bitcode the bundled comgr cannot read,
// pytorch/pytorch#189618) surfaces as miopenStatusUnknownError from the first
// forward on any architecture, so probe per device instead of hard-coding
// arch lists: the first MIOpen RNN failure on a device marks it broken and
// RNNs fall back to the native implementation for the rest of the process. A
// failure after a prior success on the same device is rethrown so real
// errors (e.g. OOM) are not masked. NB: MIOpen wrappers throw
// at::native::miopen_exception (a std::runtime_error), not c10::Error, so
// the dispatch sites below catch std::exception.
enum miopen_rnn_probe_result : int {
  miopen_rnn_untried = 0,
  miopen_rnn_ok = 1,
  miopen_rnn_broken = 2,
};

std::atomic<int>& miopen_rnn_probe(at::DeviceIndex device_index) {
  static std::array<std::atomic<int>, 64> state{};
  static std::atomic<int> overflow{miopen_rnn_untried};
  auto idx = static_cast<size_t>(device_index);
  return idx < state.size() ? state[idx] : overflow;
}

// Returns true if the failure was absorbed (fall through to native); false
// means the caller must rethrow.
bool miopen_rnn_handle_failure(at::DeviceIndex device_index, const char* what) {
  auto& probe = miopen_rnn_probe(device_index);
  if (probe.load(std::memory_order_relaxed) == miopen_rnn_ok) {
    return false;
  }
  probe.store(miopen_rnn_broken, std::memory_order_relaxed);
  TORCH_WARN(
      "MIOpen RNN failed on device ", static_cast<int>(device_index),
      "; falling back to the native RNN implementation on this device for the "
      "rest of the process. This usually indicates a broken MIOpen runtime "
      "kernel compilation environment "
      "(https://github.com/pytorch/pytorch/issues/189618). Error: ", what);
  return true;
}

// Check if pytorch is compiled with MIOpen.
bool use_miopen(const at::Tensor& input, const double dropout_state) {
    bool is_miopen_acceptable = ((input.scalar_type() == at::kFloat)|| (input.scalar_type() == at::kHalf)) &&
                                (detail::getCUDAHooks().compiledWithMIOpen()) &&
                                (input.is_cuda()) &&
                                (at::globalContext().userEnabledCuDNN());
    // MIOpen functions returns miopenStatusBadParm on empty
    // tensors. Maybe some functions actually support empty tensors, but
    // native kernels shouldn't be much slower because the output is also
    // likely empty.
    if (input.sym_numel() == 0) return false;

    // Devices where MIOpen RNN already failed fall back to the native
    // implementation; see miopen_rnn_probe above.
    if (is_miopen_acceptable &&
        miopen_rnn_probe(input.device().index()).load(std::memory_order_relaxed) ==
            miopen_rnn_broken) {
        return false;
    }

    return is_miopen_acceptable;
}

bool use_mkldnn(const Tensor& input, TensorList params, TensorList hx) {
#if AT_MKLDNN_ENABLED()
  if (!at::globalContext().userEnabledMkldnn()) {
    return false;
  }
  // XPU: oneDNN LSTM for inference (GPU primitive supports f32 and f16)
  if (input.is_xpu()) {
    return !at::GradMode::is_enabled() &&
        (input.scalar_type() == kFloat || input.scalar_type() == kHalf) &&
        input.numel() != 0;
  }
  auto is_cpu_backend = [&](const TensorList tensors) {
    bool backend_cpu = true;
    for (const auto& t : tensors) {
      if (!(t.options().backend() == at::Backend::CPU)) {
        backend_cpu = false;
        break;
      }
    }
    return backend_cpu;
  };
  return input.options().backend() == at::Backend::CPU &&
      is_cpu_backend(params) && is_cpu_backend(hx) &&
      (input.scalar_type() == kFloat ||
       (input.scalar_type() == kBFloat16 && mkldnn_bf16_device_check()) ||
       (input.scalar_type() == kHalf && !at::GradMode::is_enabled() &&
        mkldnn_fp16_device_check())) &&
      input.numel() != 0;
#else
  (void)params;
  (void)hx;
  return false;
#endif
}

bool use_cudnn(const Tensor& t) {
  bool acceptable = at::cudnn_is_acceptable(t);
  auto st = t.scalar_type();
  bool bfloat16_cond = st == kBFloat16 && at::detail::getCUDAHooks().supportsBFloat16RNNWithCuDNN();
  return acceptable && (bfloat16_cond || st == kDouble || st == kFloat || st == kHalf);
}

template<typename T>
using pair_of = std::pair<T, T>;

template<typename T>
using tpair_of = std::tuple<T, T>;

// Those could have been function pointers, but MSVC chokes on function pointers as template parameters
struct tanh_f {
  Tensor operator()(const Tensor& t) const { return at::tanh(t); }
};

struct relu_f {
  Tensor operator()(const Tensor& t) const { return at::relu(t); }
};

struct PackedSequence {
  PackedSequence() = default;
  PackedSequence(Tensor _data, Tensor _batch_sizes)
    : data(std::move(_data)), batch_sizes(std::move(_batch_sizes)) {}

  Tensor data;
  Tensor batch_sizes;
};

// Simple type for __getstate__/__setstate__ serialization
//
// Element 0 is a string key to say what kind of CellParam this is. It
// should be a valid key into cell_params_deserializers
// Element 1 is the Tensors contained within the CellParams instance
// Element 2 is the doubles (if any) contained in the CellParams instance
// Element 3 is the longs (if any) contained within the CellParams instance
// Base class so we can polymorphically handle these
struct CellParamsBase {
  CellParamsBase() = default;
  CellParamsBase(const CellParamsBase&) = default;
  CellParamsBase(CellParamsBase&&) = default;
  CellParamsBase& operator=(const CellParamsBase&) = default;
  CellParamsBase& operator=(CellParamsBase&&) = default;
  virtual ~CellParamsBase() = default;
  virtual Tensor matmul_ih(const Tensor& input) const = 0;
  virtual Tensor matmul_hh(const Tensor& h) const = 0;
  // by default doing nothing. CellParams will override this
  // to define correct behavior for LSTMs with projections.
  // This function is not pure virtual, because it's useful to
  // provide this default implementation, so that all cell params
  // that don't support projections work correctly
  virtual Tensor matmul_hr(const Tensor& h) const {
    return h;
  }
  virtual Tensor linear_ih(const Tensor& input_ih) const = 0;
  virtual Tensor linear_hh(const Tensor& input_hh) const = 0;

  virtual const Tensor& b_ih() const = 0;
  virtual const Tensor& b_hh() const = 0;

};

// Pretty much all cells we support take the same set of arguments, but threading those
// 4 arguments manually is really annoying. Their lifetime is externally managed, so we only
// pass this struct of references around. LSTMs with projections have 5th argument w_hr, for all
// other models it's always going to be undefined.
struct CellParams : public CellParamsBase {
  CellParams(
      const Tensor& _w_ih,
      const Tensor& _w_hh,
      const Tensor& _b_ih,
      const Tensor& _b_hh,
      const Tensor& _w_hr)
      : w_ih(_w_ih), w_hh(_w_hh), b_ih_(_b_ih), b_hh_(_b_hh), w_hr(_w_hr) {}

  const Tensor& w_ih;
  const Tensor& w_hh;
  const Tensor& b_ih_; /* optional */
  const Tensor& b_hh_; /* optional */
  const Tensor& w_hr;  /* only defined for LSTMs with projections */

  Tensor matmul_ih(const Tensor& input) const override {
    return at::matmul(input, w_ih.t());
  }
  Tensor matmul_hh(const Tensor& h) const override {
    return at::matmul(h, w_hh.t());
  }
  Tensor matmul_hr(const Tensor& h) const override {
    if (w_hr.defined()) {
      return at::matmul(h, w_hr.t());
    }
    return h;
  }
  Tensor linear_ih(const Tensor& input) const override {
    return at::linear(input, w_ih, b_ih_);
  }
  Tensor linear_hh(const Tensor& h) const override {
    return at::linear(h, w_hh, b_hh_);
  }
  const Tensor& b_ih() const override {
    return b_ih_;
  }
  const Tensor& b_hh() const override {
    return b_hh_;
  }
};

// Gathers every two elements of a vector in a vector of pairs
template<typename T>
std::vector<pair_of<T>> pair_vec(const std::vector<T>& vals) {
  TORCH_CHECK(vals.size() % 2 == 0, "Odd number of params or hiddens given to a bidirectional RNN");
  std::vector<pair_of<T>> result;
  result.reserve(vals.size() / 2);
  for (size_t i = 0; i < vals.size(); i += 2) {
    result.emplace_back(vals[i], vals[i + 1]);
  }
  return result;
}

// Flattens a vector of pairs
template<typename T>
std::vector<T> unpair_vec(std::vector<pair_of<T>>&& vals) {
  std::vector<T> result;
  result.reserve(vals.size() * 2);
  for (const auto i : c10::irange(vals.size())) {
    result.push_back(std::move(vals[i].first));
    result.push_back(std::move(vals[i].second));
  }
  return result;
}

// Parses a flat list of parameter tensors into a list of CellParams
std::vector<CellParams> gather_params(TensorList params, bool has_biases, bool has_projections = false) {
  static at::Tensor undefined;
  std::vector<CellParams> result;
  if (has_biases) {
    if (has_projections) {
      TORCH_CHECK(params.size() % 5 == 0, "got an incorrect number of RNN parameters");
      result.reserve(params.size() / 5);
      for (size_t i = 0; i < params.size(); i += 5) {
        result.emplace_back(params[i], params[i + 1], params[i + 2], params[i + 3], params[i + 4]);
      }
    } else {
      TORCH_CHECK(params.size() % 4 == 0, "got an incorrect number of RNN parameters");
      result.reserve(params.size() / 4);
      for (size_t i = 0; i < params.size(); i += 4) {
        result.emplace_back(params[i], params[i + 1], params[i + 2], params[i + 3], undefined);
      }
    }
  } else {
    if (has_projections) {
      TORCH_CHECK(params.size() % 3 == 0, "got an incorrect number of RNN parameters");
      result.reserve(params.size() / 3);
      for (size_t i = 0; i < params.size(); i += 3) {
        result.emplace_back(params[i], params[i + 1], undefined, undefined, params[i + 2]);
      }
    } else {
      TORCH_CHECK(params.size() % 2 == 0, "got an incorrect number of RNN parameters");
      result.reserve(params.size() / 2);
      for (size_t i = 0; i < params.size(); i += 2) {
        result.emplace_back(params[i], params[i + 1], undefined, undefined, undefined);
      }
    }
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// HIDDEN STATE FUNCTIONS
//
// Functions implemented below are implemented as templates based on hidden type,
// because they need to work both with simple RNNs and GRU (which use a single Tensor),
// as well as with LSTM (or possibly more complicated architectures in the future).
// Still, there are some operations that need to be performed on the hidden states
// alone, and for this purpose we provide an overloaded set of functions below.

Tensor hidden_as_output(const Tensor& t) { return t; }
Tensor hidden_as_output(const tpair_of<Tensor>& t) { return std::get<0>(t); }

template<size_t index>
std::vector<Tensor> project(at::ArrayRef<tpair_of<Tensor>> tuples) {
  std::vector<Tensor> result;
  result.reserve(tuples.size());
  for (auto & t : tuples) {
    result.push_back(std::get<index>(t));
  }
  return result;
}

Tensor hidden_concat(at::ArrayRef<Tensor> hiddens) { return at::cat(hiddens, 0); }
tpair_of<Tensor> hidden_concat(at::ArrayRef<tpair_of<Tensor>> hiddens) {
  return std::make_tuple(hidden_concat(project<0>(hiddens)), hidden_concat(project<1>(hiddens)));
}

Tensor hidden_slice(const Tensor& t, int64_t start, int64_t end) {
  return t.narrow(0, start, end - start);
}
tpair_of<Tensor> hidden_slice(const tpair_of<Tensor>& t, int64_t start, int64_t end) {
  return std::make_tuple(hidden_slice(std::get<0>(t), start, end),
                         hidden_slice(std::get<1>(t), start, end));
}

////////////////////////////////////////////////////////////////////////////////
// CELL IMPLEMENTATIONS
//
// Cell is a basic component of an RNN, representing a single application of the
// recurrent function. You can think of it as a function of signature
//
// (Tensor input, hidden_type hidden, CellParams) -> hidden_type
//
// which means that it consumes an input tensor, and updates the previous hidden state.
// It's a struct only because functional programming in C++ is a pain, and it's easier
// to pass around "vtable pointers" than actual function pointers.

void check_rnn_cell_forward_input(const Tensor& input, const c10::SymInt& input_size) {
  TORCH_CHECK(
    input.sym_size(1) == input_size,
    "input has inconsistent input_size: got ", input.sym_size(1), " expected ", input_size);
}

void check_rnn_cell_forward_hidden(const Tensor& input, const Tensor& hx, const c10::SymInt& hidden_size, const c10::SymInt& hidden_label) {
  TORCH_CHECK(
    input.sym_size(0) == hx.sym_size(0),
    "Input batch size ", input.sym_size(0), " doesn't match hidden", hidden_label, " batch size ", hx.sym_size(0));

  TORCH_CHECK(
    hx.sym_size(1) == hidden_size,
    "hidden", hidden_label, " has inconsistent hidden_size: got ", hx.sym_size(1), ", expected ", hidden_size);
}

template<int64_t gate_count>
inline void check_rnn_cell_forward_weights(const Tensor& w_ih, const Tensor& w_hh, const c10::SymInt& hidden_size){
    TORCH_CHECK(w_ih.size(0) == gate_count * hidden_size, "weight_ih first dim must be ", gate_count, " * hidden_size = ",
        gate_count * hidden_size, ", but got ", w_ih.size(0));
    TORCH_CHECK(w_hh.size(0) == gate_count * hidden_size, "weight_hh first dim must be ", gate_count, " * hidden_size = ",
        gate_count * hidden_size, ", but got ", w_hh.size(0));
}


template<typename hidden_type_tmpl, typename cell_params_tmpl>
struct Cell {
  using hidden_type = hidden_type_tmpl;
  using cell_params = cell_params_tmpl;

  virtual ~Cell() = default; // This is really dumb, but enables projects with
                             // -Wnon-virtual-dtor to compile...

  virtual hidden_type operator()(
      const Tensor& input,
      const hidden_type& hidden,
      const cell_params& params,
      bool pre_compute_input = false) const = 0;
};

template<typename nonlinearity, typename cell_params>
struct SimpleCell : Cell<Tensor, cell_params> {
  using hidden_type = Tensor;
  Tensor operator()(
      const Tensor& input,
      const Tensor& hidden,
      const cell_params& params,
      bool pre_compute_input = false) const override {
    return nonlinearity{}(params.linear_hh(hidden).add_(
        pre_compute_input ? input : params.linear_ih(input)));
  }
};

// TODO: can use inplace ops?
template <typename cell_params>
struct LSTMCell : Cell<std::tuple<Tensor, Tensor>, cell_params> {
  using hidden_type = std::tuple<Tensor, Tensor>;

  hidden_type operator()(
      const Tensor& input,
      const hidden_type& hidden,
      const cell_params& params,
      bool pre_compute_input = false) const override {
    const auto& [hx, cx] = hidden;

    if (input.is_cuda() || input.is_xpu() || input.is_privateuseone()) {
      TORCH_CHECK(!pre_compute_input);
      auto igates = params.matmul_ih(input);
      auto hgates = params.matmul_hh(hx);
      auto result = at::_thnn_fused_lstm_cell(
          igates, hgates, cx, params.b_ih(), params.b_hh());
      // applying projections if w_hr is defined
      auto hy = params.matmul_hr(std::get<0>(result));
      // Slice off the workspace argument (it's needed only for AD).
      return std::make_tuple(std::move(hy), std::move(std::get<1>(result)));
    }

    const auto gates = params.linear_hh(hx).add_(
        pre_compute_input ? input : params.linear_ih(input));
    auto chunked_gates = gates.unsafe_chunk(4, 1);
    auto ingate = chunked_gates[0].sigmoid_();
    auto forgetgate = chunked_gates[1].sigmoid_();
    auto cellgate = chunked_gates[2].tanh_();
    auto outgate = chunked_gates[3].sigmoid_();
    auto cy = (forgetgate * cx).add_(ingate * cellgate);
    auto hy = outgate * cy.tanh();
    hy = params.matmul_hr(hy);
    return std::make_tuple(std::move(hy), std::move(cy));
  }

};

template <typename cell_params>
struct GRUCell : Cell<Tensor, cell_params> {
  using hidden_type = Tensor;

  hidden_type operator()(
      const Tensor& input,
      const hidden_type& hidden,
      const cell_params& params,
      bool pre_compute_input = false) const override {
    if (input.is_cuda() || input.is_xpu() || input.is_privateuseone()) {
      TORCH_CHECK(!pre_compute_input);
      auto igates = params.matmul_ih(input);
      auto hgates = params.matmul_hh(hidden);
      auto result = at::_thnn_fused_gru_cell(
          igates, hgates, hidden, params.b_ih(), params.b_hh());
      // Slice off the workspace argument (it's needed only for AD).
      return std::move(std::get<0>(result));
    }
    const auto chunked_igates = pre_compute_input
        ? input.unsafe_chunk(3, 1)
        : params.linear_ih(input).unsafe_chunk(3, 1);
    auto chunked_hgates = params.linear_hh(hidden).unsafe_chunk(3, 1);
    const auto reset_gate =
        chunked_hgates[0].add_(chunked_igates[0]).sigmoid_();
    const auto input_gate =
        chunked_hgates[1].add_(chunked_igates[1]).sigmoid_();
    const auto new_gate =
        chunked_igates[2].add(chunked_hgates[2].mul_(reset_gate)).tanh_();
    return (hidden - new_gate).mul_(input_gate).add_(new_gate);
  }
};

////////////////////////////////////////////////////////////////////////////////
// LAYER IMPLEMENTATIONS
//
// Layers are scan-like higher-order functions, which take in cells, and
// transform them to functions of signature
//
// (io_type input, hidden_type hidden, param_type params) -> (io_type, hidden_type)
//
// which can apply the cell over a sequence of inputs, and produce both a new set
// of hidden states, as well as a concatenated output of each step.

template<typename output_type, typename hidden_type>
struct LayerOutput {
  output_type outputs;
  hidden_type final_hidden;
};

template<typename io_type, typename hidden_type, typename param_type>
struct Layer {
  using output_type = LayerOutput<io_type, hidden_type>;

  virtual ~Layer() = default; // This is really dumb, but enables projects with
                              // -Wnon-virtual-dtor to compile...
  virtual output_type operator()(
      const io_type& input,
      const hidden_type& input_hidden,
      const param_type& params) const = 0;
};

template<typename hidden_type, typename cell_params>
struct FullLayer : Layer<Tensor, hidden_type, cell_params> {
  using output_type =
      typename Layer<Tensor, hidden_type, cell_params>::output_type;
  using unstacked_output_type = LayerOutput<std::vector<Tensor>, hidden_type>;

  FullLayer(Cell<hidden_type, cell_params>& cell)
    : cell_(cell) {}

  unstacked_output_type operator()(
      const std::vector<Tensor>& step_inputs,
      const hidden_type& input_hidden,
      const cell_params& params,
      bool pre_compute_input = false) const {
    std::vector<Tensor> step_outputs;
    auto hidden = input_hidden;
    for (const auto& input : step_inputs) {
      hidden = cell_(input, hidden, params, pre_compute_input);
      step_outputs.emplace_back(hidden_as_output(hidden));
    }
    return {std::move(step_outputs), std::move(hidden)};
  }

  output_type operator()(
      const Tensor& inputs,
      const hidden_type& input_hidden,
      const cell_params& params) const override {
    if (inputs.device().is_cpu()) {
      const auto inputs_w = params.linear_ih(inputs);
      auto unstacked_output =
          (*this)(inputs_w.unbind(0), input_hidden, params, true);
      TORCH_CHECK(unstacked_output.outputs.size()>0, "Expected sequence length to be larger than 0 in RNN");
      return {at::stack(unstacked_output.outputs, 0),
              std::move(unstacked_output.final_hidden)};
    }
    auto unstacked_output = (*this)(inputs.unbind(0), input_hidden, params);
    TORCH_CHECK(unstacked_output.outputs.size()>0, "Expected sequence length to be larger than 0 in RNN");
    return {at::stack(unstacked_output.outputs, 0),
            std::move(unstacked_output.final_hidden)};
  }

  Cell<hidden_type, cell_params>& cell_;
};

template <typename dir_hidden_type, typename cell_params>
struct FullBidirectionalLayer
    : Layer<Tensor, pair_of<dir_hidden_type>, pair_of<cell_params>> {
  using hidden_type = pair_of<dir_hidden_type>;
  using param_type = pair_of<cell_params>;
  using output_type = typename Layer<Tensor, hidden_type, param_type>::output_type;

  FullBidirectionalLayer(Cell<dir_hidden_type, cell_params>& cell)
    : layer_(cell) {}

  output_type operator()(
      const Tensor& input,
      const hidden_type& input_hidden,
      const param_type& params) const override {
    std::vector<Tensor> step_inputs;
    if (input.device().is_cpu()) {
      auto input_w = params.first.linear_ih(input);
      step_inputs = input_w.unbind(0);
      auto fw_result = layer_(
          step_inputs, input_hidden.first, params.first, true);
      TORCH_CHECK(!fw_result.outputs.empty(), "Expected sequence length to be larger than 0 in RNN");
      auto fw_output = at::stack(fw_result.outputs, 0);
      input_w = params.second.linear_ih(input);
      step_inputs = input_w.unbind(0);
      auto rev_step_inputs = reverse(std::move(step_inputs));
      auto rev_result =
          layer_(rev_step_inputs, input_hidden.second, params.second, true);
      std::reverse(rev_result.outputs.begin(), rev_result.outputs.end());
      auto rev_output = at::stack(rev_result.outputs, 0);
      return {at::cat({fw_output, rev_output}, fw_output.dim() - 1),
              std::make_pair(
                  std::move(fw_result.final_hidden),
                  std::move(rev_result.final_hidden))};
    }

    step_inputs = input.unbind(0);
    auto fw_result = layer_(step_inputs, input_hidden.first, params.first);
    TORCH_CHECK(!fw_result.outputs.empty(), "Expected sequence length to be larger than 0 in RNN");
    auto fw_output = at::stack(fw_result.outputs, 0);
    auto rev_step_inputs = reverse(std::move(step_inputs));
    auto rev_result =
        layer_(rev_step_inputs, input_hidden.second, params.second);
    std::reverse(rev_result.outputs.begin(), rev_result.outputs.end());
    auto rev_output = at::stack(rev_result.outputs, 0);
    return {at::cat({fw_output, rev_output}, fw_output.dim() - 1),
            std::make_pair(
                std::move(fw_result.final_hidden),
                std::move(rev_result.final_hidden))};
  }

  std::vector<Tensor> reverse(std::vector<Tensor>&& x) const {
    std::reverse(x.begin(), x.end());
    return std::move(x);
  }

  FullLayer<dir_hidden_type, cell_params> layer_;
};

template<typename hidden_type, typename cell_params>
struct PackedLayer : Layer<PackedSequence, hidden_type, cell_params> {
  using output_type =
      typename Layer<PackedSequence, hidden_type, cell_params>::output_type;

  PackedLayer(Cell<hidden_type, cell_params>& cell)
    : cell_(cell) {}

  output_type operator()(
      const PackedSequence& input,
      const hidden_type& input_hidden,
      const cell_params& params) const override {

    std::vector<at::Tensor> step_outputs;
    std::vector<hidden_type> hiddens;
    int64_t input_offset = 0;
    int64_t num_steps = input.batch_sizes.size(0);
    const int64_t* batch_sizes = input.batch_sizes.const_data_ptr<int64_t>();
    int64_t last_batch_size = batch_sizes[0];

    const Tensor* input_ptr = &input.data;
    bool pre_compute_input = false;
    Tensor input_w;
    if (input.data.device().is_cpu()) {
      input_w = params.linear_ih(input.data);
      input_ptr = &input_w;
      pre_compute_input = true;
    }

    // Batch sizes is a sequence of decreasing lengths, which are offsets
    // into a 1D list of inputs. At every step we slice out batch_size elements,
    // and possibly account for the decrease in the batch size since the last step,
    // which requires us to slice the hidden state (since some sequences
    // are completed now). The sliced parts are also saved, because we will need
    // to return a tensor of final hidden state.
    auto hidden = input_hidden;
    for (const auto i : c10::irange(num_steps)) {
      const int64_t batch_size = batch_sizes[i];
      auto step_input = input_ptr->narrow(0, input_offset, batch_size);
      input_offset += batch_size;
      const int64_t dec = last_batch_size - batch_size;
      if (dec > 0) {
        hiddens.emplace_back(
            hidden_slice(hidden, last_batch_size - dec, last_batch_size));
        hidden = hidden_slice(hidden, 0, last_batch_size - dec);
      }

      last_batch_size = batch_size;
      hidden = cell_(step_input, hidden, params, pre_compute_input);
      step_outputs.push_back(hidden_as_output(hidden));
    }
    hiddens.emplace_back(hidden);
    std::reverse(hiddens.begin(), hiddens.end());

    return {PackedSequence{at::cat(step_outputs, 0), input.batch_sizes},
            hidden_concat(hiddens)};
  }

  Cell<hidden_type, cell_params>& cell_;
};

template<typename hidden_type, typename cell_params>
struct ReversedPackedLayer : Layer<PackedSequence, hidden_type, cell_params> {
  using output_type =
      typename Layer<PackedSequence, hidden_type, cell_params>::output_type;

  ReversedPackedLayer(Cell<hidden_type, cell_params>& cell)
    : cell_(cell) {}

  output_type operator()(
      const PackedSequence& input,
      const hidden_type& input_hidden,
      const cell_params& params) const override {
    std::vector<at::Tensor> step_outputs;
    int64_t input_offset = input.data.size(0);
    int64_t num_steps = input.batch_sizes.size(0);
    const int64_t* batch_sizes = input.batch_sizes.const_data_ptr<int64_t>();
    int64_t last_batch_size = batch_sizes[num_steps - 1];

    const Tensor* input_ptr = &input.data;
    bool pre_compute_input = false;
    Tensor input_w;
    if (input.data.device().is_cpu()) {
      input_w = params.linear_ih(input.data);
      input_ptr = &input_w;
      pre_compute_input = true;
    }

    // Here the situation is similar to that above, except we start out with
    // the smallest batch size (and a small set of hidden states we actually use),
    // and progressively expand the hidden states, as we move backwards over the
    // 1D list of inputs.
    auto hidden = hidden_slice(input_hidden, 0, batch_sizes[num_steps - 1]);
    for (int64_t i = num_steps - 1; i >= 0; --i) {
      const int64_t batch_size = batch_sizes[i];
      const int64_t inc = batch_size - last_batch_size;
      if (inc > 0) {
        hidden = hidden_concat(ArrayRef<hidden_type>{
            hidden, hidden_slice(input_hidden, last_batch_size, batch_size)});
      }
      auto step_input =
          input_ptr->narrow(0, input_offset - batch_size, batch_size);
      input_offset -= batch_size;
      last_batch_size = batch_size;
      hidden = cell_(step_input, hidden, params, pre_compute_input);
      step_outputs.emplace_back(hidden_as_output(hidden));
    }
    std::reverse(step_outputs.begin(), step_outputs.end());
    return {PackedSequence{at::cat(step_outputs, 0), input.batch_sizes},
            std::move(hidden)};
  }

  Cell<hidden_type, cell_params>& cell_;
};

template <typename dir_hidden_type, typename cell_params>
struct PackedBidirectionalLayer
    : Layer<PackedSequence, pair_of<dir_hidden_type>, pair_of<cell_params>> {
  using hidden_type = pair_of<dir_hidden_type>;
  using param_type = pair_of<cell_params>;
  using output_type =
      typename Layer<PackedSequence, hidden_type, param_type>::output_type;

  PackedBidirectionalLayer(Cell<dir_hidden_type, cell_params>& cell)
    : layer_(cell), rev_layer_(cell) {}

  output_type operator()(
      const PackedSequence& input,
      const hidden_type& input_hidden,
      const param_type& params) const override {
    auto fw_result = layer_(input, input_hidden.first, params.first);
    auto rev_result = rev_layer_(input, input_hidden.second, params.second);
    PackedSequence output{
        at::cat({fw_result.outputs.data, rev_result.outputs.data}, -1),
        input.batch_sizes};
    return {std::move(output),
            std::make_pair(
                std::move(fw_result.final_hidden),
                std::move(rev_result.final_hidden))};
  }

  PackedLayer<dir_hidden_type, cell_params> layer_;
  ReversedPackedLayer<dir_hidden_type, cell_params> rev_layer_;
};

////////////////////////////////////////////////////////////////////////////////
// apply_layer_stack
//
// layers are convenient, but in reality we often want to stack them. this little
// helper manages slicing of all inputs and parameters, and repeatedly feeds them
// into the given layer. returns the last layer's outputs, and a vector of final
// hidden states produced at each level.

Tensor dropout(const Tensor& input, double p) {
  return at::dropout(input, p, /*train=*/true);
}

PackedSequence dropout(const PackedSequence& input, double p) {
  return {at::dropout(input.data, p, /*train=*/true), input.batch_sizes};
}

template<typename io_type, typename hidden_type, typename weight_type>
LayerOutput<io_type, std::vector<hidden_type>>
apply_layer_stack(const Layer<io_type, hidden_type, weight_type>& layer, const io_type& input,
                  const std::vector<hidden_type>& hiddens, const std::vector<weight_type>& weights,
                  int64_t num_layers, double dropout_p, bool train) {
  TORCH_CHECK(num_layers == (int64_t)hiddens.size(), "Expected more hidden states in stacked_rnn");
  TORCH_CHECK(num_layers == (int64_t)weights.size(), "Expected more weights in stacked_rnn");

  auto layer_input = input;
  auto hidden_it = hiddens.begin();
  auto weight_it = weights.begin();
  std::vector<hidden_type> final_hiddens;
  final_hiddens.reserve(num_layers);
  for (const auto l : c10::irange(num_layers)) {
    auto layer_output = layer(layer_input, *(hidden_it++), *(weight_it++));
    final_hiddens.push_back(std::move(layer_output.final_hidden));
    layer_input = std::move(layer_output.outputs);

    if (dropout_p != 0 && train && l < num_layers - 1) {
      layer_input = dropout(layer_input, dropout_p);
    }
  }

  return {std::move(layer_input), std::move(final_hiddens)};
}

////////////////////////////////////////////////////////////////////////////////
// HELPERS SIMPLIFYING DISPATCH TO FUNCTIONS ABOVE
////////////////////////////////////////////////////////////////////////////////

template<typename CellType, template<typename,typename> class LayerT, template<typename,typename> class BidirLayerT, typename cell_params, typename io_type>
LayerOutput<io_type, std::vector<typename CellType::hidden_type>> _rnn_impl(
      const io_type& input,
      const std::vector<cell_params>& params,
      const std::vector<typename CellType::hidden_type>& hiddens,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  using hidden_type = typename CellType::hidden_type;
  CellType cell;
  if (bidirectional) {
    using BidirLayer = BidirLayerT<hidden_type, cell_params>;
    auto bidir_result = apply_layer_stack(BidirLayer{cell}, input, pair_vec(hiddens), pair_vec(params), num_layers, dropout_p, train);
    return {std::move(bidir_result.outputs), unpair_vec(std::move(bidir_result.final_hidden))};
  } else {
    return apply_layer_stack(LayerT<hidden_type,cell_params>{cell}, input, hiddens, params, num_layers, dropout_p, train);
  }
}

template<typename CellType, template<typename,typename> class LayerT, template<typename,typename> class BidirLayerT, typename cell_params, typename io_type>
std::tuple<io_type, Tensor> _rnn_impl_with_concat(
      const io_type& input,
      const std::vector<cell_params>& params,
      const std::vector<typename CellType::hidden_type>& hiddens,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  auto result = _rnn_impl<CellType, LayerT, BidirLayerT>(input, params, hiddens, num_layers, dropout_p, train, bidirectional);
  return std::make_tuple(std::move(result.outputs), at::stack(result.final_hidden, 0));
}

template<template<typename,typename> class LayerT, template<typename,typename> class BidirLayerT, typename cell_params, typename io_type>
std::tuple<io_type, Tensor, Tensor> _lstm_impl(
      const io_type& input,
      const std::vector<cell_params>& params, const Tensor& hx, const Tensor& cx,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  // It's much more useful for us to work on lists of pairs of hx and cx for each layer, so we need
  // to transpose a pair of those tensors.
  auto layer_hx = hx.unbind(0);
  auto layer_cx = cx.unbind(0);
  int64_t total_layers = layer_hx.size();
  std::vector<typename LSTMCell<cell_params>::hidden_type> hiddens;
  hiddens.reserve(total_layers);
  for (const auto i : c10::irange(total_layers)) {
    hiddens.emplace_back(std::move(layer_hx[i]), std::move(layer_cx[i]));
  }

  auto result = _rnn_impl<LSTMCell<cell_params>, LayerT, BidirLayerT>(input, params, hiddens, num_layers, dropout_p, train, bidirectional);

  // Now, we need to reverse the transposed we performed above.
  std::vector<Tensor> hy, cy;
  hy.reserve(total_layers); cy.reserve(total_layers);
  for (auto & hidden : result.final_hidden) {
    hy.push_back(std::move(std::get<0>(hidden)));
    cy.push_back(std::move(std::get<1>(hidden)));
  }

  return std::make_tuple(std::move(result.outputs), at::stack(hy, 0), at::stack(cy, 0));
}


} // anonymous namespace

bool _use_cudnn_rnn_flatten_weight() {
  return detail::getCUDAHooks().compiledWithCuDNN();
}

// NB: This a (composite) wrapper for _thnn_fused_lstm_cell_backward_impl.
//     It duplicates the outputs of this function so the non-composite version doesn't have to.
//     The point is so that we avoid triggering TensorImpl use count asserts in debug mode
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _thnn_fused_lstm_cell_backward( const std::optional<Tensor>& grad_hy_opt, const std::optional<Tensor>& grad_cy_opt,
      const Tensor& cx, const Tensor& cy,
      const Tensor& workspace, bool has_bias) {
  TORCH_INTERNAL_ASSERT(!GradMode::is_enabled());
  auto ret = at::_thnn_fused_lstm_cell_backward_impl(grad_hy_opt, grad_cy_opt, cx, cy, workspace, has_bias);
  return std::make_tuple(std::get<0>(ret), std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<2>(ret));
}


////////////////////////////////////////////////////////////////////////////////
// PUBLIC FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

#define ONE_HIDDEN_RNN(NAME, CELL)                                          \
  DEFINE_DISPATCH(NAME##_cudnn_stub);                                       \
  DEFINE_DISPATCH(NAME##_miopen_stub);                                      \
  DEFINE_DISPATCH(NAME##_packed_cudnn_stub);                                \
  DEFINE_DISPATCH(NAME##_packed_miopen_stub);                               \
  REGISTER_NO_CPU_DISPATCH(NAME##_cudnn_stub)                              \
  REGISTER_NO_CPU_DISPATCH(NAME##_miopen_stub)                             \
  REGISTER_NO_CPU_DISPATCH(NAME##_packed_cudnn_stub)                       \
  REGISTER_NO_CPU_DISPATCH(NAME##_packed_miopen_stub)                      \
                                                                            \
  std::tuple<Tensor, Tensor> NAME(                                          \
      const Tensor& _input,                                                 \
      const Tensor& hx,                                                     \
      TensorList _params,                                                   \
      bool has_biases,                                                      \
      int64_t num_layers,                                                   \
      double dropout_p,                                                     \
      bool train,                                                           \
      bool bidirectional,                                                   \
      bool batch_first) {                                                   \
    if (use_cudnn(_input)) {                                                \
      Tensor output, hy;                                                    \
      NAME##_cudnn_stub(                                                    \
          _input.device().type(),                                           \
          output,                                                           \
          hy,                                                               \
          _input,                                                           \
          hx,                                                               \
          _params,                                                          \
          has_biases,                                                       \
          num_layers,                                                       \
          dropout_p,                                                        \
          train,                                                            \
          bidirectional,                                                    \
          batch_first);                                                     \
      return std::make_tuple(std::move(output), std::move(hy));             \
    }                                                                       \
    if (use_miopen(_input, dropout_p)) {                                    \
      try {                                                                 \
        Tensor output, hy;                                                  \
        NAME##_miopen_stub(                                                 \
            _input.device().type(),                                         \
            output,                                                         \
            hy,                                                             \
            _input,                                                         \
            hx,                                                             \
            _params,                                                        \
            has_biases,                                                     \
            num_layers,                                                     \
            dropout_p,                                                      \
            train,                                                          \
            bidirectional,                                                  \
            batch_first);                                                   \
        miopen_rnn_probe(_input.device().index())                           \
            .store(miopen_rnn_ok, std::memory_order_relaxed);               \
        return std::make_tuple(std::move(output), std::move(hy));           \
      } catch (const std::exception& e) {                                   \
        if (!miopen_rnn_handle_failure(_input.device().index(), e.what()))  \
          throw;                                                            \
      }                                                                     \
    }                                                                       \
    check_attributes(_input, _params, hx);                                  \
    auto input = batch_first ? _input.transpose(0, 1) : _input;             \
    auto params = gather_params(_params, has_biases);                       \
    auto results =                                                          \
        _rnn_impl_with_concat<CELL, FullLayer, FullBidirectionalLayer>(     \
            input,                                                          \
            params,                                                         \
            hx.unbind(0),                                                   \
            num_layers,                                                     \
            dropout_p,                                                      \
            train,                                                          \
            bidirectional);                                                 \
    if (batch_first) {                                                      \
      std::get<0>(results).transpose_(0, 1);                                \
    }                                                                       \
    return results;                                                         \
  }                                                                         \
                                                                            \
  std::tuple<Tensor, Tensor> NAME(                                          \
      const Tensor& data,                                                   \
      const Tensor& batch_sizes,                                            \
      const Tensor& hx,                                                     \
      TensorList _params,                                                   \
      bool has_biases,                                                      \
      int64_t num_layers,                                                   \
      double dropout_p,                                                     \
      bool train,                                                           \
      bool bidirectional) {                                                 \
    if (use_cudnn(data)) {                                                  \
      Tensor output, hy;                                                    \
      NAME##_packed_cudnn_stub(                                             \
          data.device().type(),                                             \
          output,                                                           \
          hy,                                                               \
          data,                                                             \
          batch_sizes,                                                      \
          hx,                                                               \
          _params,                                                          \
          has_biases,                                                       \
          num_layers,                                                       \
          dropout_p,                                                        \
          train,                                                            \
          bidirectional);                                                   \
      return std::make_tuple(std::move(output), std::move(hy));             \
    }                                                                       \
    if (use_miopen(data, dropout_p)) {                                      \
      try {                                                                 \
        Tensor output, hy;                                                  \
        NAME##_packed_miopen_stub(                                          \
            data.device().type(),                                           \
            output,                                                         \
            hy,                                                             \
            data,                                                           \
            batch_sizes,                                                    \
            hx,                                                             \
            _params,                                                        \
            has_biases,                                                     \
            num_layers,                                                     \
            dropout_p,                                                      \
            train,                                                          \
            bidirectional);                                                 \
        miopen_rnn_probe(data.device().index())                             \
            .store(miopen_rnn_ok, std::memory_order_relaxed);               \
        return std::make_tuple(std::move(output), std::move(hy));           \
      } catch (const std::exception& e) {                                   \
        if (!miopen_rnn_handle_failure(data.device().index(), e.what()))    \
          throw;                                                            \
      }                                                                     \
    }                                                                       \
    PackedSequence input{data, batch_sizes};                                \
    auto params = gather_params(_params, has_biases);                       \
    auto result =                                                           \
        _rnn_impl_with_concat<CELL, PackedLayer, PackedBidirectionalLayer>( \
            input,                                                          \
            params,                                                         \
            hx.unbind(0),                                                   \
            num_layers,                                                     \
            dropout_p,                                                      \
            train,                                                          \
            bidirectional);                                                 \
    auto& packed_output = std::get<0>(result);                              \
    return std::make_tuple(                                                 \
        std::move(packed_output.data), std::move(std::get<1>(result)));     \
  }

ONE_HIDDEN_RNN(gru, GRUCell<CellParams>)

using tanf_cell_type = SimpleCell<tanh_f, CellParams>;
ONE_HIDDEN_RNN(rnn_tanh, tanf_cell_type)
using relu_cell_type = SimpleCell<relu_f, CellParams>;
ONE_HIDDEN_RNN(rnn_relu, relu_cell_type)

DEFINE_DISPATCH(lstm_cudnn_stub);
DEFINE_DISPATCH(lstm_packed_cudnn_stub);
DEFINE_DISPATCH(lstm_miopen_stub);
DEFINE_DISPATCH(lstm_packed_miopen_stub);
DEFINE_DISPATCH(lstm_mkldnn_stub);
REGISTER_NO_CPU_DISPATCH(lstm_cudnn_stub)
REGISTER_NO_CPU_DISPATCH(lstm_packed_cudnn_stub)
REGISTER_NO_CPU_DISPATCH(lstm_miopen_stub)
REGISTER_NO_CPU_DISPATCH(lstm_packed_miopen_stub)

std::tuple<Tensor, Tensor, Tensor> lstm(
      const Tensor& _input, TensorList hx,
      TensorList _params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");
  if (use_cudnn(_input)) {
    Tensor output, hy, cy;
    lstm_cudnn_stub(_input.device().type(), output, hy, cy, _input, hx, _params, has_biases,
            num_layers, dropout_p, train, bidirectional, batch_first);
    return std::make_tuple(std::move(output), std::move(hy), std::move(cy));
  }
#ifdef USE_MPS
  if (_input.is_mps()) {
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> output = at::_lstm_mps(_input, hx, _params, has_biases,
            num_layers, dropout_p, train, bidirectional, batch_first);
    std::tuple<Tensor, Tensor, Tensor> return_values = std::make_tuple(
        std::move(std::get<0>(output)),
        std::move(std::get<1>(output)),
        std::move(std::get<2>(output)));
    return return_values;
  }
#endif
  // if cells are of different size, that means projections are used
  bool has_projections = (hx[0].sym_size(2) != hx[1].sym_size(2));
  if (use_miopen(_input, dropout_p)) {
    if (!has_projections) {
      try {
        Tensor output, hy, cy;
        lstm_miopen_stub(_input.device().type(), output, hy, cy, _input, hx, _params, has_biases,
                  num_layers, dropout_p, train, bidirectional, batch_first);
        miopen_rnn_probe(_input.device().index()).store(miopen_rnn_ok, std::memory_order_relaxed);
        return std::make_tuple(std::move(output), std::move(hy), std::move(cy));
      } catch (const std::exception& e) {
        if (!miopen_rnn_handle_failure(_input.device().index(), e.what())) {
          throw;
        }
      }
    } else {
      TORCH_WARN_ONCE(
          "LSTM with projections is not supported with MIOpen. Using default implementation.");
    }
  }

  if (use_mkldnn(_input, _params, hx)) {
    if (!has_projections) {
      if (hx[0].unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
        TORCH_WARN_ONCE(
          "LSTM with symbolic sizes and strides is not supported with oneDNN. Using default implementation.");
      } else {
        Tensor output, hy, cy;
        lstm_mkldnn_stub(_input.device().type(), output, hy, cy,_input, hx, _params, has_biases,
            num_layers, dropout_p, train, bidirectional, batch_first);
        return std::make_tuple(std::move(output), std::move(hy), std::move(cy));
      }
    } else {
      TORCH_WARN_ONCE(
          "LSTM with projections is not supported with oneDNN. Using default implementation.");
    }
  }

  check_attributes(_input, _params, hx);
  auto input = batch_first ? _input.transpose(0, 1) : _input;
  auto params = gather_params(_params, has_biases, has_projections);
  auto results = _lstm_impl<FullLayer, FullBidirectionalLayer>(
      input, params, hx[0], hx[1], num_layers, dropout_p, train, bidirectional);
  if (batch_first) {
    std::get<0>(results) = std::get<0>(results).transpose(0, 1);
  }
  return results;
}

std::tuple<Tensor, Tensor, Tensor> lstm(
      const Tensor& data, const Tensor& batch_sizes, TensorList hx,
      TensorList _params, bool has_biases,
      int64_t num_layers, double dropout_p, bool train, bool bidirectional) {
  TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");
  if (use_cudnn(data)) {
    Tensor output, hy, cy;
    lstm_packed_cudnn_stub(data.device().type(), output, hy, cy, data, batch_sizes, hx,
            _params, has_biases, num_layers, dropout_p, train, bidirectional);
    return std::make_tuple(std::move(output), std::move(hy), std::move(cy));
  }
  // if cells are of different size, that means projections are used
  bool has_projections = (hx[0].size(2) != hx[1].size(2));
  if (use_miopen(data, dropout_p)) {
    if (!has_projections) {
      try {
        Tensor output, hy, cy;
        lstm_packed_miopen_stub(data.device().type(), output, hy, cy, data, batch_sizes, hx,
                _params, has_biases, num_layers, dropout_p, train, bidirectional);
        miopen_rnn_probe(data.device().index()).store(miopen_rnn_ok, std::memory_order_relaxed);
        return std::make_tuple(std::move(output), std::move(hy), std::move(cy));
      } catch (const std::exception& e) {
        if (!miopen_rnn_handle_failure(data.device().index(), e.what())) {
          throw;
        }
      }
    } else {
      TORCH_WARN_ONCE(
          "LSTM with projections is not supported with MIOpen. Using default implementation.");
    }
  }

  // If packed sequence has uniform batch size, unwrap to regular tensor
  // and dispatch to the standard lstm (enables oneDNN path for CPU/XPU)
  if (!has_projections && use_mkldnn(data, _params, hx) && batch_sizes.size(0) > 0) {
    const int64_t* bs_ptr = batch_sizes.const_data_ptr<int64_t>();
    int64_t first_batch = bs_ptr[0];
    bool uniform = true;
    for (int64_t i = 1; i < batch_sizes.size(0); i++) {
      if (bs_ptr[i] != first_batch) {
        uniform = false;
        break;
      }
    }
    if (uniform) {
      int64_t seq_len = batch_sizes.size(0);
      auto input_3d = data.reshape({seq_len, first_batch, data.size(-1)});
      auto result = at::native::lstm(input_3d, hx, _params, has_biases,
          num_layers, dropout_p, train, bidirectional, /*batch_first=*/false);
      std::get<0>(result) = std::get<0>(result).reshape(
          {seq_len * first_batch, std::get<0>(result).size(-1)});
      return result;
    }
  }

  PackedSequence input { data, batch_sizes };
  auto params = gather_params(_params, has_biases, has_projections);
  auto result = _lstm_impl<PackedLayer, PackedBidirectionalLayer>(
      input, params, hx[0], hx[1], num_layers, dropout_p, train, bidirectional);
  auto & packed_output = std::get<0>(result);
  return std::make_tuple(std::move(packed_output.data),
                         std::move(std::get<1>(result)),
                         std::move(std::get<2>(result)));
}

std::tuple<Tensor, Tensor> lstm_cell(
    const Tensor& input, TensorList hx,
    const Tensor& w_ih, const Tensor& w_hh, const std::optional<Tensor>& b_ih_opt, const std::optional<Tensor>& b_hh_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> b_ih_maybe_owned = at::borrow_from_optional_tensor(b_ih_opt);
  const Tensor& b_ih = *b_ih_maybe_owned;
  const Tensor& b_hh = b_hh_opt.value_or(Tensor());

  TORCH_CHECK(hx.size() == 2, "lstm_cell expects two hidden states");
  auto hidden_size = w_hh.sym_size(1);
  check_rnn_cell_forward_input(input, w_ih.sym_size(1));
  check_rnn_cell_forward_weights<4>(w_ih, w_hh, hidden_size);
  check_rnn_cell_forward_hidden(input, hx[0], hidden_size, 0);
  check_rnn_cell_forward_hidden(input, hx[1], std::move(hidden_size), 1);
  static at::Tensor undefined;
  return LSTMCell<CellParams>{}(input, std::make_tuple(hx[0], hx[1]), CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
_thnn_differentiable_lstm_cell_backward( const std::optional<Tensor>& grad_hy_opt, const std::optional<Tensor>& grad_cy_opt,
    const Tensor& input_gates,
    const Tensor& hidden_gates, const std::optional<Tensor>& input_bias_opt, const std::optional<Tensor>& hidden_bias_opt,
    const Tensor& cx,
    const Tensor& cy) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> grad_hy_maybe_owned = at::borrow_from_optional_tensor(grad_hy_opt);
  const Tensor& grad_hy = *grad_hy_maybe_owned;
  const Tensor& grad_cy = grad_cy_opt.value_or(Tensor());
  const Tensor& input_bias = input_bias_opt.value_or(Tensor());
  const Tensor& hidden_bias = hidden_bias_opt.value_or(Tensor());

  if (!grad_hy.defined() && !grad_cy.defined()) {
    return std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>();
  }
  Tensor gates = input_gates + hidden_gates;
  if (input_bias.defined()) {
    gates = gates + input_bias;
  }
  if (hidden_bias.defined()) {
    gates = gates + hidden_bias;
  }
  auto chunked_gates = gates.unsafe_chunk(4, 1);
  Tensor i = chunked_gates[0].sigmoid();
  Tensor f = chunked_gates[1].sigmoid();
  Tensor c = chunked_gates[2].tanh();
  Tensor o = chunked_gates[3].sigmoid();

  Tensor gcx = cy.tanh();
  Tensor gog;
  TORCH_INTERNAL_ASSERT((grad_hy.defined() || grad_cy.defined()),"either gradient with respect to hy or cy should be defined");
  if (grad_hy.defined()) {
    gog = grad_hy * gcx;
    gog = at::sigmoid_backward(gog, o);
    gcx = at::tanh_backward(grad_hy * o, gcx);
    if (grad_cy.defined()) {
      gcx = gcx + grad_cy;
    }
  } else if (grad_cy.defined()) {
    gog = at::zeros_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    gcx = grad_cy;
  }
  Tensor gig = gcx * c;
  Tensor gfg = gcx * cx;
  Tensor gcg = gcx * i;
  gcx = gcx * f;
  gig = at::sigmoid_backward(gig, i);
  gfg = at::sigmoid_backward(gfg, f);
  gcg = at::tanh_backward(gcg, c);
  Tensor grad_gates = at::cat({std::move(gig), std::move(gfg), std::move(gcg), std::move(gog)}, 1);
  Tensor grad_bias = input_bias.defined() ? grad_gates.sum(0, /*keepdim=*/false) : at::Tensor{};
  return std::make_tuple(grad_gates, grad_gates, std::move(gcx), grad_bias, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _thnn_differentiable_gru_cell_backward(
    const Tensor& grad_hy,
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& hx, const std::optional<Tensor>& input_bias_opt, const std::optional<Tensor>& hidden_bias_opt){
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> input_bias_maybe_owned = at::borrow_from_optional_tensor(input_bias_opt);
  const Tensor& input_bias = *input_bias_maybe_owned;
  const Tensor& hidden_bias = hidden_bias_opt.value_or(Tensor());

  Tensor in_g = input_gates;
  Tensor h_g = hidden_gates;
  if (input_bias.defined()){
    in_g = in_g+input_bias;
  }
  if (hidden_bias.defined()){
    h_g = h_g + hidden_bias;
  }
  auto chunked_input_gates = in_g.unsafe_chunk(3, 1);
  Tensor ir = chunked_input_gates[0];
  Tensor ii = chunked_input_gates[1];
  Tensor in = chunked_input_gates[2];
  auto chunked_hidden_gates = h_g.unsafe_chunk(3, 1);
  Tensor hr = chunked_hidden_gates[0];
  Tensor hi = chunked_hidden_gates[1];
  Tensor hn = chunked_hidden_gates[2];
  Tensor rg = (ir + hr).sigmoid();
  Tensor ig = (ii + hi).sigmoid();
  Tensor grad_hx = grad_hy * ig;
  Tensor ng = (in+rg*hn).tanh();
  Tensor gig = at::sigmoid_backward(grad_hy * (hx - ng), ig);
  Tensor gin = at::tanh_backward(grad_hy * (1 - ig), ng);
  Tensor ghn = gin * rg;
  Tensor grg = at::sigmoid_backward(gin * hn, rg);
  Tensor grad_input_gates = at::cat({grg,gig,std::move(gin)}, 1);
  Tensor grad_hidden_gates = at::cat({std::move(grg),std::move(gig),std::move(ghn)}, 1);
  Tensor grad_input_bias = input_bias.defined() ? grad_input_gates.sum(0, /*keepdim=*/false) : at::Tensor{};
  Tensor grad_hidden_bias = input_bias.defined() ? grad_hidden_gates.sum(0, /*keepdim=*/false) : at::Tensor{};
  return std::make_tuple(std::move(grad_input_gates), std::move(grad_hidden_gates),
                         std::move(grad_hx), std::move(grad_input_bias), std::move(grad_hidden_bias));
}

Tensor gru_cell(
    const Tensor& input, const Tensor& hx,
    const Tensor& w_ih, const Tensor& w_hh, const std::optional<Tensor>& b_ih_opt, const std::optional<Tensor>& b_hh_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> b_ih_maybe_owned = at::borrow_from_optional_tensor(b_ih_opt);
  const Tensor& b_ih = *b_ih_maybe_owned;
  const Tensor& b_hh = b_hh_opt.value_or(Tensor());

  check_rnn_cell_forward_input(input, w_ih.size(1));
  check_rnn_cell_forward_hidden(input, hx, w_hh.size(1), 0);
  check_rnn_cell_forward_weights<3>(w_ih, w_hh, w_hh.size(1));
  static at::Tensor undefined;
  return GRUCell<CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
}

Tensor rnn_tanh_cell(
    const Tensor& input, const Tensor& hx,
    const Tensor& w_ih, const Tensor& w_hh, const std::optional<Tensor>& b_ih_opt, const std::optional<Tensor>& b_hh_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> b_ih_maybe_owned = at::borrow_from_optional_tensor(b_ih_opt);
  const Tensor& b_ih = *b_ih_maybe_owned;
  const Tensor& b_hh = b_hh_opt.value_or(Tensor());

  static at::Tensor undefined;
  check_rnn_cell_forward_weights<1>(w_ih, w_hh, w_hh.size(1));
  check_rnn_cell_forward_input(input, w_ih.size(1));
  check_rnn_cell_forward_hidden(input, hx, w_hh.size(1), 0);
  return SimpleCell<tanh_f, CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
}

Tensor rnn_relu_cell(
    const Tensor& input, const Tensor& hx,
    const Tensor& w_ih, const Tensor& w_hh, const std::optional<Tensor>& b_ih_opt, const std::optional<Tensor>& b_hh_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> b_ih_maybe_owned = at::borrow_from_optional_tensor(b_ih_opt);
  const Tensor& b_ih = *b_ih_maybe_owned;
  const Tensor& b_hh = b_hh_opt.value_or(Tensor());

  static at::Tensor undefined;
  check_rnn_cell_forward_weights<1>(w_ih, w_hh, w_hh.size(1));
  check_rnn_cell_forward_input(input, w_ih.size(1));
  check_rnn_cell_forward_hidden(input, hx, w_hh.size(1), 0);
  return SimpleCell<relu_f, CellParams>{}(input, hx, CellParams{w_ih, w_hh, b_ih, b_hh, undefined});
}

}  // namespace at::native
