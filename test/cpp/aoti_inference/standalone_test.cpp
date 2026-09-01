#include <chrono>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <torch/all.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/csrc/jit/serialization/pickle.h>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr
        << "Usage: ./standalone_test <input file> [benchmark iter] [warm-up iter]"
        << std::endl;
    return 1;
  }
  size_t benchmark_iter = 10;
  size_t warmup_iter = 3;

  if (argc == 3) {
    benchmark_iter = std::stoul(argv[2]);
  } else if (argc == 4) {
    benchmark_iter = std::stoul(argv[2]);
    warmup_iter = std::stoul(argv[3]);
  }

  std::string data_path = argv[1];
  std::ifstream in(data_path, std::ios::binary);
  TORCH_CHECK(in.good(), "cannot open fixture: ", data_path);
  std::vector<char> bytes(
      (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  auto data_loader = torch::jit::pickle_load(bytes).toGenericDict();
  auto at = [&](const std::string& key) {
    auto it = data_loader.find(key);
    TORCH_CHECK(it != data_loader.end(), "fixture has no key: ", key);
    return it->value();
  };
  const auto& model_so_path = at("model_so_path").toStringRef();
  const auto& input_tensors = at("inputs").toTensorList().vec();
  const auto& output_tensors = at("outputs").toTensorList().vec();

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
      model_so_path);

  // Check results.
  auto actual_output_tensors = runner->run(input_tensors);
  assert(output_tensors.size() == actual_output_tensors.size());
  for (size_t i = 0; i < output_tensors.size(); i++) {
    assert(torch::allclose(output_tensors[i], actual_output_tensors[i]));
  }

  // Start benchmarking for lowered module.
  // Warm up
  for (size_t i = 0; i < warmup_iter; i++) {
    runner->run(input_tensors);
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < benchmark_iter; i++) {
    runner->run(input_tensors);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> lowered_duration = end - start;

  std::cout << "[Lowered] Time for " << benchmark_iter
            << "iter(s): " << lowered_duration.count() << " sec(s)"
            << std::endl;

  return 0;
}
