#include <ATen/ATen.h>

int main() {
  std::cout << at::ones({3,4}, at::kFloat) << '\n';
}
