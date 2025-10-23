// Try compiling this minimal example with clang
// https://docs.pytorch.org/cppdocs/installing.html#minimal-example
#include <iostream>
#include <torch/torch.h>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}