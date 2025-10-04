// Sources/ATenCXXDoctests/conv2d_backward_doctests.cpp
#include "include/doctest.h"
#include "tensor_shim.hpp"
#include <array>
#include <tuple>
#include <vector>

using i64 = int64_t;

static TTSTensor makeTensorF64(const std::vector<double> &host, const std::vector<i64> &shape)
{
  return TTSTensor::fromArray(host, shape);
}

DOCTEST_TEST_CASE("conv2d_backward returns gradients matching manual computation")
{
  TTSTensor input = makeTensorF64({1.0, 2.0,
                                   3.0, 4.0},
                                  {1, 1, 2, 2});

  TTSTensor weight = makeTensorF64({2.0}, {1, 1, 1, 1});

  TTSTensor gradOut = makeTensorF64({1.0, 1.0,
                                     1.0, 1.0},
                                    {1, 1, 2, 2});

  const std::array<i64, 2> stride{1, 1};
  const std::array<i64, 2> padding{0, 0};
  const std::array<i64, 2> dilation{1, 1};

  auto grads = TTSTensor::_conv2d_backward(
      gradOut,
      input,
      weight,
      stride.data(), stride.size(),
      padding.data(), padding.size(),
      dilation.data(), dilation.size(),
      /*groups=*/1);

  TTSTensor gradInput = std::get<0>(grads);
  TTSTensor gradWeight = std::get<1>(grads);
  TTSTensor gradBias = std::get<2>(grads);

  TTSTensor expectedInput = makeTensorF64({
                                            2.0, 2.0,
                                            2.0, 2.0
                                          },
                                          {1, 1, 2, 2});
  TTSTensor expectedWeight = makeTensorF64({10.0}, {1, 1, 1, 1});
  TTSTensor expectedBias = makeTensorF64({4.0}, {1});

  DOCTEST_CHECK(gradInput.allclose(expectedInput, 1e-12, 0.0, false));
  DOCTEST_CHECK(gradWeight.allclose(expectedWeight, 1e-12, 0.0, false));
  DOCTEST_CHECK(gradBias.allclose(expectedBias, 1e-12, 0.0, false));
}

