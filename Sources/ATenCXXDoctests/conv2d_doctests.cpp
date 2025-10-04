// Sources/ATenCXXDoctests/conv2d_doctests.cpp
#include "include/doctest.h"
#include "tensor_shim.hpp"
#include <array>
#include <vector>

using i64 = int64_t;

static TTSTensor makeTensorF64(const std::vector<double> &host, const std::vector<i64> &shape)
{
  return TTSTensor::fromArray(host, shape);
}

DOCTEST_TEST_CASE("conv2d: 1x1 in/out channel valid convolution matches hand calculation")
{
  TTSTensor input = makeTensorF64({
                                   0.0, 1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0, 7.0,
                                   8.0, 9.0, 10.0, 11.0,
                                   12.0, 13.0, 14.0, 15.0
                                 },
                                 {1, 1, 4, 4});

  TTSTensor weight = makeTensorF64({
                                    1.0, 2.0,
                                    3.0, 4.0
                                  },
                                  {1, 1, 2, 2});

  const std::array<i64, 2> stride{1, 1};
  const std::array<i64, 2> padding{0, 0};
  const std::array<i64, 2> dilation{1, 1};

  TTSTensor result = TTSTensor::_conv2d(
      input,
      weight,
      /*bias=*/nullptr,
      stride.data(), stride.size(),
      padding.data(), padding.size(),
      dilation.data(), dilation.size(),
      /*groups=*/1);

  TTSTensor expected = makeTensorF64({
                                       34.0, 44.0, 54.0,
                                       74.0, 84.0, 94.0,
                                       114.0, 124.0, 134.0
                                     },
                                     {1, 1, 3, 3});

  DOCTEST_CHECK(result.allclose(expected, 1e-12, 0.0, false));
}

DOCTEST_TEST_CASE("conv2d: bias term is broadcast across the output")
{
  TTSTensor input = makeTensorF64({
                                   1.0, 2.0,
                                   3.0, 4.0
                                 },
                                 {1, 1, 2, 2});

  TTSTensor weight = makeTensorF64({1.0}, {1, 1, 1, 1});
  TTSTensor bias = makeTensorF64({0.5}, {1});

  const std::array<i64, 2> stride{1, 1};
  const std::array<i64, 2> padding{0, 0};
  const std::array<i64, 2> dilation{1, 1};

  TTSTensor result = TTSTensor::_conv2d(
      input,
      weight,
      &bias,
      stride.data(), stride.size(),
      padding.data(), padding.size(),
      dilation.data(), dilation.size(),
      /*groups=*/1);

  TTSTensor expected = makeTensorF64({
                                       1.5, 2.5,
                                       3.5, 4.5
                                     },
                                     {1, 1, 2, 2});

  DOCTEST_CHECK(result.allclose(expected, 1e-12, 0.0, false));
}

