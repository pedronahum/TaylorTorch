// Sources/ATenCXXDoctests/avg_pool2d_doctests.cpp
#include "include/doctest.h"
#include "tensor_shim.hpp"
#include <array>
#include <vector>

using i64 = int64_t;

static TTSTensor makeTensorF64(const std::vector<double> &host, const std::vector<i64> &shape)
{
  return TTSTensor::fromArray(host, shape);
}

DOCTEST_TEST_CASE("avg_pool2d: 2x2 stride-2 matches PyTorch defaults")
{
  TTSTensor input = makeTensorF64({
                                  0.0, 1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0, 7.0,
                                  8.0, 9.0, 10.0, 11.0,
                                  12.0, 13.0, 14.0, 15.0
                                },
                                {1, 1, 4, 4});

  const std::array<i64, 2> kernel{2, 2};
  const std::array<i64, 2> stride{2, 2};
  const std::array<i64, 2> padding{0, 0};

  TTSTensor result = input.avg_pool2d(
      kernel.data(), kernel.size(),
      stride.data(), stride.size(),
      padding.data(), padding.size(),
      /*ceil_mode=*/false);

  TTSTensor expected = makeTensorF64({2.5, 4.5, 10.5, 12.5}, {1, 1, 2, 2});
  DOCTEST_CHECK(result.allclose(expected, 1e-12, 0.0, false));
}

DOCTEST_TEST_CASE("avg_pool2d: padding excluded from denominator when count_include_pad is false")
{
  TTSTensor input = makeTensorF64({1.0, 2.0, 3.0, 4.0}, {1, 1, 2, 2});

  const std::array<i64, 2> kernel{2, 2};
  const std::array<i64, 2> stride{1, 1};
  const std::array<i64, 2> padding{1, 1};

  TTSTensor result = input.avg_pool2d(
      kernel.data(), kernel.size(),
      stride.data(), stride.size(),
      padding.data(), padding.size(),
      /*ceil_mode=*/false);

  TTSTensor expected = makeTensorF64({
                                      1.0, 1.5, 2.0,
                                      2.0, 2.5, 3.0,
                                      3.0, 3.5, 4.0
                                    },
                                    {1, 1, 3, 3});

  DOCTEST_CHECK(result.allclose(expected, 1e-12, 0.0, false));
}

