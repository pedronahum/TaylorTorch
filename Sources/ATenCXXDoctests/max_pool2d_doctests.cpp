// Sources/ATenCXXDoctests/max_pool2d_doctests.cpp
#include "include/doctest.h"
#include "tensor_shim.hpp"
#include <array>
#include <vector>

using i64 = int64_t;

static TTSTensor makeTensorF64(const std::vector<double> &host, const std::vector<i64> &shape)
{
  return TTSTensor::fromArray(host, shape);
}

DOCTEST_TEST_CASE("max_pool2d: 2x2 stride-2 matches PyTorch semantics")
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
  const std::array<i64, 2> dilation{1, 1};

  TTSTensor result = input.max_pool2d(
      kernel.data(), kernel.size(),
      stride.data(), stride.size(),
      padding.data(), padding.size(),
      dilation.data(), dilation.size(),
      /*ceil_mode=*/false);

  TTSTensor expected = makeTensorF64({5.0, 7.0, 13.0, 15.0}, {1, 1, 2, 2});
  DOCTEST_CHECK(result.allclose(expected, 0.0, 0.0, false));
}

