// Sources/ATenCXXDoctests/masked_scatter_doctests.cpp
#include "include/doctest.h"
#include "tensor_shim.hpp"
#include <vector>

using i64 = int64_t;

// ✅ FIX: Reverted helpers to use the original factory methods which have default device arguments.
static TTSTensor makeTensorF32(const std::vector<float> &host, const std::vector<i64> &shape)
{
  return TTSTensor::fromArray(host, shape);
}

static TTSTensor makeTensorI64(const std::vector<i64> &host, const std::vector<i64> &shape)
{
  return TTSTensor::fromArray(host, shape);
}

static TTSTensor makeBoolFromBytes(const std::vector<uint8_t> &host, const std::vector<i64> &shape)
{
  return TTSTensor::fromMask(host, shape);
}

DOCTEST_TEST_CASE("1-D: scatter into masked positions; keep others")
{
  TTSTensor self = makeTensorF32({0, 0, 0, 0}, {4});
  TTSTensor mask = makeBoolFromBytes({1, 0, 1, 0}, {4});
  TTSTensor src = makeTensorF32({10, 20}, {2});

  TTSTensor out = self.maskedScatter(mask, src);
  TTSTensor expected = makeTensorF32({10, 0, 20, 0}, {4});

  DOCTEST_CHECK(out.allclose(expected, 0.0, 0.0, false));
}

DOCTEST_TEST_CASE("Shape check: mismatch between mask.trueCount and source.numel throws")
{
  TTSTensor self = makeTensorF32({0, 0, 0, 0}, {4});
  TTSTensor mask = makeBoolFromBytes({1, 0, 1, 0}, {4});
  TTSTensor src = makeTensorF32({10, 20, 30}, {3});

  const std::string expected_msg = "must match the number of true elements";

  try
  {
    // ✅ FIX: Changed from snake_case 'masked_scatter' to correct 'maskedScatter'.
    self.maskedScatter(mask, src);
    DOCTEST_FAIL("Expected c10::Error to be thrown, but no exception occurred.");
  }
  catch (const c10::Error &e)
  {
    std::string actual_msg = e.what();
    DOCTEST_CHECK(actual_msg.find(expected_msg) != std::string::npos);
  }
}