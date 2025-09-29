// Sources/ATenCXXDoctests/ttsd_doctest_bridge.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "include/doctest.h"
#include "tensor_shim.hpp"
#include <cstdint>
#include <vector>

// ✅ FIX: Reverted helpers to use the original factory methods.
static TTSTensor makeTensorF32(const std::vector<float> &host, const std::vector<int64_t> &shape)
{
  return TTSTensor::fromArray(host, shape);
}

static TTSTensor makeBoolFromBytes(const std::vector<uint8_t> &host, const std::vector<int64_t> &shape)
{
  return TTSTensor::fromMask(host, shape);
}

DOCTEST_TEST_CASE("indexPut with boolean mask matches masked_scatter semantics")
{
  TTSTensor self = makeTensorF32({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {2, 5});
  TTSTensor mask = makeBoolFromBytes({0, 0, 0, 0, 0, 0, 0, 0, 1, 1}, {2, 5});
  TTSTensor source = makeTensorF32({10.0, 20.0}, {2});
  TTSTensor expected = makeTensorF32({0, 0, 0, 0, 0, 0, 0, 0, 10.0, 20.0}, {2, 5});

  TTSTensor indices[1] = {mask};
  auto out = self.indexPut(indices, 1, source, false);

  DOCTEST_CHECK(out.allclose(expected, 0.0, 0.0, false));
}
