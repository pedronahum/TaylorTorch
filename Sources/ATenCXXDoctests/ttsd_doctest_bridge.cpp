// Sources/ATenCXXDoctests/ttsd_doctest_bridge.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
//#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES  // avoid clash with c10's CHECK macro
#include "include/doctest.h"

#include "aten_shim.hpp"
#include "tensor_shim.hpp"
#include <cstdint>

// Convenience: CPU device
static inline c10::Device cpu() { return make_device(c10::DeviceType::CPU); }

// Build a 2x5 boolean mask where entries >= 8 are true.
// Array form (for intuition):
//  [[0,1,2,3,4],
//   [5,6,7,8,9]] >= 8  ->  [[F,F,F,F,F],
//                            [F,F,F,T,T]]
static TTSTensor makeMask2x5() {
  auto a = TTSTensor::arange(
      c10::Scalar(int64_t{0}),
      c10::Scalar(int64_t{10}),
      c10::Scalar(int64_t{1}),
      c10::ScalarType::Long,
      cpu());

  const int64_t shape[2] = {2, 5};
  auto a25 = a.reshape(shape, /*ndims=*/2);
  return a25.geScalar(c10::Scalar(int64_t{8})); // bool tensor
}

DOCTEST_TEST_CASE("indexPut with boolean mask matches masked_scatter semantics") {
  // self: zeros([2,5]); mask: as above; source: length == mask.true_count == 2
  const int64_t shape[2] = {2, 5};

  auto self = TTSTensor::zeros(shape, /*ndims=*/2, c10::ScalarType::Float, cpu());
  auto mask = makeMask2x5();

  // source = [10.0, 20.0]
  auto source = TTSTensor::linspace(
      /*start=*/0.0, /*end=*/1.0, /*steps=*/2, c10::ScalarType::Float, cpu());
  // scale to [10, 20]
  source = source.mulScalar(c10::Scalar(10.0)).addScalar(c10::Scalar(10.0));

  // masked_scatter(self, mask, source)  ≡  index_put(self, {mask}, source, accumulate=false)
  TTSTensor indices[1] = {mask};
  auto out = self.indexPut(indices, /*n=*/1, source, /*accumulate=*/false);

  // Validate the two positions that should have been written:
  // They are in row 1, cols 3 and 4.
  auto v13 = out.select(/*dim=*/0, /*index=*/int64_t{1}).select(/*dim=*/0, /*index=*/int64_t{3});
  auto v14 = out.select(/*dim=*/0, /*index=*/int64_t{1}).select(/*dim=*/0, /*index=*/int64_t{4});

  // Work with scalars robustly via sumAll on a 0-d view (still returns a scalar tensor).
  // If you have a scalar-reader helper (see §2), you can assert directly.
  auto s13 = v13.sumAll(); // scalar tensor == 10.0
  auto s14 = v14.sumAll(); // scalar tensor == 20.0

  // Quick numeric check: (s13 - 10).abs().sumAll() == 0  and likewise for s14.
  // We keep this entirely in-ATen/TTSTensor land to avoid needing item<double>() helpers.
  auto zero13 = s13.subScalar(c10::Scalar(10.0)).abs_().sumAll();
  auto zero14 = s14.subScalar(c10::Scalar(20.0)).abs_().sumAll();

  // Compare to exact zero by turning into a boolean mask and reducing.
  DOCTEST_CHECK(zero13.eqScalar(c10::Scalar(0.0)).sumAll().toBool());
  DOCTEST_CHECK(zero14.eqScalar(c10::Scalar(0.0)).sumAll().toBool());

  // Sanity: the rest is still zero. Sum of row 0 is 0; sum of row 1 is 30.
  auto row0_sum = out.select(0, int64_t{0}).sumAll();
  auto row1_sum = out.select(0, int64_t{1}).sumAll();

  DOCTEST_CHECK(row0_sum.eqScalar(c10::Scalar(0.0)).sumAll().toBool());
  DOCTEST_CHECK(row1_sum.eqScalar(c10::Scalar(30.0)).sumAll().toBool());
}
