// Sources/ATenCXXDoctests/masked_scatter_doctests.cpp
#include "doctest.h"
#include "tensor_shim.hpp"

using i64 = int64_t;

static TTSTensor makeTensorF32(const std::vector<float>& host, const std::vector<i64>& shape) {
  return TTSTensor::fromHostArray(host.data(), shape.size(), shape.data(), c10::ScalarType::Float);
}

static TTSTensor makeTensorI64(const std::vector<i64>& host, const std::vector<i64>& shape) {
  return TTSTensor::fromHostArray(host.data(), shape.size(), shape.data(), c10::ScalarType::Long);
}

static TTSTensor makeBoolFromBytes(const std::vector<uint8_t>& host, const std::vector<i64>& shape) {
  return TTSTensor::fromHostBytesAsBool(host.data(), shape.size(), shape.data());
}



  DOCTEST_TEST_CASE("1-D: scatter into masked positions; keep others") {
    // self: [0,0,0,0]
    TTSTensor self = makeTensorF32({0,0,0,0}, {4});
    TTSTensor mask = makeBoolFromBytes({1,0,1,0}, {4});
    TTSTensor src  = makeTensorF32({10,20}, {2});

    TTSTensor out = self.maskedScatter(mask, src);
    TTSTensor expected = makeTensorF32({10,0,20,0}, {4});

    DOCTEST_CHECK(out.allclose(expected, /*rtol*/0.0, /*atol*/0.0, /*equal_nan*/false));
  }

  DOCTEST_TEST_CASE("Order: mask consumes source sequentially (row-major)") {
    TTSTensor self = makeTensorI64({0,0,0,0,0}, {5});
    TTSTensor mask = makeBoolFromBytes(
      {0,0,0,1,1,  1,1,0,1,1}, {2,5}
    );
    TTSTensor src  = makeTensorI64({0,1,2,3,4,  5,6,7,8,9}, {2,5});

    TTSTensor out = self.maskedScatter(mask, src);
    // Expected taken from your Python example:
    TTSTensor expected = makeTensorI64({0,0,0,0,1,  2,3,0,4,5}, {2,5});

    DOCTEST_CHECK(out.allclose(expected, /*rtol*/0.0, /*atol*/0.0, /*equal_nan*/false));
  }

  // FILE: masked_scatter_doctests.cpp

DOCTEST_TEST_CASE("Shape check: mismatch between mask.trueCount and source.numel throws") {
    TTSTensor self = makeTensorF32({0,0,0,0}, {4});
    TTSTensor mask = makeBoolFromBytes({1,0,1,0}, {4}); // two trues
    TTSTensor src  = makeTensorF32({10,20,30}, {3});    // three elements — mismatch

    // The expected error message (without the stack trace)
    const std::string expected_msg = "masked_scatter: number of elements in source (3) must match the number of true elements in mask (2)";

    try {
        // This line is expected to throw
        self.masked_scatter(mask, src);

        // If we get here, no exception was thrown, which is an error.
        DOCTEST_FAIL("Expected c10::Error to be thrown, but no exception occurred.");

    } catch (const c10::Error& e) {
        // The correct exception type was caught. Now check the message.
        std::string actual_msg = e.what();

        // Check if the actual message contains our expected message as a substring.
        bool message_contains_expected = (actual_msg.find(expected_msg) != std::string::npos);
        DOCTEST_CHECK(message_contains_expected);

    } catch (...) {
        // Catch any other unexpected exception types.
        DOCTEST_FAIL("An unexpected exception type was thrown.");
    }
}

