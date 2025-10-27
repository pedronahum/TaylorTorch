#pragma once

#include <c10/core/Scalar.h>
#include <cstdint>

// Helper function to explicitly construct c10::Scalar from int64_t
// This avoids C++ overload ambiguity on Linux where both long and long long are 64-bit
inline c10::Scalar make_scalar_int64(int64_t value) {
    return c10::Scalar(static_cast<int64_t>(value));
}
