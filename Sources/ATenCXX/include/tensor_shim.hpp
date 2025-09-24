#pragma once
#include <ATen/ATen.h>
#include <vector>

// Forward-declare the helper function so the class can see it
class TTSTensor;
inline TTSTensor masked_fill_tensor_helper(const TTSTensor &self, const TTSTensor &mask, const TTSTensor &value);

// A tiny, importer-friendly wrapper around at::Tensor.
class TTSTensor
{
  at::Tensor t_;

public:
  // ✅ Add this 'friend' declaration inside the class.
  // This gives the helper function access to private members.
  friend TTSTensor masked_fill_tensor_helper(const TTSTensor &self, const TTSTensor &mask, const TTSTensor &value);

  TTSTensor() = default;
  explicit TTSTensor(const at::Tensor &t) : t_(t) {}

  int64_t numel() const { return t_.numel(); }

  // ---- Factories
  static TTSTensor empty(const int64_t *sizes, size_t dim,
                         c10::ScalarType dtype, c10::Device device)
  {
    at::IntArrayRef shape(sizes, dim);
    auto opts = at::TensorOptions().dtype(dtype).device(device);
    return TTSTensor(at::empty(shape, opts));
  }

  static TTSTensor zeros(const int64_t *sizes, size_t dim,
                         c10::ScalarType dtype, c10::Device device)
  {
    at::IntArrayRef shape(sizes, dim);
    auto opts = at::TensorOptions().dtype(dtype).device(device);
    return TTSTensor(at::zeros(shape, opts));
  }

  static TTSTensor ones(const int64_t *sizes, size_t dim,
                        c10::ScalarType dtype, c10::Device device)
  {
    at::IntArrayRef shape(sizes, dim);
    auto opts = at::TensorOptions().dtype(dtype).device(device);
    return TTSTensor(at::ones(shape, opts));
  }

  static TTSTensor full(c10::Scalar value,
                        const int64_t *sizes, size_t dim,
                        c10::ScalarType dtype, c10::Device device)
  {
    at::IntArrayRef shape(sizes, dim);
    auto opts = at::TensorOptions().dtype(dtype).device(device);
    return TTSTensor(at::full(shape, value, opts));
  }

  static TTSTensor fromScalar(c10::Scalar value,
                              c10::ScalarType dtype,
                              c10::Device device)
  {
    auto opts = at::TensorOptions().dtype(dtype).device(device);
    return TTSTensor(at::scalar_tensor(value, opts));
  }

  // ---- Queries
  bool defined() const { return t_.defined(); }
  int64_t dim() const { return t_.dim(); }
  int64_t sizeAt(int64_t d) const { return t_.size(d); }
  c10::ScalarType dtype() const { return t_.scalar_type(); }
  c10::Device device() const { return t_.device(); }

  // ---- Conversions
  TTSTensor toDType(c10::ScalarType dt) const { return TTSTensor(t_.toType(dt)); }
  TTSTensor toDevice(c10::Device dev) const { return TTSTensor(t_.to(dev)); }

  // ---- Simple ops
  TTSTensor add(const TTSTensor &other, c10::Scalar alpha = 1) const
  {
    return TTSTensor(t_.add(other.t_, alpha));
  }
  TTSTensor addScalar(c10::Scalar s) const
  {
    return TTSTensor(t_.add(s));
  }

  // ---- Array I/O (NEW)

  // Create a tensor by copying 'elem_count' elements from a host buffer.
  // The copy happens on CPU and then moves to 'device' if needed.
  static TTSTensor fromHostBuffer(
      const void *data,
      size_t elem_count,
      const int64_t *sizes, size_t dim,
      c10::ScalarType dtype,
      c10::Device device)
  {
    at::IntArrayRef shape(sizes, dim);

    // Always stage on CPU first (avoids templated device copies).
    auto cpu_opts = at::TensorOptions().dtype(dtype).device(c10::DeviceType::CPU);
    at::Tensor t = at::empty(shape, cpu_opts);

    switch (dtype)
    {
    case c10::ScalarType::Float:
      std::memcpy(t.data_ptr<float>(), data, elem_count * sizeof(float));
      break;
    case c10::ScalarType::Double:
      std::memcpy(t.data_ptr<double>(), data, elem_count * sizeof(double));
      break;
    case c10::ScalarType::Int:
      std::memcpy(t.data_ptr<int32_t>(), data, elem_count * sizeof(int32_t));
      break;
    case c10::ScalarType::Long:
      std::memcpy(t.data_ptr<int64_t>(), data, elem_count * sizeof(int64_t));
      break;
    case c10::ScalarType::Short:
      std::memcpy(t.data_ptr<int16_t>(), data, elem_count * sizeof(int16_t));
      break;
    case c10::ScalarType::Byte:
      std::memcpy(t.data_ptr<uint8_t>(), data, elem_count * sizeof(uint8_t));
      break;
    case c10::ScalarType::Char:
      std::memcpy(t.data_ptr<int8_t>(), data, elem_count * sizeof(int8_t));
      break;
    case c10::ScalarType::Bool:
    {
      auto dst = t.data_ptr<bool>();
      auto src = static_cast<const uint8_t *>(data);
      for (size_t i = 0; i < elem_count; ++i)
        dst[i] = (src[i] != 0);
      break;
    }
    default:
      TORCH_CHECK(false, "fromHostBuffer: unsupported dtype");
    }

    if (device.type() != c10::DeviceType::CPU)
    {
      t = t.to(device, /*non_blocking=*/false, /*copy=*/true);
    }
    return TTSTensor(t);
  }

  // Copy tensor contents into a pre-allocated host buffer.
  // Ensures CPU, dtype, contiguous; returns false if 'out' is too small
  // or dtype unsupported in this helper.
  bool toHostBuffer(void *out, size_t out_elem_count, c10::ScalarType dtype) const
  {
    at::Tensor src = t_;
    if (src.device().type() != c10::DeviceType::CPU || src.scalar_type() != dtype)
    {
      src = src.to(c10::Device(c10::DeviceType::CPU), dtype, /*non_blocking=*/false, /*copy=*/true);
    }
    src = src.contiguous();

    size_t n = static_cast<size_t>(src.numel());
    if (n > out_elem_count)
      return false;

    switch (dtype)
    {
    case c10::ScalarType::Float:
      std::memcpy(out, src.data_ptr<float>(), n * sizeof(float));
      break;
    case c10::ScalarType::Double:
      std::memcpy(out, src.data_ptr<double>(), n * sizeof(double));
      break;
    case c10::ScalarType::Int:
      std::memcpy(out, src.data_ptr<int32_t>(), n * sizeof(int32_t));
      break;
    case c10::ScalarType::Long:
      std::memcpy(out, src.data_ptr<int64_t>(), n * sizeof(int64_t));
      break;
    case c10::ScalarType::Short:
      std::memcpy(out, src.data_ptr<int16_t>(), n * sizeof(int16_t));
      break;
    case c10::ScalarType::Byte:
      std::memcpy(out, src.data_ptr<uint8_t>(), n * sizeof(uint8_t));
      break;
    case c10::ScalarType::Char:
      std::memcpy(out, src.data_ptr<int8_t>(), n * sizeof(int8_t));
      break;
    case c10::ScalarType::Bool:
    {
      auto sp = src.data_ptr<bool>();
      auto dp = static_cast<uint8_t *>(out);
      for (size_t i = 0; i < n; ++i)
        dp[i] = sp[i] ? 1 : 0;
      break;
    }
    default:
      return false;
    }
    return true;
  }

  // ---- Indexing helpers (NEW)

private:
  static int64_t _canon_dim(const at::Tensor &t, int64_t dim)
  {
    auto d = dim < 0 ? dim + t.dim() : dim;
    TORCH_CHECK(d >= 0 && d < t.dim(), "dim out of range");
    return d;
  }

  static int64_t _canon_index(const at::Tensor &t, int64_t dim, int64_t idx)
  {
    auto d = _canon_dim(t, dim);
    auto size = t.size(d);
    auto i = idx < 0 ? idx + size : idx;
    TORCH_CHECK(i >= 0 && i < size, "index out of range");
    return i;
  }

  static void _canon_slice_bounds(const at::Tensor &t, int64_t dim,
                                  int64_t &start, int64_t &end, int64_t &step)
  {
    auto d = _canon_dim(t, dim);
    auto size = t.size(d);

    // Normalize negatives
    if (start < 0)
      start += size;
    if (end < 0)
      end += size;

    // Clamp to [0, size]
    if (start < 0)
      start = 0;
    if (start > size)
      start = size;
    if (end < 0)
      end = 0;
    if (end > size)
      end = size;

    TORCH_CHECK(step != 0, "slice step cannot be 0");
    TORCH_CHECK(step > 0, "slice with negative step not yet supported");
  }

public:
  TTSTensor select(int64_t dim, int64_t index) const
  {
    auto d = _canon_dim(t_, dim);
    auto i = _canon_index(t_, d, index);
    return TTSTensor(t_.select(d, i));
  }

  TTSTensor narrow(int64_t dim, int64_t start, int64_t length) const
  {
    auto d = _canon_dim(t_, dim);
    auto size = t_.size(d);
    if (start < 0)
      start += size;
    TORCH_CHECK(length >= 0, "narrow length must be >= 0");
    TORCH_CHECK(start >= 0 && start <= size, "narrow start out of range");
    TORCH_CHECK(start + length <= size, "narrow start+length exceeds size");
    return TTSTensor(t_.narrow(d, start, length));
  }

  TTSTensor slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const
  {
    _canon_slice_bounds(t_, dim, start, end, step);
    auto d = _canon_dim(t_, dim);
    return TTSTensor(t_.slice(d, start, end, step));
  }

  // ---- Shape ops & advanced indexing (NEW)

  // Canonicalize a dim for insertion (range: [0, t.dim()])
  static int64_t _canon_dim_inclusive(const at::Tensor &t, int64_t dim)
  {
    auto nd = t.dim();
    auto d = dim < 0 ? dim + nd + 1 : dim;
    TORCH_CHECK(d >= 0 && d <= nd, "insert dim out of range");
    return d;
  }

  TTSTensor transpose(int64_t dim0, int64_t dim1) const
  {
    auto d0 = _canon_dim(t_, dim0);
    auto d1 = _canon_dim(t_, dim1);
    return TTSTensor(t_.transpose(d0, d1));
  }

  TTSTensor permute(const int64_t *order, size_t ndims) const
  {
    std::vector<int64_t> v(order, order + ndims);
    // Normalize negatives and bounds-check
    for (auto &d : v)
    {
      d = (d < 0) ? d + t_.dim() : d;
      TORCH_CHECK(d >= 0 && d < t_.dim(), "permute: dim out of range");
    }
    return TTSTensor(t_.permute(v));
  }

  TTSTensor reshape(const int64_t *sizes, size_t ndims) const
  {
    at::IntArrayRef shape(sizes, ndims);
    // at::reshape returns a view if possible, otherwise a copy
    return TTSTensor(t_.reshape(shape));
  }

  TTSTensor squeezeAll() const { return TTSTensor(t_.squeeze()); }

  TTSTensor squeezeDim(int64_t dim) const
  {
    auto d = _canon_dim(t_, dim);
    return TTSTensor(t_.squeeze(d));
  }

  TTSTensor unsqueeze(int64_t dim) const
  {
    auto d = _canon_dim_inclusive(t_, dim);
    return TTSTensor(t_.unsqueeze(d));
  }

  TTSTensor flatten(int64_t start_dim, int64_t end_dim) const
  {
    auto sd = _canon_dim(t_, start_dim);
    auto ed = end_dim < 0 ? end_dim + t_.dim() : end_dim;
    TORCH_CHECK(ed >= 0 && ed < t_.dim(), "flatten end_dim out of range");
    return TTSTensor(t_.flatten(sd, ed));
  }

  // indexSelect(dim, indices[]) using a CPU Long tensor for indices
  TTSTensor indexSelect(int64_t dim, const int64_t *idx, size_t count) const
  {
    auto d = _canon_dim(t_, dim);
    auto opts = at::TensorOptions().dtype(c10::ScalarType::Long).device(c10::DeviceType::CPU);
    at::Tensor i = at::empty({static_cast<long>(count)}, opts);
    std::memcpy(i.data_ptr<int64_t>(), idx, count * sizeof(int64_t));
    return TTSTensor(t_.index_select(d, i));
  }

  // ---- Layout & contiguity
  bool isContiguous() const { return t_.is_contiguous(); }
  int64_t strideAt(int64_t d) const { return t_.stride(d); }
  TTSTensor contiguous() const { return TTSTensor(t_.contiguous()); }

  // ---- Joiners
  static TTSTensor cat(const TTSTensor *xs, size_t count, int64_t dim)
  {
    TORCH_CHECK(count > 0, "cat: empty input");
    std::vector<at::Tensor> v;
    v.reserve(count);
    for (size_t i = 0; i < count; ++i)
      v.push_back(xs[i].t_);
    int64_t d = dim;
    if (d < 0)
      d += v[0].dim();
    return TTSTensor(at::cat(v, d));
  }

  static TTSTensor stack(const TTSTensor *xs, size_t count, int64_t dim)
  {
    TORCH_CHECK(count > 0, "stack: empty input");
    std::vector<at::Tensor> v;
    v.reserve(count);
    for (size_t i = 0; i < count; ++i)
      v.push_back(xs[i].t_);
    int64_t d = dim;
    if (d < 0)
      d += v[0].dim() + 1; // stack inserts a new dim
    return TTSTensor(at::stack(v, d));
  }

  // ---- Random / range initializers
  static TTSTensor rand(const int64_t *sizes, size_t ndims,
                        c10::ScalarType dtype, c10::Device device)
  {
    at::IntArrayRef shape(sizes, ndims);
    auto opts = at::TensorOptions().dtype(dtype).device(device);
    return TTSTensor(at::rand(shape, opts));
  }

  static TTSTensor randn(const int64_t *sizes, size_t ndims,
                         c10::ScalarType dtype, c10::Device device)
  {
    at::IntArrayRef shape(sizes, ndims);
    auto opts = at::TensorOptions().dtype(dtype).device(device);
    return TTSTensor(at::randn(shape, opts));
  }

  static TTSTensor arange(c10::Scalar start, c10::Scalar end, c10::Scalar step,
                          c10::ScalarType dtype, c10::Device device)
  {
    auto opts = at::TensorOptions().dtype(dtype).device(device);
    return TTSTensor(at::arange(start, end, step, opts));
  }

  static TTSTensor linspace(double start, double end, int64_t steps,
                            c10::ScalarType dtype, c10::Device device)
  {
    auto opts = at::TensorOptions().dtype(dtype).device(device);
    return TTSTensor(at::linspace(start, end, steps, opts));
  }

  // ---- Math, Reductions, Comparisons, Linalg (NEW)

public:
  // Unary
  TTSTensor neg() const { return TTSTensor(at::neg(t_)); }
  TTSTensor abs_() const { return TTSTensor(at::abs(t_)); }
  TTSTensor relu() const { return TTSTensor(at::relu(t_)); }
  TTSTensor exp_() const { return TTSTensor(at::exp(t_)); }
  TTSTensor log_() const { return TTSTensor(at::log(t_)); }
  TTSTensor sqrt_() const { return TTSTensor(at::sqrt(t_)); }

  // Binary (tensor ⊗ tensor)
  TTSTensor sub(const TTSTensor &other, c10::Scalar alpha = 1) const
  {
    return TTSTensor(t_.sub(other.t_, alpha));
  }
  TTSTensor mul(const TTSTensor &other) const
  {
    return TTSTensor(t_.mul(other.t_));
  }
  TTSTensor div(const TTSTensor &other) const
  {
    return TTSTensor(t_.div(other.t_));
  }

  // Binary (tensor ⊗ scalar)
  TTSTensor subScalar(c10::Scalar s) const { return TTSTensor(t_.sub(s)); }
  TTSTensor mulScalar(c10::Scalar s) const { return TTSTensor(t_.mul(s)); }
  TTSTensor divScalar(c10::Scalar s) const { return TTSTensor(t_.div(s)); }

  // Power
  TTSTensor powScalar(c10::Scalar s) const { return TTSTensor(at::pow(t_, s)); }
  TTSTensor powTensor(const TTSTensor &other) const { return TTSTensor(at::pow(t_, other.t_)); }

  // Clamp
  TTSTensor clamp(c10::Scalar minv, c10::Scalar maxv) const
  {
    return TTSTensor(at::clamp(t_, minv, maxv));
  }

  // Reductions (all)
  TTSTensor sumAll() const { return TTSTensor(at::sum(t_)); }
  TTSTensor meanAll() const { return TTSTensor(at::mean(t_)); }

  // Reductions (along single dim, keepdim selectable)
  TTSTensor sumDim(int64_t dim, bool keepdim) const
  {
    auto d = _canon_dim(t_, dim);
    std::vector<int64_t> dims{d};
    return TTSTensor(at::sum(t_, dims, keepdim));
  }
  TTSTensor meanDim(int64_t dim, bool keepdim) const
  {
    auto d = _canon_dim(t_, dim);
    std::vector<int64_t> dims{d};
    return TTSTensor(at::mean(t_, dims, keepdim));
  }

  // Linalg
  TTSTensor matmul(const TTSTensor &other) const
  {
    return TTSTensor(at::matmul(t_, other.t_));
  }
  TTSTensor dot(const TTSTensor &other) const
  {
    return TTSTensor(at::dot(t_, other.t_));
  }

  // Comparisons (tensor ⊗ tensor) — result dtype = Bool
  TTSTensor eq(const TTSTensor &other) const { return TTSTensor(t_.eq(other.t_)); }
  TTSTensor lt(const TTSTensor &other) const { return TTSTensor(t_.lt(other.t_)); }
  TTSTensor le(const TTSTensor &other) const { return TTSTensor(t_.le(other.t_)); }
  TTSTensor gt(const TTSTensor &other) const { return TTSTensor(t_.gt(other.t_)); }
  TTSTensor ge(const TTSTensor &other) const { return TTSTensor(t_.ge(other.t_)); }

  // Comparisons (tensor ⊗ scalar)
  TTSTensor eqScalar(c10::Scalar s) const { return TTSTensor(t_.eq(s)); }
  TTSTensor ltScalar(c10::Scalar s) const { return TTSTensor(t_.lt(s)); }
  TTSTensor leScalar(c10::Scalar s) const { return TTSTensor(t_.le(s)); }
  TTSTensor gtScalar(c10::Scalar s) const { return TTSTensor(t_.gt(s)); }
  TTSTensor geScalar(c10::Scalar s) const { return TTSTensor(t_.ge(s)); }

  // Where (ternary)
  static TTSTensor where3(const TTSTensor &cond, const TTSTensor &a, const TTSTensor &b)
  {
    return TTSTensor(at::where(cond.t_, a.t_, b.t_));
  }

  // ---- Reductions that also return indices (NEW)

  // Returns the minimum VALUES
  TTSTensor minDim(int64_t dim, bool keepdim = false) const
  {
    auto d = _canon_dim(t_, dim);
    return TTSTensor(std::get<0>(at::min(t_, d, keepdim)));
  }
  // Returns the minimum INDICES
  TTSTensor argminDim(int64_t dim, bool keepdim = false) const
  {
    auto d = _canon_dim(t_, dim);
    return TTSTensor(std::get<1>(at::min(t_, d, keepdim)));
  }

  // Returns the maximum VALUES
  TTSTensor maxDim(int64_t dim, bool keepdim = false) const
  {
    auto d = _canon_dim(t_, dim);
    return TTSTensor(std::get<0>(at::max(t_, d, keepdim)));
  }
  // Returns the maximum INDICES
  TTSTensor argmaxDim(int64_t dim, bool keepdim = false) const
  {
    auto d = _canon_dim(t_, dim);
    return TTSTensor(std::get<1>(at::max(t_, d, keepdim)));
  }

  // Scalar full reductions (rank-0 tensors)
  TTSTensor minAll() const { return TTSTensor(at::min(t_)); }
  TTSTensor maxAll() const { return TTSTensor(at::max(t_)); }

  // Top-K (values and indices)
  TTSTensor topk_values(int64_t k, int64_t dim, bool largest = true, bool sorted = true) const
  {
    auto d = _canon_dim(t_, dim);
    auto tup = at::topk(t_, k, d, largest, sorted);
    return TTSTensor(std::get<0>(tup));
  }

  TTSTensor topk_indices(int64_t k, int64_t dim, bool largest = true, bool sorted = true) const
  {
    auto d = _canon_dim(t_, dim);
    auto tup = at::topk(t_, k, d, largest, sorted);
    return TTSTensor(std::get<1>(tup));
  }

  // Returns the sorted VALUES
  TTSTensor sortDim_values(int64_t dim, bool descending = false) const
  {
    auto d = _canon_dim(t_, dim);
    auto tup = at::sort(t_, d, descending); // tup is std::tuple<at::Tensor, at::Tensor>
    return TTSTensor(std::get<0>(tup));     // std::get<0> gets the first element (values)
  }

  // Returns the sorted INDICES
  TTSTensor sortDim_indices(int64_t dim, bool descending = false) const
  {
    auto d = _canon_dim(t_, dim);
    auto tup = at::sort(t_, d, descending);
    return TTSTensor(std::get<1>(tup)); // std::get<1> gets the second element (indices)
  }

  // ---- Element-wise min / max (tensor ⊗ tensor)
  TTSTensor minimum(const TTSTensor &other) const { return TTSTensor(at::minimum(t_, other.t_)); }
  TTSTensor maximum(const TTSTensor &other) const { return TTSTensor(at::maximum(t_, other.t_)); }

  // Broadcasting
  TTSTensor expand(const int64_t *sizes, size_t ndims, bool implicit = false) const
  {
    at::IntArrayRef shape(sizes, ndims);
    return TTSTensor(t_.expand(shape, implicit));
  }

  TTSTensor expandAs(const TTSTensor &other) const
  {
    return TTSTensor(t_.expand_as(other.t_));
  }

  TTSTensor broadcastTo(const int64_t *sizes, size_t ndims) const
  {
    at::IntArrayRef shape(sizes, ndims);
    return TTSTensor(at::broadcast_to(t_, shape));
  }

  // Masks: masked fill/select
  TTSTensor maskedFillScalar(const TTSTensor &mask, c10::Scalar value) const
  {
    // ✅ Correctly implement out-of-place behavior:
    // 1. Clone the original tensor.
    at::Tensor result = t_.clone();
    // 2. Apply the in-place operation (masked_fill_) to the clone.
    result.masked_fill_(mask.t_, value);
    // 3. Return the new, modified tensor.
    return TTSTensor(result);
  }

  TTSTensor maskedFillTensor(const TTSTensor &mask, const TTSTensor &value) const
  {
    // ✅ Apply the same clone-then-modify pattern here.
    at::Tensor result = t_.clone();
    result.masked_fill_(mask.t_, value.t_);
    return TTSTensor(result);
  }

  // masked_select is already out-of-place, so it's correct.
  TTSTensor maskedSelect(const TTSTensor &mask) const
  {
    return TTSTensor(at::masked_select(t_, mask.t_));
  }

  // Boolean reductions & utilities
  TTSTensor anyAll() const { return TTSTensor(at::any(t_)); }
  TTSTensor allAll() const { return TTSTensor(at::all(t_)); }

  TTSTensor anyDim(int64_t dim, bool keepdim) const
  {
    auto d = _canon_dim(t_, dim);
    std::vector<int64_t> dims{d};
    return TTSTensor(at::any(t_, dims, keepdim));
  }
  TTSTensor allDim(int64_t dim, bool keepdim) const
  {
    auto d = _canon_dim(t_, dim);
    std::vector<int64_t> dims{d};
    return TTSTensor(at::all(t_, dims, keepdim));
  }

  // View that creates sliding local blocks along a dimension.
  TTSTensor unfold(int64_t dim, int64_t size, int64_t step) const
  {
    auto d = _canon_dim(t_, dim);
    return TTSTensor(t_.unfold(d, size, step));
  }

  // Structural equality (shape, dtype, device, and elementwise exact equality).
  bool equal(const TTSTensor &other) const
  {
    return at::equal(t_, other.t_);
  }

  // Numeric closeness (for tests): allclose with rtol/atol/equal_nan.
  bool allclose(const TTSTensor &other, double rtol, double atol, bool equal_nan) const
  {
    return at::allclose(t_, other.t_, rtol, atol, equal_nan);
  }

  TTSTensor nonzero() const { return TTSTensor(at::nonzero(t_)); }

  // ---- Device queries and non-blocking toDevice
  static bool hasCUDA() { return at::hasCUDA(); }
  static bool hasMPS() { return at::hasMPS(); }

  // toDevice with non_blocking option
  TTSTensor toDeviceNB(c10::Device dev, bool non_blocking) const
  {
    // Keep dtype, allow non-blocking if backend supports it
    return TTSTensor(t_.to(dev, t_.scalar_type(), non_blocking, /*copy=*/true, c10::nullopt));
  }
};

// Helper function to create a c10::Device unambiguously
inline c10::Device make_device(c10::DeviceType type, int8_t index = -1)
{
  return c10::Device(type, index);
}

// The helper function
inline TTSTensor masked_fill_tensor_helper(const TTSTensor &self, const TTSTensor &mask, const TTSTensor &value)
{
  // ✅ Use the more robust 'at::where' function to achieve the same result.
  // The logic is: where(condition, value_if_true, value_if_false)
  return TTSTensor(at::where(mask.t_, value.t_, self.t_));
}