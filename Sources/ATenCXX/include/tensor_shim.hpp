#pragma once
#include <ATen/ATen.h>
#include <vector>
#include <utility>
#include <ATen/ops/where.h>
#include <ATen/ops/allclose.h>

#include <swift/bridging>

// Forward-declare the helper function so the class can see it
class TTSTensor;
inline TTSTensor masked_fill_tensor_helper(const TTSTensor &self, const TTSTensor &mask, const TTSTensor &value);

// A tiny, importer-friendly wrapper around at::Tensor.
class TTSTensor
{
  at::Tensor t_;

public:
  const at::Tensor &_t() const noexcept { return t_; }

  // ---- Factories: copy from host memory (safe) -------------------------------

  // ---------- Host array constructors (handy in doctests) ----------
  template <typename T>
  static TTSTensor fromHostArray(const T *data,
                                 size_t ndims,
                                 const int64_t *shape,
                                 c10::ScalarType dtype,
                                 c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    std::vector<int64_t> sizes(shape, shape + ndims);
    auto opts = at::TensorOptions().dtype(dtype).device(device);
    // from_blob does not own memory; clone() so the resulting tensor owns storage.
    at::Tensor t = at::from_blob(const_cast<T *>(data), sizes, opts).clone();
    return TTSTensor(std::move(t));
  }

  // Special-case for boolean data commonly represented as bytes in tests.
  // You can pass uint8_t* with dtype = c10::ScalarType::Bool and we will cast.
  static TTSTensor fromHostBytesAsBool(const uint8_t *data,
                                       size_t ndims,
                                       const int64_t *shape,
                                       c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    std::vector<int64_t> sizes(shape, shape + ndims);
    auto byteOpts = at::TensorOptions().dtype(c10::ScalarType::Byte).device(device);
    at::Tensor byteT = at::from_blob(const_cast<uint8_t *>(data), sizes, byteOpts).clone();
    at::Tensor boolT = byteT.to(c10::ScalarType::Bool);
    return TTSTensor(std::move(boolT));
  }

  // ---- Typed pointer overloads (common dtypes) ----
  static TTSTensor fromArray(const float *data, size_t ndim, const int64_t *sizes,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromHostArray<float>(data, ndim, sizes, c10::kFloat, device);
  }
  static TTSTensor fromArray(const double *data, size_t ndim, const int64_t *sizes,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromHostArray<double>(data, ndim, sizes, c10::kDouble, device);
  }
  static TTSTensor fromArray(const int64_t *data, size_t ndim, const int64_t *sizes,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromHostArray<int64_t>(data, ndim, sizes, c10::kLong, device);
  }
  static TTSTensor fromArray(const int32_t *data, size_t ndim, const int64_t *sizes,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromHostArray<int32_t>(data, ndim, sizes, c10::kInt, device);
  }
  static TTSTensor fromArray(const int16_t *data, size_t ndim, const int64_t *sizes,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromHostArray<int16_t>(data, ndim, sizes, c10::kShort, device);
  }
  static TTSTensor fromArray(const int8_t *data, size_t ndim, const int64_t *sizes,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromHostArray<int8_t>(data, ndim, sizes, c10::kChar, device);
  }
  static TTSTensor fromArray(const uint8_t *data, size_t ndim, const int64_t *sizes,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromHostArray<uint8_t>(data, ndim, sizes, c10::kByte, device);
  }

  // ---- Masks (prefer uint8_t 0/1 or bool*) ----
  static TTSTensor fromMask(const uint8_t *data, size_t ndim, const int64_t *sizes,
                            c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    auto options = at::TensorOptions().dtype(c10::kByte).device(device);
    at::Tensor t = at::from_blob(const_cast<uint8_t *>(data),
                                 at::IntArrayRef(sizes, ndim), options)
                       .clone();
    return TTSTensor(t.to(c10::kBool));
  }
  static TTSTensor fromMask(const bool *data, size_t ndim, const int64_t *sizes,
                            c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromHostArray<bool>(data, ndim, sizes, c10::kBool, device);
  }

  // ---- std::vector<T> convenience with shape checking ----
  template <typename T>
  static TTSTensor fromArray(const std::vector<T> &host,
                             const std::vector<int64_t> &shape,
                             c10::ScalarType dtype,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    size_t numel = 1;
    for (auto s : shape)
      numel *= static_cast<size_t>(s);
    TORCH_CHECK(numel == host.size(),
                "fromArray: host.size() (", host.size(),
                ") != product(shape) (", numel, ")");
    return fromHostArray<T>(host.data(), shape.size(), shape.data(), dtype, device);
  }

  // Typed vector overloads
  static TTSTensor fromArray(const std::vector<float> &v,
                             const std::vector<int64_t> &shape,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromArray<float>(v, shape, c10::kFloat, device);
  }
  static TTSTensor fromArray(const std::vector<double> &v,
                             const std::vector<int64_t> &shape,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromArray<double>(v, shape, c10::kDouble, device);
  }
  static TTSTensor fromArray(const std::vector<int64_t> &v,
                             const std::vector<int64_t> &shape,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromArray<int64_t>(v, shape, c10::kLong, device);
  }
  static TTSTensor fromArray(const std::vector<int32_t> &v,
                             const std::vector<int64_t> &shape,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromArray<int32_t>(v, shape, c10::kInt, device);
  }
  static TTSTensor fromArray(const std::vector<uint8_t> &v,
                             const std::vector<int64_t> &shape,
                             c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    return fromArray<uint8_t>(v, shape, c10::kByte, device);
  }

  // Vector<bool> has no data(); accept uint8_t 0/1 instead.
  static TTSTensor fromMask(const std::vector<uint8_t> &v,
                            const std::vector<int64_t> &shape,
                            c10::Device device = c10::Device(c10::DeviceType::CPU))
  {
    size_t numel = 1;
    for (auto s : shape)
      numel *= static_cast<size_t>(s);
    TORCH_CHECK(numel == v.size(),
                "fromMask: host.size() (", v.size(),
                ") != product(shape) (", numel, ")");
    return fromMask(v.data(), shape.size(), shape.data(), device);
  }

  // ---------- Broadcast/expand views ----------
  TTSTensor broadcast_to(const int64_t *sizes, size_t ndims) const
  {
    std::vector<int64_t> sz(sizes, sizes + ndims);
    return TTSTensor(t_.expand(sz));
  }

  TTSTensor broadcast_to(const std::vector<int64_t> &sizes) const
  {
    return TTSTensor(t_.expand(sizes));
  }

  TTSTensor expand_as(const TTSTensor &other) const
  {
    return TTSTensor(t_.expand_as(other.t_));
  }

  // ---------- where(cond, src, self) convenience ----------
  static TTSTensor where3(const TTSTensor &cond, const TTSTensor &a, const TTSTensor &b)
  {
    return TTSTensor(at::where(cond.t_, a.t_, b.t_));
  }

  // As a method on "self": self.where(mask, src) == where(mask, src, self)
  TTSTensor where(const TTSTensor &cond, const TTSTensor &src) const
  {
    return TTSTensor(at::where(cond.t_, src.t_, t_));
  }

  // Scalar extractor for rank-0 tensors
  double toDouble() const
  {
    return t_.item<double>();
  }

  // In struct TTSTensor (public section), add:
  double itemDouble() const { return t_.item<double>(); }
  int64_t itemInt64() const { return t_.item<int64_t>(); }
  bool itemBool() const { return t_.item<bool>(); }

  // masked_fill(self, mask, scalar)
  TTSTensor maskedFill(const TTSTensor &mask, c10::Scalar value) const
  {
    return TTSTensor(t_.masked_fill(mask.t_, value));
  }

  // masked_scatter(self, mask, source)
  TTSTensor maskedScatter(const TTSTensor &mask, const TTSTensor &source) const
  {
    return TTSTensor(t_.masked_scatter(mask.t_, source.t_));
  }

  // Extract a single boolean value from a rank-0 Bool tensor
  bool toBool() const
  {
    return t_.item<bool>();
  }

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
  // In TTSTensor class inside tensor_shim.hpp

  TTSTensor indexSelect(int64_t dim, const int64_t *idx, size_t count) const
  {
    auto d = _canon_dim(t_, dim);
    // 💡 FIX: Get the size of the dimension being indexed.
    auto dim_size = t_.size(d);

    // 💡 FIX: Create a temporary vector and fill it with normalized indices.
    std::vector<int64_t> normalized_idx(count);
    for (size_t i = 0; i < count; ++i)
    {
      normalized_idx[i] = idx[i] < 0 ? idx[i] + dim_size : idx[i];
    }

    // Create the index tensor on the CPU from the *normalized* data.
    auto opts = at::TensorOptions().dtype(c10::ScalarType::Long).device(c10::DeviceType::CPU);
    at::Tensor i_tensor = at::empty({static_cast<long>(count)}, opts);
    std::memcpy(i_tensor.data_ptr<int64_t>(), normalized_idx.data(), count * sizeof(int64_t));

    // Call the underlying ATen function, ensuring the index tensor is on the same device.
    return TTSTensor(t_.index_select(d, i_tensor.to(t_.device())));
  }

  TTSTensor masked_scatter(const TTSTensor &mask, const TTSTensor &source) const
  {
    // First, ensure the mask is boolean.
    at::Tensor bool_mask = mask.t_.is_same(t_) ? mask.t_.to(at::kBool) : mask.t_;
    if (bool_mask.scalar_type() != at::kBool)
    {
      bool_mask = bool_mask.to(at::kBool);
    }

    // Count the number of true elements in the mask.
    const int64_t true_count = at::sum(bool_mask).item<int64_t>();
    const int64_t source_numel = source.numel();

    // Throw if the counts don't match exactly.
    TORCH_CHECK(source_numel == true_count,
                "masked_scatter: number of elements in source (", source_numel,
                ") must match the number of true elements in mask (", true_count, ")");

    // Call the underlying ATen function. Note the use of the validated bool_mask.
    return TTSTensor(t_.masked_scatter(bool_mask, source.t_));
  }

  // Adds all values from the 'source' tensor into self at the indices
  // ... rest of the class definition ...

  // Adds all values from the 'source' tensor into self at the indices
  // specified in the 'index' tensor along a given 'dim'.
  TTSTensor scatterAdd(int64_t dim, const TTSTensor &index, const TTSTensor &source) const
  {
    auto d = _canon_dim(t_, dim);
    at::Tensor result = t_.clone();

    // ✅ Convert index tensor to Int64 (Long) before the operation
    at::Tensor index_long = index.t_.to(c10::ScalarType::Long);

    result.scatter_add_(d, index_long, source.t_);
    return TTSTensor(result);
  }

  // Places values from the 'values' tensor into self at locations specified
  // by 'indices'.
  TTSTensor indexPut(const TTSTensor *indices, size_t num_indices,
                     const TTSTensor &values, bool accumulate = false) const
  {
    // 1. Build a vector of the required type: std::optional<at::Tensor>
    std::vector<std::optional<at::Tensor>> optional_indices;
    optional_indices.reserve(num_indices);
    for (size_t i = 0; i < num_indices; ++i)
    {
      if (indices[i].defined())
      {
        // If the tensor is defined, add it to the list.
        optional_indices.push_back(indices[i].t_);
      }
      else
      {
        // If it's undefined (for slicing like [:]), add a nullopt.
        optional_indices.push_back(c10::nullopt);
      }
    }

    // 2. Convert the std::vector into the c10::List that the function expects.
    c10::List<std::optional<at::Tensor>> index_list(optional_indices);

    // 3. Clone self to maintain out-of-place semantics
    at::Tensor result = t_.clone();

    // 4. Call the in-place ATen function with the correctly typed list
    result.index_put_(index_list, values.t_, accumulate);

    // 5. Return the new tensor
    return TTSTensor(result);
  }

  TTSTensor indexAdd(int64_t dim, const TTSTensor &index, const TTSTensor &source, c10::Scalar alpha) const
  {
    auto d = _canon_dim(t_, dim);
    return TTSTensor(at::index_add(t_, d, index.t_, source.t_, alpha));
  }

  TTSTensor indexCopy(int64_t dim, const TTSTensor &index, const TTSTensor &source) const
  {
    auto d = _canon_dim(t_, dim);
    return TTSTensor(at::index_copy(t_, d, index.t_, source.t_));
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
  TTSTensor sin_() const { return TTSTensor(at::sin(t_)); }
  TTSTensor cos_() const { return TTSTensor(at::cos(t_)); }
  TTSTensor tan_() const { return TTSTensor(at::tan(t_)); }
  TTSTensor asin_() const { return TTSTensor(at::asin(t_)); }
  TTSTensor acos_() const { return TTSTensor(at::acos(t_)); }
  TTSTensor atan_() const { return TTSTensor(at::atan(t_)); }
  TTSTensor sinh_() const { return TTSTensor(at::sinh(t_)); }
  TTSTensor cosh_() const { return TTSTensor(at::cosh(t_)); }
  TTSTensor tanh_() const { return TTSTensor(at::tanh(t_)); }
  TTSTensor asinh_() const { return TTSTensor(at::asinh(t_)); }
  TTSTensor acosh_() const { return TTSTensor(at::acosh(t_)); }
  TTSTensor atanh_() const { return TTSTensor(at::atanh(t_)); }
  TTSTensor erf_() const { return TTSTensor(at::erf(t_)); }
  TTSTensor erfc_() const { return TTSTensor(at::erfc(t_)); }
  TTSTensor sigmoid_() const { return TTSTensor(at::sigmoid(t_)); }

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

  // ---- Reductions that also return indices (NEW)
  // This is the targeted patch to fix the crash and improve performance.

  // These functions call the underlying operation ONCE and return a std::pair,
  // which Swift's C++ Interop will bridge to a tuple.
  std::pair<TTSTensor, TTSTensor> minDimWithIndices(int64_t dim, bool keepdim) const SWIFT_RETURNS_INDEPENDENT_VALUE
  {
    auto d = _canon_dim(t_, dim);
    auto result_tuple = at::min(t_, d, keepdim);
    return {TTSTensor(std::get<0>(result_tuple)), TTSTensor(std::get<1>(result_tuple))};
  }

  std::pair<TTSTensor, TTSTensor> maxDimWithIndices(int64_t dim, bool keepdim) const SWIFT_RETURNS_INDEPENDENT_VALUE
  {
    auto d = _canon_dim(t_, dim);
    auto result_tuple = at::max(t_, d, keepdim);
    return {TTSTensor(std::get<0>(result_tuple)), TTSTensor(std::get<1>(result_tuple))};
  }

  std::pair<TTSTensor, TTSTensor> topkWithIndices(int64_t k, int64_t dim, bool largest, bool sorted) const SWIFT_RETURNS_INDEPENDENT_VALUE
  {
    auto d = _canon_dim(t_, dim);
    auto result_tuple = at::topk(t_, k, d, largest, sorted);
    return {TTSTensor(std::get<0>(result_tuple)), TTSTensor(std::get<1>(result_tuple))};
  }

  std::pair<TTSTensor, TTSTensor> sortDimWithIndices(int64_t dim, bool descending) const SWIFT_RETURNS_INDEPENDENT_VALUE
  {
    auto d = _canon_dim(t_, dim);
    auto result_tuple = at::sort(t_, d, descending);
    return {TTSTensor(std::get<0>(result_tuple)), TTSTensor(std::get<1>(result_tuple))};
  }

  // Argmin/Argmax are still needed for cases where only indices are required.
  TTSTensor argminDim(int64_t dim, bool keepdim = false) const
  {
    auto d = _canon_dim(t_, dim);
    return TTSTensor(std::get<1>(at::min(t_, d, keepdim)));
  }
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
