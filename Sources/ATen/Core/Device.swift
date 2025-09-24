import ATenCXX

/// Identifies the execution device for tensor storage and computation.
/// Mirrors the subset of PyTorch devices currently supported by the Swift bindings.
public enum Device: Sendable, Hashable, Codable {
  /// Host CPU computation.
  case cpu
  /// NVIDIA CUDA device with the provided device index.
  case cuda(Int8)
  /// Apple Metal Performance Shaders device.
  case mps
}

extension Device {
  /// Converts the Swift `Device` into the corresponding `c10::Device` instance.
  var _c10: c10.Device {
    switch self {
    case .cpu:
      // ✅ Call the unambiguous C++ helper function
      return make_device(c10.DeviceType.CPU)
    case .cuda(let idx):
      return make_device(c10.DeviceType.CUDA, idx)
    case .mps:
      return make_device(c10.DeviceType.MPS)
    }
  }
}
