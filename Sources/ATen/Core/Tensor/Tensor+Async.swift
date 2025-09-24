import ATenCXX
import Foundation

/// Errors surfaced by asynchronous tensor utilities.
public enum TensorError: Error, Sendable {
  /// Indicates that the requested `device` is not currently available on the host.
  case deviceUnavailable(Device)
}

public extension Tensor {
  /// Returns `true` when the runtime can materialize tensors on the given `device`.
  static func isAvailable(_ device: Device) -> Bool {
    switch device {
    case .cpu: return true
    case .cuda: return TTSTensor.hasCUDA()
    case .mps:  return TTSTensor.hasMPS()
    }
  }

  /// Performs a device-to-device copy on a background queue, optionally using a
  /// non-blocking transfer, and surfaces an error when the destination `device`
  /// is not available.
  func moved(to device: Device, nonBlocking: Bool = true) async throws -> Tensor {
    guard Tensor.isAvailable(device) else { throw TensorError.deviceUnavailable(device) }
    return await withCheckedContinuation { cont in
      DispatchQueue.global().async {
        let out = Tensor(_impl.toDeviceNB(device._c10, nonBlocking))
        cont.resume(returning: out)
      }
    }
  }
}
