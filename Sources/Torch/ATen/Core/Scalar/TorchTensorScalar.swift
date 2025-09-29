import ATenCXX

/// Base scalar hook used by all marker protocols and tensors.
/// Every element type (Float, Int32, Complex, Quantized, TFString, …)
/// must provide its backing `DType`.
public protocol TorchTensorScalar: Sendable {
  /// Torch dtype that identifies how values of the conforming type are materialized in ATen.
  static var torchDType: DType { get }
}

