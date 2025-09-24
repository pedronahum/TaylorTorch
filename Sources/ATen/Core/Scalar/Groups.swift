import ATenCXX

// Core stable categories:

/// Marker protocol for index dtypes (Int32/Int64) accepted by slicing and indexing APIs.
public protocol TorchIndex: TorchTensorScalar {}

/// Marker protocol for real-valued floating dtypes.
public protocol TorchFloating: TorchTensorScalar {}

/// Marker protocol for all fixed-width integer dtypes, signed or unsigned.
public protocol TorchInteger: TorchTensorScalar {}

/// Marker protocol for the signed integer subset of `TorchInteger`.
public protocol TorchSignedInteger: TorchInteger {}

/// Marker protocol for the unsigned integer subset of `TorchInteger`.
public protocol TorchUnsignedInteger: TorchInteger {}

/// Marker protocol for complex-valued dtypes.
public protocol TorchComplex: TorchTensorScalar {}

/// Marker protocol for the quantized integer dtypes used by ATen.
public protocol TorchQuantized: TorchTensorScalar {}

/// Marker protocol for real-number dtypes (integers and floats, excluding bool/string/quantized/complex).
public protocol TorchRealNumber: TorchTensorScalar {}

/// Marker protocol for dtypes that support arithmetic operations in ATen (real numbers plus complex).
public protocol TorchArithmetic: TorchTensorScalar, Numeric {
  /// Provides the Torch scalar representation used when marshalling to C++.
  var _cxxScalar: c10.Scalar { get }
}

/// Marker protocol for Boolean scalar values.
public protocol TorchBoolScalar: TorchTensorScalar {}

/// Marker protocol for string scalar values used in ragged/text APIs.
public protocol TorchStringScalar: TorchTensorScalar {}

/// Marker protocol for variant scalars used in composite backends.
public protocol TorchVariantScalar: TorchTensorScalar {}
