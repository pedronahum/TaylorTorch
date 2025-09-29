// Sources/TensorTypes/GroupUnions.swift

/// Marker protocol for operators that accept only `Float` and `Double` inputs.
public protocol TorchFloatOrDouble: TorchTensorScalar {}

/// Marker protocol for operators that accept any floating or complex dtype.
public protocol TorchFloatingOrComplex: TorchTensorScalar {}

/// Marker protocol for operators that accept `Float` or `Float16` inputs.
public protocol TorchFloatOrHalf: TorchTensorScalar {}

/// Marker protocol for operators that accept `BFloat16`, `Float16`, or `Float`.
public protocol TorchFloatingNoDouble: TorchTensorScalar {}

/// Marker protocol for operators that accept `Float16`, `Float`, or `Double`.
public protocol TorchFloatHalfOrDouble: TorchTensorScalar {}

/// Marker protocol for operators that accept floating dtypes and the index types (`Int32`, `Int64`).
public protocol TorchFloatingOrIndex: TorchTensorScalar {}

/// Marker protocol for operators that accept index dtypes used in slicing (`Int16`, `Int32`, `Int64`).
public protocol TorchSliceIndex: TorchTensorScalar {}

/// Marker protocol for operators that accept any numeric dtype including quantized variants.
public protocol TorchAnyNumericOrQuantized: TorchTensorScalar {}

/// Marker protocol for operators that accept `Int64` in addition to string-like scalars.
public protocol TorchInt64OrString: TorchTensorScalar {}

/// Marker protocol for operators that accept `Float` or `Int64` scalars as well as string-like scalars.
public protocol TorchFloatOrInt64OrString: TorchTensorScalar {}
