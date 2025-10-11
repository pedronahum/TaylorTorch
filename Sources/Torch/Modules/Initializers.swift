// Sources/Torch/Modules/Initializers.swift
import Foundation

/// Xavier/Glorot uniform for linear layers: U[-a, a], a = sqrt(6/(fanIn + fanOut))
public func glorotUniform(fanIn: Int, fanOut: Int, dtype: DType = .float32, device: Device = .cpu)
  -> Tensor
{
  let a = Foundation.sqrt(6.0 / Double(fanIn + fanOut))
  // Use `Tensor.uniform` here, not `Tensor.rand`
  return Tensor.uniform(low: -a, high: a, shape: [fanOut, fanIn], dtype: dtype, device: device)
}

public func zeros(_ n: Int, dtype: DType = .float32, device: Device = .cpu) -> Tensor {
  Tensor.zeros(shape: [n], dtype: dtype, device: device)
}
