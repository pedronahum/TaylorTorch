import Foundation
// Tests/ActivationModulesTests.swift
import Testing
import _Differentiation

@testable import Torch

// MARK: - 1) Layer forward = reference op

@Test("ReLU layer equals x.relu() (forward + input grads)")
func relu_layer_parity() throws {
  let x = Tensor(array: [-2.0, -0.5, 1.5, 3.0], shape: [4], dtype: .float64)

  // Forward parity
  let yLayer = ReLU()(x)
  let yRef = x.relu()
  #expect(yLayer.isClose(to: yRef, rtol: 0, atol: 0, equalNan: false))

  // Gradient parity (w.r.t. input) under L = sum(y)
  let (_, pbLayer) = valueWithPullback(at: x) { ReLU()($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))
  let (_, pbRef) = valueWithPullback(at: x) { $0.relu().sum() }
  let gRef = pbRef(Tensor(1.0, dtype: .float64))
  #expect(gLayer.isClose(to: gRef, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// MARK: - 2) Tanh & Sigmoid

@Test("Tanh layer equals tanh(x) (forward + grads)")
func tanh_layer_parity() throws {
  let x = Tensor(array: [-1.2, 0.3, 2.2], shape: [3], dtype: .float64)
  let (_, pbLayer) = valueWithPullback(at: x) { Tanh()($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))
  let (_, pbRef) = valueWithPullback(at: x) { $0.tanh().sum() }
  let gRef = pbRef(Tensor(1.0, dtype: .float64))
  #expect(gLayer.isClose(to: gRef, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("Sigmoid layer equals sigmoid(x) (forward + grads)")
func sigmoid_layer_parity() throws {
  let x = Tensor(array: [-2.0, 0.0, 1.0], shape: [3], dtype: .float64)
  let (_, pbLayer) = valueWithPullback(at: x) { Sigmoid()($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))
  let (_, pbRef) = valueWithPullback(at: x) { $0.sigmoid().sum() }
  let gRef = pbRef(Tensor(1.0, dtype: .float64))
  #expect(gLayer.isClose(to: gRef, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// MARK: - 3) GELU (exact and approximate)

@Test("GELU exact matches erf-based formula (forward + grads)")
func gelu_exact_parity() throws {
  let x = Tensor(array: [-1.0, -0.1, 0.2, 1.5], shape: [4], dtype: .float64)

  let layer = GELU(approximate: false)
  let (_, pbLayer) = valueWithPullback(at: x) { layer($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))

  let dtype = withoutDerivative(at: x.dtype ?? .float32)
  let device = withoutDerivative(at: x.device)
  let half = Tensor(0.5, dtype: dtype, device: device)
  let invSqrt2 = Tensor(0.7071067811865476, dtype: dtype, device: device)
  let sqrtTwoPiInv = Tensor(0.7978845608028654, dtype: dtype, device: device)  // √(2/π)
  let base = x.multiplying(invSqrt2).erf().adding(1)
  let expTerm = x.multiplying(x).dividing(-2).exp()
  let expected = half.multiplying(base).adding(x.multiplying(expTerm).multiplying(sqrtTwoPiInv))

  #expect(gLayer.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("GELU approximate matches tanh-based formula (forward + grads)")
func gelu_approx_parity() throws {
  let x = Tensor(array: [-1.0, -0.1, 0.2, 1.5], shape: [4], dtype: .float64)

  let layer = GELU(approximate: true)
  let (_, pbLayer) = valueWithPullback(at: x) { layer($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))

  let kappa = Tensor((2.0 / Double.pi).squareRoot())
  let (_, pbRef) = valueWithPullback(at: x) { x in
    let inner = x.adding(0.044715 * x.multiplying(x).multiplying(x))
    return (0.5 * x).multiplying(1.0 + (kappa.multiplying(inner)).tanh()).sum()
  }
  let gRef = pbRef(Tensor(1.0, dtype: .float64))
  #expect(gLayer.isClose(to: gRef, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// MARK: - 4) SiLU / Swish

@Test("SiLU equals x * sigmoid(x) (forward + grads)")
func silu_parity() throws {
  let x = Tensor(array: [-1.5, -0.25, 0.0, 2.0], shape: [4], dtype: .float64)

  let (_, pbLayer) = valueWithPullback(at: x) { SiLU()($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))

  let (_, pbRef) = valueWithPullback(at: x) { x in x.multiplying(x.sigmoid()).sum() }
  let gRef = pbRef(Tensor(1.0, dtype: .float64))

  #expect(gLayer.isClose(to: gRef, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// MARK: - 5) Softplus layer defers to global helper

@Test("Softplus layer equals softplus(x) helper (forward + grads)")
func softplus_layer_parity() throws {
  let x = Tensor(array: [-1.0, 0.0, 2.0], shape: [3], dtype: .float64)

  let (_, pbLayer) = valueWithPullback(at: x) { Softplus()($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))

  let (_, pbRef) = valueWithPullback(at: x) { softplus($0).sum() }
  let gRef = pbRef(Tensor(1.0, dtype: .float64))

  #expect(gLayer.isClose(to: gRef, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("ELU layer matches functional helper (forward + grads)")
func elu_layer_parity() throws {
  let x = Tensor(array: [-2.0, -0.1, 0.0, 0.1, 2.0], shape: [5], dtype: .float64)
  let alpha: Float = 1.1

  let layer = ELU(alpha: alpha)
  let (_, pbLayer) = valueWithPullback(at: x) { layer($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))

  let dtype = withoutDerivative(at: x.dtype ?? .float32)
  let device = withoutDerivative(at: x.device)
  let alphaT = Tensor(alpha, dtype: dtype, device: device)
  let zero = Tensor(0, dtype: dtype, device: device)
  let posGrad = Tensor.ones(shape: x.shape, dtype: dtype, device: device)
  let negGrad = alphaT.multiplying(x.minimum(zero).exp())
  let expected = TorchWhere.select(condition: x.gt(0), posGrad, negGrad)

  #expect(gLayer.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// MARK: - 6) LeakyReLU

@Test("LeakyReLU(alpha) matches relu(x) - alpha * relu(-x) (forward + grads)")
func leakyrelu_parity() throws {
  let x = Tensor(array: [-2.0, -0.1, 0.5], shape: [3], dtype: .float64)
  let alpha: Float = 0.2

  let (_, pbLayer) = valueWithPullback(at: x) { LeakyReLU(negativeSlope: alpha)($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))

  let (_, pbRef) = valueWithPullback(at: x) { t in
    t.relu().adding(t.negated().relu().multiplying(-alpha)).sum()
  }
  let gRef = pbRef(Tensor(1.0, dtype: .float64))
  #expect(gLayer.isClose(to: gRef, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// MARK: - 7) Integration with Sequential builder

@Test("Sequential { Linear; ReLU; Linear } equals manual composition (fwd + grads)")
func builder_sequential_with_activation() throws {
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,
      1.5, 0.0, -0.5,
    ], shape: [2, 3], dtype: .float64)

  var l1 = Linear(inputSize: 3, outputSize: 2, dtype: .float64)
  var l2 = Linear(inputSize: 2, outputSize: 2, dtype: .float64)
  l1.weight = Tensor.arange(Double(0), to: Double(3 * 2), step: 1, dtype: .float64)
    .reshaped([3, 2])
  l1.bias = Tensor(array: [0.1, -0.2], shape: [2], dtype: .float64)
  l2.weight = Tensor.arange(Double(0), to: Double(2 * 2), step: 1, dtype: .float64)
    .reshaped([2, 2])
  l2.bias = Tensor(array: [0.0, 0.3], shape: [2], dtype: .float64)

  let model = Sequential {
    l1
    ReLU()
    l2
  }

  // Manual baseline using the same parameters.
  let yManual = l2(l1(x).relu())
  let yModel = model(x)
  #expect(yModel.isClose(to: yManual, rtol: 1e-9, atol: 1e-9, equalNan: false))

  // Gradient equality for L = sum(y)
  let (_, pbModel) = valueWithPullback(at: model) { $0(x).sum() }
  let gModel = pbModel(Tensor(1.0, dtype: .float64))

  let (_, pbManual) = valueWithPullback(at: l1, l2) { first, second in
    second(first(x).relu()).sum()
  }
  let (gL1Manual, gL2Manual) = pbManual(Tensor(1.0, dtype: .float64))

  // Align gradients (builder nests layers as Chain<Chain<Linear, ReLU>, Linear>)
  let chainGrad = gModel.body
  let gL1Model = chainGrad.first.first
  let gL2Model = chainGrad.second

  #expect(gL1Model.weight.isClose(to: gL1Manual.weight, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(gL1Model.bias.isClose(to: gL1Manual.bias, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(gL2Model.weight.isClose(to: gL2Manual.weight, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(gL2Model.bias.isClose(to: gL2Manual.bias, rtol: 1e-9, atol: 1e-9, equalNan: false))
}

// MARK: - 8) Parameter traversal (empty)

@Test("Activation layers expose no trainable parameters")
func activation_layers_have_empty_parameter_lists() throws {
  #expect(ReLU().recursivelyAllWritableKeyPaths(to: Tensor.self).isEmpty)
  #expect(Tanh().recursivelyAllWritableKeyPaths(to: Tensor.self).isEmpty)
  #expect(Sigmoid().recursivelyAllWritableKeyPaths(to: Tensor.self).isEmpty)
  #expect(GELU().recursivelyAllWritableKeyPaths(to: Tensor.self).isEmpty)
  #expect(SiLU().recursivelyAllWritableKeyPaths(to: Tensor.self).isEmpty)
  #expect(Softplus().recursivelyAllWritableKeyPaths(to: Tensor.self).isEmpty)
  #expect(LeakyReLU().recursivelyAllWritableKeyPaths(to: Tensor.self).isEmpty)
}
