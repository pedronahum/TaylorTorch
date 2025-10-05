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

  // Manual reference (independent expression)
  let invSqrt2 = 1.0 / Foundation.sqrt(2.0)
  let (_, pbRef) = valueWithPullback(at: x) { x in
    (0.5 * x).multiplying(1.0 + (x.multiplying(invSqrt2)).erf()).sum()
  }
  let gRef = pbRef(Tensor(1.0, dtype: .float64))

  #expect(gLayer.isClose(to: gRef, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

@Test("GELU approximate matches tanh-based formula (forward + grads)")
func gelu_approx_parity() throws {
  let x = Tensor(array: [-1.0, -0.1, 0.2, 1.5], shape: [4], dtype: .float64)

  let layer = GELU(approximate: true)
  let (_, pbLayer) = valueWithPullback(at: x) { layer($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))

  let kappa = Tensor(Foundation.sqrt(2.0 / Double.pi))
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

// MARK: - 6) LeakyReLU

@Test("LeakyReLU(alpha) matches relu(x) - alpha * relu(-x) (forward + grads)")
func leakyrelu_parity() throws {
  let x = Tensor(array: [-2.0, -0.1, 0.5], shape: [3], dtype: .float64)
  let alpha = 0.2

  let (_, pbLayer) = valueWithPullback(at: x) { LeakyReLU(alpha: alpha)($0).sum() }
  let gLayer = pbLayer(Tensor(1.0, dtype: .float64))

  let (_, pbRef) = valueWithPullback(at: x) { t in
    t.relu().adding(t.negated().relu().multiplying(-alpha)).sum()
  }
  let gRef = pbRef(Tensor(1.0, dtype: .float64))
  #expect(gLayer.isClose(to: gRef, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// MARK: - 7) Integration with builder & Sequential

@Test("Builder: SequentialBlock { Linear; ReLU; Linear } equals manual composition (fwd + grads)")
func builder_sequential_with_activation() throws {
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,
      1.5, 0.0, -0.5,
    ], shape: [2, 3], dtype: .float64)

  // Use your Linear.glorot convenience for realistic params.
  let l1 = Linear.glorot(inFeatures: 3, outFeatures: 2, dtype: .float64)
  let l2 = Linear.glorot(inFeatures: 2, outFeatures: 2, dtype: .float64)

  let typed = Sequential(l1, l2)
  let block = SequentialBlock {
    l1
    ReLU()
    l2
  }

  // Forward equality
  let yTyped = l2(l1(x).relu())
  let yBlock = block(x)
  #expect(yBlock.isClose(to: yTyped, rtol: 1e-9, atol: 1e-9, equalNan: false))

  // Gradient equality for L = sum(y)
  let (_, pbBlock) = valueWithPullback(at: block) { $0(x).sum() }
  let gBlock = pbBlock(Tensor(1.0, dtype: .float64))

  // Compare to manual chain via the same params (no extra activation params present).
  let (_, pbManual) = valueWithPullback(at: typed) { m in m.l2(m.l1(x).relu()).sum() }
  let gManual = pbManual(Tensor(1.0, dtype: .float64))

  // Structure: block.body.(l1, ReLU, l2). Only l1/l2 have parameters.
  #expect(
    gBlock.body.l1.l1.weight.isClose(to: gManual.l1.weight, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(gBlock.body.l1.l1.bias.isClose(to: gManual.l1.bias, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(
    gBlock.body.l2.weight.isClose(to: gManual.l2.weight, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(gBlock.body.l2.bias.isClose(to: gManual.l2.bias, rtol: 1e-9, atol: 1e-9, equalNan: false))
}

// MARK: - 8) Parameter traversal (empty)

@Test("Activation layers expose no trainable parameters")
func activation_layers_have_empty_parameter_lists() throws {
  #expect(ReLU.parameterKeyPaths.isEmpty)
  #expect(Tanh.parameterKeyPaths.isEmpty)
  #expect(Sigmoid.parameterKeyPaths.isEmpty)
  #expect(GELU.parameterKeyPaths.isEmpty)
  #expect(SiLU.parameterKeyPaths.isEmpty)
  #expect(Softplus.parameterKeyPaths.isEmpty)
  #expect(LeakyReLU.parameterKeyPaths.isEmpty)
}
