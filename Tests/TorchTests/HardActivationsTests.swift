// Tests/TorchTests/HardActivationsTests.swift
import Testing
import _Differentiation

@testable import Torch

// --- HardTanh ---------------------------------------------------------------

@Test("HardTanh: forward is clamp; grad is 1 inside, 0 outside")
func hardtanh_forward_and_grad() throws {
  let x = Tensor(array: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], shape: [7], dtype: .float64)

  // Functional
  do {
    let y = Activations.hardtanh(x, minVal: -1.0, maxVal: 1.0)
    let expected = x.clamp(min: -1.0, max: 1.0)
    #expect(y.isClose(to: expected, rtol: 0, atol: 0, equalNan: false))

    let (_, pb) = valueWithPullback(at: x) {
      HardTanh(minVal: -1.0, maxVal: 1.0)($0).sum()
    }
    let g = pb(Tensor(1.0, dtype: .float64))

    // Analytic: 1 for (-1,1), 0 at/beyond clamps
    let inside = x.gt(-1.0).to(dtype: .float64).multiplying(x.lt(1.0).to(dtype: .float64))
    #expect(g.isClose(to: inside, rtol: 0, atol: 0, equalNan: false))
  }

  // Layer wrapper parity
  do {
    let layer = HardTanh(minVal: -1.0, maxVal: 1.0)
    #expect(layer(x).isClose(to: Activations.hardtanh(x), rtol: 0, atol: 0, equalNan: false))
  }
}

// --- HardSigmoid ------------------------------------------------------------

@Test("HardSigmoid: forward matches clip(x+3,0,6)/6; grad is 1/6 in (-3,3)")
func hardsigmoid_forward_and_grad() throws {
  let x = Tensor(array: [-4.0, -3.0, -1.5, 0.0, 2.0, 3.0, 4.0], shape: [7], dtype: .float64)
  let y = Activations.hardsigmoid(x)
  let expected = x.adding(3.0).clamp(min: 0.0, max: 6.0).dividing(6.0)
  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))

  let (_, pb) = valueWithPullback(at: x) { HardSigmoid()($0).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))

  // Analytic: 0 outside, 1/6 inside the linear window (-3, 3)
  let inside = x.gt(-3.0).to(dtype: .float64).multiplying(x.lt(3.0).to(dtype: .float64))
  let expectedGrad = inside.multiplying(1.0 / 6.0)
  #expect(g.isClose(to: expectedGrad, rtol: 1e-12, atol: 1e-12, equalNan: false))

  // Layer wrapper parity
  let layer = HardSigmoid()
  #expect(layer(x).isClose(to: y, rtol: 0, atol: 0, equalNan: false))
}

// --- HardSwish --------------------------------------------------------------

@Test("HardSwish: forward x*hardsigmoid(x); piece-wise derivative")
func hardswish_forward_and_grad() throws {
  let x = Tensor(array: [-4.0, -2.0, 0.0, 2.0, 4.0], shape: [5], dtype: .float64)
  let y = Activations.hardswish(x)
  let expected = x.multiplying(Activations.hardsigmoid(x))
  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))

  let (_, pb) = valueWithPullback(at: x) { HardSwish()($0).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))

  // d/dx:
  //  x <= -3 -> 0
  //  -3 < x < 3 -> (2x + 3)/6
  //  x >= 3 -> 1
  let zero = Tensor(0.0, dtype: .float64)
  let one = Tensor(1.0, dtype: .float64)

  let left = x.le(-3.0).to(dtype: .float64)
  let mid = x.gt(-3.0).to(dtype: .float64).multiplying(x.lt(3.0).to(dtype: .float64))
  let right = x.ge(3.0).to(dtype: .float64)

  let midGrad = x.multiplying(2.0).adding(3.0).dividing(6.0)  // (2x + 3)/6
  let expectedGrad = left.multiplying(zero)
    .adding(mid.multiplying(midGrad))
    .adding(right.multiplying(one))

  #expect(g.isClose(to: expectedGrad, rtol: 1e-12, atol: 1e-12, equalNan: false))

  // Layer wrapper parity
  let layer = HardSwish()
  #expect(layer(x).isClose(to: y, rtol: 0, atol: 0, equalNan: false))
}

// --- ELU --------------------------------------------------------------------

@Test("ELU: forward negative branch alpha*(exp(x)-1); grad is piece‑wise")
func elu_forward_and_grad() throws {
  let x = Tensor(array: [-2.0, -0.1, 0.0, 0.1, 2.0], shape: [5], dtype: .float64)
  let alpha = 1.1

  let y = Activations.elu(x, alpha: alpha)
  // Manual forward
  let a = Tensor(alpha, dtype: .float64)
  let pos = x.gt(0.0).to(dtype: .float64).multiplying(x)
  let neg = x.le(0.0).to(dtype: .float64).multiplying(a.multiplying(x.exp().subtracting(1.0)))
  let expected = pos.adding(neg)
  #expect(y.isClose(to: expected, rtol: 1e-12, atol: 1e-12, equalNan: false))

  let (_, pb) = valueWithPullback(at: x) { ELU(alpha: alpha)($0).sum() }
  let g = pb(Tensor(1.0, dtype: .float64))

  // Grad: 1 for x>0;  α*exp(x) for x<=0.  (Masking behavior is tested in your suite.)
  // See: clamp/where/comparison pullbacks tests.
  let posMask = x.gt(0.0).to(dtype: .float64)
  let negMask = Tensor(1.0, dtype: .float64).subtracting(posMask)
  let expectedGrad = posMask.adding(negMask.multiplying(a.multiplying(x.exp())))
  #expect(g.isClose(to: expectedGrad, rtol: 1e-12, atol: 1e-12, equalNan: false))

  // Layer wrapper parity
  let layer = ELU(alpha: alpha)
  #expect(layer(x).isClose(to: y, rtol: 0, atol: 0, equalNan: false))
}
