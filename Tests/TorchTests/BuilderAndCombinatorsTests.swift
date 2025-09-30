import Testing
import _Differentiation

@testable import Torch

// MARK: - 1) Builder vs. manual Sequential

@Test("SequentialBlock { ... } matches Sequential<Linear,Linear> (forward + grads)")
func builder_matches_typed_sequential_forward_and_grads() throws {
  // Input: [batch, in]
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,
      1.5, 0.0, -0.5,
    ], shape: [2, 3])

  // Layer 1 params: W1 [out1,in], b1 [out1]
  let W1 = Tensor(
    array: [
      1.0, 0.0, -1.0,
      0.5, 2.0, 1.0,
    ], shape: [2, 3])
  let b1 = Tensor(array: [0.1, -0.2], shape: [2])

  // Layer 2 params: W2 [out2,out1], b2 [out2]
  let W2 = Tensor(
    array: [
      2.0, -1.0,
      1.5, 0.5,
    ], shape: [2, 2])
  let b2 = Tensor(array: [0.0, 0.25], shape: [2])

  // Manual typed chain
  let typed = Sequential(
    Linear(weight: W1, bias: b1),
    Linear(weight: W2, bias: b2)
  )

  // Builder-based chain (lowered to nested Sequential under the hood)
  let block = SequentialBlock {
    Linear(weight: W1, bias: b1)
    Linear(weight: W2, bias: b2)
  }

  // Forward equality
  let yTyped = typed(x)
  let yBlock = block(x)
  #expect(yBlock.isClose(to: yTyped, rtol: 1e-9, atol: 1e-9, equalNan: false))

  // Gradient equality for scalar loss L = sum(y)
  let (_, pbTyped) = valueWithPullback(at: typed) { m in m(x).sum() }
  let gTyped = pbTyped(Tensor(1.0))

  let (_, pbBlock) = valueWithPullback(at: block) { m in m(x).sum() }
  let gBlock = pbBlock(Tensor(1.0))

  // Compare leaf-by-leaf: block.body.<layer>.* vs typed.<layer>.*
  #expect(
    gBlock.body.l1.weight.isClose(to: gTyped.l1.weight, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(gBlock.body.l1.bias.isClose(to: gTyped.l1.bias, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(
    gBlock.body.l2.weight.isClose(to: gTyped.l2.weight, rtol: 1e-9, atol: 1e-9, equalNan: false))
  #expect(gBlock.body.l2.bias.isClose(to: gTyped.l2.bias, rtol: 1e-9, atol: 1e-9, equalNan: false))
}

// MARK: - 2) Residual(x, f) == x + f(x) and grads match inner layer

@Test("Residual(Linear) forward is x + f(x) and grads match inner Linear for sum loss")
func residual_forward_and_grads_match_inner_linear() throws {
  // Choose square shapes so x and f(x) can be added.
  let x = Tensor(
    array: [
      1.0, 2.0,
      -1.0, 0.0,
    ], shape: [2, 2])

  let W = Tensor(
    array: [
      0.3, -0.1,
      0.2, 0.5,
    ], shape: [2, 2])
  let b = Tensor(array: [0.05, -0.2], shape: [2])

  let inner = Linear(weight: W, bias: b)
  let innerCopy = inner  // for the baseline comparison
  let resid = Residual(inner)

  // Forward: Residual == x + f(x)
  let yResid = resid(x)
  let yManual = x.adding(inner(x))
  #expect(yResid.isClose(to: yManual, rtol: 1e-12, atol: 1e-12, equalNan: false))

  // Grads under L = sum(y): d/dθ sum(x + f(x)) == d/dθ sum(f(x))
  let (_, pbResid) = valueWithPullback(at: resid) { m in m(x).sum() }
  let gResid = pbResid(Tensor(1.0))

  let (_, pbInner) = valueWithPullback(at: innerCopy) { l in l(x).sum() }
  let gInner = pbInner(Tensor(1.0))

  #expect(gResid.layer.weight.isClose(to: gInner.weight, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gResid.layer.bias.isClose(to: gInner.bias, rtol: 1e-12, atol: 1e-12, equalNan: false))
}

// MARK: - 3) Concat module matches manual Tensor.cat (forward + parameter pullbacks)

@Test("Concat(l1,l2,dim) matches Tensor.cat([l1(x), l2(x)], dim) (forward + grads)")
func concat_module_matches_manual_cat_forward_and_grads() throws {
  // Input: [B, in]
  let x = Tensor(
    array: [
      0.5, -1.0, 2.0,
      1.5, 0.0, -0.5,
    ], shape: [2, 3])

  // Branch 1: out1=2; Branch 2: out2=1
  let W1 = Tensor(
    array: [
      1.0, 0.0, -1.0,
      0.5, 2.0, 1.0,
    ], shape: [2, 3])
  let b1 = Tensor(array: [0.1, -0.2], shape: [2])

  let W2 = Tensor(
    array: [  // 1 x 3
      2.0, -1.0, 0.5,
    ], shape: [1, 3])
  let b2 = Tensor(array: [0.25], shape: [1])

  let l1 = Linear(weight: W1, bias: b1)
  let l2 = Linear(weight: W2, bias: b2)
  let model = Concat(l1, l2, dim: -1)  // concat along last dimension

  // Forward equality
  let yModule = model(x)
  let yManual = Tensor.cat([l1(x), l2(x)], dim: -1)
  #expect(yModule.isClose(to: yManual, rtol: 1e-12, atol: 1e-12, equalNan: false))

  // Gradient equality for L = sum(concat(...))
  let (_, pbModule) = valueWithPullback(at: model) { m in m(x).sum() }
  let gModule = pbModule(Tensor(1.0))

  let (_, pbManual) = valueWithPullback(at: l1, l2) { a, b in
    Tensor.cat([a(x), b(x)], dim: -1).sum()
  }
  let (g1, g2) = pbManual(Tensor(1.0))

  #expect(gModule.l1.weight.isClose(to: g1.weight, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gModule.l1.bias.isClose(to: g1.bias, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gModule.l2.weight.isClose(to: g2.weight, rtol: 1e-12, atol: 1e-12, equalNan: false))
  #expect(gModule.l2.bias.isClose(to: g2.bias, rtol: 1e-12, atol: 1e-12, equalNan: false))
}
