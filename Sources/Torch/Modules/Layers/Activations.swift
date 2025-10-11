import Foundation
import _Differentiation

// MARK: - ReLU
public struct ReLU: Layer {
  public typealias Input = Tensor
  public typealias Output = Tensor
  public init() {}

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.relu() }  // elementwise op. :contentReference[oaicite:6]{index=6}
}

extension ReLU {
  // Manual VJP via free closure to avoid the "curried self" path.
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (ReLU.TangentVector, Tensor))
  {
    func primal(_ s: ReLU, _ i: Tensor) -> Tensor { i.relu() }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (ReLU.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> ReLU.TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - LeakyReLU (parameterless; slope is a hyper-parameter)
public struct LeakyReLU: Layer {
  @noDerivative public let negativeSlope: Float
  public init(negativeSlope: Float = 0.01) { self.negativeSlope = negativeSlope }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // leaky_relu(x) = relu(x) - a * relu(-x)  (no non-diff max/min needed). :contentReference[oaicite:7]{index=7}
    x.relu().subtracting(x.negated().relu().multiplying(negativeSlope))
  }
}

extension LeakyReLU {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (LeakyReLU.TangentVector, Tensor))
  {
    func primal(_ s: LeakyReLU, _ i: Tensor) -> Tensor {
      i.relu().subtracting(i.negated().relu().multiplying(s.negativeSlope))
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (LeakyReLU.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> LeakyReLU.TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - SiLU / Swish: x * sigmoid(x)
public struct SiLU: Layer {
  public init() {}
  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor { x.multiplying(x.sigmoid()) }  // :contentReference[oaicite:8]{index=8}
}

extension SiLU {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (SiLU.TangentVector, Tensor))
  {
    func primal(_ s: SiLU, _ i: Tensor) -> Tensor { i.multiplying(i.sigmoid()) }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (SiLU.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> SiLU.TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - GELU (exact via erf, or fast tanh-approx)
public struct GELU: Layer {
  @noDerivative public let approximate: Bool
  public init(approximate: Bool = true) { self.approximate = approximate }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    if approximate {
      // 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
      let k: Float = 0.7978845608028654  // √(2/π)
      let c: Float = 0.044715
      let inner = x.adding(x.pow(3).multiplying(c)).multiplying(k)  // pow has VJP. :contentReference[oaicite:9]{index=9}
      return x.multiplying(0.5).multiplying(inner.tanh().adding(1))  // tanh VJP exists. :contentReference[oaicite:10]{index=10}
    } else {
      // 0.5 * x * (1 + erf(x / √2))
      let invRt2: Float = 0.7071067811865476
      return x.multiplying(0.5).multiplying(x.multiplying(invRt2).erf().adding(1))  // erf VJP exists. :contentReference[oaicite:11]{index=11}
    }
  }
}

extension GELU {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (GELU.TangentVector, Tensor))
  {
    let y = callAsFunction(x)

    let derivative: Tensor
    if approximate {
      let k = Tensor(0.7978845608028654, dtype: x.dtype ?? .float32, device: x.device)  // √(2/π)
      let c = Tensor(0.044715, dtype: x.dtype ?? .float32, device: x.device)
      let half = Tensor(0.5, dtype: x.dtype ?? .float32, device: x.device)

      let xSquared = x.multiplying(x)
      let inner = k.multiplying(x.adding(xSquared.multiplying(x).multiplying(c)))
      let tanhInner = inner.tanh()
      let sech2 = Tensor(1, dtype: x.dtype ?? .float32, device: x.device).subtracting(tanhInner.multiplying(tanhInner))
      let fPrime = k.multiplying(Tensor(1, dtype: x.dtype ?? .float32, device: x.device).adding(xSquared.multiplying(c).multiplying(3)))
      derivative = half.multiplying(tanhInner.adding(1)).adding(half.multiplying(x).multiplying(sech2).multiplying(fPrime))
    } else {
      let dtype = x.dtype ?? .float32
      let device = x.device
      let half = Tensor(0.5, dtype: dtype, device: device)
      let invRt2 = Tensor(0.7071067811865476, dtype: dtype, device: device)
      let sqrt2OverPi = Tensor(0.7978845608028654, dtype: dtype, device: device)
      let base = x.multiplying(invRt2).erf().adding(1)
      let expTerm = x.multiplying(x).dividing(-2).exp()
      derivative = half.multiplying(base).adding(x.multiplying(expTerm).multiplying(sqrt2OverPi))
    }

    return (
      y,
      { v in
        let dx = v.multiplying(derivative)
        return (GELU.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> GELU.TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - Tanh / Sigmoid (thin wrappers)
public struct Tanh: Layer {
  public init() {}
  public typealias Input = Tensor
  public typealias Output = Tensor
  @differentiable(reverse) public func callAsFunction(_ x: Tensor) -> Tensor { x.tanh() }  // :contentReference[oaicite:12]{index=12}
}
extension Tanh {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (Tanh.TangentVector, Tensor))
  {
    func primal(_ s: Tanh, _ i: Tensor) -> Tensor { i.tanh() }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (Tanh.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> Tanh.TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

public struct Sigmoid: Layer {
  public init() {}
  public typealias Input = Tensor
  public typealias Output = Tensor
  @differentiable(reverse) public func callAsFunction(_ x: Tensor) -> Tensor { x.sigmoid() }  // :contentReference[oaicite:13]{index=13}
}
extension Sigmoid {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (Sigmoid.TangentVector, Tensor))
  {
    func primal(_ s: Sigmoid, _ i: Tensor) -> Tensor { i.sigmoid() }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (Sigmoid.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> Sigmoid.TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - ELU (alpha is hyper-parameter)
public struct ELU: Layer {
  @noDerivative public let alpha: Float
  public init(alpha: Float = 1.0) { self.alpha = alpha }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // where x>0 ? x : alpha*(exp(x)-1) — route grad via TorchWhere.select. :contentReference[oaicite:14]{index=14}
    let cond = withoutDerivative(at: x.gt(0))  // boolean mask
    let pos = x
    let neg = x.exp().subtracting(1).multiplying(alpha)
    return TorchWhere.select(condition: cond, pos, neg)  // differentiable select. :contentReference[oaicite:15]{index=15}
  }
}
extension ELU {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (ELU.TangentVector, Tensor))
  {
    func primal(_ s: ELU, _ i: Tensor) -> Tensor {
      let cond = withoutDerivative(at: i.gt(0))
      let pos = i
      let neg = i.exp().subtracting(1).multiplying(s.alpha)
      return TorchWhere.select(condition: cond, pos, neg)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (ELU.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> ELU.TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - Softplus: y = (1/β) * log(1 + exp(βx)) with stable branching
public struct Softplus: Layer {
  @noDerivative public let beta: Float
  public init(beta: Float = 1.0) { self.beta = beta }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    // Stable form:
    // z = βx
    // if z > 0: z + log(1 + exp(-z)) else: log(1 + exp(z))
    let z = x.multiplying(beta)
    let one = Tensor(1)
    let pos = z.adding((z.negated()).exp().adding(one).log())
    let neg = z.exp().adding(one).log()
    let cond = withoutDerivative(at: z.gt(0))
    return TorchWhere.select(condition: cond, pos, neg).dividing(beta)  // select has VJP. :contentReference[oaicite:16]{index=16}
  }
}
extension Softplus {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (Softplus.TangentVector, Tensor))
  {
    func primal(_ s: Softplus, _ i: Tensor) -> Tensor {
      let z = i.multiplying(s.beta)
      let one = Tensor(1)
      let pos = z.adding((z.negated()).exp().adding(one).log())
      let neg = z.exp().adding(one).log()
      let cond = withoutDerivative(at: z.gt(0))
      return TorchWhere.select(condition: cond, pos, neg).dividing(s.beta)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (Softplus.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> Softplus.TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - Softmax (axis-aware; stable with detached shift)
public struct Softmax: Layer {
  @noDerivative public let axis: Int
  public init(axis: Int = -1) { self.axis = axis }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let a = withoutDerivative(at: _normalizeDimension(axis, rank: x.rank))  // supports negatives. :contentReference[oaicite:17]{index=17}
    // Detach the shift (max) so we do not need a derivative for max. :contentReference[oaicite:18]{index=18}
    let shift = withoutDerivative(at: x.max(dim: a, keepdim: true).values)
    let expX = (x - shift).exp()  // exp VJP. :contentReference[oaicite:19]{index=19}
    let denom = expX.sum(dim: a, keepdim: true)  // sum VJP. :contentReference[oaicite:20]{index=20}
    return expX.dividing(denom)
  }
}
extension Softmax {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (Softmax.TangentVector, Tensor))
  {
    func primal(_ s: Softmax, _ i: Tensor) -> Tensor {
      let a = withoutDerivative(at: _normalizeDimension(s.axis, rank: i.rank))
      let shift = withoutDerivative(at: i.max(dim: a, keepdim: true).values)
      let expX = (i - shift).exp()
      let denom = expX.sum(dim: a, keepdim: true)
      return expX.dividing(denom)
    }
    let (y, pb) = valueWithPullback(at: self, x, of: primal)
    return (
      y,
      { v in
        let (_, dx) = pb(v)
        return (.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> Softmax.TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}

// MARK: - LogSoftmax (axis-aware; stable with detached shift)
public struct LogSoftmax: Layer {
  @noDerivative public let axis: Int
  public init(axis: Int = -1) { self.axis = axis }

  public typealias Input = Tensor
  public typealias Output = Tensor

  @differentiable(reverse)
  public func callAsFunction(_ x: Tensor) -> Tensor {
    let a = _normalizeDimension(axis, rank: x.rank)  // :contentReference[oaicite:21]{index=21}
    let shift = withoutDerivative(at: x.max(dim: a, keepdim: true).values)  // :contentReference[oaicite:22]{index=22}
    let z = x - shift
    let lse = z.exp().sum(dim: a, keepdim: true).log()  // log VJP exists. :contentReference[oaicite:23]{index=23}
    return z - lse
  }
}
extension LogSoftmax {
  @derivative(of: callAsFunction, wrt: (self, x))
  public func _vjpCallAsFunction(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> (LogSoftmax.TangentVector, Tensor))
  {
    let axisNorm = withoutDerivative(at: _normalizeDimension(axis, rank: x.rank))
    let y = callAsFunction(x)
    let soft = y.exp()
    return (
      y,
      { v in
        let sumV = v.sum(dim: axisNorm, keepdim: true)
        let dx = v - soft.multiplying(sumV)
        return (LogSoftmax.TangentVector.zero, dx)
      }
    )
  }
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ x: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> LogSoftmax.TangentVector)
  {
    let (y, pbBoth) = _vjpCallAsFunction(x)
    return (
      y,
      { v in
        let (dSelf, _) = pbBoth(v)
        return dSelf
      }
    )
  }
}
