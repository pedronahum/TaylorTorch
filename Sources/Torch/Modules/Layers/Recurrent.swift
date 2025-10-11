// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import _Differentiation

/// An input to a recurrent neural network.
public struct RNNCellInput<Input: Differentiable, State: Differentiable>: Differentiable {
  /// The input at the current time step.
  public var input: Input
  /// The previous state.
  public var state: State

  @differentiable(reverse)
  public init(input: Input, state: State) {
    self.input = input
    self.state = state
  }
}

extension RNNCellInput: EuclideanDifferentiable
where Input: EuclideanDifferentiable, State: EuclideanDifferentiable {}

/// An output to a recurrent neural network.
public struct RNNCellOutput<Output: Differentiable, State: Differentiable>: Differentiable {
  /// The output at the current time step.
  public var output: Output
  /// The current state.
  public var state: State

  @differentiable(reverse)
  public init(output: Output, state: State) {
    self.output = output
    self.state = state
  }
}

extension RNNCellOutput: EuclideanDifferentiable
where Output: EuclideanDifferentiable, State: EuclideanDifferentiable {}

/// A recurrent layer cell.
public protocol RecurrentLayerCell: Layer
where
  Input == RNNCellInput<TimeStepInput, State>,
  Output == RNNCellOutput<TimeStepOutput, State>
{
  /// The input at a time step.
  associatedtype TimeStepInput: Differentiable
  /// The output at a time step.
  associatedtype TimeStepOutput: Differentiable
  /// The state that may be preserved across time steps.
  associatedtype State: Differentiable

  /// Returns a zero-valued state with shape compatible with the provided input.
  func zeroState(for input: TimeStepInput) -> State
}

extension RecurrentLayerCell {
  /// Returns the new state obtained from applying the recurrent layer cell to the input at the
  /// current time step and the previous state.
  ///
  /// - Parameters:
  ///   - timeStepInput: The input at the current time step.
  ///   - previousState: The previous state of the recurrent layer cell.
  /// - Returns: The output.
  @differentiable(reverse)
  public func callAsFunction(
    input: TimeStepInput,
    state: State
  ) -> RNNCellOutput<TimeStepOutput, State> {
    self(RNNCellInput(input: input, state: state))
  }

  @differentiable(reverse)
  public func call(input: TimeStepInput, state: State) -> RNNCellOutput<TimeStepOutput, State> {
    self(RNNCellInput(input: input, state: state))
  }
}

/// A basic RNN cell.
public struct BasicRNNCell: RecurrentLayerCell {

  public var weight: Tensor
  public var bias: Tensor

  public typealias State = Tensor
  public typealias TimeStepInput = Tensor
  public typealias TimeStepOutput = State
  public typealias Input = RNNCellInput<TimeStepInput, State>
  public typealias Output = RNNCellOutput<TimeStepOutput, State>

  /// Creates a `SimpleRNNCell` with the specified input size and hidden state size.
  ///
  /// - Parameters:
  ///   - inputSize: The number of features in 2-D input tensors.
  ///   - hiddenSize: The number of features in 2-D hidden states.
  ///   - seed: The random seed for initialization. The default value is random.
  public init(inputSize: Int, hiddenSize: Int, seed: Int = Context.local.randomSeed) {

    self.weight = Tensor.uniform(
      low: -5, high: 5, shape: [inputSize, hiddenSize], dtype: .float32, device: .cpu)

    self.bias = Tensor.zeros(shape: [hiddenSize], dtype: .float32)
  }

  /// Returns a zero-valued state with shape compatible with the provided input.
  public func zeroState(for input: Tensor) -> State {
    Tensor.zeros(
      shape: [input.shape[0], weight.shape[1]], dtype: .float32)
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The hidden state.
  @differentiable(reverse)
  public func callAsFunction(_ input: Input) -> Output {
    let concatenatedInput = Tensor.cat([input.input, input.state], dim: 1)
    let newState = concatenatedInput.matmul(weight) + bias
    return Output(output: newState, state: newState)
  }

  // MARK: - Manual TangentVector to avoid synthesis pitfalls
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,  // gives `zero`, `+`, `-`
    KeyPathIterable,  // needed by Module constraint
    VectorProtocol,  // implies AdditiveArithmetic; scalar ops via defaults
    PointwiseMultiplicative  // default reflection-powered impl
  {
    public typealias VectorSpaceScalar = Float

    public var weight: Tensor
    public var bias: Tensor

    public init(weight: Tensor = Tensor(0), bias: Tensor = Tensor(0)) {
      self.weight = weight
      self.bias = bias
    }

    // AdditiveArithmetic (spelled out, no synthesis)
    public static var zero: Self { Self() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(weight: lhs.weight + rhs.weight, bias: lhs.bias + rhs.bias)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(weight: lhs.weight - rhs.weight, bias: lhs.bias - rhs.bias)
    }

    // VectorProtocol & PointwiseMultiplicative get their behavior
    // from reflection-based defaults, since the stored properties
    // (`Tensor`) conform to the internal “field” protocols used there.
  }

  // Required by `Differentiable` conformance when `TangentVector` is manually defined.
  public mutating func move(by direction: TangentVector) {
    weight += direction.weight
    bias += direction.bias
  }
}

/// An LSTM cell.
public struct LSTMCell: RecurrentLayerCell {
  public var fusedWeight: Tensor
  public var fusedBias: Tensor

  // MARK: - Manual TangentVector (avoid synthesis)
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,  // gives zero, +, -
    KeyPathIterable,  // required by Module
    VectorProtocol,  // required by Module
    PointwiseMultiplicative  // required by Module
  {
    public typealias VectorSpaceScalar = Float

    // ⚠️ Names and order MUST match the primal's differentiable stored properties.
    public var fusedWeight: Tensor
    public var fusedBias: Tensor

    public init(
      fusedWeight: Tensor = Tensor.zeros(shape: [0, 0], dtype: .float32),
      fusedBias: Tensor = Tensor.zeros(shape: [0], dtype: .float32)
    ) {
      self.fusedWeight = fusedWeight
      self.fusedBias = fusedBias
    }

    // AdditiveArithmetic (explicit — no synthesis)
    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(
        fusedWeight: lhs.fusedWeight + rhs.fusedWeight,
        fusedBias: lhs.fusedBias + rhs.fusedBias
      )
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(
        fusedWeight: lhs.fusedWeight - rhs.fusedWeight,
        fusedBias: lhs.fusedBias - rhs.fusedBias
      )
    }
  }

  // MARK: - Avoid synthesized move(by:)
  public mutating func move(by direction: TangentVector) {
    fusedWeight += direction.fusedWeight
    fusedBias += direction.fusedBias
  }

  public typealias TimeStepInput = Tensor
  public typealias TimeStepOutput = Tensor
  public typealias Input = RNNCellInput<TimeStepInput, State>
  public typealias Output = RNNCellOutput<TimeStepOutput, State>

  private var hiddenSize: Int { fusedWeight.shape[1] / 4 }

  public init(inputSize: Int, hiddenSize: Int, seed: Int = Context.local.randomSeed) {
    self.fusedWeight = Tensor.uniform(
      low: -5,
      high: 5,
      shape: [inputSize + hiddenSize, 4 * hiddenSize],
      dtype: .float32,
      device: .cpu)
    self.fusedBias = Tensor.zeros(shape: [4 * hiddenSize], dtype: .float32)
  }

  public func zeroState(for input: Tensor) -> State {
    let batchSize = input.shape[0]
    let cellZero = Tensor.zeros(shape: [batchSize, hiddenSize], dtype: .float32)
    let hiddenZero = Tensor.zeros(shape: [batchSize, hiddenSize], dtype: .float32)
    return State(cell: cellZero, hidden: hiddenZero)
  }

  @differentiable(reverse)
  public func callAsFunction(_ input: Input) -> Output {
    let concatenatedInput = Tensor.cat([input.input, input.state.hidden], dim: 1)
    let gateLinear = concatenatedInput.matmul(fusedWeight).adding(fusedBias)
    let hiddenSize = withoutDerivative(at: self.hiddenSize)
    let inputGateLinear = gateLinear.narrow(dim: 1, start: 0, length: hiddenSize)
    let candidateGateLinear = gateLinear.narrow(dim: 1, start: hiddenSize, length: hiddenSize)
    let forgetGateLinear = gateLinear.narrow(dim: 1, start: 2 * hiddenSize, length: hiddenSize)
    let outputGateLinear = gateLinear.narrow(dim: 1, start: 3 * hiddenSize, length: hiddenSize)

    let inputGate = inputGateLinear.sigmoid()
    let candidateGate = candidateGateLinear.tanh()
    let forgetGate = forgetGateLinear.sigmoid()
    let outputGate = outputGateLinear.sigmoid()

    let newCell =
      forgetGate.multiplying(input.state.cell).adding(inputGate.multiplying(candidateGate))
    let newHidden = outputGate.multiplying(newCell.tanh())
    let newState = State(cell: newCell, hidden: newHidden)
    return Output(output: newHidden, state: newState)
  }

  public var inputWeight: Tensor {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedWeight.narrow(dim: 1, start: 0, length: hiddenSize)
  }

  public var updateWeight: Tensor {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedWeight.narrow(dim: 1, start: hiddenSize, length: hiddenSize)
  }

  public var forgetWeight: Tensor {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedWeight.narrow(dim: 1, start: 2 * hiddenSize, length: hiddenSize)
  }

  public var outputWeight: Tensor {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedWeight.narrow(dim: 1, start: 3 * hiddenSize, length: hiddenSize)
  }

  public var inputBias: Tensor {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedBias.narrow(dim: 0, start: 0, length: hiddenSize)
  }

  public var updateBias: Tensor {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedBias.narrow(dim: 0, start: hiddenSize, length: hiddenSize)
  }

  public var forgetBias: Tensor {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedBias.narrow(dim: 0, start: 2 * hiddenSize, length: hiddenSize)
  }

  public var outputBias: Tensor {
    let hiddenSize = fusedWeight.shape[1] / 4
    return fusedBias.narrow(dim: 0, start: 3 * hiddenSize, length: hiddenSize)
  }

  /// Creates a `LSTMCell` with the specified input size and hidden state size.
  ///
  /// - Parameters:
  ///   - inputSize: The number of features in 2-D input tensors.
  ///   - hiddenSize: The number of features in 2-D hidden states.
  public init(inputSize: Int, hiddenSize: Int) {

    self.fusedWeight = Tensor.uniform(
      low: -5,
      high: 5,
      shape: [inputSize + hiddenSize, 4 * hiddenSize],
      dtype: .float32,
      device: .cpu)
    self.fusedBias = Tensor.zeros(shape: [4 * hiddenSize], dtype: .float32)

  }

  public struct State: Equatable, Differentiable, VectorProtocol, KeyPathIterable, Mergeable {
    public var cell: Tensor
    public var hidden: Tensor

    // MARK: - AdditiveArithmetic (explicit; avoid synthesis of `+`)
    public static var zero: Self {
      // Scalars are safe here — broadcasting handles real shapes at use sites.
      Self(cell: Tensor(0), hidden: Tensor(0))
    }

    public static func + (lhs: Self, rhs: Self) -> Self {
      Self(
        cell: lhs.cell + rhs.cell,
        hidden: lhs.hidden + rhs.hidden)
    }

    public static func - (lhs: Self, rhs: Self) -> Self {
      Self(
        cell: lhs.cell - rhs.cell,
        hidden: lhs.hidden - rhs.hidden)
    }

    // MARK: - Manual derivatives for `+` and `-`
    @derivative(of: +)
    public static func _vjpAdd(_ lhs: Self, _ rhs: Self)
      -> (value: Self, pullback: (TangentVector) -> (TangentVector, TangentVector))
    {
      let y = lhs + rhs
      return (y, { v in (v, v) })
    }

    @derivative(of: -)
    public static func _vjpSub(_ lhs: Self, _ rhs: Self)
      -> (value: Self, pullback: (TangentVector) -> (TangentVector, TangentVector))
    {
      let y = lhs - rhs
      return (y, { v in (v, .zero - v) })
    }

    // MARK: - Explicit manual TangentVector (no synthesis)
    public struct TangentVector:
      Differentiable,
      AdditiveArithmetic,  // explicit zero/+/- to avoid synthesis pitfalls
      KeyPathIterable,  // enables reflection-based vector ops
      VectorProtocol,  // provides adding(_:) / scaled(by:) defaults
      PointwiseMultiplicative,  // <= provides multiplying(_:) default
      Mergeable  // <= provides merging defaults
    {

      public typealias VectorSpaceScalar = Float
      // ⚠️ Field names & order MUST mirror the primal:
      public var cell: Tensor
      public var hidden: Tensor

      @differentiable(reverse)
      public static func concatenate(
        _ lhs: LSTMCell.State.TangentVector, _ rhs: LSTMCell.State.TangentVector
      ) -> LSTMCell.State.TangentVector {
        let rank = withoutDerivative(at: lhs.cell.rank)
        let axis = withoutDerivative(at: _normalizeDimension(-1, rank: rank))
        let cell = Tensor.cat([lhs.cell, rhs.cell], dim: axis)
        let hidden = Tensor.cat([lhs.hidden, rhs.hidden], dim: axis)
        return .init(cell: cell, hidden: hidden)
      }

      @differentiable(reverse)
      public static func sum(
        _ lhs: LSTMCell.State.TangentVector, _ rhs: LSTMCell.State.TangentVector
      ) -> LSTMCell.State.TangentVector {
        .init(
          cell: lhs.cell.adding(rhs.cell),
          hidden: lhs.hidden.adding(rhs.hidden))
      }

      @differentiable(reverse)
      public static func average(
        _ lhs: LSTMCell.State.TangentVector, _ rhs: LSTMCell.State.TangentVector
      ) -> LSTMCell.State.TangentVector {
        let half: Float = 0.5
        let cell = lhs.cell.adding(rhs.cell).multiplying(half)
        let hidden = lhs.hidden.adding(rhs.hidden).multiplying(half)
        return .init(cell: cell, hidden: hidden)
      }

      @differentiable(reverse)
      public static func multiply(
        _ lhs: LSTMCell.State.TangentVector, _ rhs: LSTMCell.State.TangentVector
      ) -> LSTMCell.State.TangentVector {
        .init(
          cell: lhs.cell.multiplying(rhs.cell),
          hidden: lhs.hidden.multiplying(rhs.hidden))
      }

      @differentiable(reverse)
      public static func stack(
        _ lhs: LSTMCell.State.TangentVector, _ rhs: LSTMCell.State.TangentVector
      ) -> LSTMCell.State.TangentVector {
        let cell = Tensor.stack([lhs.cell, rhs.cell])
        let hidden = Tensor.stack([lhs.hidden, rhs.hidden])
        return .init(cell: cell, hidden: hidden)
      }

      // Use scalar zeros so broadcasting works when added to shaped tensors.
      public init(cell: Tensor = Tensor(0), hidden: Tensor = Tensor(0)) {
        self.cell = cell
        self.hidden = hidden
      }

      // AdditiveArithmetic — spelled out (no synthesis).
      public static var zero: Self { .init() }

      public static func + (lhs: Self, rhs: Self) -> Self {
        .init(
          cell: lhs.cell + rhs.cell,
          hidden: lhs.hidden + rhs.hidden)
      }

      public static func - (lhs: Self, rhs: Self) -> Self {
        .init(
          cell: lhs.cell - rhs.cell,
          hidden: lhs.hidden - rhs.hidden)
      }
    }

    // MARK: - Avoid synthesized move(by:)
    public mutating func move(by direction: TangentVector) {
      cell += direction.cell
      hidden += direction.hidden
    }

    @differentiable(reverse)
    public init(cell: Tensor, hidden: Tensor) {
      self.cell = cell
      self.hidden = hidden
    }

    /// Concatenates two values.
    @differentiable(reverse)
    public static func concatenate(_ lhs: Self, _ rhs: Self) -> Self {
      let rank = withoutDerivative(at: lhs.cell.rank)
      let axis = withoutDerivative(at: _normalizeDimension(-1, rank: rank))
      let cell = Tensor.cat([lhs.cell, rhs.cell], dim: axis)
      let hidden = Tensor.cat([lhs.hidden, rhs.hidden], dim: axis)
      return Self(cell: cell, hidden: hidden)
    }

    /// Adds two values and produces their sum.
    @differentiable(reverse)
    public static func sum(_ lhs: Self, _ rhs: Self) -> Self {
      Self(
        cell: lhs.cell.adding(rhs.cell),
        hidden: lhs.hidden.adding(rhs.hidden))
    }

    /// Averages two values.
    @differentiable(reverse)
    public static func average(_ lhs: Self, _ rhs: Self) -> Self {
      let half: Float = 0.5
      let cell = lhs.cell.adding(rhs.cell).multiplying(half)
      let hidden = lhs.hidden.adding(rhs.hidden).multiplying(half)
      return Self(cell: cell, hidden: hidden)
    }

    /// Multiplies two values.
    @differentiable(reverse)
    public static func multiply(_ lhs: Self, _ rhs: Self) -> Self {
      Self(
        cell: lhs.cell.multiplying(rhs.cell),
        hidden: lhs.hidden.multiplying(rhs.hidden))
    }

    /// Stack two values.
    @differentiable(reverse)
    public static func stack(_ lhs: Self, _ rhs: Self) -> Self {
      let cell = Tensor.stack([lhs.cell, rhs.cell])
      let hidden = Tensor.stack([lhs.hidden, rhs.hidden])
      return Self(cell: cell, hidden: hidden)
    }
  }

}

// MARK: - GRUCell
public struct GRUCell: RecurrentLayerCell {
  public var fusedWeight: Tensor
  public var fusedBias: Tensor

  public typealias State = Tensor
  public typealias TimeStepInput = Tensor
  public typealias TimeStepOutput = Tensor
  public typealias Input = RNNCellInput<TimeStepInput, State>
  public typealias Output = RNNCellOutput<TimeStepOutput, State>

  // 3 * hidden (update z, reset r, candidate n)
  @inline(__always)
  private var hiddenSize: Int { fusedWeight.shape[1] / 3 }

  // MARK: - Inits
  public init(inputSize: Int, hiddenSize: Int, seed: Int = Context.local.randomSeed) {
    self.fusedWeight = Tensor.uniform(
      low: -5, high: 5,
      shape: [inputSize + hiddenSize, 3 * hiddenSize],
      dtype: .float32, device: .cpu)
    self.fusedBias = Tensor.zeros(shape: [3 * hiddenSize], dtype: .float32)
  }

  public init(inputSize: Int, hiddenSize: Int) {
    self.fusedWeight = Tensor.uniform(
      low: -5, high: 5,
      shape: [inputSize + hiddenSize, 3 * hiddenSize],
      dtype: .float32, device: .cpu)
    self.fusedBias = Tensor.zeros(shape: [3 * hiddenSize], dtype: .float32)
  }

  // MARK: - Manual TangentVector (avoid synthesis pitfalls)
  public struct TangentVector:
    Differentiable,
    AdditiveArithmetic,  // explicit zero/+/- to avoid solver synthesis
    KeyPathIterable,  // required by Module
    VectorProtocol,  // required by Module
    PointwiseMultiplicative  // required by Module
  {
    public typealias VectorSpaceScalar = Float
    // Match primal field names & order exactly:
    public var fusedWeight: Tensor
    public var fusedBias: Tensor

    public init(
      fusedWeight: Tensor = Tensor.zeros(shape: [0, 0], dtype: .float32),
      fusedBias: Tensor = Tensor.zeros(shape: [0], dtype: .float32)
    ) {
      self.fusedWeight = fusedWeight
      self.fusedBias = fusedBias
    }

    public static var zero: Self { .init() }
    public static func + (lhs: Self, rhs: Self) -> Self {
      .init(
        fusedWeight: lhs.fusedWeight + rhs.fusedWeight,
        fusedBias: lhs.fusedBias + rhs.fusedBias)
    }
    public static func - (lhs: Self, rhs: Self) -> Self {
      .init(
        fusedWeight: lhs.fusedWeight - rhs.fusedWeight,
        fusedBias: lhs.fusedBias - rhs.fusedBias)
    }
  }

  // Required when defining a manual TangentVector
  public mutating func move(by direction: TangentVector) {
    fusedWeight += direction.fusedWeight
    fusedBias += direction.fusedBias
  }

  // MARK: - Split views into gates
  public var updateWeight: Tensor {
    let h = hiddenSize
    return fusedWeight.narrow(dim: 1, start: 0, length: h)
  }
  public var resetWeight: Tensor {
    let h = hiddenSize
    return fusedWeight.narrow(dim: 1, start: h, length: h)
  }
  public var candidateWeight: Tensor {
    let h = hiddenSize
    return fusedWeight.narrow(dim: 1, start: 2 * h, length: h)
  }
  public var updateBias: Tensor {
    let h = hiddenSize
    return fusedBias.narrow(dim: 0, start: 0, length: h)
  }
  public var resetBias: Tensor {
    let h = hiddenSize
    return fusedBias.narrow(dim: 0, start: h, length: h)
  }
  public var candidateBias: Tensor {
    let h = hiddenSize
    return fusedBias.narrow(dim: 0, start: 2 * h, length: h)
  }

  // MARK: - API
  public func zeroState(for input: Tensor) -> State {
    let batch = input.shape[0]
    return Tensor.zeros(shape: [batch, hiddenSize], dtype: .float32)
  }

  /// GRU forward:
  /// z = σ([x, h] Wz + bz)
  /// r = σ([x, h] Wr + br)
  /// n = tanh([x, (r ⊙ h)] Wn + bn)
  /// h' = z ⊙ h + (1 - z) ⊙ n
  @differentiable(reverse)
  public func callAsFunction(_ input: Input) -> Output {
    let h = input.state
    let xh = Tensor.cat([input.input, h], dim: 1)

    // Compute z and r from a single matmul
    let gateLinear = xh.matmul(fusedWeight).adding(fusedBias)
    let hsize = withoutDerivative(at: hiddenSize)
    let z = gateLinear.narrow(dim: 1, start: 0, length: hsize).sigmoid()
    let r = gateLinear.narrow(dim: 1, start: hsize, length: hsize).sigmoid()

    // Candidate uses reset-gated hidden
    let rh = h.multiplying(r)
    let xrh = Tensor.cat([input.input, rh], dim: 1)
    let n = xrh.matmul(candidateWeight).adding(candidateBias).tanh()

    let newHidden = z.multiplying(h).adding((Tensor(1) - z).multiplying(n))
    return Output(output: newHidden, state: newHidden)
  }
}

// MARK: - Manual derivatives to avoid “curried self” solver path
extension GRUCell {
  @derivative(of: callAsFunction, wrt: (self, input))
  public func _vjpCallAsFunction(_ input: Input)
    -> (
      value: Output,
      pullback: (Output.TangentVector) -> (TangentVector, Input.TangentVector)
    )
  {
    func primal(_ s: GRUCell, _ i: Input) -> Output {
      let h = i.state
      let xh = Tensor.cat([i.input, h], dim: 1)

      let hsize = withoutDerivative(at: s.fusedWeight.shape[1] / 3)
      let gateLinear = xh.matmul(s.fusedWeight).adding(s.fusedBias)
      let z = gateLinear.narrow(dim: 1, start: 0, length: hsize).sigmoid()
      let r = gateLinear.narrow(dim: 1, start: hsize, length: hsize).sigmoid()

      let rh = h.multiplying(r)
      let xrh = Tensor.cat([i.input, rh], dim: 1)
      let candidateW = s.fusedWeight.narrow(dim: 1, start: 2 * hsize, length: hsize)
      let candidateB = s.fusedBias.narrow(dim: 0, start: 2 * hsize, length: hsize)
      let n = xrh.matmul(candidateW).adding(candidateB).tanh()

      let newHidden = z.multiplying(h).adding((Tensor(1) - z).multiplying(n))
      return Output(output: newHidden, state: newHidden)
    }

    let (y, pb) = valueWithPullback(at: self, input, of: primal)
    return (y, pb)
  }

  // Some call sites only request ∂/∂self.
  @derivative(of: callAsFunction, wrt: (self))
  public func _vjpCallAsFunction_wrtSelf(_ input: Input)
    -> (value: Output, pullback: (Output.TangentVector) -> TangentVector)
  {
    let (y, pbFull) = _vjpCallAsFunction(input)
    return (
      y,
      { v in
        let (dSelf, _) = pbFull(v)
        return dSelf
      }
    )
  }
}
