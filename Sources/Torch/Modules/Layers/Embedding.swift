import _Differentiation

/// Embedding: a trainable lookup table.
///
/// ### Why do we need this?
/// - Turns categorical integer IDs into dense, learnable vectors.
/// - Ubiquitous in NLP, recommendation, tabular models, etc.
/// - A clean `Layer` wrapper keeps the parameter surface minimal (`weight` only)
///   while integrating seamlessly with your optimizers via `ParameterIterable`.
///
/// ### Semantics
/// - Input: an integer Tensor of arbitrary shape (e.g. `[batch, time]`).
/// - Output: a floating Tensor with shape `indices.shape + [embeddingDim]`.
/// - Gradient: upstream gradients are **scattered** back onto the rows of `weight`
///   referenced by `indices` (accumulating for repeated IDs).
/// - `paddingIndex`: if set, the gradient for `weight[paddingIndex]` is **zeroed**,
///   mirroring the behavior of common DL libraries (keeps the pad vector fixed).
///   See PyTorch docs for `padding_idx`.  // mirrors: nn.Embedding's padding_idx semantics.
///
/// This layer relies on core tensor ops you already test heavily:
///  - `indexSelect`/gather and shape transforms for the forward,
///  - `scatterAdd` for assembling the weight gradient (as used in your VJPs).
///
/// Example:
/// ```swift
/// let emb = Embedding(numEmbeddings: 10, embeddingDim: 4, paddingIndex: 0)
/// let ids = Tensor(array: [2, 0, 2, 5], shape: [2, 2]).to(dtype: .int64)
/// let y = emb(ids) // shape: [2, 2, 4]
/// ```
public struct Embedding: Layer {
  /// Trainable lookup table: [numEmbeddings, embeddingDim]
  public var weight: Tensor

  /// If set, the row at `paddingIndex` receives **no gradient**.
  public let paddingIndex: Int?

  /// Designated initializer from an explicit weight matrix.
  /// - Parameters:
  ///   - weight: A floating tensor of shape `[numEmbeddings, embeddingDim]`.
  ///   - paddingIndex: Optional index to exclude from gradient updates.
  public init(weight: Tensor, paddingIndex: Int? = nil) {
    precondition(weight.rank == 2, "Embedding weight must be rank-2 [numEmbeddings, embeddingDim]")
    if let pad = paddingIndex {
      precondition(pad >= 0 && pad < weight.shape[0], "paddingIndex out of bounds")
    }
    self.weight = weight
    self.paddingIndex = paddingIndex
  }

  /// Convenience initializer that builds a weight matrix (zeros by default).
  /// Pass a custom initializer if you want random init.
  /// - Parameters:
  ///   - numEmbeddings: vocabulary size.
  ///   - embeddingDim: embedding dimensionality.
  ///   - paddingIndex: optional pad row (no gradient).
  ///   - initializer: closure to create the weight tensor (shape `[numEmbeddings, embeddingDim]`).
  public init(
    numEmbeddings: Int,
    embeddingDim: Int,
    paddingIndex: Int? = nil,
    initializer: (Int, Int) -> Tensor = { n, d in
      // Conservative default that is guaranteed to compile everywhere.
      Tensor.zeros(shape: [n, d], dtype: .float32)
      // You can plug your Initializers.swift here, e.g. glorotUniform / normal.
    }
  ) {
    self.weight = initializer(numEmbeddings, embeddingDim)
    if self.weight.dtype == nil {
      self.weight = self.weight.to(dtype: .float32)
    }
    self.paddingIndex = paddingIndex
  }

  /// Convenience: zeros init with dtype/device.
  public init(
    numEmbeddings: Int,
    embeddingDim: Int,
    paddingIndex: Int? = nil,
    dtype: DType = .float32,
    device: Device = .cpu
  ) {
    self.init(
      numEmbeddings: numEmbeddings,
      embeddingDim: embeddingDim,
      paddingIndex: paddingIndex,
      initializer: { n, d in Tensor.zeros(shape: [n, d], dtype: dtype, device: device) }
    )
  }

  /// Forward: gather rows for every index (works with any-rank index tensors).
  @differentiable(reverse)
  public func callAsFunction(_ indices: @noDerivative Tensor) -> Tensor {
    let idx = prepareIndices(indices)
    let flat = idx.flattened()
    let hostIdx = withoutDerivative(at: flat.to(device: .cpu).toArray(as: Int64.self))
    let rows = weight.indexSelect(dim: 0, indices: hostIdx)
    var outShape = withoutDerivative(at: idx.shape)
    let embeddingDim = withoutDerivative(at: weight.shape[1])
    outShape.append(embeddingDim)
    return rows.reshaped(outShape)
  }

  @differentiable(reverse)
  public func call(_ input: @noDerivative Tensor, context: ForwardContext) -> Tensor {
    callAsFunction(input)
  }

  public mutating func move(by offset: TangentVector) {
    weight.move(by: offset.weight)
  }

  private func prepareIndices(_ indices: Tensor) -> Tensor {
    var idx = withoutDerivative(at: indices.to(dtype: .int64))
    if idx.device != weight.device {
      idx = withoutDerivative(at: idx.to(device: weight.device))
    }
    return idx
  }
}

extension Embedding {
  // MARK: - ParameterIterableModel conformance
  public static var parameterKeyPaths: [WritableKeyPath<Embedding, Tensor>] { [\Embedding.weight] }

  // MARK: - TangentVector: parameter-wise operations for optimizers
  public struct TangentVector: Differentiable, AdditiveArithmetic, ParameterIterable {
    public var weight: Tensor

    public init(weight: Tensor) { self.weight = weight }

    public static var zero: TangentVector { .init(weight: .zero) }
    public static func + (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(weight: lhs.weight.adding(rhs.weight))
    }
    public static func - (lhs: TangentVector, rhs: TangentVector) -> TangentVector {
      .init(weight: lhs.weight.subtracting(rhs.weight))
    }

    public static var parameterKeyPaths: [WritableKeyPath<TangentVector, Tensor>] { [\Self.weight] }
  }
}

// MARK: - Custom VJP to (1) scatter-add gradients, (2) zero out paddingIndex row.

extension Embedding {
  @derivative(of: callAsFunction, wrt: self)
  public func _vjpCallAsFunction(_ indices: Tensor)
    -> (value: Tensor, pullback: (Tensor) -> TangentVector)
  {
    let idx = prepareIndices(indices)
    let flat = idx.flattened()
    let hostIdx = withoutDerivative(at: flat.to(device: .cpu).toArray(as: Int64.self))
    let rows = weight.indexSelect(dim: 0, indices: hostIdx)
    var outShape = withoutDerivative(at: idx.shape)
    outShape.append(weight.shape[1])
    let value = rows.reshaped(outShape)

    return (
      value,
      { upstream in
        let embeddingDim = withoutDerivative(at: self.weight.shape[1])
        let upstream2D = upstream.reshaped([flat.shape[0], embeddingDim])
        let weightShape = withoutDerivative(at: self.weight.shape)
        var gradW = Tensor.zeros(
          shape: weightShape, dtype: self.weight.dtype!, device: self.weight.device)
        gradW = gradW.indexAdd(dim: 0, index: flat, source: upstream2D)

        if let pad = self.paddingIndex {
          let padIdx = Tensor(Int64(pad), dtype: .int64, device: gradW.device)
          let zerosRow = Tensor.zeros(
            shape: [embeddingDim], dtype: gradW.dtype!, device: gradW.device)
          gradW = gradW.indexPut(indices: [padIdx], values: zerosRow, accumulate: false)
        }

        return TangentVector(weight: gradW)
      }
    )
  }
}
