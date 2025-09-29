import _Differentiation

// MARK: - indexSelect(dim:indices:)

// y = self.indexSelect(dim: d, indices)
// ∂L/∂self = scatter_add_d(upstream, indices)  (accumulates for repeated indices)
// (no gradient wrt indices)
extension Tensor {
  @derivative(of: indexSelect(dim:indices:), wrt: self)
  @inlinable
  public func _vjpIndexSelect<T: TorchSliceIndex & FixedWidthInteger>(
    dim: Int, indices: [T]
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let y = indexSelect(dim: dim, indices: indices)
    return (
      y,
      { v in
        let axis = _normalizeDimension(dim, rank: self.rank)
        let dimSize = self.shape[axis]

        // Normalise negatives into [0, dimSize)
        let norm: [Int64] = indices.map { raw in
          let i = Int64(raw)
          return i < 0 ? i &+ Int64(dimSize) : i
        }

        let idx = Tensor(array: norm, shape: [norm.count], device: self.device)
        let dtype = _resolveFloatingDType(v.dtype, self.dtype)
        let vCast = v.to(dtype: dtype)

        // Scatter-add upstream back into self’s shape
        let base = Tensor.zeros(shape: self.shape, dtype: dtype, device: self.device)
        return base.indexAdd(dim: axis, index: idx, source: vCast)
      }
    )
  }
}

// MARK: - indexAdd(dim:index:source:alpha:)

// y = self + scatter_add_d(alpha * source, index)
// ∂L/∂self   = upstream
// ∂L/∂source = gather_d(upstream, index) * alpha
extension Tensor {
  @derivative(of: indexAdd(dim:index:source:alpha:), wrt: (self, source))
  @inlinable
  internal func _vjpIndexAdd(
    dim: Int, index: Tensor, source: Tensor, alpha: Scalar
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    let y = indexAdd(dim: dim, index: index, source: source, alpha: alpha)
    return (
      y,
      { v in
        let axis = _normalizeDimension(dim, rank: self.rank)
        let idxHost: [Int64] = index.toArray(as: Int64.self)
        let gathered = v.indexSelect(dim: axis, indices: idxHost)
        let gradSource = gathered.multiplying(_scalarToDouble(alpha))
        let gradSelf = v
        return (gradSelf, gradSource)
      }
    )
  }
}

// MARK: - indexCopy(dim:index:source:)

// y = self with y[dim, index] ← source (overwrite)
// ∂L/∂source = gather_d(upstream, index)
// ∂L/∂self   = upstream - scatter_add_d(∂L/∂source, index)
extension Tensor {
  @derivative(of: indexCopy(dim:index:source:), wrt: (self, source))
  @inlinable
  internal func _vjpIndexCopy(
    dim: Int, index: Tensor, source: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    let y = indexCopy(dim: dim, index: index, source: source)
    return (
      y,
      { v in
        let axis = _normalizeDimension(dim, rank: self.rank)
        let dtype = _resolveFloatingDType(v.dtype, self.dtype, source.dtype)
        let vCast = v.to(dtype: dtype)

        let idxHost: [Int64] = index.toArray(as: Int64.self)
        let gradSource = vCast.indexSelect(dim: axis, indices: idxHost)

        let zerosSelf = Tensor.zeros(shape: self.shape, dtype: dtype, device: self.device)
        let overwritten = zerosSelf.indexAdd(dim: axis, index: index, source: gradSource)
        let gradSelf = vCast.subtracting(overwritten)

        return (gradSelf, gradSource)
      }
    )
  }
}

// MARK: - indexPut(indices:values:accumulate:)

// Advanced N-D indexing:
// accumulate == true  (add):  y = self + scatter_add(values, indices)
//   ∂L/∂self   = upstream
//   ∂L/∂values = index(upstream, indices)
// accumulate == false (overwrite): y = self with y[indices] ← values
//   ∂L/∂values = index(upstream, indices)
//   ∂L/∂self   = upstream - scatter(index(upstream, indices), indices)
extension Tensor {
  @derivative(of: indexPut(indices:values:accumulate:), wrt: (self, values))
  @inlinable
  public func _vjpIndexPut(
    indices: [Tensor], values: Tensor, accumulate: Bool
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    let y = indexPut(indices: indices, values: values, accumulate: accumulate)

    return (
      y,
      { v in
        precondition(!indices.isEmpty, "indexPut pullback: empty indices")

        // We’ll accumulate in a floating dtype whenever possible.
        let dtype = _resolveFloatingDType(v.dtype, self.dtype, values.dtype)
        let vCast = v.to(dtype: dtype)

        // -------------------------------
        // 1) Build ordinal map of writes
        // -------------------------------
        // Assign ordinals 1..N (so we can derive a boolean mask with > 0),
        // scatter them into a fresh int64 tensor with the same shape as `self`.
        let n = values.count
        let ordVals = Tensor.arange(
          Int64(1), to: Int64(n) &+ 1, step: Int64(1),
          dtype: .int64, device: values.device
        )
        .reshaped(values.shape)

        let ordBase = Tensor.zeros(shape: self.shape, dtype: .int64, device: self.device)
        let ordScattered = ordBase.indexPut(indices: indices, values: ordVals, accumulate: false)

        // mask of selected positions in `self`
        let selMaskBool = ordScattered.gt(Int64(0))  // Bool mask
        // corresponding ordinal ids in 0..N-1 (packed to 1-D)
        let ordPacked = ordScattered.maskedSelect(selMaskBool).to(dtype: .int64).subtracting(
          Int64(1))

        // upstream values at those positions, packed to 1-D (same order as mask)
        let upPacked = vCast.maskedSelect(selMaskBool)

        // ----------------------------------------------------------
        // 2) Reorder packed upstream to match `values` flatten order
        // ----------------------------------------------------------
        // Sort ordinals so we can align with the original values order (0..N-1)
        let sortPair = ordPacked.sort(dim: 0, descending: false)

        // Reorder the packed upstream by that permutation
        let permHost: [Int64] = sortPair.indices.toArray(as: Int64.self)
        let upSorted = upPacked.indexSelect(dim: 0, indices: permHost)

        // Positions (0-based) in the values vector that actually survived (last-wins)
        let keptOrdinalsHost: [Int64] = sortPair.values.toArray(as: Int64.self)
        let keptOrdinals = Tensor(
          array: keptOrdinalsHost, shape: [keptOrdinalsHost.count], device: values.device
        ).to(dtype: .int64)

        // Scatter the sorted pieces into a length-N vector (zeros for overwritten)
        let zerosN = Tensor.zeros(shape: [n], dtype: dtype, device: values.device)
        let gradValues1D = zerosN.indexAdd(dim: 0, index: keptOrdinals, source: upSorted)

        // Reshape back to `values.shape`
        let gradValues = gradValues1D.reshaped(values.shape)

        // -------------------------
        // 3) Grad w.r.t. `self`
        // -------------------------
        let gradSelf: Tensor
        if accumulate {
          // additive variant: identity path
          gradSelf = vCast
        } else {
          // overwrite variant: subtract whatever flowed into the written region
          let zerosSelf = Tensor.zeros(shape: self.shape, dtype: dtype, device: self.device)
          let removed = zerosSelf.indexPut(indices: indices, values: gradValues, accumulate: false)
          gradSelf = vCast.subtracting(removed)
        }

        return (gradSelf, gradValues)
      }
    )
  }

}
