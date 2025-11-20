import Testing
import _Differentiation

@testable import Torch

@Test("Differentiation: axis-based select shares pullback with integer variant")
func axisSelectGradientMatchesIntegerVariant() throws {
  let input = Tensor(array: (0..<6).map(Double.init), shape: [2, 3])
  let axis: Axis = .last
  let index = 1

  let (valueAxis, pullbackAxis) = valueWithPullback(at: input) { tensor in
    tensor.select(dim: axis, index: index)
  }

  let resolved = axis.resolve(forRank: input.rank)
  let (valueInt, pullbackInt) = valueWithPullback(at: input) { tensor in
    tensor.select(dim: resolved, index: index)
  }

  #expect(valueAxis.isClose(to: valueInt, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: [0.75, -1.25], shape: [2])
  let gradAxis = pullbackAxis(upstream)
  let gradInt = pullbackInt(upstream)
  #expect(gradAxis.isClose(to: gradInt, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: axis-based narrow mirrors integer pullback")
func axisNarrowGradientMatchesIntegerVariant() throws {
  let input = Tensor(array: (0..<24).map(Double.init), shape: [2, 3, 4])
  let axis: Axis = .penultimate
  let start = 1
  let length = 2

  let (valueAxis, pullbackAxis) = valueWithPullback(at: input) { tensor in
    tensor.narrow(dim: axis, start: start, length: length)
  }

  let resolved = axis.resolve(forRank: input.rank)
  let (valueInt, pullbackInt) = valueWithPullback(at: input) { tensor in
    tensor.narrow(dim: resolved, start: start, length: length)
  }

  #expect(valueAxis.isClose(to: valueInt, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: (0..<16).map { Double($0) * 0.1 }, shape: [2, 2, 4])
  let gradAxis = pullbackAxis(upstream)
  let gradInt = pullbackInt(upstream)
  #expect(gradAxis.isClose(to: gradInt, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: axis-based slice matches integer implementation")
func axisSliceGradientMatchesIntegerVariant() throws {
  let input = Tensor(array: (0..<12).map(Double.init), shape: [3, 4])
  let axis: Axis = .batch
  let start = 0
  let end = 3
  let step = 2

  let (valueAxis, pullbackAxis) = valueWithPullback(at: input) { tensor in
    tensor.slice(dim: axis, start: start, end: end, step: step)
  }

  let resolved = axis.resolve(forRank: input.rank)
  let (valueInt, pullbackInt) = valueWithPullback(at: input) { tensor in
    tensor.slice(dim: resolved, start: start, end: end, step: step)
  }

  #expect(valueAxis.isClose(to: valueInt, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: (0..<8).map { Double($0) - 2.0 }, shape: [2, 4])
  let gradAxis = pullbackAxis(upstream)
  let gradInt = pullbackInt(upstream)
  #expect(gradAxis.isClose(to: gradInt, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

@Test("Differentiation: axis-based transpose swaps gradients identically")
func axisTransposeGradientMatchesIntegerVariant() throws {
  let input = Tensor(array: (0..<6).map(Double.init), shape: [2, 3])
  let first: Axis = .batch
  let second: Axis = .feature

  let (valueAxis, pullbackAxis) = valueWithPullback(at: input) { tensor in
    tensor.transposed(first, second)
  }

  let (valueInt, pullbackInt) = valueWithPullback(at: input) { tensor in
    tensor.transposed(0, 1)
  }

  #expect(valueAxis.isClose(to: valueInt, rtol: 1e-6, atol: 1e-6, equalNan: false))

  let upstream = Tensor(array: (0..<6).map { Double($0) * 0.2 }, shape: [3, 2])
  let gradAxis = pullbackAxis(upstream)
  let gradInt = pullbackInt(upstream)
  #expect(gradAxis.isClose(to: gradInt, rtol: 1e-6, atol: 1e-6, equalNan: false))
}

// DISABLED: This test causes Swift compiler crashes with automatic differentiation on Linux
// The for-in loop inside valueWithPullback triggers a "Global is external, but doesn't have external or weak linkage" error
// @Test("Differentiation: axis-based reductions replay integer pullbacks")
// func axisReductionsGradientMatchIntegerVariants() throws {
//   let input = Tensor(array: (0..<24).map(Double.init), shape: [2, 3, 4])
//   let axes: [Axis] = [.batch, .last]
//
//   do {
//     let keepdim = false
//     let (valueAxis, pullbackAxis) = valueWithPullback(at: input) { tensor in
//       tensor.sum(along: axes, keepdim: keepdim)
//     }
//
//     var dimsToReduce: [Int] = []
//     var currentRank = input.rank
//     for axis in axes {
//       let resolvedDim = axis.resolve(forRank: currentRank)
//       dimsToReduce.append(resolvedDim)
//       if !keepdim {
//         currentRank -= 1
//       }
//     }
//
//     let (valueInt, pullbackInt) = valueWithPullback(at: input) { tensor in
//       var current = tensor
//       for dim in dimsToReduce {
//         current = current.sum(dim: dim, keepdim: keepdim)
//       }
//       return current
//     }
//
//     #expect(valueAxis.isClose(to: valueInt, rtol: 1e-6, atol: 1e-6, equalNan: false))
//
//     let upstream = Tensor(array: [0.5, -1.0, 2.0], shape: [3])
//     let gradAxis = pullbackAxis(upstream)
//     let gradInt = pullbackInt(upstream)
//     #expect(gradAxis.isClose(to: gradInt, rtol: 1e-6, atol: 1e-6, equalNan: false))
//   }
//
//   do {
//     let keepdim = true
//     let (valueAxis, pullbackAxis) = valueWithPullback(at: input) { tensor in
//       tensor.mean(along: axes, keepdim: keepdim)
//     }
//
//     var dimsToReduce: [Int] = []
//     var currentRank = input.rank
//     for axis in axes {
//       let resolvedDim = axis.resolve(forRank: currentRank)
//       dimsToReduce.append(resolvedDim)
//       if !keepdim {
//         currentRank -= 1
//       }
//     }
//
//     let (valueInt, pullbackInt) = valueWithPullback(at: input) { tensor in
//       var current = tensor
//       for dim in dimsToReduce {
//         current = current.mean(dim: dim, keepdim: keepdim)
//       }
//       return current
//     }
//
//     #expect(valueAxis.isClose(to: valueInt, rtol: 1e-6, atol: 1e-6, equalNan: false))
//
//     let upstream = Tensor(array: [0.1, -0.2, 0.3], shape: [1, 3, 1])
//     let gradAxis = pullbackAxis(upstream)
//     let gradInt = pullbackInt(upstream)
//     #expect(gradAxis.isClose(to: gradInt, rtol: 1e-6, atol: 1e-6, equalNan: false))
//   }
// }
