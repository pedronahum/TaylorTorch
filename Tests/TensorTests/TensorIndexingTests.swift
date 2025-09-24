import Testing
@testable import ATen

@Test("Basic select and slice operations")
func selectAndSliceWork() throws {
  let tensor = Tensor.arange(Int64(0), to: Int64(12), step: Int64(1)).reshaped([3, 4])

  let firstRow = tensor.select(dim: 0, index: Int64(0))
  #expect(firstRow.shape == [4])
  let rowValues = firstRow.toArray(as: Int64.self)
  #expect(rowValues == [0, 1, 2, 3])

  let narrowed = tensor.narrow(dim: 1, start: Int64(1), length: Int64(2))
  #expect(narrowed.shape == [3, 2])

  let sliced = tensor.slice(dim: 1, start: 0, end: 3, step: 2)
  #expect(sliced.shape == [3, 2])
  let sliceValues = sliced.toArray(as: Int64.self)
  #expect(sliceValues == [0, 2, 4, 6, 8, 10])
}

@Test("Subscripts mirror slicing semantics")
func subscriptsProvideConvenience() throws {
  let tensor = Tensor.arange(Int64(0), to: Int64(9), step: Int64(1)).reshaped([3, 3])

  #expect(tensor[Int64(1)].shape == [3])
  let range = tensor[dim: 0, 1..<3]
  #expect(range.shape == [2, 3])
  let closed = tensor[dim: 1, 0...1]
  #expect(closed.shape == [3, 2])

  let partial = tensor[dim: 0, 1...]
  #expect(partial.shape == [2, 3])
}

@Test("Split and chunk partition tensors into views")
func splitAndChunkReturnExpectedSegments() throws {
  let tensor = Tensor.arange(Int64(0), to: Int64(10), step: Int64(1))
  let splits = tensor.split(size: 4)
  #expect(splits.count == 3)
  #expect(splits[0].shape == [4])
  #expect(splits[2].shape == [2])

  let chunks = tensor.chunk(3)
  #expect(chunks.count == 3)
  #expect(chunks[0].shape == [4])
  #expect(chunks[1].shape == [3])
  #expect(chunks[2].shape == [3])
}

@Test("TensorIndex-based multi-axis slicing works")
func tensorIndexVariadicSubscriptWorks() throws {
  let tensor = Tensor.arange(Int64(0), to: Int64(24), step: Int64(1)).reshaped([2, 3, 4])
  let view = tensor[.i(1), .ellipsis]
  #expect(view.shape == [3, 4])

  let expanded = tensor[.newAxis, .i(0), .range(1..<3), .closed(0...2)]
  
  // ✅ Corrected the expected shape and data
  #expect(expanded.shape == [1, 2, 3])
  let data = expanded.toArray(as: Int64.self)
  #expect(data == [4, 5, 6, 8, 9, 10])
}

@Test("Index select gathers rows")
func indexSelectGathersRows() throws {
  let tensor = Tensor.arange(Int64(0), to: Int64(12), step: Int64(1)).reshaped([3, 4])
  let selected = tensor.indexSelect(dim: 0, indices: [Int32(2), Int32(0)])
  #expect(selected.shape == [2, 4])
  let values = selected.toArray(as: Int64.self)
  #expect(values == [8, 9, 10, 11, 0, 1, 2, 3])
}