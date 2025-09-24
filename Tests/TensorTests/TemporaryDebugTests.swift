import Testing

@testable import ATen

@Test("Debug: maskedSelect")
func debugMaskedSelect() throws {
  //print("➡️ Starting debugMaskedSelect...")
  let base = Tensor.arange(Double(0), to: Double(4), step: Double(1))
  let mask = tensor([true, false, true, false], shape: [4])
  let selected = base.maskedSelect(where: mask)
  #expect(selected.shape == [2])
  #expect(selected.toArray(as: Double.self) == [0, 2])
  //print("✅ debugMaskedSelect PASSED")
}

@Test("Debug: maskedFill with Tensor")
func debugMaskedFillTensor() throws {
  //print("➡️ Starting debugMaskedFillTensor...")
  let base = Tensor.arange(Double(0), to: Double(4), step: Double(1))
  let mask = tensor([true, false, true, false], shape: [4])
  let tensorFill = tensor([-1.0, -2.0, -3.0, -4.0], shape: [4])
  let filledByTensor = base.maskedFill(where: mask, with: tensorFill)
  #expect(filledByTensor.toArray(as: Double.self) == [-1, 1, -3, 3])
  //print("✅ debugMaskedFillTensor PASSED")
}
