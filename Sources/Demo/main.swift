import ATen
import Foundation // Needed for Codable tests

print("Swift successfully imported ATen via C++ interop ✅")
print("--- Basic Ops ---")

let s = Tensor(Int32(3))
let z = Tensor.zeros(shape: [2, 3], dtype: .float32)
let z2 = z + 2.5
print("z2:", z2)


print("\n--- Array I/O ---")
let t_io = Tensor(array: Array(Int32(0)..<Int32(6)), shape: [2, 3])
let back_io: [Int32] = t_io.toArray()
print("Array I/O test passed: \(back_io == [0, 1, 2, 3, 4, 5])")


print("\n--- Convenience initializers ---")
let t_literal = tensor([1.0, 2.0, 3.0, 4.0], 2, 2)
let t_variadic = Tensor.ones(2, 3, dtype: .float32)
print("tensor() literal:", t_literal)
print("Variadic ones():", t_variadic)


print("\n--- Operators ---")
let opA = Tensor(array: [1, 2, 3], shape: [3])
let opB = Tensor(array: [4, 5, 6], shape: [3])
var opC = opA
opC += 1
print("A + B =", opA + opB)
print("A * 3 =", opA * 3)
print("C (compound) =", opC)
print("A .< 2 =", opA .< 2)



print("\n--- Advanced Subscripting ---")
let t_adv = Tensor.arange(0, to: 24, step: 1).reshaped([2, 3, 4])
print("Advanced indexing on shape:", t_adv.shape)
print("t_adv[0].shape:             ", t_adv[0].shape)

// ✅ Explicitly use the .range case for the subscript
print("t_adv[0, .range(1..<3)].shape:", t_adv[0, .range(1..<3)].shape)

print("t_adv[.all, 0, .all].shape: ", t_adv[.all, 0, .all].shape)

print("\n--- Codable (Serialization) ---")
let t_codable = Tensor.arange(0.0, to: 10.0, step: 1.0)
let codable_wrapper = CodableTensor(t_codable)
let encoder = JSONEncoder()
let decoder = JSONDecoder()

do {
    let jsonData = try encoder.encode(codable_wrapper)
    print("Serialized to \(jsonData.count) bytes of JSON")
    let decoded_wrapper = try decoder.decode(CodableTensor.self, from: jsonData)
    let t_decoded = decoded_wrapper.makeTensor()
    print("Serialization round-trip successful: \(t_codable.isClose(to: t_decoded))")
} catch {
    print("Serialization failed: \(error)")
}


print("\n--- Host Buffer Access ---")
let t_host = Tensor(array: [10, 20, 30], shape: [3])
let (_, new_t_host) = t_host.withMutableHostBuffer(as: Int32.self) { buffer in
    for i in 0..<buffer.count {
        buffer[i] *= 2
    }
}
print("Original host tensor:", t_host)
print("Mutated host tensor: ", new_t_host)


// --- All previous tests from here on ---

print("\n--- Indexing & Slicing ---")
let t_idx = Tensor(array: Array(Int32(0)..<Int32(6)), shape: [2, 3])
let row1 = t_idx[1]
let col2 = t_idx[dim: 1, 2]
print("row1 shape:", row1.shape, "| col2 shape:", col2.shape)


print("\n--- Shape Ops & Advanced Indexing ---")
let t_shape = Tensor(array: Array(Int32(0)..<Int32(12)), shape: [3, 4])
let r_reshaped = t_shape.reshaped(inferring: [-1, 6])
let f = r_reshaped.flattened(startDim: 1)
print("reshaped(inferring:):", r_reshaped.shape, "| flattened:", f.shape)


print("\n--- Joiners ---")
let t0 = Tensor.arange(0, to: 6, step: 1).reshaped([2,3])
let t1 = Tensor.arange(6, to: 12, step: 1).reshaped([2,3])
let cat0 = Tensor.cat([t0, t1], dim: 0)
let stk  = Tensor.stack([t0, t1], dim: 0)
print("cat0 shape:", cat0.shape, "| stack shape:", stk.shape)


print("\n--- Math, Reductions, Comparisons, Linalg ---")
let A = Tensor.arange(0, to: 6, step: 1).reshaped([2,3])
let B = Tensor.ones(2, 3, dtype: .float32)
let C = (A.to(dtype: .float32) + 2) * 3
let D = C / 2.0
print("D (after math ops):", D)
let E = A.to(dtype: .float32).matmul(A.transposed(0, 1).to(dtype: .float32))
print("E.shape (matmul):", E.shape)
print("where(A.<3, B, C):", TorchWhere.select(condition: A .< 3, B, C))


print("\n--- Reductions with indices ---")
let X = Tensor.arange(0, to: 12, step: 1).reshaped([3, 4])
let min01 = X.min(dim: 1)
print("min along dim=1 -> values:", min01.values.shape, "indices:", min01.indices.shape)


print("\n--- Broadcasting ---")
let a_bc = Tensor.arange(0, to: 6, step: 1).reshaped([2, 3])
let b_bc = Tensor.ones(1, 3, dtype: .float32)
let sum_bc = a_bc.to(dtype: .float32) + b_bc
print("Broadcast sum shape:", sum_bc.shape)


print("\n--- Masks ---")
let mask1 = a_bc .< 3
let filled = a_bc.to(dtype: .float32).maskedFill(where: mask1, with: Float(100))
print("filled tensor:", filled)