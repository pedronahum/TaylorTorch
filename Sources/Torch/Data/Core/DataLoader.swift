import Foundation

/// A minimal, eager DataLoader that yields contiguous batches from a RandomAccessCollection.
/// Uses in-memory batching; add prefetch/parallelism later as needed.
public final class DataLoader<D: RandomAccessCollection>: Sequence where D.Index == Int {

    public typealias Sample = D.Element
    public typealias Batch = [Sample]

    private let data: D
    private let batchSize: Int
    private let shuffle: Bool
    private let dropLast: Bool
    private let seed: UInt64?

    public init(
        dataset: D, batchSize: Int, shuffle: Bool = true, dropLast: Bool = false,
        seed: UInt64? = nil
    ) {
        self.data = dataset
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.dropLast = dropLast
        self.seed = seed
    }

    public func makeIterator() -> AnyIterator<Batch> {
        // Build an index list so we can shuffle without touching the underlying dataset.
        var indices = Array(data.indices)
        if shuffle {
            var rng = SeededRandomNumberGenerator(seed: seed ?? 0xC0FFEE)
            indices.shuffle(using: &rng)
        }

        var cursor = 0
        let n = indices.count
        let step = batchSize

        return AnyIterator { [self] in
            if cursor >= n { return nil }
            let upper = Swift.min(cursor + step, n)
            if self.dropLast && upper - cursor < self.batchSize {
                cursor = n
                return nil
            }
            let batchIdxs = indices[cursor..<upper]
            cursor = upper
            return batchIdxs.map { self.data[$0] }
        }
    }
}

/// A deterministic RNG so shuffling can be repeatable across runs.
public struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64
    public init(seed: UInt64) { self.state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed }
    public mutating func next() -> UInt64 {
        // XorShift64*
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
