// Sources/TaylorTorchDatasets/CIFAR10.swift
import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

// A single CIFAR-10 sample: image in NCHW [3, 32, 32] and the integer label [0-9].
public struct CIFAR10Example {
  public let image: Tensor
  public let label: Int
}

/// Loader for the CIFAR-10 *binary* dataset.
/// Downloads & extracts to ~/.taylortorch/datasets/cifar10 by default, caches thereafter.
public struct CIFAR10 {
  public let train: ArrayDataset<CIFAR10Example>
  public let test: ArrayDataset<CIFAR10Example>

  /// Canonical class names (index == label id).
  public static let classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
  ]

  public init(
    root: URL = DataHome.root.appendingPathComponent("cifar10", isDirectory: true),
    normalize: Bool = true
  ) throws {
    try Downloader.ensureDir(root)

    // 1) Download (cached)
    let url = URL(string: "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")!
    let archive = root.appendingPathComponent("cifar-10-binary.tar.gz")
    _ = try Downloader.fetch(url: url, to: archive)  // uses your Downloader/cache
    // 2) Extract (idempotent)
    let extractedDir = root.appendingPathComponent("cifar-10-batches-bin", isDirectory: true)
    if !FileManager.default.fileExists(atPath: extractedDir.path) {
      _ = try Tar.extract(archive: archive, to: root)
    }

    // 3) Parse binary batches
    var trainExamples: [CIFAR10Example] = []
    trainExamples.reserveCapacity(50_000)
    for i in 1...5 {
      let f = extractedDir.appendingPathComponent("data_batch_\(i).bin")
      trainExamples += try Self.loadBinaryBatch(from: f, normalize: normalize)
    }
    let testExamples = try Self.loadBinaryBatch(
      from: extractedDir.appendingPathComponent("test_batch.bin"),
      normalize: normalize
    )

    self.train = ArrayDataset(trainExamples)
    self.test = ArrayDataset(testExamples)
  }

  // MARK: - Parsing

  private static func loadBinaryBatch(from file: URL, normalize: Bool) throws -> [CIFAR10Example] {
    let bytes = try Data(contentsOf: file)
    let recordBytes = 1 + 32 * 32 * 3  // label + pixels (R 1024, G 1024, B 1024)
    precondition(bytes.count % recordBytes == 0, "Unexpected CIFAR-10 file size")
    let count = bytes.count / recordBytes

    var out: [CIFAR10Example] = []
    out.reserveCapacity(count)

    try bytes.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
      guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
        throw NSError(
          domain: "CIFAR10", code: -1, userInfo: [NSLocalizedDescriptionKey: "Nil buffer"])
      }
      let plane = 32 * 32
      for i in 0..<count {
        let row = base.advanced(by: i * recordBytes)
        let label = Int(row.pointee)
        let imgStart = row.advanced(by: 1)

        // Build [3, 32, 32] in NCHW order, scaled to [0, 1].
        var floats = [Float](repeating: 0, count: 3 * plane)
        for c in 0..<3 {
          let ch = imgStart.advanced(by: c * plane)
          for j in 0..<plane {
            floats[c * plane + j] = Float(ch.advanced(by: j).pointee) / 255.0
          }
        }

        var image = Tensor(array: floats, shape: [3, 32, 32], dtype: .float32)
        if normalize {
          image = Self.normalizeCIFAR10(image)
        }
        out.append(CIFAR10Example(image: image, label: label))
      }
    }
    return out
  }

  /// Per-channel normalization with standard CIFAR-10 mean/std.
  /// (Common practice in reference implementations.)
  public static func normalizeCIFAR10(_ x: Tensor) -> Tensor {
    let mean = Tensor(array: [0.4914, 0.4822, 0.4465], shape: [3, 1, 1], dtype: .float32)
    let std = Tensor(array: [0.2470, 0.2435, 0.2616], shape: [3, 1, 1], dtype: .float32)
    return (x - mean) / std
  }
}
