import Foundation

// MARK: - Public Types

public struct MNISTExample {
    public let image: Tensor  // shape: [1, 28, 28]
    public let label: Int32  // 0...9
    public init(image: Tensor, label: Int32) {
        self.image = image
        self.label = label
    }
}

public struct MNIST {
    public let train: ArrayDataset<MNISTExample>
    public let test: ArrayDataset<MNISTExample>

    public init(root: URL? = nil, download: Bool = true, normalize: Bool = true) throws {
        let rootDir = (root ?? DataHome.root).appendingPathComponent("mnist", isDirectory: true)
        try Downloader.ensureDir(rootDir)

        let urls = MNISTSources.urls

        let trainImgPath = rootDir.appendingPathComponent("train-images-idx3-ubyte")
        let trainLblPath = rootDir.appendingPathComponent("train-labels-idx1-ubyte")
        let testImgPath = rootDir.appendingPathComponent("t10k-images-idx3-ubyte")
        let testLblPath = rootDir.appendingPathComponent("t10k-labels-idx1-ubyte")

        // Download + decompress if needed.
        if download {
            func ensureFileExists(at rawPath: URL, remote: URL) throws {
                if FileManager.default.fileExists(atPath: rawPath.path) {
                    print("[MNIST] Using cached \(rawPath.lastPathComponent)")
                    return
                }
                let gzPath = rawPath.appendingPathExtension("gz")
                print("[MNIST] Preparing \(rawPath.lastPathComponent)")
                let gz = try Downloader.fetch(url: remote, to: gzPath)
                print("[MNIST] Decompressing \(gz.lastPathComponent) → \(rawPath.lastPathComponent)")
                let data = try GZip.gunzipFile(at: gz)
                try data.write(to: rawPath, options: .atomic)
                let sizeMB = Double(data.count) / 1_048_576.0
                print(String(format: "[MNIST] Wrote %@ (%.1f MB)", rawPath.lastPathComponent, sizeMB))
            }

            try ensureFileExists(at: trainImgPath, remote: urls.trainImages)
            try ensureFileExists(at: trainLblPath, remote: urls.trainLabels)
            try ensureFileExists(at: testImgPath, remote: urls.testImages)
            try ensureFileExists(at: testLblPath, remote: urls.testLabels)
        }

        let trainSet = try MNISTParser.parse(
            imagesURL: trainImgPath, labelsURL: trainLblPath, normalize: normalize)
        let testSet = try MNISTParser.parse(
            imagesURL: testImgPath, labelsURL: testLblPath, normalize: normalize)

        self.train = ArrayDataset(trainSet)
        self.test = ArrayDataset(testSet)
    }
}

// MARK: - Source URLs

enum MNISTSources {
    // Canonical source: Yann LeCun's page. Many mirrors exist.
    static let base = URL(string: "https://raw.githubusercontent.com/fgnt/mnist/master/")!

    static let urls: (trainImages: URL, trainLabels: URL, testImages: URL, testLabels: URL) = (
        base.appendingPathComponent("train-images-idx3-ubyte.gz"),
        base.appendingPathComponent("train-labels-idx1-ubyte.gz"),
        base.appendingPathComponent("t10k-images-idx3-ubyte.gz"),
        base.appendingPathComponent("t10k-labels-idx1-ubyte.gz")
    )
}

// MARK: - Parser

enum MNISTParserError: Error {
    case badMagic(String)
    case countMismatch
}

enum MNISTParser {

    static func parse(imagesURL: URL, labelsURL: URL, normalize: Bool) throws -> [MNISTExample] {
        let imgData = try Data(contentsOf: imagesURL)
        let lblData = try Data(contentsOf: labelsURL)

        var imgCursor = 0
        var lblCursor = 0

        func readBE32(_ data: Data, _ cursor: inout Int) -> Int {
            let v: UInt32 = data.withUnsafeBytes {
                $0.load(fromByteOffset: cursor, as: UInt32.self)
            }
            cursor += 4
            return Int(UInt32(bigEndian: v))
        }

        // Headers (big-endian), per IDX spec:
        // images: [magic=0x00000803, n, rows, cols], then pixels as bytes
        // labels: [magic=0x00000801, n], then labels as bytes
        let imgMagic = readBE32(imgData, &imgCursor)
        guard imgMagic == 2051 else { throw MNISTParserError.badMagic("images: \(imgMagic)") }
        let imgCount = readBE32(imgData, &imgCursor)
        let rows = readBE32(imgData, &imgCursor)
        let cols = readBE32(imgData, &imgCursor)

        let lblMagic = readBE32(lblData, &lblCursor)
        guard lblMagic == 2049 else { throw MNISTParserError.badMagic("labels: \(lblMagic)") }
        let lblCount = readBE32(lblData, &lblCursor)

        guard imgCount == lblCount else { throw MNISTParserError.countMismatch }

        let imageSize = rows * cols
        var out: [MNISTExample] = []
        out.reserveCapacity(imgCount)

        for _ in 0..<imgCount {
            // Read one image
            let slice = imgData[imgCursor..<(imgCursor + imageSize)]
            imgCursor += imageSize

            // Convert bytes [0,255] to Float [0,1], shape [1,rows,cols]
            let floats: [Float]
            if normalize {
                floats = slice.map { Float($0) / 255.0 }
            } else {
                floats = slice.map { Float($0) }
            }

            let image = Tensor(array: floats, shape: [1, rows, cols], dtype: .float32)

            let labelByte = lblData[lblCursor]
            lblCursor += 1
            let label = Int32(labelByte)

            out.append(MNISTExample(image: image, label: label))
        }

        return out
    }
}
