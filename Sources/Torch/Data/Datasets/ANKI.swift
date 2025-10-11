import Foundation

// MARK: - Unzip Utility (added for handling .zip archives, similar to Tar)
// (Assuming this is already present from previous code)

public enum UnzipError: Error { case extractionFailed(String) }

public enum Unzip {
  @discardableResult
  public static func extract(archive: URL, to destination: URL) throws -> URL {
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
    proc.arguments = ["-o", archive.path, "-d", destination.path]

    let stderrPipe = Pipe()
    proc.standardError = stderrPipe

    try proc.run()
    proc.waitUntilExit()
    if proc.terminationStatus != 0 {
      let err =
        String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)
        ?? "unknown error"
      throw UnzipError.extractionFailed(err)
    }
    return destination
  }
}

// MARK: - FileHandle Extension for Line Reading
// (Assuming this is already present)

extension FileHandle {
  func readLine() -> String? {
    var buf = Data()
    while true {
      do {
        if let ch = try read(upToCount: 1), ch.count > 0 {
          buf.append(ch)
          if ch.first == 10 {  // \n
            break
          }
        } else {
          if buf.count == 0 { return nil }
          break
        }
      } catch {
        return nil
      }
    }
    return String(data: buf, encoding: .utf8)?.trimmingCharacters(in: .newlines)
  }
}

// MARK: - Anki Example
public struct AnkiExample {
  public let english: String
  public let spanish: String
  public init(english: String, spanish: String) {
    self.english = english
    self.spanish = spanish
  }
}

// MARK: - Tatoeba English to Spanish Loader (Smaller Dataset)

/// Loader for the smaller Tatoeba-based English-Spanish parallel corpus from manythings.org.
/// Downloads & extracts spa-eng.zip (~142k sentence pairs) to ~/.taylortorch/datasets/tatoeba-en-es by default.
/// Subsamples to maxSamples sentence pairs (default 100k) if specified; full size is small enough for demos.
public struct TatoebaEnglishToSpanish {
  public let train: ArrayDataset<AnkiExample>

  public init(root: URL? = nil, maxSamples: Int? = 100_000, download: Bool = true) throws {
    let rootDir = (root ?? DataHome.root).appendingPathComponent("tatoeba-en-es", isDirectory: true)
    try Downloader.ensureDir(rootDir)

    let url = URL(string: "https://www.manythings.org/anki/spa-eng.zip")!
    let archive = rootDir.appendingPathComponent("spa-eng.zip")
    if download {
      _ = try Downloader.fetch(url: url, to: archive)
    }

    let extractedDir = rootDir.appendingPathComponent("extracted", isDirectory: true)
    if !FileManager.default.fileExists(atPath: extractedDir.path) {
      _ = try Unzip.extract(archive: archive, to: extractedDir)
    }

    let candidatePaths = [
      extractedDir.appendingPathComponent("spa-eng/spa.txt"),
      extractedDir.appendingPathComponent("spa.txt")
    ]
    guard let tsvPath = candidatePaths.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
      throw NSError(
        domain: NSCocoaErrorDomain,
        code: NSFileReadNoSuchFileError,
        userInfo: [
          NSFilePathErrorKey: candidatePaths.first!.path,
          NSLocalizedDescriptionKey: "spa.txt not found after extracting spa-eng.zip"
        ])
    }

    // Parse tab-separated lines, up to maxSamples
    let pairs = try Self.readPairedFromTSV(file: tsvPath, max: maxSamples)
    let examples = pairs.map { AnkiExample(english: $0.0, spanish: $0.1) }

    self.train = ArrayDataset(examples)
  }

  // MARK: - Parsing Helper

  private static func readPairedFromTSV(file: URL, max: Int?) throws -> [(String, String)] {
    let handle = try FileHandle(forReadingFrom: file)

    var pairs: [(String, String)] = []
    if let m = max {
      pairs.reserveCapacity(m)
      print("[Tatoeba] Subsamping to \(m) sentence pairs.")
    } else {
      print("[Tatoeba] Loading full dataset (~142k pairs).")
    }

    var lineNum = 0
    while true {
      guard let line = handle.readLine() else { break }
      let parts = line.components(separatedBy: "\t")
      if parts.count >= 2 {
        let eng = parts[0].trimmingCharacters(in: .whitespaces)
        let spa = parts[1].trimmingCharacters(in: .whitespaces)
        if !eng.isEmpty && !spa.isEmpty {
          pairs.append((eng, spa))
          lineNum += 1
          if let m = max, lineNum >= m { break }
        }
      }
    }

    handle.closeFile()

    print("[Tatoeba] Loaded \(pairs.count) sentence pairs.")
    return pairs
  }
}
