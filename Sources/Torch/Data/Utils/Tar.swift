import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

public enum TarError: Error { case extractionFailed(String) }

public enum Tar {
  /// Extracts a .tar or .tar.gz archive using the system `tar`.
  /// If the destination already contains extracted files, this is a no-op.
  @discardableResult
  public static func extract(archive: URL, to destination: URL) throws -> URL {
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    proc.arguments = ["tar", "-xzf", archive.path, "-C", destination.path]

    let stderrPipe = Pipe()
    proc.standardError = stderrPipe

    try proc.run()
    proc.waitUntilExit()
    if proc.terminationStatus != 0 {
      let err =
        String(
          data: stderrPipe.fileHandleForReading.readDataToEndOfFile(),
          encoding: .utf8) ?? "unknown error"
      throw TarError.extractionFailed(err)
    }
    return destination
  }
}
