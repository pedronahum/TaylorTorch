
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

enum DownloadError: Error { case network(String) }

public enum DataHome {
    /// Default cache root: ~/.taylortorch/datasets
    public static var root: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".taylortorch/datasets", isDirectory: true)
    }
}

public struct Downloader {

    public static func ensureDir(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    @discardableResult
    public static func fetch(url: URL, to dest: URL) throws -> URL {
        try ensureDir(dest.deletingLastPathComponent())
        if FileManager.default.fileExists(atPath: dest.path) {
            print("[Downloader] Using cached \(dest.lastPathComponent)")
            return dest
        }
        print("[Downloader] Fetching \(url.absoluteString)")
        let start = Date()
        // Small & simple for MVP: Data(contentsOf:) works on macOS & Linux via FoundationNetworking.
        let data = try Data(contentsOf: url)
        let elapsed = Date().timeIntervalSince(start)
        try data.write(to: dest, options: .atomic)
        let sizeMB = Double(data.count) / 1_048_576.0
        print(String(format: "[Downloader] Saved %@ (%.1f MB) in %.1fs", dest.lastPathComponent, sizeMB, elapsed))
        return dest
    }
}
