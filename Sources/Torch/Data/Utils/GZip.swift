
import Foundation
#if canImport(Compression)
import Compression
#endif

public enum GZip {

    enum Error: Swift.Error {
        case initializationFailed
        case processFailed(Int32)
    }

    public static func gunzipFile(at url: URL) throws -> Data {
        if url.pathExtension.lowercased() != "gz" {
            return try Data(contentsOf: url)
        }

        let compressed = try Data(contentsOf: url)

        #if canImport(Compression)
        if let decoded = try? decompressUsingCompression(compressed) {
            return decoded
        } else {
            fputs("[GZip] Compression framework decompression failed, falling back to gunzip.\n", stderr)
        }
        #endif
        return try gunzipViaProcess(url: url)
    }

    #if canImport(Compression)
    private static func decompressUsingCompression(_ compressed: Data) throws -> Data {
        return try compressed.withUnsafeBytes { rawBuffer -> Data in
            guard let baseAddress = rawBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                return Data()
            }

            var stream = compression_stream()
            var status = compression_stream_init(&stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB)
            guard status != COMPRESSION_STATUS_ERROR else { throw Error.initializationFailed }
            defer { compression_stream_destroy(&stream) }

            stream.src_ptr = baseAddress
            stream.src_size = compressed.count

            let chunkSize = 64 * 1024
            let dstPointer = UnsafeMutablePointer<UInt8>.allocate(capacity: chunkSize)
            defer { dstPointer.deallocate() }

            var output = Data()
            repeat {
                stream.dst_ptr = dstPointer
                stream.dst_size = chunkSize

                let flags: Int32 = stream.src_size == 0 ? Int32(COMPRESSION_STREAM_FINALIZE.rawValue) : 0
                status = compression_stream_process(&stream, flags)
                let produced = chunkSize - stream.dst_size
                if produced > 0 {
                    output.append(dstPointer, count: produced)
                }
            } while status == COMPRESSION_STATUS_OK

            guard status == COMPRESSION_STATUS_END else {
                throw Error.processFailed(status.rawValue)
            }

            return output
        }
    }
    #endif

    private static func gunzipViaProcess(url: URL) throws -> Data {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = ["gunzip", "-c", url.path]

        let pipe = Pipe()
        proc.standardOutput = pipe
        let errorPipe = Pipe()
        proc.standardError = errorPipe
        try proc.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()

        let status = proc.terminationStatus
        if status != 0 {
            let errData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            let msg = String(data: errData, encoding: .utf8) ?? "unknown"
            throw NSError(domain: "GZip", code: Int(status), userInfo: [NSLocalizedDescriptionKey: "gunzip failed: \(msg)"])
        }
        return data
    }
}
