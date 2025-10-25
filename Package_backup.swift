// swift-tools-version:6.1
import PackageDescription
import Foundation

// Define constants for paths to avoid repetition
// Check for environment variables first (for container/CI), fallback to local paths
let swiftToolchainDir = ProcessInfo.processInfo.environment["SWIFT_TOOLCHAIN_DIR"] ??
    "/Users/pedro/Library/Developer/Toolchains/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a.xctoolchain/usr"
let pytorchInstallDir = ProcessInfo.processInfo.environment["PYTORCH_INSTALL_DIR"] ??
    "/Users/pedro/programming/pytorch/install"

let sdkRoot = ProcessInfo.processInfo.environment["SDKROOT"]

func firstExistingPath(_ candidates: [String?]) -> String? {
    let fileManager = FileManager.default
    for candidate in candidates {
        if let path = candidate, fileManager.fileExists(atPath: path) {
            return path
        }
    }
    return nil
}

// Derived paths
let swiftLibDir = "\(swiftToolchainDir)/lib/swift"
let swiftClangIncludeDir = "\(swiftLibDir)/clang/include"
let swiftIncludeDir = "\(swiftToolchainDir)/include"

let swiftBridgingIncludeDir: String? = {
    let fileManager = FileManager.default
    
    // Check environment variable first
    if let envPath = ProcessInfo.processInfo.environment["SWIFT_BRIDGING_INCLUDE_DIR"] {
        if fileManager.fileExists(atPath: "\(envPath)/swift/bridging") ||
           fileManager.fileExists(atPath: "\(envPath)/swift/bridging.h") {
            print("Found bridging header via SWIFT_BRIDGING_INCLUDE_DIR: \(envPath)")
            return envPath
        }
    }
    
    let candidates: [String?] = [
        swiftClangIncludeDir,  // /usr/lib/swift/clang/include (Linux) or toolchain equivalent
        swiftIncludeDir,
        "\(swiftLibDir)/clang/include",
        swiftLibDir,
        "\(swiftLibDir)/swiftToCxx",
        "/usr/lib/swift/clang/include",  // Linux system install
        "/usr/include",
        sdkRoot.map { "\($0)/usr/include" }
    ]
    
    for candidate in candidates {
        guard let base = candidate else { continue }
        let bridgingHeader = "\(base)/swift/bridging"
        let bridgingHeaderWithExt = "\(base)/swift/bridging.h"
        if fileManager.fileExists(atPath: bridgingHeader) || 
           fileManager.fileExists(atPath: bridgingHeaderWithExt) {
            print("Found Swift bridging header at: \(base)")
            return base
        }
    }
    
    print("WARNING: Swift bridging header not found. Searched:")
    for candidate in candidates {
        if let path = candidate {
            print("  - \(path)/swift/bridging")
        }
    }
    
    return nil
}()

let sdkIncludeDir = sdkRoot.map { "\($0)/usr/include" }

// Enhanced Darwin module map detection with better fallback paths
let darwinModuleMap: String? = {
    let candidates: [String?] = [
        sdkRoot.map { "\($0)/usr/include/module.modulemap" },
        "\(swiftClangIncludeDir)/module.modulemap",
        "\(swiftIncludeDir)/module.modulemap",
        // Check Swift toolchain's Darwin overlay
        "\(swiftLibDir)/macosx/Darwin.swiftmodule/module.modulemap"
    ]
    return firstExistingPath(candidates)
}()

// Enhanced C standard library module map detection
let cStandardLibraryModuleMap: String? = {
    let candidates: [String?] = [
        sdkRoot.map { "\($0)/usr/include/c_standard_library.modulemap" },
        "\(swiftClangIncludeDir)/c_standard_library.modulemap",
        // Some toolchains might have it here
        "\(swiftClangIncludeDir)/module.modulemap"
    ]
    return firstExistingPath(candidates)
}()

let pytorchIncludeDir = "\(pytorchInstallDir)/include"
let pytorchApiIncludeDir = "\(pytorchInstallDir)/include/torch/csrc/api/include"
let pytorchLibDir = "\(pytorchInstallDir)/lib"

// Common compiler & linker settings
var commonSwiftSettings: [SwiftSetting] = [
    .interoperabilityMode(.Cxx),
    .unsafeFlags(["-Xcc", "-I\(swiftIncludeDir)"]),
    .unsafeFlags(["-Xcc", "-I\(swiftClangIncludeDir)"]),
    .unsafeFlags(["-Xcc", "-DSWIFT_INTEROP_ENABLED"]),
    .unsafeFlags(["-Xcc", "-I\(pytorchIncludeDir)"]),
    .unsafeFlags(["-Xcc", "-I\(pytorchApiIncludeDir)"]),
]

// Add bridging header if found
if let swiftBridgingIncludeDir {
    commonSwiftSettings.append(.unsafeFlags(["-Xcc", "-I\(swiftBridgingIncludeDir)"]))
}

// CRITICAL FIX FOR MACOS: Use -isystem for SDK includes to ensure system headers are found
// This fixes the "no such module 'tgmath_h'" error
if let sdkIncludeDir {
    commonSwiftSettings.append(.unsafeFlags(["-Xcc", "-isystem", "-Xcc", sdkIncludeDir]))
}

// Add explicit SDK flag for Swift compiler (helps with module resolution on macOS)
if let sdkRoot {
    commonSwiftSettings.append(.unsafeFlags(["-sdk", sdkRoot]))
}

// Add module maps
if let darwinModuleMap {
    commonSwiftSettings.append(.unsafeFlags(["-Xcc", "-fmodule-map-file=\(darwinModuleMap)"]))
}
if let cStandardLibraryModuleMap {
    commonSwiftSettings.append(.unsafeFlags(["-Xcc", "-fmodule-map-file=\(cStandardLibraryModuleMap)"]))
}

let commonLinkerSettings: [LinkerSetting] = [
    .unsafeFlags(["-L", pytorchLibDir]),
    .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", pytorchLibDir]),
    .linkedLibrary("c10"),
    .linkedLibrary("torch"),
    .linkedLibrary("torch_cpu"),
]

var atenCxxSettings: [CXXSetting] = [
    .unsafeFlags(["-I", swiftIncludeDir]),
    .unsafeFlags(["-I", swiftClangIncludeDir]),
    .unsafeFlags(["-I", swiftLibDir]),
    .unsafeFlags(["-I", pytorchIncludeDir]),
    .unsafeFlags(["-I", pytorchApiIncludeDir]),
]
if let swiftBridgingIncludeDir {
    atenCxxSettings.append(.unsafeFlags(["-I", swiftBridgingIncludeDir]))
}
if let sdkIncludeDir {
    atenCxxSettings.append(.unsafeFlags(["-I", sdkIncludeDir]))
}
if let darwinModuleMap {
    atenCxxSettings.append(.unsafeFlags(["-fmodule-map-file=\(darwinModuleMap)"]))
}
if let cStandardLibraryModuleMap {
    atenCxxSettings.append(.unsafeFlags(["-fmodule-map-file=\(cStandardLibraryModuleMap)"]))
}

var atenCxxDoctestSettings: [CXXSetting] = [
    .define("DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES"),
    .unsafeFlags(["-I", swiftIncludeDir]),
    .unsafeFlags(["-I", swiftClangIncludeDir]),
    .unsafeFlags(["-I", pytorchIncludeDir]),
    .unsafeFlags(["-I", pytorchApiIncludeDir]),
    .unsafeFlags(["-std=c++17"]),
]
if let swiftBridgingIncludeDir {
    atenCxxDoctestSettings.append(.unsafeFlags(["-I", swiftBridgingIncludeDir]))
}
if let sdkIncludeDir {
    atenCxxDoctestSettings.append(.unsafeFlags(["-I", sdkIncludeDir]))
}
if let darwinModuleMap {
    atenCxxDoctestSettings.append(.unsafeFlags(["-fmodule-map-file=\(darwinModuleMap)"]))
}
if let cStandardLibraryModuleMap {
    atenCxxDoctestSettings.append(.unsafeFlags(["-fmodule-map-file=\(cStandardLibraryModuleMap)"]))
}

let package = Package(
    name: "TaylorTorch",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "Torch", targets: ["Torch"]),
        .executable(name: "MNISTExample", targets: ["MNISTExample"]),
        .executable(name: "ANKIExample", targets: ["ANKIExample"]),
        .executable(name: "KARATEExample", targets: ["KARATEExample"]),
    ],
    targets: [
        // ----------------- C++ Targets -----------------
        .target(
            name: "ATenCXX",
            path: "Sources/ATenCXX",
            publicHeadersPath: "include",
            cxxSettings: atenCxxSettings
        ),
        .executableTarget(
            name: "ATenCXXDoctests",
            dependencies: ["ATenCXX"],
            path: "Sources/ATenCXXDoctests",
            cxxSettings: atenCxxDoctestSettings,
            linkerSettings: [
                .unsafeFlags(["-L", pytorchLibDir]),
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", pytorchLibDir]),
                .linkedLibrary("c10"),
                .linkedLibrary("torch_cpu"),
            ]
        ),

        // ----------------- Swift Targets -----------------
        .target(
            name: "Torch",
            dependencies: ["ATenCXX"],
            exclude: [
                "readme.md", "ATen/readme.md", "ATen/Core/Tensor/readme.md", "Core/readme.md",
                "Optimizers/readme.md", "Modules/readme.md",
                "Modules/Context/readme.md", "Modules/Layers/readme.md", "Modules/Graph/readme.md",
                "Data/README.md",
            ],
            swiftSettings: commonSwiftSettings,
            linkerSettings: commonLinkerSettings
        ),

        // ----------------- Example Targets -----------------
        .executableTarget(
            name: "MNISTExample",
            dependencies: ["Torch"],
            path: "Examples/MNIST",
            swiftSettings: commonSwiftSettings,
            linkerSettings: commonLinkerSettings
        ),
        .executableTarget(
            name: "ANKIExample",
            dependencies: ["Torch"],
            path: "Examples/ANKI",
            swiftSettings: commonSwiftSettings,
            linkerSettings: commonLinkerSettings
        ),
        .executableTarget(
            name: "KARATEExample",
            dependencies: ["Torch"],
            path: "Examples/KARATE",
            swiftSettings: commonSwiftSettings,
            linkerSettings: commonLinkerSettings
        ),

        // ----------------- Test Targets -----------------
        .testTarget(
            name: "TensorTests",
            dependencies: ["Torch"],
            path: "Tests/TensorTests",
            swiftSettings: commonSwiftSettings,
            linkerSettings: commonLinkerSettings
        ),
        .testTarget(
            name: "TorchTests",
            dependencies: ["Torch"],
            path: "Tests/TorchTests",
            swiftSettings: commonSwiftSettings,
            linkerSettings: commonLinkerSettings
        ),
    ],
    cxxLanguageStandard: .cxx17
)