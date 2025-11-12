import Foundation
// swift-tools-version:6.1
import PackageDescription

// Define constants for paths to avoid repetition
// Check for environment variables first (for container/CI), fallback to local paths
let swiftToolchainDir =
    ProcessInfo.processInfo.environment["SWIFT_TOOLCHAIN_DIR"]
    ?? "/Users/pedro/Library/Developer/Toolchains/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a.xctoolchain/usr"
let pytorchInstallDir =
    ProcessInfo.processInfo.environment["PYTORCH_INSTALL_DIR"]
    ?? "/Users/pedro/programming/pytorch/install"

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
    let candidates: [String?] = [
        ProcessInfo.processInfo.environment["SWIFT_BRIDGING_INCLUDE_DIR"],
        swiftIncludeDir,
        swiftClangIncludeDir,
        "\(swiftLibDir)/swiftToCxx",
        swiftLibDir,
        sdkRoot.map { "\($0)/usr/include" },
    ]
    let fileManager = FileManager.default
    for candidate in candidates {
        guard let base = candidate else { continue }
        let bridgingHeader = "\(base)/swift/bridging"
        let bridgingHeaderWithExt = "\(base)/swift/bridging.h"
        if fileManager.fileExists(atPath: bridgingHeader)
            || fileManager.fileExists(atPath: bridgingHeaderWithExt)
        {
            return base
        }
    }
    return nil
}()
let sdkIncludeDir = sdkRoot.map { "\($0)/usr/include" }
let darwinModuleMap = firstExistingPath([
    sdkRoot.map { "\($0)/usr/include/module.modulemap" },
    "\(swiftClangIncludeDir)/module.modulemap",
    "\(swiftIncludeDir)/module.modulemap",
])
let cStandardLibraryModuleMap = firstExistingPath([
    sdkRoot.map { "\($0)/usr/include/c_standard_library.modulemap" },
    "\(swiftClangIncludeDir)/c_standard_library.modulemap",
])
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
if let swiftBridgingIncludeDir {
    commonSwiftSettings.append(.unsafeFlags(["-Xcc", "-I\(swiftBridgingIncludeDir)"]))
}

if let sdkIncludeDir {
    commonSwiftSettings.append(.unsafeFlags(["-Xcc", "-I\(sdkIncludeDir)"]))
}
if let darwinModuleMap {
    commonSwiftSettings.append(.unsafeFlags(["-Xcc", "-fmodule-map-file=\(darwinModuleMap)"]))
}
if let cStandardLibraryModuleMap {
    commonSwiftSettings.append(
        .unsafeFlags(["-Xcc", "-fmodule-map-file=\(cStandardLibraryModuleMap)"]))
}

// On Linux, use --whole-archive to force inclusion of all PyTorch operator symbols
// These symbols are in static registration sections that get optimized out without this flag
#if os(Linux)
    let commonLinkerSettings: [LinkerSetting] = [
        // CRITICAL: Every flag must be passed through -Xlinker to prevent swiftc reordering
        .unsafeFlags([
            "-L", pytorchLibDir,
            "-Xlinker", "-rpath", "-Xlinker", pytorchLibDir,
            // C++ libraries - using libstdc++ (what PyTorch actually uses in Docker)
            "-Xlinker", "-lstdc++",
            "-Xlinker", "-lm",
            // PyTorch libraries in --whole-archive block
            "-Xlinker", "--whole-archive",
            "-Xlinker", "-ltorch_cpu",
            "-Xlinker", "-ltorch",
            "-Xlinker", "-lc10",
            "-Xlinker", "--no-whole-archive",
            // Additional dependencies
            "-Xlinker", "-ltorch_global_deps",
        ])
    ]
#else
    let commonLinkerSettings: [LinkerSetting] = [
        .unsafeFlags(["-L", pytorchLibDir]),
        .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", pytorchLibDir]),
        .linkedLibrary("torch_cpu"),
        .linkedLibrary("torch"),
        .linkedLibrary("c10"),
    ]
#endif

// Platform-specific linker settings for ATenCXXDoctests
#if os(Linux)
    let platformLinkerSettings: [LinkerSetting] = [
        .linkedLibrary("stdc++"),
        .linkedLibrary("m"),
    ]

    // ATenCXXDoctests - needs --whole-archive wrapper like main target
    let atenDoctestsLinkerSettings: [LinkerSetting] = [
        .unsafeFlags([
            "-L", pytorchLibDir,
            "-Xlinker", "-rpath", "-Xlinker", pytorchLibDir,
            "-Xlinker", "-lstdc++",
            "-Xlinker", "-lm",
            // PyTorch libraries in --whole-archive block
            "-Xlinker", "--whole-archive",
            "-Xlinker", "-ltorch_cpu",
            "-Xlinker", "-ltorch",
            "-Xlinker", "-lc10",
            "-Xlinker", "--no-whole-archive",
            "-Xlinker", "-ltorch_global_deps",
        ])
    ]
#else
    let platformLinkerSettings: [LinkerSetting] = []

    // On macOS, keep original structure - it works fine!
    let atenDoctestsLinkerSettings: [LinkerSetting] =
        [
            .unsafeFlags(["-L", pytorchLibDir]),
            .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", pytorchLibDir]),
            .linkedLibrary("c10"),
            .linkedLibrary("torch_cpu"),
        ] + platformLinkerSettings
#endif

// Combined linker settings for Torch target
let allLinkerSettings = commonLinkerSettings

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

// Platform-specific CXX settings for Linux
#if os(Linux)
    let platformCxxSettings: [CXXSetting] = [
        // Use libstdc++ (what PyTorch actually uses in Docker)
        .unsafeFlags(["-stdlib=libstdc++"]),
        // Use old ABI (ABI=0) to match Docker PyTorch build
        .define("_GLIBCXX_USE_CXX11_ABI", to: "1"),
    ]
#else
    let platformCxxSettings: [CXXSetting] = []
#endif

// Combined CXX settings
let allAtenCxxSettings = atenCxxSettings + platformCxxSettings

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

// Combined CXX doctest settings
let allAtenCxxDoctestSettings = atenCxxDoctestSettings + platformCxxSettings

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
    dependencies: [
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0")
    ],
    targets: [
        // ----------------- C++ Targets -----------------
        .target(
            name: "ATenCXX",
            path: "Sources/ATenCXX",
            publicHeadersPath: "include",
            cxxSettings: allAtenCxxSettings
        ),
        .executableTarget(
            name: "ATenCXXDoctests",
            dependencies: ["ATenCXX"],
            path: "Sources/ATenCXXDoctests",
            cxxSettings: allAtenCxxDoctestSettings,
            linkerSettings: atenDoctestsLinkerSettings
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
            cxxSettings: allAtenCxxSettings,
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),

        // ----------------- Example Targets -----------------
        .executableTarget(
            name: "MNISTExample",
            dependencies: ["Torch"],
            path: "Examples/MNIST",
            cxxSettings: allAtenCxxSettings,
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),
        .executableTarget(
            name: "ANKIExample",
            dependencies: ["Torch"],
            path: "Examples/ANKI",
            cxxSettings: allAtenCxxSettings,
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),
        .executableTarget(
            name: "KARATEExample",
            dependencies: ["Torch"],
            path: "Examples/KARATE",
            cxxSettings: allAtenCxxSettings,
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),
        // ----------------- Test Targets -----------------
        .testTarget(
            name: "TensorTests",
            dependencies: ["Torch"],
            path: "Tests/TensorTests",
            cxxSettings: allAtenCxxSettings,
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),
        .testTarget(
            name: "TorchTests",
            dependencies: ["Torch"],
            path: "Tests/TorchTests",
            cxxSettings: allAtenCxxSettings,
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),
    ],
    cxxLanguageStandard: .cxx17
)
