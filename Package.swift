// swift-tools-version:6.1
import PackageDescription
import Foundation

// Define constants for paths to avoid repetition
// Check for environment variables first (for container/CI), fallback to local paths
let swiftToolchainDir = ProcessInfo.processInfo.environment["SWIFT_TOOLCHAIN_DIR"] ??
    "/Users/pedro/Library/Developer/Toolchains/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a.xctoolchain/usr"
let pytorchInstallDir = ProcessInfo.processInfo.environment["PYTORCH_INSTALL_DIR"] ??
    "/Users/pedro/programming/pytorch/install"

// Derived paths
let swiftLibDir = "\(swiftToolchainDir)/lib/swift"
let swiftClangIncludeDir = "\(swiftLibDir)/clang/include"
let swiftIncludeDir = "\(swiftToolchainDir)/include"
let swiftBridgingIncludeDir = "\(swiftIncludeDir)/swift"
let pytorchIncludeDir = "\(pytorchInstallDir)/include"
let pytorchApiIncludeDir = "\(pytorchInstallDir)/include/torch/csrc/api/include"
let pytorchLibDir = "\(pytorchInstallDir)/lib"

// Common compiler & linker settings
let commonSwiftSettings: [SwiftSetting] = [
    .interoperabilityMode(.Cxx),
    .unsafeFlags(["-Xcc", "-I\(swiftIncludeDir)"]),
    .unsafeFlags(["-Xcc", "-I\(swiftBridgingIncludeDir)"]),
    .unsafeFlags(["-Xcc", "-I\(swiftClangIncludeDir)"]),
    .unsafeFlags(["-Xcc", "-DSWIFT_INTEROP_ENABLED"]),
    .unsafeFlags(["-Xcc", "-I\(pytorchIncludeDir)"]),
    .unsafeFlags(["-Xcc", "-I\(pytorchApiIncludeDir)"]),
]

let commonLinkerSettings: [LinkerSetting] = [
    .unsafeFlags(["-L", pytorchLibDir]),
    .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", pytorchLibDir]),
    .linkedLibrary("c10"),
    .linkedLibrary("torch"),
    .linkedLibrary("torch_cpu"),
]

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
            cxxSettings: [
                .unsafeFlags(["-I", swiftIncludeDir]),
                .unsafeFlags(["-I", swiftBridgingIncludeDir]),
                .unsafeFlags(["-I", swiftClangIncludeDir]),
                .unsafeFlags(["-I", swiftLibDir]),
                .unsafeFlags(["-I", pytorchIncludeDir]),
                .unsafeFlags(["-I", pytorchApiIncludeDir]),
            ]
        ),
        .executableTarget(
            name: "ATenCXXDoctests",
            dependencies: ["ATenCXX"],
            path: "Sources/ATenCXXDoctests",
            cxxSettings: [
                .define("DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES"),
                .unsafeFlags(["-I", swiftIncludeDir]),
                .unsafeFlags(["-I", swiftBridgingIncludeDir]),
                .unsafeFlags(["-I", swiftClangIncludeDir]),
                .unsafeFlags(["-I", pytorchIncludeDir]),
                .unsafeFlags(["-I", pytorchApiIncludeDir]),
                .unsafeFlags(["-std=c++17"]),
            ],
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
