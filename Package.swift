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

// On Linux, configure Swift to use libstdc++ properly
#if os(Linux)
commonSwiftSettings += [
    // Add libstdc++ include paths before Swift's clang includes
    .unsafeFlags(["-Xcc", "-isystem/usr/include/c++/13"]),
    .unsafeFlags(["-Xcc", "-isystem/usr/include/x86_64-linux-gnu/c++/13"]),
    .unsafeFlags(["-Xcc", "-isystem/usr/include/c++/13/backward"]),
    .unsafeFlags(["-Xcc", "-isystem/usr/lib/gcc/x86_64-linux-gnu/13/include"]),
    .unsafeFlags(["-Xcc", "-isystem/usr/include"]),
    .unsafeFlags(["-Xcc", "-isystem/usr/include/x86_64-linux-gnu"]),
]
#endif

// On Linux, use --whole-archive to force inclusion of all PyTorch operator symbols
// These symbols are in static registration sections that get optimized out without this flag
#if os(Linux)
    let commonLinkerSettings: [LinkerSetting] = [
        // CRITICAL: Every flag must be passed through -Xlinker to prevent swiftc reordering
        .unsafeFlags([
            "-L", pytorchLibDir,
            "-Xlinker", "-rpath", "-Xlinker", pytorchLibDir,
            // C++ libraries - using libstdc++ (what PyTorch is built with)
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
        // libstdc++ headers
        .unsafeFlags(["-isystem", "/usr/include/c++/13"]),
        .unsafeFlags(["-isystem", "/usr/include/x86_64-linux-gnu/c++/13"]),
        .unsafeFlags(["-isystem", "/usr/include/c++/13/backward"]),
        // GCC internal includes
        .unsafeFlags(["-isystem", "/usr/lib/gcc/x86_64-linux-gnu/13/include"]),
        // System C includes
        .unsafeFlags(["-isystem", "/usr/include"]),
        .unsafeFlags(["-isystem", "/usr/include/x86_64-linux-gnu"]),
    ]
#else
    let platformCxxSettings: [CXXSetting] = []
#endif

// Combined CXX settings - platform settings first for correct include order
let allAtenCxxSettings = platformCxxSettings + atenCxxSettings

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

// Combined CXX doctest settings - platform settings first for correct include order
let allAtenCxxDoctestSettings = platformCxxSettings + atenCxxDoctestSettings

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
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0"),
        .package(url: "https://github.com/differentiable-swift/swift-numerics-differentiable", from: "1.3.0"),
    ],
    targets: {
        var targets: [Target] = [
            // ----------------- C++ Targets -----------------
            .target(
                name: "ATenCXX",
                path: "Sources/ATenCXX",
                publicHeadersPath: "include",
                cxxSettings: allAtenCxxSettings
            ),
        ]

        // ATenCXXDoctests 
        
        targets.append(
            .executableTarget(
                name: "ATenCXXDoctests",
                dependencies: ["ATenCXX"],
                path: "Sources/ATenCXXDoctests",
                cxxSettings: allAtenCxxDoctestSettings,
                linkerSettings: atenDoctestsLinkerSettings
            )
        )
        

        // ----------------- Swift Targets -----------------
        targets += [
        .target(
            name: "Torch",
            dependencies: [
                "ATenCXX",
                .product(name: "RealModuleDifferentiable", package: "swift-numerics-differentiable"),
            ],
            exclude: [
                "readme.md", "ATen/readme.md", "ATen/Core/Tensor/readme.md", "Core/readme.md",
                "Optimizers/readme.md", "Modules/readme.md",
                "Modules/Context/readme.md", "Modules/Layers/readme.md", "Modules/Graph/readme.md",
                "Data/README.md",
            ],
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),

        // ----------------- Example Targets -----------------
        .executableTarget(
            name: "MNISTExample",
            dependencies: ["Torch"],
            path: "Examples/MNIST",
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),
        .executableTarget(
            name: "ANKIExample",
            dependencies: ["Torch"],
            path: "Examples/ANKI",
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),
        .executableTarget(
            name: "KARATEExample",
            dependencies: ["Torch"],
            path: "Examples/KARATE",
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),

        // ----------------- Test Targets -----------------
        .testTarget(
            name: "TensorTests",
            dependencies: [
                "Torch",
                .product(name: "RealModuleDifferentiable", package: "swift-numerics-differentiable"),
            ],
            path: "Tests/TensorTests",
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),
        .testTarget(
            name: "TorchTests",
            dependencies: [
                "Torch",
                .product(name: "RealModuleDifferentiable", package: "swift-numerics-differentiable"),
            ],
            path: "Tests/TorchTests",
            swiftSettings: commonSwiftSettings,
            linkerSettings: allLinkerSettings
        ),
        ]

        return targets
    }(),
    cxxLanguageStandard: .cxx17
)
