// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "TaylorTorch",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "Torch", targets: ["Torch"])
    ],
    targets: [
        // ----------------- C++ Targets -----------------
        .target(
            name: "ATenCXX",
            path: "Sources/ATenCXX",
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags([
                    "-I",
                    "/Users/pedro/Library/Developer/Toolchains/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a.xctoolchain/usr/lib/swift",
                ]),
                .unsafeFlags(["-I", "/Users/pedro/programming/pytorch/install/include"]),
                .unsafeFlags([
                    "-I", "/Users/pedro/programming/pytorch/install/include/torch/csrc/api/include",
                ]),
            ]
        ),
        .executableTarget(
            name: "ATenCXXDoctests",
            dependencies: ["ATenCXX"],
            path: "Sources/ATenCXXDoctests",
            cxxSettings: [
                .unsafeFlags([
                    "-I",
                    "/Users/pedro/Library/Developer/Toolchains/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a.xctoolchain/usr/include",
                ]),
                .unsafeFlags(["-I", "/Users/pedro/programming/pytorch/install/include"]),
                .unsafeFlags([
                    "-I", "/Users/pedro/programming/pytorch/install/include/torch/csrc/api/include",
                ]),
                .unsafeFlags(["-std=c++17"]),
            ],
            linkerSettings: [

                .unsafeFlags(["-L", "/Users/pedro/programming/pytorch/install/lib"]),
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker",
                    "/Users/pedro/programming/pytorch/install/lib",
                ]),
                .linkedLibrary("c10"),
                .linkedLibrary("torch_cpu"),
            ]
        ),

        // ----------------- Swift Targets -----------------

        .target(
            name: "Torch",
            dependencies: ["ATenCXX"],
            exclude: ["ATen/Core/Tensor/readme.md"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags([
                    "-Xcc",
                    "-I/Users/pedro/Library/Developer/Toolchains/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a.xctoolchain/usr/include",
                ]),
                .unsafeFlags(["-Xcc", "-DSWIFT_INTEROP_ENABLED"]),
                // ✅ FIX: Added the missing include paths for the Swift compiler's Clang importer.
                .unsafeFlags(["-Xcc", "-I/Users/pedro/programming/pytorch/install/include"]),
                .unsafeFlags([
                    "-Xcc",
                    "-I/Users/pedro/programming/pytorch/install/include/torch/csrc/api/include",
                ]),
            ]
        ),

        // ----------------- Test Targets -----------------
        .testTarget(
            name: "TensorTests",
            dependencies: ["Torch"],
            path: "Tests/TensorTests",
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags([
                    "-Xcc",
                    "-I/Users/pedro/Library/Developer/Toolchains/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a.xctoolchain/usr/include",
                ]),
                //.unsafeFlags(["-Xcc", "-I/Library/Developer/CommandLineTools/usr/include"]),
                .unsafeFlags(["-Xcc", "-DSWIFT_INTEROP_ENABLED"]),
                // ✅ FIX: Added the missing include paths for the Swift compiler's Clang importer.
                .unsafeFlags(["-Xcc", "-I/Users/pedro/programming/pytorch/install/include"]),
                .unsafeFlags([
                    "-Xcc",
                    "-I/Users/pedro/programming/pytorch/install/include/torch/csrc/api/include",
                ]),
            ],
            linkerSettings: [

                .unsafeFlags(["-L", "/Users/pedro/programming/pytorch/install/lib"]),
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker",
                    "/Users/pedro/programming/pytorch/install/lib",
                ]),
                .linkedLibrary("c10"),
                .linkedLibrary("torch"),
                .linkedLibrary("torch_cpu"),
            ],
        ),
        .testTarget(
            name: "TorchTests",
            dependencies: ["Torch"],
            path: "Tests/TorchTests",
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags([
                    "-Xcc",
                    "-I/Users/pedro/Library/Developer/Toolchains/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a.xctoolchain/usr/include",
                ]),
                //.unsafeFlags(["-Xcc", "-I/Library/Developer/CommandLineTools/usr/include"]),
                .unsafeFlags(["-Xcc", "-DSWIFT_INTEROP_ENABLED"]),
                // ✅ FIX: Added the missing include paths for the Swift compiler's Clang importer.
                .unsafeFlags(["-Xcc", "-I/Users/pedro/programming/pytorch/install/include"]),
                .unsafeFlags([
                    "-Xcc",
                    "-I/Users/pedro/programming/pytorch/install/include/torch/csrc/api/include",
                ]),
            ],
            linkerSettings: [

                .unsafeFlags(["-L", "/Users/pedro/programming/pytorch/install/lib"]),
                .unsafeFlags([
                    "-Xlinker", "-rpath", "-Xlinker",
                    "/Users/pedro/programming/pytorch/install/lib",
                ]),
                .linkedLibrary("c10"),
                .linkedLibrary("torch"),
                .linkedLibrary("torch_cpu"),
            ],
        ),
    ],
    cxxLanguageStandard: .cxx17
)
