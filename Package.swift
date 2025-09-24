// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import Foundation

let package = Package(
    name: "TaylorTorch",
    // ✅ Best Practice: Add a platforms field
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "Demo", targets: ["Demo"])
    ],
    targets: [
        .target(
            name: "ATenCXX",
            path: "Sources/ATenCXX",
            publicHeadersPath: "include",
            
              cxxSettings: [
                    .unsafeFlags(["-I", "/Users/pedro/programming/pytorch/install/include"]),
                    .unsafeFlags(["-I", "/Users/pedro/programming/pytorch/install/include/torch/csrc/api/include"]),
                ]
            
        ),
       .target(
            name: "ATen",
            dependencies: ["ATenCXX"],
            exclude: ["Core/Tensor/readme.md"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(["-Xcc", "-I/Users/pedro/programming/pytorch/install/include"]),
                .unsafeFlags(["-Xcc", "-I/Users/pedro/programming/pytorch/install/include/torch/csrc/api/include"]),
            ]
        ),
        .executableTarget(
            name: "Demo",
            dependencies: ["ATen"],
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ],
            linkerSettings: [
                .unsafeFlags(["-L", "/Users/pedro/programming/pytorch/install/lib"]),
                .unsafeFlags([
                    "-Xlinker", "-rpath",
                    "-Xlinker", "/Users/pedro/programming/pytorch/install/lib"
                ]),
                // ✅ Recommendation: Uncomment ATen for robust linking
                .linkedLibrary("c10"),
                //.linkedLibrary("Aten"),
                .linkedLibrary("torch"),
                .linkedLibrary("torch_cpu")
            ]
        ),
        .testTarget(
            name: "TensorTests",
            dependencies: ["ATen"],
            path: "Tests/TensorTests",
            swiftSettings: [
                .interoperabilityMode(.Cxx)
            ],
            linkerSettings: [
                .unsafeFlags(["-L", "/Users/pedro/programming/pytorch/install/lib"]),
                .unsafeFlags([
                    "-Xlinker", "-rpath",
                    "-Xlinker", "/Users/pedro/programming/pytorch/install/lib"
                ]),
                // ✅ Recommendation: Uncomment ATen for robust linking
                .linkedLibrary("c10"),
                //.linkedLibrary("Aten"),
                .linkedLibrary("torch"),
                .linkedLibrary("torch_cpu")
            ]
        )
    ],
    cxxLanguageStandard: .cxx17
)
