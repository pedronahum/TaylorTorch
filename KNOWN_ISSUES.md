# TaylorTorch Known Issues

This document describes known issues encountered when building TaylorTorch on Linux with Swift's automatic differentiation, and the workarounds implemented.

> **Note**: These issues are **specific to Linux (Ubuntu 24.04)**. macOS builds do not experience these problems and can use standard C library math functions without issues.

## Swift SIL Linker Assertion Failures with C Library Math Functions

### Problem

When using C library math functions (`exp`, `log`, `sqrt`, `pow`, `powf`) in code that undergoes Swift automatic differentiation, the Swift compiler crashes with a SIL (Swift Intermediate Language) linker assertion:

```
Assertion failed: googGV->isDeclaration() && "global variable already has initializer"
```

or

```
LLVM ERROR: Global is external, but doesn't have external or weak linkage
```

These errors occur because the Swift autodiff system generates derivative code that references C library function symbols in ways that conflict with Swift's SIL linker expectations on Linux.

### Affected Functions

- `exp()`, `expf()` - exponential
- `log()`, `logf()`, `log1p()` - logarithm
- `sqrt()`, `sqrtf()` - square root
- `pow()`, `powf()` - power

### Solution: Use swift-numerics-differentiable

The recommended solution is to use the [swift-numerics-differentiable](https://github.com/differentiable-swift/swift-numerics-differentiable) package, which provides pure Swift implementations of math functions with properly registered derivatives for autodiff.

#### Installation

Add to `Package.swift`:
```swift
dependencies: [
    .package(url: "https://github.com/differentiable-swift/swift-numerics-differentiable", from: "1.3.0"),
],
```

Add dependency to targets:
```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "RealModuleDifferentiable", package: "swift-numerics-differentiable"),
    ],
),
```

#### Usage

TaylorTorch re-exports `RealModuleDifferentiable`, so simply `import Torch` gives you access to differentiable scalar math:

```swift
import Torch

// Instead of Foundation.exp (crashes):
let result = Double.exp(x)

// Instead of Foundation.log (crashes):
let result = Double.log(x)

// Instead of log1p (crashes):
let result = Double.log(onePlus: x)

// Instead of pow (crashes):
let result = Double.pow(x, y)
```

These functions work correctly with Swift autodiff because they are pure Swift implementations with derivatives properly registered via `@derivative` attributes.

### Alternative Workarounds

If you cannot use swift-numerics-differentiable, these workarounds are still available:

#### 1. Replace `sqrt` with `.squareRoot()`

Swift's native `FloatingPoint.squareRoot()` method works correctly with autodiff:

```swift
// Before (causes SIL crash)
let a = sqrt(x)

// After (works)
let a = x.squareRoot()
```

#### 2. Replace `pow(x, -0.5)` with `1.0 / x.squareRoot()`

```swift
// Before (causes SIL crash)
let a = powf(x, -0.5)
let b = powf(x, -1.5)

// After (works)
let a = 1.0 / x.squareRoot()
let b = 1.0 / (x.squareRoot() * x)
```

### Files Modified

- `Package.swift` - Added `swift-numerics-differentiable` dependency
- `Sources/Torch/Core/EuclideanDifferentiable.swift` - Re-exports `RealModuleDifferentiable` so users get it automatically
- `Examples/ANKI/main.swift` - Replaced `powf` with `.squareRoot()`
- `Sources/Torch/Modules/Initializers.swift` - Replaced `sqrt` with `.squareRoot()`
- `Tests/TensorTests/TensorMathTests.swift` - Uses `Double.exp()` from RealModuleDifferentiable
- `Tests/TorchTests/LossTests.swift` - Uses `Double.exp()` and `Double.log(onePlus:)` from RealModuleDifferentiable
- `Tests/TorchTests/ActivationModulesTests.swift` - Replaced `Foundation.sqrt` with `.squareRoot()`

---

## Swift Autodiff Crash with For-In Loops

### Problem

Swift's automatic differentiation crashes when a `for-in` loop is used inside a `valueWithPullback` closure on Linux:

```
LLVM ERROR: Global is external, but doesn't have external or weak linkage
```

### Example

```swift
// This crashes the compiler
let (value, pullback) = valueWithPullback(at: input) { tensor in
    var current = tensor
    for dim in dims {  // <-- for-in loop causes crash
        current = current.sum(dim: dim)
    }
    return current
}
```

### Workaround

Comment out or disable tests that use for-in loops inside differentiated closures. This is a Swift compiler bug that needs to be fixed upstream.

### Files Modified

- `Tests/TensorTests/TensorAxisSugarDifferentiationTests.swift` - Commented out `axisReductionsGradientMatchIntegerVariants` test

---

## Adam Optimizer KeyPath Crashes with Complex Models

### Problem

The Adam optimizer crashes at runtime when used with complex nested models (like Transformers) on Linux. The crash occurs in `recursivelyAllWritableKeyPaths` when iterating over the TangentVector structure.

```
Swift/KeyPath.swift:1051: Fatal error: Could not extract a String from KeyPath Swift.KeyPath<...>
```

This appears to be related to how Swift handles KeyPath operations on complex nested generic types on Linux.

### Workaround

Use SGD with momentum instead of Adam for complex models:

```swift
// Instead of:
let opt = Adam(for: model, learningRate: 0.01)

// Use:
var opt = SGD(for: model, learningRate: 0.01, momentum: 0.9)
```

### Files Modified

- `Examples/ANKI/main.swift` - Switched from Adam to SGD optimizer
- `Examples/KARATE/main.swift` - Switched from Adam to SGD with LR 0.001 (higher rates cause NaN)
- `Tests/TorchTests/SequentialLayerTests.swift` - Disabled `sequential_parameter_keypaths_and_flattening` test

---

## Environment Variables Required for Building

### Problem

Building TaylorTorch fails with `'swift/bridging' file not found` if environment variables are not set.

### Solution

Set these environment variables before building:

```bash
export SWIFT_TOOLCHAIN_DIR="/path/to/swiftly/toolchains/main-snapshot-2025-11-03/usr"
export PYTORCH_INSTALL_DIR="/opt/pytorch"
export PATH="/path/to/swiftly/bin:$PATH"
```

Or source the environment files created by the install script:

```bash
source /etc/profile.d/swift.sh
source /etc/profile.d/pytorch.sh
```

---

## Platform

These issues are specific to:
- **OS**: Linux (Ubuntu 24.04)
- **Swift**: Development snapshots (main-snapshot-2025-11-03)
- **C++ Standard Library**: libstdc++ (GCC 13)

macOS builds are not affected by most of these issues.
