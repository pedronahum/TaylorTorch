// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
@_spi(Reflection) import Swift
import _Differentiation

// Re-export RealModuleDifferentiable so users get access to differentiable
// scalar math functions (Double.exp, Float.log, etc.) without SIL linker crashes.
// See KNOWN_ISSUES.md for details on why this is needed on Linux.
@_exported import RealModuleDifferentiable

// Global lock and caches to ensure thread safety for reflection operations and
// to avoid recomputing reflection results for better performance.
private let reflectionLock = NSRecursiveLock()
nonisolated(unsafe) private var listFieldsCache: [ObjectIdentifier: Any] = [:]
nonisolated(unsafe) private var differentiableFieldsCache: [ObjectIdentifier: Any] = [:]

/// A thread-safe, cached function that lists all fields of a given type using reflection.
// listFields(of:)
func listFields<Root>(of type: Root.Type) -> [(String, PartialKeyPath<Root>)] {
  let typeID = ObjectIdentifier(type)
  reflectionLock.lock()
  defer { reflectionLock.unlock() }
  if let cached = listFieldsCache[typeID] as? [(String, PartialKeyPath<Root>)] { return cached }

  var out = [(String, PartialKeyPath<Root>)]()
  _forEachFieldWithKeyPath(of: type, options: []) { cname, kp in
    out.append((String(validatingCString: cname)!, kp))
    return true
  }

  listFieldsCache[typeID] = out
  return out
}

extension Differentiable {
  /// A thread-safe, cached computed property that provides the names and key paths
  /// for fields of a type that are present in its tangent vector.
  // Differentiable.differentiableFields
  static var differentiableFields: [(String, PartialKeyPath<Self>, PartialKeyPath<TangentVector>)] {
    let typeID = ObjectIdentifier(Self.self)
    reflectionLock.lock()
    defer { reflectionLock.unlock() }
    if let cached = differentiableFieldsCache[typeID]
      as? [(String, PartialKeyPath<Self>, PartialKeyPath<TangentVector>)]
    {
      return cached
    }

    let tangentFields = listFields(of: TangentVector.self)
    var i = 0
    var out = [(String, PartialKeyPath<Self>, PartialKeyPath<TangentVector>)]()
    _forEachFieldWithKeyPath(of: Self.self, options: []) { cname, kp in
      if i >= tangentFields.count { return false }
      let name = String(validatingCString: cname)!
      if tangentFields[i].0 == name {
        out.append((name, kp, tangentFields[i].1))
        i += 1
      }
      return true
    }
    differentiableFieldsCache[typeID] = out
    return out
  }
}

public protocol _EuclideanDifferentiable {
  static func _copyWeightsToTangentVector<Root: Differentiable>(
    _ base: Root, _ out: inout Root.TangentVector,
    _ keyPathBase: PartialKeyPath<Root>,
    _ keyPathOut: PartialKeyPath<Root.TangentVector>
  )
}

public protocol EuclideanDifferentiable: _EuclideanDifferentiable & Differentiable {
  var differentiableVectorView: TangentVector { get }
  func _copyWeightsToTangentVector(_ out: inout TangentVector)
}

extension EuclideanDifferentiable where TangentVector == Self {
  public var differentiableVectorView: TangentVector { _read { yield self } }
  public func _copyWeightsToTangentVector(_ out: inout TangentVector) {
    out = differentiableVectorView
  }
}

extension EuclideanDifferentiable {
  public static func _copyWeightsToTangentVector<Root: Differentiable>(
    _ base: Root, _ out: inout Root.TangentVector,
    _ keyPathBase: PartialKeyPath<Root>,
    _ keyPathOut: PartialKeyPath<Root.TangentVector>
  ) {
    guard let keyPathBase = keyPathBase as? WritableKeyPath<Root, Self>,
      let keyPathOut = keyPathOut as? WritableKeyPath<Root.TangentVector, Self.TangentVector>
    else {
      fatalError("Failure to build differentiableVectorView via reflection: \(Self.self)")
    }
    base[keyPath: keyPathBase]._copyWeightsToTangentVector(&out[keyPath: keyPathOut])
  }
  public var differentiableVectorView: TangentVector {
    var out = TangentVector.zero
    _copyWeightsToTangentVector(&out)
    return out
  }
  public func _copyWeightsToTangentVector(_ out: inout TangentVector) {
    // This now safely accesses the cached `differentiableFields`.
    for (_, keyPathBase, keyPathOut) in Self.differentiableFields {
      let valueType = type(of: keyPathBase).valueType
      if let valueType = valueType as? _EuclideanDifferentiable.Type {
        valueType._copyWeightsToTangentVector(self, &out, keyPathBase, keyPathOut)
      } else {
        fatalError("Failure to build differentiableVectorView via reflection: \(valueType)")
      }
    }
  }
}

extension Float: EuclideanDifferentiable {}
extension Double: EuclideanDifferentiable {}

extension Array: EuclideanDifferentiable & _EuclideanDifferentiable
where Element: EuclideanDifferentiable {
  public func _copyWeightsToTangentVector(_ out: inout TangentVector) {
    out = Array.DifferentiableView.TangentVector(self.map { $0.differentiableVectorView })
  }
}
extension Array.DifferentiableView: EuclideanDifferentiable & _EuclideanDifferentiable
where Element: EuclideanDifferentiable {
  public func _copyWeightsToTangentVector(_ out: inout TangentVector) {
    out = Array.DifferentiableView.TangentVector(self.base.map { $0.differentiableVectorView })
  }
}

extension RNNCellInput: _EuclideanDifferentiable
where Input: EuclideanDifferentiable, State: EuclideanDifferentiable {}

extension RNNCellOutput: _EuclideanDifferentiable
where Output: EuclideanDifferentiable, State: EuclideanDifferentiable {}

extension Tensor: EuclideanDifferentiable & _EuclideanDifferentiable {
  public static func _copyWeightsToTangentVector<Root: Differentiable>(
    _ base: Root, _ out: inout Root.TangentVector,
    _ keyPathBase: PartialKeyPath<Root>,
    _ keyPathOut: PartialKeyPath<Root.TangentVector>
  ) {
    guard
      let baseKeyPath = keyPathBase as? WritableKeyPath<Root, Tensor>,
      let outKeyPath = keyPathOut as? WritableKeyPath<Root.TangentVector, Tensor>
    else {
      fatalError("Failure to build differentiableVectorView via reflection: \(Tensor.self)")
    }
    out[keyPath: outKeyPath] = base[keyPath: baseKeyPath]
  }

  public var differentiableVectorView: TangentVector { self }

  public func _copyWeightsToTangentVector(_ out: inout TangentVector) {
    out = self
  }
}
