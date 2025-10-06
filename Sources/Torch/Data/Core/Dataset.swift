
import Foundation

// MARK: - Dataset Protocols

public protocol Dataset {
    associatedtype Element
    var count: Int { get }
    subscript(_ index: Int) -> Element { get }
}

/// A simple concrete Dataset backed by an Array.
public struct ArrayDataset<Element>: Dataset, RandomAccessCollection {
    public typealias Index = Int
    public let elements: [Element]

    public init(_ elements: [Element]) {
        self.elements = elements
    }

    // Collection
    public var startIndex: Int { elements.startIndex }
    public var endIndex: Int { elements.endIndex }
    public func index(after i: Int) -> Int { elements.index(after: i) }
    public func index(before i: Int) -> Int { elements.index(before: i) }
    public subscript(_ index: Int) -> Element { elements[index] }

    // Dataset
    public var count: Int { elements.count }
}

// MARK: - Transforms

/// A light-weight, composable transform for dataset elements.
public struct Compose<T> {
    public let transforms: [(T) -> T]
    public init(_ transforms: [(T) -> T]) { self.transforms = transforms }
    public func callAsFunction(_ x: T) -> T { transforms.reduce(x) { $1($0) } }
}
