# Core infrastructure

The files in this directory provide the protocol extensions, threading
utilities, and device helpers that glue together TaylorTorch’s Layer/optimizer
stack. They rarely need to be touched directly by end users, but they underpin
most higher-level APIs.

## Files at a glance

| File | Highlights |
| --- | --- |
| `VectorProtocol.swift` | Defines a lightweight vector-space protocol that all tangent vectors conform to. Provides scalar arithmetic (`adding`, `scaled(by:)`) and reflection helpers used by layers and optimisers. |
| `PointwiseMultiplicative.swift` | Adds element-wise multiplication requirements to tangent vectors so optimisers like Adam can combine gradients and moments safely. |
| `EuclideanDifferentiable.swift` | Supplements Swift’s `Differentiable` protocol with euclidean conveniences (e.g. a default `differentiableVectorView`) used across modules. |
| `Mergeable.swift` | Trait for types whose tangents can be merged (e.g. recurrent states); exposed via protocols instead of relying on compiler synthesis. |
| `KeyPathIterable.swift` | Swift reflection helpers for enumerating writable key paths—critical for walking tensor leaves when copying to devices or updating optimizer state. |
| `CopyableToDevice.swift` | Defines the `CopyableToDevice` trait and its default implementation, enabling model/optimizer state to be moved between devices. |
| `Threading.swift` | Minimal thread-local storage and context management used by `Context` and the optimiser stack. |

## Why it matters

- **Optimiser support** – `VectorProtocol`, `PointwiseMultiplicative`, and
  `KeyPathIterable` power the optimiser implementations (`SGD`, `Adam`). They
  let TaylorTorch walk nested structures, scale gradients, and maintain moments
  without fragile casting.
- **Differentiation ergonomics** – manual tangent vectors built throughout the
  library lean on these protocols for default implementations (`add`, `scale`),
  ensuring they compose with Swift autodiff.
- **Device copying** – `CopyableToDevice` and the key-path helpers are how the
  library migrates tensors between CPU/GPU (even though only CPU is currently
  wired up, the architecture assumes device portability).
- **Context management** – thread-local utilities underpin `Context.local` and
  `withLearningPhase`, keeping training/inference orthogonal to the rest of the
  API.

Taken together, these core utilities form the low-level framework that the
Modules/Layers and Modules/Graph packages rely on. Even though most end users
won’t touch them directly, they are the reason higher-level TaylorTorch code can
stay purely in Swift without having to micromanage C++ interop details.
