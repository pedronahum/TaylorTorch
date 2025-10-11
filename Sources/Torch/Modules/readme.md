# Modules overview

This subtree assembles everything above the tensor primitives into reusable,
differentiable components. It also wires training/inference context through the
layer stack so higher-level examples can stay in pure Swift.

## Files and folders

| Path | Highlights |
| --- | --- |
| `Layer.swift` | Declares the `Layer` protocol (alias of `Module`) that every trainable component adopts. Provides contextual calls, default `forward`, and ensures parameters are discoverable via key paths. |
| `Initializers.swift` | Shared tensor initializer routines (e.g. Glorot/Xavier) used by layers when constructing weight tensors. |
| `Layers/` | Concrete layers: dense blocks, convolutions/pooling, embeddings, normalisation, activations, dropout, attention, recurrent cells, and the `Sequential` builder. See `Layers/readme.md` for details. |
| `Graph/` | Graph neural network primitives—`Graphs`, batching helpers, segment ops, and the message-passing `GraphNetwork`. Documented in `Graph/readme.md`. |
| `Context/` | Houses the thread-local `ForwardContext` and helpers that toggle training vs inference. Several layers read `Context.local` to adjust behaviour. |

## Why this structure matters

- **Single contract** – Every module conforms to `Layer`, so optimisers, builders,
  and compositional helpers can treat them uniformly.
- **Separation of concerns** – Core algebra lives under `Torch/Core`, tensor
  bindings under `Torch/ATen`; `Modules/` focuses solely on differentiable
  building blocks.
- **Composable design** – With `Sequential`, result builders, and the composable
  graph layers, you can express complex architectures purely in Swift without
  touching C++ interop.

Most downstream code imports `Torch`, picks the modules needed, and plugs them
into optimisers defined elsewhere in the project. The README files inside each
subfolder provide deeper dives into the available components.
