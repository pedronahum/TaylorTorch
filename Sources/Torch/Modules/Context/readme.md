# Context

Layers that need to alter behaviour between training and inference share a common entry point: `ForwardContext`.

- **ForwardContext.swift** – Defines the simple value type (`training: Bool`) passed down through contextual `call(_:context:)` overloads. Layers such as Dropout, BatchNorm, and Sequential use it to decide whether to update running statistics or sample stochastic masks.

If you add new stateful modules, accept a `ForwardContext` so they slot into existing training loops without additional glue code.
