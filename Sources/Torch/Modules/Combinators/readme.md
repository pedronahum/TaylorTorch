# Combinators

Reusable higher-order modules that orchestrate the flow between layers live here.

- **Identity.swift** – Passes inputs through unchanged; useful when toggling optional layers in a builder.
- **Residual.swift** – Implements residual connections (`x + f(x)`) for any inner layer that takes and returns a tensor of matching shape.
- **ParallelAdd.swift** – Routes the same input through multiple branches and sums the outputs.
- **Concat.swift** – Concatenates branch outputs along a chosen dimension to build multi-stream feature extractors.

Each combinator conforms to `Layer`, so they compose seamlessly with other modules and expose differentiable tangents when internal branches carry parameters.
