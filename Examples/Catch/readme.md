# Catch Example

This example is adapted from the Catch game in the [Swift Models](https://github.com/tensorflow/swift-models) repository. The original environment and agent implementation were reworked to run on top of TaylorTorch and its optimizer stack.

Key differences from the upstream version:
- Uses TaylorTorch tensors, layers, and the AdamW optimizer instead of the Swift for TensorFlow runtime.
- Replaces the Philox RNG with a lightweight deterministic SplitMix64 generator written in Swift.
- Normalizes observations and rewards as tensors so they flow directly through the TaylorTorch autodiff pipeline.

Run the example with `swift run CatchExample`. The training loop prints rolling win rates while the simplified policy updates on every timestep.

Credit for the task design and original Swift implementation goes to the Swift for TensorFlow team.
