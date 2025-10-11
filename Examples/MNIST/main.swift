import Foundation
import Torch
import _Differentiation

/// CLI-adjustable hyperparameters and runtime options for the MNIST example.
struct TrainingConfig {
  /// Number of full passes over the training split.
  var epochs: Int = 3
  /// Mini-batch size used during training.
  var batchSize: Int = 128
  /// Mini-batch size used during evaluation.
  var evalBatchSize: Int = 1024
  /// Frequency (in optimizer steps) at which progress is logged.
  var logInterval: Int = 100
  /// Optimizer learning rate.
  var learningRate: Double = 1e-3
  /// Seed for data shuffling so runs are reproducible.
  var shuffleSeed: UInt64 = 0xfeed_cafe
  /// Optional cap on the number of batches processed per epoch (useful for debugging).
  var maxBatchesPerEpoch: Int? = nil
}

/// Parses command-line overrides into a `TrainingConfig`.
/// Accepts arguments such as `--epochs=5` or `--learning-rate=0.0005`.
/// - Returns: Configuration populated with any provided overrides.
func parseConfig() -> TrainingConfig {
  var config = TrainingConfig()
  for argument in CommandLine.arguments.dropFirst() {
    if let value = parseInt(argument, prefix: "--epochs=") {
      config.epochs = value
    } else if let value = parseInt(argument, prefix: "--batch-size=") {
      config.batchSize = value
    } else if let value = parseInt(argument, prefix: "--eval-batch-size=") {
      config.evalBatchSize = value
    } else if let value = parseInt(argument, prefix: "--log-interval=") {
      config.logInterval = value
    } else if let value = parseDouble(argument, prefix: "--learning-rate=") {
      config.learningRate = value
    } else if let value = parseUInt64(argument, prefix: "--seed=") {
      config.shuffleSeed = value
    } else if let value = parseInt(argument, prefix: "--max-batches=") {
      config.maxBatchesPerEpoch = value
    }
  }
  return config
}

/// Attempts to parse an integer value from `argument` when prefixed with `prefix`.
/// - Parameters:
///   - argument: Raw CLI argument.
///   - prefix: Expected argument prefix, including `--` and the trailing `=`.
/// - Returns: Parsed integer when the prefix matches, otherwise `nil`.
func parseInt(_ argument: String, prefix: String) -> Int? {
  guard argument.hasPrefix(prefix) else { return nil }
  return Int(argument.dropFirst(prefix.count))
}

/// Attempts to parse a floating-point value from `argument` when prefixed with `prefix`.
/// - Parameters:
///   - argument: Raw CLI argument.
///   - prefix: Expected argument prefix, including `--` and the trailing `=`.
/// - Returns: Parsed double when the prefix matches, otherwise `nil`.
func parseDouble(_ argument: String, prefix: String) -> Double? {
  guard argument.hasPrefix(prefix) else { return nil }
  return Double(argument.dropFirst(prefix.count))
}

/// Attempts to parse an unsigned 64-bit integer from `argument` when prefixed with `prefix`.
/// - Parameters:
///   - argument: Raw CLI argument.
///   - prefix: Expected argument prefix, including `--` and the trailing `=`.
/// - Returns: Parsed unsigned integer when the prefix matches, otherwise `nil`.
func parseUInt64(_ argument: String, prefix: String) -> UInt64? {
  guard argument.hasPrefix(prefix) else { return nil }
  return UInt64(argument.dropFirst(prefix.count))
}

/// Packs a batch of `MNISTExample` samples into dense tensors.
/// - Parameter batch: Collection of MNIST examples to stack.
/// - Returns: Tuple containing image tensors (`[batch, 1, 28, 28]`) and integer labels.
func makeBatch(_ batch: [MNISTExample]) -> (images: Tensor, labels: Tensor) {
  let images = Tensor.stack(batch.map { $0.image }, dim: 0)
  let labelScalars = batch.map { Int64($0.label) }
  let labels = Tensor(array: labelScalars, shape: [labelScalars.count], dtype: .int64)
  return (images, labels)
}

/// Computes the number of correct predictions in a batch.
/// - Parameters:
///   - logits: Raw model outputs shaped `[batch, numClasses]`.
///   - labels: Ground-truth class indices shaped `[batch]`.
/// - Returns: Tuple containing the count of correct predictions and total samples.
func batchAccuracy(logits: Tensor, labels: Tensor) -> (correct: Int, total: Int) {
  let predictions = logits.argmax(dim: 1)
  let matches = predictions.eq(labels)
  let correctTensor = matches.to(dtype: .int32).sum()
  let correct = Int(correctTensor.toArray(as: Int32.self)[0])
  return (correct, labels.shape[0])
}

/// Evaluates a model over the provided data loader.
/// - Parameters:
///   - model: Model to evaluate (any `Layer` conformer).
///   - loader: Loader yielding evaluation batches.
/// - Returns: Mean softmax-cross-entropy loss and top-1 accuracy.
func evaluate<Model: Layer>(
  _ model: Model,
  loader: DataLoader<ArrayDataset<MNISTExample>>
) -> (loss: Double, accuracy: Double) {
  var totalLoss: Double = 0
  var totalCorrect = 0
  var totalSamples = 0

  for batch in loader {
    let (images, labels) = makeBatch(batch)
    let logits = model(images as! Model.Input)
    let loss = softmaxCrossEntropy(logits: logits as! Tensor, labels: labels)
    let lossValue = loss.toArray(as: Float.self)[0]
    let (correct, batchTotal) = batchAccuracy(logits: logits as! Tensor, labels: labels)

    totalLoss += Double(lossValue) * Double(batchTotal)
    totalCorrect += correct
    totalSamples += batchTotal
  }

  let meanLoss = totalLoss / Double(totalSamples)
  let accuracy = Double(totalCorrect) / Double(totalSamples)
  return (meanLoss, accuracy)
}

do {
  let config = parseConfig()
  print("TaylorTorch MNIST • epochs: \(config.epochs), batch: \(config.batchSize)")

  print("Preparing MNIST dataset…")
  let mnist = try MNIST()
  print("Loaded MNIST with \(mnist.train.count) training and \(mnist.test.count) test samples")

  /// Batched accessor used to evaluate on the held-out test split.
  let testLoader = DataLoader(
    dataset: mnist.test,
    batchSize: config.evalBatchSize,
    shuffle: false,
    dropLast: false,
    seed: nil
  )

  /*
  /// Shallow convolutional stem replicating the classic LeNet-style feature extractor.
  var model = Sequential {
    Conv2D(kaimingUniformInChannels: 1, outChannels: 6, kernelSize: (5, 5))
    ReLU()
    MaxPool2D(kernelSize: (2, 2))
    Conv2D(kaimingUniformInChannels: 6, outChannels: 16, kernelSize: (5, 5))
    ReLU()
    MaxPool2D(kernelSize: (2, 2))
  
    Flatten()
    Dense.relu(inputSize: 16 * 4 * 4, outputSize: 120)
    Dense.relu(inputSize: 120, outputSize: 84)
    Dense(inputSize: 84, outputSize: 10)
  }*/

  var model = Sequential {
    // ── Block 1 ──────────────────────────────────────────────────────────────
    Conv2D(
      kaimingUniformInChannels: 1, outChannels: 32,
      kernelSize: (3, 3), padding: (1, 1))
    Dropout(probability: 0.025)  // Flax applies dropout before BN here
    BatchNorm(featureCount: 32)  // NCHW => axis 1
    ReLU()
    AvgPool2D(kernelSize: (2, 2), stride: (2, 2))

    // ── Block 2 ──────────────────────────────────────────────────────────────
    Conv2D(
      kaimingUniformInChannels: 32, outChannels: 64,
      kernelSize: (3, 3), padding: (1, 1))
    BatchNorm(featureCount: 64)
    ReLU()
    AvgPool2D(kernelSize: (2, 2), stride: (2, 2))

    // ── Head ─────────────────────────────────────────────────────────────────
    Flatten(startDim: 1)  // [N, 64*7*7] = [N, 3136]
    Linear(inputSize: 64 * 7 * 7, outputSize: 256)
    Dropout(probability: 0.025)
    ReLU()
    Linear(inputSize: 256, outputSize: 10)
  }

  /// SGD optimizer configured with the requested learning rate.
  let optimizer = SGD(for: model, learningRate: Float(config.learningRate))
  let stepsPerEpoch = (mnist.train.count + config.batchSize - 1) / config.batchSize
  let startTime = Date()

  for epoch in 1...config.epochs {
    /// Shuffled loader that feeds training batches for the current epoch.
    let trainLoader = DataLoader(
      dataset: mnist.train,
      batchSize: config.batchSize,
      shuffle: true,
      dropLast: false,
      seed: config.shuffleSeed &+ UInt64(epoch)
    )

    var runningLoss: Double = 0
    var runningCorrect = 0
    var runningSamples = 0
    var blockLoss: Double = 0
    var blockCorrect = 0
    var blockSamples = 0
    var step = 0

    for batch in trainLoader {
      step += 1
      if let limit = config.maxBatchesPerEpoch, step > limit { break }
      let (images, labels) = makeBatch(batch)

      let (lossTensor, pullback) = valueWithPullback(at: model) { current -> Tensor in
        let logits = current(images)
        return softmaxCrossEntropy(logits: logits, labels: labels)
      }
      let grad = pullback(Tensor(1.0, dtype: .float32))

      let logits = model(images)
      let (correct, batchTotal) = batchAccuracy(logits: logits, labels: labels)
      let lossValue = lossTensor.toArray(as: Float.self)[0]

      optimizer.update(&model, along: grad)

      runningLoss += Double(lossValue) * Double(batchTotal)
      runningCorrect += correct
      runningSamples += batchTotal

      blockLoss += Double(lossValue) * Double(batchTotal)
      blockCorrect += correct
      blockSamples += batchTotal

      if step % config.logInterval == 0 {
        let avgLoss = blockLoss / Double(blockSamples)
        let avgAcc = Double(blockCorrect) / Double(blockSamples)
        let elapsed = Date().timeIntervalSince(startTime)
        print(
          String(
            format: "epoch %d • step %d/%d • loss %.4f • acc %.2f%% • %.1fs",
            epoch, step, stepsPerEpoch, avgLoss, avgAcc * 100, elapsed))
        blockLoss = 0
        blockCorrect = 0
        blockSamples = 0
      }
    }

    if blockSamples > 0 {
      let avgLoss = blockLoss / Double(blockSamples)
      let avgAcc = Double(blockCorrect) / Double(blockSamples)
      let elapsed = Date().timeIntervalSince(startTime)
      print(
        String(
          format: "epoch %d • step %d/%d • loss %.4f • acc %.2f%% • %.1fs",
          epoch, step, stepsPerEpoch, avgLoss, avgAcc * 100, elapsed))
    }

    let epochLoss = runningLoss / Double(runningSamples)
    let epochAcc = Double(runningCorrect) / Double(runningSamples)
    let (valLoss, valAcc) = evaluate(model, loader: testLoader)
    print(
      String(
        format:
          "epoch %d done • train loss %.4f • train acc %.2f%% • val loss %.4f • val acc %.2f%%",
        epoch, epochLoss, epochAcc * 100, valLoss, valAcc * 100))
  }

  let totalElapsed = Date().timeIntervalSince(startTime)
  print(String(format: "Training finished in %.1fs", totalElapsed))
} catch {
  let message = "MNIST example failed: \(error)\n"
  if let data = message.data(using: .utf8) {
    FileHandle.standardError.write(data)
  }
  exit(EXIT_FAILURE)
}
