import Foundation
import Torch
import _Differentiation

// ---- Config & CLI parsing ----
struct TrainingConfig {
  var epochs: Int = 10
  var batchSize: Int = 128
  var evalBatchSize: Int = 1024
  var logInterval: Int = 100
  var learningRate: Double = 1e-3
  var shuffleSeed: UInt64 = 0xfeed_cafe
  var maxBatchesPerEpoch: Int? = nil
}
func parseConfig() -> TrainingConfig {
  var c = TrainingConfig()
  for arg in CommandLine.arguments.dropFirst() {
    if let v = parseInt(arg, prefix: "--epochs=") {
      c.epochs = v
    } else if let v = parseInt(arg, prefix: "--batch-size=") {
      c.batchSize = v
    } else if let v = parseInt(arg, prefix: "--eval-batch-size=") {
      c.evalBatchSize = v
    } else if let v = parseInt(arg, prefix: "--log-interval=") {
      c.logInterval = v
    } else if let v = parseDouble(arg, prefix: "--learning-rate=") {
      c.learningRate = v
    } else if let v = parseUInt64(arg, prefix: "--seed=") {
      c.shuffleSeed = v
    } else if let v = parseInt(arg, prefix: "--max-batches=") {
      c.maxBatchesPerEpoch = v
    }
  }
  return c
}
func parseInt(_ a: String, prefix: String) -> Int? {
  a.hasPrefix(prefix) ? Int(a.dropFirst(prefix.count)) : nil
}
func parseDouble(_ a: String, prefix: String) -> Double? {
  a.hasPrefix(prefix) ? Double(a.dropFirst(prefix.count)) : nil
}
func parseUInt64(_ a: String, prefix: String) -> UInt64? {
  a.hasPrefix(prefix) ? UInt64(a.dropFirst(prefix.count)) : nil
}

// ---- Batching & metrics ----
struct CIFARBatch {
  let images: Tensor
  let labels: Tensor
}
func makeBatch(_ batch: [CIFAR10Example]) -> CIFARBatch {
  let images = Tensor.stack(batch.map { $0.image }, dim: 0)  // [N, 3, 32, 32]
  let labelScalars = batch.map { Int64($0.label) }
  let labels = Tensor(array: labelScalars, shape: [labelScalars.count], dtype: .int64)
  return CIFARBatch(images: images, labels: labels)
}
func batchAccuracy(logits: Tensor, labels: Tensor) -> (correct: Int, total: Int) {
  let predictions = logits.argmax(dim: 1)
  let matches = predictions.eq(labels)
  let correctTensor = matches.to(dtype: .int32).sum()
  let correct = Int(correctTensor.toArray(as: Int32.self)[0])
  return (correct, labels.shape[0])
}
func evaluate<Model: Layer>(_ model: Model, loader: DataLoader<ArrayDataset<CIFAR10Example>>) -> (
  loss: Double, accuracy: Double
) {
  var totalLoss: Double = 0
  var totalCorrect = 0
  var total = 0
  let evalContext = ForwardContext(training: false)
  for batch in loader {
    let b = makeBatch(batch)
    let logits = model.call(b.images, context: evalContext)
    let loss = softmaxCrossEntropy(logits: logits, labels: b.labels)
    let (correct, n) = batchAccuracy(logits: logits, labels: b.labels)
    totalLoss += Double(loss.toArray(as: Float.self)[0]) * Double(n)
    totalCorrect += correct
    total += n
  }
  return (totalLoss / Double(total), Double(totalCorrect) / Double(total))
}

// ---- Main ----
do {
  let cfg = parseConfig()
  print("TaylorTorch CIFAR-10 • epochs: \(cfg.epochs), batch: \(cfg.batchSize)")

  print("Preparing CIFAR-10 dataset…")
  let cifar = try CIFAR10(normalize: true)
  print("Loaded CIFAR-10 with \(cifar.train.count) training and \(cifar.test.count) test samples")

  let testLoader = DataLoader(
    dataset: cifar.test, batchSize: cfg.evalBatchSize, shuffle: false, dropLast: false, seed: nil)

  // Compose the entire network inside one SequentialBlock.
  let model = SequentialBlock {
    // ---- Feature extractor (LeNet-ish) ----
    Conv2D.kaimingUniform(inC: 3, outC: 6, kH: 5, kW: 5, padding: .valid)
    ReLU()
    Dropout(rate: 0.3)

    MaxPool2D(kernel: (2, 2))
    Conv2D.kaimingUniform(inC: 6, outC: 16, kH: 5, kW: 5, padding: .valid)
    ReLU()
    Dropout(rate: 0.3)

    MaxPool2D(kernel: (2, 2))
    Conv2D.kaimingUniform(inC: 16, outC: 32, kH: 3, kW: 3, padding: .valid)
    ReLU()
    Dropout(rate: 0.3)

    // ---- Classifier ----
    Flatten()
    Dense(inFeatures: 32 * 3 * 3, outFeatures: 120, dtype: .float32, device: .cpu)
    ReLU()
    Dropout(rate: 0.5)

    Dense(inFeatures: 120, outFeatures: 84, dtype: .float32, device: .cpu)
    ReLU()
    Dropout(rate: 0.5)

    Dense(
      inFeatures: 84,
      outFeatures: 10,
      dtype: .float32,
      device: .cpu
    )
  }

  var modelCopy = model  // mutable copy for training
  var opt = AdamW(for: modelCopy, learningRate: cfg.learningRate)
  let stepsPerEpoch = (cifar.train.count + cfg.batchSize - 1) / cfg.batchSize
  let start = Date()

  for epoch in 1...cfg.epochs {
    let trainLoader = DataLoader(
      dataset: cifar.train,
      batchSize: cfg.batchSize,
      shuffle: true,
      dropLast: false,
      seed: cfg.shuffleSeed &+ UInt64(epoch)
    )
    let trainContext = ForwardContext(training: true)
    let inferenceContext = ForwardContext(training: false)

    var runningLoss: Double = 0
    var runningCorrect = 0
    var runningTotal = 0
    var blockLoss: Double = 0
    var blockCorrect = 0
    var blockTotal = 0
    var step = 0

    for batch in trainLoader {
      step += 1
      if let limit = cfg.maxBatchesPerEpoch, step > limit { break }

      let b = makeBatch(batch)
      let (lossTensor, pullback) = valueWithPullback(at: modelCopy) { current -> Tensor in
        let logits = current.call(b.images, context: trainContext)
        return softmaxCrossEntropy(logits: logits, labels: b.labels)
      }
      let grad = pullback(Tensor(1.0, dtype: .float32))
      opt.update(&modelCopy, along: grad)

      let logits = modelCopy.call(b.images, context: inferenceContext)
      let (correct, n) = batchAccuracy(logits: logits, labels: b.labels)
      let lossVal = Double(lossTensor.toArray(as: Float.self)[0])

      runningLoss += lossVal * Double(n)
      runningCorrect += correct
      runningTotal += n
      blockLoss += lossVal * Double(n)
      blockCorrect += correct
      blockTotal += n

      if step % cfg.logInterval == 0 {
        let elapsed = Date().timeIntervalSince(start)
        print(
          String(
            format: "epoch %d • step %d/%d • loss %.4f • acc %.2f%% • %.1fs",
            epoch, step, stepsPerEpoch, blockLoss / Double(blockTotal),
            Double(blockCorrect) / Double(blockTotal) * 100, elapsed))
        blockLoss = 0
        blockCorrect = 0
        blockTotal = 0
      }
    }

    if blockTotal > 0 {
      let elapsed = Date().timeIntervalSince(start)
      print(
        String(
          format: "epoch %d • step %d/%d • loss %.4f • acc %.2f%% • %.1fs",
          epoch, step, stepsPerEpoch, blockLoss / Double(blockTotal),
          Double(blockCorrect) / Double(blockTotal) * 100, elapsed))
    }

    let trainLoss = runningLoss / Double(runningTotal)
    let trainAcc = Double(runningCorrect) / Double(runningTotal)
    let (valLoss, valAcc) = evaluate(modelCopy, loader: testLoader)
    print(
      String(
        format:
          "epoch %d done • train loss %.4f • train acc %.2f%% • val loss %.4f • val acc %.2f%%",
        epoch, trainLoss, trainAcc * 100, valLoss, valAcc * 100))
  }

  print(String(format: "Training finished in %.1fs", Date().timeIntervalSince(start)))
} catch {
  let msg = "CIFAR-10 example failed: \(error)\n"
  FileHandle.standardError.write(msg.data(using: .utf8)!)
  exit(EXIT_FAILURE)
}
