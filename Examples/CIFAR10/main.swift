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

  // New: architecture & width multiplier for MobileNet
  var arch: String = "mobilenetv1"  // lenet | alexnet | vgg11 | vgg16 | mobilenetv1 | mlp
  var alpha: Double = 1.0  // only used by mobilenetv1
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
    } else if let v = parseString(arg, prefix: "--arch=") {
      c.arch = v.lowercased()
    } else if let v = parseDouble(arg, prefix: "--alpha=") {
      c.alpha = v
    }
  }
  return c
}
@inline(__always) func parseInt(_ a: String, prefix: String) -> Int? {
  a.hasPrefix(prefix) ? Int(a.dropFirst(prefix.count)) : nil
}
@inline(__always) func parseDouble(_ a: String, prefix: String) -> Double? {
  a.hasPrefix(prefix) ? Double(a.dropFirst(prefix.count)) : nil
}
@inline(__always) func parseUInt64(_ a: String, prefix: String) -> UInt64? {
  a.hasPrefix(prefix) ? UInt64(a.dropFirst(prefix.count)) : nil
}
@inline(__always) func parseString(_ a: String, prefix: String) -> String? {
  a.hasPrefix(prefix) ? String(a.dropFirst(prefix.count)) : nil
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

// ---- Generic training loop (works for any Layer) ----
func run<Model: Layer>(model: Model, cfg: TrainingConfig, cifar: CIFAR10) {
  // For parity with your examples, reuse the same optimizer and logging.
  var model = model
  var opt = AdamW(for: model, learningRate: cfg.learningRate)
  let stepsPerEpoch = (cifar.train.count + cfg.batchSize - 1) / cfg.batchSize

  let testLoader = DataLoader(
    dataset: cifar.test, batchSize: cfg.evalBatchSize, shuffle: false, dropLast: false, seed: nil)
  let start = Date()

  // Parameter count (sum of shapes).
  @inline(__always) func paramCount<M: Layer>(_ m: M) -> Int {
    var total = 0
    for kp in M.parameterKeyPaths {
      let t = m[keyPath: kp]
      total += t.shape.reduce(1, *)
    }
    return total
  }
  print("Model parameters: \(paramCount(model))")

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
      let (lossTensor, pullback) = valueWithPullback(at: model) { current -> Tensor in
        let logits = current.call(b.images, context: trainContext)
        return softmaxCrossEntropy(logits: logits, labels: b.labels)
      }
      let grad = pullback(Tensor(1.0, dtype: .float32))
      opt.update(&model, along: grad)

      let logits = model.call(b.images, context: inferenceContext)
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
    let (valLoss, valAcc) = evaluate(model, loader: testLoader)
    print(
      String(
        format:
          "epoch %d done • train loss %.4f • train acc %.2f%% • val loss %.4f • val acc %.2f%%",
        epoch, trainLoss, trainAcc * 100, valLoss, valAcc * 100))
  }

  print(String(format: "Training finished in %.1fs", Date().timeIntervalSince(start)))
}

// ---- Main (select architecture, then run) ----
do {
  var cfg = parseConfig()
  print("TaylorTorch CIFAR-10 • arch: \(cfg.arch) • epochs: \(cfg.epochs), batch: \(cfg.batchSize)")

  print("Preparing CIFAR-10 dataset…")
  let cifar = try CIFAR10(normalize: true)
  print("Loaded CIFAR-10 with \(cifar.train.count) training and \(cifar.test.count) test samples")

  // Dispatch to the selected architecture. Each branch specializes run<Model: Layer>.
  switch cfg.arch {
  case "lenet":
    run(model: CIFARArchitectures.leNet(), cfg: cfg, cifar: cifar)

  case "alexnet":
    run(model: CIFARArchitectures.alexNetCIFAR(), cfg: cfg, cifar: cifar)

  case "vgg11":
    run(model: CIFARArchitectures.vgg11CIFAR(), cfg: cfg, cifar: cifar)

  case "vgg16":
    run(model: CIFARArchitectures.vgg16CIFAR(), cfg: cfg, cifar: cifar)

  case "mobilenetv1":
    run(model: CIFARArchitectures.mobileNetV1CIFAR(alpha: cfg.alpha), cfg: cfg, cifar: cifar)

  case "mlp":
    run(model: CIFARArchitectures.mlp(), cfg: cfg, cifar: cifar)

  case "mini-cnn":
    run(model: CIFARMiniArchitectures.miniCNN(), cfg: cfg, cifar: cifar)

  case "tiny-vgg":
    run(model: CIFARMiniArchitectures.tinyVGG(), cfg: cfg, cifar: cifar)

  case "micro-mobilenet":
    run(model: CIFARMiniArchitectures.microMobileNetV1(alpha: cfg.alpha), cfg: cfg, cifar: cifar)

  case "nano-mlp":
    run(model: CIFARMiniArchitectures.nanoMLP(width: 64), cfg: cfg, cifar: cifar)

  default:
    print(
      "Unknown --arch=\(cfg.arch). Supported: lenet | alexnet | vgg11 | vgg16 | mobilenetv1 | mlp")
    print("Falling back to lenet.")
    run(model: CIFARArchitectures.leNet(), cfg: cfg, cifar: cifar)
  }
} catch {
  let msg = "CIFAR-10 example failed: \(error)\n"
  FileHandle.standardError.write(msg.data(using: .utf8)!)
  exit(EXIT_FAILURE)
}
