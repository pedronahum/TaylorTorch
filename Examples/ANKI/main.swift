import Foundation
import Torch
import _Differentiation

// MARK: - Tiny tokenizer & vocabulary

struct Vocab {
  let stoi: [String: Int]
  let itos: [String]
  let pad: Int
  let bos: Int
  let eos: Int
  let unk: Int

  init(
    from texts: [String], minFreq: Int = 2,
    specials: [String] = ["<pad>", "<bos>", "<eos>", "<unk>"]
  ) {
    var freq: [String: Int] = [:]
    for t in texts { for tok in Vocab.tokenize(t) { freq[tok, default: 0] += 1 } }
    var itos = specials
    for (tok, c) in freq where c >= minFreq && !specials.contains(tok) { itos.append(tok) }
    var stoi: [String: Int] = [:]
    for (i, tok) in itos.enumerated() { stoi[tok] = i }
    self.stoi = stoi
    self.itos = itos
    self.pad = stoi["<pad>"]!
    self.bos = stoi["<bos>"]!
    self.eos = stoi["<eos>"]!
    self.unk = stoi["<unk>"]!
  }

  static func tokenize(_ s: String) -> [String] {
    let lowered = s.lowercased()
    let spaced = lowered.replacingOccurrences(
      of: #"([.,!?;:()"])"#,
      with: " $1 ",
      options: .regularExpression)
    return spaced.split { $0.isWhitespace }.map(String.init)
  }

  func encode(_ s: String, addBos: Bool = false, addEos: Bool = false) -> [Int64] {
    var out: [Int64] = []
    if addBos { out.append(Int64(bos)) }
    for tok in Self.tokenize(s) { out.append(Int64(stoi[tok] ?? unk)) }
    if addEos { out.append(Int64(eos)) }
    return out
  }
}

// MARK: - Batch collation

struct EncodedPair {
  let src: [Int64]
  let tgt: [Int64]
}

struct Batch {
  let src: Tensor  // [N, Ls] int64
  let tgtIn: Tensor  // [N, Lt] int64 (BOS + tokens)
  let tgtOut: Tensor  // [N, Lt] int64 (tokens + EOS)
  let srcMask: Tensor  // [N, 1, 1, Ls] bool
  let tgtMask: Tensor  // [N, 1, Lt, Lt] bool
}

func makeMasks(src: Tensor, tgtIn: Tensor, pad: Int) -> (Tensor, Tensor) {
  // Padding masks
  let padTensor = Tensor(Int64(pad), dtype: .int64, device: src.device)
  let srcPad = src.eq(padTensor).unsqueezed(dim: 1).unsqueezed(dim: 1)  // [N,1,1,Ls]
  let tgtPad = tgtIn.eq(padTensor).unsqueezed(dim: 1).unsqueezed(dim: 1)  // [N,1,1,Lt]

  // Causal mask for decoder self-attn: mask positions with k > q (upper triangle).
  let Lt = tgtIn.shape[1]
  let ar = Tensor.arange(Double(0), to: Double(Lt), step: 1.0, dtype: .float64)
  let i = ar.unsqueezed(dim: 1)  // [Lt,1]
  let j = ar.unsqueezed(dim: 0)  // [1,Lt]
  let future = j.gt(i)  // [Lt,Lt] Bool, true above diagonal
  let future4D = future.unsqueezed(dim: 0).unsqueezed(dim: 0)  // [1,1,Lt,Lt]
    .broadcasted(to: [tgtIn.shape[0], 1, Lt, Lt])

  // Combine: mask if padding OR future.
  let tgtPad4D = tgtPad.broadcasted(to: [tgtIn.shape[0], 1, Lt, Lt])
  let combinedInt = tgtPad4D.to(dtype: .int64) + future4D.to(dtype: .int64)
  let zero = Tensor(Int64(0), dtype: .int64, device: combinedInt.device)
  let tgtMask = combinedInt.gt(zero)
  return (srcPad, tgtMask)
}

func collate(_ group: [EncodedPair], pad: Int, maxSrc: Int, maxTgt: Int) -> Batch {
  let N = group.count
  let Ls = min(maxSrc, group.map { $0.src.count }.max() ?? 1)
  // We will form pairs [BOS, ...] -> [..., EOS], so the per-example target length
  // already includes BOS/EOS in ex.tgt. Use the max over the group, bounded by maxTgt.
  let Lt = min(maxTgt, group.map { $0.tgt.count }.max() ?? 2)

  var srcArr = [Int64](repeating: Int64(pad), count: N * Ls)
  var tgtInArr = [Int64](repeating: Int64(pad), count: N * Lt)
  var tgtOutArr = [Int64](repeating: Int64(pad), count: N * Lt)

  for (n, ex) in group.enumerated() {
    // Truncate source to Ls
    let s = Array(ex.src.prefix(Ls))

    // Keep up to Lt tokens from the *full* target (which already has BOS/EOS)
    // Then make aligned teacher-forcing pairs:
    //   tgtIn  = full[..<last]  (BOS + tokens)
    //   tgtOut = full[1..]      (tokens + EOS)
    let full = Array(ex.tgt.prefix(Lt))
    let tIn = Array(full.dropLast())
    let tOut = Array(full.dropFirst())

    for (i, v) in s.enumerated() { srcArr[n * Ls + i] = v }
    for (i, v) in tIn.enumerated() { tgtInArr[n * Lt + i] = v }
    for (i, v) in tOut.enumerated() { tgtOutArr[n * Lt + i] = v }
  }

  let src = Tensor(array: srcArr, shape: [N, Ls], dtype: .int64)
  let tgtIn = Tensor(array: tgtInArr, shape: [N, Lt], dtype: .int64)
  let tgtOut = Tensor(array: tgtOutArr, shape: [N, Lt], dtype: .int64)
  let (srcMask, tgtMask) = makeMasks(src: src, tgtIn: tgtIn, pad: pad)
  return Batch(src: src, tgtIn: tgtIn, tgtOut: tgtOut, srcMask: srcMask, tgtMask: tgtMask)
}

@inlinable
func noamScale(step: Int, dModel: Int, warmup: Int = 4000) -> Float {
  // d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
  let s = Float(step + 1)
  let dm = powf(Float(dModel), -0.5)
  let a = powf(s, -0.5)
  let b = s * powf(Float(warmup), -1.5)
  return dm * min(a, b)
}

// MARK: - Sequence cross-entropy with padding ignored
// logits: [N, Lt, V]; targets: [N, Lt] (int64); returns mean loss over non-pad tokens.
@differentiable(reverse)
func sequenceCrossEntropy(logits: Tensor, targets: Tensor, padIndex: Int) -> Tensor {
  let V = withoutDerivative(at: logits.shape[2])
  let logProbs = LogSoftmax(axis: -1)(logits)  // you ship Softmax/LogSoftmax

  let dtype = withoutDerivative(at: logits.dtype ?? .float32)
  let device = withoutDerivative(at: logits.device)

  let indices = withoutDerivative(at: targets.to(dtype: .int64))
  let rank = withoutDerivative(at: indices.rank)

  //print(
  //  "[Debug] logProbs shape: \(logProbs.shape), targets shape: \(targets.shape), depth: \(V), rank: \(rank)"
  //)

  var classIndicesTensor = Tensor.arange(
    Int64(0), to: Int64(V), step: Int64(1), dtype: .int64, device: device)
  for _ in 0..<rank { classIndicesTensor = classIndicesTensor.unsqueezed(dim: 0) }
  let classIndices = withoutDerivative(at: classIndicesTensor)
  let expanded = withoutDerivative(at: indices.unsqueezed(dim: rank))
  let mask = withoutDerivative(at: expanded.eq(classIndices))
  let oneHot = TorchWhere.select(
    condition: mask,
    Tensor(1.0, dtype: dtype, device: device),
    Tensor(0.0, dtype: dtype, device: device))

  let reduceDim = withoutDerivative(at: logProbs.rank - 1)
  let perToken = logProbs.multiplying(oneHot).sum(dim: reduceDim).negated()  // [N, Lt]

  let padTensor = withoutDerivative(at: Tensor(Int64(padIndex), dtype: .int64, device: device))
  let nonPad = withoutDerivative(
    at: Tensor.ones(shape: targets.shape, dtype: dtype, device: device)
      .subtracting(targets.eq(padTensor).to(dtype: dtype)))

  let lossPerTok = perToken.multiplying(nonPad)
  let denom = nonPad.sum() + Tensor(1e-6, dtype: dtype, device: device)
  return lossPerTok.sum().dividing(denom)
}

// MARK: - Demo training

struct Config {
  var epochs = 50
  var batchSize = 64
  var maxSrcLen = 48
  var maxTgtLen = 48
  var learningRate: Float = 0.0005
  var maxSamples = 20_000
}

do {
  let cfg = Config()
  print("Preparing Tatoeba (Anki) EN→ES…")
  let tatoeba = try TatoebaEnglishToSpanish(maxSamples: cfg.maxSamples)  // pairs of strings

  // Build vocabularies (normalized) – lowercased & trimmed
  let engTexts = tatoeba.train.map { $0.english.lowercased().trimmingCharacters(in: .whitespaces) }
  let spaTexts = tatoeba.train.map { $0.spanish.lowercased().trimmingCharacters(in: .whitespaces) }
  let srcV = Vocab(from: engTexts, minFreq: 1)
  let tgtV = Vocab(from: spaTexts, minFreq: 1)

  // Ensure specials exist in target encodings: we’ll use BOS/EOS explicitly.
  // (They are included by construction: <pad>, <bos>, <eos>, <unk>.)

  // Encode + filter by max lengths
  var encoded: [EncodedPair] = []
  encoded.reserveCapacity(tatoeba.train.count)
  for ex in tatoeba.train {
    let s = srcV.encode(ex.english, addBos: false, addEos: false)
    let t = tgtV.encode(ex.spanish, addBos: true, addEos: true)  // keep BOS/EOS in stream
    if s.count <= cfg.maxSrcLen && t.count <= cfg.maxTgtLen {
      encoded.append(EncodedPair(src: s, tgt: t))
    }
  }
  let dataset = ArrayDataset(encoded)
  print("Loaded \(dataset.count) EN–ES sentence pairs.")

  // Model
  let dModel = 128
  let heads = 4
  let ff = 256
  var model = TinyTransformer(
    srcVocab: srcV.itos.count,
    tgtVocab: tgtV.itos.count,
    dModel: dModel, heads: heads, ff: ff, maxLength: max(cfg.maxSrcLen, cfg.maxTgtLen)
  )
  print("Model initialized (parameters: \(dModel) dims, \(heads) heads).")

  //var opt = SGD(for: model, learningRate: cfg.learningRate)
  let opt = Adam(for: model, learningRate: cfg.learningRate)
  print("Optimizer ready; starting training…")

  // Training loop (MNIST-style scaffold)
  let stepsPerEpoch = (dataset.count + cfg.batchSize - 1) / cfg.batchSize
  print(
    "Training plan: \(cfg.epochs) epochs × \(stepsPerEpoch) steps (dataset count \(dataset.count))."
  )
  let startTime = Date()

  withLearningPhase(.training) {
    for epoch in 1...cfg.epochs {
      var rng = SystemRandomNumberGenerator()
      let order = Array(0..<dataset.count).shuffled(using: &rng)

      var step = 0
      var runningLoss: Double = 0
      var runningToks: Int = 0

      while step * cfg.batchSize < dataset.count {
        let lo = step * cfg.batchSize
        let hi = min(dataset.count, lo + cfg.batchSize)
        step += 1
        let group = order[lo..<hi].map { dataset[$0] }
        let batch = collate(group, pad: srcV.pad, maxSrc: cfg.maxSrcLen, maxTgt: cfg.maxTgtLen)
        //print("Step \(step): src shape \(batch.src.shape), tgtIn \(batch.tgtIn.shape)")
        //print("Masks src \(batch.srcMask.shape) tgt \(batch.tgtMask.shape)")

        let inferenceInput = TinyTransformer.Input(
          src: batch.src,
          tgtIn: batch.tgtIn,
          srcMask: batch.srcMask,
          tgtMask: batch.tgtMask)

        let result = valueWithPullback(at: model, inferenceInput) { current, input -> Tensor in
          let logits = current(input)
          return sequenceCrossEntropy(
            logits: logits, targets: batch.tgtOut, padIndex: tgtV.pad)
        }
        let lossTensor = result.value
        let pbModel = result.pullback(Tensor(1.0, dtype: .float32))
        let grad = pbModel.0
        opt.update(&model, along: grad)

        let loss = Double(lossTensor.toArray(as: Float.self)[0])
        // Approximate token count (non-pad) for reporting
        let padTensor = Tensor(Int64(tgtV.pad), dtype: .int64, device: batch.tgtOut.device)
        let ones = Tensor.ones(
          shape: batch.tgtOut.shape, dtype: .float32, device: batch.tgtOut.device)
        let isPad = batch.tgtOut.eq(padTensor).to(dtype: .float32)
        let nonPad = (ones - isPad).to(dtype: .int32).sum()
        let tokCount = Int(nonPad.toArray(as: Int32.self)[0])

        runningLoss += loss * Double(tokCount)
        runningToks += tokCount

        if step % 50 == 0 || step == stepsPerEpoch {
          let elapsed = Date().timeIntervalSince(startTime)
          let avgLoss = runningLoss / max(1, Double(runningToks))
          print(
            String(
              format: "epoch %d • step %d/%d • token-mean NLL %.4f • %.1fs",
              epoch, step, stepsPerEpoch, avgLoss, elapsed))
          runningLoss = 0
          runningToks = 0
        }
        if step >= stepsPerEpoch { break }
      }
    }
  }

  // quick qualitative samples (inference mode)
  withLearningPhase(.inference) {
    let samples = [
      "how are you?",
      "good morning!",
      "what time is it?",
      "where is the train station?",
    ]
    for s in samples {
      let src = srcV.encode(s)
      // greedy decode for 1 sentence
      let (tr, _) = greedyDecode(
        model: model, src: src, srcV: srcV, tgtV: tgtV, maxLen: cfg.maxTgtLen)
      print("EN: \(s)")
      print("ES: \(tr)")
    }
  }
} catch {
  // Use FileHandle.standardError for Swift 6 concurrency safety
  if let data = "ANKI example failed: \(error)\n".data(using: .utf8) {
    try? FileHandle.standardError.write(contentsOf: data)
  }
  exit(EXIT_FAILURE)
}

// MARK: - Tiny greedy decoder (for qualitative checks only)
func greedyDecode(
  model: TinyTransformer, src: [Int64],
  srcV: Vocab, tgtV: Vocab, maxLen: Int
) -> (String, [Int64]) {
  let N = 1
  let Ls = src.count
  let srcT = Tensor(array: src, shape: [N, Ls], dtype: .int64)
  let srcPad = Tensor(Int64(srcV.pad), dtype: .int64, device: srcT.device)
  let srcMask = srcT.eq(srcPad)
    .unsqueezed(dim: 1).unsqueezed(dim: 1)  // [1,1,1,Ls]

  var ys: [Int64] = [Int64(tgtV.bos)]
  for _ in 0..<maxLen {
    let Lt = ys.count
    let tgtIn = Tensor(array: ys, shape: [1, Lt], dtype: .int64)
    let (_, tgtMask) = makeMasks(src: srcT, tgtIn: tgtIn, pad: tgtV.pad)
    let logits = model(.init(src: srcT, tgtIn: tgtIn, srcMask: srcMask, tgtMask: tgtMask))
    let last = logits.select(dim: 1, index: Lt - 1)  // [1, V]
    let next = last.argmax(dim: 1).toArray(as: Int64.self)[0]
    ys.append(next)
    if next == Int64(tgtV.eos) { break }
  }
  // strip BOS/EOS
  let tokens = ys.dropFirst().prefix { $0 != Int64(tgtV.eos) }
  let words = tokens.map { tgtV.itos[Int($0)] }
  return (words.joined(separator: " "), Array(tokens))
}
