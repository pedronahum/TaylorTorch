# Building a Neural Machine Translation Model

Learn to build a sequence-to-sequence model with attention for translating text between languages.

## Overview

In this tutorial, you'll build a neural machine translation (NMT) system using transformer-based attention mechanisms. You'll learn how to handle variable-length sequences, implement encoder-decoder architectures, and use attention to align source and target languages.

**What you'll learn:**
- Sequence-to-sequence (seq2seq) architecture
- Encoder-decoder pattern with transformers
- Multi-head attention mechanisms
- Handling variable-length sequences
- Teacher forcing during training
- Beam search for inference (basic greedy decoding)

**Time to complete:** 30-40 minutes

**Prerequisites:**
- Completed the MNIST tutorial (recommended)
- Understanding of attention mechanisms (helpful)
- Basic knowledge of NLP concepts

## Understanding Neural Machine Translation

**Goal:** Translate sentences from one language to another (e.g., English → French)

**Example:**
```
Input:  "I love deep learning"
Output: "J'aime l'apprentissage profond"
```

**Key Challenges:**
- Variable-length input and output
- Different word orders between languages
- Handling rare words and context
- Alignment between source and target

**Our Approach:** Transformer-based encoder-decoder with attention

## Architecture Overview

```
Source: "I love deep learning"
   ↓
Embedding → [batch, src_len, embed_dim]
   ↓
Transformer Encoder (self-attention)
   ↓
Context vectors → [batch, src_len, model_dim]
   ↓
Transformer Decoder (cross-attention + self-attention)
   ↓
Output: "J' aime l' apprentissage profond"
```

## Step 1: Import and Setup

```swift
import Torch
import _Differentiation

// Set random seed for reproducibility
Tensor.setRandomSeed(42)

// Vocabulary and constants
let srcVocabSize = 10000  // English vocabulary
let tgtVocabSize = 10000  // French vocabulary
let maxSeqLen = 50        // Maximum sequence length
let padIdx = 0            // Padding token index
let sosIdx = 1            // Start-of-sequence token
let eosIdx = 2            // End-of-sequence token
```

## Step 2: Build the Encoder

The encoder processes the source sentence and creates context vectors:

```swift
struct TransformerEncoder: Layer {
    var embedding: Embedding
    var positionEncoding: Tensor
    var encoderLayers: [TransformerEncoderLayer]
    var norm: LayerNorm

    init(vocabSize: Int, modelDim: Int, numHeads: Int, numLayers: Int, maxSeqLen: Int) {
        // Token embeddings
        embedding = Embedding(vocabularySize: vocabSize, embeddingSize: modelDim)

        // Positional encoding (sinusoidal)
        positionEncoding = createPositionalEncoding(maxSeqLen: maxSeqLen, modelDim: modelDim)

        // Stack of transformer encoder layers
        encoderLayers = (0..<numLayers).map { _ in
            TransformerEncoderLayer(modelDim: modelDim, numHeads: numHeads)
        }

        // Final layer normalization
        norm = LayerNorm(featureCount: modelDim)
    }

    @differentiable
    func callAsFunction(_ input: Tensor, mask: Tensor? = nil) -> Tensor {
        // input: [batch, seqLen] (token indices)

        // Embed tokens
        var x = embedding(input)  // [batch, seqLen, modelDim]

        // Add positional encoding
        let seqLen = x.shape[1]
        let pos = positionEncoding[0..<seqLen, :]  // [seqLen, modelDim]
        x = x + pos.unsqueezed(dim: 0)  // Broadcast across batch

        // Apply transformer encoder layers
        for layer in encoderLayers {
            x = layer(x, mask: mask)
        }

        // Final normalization
        return norm(x)  // [batch, seqLen, modelDim]
    }
}

// Helper: Create sinusoidal positional encoding
func createPositionalEncoding(maxSeqLen: Int, modelDim: Int) -> Tensor {
    var encoding = Tensor.zeros([maxSeqLen, modelDim])

    for pos in 0..<maxSeqLen {
        for i in 0..<(modelDim / 2) {
            let angle = Float(pos) / pow(10000.0, Float(2 * i) / Float(modelDim))
            encoding[pos, 2 * i] = Tensor(sin(angle))
            encoding[pos, 2 * i + 1] = Tensor(cos(angle))
        }
    }

    return encoding
}
```

**What it does:**
1. **Embed tokens**: Convert word indices to dense vectors
2. **Add position**: Inject sequence order information
3. **Self-attention**: Let each word attend to all other words
4. **Output**: Context-aware representations of source sentence

## Step 3: Build the Decoder

The decoder generates the target sentence one word at a time:

```swift
struct TransformerDecoder: Layer {
    var embedding: Embedding
    var positionEncoding: Tensor
    var decoderLayers: [TransformerDecoderLayer]
    var norm: LayerNorm
    var outputProjection: Dense

    init(vocabSize: Int, modelDim: Int, numHeads: Int, numLayers: Int, maxSeqLen: Int) {
        // Token embeddings
        embedding = Embedding(vocabularySize: vocabSize, embeddingSize: modelDim)

        // Positional encoding
        positionEncoding = createPositionalEncoding(maxSeqLen: maxSeqLen, modelDim: modelDim)

        // Stack of transformer decoder layers
        decoderLayers = (0..<numLayers).map { _ in
            TransformerDecoderLayer(modelDim: modelDim, numHeads: numHeads)
        }

        // Final layer normalization
        norm = LayerNorm(featureCount: modelDim)

        // Project to vocabulary
        outputProjection = Dense(inputSize: modelDim, outputSize: vocabSize)
    }

    @differentiable
    func callAsFunction(
        _ target: Tensor,
        encoderOutput: Tensor,
        targetMask: Tensor? = nil,
        sourceMask: Tensor? = nil
    ) -> Tensor {
        // target: [batch, tgtLen] (token indices)
        // encoderOutput: [batch, srcLen, modelDim]

        // Embed target tokens
        var x = embedding(target)  // [batch, tgtLen, modelDim]

        // Add positional encoding
        let seqLen = x.shape[1]
        let pos = positionEncoding[0..<seqLen, :]
        x = x + pos.unsqueezed(dim: 0)

        // Apply transformer decoder layers
        for layer in decoderLayers {
            x = layer(
                x,
                encoderOutput: encoderOutput,
                targetMask: targetMask,
                sourceMask: sourceMask
            )
        }

        // Final normalization
        x = norm(x)  // [batch, tgtLen, modelDim]

        // Project to vocabulary logits
        return outputProjection(x)  // [batch, tgtLen, vocabSize]
    }
}
```

**What it does:**
1. **Embed target tokens**: Convert output words to vectors
2. **Self-attention**: Attend to previously generated words (causal)
3. **Cross-attention**: Attend to source sentence (encoder output)
4. **Output**: Logits over vocabulary for each position

## Step 4: Complete Seq2Seq Model

Combine encoder and decoder:

```swift
struct Seq2SeqTransformer: Layer {
    var encoder: TransformerEncoder
    var decoder: TransformerDecoder

    init(
        srcVocabSize: Int,
        tgtVocabSize: Int,
        modelDim: Int = 512,
        numHeads: Int = 8,
        numLayers: Int = 6,
        maxSeqLen: Int = 100
    ) {
        encoder = TransformerEncoder(
            vocabSize: srcVocabSize,
            modelDim: modelDim,
            numHeads: numHeads,
            numLayers: numLayers,
            maxSeqLen: maxSeqLen
        )

        decoder = TransformerDecoder(
            vocabSize: tgtVocabSize,
            modelDim: modelDim,
            numHeads: numHeads,
            numLayers: numLayers,
            maxSeqLen: maxSeqLen
        )
    }

    @differentiable
    func callAsFunction(
        source: Tensor,
        target: Tensor,
        sourceMask: Tensor? = nil,
        targetMask: Tensor? = nil
    ) -> Tensor {
        // Encode source
        let encoderOutput = encoder(source, mask: sourceMask)

        // Decode to target
        let logits = decoder(
            target,
            encoderOutput: encoderOutput,
            targetMask: targetMask,
            sourceMask: sourceMask
        )

        return logits  // [batch, tgtLen, tgtVocabSize]
    }
}
```

## Step 5: Create Attention Masks

Masks prevent attending to padding and future tokens:

```swift
// Padding mask: Prevent attention to <PAD> tokens
func createPaddingMask(tokens: Tensor, padIdx: Int) -> Tensor {
    // tokens: [batch, seqLen]
    // Returns: [batch, seqLen] with 1 for real tokens, 0 for padding
    return (tokens != Tensor(Float(padIdx))).to(dtype: .float32)
}

// Causal mask: Prevent attention to future tokens (for decoder self-attention)
func createCausalMask(seqLen: Int) -> Tensor {
    // Returns: [seqLen, seqLen] lower triangular matrix
    var mask = Tensor.zeros([seqLen, seqLen])
    for i in 0..<seqLen {
        for j in 0...i {
            mask[i, j] = Tensor(1.0)
        }
    }
    return mask
}

// Combined mask for decoder
func createDecoderMask(target: Tensor, padIdx: Int) -> Tensor {
    let seqLen = target.shape[1]

    // Padding mask
    let padMask = createPaddingMask(tokens: target, padIdx: padIdx)  // [batch, seqLen]

    // Causal mask
    let causalMask = createCausalMask(seqLen: seqLen)  // [seqLen, seqLen]

    // Combine: broadcast and multiply
    // padMask: [batch, 1, seqLen] * causalMask: [1, seqLen, seqLen]
    let combined = padMask.unsqueezed(dim: 1) * causalMask.unsqueezed(dim: 0)

    return combined  // [batch, seqLen, seqLen]
}
```

## Step 6: Training with Teacher Forcing

Teacher forcing: Feed ground truth target tokens during training (not predictions):

```swift
// Initialize model
var model = Seq2SeqTransformer(
    srcVocabSize: srcVocabSize,
    tgtVocabSize: tgtVocabSize,
    modelDim: 512,
    numHeads: 8,
    numLayers: 6
)

// Initialize Adam optimizer (standard for transformers)
var optimizer = Adam(
    for: model,
    learningRate: 1e-4,
    beta1: 0.9,
    beta2: 0.98,  // Common for transformers
    weightDecay: 0.01,
    adamW: true
)

print("Model initialized with ~\(model.parameterCount) parameters")

// Training loop
let epochs = 20
let batchSize = 32

print("Starting training...")

for epoch in 1...epochs {
    var totalLoss: Float = 0
    var numBatches = 0

    for (srcBatch, tgtBatch) in trainingData.batched(batchSize) {
        // srcBatch: [batch, srcLen] - source sentences
        // tgtBatch: [batch, tgtLen] - target sentences

        // Prepare decoder input: shift target right and add <SOS>
        let tgtInput = tgtBatch[:, 0..<(tgtBatch.shape[1] - 1)]  // Remove last token
        let tgtOutput = tgtBatch[:, 1...]  // Remove <SOS>, shift left

        // Create masks
        let srcMask = createPaddingMask(tokens: srcBatch, padIdx: padIdx)
        let tgtMask = createDecoderMask(target: tgtInput, padIdx: padIdx)

        // Forward pass + compute gradients
        let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
            // Get logits
            let logits = model(
                source: srcBatch,
                target: tgtInput,
                sourceMask: srcMask,
                targetMask: tgtMask
            )  // [batch, tgtLen-1, vocabSize]

            // Compute cross-entropy loss
            // Ignore padding tokens in loss
            let flatLogits = logits.reshaped([-1, tgtVocabSize])
            let flatTargets = tgtOutput.reshaped([-1])

            return softmaxCrossEntropy(logits: flatLogits, labels: flatTargets)
        }

        // Update parameters
        optimizer.update(&model, along: gradients)

        totalLoss += loss.item()
        numBatches += 1
    }

    let avgLoss = totalLoss / Float(numBatches)
    print("Epoch \(epoch)/\(epochs): Loss = \(String(format: "%.4f", avgLoss))")

    // Learning rate warmup (first 4000 steps)
    if epoch <= 4 {
        optimizer.learningRate *= 1.1  // Increase gradually
    } else if epoch == 10 {
        optimizer.learningRate *= 0.5  // Decay after plateau
    }
}

print("Training complete!")
```

**Teacher Forcing:**
```
Target:    <SOS> J'    aime  l'    apprentissage  profond  <EOS>
Input:     <SOS> J'    aime  l'    apprentissage  profond
Output:          J'    aime  l'    apprentissage  profond  <EOS>
```

At each step, feed the ground truth previous token (not model's prediction).

## Step 7: Inference (Greedy Decoding)

Generate translations using greedy decoding:

```swift
func translate(
    model: Seq2SeqTransformer,
    source: Tensor,  // [1, srcLen]
    maxLen: Int = 50,
    sosIdx: Int = 1,
    eosIdx: Int = 2
) -> Tensor {
    // Encode source
    let srcMask = createPaddingMask(tokens: source, padIdx: padIdx)
    let encoderOutput = model.encoder(source, mask: srcMask)

    // Start with <SOS> token
    var output = Tensor([sosIdx]).reshaped([1, 1])  // [1, 1]

    // Generate tokens one by one
    for _ in 0..<maxLen {
        // Create decoder mask
        let tgtMask = createCausalMask(seqLen: output.shape[1])

        // Get next token logits
        let logits = model.decoder(
            output,
            encoderOutput: encoderOutput,
            targetMask: tgtMask,
            sourceMask: srcMask
        )  // [1, currentLen, vocabSize]

        // Get last token prediction
        let nextTokenLogits = logits[0, -1, :]  // [vocabSize]
        let nextToken = nextTokenLogits.argmax(dim: 0)  // [1]

        // Append to output
        output = Tensor.cat([output, nextToken.unsqueezed(dim: 0).unsqueezed(dim: 0)], dim: 1)

        // Stop if <EOS> generated
        if nextToken.item() == eosIdx {
            break
        }
    }

    return output  // [1, outputLen]
}

// Example usage
let sourceSentence = Tensor([1, 45, 234, 67, 89, 2])  // "I love deep learning" encoded
let translation = translate(model: model, source: sourceSentence.unsqueezed(dim: 0))

print("Source:", sourceSentence)
print("Translation:", translation)
// Would decode back to: "J' aime l' apprentissage profond"
```

**Greedy Decoding:**
1. Start with `<SOS>` token
2. Predict next token (argmax)
3. Append prediction to input
4. Repeat until `<EOS>` or max length

## Step 8: Evaluate Translation Quality

Compute BLEU score (standard metric for translation):

```swift
// Simplified BLEU-1 (unigram precision)
func computeBLEU1(
    predictions: [Tensor],  // List of predicted sequences
    references: [Tensor]     // List of reference translations
) -> Float {
    var totalPrecision: Float = 0

    for (pred, ref) in zip(predictions, references) {
        let predTokens = Set(pred.toArray(as: Int.self))
        let refTokens = Set(ref.toArray(as: Int.self))

        let intersection = predTokens.intersection(refTokens)
        let precision = Float(intersection.count) / Float(predTokens.count)

        totalPrecision += precision
    }

    return totalPrecision / Float(predictions.count) * 100
}

// Evaluate on test set
print("\nEvaluating on test set...")
var predictions: [Tensor] = []
var references: [Tensor] = []

for (src, tgt) in testData {
    let pred = translate(model: model, source: src)
    predictions.append(pred)
    references.append(tgt)
}

let bleu = computeBLEU1(predictions: predictions, references: references)
print("BLEU-1 Score: \(String(format: "%.2f", bleu))%")
```

**BLEU Score:**
- Measures n-gram overlap between prediction and reference
- Range: 0-100 (higher is better)
- BLEU-1: Unigram precision (simplest)
- Production systems use BLEU-4 (4-gram)

**Expected Results:**
- BLEU-1: 50-70% (good)
- BLEU-4: 20-40% (competitive)

## Understanding Attention

Attention allows the decoder to focus on relevant source words:

```
Source: "I    love  deep    learning"
Target: "J'   aime  l'      apprentissage  profond"

When generating "aime":
  Attention to: "love" (high), "I" (medium), others (low)

When generating "apprentissage":
  Attention to: "learning" (high), "deep" (high), others (low)
```

**Visualization (conceptual):**
```swift
// Get attention weights from last decoder layer
let attentionWeights = model.decoder.decoderLayers.last?.crossAttention.attentionWeights

// attentionWeights: [batch, numHeads, tgtLen, srcLen]
// Shows which source tokens each target token attends to
```

## Advanced: Beam Search

Beam search explores multiple translation hypotheses:

```swift
func translateBeamSearch(
    model: Seq2SeqTransformer,
    source: Tensor,
    beamWidth: Int = 5,
    maxLen: Int = 50
) -> Tensor {
    // Keep top-k hypotheses at each step
    // Score = sum of log probabilities

    // Pseudocode:
    // 1. Start with k = 1 hypothesis: [<SOS>]
    // 2. Expand each hypothesis with all possible next tokens
    // 3. Score each expansion (log prob)
    // 4. Keep top-k highest scoring sequences
    // 5. Repeat until all k sequences end with <EOS>

    // Implementation left as exercise (complex)
    fatalError("Beam search implementation")
}
```

Beam search typically improves BLEU by 2-5 points over greedy decoding.

## Troubleshooting

### Problem: "Loss not decreasing"

**Solutions:**
- Implement learning rate warmup (gradual increase for first 4k steps)
- Check gradient norms (should be < 5.0)
- Verify masks are correct (especially causal mask)
- Reduce learning rate to 5e-5

### Problem: "Model outputs gibberish"

**Causes:**
- Training not converged yet (train longer)
- Learning rate too high
- Data not properly preprocessed

**Solutions:**
- Train for at least 20 epochs
- Monitor validation loss (should decrease steadily)
- Verify vocabularies are correct

### Problem: "Translations always the same"

**Cause:** Model learned to output most common sentence.

**Solutions:**
- Check label smoothing (reduces overconfidence)
- Increase model capacity (more layers/heads)
- More training data
- Use beam search instead of greedy

### Problem: "Out of memory"

**Solutions:**
```swift
// Reduce batch size
let batchSize = 16  // Instead of 32

// Reduce model size
let modelDim = 256  // Instead of 512
let numLayers = 4   // Instead of 6

// Gradient accumulation (simulate larger batch)
let accumSteps = 4
if step % accumSteps == 0 {
    optimizer.update(&model, along: gradients)
}
```

## Summary

Congratulations! You've built a neural machine translation system with attention. Here's what you learned:

✅ **Encoder-decoder architecture**: Process source and generate target
✅ **Attention mechanisms**: ``MultiHeadAttention`` for alignment
✅ **Variable-length sequences**: Handling different sentence lengths
✅ **Teacher forcing**: Training technique for seq2seq
✅ **Greedy decoding**: Inference with autoregressive generation
✅ **Masking**: Padding and causal masks

## Next Steps

- **Implement beam search**: Better translation quality
- **Add byte-pair encoding (BPE)**: Handle subwords
- **Pre-training**: Use pretrained embeddings (Word2Vec, GloVe)
- **Multi-lingual**: Train on multiple language pairs
- **Back-translation**: Data augmentation technique
- **Fine-tune BERT**: Use modern pretrained models

## See Also

### Attention & Transformers
- ``MultiHeadAttention`` - Multi-head attention mechanism
- ``TransformerEncoderLayer`` - Single transformer encoder layer
- ``TransformerDecoderLayer`` - Single transformer decoder layer

### Core Components
- ``Embedding`` - Token embeddings
- ``LayerNorm`` - Layer normalization
- ``Dense`` - Fully connected layers

### Training
- ``Adam`` - Adam optimizer (standard for transformers)
- ``softmaxCrossEntropy(logits:labels:reduction:)`` - Sequence loss

### Advanced Topics
- Label smoothing for better generalization
- Learning rate scheduling (warmup + decay)
- Gradient clipping for stability

---

**Estimated training time:** 2-4 hours (depends on dataset size)
**Expected BLEU score:** 25-40 (BLEU-4 on standard benchmarks)
**Difficulty:** Advanced 🔴
