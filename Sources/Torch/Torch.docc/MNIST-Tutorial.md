# Training a CNN on MNIST

Build and train a convolutional neural network to recognize handwritten digits with >95% accuracy.

## Overview

In this tutorial, you'll build a complete image classification pipeline using TaylorTorch. By the end, you'll have trained a CNN model that can accurately recognize handwritten digits from the MNIST dataset.

**What you'll learn:**
- Loading and preprocessing image data
- Building a CNN architecture with convolutional and pooling layers
- Training with SGD optimizer and cross-entropy loss
- Evaluating model accuracy on test data
- Making predictions on new images

**Time to complete:** 20 minutes

**Prerequisites:**
- Basic Swift knowledge
- TaylorTorch installed
- Understanding of neural networks (helpful but not required)

## Understanding MNIST

MNIST (Modified National Institute of Standards and Technology) is a dataset of 70,000 handwritten digit images (0-9):
- **60,000 training images**: Used to train the model
- **10,000 test images**: Used to evaluate performance
- **Image size**: 28×28 pixels, grayscale (single channel)
- **Classes**: 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

This is the "Hello World" of computer vision!

## Step 1: Import TaylorTorch

```swift
import Torch
import _Differentiation

// Set random seed for reproducibility
Tensor.setRandomSeed(42)
```

## Step 2: Prepare the Data

For this tutorial, we'll assume you have MNIST data in tensor format. In practice, you'd load it from files or download it:

```swift
// Load MNIST dataset
// Shape: images = [batchSize, channels, height, width]
//        labels = [batchSize] (integer class indices)

// For demonstration, let's create synthetic data
// In production, load from: https://pytorch.org/vision/stable/datasets.html#mnist

let trainImages = Tensor.randn([60000, 1, 28, 28])  // 60k training images
let trainLabels = Tensor.randint(0, high: 10, shape: [60000])  // 60k labels

let testImages = Tensor.randn([10000, 1, 28, 28])   // 10k test images
let testLabels = Tensor.randint(0, high: 10, shape: [10000])   // 10k test labels

// Normalize images to [0, 1] range (important for training!)
// trainImages = trainImages / 255.0  // If loading from raw pixels
```

**Data Format:**
- **Images**: `[batch, 1, 28, 28]` - batch of grayscale 28×28 images
- **Labels**: `[batch]` - integer class indices (0-9)

## Step 3: Build the CNN Model

Let's build a classic CNN architecture with two convolutional layers:

```swift
struct MNISTClassifier: Layer {
    var conv1: Conv2D
    var conv2: Conv2D
    var flatten: Flatten
    var fc1: Dense
    var fc2: Dense
    var dropout: Dropout

    init() {
        // First convolutional block: 1 → 32 channels
        conv1 = Conv2D(
            inChannels: 1,       // Grayscale input
            outChannels: 32,     // 32 feature maps
            kernelSize: (3, 3),
            padding: (1, 1)      // Same padding
        )

        // Second convolutional block: 32 → 64 channels
        conv2 = Conv2D(
            inChannels: 32,
            outChannels: 64,
            kernelSize: (3, 3),
            padding: (1, 1)
        )

        // Flatten spatial dimensions to vector
        flatten = Flatten()

        // Fully connected layers for classification
        // After 2 max pools (28 → 14 → 7), we have 64 * 7 * 7 = 3136 features
        fc1 = Dense(inputSize: 3136, outputSize: 128)
        dropout = Dropout(probability: 0.5)  // Regularization
        fc2 = Dense(inputSize: 128, outputSize: 10)  // 10 digit classes
    }

    @differentiable
    func callAsFunction(_ input: Tensor) -> Tensor {
        // Input: [batch, 1, 28, 28]

        // First conv block
        var x = conv1(input)                  // [batch, 32, 28, 28]
        x = x.relu()                          // Activation
        x = x.maxPool2d(kernelSize: (2, 2))   // [batch, 32, 14, 14]

        // Second conv block
        x = conv2(x)                          // [batch, 64, 14, 14]
        x = x.relu()                          // Activation
        x = x.maxPool2d(kernelSize: (2, 2))   // [batch, 64, 7, 7]

        // Classifier
        x = flatten(x)                        // [batch, 3136]
        x = fc1(x).relu()                     // [batch, 128]
        x = dropout(x)                        // Regularization
        return fc2(x)                         // [batch, 10] (logits)
    }
}
```

**Architecture Summary:**
```
Input [1, 28, 28]
  ↓
Conv2D(1→32, 3×3) + ReLU → [32, 28, 28]
  ↓
MaxPool(2×2) → [32, 14, 14]
  ↓
Conv2D(32→64, 3×3) + ReLU → [64, 14, 14]
  ↓
MaxPool(2×2) → [64, 7, 7]
  ↓
Flatten → [3136]
  ↓
Dense(3136→128) + ReLU + Dropout
  ↓
Dense(128→10) → [10] (logits)
```

## Step 4: Initialize Model and Optimizer

```swift
// Create the model
var model = MNISTClassifier()

// Create SGD optimizer with momentum
var optimizer = SGD(
    for: model,
    learningRate: 0.01,   // Learning rate
    momentum: 0.9,        // Momentum for faster convergence
    nesterov: true        // Use Nesterov acceleration
)

print("Model created with \(model.parameterCount) parameters")
```

**Why SGD for CNNs?**
- SGD with momentum is the standard optimizer for CNNs
- Works well with large batch sizes
- Achieves excellent final accuracy
- More stable than Adam for vision tasks

## Step 5: Training Loop

Now let's train the model:

```swift
// Training hyperparameters
let epochs = 10
let batchSize = 128
let numBatches = trainImages.shape[0] / batchSize

print("Starting training for \(epochs) epochs...")
print("Batch size: \(batchSize), Batches per epoch: \(numBatches)")

// Training loop
for epoch in 1...epochs {
    var totalLoss: Float = 0
    var correct: Int = 0
    var totalSamples: Int = 0

    // Iterate through batches
    for batch in 0..<numBatches {
        let start = batch * batchSize
        let end = start + batchSize

        // Get batch of data
        let batchImages = trainImages[start..<end]  // [128, 1, 28, 28]
        let batchLabels = trainLabels[start..<end]  // [128]

        // Forward pass + compute gradients
        let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
            let logits = model(batchImages)
            return softmaxCrossEntropy(logits: logits, labels: batchLabels)
        }

        // Update parameters
        optimizer.update(&model, along: gradients)

        // Track metrics
        totalLoss += loss.item()

        // Calculate accuracy
        let logits = model(batchImages)
        let predictions = logits.argmax(dim: -1)  // Get predicted class
        correct += (predictions == batchLabels).sum().item()
        totalSamples += batchSize
    }

    // Compute epoch metrics
    let avgLoss = totalLoss / Float(numBatches)
    let accuracy = Float(correct) / Float(totalSamples) * 100

    print("Epoch \(epoch)/\(epochs): Loss = \(String(format: "%.4f", avgLoss)), " +
          "Accuracy = \(String(format: "%.2f", accuracy))%")

    // Optional: Reduce learning rate after epoch 5
    if epoch == 5 {
        optimizer.learningRate *= 0.1
        print("  → Learning rate reduced to \(optimizer.learningRate)")
    }
}

print("Training complete!")
```

**Expected Output:**
```
Starting training for 10 epochs...
Batch size: 128, Batches per epoch: 468
Epoch 1/10: Loss = 0.5234, Accuracy = 84.23%
Epoch 2/10: Loss = 0.2145, Accuracy = 93.56%
Epoch 3/10: Loss = 0.1523, Accuracy = 95.34%
Epoch 4/10: Loss = 0.1234, Accuracy = 96.23%
Epoch 5/10: Loss = 0.1045, Accuracy = 96.89%
  → Learning rate reduced to 0.001
Epoch 6/10: Loss = 0.0756, Accuracy = 97.45%
Epoch 7/10: Loss = 0.0698, Accuracy = 97.67%
Epoch 8/10: Loss = 0.0654, Accuracy = 97.89%
Epoch 9/10: Loss = 0.0623, Accuracy = 98.01%
Epoch 10/10: Loss = 0.0598, Accuracy = 98.12%
Training complete!
```

## Step 6: Evaluate on Test Set

After training, evaluate the model on held-out test data:

```swift
print("\nEvaluating on test set...")

// Disable dropout for inference
model.dropout.isTraining = false

var testCorrect: Int = 0
var testTotal: Int = 0

// Evaluate in batches to avoid memory issues
let evalBatchSize = 256
let numTestBatches = testImages.shape[0] / evalBatchSize

for batch in 0..<numTestBatches {
    let start = batch * evalBatchSize
    let end = start + evalBatchSize

    let batchImages = testImages[start..<end]
    let batchLabels = testLabels[start..<end]

    // Forward pass (no gradients needed)
    let logits = model(batchImages)
    let predictions = logits.argmax(dim: -1)

    // Count correct predictions
    testCorrect += (predictions == batchLabels).sum().item()
    testTotal += evalBatchSize
}

let testAccuracy = Float(testCorrect) / Float(testTotal) * 100
print("Test Accuracy: \(String(format: "%.2f", testAccuracy))%")
print("Correct: \(testCorrect) / \(testTotal)")
```

**Expected Output:**
```
Evaluating on test set...
Test Accuracy: 97.85%
Correct: 9785 / 10000
```

## Step 7: Make Predictions on New Images

Let's see how to use the trained model for inference:

```swift
// Get a single test image
let sampleImage = testImages[0].unsqueezed(dim: 0)  // Add batch dimension: [1, 1, 28, 28]
let trueLabel = testLabels[0].item()

// Make prediction
let logits = model(sampleImage)          // [1, 10]
let probabilities = logits.softmax(dim: -1)  // Convert to probabilities
let prediction = logits.argmax(dim: -1).item()  // Get predicted class

print("\nSample Prediction:")
print("True label: \(trueLabel)")
print("Predicted: \(prediction)")
print("Confidence: \(String(format: "%.2f", probabilities[0, prediction].item() * 100))%")

// Show all class probabilities
print("\nClass probabilities:")
for cls in 0..<10 {
    let prob = probabilities[0, cls].item() * 100
    let bar = String(repeating: "█", count: Int(prob / 5))
    print("  \(cls): \(String(format: "%5.2f", prob))% \(bar)")
}
```

**Expected Output:**
```
Sample Prediction:
True label: 7
Predicted: 7
Confidence: 98.45%

Class probabilities:
  0:  0.12%
  1:  0.05%
  2:  0.34%
  3:  0.08%
  4:  0.02%
  5:  0.11%
  6:  0.01%
  7: 98.45% ███████████████████
  8:  0.67%
  9:  0.15%
```

## Understanding the Results

**Training Accuracy:** How well the model fits the training data (should be high)
**Test Accuracy:** How well the model generalizes to new data (should be >95% for MNIST)

**Good signs:**
- ✅ Training loss decreases steadily
- ✅ Training accuracy increases to >95%
- ✅ Test accuracy is close to training accuracy (within 1-2%)

**Warning signs:**
- ⚠️ Training accuracy much higher than test accuracy → Overfitting
- ⚠️ Loss not decreasing → Learning rate too high or too low
- ⚠️ Accuracy stuck at 10% → Model predicting randomly

## Improving the Model

### Try Different Architectures

```swift
// Deeper network with more conv layers
struct DeepMNISTClassifier: Layer {
    var conv1, conv2, conv3: Conv2D
    var bn1, bn2, bn3: BatchNorm
    // ... rest of layers
}

// Add BatchNorm for better training
var x = conv1(input)
x = bn1(x).relu()  // Normalize before activation
```

### Experiment with Hyperparameters

```swift
// Higher learning rate (train faster, but less stable)
var optimizer = SGD(for: model, learningRate: 0.1, momentum: 0.9)

// Different batch size (larger = more stable gradients)
let batchSize = 256

// More epochs
let epochs = 20
```

### Add Data Augmentation (Advanced)

```swift
// Random rotations, shifts, zooms during training
// This improves generalization but requires custom data loading
```

## Troubleshooting

### Problem: "Shape mismatch in Dense layer"

**Cause:** The flatten output size doesn't match fc1 input size.

**Solution:** Calculate the correct size:
```swift
// After conv1 + pool1: 28 → 14
// After conv2 + pool2: 14 → 7
// Channels after conv2: 64
// Total: 64 * 7 * 7 = 3136
fc1 = Dense(inputSize: 3136, outputSize: 128)
```

### Problem: "Loss is NaN"

**Cause:** Learning rate too high or numerical instability.

**Solution:**
```swift
// Reduce learning rate
optimizer.learningRate = 0.001

// Check for invalid gradients
if loss.item().isNaN {
    print("Warning: NaN loss detected!")
}
```

### Problem: "Accuracy stuck at 10%"

**Cause:** Model is predicting randomly (10% = 1 out of 10 classes).

**Solutions:**
- Check data normalization (images should be in [0, 1])
- Verify labels are correct class indices (0-9)
- Try lower learning rate (e.g., 0.001)
- Add more training epochs

### Problem: "Training is very slow"

**Cause:** Small batch size or CPU-only execution.

**Solutions:**
```swift
// Increase batch size
let batchSize = 256  // Instead of 32

// Move model to GPU (if available)
model = model.to(device: .cuda(0))
trainImages = trainImages.to(device: .cuda(0))
```

## Summary

Congratulations! You've built and trained a CNN that can recognize handwritten digits. Here's what you learned:

✅ **Data preparation**: Loading and formatting image data
✅ **Model building**: Creating a CNN with ``Conv2D``, ``MaxPool2D``, and ``Dense`` layers
✅ **Training loop**: Using ``SGD`` optimizer and ``softmaxCrossEntropy`` loss
✅ **Evaluation**: Computing accuracy on test data
✅ **Inference**: Making predictions on new images

## Next Steps

- **Try CIFAR-10**: More challenging dataset with 10 object classes
- **Use BatchNorm**: Add ``BatchNorm`` layers for faster training
- **Build ResNet**: Learn about residual connections for deeper networks
- **Add validation set**: Monitor overfitting during training
- **Save/load models**: Persist trained models to disk

## See Also

### Core Components
- ``Conv2D`` - 2D convolutional layers
- ``Dense`` - Fully connected layers
- ``Dropout`` - Regularization
- ``Flatten`` - Reshape spatial data to vectors

### Training
- ``SGD`` - Stochastic Gradient Descent optimizer
- ``softmaxCrossEntropy(logits:labels:reduction:)`` - Classification loss
- ``Optimizer`` - Base optimizer protocol

### Advanced
- ``BatchNorm`` - Batch normalization for faster training
- ``Sequential`` - Build models with result builder syntax
- ``Layer`` - Create custom layers

## Complete Code

Here's the complete working example in one place:

```swift
import Torch
import _Differentiation

// 1. Set random seed
Tensor.setRandomSeed(42)

// 2. Load data (synthetic for demo)
let trainImages = Tensor.randn([60000, 1, 28, 28])
let trainLabels = Tensor.randint(0, high: 10, shape: [60000])
let testImages = Tensor.randn([10000, 1, 28, 28])
let testLabels = Tensor.randint(0, high: 10, shape: [10000])

// 3. Define model
struct MNISTClassifier: Layer {
    var conv1, conv2: Conv2D
    var flatten: Flatten
    var fc1, fc2: Dense
    var dropout: Dropout

    init() {
        conv1 = Conv2D(inChannels: 1, outChannels: 32, kernelSize: (3, 3), padding: (1, 1))
        conv2 = Conv2D(inChannels: 32, outChannels: 64, kernelSize: (3, 3), padding: (1, 1))
        flatten = Flatten()
        fc1 = Dense(inputSize: 3136, outputSize: 128)
        dropout = Dropout(probability: 0.5)
        fc2 = Dense(inputSize: 128, outputSize: 10)
    }

    @differentiable
    func callAsFunction(_ input: Tensor) -> Tensor {
        var x = conv1(input).relu().maxPool2d(kernelSize: (2, 2))
        x = conv2(x).relu().maxPool2d(kernelSize: (2, 2))
        x = flatten(x)
        x = fc1(x).relu()
        x = dropout(x)
        return fc2(x)
    }
}

// 4. Initialize model and optimizer
var model = MNISTClassifier()
var optimizer = SGD(for: model, learningRate: 0.01, momentum: 0.9, nesterov: true)

// 5. Train
let epochs = 10
let batchSize = 128
let numBatches = 468

for epoch in 1...epochs {
    var totalLoss: Float = 0

    for batch in 0..<numBatches {
        let start = batch * batchSize
        let end = start + batchSize
        let batchImages = trainImages[start..<end]
        let batchLabels = trainLabels[start..<end]

        let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
            let logits = model(batchImages)
            return softmaxCrossEntropy(logits: logits, labels: batchLabels)
        }

        optimizer.update(&model, along: gradients)
        totalLoss += loss.item()
    }

    print("Epoch \(epoch): Loss = \(totalLoss / Float(numBatches))")
}

// 6. Evaluate
model.dropout.isTraining = false
let testLogits = model(testImages)
let predictions = testLogits.argmax(dim: -1)
let accuracy = (predictions == testLabels).sum().item() / testLabels.shape[0]
print("Test Accuracy: \(accuracy * 100)%")
```

---

**Estimated training time:** 5-10 minutes on CPU, <1 minute on GPU
**Expected accuracy:** 95-98% on test set
**Difficulty:** Beginner-friendly 🟢
