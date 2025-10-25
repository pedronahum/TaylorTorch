# Building Neural Network Models

Learn how to construct deep learning architectures using TaylorTorch's layer composition system.

## Overview

TaylorTorch provides powerful tools for building neural networks, from simple feedforward networks to complex architectures like transformers and graph neural networks. This guide covers the essential patterns and best practices for model construction.

## Sequential Models

The ``Sequential`` container is the simplest way to build models by stacking layers:

```swift
import Torch

let model = Sequential {
    Linear(inputSize: 784, outputSize: 256)
    ReLU()
    Dropout(probability: 0.2)
    Linear(inputSize: 256, outputSize: 128)
    ReLU()
    Linear(inputSize: 128, outputSize: 10)
}
```

### How Sequential Works

``Sequential`` uses Swift's result builder syntax to create a pipeline where:
1. Each layer's output becomes the next layer's input
2. Type checking ensures compatibility at compile time
3. Gradients flow backwards through all layers automatically

```swift
// Input shape: [batch, 784]
let input = Tensor.randn([32, 784])

// Forward pass through all layers
let output = model(input)  // Shape: [32, 10]

// Backward pass computes gradients for all parameters
let gradients = gradient(at: model) { m in
    let pred = m(input)
    return crossEntropy(pred, labels)
}
```

## Convolutional Neural Networks

Build CNNs for image processing tasks:

```swift
let cnn = Sequential {
    // First conv block
    Conv2D(inChannels: 3, outChannels: 32, kernelSize: (3, 3), padding: (1, 1))
    BatchNorm(featureCount: 32)
    ReLU()
    MaxPool2D(kernelSize: (2, 2))

    // Second conv block
    Conv2D(inChannels: 32, outChannels: 64, kernelSize: (3, 3), padding: (1, 1))
    BatchNorm(featureCount: 64)
    ReLU()
    MaxPool2D(kernelSize: (2, 2))

    // Classifier
    Flatten(startDim: 1)
    Linear(inputSize: 64 * 7 * 7, outputSize: 10)
}

// Process a batch of RGB images
let images = Tensor.randn([32, 3, 28, 28])  // [N, C, H, W]
let predictions = cnn(images)  // [32, 10]
```

### Understanding Convolution Parameters

- **inChannels**: Number of input feature maps (3 for RGB, 1 for grayscale)
- **outChannels**: Number of filters/feature maps to produce
- **kernelSize**: Size of the convolution window (height, width)
- **stride**: Step size for sliding the kernel (default: 1)
- **padding**: Zero-padding added to input borders
- **dilation**: Spacing between kernel elements (default: 1)

```swift
// Same padding - output size matches input size (with stride=1)
Conv2D(inChannels: 64, outChannels: 64, kernelSize: (3, 3), padding: (1, 1))

// Valid padding - no padding, output shrinks
Conv2D(inChannels: 64, outChannels: 64, kernelSize: (3, 3), padding: (0, 0))

// Strided convolution - downsamples spatially
Conv2D(inChannels: 64, outChannels: 128, kernelSize: (3, 3), stride: (2, 2))
```

## Custom Layer Architectures

For complex models, create custom types conforming to ``Layer``:

```swift
struct ResidualBlock: Layer {
    var conv1: Conv2D
    var bn1: BatchNorm
    var conv2: Conv2D
    var bn2: BatchNorm

    init(channels: Int) {
        self.conv1 = Conv2D(
            inChannels: channels,
            outChannels: channels,
            kernelSize: (3, 3),
            padding: (1, 1)
        )
        self.bn1 = BatchNorm(featureCount: channels)
        self.conv2 = Conv2D(
            inChannels: channels,
            outChannels: channels,
            kernelSize: (3, 3),
            padding: (1, 1)
        )
        self.bn2 = BatchNorm(featureCount: channels)
    }

    @differentiable
    func callAsFunction(_ input: Tensor) -> Tensor {
        var x = input
        x = conv1(x)
        x = bn1(x)
        x = relu(x)
        x = conv2(x)
        x = bn2(x)
        // Residual connection
        x = x + input
        return relu(x)
    }
}

// Use in a larger model
let resnet = Sequential {
    Conv2D(inChannels: 3, outChannels: 64, kernelSize: (7, 7), stride: (2, 2), padding: (3, 3))
    BatchNorm(featureCount: 64)
    ReLU()
    MaxPool2D(kernelSize: (3, 3), stride: (2, 2))

    ResidualBlock(channels: 64)
    ResidualBlock(channels: 64)
    ResidualBlock(channels: 64)

    GlobalAvgPool()
    Flatten()
    Linear(inputSize: 64, outputSize: 1000)
}
```

## Recurrent Neural Networks

Process sequential data with RNN cells:

```swift
struct SequenceClassifier: Layer {
    var embedding: Embedding
    var lstm: LSTMCell
    var output: Linear

    init(vocabSize: Int, embeddingDim: Int, hiddenSize: Int, numClasses: Int) {
        self.embedding = Embedding(vocabularySize: vocabSize, embeddingSize: embeddingDim)
        self.lstm = LSTMCell(inputSize: embeddingDim, hiddenSize: hiddenSize)
        self.output = Linear(inputSize: hiddenSize, outputSize: numClasses)
    }

    @differentiable
    func callAsFunction(_ input: Tensor) -> Tensor {
        // input shape: [batch, sequence_length]
        let batchSize = input.shape[0]
        let seqLength = input.shape[1]

        // Embed tokens
        let embedded = embedding(input)  // [batch, seq_len, embed_dim]

        // Initialize hidden and cell states
        var hidden = Tensor.zeros([batchSize, hiddenSize])
        var cell = Tensor.zeros([batchSize, hiddenSize])

        // Process sequence
        for t in 0..<seqLength {
            let x = embedded[.all, t, .all]  // [batch, embed_dim]
            (hidden, cell) = lstm(x, hidden: hidden, cell: cell)
        }

        // Use final hidden state for classification
        return output(hidden)
    }
}
```

## Transformer Models

Build attention-based architectures:

```swift
struct TransformerEncoder: Layer {
    var attention: MultiHeadAttention
    var norm1: LayerNorm
    var feedForward: Sequential
    var norm2: LayerNorm
    var dropout: Dropout

    init(embeddingDim: Int, numHeads: Int, hiddenDim: Int, dropoutProb: Float) {
        self.attention = MultiHeadAttention(
            embeddingDim: embeddingDim,
            numHeads: numHeads
        )
        self.norm1 = LayerNorm(normalizedShape: [embeddingDim])
        self.feedForward = Sequential {
            Linear(inputSize: embeddingDim, outputSize: hiddenDim)
            GELU()
            Linear(inputSize: hiddenDim, outputSize: embeddingDim)
        }
        self.norm2 = LayerNorm(normalizedShape: [embeddingDim])
        self.dropout = Dropout(probability: dropoutProb)
    }

    @differentiable
    func callAsFunction(_ input: Tensor) -> Tensor {
        // Self-attention with residual connection
        var x = input
        let attnOutput = attention(query: x, key: x, value: x)
        x = norm1(x + dropout(attnOutput))

        // Feed-forward with residual connection
        let ffOutput = feedForward(x)
        x = norm2(x + dropout(ffOutput))

        return x
    }
}
```

## Graph Neural Networks

Work with graph-structured data:

```swift
import Torch

struct GNNModel: Layer {
    var graphNet: GraphNetwork
    var nodeOutput: Linear

    init(nodeFeatureDim: Int, edgeFeatureDim: Int, hiddenDim: Int, outputDim: Int) {
        self.graphNet = GraphNetwork(
            nodeModelInputSize: nodeFeatureDim,
            edgeModelInputSize: edgeFeatureDim,
            nodeModelOutputSize: hiddenDim,
            edgeModelOutputSize: hiddenDim
        )
        self.nodeOutput = Linear(inputSize: hiddenDim, outputSize: outputDim)
    }

    @differentiable
    func callAsFunction(
        nodeFeatures: Tensor,
        edgeFeatures: Tensor,
        senders: Tensor,
        receivers: Tensor
    ) -> Tensor {
        // Message passing
        let updated = graphNet(
            nodeFeatures: nodeFeatures,
            edgeFeatures: edgeFeatures,
            senders: senders,
            receivers: receivers
        )

        // Node-level predictions
        return nodeOutput(updated.nodes)
    }
}
```

See <doc:GraphNeuralNetworks> for a complete guide to graph neural networks.

## Weight Initialization

Proper initialization is crucial for training deep networks:

```swift
struct CustomLayer: Layer {
    var weight: Tensor
    var bias: Tensor

    init(inputSize: Int, outputSize: Int) {
        // Kaiming uniform initialization (good for ReLU)
        self.weight = Tensor.kaimingUniform(
            [inputSize, outputSize],
            nonlinearity: .relu
        )

        // Zero initialization for biases
        self.bias = Tensor.zeros([outputSize])
    }

    @differentiable
    func callAsFunction(_ input: Tensor) -> Tensor {
        return input.matmul(weight) + bias
    }
}
```

Common initialization strategies:

```swift
// Xavier/Glorot uniform (good for tanh, sigmoid)
let weight = Tensor.xavierUniform([fanIn, fanOut])

// Kaiming/He initialization (good for ReLU)
let weight = Tensor.kaimingUniform([fanIn, fanOut], nonlinearity: .relu)

// Normal distribution
let weight = Tensor.randn([fanIn, fanOut]) * 0.01

// Uniform distribution in range
let weight = Tensor.rand([fanIn, fanOut]) * 2.0 - 1.0  // [-1, 1]
```

## Best Practices

### 1. Use Normalization Layers

Normalization improves training stability:

```swift
let model = Sequential {
    Linear(inputSize: 512, outputSize: 512)
    LayerNorm(normalizedShape: [512])  // or BatchNorm
    ReLU()
}
```

### 2. Add Dropout for Regularization

Prevent overfitting with dropout:

```swift
let model = Sequential {
    Linear(inputSize: 784, outputSize: 512)
    ReLU()
    Dropout(probability: 0.5)  // Drop 50% of activations during training
    Linear(inputSize: 512, outputSize: 10)
}
```

### 3. Choose Appropriate Activations

- **ReLU**: Default choice, fast and effective
- **GELU**: Used in transformers (BERT, GPT)
- **SiLU/Swish**: Smooth alternative to ReLU
- **Tanh**: For outputs in (-1, 1)
- **Sigmoid**: For binary outputs

```swift
// Modern activation: GELU
let transformer = Sequential {
    Linear(inputSize: 512, outputSize: 2048)
    GELU()
    Linear(inputSize: 2048, outputSize: 512)
}
```

### 4. Match Input/Output Dimensions

Ensure dimensions align through your network:

```swift
// Image: [N, C, H, W] = [32, 3, 224, 224]
Conv2D(inChannels: 3, outChannels: 64, ...)  // ✅ Matches input channels

// After pooling: [32, 64, 7, 7]
Flatten(startDim: 1)  // → [32, 64 * 7 * 7] = [32, 3136]

Linear(inputSize: 3136, outputSize: 1000)  // ✅ Matches flattened size
```

### 5. Organize Complex Models

Break large models into logical components:

```swift
struct VGGBlock: Layer {
    var layers: Sequential

    init(inChannels: Int, outChannels: Int, numConvs: Int) {
        var blockLayers: [any Layer] = []
        for i in 0..<numConvs {
            let channels = i == 0 ? inChannels : outChannels
            blockLayers.append(Conv2D(
                inChannels: channels,
                outChannels: outChannels,
                kernelSize: (3, 3),
                padding: (1, 1)
            ))
            blockLayers.append(ReLU())
        }
        blockLayers.append(MaxPool2D(kernelSize: (2, 2)))
        self.layers = Sequential(blockLayers)
    }

    @differentiable
    func callAsFunction(_ input: Tensor) -> Tensor {
        return layers(input)
    }
}
```

## Debugging Models

### Print Shapes

Track tensor shapes through your network:

```swift
func debugModel(_ input: Tensor) {
    var x = input
    print("Input:", x.shape)

    x = conv1(x)
    print("After conv1:", x.shape)

    x = pool(x)
    print("After pool:", x.shape)

    x = flatten(x)
    print("After flatten:", x.shape)

    x = linear(x)
    print("Output:", x.shape)
}
```

### Check for NaN/Inf

Monitor for numerical instability:

```swift
func checkTensor(_ tensor: Tensor, name: String) {
    if tensor.isnan().any().scalarValue {
        print("Warning: NaN detected in \(name)")
    }
    if tensor.isinf().any().scalarValue {
        print("Warning: Inf detected in \(name)")
    }
}
```

## Next Steps

- <doc:Examples-MNIST> - Complete CNN example
- <doc:Examples-Translation> - Sequence-to-sequence model
- <doc:AutomaticDifferentiation> - Understanding gradients
- <doc:GraphNeuralNetworks> - Working with graphs

## See Also

- ``Sequential``
- ``Layer``
- ``Linear``
- ``Conv2D``
- ``MultiHeadAttention``
- ``GraphNetwork``
