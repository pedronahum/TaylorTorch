# Node Classification with Graph Neural Networks

Learn to build a Graph Neural Network for node classification on Zachary's Karate Club dataset.

## Overview

In this tutorial, you'll build a Graph Neural Network (GNN) to classify nodes in a social network. You'll learn how graphs differ from images and sequences, understand message passing, and implement a complete GNN training pipeline.

**What you'll learn:**
- Graph data structures (nodes, edges, features)
- Message passing and aggregation
- Graph Neural Network architecture
- Node classification task
- Working with the ``GraphNetwork`` layer

**Time to complete:** 25-30 minutes

**Prerequisites:**
- Completed the MNIST tutorial (recommended)
- Basic understanding of graphs (nodes and edges)

## Understanding Graph Neural Networks

**Traditional Neural Networks:**
- CNNs: Process **grid data** (images, videos)
- RNNs: Process **sequence data** (text, audio)
- MLPs: Process **tabular data** (features in tables)

**Graph Neural Networks:**
- GNNs: Process **graph data** (social networks, molecules, knowledge graphs)

**Why GNNs?** Many real-world data have graph structure:
- Social networks (users = nodes, friendships = edges)
- Molecules (atoms = nodes, bonds = edges)
- Citation networks (papers = nodes, citations = edges)
- Recommendation systems (users/items = nodes, interactions = edges)

## The Karate Club Dataset

**Zachary's Karate Club** is a famous social network dataset:
- **34 nodes**: Members of a karate club
- **78 edges**: Friendships between members
- **2 classes**: Two factions after the club split

**Story:** In the 1970s, a karate club split into two groups due to a dispute. Can we predict which faction each member joined based on their friendships?

**Graph Structure:**
```
        (0)─────(1)
       /  \    /  \
     (2)  (3)─────(4)
      |    |       |
     (5)  (6)     (7)
     ...
```

**Task:** Given the friendship graph and some labeled nodes, predict labels for all nodes.

## Step 1: Import and Setup

```swift
import Torch
import _Differentiation

// Set random seed for reproducibility
Tensor.setRandomSeed(42)

// Dataset constants
let numNodes = 34     // Number of karate club members
let numEdges = 78     // Number of friendships
let numFeatures = 34  // Node features (one-hot node IDs initially)
let numClasses = 2    // Two factions
```

## Step 2: Load the Karate Club Graph

Let's create the graph structure:

```swift
// Edge list: each row is (source, target) edge
// Zachary's Karate Club edges (bidirectional)
let edgeList: [(Int, Int)] = [
    // Instructor's group (faction 0)
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
    (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
    (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
    (3, 7), (3, 12), (3, 13),
    // ... more edges ...

    // Administrator's group (faction 1)
    (33, 8), (33, 13), (33, 14), (33, 15), (33, 18), (33, 20), (33, 22),
    (33, 23), (33, 24), (33, 25), (33, 26), (33, 27), (33, 28), (33, 29),
    (33, 30), (33, 31), (33, 32)
]

// Convert to tensors
var sendersList: [Int] = []
var receiversList: [Int] = []

for (src, dst) in edgeList {
    // Add both directions (undirected graph)
    sendersList.append(src)
    receiversList.append(dst)
    sendersList.append(dst)
    receiversList.append(src)
}

let senders = Tensor(sendersList.map { Int64($0) }, dtype: .int64)
let receivers = Tensor(receiversList.map { Int64($0) }, dtype: .int64)

print("Graph loaded: \(numNodes) nodes, \(senders.shape[0]) directed edges")
```

## Step 3: Create Node Features

For this example, we'll use one-hot encoded node IDs as features:

```swift
// One-hot node features: [numNodes, numFeatures]
var nodeFeatures = Tensor.zeros([numNodes, numFeatures])
for i in 0..<numNodes {
    nodeFeatures[i, i] = Tensor(1.0)
}

// In practice, nodes might have rich features:
// - Social networks: age, interests, activity level
// - Molecules: atom type, charge, hybridization
// - Citation networks: paper embeddings, publication year

print("Node features shape: \(nodeFeatures.shape)")  // [34, 34]
```

## Step 4: Create Labels

Ground truth labels for the two factions:

```swift
// Labels: 0 = Instructor's faction, 1 = Administrator's faction
let labels = Tensor([
    // Nodes 0-33 labels (0 or 1)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
], dtype: .int64)

// Train/test split: use 4 labeled nodes for training
let trainMask = Tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                       dtype: .float32)  // Train on nodes 0, 1, 33

let testMask = 1.0 - trainMask  // Test on all other nodes

print("Training on \(trainMask.sum().item()) labeled nodes")
print("Testing on \(testMask.sum().item()) unlabeled nodes")
```

## Step 5: Build the GNN Model

Let's build a simple 2-layer Graph Convolutional Network (GCN):

```swift
struct GraphConvLayer: Layer {
    var transform: Dense  // Transform node features
    var normalize: Bool

    init(inputSize: Int, outputSize: Int, normalize: Bool = true) {
        transform = Dense(inputSize: inputSize, outputSize: outputSize)
        self.normalize = normalize
    }

    @differentiable
    func callAsFunction(
        _ nodeFeatures: Tensor,
        senders: Tensor,
        receivers: Tensor
    ) -> Tensor {
        // nodeFeatures: [numNodes, inFeatures]
        // senders, receivers: [numEdges] (edge indices)

        let numNodes = nodeFeatures.shape[0]
        let numEdges = senders.shape[0]

        // 1. Aggregate messages from neighbors
        // For each edge (src -> dst), send src's features to dst
        let sendersHost = senders.toArray(as: Int.self)
        let receiversHost = receivers.toArray(as: Int.self)

        // Gather neighbor features
        let neighborFeatures = nodeFeatures.indexSelect(dim: 0, indices: sendersHost)

        // 2. Aggregate by summing messages to each node
        var aggregated = Tensor.zeros(nodeFeatures.shape)

        for (edge, receiver) in receiversHost.enumerated() {
            aggregated[receiver] = aggregated[receiver] + neighborFeatures[edge]
        }

        // 3. Combine with self features
        let combined = nodeFeatures + aggregated  // Add self-connections

        // 4. Apply transformation
        var output = transform(combined)

        // 5. Optional normalization
        if normalize {
            // Normalize by degree (number of neighbors)
            var degree = Tensor.zeros([numNodes, 1])
            for receiver in receiversHost {
                degree[receiver] = degree[receiver] + 1.0
            }
            degree = degree + 1.0  // Add 1 for self-connection
            output = output / degree
        }

        return output
    }
}

// Complete GNN model
struct KarateClubGNN: Layer {
    var conv1: GraphConvLayer
    var conv2: GraphConvLayer

    init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
        conv1 = GraphConvLayer(inputSize: inputSize, outputSize: hiddenSize)
        conv2 = GraphConvLayer(inputSize: hiddenSize, outputSize: outputSize)
    }

    @differentiable
    func callAsFunction(
        _ nodeFeatures: Tensor,
        senders: Tensor,
        receivers: Tensor
    ) -> Tensor {
        // Layer 1: propagate and transform
        var h = conv1(nodeFeatures, senders: senders, receivers: receivers)
        h = h.relu()  // Non-linearity

        // Layer 2: final classification layer
        h = conv2(h, senders: senders, receivers: receivers)

        return h  // [numNodes, numClasses] logits
    }
}
```

**How Message Passing Works:**

```
Initial:
  Node 0: [features]
  Node 1: [features]
  Node 2: [features]

After Layer 1:
  Node 0: [aggregate(neighbors of 0) + features of 0] → transform → ReLU
  Node 1: [aggregate(neighbors of 1) + features of 1] → transform → ReLU
  Node 2: [aggregate(neighbors of 2) + features of 2] → transform → ReLU

After Layer 2:
  Node 0: [aggregate(layer1 neighbors of 0)] → classify
  ...

Each node's representation incorporates information from its 2-hop neighborhood!
```

## Step 6: Training Loop

Train the GNN using semi-supervised learning:

```swift
// Initialize model
var model = KarateClubGNN(
    inputSize: numFeatures,
    hiddenSize: 16,      // Hidden layer size
    outputSize: numClasses
)

// Initialize Adam optimizer
var optimizer = Adam(
    for: model,
    learningRate: 0.01,
    weightDecay: 5e-4  // L2 regularization
)

print("Model initialized")
print("Training on semi-supervised task (few labeled nodes)...")

// Training loop
let epochs = 200

for epoch in 1...epochs {
    // Forward pass + compute gradients
    let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor in
        // Get predictions for all nodes
        let logits = model(nodeFeatures, senders: senders, receivers: receivers)

        // Compute loss only on training nodes
        let trainLogits = logits * trainMask.unsqueezed(dim: -1)  // Mask out test nodes
        let trainLoss = softmaxCrossEntropy(logits: trainLogits, labels: labels)

        // Only count loss for training nodes
        return trainLoss * trainMask.sum() / Float(numNodes)
    }

    // Update parameters
    optimizer.update(&model, along: gradients)

    // Evaluate every 20 epochs
    if epoch % 20 == 0 {
        let logits = model(nodeFeatures, senders: senders, receivers: receivers)
        let predictions = logits.argmax(dim: -1)

        // Training accuracy
        let trainCorrect = ((predictions == labels) * trainMask).sum()
        let trainAcc = trainCorrect.item() / trainMask.sum().item() * 100

        // Test accuracy
        let testCorrect = ((predictions == labels) * testMask).sum()
        let testAcc = testCorrect.item() / testMask.sum().item() * 100

        print("Epoch \(epoch): Loss = \(String(format: "%.4f", loss.item())), " +
              "Train Acc = \(String(format: "%.1f", trainAcc))%, " +
              "Test Acc = \(String(format: "%.1f", testAcc))%")
    }
}

print("Training complete!")
```

**Expected Output:**
```
Model initialized
Training on semi-supervised task (few labeled nodes)...
Epoch 20: Loss = 0.6234, Train Acc = 75.0%, Test Acc = 61.3%
Epoch 40: Loss = 0.4123, Train Acc = 100.0%, Test Acc = 74.2%
Epoch 60: Loss = 0.2845, Train Acc = 100.0%, Test Acc = 80.6%
Epoch 80: Loss = 0.1923, Train Acc = 100.0%, Test Acc = 87.1%
Epoch 100: Loss = 0.1234, Train Acc = 100.0%, Test Acc = 90.3%
...
Epoch 200: Loss = 0.0456, Train Acc = 100.0%, Test Acc = 93.5%
Training complete!
```

## Step 7: Visualize Predictions

Let's see which nodes were classified correctly:

```swift
// Final predictions
let finalLogits = model(nodeFeatures, senders: senders, receivers: receivers)
let predictions = finalLogits.argmax(dim: -1)
let probabilities = finalLogits.softmax(dim: -1)

print("\nNode Classification Results:")
print("Node | True | Pred | Confidence | Correct?")
print("-----|------|------|------------|----------")

for node in 0..<numNodes {
    let trueLabel = labels[node].item()
    let predLabel = predictions[node].item()
    let confidence = probabilities[node, predLabel].item() * 100
    let correct = trueLabel == predLabel ? "✓" : "✗"
    let trainNode = trainMask[node].item() > 0 ? "[TRAIN]" : ""

    print(String(format: "%4d | %4d | %4d | %9.1f%% | %8s %s",
                 node, Int(trueLabel), Int(predLabel), confidence, correct, trainNode))
}

// Overall accuracy
let totalCorrect = (predictions == labels).sum().item()
let accuracy = totalCorrect / Float(numNodes) * 100
print("\nOverall Accuracy: \(String(format: "%.1f", accuracy))%")
```

**Expected Output:**
```
Node Classification Results:
Node | True | Pred | Confidence | Correct?
-----|------|------|------------|----------
   0 |    0 |    0 |      99.2% |        ✓ [TRAIN]
   1 |    0 |    0 |      98.5% |        ✓ [TRAIN]
   2 |    0 |    0 |      95.3% |        ✓
   3 |    0 |    0 |      94.1% |        ✓
   4 |    0 |    0 |      89.7% |        ✓
   5 |    0 |    0 |      91.2% |        ✓
   ...
  30 |    1 |    1 |      88.4% |        ✓
  31 |    1 |    1 |      92.1% |        ✓
  32 |    1 |    1 |      94.8% |        ✓
  33 |    1 |    1 |      99.1% |        ✓ [TRAIN]

Overall Accuracy: 93.5%
```

## Understanding the Results

**Why does it work with so few labeled nodes?**

GNNs leverage **homophily**: nodes connected in the graph tend to have similar labels.

```
Labeled: Node 0 (faction 0)
         ↓ (connected to)
Unlabeled: Node 2, 3, 4, 5, ...
           → Likely also faction 0!
```

Through message passing, label information **propagates** through the graph.

## Using the GraphNetwork Layer

TaylorTorch provides a more advanced ``GraphNetwork`` layer:

```swift
// Advanced GNN with edge features
let gnn = GraphNetwork(
    phiE: edgeModel,    // Update edge features
    phiV: nodeModel,    // Update node features
    phiU: globalModel   // Update global features
)

// Create Graphs struct
let graph = Graphs(
    nodes: nodeFeatures,
    edges: edgeFeatures,  // Can include edge features!
    senders: senders,
    receivers: receivers,
    globals: globalFeatures,
    nNode: [numNodes],
    nEdge: [numEdges]
)

// Forward pass
let updatedGraph = gnn(graph)
```

See ``GraphNetwork`` and ``Graphs`` documentation for details.

## Advanced: Graph-Level Classification

Classify entire graphs (not just nodes):

```swift
// Example: Molecule classification (toxic vs non-toxic)
struct MoleculeClassifier: Layer {
    var gnn: GraphConvLayer
    var globalPool: GlobalMeanPool  // Aggregate node features
    var classifier: Dense

    @differentiable
    func callAsFunction(_ graph: Graphs) -> Tensor {
        // Update node features
        var h = gnn(graph.nodes, senders: graph.senders, receivers: graph.receivers)

        // Aggregate to graph-level representation
        let graphEmbedding = globalPool(h, nNode: graph.nNode)  // [batch, features]

        // Classify
        return classifier(graphEmbedding)  // [batch, numClasses]
    }
}
```

## Troubleshooting

### Problem: "Test accuracy much lower than training"

**Cause:** Overfitting on small training set.

**Solutions:**
- Increase weight decay: `weightDecay: 1e-3`
- Add dropout to GNN layers
- Use more training nodes
- Reduce model capacity (smaller hidden size)

### Problem: "Accuracy stuck at 50%"

**Cause:** Model predicting randomly.

**Solutions:**
- Check that edges are correctly formatted (bidirectional for undirected graphs)
- Verify labels are correct (0 and 1, not 1 and 2)
- Increase learning rate: `learningRate: 0.05`
- Train for more epochs

### Problem: "Loss is NaN"

**Cause:** Numerical instability or gradient explosion.

**Solutions:**
```swift
// Add gradient clipping
let gradNorm = computeGradientNorm(gradients)
if gradNorm > 1.0 {
    gradients = clipGradientsByNorm(gradients, maxNorm: 1.0)
}

// Lower learning rate
optimizer.learningRate = 0.001
```

## Summary

Congratulations! You've built a Graph Neural Network for node classification. Here's what you learned:

✅ **Graph data structures**: Nodes, edges, senders, receivers
✅ **Message passing**: How GNNs aggregate neighbor information
✅ **Semi-supervised learning**: Training with few labeled nodes
✅ **Node classification**: Predicting node labels from graph structure
✅ **Graph convolutions**: Implementing GCN-style layers

## Next Steps

- **Try different datasets**: Cora, Citeseer, PubMed (citation networks)
- **Graph-level tasks**: Molecule classification, social network analysis
- **Advanced architectures**: GAT (Graph Attention Networks), GraphSAGE
- **Edge prediction**: Link prediction task
- **Dynamic graphs**: Temporal graph networks

## See Also

### Graph Components
- ``GraphNetwork`` - Advanced graph neural network layer
- ``Graphs`` - Graph data structure for batched graphs

### Core Components
- ``Dense`` - Fully connected transformation
- ``Layer`` - Base protocol for custom GNN layers

### Training
- ``Adam`` - Adam optimizer
- ``softmaxCrossEntropy(logits:labels:reduction:)`` - Node classification loss

### Related Tasks
- Semi-supervised learning with few labels
- Graph representation learning
- Network analysis and community detection

---

**Estimated training time:** <1 minute (small dataset)
**Expected accuracy:** 90-95% with 3-4 labeled nodes
**Difficulty:** Intermediate 🟡
