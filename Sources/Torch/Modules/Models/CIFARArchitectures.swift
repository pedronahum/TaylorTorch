import Foundation
import _Differentiation

// Public factories for classic CIFAR-10 networks.
// Conventions:
// - NCHW data format (default in your Conv2D).
// - `.same` padding for AlexNet/VGG/MobileNet so spatial halving is controlled by pooling/stride.
// - No stored differentiable closures; compose ReLU/Dropout as layers.

public enum CIFARArchitectures {

  // MARK: - LeNet (your existing layout)
  @inlinable
  public static func leNet(numClasses: Int = 10) -> some Layer {
    SequentialBlock {
      // feature extractor
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

      // classifier (32*3*3 = 288)
      Flatten()
      Dense(inFeatures: 32 * 3 * 3, outFeatures: 120)
      ReLU()
      Dropout(rate: 0.5)

      Dense(inFeatures: 120, outFeatures: 84)
      ReLU()
      Dropout(rate: 0.5)

      Dense(inFeatures: 84, outFeatures: numClasses)  // identity head
    }
  }

  // MARK: - AlexNet (CIFAR-sized)
  @inlinable
  public static func alexNetCIFAR(numClasses: Int = 10, pDrop: Double = 0.5) -> some Layer {
    SequentialBlock {
      Conv2D.kaimingUniform(inC: 3, outC: 64, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: 64, outC: 64, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 32→16

      Conv2D.kaimingUniform(inC: 64, outC: 192, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: 192, outC: 192, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 16→8

      Conv2D.kaimingUniform(inC: 192, outC: 384, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: 384, outC: 256, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: 256, outC: 256, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 8→4

      Flatten()  // 256 * 4 * 4 = 4096
      Dense(inFeatures: 4096, outFeatures: 4096)
      ReLU()
      Dropout(rate: pDrop)
      Dense(inFeatures: 4096, outFeatures: 4096)
      ReLU()
      Dropout(rate: pDrop)
      Dense(inFeatures: 4096, outFeatures: numClasses)
    }
  }

  // MARK: - VGG-11 (CIFAR-sized, loop-free builder)
  // Five max-pool layers reduce 32×32 → 1×1; head is a single Dense(c5 → classes).
  @inlinable
  public static func vgg11CIFAR(numClasses: Int = 10, base: Int = 64) -> some Layer {
    let c1 = base
    let c2 = base * 2
    let c3 = base * 4
    let c4 = base * 8
    let c5 = base * 16

    return SequentialBlock {
      // Stage 1: 1× conv
      Conv2D.kaimingUniform(inC: 3, outC: c1, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 32→16

      // Stage 2: 1× conv
      Conv2D.kaimingUniform(inC: c1, outC: c2, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 16→8

      // Stage 3: 2× conv
      Conv2D.kaimingUniform(inC: c2, outC: c3, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c3, outC: c3, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 8→4

      // Stage 4: 2× conv
      Conv2D.kaimingUniform(inC: c3, outC: c4, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c4, outC: c4, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 4→2

      // Stage 5: 2× conv
      Conv2D.kaimingUniform(inC: c4, outC: c5, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c5, outC: c5, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 2→1

      // Head
      Flatten()
      Dropout(rate: 0.5)
      Dense(inFeatures: c5, outFeatures: numClasses)
    }
  }

  // MARK: - VGG-16 (CIFAR-sized, loop-free builder)
  @inlinable
  public static func vgg16CIFAR(numClasses: Int = 10, base: Int = 64) -> some Layer {
    let c1 = base
    let c2 = base * 2
    let c3 = base * 4
    let c4 = base * 8
    let c5 = base * 16

    return SequentialBlock {
      // Stage 1: 2× conv
      Conv2D.kaimingUniform(inC: 3, outC: c1, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c1, outC: c1, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 32→16

      // Stage 2: 2× conv
      Conv2D.kaimingUniform(inC: c1, outC: c2, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c2, outC: c2, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 16→8

      // Stage 3: 3× conv
      Conv2D.kaimingUniform(inC: c2, outC: c3, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c3, outC: c3, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c3, outC: c3, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 8→4

      // Stage 4: 3× conv
      Conv2D.kaimingUniform(inC: c3, outC: c4, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c4, outC: c4, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c4, outC: c4, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 4→2

      // Stage 5: 3× conv
      Conv2D.kaimingUniform(inC: c4, outC: c5, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c5, outC: c5, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c5, outC: c5, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 2→1

      // Head
      Flatten()
      Dropout(rate: 0.5)
      Dense(inFeatures: c5, outFeatures: numClasses)
    }
  }

  // MARK: - MobileNetV1 (CIFAR-sized) using DepthwiseSeparableConv2D
  @inlinable
  public static func mobileNetV1CIFAR(numClasses: Int = 10, alpha: Double = 1.0) -> some Layer {
    @inline(__always) func ch(_ c: Int) -> Int { max(1, Int((Double(c) * alpha).rounded())) }
    @inline(__always) func ds(_ inC: Int, _ outC: Int, _ s: Int) -> some Layer {
      SequentialBlock {
        DepthwiseSeparableConv2D(
          inC: inC, outC: outC, kH: 3, kW: 3,
          stride: (s, s), padding: .same, dilation: (1, 1),
          dataFormat: .nchw)
        ReLU()
      }
    }

    return SequentialBlock {
      // stem
      Conv2D.kaimingUniform(inC: 3, outC: ch(32), kH: 3, kW: 3, stride: (1, 1), padding: .same)
      ReLU()

      // depthwise separable stages
      ds(ch(32), ch(64), 1)
      ds(ch(64), ch(128), 2)
      ds(ch(128), ch(128), 1)
      ds(ch(128), ch(256), 2)
      ds(ch(256), ch(256), 1)
      ds(ch(256), ch(512), 2)
      ds(ch(512), ch(512), 1)
      ds(ch(512), ch(512), 1)
      ds(ch(512), ch(512), 1)
      ds(ch(512), ch(512), 1)
      ds(ch(512), ch(512), 1)
      ds(ch(512), ch(1024), 2)
      ds(ch(1024), ch(1024), 1)

      // CIFAR head: 2×2 → 1×1
      AvgPool2D(kernel: (2, 2))
      Flatten()
      Dense(inFeatures: ch(1024), outFeatures: numClasses)
    }
  }

  // MARK: - Compact MLP (baseline)
  @inlinable
  public static func mlp(numClasses: Int = 10, width: Int = 512) -> some Layer {
    SequentialBlock {
      Flatten()
      Dense(inFeatures: 3 * 32 * 32, outFeatures: width)
      ReLU()
      Dropout(rate: 0.5)
      Dense(inFeatures: width, outFeatures: numClasses)
    }
  }
}
