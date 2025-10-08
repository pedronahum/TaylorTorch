import _Differentiation

/// Ultra‑lightweight CIFAR‑10 architectures for small PCs.
/// 32×32 inputs, NCHW layout (mirrors Conv2D’s default). :
/// Uses only existing layers & your typed result‑builder.
/// (No loops inside builder blocks.)
public enum CIFARMiniArchitectures {

  // MARK: MiniCNN (~24k params)
  // 32→16→8 via two 2×2 maxpools, then global avg (8×8) → Dense(64→numClasses).
  // Channels: 16 → 32 → 64. All 3×3, `.same` padding.
  @inlinable
  public static func miniCNN(
    numClasses: Int = 10, base: Int = 16,
    pDrop: Double = 0.10
  ) -> some Layer {
    let c1 = base
    let c2 = base * 2
    let c3 = base * 4
    return SequentialBlock {
      // stage 1
      Conv2D.kaimingUniform(inC: 3, outC: c1, kH: 3, kW: 3, padding: .same)  // 32→32
      ReLU()
      Dropout(rate: pDrop)
      MaxPool2D(kernel: (2, 2))  // 32→16

      // stage 2
      Conv2D.kaimingUniform(inC: c1, outC: c2, kH: 3, kW: 3, padding: .same)  // 16→16
      ReLU()
      Dropout(rate: pDrop)
      MaxPool2D(kernel: (2, 2))  // 16→8

      // stage 3
      Conv2D.kaimingUniform(inC: c2, outC: c3, kH: 3, kW: 3, padding: .same)  // 8→8
      ReLU()

      // 8×8 → 1×1 (global avg for CIFAR: kernel must match spatial extent)
      AvgPool2D(kernel: (8, 8))
      Flatten()
      Dense(inFeatures: c3, outFeatures: numClasses)  // identity head
    }
  }

  // MARK: TinyVGG (~72k params)
  // VGG‑style with 3 stages × (2 conv 3×3 + ReLU) + pool. Channels: 16→32→64.
  // 32→16→8→4; global avg (4×4) → Dense(64→numClasses).
  @inlinable
  public static func tinyVGG(
    numClasses: Int = 10, base: Int = 16,
    pDrop: Double = 0.10
  ) -> some Layer {
    let c1 = base
    let c2 = base * 2
    let c3 = base * 4
    return SequentialBlock {
      // stage 1
      Conv2D.kaimingUniform(inC: 3, outC: c1, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c1, outC: c1, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 32→16
      Dropout(rate: pDrop)

      // stage 2
      Conv2D.kaimingUniform(inC: c1, outC: c2, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c2, outC: c2, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 16→8
      Dropout(rate: pDrop)

      // stage 3
      Conv2D.kaimingUniform(inC: c2, outC: c3, kH: 3, kW: 3, padding: .same)
      ReLU()
      Conv2D.kaimingUniform(inC: c3, outC: c3, kH: 3, kW: 3, padding: .same)
      ReLU()
      MaxPool2D(kernel: (2, 2))  // 8→4

      AvgPool2D(kernel: (4, 4))  // 4×4 → 1×1
      Flatten()
      Dense(inFeatures: c3, outFeatures: numClasses)
    }
  }

  // MARK: MicroMobileNetV1 (α≈0.25 by default)  — very few params (~2k)
  // Strided depthwise‑separable stacks reduce 32→16→8→4; global avg → Dense.
  // Uses your `DepthwiseSeparableConv2D` + ReLU blocks.
  @inlinable
  public static func microMobileNetV1(
    numClasses: Int = 10,
    alpha: Double = 0.25
  ) -> some Layer {
    @inline(__always) func ch(_ c: Int) -> Int { max(1, Int((Double(c) * alpha).rounded())) }
    @inline(__always) func ds(_ inC: Int, _ outC: Int, _ stride: Int) -> some Layer {
      SequentialBlock {
        DepthwiseSeparableConv2D(
          inC: inC, outC: outC, kH: 3, kW: 3,
          stride: (stride, stride), padding: .same, dilation: (1, 1),
          dataFormat: .nchw
        )
        ReLU()
      }
    }
    // 32→16 (s2), 16→8 (s2), 8→8 (s1), 8→4 (s2)
    return SequentialBlock {
      // stem
      Conv2D.kaimingUniform(inC: 3, outC: ch(16), kH: 3, kW: 3, padding: .same)
      ReLU()

      ds(ch(16), ch(32), 2)
      ds(ch(32), ch(64), 2)
      ds(ch(64), ch(64), 1)
      ds(ch(64), ch(128), 2)

      AvgPool2D(kernel: (4, 4))  // 4×4 → 1×1
      Flatten()
      Dense(inFeatures: ch(128), outFeatures: numClasses)
    }
  }

  // MARK: NanoMLP (~197k params at width=64; choose width=32 for ~98k)
  @inlinable
  public static func nanoMLP(
    numClasses: Int = 10,
    width: Int = 64,
    pDrop: Double = 0.2
  ) -> some Layer {
    SequentialBlock {
      Flatten()  // 3*32*32 = 3072
      Dense(inFeatures: 3 * 32 * 32, outFeatures: width)
      ReLU()
      Dropout(rate: pDrop)
      Dense(inFeatures: width, outFeatures: numClasses)  // identity head
    }
  }
}
