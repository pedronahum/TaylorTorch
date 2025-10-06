/// Data format for 2D layers.
/// - `.nchw`: Batch, channel, height, width layout.
/// - `.nhwc`: Batch, height, width, channel layout.
public enum DataFormat { case nchw, nhwc }
