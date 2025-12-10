#!/usr/bin/env python3
"""
PyTorch MNIST benchmark matching the TaylorTorch Swift implementation exactly.

This script mirrors the Swift MNIST example for performance comparison:
- Same model architecture (Conv2D -> Dropout -> BatchNorm -> ReLU -> AvgPool2D -> ...)
- Same hyperparameters (batch_size=128, lr=1e-3, epochs=3)
- Same training loop structure
- Same timing measurements (forward, backward, optimizer)

Note: This uses pure PyTorch without torchvision, loading MNIST directly.
"""

import argparse
import gzip
import os
import struct
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingConfig:
    """CLI-adjustable hyperparameters matching the Swift TrainingConfig."""
    epochs: int = 3
    batch_size: int = 128
    eval_batch_size: int = 1024
    log_interval: int = 100
    learning_rate: float = 1e-3
    shuffle_seed: int = 0xFEED_CAFE
    max_batches_per_epoch: int | None = None


def download_mnist(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Downloads and loads MNIST dataset without torchvision."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Original URL is down, using ossci-datasets mirror (same as PyTorch uses)
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    def download_file(filename: str) -> Path:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        return filepath

    def read_images(filepath: Path) -> torch.Tensor:
        with gzip.open(filepath, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            assert magic == 2051, f"Invalid magic number {magic} for images"
            data = f.read()
            images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            images = images.reshape(num, 1, rows, cols).float() / 255.0
            return images

    def read_labels(filepath: Path) -> torch.Tensor:
        with gzip.open(filepath, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            assert magic == 2049, f"Invalid magic number {magic} for labels"
            data = f.read()
            labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
            return labels

    train_images = read_images(download_file(files["train_images"]))
    train_labels = read_labels(download_file(files["train_labels"]))
    test_images = read_images(download_file(files["test_images"]))
    test_labels = read_labels(download_file(files["test_labels"]))

    return train_images, train_labels, test_images, test_labels


class MNISTModel(nn.Module):
    """
    CNN architecture matching the Swift Sequential model exactly:

    Block 1:
      - Conv2D(1 -> 32, kernel=3x3, padding=1)
      - Dropout(p=0.025)
      - BatchNorm(32)
      - ReLU
      - AvgPool2D(kernel=2x2, stride=2x2)

    Block 2:
      - Conv2D(32 -> 64, kernel=3x3, padding=1)
      - BatchNorm(64)
      - ReLU
      - AvgPool2D(kernel=2x2, stride=2x2)

    Head:
      - Flatten -> [N, 64*7*7] = [N, 3136]
      - Linear(3136 -> 256)
      - Dropout(p=0.025)
      - ReLU
      - Linear(256 -> 10)
    """

    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(p=0.025)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Head
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.dropout2 = nn.Dropout(p=0.025)
        self.fc2 = nn.Linear(256, 10)

        # Initialize weights with Kaiming uniform (matching Swift kaimingUniformInChannels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Head
        x = x.flatten(start_dim=1)  # [N, 64*7*7]
        x = self.fc1(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """Computes the number of correct predictions in a batch."""
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return int(correct), labels.shape[0]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluates a model over the provided data loader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction='sum')
            correct, batch_total = batch_accuracy(logits, labels)

            total_loss += loss.item()
            total_correct += correct
            total_samples += batch_total

    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return mean_loss, accuracy


def parse_args() -> tuple[TrainingConfig, str]:
    """Parses command-line arguments into a TrainingConfig."""
    parser = argparse.ArgumentParser(description='PyTorch MNIST Benchmark')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=1024)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0xFEED_CAFE)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])

    args = parser.parse_args()
    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        log_interval=args.log_interval,
        learning_rate=args.learning_rate,
        shuffle_seed=args.seed,
        max_batches_per_epoch=args.max_batches,
    ), args.device


def main():
    config, device_name = parse_args()

    # Select device
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_name == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"PyTorch MNIST * epochs: {config.epochs}, batch: {config.batch_size}, device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Set seed for reproducibility
    torch.manual_seed(config.shuffle_seed)

    print("Preparing MNIST dataset...")
    data_dir = Path(__file__).parent / "data"
    train_images, train_labels, test_images, test_labels = download_mnist(data_dir)

    print(f"Loaded MNIST with {len(train_images)} training and {len(test_images)} test samples")

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = MNISTModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    steps_per_epoch = (len(train_dataset) + config.batch_size - 1) // config.batch_size
    start_time = time.perf_counter()

    # Performance timing accumulators
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_optimizer_time = 0.0
    total_batches = 0

    for epoch in range(1, config.epochs + 1):
        # Create shuffled loader for this epoch with different seed
        generator = torch.Generator()
        generator.manual_seed(config.shuffle_seed + epoch)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        block_loss = 0.0
        block_correct = 0
        block_samples = 0
        step = 0

        for images, labels in train_loader:
            step += 1
            if config.max_batches_per_epoch is not None and step > config.max_batches_per_epoch:
                break

            images, labels = images.to(device), labels.to(device)

            # Time the forward pass (includes loss computation)
            forward_start = time.perf_counter()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            forward_end = time.perf_counter()

            # Time the backward pass (gradient computation)
            backward_start = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            backward_end = time.perf_counter()

            # Compute accuracy (using same forward pass result)
            correct, batch_total = batch_accuracy(logits, labels)
            loss_value = loss.item()

            # Time the optimizer update
            optimizer_start = time.perf_counter()
            optimizer.step()
            optimizer_end = time.perf_counter()

            # Accumulate timing statistics
            total_forward_time += forward_end - forward_start
            total_backward_time += backward_end - backward_start
            total_optimizer_time += optimizer_end - optimizer_start
            total_batches += 1

            running_loss += loss_value * batch_total
            running_correct += correct
            running_samples += batch_total

            block_loss += loss_value * batch_total
            block_correct += correct
            block_samples += batch_total

            if step % config.log_interval == 0:
                avg_loss = block_loss / block_samples
                avg_acc = block_correct / block_samples
                elapsed = time.perf_counter() - start_time
                print(f"epoch {epoch} * step {step}/{steps_per_epoch} * "
                      f"loss {avg_loss:.4f} * acc {avg_acc * 100:.2f}% * {elapsed:.1f}s")
                block_loss = 0.0
                block_correct = 0
                block_samples = 0

        if block_samples > 0:
            avg_loss = block_loss / block_samples
            avg_acc = block_correct / block_samples
            elapsed = time.perf_counter() - start_time
            print(f"epoch {epoch} * step {step}/{steps_per_epoch} * "
                  f"loss {avg_loss:.4f} * acc {avg_acc * 100:.2f}% * {elapsed:.1f}s")

        epoch_loss = running_loss / running_samples
        epoch_acc = running_correct / running_samples
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch} done * train loss {epoch_loss:.4f} * train acc {epoch_acc * 100:.2f}% * "
              f"val loss {val_loss:.4f} * val acc {val_acc * 100:.2f}%")

    total_elapsed = time.perf_counter() - start_time
    print(f"Training finished in {total_elapsed:.1f}s")

    # Print performance timing summary
    print("\n=== Performance Timing Summary ===")
    print(f"Total batches processed: {total_batches}")
    print(f"Total forward pass time:   {total_forward_time:.3f}s")
    print(f"Total backward pass time:  {total_backward_time:.3f}s")
    print(f"Total optimizer time:      {total_optimizer_time:.3f}s")
    print(f"Avg forward pass time:     {(total_forward_time / total_batches) * 1000:.4f}ms")
    print(f"Avg backward pass time:    {(total_backward_time / total_batches) * 1000:.4f}ms")
    print(f"Avg optimizer time:        {(total_optimizer_time / total_batches) * 1000:.4f}ms")
    total_compute_time = total_forward_time + total_backward_time + total_optimizer_time
    print(f"Total compute time:        {total_compute_time:.3f}s ({(total_compute_time / total_elapsed) * 100:.1f}% of wall time)")


if __name__ == '__main__':
    main()
