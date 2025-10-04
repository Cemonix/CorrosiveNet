# CorrosiveNet

A high-performance neural network library written in Rust with optional CUDA acceleration for GPU training.

## Overview

CorrosiveNet provides a PyTorch-inspired API for building and training neural networks in Rust. It features a flexible tensor library with multi-dimensional array operations, automatic differentiation, and modular neural network components.

### Key Features

- **Flexible Tensor Operations**: Multi-dimensional arrays with broadcasting, reshaping, and linear algebra operations
- **GPU Acceleration**: Optional CUDA support for GPU-accelerated training and inference
- **Modular Architecture**: Composable layers, activation functions, loss functions, and optimizers
- **Memory Efficient**: PyTorch-style exclusive device storage - tensors live on a single device at a time
- **Type Safety**: Leverages Rust's type system for safe, zero-cost abstractions

## Project Structure

The library is organized into three main crates:

- **corrosive-tensor**: Core tensor operations and multi-dimensional array manipulation
- **corrosive-nn**: Neural network building blocks (layers, activations, losses, optimizers)
- **corrosive-cuda**: CUDA backend for GPU acceleration (optional)

## Installation

Add CorrosiveNet to your `Cargo.toml`:

```toml
[dependencies]
corrosive-nn = "0.1.0"
```

For CUDA support, enable the `cuda` feature:

```toml
[dependencies]
corrosive-nn = { version = "0.1.0", features = ["cuda"] }
```

## Quick Start

### CPU Training

```rust
use corrosive_nn::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a simple neural network
    let mut model = Sequential::new()
        .add(Linear::new(784, 128))
        .add(ReLU::new())
        .add(Linear::new(128, 10));

    // Create optimizer and loss function
    let mut optimizer = SGD::new(0.01);
    let loss_fn = CrossEntropyLoss::new();

    // Training loop
    for epoch in 0..10 {
        for (inputs, targets) in dataloader {
            let outputs = model.forward(&inputs)?;
            let loss = loss_fn.forward(&outputs, &targets)?;

            let grad = loss_fn.backward()?;
            model.backward(&grad)?;

            optimizer.step(&mut model.parameters())?;
            optimizer.zero_grad();
        }
    }

    Ok(())
}
```

### GPU Training

```rust
use corrosive_nn::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Select device
    let device = if cuda_available() {
        Device::cuda()
    } else {
        Device::CPU
    };

    // Build and move model to GPU
    let mut model = Sequential::new()
        .add(Linear::new(784, 128))
        .add(ReLU::new())
        .add(Linear::new(128, 10))
        .to(device)?;

    let mut optimizer = SGD::new(0.01);
    let loss_fn = CrossEntropyLoss::new();

    for epoch in 0..10 {
        for (inputs, targets) in dataloader {
            // Move batch to GPU
            let inputs = inputs.to(device)?;
            let targets = targets.to(device)?;

            // All operations happen on GPU
            let outputs = model.forward(&inputs)?;
            let loss = loss_fn.forward(&outputs, &targets)?;

            let grad = loss_fn.backward()?;
            model.backward(&grad)?;

            optimizer.step(&mut model.parameters())?;
            optimizer.zero_grad();
        }
    }

    Ok(())
}
```

## Building

### CPU-only Build

```bash
cargo build
cargo test
```

### CUDA Build

Requires NVIDIA CUDA Toolkit 12.8 or later.

```bash
cargo build --features cuda
cargo test --features cuda
```

## Development Status

CorrosiveNet is under active development. Current implementation status:

- **Tensor Operations**: Element-wise operations, broadcasting, reshaping, linear algebra (in progress)
- **CUDA Support**: Element-wise operations, scalar operations, device transfers (in progress)
- **Neural Network Components**: Layers, activations, losses, optimizers (in progress)
- **Training Infrastructure**: Backpropagation, parameter updates (planned)

## License

Licensed under the GNU General Public License v3.0 or later. See [LICENSE](LICENSE) for details.

## Requirements

- Rust 2024 edition or later
- For CUDA support: NVIDIA GPU with CUDA Toolkit 12.8+
