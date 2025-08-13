# NA_transformer - Neural Architecture Transformer

A PyTorch implementation of a multi-layer transformer network with Apple NPU (Neural Processing Unit) acceleration support.

## Project Structure

```
NA_transformer/
├── README.md
├── libs/
│   ├── transformer.py      # Single-layer transformer implementation
│   └── network.py          # Multi-layer network with training capabilities
└── training_example.py     # Example training script
```

## Features

### 🚀 Core Components

- **Single-layer Transformer** (`transformer.py`):
  - Multi-head self-attention mechanism
  - Position-wise feed-forward networks
  - Layer normalization and residual connections
  - Positional encoding

- **Multi-layer Network** (`network.py`):
  - Stacks multiple transformer layers
  - Complete training pipeline with backpropagation
  - Adam optimizer with weight decay
  - Text generation capabilities
  - Support for causal and padding masks

### ⚡ Hardware Acceleration

- **Apple NPU Support**: Optimized for Apple Silicon with Metal Performance Shaders (MPS)
- **CUDA Support**: Compatible with NVIDIA GPUs
- **CPU Fallback**: Works on any system

### 🎯 Training Features

- **Backpropagation**: Full gradient computation and parameter updates
- **Adam Optimizer**: With configurable learning rate, weight decay, and beta parameters
- **Mixed Precision Training**: Optional for faster training on supported hardware
- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: Easy to integrate custom schedulers

## Installation

1. **Install PyTorch** with NPU support:
```bash
pip3 install --user --break-system-packages torch torchvision torchaudio
```

2. **Verify NPU support**:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

## Quick Start

### Basic Usage

```python
from libs.network import Network
import torch

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Create network
network = Network(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    max_seq_length=256
).to(device)

# Setup optimizer
optimizer = network.setup_optimizer(learning_rate=1e-4)

print(f"Network has {sum(p.numel() for p in network.parameters()):,} parameters")
```

### Training

```python
# Prepare your data
batch = {
    'input_ids': torch.randint(1, vocab_size, (batch_size, seq_length)).to(device),
    'target_ids': torch.randint(1, vocab_size, (batch_size, seq_length)).to(device)
}

# Training step with backpropagation
network.train()
loss = network.train_step(batch, optimizer)
print(f"Loss: {loss:.4f}")

# Full epoch training
train_loss = network.train_epoch(
    dataloader=train_loader,
    optimizer=optimizer,
    device=device,
    gradient_clip_norm=1.0
)
```

### Text Generation

```python
# Generate text
network.eval()
input_sequence = torch.randint(1, 100, (1, 10)).to(device)

# Greedy generation
generated = network.generate(
    input_sequence, 
    max_length=50, 
    do_sample=False
)

# Sampling with temperature
generated = network.generate(
    input_sequence, 
    max_length=50, 
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    do_sample=True
)
```

## Architecture Details

### Network Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Vocabulary size | Required |
| `d_model` | Model dimension | 512 |
| `num_heads` | Number of attention heads | 8 |
| `d_ff` | Feed-forward dimension | 2048 |
| `num_layers` | Number of transformer layers | 6 |
| `max_seq_length` | Maximum sequence length | 1000 |
| `dropout` | Dropout rate | 0.1 |

### Model Components

1. **Token Embedding**: Converts token IDs to dense vectors
2. **Positional Encoding**: Adds position information using sine/cosine functions
3. **Transformer Layers**: Stack of self-attention and feed-forward blocks
4. **Layer Normalization**: Applied before each sub-layer
5. **Residual Connections**: Skip connections around each sub-layer
6. **Output Projection**: Maps hidden states to vocabulary logits

## Performance

### NPU Acceleration Results

- **Device**: Apple NPU (MPS)
- **Model Size**: ~29M parameters (6 layers, 512d model)
- **Training Speed**: ~3.4s per epoch (125 batches)
- **Memory Efficiency**: Optimized for Apple Silicon

### Benchmark (6-layer network, 512d model)

| Batch Size | Sequence Length | NPU Time/Batch | Parameters |
|------------|-----------------|-----------------|------------|
| 8 | 64 | ~27ms | 29.2M |
| 4 | 128 | ~45ms | 29.2M |
| 2 | 256 | ~80ms | 29.2M |

## Example Training Output

```
Using Apple NPU (MPS) acceleration
Network created with 5,724,552 parameters

==================================================
Starting Training
==================================================

Epoch 1/3
------------------------------
Batch    0/ 125 | Loss: 8.5812 | Time: 0.63s
Batch   20/ 125 | Loss: 8.5649 | Time: 1.16s
...

Epoch 1 Results:
  Training Loss:   8.5592
  Validation Loss: 8.5454
```

## Advanced Features

### Custom Training Loop

```python
# Custom training with validation
for epoch in range(num_epochs):
    # Training
    train_loss = network.train_epoch(train_loader, optimizer, device)
    
    # Validation
    val_loss = network.evaluate(val_loader, device)
    
    print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
```

### Mixed Precision Training

```python
# Enable mixed precision (CUDA only)
train_loss = network.train_epoch(
    dataloader=train_loader,
    optimizer=optimizer,
    device=device,
    use_mixed_precision=True,  # Requires CUDA
    gradient_clip_norm=1.0
)
```

### Generation with Different Strategies

```python
# Greedy decoding
greedy = network.generate(input_ids, do_sample=False)

# Temperature sampling
temp = network.generate(input_ids, temperature=0.8, do_sample=True)

# Top-k sampling
topk = network.generate(input_ids, top_k=50, do_sample=True)

# Nucleus (top-p) sampling
nucleus = network.generate(input_ids, top_p=0.9, do_sample=True)
```

## Running Examples

### Test Single Layer
```bash
python3 libs/transformer.py
```

### Test Multi-Layer Network
```bash
python3 libs/network.py
```

### Run Training Example
```bash
python3 training_example.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Apple Silicon Mac (for NPU acceleration) or CUDA GPU

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!

---

**Note**: This implementation is optimized for Apple's Neural Processing Unit (NPU) through Metal Performance Shaders (MPS), providing significant acceleration on Apple Silicon devices.
