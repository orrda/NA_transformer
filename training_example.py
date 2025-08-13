"""
Example training script for the Network class
Demonstrates how to train the multi-layer transformer network
"""

import torch
import torch.utils.data as data
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'libs'))
from network import Network


class SimpleDataset(data.Dataset):
    """Simple dataset for demonstration purposes"""
    
    def __init__(self, vocab_size, seq_length, num_samples):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequences (in practice, this would be real text data)
        input_ids = torch.randint(1, self.vocab_size, (self.seq_length,))
        
        # For language modeling, target is input shifted by one position
        target_ids = torch.cat([input_ids[1:], torch.randint(1, self.vocab_size, (1,))])
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }


def train_example():
    """Example training loop"""
    
    # Check device availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple NPU (MPS) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Model parameters
    vocab_size = 5000
    d_model = 256
    num_heads = 8
    d_ff = 1024
    num_layers = 4
    max_seq_length = 128
    
    # Training parameters
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 3
    seq_length = 64
    
    print(f"Training Parameters:")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Model dimension: {d_model}")
    print(f"- Number of layers: {num_layers}")
    print(f"- Number of heads: {num_heads}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Sequence length: {seq_length}")
    
    # Create network
    network = Network(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_length=max_seq_length,
        dropout=0.1
    ).to(device)
    
    print(f"\nNetwork created with {sum(p.numel() for p in network.parameters()):,} parameters")
    
    # Setup optimizer
    optimizer = network.setup_optimizer(learning_rate=learning_rate)
    
    # Create datasets
    train_dataset = SimpleDataset(vocab_size, seq_length, 1000)  # 1000 training samples
    val_dataset = SimpleDataset(vocab_size, seq_length, 200)     # 200 validation samples
    
    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0  # Set to 0 for MPS compatibility
    )
    
    val_loader = data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        # Training
        train_loss = network.train_epoch(
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            use_mixed_precision=False,  # Set to True if using CUDA
            gradient_clip_norm=1.0,
            log_interval=20
        )
        
        # Validation
        val_loss = network.evaluate(val_loader, device)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Training Loss:   {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        
        # Store training history
        network.training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    # Demonstrate text generation after training
    print("\nDemonstrating text generation after training...")
    network.eval()
    
    # Generate with different sampling strategies
    input_sequence = torch.randint(1, 100, (1, 10)).to(device)
    
    print(f"Input sequence shape: {input_sequence.shape}")
    
    # Greedy generation
    generated_greedy = network.generate(
        input_sequence, 
        max_length=30, 
        do_sample=False
    )
    print(f"Greedy generation length: {generated_greedy.size(1)}")
    
    # Sampling with temperature
    generated_sample = network.generate(
        input_sequence, 
        max_length=30, 
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )
    print(f"Sampled generation length: {generated_sample.size(1)}")
    
    # Print training history
    print("\nTraining History:")
    for record in network.training_history:
        print(f"Epoch {record['epoch']}: "
              f"Train Loss = {record['train_loss']:.4f}, "
              f"Val Loss = {record['val_loss']:.4f}")
    
    return network


if __name__ == "__main__":
    trained_network = train_example()
    print("\nTraining example completed successfully!")
