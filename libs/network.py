import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformer import TransformerLayer
import math
import time
from typing import Optional, Tuple, List


class Network(nn.Module):
    """
    Multi-layer Transformer Network with training capabilities
    Built by stacking multiple TransformerLayer instances
    """
    
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_seq_length: int = 1000,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super(Network, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = self.create_positional_encoding(max_seq_length, d_model)
        
        # Stack of transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization and output projection
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Training state
        self.training_history = []
        
    def create_positional_encoding(self, max_seq_length: int, d_model: int) -> torch.Tensor:
        """Create positional encoding matrix"""
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask to ignore padding tokens"""
        # x shape: (batch_size, seq_length)
        # Returns mask where True indicates padding positions
        return (x == self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        return mask.bool()
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input token IDs (batch_size, seq_length)
            attention_mask: Optional attention mask
            use_causal_mask: Whether to use causal masking for autoregressive tasks
        
        Returns:
            Output logits (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = x.shape
        device = x.device
        
        # Create attention mask
        if attention_mask is None:
            attention_mask = self.create_padding_mask(x)
        
        if use_causal_mask:
            causal_mask = self.create_causal_mask(seq_length, device)
            attention_mask = attention_mask | causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Token embeddings + positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :seq_length, :].to(device)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        # Final normalization and output projection
        x = self.final_norm(x)
        output = self.output_projection(x)
        
        return output
    
    def compute_loss(
        self, 
        input_ids: torch.Tensor, 
        target_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True
    ) -> torch.Tensor:
        """
        Compute loss for training
        
        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            target_ids: Target token IDs (batch_size, seq_length)
            attention_mask: Optional attention mask
            use_causal_mask: Whether to use causal masking
        
        Returns:
            Cross-entropy loss
        """
        logits = self.forward(input_ids, attention_mask, use_causal_mask)
        
        # Reshape for loss computation
        logits = logits.view(-1, self.vocab_size)
        targets = target_ids.view(-1)
        
        # Compute cross-entropy loss, ignoring padding tokens
        loss = F.cross_entropy(logits, targets, ignore_index=self.pad_token_id)
        
        return loss
    
    def setup_optimizer(
        self, 
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999)
    ) -> torch.optim.Optimizer:
        """
        Setup Adam optimizer with weight decay
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            betas: Betas for Adam optimizer
        
        Returns:
            Configured optimizer
        """
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() 
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() 
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas
        )
        
        return optimizer
    
    def train_step(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        use_mixed_precision: bool = False
    ) -> float:
        """
        Perform a single training step with backpropagation
        
        Args:
            batch: Dictionary containing 'input_ids' and 'target_ids'
            optimizer: Optimizer instance
            scaler: GradScaler for mixed precision training
            use_mixed_precision: Whether to use mixed precision training
        
        Returns:
            Loss value for this step
        """
        self.train()
        optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        attention_mask = batch.get('attention_mask', None)
        
        if use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(input_ids, target_ids, attention_mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = self.compute_loss(input_ids, target_ids, attention_mask)
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def train_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_mixed_precision: bool = False,
        gradient_clip_norm: float = 1.0,
        log_interval: int = 100
    ) -> float:
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader for training data
            optimizer: Optimizer instance
            device: Device to train on
            use_mixed_precision: Whether to use mixed precision
            gradient_clip_norm: Gradient clipping norm
            log_interval: Interval for logging progress
        
        Returns:
            Average loss for the epoch
        """
        self.train()
        total_loss = 0.0
        num_batches = 0
        
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            loss = self.train_step(batch, optimizer, scaler, use_mixed_precision)
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_norm)
            
            total_loss += loss
            num_batches += 1
            
            # Logging
            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                print(f'Batch {batch_idx:4d}/{len(dataloader):4d} | '
                      f'Loss: {loss:.4f} | '
                      f'Time: {elapsed:.2f}s')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(
        self,
        dataloader,
        device: torch.device
    ) -> float:
        """
        Evaluate the model on validation data
        
        Args:
            dataloader: DataLoader for validation data
            device: Device to evaluate on
        
        Returns:
            Average validation loss
        """
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                input_ids = batch['input_ids']
                target_ids = batch['target_ids']
                attention_mask = batch.get('attention_mask', None)
                
                loss = self.compute_loss(input_ids, target_ids, attention_mask)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text using the trained model
        
        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
        
        Returns:
            Generated token IDs
        """
        self.eval()
        
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize generated sequence with input
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                logits = self.forward(generated, use_causal_mask=True)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end-of-sequence token
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated


# Example usage and training demonstration
if __name__ == "__main__":
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
    
    # Network parameters
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    max_seq_length = 256
    
    # Create network
    network = Network(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_length=max_seq_length
    ).to(device)
    
    # Setup optimizer
    optimizer = network.setup_optimizer(learning_rate=1e-4)
    
    print(f"Network created with {sum(p.numel() for p in network.parameters()):,} parameters")
    print(f"Device: {device}")
    
    # Create dummy training data
    batch_size = 4
    seq_length = 64
    
    # Simulate a training batch
    dummy_batch = {
        'input_ids': torch.randint(1, vocab_size, (batch_size, seq_length)).to(device),
        'target_ids': torch.randint(1, vocab_size, (batch_size, seq_length)).to(device)
    }
    
    # Demonstrate training step
    print("\nDemonstrating training step...")
    network.train()
    loss = network.train_step(dummy_batch, optimizer)
    print(f"Training loss: {loss:.4f}")
    
    # Demonstrate generation
    print("\nDemonstrating text generation...")
    network.eval()
    input_sequence = torch.randint(1, 100, (1, 10)).to(device)
    generated = network.generate(input_sequence, max_length=20, do_sample=True)
    print(f"Input length: {input_sequence.size(1)}")
    print(f"Generated length: {generated.size(1)}")
    
    print("\nNetwork setup complete!")
