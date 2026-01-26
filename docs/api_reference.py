

"""
API Reference for Transformer Implementation

This file contains documented classes and functions from the transformer implementation
with detailed docstrings for easy reference when revisiting the code.
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional


class InputEmbeddings(nn.Module):
    """
    Converts input token IDs into dense vector representations.
    
    The embeddings are scaled by the square root of the model dimension
    as described in the original Transformer paper to improve training
    stability.
    
    Attributes:
        d_model (int): Dimension of the embedding vectors
        vocab_size (int): Size of the vocabulary
        embedding (nn.Embedding): PyTorch embedding layer
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the input embeddings layer.
        
        Args:
            d_model (int): Dimension of the embedding vectors
            vocab_size (int): Size of the vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the input embeddings.
        
        Args:
            x (Tensor): Input tensor of token IDs, shape (batch, seq_len)
            
        Returns:
            Tensor: Embedded tensor, shape (batch, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Adds positional information to input embeddings.
    
    Uses sinusoidal functions with different frequencies to encode
    position information. This enables the model to distinguish
    between different positions in the sequence.
    
    Attributes:
        d_model (int): Dimensionality of the model
        seq_len (int): Maximum sequence length
        dropout (nn.Dropout): Dropout layer
        pe (Tensor): Precomputed positional encodings
    """
    
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
        Initialize the positional encoding layer.
        
        Args:
            d_model (int): Dimensionality of the model
            seq_len (int): Maximum sequence length
            dropout (float): Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute angles for each position
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, seq_len, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the positional encoding.
        
        Args:
            x (Tensor): Input tensor, shape (batch, seq_len, d_model)
            
        Returns:
            Tensor: Tensor with positional encodings added, shape (batch, seq_len, d_model)
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):
    """
    Implements multi-head attention mechanism.
    
    Allows the model to jointly attend to information from different
    representation subspaces at different positions. The attention
    mechanism computes how much focus to place on other parts of the
    sequence when processing each position.
    
    Attributes:
        d_k (int): Dimension of each attention head
        w_q (nn.Linear): Linear transformation for queries
        w_k (nn.Linear): Linear transformation for keys
        w_v (nn.Linear): Linear transformation for values
        w_o (nn.Linear): Final linear transformation
        dropout (nn.Dropout): Dropout layer
        attention_weights (Tensor): Stored attention weights for inspection
    """
    
    def __init__(self, d_model: int, h: int, dropout: float):
        """
        Initialize the multi-head attention block.
        
        Args:
            d_model (int): Dimensionality of the model
            h (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        # Verify that d_model is divisible by h
        assert d_model % h == 0
        
        # Depth of each attention head
        self.d_k = d_model // h
        
        # Linear transformations for queries, keys, values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Final linear transformation
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query: Tensor, key: Tensor, value: Tensor, 
                  mask: Optional[Tensor], dropout: nn.Dropout):
        """
        Static method to compute scaled dot-product attention.
        
        Args:
            query (Tensor): Query tensor
            key (Tensor): Key tensor
            value (Tensor): Value tensor
            mask (Optional[Tensor]): Mask to apply to attention scores
            dropout (nn.Dropout): Dropout layer
            
        Returns:
            Tuple[Tensor, Tensor]: Output tensor and attention weights
        """
        d_k = query.shape[-1]
        
        # Compute attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)  # Replace masked positions with large negative value
        
        # Apply softmax to get attention weights
        attention_weights = attention_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        
        # Apply attention weights to values
        return (attention_weights @ value), attention_weights

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Forward pass of the multi-head attention.
        
        Args:
            q (Tensor): Query tensor
            k (Tensor): Key tensor
            v (Tensor): Value tensor
            mask (Optional[Tensor]): Mask to apply to attention scores
            
        Returns:
            Tensor: Output tensor after multi-head attention
        """
        query = self.w_q(q)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)    # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        
        # Split into multiple heads
        # Reshape to (batch, seq_len, h, d_k) then transpose to (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # Apply attention mechanism
        x, self.attention_weights = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        
        # Concatenate heads back together
        # Transpose back to (batch, seq_len, h, d_k) and reshape to (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], -1, self.h * self.d_k
        )
        
        # Apply final linear transformation
        return self.w_o(x)


class FeedForwardBlock(nn.Module):
    """
    Implements position-wise feed-forward networks.
    
    A simple two-layer feed-forward network applied to each position
    separately and identically. Consists of two linear transformations
    with a ReLU activation in between.
    
    Attributes:
        linear_1 (nn.Linear): First linear transformation
        relu (nn.ReLU): ReLU activation function
        linear_2 (nn.Linear): Second linear transformation
        dropout (nn.Dropout): Dropout layer
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Initialize the feed-forward block.
        
        Args:
            d_model (int): Input/output dimensionality
            d_ff (int): Hidden layer dimensionality
            dropout (float): Dropout probability
        """
        super().__init__()
        # First linear transformation: expand to d_ff dimensions
        self.linear_1 = nn.Linear(d_model, d_ff)
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Second linear transformation: compress back to d_model dimensions
        self.linear_2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the feed-forward network.
        
        Args:
            x (Tensor): Input tensor, shape (batch, seq_len, d_model)
            
        Returns:
            Tensor: Output tensor, shape (batch, seq_len, d_model)
        """
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))


class LayerNormalization(nn.Module):
    """
    Implements layer normalization.
    
    Normalizes the activations across the feature dimension for each
    sample independently. Helps with training stability and speed.
    
    Attributes:
        eps (float): Small value to prevent division by zero
        alpha (nn.Parameter): Learnable scaling parameter
        bias (nn.Parameter): Learnable bias parameter
    """
    
    def __init__(self, eps: float = 10**-6):
        """
        Initialize the layer normalization.
        
        Args:
            eps (float): Small value to prevent division by zero
        """
        super().__init__()
        self.eps = eps  # Small value to prevent division by zero
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable scaling parameter
        self.bias = nn.Parameter(torch.zeros(1))   # Learnable bias parameter
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of layer normalization.
        
        Args:
            x (Tensor): Input tensor, shape (batch, seq_len, d_model)
            
        Returns:
            Tensor: Normalized tensor, shape (batch, seq_len, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)  # Mean across the last dimension
        std = x.std(dim=-1, keepdim=True)    # Standard deviation across the last dimension
        
        # Normalize and apply learnable parameters
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):
    """
    Implements residual connections followed by layer normalization.
    
    This is the "pre-norm" variant where normalization is applied
    before the sublayer, which tends to improve training stability.
    
    Attributes:
        dropout (nn.Dropout): Dropout layer
        norm (LayerNormalization): Layer normalization
    """
    
    def __init__(self, d_model: int, dropout: float):
        """
        Initialize the residual connection.
        
        Args:
            d_model (int): Dimensionality of the model
            dropout (float): Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x: Tensor, sublayer) -> Tensor:
        """
        Forward pass of the residual connection.
        
        Args:
            x (Tensor): Input tensor
            sublayer: Sublayer function to apply
            
        Returns:
            Tensor: Output tensor after residual connection
        """
        # Apply normalization, then sublayer, then dropout, then add residual connection
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    Represents a single encoder block in the Transformer architecture.
    
    Each encoder block consists of a multi-head self-attention mechanism
    and a position-wise feed-forward network, each followed by residual
    connections and layer normalization.
    
    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Self-attention mechanism
        feed_forward_block (FeedForwardBlock): Feed-forward network
        residual_connections (nn.ModuleList): Residual connections for each sublayer
    """
    
    def __init__(
        self, 
        d_model: int, 
        self_attention_block: MultiHeadAttentionBlock, 
        feed_forward_block: FeedForwardBlock, 
        dropout: float
    ):
        """
        Initialize the encoder block.
        
        Args:
            d_model (int): Dimensionality of the model
            self_attention_block (MultiHeadAttentionBlock): Self-attention mechanism
            feed_forward_block (FeedForwardBlock): Feed-forward network
            dropout (float): Dropout probability
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])
    
    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Forward pass of the encoder block.
        
        Args:
            x (Tensor): Input tensor, shape (batch, seq_len, d_model)
            mask (Optional[Tensor]): Mask to apply to attention scores
            
        Returns:
            Tensor: Output tensor, shape (batch, seq_len, d_model)
        """
        # First residual connection with self-attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        
        # Second residual connection with feed-forward network
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x


class Encoder(nn.Module):
    """
    Complete encoder stack in the Transformer architecture.
    
    Comprises multiple identical encoder blocks stacked sequentially,
    followed by a final layer normalization.
    
    Attributes:
        layers (nn.ModuleList): List of encoder blocks
        norm (LayerNormalization): Final layer normalization
    """
    
    def __init__(self, layers: nn.ModuleList):
        """
        Initialize the encoder stack.
        
        Args:
            layers (nn.ModuleList): List of encoder blocks
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Forward pass of the encoder stack.
        
        Args:
            x (Tensor): Input tensor, shape (batch, seq_len, d_model)
            mask (Optional[Tensor]): Mask to apply to attention scores
            
        Returns:
            Tensor: Output tensor, shape (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def build_transformer(
    src_vocab_size: int, 
    tgt_vocab_size: int, 
    src_seq_len: int, 
    tgt_seq_len: int, 
    d_model: int = 512, 
    N: int = 6, 
    h: int = 8, 
    dropout: float = 0.1, 
    d_ff: int = 2048
) -> "Transformer":
    """
    Factory function to build a complete Transformer model.
    
    Args:
        src_vocab_size (int): Size of source vocabulary
        tgt_vocab_size (int): Size of target vocabulary
        src_seq_len (int): Maximum length of source sequences
        tgt_seq_len (int): Maximum length of target sequences
        d_model (int): Dimension of the model (default 512)
        N (int): Number of encoder and decoder layers (default 6)
        h (int): Number of attention heads (default 8)
        dropout (float): Dropout probability (default 0.1)
        d_ff (int): Inner-layer dimension of feed-forward nets (default 2048)
        
    Returns:
        Transformer: Complete Transformer model
    """
    # Implementation details would go here
    pass