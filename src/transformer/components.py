import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Core components module for the Transformer model.

This module implements the fundamental building blocks of the Transformer architecture as described in the "Attention is All You Need" paper.
Each component serves a specific purpose in the encoder-decoder architecture and can be used independently or combined to form complete models.

Key components include:
- Input embeddings with scaling factor
- Positional encoding using sinusoidal functions
- Multi-head attention mechanism
- Feed-forward networks
- Layer normalization
- Residual connections with layer normalization
- Position-wise feed-forward blocks

Architecture overview:
1. Input embeddings convert tokens to dense vectors
2. Positional encoding adds sequential information
3. Multi-head attention enables parallel attention computation
4. Feed-forward networks process each position independently
5. Residual connections and layer normalization stabilize training
6. Stacked layers form the encoder and decoder blocks

For detailed implementation notes and design decisions, refer to the project documentation
in the /docs directory, particularly the 'Transformer Components' and 'Attention Mechanism' sections.
"""

# ---------------------------------------------------------
# 1. INPUT EMBEDDINGS
# ---------------------------------------------------------
class InputEmbeddings(nn.Module):
    """
    Converts discrete tokens into dense vectors.
    According to the paper, we multiply weights by sqrt(d_model).
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> output: (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

# ---------------------------------------------------------
# 2. POSITIONAL ENCODING
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Adds information about the position of tokens using Sine and Cosine functions.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Calculate the division term (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices (0, 2, 4...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices (1, 3, 5...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

# ---------------------------------------------------------
# 3. LAYER NORMALIZATION
# ---------------------------------------------------------
class LayerNormalization(nn.Module):
    """
    Normalizes the features across the d_model dimension.
    """
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features)) # Multiplicative
        self.beta = nn.Parameter(torch.zeros(features)) # Additive
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# ---------------------------------------------------------
# 4. FEED FORWARD NETWORK
# ---------------------------------------------------------
class FeedForwardBlock(nn.Module):
    """
    A simple two-layer linear transformation with ReLU.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # Expansion
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # Contraction

    def forward(self, x):
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# ---------------------------------------------------------
# 5. MULTI-HEAD ATTENTION
# ---------------------------------------------------------
class MultiHeadAttentionBlock(nn.Module):
    """
    Splits the d_model into 'h' heads to allow the model to 
    attend to information from different representation subspaces.
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch, h, seq_len, d_k) @ (Batch, h, d_k, seq_len) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Masking out values (e.g., padding or future tokens)
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # (Batch, h, seq_len, seq_len) @ (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        
        query = self.w_q(q) # (Batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split into heads: (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_weights = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concatenate heads: (Batch, h, seq_len, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

# ---------------------------------------------------------
# 6. RESIDUAL CONNECTION & SUB-LAYER
# ---------------------------------------------------------
class ResidualConnection(nn.Module):
    """
    Implements LayerNorm(x + Sublayer(x)) or x + Sublayer(LayerNorm(x)).
    The paper uses the former, but many implementations prefer the latter (Pre-Norm).
    We will use the Paper version: Norm(x + Dropout(Sublayer(x)))
    """
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # We pass a lambda function as sublayer to handle MHA or FeedForward
        return x + self.dropout(sublayer(self.norm(x)))
