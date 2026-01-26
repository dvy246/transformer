# Transformer Components Documentation

This document provides detailed explanations of each component in the Transformer architecture implementation.

## Table of Contents
- [Input Embeddings](#input-embeddings)
- [Positional Encoding](#positional-encoding)
- [Multi-Head Attention](#multi-head-attention)
- [Feed-Forward Networks](#feed-forward-networks)
- [Skip Connections](#skip-connections)
- [Layer Normalization](#layer-normalization)
- [Projection Layer](#projection-layer)
- [Encoder Components](#encoder-components)
- [Decoder Components](#decoder-components)

## Input Embeddings

### Class: `InputEmbeddings`

```python
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

**Purpose**: Converts input token IDs into dense vector representations of dimension `d_model`.

**Key Features**:
- Uses PyTorch's `nn.Embedding` layer
- Scales embeddings by `sqrt(d_model)` as described in the original paper
- Maps discrete tokens to continuous vector space

**Parameters**:
- `d_model`: Dimensionality of the embedding vectors
- `vocab_size`: Size of the vocabulary

## Positional Encoding

### Class: `PositionalEncoding`

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
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
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
```

**Purpose**: Adds positional information to the input embeddings since the model has no inherent notion of sequence order.

**Key Features**:
- Uses sinusoidal functions with different frequencies
- Even positions use sine functions, odd positions use cosine functions
- Learned positional embeddings are possible but this approach generalizes to longer sequences
- Dropout applied after adding positional encoding

**Parameters**:
- `d_model`: Dimensionality of the model
- `seq_len`: Maximum sequence length
- `dropout`: Dropout probability

## Multi-Head Attention

### Class: `MultiHeadAttentionBlock`

```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
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
    def attention(query, key, value, mask, dropout: nn.Dropout):
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

    def forward(self, q, k, v, mask):
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
```

**Purpose**: Allows the model to attend to different positions of the input sequence simultaneously by using multiple attention heads.

**Key Features**:
- Computes scaled dot-product attention across multiple heads
- Applies linear transformations to create queries, keys, and values
- Concatenates outputs from all heads and applies final linear transformation
- Includes masking capability for decoder attention

**Types of Attention**:
1. **Self-Attention in Encoder**: Q, K, V all come from the same source (encoder input)
2. **Self-Attention in Decoder**: Q, K, V all come from the same source (decoder input), with causal masking
3. **Encoder-Decoder Attention**: Q comes from decoder, K and V come from encoder

**Parameters**:
- `d_model`: Dimensionality of the model
- `h`: Number of attention heads
- `dropout`: Dropout probability

## Feed-Forward Networks

### Class: `FeedForwardBlock`

```python
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # First linear transformation: expand to d_ff dimensions
        self.linear_1 = nn.Linear(d_model, d_ff)
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Second linear transformation: compress back to d_model dimensions
        self.linear_2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))
```

**Purpose**: Applies position-wise fully connected feed-forward network to each position separately and identically.

**Key Features**:
- Two linear transformations with ReLU activation in between
- Applied independently to each position in the sequence
- Expands to higher dimension `d_ff` then contracts back to `d_model`

**Parameters**:
- `d_model`: Input/output dimensionality
- `d_ff`: Hidden layer dimensionality (typically 2048 in standard implementation)
- `dropout`: Dropout probability

## Skip Connections

### Class: `ResidualConnection`

```python
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, sublayer):
        # Apply normalization, then sublayer, then dropout, then add residual connection
        return x + self.dropout(sublayer(self.norm(x)))
```

**Purpose**: Implements residual connections followed by layer normalization, as described in the original paper.

**Key Features**:
- Applies layer normalization before the sublayer (pre-norm variant)
- Adds the input to the output of the sublayer (residual connection)
- Helps with gradient flow and training stability

**Parameters**:
- `d_model`: Dimensionality of the model
- `dropout`: Dropout probability

## Layer Normalization

### Class: `LayerNormalization`

```python
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps  # Small value to prevent division by zero
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable scaling parameter
        self.bias = nn.Parameter(torch.zeros(1))   # Learnable bias parameter
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Mean across the last dimension
        std = x.std(dim=-1, keepdim=True)    # Standard deviation across the last dimension
        
        # Normalize and apply learnable parameters
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
```

**Purpose**: Normalizes the activations across the feature dimension.

**Key Features**:
- Normalizes each sample independently across the feature dimension
- Uses learnable parameters alpha and bias for flexibility
- Prevents extreme values with epsilon parameter

**Parameters**:
- `eps`: Small value to prevent division by zero (default 1e-6)

## Projection Layer

### Class: `ProjectionLayer`

```python
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
```

**Purpose**: Projects the decoder output from `d_model` dimensions to `vocab_size` dimensions.

**Key Features**:
- Final linear transformation before computing output probabilities
- Converts model outputs to logits over the vocabulary
- Used with softmax to compute output probability distribution

**Parameters**:
- `d_model`: Input dimensionality
- `vocab_size`: Output dimensionality (size of target vocabulary)

## Encoder Components

### Class: `Encoder`

```python
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

**Purpose**: Stacks multiple encoder layers and applies final normalization.

### Class: `EncoderLayer`

```python
class EncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        self_attention_block: MultiHeadAttentionBlock, 
        feed_forward_block: FeedForwardBlock, 
        dropout: float
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])
    
    def forward(self, x, mask):
        # First residual connection with self-attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        
        # Second residual connection with feed-forward network
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x
```

**Purpose**: Implements a single encoder layer with self-attention and feed-forward network.

## Decoder Components

### Class: `Decoder`

```python
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
```

**Purpose**: Stacks multiple decoder layers and applies final normalization.

### Class: `DecoderLayer`

```python
class DecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        self_attention_block: MultiHeadAttentionBlock, 
        cross_attention_block: MultiHeadAttentionBlock, 
        feed_forward_block: FeedForwardBlock, 
        dropout: float
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(3)
        ])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # First residual connection with masked self-attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        
        # Second residual connection with encoder-decoder attention
        x = self.residual_connections[1](
            x, 
            lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        
        # Third residual connection with feed-forward network
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x
```

**Purpose**: Implements a single decoder layer with masked self-attention, encoder-decoder attention, and feed-forward network.

**Mask Types**:
- `tgt_mask`: Masks future positions in the target sequence (causal masking)
- `src_mask`: Masks padding positions in the source sequence