# Transformer Model Architecture Documentation

This document provides a detailed explanation of the Transformer model architecture implementation.

## Table of Contents
- [Overview](#overview)
- [Model Builder](#model-builder)
- [Encoder Architecture](#encoder-architecture)
- [Decoder Architecture](#decoder-architecture)
- [Transformer Class](#transformer-class)
- [Forward Pass Flow](#forward-pass-flow)
- [Attention Mechanisms](#attention-mechanisms)

## Overview

The Transformer model is composed of an encoder stack and a decoder stack, each containing multiple identical layers. Each layer employs residual connections and layer normalization. The model uses attention mechanisms instead of recurrence to capture dependencies between input and output sequences.

The implementation follows the architecture described in the paper "Attention Is All You Need" by Vaswani et al.

## Model Builder

### Function: `build_transformer`

Located in `src/transformer/transformer.py`, this function constructs a complete Transformer model with proper initialization.

```python
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
):
```

#### Parameters:
- `src_vocab_size`: Size of the source language vocabulary
- `tgt_vocab_size`: Size of the target language vocabulary
- `src_seq_len`: Maximum length of source sequences
- `tgt_seq_len`: Maximum length of target sequences
- `d_model`: Dimension of the model (default 512)
- `N`: Number of encoder and decoder layers (default 6)
- `h`: Number of attention heads (default 8)
- `dropout`: Dropout probability (default 0.1)
- `d_ff`: Inner-layer dimension of feed-forward networks (default 2048)

#### Construction Process:
1. Creates source and target embeddings
2. Creates positional encodings for both source and target
3. Builds N encoder layers with self-attention and feed-forward blocks
4. Builds N decoder layers with self-attention, cross-attention, and feed-forward blocks
5. Creates encoder and decoder with their respective layer lists
6. Creates projection layer to map decoder output to target vocabulary
7. Assembles all components into a Transformer instance
8. Initializes model parameters with Xavier uniform initialization

## Encoder Architecture

The encoder maps an input sequence of symbol representations `(x1, ..., xn)` to a sequence of continuous representations `z = (z1, ..., zn)`.

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

The encoder consists of a stack of `N=6` identical layers. Each layer has two sub-layers:
1. Multi-head self-attention mechanism
2. Position-wise fully connected feed-forward network

Each sub-layer has a residual connection followed by layer normalization.

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

Each encoder layer applies:
1. Self-attention where Q, K, and V all come from the previous encoder layer
2. A position-wise feed-forward network

The encoder does not have masking in its self-attention mechanism since it can attend to all positions in the input sequence.

## Decoder Architecture

The decoder generates the output sequence one element at a time, with each element attending to the previously generated elements.

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

The decoder also consists of a stack of `N=6` identical layers. Each layer has three sub-layers:
1. Masked multi-head self-attention mechanism
2. Multi-head attention over encoder output
3. Position-wise fully connected feed-forward network

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

Each decoder layer applies:
1. Masked self-attention allowing positions to attend to previous positions only
2. Encoder-decoder attention where Q comes from the decoder and K, V come from the encoder
3. A position-wise feed-forward network

## Transformer Class

### Class: `Transformer`

```python
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
```

The Transformer class combines all components:
- Encoder and decoder stacks
- Source and target embeddings
- Source and target positional encodings
- Projection layer to convert output to vocabulary space

## Forward Pass Flow

The complete forward pass follows this sequence:

1. **Encoding Phase**:
   - Input source sequence through source embedding
   - Add positional encoding to source embeddings
   - Pass through encoder stack with source mask

2. **Decoding Phase**:
   - Input target sequence through target embedding
   - Add positional encoding to target embeddings
   - Pass through decoder stack with encoder output, source mask, and target mask
   - Project decoder output to vocabulary space

The forward pass can be summarized as:
```python
def forward(self, src, tgt, src_mask, tgt_mask):
    # Encode source sequence
    encoder_output = self.encode(src, src_mask)
    
    # Decode target sequence using encoder output
    decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
    
    # Project to vocabulary space
    return self.project(decoder_output)
```

## Attention Mechanisms

The model uses three distinct types of attention:

1. **Encoder Self-Attention**: Each position in the encoder can attend to all positions in the previous layer of the encoder
2. **Decoder Self-Attention**: Each position in the decoder can attend to all positions up to and including that position in the decoder (masked)
3. **Encoder-Decoder Attention**: Each position in the decoder can attend to all positions in the encoder output

### Masking

Two types of masking are used:
- **Padding Mask**: Prevents attention to padding tokens in both encoder and decoder
- **Look-ahead Mask**: Prevents decoder from attending to subsequent positions during training (causal masking)