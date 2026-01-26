# Transformer Encoder Documentation

## Overview
The encoder processes the source sequence through multiple layers to generate contextualized representations. It handles:
- Input tokenization and embedding
- Positional encoding
- Multi-head self-attention
- Position-wise feed-forward networks

**Input**: `(batch_size, seq_len)` token IDs  
**Output**: `(batch_size, seq_len, d_model)` contextualized features

## Step-by-Step Flow

### 1. Input Processing
```python
src_embed = InputEmbeddings(d_model, src_vocab_size)  # [src/transformer/components.py]
src_pos = PositionalEncoding(d_model, src_seq_len, dropout)  # [src/transformer/components.py]
```
- Converts token IDs → dense vectors (`d_model` dimensions)
- Adds positional information using sine/cosine functions

### 2. Encoder Layers (N=6 by default)
Each layer ([EncoderLayer](file:///Users/divyyadav/transformer/src/transformer/model.py#L4-L22)) contains:

#### a) Multi-Head Self-Attention
```python
self.self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # [src/transformer/components.py]
```
- Computes attention scores between all positions
- Masked to prevent future token leakage (not needed in encoder)
- Output shape: `(batch_size, seq_len, d_model)`

#### b) Residual Connection & LayerNorm
```python
self.residual_connections[0](x, lambda x: self_attention(x))
```
- Prevents vanishing gradients
- Stabilizes training

#### c) Position-wise Feed-Forward Network
```python
self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # [src/transformer/components.py]
```
- Two linear layers with ReLU activation
- Projects to `d_ff=2048` then back to `d_model`

### 3. Final Normalization
```python
self.norm = LayerNormalization(features)  # [src/transformer/components.py]
```
- Ensures stable feature distributions

## Time Complexity Analysis
| Component | Complexity | Explanation |
|-----------|------------|-------------|
| Self-Attention | O(seq_len² × d_model) | Pairwise token interactions |
| Feed-Forward | O(seq_len × d_model²) | Per-token transformations |
| **Total (per layer)** | O(seq_len²d + seq_lend²) | Dominated by attention for long sequences |
| **Full Encoder** | O(N × (seq_len²d + seq_lend²)) | N=6 layers by default |

## Key Implementation Notes
- **Masking**: Encoder uses `src_mask` to ignore padding tokens
- **Initialization**: Xavier uniform for stable training
- **Critical Files**:
  - [model.py](file:///Users/divyyadav/transformer/src/transformer/model.py)
  - [components.py](file:///Users/divyyadav/transformer/src/transformer/components.py)

> 💡 **Revision Tip**: Focus on how residual connections prevent gradient vanishing and why attention dominates time complexity for long sequences.