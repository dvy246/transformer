# Transformer Decoder Documentation

## Overview
The decoder generates target sequence tokens autoregressively using encoder context. It handles:
- Target token embedding and positional encoding
- Masked self-attention (prevents cheating)
- Cross-attention with encoder output
- Position-wise feed-forward networks

**Input**: `(batch_size, tgt_seq_len)` token IDs  
**Output**: `(batch_size, tgt_seq_len, d_model)` contextualized features

## Step-by-Step Flow

### 1. Input Processing
```python
tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)  # [src/transformer/components.py]
tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)  # [src/transformer/components.py]
```
- Converts target token IDs → dense vectors
- Adds positional information

### 2. Decoder Layers (N=6 by default)
Each layer ([DecoderLayer](file:///Users/divyyadav/transformer/src/transformer/model.py#L25-L40)) contains:

#### a) Masked Multi-Head Self-Attention
```python
self.self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # [src/transformer/components.py]
```
- Uses `tgt_mask` to prevent attending to future positions
- Critical for autoregressive generation

#### b) Cross-Attention with Encoder Output
```python
self.cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # [src/transformer/components.py]
```
- Queries from decoder, keys/values from encoder output
- Bridges source and target representations

#### c) Residual Connections & LayerNorm
- Applied after each sub-layer (self-attention, cross-attention, feed-forward)

#### d) Position-wise Feed-Forward Network
```python
self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # [src/transformer/components.py]
```
- Same structure as encoder FFN

### 3. Final Normalization
```python
self.norm = LayerNormalization(features)  # [src/transformer/components.py]
```
- Ensures stable feature distributions before projection

## Time Complexity Analysis
| Component | Complexity | Explanation |
|-----------|------------|-------------|
| Masked Self-Attention | O(tgt_seq_len² × d_model) | Causal attention prevents future token access |
| Cross-Attention | O(tgt_seq_len × src_seq_len × d_model) | Query-target interactions with encoder keys/values |
| Feed-Forward | O(tgt_seq_len × d_model²) | Per-token transformations |
| **Total (per layer)** | O(tgt²d + tgt×src×d + tgt×d²) | Cross-attention dominates when src_seq_len > tgt_seq_len |
| **Full Decoder** | O(N × (tgt²d + tgt×src×d + tgt×d²)) | N=6 layers by default |

## Key Implementation Notes
- **Masking**: `tgt_mask` is causal (triangular), `src_mask` handles padding
- **Projection**: Final `ProjectionLayer` converts to vocabulary logits
- **Critical Files**:
  - [model.py](file:///Users/divyyadav/transformer/src/transformer/model.py)
  - [components.py](file:///Users/divyyadav/transformer/src/transformer/components.py)

> 💡 **Revision Tip**: Remember that cross-attention is the 'bridge' between encoder and decoder—without it, the decoder couldn't access source context!