# Quick Reference Guide for Transformer Implementation

This guide serves as a quick reference for understanding and navigating the transformer implementation when returning to the code.

## Project Structure
```
transformer/
├── src/
│   ├── config/
│   │   └── config.py          # Configuration parameters
│   ├── training/
│   │   ├── data.py            # Dataset implementation
│   │   ├── tokenizer.py       # Tokenizer utilities
│   │   └── train.py           # Training pipeline
│   └── transformer/
│       ├── components.py      # Core building blocks
│       ├── model.py           # Model architecture classes
│       └── transformer.py     # Model builder
├── docs/                      # Documentation (you are here)
│   ├── components.md          # Component explanations
│   ├── training.md            # Training process
│   ├── model.md               # Model architecture
│   └── quick_reference.md     # This file
└── README.md                  # Project overview
```

## Key Classes and Functions

### Core Components (`src/transformer/components.py`)
- `InputEmbeddings`: Converts tokens to vectors
- `PositionalEncoding`: Adds positional info to embeddings
- `MultiHeadAttentionBlock`: Multi-head attention mechanism
- `FeedForwardBlock`: Position-wise feed-forward network
- `ResidualConnection`: Residual connections with normalization
- `LayerNormalization`: Normalization across feature dimension
- `ProjectionLayer`: Projects to vocabulary space

### Model Architecture (`src/transformer/model.py`)
- `Encoder`: Stack of encoder layers
- `EncoderLayer`: Single encoder layer
- `Decoder`: Stack of decoder layers
- `DecoderLayer`: Single decoder layer
- `Transformer`: Combines encoder/decoder with embeddings/projection

### Model Builder (`src/transformer/transformer.py`)
- `build_transformer`: Constructs complete model

### Training Utilities (`src/training/`)
- `CustomDataset`: Handles data preprocessing
- `get_or_build_tokenizer`: Creates tokenizers
- `get_ds`: Prepares datasets
- `train_model`: Main training function

## Key Formulas & Dimensions

### Attention Formula
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

### Dimensions
- Input/Output Embeddings: `(batch, seq_len, d_model)`
- Attention Weights: `(batch, h, seq_len, seq_len)`
- Final Output: `(batch, seq_len, tgt_vocab_size)`

### Standard Values
- `d_model`: 512
- `d_ff`: 2048
- `h` (attention heads): 8
- `N` (layers): 6

## Key Implementation Details

### Encoder Layer Flow
```
x → Norm → Self-Attention → Dropout → Add → 
  → Norm → FFN → Dropout → Add → Output
```

### Decoder Layer Flow
```
x → Norm → Masked Self-Att → Dropout → Add →
  → Norm → Cross-Att (with encoder) → Dropout → Add → 
  → Norm → FFN → Dropout → Add → Output
```

### Masking
- Padding masks: Hide padding tokens (value=0)
- Look-ahead masks: Hide future positions in target sequence

## Configuration Parameters (`src/config/config.py`)
- `seq_lenght`: 350 (sequence length)
- `batch_size`: 350
- `d_model`: 512
- `lr`: 1e-4 (learning rate)
- `epochs`: 20

## Training Process
1. Load and preprocess data
2. Build model with `build_transformer`
3. Train for N epochs:
   - Forward pass through encoder/decoder
   - Compute loss against targets
   - Backpropagate gradients
   - Update parameters
4. Validate after each epoch
5. Save model checkpoints

## Common Issues & Solutions

### Memory Issues
- Reduce `batch_size` or `seq_lenght` if running out of memory
- Consider gradient accumulation for larger effective batch sizes

### Training Instability
- Lower learning rate if loss is diverging
- Ensure proper initialization of weights

### Poor Performance
- Check if model is properly masked during inference
- Verify data preprocessing and tokenization
- Consider training for more epochs

## Quick Commands
```bash
# Train model
python -m src.training.train

# With uv
uv run python -m src.training.train
```

## Key Concepts to Remember

1. **Pre-norm vs Post-norm**: This implementation uses pre-norm (normalization before sub-layers)
2. **Scaled Dot-Product Attention**: Attention scores are scaled by sqrt(d_k)
3. **Causal Masking**: Decoder only attends to previous positions during training
4. **Residual Connections**: Applied around each sub-layer with layer norm
5. **Shared Embeddings**: Encoder and decoder may share embedding weights (not implemented here)

## Math Behind Key Components

### Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Feed-Forward Network
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```