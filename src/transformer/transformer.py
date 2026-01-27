import torch.nn as nn
from src.transformer.model import Encoder,EncoderLayer,Decoder,DecoderLayer,ProjectionLayer,Transformer
from src.transformer.components import MultiHeadAttentionBlock,FeedForwardBlock,PositionalEncoding,InputEmbeddings

# ---------------------------------------------------------
# 9. MODEL BUILDER
# ---------------------------------------------------------
"""
Transformer builder module.

This module provides utilities for constructing complete Transformer models according to the
architecture specified in the "Attention is All You Need" paper. It implements a factory
function that assembles all components into a ready-to-use Transformer model.

The builder creates a complete Transformer with:
1. Configurable vocabulary sizes for source and target languages
2. Customizable sequence lengths for encoder and decoder
3. Standard Transformer hyperparameters (d_model, N, h, dropout, d_ff)
4. Proper initialization of all model parameters
5. Complete encoder-decoder architecture with attention mechanisms

Key features:
- Flexible model configuration through parameters
- Xavier uniform initialization for stable training
- Complete encoder-decoder architecture
- Parameterizable model dimensions and depth
- Integration of all transformer components

For detailed implementation notes and design decisions, refer to the project documentation
in the /docs directory, particularly the 'Model Construction' and 'Hyperparameter Configuration' sections.
"""

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # 1. Create Embeddings
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # 2. Create Positional Encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # 3. Create Encoder Blocks
    encoder_layers = []
    for _ in range(N):
        encoder_self_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_layers.append(EncoderLayer(d_model, encoder_self_attn, feed_forward, dropout))
    
    # 4. Create Decoder Blocks
    decoder_layers = []
    for _ in range(N):
        decoder_self_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_layers.append(DecoderLayer(d_model, decoder_self_attn, decoder_cross_attn, feed_forward, dropout))

    # 5. Build Encoder/Decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_layers))
    decoder = Decoder(d_model, nn.ModuleList(decoder_layers))

    # 6. Projection
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # 7. Create Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize Parameters (Xavier)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer












































