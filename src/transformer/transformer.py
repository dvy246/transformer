import torch.nn as nn
from src.transformer.model import Encoder,EncoderLayer,Decoder,DecoderLayer,ProjectionLayer,Transformer
from src.transformer.components import MultiHeadAttentionBlock,FeedForwardBlock,PositionalEncoding,InputEmbeddings

# ---------------------------------------------------------
# 9. MODEL BUILDER
# ---------------------------------------------------------
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












































