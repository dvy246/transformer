import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer.components import MultiHeadAttentionBlock,FeedForwardBlock,PositionalEncoding,ResidualConnection,LayerNormalization,InputEmbeddings



class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder that combines self-attention and feed-forward operations with residual connections.

    This layer is a fundamental building block of the Transformer encoder, implementing the core operations:
    - Multi-head self-attention mechanism to capture dependencies between different positions in the input sequence
    - Position-wise feed-forward network for additional non-linear transformations
    - Residual connections around each sub-layer followed by layer normalization

    Typical usage in a Transformer encoder:
        encoder_layer = EncoderLayer(
            features=512,
            self_attention_block=MultiHeadAttentionBlock(...),
            feed_forward_block=FeedForwardBlock(...),
            dropout=0.1
        )
        output = encoder_layer(input_tensor, src_mask)

    Args:
        features (int): The number of expected features in the input (dimensionality of the model).
        self_attention_block (MultiHeadAttentionBlock): The multi-head attention mechanism module.
        feed_forward_block (FeedForwardBlock): The position-wise feed-forward network module.
        dropout (float): The dropout probability for residual connections.
    
    Note:
        - The input tensor should have shape (batch_size, seq_length, features)
        - The src_mask is used to mask out certain positions in the attention mechanism
        - This implementation assumes the residual connections include layer normalization
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

# ---------------------------------------------------------
# 8. ASSEMBLY (ENCODER, DECODER, TRANSFORMER)
# ---------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
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
