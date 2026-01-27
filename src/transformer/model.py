import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer.components import MultiHeadAttentionBlock,FeedForwardBlock,PositionalEncoding,ResidualConnection,LayerNormalization,InputEmbeddings
"""
Transformer model assembly module.

This module defines the complete Transformer architecture by assembling the individual components
into encoder and decoder blocks, and ultimately a full Transformer model. It implements the
encoder-decoder structure with multi-head self-attention and feed-forward networks.

The model follows the architecture described in "Attention is All You Need":
1. Encoder: Stack of N identical layers with self-attention and feed-forward sub-layers
2. Decoder: Stack of N identical layers with self-attention, encoder-decoder attention, and feed-forward sub-layers
3. Position-wise feed-forward networks in each layer
4. Residual connections around each sub-layer with layer normalization
5. Input embeddings with positional encodings
6. Final linear and softmax layer for output prediction

Key features:
- Modular design allowing flexible configuration of layer count and dimensions
- Encoder-Decoder architecture with cross-attention
- Input embedding with positional encoding
- Projection layer for output vocabulary mapping
- Masking support for causal decoding and padding handling

For detailed implementation notes and design decisions, refer to the project documentation
in the /docs directory, particularly the 'Transformer Architecture' and 'Model Assembly' sections.
"""



class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.
    
    This class implements a single layer of the Transformer encoder, combining multi-head self-attention
    with a position-wise feed-forward network, both wrapped with residual connections and layer normalization.
    
    Architecture:
        Input -> Self-Attention -> Feed-Forward -> Output
               ↖________________ ↖______________
                    Residual Connections
    
    Design Choices:
        - Uses modular components for attention and feed-forward networks
        - Implements residual connections with layer normalization
        - Flexible to different attention mechanisms and feed-forward configurations
    
    Args:
        features (int): Dimensionality of the model (input and output feature size)
        self_attention_block (MultiHeadAttentionBlock): Multi-head self-attention module
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward network module
        dropout (float): Dropout probability for residual connections
    
    Shape Conventions:
        Input: (batch_size, seq_length, features)
        Source Mask: (batch_size, 1, seq_length) or broadcastable shape
        Output: (batch_size, seq_length, features)
    
    Example:
        >>> attention = MultiHeadAttentionBlock(512, 8)
        >>> ff = FeedForwardBlock(512, 2048, 0.1)
        >>> encoder_layer = EncoderLayer(512, attention, ff, 0.1)
        >>> x = torch.randn(32, 10, 512)  # batch of 32 sequences of 10 tokens
        >>> mask = torch.ones(32, 1, 10)  # mask for all tokens
        >>> output = encoder_layer(x, mask)
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
    """
    Transformer Decoder Layer.
    
    This class implements a single layer of the Transformer decoder, which consists of three main components:
    1. Masked multi-head self-attention for target sequence processing
    2. Multi-head encoder-decoder attention to attend to encoder outputs
    3. Position-wise feed-forward network for additional transformations
    
    All sub-layers are wrapped with residual connections and layer normalization.
    
    Architecture:
        Input -> Masked Self-Attention -> Encoder-Decoder Attention -> Feed-Forward -> Output
               ↖_______________________ ↖__________________________ ↖______________
                    Residual Connections
    
    Design Choices:
        - Uses modular components for different attention mechanisms
        - Maintains consistent feature dimensions throughout the layer
        - Flexible to different attention heads and feed-forward configurations
    
    Args:
        features (int): Dimensionality of the model (input and output feature size)
        self_attention_block (MultiHeadAttentionBlock): Masked multi-head self-attention module
        cross_attention_block (MultiHeadAttentionBlock): Multi-head encoder-decoder attention module
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward network module
        dropout (float): Dropout probability for residual connections
    
    Shape Conventions:
        Input: (batch_size, tgt_seq_length, features)
        Encoder Output: (batch_size, src_seq_length, features)
        Source Mask: (batch_size, 1, src_seq_length) or broadcastable shape
        Target Mask: (batch_size, tgt_seq_length, tgt_seq_length) or broadcastable shape
        Output: (batch_size, tgt_seq_length, features)
    
    Example:
        >>> self_attn = MultiHeadAttentionBlock(512, 8)
        >>> cross_attn = MultiHeadAttentionBlock(512, 8)
        >>> ff = FeedForwardBlock(512, 2048, 0.1)
        >>> decoder_layer = DecoderLayer(512, self_attn, cross_attn, ff, 0.1)
        >>> x = torch.randn(32, 20, 512)  # batch of 32 sequences of 20 target tokens
        >>> encoder_out = torch.randn(32, 10, 512)  # corresponding encoder outputs
        >>> src_mask = torch.ones(32, 1, 10)  # source sequence mask
        >>> tgt_mask = torch.tril(torch.ones(20, 20))  # target sequence mask (causal)
        >>> output = decoder_layer(x, encoder_out, src_mask, tgt_mask)
    """
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
    """
    Transformer Encoder.
    
    This class implements the full encoder section of the Transformer model, consisting of:
    1. A stack of N identical encoder layers
    2. Final layer normalization after all encoder layers
    
    The encoder processes the input sequence and generates context-aware representations
    using self-attention mechanisms to capture dependencies between different positions.
    
    Architecture:
        Input -> [EncoderLayer × N] -> LayerNorm -> Output
    
    Design Choices:
        - Modular design allowing flexible configuration of layer count and dimensions
        - Consistent feature dimension throughout the encoder
        - Input masking support for variable-length sequences
    
    Args:
        features (int): Dimensionality of the model (input and output feature size)
        layers (nn.ModuleList): List of EncoderLayer instances to use in the encoder
    
    Shape Conventions:
        Input: (batch_size, seq_length, features)
        Mask: (batch_size, 1, seq_length) or broadcastable shape
        Output: (batch_size, seq_length, features)
    
    Example:
        >>> # Create a 6-layer encoder
        >>> layers = nn.ModuleList([EncoderLayer(...) for _ in range(6)])
        >>> encoder = Encoder(512, layers)
        >>> x = torch.randn(32, 10, 512)  # batch of 32 sequences of 10 tokens
        >>> mask = torch.ones(32, 1, 10)  # mask for all tokens
        >>> output = encoder(x, mask)
    """
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    """
    Transformer Decoder.
    
    This class implements the full decoder section of the Transformer model, consisting of:
    1. A stack of N identical decoder layers
    2. Final layer normalization after all decoder layers
    
    The decoder generates target sequences by attending to both its own previous outputs
    (masked self-attention) and the encoder's outputs (encoder-decoder attention).
    
    Architecture:
        Input -> [DecoderLayer × N] -> LayerNorm -> Output
    
    Design Choices:
        - Modular design allowing flexible configuration of layer count and dimensions
        - Maintains consistent feature dimensions throughout the decoder
        - Supports both source and target sequence masking
    
    Args:
        features (int): Dimensionality of the model (input and output feature size)
        layers (nn.ModuleList): List of DecoderLayer instances to use in the decoder
    
    Shape Conventions:
        Input: (batch_size, tgt_seq_length, features)
        Encoder Output: (batch_size, src_seq_length, features)
        Source Mask: (batch_size, 1, src_seq_length) or broadcastable shape
        Target Mask: (batch_size, tgt_seq_length, tgt_seq_length) or broadcastable shape
        Output: (batch_size, tgt_seq_length, features)
    
    Example:
        >>> # Create a 6-layer decoder
        >>> layers = nn.ModuleList([DecoderLayer(...) for _ in range(6)])
        >>> decoder = Decoder(512, layers)
        >>> x = torch.randn(32, 20, 512)  # batch of 32 sequences of 20 target tokens
        >>> encoder_out = torch.randn(32, 10, 512)  # corresponding encoder outputs
        >>> src_mask = torch.ones(32, 1, 10)  # source sequence mask
        >>> tgt_mask = torch.tril(torch.ones(20, 20))  # target sequence mask (causal)
        >>> output = decoder(x, encoder_out, src_mask, tgt_mask)
    """
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Projection Layer for Transformer Output.
    
    This layer maps the final feature representation of the Transformer to the output vocabulary space.
    It consists of a linear transformation followed by log-softmax activation to produce
    log-probabilities over the vocabulary for each token position.
    
    Key Features:
        - Linear projection from model dimension to vocabulary size
        - Log-softmax activation for numerical stability
        - Position-wise application (independent for each position in sequence)
    
    Args:
        d_model (int): Dimensionality of the model's feature representation
        vocab_size (int): Size of the output vocabulary
    
    Shape Conventions:
        Input: (batch_size, seq_length, d_model)
        Output: (batch_size, seq_length, vocab_size)
    
    Example:
        >>> proj = ProjectionLayer(512, 10000)  # 512-d model, 10k vocab
        >>> x = torch.randn(32, 20, 512)  # batch of 32 sequences of 20 tokens
        >>> log_probs = proj(x)  # log probabilities over 10k vocabulary items
        >>> assert log_probs.shape == (32, 20, 10000)
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    """
    Full Transformer Model.
    
    This class assembles all components into a complete Transformer model for sequence-to-sequence tasks.
    It implements the standard encoder-decoder architecture with the following components:
    1. Source and target input embeddings
    2. Positional encoding for sequence order information
    3. Encoder with multiple layers
    4. Decoder with multiple layers
    5. Final projection layer to vocabulary space
    
    Key Features:
        - Modular design with interchangeable components
        - Support for sequence-to-sequence tasks (e.g., machine translation)
        - Input masking for variable-length sequences
        - Separate processing for encoder and decoder phases
    
    Args:
        encoder (Encoder): Encoder module
        decoder (Decoder): Decoder module
        src_embed (InputEmbeddings): Input embeddings for source language
        tgt_embed (InputEmbeddings): Input embeddings for target language
        src_pos (PositionalEncoding): Positional encoding for source sequence
        tgt_pos (PositionalEncoding): Positional encoding for target sequence
        projection_layer (ProjectionLayer): Final projection to vocabulary space
    
    Shape Conventions:
        Source Input: (batch_size, src_seq_length)
        Target Input: (batch_size, tgt_seq_length)
        Source Mask: (batch_size, 1, src_seq_length) or broadcastable shape
        Target Mask: (batch_size, tgt_seq_length, tgt_seq_length) or broadcastable shape
        Output: (batch_size, tgt_seq_length, vocab_size)
    
    Example:
        >>> # Create a complete Transformer model
        >>> transformer = Transformer(
        >>>     encoder=Encoder(...),
        >>>     decoder=Decoder(...),
        >>>     src_embed=InputEmbeddings(...),
        >>>     tgt_embed=InputEmbeddings(...),
        >>>     src_pos=PositionalEncoding(...),
        >>>     tgt_pos=PositionalEncoding(...),
        >>>     projection_layer=ProjectionLayer(...)
        >>> )
        >>> 
        >>> # Encode source sequence
        >>> src = torch.randint(0, 10000, (32, 10))  # batch of 32 sequences of 10 tokens
        >>> src_mask = (src != 0).unsqueeze(1)  # create mask for padding tokens
        >>> encoder_output = transformer.encode(src, src_mask)
        >>> 
        >>> # Decode target sequence
        >>> tgt = torch.randint(0, 10000, (32, 20))  # batch of 32 sequences of 20 tokens
        >>> tgt_mask = torch.tril(torch.ones(20, 20))  # causal mask for decoding
        >>> decoder_output = transformer.decode(encoder_output, src_mask, tgt, tgt_mask)
        >>> 
        >>> # Get final output probabilities
        >>> output = transformer.project(decoder_output)
        >>> assert output.shape == (32, 20, 10000)  # (batch, seq_len, vocab_size)
    """
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
