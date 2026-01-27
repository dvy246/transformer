from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, Any, Tuple


class CustomDataset(Dataset):
    """
    Data module for the Transformer model.
    
    This module implements a custom PyTorch Dataset class for sequence-to-sequence tasks.
    It handles tokenization, padding, and adding special tokens (SOS, EOS) to the input sequences.
    The module also provides utilities for creating attention masks needed by the Transformer architecture.
    
    Key features:
    - Custom dataset implementation for sequence-to-sequence tasks
    - Proper handling of source and target tokenization
    - Creation of attention masks (encoder and causal decoder masks)
    - Fixed sequence length management with padding
    - Special token handling (SOS, EOS, PAD)
    
    For detailed implementation notes and design decisions, refer to the project documentation
    in the /docs directory, particularly the 'Data Pipeline' and 'Attention Mechanism' sections.
    
    Attributes:
        data: The raw dataset containing source and target text pairs.
        src_tokenizer: Tokenizer for the source language.
        tgt_tokenizer: Tokenizer for the target language.
        src_lang (str): The key for the source language in the dataset.
        tgt_lang (str): The key for the target language in the dataset.
        seq_length (int): The fixed sequence length for padding/truncation.
        eos_token (torch.Tensor): Tensor containing the EOS token ID.
        pad_token (torch.Tensor): Tensor containing the PAD token ID.
        sos_token (torch.Tensor): Tensor containing the SOS token ID.
    """
    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_length):
        """
        Initializes the CustomDataset.
        
        This constructor sets up the dataset with all necessary components for data processing.
        It stores tokenizers for both source and target languages and prepares special tokens
        for sequence construction.

        Args:
            ds: The raw dataset containing source and target text pairs.
            src_tokenizer: Tokenizer for the source language.
            tgt_tokenizer: Tokenizer for the target language.
            src_lang (str): The key for the source language in the dataset.
            tgt_lang (str): The key for the target language in the dataset.
            seq_length (int): The fixed sequence length for padding/truncation.
        """
        super().__init__()

        self.data = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_length = seq_length

        # Get special tokens from the tokenizer
        self.eos_token = torch.tensor([self.src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([self.src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)
        self.sos_token = torch.tensor([self.src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        
        # Validate that all special tokens exist in the tokenizer
        if any(token is None for token in [self.eos_token, self.pad_token, self.sos_token]):
            missing_tokens = [name for name, token in 
                             zip(['[EOS]', '[PAD]', '[SOS]'], 
                                 [self.eos_token, self.pad_token, self.sos_token]) 
                             if token is None]
            raise ValueError(f"Missing required special tokens in tokenizer: {missing_tokens}")
        
    def __len__(self) -> int:
        """
        Returns the total number of items in the dataset.
        
        Returns:
            int: The number of samples in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset at the given index.
        
        This method processes the raw text data through tokenization, adds special tokens,
        applies padding/truncation to the fixed sequence length, and creates attention masks.
        The processed data is returned in a dictionary format suitable for model training.
        
        Design decisions:
        - SOS token is added at the beginning of encoder input
        - EOS token is added at the end of encoder input
        - Decoder input is prefixed with SOS token
        - Output labels are suffixed with EOS token
        - Padding tokens are added to reach the fixed sequence length
        
        Args:
            idx (int): The index of the sample to retrieve
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'encoder_input': Tensor for encoder input (SOS + tokens + EOS + PAD)
                - 'decoder_input': Tensor for decoder input (SOS + tokens + PAD)
                - 'encoder_mask': Mask for encoder input (hides PAD tokens)
                - 'decoder_mask': Mask for decoder input (hides PAD tokens and future tokens)
                - 'output_label': Tensor for target labels (tokens + EOS + PAD)
                - 'src_text': Original source text
                - 'tgt_text': Original target text
                
        Raises:
            ValueError: If the sentence length exceeds the maximum sequence length
        """
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'encoder_input': Tensor for encoder input (SOS + tokens + EOS + PAD).
                - 'decoder_input': Tensor for decoder input (SOS + tokens + PAD).
                - 'encoder_mask': Mask for encoder input (hides PAD tokens).
                - 'decoder_mask': Mask for decoder input (hides PAD tokens and future tokens).
                - 'output_label': Tensor for target labels (tokens + EOS + PAD).
                - 'src_text': Original source text.
                - 'tgt_text': Original target text.
        """
        
        # Get the input and output texts from the dataset
        src_text = self.data[idx][self.src_lang]
        tgt_text = self.data[idx][self.tgt_lang]

        # Tokenize and encode them
        encoder_input_tokens = self.src_tokenizer.encode(src_text).ids
        decoder_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        # Calculate padding requirements
        enc_pad = self.seq_length - len(encoder_input_tokens) - 2  # SOS + EOS
        dec_pad = self.seq_length - len(decoder_input_tokens) - 1  # SOS only

        if enc_pad < 0 or dec_pad < 0:
            raise ValueError("The sentence is too long")

        # Build encoder input with special tokens and padding
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(encoder_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * abs(enc_pad), dtype=torch.int64)
        ])

        # Build decoder input with special tokens and padding
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(decoder_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_pad, dtype=torch.int64)
        ])

        # Build output labels with EOS and padding
        output_label = torch.cat([
            torch.tensor(decoder_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_pad, dtype=torch.int64)
        ])

        # Verify sequence lengths
        assert encoder_input.size(0) == self.seq_length
        assert decoder_input.size(0) == self.seq_length
        assert output_label.size(0) == self.seq_length

        # Create encoder attention mask (1 for non-pad tokens, 0 for pad tokens)
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        
        # Create decoder attention mask (combines padding mask and causal mask)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'output_label': output_label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def causal_mask(size: int) -> torch.Tensor:
    """
    Creates a causal mask (upper triangular matrix) for the decoder to prevent attending to future tokens.
    
    The causal mask ensures that each position in the decoder can only attend to previous positions,
    maintaining the auto-regressive property of the model during training.
    
    Design decisions:
    - Using upper triangular matrix (exclusive) for masking future tokens
    - Returning boolean mask where True indicates positions to attend to
    - Adding batch dimension for compatibility with multi-head attention

    Args:
        size (int): The size of the square mask (sequence length).
        
    Returns:
        torch.Tensor: A boolean mask of shape (1, size, size) where True indicates allowed positions.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.bool)
    return ~mask  # Invert the mask so True means "attend to this position"