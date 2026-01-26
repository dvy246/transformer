# Training Process Documentation

This document explains the training pipeline for the Transformer model implementation.

## Table of Contents
- [Overview](#overview)
- [Data Pipeline](#data-pipeline)
- [Tokenization](#tokenization)
- [Training Loop](#training-loop)
- [Validation](#validation)
- [Model Checkpoints](#model-checkpoints)
- [Hyperparameters](#hyperparameters)

## Overview

The training process for the Transformer model involves several key components:
1. Data preparation and preprocessing
2. Tokenization
3. Model initialization
4. Training loop with validation
5. Model checkpoint saving

The main training script is located in `src/training/train.py`.

## Data Pipeline

### Class: `CustomDataset`

The data pipeline is implemented in `src/training/data.py` and defines a custom PyTorch Dataset.

#### Purpose
- Handles tokenization, padding, and special token insertion
- Creates appropriate masks for encoder and decoder attention
- Provides labeled examples for training

#### Key Methods
- `__init__()`: Initializes the dataset with tokenizers, languages, and sequence length
- `__len__()`: Returns the total number of items in the dataset
- `__getitem__()`: Retrieves a sample with all necessary components

#### Output Format
Each item returned by the dataset contains:
- `encoder_input`: Tensor for encoder input (SOS + tokens + EOS + PAD)
- `decoder_input`: Tensor for decoder input (SOS + tokens + PAD)
- `encoder_mask`: Mask for encoder input (hides PAD tokens)
- `decoder_mask`: Mask for decoder input (hides PAD tokens and future tokens)
- `output_label`: Tensor for target labels (tokens + EOS + PAD)
- `src_text`: Original source text
- `tgt_text`: Original target text

### Function: `casual_mask`

Creates a causal mask (upper triangular matrix) for the decoder to prevent attending to future tokens.

## Tokenization

### Functions in `src/training/tokenizer.py`

#### `get_all_sentences(ds, lang)`
Iterates over the dataset and yields sentences for a specific language.

#### `get_or_build_tokenizer(config, ds, lang)`
Retrieves an existing tokenizer from the file path defined in config, or builds and trains a new one.

**Process**:
1. Checks if tokenizer exists at the specified path
2. If not, creates a WordLevel tokenizer with Whitespace pre-tokenizer
3. Trains the tokenizer with min_frequency=2 and special tokens [PAD], [UNK], [SOS], [EOS]
4. Saves the tokenizer to the specified path
5. Returns the loaded tokenizer

#### `get_ds(config)`
Loads the dataset and prepares it for training.

**Process**:
1. Loads the "Aarif1430/english-to-hindi" dataset
2. Builds tokenizers for both source and target languages
3. Splits the dataset into training (90%) and validation (10%) sets
4. Creates CustomDataset instances for both training and validation
5. Calculates and prints the maximum sequence lengths

## Training Loop

### Function: `train_model()` in `src/training/train.py`

#### Process Steps

1. **Load Configuration**
   ```python
   cfg = get_config()
   ```

2. **Prepare Data**
   ```python
   train_ds, val_ds, train_tokenizer, val_tokenizer = get_ds(cfg)
   train_dataloader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
   val_dataloader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)
   ```

3. **Initialize Model**
   - Get vocabulary sizes from tokenizers
   - Determine device (MPS, CUDA, or CPU)
   - Build the transformer model with specified parameters
   - Create optimizer (Adam with specified learning rate)
   - Define loss function (CrossEntropy with ignore_index and label_smoothing)

4. **Training Epochs**
   For each epoch:
   - Set model to training mode
   - Iterate through training batches
   - Perform forward pass through encoder and decoder
   - Calculate loss
   - Perform backward pass and update weights
   - Run validation
   - Save model checkpoint

5. **Forward Pass Details**
   ```python
   encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
   decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
   proj_output = model.project(decoder_output)  # (B, seq_len, tgt_vocab_size)
   ```

6. **Loss Calculation**
   ```python
   loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
   ```

## Validation

### Function: `run_validation(model, val_dataloader, device, loss_fn)`

#### Process
1. Sets model to evaluation mode
2. Disables gradient computation
3. Iterates through validation batches
4. Performs forward pass
5. Calculates average validation loss
6. Prints validation loss and returns model to training mode

## Model Checkpoints

After each epoch, the model saves a checkpoint with:
- Current epoch number
- Model state dictionary
- Optimizer state dictionary
- Current loss value

Saved to: `{model_folder}/{model_basename}_{epoch+1}.pt`

## Hyperparameters

The configuration is defined in `src/config/config.py`:

```python
def get_config():
    return {
        'src_lang': 'english_sentence',    # Source language column in dataset
        'tgt_lang': 'hinid_sentence',      # Target language column in dataset (typo intentional)
        'lr': 10**-4,                     # Learning rate
        'batch_size': 350,                # Batch size for training
        'seq_lenght': 350,                # Sequence length (typo intentional)
        'd_model': 512,                   # Model dimension
        'epochs': 20,                     # Number of training epochs
        'model_folder': 'weights',        # Directory to save model weights
        'model_basename': 'base_name',    # Base name for saved models
        'preload': True,                  # Whether to preload existing models
        'experiment_name': 'runs/model',  # Name for experiment tracking
        'tokenizer_path': './tokenizer_{0}.json'  # Path template for tokenizer files
    }
```

### Key Hyperparameters Explained

- **d_model**: Dimensionality of the model embeddings and sub-layers (512 in the original paper)
- **batch_size**: Number of sequences processed together in each training step
- **seq_lenght**: Maximum length of sequences (affects memory usage and performance)
- **lr**: Learning rate for the Adam optimizer (relatively low to ensure stable training)
- **epochs**: Number of times the entire training dataset is passed through the model