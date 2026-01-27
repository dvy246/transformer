# Transformer Implementation from Scratch

This project implements a Transformer model from scratch using PyTorch, following the original "Attention Is All You Need" paper. 

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training Process](#training-process)
- [Components Breakdown](#components-breakdown)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Transformer model revolutionized Natural Language Processing by relying entirely on attention mechanisms, dispensing with recurrence and convolutions. This implementation includes:

- Complete encoder-decoder architecture
- Multi-head attention mechanism
- Positional encoding
- Feed-forward networks
- Layer normalization
- Residual connections

### Key Features

- Pure PyTorch implementation (no external NLP libraries)
- Modular design for easy understanding
- Configurable hyperparameters
- Custom tokenization pipeline
- Complete training loop with validation

## Architecture

The Transformer consists of an encoder stack and a decoder stack with the following components:

### Encoder
- Input embedding layer
- Positional encoding
- N identical layers, each containing:
  - Multi-head self-attention mechanism
  - Position-wise feed-forward networks
  - Residual connections and layer normalization

### Decoder
- Output embedding layer
- Positional encoding
- N identical layers, each containing:
  - Masked multi-head self-attention mechanism
  - Multi-head encoder-decoder attention
  - Position-wise feed-forward networks
  - Residual connections and layer normalization

## Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 2.10.0

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd transformer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# Using pip
pip install torch torchvision torchaudio
pip install datasets pandas python-dotenv tokenizers transformers

# Or using uv (if available)
uv pip install torch torchvision torchaudio
uv pip install datasets pandas python-dotenv tokenizers transformers
```

## Usage

### Training a Model

To train the model:
```bash
python -m src.training.train
```

Or using uv:
```bash
uv run python -m src.training.train
```

### Custom Training

You can customize the training by modifying the configuration in `src/config/config.py`.

## Project Structure

```
transformer/
├── README.md                           # This file
├── pyproject.toml                      # Project dependencies
├── main.py                            # Entry point (optional)
├── src/
│   ├── config/
│   │   └── config.py                  # Configuration parameters
│   ├── training/
│   │   ├── data.py                    # Dataset handling
│   │   ├── tokenizer.py               # Tokenizer implementation
│   │   └── train.py                   # Training pipeline
│   ├── transformer/
│   │   ├── components.py              # Core transformer components
│   │   ├── model.py                   # Base model classes
│   │   └── transformer.py             # Transformer builder
│   ├── data.ipynb                     # Data exploration notebook
│   └── notebook.ipynb                 # Experimentation notebook
├── docs/                              # Detailed documentation
│   ├── components.md                  # Component breakdown
│   ├── model.md                       # Model architecture
│   ├── training.md                    # Training process
│   └── quick_reference.md             # Quick reference guide
└── transformers/                      # Virtual environment (should be renamed to avoid confusion)
```

## Configuration

The configuration is managed in `src/config/config.py`. Key parameters include:

- `src_lang`: Source language column name in dataset
- `tgt_lang`: Target language column name in dataset
- `lr`: Learning rate
- `batch_size`: Batch size for training
- `seq_lenght`: Sequence length (note: spelling intentional)
- `d_model`: Dimension of the model
- `epochs`: Number of training epochs
- `model_folder`: Directory to save model weights
- `model_basename`: Base name for saved models
- `preload`: Whether to preload existing models
- `experiment_name`: Name for experiment tracking
- `tokenizer_path`: Path template for tokenizer files

## Training Process

The training process involves:

1. **Data Preparation**: Loading and preprocessing the dataset
2. **Tokenization**: Creating and using custom tokenizers
3. **Model Initialization**: Building the transformer model
4. **Training Loop**: Iterating through epochs and batches
5. **Validation**: Running validation after each epoch
6. **Checkpoint Saving**: Saving model checkpoints periodically

### Data Pipeline

The data pipeline (`src/training/data.py`) creates a custom dataset that:
- Handles tokenization
- Adds special tokens (SOS, EOS, PAD)
- Creates appropriate masks for attention
- Pads sequences to consistent lengths

### Tokenization

The tokenization system (`src/training/tokenizer.py`) builds word-level tokenizers and:
- Processes both source and target languages
- Handles special tokens
- Manages vocabulary creation
- Saves tokenizers for later use

## Components Breakdown

### Core Components (`src/transformer/components.py`)
- `InputEmbeddings`: Converts token IDs to dense vectors
- `PositionalEncoding`: Adds positional information to embeddings
- `MultiHeadAttentionBlock`: Implements multi-head attention mechanism
- `FeedForwardBlock`: Position-wise feed-forward network
- `SkipConnection`: Implements residual connections with dropout
- `LayerNormalization`: Layer normalization implementation
- `ProjectionLayer`: Projects final decoder output to vocabulary space

### Model Classes (`src/transformer/model.py`)
- `Encoder`: Stack of encoder layers
- `EncoderLayer`: Single encoder layer with self-attention and FFN
- `Decoder`: Stack of decoder layers
- `DecoderLayer`: Single decoder layer with self-attention, encoder-decoder attention, and FFN
- `Transformer`: Combines encoder and decoder with embeddings and projection

### Model Builder (`src/transformer/transformer.py`)
- `build_transformer`: Constructs a complete transformer model with proper initialization

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [components.md](docs/components.md): Detailed explanation of each transformer component
- [model.md](docs/model.md): In-depth architecture overview
- [training.md](docs/training.md): Complete training process documentation
- [quick_reference.md](docs/quick_reference.md): Quick reference for when you return to the code

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
