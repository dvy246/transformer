# Transformer Implementation Documentation

Welcome to the documentation for the Transformer model implementation from scratch. This project provides a complete implementation of the original Transformer architecture described in "Attention Is All You Need" by Vaswani et al.

## Table of Contents

### Getting Started
- [README](../README.md) - Main project overview and setup instructions

### Detailed Documentation
- [Components Breakdown](components.md) - In-depth explanation of each transformer component
- [Model Architecture](model.md) - Complete architecture overview
- [Training Process](training.md) - Full training pipeline explanation
- [API Reference](api_reference.py) - Python docstrings for main classes and functions

### Quick References
- [Quick Reference Guide](quick_reference.md) - Fast lookup when returning to the code

## About This Implementation

This Transformer implementation focuses on educational clarity while maintaining the core architectural elements from the original paper:

- Complete encoder-decoder architecture
- Multi-head attention mechanisms
- Positional encodings
- Residual connections and layer normalization
- Masked self-attention for autoregressive generation
- Encoder-decoder attention for translation tasks

## Key Files

- `src/transformer/components.py` - Core building blocks
- `src/transformer/model.py` - Model architecture classes
- `src/transformer/transformer.py` - Model builder
- `src/training/train.py` - Training pipeline
- `src/config/config.py` - Configuration parameters

## Purpose

This implementation is designed for learning and understanding the Transformer architecture. It avoids some optimizations found in production systems to prioritize clarity and educational value.

## Next Steps

1. Start with the [README](../README.md) for setup instructions
2. Review the [Components Breakdown](components.md) for understanding individual parts
3. Study the [Model Architecture](model.md) for how components fit together
4. Examine the [Training Process](training.md) to understand how the model learns
5. Use the [Quick Reference Guide](quick_reference.md) when returning to the code later