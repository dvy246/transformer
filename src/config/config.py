"""
Configuration module for the Transformer model.

This module defines the configuration parameters and utility functions needed to set up
and run the Transformer model. It includes hyperparameters, paths, and other settings
that control the model's behavior during training, evaluation, and inference.

The configuration encompasses:
1. Language specifications for source and target languages
2. Model hyperparameters (dimensions, layers, attention heads)
3. Training parameters (learning rate, batch size, epochs)
4. File paths for model weights, tokenizer, and experiment logs
5. Utility functions for generating file paths

Key features:
- Centralized configuration management
- Default hyperparameters based on the original Transformer paper
- Flexible path management for model artifacts
- Consistent parameter interface across the project

For detailed implementation notes and design decisions, refer to the project documentation
in the /docs directory, particularly the 'Configuration Management' and 'Hyperparameter Tuning' sections.
"""

from pathlib import Path


def get_config():
    '''
    to get the configuration from the config file
    
    '''
    return {

        'src_lang':'english_sentence',
        'tgt_lang':'hindi_sentence',
        'lr':10**-4,
        'batch_size':350,
        'seq_length':350,  # Fixed typo: was 'seq_lenght'
        'd_model':512,
        'epochs':20,
        'model_folder':'weights',
        'model_basename':'base_name',
        'preload':False,
        'experiment_name':'runs/model',
        'tokenizer_path':'./tokenizer_{0}.json'

    }


def model_weights_file_path(config,epochs):
    '''
    to get the model weights file path

    '''
    return Path(config['model_folder']).joinpath(config['model_basename']+'_{0}.pth'.format(epochs))