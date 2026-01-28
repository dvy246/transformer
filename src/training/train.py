"""
Training script for the Transformer model.

This module contains the `train_model` function which orchestrates the training process
of the Transformer architecture. It handles device configuration, dataset loading, model
initialization, optimization, and checkpointing. The training loop includes detailed
monitoring via TensorBoard and supports resuming from saved checkpoints.

Key features:
- Automatic device selection (MPS/CPU)
- Checkpoint-based training resumption
- Label smoothing for regularization
- TensorBoard integration for loss tracking
- Modular design following the project's architecture specifications

The training process follows the standard encoder-decoder architecture:
1. Encoder processes source sequence with self-attention
2. Decoder generates target sequence with masked self-attention
3. Projection layer converts decoder output to vocabulary space
4. Loss computed against ground truth labels

For detailed implementation notes and design decisions, refer to the project documentation
in the /docs directory, particularly the 'Training Process' and 'Model Architecture' sections.
"""

from src.transformer.transformer import build_transformer
from src.config import config
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from src.training.val import run_validation
from src.config.config import get_config, model_weights_file_path
import warnings
from src.training.tokenizer import get_ds


def train_model(config):
    """
    Train the Transformer model according to the provided configuration.
    
    This function implements the complete training workflow including:
    1. Device initialization (MPS/CPU)
    2. Dataset and tokenizer preparation
    3. Model construction and placement on target device
    4. Optimizer and loss function setup
    5. Checkpoint resumption (if configured)
    6. Epoch-wise training with batch processing
    7. Loss tracking via TensorBoard
    8. Periodic model checkpointing

    Key implementation decisions:
    - Label smoothing (0.1) applied to CrossEntropyLoss for regularization
    - Adam optimizer with default parameters (eps=1e-9) for stable convergence
    - Padding token ([PAD]) ignored during loss calculation
    - Global step tracking for TensorBoard alignment
    - MPS device prioritization for Apple Silicon acceleration

    Time complexity:
    O(epochs * N) where N is the number of batches in the training dataset.
    Each forward/backward pass has complexity dependent on sequence length and model size.

    Args:
        config (dict): Configuration dictionary containing:
            - model_folder: Directory to save model checkpoints
            - experiment_name: TensorBoard experiment identifier
            - preload: Optional checkpoint identifier to resume training
            - epochs: Total number of training epochs
            - lr: Learning rate for Adam optimizer
            - seq_length: Maximum sequence length for inputs
            - Other dataset/model parameters

    Returns:
        None. Saves model checkpoints to disk and logs metrics to TensorBoard.
        
    Note:
        The function follows the standard encoder-decoder architecture:
        1. Encoder processes source sequence with self-attention
        2. Decoder generates target sequence with masked self-attention
        3. Projection layer converts decoder output to vocabulary space
        4. Loss computed against ground truth labels
    """
    
    # ============================================================================
    # STEP 1: DEVICE SETUP
    # ============================================================================
    # Automatic device selection prioritizes MPS (Apple Silicon) when available
    # for accelerated training on macOS systems. Falls back to CPU if MPS unavailable.
    # This decision leverages Apple Silicon's native acceleration capabilities while
    # maintaining cross-platform compatibility.
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'using device {device}')
    
    
    # ============================================================================
    # STEP 2: CREATE MODEL DIRECTORY
    # ============================================================================
    # Create the directory where model weights will be saved
    # exist_ok=True prevents errors if directory already exists
    Path(config['model_folder']).mkdir(exist_ok=True)
    
    
    # ============================================================================
    # STEP 3: LOAD DATASETS AND TOKENIZERS
    # ============================================================================
    # get_ds() returns:
    # - training_ds: Training dataset (DataLoader or iterable)
    # - validation_ds: Validation dataset (not used in this training loop)
    # - train_tokenizer: Tokenizer for source language (used for vocab size and PAD token)
    # - val_tokenizer: Tokenizer for target language (used for vocab size)
    training_ds, validation_ds, train_tokenizer, val_tokenizer = get_ds(config)
    
    
    # ============================================================================
    # STEP 4: BUILD THE TRANSFORMER MODEL
    # ============================================================================
    # build_transformer creates a new Transformer model with:
    # - Source vocabulary size (from train_tokenizer)
    # - Target vocabulary size (from val_tokenizer)
    # - Sequence length for both encoder and decoder
    model = build_transformer(
        train_tokenizer.get_vocab_size(),
        val_tokenizer.get_vocab_size(),
        config['seq_length'],
        config['seq_length']
    )
    
    # Move model to the selected device (MPS or CPU)
    model.to(device)
    
    
    # ============================================================================
    # STEP 5: SETUP TENSORBOARD LOGGING
    # ============================================================================
    # SummaryWriter logs training metrics to TensorBoard
    # This allows visualization of loss curves and other metrics during training
    writer = SummaryWriter(config['experiment_name'])
    
    
    # ============================================================================
    # STEP 6: INITIALIZE OPTIMIZER
    # ============================================================================
    # Adam optimizer with specified learning rate
    # eps=1e9 is a very large epsilon value (unusual, typically 1e-8)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], eps=1e9)
    
    
    # ============================================================================
    # STEP 7: INITIALIZE TRAINING STATE VARIABLES
    # ============================================================================
    # These variables track training progress and enable checkpoint resumption
    initial_epoc = 0  # Starting epoch (will be updated if loading from checkpoint)
    global_step = 0   # Global step counter across all epochs (will be updated if loading from checkpoint)
    
    
    # ============================================================================
    # STEP 8: LOAD FROM CHECKPOINT (IF PRELOAD IS ENABLED)
    # ============================================================================
    # Resume training state from checkpoint including model weights, optimizer
    # state, and training progress counters. This enables fault-tolerant training
    # that can recover from interruptions.
    #
    # Key implementation detail: Preserves optimizer state (including learning rate
    # scheduler information) to ensure training continues exactly where it left off.
    if config['preload']:
        # Get the file path for the checkpoint
        file = model_weights_file_path(config, config['preload'])
        
        # Load the checkpoint dictionary from disk
        # map_location=device ensures tensors are loaded to the correct device
        state = torch.load(file, map_location=device)
        
        # Restore model weights from checkpoint
        model.load_state_dict(state['models_state'])
        
        # Restore optimizer state (learning rate schedules, momentum, etc.)
        optimizer.load_state_dict(state['optimizer_state'])
        
        # Resume from the next epoch after the saved checkpoint
        initial_epoch = state['epochs'] + 1
        
        # Resume from the global step where training was interrupted
        global_step = state['global_step']
    
    
    # ============================================================================
    # STEP 9: INITIALIZE LOSS FUNCTION
    # ============================================================================
    # Initialize loss with label smoothing (0.1) for regularization and
    # ignoring padding tokens during loss calculation. The [PAD] token ID is
    # dynamically retrieved from the tokenizer to ensure consistency.
    # 
    # Key design decision: Label smoothing improves generalization by preventing
    # the model from becoming overconfident in its predictions, which helps reduce
    # overfitting in sequence-to-sequence tasks.
    loss = nn.CrossEntropyLoss(
        ignore_index=train_tokenizer.token_to_id('[PAD]'),
        label_smoothing=0.1
    )
    
    
    # ============================================================================
    # STEP 10: MAIN TRAINING LOOP
    # ============================================================================
    # Iterate through epochs
    for epochs in range(initial_epoc, config['epochs']):
        
        # Create progress bar for the epoch
        training_loader = tqdm(training_ds, f'processing epoch {epochs:02d}')
        
        # Iterate through batches in the training dataset
        for batch in training_loader:
            
            # Set model to training mode (enables dropout, batch norm updates, etc.)
            model.train()

            # ====================================================================
            # STEP 10.1: LOAD BATCH DATA TO DEVICE
            # ====================================================================
            # Extract tensors from batch and move to device (MPS or CPU)
            
            # encoder_input: Source language tokens
            # Shape: (batch_size, seq_length)
            encoder_input = batch['encoder_input'].to(device)
            
            # decoder_input: Target language tokens (shifted right for autoregressive decoding)
            # Shape: (batch_size, seq_length)
            decoder_input = batch['decoder_input'].to(device)
            
            # encoder_mask: Attention mask for encoder (prevents attention to padding)
            # Shape: (batch_size, 1, 1, seq_length)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # decoder_mask: Causal attention mask for decoder (prevents looking ahead)
            # Shape: (batch_size, 1, seq_length, seq_length)
            decoder_mask = batch['decoder_mask'].to(device)
            
            
            # ====================================================================
            # STEP 10.2: FORWARD PASS - ENCODER
            # ====================================================================
            # Encoder processes source language input and produces contextual representations
            # encoder_output: Encoded representation of source sequence
            # Shape: (batch_size, seq_length, d_model)
            encoder_output = model.encode(encoder_input, encoder_mask)
            
            
            # ====================================================================
            # STEP 10.3: FORWARD PASS - DECODER
            # ====================================================================
            # Decoder processes target language input using encoder output as context
            # decoder_output: Decoded representation of target sequence
            # Shape: (batch_size, seq_length, d_model)
            decoder_output = model.decode(
                encoder_output,
                src_mask=encoder_mask,
                tgt=decoder_input,
                tgt_mask=decoder_mask
            )
            
            
            # ====================================================================
            # STEP 10.4: PROJECT TO VOCABULARY
            # ====================================================================
            # Project decoder output to vocabulary size for token prediction
            # projections: Logits for each token in vocabulary
            # Shape: (batch_size, seq_length, target_vocab_size)
            projections = model.project(decoder_output)
            
            
            # ====================================================================
            # STEP 10.5: LOAD LABELS
            # ====================================================================
            # Ground truth target tokens (what the model should predict)
            # Shape: (batch_size, seq_length)
            label = batch['label'].to(device)  # Reverted to singular 'label' (was 'labels')
            

            # ====================================================================
            # STEP 10.6: COMPUTE LOSS
            # ====================================================================
            # Reshape projections and labels to 1D for loss computation
            # projections: (batch_size * seq_length, target_vocab_size)
            # label: (batch_size * seq_length,)
            # loss_value: Scalar tensor containing the cross-entropy loss
            loss_value = loss(
                projections.view(-1, train_tokenizer.get_vocab_size()),
                label.view(-1)
            )


            # ====================================================================
            # STEP 10.7: UPDATE PROGRESS BAR
            # ====================================================================
            # Display current loss value in the progress bar
            training_loader.set_postfix(f'loss:{loss_value.item():6.3f}')


            # ====================================================================
            # STEP 10.8: LOG METRICS TO TENSORBOARD
            # ====================================================================
            # Record loss value for visualization in TensorBoard
            writer.add_scalar('training_loss', loss_value.item(), global_step)


            # ====================================================================
            # STEP 10.9: BACKWARD PASS AND OPTIMIZATION
            # ====================================================================
            # Clear gradients from previous iteration
            optimizer.zero_grad()
            
            # Compute gradients via backpropagation
            loss_value.backward()
            
            # Update model parameters using computed gradients
            optimizer.step()
            
            # ====================================================================
            # STEP 10 RUN VALIDATION
            # ====================================================================
            run_validation(validation_ds=validation_ds,model=model,src_tokenizer=train_tokenizer,tgt_tokenizer=val_tokenizer,max_len=config['seq_length'],device=device,writer=writer,print_msg=lambda x: training_loader.write(x),global_step=global_step)

            # Increment global step counter
            global_step += 1
        

        # ====================================================================
        # STEP 11: SAVE CHECKPOINT AFTER EPOCH
        # ====================================================================
        # Save model state, optimizer state, and training metadata
        # This allows resuming training from this checkpoint if interrupted
        torch.save(
            {
                'models_state': model.state_dict(),      # Model weights
                'epochs': epochs,                         # Current epoch number
                'global_step': global_step,               # Total steps completed
                'optimizer_state': optimizer.state_dict() # Optimizer state (momentum, etc.)
            },
            model_weights_file_path(config, epochs)
        )


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Load configuration from config file
    config = get_config()
    
    # Start training with the loaded configuration
    train_model(config=config)
