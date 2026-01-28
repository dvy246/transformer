import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.training.data import causal_mask
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os
import time
from src.config.config import get_config
"""
Validation module for the Transformer model.

This module contains utilities and functions for validating the Transformer model during or after training.
It provides evaluation metrics, validation loops, and performance assessment tools.

Currently includes:
- PyTorch imports for neural network operations
- TensorBoard logging utilities for validation metrics
- Framework for implementing validation loops
- Evaluation metrics computation

For detailed implementation notes and design decisions, refer to the project documentation
in the /docs directory, particularly the 'Validation Strategy' and 'Evaluation Metrics' sections.
"""



def greedy_decoder(model, source,source_mask,tokenizer_src,max_len,tokenizer_tgt,device,global_step):

    """
    Greedy decoding function for the Transformer model.
    """
    sos_input_idx=tokenizer_src.token_to_id('[SOS]')
    eos_input_idx=tokenizer_tgt.token_to_id('[EOS]')
    
    #precompute the encoder output 
    enocder_output=model.encode(source,source_mask)

    #initalize the decoder model with sos token
    decoder_input=torch.empty(1,1).fill_(sos_input_idx).type_as(source).to(device)
    
    while True:

        if decoder_input.size(0)==max_len:
            break
         
        #build the decoder mask
        decoder_mask=causal_mask(decoder_input.size(1)).type_as(source).to(device)

        #calculate the decoder output
        decoder_output=model.decode(enocder_output,source_mask,decoder_input,decoder_mask)

        #get the next token 
        logits=model.project(decoder_output[-1])

        #select the token with the highest probability because its greedy decoder
        _,max=torch.max(logits,dim=1)

        decoder_input=torch.cat(
            [
                decoder_input,
                torch.empty(1,1).fill_(max.item()).type_as(source).to(device).to(device=device)
            ],
            dim=1
        )
        
        if decoder_input==sos_input_idx:
            break

    #squeeze to remove the batch dimention at 0
    return decoder_input.squeeze(0)


def run_validation(validation_ds,model,src_tokenizer,tgt_tokenizer,max_len,device,writer,print_msg,num_examples=2):
    """
    Run validation on the provided dataset using the given model.

    """

    #put the model into evaluation mode
    model.eval()
    count=0

    source_texts=[]
    expected_texts=[]
    predicted_texts=[]


    #control window size
    control_window=80

    with torch.no_grad():

        for batch in validation_ds:
            count+=1

            encoder_input=batch['econder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)


            assert encoder_input.size(0)==1 #batch size must be 1 for validation

            #when we need to inference the model  we need to calculate the encoder output only once 
            output=greedy_decoder(model=model,tokenizer_src=src_tokenizer,tokenizer_tgt=tgt_tokenizer,device=device,source=encoder_input,source_mask=encoder_mask,max_len=max_len)


            src_text=batch['src_text'][0]
            tgt_text=batch['tgt_text'][0]
            model_output_text=src_tokenizer.decode(output.detach().cpu().numpy())


            source_texts.append(src_text)
            expected_texts.append(tgt_text)
            predicted_texts.append(model_output_text)


            #print_msg because in training loop we are using tqdm progress bar and not suggested to print on console while the tqdm is running so we use print_msg
            print_msg(f'Source: {src_text}\nTarget: {tgt_text}\nPrediction: {model_output_text}\n')


            if count==num_examples:
                break
             






