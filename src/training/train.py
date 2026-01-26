from src.transformer.transformer import build_transformer
from src.config import config
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from src.config.config import get_config,model_weights_file_path
import warnings
from src.training.tokenizer import get_ds



def train_model(config):
         
        #check that the device is available or not
        device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        print(f'using device {device}')
        
        #make the model folder if it does not exist
        Path(config['model_folder']).mkdir(exist_ok=True)

        #get the dataset and the tokenizers
        training_ds,validation_ds,train_tokenizer,val_tokenizer=get_ds(config)

        #get the model
        model=build_transformer(train_tokenizer.get_vocab_size(),val_tokenizer.get_vocab_size(),config['seq_lenght'],config['seq_lenght'])
        
        #move the model to mps
        model.to(device)
        
        #use the tensorboard for making experiments 
        writer=SummaryWriter(config['experiment_name'])

        #optimizer is adam
        optimizer=optim.Adam(model.parameters(),lr=config['lr'],eps=1e9)
        
        #to restore the state of the model and the optimizer in case the treaining crashes 
        initial_epoc=0
        global_epoc=0

        if config['preload']:
                #load the model from the file
                file=model_weights_file_path(config,config['preload'])

                #load the model weights from the file
                state=torch.load(file,map_location=device)

                #load the model state
                model.load_state_dict(state['models_state'])

                #loading the optimizer state from the file
                optimizer.load_state_dict(state['optimizer_state'])

                #initial_epoch and global_step from the saved model state 
                initial_epoch=state['epochs']+1

                #global step
                global_step=state['global_step']
        
        #initialize loss with ignore index of the pad token BY CONVERTING TO ID
        loss=nn.CrossEntropyLoss(ignore_index=train_tokenizer.token_to_id('[PAD]').id,label_smoothing=0.1)
        
        for epochs in range(initial_epoc,config['epochs']):
                model.train()
                training_loader=tqdm(training_ds,f'processing epoch {epochs:02d}')
                for batch in training_loader:

                        encoder_input=batch['encoder_input'].to(device) # (batch_size, seq_length)
                        decoder_input=batch['decoder_input'].to(device) # (batch_size,seq_lenght)
                        encoder_mask=batch['encoder_mask'].to(device) # (batch,1,1,seq_length)
                        decoder_mask=batch['decoder_mask'].to(device) # (barch,1,seq_length,seq_length)


                        #forward pass
                        encoder_output=model.encode(encoder_input,encoder_mask) #(B,seq_length,d_model)

                        #decoder output
                        decoder_output=model.decode(encoder_output,src_mask=encoder_mask,tgt=decoder_input,tgt_mask=decoder_mask) #(batch_size,seq_length,d_model)
                        #projections
                        projections=model.project(decoder_output)  #(batch_size,seq_length,tgt_vocab_size)

                        #labels
                        label=batch['labels'].to(device) #(b,seq_length)

                        #loss
                        loss_value=loss(projections.view(-1,train_tokenizer.get_vocab_size()),label.view(-1))
                        #show the progress bar
                        training_loader.set_postfix(f'loss:{loss_value.item():6.3f}')

                        #Log the loss
                        writer.add_scalar('training_loss',loss_value.item(),global_step)

                        #backward_optimization
                        optimizer.zero_grad()
                        loss.backward()
                        
                        #optimize
                        optimizer.step()

                        global_step += 1

                #save the model and its things after every epoch
                torch.save(
                        {
                                'models_state':model.state_dict(),
                                'epochs':epochs,
                                'global_step':global_step,
                                'optimizer_state':optimizer.state_dict()
                        },model_weights_file_path(config,epochs)
                )





if __name__ == '__main__':
        warnings.filterwarnings('ignore')
        config = get_config()
        train_model(config=config)
