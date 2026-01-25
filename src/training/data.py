from torch.utils.data import Dataset,DataLoader
import torch


class CustomDataset(Dataset):
    def __init__(self, ds,src_tokenizer,tgt_tokenizer,src_lang,tgt_lang,seq_lenght):
        super().__init__()

        self.data = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_lenght = seq_lenght


        self.eos_token=torch.tensor(self.src_tokenizer.token_to_id([['EOS']]),dtype=torch.int64)
        self.pad_token=torch.tensor(self.src_tokenizer.token_to_id([['PAD']]),dtype=torch.int64)
        self.sos_token=torch.tensor(self.src_tokenizer.token_to_id([['SOS']]),dtype=torch.int64)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        #get the input and output texts from the dataset
        src_text=self.data[idx][self.src_lang]
        tgt_text=self.data[idx][self.tgt_lang]

        #tokenize and encode them
        encoder_input_tokens=self.src_tokenizer.encode(src_text).ids
        decoder_input__tokens=self.tgt_tokenizer.encode(tgt_text).ids


        #pad the input and output texts
        enc_padd=self.seq_lenght-len(encoder_input_tokens)-2 #-2 cause of sos and eos
        dec_padd=self.seq_lenght-len(decoder_input__tokens)-1 #-1 speical taoken only

        if enc_padd < 0 and dec_padd <0:
            raise ValueError("The sentence is too long")
        
        #encoder_input
        #ADD SOS and EOS Tokens to the encoder input
        encoder_input=torch.cat(

            [self.sos_token,
             torch.tensor(encoder_input_tokens,dtype=torch.int64),
             self.eos_token,
             torch.tensor([self.pad_token]*abs(enc_padd),dtype=torch.int64)
             ]
        )

        #decoder_input
        #ADD SOS Token to the decoder input
        decoder_input=torch.cat(
            [
                self.eos_token,
                torch.tensor(decoder_input__tokens,dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_padd,dtype=torch.int64)

            ]
        )

        #label 
        #add the eos token to the label
        output_label=torch.cat(
            [
                torch.tensor(decoder_input__tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_padd,dtype=torch.int64)


            ]
        )

        assert encoder_input.size(0)==self.seq_lenght
        assert decoder_input.size(0)==self.seq_lenght
        assert output_label.size(0)==self.seq_lenght
            
        return {
            'encoder_input':self.encoder_input,
            'decoder_input':self.decoder_input,
            'label':self.output_label
        }
