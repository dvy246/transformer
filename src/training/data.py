from torch.utils.data import Dataset,DataLoader
import torch


class CustomDataset(Dataset):
<<<<<<< HEAD
    """
    A custom PyTorch Dataset for sequence-to-sequence tasks.
    It handles tokenization, padding, and adding special tokens (SOS, EOS) to the input sequences.
    """
    def __init__(self, ds,src_tokenizer,tgt_tokenizer,src_lang,tgt_lang,seq_lenght):
        """
        Initializes the CustomDataset.

        Args:
            ds: The raw dataset containing source and target text pairs.
            src_tokenizer: Tokenizer for the source language.
            tgt_tokenizer: Tokenizer for the target language.
            src_lang (str): The key for the source language in the dataset.
            tgt_lang (str): The key for the target language in the dataset.
            seq_lenght (int): The fixed sequence length for padding/truncation.
        """
=======
    def __init__(self, ds,src_tokenizer,tgt_tokenizer,src_lang,tgt_lang,seq_lenght):
>>>>>>> 837e97f1282eeabab6674f956e8c28d0cfdc63b5
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
<<<<<<< HEAD
        """
        Returns the total number of items in the dataset.
        """

        return len(self.data)
    
    def __getitem__(self, idx):
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
=======
        return len(self.data)
    
    def __getitem__(self, idx):
>>>>>>> 837e97f1282eeabab6674f956e8c28d0cfdc63b5
        
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
<<<<<<< HEAD
            'encoder_input':encoder_input,
            'decoder_input':decoder_input,
            'encoder_mask':(encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask':(decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            'output_label':output_label,
            'src_text':src_text,
            'tgt_text':tgt_text
        }


def casual_mask(size):
    """
    Creates a causal mask (upper triangular matrix) for the decoder to prevent attending to future tokens.

    Args:
        size (int): The size of the square mask (sequence length).

    Returns:
        torch.Tensor: A boolean mask where True indicates allowed positions.
    """
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask==0
=======
            'encoder_input':self.encoder_input,
            'decoder_input':self.decoder_input,
            'label':self.output_label
        }
>>>>>>> 837e97f1282eeabab6674f956e8c28d0cfdc63b5
