import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,random_split

from datasets import load_dataset
from src.training.data import CustomDataset
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


def get_all_sentences(ds,lang):
    """
    Iterates over the dataset and yields sentences for a specific language.

    Args:
        ds: The dataset object containing the data.
        lang (str): The key (column name) for the language in the dataset items.

    Yields:
        str: A sentence from the dataset for the specified language.
    """
    for items in ds:
        yield items[lang]


def get_or_build_tokenizer(config,ds,lang):
    """
    Retrieves an existing tokenizer from the file path defined in config, or builds and trains a new one.

    Args:
        config (dict): Configuration dictionary containing 'tokenizer_path'.
        ds: The dataset used to train the tokenizer if it doesn't exist.
        lang (str): The language key to extract sentences from the dataset.

    Returns:
        Tokenizer: The loaded or newly trained tokenizer object.
    """
    #define the tokenizer path
    path=Path(config["tokenizer_path"].format(lang))

    if not path.exists():
        #build the tokenizer
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        #Pre tokenizer
        tokenizer.pre_tokenizer=Whitespace()
        #Trainer
        trainer=WordLevelTrainer(min_frequency=2,special_tokens=['[PAD]','[UNK]','[SOS]','[EOS]'])

        #train the tokenizer using the trainer
        tokenizer.train_from_iterator(get_all_sentences(ds),trainer=trainer)

        #save the tokenizer
        tokenizer.save(path)

    else:
        tokenizer=Tokenizer.from_file(path)

    return tokenizer

def get_ds(config):
    """
    This function loads the dataset and splits it into a training set and a validation set.
    
    Parameters:
    config (dict): The configuration dictionary containing the path to the tokenizer.
    
    Returns:
    A tuple of two datasets, the first is the training set and the second is the validation set.
    """

    ds=load_dataset("Aarif1430/english-to-hindi",split='train')
    
    #build tokenizer
    train_tokenizer=get_or_build_tokenizer(config=config,ds=ds,lang="english_sentence")
    val_tokenizer=get_or_build_tokenizer(config=config,ds=ds,lang="hindi_sentence")

    # train test split
    train_ds_size=int(0.9*len(ds))
    test_ds_size=int(0.1* len(ds))
    

    #split the data 
    train_ds_raw,train_ds_val=random_split(ds,[train_ds_size,test_ds_size])
    
    #create the instance of the dataset
    training_ds=CustomDataset(train_ds_raw,train_tokenizer,val_tokenizer,config['src_lang'],tgt_lang=config['tgt_lang'],seq_lenght=config['seq_lenght'])
    validation_ds=CustomDataset(train_ds_val,train_tokenizer,val_tokenizer,config['src_lang'],tgt_lang=config['tgt_lang'],seq_lenght=config['seq_lenght'])

    
    #we will check the size of the sequence in the dataset and adjust the sequence lenght

    max_len_src=0
    max_len_tgt=0
    

    for item in training_ds:
        src_ids=train_tokenizer.encode(item[config['src_lang']]).ids
        tgt_ids=val_tokenizer.encode(item[config['tgt_lang']]).ids

        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))

    print(f"max len src {max_len_src} max len tgt {max_len_tgt}")

    return training_ds,validation_ds,train_tokenizer,val_tokenizer