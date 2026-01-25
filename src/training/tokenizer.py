import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,random_split

from datasets import load_dataset
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


def get_all_sentences(ds,lang):
    for items in ds:
        yield items[lang]


def get_or_build_tokenizer(config,ds,lang):
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

    

