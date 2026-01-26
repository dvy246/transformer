from pathlib import Path


def get_config():
    '''
    to get the configuration from the config file
    
    '''
    return {

        'src_lang':'english_sentence',
        'tgt_lang':'hinid_sentence',
        'lr':10**-4,
        'batch_size':350,
        'seq_lenght':350,
        'd_model':512,
        'epochs':20,
        'model_folder':'weights',
        'model_basename':'base_name',
        'preload':True,
        'experiment_name':'runs/model',
        'tokenizer_path':'./tokenizer_{0}.json'

    }


def model_weights_file_path(config,epochs):
    '''
    to get the model weights file path

    '''
    return Path(config['model_folder']).joinpath(config['model_basename']+'_{0}.pth'.format(epochs))
