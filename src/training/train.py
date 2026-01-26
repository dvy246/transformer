from src.transformer.transformer import build_transformer
from src.config import config
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')





if __name__ == '__main__':
        print(device)
