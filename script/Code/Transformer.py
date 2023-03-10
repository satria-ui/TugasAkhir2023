import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, embed_size, heads):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads