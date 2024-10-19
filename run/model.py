import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def embedding_layer(vocab_size, embedding_size):
    embedding = nn.Embedding(vocab_size, embedding_size)
    

class RNNModel():
    def __init__ (self, hidden_size=512, output_size=2, time_steps=0):
        self.hidden_size = hidden_size
        self.output_size = output_size            # o(negetive) or 1(positive)
        self.time_steps = time_steps

        # Weight Initialization
        self.W_hh = torch.randn(self.hidden_size, self.hidden_size)
        self.W_xh = torch.randn(self.input_size, self.hidden_size)
        self.W_hy = torch.randn(self.hidden_size, self.output_size)

    def init_hidden_state(self):
        return torch.zeros(1, self.hidden_size)

    def forward (self, input):
        x = self.embedding(input)
        h_0 = self.init_hidden_state()

    def backward (self, l, y):
        pass