import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, buffer_name, max_len=1024):
        super().__init__()
        self.buffer_name = buffer_name
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer(buffer_name, pe)

    def forward(self, x):
        # x: batch_size * seq_len * word_dim
        return self.get_buffer(self.buffer_name)[:, :x.size(-2)]
