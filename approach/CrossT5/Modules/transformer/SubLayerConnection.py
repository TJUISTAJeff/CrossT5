import torch.nn as nn
from Modules.transformer.LayerNorm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # nn.MultiHeadAttention will return a tuple includes output(idx=0) and attention weight(idx=1)
        return x + self.dropout(sublayer(self.norm(x))[0])