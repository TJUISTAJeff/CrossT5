import torch
import torch.nn as nn

from Modules.transformer.Attention import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=None)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


if __name__ == '__main__':
    attention = nn.MultiheadAttention(num_heads=4, embed_dim=100, dropout=0.1, batch_first=True)
    from utils.processing import getAntiMask
    import numpy as np

    query = torch.randn((16, 512, 100)).float()
    query_masks = torch.BoolTensor(16, 512)
    for i in range(16):
        randn = np.random.randint(0, 511)
        for j in range(randn):
            query_masks[i][j] = 0

    antiMask = torch.from_numpy(getAntiMask(512)).lt(1)
    outputs = attention.forward(query, query, query, key_padding_mask=query_masks, attn_mask=antiMask)
    print(outputs.shape)
    pass
