import numpy as np
import torch
from torch import nn
from torch.nn import init


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        if self.training:
            att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out, att


class AttentionBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(AttentionBlock, self).__init__()
        self.attention = ScaledDotProductAttention(d_model, d_k, d_v, h, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 3 * d_model),
            nn.ReLU(),
            nn.Linear(3 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, 1),
            nn.Softmax()
        )

    def forward(self, inputs):
        attention_out, att_score = self.attention(inputs, inputs, inputs)
        normalized_out = self.norm(attention_out)
        return self.feed_forward(normalized_out)


class CopyNet(nn.Module):
    def __init__(self, d_model):
        super(CopyNet, self).__init__()
        self.embedding_size = d_model
        self.LinearSource = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.LinearTarget = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.LinearRes = nn.Linear(self.embedding_size, 1)
        self.LinearProb = nn.Linear(self.embedding_size, 3)

    def forward(self, source, traget):
        sourceLinear = self.LinearSource(source)
        targetLinear = self.LinearTarget(traget)
        genP = self.LinearRes(torch.nn.functional.tanh(sourceLinear.unsqueeze(1) + targetLinear.unsqueeze(2))).squeeze(
            -1)

        prob = torch.nn.functional.softmax(self.LinearProb(traget), dim=-1)  # .squeeze(-1))
        return genP, prob


if __name__ == '__main__':
    # input = torch.randn(50, 49, 512)
    # sa = AttentionBlock(d_model=512, d_k=512, d_v=512, h=8)
    # output = sa(input)
    # print(output.shape)
    net = CopyNet(1000)
    source = torch.randn(2, 48, 1, 1000)
    target = torch.randn(3, 1, 726, 1000)
    res = net(source, target)
