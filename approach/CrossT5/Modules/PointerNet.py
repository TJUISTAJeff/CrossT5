import torch
import torch.nn as nn
import torch.nn.functional as F


class CopyNet(nn.Module):
    def __init__(self, hidden_size, down_size=None):
        super(CopyNet, self).__init__()
        self.embedding_size = hidden_size
        if not down_size:
            self.down_size = hidden_size
        else:
            self.down_size = down_size
        self.LinearSource = nn.Linear(self.embedding_size, self.down_size, bias=False)
        self.LinearTarget = nn.Linear(self.embedding_size, self.down_size, bias=False)
        self.LinearRes = nn.Linear(self.embedding_size, 1)
        # self.LinearProb = nn.Linear(self.embedding_size, 2)

    def forward(self, source, target):
        sourceLinear = self.LinearSource(source)
        targetLinear = self.LinearTarget(target)
        genP = self.LinearRes(torch.tanh(sourceLinear.unsqueeze(1) + targetLinear.unsqueeze(2))).squeeze(-1)
        # prob = F.softmax(self.LinearProb(target), dim=-1)  # .squeeze(-1))
        return genP, []

    def updateParam4ATLAS(self, hidden_size):
        self.LinearSource = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.LinearTarget = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self.LinearRes = nn.Linear(hidden_size * 2, 1)
        self.LinearProb = nn.Linear(hidden_size * 2, 2)


class Attention_based_CpyNet(nn.Module):
    def __init__(self, hidden_size, down_size=None):
        super(Attention_based_CpyNet, self).__init__()
        self.embedding_size = hidden_size
        if not down_size:
            self.down_size = hidden_size
        else:
            self.down_size = down_size
        self.generator = nn.MultiheadAttention(embed_dim=self.embedding_size,
                                               num_heads=1,
                                               batch_first=True)
        # self.LinearRes = nn.Linear(self.embedding_size, 1)
        # self.LinearProb = nn.Linear(self.embedding_size, 2)

    def forward(self, encoded_outputs, vocab_embeds, vocab_masks):
        _, attn_output_weights = self.generator(query=encoded_outputs, key=vocab_embeds, value=vocab_embeds,
                                                key_padding_mask=vocab_masks)
        return attn_output_weights


if __name__ == '__main__':
    net = Attention_based_CpyNet(256)
    source = torch.randn(10, 1000, 256)  # batch_size * seq_len * 1 * hidden_dim
    target = torch.randn(10, 3000, 256)  # batch_size * 1 * vocab_size * hidden_dim
    masks = torch.ones(10, 3000)
    for i in range(10):
        masks[i][0] = 0
    masks[0][1] = 0
    masks = masks.eq(1)
    genP, prob = net(source, target, masks)
    print(genP.shape, prob.shape)
    pass
