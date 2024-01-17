import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, char_vocab_size, max_char_seq_len, embed_dim):
        super(TokenEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_char_seq_len = max_char_seq_len
        self.char_vocab_size = char_vocab_size
        self.character_weights = Parameter(torch.FloatTensor(char_vocab_size, embed_dim), requires_grad=True)
        self.feed_forward_hidden = 2 * embed_dim
        self.conv = nn.Conv2d(self.embed_dim, self.embed_dim, (1, self.max_char_seq_len))
        self.reset_parameters()

    def reset_parameters(self):
        self.character_weights.data.normal_(0, 1)

    def forward(self, characters):
        char_embeds = F.embedding(characters.long(), self.character_weights)
        char_embeds = self.conv(char_embeds.permute(0, 3, 1, 2))
        outputs = char_embeds.permute(0, 2, 3, 1).squeeze(dim=-2)
        return outputs


# if __name__ == '__main__':
#     import sys
#
#     sys.path.extend(['.', '..'])
#     embedding = TokenEmbedding(100, 15, 256, 0)
#     import numpy as np
#
#     input = torch.from_numpy(np.random.random_integers(0, 99, (1, 512, 15))).long()
#     output = embedding(input)
#     pass
