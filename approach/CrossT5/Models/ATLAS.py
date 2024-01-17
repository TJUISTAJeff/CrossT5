import torch

from Modules.CPUEmbedding import CPUEmbedding
from Modules.PointerNet import CopyNet
from Modules.TokenEmbeddding import TokenEmbedding
from Modules.transformer.DenseLayer import DenseLayer
import sys

sys.path.extend([".", ".."])
from CONSTANTS import *

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
class ATLASEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(ATLASEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)

    def forward(self, inputs):
        inputs = self.dropout(inputs)

        # better keep batch size the first dim.
        inputs = inputs.permute(1, 0, 2)

        # input shape should be: (len, batch_size, H_in)
        output, (h_0, c_0) = self.rnn(inputs)
        # output shape: (len, batch_size, D * H_out)
        # h_0 shape: (D∗num_layers,N,H_out)
        return output, (h_0, c_0)


class ATLASDecoder(nn.Module):
    def __init__(self, input_size, hidden_dim, dropout=0.2):
        super(ATLASDecoder, self).__init__()
        self.attention = nn.MultiheadAttention(num_heads=1, embed_dim=hidden_dim, batch_first=False)
        self.rnn = nn.LSTM(input_size=hidden_dim + input_size, hidden_size=hidden_dim, dropout=dropout,
                           num_layers=2)
        self.hidden_size = hidden_dim


    def forward(self, inputs, enc_outputs, hidden_state, mask):
        inputs = inputs.permute(1, 0, 2)
        outputs = []
        c_0 = hidden_state[1]
        c = torch.cat([c_0[0], c_0[1]], dim=1).unsqueeze(dim=0)
        c = torch.cat([c, c], dim=0)
        h = torch.zeros_like(c)
        hidden_state = (h, c)
        for x in inputs:
            # query shape : target_len * batch_size * embedding_size
            query = torch.unsqueeze(hidden_state[1][-1], dim=0)
            # key & value size : source_len * batch_size * embedding_size
            context, _ = self.attention.forward(query=query, key=enc_outputs, value=enc_outputs, key_padding_mask=mask)
            x = torch.cat((context, torch.unsqueeze(x, dim=0)), dim=-1)
            out, hidden_state = self.rnn(x, hidden_state)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        return outputs.permute(1, 0, 2)

    def val(self, inputs, enc_outputs, hidden_state, mask):
        inputs = inputs.permute(1, 0, 2)
        query = torch.unsqueeze(hidden_state[1][-1], dim=0)
        # key & value size : source_len * batch_size * embedding_size
        context, _ = self.attention.forward(query=query, key=enc_outputs, value=enc_outputs, key_padding_mask=mask)
        x = torch.cat((context, inputs), dim=-1)
        out, hidden_state = self.rnn(x, hidden_state)
        return out.permute(1, 0, 2), hidden_state


class ATLAS(nn.Module):
    def __init__(self, embedding_size, hidden_size, global_vocab_size, assert_max_len, enc_dropout, dec_dropout):
        super(ATLAS, self).__init__()
        # hyper-parameters
        self.emb_dim = embedding_size
        self.hidden_size = hidden_size
        self.global_vocab_size = global_vocab_size
        self.dec_dropout = dec_dropout
        self.assert_max_len = assert_max_len
        self.enc_dropout = enc_dropout
        # Model
        self.embedding = CPUEmbedding(self.global_vocab_size, self.emb_dim)
        self.encoder = ATLASEncoder(self.emb_dim, self.hidden_size, dropout=self.enc_dropout)
        self.decoder = ATLASDecoder(self.emb_dim, self.hidden_size * 2, self.dec_dropout)
        # Pointer Net
        self.generator = CopyNet(self.emb_dim)
        self.loss = nn.CrossEntropyLoss()
        # self.dense = nn.Linear(self.hidden_size, self.assert_vocab_size)

    def forward(self, inputs, targets):
        encoder_inputs, encoder_masks, decoder_inputs, decoder_masks, vocab, vocab_mask, antimasks = inputs
        # Token embedding & Layer Norm
        encoder_inputs = self.embedding(encoder_inputs)
        decoder_inputs = self.embedding(decoder_inputs)
        # vocab_embedded = self.token_embedding(vocabs)  # batch_size * vocab_size * embed_size
        # Encoder
        encoder_output, hidden_state = self.encoder.forward(encoder_inputs)
        # Decoder
        dec_output = self.decoder.forward(decoder_inputs, encoder_output, hidden_state, encoder_masks == 0)
        # batch_size * context_seq_len * vocab_size
        vocab_embedding = self.embedding(vocab)
        tag_logits, _ = self.generator(vocab_embedding, dec_output)  # batch_size * context_seq_len * vocab_size
        vocab_mask = vocab_mask.repeat(1, tag_logits.size(1), 1)
        # using vocab mask to ignore impossible choices
        tag_logits = tag_logits.masked_fill(vocab_mask == 0, -1e9)

        # softmax
        probs = F.softmax(tag_logits, dim=-1)
        if not self.training:
            return probs
        loss = -torch.log(torch.gather(probs, -1, targets.unsqueeze(-1)).squeeze(-1))
        loss = loss.masked_fill(decoder_masks == 0, 0.0)
        resTruelen = torch.sum(decoder_masks, dim=-1).float()
        totalloss = torch.mean(loss, dim=-1) * self.assert_max_len / resTruelen
        return totalloss, probs

    def predict(self, inputs, targets, assert_vocab):
        encoder_inputs, encoder_masks, decoder_inputs, decoder_masks, vocab, vocab_mask, antimasks = inputs
        # Token embedding & Layer Norm
        encoder_inputs = self.embedding(encoder_inputs)
        vocab_embedding = self.embedding(vocab)
        encoder_output, hidden_state = self.encoder.forward(encoder_inputs)
        c_0 = hidden_state[1]
        c = torch.cat([c_0[0], c_0[1]], dim=1).unsqueeze(dim=0)
        c = torch.cat([c, c], dim=0)
        h = torch.zeros_like(c)
        dec_state = (h, c)
        dec_X = torch.unsqueeze(torch.tensor(
            [assert_vocab.token2id("<BOS>")], dtype=torch.long), dim=0)
        dec_X = dec_X.repeat(encoder_inputs.shape[0], 1)
        tag_logits = []
        for index in range(self.assert_max_len):
            dec_X = self.embedding(dec_X)
            Y, dec_state = self.decoder.val(dec_X, encoder_output, dec_state, encoder_masks == 0)
            tag_logit, _ = self.generator(vocab_embedding, Y)
            tag_logit = tag_logit.masked_fill(vocab_mask == 0, -1e9)
            dec_X = torch.gather(vocab, -1, tag_logit.argmax(dim=-1))
            tag_logits.append(tag_logit)
        tag_logits = torch.concat(tag_logits, dim=-1)
        probs = F.softmax(tag_logits, dim=-1)
        if not self.training:
            return probs
        loss = -torch.log(torch.gather(probs, -1, targets.unsqueeze(-1)).squeeze(-1))
        loss = loss.masked_fill(decoder_masks == 0, 0.0)
        resTruelen = torch.sum(decoder_masks, dim=-1).float()
        totalloss = torch.mean(loss, dim=-1) * self.assert_max_len / resTruelen
        return totalloss, probs

if __name__ == "__main__":
    model = ATLAS(embedding_size=512, hidden_size=256, global_vocab_size=70000,
                  assert_max_len=1000, enc_dropout=0.2, dec_dropout=0.2)
    model = model.to(device)
    print("Hello world")