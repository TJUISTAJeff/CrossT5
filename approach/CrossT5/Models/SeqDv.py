from CONSTANTS import *
from Modules.PointerNet import CopyNet
from Modules.TokenEmbeddding import TokenEmbedding
from Modules.transformer.LayerNorm import LayerNorm
from Modules.transformer.PositionalEncoding import PositionalEmbedding


class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # embedded = self.dropout(self.embedding(src))
        embedded = self.dropout(src)
        embedded = embedded.permute(1, 0, 2)
        src_len = src_len.shape[1] - src_len.sum(-1)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
        packed_outputs, (hidden, c) = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, total_length=src.shape[1])
        # hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        c = torch.tanh(self.fc(torch.cat((c[-2, :, :], c[-1, :, :]), dim=1)))
        return outputs, c


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask, -1e10)
        return F.softmax(attention, dim=1)


class CopyNetDecoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        # self.output_dim = output_dim
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=2)
        # self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # self.copy_mask = None
        # self.log_softmax = nn.LogSoftmax(dim=1)
        # self.src_pad_idx = src_pad_idx

    def __decoder_step(self, embedded, hidden, c, encoder_outputs, mask):
        attentive_weights = self.attention(hidden[-1], encoder_outputs, mask)
        attentive_weights = attentive_weights.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attentive_read = torch.bmm(attentive_weights, encoder_outputs)
        attentive_read = attentive_read.permute(1, 0, 2)
        rnn_input = torch.cat((
            embedded,
            attentive_read), dim=2)
        output, (hidden, c) = self.rnn(rnn_input, (hidden, c))
        assert (output == hidden[-1]).all()
        return output, hidden, c

    def forward(self, trg, c, encoder_outputs, mask):
        trg = trg.permute(1, 0, 2)
        # mask = self.create_mask(src)
        c = torch.cat([c.unsqueeze(0), c.unsqueeze(0)], dim=0)
        hidden = torch.zeros_like(c)
        outputs = []
        trg_len = trg.shape[0]
        for t in range(0, trg_len):
            input = trg[t]
            input = input.unsqueeze(0)
            embedded = self.dropout(input)
            output, hidden, c = self.__decoder_step(
                embedded,
                hidden,
                c,
                encoder_outputs,
                mask
            )
            outputs.append(output.squeeze(0))
        outputs = torch.stack(outputs, dim=0)
        return outputs.permute(1, 0, 2)


class SeqDv(nn.Module):
    def __init__(self, config):
        super(SeqDv, self).__init__()
        # hyper-parameters
        self.emb_dim = config.word_dims
        # self.feed_forward_hidden = 4 * self.emb_dim
        self.char_vocab_size = config.char_vocab_size
        # self.num_heads = config.num_heads
        self.input_dropout = config.input_dropout
        self.query_max_len = config.char_seq_max_len
        # Modules
        self.token_embedding = TokenEmbedding(self.char_vocab_size, self.query_max_len, self.emb_dim)
        # self.query_pe = PositionalEmbedding(self.emb_dim, 'query_pe', max_len=config.query_max_len)
        # self.context_pe = PositionalEmbedding(self.emb_dim, 'context_pe', max_len=config.context_max_len)
        # self.layer_norm = LayerNorm(self.emb_dim)
        # self.dropout = nn.Dropout(p=self.input_dropout)
        self.encoder = Encoder(self.emb_dim, self.emb_dim * 2, self.emb_dim, self.input_dropout)
        self.decoder = CopyNetDecoder(self.emb_dim, self.emb_dim * 2, self.emb_dim, self.input_dropout,)
        # Pointer Net
        self.generator = CopyNet(self.emb_dim)
        # self.loss = nn.CrossEntropyLoss()

    def getAntiMask(self, inputrule):
        batch_size, seq_length = inputrule.size()
        seq_ids = torch.arange(seq_length, dtype=torch.long, device=inputrule.device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        return ~causal_mask[0]

    def forward(self, inputs, train=False):
        context_inputs = inputs['context']
        query_inputs = inputs['query']
        vocabs = inputs['vocab']
        targets = None
        context = context_inputs.sum(dim=-1)
        context_masks = context.eq(0)
        queries = query_inputs.sum(dim=-1)
        query_masks = queries.eq(0)
        vocab = vocabs.sum(dim=-1)
        vocab_masks = vocab.eq(0)
        # antimasks = self.getAntiMask(queries)

        # batch_size = context_inputs.shape[0]
        # Token embedding & Layer Norm
        context_inputs = self.token_embedding(context_inputs)
        query_inputs = self.token_embedding(query_inputs)  # + self.query_pe(query_inputs)

        vocab_embedded = self.token_embedding(vocabs)  # batch_size * vocab_size * embed_size

        # Encoder
        encoder_outputs, c = self.encoder(context_inputs, context_masks)
        # Decoder
        decoder_outputs = self.decoder(query_inputs, c, encoder_outputs, context_masks)
        # TODO: 感觉应该加一个encoded output的mask，因为不是每个query都是那么长的，不过感觉影响不太大？

        # Pointer net
        # vocab_embedded: batch_size * vocab_size * embed_dim
        # encoded_output : batch_size * context_seq_len * embed_dim

        tag_logits, _ = self.generator(vocab_embedded, decoder_outputs)  # batch_size * context_seq_len * vocab_size
        vocab_masks = vocab_masks.unsqueeze(1).repeat(1, tag_logits.size(1), 1)
        # using vocab mask to ignore impossible choices
        tag_logits = tag_logits.masked_fill(vocab_masks, -1e9)
        # softmax
        if 'res' in inputs:
            targets = inputs['res']
        if not self.training and targets is None:
            return F.softmax(tag_logits, dim=-1)
        probs = F.softmax(tag_logits, dim=-1)
        loss = -torch.log(torch.gather(probs, -1, targets.unsqueeze(-1)).squeeze(-1))
        loss = loss.masked_fill(query_masks, 0.0)
        if train:
            return loss
        return loss, probs
