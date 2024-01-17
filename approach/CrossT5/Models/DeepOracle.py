from CONSTANTS import *
from Modules.PointerNet import CopyNet
from Modules.TokenEmbeddding import TokenEmbedding
from Modules.transformer.DenseLayer import DenseLayer
from Modules.transformer.LayerNorm import LayerNorm
from Modules.transformer.PositionalEncoding import PositionalEmbedding
from Modules.transformer.SubLayerConnection import SublayerConnection


class EncodeTransformerBlock(nn.Module):
    '''
    Self Attention encoder for contextual inputs
    '''

    def __init__(self, hidden_dim, num_heads, feed_forward_hidden, dropout=0.1):
        super(EncodeTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(num_heads=num_heads, embed_dim=hidden_dim, batch_first=True,
                                               dropout=dropout)
        # self.attention = MultiHeadedAttention(h=num_heads, d_model=hidden_dim)
        self.feed_forward = DenseLayer(d_model=hidden_dim, d_ff=feed_forward_hidden, dropout=dropout)
        self.sublayer = SublayerConnection(size=hidden_dim, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden_dim, dropout=dropout)

    def forward(self, inputs, mask):
        # Self Attention + Add&Norm
        outputs = self.sublayer(inputs,
                                lambda _x: self.attention.forward(query=_x, key=_x, value=_x, key_padding_mask=mask,
                                                                  need_weights=False))
        # Feed Forward + Add&Norm
        outputs = self.sublayer2(outputs, self.feed_forward)
        return outputs


class DecoderTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, feed_forward_hidden, dropout=0.1):
        super(DecoderTransformerBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(num_heads=num_heads, embed_dim=hidden_dim, batch_first=True,
                                                    dropout=dropout)
        self.en_in_attention = nn.MultiheadAttention(num_heads=num_heads, embed_dim=hidden_dim, batch_first=True,
                                                     dropout=dropout)
        self.feed_forward = DenseLayer(d_model=hidden_dim, d_ff=feed_forward_hidden, dropout=dropout)
        self.sublayer1 = SublayerConnection(size=hidden_dim, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden_dim, dropout=dropout)
        self.sublayer3 = SublayerConnection(size=hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, input_query_masks, anti_masks, context_encode, context_mask):
        # Self-Attention + Add&Norm
        outputs = self.sublayer1(queries, lambda _x: self.self_attention.forward(query=_x, key=_x, value=_x,
                                                                                 key_padding_mask=input_query_masks,
                                                                                 need_weights=False,
                                                                                 attn_mask=anti_masks))
        # Attention with encoder outputs + Add&Norm
        outputs = self.sublayer2(outputs, lambda _x: self.en_in_attention.forward(query=_x, key=context_encode,
                                                                                  value=context_encode,
                                                                                  key_padding_mask=context_mask,
                                                                                  need_weights=False))
        # Feed forward + Add&Norm
        outputs = self.sublayer3(outputs, self.feed_forward)
        return self.dropout(outputs)


class DeepOracle(nn.Module):
    def __init__(self, config):
        super(DeepOracle, self).__init__()
        # hyper-parameters
        self.emb_dim = config.word_dims
        self.feed_forward_hidden = 4 * self.emb_dim
        self.encoder_layers = config.encoder_layers
        self.decoder_layers = config.decoder_layers
        self.char_vocab_size = config.char_vocab_size
        self.num_heads = config.num_heads
        self.input_dropout = config.input_dropout
        self.query_max_len = config.char_seq_max_len
        # Modules
        self.token_embedding = TokenEmbedding(self.char_vocab_size, config.char_seq_max_len, self.emb_dim)
        self.query_pe = PositionalEmbedding(self.emb_dim, 'query_pe', max_len=config.query_max_len)
        self.context_pe = PositionalEmbedding(self.emb_dim, 'context_pe', max_len=config.context_max_len)
        self.layer_norm = LayerNorm(self.emb_dim)
        self.dropout = nn.Dropout(p=self.input_dropout)
        self.encoder = nn.ModuleList(
            [EncodeTransformerBlock(self.emb_dim, self.num_heads, self.feed_forward_hidden, self.input_dropout)
             for _ in range(self.encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderTransformerBlock(self.emb_dim, self.num_heads, self.feed_forward_hidden, self.input_dropout)
             for _ in range(self.decoder_layers)]
        )
        # Final MLP Layer for classification
        #self.finalLinear = DenseLayer(self.emb_dim, self.feed_forward_hidden)
        # Pointer Net
        self.generator = CopyNet(self.emb_dim)
        self.loss = nn.CrossEntropyLoss()
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
        context =  context_inputs.sum(dim=-1)
        context_masks = context.eq(0)
        queries = query_inputs.sum(dim=-1)
        query_masks = queries.eq(0)
        vocab = vocabs.sum(dim=-1)
        vocab_masks = vocab.eq(0)
        antimasks = self.getAntiMask(queries)

        batch_size = context_inputs.shape[0]
        # Token embedding & Layer Norm
        context_inputs = self.token_embedding(context_inputs)
        context_inputs = self.layer_norm(context_inputs)  # batch_size * context_len(1024) * emb_size

        # Positional Embedding & Dropoout
        context_inputs = context_inputs + self.context_pe(context_inputs)
        context_inputs = self.dropout(context_inputs)

        query_inputs = self.token_embedding(query_inputs) + self.query_pe(query_inputs)
        context_inputs = self.dropout(context_inputs)

        vocab_embedded = self.token_embedding(vocabs)  # batch_size * vocab_size * embed_size

        # Encoder
        context_encoder_output = context_inputs
        for layer in self.encoder:
            context_encoder_output = layer.forward(context_encoder_output, context_masks)

        # Decoder
        encoded_output = query_inputs
        for layer in self.decoder:
            encoded_output = layer.forward(encoded_output, query_masks, antimasks, context_encoder_output,
                                           context_masks)

        #TODO: 感觉应该加一个encoded output的mask，因为不是每个query都是那么长的，不过感觉影响不太大？


        # Pointer net
        # vocab_embedded: batch_size * vocab_size * embed_dim
        # encoded_output : batch_size * context_seq_len * embed_dim

        tag_logits, _ = self.generator(vocab_embedded, encoded_output)  # batch_size * context_seq_len * vocab_size
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
