import copy
from torch import nn
from math import sqrt
import torch
import warnings
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Modules.MATL import config
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput



def init_rnn_wt(rnn):
    for names in rnn._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(rnn, name)
                wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.init_normal_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.init_normal_std)


def init_wt_normal(wt):
    wt.data.normal_(std=config.init_normal_std)


def init_wt_uniform(wt):
    wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)

class Encoder(nn.Module):

    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        embedded = self.embedding(inputs)

        packed = pack_padded_sequence(embedded, seq_lens.cpu(), enforce_sorted=False)
        outputs, hidden = self.gru(packed)

        outputs, _ = pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        # use mask
        # attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention, V)
        return attention


class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query和key，第二个参数用于计算value
    """

    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size  # 输入维度
        self.all_head_size = all_head_size  # 输出维度
        self.num_heads = head_num  # 注意头的数量
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, 512)

        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size, self.all_head_size)
        print(self.linear_k, self.linear_q, self.linear_v)

    def forward(self, x, y, attention_mask):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q和k的输入，y作为v的输入
        """
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention_mask = attention_mask.eq(0)

        # import ipdb
        # ipdb.set_trace()

        attention = CalculateAttention()(q_s, k_s, v_s, attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output

__HEAD_MASK_WARNING_MSG = """The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

a =1

class ExtraT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config_, t5_model_path, model_file_path=None,is_eval=False):
        super().__init__(config_)
        self.config_ = config_
        self.ast_vocab_size = 24188
        self.is_eval = is_eval
        self.sent_dim = 128

        self.ast_encoder_tf_0 = Encoder(self.ast_vocab_size)
        self.ast_encoder_tf_1 = Encoder(self.ast_vocab_size)
        self.ast_encoder_tf_2 = Encoder(self.ast_vocab_size)
        self.ast_encoder_tf_3 = Encoder(self.ast_vocab_size)
        self.ast_encoder_tf_4 = Encoder(self.ast_vocab_size)

        self.ast_encoder_torch_0 = Encoder(self.ast_vocab_size)
        self.ast_encoder_torch_1 = Encoder(self.ast_vocab_size)
        self.ast_encoder_torch_2 = Encoder(self.ast_vocab_size)
        self.ast_encoder_torch_3 = Encoder(self.ast_vocab_size)
        self.ast_encoder_torch_4 = Encoder(self.ast_vocab_size)

        # self.sent_dim = config.hidden_size
        # self.atten_guide_torch = nn.Parameter(torch.Tensor(self.sent_dim).cuda())
        # self.atten_guide_torch.data.normal_(0, 1)
        self.atten = Multi_CrossAttention(self.sent_dim, self.sent_dim, 4)

        self.linear_before_decoder = nn.Linear(5 * 512, 511 * 512)
        init_linear_wt(self.linear_before_decoder)
        self.batch_norm_before_decoder = torch.nn.BatchNorm1d(512)


        device_ids = [0]
        # self.model = self.model.cuda()

        self.ast_encoder_tf_0 = torch.nn.DataParallel(self.ast_encoder_tf_0, device_ids=device_ids).cuda()
        self.ast_encoder_tf_1 = torch.nn.DataParallel(self.ast_encoder_tf_1, device_ids=device_ids).cuda()
        self.ast_encoder_tf_2 = torch.nn.DataParallel(self.ast_encoder_tf_2, device_ids=device_ids).cuda()
        self.ast_encoder_tf_3 = torch.nn.DataParallel(self.ast_encoder_tf_3, device_ids=device_ids).cuda()
        self.ast_encoder_tf_4 = torch.nn.DataParallel(self.ast_encoder_tf_4, device_ids=device_ids).cuda()

        self.ast_encoder_torch_0 = torch.nn.DataParallel(self.ast_encoder_torch_0, device_ids=device_ids).cuda()
        self.ast_encoder_torch_1 = torch.nn.DataParallel(self.ast_encoder_torch_1, device_ids=device_ids).cuda()
        self.ast_encoder_torch_2 = torch.nn.DataParallel(self.ast_encoder_torch_2, device_ids=device_ids).cuda()
        self.ast_encoder_torch_3 = torch.nn.DataParallel(self.ast_encoder_torch_3, device_ids=device_ids).cuda()
        self.ast_encoder_torch_4 = torch.nn.DataParallel(self.ast_encoder_torch_4, device_ids=device_ids).cuda()

        self.atten = torch.nn.DataParallel(self.atten, device_ids=device_ids).cuda()

        self.linear_before_decoder = torch.nn.DataParallel(self.linear_before_decoder, device_ids=device_ids).cuda()
        self.batch_norm_before_decoder = torch.nn.DataParallel(self.batch_norm_before_decoder, device_ids=device_ids).cuda()

        state_t5 = torch.load(t5_model_path)
        model_dict = self.state_dict()
        state_dict = {k: v for k, v in state_t5.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

        if model_file_path:
            state = torch.load(model_file_path, map_location={'cuda:1':'cuda:0'})
            self.set_state_dict(state)

        if is_eval:
            self.ast_encoder_tf_0.eval()
            self.ast_encoder_tf_1.eval()
            self.ast_encoder_tf_2.eval()
            self.ast_encoder_tf_3.eval()
            self.ast_encoder_tf_4.eval()

            self.ast_encoder_torch_0.eval()
            self.ast_encoder_torch_1.eval()
            self.ast_encoder_torch_2.eval()
            self.ast_encoder_torch_3.eval()
            self.ast_encoder_torch_4.eval()

            self.atten.eval()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        ast_batch_tf = input_ids['para_batch_1']
        ast_seq_lens_tf = input_ids['seq_len_batch_1']
        attention_mask_batch_tf =input_ids['para_mask_batch_1']

        ast_batch_torch =input_ids['para_batch_2']
        ast_seq_lens_torch =input_ids['seq_len_batch_2']
        attention_mask_batch_torch =input_ids['para_mask_batch_2']

        # attention_mask_batch_tf = torch.tensor(attention_mask_batch_tf, device='cuda').long()
        # attention_mask_batch_torch = torch.tensor(attention_mask_batch_torch, device='cuda').long()
        # import ipdb
        # ipdb.set_trace()

        ast_outputs_0_tf, ast_hidden_0_tf = self.ast_encoder_tf_0(ast_batch_tf[0], ast_seq_lens_tf[0])
        ast_outputs_1_tf, ast_hidden_1_tf = self.ast_encoder_tf_0(ast_batch_tf[1], ast_seq_lens_tf[1])
        ast_outputs_2_tf, ast_hidden_2_tf = self.ast_encoder_tf_0(ast_batch_tf[2], ast_seq_lens_tf[2])
        ast_outputs_3_tf, ast_hidden_3_tf = self.ast_encoder_tf_0(ast_batch_tf[3], ast_seq_lens_tf[3])
        ast_outputs_4_tf, ast_hidden_4_tf = self.ast_encoder_tf_0(ast_batch_tf[4], ast_seq_lens_tf[4])

        ast_outputs_tf = torch.cat(
            (ast_outputs_0_tf, ast_outputs_1_tf, ast_outputs_2_tf, ast_outputs_3_tf, ast_outputs_4_tf), 0)

        ast_hidden_0_tf = ast_hidden_0_tf[0] + ast_hidden_0_tf[1]  # [B, H]
        ast_hidden_1_tf = ast_hidden_1_tf[0] + ast_hidden_1_tf[1]  # [B, H]
        ast_hidden_2_tf = ast_hidden_2_tf[0] + ast_hidden_2_tf[1]  # [B, H]
        ast_hidden_3_tf = ast_hidden_3_tf[0] + ast_hidden_3_tf[1]  # [B, H]
        ast_hidden_4_tf = ast_hidden_4_tf[0] + ast_hidden_4_tf[1]  # [B, H]
        ast_hidden_tf = torch.stack(
            (ast_hidden_0_tf, ast_hidden_1_tf, ast_hidden_2_tf, ast_hidden_3_tf, ast_hidden_4_tf), dim=1)

        ast_outputs_0_torch, ast_hidden_0_torch = self.ast_encoder_torch_0(ast_batch_torch[0], ast_seq_lens_torch[0])
        ast_outputs_1_torch, ast_hidden_1_torch = self.ast_encoder_torch_0(ast_batch_torch[1], ast_seq_lens_torch[1])
        ast_outputs_2_torch, ast_hidden_2_torch = self.ast_encoder_torch_0(ast_batch_torch[2], ast_seq_lens_torch[2])
        ast_outputs_3_torch, ast_hidden_3_torch = self.ast_encoder_torch_0(ast_batch_torch[3], ast_seq_lens_torch[3])
        ast_outputs_4_torch, ast_hidden_4_torch = self.ast_encoder_torch_0(ast_batch_torch[4], ast_seq_lens_torch[4])

        ast_outputs_torch = torch.cat(
            (ast_outputs_0_torch, ast_outputs_1_torch, ast_outputs_2_torch, ast_outputs_3_torch, ast_outputs_4_torch),
            0)

        ast_hidden_0_torch = ast_hidden_0_torch[0] + ast_hidden_0_torch[1]  # [B, H]
        ast_hidden_1_torch = ast_hidden_1_torch[0] + ast_hidden_1_torch[1]  # [B, H]
        ast_hidden_2_torch = ast_hidden_2_torch[0] + ast_hidden_2_torch[1]  # [B, H]
        ast_hidden_3_torch = ast_hidden_3_torch[0] + ast_hidden_3_torch[1]  # [B, H]
        ast_hidden_4_torch = ast_hidden_4_torch[0] + ast_hidden_4_torch[1]  # [B, H]

        ast_hidden_torch = torch.stack(
            (ast_hidden_0_torch, ast_hidden_1_torch, ast_hidden_2_torch, ast_hidden_3_torch, ast_hidden_4_torch),
            dim=1)  # [B, 5, H]

        cross_output = self.atten(ast_hidden_tf, ast_hidden_torch, attention_mask_batch_torch)

        use_cache = use_cache if use_cache is not None else self.config_.use_cache
        return_dict = return_dict if return_dict is not None else self.config_.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config_.num_layers == self.config_.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids['context'],
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]


        # cross_output = cross_output.view(8, -1)
        # cross_output = self.linear_before_decoder(cross_output)
        # cross_output = cross_output.view(8, 511, 512)
        hidden_states_length = hidden_states.size(1)
        semantic_mean = torch.mean(cross_output, dim=1)
        semantic_hidden = semantic_mean.unsqueeze(1).repeat(1, hidden_states_length, 1)
        # cross_output = self.batch_norm_before_decoder(cross_output)

        #hidden_states = hidden_states + semantic_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config_.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def set_state_dict(self, state_dict):
        self.ast_encoder_tf_0.load_state_dict(state_dict["ast_encoder_0"])
        self.ast_encoder_tf_1.load_state_dict(state_dict["ast_encoder_1"])
        self.ast_encoder_tf_2.load_state_dict(state_dict["ast_encoder_2"])
        self.ast_encoder_tf_3.load_state_dict(state_dict["ast_encoder_3"])
        self.ast_encoder_tf_4.load_state_dict(state_dict["ast_encoder_4"])

        self.ast_encoder_torch_0.load_state_dict(state_dict["ast_encoder_0"])
        self.ast_encoder_torch_1.load_state_dict(state_dict["ast_encoder_1"])
        self.ast_encoder_torch_2.load_state_dict(state_dict["ast_encoder_2"])
        self.ast_encoder_torch_3.load_state_dict(state_dict["ast_encoder_3"])
        self.ast_encoder_torch_4.load_state_dict(state_dict["ast_encoder_4"])

