import operator
import os
import json
import io
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

# from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
# from torchtext.data import Field, BucketIterator, TabularDataset
# from torchtext import data

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
import json

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='ISO-8859-1')
# dataset_path = "/data/Dataset"
dataset_path = "F:/Asuna/Desktop/Dataset"
train_path = "Training"
eval_path = "Eval"
test_path = "Testing"
method_path = "testMethods.txt"
assert_path = "assertLines.txt"
train_json_path = "train.json"
eval_json_path = "valid.json"
test_json_path = "test.json"
wandb_flag = False
train_model = False
eval_model = True


def to_json(method_path, assertion_path, keep_path, max_assert_len=100):
    data = []
    # print("Hello world")
    with open(method_path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
    assertion = []
    with open(assertion_path, 'r', encoding="ISO-8859-1") as f:
        assertion = f.readlines()
    dataset = []
    # print("Hello world")
    for x, y in zip(data, assertion):
        src =  [i.strip() for i in x.strip().split()]
        trg = [i.strip() for i in y.strip().split()]
        if len(trg)>max_assert_len:
            continue
        data_ = {'src': src, 'trg': trg}
        dataset.append(data_)
    # print(dataset)
    # print("Hello world")
    if not os.path.exists(keep_path):
        with open(keep_path, 'w', encoding="ISO-8859-1") as f:
            print(f"Create keep file: {keep_path}")
    with open(keep_path, 'a+', encoding="ISO-8859-1") as f:
        f.seek(0)
        f.truncate()
        for i in dataset:
            json.dump(i, f)
            f.write("\n")



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'))
        packed_outputs, (hidden, c) = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        c = torch.tanh(self.fc(torch.cat((c[-2, :, :], c[-1, :, :]), dim=1)))
        return outputs, hidden, c


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)


class CopyNetDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                 dropout, attention, src_vocab, trg_vocab, cpy_vocab):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.out_len = len(cpy_vocab)
        self.embedding = nn.Embedding(self.out_len, emb_dim)
        self.rnn = nn.LSTM(2 * (enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=2)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.copy_out = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.trg_to_cpy = torch.zeros((len(trg_vocab),), dtype=torch.long, device=device)
        self.src_to_cpy = torch.zeros((len(src_vocab),), dtype=torch.long, device=device)
        for i in range(len(trg_vocab)):
            self.trg_to_cpy[i] = cpy_vocab.stoi[trg_vocab.itos[i]]
        for i in range(len(src_vocab)):
            self.src_to_cpy[i] = cpy_vocab.stoi[src_vocab.itos[i]]
        self.selective_weights = None
        self.copy_mask = None
        self.log_softmax = nn.LogSoftmax(dim=1)

    def __decoder_step(self, input, embedded, hidden, c, encoder_outputs, mask):

        attentive_weights = self.attention(hidden[-1], encoder_outputs, mask)
        attentive_weights = attentive_weights.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attentive_read = torch.bmm(attentive_weights, encoder_outputs)
        attentive_read = attentive_read.permute(1, 0, 2)
        self.selective_weights = self.selective_weights.masked_fill(mask[:, 1:-1] == 0, -1e10)
        self.selective_weights = self.selective_weights.unsqueeze(1)
        selective_read = torch.bmm(self.selective_weights, encoder_outputs[:, 1:-1, :])
        selective_read = selective_read.permute(1, 0, 2)
        rnn_input = torch.cat((
            embedded,
            attentive_read,
            selective_read), dim=2)
        output, (hidden, c) = self.rnn(rnn_input, (hidden, c))
        assert (output == hidden[-1]).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        attentive_read = attentive_read.squeeze(0)

        generation_scores = self.fc_out(torch.cat((
            output,
            attentive_read,
            embedded), dim=1))
        return generation_scores, hidden, c, attentive_weights.squeeze(1)

    def __copy_scores(self, hidden, encoder_outputs, mask):
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        trimmed_encoder_outputs = encoder_outputs[:, 1:-1, :]
        copy_projection = self.copy_out(trimmed_encoder_outputs)
        copy_projection = torch.tanh(copy_projection)
        copy_scores = copy_projection.bmm(hidden.unsqueeze(-1)).squeeze(-1)
        copy_scores = copy_scores.masked_fill(mask[:, 1:-1] == 0, -1e10)
        return copy_scores

    def init_selective_weights(self, batch_size, trimmed_src_len, hidden):
        self.selective_weights = hidden.new_zeros(size=(batch_size, trimmed_src_len))
        self.copy_mask = hidden.new_zeros(size=(batch_size, trimmed_src_len))

    def update_selective_weights(self, src, prev_pred):
        # src = [src len, batch size]
        converted_src = self.src_to_cpy[src[1:-1, :]]
        self.copy_mask = converted_src.eq(prev_pred.squeeze(0)).permute(1, 0)
        # print(self.selective_weights.size())
        self.selective_weights = self.selective_weights.masked_fill(self.copy_mask == 0, -1e10)
        self.selective_weights = F.softmax(self.selective_weights, dim=1)

    def forward(self, src, input, hidden, c, encoder_outputs, mask):

        assert (self.selective_weights != None)
        source_length = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        # mask = [batch size, src len]
        input = input.unsqueeze(0)
        self.update_selective_weights(src, input)
        embedded = self.dropout(self.embedding(input))
        generation_scores, hidden, c, attentive_weights = self.__decoder_step(
            input,
            embedded,
            hidden,
            c,
            encoder_outputs,
            mask
        )
        copy_scores = self.__copy_scores(
            hidden[-1],
            encoder_outputs,
            mask
        )
        self.selective_weights = copy_scores

        batch_row = torch.LongTensor(list(range(batch_size))).view(-1, 1)
        src = src.permute(1, 0)
        src = self.src_to_cpy[src[:, 1:-1]]
        predictions = generation_scores.new_zeros(batch_size, self.out_len)
        predictions[:, :self.output_dim] = generation_scores
        predictions[batch_row, src] = copy_scores
        return predictions, hidden, c, attentive_weights, self.selective_weights


def vec(l):
    t = torch.stack(tuple(l), -1)
    return t.view(-1)


def unravel_topk_indices(idxs, rows):
    r = torch.clone(idxs)
    for i in range(rows):
        r[i, :] = i
    return (r, idxs)


class CopyNetSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        source_length = src.shape[0]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, self.decoder.out_len).to(device)
        encoder_outputs, hidden, c = self.encoder(src, src_len)
        self.decoder.init_selective_weights(batch_size, source_length - 2, hidden)
        input = trg[0, :]
        mask = self.create_mask(src)
        c = torch.cat([c.unsqueeze(0), c.unsqueeze(0)], dim=0)
        hidden = torch.zeros_like(c)
        for t in range(1, trg_len):
            output, hidden, c, _, _ = self.decoder(src, input, hidden, c, encoder_outputs, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src, src_len = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, src_len, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, beam_search=False, beam_width=10):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg
            output = model(src, src_len, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    # if isinstance(sentence, str):
    #     nlp = spacy.load('de')
    #     tokens = [token.text.lower() for token in nlp(sentence)]
    # else:
    #     tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + sentence + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])

    with torch.no_grad():
        encoder_outputs, hidden, c = model.encoder(src_tensor, src_len)
        model.decoder.init_selective_weights(1, src_len - 2, hidden)
    mask = model.create_mask(src_tensor)
    c = torch.cat([c.unsqueeze(0), c.unsqueeze(0)], dim=0)
    hidden = torch.zeros_like(c)
    # hidden = torch.cat([hidden.unsqueeze(0), hidden.unsqueeze(0)], dim=0)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    select_attns = torch.zeros(max_len, 1, len(src_indexes) - 2).to(device)

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, c, attention, select_attn = model.decoder(src_tensor,
                                                                   trg_tensor,
                                                                   hidden,
                                                                   c,
                                                                   encoder_outputs,
                                                                   mask)
            output = F.log_softmax(output, dim=1)

            attentions[i] = attention
            select_attns[i, :] = select_attn

            pred_token = output.argmax(1).item()

            trg_indexes.append(pred_token)

            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break
    try:
        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    except:
        print(f"out of index:{sentence} {trg_indexes}")
        return [], [], []
    return trg_tokens[1:], attentions[:len(trg_tokens) - 1], select_attns[:len(trg_tokens) - 2]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    # print("Hello world")
    # to_json(
    #     os.path.join(dataset_path, train_path, method_path),
    #     os.path.join(dataset_path, train_path, assert_path),
    #     os.path.join(dataset_path, train_path, train_json_path)
    # )
    # to_json(
    #     os.path.join(dataset_path, eval_path, method_path),
    #     os.path.join(dataset_path, eval_path, assert_path),
    #     os.path.join(dataset_path, eval_path, eval_json_path)
    # )
    # to_json(
    #     os.path.join(dataset_path, test_path, method_path),
    #     os.path.join(dataset_path, test_path, assert_path),
    #     os.path.join(dataset_path, test_path, test_json_path)
    # )
    if wandb_flag:
        wandb.init(project="copynet_sampled")
    INIT = True
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    SRC = Field(
        #   tokenize = tokenize_de,
        init_token='<sos>',
        eos_token='<eos>',
        include_lengths=True
    )

    TRG = Field(
        # tokenize = tokenize_en,
        init_token='<sos>',
        eos_token='<eos>',
    )

    CPY = Field(
        init_token='<sos>',
        eos_token='<eos>',
    )

    fields = {'src': ('src', SRC), 'trg': ('trg', TRG)}

    train_data, valid_data, test_data = TabularDataset.splits(
        path=os.path.join(dataset_path),
        train=os.path.join(train_path, train_json_path),
        validation=os.path.join(eval_path, eval_json_path),
        test=os.path.join(test_path, test_json_path),
        format='json',
        fields=fields
    )

    SRC.build_vocab(train_data, min_freq=10)
    TRG.build_vocab(train_data, max_size=1000)
    CPY.build_vocab(train_data.trg, max_size=1000)
    CPY.vocab.extend(SRC.vocab)

    # train_data, valid_data, test_data = TabularDataset.splits(
    #     path = 'F:/Asuna/Desktop/Copynet-main',
    #     train = 'train.json',
    #     validation = 'valid.json',
    #     test = 'test.json',
    #     format = 'json',
    #     fields = fields
    # )

    # SRC.build_vocab(train_data, min_freq = 2)
    # TRG.build_vocab(train_data, min_freq = 2)
    # CPY.build_vocab(train_data.trg, min_freq=2)
    # CPY.vocab.extend(SRC.vocab)

    print("SRC vocab contains %d tokens" % len(SRC.vocab))
    print("TRG vocab contains %d tokens" % len(TRG.vocab))
    print("COPY_TRG vocab contains %d tokens" % len(CPY.vocab))

    # BATCH_SIZE = 2
    BATCH_SIZE = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device)

    if train_model:
        EOS_IDX = SRC.vocab.stoi[SRC.eos_token]
        SOS_IDX = SRC.vocab.stoi[SRC.init_token]
        TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
        TRG_SOS = TRG.eos_token
        INPUT_DIM = len(SRC.vocab)
        OUTPUT_DIM = len(TRG.vocab)
        COPY_OUT_DIM = len(CPY.vocab)
        ENC_EMB_DIM = 512
        DEC_EMB_DIM = 512
        ENC_HID_DIM = 256
        DEC_HID_DIM = 256
        ENC_DROPOUT = 0.2
        DEC_DROPOUT = 0.2
        SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = CopyNetDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM,
                             DEC_DROPOUT, attn, SRC.vocab, TRG.vocab, CPY.vocab)

        model = CopyNetSeq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

        model.apply(init_weights)

        print(f'The model has {count_parameters(model):,} trainable parameters')

        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

        N_EPOCHS = 1000
        CLIP = 1

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'tut4-model.pt')
            if wandb_flag: wandb.log({"Train Loss": train_loss})
            if wandb_flag: wandb.log({"Train PPL": math.exp(train_loss)})
            if wandb_flag: wandb.log({"Val. Loss": valid_loss})
            if wandb_flag: wandb.log({"Val. PPL": math.exp(valid_loss)})
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

            if epoch % 20 == 0:
                count = 0
                total_num = 0
                # start_time = time.time()
                for i in test_data.examples:
                    total_num += 1
                    src = vars(i)['src']
                    trg = vars(i)['trg']
                    translation, attention, s_attn = translate_sentence(src, SRC, CPY, model, device, max_len=100)
                    if len(translation) > 0 and translation[-1] == TRG_SOS:
                        translation = translation[:-1]
                    if operator.eq(trg, translation):
                        count += 1
                    # if total_num%100==0:
                    #     print(f"Finish {total_num} test case.")
                # end_time = time.time()
                # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                # print(f'Time: {epoch_mins}m {epoch_secs}s')
                if wandb_flag: wandb.log({"test acc": (count / total_num) * 100})
                test_loss = evaluate(model, test_iterator, criterion)
                if wandb_flag: wandb.log({"Test. Loss": test_loss})
                if wandb_flag: wandb.log({"Test. PPL": math.exp(test_loss)})

        model.load_state_dict(torch.load('tut4-model.pt'))

        test_loss = evaluate(model, test_iterator, criterion)

        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    if eval_model:
        EOS_IDX = SRC.vocab.stoi[SRC.eos_token]
        SOS_IDX = SRC.vocab.stoi[SRC.init_token]
        TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
        TRG_SOS = TRG.eos_token
        INPUT_DIM = len(SRC.vocab)
        OUTPUT_DIM = len(TRG.vocab)
        COPY_OUT_DIM = len(CPY.vocab)
        ENC_EMB_DIM = 512
        DEC_EMB_DIM = 512
        ENC_HID_DIM = 256
        DEC_HID_DIM = 256
        ENC_DROPOUT = 0.2
        DEC_DROPOUT = 0.2
        SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = CopyNetDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM,
                             DEC_DROPOUT, attn, SRC.vocab, TRG.vocab, CPY.vocab)

        model = CopyNetSeq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)
        model.load_state_dict(torch.load(os.path.join(dataset_path, 'tut4-model.pt'), map_location=device))
        count = 0
        total_num = 0
        start_time = time.time()
        print("begin test!")
        for i in test_data.examples:
            total_num += 1
            src = vars(i)['src']
            trg = vars(i)['trg']
            translation, attention, s_attn = translate_sentence(src, SRC, CPY, model, device, max_len=100)
            if len(translation) > 0 and translation[-1] == TRG_SOS:
                translation = translation[:-1]
            if operator.eq(trg, translation):
                count += 1
            if total_num % 100 == 0:
                print(f"Finish {total_num} test case.")
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Time: {epoch_mins}m {epoch_secs}s')
        print(f"acc: {(count / total_num) * 100:7.2f}%")