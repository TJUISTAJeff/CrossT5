import pickle

import torch
import torch.utils.data as data
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
tokenlen = len(tokenizer.get_vocab())
args = {}
voc = {}
PAD_token = tokenizer.pad_token_id


def pad_seq(seq, maxlen):
    act_len = len(seq)
    if act_len < maxlen:
        seq = seq + [0] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq


def pad_list(seq, maxlen1, maxlen2):
    if len(seq) < maxlen1:
        seq = seq + [[0] * maxlen2] * maxlen1
        seq = seq[:maxlen1]
    else:
        seq = seq[:maxlen1]
    return seq


def pad_list_1d(seq, maxlen1):
    if len(seq) < maxlen1:
        seq = seq + [0] * maxlen1
        seq = seq[:maxlen1]
    else:
        seq = seq[:maxlen1]
    return seq


# def readpickle(filename, debug=False):
#     data = []
#     pbar = tqdm()
#     with open(filename, 'rb') as f:
#         while True:
#             try:
#                 data.append(pickle.load(f))
#             except EOFError:
#                 break
#             if debug and len(data) > 100:
#                 break
#             pbar.update(1)
#     pbar.close()
#     return data

#TODO: 还是一个一个的pickle比较好，需要修改一下dataloader
def readpickle(filename, debug=False):
    data = []
    with open(filename, 'rb') as f:
        data=pickle.load(f)
        if debug:
            data = data[:100]
    return data

def rs_collate_fn(batch):
    global args
    rbatch = {}
    maxcontextlen = 0
    maxquerylen = 0
    maxvocablen = 0
    bcontext = []
    bquery = []
    blocalvocab = []
    bres = []

    for k in (range(len(batch))):
        bcontext.append(batch[k].context_tokens)
        bquery.append(batch[k].query_tokens[:-1])
        blocalvocab.append(batch[k].local_vocab)
        bres.append(batch[k].assertion[1:])
        maxcontextlen = max(maxcontextlen, len(batch[k].context_tokens))
        maxquerylen = max(maxquerylen, len(batch[k].query_tokens) - 1)
        maxvocablen = max(maxvocablen, len(batch[k].local_vocab))

    maxcontextlen = min(maxcontextlen, args.context_max_len)
    maxquerylen = min(maxquerylen, args.query_max_len)
    for i in range(len(bcontext)):
        bcontext[i] = pad_list(bcontext[i], maxcontextlen, args.char_seq_max_len)
        bquery[i] = pad_list(bquery[i], maxquerylen, args.char_seq_max_len)
        blocalvocab[i] = pad_list(blocalvocab[i], maxvocablen, args.char_seq_max_len)
        bres[i] = pad_seq(bres[i], maxquerylen)
    rbatch['context'] = torch.tensor(bcontext)
    rbatch['query'] = torch.tensor(bquery)
    rbatch['vocab'] = torch.tensor(blocalvocab)
    rbatch['res'] = torch.tensor(bres)
    return rbatch


def rs_collate_fn_t5(batch):
    global args
    rbatch = {}
    maxcontextlen = 0
    maxquerylen = 0
    maxvocablen = 0
    bcontext = []
    bquery = []
    blocalvocab = []
    bres = []

    for k in (range(len(batch))):
        bcontext.append(batch[k].context_tokens)
        bquery.append(batch[k].query_tokens[:-1])
        blocalvocab.append(batch[k].local_vocab)
        bres.append(batch[k].assertion[1:])
        maxcontextlen = max(maxcontextlen, len(batch[k].context_tokens))
        maxquerylen = max(maxquerylen, len(batch[k].query_tokens) - 1)
        maxvocablen = max(maxvocablen, len(batch[k].local_vocab))

    maxcontextlen = min(maxcontextlen, args.context_max_len)
    maxquerylen = min(maxquerylen, args.query_max_len)
    for i in range(len(bcontext)):
        bcontext[i] = pad_list_1d(bcontext[i], maxcontextlen)
        bquery[i] = pad_list(bquery[i], maxquerylen, args.char_seq_max_len)
        blocalvocab[i] = pad_list(blocalvocab[i], maxvocablen, args.char_seq_max_len)
        bres[i] = pad_seq(bres[i], maxquerylen)
    rbatch['context'] = torch.tensor(bcontext)
    rbatch['query'] = torch.tensor(bquery)
    rbatch['vocab'] = torch.tensor(blocalvocab)
    rbatch['res'] = torch.tensor(bres)
    return rbatch

def rs_collate_fn_our_data(batch):
    global args
    rbatch = {}
    maxcontextlen = 0
    maxquerylen = 0
    maxvocablen = 0
    bcontext = []
    bquery = []
    blocalvocab = []
    bres = []

    for k in (range(len(batch))):
        bcontext.append(batch[k]['context'])
        bquery.append(batch[k]['query'][:-1])
        blocalvocab.append(batch[k]['vocab'])
        bres.append(batch[k]['res'][1:])
        maxcontextlen = max(maxcontextlen, len(batch[k]['context']))
        maxquerylen = max(maxquerylen, len(batch[k]['query']) - 1)
        maxvocablen = max(maxvocablen, len(batch[k]['vocab']))

    maxcontextlen = min(maxcontextlen, args.context_max_len)
    maxquerylen = min(maxquerylen, args.query_max_len)
    for i in range(len(bcontext)):
        bcontext[i] = pad_seq(bcontext[i], maxcontextlen)
        bquery[i] = pad_list(bquery[i], maxquerylen, args.char_seq_max_len)
        blocalvocab[i] = pad_list(blocalvocab[i], maxvocablen, args.char_seq_max_len)
        bres[i] = pad_seq(bres[i], maxquerylen)
    rbatch['context'] = torch.tensor(bcontext)
    rbatch['query'] = torch.tensor(bquery)
    rbatch['vocab'] = torch.tensor(blocalvocab)
    rbatch['res'] = torch.tensor(bres)
    return rbatch

class SumDataset(data.Dataset):
    def __init__(self, config, data):
        global args
        args = config
        self.data = data

    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_list(self, seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def __getitem__(self, offset):
        return self.data[offset]

    def __len__(self):
        return len(self.data)
