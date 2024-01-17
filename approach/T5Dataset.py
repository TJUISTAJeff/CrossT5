import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('.')
import pickle
import re
import torch
import dill
import torch.utils.data as data
from tqdm import tqdm
from utils import *
from Modules.MATL import config



the_ast_vocab = load_vocab_pk("ast_vocab_new.pk")
#print(11111111, len(the_ast_vocab))

def pad_seq(seq, maxlen):
    act_len = len(seq)
    if act_len < maxlen:
        seq = seq + [0] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq


def readpickle(filename, debug=False):
    data = []
    pbar = tqdm()
    with open(filename, 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
            if debug and len(data) > 1000:
                break
            pbar.update(1)
    pbar.close()
    return data

_PAD = '<PAD>'
_SOS = '<s>'    # start of sentence
_EOS = '</s>'   # end of sentence
_UNK = '<UNK>'  # OOV word

import itertools

def pad_one_batch(batch: list, vocab: Vocab) -> torch.Tensor:
    """
    pad batch using _PAD token and get the sequence lengths
    :param batch: one batch, [B, T]
    :param vocab: corresponding vocab
    :return:
    """
    if len(max(*batch, key=lambda v: len(v))) != 0:
        batch = list(itertools.zip_longest(*batch, fillvalue=vocab.word2index[_PAD]))
        batch = [list(b) for b in batch]
    else:
        batch[0] = [vocab.word2index[_PAD]]
        batch = list(itertools.zip_longest(*batch, fillvalue=vocab.word2index[_PAD]))
        batch = [list(b) for b in batch]
    use_cuda = torch.cuda.is_available()
    return torch.tensor(batch, device=torch.device('cuda' if use_cuda else 'cpu')).long()

def indices_from_batch_for_signature(batch: list, vocab: Vocab) -> list:
    """
    translate the word in batch to corresponding index by given vocab, then append the EOS token to each sentence
    :param batch: batch to be translated, [B, T]
    :param vocab: Vocab
    :return: translated batch, [B, T]
    """
    indices = []
    for line in batch:
        params_indices = []
        for param in line:
            indices_sentence = []
            for word in param:
                if word not in vocab.word2index:
                    indices_sentence.append(vocab.word2index[_UNK])
                else:
                    indices_sentence.append(vocab.word2index[word])
            indices_sentence.append(vocab.word2index[_EOS])
            params_indices.append(indices_sentence)
        indices.append(params_indices)

    result = []
    mask_for_attention = []
    result.append([])
    result.append([])
    result.append([])
    result.append([])
    result.append([])
    for line in indices:
        line_mask = []
        i = 0
        while i < 5:
            if i < len(line):
                result[i].append(line[i])
                line_mask.append(1)
            else:
                result[i].append([])
                line_mask.append(0)
            i += 1
        mask_for_attention.append(line_mask)

    return result, torch.tensor(mask_for_attention)

def get_seq_lens(batch: list) -> list:
    """
    get sequence lengths of given batch
    :param batch: [B, T]
    :return: sequence lengths
    """
    seq_lens = []
    for seq in batch:
        if len(seq) == 0:
            seq_lens.append(1)
        else:
            seq_lens.append(len(seq))
    return torch.tensor(seq_lens)

def rs_collate_fn(batch):
    global args
    rbatch = {}
    maxcontextlen = 0
    maxquerylen = 0
    bcontext = []
    bquery = []
    bres = []
    bpara_1 = []
    bpara_2 = []

    for k in (range(len(batch))):
        bcontext.append(batch[k]['context'])
        bquery.append(batch[k]['query'][:-1])
        bres.append(batch[k]['query'][1:])
        bpara_1.append(load_dataset_for_signature(batch[k]['para_1']))
        bpara_2.append(load_dataset_for_signature(batch[k]['para_2']))
        maxcontextlen = max(maxcontextlen, len(batch[k]['context']))
        maxquerylen = max(maxquerylen, len(batch[k]['query']) - 1)
    
    #import ipdb
    #ipdb.set_trace()
    
    ast_vocab = the_ast_vocab

    para_batch_1, attention_mask_batch_1 = indices_from_batch_for_signature(bpara_1, ast_vocab)  # [B, T]
    para_batch_2, attention_mask_batch_2 = indices_from_batch_for_signature(bpara_2, ast_vocab)  # [B, T]

    ast_seq_lens_tf = []
    ast_seq_lens_tf.append(get_seq_lens(para_batch_1[0]))
    ast_seq_lens_tf.append(get_seq_lens(para_batch_1[1]))
    ast_seq_lens_tf.append(get_seq_lens(para_batch_1[2]))
    ast_seq_lens_tf.append(get_seq_lens(para_batch_1[3]))
    ast_seq_lens_tf.append(get_seq_lens(para_batch_1[4]))

    ast_seq_lens_torch = []
    ast_seq_lens_torch.append(get_seq_lens(para_batch_2[0]))
    ast_seq_lens_torch.append(get_seq_lens(para_batch_2[1]))
    ast_seq_lens_torch.append(get_seq_lens(para_batch_2[2]))
    ast_seq_lens_torch.append(get_seq_lens(para_batch_2[3]))
    ast_seq_lens_torch.append(get_seq_lens(para_batch_2[4]))
        
    #import ipdb
    #ipdb.set_trace()

    para_batch_1[0] = (pad_one_batch(para_batch_1[0], ast_vocab))
    para_batch_1[1] = (pad_one_batch(para_batch_1[1], ast_vocab))
    para_batch_1[2] = (pad_one_batch(para_batch_1[2], ast_vocab))
    para_batch_1[3] = (pad_one_batch(para_batch_1[3], ast_vocab))
    para_batch_1[4] = (pad_one_batch(para_batch_1[4], ast_vocab))

    para_batch_2[0] = (pad_one_batch(para_batch_2[0], ast_vocab))
    para_batch_2[1] = (pad_one_batch(para_batch_2[1], ast_vocab))
    para_batch_2[2] = (pad_one_batch(para_batch_2[2], ast_vocab))
    para_batch_2[3] = (pad_one_batch(para_batch_2[3], ast_vocab))
    para_batch_2[4] = (pad_one_batch(para_batch_2[4], ast_vocab))

    maxcontextlen = min(maxcontextlen, args.context_max_len)
    maxquerylen = min(maxquerylen, args.query_max_len)
    for i in range(len(bcontext)):
        bcontext[i] = pad_seq(bcontext[i], maxcontextlen)
        bquery[i] = pad_seq(bquery[i], maxquerylen)
        bres[i] = pad_seq(bres[i], maxquerylen)
    rbatch['context'] = torch.tensor(bcontext)
    rbatch['query'] = torch.tensor(bquery)
    rbatch['res'] = torch.tensor(bres)
    
    rbatch['para_batch_1'] = (para_batch_1)
    rbatch['seq_len_batch_1'] = (ast_seq_lens_tf)
    rbatch['para_mask_batch_1'] = (attention_mask_batch_1)
    rbatch['para_batch_2'] = (para_batch_2)
    rbatch['seq_len_batch_2'] = (ast_seq_lens_torch)
    rbatch['para_mask_batch_2'] = (attention_mask_batch_2)
    #import ipdb
    #ipdb.set_trace()
    return rbatch


def atlas_rs_collate_fn(batch):
    global args
    rbatch = {}
    maxcontextlen = 0
    maxquerylen = 0
    bcontext = []
    bquery = []
    bres = []
    for k in (range(len(batch))):
        bcontext.append(batch[k][0])
        bquery.append(batch[k][1][:-1])
        bres.append(batch[k][1][1:])
        maxcontextlen = max(maxcontextlen, len(batch[k][0]))
        maxquerylen = max(maxquerylen, len(batch[k][1]) - 1)
    maxcontextlen = min(maxcontextlen, args.context_max_len)
    maxquerylen = min(maxquerylen, args.query_max_len)
    for i in range(len(bcontext)):
        bcontext[i] = pad_seq(bcontext[i], maxcontextlen)
        bquery[i] = pad_seq(bquery[i], maxquerylen)
        bres[i] = pad_seq(bres[i], maxquerylen)
    rbatch['context'] = torch.tensor(bcontext)
    rbatch['query'] = torch.tensor(bquery)
    rbatch['res'] = torch.tensor(bres)
    return rbatch

def load_dataset_for_signature(a_piece_if_api_para) -> list:
    """
    load the dataset from given path
    :param dataset_path: path of dataset
    :return: lines from the dataset
    """
    param_items = a_piece_if_api_para.strip().split('\n')
    one_api = []
    for param in param_items:
        if param != '':
            words = param.strip().split()
            one_api.append(words)

    return one_api

class T5Dataset(data.Dataset):
    def __init__(self, config, data):
        global args
        args = config
        self.data = data

    def __getitem__(self, offset):
        return self.data[offset]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    pass
