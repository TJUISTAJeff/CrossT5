import torch
from sklearn.metrics import accuracy_score
import numpy as np
from entities.instance import TensorInstance


def getAntiMask(size):
    ans = np.zeros([size, size])
    for i in range(size):
        for j in range(0, i + 1):
            ans[i, j] = 1.0
    return ans


def generate_batched_tensor(batched_instances, config, char_vocab, decode=False):
    batch_size = len(batched_instances)
    char_seq_len = config.char_seq_max_len
    inputs_seq_len = config.context_max_len
    queries_seq_len = config.query_max_len
    vocab_len = 0
    for inst in batched_instances:
        vocab_len = inst.vocab_len if inst.vocab_len > vocab_len else vocab_len

    tinst = TensorInstance(batch_size, inputs_seq_len, queries_seq_len, vocab_len, char_seq_len)

    for batch_id, inst in enumerate(batched_instances):
        for seq_id, token in enumerate(inst.context_tokens):
            if seq_id >= inputs_seq_len:
                break
            for char_seq_id, char in enumerate(token):
                if char_seq_id >= char_seq_len:
                    break
                char_id = char_vocab.char2id(char)
                tinst.input_context_char_seq[batch_id][seq_id][char_seq_id] = char_id

            tinst.input_context_mask[batch_id][seq_id] = 0
        if decode:
            r = len(inst.assertion)
        else:
            r = len(inst.assertion) - 1
        for seq_id in range(r):
            # the last token is <EOS>
            if seq_id >= queries_seq_len:
                break
            token = inst.assertion[seq_id]
            for char_seq_id, char in enumerate(token):
                if char_seq_id >= char_seq_len:
                    break
                char_id = char_vocab.char2id(char)
                tinst.input_queries_char_seq[batch_id][seq_id][char_seq_id] = char_id
            tinst.input_queries_mask[batch_id][seq_id] = 1
            if not decode: tinst.targets[batch_id][seq_id] = inst.local_vocab.token2id(inst.assertion[seq_id + 1])

        for seq_id, token in enumerate(inst.local_vocab.tokens):
            for char_seq_id, char in enumerate(token):
                if char_seq_id >= char_seq_len:
                    break
                char_id = char_vocab.char2id(char)
                tinst.input_vocab_char_seq[batch_id][seq_id][char_seq_id] = char_id
            tinst.input_vocab_mask[batch_id][0][seq_id] = 1
    tinst.antiMask = torch.from_numpy(getAntiMask(config.query_max_len)).lt(1)
    return tinst


def _batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        insts = [data[i * batch_size + b] for b in range(cur_batch_size)]
        yield insts


def data_iter(data, batch_size, shuffle=False):
    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(_batch_slice(data, batch_size)))
    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def compute_accuracy(pred, truth, normalize=True):
    assert len(pred) == len(truth)
    return accuracy_score(truth, pred, normalize=normalize)
