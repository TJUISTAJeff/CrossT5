# -*- coding : utf-8-*-
# coding:unicode_escape
import os.path
import sys
import io
import copy

from entities.instance import atlasInstance

sys.path.extend(['.', '..'])

from CONSTANTS import *
from data.vocab import ATLASVocab
import codecs


def load_vocab_tokens(file=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Training/context_token_min_freq10.txt")):
    bfile = codecs.open(file, 'r', 'ISO-8859-1')
    data = bfile.readlines()
    bfile.close()
    token_ = [x.strip().split('<separator>')[0].strip() for x in data]
    return token_


def load_data(file=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Training/testMethods.txt"), assertion=False):
    bfile = codecs.open(file, 'r', 'ISO-8859-1')
    data = bfile.readlines()
    bfile.close()
    contexts = []
    for context in data:
        tokens = [x.strip() for x in context.strip().split(' ')]
        if assertion:
            tokens.append("<EOS>")
        contexts.append(tokens)
    return contexts


def prepare_vocab(
        context_token_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Training/context_token_min_freq10.txt"),
        assert_token_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Training/assert_token.txt"),
        keep_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/vocab.pkl")
):
    context_tokens = load_vocab_tokens(context_token_path)
    assert_vocab_tokens = load_vocab_tokens(assert_token_path)
    global_vocab = ATLASVocab(list(set(context_tokens) | set(assert_vocab_tokens)))
    keep_f = open(keep_path, 'wb')
    pickle.dump(global_vocab, keep_f, protocol=pickle.HIGHEST_PROTOCOL)
    keep_f.close()


def prepare_data(
        context_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Training/testMethods.txt"),
        assert_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Training/assertLines.txt"),
        assert_token_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Training/assert_token.txt"),
        keep_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/train.pkl")
):
    contexts = load_data(context_path)
    # context_tokens = load_vocab_tokens(context_token_path)
    asserts = load_data(assert_path, True)
    assert_vocab_tokens = load_vocab_tokens(assert_token_path)
    # global_vocab = ATLASVocab(list(set(context_tokens) | set(assert_vocab_tokens)))
    assert_vocab = ATLASVocab(assert_vocab_tokens)
    insts = []
    for x, y in zip(contexts, asserts):
        inst = atlasInstance(x, y, copy.deepcopy(assert_vocab))
        inst.update_vocab(set(x))
        insts.append(inst)
    keep_f = open(keep_path, 'wb')
    for inst in tqdm(insts, desc='Writing instances.'):
        pickle.dump(inst, keep_f, protocol=pickle.HIGHEST_PROTOCOL)
    keep_f.close()


def prepare_all_data():
    vocab_keep_path = os.path.join(PROJECT_ROOT, "datasets/ATLAS/vocab.pkl")
    if not os.path.exists(vocab_keep_path):
        prepare_vocab()
    train_data_path = os.path.join(PROJECT_ROOT, "datasets/ATLAS/train.pkl")
    if not os.path.exists(train_data_path):
        prepare_data()
    dev_data_path = os.path.join(PROJECT_ROOT, "datasets/ATLAS/dev.pkl")
    if not os.path.exists(dev_data_path):
        prepare_data(context_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Eval/testMethods.txt"),
                     assert_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Eval/assertLines.txt"),
                     keep_path=dev_data_path)
    test_data_path = os.path.join(PROJECT_ROOT, "datasets/ATLAS/test.pkl")
    if not os.path.exists(test_data_path):
        prepare_data(context_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Testing/testMethods.txt"),
                     assert_path=os.path.join(PROJECT_ROOT, "datasets/ATLAS/Testing/assertLines.txt"),
                     keep_path=test_data_path)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='ISO-8859-1')
    prepare_all_data()
