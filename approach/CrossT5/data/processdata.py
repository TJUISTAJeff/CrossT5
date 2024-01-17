import os
import pickle

from CONSTANTS import PROJECT_ROOT
from entities.instance import Instance
from data.vocab import AssertionVocab
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
from tqdm import tqdm
from utils.Config import Configurable
import multiprocessing

config = Configurable('../config/default.ini')

middle_res_base = 'datasets/middle_res'
if not os.path.exists(middle_res_base):
    os.makedirs(middle_res_base)


def pad_seq(seq, maxlen):
    act_len = len(seq)
    if len(seq) < maxlen:
        seq = seq + [0] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq


def splitdata():
    train_reader = open(os.path.join(PROJECT_ROOT, config.train_file), 'rb')
    ori_train = pickle.load(train_reader)
    train_reader.close()
    train = [inst for inst in ori_train if len(inst.local_vocab) < 3000]
    print(len(train))
    chunk_size = len(train) // 30
    for i in range(30):
        open('traindata%d.pkl' % i, 'wb').write(pickle.dumps(train[i * chunk_size: (i + 1) * chunk_size]))


def splitdatadev():
    dev_reader = open(os.path.join(PROJECT_ROOT, config.dev_file), 'rb')
    ori_dev = pickle.load(dev_reader)
    dev = [inst for inst in ori_dev if len(inst.local_vocab) < 3000]
    dev_reader.close()
    chunk_size = len(dev) // 30
    for i in range(30):
        open(os.path.join(PROJECT_ROOT, f'devdata{i}.pkl'), 'wb').write(pickle.dumps(dev[i * chunk_size: (i + 1) * chunk_size]))


def splitdatatest():
    dev_reader = open(os.path.join(PROJECT_ROOT, config.test_file), 'rb')
    ori_dev = pickle.load(dev_reader)
    dev = [inst for inst in ori_dev if len(inst.local_vocab) < 3000]
    dev_reader.close()
    chunk_size = len(dev) // 30
    for i in range(30):
        open(os.path.join(PROJECT_ROOT, f'testdata{i}.pkl'), 'wb').write(pickle.dumps(dev[i * chunk_size: (i + 1) * chunk_size]))


def process(idx):
    train_pkl_reader = open(os.path.join(middle_res_base,'traindata%d.pkl' % idx), 'rb')
    ori_train = pickle.load(train_pkl_reader)
    train = ori_train  # [inst for inst in ori_train if len(inst.local_vocab) < 3000]
    train_pkl_reader.close()
    # transfer data
    newdata = []
    for inst in tqdm(train):
        newcontext = []

        for token in inst.context_tokens:
            lst = tokenizer.tokenize(token)
            lst = tokenizer.convert_tokens_to_ids(lst)
            newcontext.append(pad_seq(lst, config.char_seq_max_len))
        inst.context_tokens = newcontext

        newquery = []
        for token in inst.assertion:
            lst = tokenizer.tokenize(token)
            lst = tokenizer.convert_tokens_to_ids(lst)
            newquery.append(pad_seq(lst, config.char_seq_max_len))
        inst.query_tokens = newquery

        newlocal_vocab = []
        for token in inst.local_vocab.tokens:
            lst = tokenizer.tokenize(token)
            lst = tokenizer.convert_tokens_to_ids(lst)
            newlocal_vocab.append(pad_seq(lst, config.char_seq_max_len))

        lst = [inst.local_vocab.token2id(ass) for ass in inst.assertion]
        inst.local_vocab = newlocal_vocab

        inst.assertion = lst

        newdata.append(inst)
    open(os.path.join(middle_res_base,'newtraindata%d.pkl' % idx), 'wb').write(pickle.dumps(newdata))


def sumdata(num_cards=2):
    newdata = []
    for i in tqdm(range(30)):
        train_pkl_reader = open(os.path.join(PROJECT_ROOT, f'newtraindata{i}.pkl'), 'rb')
        ori_train = pickle.load(train_pkl_reader)
        newdata.extend(ori_train)
        train_pkl_reader.close()
        os.remove('newtraindata%d.pkl' % i)
    chunk_size = len(newdata) // num_cards
    for i in range(num_cards):
        f = open(os.path.join(PROJECT_ROOT, f'processtraindata{i}.pkl'), 'wb')
        for inst in newdata[i * chunk_size: (i + 1) * chunk_size]:
            f.write(pickle.dumps(inst))
        f.close()


def sumdatadev():
    newdata = []
    for i in range(30):
        dev_reader = open('processdevdata%d.pkl' % i, 'rb')
        ori_dev = pickle.load(dev_reader)
        newdata.extend(ori_dev)
        dev_reader.close()
    f = open('processdevdata.pkl', 'wb')
    for inst in newdata:
        f.write(pickle.dumps(inst))
    f.close()


def sumdatatest():
    newdata = []
    testvocb = []
    testassertion = []
    for i in range(30):
        dev_reader = open('processtestdata%d.pkl' % i, 'rb')
        ori_dev = pickle.load(dev_reader)
        newdata.extend(ori_dev)
        dev_reader.close()
        vocabreader = open('testvocab%d.pkl' % i, 'rb')
        vocabs = pickle.load(vocabreader)
        testvocb.extend(vocabs)
        vocabreader.close()
        assertionreader = open('testassertion%d.pkl' % i, 'rb')
        assertions = pickle.load(assertionreader)
        testassertion.extend(assertions)
        assertionreader.close()
    f = open('processtestdata.pkl', 'wb')
    for inst in newdata:
        f.write(pickle.dumps(inst))
    f.close()
    f = open('testvocab.pkl', 'wb')
    pickle.dump(testvocb, f)
    f.close()
    f = open('testassertion.pkl', 'wb')
    pickle.dump(testassertion, f)
    f.close()


def processdev(idx):
    dev_reader = open('devdata%d.pkl' % idx, 'rb')
    ori_dev = pickle.load(dev_reader)
    dev_reader.close()
    newdata = []
    for inst in tqdm(ori_dev):
        newcontext = []

        for token in inst.context_tokens:
            lst = tokenizer.tokenize(token)
            if len(lst) > 512:
                print(token)
            lst = tokenizer.convert_tokens_to_ids(lst)
            newcontext.append(pad_seq(lst, config.char_seq_max_len))
        inst.context_tokens = newcontext

        newquery = []
        for token in inst.assertion:
            lst = tokenizer.tokenize(token)
            if len(lst) > 512:
                print(token)
            lst = tokenizer.convert_tokens_to_ids(lst)
            newquery.append(pad_seq(lst, config.char_seq_max_len))
        inst.query_tokens = newquery

        newlocal_vocab = []
        for token in inst.local_vocab.tokens:
            lst = tokenizer.tokenize(token)
            lst = tokenizer.convert_tokens_to_ids(lst)
            newlocal_vocab.append(pad_seq(lst, config.char_seq_max_len))

        lst = [inst.local_vocab.token2id(ass) for ass in inst.assertion]
        inst.local_vocab = newlocal_vocab

        inst.assertion = lst

        newdata.append(inst)
    open('processdevdata%d.pkl' % idx, 'wb').write(pickle.dumps(newdata))


def processtest(idx):
    dev_reader = open('testdata%d.pkl' % idx, 'rb')
    ori_dev = pickle.load(dev_reader)
    dev_reader.close()
    newdata = []
    testvocb = []
    assertion = []
    for inst in tqdm(ori_dev):
        newcontext = []
        testvocb.append(inst.local_vocab)
        for token in inst.context_tokens:
            lst = tokenizer.tokenize(token)
            if len(lst) > 512:
                print(token)
            lst = tokenizer.convert_tokens_to_ids(lst)
            newcontext.append(pad_seq(lst, config.char_seq_max_len))
        inst.context_tokens = newcontext
        assertion.append(inst.assertion)
        newquery = []
        for token in inst.assertion:
            lst = tokenizer.tokenize(token)
            if len(lst) > 512:
                print(token)
            lst = tokenizer.convert_tokens_to_ids(lst)
            newquery.append(pad_seq(lst, config.char_seq_max_len))
        inst.query_tokens = newquery

        newlocal_vocab = []
        for token in inst.local_vocab.tokens:
            lst = tokenizer.tokenize(token)
            lst = tokenizer.convert_tokens_to_ids(lst)
            newlocal_vocab.append(pad_seq(lst, config.char_seq_max_len))

        lst = [inst.local_vocab.token2id(ass) for ass in inst.assertion]
        inst.local_vocab = newlocal_vocab

        inst.assertion = lst

        newdata.append(inst)
    open('processtestdata%d.pkl' % idx, 'wb').write(pickle.dumps(newdata))
    open('testvocab%d.pkl' % idx, 'wb').write(pickle.dumps(testvocb))
    open('testassertion%d.pkl' % idx, 'wb').write(pickle.dumps(assertion))


if __name__ == '__main__':
    print('Processing Training data:...')
    multiprocessing.freeze_support()
    # splitdata()
    p = []
    for i in range(30):
        proc = multiprocessing.Process(target=process, args=(i,))
        proc.start()
        p.append(proc)
    for proc in p:
        proc.join()
    sumdata()

    splitdatatest()
    p = []
    for i in range(30):
        proc = multiprocessing.Process(target=processtest, args=(i,))
        proc.start()
        p.append(proc)
    for proc in p:
        proc.join()
    sumdatatest()

    # dev_reader = open('testassertion.pkl', 'rb')
    # ori_dev = pickle.load(dev_reader)
    # dev_reader.close()
    # print(ori_dev[0])
    # splitdatatest()
    # sumdata()
    # sumdatatest()
