import sys
sys.path.extend(['.', '..'])
from CONSTANTS import *
from utils.preprocessing import pad_seq, load_project_list, data_splitter, load_ori_insts
from transformers import AutoTokenizer
from utils.Config import Configurable


tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
config = Configurable('config/t5encoder_small_transfromer_no_emb.ini')

def process_test(ori_insts):
    newdata = []
    testvocb = []
    assertion = []
    for inst in tqdm(ori_insts):
        newcontext = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(' '.join(inst.context_tokens)))
        testvocb.append(inst.local_vocab)
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
    output_base = config.data_dir
    f = open(os.path.join(output_base, 'processtestdata.pkl'), 'wb')
    for inst in newdata:
        f.write(pickle.dumps(inst))
    f.close()
    f = open(os.path.join(output_base, 'testvocab.pkl'), 'wb')
    pickle.dump(testvocb, f)
    f.close()
    f = open(os.path.join(output_base, 'testassertion.pkl'), 'wb')
    pickle.dump(assertion, f)
    f.close()

def prepare_data():
    output_base = config.middle_res_dir
    total = 0
    instances = []
    projects = load_project_list()
    num_chunks = 20
    need_to_parse = []

    # 首先检查所有线程的预处理结果
    for i in range(num_chunks):
        file = os.path.join(output_base, 'CompleteData%d.pkl' % i)
        if not os.path.exists(file):
            need_to_parse.append(i)

    if len(need_to_parse) != 0:
        #如果某个文件缺失了，重新分析对应的project json file
        chunk_size = len(projects) // num_chunks
        processes = []
        for idx in need_to_parse:
            sub_projects = [
                p.split('/')[1] for p in projects[chunk_size * idx:chunk_size * (idx + 1)]]
            p = multiprocessing.Process(
                target=load_ori_insts, args=(idx, sub_projects))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # 首先根据id筛选train，dev和test数据的index
    train_ids, dev_ids, test_ids = data_splitter(
        list(range(187214)), [8, 1, 1])
    train_ids = set(train_ids)
    dev_ids = set(dev_ids)
    test_ids = set(test_ids)
    print('Total instances: %d' % 187214)
    print('Num train: %d' % (len(train_ids)))
    print('Num dev: %d' % (len(dev_ids)))
    print('Num test: %d' % (len(test_ids)))

    # 逐个文件读取，按照id累加的方式计算每个instance应该属于哪个数据集
    id = 0
    train_data = []
    dev_data = []
    test = []
    for idx in range(num_chunks):
        data = {
            "train":[],
            "dev":[],
            "test":[]
        }

        file = os.path.join(output_base, 'CompleteData%d.pkl' % idx)
        f = open(file, 'rb')
        instances = pickle.load(f)
        f.close()
        for inst in tqdm(instances, desc='Chunk #%d' % idx):
            if id in train_ids or id in dev_ids:
                # 训练集或验证集不需要保留token instance
                # 这里直接处理成Dataset需要的形式
                start = time.time()
                inst_data = {}
                inst_data['context'] = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(' '.join(inst.context_tokens)))
                newquery = []
                for token in inst.assertion:
                    lst = tokenizer.tokenize(token)
                    lst = tokenizer.convert_tokens_to_ids(lst)
                    newquery.append(pad_seq(lst, config.char_seq_max_len))
                inst_data['query'] = newquery
                newlocal_vocab = []
                for token in inst.local_vocab.tokens:
                    lst = tokenizer.tokenize(token)
                    lst = tokenizer.convert_tokens_to_ids(lst)
                    if len(lst) > config.char_seq_max_len:
                        continue
                    newlocal_vocab.append(
                        pad_seq(lst, config.char_seq_max_len))
                lst = [inst.local_vocab.token2id(ass) for ass in inst.assertion]
                inst_data['vocab'] = newlocal_vocab
                inst_data['res'] = lst
                if id in train_ids:
                    data["train"].append(inst_data)
                else:
                    data["dev"].append(inst_data)
                pass
            else:
                #测试集需要保留token instances
                data['test'].append(pickle.loads(pickle.dumps(inst)))
                pass
            id += 1
            pass
        with open(os.path.join(config.middle_res_dir,"processed_data%d.pkl"%idx),'wb') as f:
            pickle.dump(data, f)
        data = None
        instances = None
        gc.collect()
        pass

if __name__ == '__main__':
    # prepare_data()
    train, dev, test  = [],[],[]
    for i in tqdm(range(20)):
        file = os.path.join(config.middle_res_dir,'processed_data%d.pkl'%i)
        if not os.path.exists(file):
            continue
        with open(file,'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    train.extend(data['train'])
                    dev.extend(data['dev'])
                    test.extend(data['test'])
                except EOFError:
                    break


    # 返回的train和dev是已经处理好的数据，直接dump到pkl文件中即可
    # train需要根据卡的数量进行分割
    chunk_size = int(len(train)/2)+1
    output_base = config.data_dir
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    for i in range(2):
        f = open(os.path.join(output_base, 'processtraindata%d.pkl' % i), 'wb')
        cur_data = train[i * chunk_size: (i + 1) * chunk_size]
        pickle.dump(cur_data,f)
        f.close()

    f = open(os.path.join(output_base, 'processdevdata.pkl'), 'wb')
    pickle.dump(dev,f)
    f.close()

    # 测试数据单独处理
    process_test(test)
    pass
