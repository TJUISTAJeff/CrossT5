import sys
sys.path.extend(['.', '..'])
from CONSTANTS import *
from utils.preprocessing import load_project_list, data_splitter, load_ori_insts
from transformers import AutoTokenizer
from utils.Config import Configurable

config = Configurable('config/t5encoder_decoder.ini')
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')


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
    test_data = []
    for idx in range(num_chunks):
        file = os.path.join(output_base, 'CompleteData%d.pkl' % idx)
        f = open(file, 'rb')
        instances = pickle.load(f)
        for inst in tqdm(instances, desc='Chunk #%d' % idx):
            # 训练集或验证集不需要保留token instance
            # 这里直接处理成Dataset需要的形式
            inst_data = {}
            inst_data['context'] = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(' '.join(inst.context_tokens)))
            inst_data['query'] = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(' '.join(inst.assertion)))
            inst_data['res'] = inst_data['query']
            if id in train_ids:
                train_data.append(inst_data)
                pass
            elif id in dev_ids:
                dev_data.append(inst_data)
                pass
            else:
                test_data.append(inst_data)
                pass
            id += 1
            instances = None
            gc.collect()
            pass
        f.close()
    pass
    return train_data, dev_data, test_data


if __name__ == '__main__':
    train, dev, test = prepare_data()
    # 返回的是已经处理好的数据，直接dump到pkl文件中即可
    # train需要根据卡的数量进行分割
    chunk_size = int(len(train)/2)+1
    output_base = config.data_dir
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    for i in range(2):
        f = open(os.path.join(output_base, 'processtraindata%d.pkl' % i), 'wb')
        for inst in train[i * chunk_size: (i + 1) * chunk_size]:
            f.write(pickle.dumps(inst))
        f.close()
    # 处理Dev
    f = open(os.path.join(output_base, 'processdevdata.pkl'), 'rb')
    for inst in dev:
        f.write(pickle.dumps(inst))
    f.close()
    # 处理Test
    f = open(os.path.join(output_base, 'processtestdata.pkl'), 'wb')
    for inst in test:
        f.write(pickle.dumps(inst))
    f.close()

    pass
