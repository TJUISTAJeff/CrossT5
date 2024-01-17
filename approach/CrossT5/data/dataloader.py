import sys
sys.path.extend(['.', '..'])

from CONSTANTS import *
import multiprocessing
from utils.Config import Configurable
from data.vocab import TokenVocab, DynamicVocab
from entities.instance import Instance, TokenInstance

config = Configurable( 'config/default.ini')

# load global
global_vocab_tokens = set()
with open( 'datasets/vocabs/global_dict', 'r', encoding='iso8859-1') as reader:
    for line in reader.readlines():
        token, _ = line.strip().split()
        token = token.replace("<space>", " ")
        global_vocab_tokens.add(token)
        if len(global_vocab_tokens) == 1000:
            break

api_vocab_dict = {}
for i in range(4):
    with open( 'datasets/vocabs/api_' + str(i) + '.json'), 'r') as reader:
        jsonObj = json.load(reader)
        for obj in jsonObj['apis']:
            type = obj['type']
            token = obj['name']
            if type not in api_vocab_dict.keys():
                api_vocab_dict[type] = []
            api_vocab_dict[type].append(token)


def clear_signature(tokens):
    cleaned_tokens = set()
    if len(tokens) == 0:
        return cleaned_tokens
    pattern1 = re.compile('@.+;')
    pattern2 = re.compile('@.+')

    for token in tokens:
        if '#' in token:
            if '@' in token:
                if ';' in token:
                    # resolved method
                    res = pattern1.findall(token)
                    for tmp_token in res:
                        cleaned_tokens.add(tmp_token[1:-1])

                else:
                    # others
                    res = pattern2.findall(token)
                    for tmp_token in res:
                        cleaned_tokens.add(tmp_token[1:])

            else:
                # Type name
                class_signature = token[1:]
                cleaned_tokens.add(class_signature.split('.')[-1])
        elif token.startswith('@'):
            # Unresolved method name
            cleaned_tokens.add(token[1:])
        else:
            # tokens
            cleaned_tokens.add(token)

    return cleaned_tokens





# def load_json_data(file= 'datasets/Processed_Data/esjc.json'), ):
#     instances = []
#     pattern = re.compile('assert(Equals|Null|NotNull|True|False)')
#     missed = {
#         'Exceed_max_length': 0,
#         'Contains OOV': 0,
#         'Unsupported Assertion': 0
#     }
#     try:
#         with open(file, 'r', encoding='utf-8') as reader:
#             data = json.load(reader)
#         # Remove old version json data files.
#         if 'project_static_tokens' not in data.keys():
#             data = None
#             os.remove(file)
#             return instances, 0
#         # project_tokens = clear_signature(data['project_static_tokens'].split())
#         # project_tokens = project_tokens | clear_signature(data['project_enum_tokens'].split())
#         data = data['data']
#         for jsonObj in data:
#             try:
#                 assertion = jsonObj['statement']
#                 res = re.findall(pattern, assertion)
#                 if len(res) == 1:
#                     if not assertion.startswith('assert' + res[0]):
#                         if assertion.startswith('Assert.assert' + res[0]):
#                             assertion = assertion[7:]
#                         elif assertion.startswith('org.junit.Assert.assert' + res[0]):
#                             assertion = assertion[17:]
#                         else:
#                             missed['Unsupported Assertion'] += 1
#                             raise NotImplementedError("Illegal start token for assertion found in %s" % assertion)
#                             pass
#                 else:
#                     raise Exception("More than one assertion found in %s" % assertion)
#                 context = jsonObj['context']
#                 tmp_tks = [x for x in context.split() if x not in [' ', '', '\n']]
#                 if tmp_tks[-1].startswith("!"):
#                     context = ' '.join(tmp_tks[:-1])
#                 # remove redundant <space>
#                 assertion = assertion.replace("<space>", " ")
#                 assertion_tokens = ['<s>'] + [x.value for x in javalang.tokenizer.tokenize(assertion)]
#                 assert_token_set = set(assertion_tokens[2:])
#                 assertion_tokens.append('</s>')
#                 context_tokens = [x.value for x in javalang.tokenizer.tokenize(context)]
#                 del assertion, context

#                 local_tokens = set(context_tokens)
#                 # read other dynamic vocab tokens
#                 local_vocabs = jsonObj['local_vocab']
#                 for key, value in local_vocabs.items():
#                     # if key == 'focal_method' and value == '':
#                     #     raise Exception('no focal method')
#                     if key == 'import_classes':
#                         class_names = [x[1:] for x in value.strip().split()]
#                         for class_name in class_names:
#                             if class_name in api_vocab_dict.keys():
#                                 local_tokens = local_tokens | set(api_vocab_dict[class_name])
#                     else:
#                         tmp_tokens = clear_signature(value.split())
#                         local_tokens = local_tokens | set(tmp_tokens)
#                 # local_tokens = local_tokens | project_tokens
#                 local_tokens = local_tokens | global_vocab_tokens
#                 left_tokens = assert_token_set - local_tokens
#                 if len(local_tokens) < 3000:
#                     if len(left_tokens) == 0:
#                         dynamic_vocab = DynamicVocab(local_tokens)
#                         del local_tokens
#                         inst = Instance(assertion_tokens, context_tokens, dynamic_vocab)
#                         instances.append(inst)
#                     else:
#                         # print(left_tokens)
#                         missed['Contains OOV'] += 1
#                         raise NotImplementedError('Contains OOV.')
#                 else:
#                     missed['Exceed_max_length'] += 1
#                     raise NotImplementedError('Exceed max vocab length limitation.')
#             except Exception:
#                 continue

#     except UnicodeDecodeError:
#         print('Error: Encoding UTF-8 does not fit. File %s' % file)
#         exit(-1)
#     except FileNotFoundError:
#         print('Error: File %s does not found, please check' % file)
#         exit(-2)
#     except Exception as exception:
#         print('Error: Unexpceted exception %s has occurred when processing file %s, please check.' % (file, exception))
#         return instances, 0
#     finally:
#         print(
#             'Finished parsing file %s, found %d instances in total. %s.' % (file, len(instances), str(missed)))
#         gc.collect()

#     return instances, missed




def process_test(place_holder, ori_insts):
    newdata = []
    testvocb = []
    assertion = []
    for inst in tqdm(ori_insts):
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
    output_base =  config.data_dir)
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


def process_dev(placeholder, ori_insts):
    newdata = []
    for inst in tqdm(ori_insts):
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
    output_base =  config.data_dir)
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    f = open(os.path.join(output_base, 'processdevdata.pkl'), 'wb')
    for inst in newdata:
        f.write(pickle.dumps(inst))
    f.close()


def tokenize_insts(idx, instances):
    newdata = []
    for inst in tqdm(instances):
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
    output_base =  config.middle_res_dir)
    print('Process %d has analyzed %d instance.' % (idx, len(newdata)))
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    open(os.path.join(output_base, 'processed_train%d.pkl' % idx), 'wb').write(pickle.dumps(newdata))


def process_train(ori_insts, num_cards):
    num_chunks = 10
    chunk_size = len(ori_insts) // num_chunks
    processes = []
    for idx in range(num_chunks):
        data = [inst for inst in ori_insts[chunk_size * idx:chunk_size * (idx + 1)]]
        p = multiprocessing.Process(target=tokenize_insts, args=(idx, data))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finished preparing training data, start writing train file.')
    new_data = []
    output_base =  config.middle_res_dir
    for idx in tqdm(range(num_chunks)):
        file = os.path.join(output_base, 'processed_train%d.pkl' % idx)
        with open(file, 'rb') as f:
            new_data.extend(pickle.load(f))
        os.remove(file)
    print('Found %d instances in total.' % len(new_data))
    chunk_size = len(new_data) // num_cards
    output_base =  config.data_dir
    if not os.path.exists(output_base): os.makedirs(output_base)
    for i in range(num_cards):
        f = open(os.path.join(output_base, 'processtraindata%d.pkl' % i), 'wb')
        for inst in new_data[i * chunk_size: (i + 1) * chunk_size]:
            f.write(pickle.dumps(inst))
        f.close()


# def prepare_data(num_cards):
#     pickle_file =  'datasets/Complete_Data.pkl')
#     total = 0
#     instances = []
#     # if the complete data pkl exists, then go to sampling the train, dev and test idxes.
#     if True and os.path.exists(pickle_file):
#         f = open(pickle_file, 'rb')
#         while True:
#             try:
#                 instances = pickle.load(f)
#                 total += len(instances)
#             except:
#                 break
#         pass
#         f.close()
#     # Or it does not exist, prepare one from raw data.
#     else:
#         projects = load_project_list()
#         num_chunks = 20
#         chunk_size = len(projects) // num_chunks
#         print(chunk_size)
#         processes = []
#         for idx in range(num_chunks):
#             sub_projects = [p.split('/')[1] for p in projects[chunk_size * idx:chunk_size * (idx + 1)]]
#             p = multiprocessing.Process(target=load_ori_insts, args=(idx, sub_projects))
#             p.start()
#             processes.append(p)
#         for p in processes:
#             p.join()
#         output_base =  config.middle_res_dir)
#         for idx in tqdm(range(num_chunks)):
#             file = os.path.join(output_base, 'CompleteData%d.pkl' % idx)
#             if os.path.exists(file):
#                 f = open(file, 'rb')
#                 instances.extend(pickle.load(f))
#                 f.close()
#                 os.remove(file)
#         if len(instances) != 0:
#             f = open(pickle_file, 'wb')
#             pickle.dump(instances, f)
#             f.close()
#             print('Load %d instances finished.' % len(instances))
#         pass

#     if len(instances) != 0:
#         # After the above step, the Complete_Data.pkl should be prepared, sample train, dev and test instances.
#         train, dev, test = data_splitter(instances, [8, 1, 1])

#         # For TOGA comparison.
#         new_train = []
#         new_dev = []
#         new_test = []
#         test_valid_idxes = set()
#         train_valid_idxes = set()
#         dev_valid_idxes = set()
#         with open(os.path.join(config.data_dir, 'train_valid_idx.txt'), 'r', encoding='utf-8') as reader:
#             for line in reader.readlines():
#                 line = line.strip()
#                 tks = line.split()
#                 if int(tks[1]) == 1:
#                     train_valid_idxes.add(int(tks[0]))

#         for idx, inst in enumerate(train):
#             if idx in train_valid_idxes:
#                 new_train.append(inst)

#         with open(os.path.join(config.data_dir, 'test_valid_idx.txt'), 'r', encoding='utf-8') as reader:
#             for line in reader.readlines():
#                 line = line.strip()
#                 tks = line.split()
#                 test_valid_idxes.add(int(tks[0]))

#         for idx, inst in enumerate(test):
#             if idx in test_valid_idxes:
#                 new_test.append(inst)

#         with open(os.path.join(config.data_dir, 'valid_valid_idx.txt'), 'r', encoding='utf-8') as reader:
#             for line in reader.readlines():
#                 line = line.strip()
#                 tks = line.split()
#                 if int(tks[1]) == 1:
#                     dev_valid_idxes.add(int(tks[0]))

#         for idx, inst in enumerate(dev):
#             if idx in dev_valid_idxes:
#                 new_dev.append(inst)

#         train = new_train
#         dev = new_dev
#         test = new_test
#         # ===================================================================

#         print('Total instances: %d' % (len(instances)))
#         print('Num train: %d' % (len(train)))
#         print('Num dev: %d' % (len(dev)))
#         print('Num test: %d' % (len(test)))
#         with open('datasets/train.pkl', 'wb') as f:
#             pickle.dump(train, f)
#         with open('datasets/dev.pkl', 'wb') as f:
#             pickle.dump(dev, f)
#         with open('datasets/test.pkl', 'wb') as f:
#             pickle.dump(test, f)
#         process_train(train, num_cards)
#         processes = []
#         p1 = multiprocessing.Process(target=process_dev, args=(None, dev))
#         p1.start()
#         processes.append(p1)
#         p2 = multiprocessing.Process(target=process_test, args=(None, test))
#         p2.start()
#         processes.append(p2)
#         for p in processes:
#             p.join()

def prepare_data():
    # pickle_file =  'datasets/Complete_Data.pkl')
    output_base =  config.middle_res_dir)
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

    if len(need_to_parse) !=0:
        #如果某个文件缺失了，重新分析对应的project json file
        chunk_size = len(projects) // num_chunks
        processes = []
        for idx in need_to_parse:
            sub_projects = [p.split('/')[1] for p in projects[chunk_size * idx:chunk_size * (idx + 1)]]
            p = multiprocessing.Process(target=load_ori_insts, args=(idx, sub_projects))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    
    # After the above step, the Complete_Data.pkl should be prepared, sample train, dev and test instances.
    train_ids, dev_ids, test_ids = data_splitter(list(range(187214)),[8,1,1])
    train_ids = set(train_ids)
    dev_ids = set(dev_ids)
    test_ids = set(test_ids)
    # train, dev, test = data_splitter(instances, [8, 1, 1])
    print('Total instances: %d' % 187214)
    print('Num train: %d' % (len(train_ids)))
    print('Num dev: %d' % (len(dev_ids)))
    print('Num test: %d' % (len(test_ids)))
    id = 0
    train, dev, test = [], [], []
    train_data = []
    dev_data = []
    for idx in range(num_chunks):
        file = os.path.join(output_base, 'CompleteData%d.pkl' % idx)
        f = open(file, 'rb')
        instances = pickle.load(f)
        for inst in tqdm(instances,desc='Chunk #%d'%idx):
            if id in train_ids or id in dev_ids:
                # 训练集或验证集不需要保留token instance
                # 这里直接处理成Dataset需要的形式
                inst_data = {}
                inst_data['context'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(inst.context_tokens)))
                newquery = []
                for token in inst.assertion:
                    lst = tokenizer.tokenize(token)
                    lst = tokenizer.convert_tokens_to_ids(lst)
                    newquery.append(pad_seq(lst, config.char_seq_max_len))
                inst_data['query']= newquery
                newlocal_vocab = []
                for token in inst.local_vocab.tokens:
                    lst = tokenizer.tokenize(token)
                    lst = tokenizer.convert_tokens_to_ids(lst)
                    if len(lst) > config.char_seq_max_len:
                        continue
                    newlocal_vocab.append(pad_seq(lst, config.char_seq_max_len))
                lst = [inst.local_vocab.token2id(ass) for ass in inst.assertion]
                inst_data['vocab'] = newlocal_vocab
                inst_data['res'] = lst
                if id in train_ids:
                    train_data.append(inst_data)
                else:
                    dev_data.append(inst_data)
                pass
            else:
                #测试集需要保留token instances
                test.append(pickle.loads(pickle.dumps(inst)))
                pass                
            id +=1
            instances=None
            gc.collect()
            pass
        f.close()
    pass
    return train_data, dev_data, test





if __name__ == '__main__':
    i,m = load_json("datasets/Processed_Data/yoga.json")
    print(len(i))
    print(len(m))