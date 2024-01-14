import pickle
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('D:\Pycharm WorkSpace\codet5-small')


def get_tf2torch():
    with open(r'D:\Pycharm WorkSpace\Github_crap\tf_and_torch\tf_and_torch.pkl', 'rb') as pck:
        data = pickle.load(pck)


        dataset = []
        for item_torch, item_tf in zip(data[0], data[1]):
            api_torch = item_torch['query'].split()[-1].strip('-()')
            api_tf = item_tf['query'].split()[-1].strip('-()')
            # print(api_tf, api_torch)

            # query_tokens = tokenizer.tokenize(item_torch['query'] + " || " + item_tf['assertion'])
            query_tokens = tokenizer.tokenize(item_torch['query'])
            assert_tokens = tokenizer.tokenize('<s> ' + item_torch['assertion'].strip('()') + ' ) </s>')

            with open(r'.\tf_para_doc.pkl', 'rb') as pck:
                para_tf = pickle.load(pck)

            with open(r'.\torch_para_doc.pkl', 'rb') as pck:
                para_torch = pickle.load(pck)

            if len(query_tokens) >= 512:
                query_tokens = query_tokens[-511:]

            inst_data = {}
            inst_data['context'] = tokenizer.convert_tokens_to_ids(query_tokens)
            inst_data['assert'] = tokenizer.convert_tokens_to_ids(assert_tokens)
            inst_data['para_1'] = para_tf[api_tf]
            inst_data['para_2'] = para_torch[api_torch]

            dataset.append(inst_data)

        with open(r'D:\Pycharm WorkSpace\Github_crap\download_projects\pycodegptdataset_tf2torch.pkl', 'wb') as f:
            pickle.dump(dataset, f)

# get_tf2torch()

def get_torch2tf():
    with open(r'D:\Pycharm WorkSpace\Github_crap\tf_and_torch\tf_and_torch.pkl', 'rb') as pck:
        data = pickle.load(pck)

        dataset = []
        for item_torch, item_tf in zip(data[0], data[1]):
            api_torch = item_torch['query'].split()[-1].strip('-()')
            api_tf = item_tf['query'].split()[-1].strip('-()')
            # print(api_tf, api_torch)

            query_tokens = tokenizer.tokenize(item_tf['query'])
            assert_tokens = tokenizer.tokenize('<s> ' + item_tf['assertion'].strip('()') + ' ) </s>')

            with open(r'.\tf_para_doc.pkl', 'rb') as pck:
                para_tf = pickle.load(pck)

            with open(r'.\torch_para_doc.pkl', 'rb') as pck:
                para_torch = pickle.load(pck)

            if len(query_tokens) >= 512:
                query_tokens = query_tokens[-511:]

            inst_data = {}
            inst_data['context'] = tokenizer.convert_tokens_to_ids(query_tokens)
            inst_data['assert'] = tokenizer.convert_tokens_to_ids(assert_tokens)
            inst_data['para_1'] = para_torch[api_torch]
            inst_data['para_2'] = para_tf[api_tf]

            dataset.append(inst_data)

        with open(r'D:\Pycharm WorkSpace\Github_crap\download_projects\t5dataset_torch2tf.pkl', 'wb') as f:
            pickle.dump(dataset, f)

# get_tf2torch()
get_torch2tf()
# with open(r'D:\Pycharm WorkSpace\Github_crap\download_projects\t5dataset_torch2tf.pkl', 'rb') as f:
#     data = pickle.load(f)
#     # for inst in data:
#     #     print(inst['context'])
#     #     print(inst['assert'])
#     #     print(inst['para_1'],'\n')
#     #     print(inst['para_2'],'\n')
#
#     print(len(data))

# with open(r'D:\Pycharm WorkSpace\Github_crap\tf_and_torch\tf_and_torch.pkl', 'rb') as pck:
#     data = pickle.load(pck)
#
#
#     dataset = []
#     for item_torch, item_tf in zip(data[0], data[1]):
#         print(item_torch['query'])
#         print(item_torch['assertion'])
#         break