import os.path
import pickle
import sys

sys.path.extend(['.', '..'])

from transformers import AutoTokenizer
from utils.Config import Configurable
from tqdm import *

config = Configurable('config/t5encoder_decoder_atlas.ini')
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")


def prepare_data(num_cards):
    groups = ['Training', 'Eval', 'Testing']
    for group in groups:
        data = []
        tm_reader = open('datasets/Raw_Dataset/%s/testMethods.txt' % group, 'r', encoding='iso8859-1')
        as_reader = open('datasets/Raw_Dataset/%s/assertLines.txt' % group, 'r', encoding='iso8859-1')
        tms = tm_reader.readlines()
        assertions = as_reader.readlines()
        assert len(tms) == len(assertions)
        for i in tqdm(range(len(tms))):
            tm = tms[i]
            assertion = assertions[i]
            input = tokenizer.encode(tm.strip(), add_special_tokens=False)
            query = tokenizer.encode(assertion.strip(), add_special_tokens=True)
            data.append((input, query))

        if group == 'Training':
            num_per_chunk = int((len(data) / num_cards) + 1)
            for cid in range(num_cards):
                dumped_data = data[cid * num_per_chunk:(cid + 1) * num_per_chunk]

                if not os.path.exists(config.data_dir):
                    os.makedirs(config.data_dir)

                with open(os.path.join(config.data_dir, '%s_%d.pkl' % (group, cid)), 'wb') as f:
                    pickle.dump(dumped_data, f)

            data.clear()
        else:
            with open(os.path.join(config.data_dir, '%s.pkl' % group), 'wb') as f:
                pickle.dump(data, f)
            data.clear()
        tm_reader.close()
        as_reader.close()


if __name__ == '__main__':
    prepare_data(2)
    pass
