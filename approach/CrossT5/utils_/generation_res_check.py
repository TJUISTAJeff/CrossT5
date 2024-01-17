import os.path

import javalang.tokenizer
from tqdm import *

from utils.Config import Configurable
from transformers import AutoTokenizer

if __name__ == '__main__':
    configs = ['config/t5encoder_small_transformer_attention-based_cpynet_no_emb.ini',
               'config/t5encoder_decoder.ini']
    need_remove_head_and_tail = [True, False]
    comparable_generations = {}
    overall_generations = {}
    all_ground_truths = set()
    preds = []
    for config_file, flag in zip(configs, need_remove_head_and_tail):
        if config_file not in comparable_generations.keys():
            comparable_generations[config_file] = {}
            overall_generations[config_file] = {}
        config = Configurable(config_file)
        base = os.path.join(config.generation_dir, 'Beam-5')
        for i in range(2):
            cur_gt_file = os.path.join(base, 'ground_truth%d.txt' % i)
            cur_gen_file = os.path.join(base, 'predictions_5_%d.txt' % i)
            gt_reader = open(cur_gt_file, 'r', encoding='iso8859-1')
            with open(cur_gen_file, 'r', encoding='iso8859-1') as reader:
                context = []
                line = ''
                while True:
                    line = reader.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line == '':
                        if len(context) == 2 and context[0] == 'Write ground truth failed.':
                            context.clear()
                            continue
                        elif len(context) == 1 and context[0] == 'Write generated assertion failed.':
                            gt_reader.readline()
                            context.clear()
                            continue
                        cur_generations = [' '.join(cline.strip().split()[1:-1]) for cline in
                                           context] if flag else context
                        gt_tokens = gt_reader.readline().strip().split()
                        gt = ' '.join(gt_tokens[1:-1]) if flag else ' '.join(gt_tokens)
                        tokenized_gt = ' '.join([x.value for x in javalang.tokenizer.tokenize(gt)])
                        all_ground_truths.add(hash(tokenized_gt))
                        res = -1
                        for j, gen_ass in enumerate(cur_generations):
                            if hash(gt) == hash(gen_ass):
                                res = j
                                break
                        preds.append(res)
                        comparable_generations[config_file][hash(tokenized_gt)] = res
                        overall_generations[config_file][hash(tokenized_gt)] = cur_generations
                        context = []
                        pass
                    else:
                        context.append(line)
            gt_reader.close()

    cnt = 0
    for gt in tqdm(all_ground_truths):
        if gt not in comparable_generations[configs[0]].keys() or gt not in comparable_generations[configs[1]].keys():
            continue
        if comparable_generations[configs[0]][gt] == -1 and comparable_generations[configs[1]][gt] == 0:
            cnt += 1
            print('\n'.join(overall_generations[configs[0]][gt]))
            print('-------------------------------------')
            # print(comparable_generations[configs[1]][gt], '\n'.join(overall_generations[configs[1]][gt]))
            print(overall_generations[configs[1]][gt][0])
            print('*************************************')
            print()
            pass
    print(cnt)
    pass
