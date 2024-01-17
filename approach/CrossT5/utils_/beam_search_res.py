import os.path

import javalang.tokenizer

from utils.Config import Configurable

config_file = 'config/t5encoder_small_transformer_attention-based_cpynet_no_emb.ini'
config = Configurable(config_file)
# if __name__ == '__main__':
#     base = os.path.join(config.generation_dir, 'Beam-%d' % config.beam_size)
#     preds = []
#     for i in range(2):
#         gt_file = os.path.join(base, 'ground_truth%d.txt' % i)
#         gen_file = os.path.join(base, 'predictions_%d_%d.txt' % (config.beam_size, i))
#
#         gt_reader = open(gt_file, 'r', encoding='iso8859-1')
#         gen_reader = open(gen_file, 'r', encoding='iso8859-1')
#         line = ""
#         context = []
#         while True:
#             line = gen_reader.readline()
#             if not line:
#                 break
#             line = line.strip()
#             if line == '':
#                 if len(context) == 2 and context[1] == 'Write generated assertion failed.':
#                     context.clear()
#                     continue
#                 elif len(context) == 1 and context[0] == 'Write generated assertion failed.':
#                     gt_reader.readline()
#                     context.clear()
#                     continue
#                 else:
#                     gt = gt_reader.readline().strip()
#                     res = config.beam_size + 1
#                     for i, gen in enumerate(context):
#                         if hash(gt) == hash(gen):
#                             res = i
#                             break
#                     preds.append(res)
#                     context.clear()
#                 pass
#             else:
#                 context.append(line)
#
#     top_1 = len([x for x in preds if x < 1])
#     top_2 = len([x for x in preds if x < 2])
#     top_3 = len([x for x in preds if x < 3])
#     top_4 = len([x for x in preds if x < 4])
#     top_5 = len([x for x in preds if x < 5])
#     top_10 = len([x for x in preds if x < 10])
#     top_15 = len([x for x in preds if x < 15])
#     top_20 = len([x for x in preds if x < 20])
#
#     print('Top1 : %d/%d = %.4f' % (top_1, len(preds), 100 * top_1 / len(preds)))
#     print('Top2 : %d/%d = %.4f' % (top_2, len(preds), 100 * top_2 / len(preds)))
#     print('Top3 : %d/%d = %.4f' % (top_3, len(preds), 100 * top_3 / len(preds)))
#     print('Top4 : %d/%d = %.4f' % (top_4, len(preds), 100 * top_4 / len(preds)))
#     print('Top5 : %d/%d = %.4f' % (top_5, len(preds), 100 * top_5 / len(preds)))
#     print('Top10 : %d/%d = %.4f' % (top_10, len(preds), 100 * top_10 / len(preds)))
#     print('Top15 : %d/%d = %.4f' % (top_15, len(preds), 100 * top_15 / len(preds)))
#     print('Top20 : %d/%d = %.4f' % (top_20, len(preds), 100 * top_20 / len(preds)))
#     pass

if __name__ == '__main__':
    tokens = javalang.tokenizer.tokenize('assertEquals(true,actual);')
    for token in tokens:
        print(token)

    import numpy as np

    # a = np.matrix([[-1, 0, 0], [0, 1, 2], [0, 2, 5]])
    # print(np.linalg.norm(a, ord=1))
    # print(np.linalg.norm(a, ord=2))
    # print(np.linalg.norm(a, ord=np.inf))
    # print(np.linalg.norm(a, ord='fro'))
    # print(np.linalg.eig(a.T * a)[0])
    # print(a.T * a)
