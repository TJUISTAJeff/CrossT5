bleus = []
ranks = []
for i in range(2):
    bleu = eval(open('bleu%d.txt'%i).read())
    rank = eval(open('rank%d.txt'%i).read())
    bleus.extend(bleu)
    ranks.extend(rank)
    break
import numpy as np
print('bleu', np.mean(bleus))
print('top1 %.4f'%(ranks.count(0)/ len(bleus)))
print('top3 %.4f'%(sum([1 if x < 3 else 0 for x in ranks]) / len(bleus)))
print('top5 %.4f'%(sum([1 if x < 5 else 0 for x in ranks]) / len(bleus)))