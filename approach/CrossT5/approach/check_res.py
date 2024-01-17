import random
import sys

sys.path.extend(['.', '..'])
from CONSTANTS import *

gt_reader = open(os.path.join(PROJECT_ROOT, 'datasets/generations/Beam-5/ground_truth.txt'), 'r', encoding='iso8859-1')
prediction_reader = open(os.path.join(PROJECT_ROOT, 'datasets/generations/Beam-5/predictions_5.txt'), 'r',
                         encoding='iso8859-1')
insts_reader = open(os.path.join(PROJECT_ROOT, 'datasets/generations/Beam-5/raw_insts.txt'), 'rb')
test_insts = pickle.load(insts_reader)
insts_reader.close()

total = 0
corrected = 0
different = 0
false_predicted_when_no_oov = 0
gts = []
prediction_groups = []


def check_ooc(instance):
    assert_tokens = set(instance.assertion[1:-1])
    context_tokens = set(instance.context_tokens)
    left = assert_tokens - context_tokens
    if len(left) == 0:
        return False
    else:
        return True


def check_oov(instance):
    for token in set(instance.assertion[1:-1]):
        if instance.local_vocab.token2id(token) == 1:
            return True
    return False


while True:
    gt = gt_reader.readline().strip()
    total += 1
    if not gt:
        break
    predictions = []
    while True:
        prediction_line = prediction_reader.readline()
        if prediction_line.strip() == '':
            break
        else:
            predictions.append(prediction_line.strip())
    flag = False
    prediction_groups.append(predictions)
    gts.append(gt)

output_writer = open(os.path.join(PROJECT_ROOT, 'datasets/generations/Beam-5/overall.txt'), 'w', encoding='iso8859-1')
total = 0
corrected = 0
contains_oov = 0

no_ooc = 0
corrected_no_ooc = 0
contains_ooc = 0
corrected_ooc = 0
samples = list(range(len(test_insts)))
samples = random.sample(samples, 400)

j = 0
for i in range(len(test_insts)):
    inst = test_insts[i]
    gt = gts[j]
    if gt != ' '.join(inst.assertion):
        continue
    total += 1
    isooc = check_ooc(inst)
    isoov = check_oov(inst)
    if isooc:
        contains_ooc += 1
    else:
        no_ooc += 1
    if isoov:
        contains_oov += 1
    predictions = prediction_groups[j]
    for prediction in predictions:
        if hash(prediction) == hash(gt):
            corrected += 1
            if isooc:
                corrected_ooc += 1
            else:
                corrected_no_ooc += 1
            break
        # break
    if i in samples:
        try:
            output_writer.write('Context: ' + ' '.join(inst.context_tokens) + '\n')
        except UnicodeError as e:
            output_writer.write('Context: ' + 'Unable to output. \n')
        try:
            output_writer.write('Vocab: ' + ' '.join(inst.local_vocab._id2token) + '\n')
        except UnicodeError as e:
            output_writer.write('Vocab: ' + 'Unable to output. \n')
        output_writer.write('Assertion: ' + gt + '\n')
        output_writer.write('Predictions: ***' + '\n')
        for prediction in predictions:
            output_writer.write(prediction + '\n')
        output_writer.write('             ***' + '\n\n')
    j += 1
output_writer.close()

print('Total acc: %d / %d = %.6f' % (corrected, total, corrected / total))
print('In context acc: %d / %d = %.6f' % (corrected_no_ooc, no_ooc, corrected_no_ooc / no_ooc))
print('Out of context acc: %d / %d = %.6f' % (corrected_ooc, contains_ooc, corrected_ooc / contains_ooc))
print('Out of vocabulary ratio: %d / %d = %.6f' % (contains_oov, total, contains_oov / total))
