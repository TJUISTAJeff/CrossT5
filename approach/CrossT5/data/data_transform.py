import os
import pickle
from tqdm import *
import javalang
PROJECT_ROOT = "F:\\Code\\Python\\DeepOracle"

def data_tansfer(output_base, src_pkl, tgt_folder):
    # output_base = os.path.join(PROJECT_ROOT, 'datasets/Toga_Transformed_Dataset')
    # for src_pkl, tgt_folder in zip(['train.pkl', 'dev.pkl', 'test.pkl'], ['Training', 'Eval', 'Testing']):
    with open(os.path.join(PROJECT_ROOT, '../approach/datasets', src_pkl), 'rb') as reader:
        instances = pickle.load(reader)
    output_folder = os.path.join(output_base, tgt_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    testMethodWriter = open(os.path.join(output_folder, 'testMethods.txt'), 'w', encoding='utf-8')
    assertLinesWriter = open(os.path.join(output_folder, 'assertLines.txt'), 'w', encoding='utf-8')

    found = 0
    datas = []
    map_idx = []
    for idx, inst in tqdm(enumerate(instances), desc='Transferring...'):
        try:
            assertLine = ' '.join([x for x in inst.assertion if x != '\n' and x != '<SOS>']).encode('utf-8').decode('utf-8')
            context = ' '.join([x for x in inst.context_tokens if x != '\n']).encode('utf-8').decode('utf-8')
            padded_cls = 'class A { ' + ' '.join(inst.context_tokens) + ' }'
            tree = javalang.parse.parse('class A { ' + ' '.join(inst.context_tokens) + ' }')
            if len(tree.types[0].body) == 2:
                if len(tree.types[0].body[1].annotations) != 0:
                    fm_start_loc = tree.types[0].body[1].annotations[0].position.column - 1
                else:
                    fm_start_loc = tree.types[0].body[1].position.column - 1
                tm_start_loc = tree.types[0].body[0].position.column - 1
                tm_end_loca = padded_cls[tm_start_loc:fm_start_loc].rfind('}')
                if tm_end_loca != -1:
                    tm = padded_cls[tm_start_loc:(tm_start_loc + tm_end_loca + 1)]
                    fm = padded_cls[fm_start_loc:-1]
                    context = (tm + " \"<AssertPlaceHolder>\" ; } " + fm).encode('utf-8').decode('utf-8')
                    found += 1
            testMethodWriter.write(context + '\n')
            assertLinesWriter.write(assertLine + '\n')
            datas.append(inst)
            map_idx.append(idx)
        except:
            continue
    print(found)
    print(len(datas))
    testMethodWriter.close()
    assertLinesWriter.close()
    with open(os.path.join(output_base, "map_idx.txt"), "w", encoding="UTF-8") as f:
        for idx, i in enumerate(map_idx):
            f.write(str(idx) + "\t" + str(i)+"\n")
    f = open(os.path.join(output_base, src_pkl), 'wb')
    pickle.dump(datas, f, protocol=pickle.HIGHEST_PROTOCOL)
    # 写入
    f.close()
    pass


if __name__ == '__main__':
    data_tansfer(os.path.join(PROJECT_ROOT, 'datasets/Our_Toga_Transformed_Dataset'), "our_toga_test.pkl", "our_toga_test")
