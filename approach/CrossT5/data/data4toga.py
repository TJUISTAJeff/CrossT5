import os, sys, pickle
import numpy as np

def get_toga_data(file, idx_path, outdir):
    with open(idx_path, 'r', encoding="utf-8") as f:
        valid_idx = f.readlines()
    reader = open(file, 'rb')
    insts = pickle.load(reader, encoding="ISO-8859-1")
    reader.close()
    valid_data = []
    valid_idx = [i.strip() for i in valid_idx]
    valid_idx = set(valid_idx)
    for idx, inst in enumerate(insts):
        if str(idx) in valid_idx:
            valid_data.append(inst)
    f = open(outdir, 'wb')
    pickle.dump(valid_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    print(f"valid indexes : {len(valid_idx)}")
    print(f"finish deal: {len(valid_data)} instances")

def idx_map(trans_map_idx_path, toga_filter_idx, output_path):
    # toga过滤出来的idx映射到没有经过data_transform的pkl上去
    with open(trans_map_idx_path, 'r', encoding="utf-8") as f:
        map_idx = [x.strip().split("\t") for x in f.readlines() if x != "\n"]
    dict_map_idx = {}
    for i in map_idx:
        dict_map_idx[int(i[0])] = int(i[1])
    with open(toga_filter_idx, 'r', encoding="utf-8") as f:
        valid_idx = [int(x.strip()) for x in f.readlines() if x != "\n"]
    idx = [dict_map_idx[i] for i in valid_idx]
    # print(len(map_idx), len(valid_idx), len(idx))
    # print(map_idx[-10:])
    # print(valid_idx[-10:])
    # print(idx[-10:])
    with open(output_path, 'w', encoding="utf-8") as f:
        for i in idx:
            f.write(str(i)+"\n")

if __name__ == "__main__":
    PROJECT_ROOT = "F:\\Code\\Python\\DeepOracle"
    # DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "toga_Transformed_Dataset")
    # IDX_DIR = os.path.join(DATA_DIR, "our_data")
    # for x, y in zip(['test', 'valid', "train"], ['test', 'dev', 'train']):
    #     print(f"begin to deal {x}")
    #     f_pkl = os.path.join(DATA_DIR, f"{y}.pkl")
    #     f_idx = os.path.join(IDX_DIR, f"{x}_valid_idx.txt")
    #     f_out = os.path.join(IDX_DIR, f"toga_{y}.pkl")
    #     get_toga_data(f_pkl, f_idx, f_out)
    DATA_DIR = os.path.join(PROJECT_ROOT, '../approach/datasets', 'Our_Toga_Transformed_Dataset')
    map_idx = os.path.join(DATA_DIR, 'map_idx.txt')
    valid_idx = os.path.join(DATA_DIR, 'our_data', 'our_toga_test_valid_idx.txt')
    output_path = os.path.join(DATA_DIR, 'our_data', "our_toga_mapped_test_valid_idx.txt")
    idx_map(map_idx, valid_idx, output_path)