import operator
import os.path
import sys

sys.path.extend([".", ".."])
import pickle


def change2atlas(file, output_dir) -> None:
    reader = open(os.path.join(file), 'rb')
    insts = pickle.load(reader, encoding="ISO-8859-1")
    testmethods = []
    assertlines = []
    asserts = ['assertTrue', 'assertFalse', 'assertEquals', 'assertNull', 'assertNotNull']
    for i in insts:
        if len(i.local_vocab) >= 3000:
            # print(i)
            continue

        testmethods.append(str.join(" ", i.context_tokens))
        assertion = i.assertion
        assertion.pop()
        all_tokens = set(i.context_tokens)|set([x.split("@")[1] if "@" in x else x for x in i.local_vocab.tokens])
        oov_token = set(assertion) - all_tokens
        if len(oov_token) > 0:
            if len(oov_token)==1 and list(oov_token)[0] in asserts:
                pass
            else:
                continue
        assertlines.append(str.join(" ", assertion))
    print(f"total test instances: {len(insts)}")
    print(f"vocab below 3000 instances: {len(assertlines)}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    wrong_method_index = []
    with open(os.path.join(output_dir, "testMethods.txt"), 'w+', encoding="UTF-8") as f:
        for i in range(len(testmethods)):
            try:
                f.write(testmethods[i] + "\n")
            except:
                wrong_method_index.append(i)
                continue
    wrong_assert_index = []
    with open(os.path.join(output_dir, "assertLines.txt"), 'w+', encoding="UTF-8") as f:
        for i in range(len(assertlines)):
            try:
                f.write(assertlines[i] + "\n")
            except:
                wrong_assert_index.append(i)
                continue
    wrong_index = set(wrong_method_index) | set(wrong_assert_index)
    print(f"right coded instances:{len(assertlines) - len(wrong_index)}")
    # print(
    #     f"wrong test methods count: {len(wrong_method_index)}\nwrong test assertion count: {len(wrong_assert_index)}\n" f"total wrong count: {len(set(wrong_method_index) | set(wrong_assert_index))}\trate: {100 * len(set(wrong_method_index) | set(wrong_assert_index)) / len(testmethods):7.2f}%")
    with open(os.path.join(output_dir, "testMethods.txt"), 'w+', encoding="UTF-8") as f:
        for i in range(len(testmethods)):
            if i in wrong_index:
                continue
            f.write(testmethods[i] + "\n")
    with open(os.path.join(output_dir, "assertLines.txt"), 'w+', encoding="UTF-8") as f:
        for i in range(len(assertlines)):
            if i in wrong_index:
                continue
            f.write(assertlines[i] + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        source_path = sys.argv[1]
        target_path = sys.argv[2]
    else:
        source_path = "F:\\Asuna\\Desktop\\datasets"
        target_path = "F:\\Asuna\\Desktop\\datasets"
    dev_file_path = os.path.join(source_path, "dev.pkl")
    train_file_path = os.path.join(source_path, "train.pkl")
    test_file_path = os.path.join(source_path, "test.pkl")
    dev_output_path = os.path.join(target_path, "Eval")
    train_output_path = os.path.join(target_path, "Training")
    test_output_path = os.path.join(target_path, "Test")
    change2atlas(test_file_path, test_output_path)
    # change2atlas(train_file_path, train_output_path)
    # change2atlas(dev_file_path, dev_output_path)
