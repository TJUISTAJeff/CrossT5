
from data.vocab import TokenVocab
from entities.instance import Instance, TokenInstance
from CONSTANTS import *
import sys
sys.path.extend(['.', '..'])

# load global
global_vocab_tokens = set()
with open('datasets/vocabs/global_dict', 'r', encoding='iso8859-1') as reader:
    for line in reader.readlines():
        token, _ = line.strip().split()
        token = token.replace("<space>", " ")
        global_vocab_tokens.add(token)
        if len(global_vocab_tokens) == 1000:
            break

api_vocab_dict = {}
for i in range(4):
    with open('datasets/vocabs/api_' + str(i) + '.json', 'r') as reader:
        jsonObj = json.load(reader)
        for obj in jsonObj['apis']:
            type = obj['type']
            token = obj['name']
            if type not in api_vocab_dict.keys():
                api_vocab_dict[type] = []
            api_vocab_dict[type].append(token)


def pad_seq(seq, maxlen):
    act_len = len(seq)
    if len(seq) < maxlen:
        seq = seq + [0] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq


def load_project_list(file='datasets/ProjectList'):
    projects = []
    with open(file, 'r', encoding='utf-8') as reader:
        for line in reader.readlines():
            projects.append(line.strip())
    return projects


def load_ori_insts(idx, projects):
    print('Process %d started.' % idx)
    base = 'datasets/Processed_Data'
    insts = []
    file_not_found = 0
    for project in projects:
        file = os.path.join(base, project + ".json")
        if os.path.exists(file):
            data, _ = load_json(file)
            insts.extend(data)
        else:
            file_not_found += 1
    output_base = config.middle_res_dir
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    output_file = os.path.join(output_base, 'CompleteData%d.pkl' % idx)
    f = open(output_file, 'wb')
    pickle.dump(insts, f)
    f.close()
    print("Missing %d files." % file_not_found)


def data_splitter(instances, ratios, shuffle=True):
    total_insts = len(instances)
    if len(ratios) == 2:
        if shuffle:
            random.shuffle(instances)

        total_share = ratios[0] + ratios[1]
        inst_per_share = int(total_insts / total_share)
        train = instances[:inst_per_share * ratios[0]]
        test = instances[inst_per_share * ratios[0]:]
        return train, None, test
        pass
    elif len(ratios) == 3:
        if shuffle:
            random.shuffle(instances)
        total_share = ratios[0] + ratios[1] + ratios[2]
        inst_per_share = int(total_insts / total_share)
        train = instances[:inst_per_share * ratios[0]]
        dev = instances[(inst_per_share * ratios[0])
                         :(inst_per_share * (ratios[0] + ratios[1]))]
        test = instances[(inst_per_share * (ratios[0] + ratios[1])):]
        return train, dev, test
        pass
    else:
        print('Error: Unsupported ratios: ' + str(ratios))
        exit(-2)


def load_json(file='datasets/Processed_Data/esjc.json'):
    instances = []
    t = {}
    pattern = re.compile('assert(Equals|Null|NotNull|True|False)')
    missed = {
        'Exceed_max_length': 0,
        'Contains OOV': 0,
        'Unsupported Assertion': 0
    }
    try:
        with open(file, 'r', encoding='utf-8') as reader:
            data = json.load(reader)
        # Remove old version json data files.
        if 'project_static_tokens' not in data.keys():
            data = None
            os.remove(file)
            return instances, 0
        # project_tokens = clear_signature(data['project_static_tokens'].split())
        # project_tokens = project_tokens | clear_signature(data['project_enum_tokens'].split())
        data = data['data']
        for jsonObj in data:
            try:
                assertion = jsonObj['statement']
                res = re.findall(pattern, assertion)
                if len(res) == 1:
                    if not assertion.startswith('assert' + res[0]):
                        if assertion.startswith('Assert.assert' + res[0]):
                            assertion = assertion[7:]
                        elif assertion.startswith('org.junit.Assert.assert' + res[0]):
                            assertion = assertion[17:]
                        else:
                            missed['Unsupported Assertion'] += 1
                            raise NotImplementedError(
                                "Illegal start token for assertion found in %s" % assertion)
                            pass
                else:
                    raise Exception(
                        "More than one assertion found in %s" % assertion)
                context = jsonObj['context']
                tmp_tks = [x for x in context.split() if x not in [
                    ' ', '', '\n']]
                if tmp_tks[-1].startswith("!"):
                    context = ' '.join(tmp_tks[:-1])
                # remove redundant <space>
                assertion = assertion.replace("<space>", " ")
                assertion_tokens = [
                    '<s>'] + [x.value for x in javalang.tokenizer.tokenize(assertion)]
                assert_token_set = set(assertion_tokens[2:])
                assertion_tokens.append('</s>')
                context_tokens = [
                    x.value for x in javalang.tokenizer.tokenize(context)]
                del assertion, context
                token_insts = list()
                local_tokens = set()
                # local_tokens = set()
                # read other dynamic vocab tokens
                local_vocabs = jsonObj['local_vocab']
                for key, value in local_vocabs.items():
                    # if key == 'focal_method' and value == '':
                    #     raise Exception('no focal method')
                    if key == 'import_classes':
                        class_names = [x[1:]
                                       for x in value.strip().split(" <<<delimiter>>> ")]
                        for class_name in class_names:
                            if class_name in api_vocab_dict.keys():
                                api_names = set(
                                    [f"#{class_name}@0${i};" for i in api_vocab_dict[class_name]])
                                local_tokens = local_tokens | api_names
                    else:
                        values = [m.strip() for m in value.split(
                            " <<<delimiter>>> ") if m != ""]
                        local_tokens = local_tokens | set(values)
                # local_tokens = local_tokens | project_tokens
                context_tokens = set(context_tokens)
                local_tokens = local_tokens | global_vocab_tokens
                local_token_names = set()
                for i in local_tokens:
                    tmp_inst = TokenInstance(i)
                    token_insts.append(tmp_inst)
                    # if tmp_inst.name in local_token_names:
                    #     print(i, tmp_inst.name)
                    local_token_names.add(tmp_inst.name)
                # a = len(local_token_names)
                # b = len(token_insts)
                # c = len(local_tokens)
                # if a!=b or a!=c:
                #     print(a, b, c)
                context_left_tokens = context_tokens - local_token_names
                local_token_names = local_token_names | context_left_tokens
                for i in context_left_tokens:
                    tmp_inst = TokenInstance(i)
                    token_insts.append(tmp_inst)
                token_insts = set(token_insts)
                # e = len(token_insts)
                # f = len(local_token_names)
                # if e!=f:
                #     print(e, f)
                left_tokens = assert_token_set - local_token_names
                if len(token_insts) < 3000:
                    if len(left_tokens) == 0:
                        token_vocab = TokenVocab(token_insts)
                        del local_tokens, token_insts
                        inst = Instance(assertion_tokens,
                                        context_tokens, token_vocab)
                        instances.append(inst)
                        pass
                    else:
                        # print(left_tokens)
                        missed['Contains OOV'] += 1
                        raise NotImplementedError('Contains OOV.')
                else:
                    missed['Exceed_max_length'] += 1
                    raise NotImplementedError(
                        'Exceed max vocab length limitation.')
            except Exception:
                # traceback.print_exc()
                continue

    except UnicodeDecodeError:
        print('Error: Encoding UTF-8 does not fit. File %s' % file)
        exit(-1)
    except FileNotFoundError:
        print('Error: File %s does not found, please check' % file)
        exit(-2)
    except Exception as exception:
        print('Error: Unexpceted exception %s has occurred when processing file %s, please check.' % (
            file, exception))
        return instances, 0
    finally:
        print(
            'Finished parsing file %s, found %d instances in total. %s.' % (file, len(instances), str(missed)))
        gc.collect()
    return instances, missed
