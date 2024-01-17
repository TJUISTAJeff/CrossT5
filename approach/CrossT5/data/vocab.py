import pickle


class CharVocab():
    def __init__(self):
        self._id2char = [
            '<pad>', '<unk>', '<space>', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '@',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            ':', ';', '<', '=', '>', '?', '@',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z',
            '[', '\\', ']', '^', '_', '`',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z',
            '{', '|', '}', '~'
        ]
        self._char2id = {char: idx for idx, char in enumerate(self._id2char)}

    def id2char(self, id):
        if id < len(self._id2char):
            return self._id2char[id]
        else:
            return '<unk>'

    def char2id(self, char):
        if char in self._char2id.keys():
            return self._char2id[char]
        else:
            return 1

    @property
    def vocab_size(self):
        return len(self._id2char)


class AssertionVocab():
    def __init__(self):
        self._stopIds = [0, 1]
        self._PAD = 0
        self._UNK = 1
        self._id2Assertion = {
            0: 'assertEquals',
            1: 'assertTrue',
            2: 'assertFalse',
            3: 'assertNull',
            4: 'assertNotNull',
        }
        self._assertion2Id = {v: k for k, v in self._id2Assertion.items()}

    def id2Assertion_type(self, id):
        return self._id2Assertion[id]

    def assertion_type2id(self, assertion_type):
        return self._assertion2Id[assertion_type]


class DynamicVocab():
    def __init__(self, vocab_tokens):
        self._PAD = '<pad>'
        self._UNK = '<unk>'
        self._END = '</s>'
        self._BOS = '<s>'
        self.accepted_assert_types = [
            'assertEquals', 'assertNull', 'assertNotNull', 'assertTrue', 'assertFalse']
        self._id2token = [self._PAD, self._BOS, self._UNK,
                          self._END] + self.accepted_assert_types
        self._pre_defined_ids = len(self._id2token)
        self._id2token.extend(vocab_tokens)
        self._token2id = {token: id for id, token in enumerate(self._id2token)}

    def token2id(self, token):
        if token in self._token2id.keys():
            return self._token2id[token]
        else:
            return self._token2id[self._UNK]

    def id2token(self, id):
        if id < self.vocab_size:
            return self._id2token[id]
        else:
            return self._UNK

    def update(self, tokens):
        for token in tokens:
            if token not in self._token2id.keys():
                self._id2token.append(token)
            else:
                pass
        self._token2id = {token: id for id, token in enumerate(self._id2token)}

    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self._id2token)

    @property
    def tokens(self):
        return self._id2token

    @property
    def padding(self):
        return self._PAD

    @property
    def eos(self):
        return self._END

    @property
    def bos(self):
        return self._BOS

    @property
    def unk(self):
        return self._UNK

    @property
    def unk_id(self):
        return 1

    def contains(self, token):
        return token in self._token2id.keys()


class TokenVocab(DynamicVocab):
    def __init__(self, vocab_tokens):
        self.token_map = {}
        self.vocab_tokens = vocab_tokens
        self.separators = {'(', ')', '.', ',', '[', ']', '{', '}', ';'}
        self.operators = set(list({'+': 8501, '<': 2974, '>': 2433, '==': 1554, '::': 995, '->': 764, '-': 649, '!=': 597, '*': 466, '>=': 278,
                             '<=': 226, '<<': 36, '/': 216, '&&': 139, '++': 129, '%': 102, '||': 66, '=': 35, '?': 22, ':': 19, '&': 18, '|': 14, '--': 13, '^': 2}.keys()))
        
        tokens, self.token_map, self.type_map = self.data_filter(vocab_tokens) 
        super().__init__(tokens)

    def data_filter(self, vocab_tokens):
        tokens = list()
        token_map = {}
        type_map = {}
        for i in vocab_tokens:
            name = i.name
            if name not in token_map.keys():
                token_map[name] = [i]
            else:
                token_map[name].append(i)
            tokens.append(name)
            type = i.belonged_type
            if type == None:
                type = "None"
            if type not in type_map.keys():
                type_map[type] = [i]
            else:
                type_map[type].append(i)
        return set(tokens), token_map, type_map

    def name2insts(self, name):
        return self.token_map.get(name,[])

    def type2tokens(self, type):
        return self.type_map.get(type,[])

    def get_valid_token_ids(self, token_inst):
        if token_inst is None:
            # 如果没有指定token的话（多半是初始token），返回None，表示所有的token都要纳入考量
            return None
        else:
            # 如果有指定类型的token，根据类型来进行判断
            # self.is_type = False
            # self.is_method = False
            # self.is_param = False
            # self.is_keyword = False
            # self.is_literal = False

            if token_inst.is_param or token_inst.is_method:
                # 首先去掉Identifier
                after_remove_identifier = []
                for x in self.tokens:
                    if not(x.is_param or x.is_method or x.is_type):
                        after_remove_identifier.append(x)

                if token_inst.is_param:
                    # 其次，如果是变量的话，给operators 和 instanceof 加上type约束
                    after_operator_limits = []
                    after_operator_limit_ids = []
                    for x in after_remove_identifier:
                        if x.name in self.operators:
                            new_inst = pickle.loads(pickle.dumps(x))
                            new_inst.return_type = token_inst.return_type
                            after_operator_limits.append(x)
                            after_operator_limit_ids.append(self.token2id(x.name))
                            pass
                        elif x.name == 'instanceof' or x.name =='.':
                            new_inst = pickle.loads(pickle.dumps(x))
                            new_inst.return_type = token_inst.return_type
                            after_operator_limits.append(x)
                            after_operator_limit_ids.append(self.token2id(x.name))
                            pass
                        else:
                            after_operator_limits.append(x)
                            after_operator_limit_ids.append(self.token2id(x.name))
                
                else:
                    after_operator_limits = after_remove_identifier
                    for x in after_operator_limits:
                        after_operator_limit_ids.append(self.token2id(x.name))

                return after_operator_limits, after_operator_limit_ids

            elif token_inst.is_type:
                valid_tokens = {'&&', ')', '<', '||', '.', ',', '['}
                # assertTrue ( "Coercing to BINARY failed for PDataType " + p , obj instanceof byte [ ] ) ;
                # assertTrue ( it . getExported ( ) instanceof RemoteIterator < ? > ) ;
                ret_insts, ret_ids = [], []
                filtered_token_insts = [
                    inst for inst in self.vocab_tokens if inst.name in valid_tokens]
                for inst in filtered_token_insts:
                    if inst.name == '.':
                        copy_inst = pickle.loads(pickle.dumps(inst))
                        copy_inst.return_type = token_inst.name
                        copy_inst.requires_static = True
                        ret_insts.append(copy_inst)
                        ret_ids.append(self.token2id(copy_inst.name))
                    else:
                        ret_insts.append(inst)
                        ret_ids.append(self.token2id(inst.name))
                return ret_insts, ret_ids

            elif token_inst.is_keyword:
                required_type = token_inst.return_type
                if required_type is None:
                # 并不知道有没有什么约束条件
                    return None
                if token_inst.name == 'instanceof':
                    # Types only
                    ret_insts, ret_ids = [], []
                    filtered_token_insts = [
                        inst for inst in self.vocab_tokens if inst.is_type and inst.belonged_type == required_type]
                    for inst in filtered_token_insts:
                        ret_insts.append(inst)
                        ret_ids.append(self.token2id(inst.name))
                    return ret_insts, ret_ids

                elif token_inst.name == '.':
                    # required_type = token_inst.return_type
                    if token_inst.requires_static:
                        filtered_token_insts = [
                            inst for inst in self.vocab_tokens if inst.is_static]
                        
                    ret_insts, ret_ids = [], []
                    filtered_token_insts = [inst for inst in self.vocab_tokens if
                                            inst.belonged_type == required_type or inst.name == 'class']
                    for inst in filtered_token_insts:
                        ret_insts.append(inst)
                        ret_ids.append(self.token2id(inst.name))
                    return ret_insts, ret_ids
                        

                else:
                    # TODO: 添加其他类型的keyword的约束：
                    pass
            
            elif token_inst.is_literal:
                # 仅考虑后接operator, separator, 或者 instanceof （可以直接完整生成了感觉），当后接为 . 时，记录return_type
                filtered_tokens = []
                filtered_ids =[]
                for inst in self.tokens:
                    if inst.name in self.operators or inst.name in self.separators:
                        if inst.name =='.':
                            cpy_inst = pickle.loads(pickle.dumps(inst))
                            cpy_inst.return_type = token_inst.return_type
                            filtered_tokens.append(cpy_inst)
                            filtered_ids.append(self.token2id(cpy_inst.name))
                            pass
                        else:
                            filtered_tokens.append(inst)
                            filtered_ids.append(self.token2id(inst.name))
                    elif inst.name == 'instanceof':
                        cpy_inst = pickle.loads(pickle.dumps(inst))
                        cpy_inst.return_type = token_inst.return_type
                        filtered_tokens.append(cpy_inst)
                        filtered_ids.append(self.token2id(cpy_inst.name))
                    else:
                        pass
                
                return filtered_tokens, filtered_ids
            else:
                return None


if __name__ == '__main__':
    l = []
    l.append(None)
    print(l)
