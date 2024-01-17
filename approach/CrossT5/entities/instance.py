import json
import os, re
import sys

import torch
from torch.autograd import Variable

keyword = {'transitive', '}', 'float', '[', '&&', 'void', '*', 'double', 'char', '+', '^', ']', 'long', '/',
            'throw', ':', '@', '>', '!', '{', 'this', '=', '&', '.', 'try', 'class', 'boolean', '(', 'return', '<=',
            'byte', '>=', 'catch', 'if', 'new', 'short', '||', 'super', ';', '-', '%', '!=', '$', 'instanceof',
            'int', ')', 'final', '|', '++', 'public', '<<', '<', '=='}
literal_type = {"String", "Character", "Boolean", "Int", "Float", "Null"}
bool_value = {"true", "false", "null"}

class Instance():
    def __init__(self, assertion_tokens, context_tokens, vocab):
        self.assertion = assertion_tokens
        self.context_tokens = context_tokens
        self.local_vocab = vocab
        self.vocab_len = len(vocab)
        self.context_len = len(context_tokens)
        self.assertion_len = len(assertion_tokens)
        self.target = ''

    def set_decoder_target(self, complete):
        self.target = complete

    def update_vocab(self, tokens):
        self.local_vocab.update(tokens)
        self.vocab_len = self.local_vocab.vocab_size

    def __len__(self):
        return len(self.assertion)

    def count_oov(self):
        contains_oov = False
        num_oov_steps = 0
        for token in self.assertion[1:]:
            if self.local_vocab.token2id(token) == self.local_vocab.token2id(self.local_vocab.unk):
                contains_oov = True
                num_oov_steps += 1
        return contains_oov, num_oov_steps

    def contains_oov(self):
        for token in self.assertion[1:]:
            if self.local_vocab.token2id(token) == self.local_vocab.unk_id:
                return True
        return False


class TensorInstance():
    def __init__(self, batch_size, input_seq_len, queries_seq_len, vocab_len, char_seq_len):
        self.input_context_char_seq = Variable(torch.LongTensor(batch_size, input_seq_len, char_seq_len).zero_(),
                                               requires_grad=False)
        self.input_context_mask = Variable(torch.BoolTensor(batch_size, input_seq_len).fill_(1), requires_grad=False)
        self.input_queries_char_seq = Variable(torch.LongTensor(batch_size, queries_seq_len, char_seq_len).zero_(),
                                               requires_grad=False)
        self.input_queries_mask = Variable(torch.LongTensor(batch_size, queries_seq_len).zero_(), requires_grad=False)
        self.input_vocab_char_seq = Variable(torch.LongTensor(batch_size, vocab_len, char_seq_len).zero_(),
                                             requires_grad=False)
        self.input_vocab_mask = Variable(torch.LongTensor(batch_size, 1, vocab_len).zero_(),
                                         requires_grad=False)
        self.targets = Variable(torch.LongTensor(batch_size, queries_seq_len).zero_(), requires_grad=False)
        self.antiMask = None

    @property
    def inputs(self):
        return self.input_context_char_seq, self.input_context_mask, \
               self.input_queries_char_seq, self.input_queries_mask, \
               self.input_vocab_char_seq, self.input_vocab_mask, \
               self.antiMask

    @property
    def outputs(self):
        return self.targets

    def to_cuda(self, device):
        self.input_context_char_seq = self.input_context_char_seq.to(device)
        self.input_queries_char_seq = self.input_queries_char_seq.to(device)
        self.input_context_mask = self.input_context_mask.to(device)
        self.input_queries_mask = self.input_queries_mask.to(device)
        self.input_vocab_char_seq = self.input_vocab_char_seq.to(device)
        self.input_vocab_mask = self.input_vocab_mask.to(device)
        self.targets = self.targets.to(device)
        self.antiMask = self.antiMask.to(device)


class TokenInstance:

    def __init__(self, token):
        # initialize
        self.token = token
        self.belonged_type = None
        self.type = None
        self.return_type = None
        self.params = list()
        self.name = None
        self.is_type = False
        self.is_method = False
        self.is_param = False 
        self.is_keyword = False
        self.is_literal = False
        self.is_api = False
        self.is_static = False
        self.requires_static = False


        if '#' in token and '@' in token and ';' in token and not "&" in token or token.startswith("@") and len(
                token) > 1:
            # method
            self.clean_method(token)
        elif '#' in token and '@' in token and not ';' in token and not "&" in token:
            # variable
            self.clean_variable(token)
        elif '#' in token and '@' not in token and not ';' in token and not "&" in token:
            self.clean_type(token)
        elif '&' in token:
            self.clean_literal(token)
        else:
            self.clean_keyword(token)

    def __str__(self):
        token = f"token:\t{self.token}\nbelonged_type: {self.belonged_type}\nreturn type: {self.return_type}\n"
        token += f"type: {self.type}\n"
        token += f"name: {self.name}\nparams: {self.params}\nis_param: {self.is_param}\nis_keyword: {self.is_keyword}\n"
        token += f"is_method: {self.is_method}\nis_literal: {self.is_literal}\nis_type: {self.is_type}\nis_static: {self.is_static}\n"
        token += f"is_api: {self.is_api}"
        return token

    def clean_keyword(self, token: str):
        self.name = token
        global keyword, bool_value
        if token[0] == "\"":
            self.is_literal = True
            self.type = "String"
            self.belonged_type = self.type
            self.return_type = self.type
        elif token[0] == "\'":
            self.is_literal = True
            self.type = "Character"
            self.belonged_type = self.type
            self.return_type = self.type
        elif token in keyword:
            self.is_keyword = True
        elif token in bool_value:
            self.is_literal = True
            self.type = "Boolean"
            self.belonged_type = self.type
            self.return_type = self.type
        else:
            try:
                if '.' in token:
                    value = float(token)
                    self.type = "Float"
                else:
                    value = int(token)
                    self.type = "Int"
                self.belonged_type = self.type
                self.return_type = self.type
                self.is_literal = True
            except:
                self.is_param = True

    def clean_literal(self, token):
        global literal_type
        pattern = re.compile("(.*?)&(.*)")
        m = pattern.match(token)
        type = m.group(1)
        self.name = m.group(2)
        if not type in literal_type:
            type = None
        self.type = type
        self.return_type = self.type
        self.belonged_type = self.type
        self.is_literal = True

    def clean_type(self, token: str):
        self.is_type = True
        self.name = token.split('.')[-1].strip()
        self.type = self.name
        self.return_type = self.name
        self.belonged_type = self.type

    def clean_method(self, token: str):
        self.is_method = True
        if token.startswith('@'):
            pattern = re.compile("@\d\$(.*?);")
            m = pattern.match(token)
            self.name = m.group(1)
        else:
            token = token.strip()
            if "\n" in token or "\r\n" in token or "\r" in token:
                data = token.split()
                token = [i.strip() for i in data if i != ""]
                token = token[-1]
            pattern = re.compile("(.*)#(.*)@(\d)\$(.*?);(.*)")
            m = pattern.match(token)
            return_type = m.group(1)
            if not return_type == None:
                return_type = re.sub("<.*>", "<>", return_type)
            self.return_type = return_type
            belonged_type = m.group(2)
            if belonged_type.startswith("java"):
                self.is_api = True
            self.belonged_type = self.deal_belonged_type(belonged_type)
            self.is_static = int(m.group(3)) == 1
            self.name = m.group(4)
            params = m.group(5)
            if not params == "":
                params = params.split(":")
                for i in params:
                    if i != "":
                        if '<' in i:
                            para = re.sub("<.*?>?>", "<>", i.strip())
                        else:
                            para = i
                        self.params.append(para.strip())
        self.is_method = True

    def clean_variable(self, token: str) -> None:
        pattern = re.compile("(.*?)#(.*?)@(\d)\$(.*)")
        m = pattern.match(token)
        self.return_type = re.sub("<.*>", "<>", m.group(1))
        self.belonged_type = self.deal_belonged_type(m.group(2))
        self.is_static = int(m.group(3)) == 1
        self.name = m.group(4)
        self.type = self.return_type
        self.is_param = True

    @property
    def value(self):
        return self.name

    def deal_belonged_type(self, belonged_type) -> str:
        type = re.sub("<.*?>?>", "<>", belonged_type)
        type = type.strip().split(".")[-1]
        return type.strip()


if __name__ == "__main__":
    data_dir = "F:\\Code\\Java\\ATLAS_Ext\\datasets\\Processed_Data\\ATLAS"
    for i in os.listdir(data_dir):
        import time
        print(f"============================Project {i}====================")
        a = time.time()
        # short running
        data_path = os.path.join(data_dir, i)
        test_data_path = open(data_path)
        dat = json.load(test_data_path)
        test_data_path.close()
        data = dat['data']
        print(f"Assertion counts: {len(data)}")
        vocabs = []
        tokens = []
        for i in data:
            local_vocab = i['local_vocab']
            for key, value in local_vocab.items():
                vocabs.append(value)
        for j in vocabs:
            # pattern = re.compile(r'\S+?\"[\S ]*?\"|\S+?\'[\S ]*?\'|[\S]*')
            # split_data = re.findall(pattern, j)
            # tokens=[m.strip() for m in split_data if m!=""]
            tokens.extend([m.strip() for m in j.split(" <<<delimiter>>> ") if m != ""])
        tokens = set(tokens)
        for k in tokens:
            try:
                x = TokenInstance(k)
            except:
                print(k)
                    # print(j)
                    # print(i['context'])
                    # print(data_path)
                continue
        b = time.time()
        seconds = b - a
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print("%02d:%02d:%02d" % (h, m, s))


    # data_path = os.path.join(data_dir, "ceylon.json")
    # test_data_path = open(data_path)
    # dat = json.load(test_data_path)
    # test_data_path.close()
    # data = dat['data']
    # vocabs = []
    # for i in data:
    #     local_vocab = i['local_vocab']
    #     for key, value in local_vocab.items():
    #         vocabs.append(value)
    #     for j in vocabs:
    #         # pattern = re.compile(r'\S+?\"[\S ]*?\"|\S+?\'[\S ]*?\'|[\S]*')
    #         # split_data = re.findall(pattern, j)
    #         # tokens=[m.strip() for m in split_data if m!=""]
    #         tokens = [m.strip() for m in j.split(" <<<delimiter>>> ") if m!=""]
    #         for k in tokens:
    #             try:
    #                 a = TokenInstance(k)
    #             except:
    #                 print(k)
    #                 # print(j)
    #                 # print(i['context'])
    #                 # print(data_path)
    #                 break
    # test cases for method
    m = ["a#b.c@1$d;", "#a.b@0$c;d", "#a.b@1$c;d:e:f", "a#b.c@0$d;e:f", "@a"]
    v = ["a#b.c@0$d", "a#b.c.d.e@1$f"]
    c = ["#main.b", "#a.b.c.d"]
    l = ["Boolean&true", "Boolean&false", "Int&114", "Float&114.5", "String&a", "Null&null", "Character&a"]
    k = ["+", "-", "*", "/", ".class", "}", "{", "(", ")", "void", "public", "static", "=", "==", "!", "default"]
    b = ['null', 'true', 'false', '@']
    x = ['String&"~!@#$%^&*()',
         "ValueConstructorDeclaration$impl#org.eclipse.ceylon.compiler.java.runtime.metamodel.decl.ValueConstructorDeclarationImpl@0$$ceylon$language$meta$declaration$ValueConstructorDeclaration$impl;114:514:1919810",
         "a<djksl,kfdlsdf<ab,cdes,dsfdl>>#b.c.d.s<dd,e<ss,ds>>@0$ad;kfdsl<dkls>:fkdls<kdfjls,fjdkls>:kfd<kdfs,<kdfs,jfkds>>", "dkfs#java.dk@0$dfslf;kfds:fjds"]
    for i in x:
        a = TokenInstance(i)
        print(a)
