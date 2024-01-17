import copy
import os.path
import re
import sys
import traceback

sys.path.extend(['.', '..'])
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoTokenizer

# from data.vocab import separators, operators
from entities.instance import TokenInstance, keyword

tokenizer = AutoTokenizer.from_pretrained('../approach/codet5-small')


def equivalent(type1, last_type):
    # 数字类型统一可以相等
    numbers = ['int', 'long', 'float', 'double']
    booleans = ['boolean', 'bool', 'Boolean']
    if last_type is None or last_type == [None]:
        return True
    if type1 is None or type1 == [None]:
        return True

    if isinstance(type1, list) and len(type1) == 1:
        type1 = type1[0]
    if isinstance(last_type, list) and len(last_type) == 1:
        last_type = last_type[0]

    if isinstance(type1, str) and isinstance(last_type, str):
        if type1 == last_type or type1.lower() == last_type.lower():
            return True
        if type1.lower() in numbers and last_type.lower() in numbers:
            return True
        if type1.lower() in booleans and last_type.lower() in booleans:
            return True

    if '<T>' in last_type or 'T' in last_type:
        return True
    if type1 == "<T>" or type1 == 'T' or type1 == 'Object':
        return True
    if type1 == '<no_param>':
        return True
    if isinstance(type1, str) and isinstance(last_type, str) and type1 in numbers and last_type in numbers:
        return True
    if isinstance(type1, str) and type1.lower() in numbers:
        for t in last_type:
            if t.lower() in numbers:
                return True
    if last_type == 'number':
        if isinstance(type1, str):
            if type1.lower() in numbers:
                return True
        elif isinstance(type1, list):
            for type in type1:
                if type.lower() in numbers:
                    return True
        return False
    if type1 in last_type:
        return True
    if type1.lower() in [t.lower() for t in last_type]:
        return True
    if len(last_type) == 1:
        if last_type[0].endswith(type1):
            return True
    return False


def has_common_element(list1, list2):
    numbers = ['int', 'long', 'float', 'double']
    if list1 == [None] or list2 == [None]:
        return True
    if 'T' in list1 or 'T' in list2:
        return True
    if 'Object' in list1 or 'Object' in list2:
        return True
    tmp1 = []
    tmp2 = []
    for type in list1:
        new_type = type
        res = re.search('<T>', type)
        if res:
            start_loc = res.span()[0]
            new_type = type[:start_loc]
        if type in numbers or type.lower() in numbers:
            new_type = 'number'
        tmp1.append(new_type)
    for type in list2:
        new_type = type
        res = re.search('<T>', type)
        if res:
            start_loc = res.span()[0]
            new_type = type[:start_loc]
        if type in numbers or type.lower() in numbers:
            new_type = 'number'
        tmp2.append(new_type)

    return len(set(tmp1).intersection(set(tmp2))) > 0


def check_type_constrain(last_type, req_type):
    return False


def copy_and_update(node, token_inst):
    copynode = pickle.loads(pickle.dumps(node))
    copynode.update(token_inst.name, token_inst)
    return copynode


def update_without_token_inst(node, idx):
    new_beams = []
    tok = node.local_vocab.id2token(idx)
    possible_tokens = node.local_vocab.name2insts(tok)
    for token_inst in possible_tokens:
        new_beam = check_type(node, token_inst)
        if new_beam is not None:
            new_beams.append(new_beam)
    return new_beams


def check_type(node, token_inst):
    tok = token_inst.name

    if token_inst.is_api:
        # 如果当前生成的token是api类型
        # 部分要求继承is_method
        node.force_left_brace = True
        node.stack.append('<api>')
        node.stack.append(None)  # return type
        node.stack.append("*")  # param type
        copynode = copy_and_update(node, token_inst)
        return copynode

    elif token_inst.is_method:
        # 判断函数调用是否合法
        # 如果前一个token是None，则直接update，不管其他
        if node.last_token is None:
            pass
        else:
            # 如果前一个token不是 . ，且不是api的情况下，这个函数调用是无效的
            if node.last_token.name != '.':
                if not token_inst.is_local:
                    return None
                else:
                    pass

            # 如果前一个是 . ，检查一下类型
            elif node.last_token.name == '.':
                if not equivalent(token_inst.belonged_type, node.last_token.return_type):
                    return None
                else:
                    pass

        # 如果是函数调用，下一个token强制为左括号
        node.force_left_brace = True

        # 然后入栈type信息
        node.stack.append("<m>")
        node.stack.append(token_inst.return_type)
        if len(token_inst.params) != 0:
            # params 非空，里面存的是各个参数值的类型
            # 逆序入栈
            for i in reversed(token_inst.params):
                node.stack.append(i)

        copynode = pickle.loads(pickle.dumps(node))
        copynode.update(token_inst.name, token_inst)
        return copynode

    elif token_inst.is_keyword:
        # 这里主要进行类型检查
        if token_inst.name == ",":

            # 如果当前生成了一个逗号

            # 如果前一个token是一个type，那么这个逗号是非法的。
            if node.last_token is not None and node.last_token.is_type:
                return None

            # 当前没有pending中的 method call，那么说明逗号不应该被生成
            if len(node.stack) == 0:
                return None

            # 如果这个逗号是生成在assertEquals里，同时，当前栈内没有其他未关闭的函数调用
            if len(node.stack) == 2 and node.stack[0] == 'assertEquals':
                node.pending_assertEquals = False
                # 那么，此时对下一个参数产生约束，像栈中append一条约束
                if node.last_token is not None and node.last_token.return_type is not None:
                    node.stack.append(node.last_token.return_type)
                    return copy_and_update(node, token_inst)
                else:
                    # 这种情况下对应的是，上一个token的来源完全是global dict，因此在上一步被过滤掉了，这里last_token为None
                    node.stack.append('assertEqualsAny')
                    return copy_and_update(node, token_inst)

            elif len(node.stack) > 3 and node.stack[-3] == '<api>' and node.stack[-1] == '*':
                # 如果是api调用中的一个参数，那么不作处理，继续生成即可
                node.force_left_brace = False
                return copy_and_update(node, token_inst)

            else:
                # 就一个普通token的话，那么匹配当前已经生成的类型和要求的类型是否一致
                req_type = node.stack[-1]
                if (node.last_type is not None and
                    equivalent(node.last_type, req_type)) \
                        or req_type == 'nullableObj' \
                        or node.last_type == '<api>' \
                        or req_type == 'assertEqualsAny':  # * 只为了assertNull/NotNull, <api>是为了api, assertEqualsAny是为了适配assertEquals
                    # 一致，则出栈顶端的元素
                    node.stack.pop()
                    node.force_left_brace = False
                    return copy_and_update(node, token_inst)
                else:
                    # 不一致，则返回False，是无效的generation step
                    return None

        elif token_inst.name == ')':
            method_call_placeholders = {'<m>', '<api>'}
            # 首先看这个右括号能不能括上。

            # 如果上一个token是一个type token，那么这个括号括不上
            if node.last_token is not None and node.last_token.is_type:
                return None

            # 如果当前生成了一个右括号，那么首先匹配当前已经生成的类型和要求的类型是否一致

            # 这里先去除掉模型自己生成了一个左括号，我们没有push stack的情况，匹配掉栈顶的左括号
            if len(node.stack) != 0 and node.stack[-1] == "(":
                node.stack.pop()
                return copy_and_update(node, token_inst)

            if len(node.stack) >= 3 and node.stack[-3] == '<api>' and node.stack[-1] == '*':
                # 如果现在是一个api call，不做require type的检查，直接pop
                copynode = pickle.loads(pickle.dumps(node))
                copynode.stack.pop()
                copynode.stack.pop()
                copynode.stack.pop()
                token_inst.return_type = '<api>'
                copynode.last_token = token_inst
                copynode.last_type = '<api>'
                copynode.update(token_inst.name, token_inst)
                return copynode

            else:
                try:
                    req_type = node.stack[-1]
                    if node.last_type is not None and equivalent(node.last_type, req_type):
                        # 一致，则出栈顶端的元素
                        node.stack.pop()
                    elif node.last_type == '<api>':
                        # 如果是API，因为没有收集到返回值类型，放一个占位符表示这里完整生成了一个token，不做类型匹配
                        node.stack.pop()
                    elif req_type == 'assertEqualsAny':
                        # 这个是给没有分析出类型的assertEquals参数用的，直接pop
                        node.stack.pop()
                    else:
                        # 除了以上的情况之外，说明类型不符合，当前这个函数不能够被close掉。
                        return None

                    # pop 结束之后，检查当前函数是否生成结束，结束的话，更新栈
                    if len(node.stack) >= 2 and node.stack[-2] in method_call_placeholders:
                        last_type = node.stack.pop()
                        node.stack.pop()
                        copynode = pickle.loads(pickle.dumps(node))
                        copynode.last_type = last_type
                        copynode.last_token = token_inst
                        copynode.update(token_inst.name, token_inst)
                        return copynode
                    # # 这里是已经生成完了，当前这个右括号是assertion的右括号
                    if len(node.stack) == 2 and node.stack[-2] in node.local_vocab.accepted_assert_types:
                        if node.pending_assertEquals:
                            return None
                        node.force_semicolon = True
                        node.stack.clear()
                        return copy_and_update(node, token_inst)

                except Exception:
                    print(node.ans)
                    print(token_inst.name)
                    return None

        elif token_inst.name == ';':
            # 分号强制结束
            node.force_end = True

    elif token_inst.is_type:
        # new Type 下面应该接构造函数了吧
        if node.last_token.name == 'new':
            pass

    # 除了以上之外的情况，默认直接按照模型的预测结果更新node
    if token_inst.name == "(":
        node.stack.append("(")
    copynode = pickle.loads(pickle.dumps(node))
    copynode.last_token = token_inst
    copynode.last_type = token_inst.return_type
    lst = tokenizer.tokenize(tok)
    lst = tokenizer.convert_tokens_to_ids(lst)
    copynode.assertids.append(pad_seq(lst, copynode.maxlen))
    copynode.ans.append(tok)
    return copynode


def update_with_type(node, idx):
    tok = node.local_vocab.id2token(idx)
    print(' '.join(node.ans))
    if ' '.join(
            node.ans) == '<s> assertNotNull ( fileid . getFileId ( test , new DisabledListProgressListener ( )' and tok == ')':
        print(1)

    # 如果是强制左括号的话，那么如果id转换为token之后并不是左括号，这个node应该被舍弃
    if node.force_left_brace:
        if tok != '(':
            return []
        else:
            copynode = pickle.loads(pickle.dumps(node))
            token_inst = TokenInstance('(')
            token_inst.is_param = False
            token_inst.is_keyword = True
            token_inst.return_type = node.last_token.return_type
            copynode.update('(', token_inst)
            copynode.force_left_brace = False
            return [copynode]
    elif node.force_end:
        if tok != '</s>':
            return []
        else:
            copynode = pickle.loads(pickle.dumps(node))
            token_inst = TokenInstance('(')
            token_inst.is_param = False
            token_inst.is_keyword = True
            token_inst.return_type = node.last_token.return_type
            copynode.update(tok, token_inst)
            copynode.force_left_brace = False
            return [copynode]
    if tok in node.local_vocab.accepted_assert_types:
        # 先入栈一个assertType的func_call
        node.stack.append(tok)
        # 然后入栈一个返回值类型，为空
        node.stack.append(None)

        # 最后判断param list type，并入栈
        if tok in ["assertTrue", "assertFalse"]:
            node.stack.append("Boolean")
        elif tok in ['assertNull', 'assertNotNull']:
            node.stack.append('nullableObj')
        # 更新Token
        assertInst = TokenInstance(tok)
        assertInst.is_method = True
        assertInst.is_param = False
        copynode = pickle.loads(pickle.dumps(node))
        copynode.update(tok, assertInst)
        # assert之后，强制生成左括号
        copynode.force_left_brace = True
        if tok == 'assertEquals':
            copynode.pending_assertEquals = True
        return [copynode]
    else:
        return update_without_token_inst(node, idx)


def pad_seq(seq, maxlen):
    act_len = len(seq)
    if len(seq) < maxlen:
        seq = seq + [0] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq


def pad_list(seq, maxlen1):
    if len(seq) < maxlen1:
        seq = seq + [0] * maxlen1
        seq = seq[:maxlen1]
    else:
        seq = seq[:maxlen1]
    return seq


class SearchNode:
    def __init__(self, inst, local_vocab, config, idx):
        self.prob = 0
        self.finish = False
        self.assertids = []
        self.ans = []
        self.local_vocab = local_vocab[idx]
        self.maxlen = config.char_seq_max_len

    def update(self, idx):
        tok = self.local_vocab.id2token(idx)
        lst = tokenizer.tokenize(tok)
        lst = tokenizer.convert_tokens_to_ids(lst)
        self.assertids.append(pad_seq(lst, self.maxlen))
        self.ans.append(tok)

    def init(self, word):
        lst = tokenizer.tokenize(word)
        lst = tokenizer.convert_tokens_to_ids(lst)
        self.assertids.append(pad_seq(lst, self.maxlen))
        self.ans.append(word)


class SearchNodeWithType:
    def __init__(self, local_vocab, config, closure_dict):
        self.prob = 0
        self.finish = False
        self.assertids = []
        self.ans = []
        self.closure_dict = closure_dict
        # self.type_req = 'None'
        self.local_vocab = local_vocab
        self.maxlen = config.char_seq_max_len
        self.last_token = None
        self.stack = list()
        self.last_type = []
        self.force_left_brace = False
        self.force_end = False
        self.force_semicolon = False
        self.pending_assertEquals = False
        self.expression_end = False
        self.drop_constrain = False
        self.callstack = list()
        self.requires_static = False

    def update(self, tok, inst=None):
        # self.last_token = inst
        # self.last_type = inst.return_type if inst else None
        lst = tokenizer.tokenize(tok)
        lst = tokenizer.convert_tokens_to_ids(lst)
        self.assertids.append(pad_seq(lst, self.maxlen))
        self.ans.append(tok)

    def init_toga(self, assertion: list):
        if len(assertion) <= 3:
            print("Assertion too short, Wrong data")
        else:
            for i in range(3):
                if i == 0:
                    self.init(assertion[0])
                else:
                    tok = assertion[i]
                    tok_id = self.local_vocab.token2id(tok)

                    self.update_with_type_toga(tok_id)

    def update_with_type_toga(self, id):
        tok = self.local_vocab.id2token(id)
        # if ' '.join(self.ans) == '<s> assertTrue ( correctStreamLength > reportedEvent . getFileLength ( ) + headerPdfVersion':
        #     print(1)
        # 首先过滤assertion type
        if self.drop_constrain:
            self.update(tok)
            # 对两种可能结束的token进行判断
            if tok == ',' or tok == ')':
                # 判断当前的括号层级情况，从而判断是否让当前节点继续生成
                cur_assertion = self.ans
                left_bracket = 0
                right_bracket = 0
                for i in cur_assertion:
                    if i == '(':
                        left_bracket += 1
                    elif i == ')':
                        right_bracket += 1
                if tok == ',' and left_bracket - right_bracket == 1:
                    # 找到第一个参数，停止生成
                    self.ans.append('</s>')
                    self.last_token = '</s>'
                if tok == ')' and left_bracket == right_bracket:
                    # 完成了该断言的生成
                    self.ans.append('</s>')
                    self.last_token = '</s>'
            return True
        if self.last_token == '<s>':
            if tok not in self.local_vocab.accepted_assert_types:
                return False, "invalid assertionType"
            self.top_expression_type = 'assertion'
            if tok == 'assertEquals':
                self.stack.append('<assertion>')
                self.stack.append('none')
                self.stack.append('<default>')
                pass
            elif tok == 'assertTrue' or tok == 'assertFalse':
                self.stack.append('<assertion>')
                self.stack.append('none')
                self.stack.append('Boolean')
                pass
            else:
                self.stack.append('<assertion>')
                self.stack.append('none')
                self.stack.append('Object')
                pass
            self.last_type = ["<method>"]
            self.update(tok)
            self.last_token = tok
            return True
        elif tok == '</s>':
            self.update(tok)
            return True
        elif len(self.callstack) > 0 and self.callstack[-1] == '{' and tok != '}':
            self.update(tok)
            return True
        elif tok in separators:
            if tok == '.':
                if isinstance(self.last_token, TokenInstance):
                    if self.last_token.is_param:
                        self.last_type = [self.last_token.return_type]
                        self.last_token = '.'
                        self.update(tok)
                        return True
                    elif self.last_token.is_type:
                        self.last_type = [self.last_token.name]
                        self.last_token = '.'
                        self.update(tok)
                        self.requires_static = True
                        return True
                    elif self.last_token.is_literal:
                        self.last_type = [self.last_token.return_type]
                        self.last_token = '.'
                        self.update(tok)
                        return True
                    else:
                        return False, "invalid ."
                elif self.last_token == 'this':
                    self.update(tok)
                    self.last_token = '.'
                    self.last_type = ['local']
                    return True
                elif self.last_token in [']', ')']:
                    self.update(tok)
                    self.last_type = self.last_type
                    self.last_token = '.'
                    return True
                else:
                    return False, 'invalid .'

            if tok == ',':
                last_callstack = []
                while self.callstack[-1] in operators:
                    # print(', while loop')
                    req_type = self.stack[-1]
                    if equivalent(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == self.callstack.pop()
                        self.last_type = [ret_type]
                    else:
                        return False, "type mismatch"
                # 这里处理的是assertEquals遇到，的情况
                if len(self.stack) == 3 and self.stack[-1] == '<default>' and self.stack[0] == '<assertion>':
                    self.stack.pop()
                    self.stack.append(self.last_type)
                    self.update(tok)
                    self.last_token = tok
                    self.last_type = None
                    # 找到第一个参数，停止生成
                    self.ans.append('</s>')
                    self.last_token = '</s>'
                    return True
                elif self.stack[0] == '<assertion>':
                    self.update(tok)
                    self.drop_constrain = True
                    return True
                else:
                    req_type = self.stack[-1]
                    if req_type == '<*>':
                        # unlimited param numbers for api
                        self.update(tok)
                        self.last_token = token
                        self.last_type = None
                        return True
                    if equivalent(req_type, self.last_type):
                        if self.stack[-3] != '<method>':
                            self.stack.pop()
                            self.update(tok)
                            self.last_token = tok
                            self.last_type = None
                            return True
                        else:
                            # 等于的情况，相当于是当前的method整个都生成完了，这里不应该是逗号，而应该是右括号
                            return False, 'invalid , after finishing methodCall'
                    else:
                        return False, 'type mismatch'

            if tok == ')':
                while self.callstack[-1] in operators:
                    # print(') whil loop.')
                    req_type = self.stack[-1]
                    if equivalent(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == self.callstack.pop()
                        self.last_type = [ret_type]
                        # self.update(tok)
                        # self.last_token = tok
                    else:
                        return False, "type mismatch when generating )"

                if self.callstack[-1] == '<assertion>':
                    req_type = self.stack[-1]
                    if isinstance(req_type, list) and has_common_element(req_type, self.last_type):
                        self.stack.clear()
                        self.callstack.pop()
                        self.update(tok)
                        self.last_token = tok
                        # 生成完整的断言，停止继续生成
                        self.ans.append('</s>')
                        self.last_token = '</s>'
                        return True

                    elif req_type == 'Object':
                        self.stack.clear()
                        self.callstack.pop()
                        self.update(tok)
                        self.last_token = tok
                        # 生成完整的断言，停止继续生成
                        self.ans.append('</s>')
                        self.last_token = '</s>'
                        return True
                    elif isinstance(req_type, str) and equivalent(req_type, self.last_type):
                        self.stack.clear()
                        self.callstack.pop()
                        self.update(tok)
                        self.last_token = tok
                        # 生成完整的断言，停止继续生成
                        self.ans.append('</s>')
                        self.last_token = '</s>'
                        return True
                    else:
                        return False, "type mismatch"
                if self.stack[-3] == '<method>' and self.callstack[-1] == '<method>':
                    req_type = self.stack[-1]
                    if isinstance(req_type, str) and equivalent(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == '<method>' == self.callstack.pop()
                        self.last_type = [ret_type]
                        self.update(tok)
                        self.last_token = tok
                        return True
                    elif req_type == 'Object':
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == '<method>' == self.callstack.pop()
                        self.last_type = [ret_type]
                        self.update(tok)
                        self.last_token = tok
                        return True
                    elif isinstance(req_type, list) and has_common_element(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == '<method>' == self.callstack.pop()
                        self.last_type = [ret_type]
                        self.update(tok)
                        self.last_token = tok
                        return True
                    else:
                        return False, 'type mismatch'
                elif self.callstack[-1] == '(':
                    self.update(tok)
                    self.last_token = tok
                    self.callstack.pop()
                    self.last_type = ['T']
                    return True
                else:
                    return False, "invalid )"

            if tok == '(':
                if isinstance(self.last_token, TokenInstance) and (self.last_token.is_method or self.last_token.is_api):
                    prev_token = pickle.loads(pickle.dumps(self.last_token))
                    self.callstack.append('<method>')
                    self.stack.append('<method>')

                    self.stack.append(self.last_token.return_type)

                    if len(self.last_token.params) == 0:
                        self.stack.append('<no_param>')
                    else:
                        for type in reversed(self.last_token.params):
                            self.stack.append(type)

                    self.update(tok)
                    self.last_type = []
                    self.last_token = TokenInstance('(')
                    self.last_token.params = prev_token.params
                    self.last_token.return_type = prev_token.return_type
                    self.last_token.is_keyword = True
                    self.last_token.belonged_type = prev_token.belonged_type
                    prev_token = None
                    return True
                elif isinstance(self.last_token, TokenInstance) and self.last_token.is_type:
                    self.update(tok)
                    self.callstack.append('<method>')
                    self.stack.append('<method>')
                    self.stack.append(self.last_token.name)
                    self.stack.append('<no_param>')
                    self.last_token = '('
                    return True
                elif self.last_token in separators:
                    self.update(tok)
                    self.callstack.append('(')
                    self.last_token = '('
                    return True
                elif self.last_token in self.local_vocab.accepted_assert_types:
                    self.update(tok)
                    self.callstack.append('<assertion>')
                    self.last_token = '('
                    return True
                else:
                    return False, "invalid methodCall exp."

            if tok == '{':
                self.stack.append(('{', self.last_type))
                self.callstack.append('{')
                # self.stack.append(self.last_type)
                self.update(tok)
                self.last_token = tok
                self.last_type = None
                return True

            if tok == '}':
                if self.callstack[-1] == '{':
                    self.last_type = [self.stack.pop()[1]]
                    self.stack.pop()
                    self.callstack.pop()
                    self.update(tok)
                    self.last_token = tok
                    return True
                else:
                    return False, "invalid }"

            if tok == ';':
                if len(self.stack) == 0:
                    self.update(tok)
                    self.ans.append('</s>')
                    self.last_token = '</s>'
                    return True
                else:
                    return False, 'invalid ;'

            if tok == '[':
                if self.last_type[0].endswith('[]'):
                    type = self.last_type[0]
                    type = type.replace(' ', '')
                    type = type[:-2]
                    req_type = self.stack[-1]
                    if equivalent(req_type, type):
                        self.update(tok)
                        self.stack.append(('[', type))
                        self.callstack.append('[')
                        return True
                    else:
                        return False, 'type mismatch'
                else:
                    return False, 'invalid ['

            if tok == ']':
                if self.callstack[-1] == '[':
                    if equivalent('int', self.last_type) or 'Number' in self.last_type:
                        item = self.stack.pop()
                        self.update(tok)
                        self.last_type = [item[1]]
                        return True
                    else:
                        return False, 'type mismatch'
                else:
                    return False, 'invalid ]'

        elif tok in operators:
            if tok == '++' or tok == '--':
                if self.last_token in separators:
                    self.update(tok)
                    self.last_token = tok
                    self.last_type = ['int']
                    return True
                elif isinstance(self.last_token,
                                TokenInstance) and self.last_token.is_param and self.last_token.type.lower() in ['int',
                                                                                                                 'integer']:
                    self.last_type = [self.last_token.return_type]
                    self.last_token = tok
                    self.update(tok)
                    return True
                else:
                    return False, 'invalid %s' % tok
            if tok == '!':
                if self.last_token in separators:
                    if self.stack[-1] == 'Boolean':
                        self.update(tok)
                        self.last_token = tok
                        self.last_type = ['Boolean']
                    else:
                        return False, 'type mismatch'
                else:
                    return False, 'invalid !'
            if tok == ':' or tok == '::':
                self.update(tok)
                self.drop_constrain = True
                return True
            if tok in ['+', '-']:
                if isinstance(self.last_token, TokenInstance):
                    self.stack.append(tok)
                    self.callstack.append(tok)
                    self.stack.append(self.last_token.return_type)
                    self.stack.append(self.last_token.return_type)
                    self.update(tok)
                    self.last_token = '+'
                    return True
                elif self.last_token in [')', ']']:
                    self.stack.append(tok)
                    self.callstack.append(tok)
                    self.stack.append(self.last_type[0])
                    self.stack.append(self.last_type[0])
                    self.update(tok)
                    self.last_token = '+'
                    return True
                else:
                    return False, 'type mismatch'
            if tok in ['&&', '||']:
                if 'Boolean' in self.last_type:
                    self.update(tok)
                    self.callstack.append(tok)
                    self.stack.append(tok)
                    self.stack.append('Boolean')
                    self.stack.append('Boolean')
                    self.last_token = '&&'
                    self.last_type = ['Boolean']
                    return True
                else:
                    return False, 'type mismatch'
                pass

            if tok in ['>', '<', '>=', '<=','==']:
                if isinstance(self.last_token, TokenInstance):
                    while self.callstack[-1] in ['+', '-', '*', '/', '%']:
                        req_type = self.stack[-1]
                        if equivalent(req_type, self.last_type):
                            self.stack.pop()
                            ret_type = self.stack.pop()
                            assert self.stack.pop() == self.callstack.pop()
                            self.last_type = ret_type
                        else:
                            return False, "type mismatch"

                    self.callstack.append(tok)
                    self.stack.append(tok)
                    self.stack.append('Boolean')
                    self.stack.append(self.last_type)
                    self.update(tok)
                    self.last_token = tok
                    return True
                elif self.last_token in [')', ']']:
                    self.callstack.append(tok)
                    self.stack.append(tok)
                    self.stack.append('Boolean')
                    self.stack.append(self.last_type[0])
                    self.update(tok)
                    self.last_token = tok
                    return True
                else:
                    return False, 'type mismatch'

            pass

        elif tok in keyword:
            if tok == 'new':
                self.last_type = [self.stack[-1]]
                self.last_token = 'new'
                self.update(tok)
                return True
            elif tok == 'instanceof':
                req_type = self.stack[-1]
                if equivalent(req_type, 'Boolean'):
                    self.update(tok)
                    self.last_type = self.last_token.return_type if isinstance(self.last_token, TokenInstance) else [
                        '<T>']
                    self.last_token = 'instanceof'
                    return True
                else:
                    return False, 'type mismatch'
            elif tok == 'class':
                if self.last_token == '.':
                    self.last_token = 'class'
                    self.update(tok)
                    return True
                else:
                    return False, 'type mismatch'
            elif tok == 'this':
                self.last_type = ['local']
                self.last_token = 'this'
                self.update(tok)
                return True
            else:
                return False, 'invalid %s' % tok

        else:
            candidates = self.local_vocab.name2insts(tok)

            # Filter
            filtered = []
            if len(candidates) == 0:
                return False, 'token not found'
            elif len(candidates) == 1:
                filtered = [candidates[0]]
            else:
                if self.last_token == 'new':
                    for candidate in candidates:
                        if candidate.is_constructor:
                            filtered.append(candidate)
                elif self.last_token == 'instanceof':
                    for candidate in candidates:
                        if candidate.is_type:
                            filtered.append(candidate)
                elif self.last_token == '.':
                    if 'local' in self.last_type:
                        for candidate in candidates:
                            if candidate.is_local or candidate.is_literal:
                                filtered.append(candidate)
                    else:
                        for candidate in candidates:
                            if equivalent(candidate.belonged_type, self.last_type) or candidate.is_api:
                                filtered.append(candidate)
                                pass

                elif self.last_token in separators or self.last_token in operators:
                    for candidate in candidates:
                        if candidate.is_local or candidate.is_literal or candidate.is_type:
                            if candidate.return_type is not None:
                                filtered.append(candidate)


                else:
                    filtered = candidates
            # Update
            candidates = filtered

            if len(candidates) >= 2:
                self.update(tok)
                self.drop_constrain = True
                return True
            elif len(candidates) == 0:
                self.update(tok)
                self.drop_constrain = True
                return True
            candidate = candidates[0]
            if isinstance(candidate.return_type, str):
                need_special_process = re.match(r'\[MISSING:(.+)\]', candidate.return_type)
                if need_special_process is not None:
                    before = need_special_process.group()
                    after = before[9:-1]
                    candidate.return_type = after
            elif isinstance(candidate.return_type, list):
                new_return_type = []
                for type in candidate.return_type:
                    need_special_process = re.match(r'\[MISSING:(.+)\]', type)
                    if need_special_process is not None:
                        before = need_special_process.group()
                        after = before[9:-1]
                        new_return_type.append(after)
                candidate.return_type = new_return_type
            else:
                if candidate.return_type is None and candidate.name == candidate.token:
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = ['T']
                    return True
                pass

            if candidate.is_param:

                if self.callstack[-1] == '(':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = [candidate.return_type]
                    return True

                elif self.callstack[-1] == '[':
                    if candidate.return_type not in self.closure_dict.keys():
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    if 'int' in self.closure_dict[candidate.return_type] or 'Integer' in self.closure_dict[
                        candidate.return_type]:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        return False, 'type mismatch(dv)'

                elif self.callstack[-1] == '{':
                    self.update(tok)
                    return True

                elif self.callstack[-1] == '<assertion>' and self.stack[-1] == '<default>':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = [candidate.return_type]
                    return True

                else:
                    if self.last_token == '!':
                        if 'boolean' in self.closure_dict[candidate.return_type] or 'Boolean' in self.closure_dict[
                            candidate.return_type]:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [candidate.return_type]
                            return True
                        else:
                            return False, 'type mismatch(dv)'

                    elif self.last_token in ['--', '++']:
                        if candidate.return_type in ['int', 'Integer', 'Long', 'long']:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [candidate.return_type]
                            return True
                        else:
                            return False, 'type mismatch(dv)'

                    elif self.last_token == '.':
                        if candidate.belonged_type in self.last_type:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [candidate.return_type]
                            return True
                        else:
                            return False, 'type mismatch(dv)'

                    else:
                        req_type = self.stack[-1]
                        if '<' in candidate.return_type and '>' in candidate.return_type:
                            return_type = candidate.return_type.split('<')[0].strip()
                            pass
                        else:
                            return_type = candidate.return_type
                        if return_type in self.closure_dict.keys():
                            if len(self.closure_dict[return_type]) == 1:
                                self.update(tok)
                                self.last_token = candidate
                                self.last_type = [return_type]
                                return True
                            if (isinstance(req_type, list) and has_common_element(req_type,
                                                                                  self.closure_dict[return_type])) or (
                                    isinstance(req_type, str) and equivalent(req_type, self.closure_dict[return_type])):
                                self.update(tok)
                                self.last_token = candidate
                                self.last_type = [return_type]
                                return True
                            else:
                                return False, 'type mismatch(dv)'
                        else:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [return_type]
                            return True
            elif candidate.is_method or candidate.is_api:
                if self.last_token == '!':
                    if 'boolean' in self.closure_dict[candidate.return_type] or 'Boolean' in self.closure_dict[
                        candidate.return_type]:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        return False, 'type mismatch(dv)'
                else:
                    req_type = self.stack[-1]
                    tmp_closure = []
                    if candidate.return_type not in self.closure_dict.keys():
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        tmp_closure = self.closure_dict[candidate.return_type]
                    if has_common_element(req_type, tmp_closure):
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    elif candidate.is_api:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    elif len(tmp_closure) == 1:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        return False, 'type mismatch(dv)'
            elif candidate.is_type:
                if self.last_token == 'instanceof':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = 'Boolean'
                    return True
                elif self.last_token in separators:
                    if self.last_token == '(' or self.last_token == ',':
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = candidate.name
                        return True
                    else:
                        return False, 'type mismatch(dv)'
                elif self.last_token == 'new':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = candidate.name
                    return True
                else:
                    return False, 'type mismatch(dv)'
            elif candidate.is_literal:
                req_type = self.stack[-1]
                if candidate.return_type is None:
                    candidate.return_type = candidate.token.split('&')[0]

                if self.callstack[-1] == '[':
                    req_type = 'number'

                if req_type == '<default>' or req_type == 'T' or req_type == '<T>':
                    self.update(tok)
                    self.last_token = candidate
                    if candidate.name == 'null':
                        candidate.return_type = 'Object'
                    self.last_type = [candidate.return_type]
                    return True
                elif equivalent(req_type, candidate.return_type):
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = req_type
                    return True
                else:
                    return False, 'type mismatch(dv)'

        pass

    def init(self, word):
        lst = tokenizer.tokenize(word)
        lst = tokenizer.convert_tokens_to_ids(lst)
        self.assertids.append(pad_seq(lst, self.maxlen))
        self.ans.append(word)
        self.last_token = word

    def update_with_type(self, id):
        tok = self.local_vocab.id2token(id)
        # if ' '.join(self.ans) == '<s> assertTrue ( correctStreamLength > reportedEvent . getFileLength ( ) + headerPdfVersion':
        #     print(1)
        # 首先过滤assertion type
        if self.drop_constrain:
            self.update(tok)
            return True
        if self.last_token == '<s>':
            if tok not in self.local_vocab.accepted_assert_types:
                return False, "invalid assertionType"
            self.top_expression_type = 'assertion'
            if tok == 'assertEquals':
                self.stack.append('<assertion>')
                self.stack.append('none')
                self.stack.append('<default>')
                pass
            elif tok == 'assertTrue' or tok == 'assertFalse':
                self.stack.append('<assertion>')
                self.stack.append('none')
                self.stack.append('Boolean')
                pass
            else:
                self.stack.append('<assertion>')
                self.stack.append('none')
                self.stack.append('Object')
                pass
            self.last_type = ["<method>"]
            self.update(tok)
            self.last_token = tok
            return True
        elif tok == '</s>':
            self.update(tok)
            return True
        elif len(self.callstack) > 0 and self.callstack[-1] == '{' and tok != '}':
            self.update(tok)
            return True
        elif tok in separators:
            if tok == '.':
                if isinstance(self.last_token, TokenInstance):
                    if self.last_token.is_param:
                        self.last_type = [self.last_token.return_type]
                        self.last_token = '.'
                        self.update(tok)
                        return True
                    elif self.last_token.is_type:
                        self.last_type = [self.last_token.name]
                        self.last_token = '.'
                        self.update(tok)
                        self.requires_static = True
                        return True
                    elif self.last_token.is_literal:
                        self.last_type = [self.last_token.return_type]
                        self.last_token = '.'
                        self.update(tok)
                        return True
                    else:
                        return False, "invalid ."
                elif self.last_token == 'this':
                    self.update(tok)
                    self.last_token = '.'
                    self.last_type = ['local']
                    return True
                elif self.last_token in [']', ')']:
                    self.update(tok)
                    self.last_type = self.last_type
                    self.last_token = '.'
                    return True
                else:
                    return False, 'invalid .'

            if tok == ',':
                last_callstack = []
                while self.callstack[-1] in operators:
                    # print(', while loop')
                    req_type = self.stack[-1]
                    if equivalent(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == self.callstack.pop()
                        self.last_type = [ret_type]
                    else:
                        return False, "type mismatch"
                # 这里处理的是assertEquals遇到，的情况
                if len(self.stack) == 3 and self.stack[-1] == '<default>' and self.stack[0] == '<assertion>':
                    self.stack.pop()
                    self.stack.append(self.last_type)
                    self.update(tok)
                    self.last_token = tok
                    self.last_type = None
                    return True
                elif self.stack[0] == '<assertion>':
                    self.update(tok)
                    self.drop_constrain = True
                    return True
                else:
                    req_type = self.stack[-1]
                    if req_type == '<*>':
                        # unlimited param numbers for api
                        self.update(tok)
                        self.last_token = token
                        self.last_type = None
                        return True
                    if equivalent(req_type, self.last_type):
                        if self.stack[-3] != '<method>':
                            self.stack.pop()
                            self.update(tok)
                            self.last_token = tok
                            self.last_type = None
                            return True
                        else:
                            # 等于的情况，相当于是当前的method整个都生成完了，这里不应该是逗号，而应该是右括号
                            return False, 'invalid , after finishing methodCall'
                    else:
                        return False, 'type mismatch'

            if tok == ')':
                while self.callstack[-1] in operators:
                    # print(') whil loop.')
                    req_type = self.stack[-1]
                    if equivalent(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == self.callstack.pop()
                        self.last_type = [ret_type]
                        # self.update(tok)
                        # self.last_token = tok
                    else:
                        return False, "type mismatch when generating )"

                if self.callstack[-1] == '<assertion>':
                    req_type = self.stack[-1]
                    if isinstance(req_type, list) and has_common_element(req_type, self.last_type):
                        self.stack.clear()
                        self.callstack.pop()
                        self.update(tok)
                        self.last_token = tok
                        return True
                    elif req_type == 'Object':
                        self.stack.clear()
                        self.callstack.pop()
                        self.update(tok)
                        self.last_token = tok
                        return True
                    elif isinstance(req_type, str) and equivalent(req_type, self.last_type):
                        self.stack.clear()
                        self.callstack.pop()
                        self.update(tok)
                        self.last_token = tok
                        return True
                    else:
                        return False, "type mismatch"
                if self.stack[-3] == '<method>' and self.callstack[-1] == '<method>':
                    req_type = self.stack[-1]
                    if isinstance(req_type, str) and equivalent(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == '<method>' == self.callstack.pop()
                        self.last_type = [ret_type]
                        self.update(tok)
                        self.last_token = tok
                        return True
                    elif req_type == 'Object':
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == '<method>' == self.callstack.pop()
                        self.last_type = [ret_type]
                        self.update(tok)
                        self.last_token = tok
                        return True
                    elif isinstance(req_type, list) and has_common_element(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == '<method>' == self.callstack.pop()
                        self.last_type = [ret_type]
                        self.update(tok)
                        self.last_token = tok
                        return True
                    else:
                        return False, 'type mismatch'
                elif self.callstack[-1] == '(':
                    self.update(tok)
                    self.last_token = tok
                    self.callstack.pop()
                    self.last_type = ['T']
                    return True
                else:
                    return False, "invalid )"

            if tok == '(':
                if isinstance(self.last_token, TokenInstance) and (self.last_token.is_method or self.last_token.is_api):
                    prev_token = pickle.loads(pickle.dumps(self.last_token))
                    self.callstack.append('<method>')
                    self.stack.append('<method>')

                    self.stack.append(self.last_token.return_type)

                    if len(self.last_token.params) == 0:
                        self.stack.append('<no_param>')
                    else:
                        for type in reversed(self.last_token.params):
                            self.stack.append(type)

                    self.update(tok)
                    self.last_type = []
                    self.last_token = TokenInstance('(')
                    self.last_token.params = prev_token.params
                    self.last_token.return_type = prev_token.return_type
                    self.last_token.is_keyword = True
                    self.last_token.belonged_type = prev_token.belonged_type
                    prev_token = None
                    return True
                elif isinstance(self.last_token, TokenInstance) and self.last_token.is_type:
                    self.update(tok)
                    self.callstack.append('<method>')
                    self.stack.append('<method>')
                    self.stack.append(self.last_token.name)
                    self.stack.append('<no_param>')
                    self.last_token = '('
                    return True
                elif self.last_token in separators:
                    self.update(tok)
                    self.callstack.append('(')
                    self.last_token = '('
                    return True
                elif self.last_token in self.local_vocab.accepted_assert_types:
                    self.update(tok)
                    self.callstack.append('<assertion>')
                    self.last_token = '('
                    return True
                else:
                    return False, "invalid methodCall exp."

            if tok == '{':
                if isinstance(self.last_token, TokenInstance):
                    if self.last_token.is_param:
                        return False
                self.stack.append(('{', self.last_type))
                self.callstack.append('{')
                # self.stack.append(self.last_type)
                self.update(tok)
                self.last_token = tok
                self.last_type = None
                return True

            if tok == '}':
                if self.callstack[-1] == '{':
                    self.last_type = [self.stack.pop()[1]]
                    self.stack.pop()
                    self.callstack.pop()
                    self.update(tok)
                    self.last_token = tok
                    return True
                else:
                    return False, "invalid }"

            if tok == ';':
                if len(self.stack) == 0:
                    self.update(tok)
                    self.ans.append('</s>')
                    self.last_token = '</s>'
                    return True
                else:
                    return False, 'invalid ;'

            if tok == '[':
                if self.last_type[0].endswith('[]'):
                    type = self.last_type[0]
                    type = type.replace(' ', '')
                    type = type[:-2]
                    req_type = self.stack[-1]
                    if equivalent(req_type, type):
                        self.update(tok)
                        self.stack.append(('[', type))
                        self.callstack.append('[')
                        return True
                    else:
                        return False, 'type mismatch'
                else:
                    return False, 'invalid ['

            if tok == ']':
                if self.callstack[-1] == '[':
                    if equivalent('int', self.last_type) or 'Number' in self.last_type:
                        item = self.stack.pop()
                        self.update(tok)
                        self.last_type = [item[1]]
                        return True
                    else:
                        return False, 'type mismatch'
                else:
                    return False, 'invalid ]'

        elif tok in operators:
            if tok == '++' or tok == '--':
                if self.last_token in separators:
                    self.update(tok)
                    self.last_token = tok
                    self.last_type = ['int']
                    return True
                elif isinstance(self.last_token,
                                TokenInstance) and self.last_token.is_param and self.last_token.type.lower() in ['int',
                                                                                                                 'integer']:
                    self.last_type = [self.last_token.return_type]
                    self.last_token = tok
                    self.update(tok)
                    return True
                else:
                    return False, 'invalid %s' % tok
            if tok == '!':
                if self.last_token in separators:
                    if self.stack[-1] == 'Boolean':
                        self.update(tok)
                        self.last_token = tok
                        self.last_type = ['Boolean']
                    else:
                        return False, 'type mismatch'
                else:
                    return False, 'invalid !'
            if tok == ':' or tok == '::':
                self.update(tok)
                self.drop_constrain = True
                return True
            if tok in ['+', '-']:
                if isinstance(self.last_token, TokenInstance):
                    self.stack.append(tok)
                    self.callstack.append(tok)
                    self.stack.append(self.last_token.return_type)
                    self.stack.append(self.last_token.return_type)
                    self.update(tok)
                    self.last_token = '+'
                    return True
                elif self.last_token in [')', ']']:
                    self.stack.append(tok)
                    self.callstack.append(tok)
                    self.stack.append(self.last_type[0])
                    self.stack.append(self.last_type[0])
                    self.update(tok)
                    self.last_token = '+'
                    return True
                else:
                    return False, 'type mismatch'
            if tok in ['&&', '||']:
                if 'Boolean' in self.last_type:
                    self.update(tok)
                    self.callstack.append(tok)
                    self.stack.append(tok)
                    self.stack.append('Boolean')
                    self.stack.append('Boolean')
                    self.last_token = '&&'
                    self.last_type = ['Boolean']
                    return True
                else:
                    return False, 'type mismatch'
                pass

            if tok in ['>', '<', '>=', '<=']:
                if isinstance(self.last_token, TokenInstance):
                    while self.callstack[-1] in ['+', '-', '*', '/', '%']:
                        req_type = self.stack[-1]
                        if equivalent(req_type, self.last_type):
                            self.stack.pop()
                            ret_type = self.stack.pop()
                            assert self.stack.pop() == self.callstack.pop()
                            self.last_type = ret_type
                        else:
                            return False, "type mismatch"

                    self.callstack.append(tok)
                    self.stack.append(tok)
                    self.stack.append('Boolean')
                    self.stack.append(self.last_type)
                    self.update(tok)
                    self.last_token = tok
                    return True
                elif self.last_token in [')', ']']:
                    self.callstack.append(tok)
                    self.stack.append(tok)
                    self.stack.append('Boolean')
                    self.stack.append(self.last_type[0])
                    self.update(tok)
                    self.last_token = tok
                    return True
                else:
                    return False, 'type mismatch'

            pass

        elif tok in keyword:
            if tok == 'new':
                self.last_type = [self.stack[-1]]
                self.last_token = 'new'
                self.update(tok)
                return True
            elif tok == 'instanceof':
                req_type = self.stack[-1]
                if equivalent(req_type, 'Boolean'):
                    self.update(tok)
                    self.last_type = self.last_token.return_type if isinstance(self.last_token, TokenInstance) else [
                        '<T>']
                    self.last_token = 'instanceof'
                    return True
                else:
                    return False, 'type mismatch'
            elif tok == 'class':
                if self.last_token == '.':
                    self.last_token = 'class'
                    self.update(tok)
                    return True
                else:
                    return False, 'type mismatch'
            elif tok == 'this':
                self.last_type = ['local']
                self.last_token = 'this'
                self.update(tok)
                return True
            else:
                return False, 'invalid %s' % tok

        else:
            candidates = self.local_vocab.name2insts(tok)

            # Filter
            filtered = []
            if len(candidates) == 0:
                return False, 'token not found'
            elif len(candidates) == 1:
                filtered = [candidates[0]]
            else:
                if self.last_token == 'new':
                    for candidate in candidates:
                        if candidate.is_constructor:
                            filtered.append(candidate)
                elif self.last_token == 'instanceof':
                    for candidate in candidates:
                        if candidate.is_type:
                            filtered.append(candidate)
                elif self.last_token == '.':
                    if 'local' in self.last_type:
                        for candidate in candidates:
                            if candidate.is_local or candidate.is_literal:
                                filtered.append(candidate)
                    else:
                        for candidate in candidates:
                            if equivalent(candidate.belonged_type, self.last_type) or candidate.is_api:
                                filtered.append(candidate)
                                pass

                elif self.last_token in separators or self.last_token in operators:
                    for candidate in candidates:
                        if candidate.is_local or candidate.is_literal or candidate.is_type:
                            if candidate.return_type is not None:
                                filtered.append(candidate)


                else:
                    filtered = candidates
            # Update
            candidates = filtered

            if len(candidates) >= 2:
                self.update(tok)
                self.drop_constrain = True
                return True
            elif len(candidates) == 0:
                self.update(tok)
                self.drop_constrain = True
                return True
            candidate = candidates[0]
            if isinstance(candidate.return_type, str):
                need_special_process = re.match(r'\[MISSING:(.+)\]', candidate.return_type)
                if need_special_process is not None:
                    before = need_special_process.group()
                    after = before[9:-1]
                    candidate.return_type = after
            elif isinstance(candidate.return_type, list):
                new_return_type = []
                for type in candidate.return_type:
                    need_special_process = re.match(r'\[MISSING:(.+)\]', type)
                    if need_special_process is not None:
                        before = need_special_process.group()
                        after = before[9:-1]
                        new_return_type.append(after)
                candidate.return_type = new_return_type
            else:
                if candidate.return_type is None and candidate.name == candidate.token:
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = ['T']
                    return True
                pass

            if candidate.is_param:

                if self.callstack[-1] == '(':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = [candidate.return_type]
                    return True

                elif self.callstack[-1] == '[':
                    if candidate.return_type not in self.closure_dict.keys():
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    if 'int' in self.closure_dict[candidate.return_type] or 'Integer' in self.closure_dict[
                        candidate.return_type]:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        return False, 'type mismatch(dv)'

                elif self.callstack[-1] == '{':
                    self.update(tok)
                    return True

                elif self.callstack[-1] == '<assertion>' and self.stack[-1] == '<default>':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = [candidate.return_type]
                    return True

                else:
                    if self.last_token == '!':
                        if 'boolean' in self.closure_dict[candidate.return_type] or 'Boolean' in self.closure_dict[
                            candidate.return_type]:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [candidate.return_type]
                            return True
                        else:
                            return False, 'type mismatch(dv)'

                    elif self.last_token in ['--', '++']:
                        if candidate.return_type in ['int', 'Integer', 'Long', 'long']:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [candidate.return_type]
                            return True
                        else:
                            return False, 'type mismatch(dv)'

                    elif self.last_token == '.':
                        if candidate.belonged_type in self.last_type:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [candidate.return_type]
                            return True
                        else:
                            return False, 'type mismatch(dv)'

                    else:
                        req_type = self.stack[-1]
                        if '<' in candidate.return_type and '>' in candidate.return_type:
                            return_type = candidate.return_type.split('<')[0].strip()
                            pass
                        else:
                            return_type = candidate.return_type
                        if return_type in self.closure_dict.keys():
                            if len(self.closure_dict[return_type]) == 1:
                                self.update(tok)
                                self.last_token = candidate
                                self.last_type = [return_type]
                                return True
                            if (isinstance(req_type, list) and has_common_element(req_type,
                                                                                  self.closure_dict[return_type])) or (
                                    isinstance(req_type, str) and equivalent(req_type, self.closure_dict[return_type])):
                                self.update(tok)
                                self.last_token = candidate
                                self.last_type = [return_type]
                                return True
                            else:
                                return False, 'type mismatch(dv)'
                        else:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [return_type]
                            return True
            elif candidate.is_method or candidate.is_api:
                if self.last_token == '!':
                    if 'boolean' in self.closure_dict[candidate.return_type] or 'Boolean' in self.closure_dict[
                        candidate.return_type]:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        return False, 'type mismatch(dv)'
                else:
                    req_type = self.stack[-1]
                    tmp_closure = []
                    if candidate.return_type not in self.closure_dict.keys():
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        tmp_closure = self.closure_dict[candidate.return_type]
                    if has_common_element(req_type, tmp_closure):
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    elif candidate.is_api:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    elif len(tmp_closure) == 1:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        return False, 'type mismatch(dv)'
            elif candidate.is_type:
                if self.last_token == 'instanceof':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = 'Boolean'
                    return True
                elif self.last_token in separators:
                    if self.last_token == '(' or self.last_token == ',':
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = candidate.name
                        return True
                    else:
                        return False, 'type mismatch(dv)'
                elif self.last_token == 'new':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = candidate.name
                    return True
                else:
                    return False, 'type mismatch(dv)'
            elif candidate.is_literal:
                req_type = self.stack[-1]
                if candidate.return_type is None:
                    candidate.return_type = candidate.token.split('&')[0]

                if self.callstack[-1] == '[':
                    req_type = 'number'

                if req_type == '<default>' or req_type == 'T' or req_type == '<T>':
                    self.update(tok)
                    self.last_token = candidate
                    if candidate.name == 'null':
                        candidate.return_type = 'Object'
                    self.last_type = [candidate.return_type]
                    return True
                elif equivalent(req_type, candidate.return_type):
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = req_type
                    return True
                else:
                    return False, 'type mismatch(dv)'

        pass


class SearchNode_T5:
    def __init__(self, config):
        self.prob = 0
        self.finish = False
        self.assertids = []
        self.ans = []
        self.max_query_len = config.query_max_len
        self.token_trace_back = []

    def update(self, idx):
        self.ans.append(idx)
        tmp_token = tokenizer.convert_ids_to_tokens(idx)
        self.assertids = pad_list(self.ans, self.max_query_len)

    def init(self, idx):
        self.ans.append(idx)
        self.assertids = pad_list(self.ans, self.max_query_len)


class SearchNodeWithType_T5:
    def __init__(self, local_vocab, config, closure_dict):
        self.token_trace_back = []
        self.idx_trace_back = []
        self.prob = 0
        self.finish = False
        self.assertids = []
        self.ans = []
        self.ans_ids = []
        self.closure_dict = closure_dict
        # self.type_req = 'None'
        self.max_query_len = config.query_max_len
        self.local_vocab = local_vocab
        self.maxlen = config.char_seq_max_len
        self.last_token = None
        self.stack = list()
        self.last_type = []
        self.force_left_brace = False
        self.force_end = False
        self.force_semicolon = False
        self.pending_assertEquals = False
        self.expression_end = False
        self.drop_constrain = False
        self.callstack = list()
        self.requires_static = False
        self.query = []
        self.quote = False

    def update(self, token):
        self.ans.append(token)
        self.ans_ids.extend(self.idx_trace_back)
        self.query = copy.deepcopy(self.ans_ids)
        self.assertids = pad_list(self.query, self.max_query_len)

    def update_query(self, idx):
        self.query.append(idx)
        self.assertids = pad_list(self.query, self.max_query_len)

    def update_with_type(self, idx):
        token = tokenizer.convert_ids_to_tokens(idx)
        if token.startswith('\u0120'):
            if len(self.token_trace_back) == 0:
                if len(token) == 2 and token[1] == '"':
                    self.quote = True
                self.token_trace_back.append(token[1:])
                self.idx_trace_back.append(idx)
                self.update_query(idx)
                return True
            else:
                if self.quote:
                    if len(token) == 2 and token[1] == '"':
                        self.quote = False
                    self.token_trace_back.append(' ' + token[1:])
                    self.idx_trace_back.append(idx)
                    self.update_query(idx)
                    return True
                else:
                    flag = self._inner_check_update_with_type()
                    if flag == True:
                        if len(token) == 2 and token[1] == '"':
                            self.quote = True
                        self.token_trace_back.clear()
                        self.idx_trace_back.clear()
                        self.idx_trace_back.append(idx)
                        self.token_trace_back.append(token[1:])
                        self.update_query(idx)
                    return flag
        else:
            if len(self.token_trace_back) == 0 and token != '</s>':
                return False
            if self.quote and token.endswith('"'):
                self.quote = False
            if token == '</s>':
                if len(self.token_trace_back) ==0:
                    self.token_trace_back.clear()
                    self.idx_trace_back.clear()
                    self.idx_trace_back.append(idx)
                    self.update(token)
                    self.last_token = '</s>'
                    self.finish = True
                    flag = True
                else:
                    flag = self._inner_check_update_with_type()
                    if flag == True:
                        self.token_trace_back.clear()
                        self.idx_trace_back.clear()
                        self.idx_trace_back.append(idx)
                        self.update(token)
                        self.last_token = '</s>'
                        self.finish = True
                return flag
            self.token_trace_back.append(token)
            self.idx_trace_back.append(idx)
            self.update_query(idx)
            return True

    def _inner_check_update_with_type(self):
        tok = ''.join(self.token_trace_back)
        if tok == '':
            return True
        # if ' '.join(self.ans) == '<s> assertTrue ( correctStreamLength > reportedEvent . getFileLength ( ) + headerPdfVersion':
        #     print(1)
        # 首先过滤assertion type
        if self.drop_constrain:
            self.update(tok)
            return True
        if self.last_token == '<s>':
            if tok not in self.local_vocab.accepted_assert_types:
                return False, "invalid assertionType"
            self.top_expression_type = 'assertion'
            if tok == 'assertEquals':
                self.stack.append('<assertion>')
                self.stack.append('none')
                self.stack.append('<default>')
                pass
            elif tok == 'assertTrue' or tok == 'assertFalse':
                self.stack.append('<assertion>')
                self.stack.append('none')
                self.stack.append('Boolean')
                pass
            else:
                self.stack.append('<assertion>')
                self.stack.append('none')
                self.stack.append('Object')
                pass
            self.last_type = ["<method>"]
            self.update(tok)
            self.last_token = tok
            return True
        elif tok == '</s>':
            self.update(tok)
            return True
        elif len(self.callstack) > 0 and self.callstack[-1] == '{' and tok != '}':
            self.update(tok)
            return True
        elif tok in separators:
            if tok == '.':
                if isinstance(self.last_token, TokenInstance):
                    if self.last_token.is_param:
                        self.last_type = [self.last_token.return_type]
                        self.last_token = '.'
                        self.update(tok)
                        return True
                    elif self.last_token.is_type:
                        self.last_type = [self.last_token.name]
                        self.last_token = '.'
                        self.update(tok)
                        self.requires_static = True
                        return True
                    elif self.last_token.is_literal:
                        self.last_type = [self.last_token.return_type]
                        self.last_token = '.'
                        self.update(tok)
                        return True
                    else:
                        return False, "invalid ."
                elif self.last_token == 'this':
                    self.update(tok)
                    self.last_token = '.'
                    self.last_type = ['local']
                    return True
                elif self.last_token in [']', ')']:
                    self.update(tok)
                    self.last_type = self.last_type
                    self.last_token = '.'
                    return True
                else:
                    return False, 'invalid .'

            if tok == ',':
                last_callstack = []
                while self.callstack[-1] in operators:
                    # print(', while loop')
                    req_type = self.stack[-1]
                    if equivalent(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == self.callstack.pop()
                        self.last_type = [ret_type]
                    else:
                        return False, "type mismatch"

                if len(self.stack) == 3 and self.stack[-1] == '<default>' and self.stack[0] == '<assertion>':
                    self.stack.pop()
                    self.stack.append(self.last_type)
                    self.update(tok)
                    self.last_token = tok
                    self.last_type = None
                    return True
                elif self.stack[0] == '<assertion>':
                    self.update(tok)
                    self.drop_constrain = True
                    return True
                else:
                    req_type = self.stack[-1]
                    if req_type == '<*>':
                        # unlimited param numbers for api
                        self.update(tok)
                        self.last_token = token
                        self.last_type = None
                        return True
                    if equivalent(req_type, self.last_type):
                        if self.stack[-3] != '<method>':
                            self.stack.pop()
                            self.update(tok)
                            self.last_token = tok
                            self.last_type = None
                            return True
                        else:
                            # 等于的情况，相当于是当前的method整个都生成完了，这里不应该是逗号，而应该是右括号
                            return False, 'invalid , after finishing methodCall'
                    else:
                        return False, 'type mismatch'

            if tok == ')':
                while self.callstack[-1] in operators:
                    # print(') whil loop.')
                    req_type = self.stack[-1]
                    if equivalent(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == self.callstack.pop()
                        self.last_type = [ret_type]
                        # self.update(tok)
                        # self.last_token = tok
                    else:
                        return False, "type mismatch when generating )"

                if self.callstack[-1] == '<assertion>':
                    req_type = self.stack[-1]
                    if isinstance(req_type, list) and has_common_element(req_type, self.last_type):
                        self.stack.clear()
                        self.callstack.pop()
                        self.update(tok)
                        self.last_token = tok
                        return True
                    elif req_type == 'Object':
                        self.stack.clear()
                        self.callstack.pop()
                        self.update(tok)
                        self.last_token = tok
                        return True
                    elif isinstance(req_type, str) and equivalent(req_type, self.last_type):
                        self.stack.clear()
                        self.callstack.pop()
                        self.update(tok)
                        self.last_token = tok
                        return True
                    else:
                        return False, "type mismatch"
                if self.stack[-3] == '<method>' and self.callstack[-1] == '<method>':
                    req_type = self.stack[-1]
                    if isinstance(req_type, str) and equivalent(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == '<method>' == self.callstack.pop()
                        self.last_type = [ret_type]
                        self.update(tok)
                        self.last_token = tok
                        return True
                    elif req_type == 'Object':
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == '<method>' == self.callstack.pop()
                        self.last_type = [ret_type]
                        self.update(tok)
                        self.last_token = tok
                        return True
                    elif isinstance(req_type, list) and has_common_element(req_type, self.last_type):
                        self.stack.pop()
                        ret_type = self.stack.pop()
                        assert self.stack.pop() == '<method>' == self.callstack.pop()
                        self.last_type = [ret_type]
                        self.update(tok)
                        self.last_token = tok
                        return True
                    else:
                        return False, 'type mismatch'
                elif self.callstack[-1] == '(':
                    self.update(tok)
                    self.last_token = tok
                    self.callstack.pop()
                    self.last_type = ['T']
                    return True
                else:
                    return False, "invalid )"

            if tok == '(':
                if isinstance(self.last_token, TokenInstance) and (self.last_token.is_method or self.last_token.is_api):
                    prev_token = pickle.loads(pickle.dumps(self.last_token))
                    self.callstack.append('<method>')
                    self.stack.append('<method>')

                    self.stack.append(self.last_token.return_type)

                    if len(self.last_token.params) == 0:
                        self.stack.append('<no_param>')
                    else:
                        for type in reversed(self.last_token.params):
                            self.stack.append(type)

                    self.update(tok)
                    self.last_type = []
                    self.last_token = TokenInstance('(')
                    self.last_token.params = prev_token.params
                    self.last_token.return_type = prev_token.return_type
                    self.last_token.is_keyword = True
                    self.last_token.belonged_type = prev_token.belonged_type
                    prev_token = None
                    return True
                elif isinstance(self.last_token, TokenInstance) and self.last_token.is_type:
                    self.update(tok)
                    self.callstack.append('<method>')
                    self.stack.append('<method>')
                    self.stack.append(self.last_token.name)
                    self.stack.append('<no_param>')
                    self.last_token = '('
                    return True
                elif self.last_token in separators:
                    self.update(tok)
                    self.callstack.append('(')
                    self.last_token = '('
                    return True
                elif self.last_token in self.local_vocab.accepted_assert_types:
                    self.update(tok)
                    self.callstack.append('<assertion>')
                    self.last_token = '('
                    return True
                else:
                    return False, "invalid methodCall exp."

            if tok == '{':
                if isinstance(self.last_token, TokenInstance):
                    if self.last_token.is_param:
                        return False
                self.stack.append(('{', self.last_type))
                self.callstack.append('{')
                # self.stack.append(self.last_type)
                self.update(tok)
                self.last_token = tok
                self.last_type = None
                return True

            if tok == '}':
                if self.callstack[-1] == '{':
                    self.last_type = [self.stack.pop()[1]]
                    self.stack.pop()
                    self.callstack.pop()
                    self.update(tok)
                    self.last_token = tok
                    return True
                else:
                    return False, "invalid }"

            if tok == ';':
                if len(self.stack) == 0:
                    self.update(tok)
                    self.ans.append('</s>')
                    self.last_token = '</s>'
                    return True
                else:
                    return False, 'invalid ;'

            if tok == '[':
                if self.last_type[0].endswith('[]'):
                    type = self.last_type[0]
                    type = type.replace(' ', '')
                    type = type[:-2]
                    req_type = self.stack[-1]
                    if equivalent(req_type, type):
                        self.update(tok)
                        self.stack.append(('[', type))
                        self.callstack.append('[')
                        return True
                    else:
                        return False, 'type mismatch'
                else:
                    return False, 'invalid ['

            if tok == ']':
                if self.callstack[-1] == '[':
                    if equivalent('int', self.last_type) or 'Number' in self.last_type:
                        item = self.stack.pop()
                        self.update(tok)
                        self.last_type = [item[1]]
                        return True
                    else:
                        return False, 'type mismatch'
                else:
                    return False, 'invalid ]'

        elif tok in operators:
            if tok == '++' or tok == '--':
                if self.last_token in separators:
                    self.update(tok)
                    self.last_token = tok
                    self.last_type = ['int']
                    return True
                elif isinstance(self.last_token,
                                TokenInstance) and self.last_token.is_param and self.last_token.type.lower() in ['int',
                                                                                                                 'integer']:
                    self.last_type = [self.last_token.return_type]
                    self.last_token = tok
                    self.update(tok)
                    return True
                else:
                    return False, 'invalid %s' % tok
            if tok == '!':
                if self.last_token in separators:
                    if self.stack[-1] == 'Boolean':
                        self.update(tok)
                        self.last_token = tok
                        self.last_type = ['Boolean']
                    else:
                        return False, 'type mismatch'
                else:
                    return False, 'invalid !'
            if tok == ':' or tok == '::':
                self.update(tok)
                self.drop_constrain = True
                return True
            if tok in ['+', '-']:
                if isinstance(self.last_token, TokenInstance):
                    self.stack.append(tok)
                    self.callstack.append(tok)
                    self.stack.append(self.last_token.return_type)
                    self.stack.append(self.last_token.return_type)
                    self.update(tok)
                    self.last_token = tok
                    return True
                elif self.last_token in [')', ']']:
                    self.stack.append(tok)
                    self.callstack.append(tok)
                    self.stack.append(self.last_type[0])
                    self.stack.append(self.last_type[0])
                    self.update(tok)
                    self.last_token = tok
                    return True
                else:
                    return False, 'type mismatch'
            if tok in ['*','/']:
                if isinstance(self.last_token, TokenInstance):
                    self.stack.append(tok)
                    self.callstack.append(tok)
                    self.stack.append(self.last_token.return_type)
                    self.stack.append(self.last_token.return_type)
                    self.update(tok)
                    self.last_token = tok
                    return True
                elif self.last_token in [')', ']']:
                    self.stack.append(tok)
                    self.callstack.append(tok)
                    self.stack.append(self.last_type[0])
                    self.stack.append(self.last_type[0])
                    self.update(tok)
                    self.last_token = tok
                    return True
            if tok in ['&&', '||']:
                if 'Boolean' in self.last_type:
                    self.update(tok)
                    self.callstack.append(tok)
                    self.stack.append(tok)
                    self.stack.append('Boolean')
                    self.stack.append('Boolean')
                    self.last_token = '&&'
                    self.last_type = ['Boolean']
                    return True
                else:
                    return False, 'type mismatch'
                pass

            if tok in ['>', '<', '>=', '<=','==','!=']:
                if isinstance(self.last_token, TokenInstance):
                    while self.callstack[-1] in ['+', '-', '*', '/', '%']:
                        req_type = self.stack[-1]
                        if equivalent(req_type, self.last_type):
                            self.stack.pop()
                            ret_type = self.stack.pop()
                            assert self.stack.pop() == self.callstack.pop()
                            self.last_type = ret_type
                        else:
                            return False, "type mismatch"

                    self.callstack.append(tok)
                    self.stack.append(tok)
                    self.stack.append('Boolean')
                    self.stack.append(self.last_type)
                    self.update(tok)
                    self.last_token = tok
                    return True
                elif self.last_token in [')', ']']:
                    self.callstack.append(tok)
                    self.stack.append(tok)
                    self.stack.append('Boolean')
                    self.stack.append(self.last_type[0])
                    self.update(tok)
                    self.last_token = tok
                    return True
                else:
                    return False, 'type mismatch'
            return False, 'Unknown Operator'
            pass

        elif tok in keyword:
            if tok == 'new':
                self.last_type = [self.stack[-1]]
                self.last_token = 'new'
                self.update(tok)
                return True
            elif tok == 'instanceof':
                req_type = self.stack[-1]
                if equivalent(req_type, 'Boolean'):
                    self.update(tok)
                    self.last_type = self.last_token.return_type if isinstance(self.last_token, TokenInstance) else [
                        '<T>']
                    self.last_token = 'instanceof'
                    return True
                else:
                    return False, 'type mismatch'
            elif tok == 'class':
                if self.last_token == '.':
                    self.last_token = 'class'
                    self.update(tok)
                    return True
                else:
                    return False, 'type mismatch'
            elif tok == 'this':
                self.last_type = ['local']
                self.last_token = 'this'
                self.update(tok)
                return True
            else:
                return False, 'invalid %s' % tok

        else:
            candidates = self.local_vocab.name2insts(tok)

            # Filter
            filtered = []
            if len(candidates) == 0:
                return False, 'token not found'
            elif len(candidates) == 1:
                filtered = [candidates[0]]
            else:
                if self.last_token == 'new':
                    for candidate in candidates:
                        if candidate.is_constructor:
                            filtered.append(candidate)
                elif self.last_token == 'instanceof':
                    for candidate in candidates:
                        if candidate.is_type:
                            filtered.append(candidate)
                elif self.last_token == '.':
                    if 'local' in self.last_type:
                        for candidate in candidates:
                            if candidate.is_local or candidate.is_literal:
                                filtered.append(candidate)
                    else:
                        for candidate in candidates:
                            if equivalent(candidate.belonged_type, self.last_type) or candidate.is_api:
                                filtered.append(candidate)
                                pass

                elif self.last_token in separators or self.last_token in operators:
                    for candidate in candidates:
                        if candidate.is_local or candidate.is_literal or candidate.is_type:
                            if candidate.return_type is not None:
                                filtered.append(candidate)


                else:
                    filtered = candidates
            # Update
            candidates = filtered

            if len(candidates) >= 2:
                self.update(tok)
                self.drop_constrain = True
                return True
            elif len(candidates) == 0:
                self.update(tok)
                self.drop_constrain = True
                return True
            candidate = candidates[0]
            if isinstance(candidate.return_type, str):
                need_special_process = re.match(r'\[MISSING:(.+)\]', candidate.return_type)
                if need_special_process is not None:
                    before = need_special_process.group()
                    after = before[9:-1]
                    candidate.return_type = after
            elif isinstance(candidate.return_type, list):
                new_return_type = []
                for type in candidate.return_type:
                    need_special_process = re.match(r'\[MISSING:(.+)\]', type)
                    if need_special_process is not None:
                        before = need_special_process.group()
                        after = before[9:-1]
                        new_return_type.append(after)
                candidate.return_type = new_return_type
            else:
                if candidate.return_type is None and candidate.name == candidate.token:
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = ['T']
                    return True
                pass

            if candidate.is_param:

                if self.callstack[-1] == '(':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = [candidate.return_type]
                    return True

                elif self.callstack[-1] == '[':
                    if candidate.return_type not in self.closure_dict.keys():
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    if 'int' in self.closure_dict[candidate.return_type] or 'Integer' in self.closure_dict[
                        candidate.return_type]:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        return False, 'type mismatch(dv)'

                elif self.callstack[-1] == '{':
                    self.update(tok)
                    return True

                elif self.callstack[-1] == '<assertion>' and self.stack[-1] == '<default>':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = [candidate.return_type]
                    return True

                else:
                    if self.last_token == '!':
                        if 'boolean' in self.closure_dict[candidate.return_type] or 'Boolean' in self.closure_dict[
                            candidate.return_type]:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [candidate.return_type]
                            return True
                        else:
                            return False, 'type mismatch(dv)'

                    elif self.last_token in ['--', '++']:
                        if candidate.return_type in ['int', 'Integer', 'Long', 'long']:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [candidate.return_type]
                            return True
                        else:
                            return False, 'type mismatch(dv)'

                    elif self.last_token == '.':
                        if candidate.belonged_type in self.last_type:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [candidate.return_type]
                            return True
                        else:
                            return False, 'type mismatch(dv)'

                    else:
                        req_type = self.stack[-1]
                        if '<' in candidate.return_type and '>' in candidate.return_type:
                            return_type = candidate.return_type.split('<')[0].strip()
                            pass
                        else:
                            return_type = candidate.return_type
                        if return_type in self.closure_dict.keys():
                            if len(self.closure_dict[return_type]) == 1:
                                self.update(tok)
                                self.last_token = candidate
                                self.last_type = [return_type]
                                return True
                            if (isinstance(req_type, list) and has_common_element(req_type,
                                                                                  self.closure_dict[return_type])) or (
                                    isinstance(req_type, str) and equivalent(req_type, self.closure_dict[return_type])):
                                self.update(tok)
                                self.last_token = candidate
                                self.last_type = [return_type]
                                return True
                            else:
                                return False, 'type mismatch(dv)'
                        else:
                            self.update(tok)
                            self.last_token = candidate
                            self.last_type = [return_type]
                            return True
            elif candidate.is_method or candidate.is_api:
                if self.last_token == '!':
                    if 'boolean' in self.closure_dict[candidate.return_type] or 'Boolean' in self.closure_dict[
                        candidate.return_type]:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        return False, 'type mismatch(dv)'
                else:
                    req_type = self.stack[-1]
                    tmp_closure = []
                    if candidate.return_type not in self.closure_dict.keys():
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        tmp_closure = self.closure_dict[candidate.return_type]
                    if has_common_element(req_type, tmp_closure):
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    elif candidate.is_api:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    elif len(tmp_closure) == 1:
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = [candidate.return_type]
                        return True
                    else:
                        return False, 'type mismatch(dv)'
            elif candidate.is_type:
                if self.last_token == 'instanceof':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = 'Boolean'
                    return True
                elif self.last_token in separators:
                    if self.last_token == '(' or self.last_token == ',':
                        self.update(tok)
                        self.last_token = candidate
                        self.last_type = candidate.name
                        return True
                    else:
                        return False, 'type mismatch(dv)'
                elif self.last_token == 'new':
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = candidate.name
                    return True
                else:
                    return False, 'type mismatch(dv)'
            elif candidate.is_literal:
                req_type = self.stack[-1]
                if candidate.return_type is None:
                    candidate.return_type = candidate.token.split('&')[0]

                if self.callstack[-1] == '[':
                    req_type = 'number'

                if req_type == '<default>' or req_type == 'T' or req_type == '<T>':
                    self.update(tok)
                    self.last_token = candidate
                    if candidate.name == 'null':
                        candidate.return_type = 'Object'
                    self.last_type = [candidate.return_type]
                    return True
                elif equivalent(req_type, candidate.return_type):
                    self.update(tok)
                    self.last_token = candidate
                    self.last_type = req_type
                    return True
                else:
                    return False, 'type mismatch(dv)'

        pass

    def init(self, idx):
        word = tokenizer.convert_ids_to_tokens(idx)
        self.ans.append(word)
        self.ans_ids.append(idx)
        self.assertids = pad_list([idx], self.max_query_len)
        self.query = [idx]
        self.last_token = word


def BeamSearch_Pure_T5(model, batches, assertions, beamsize, config, device):
    batch_size = len(batches['context'])
    #import ipdb
    #ipdb.set_trace()
    codelen = config.query_max_len
    with torch.no_grad():
        beams = {}
        for i in range(batch_size):
            beams[i] = [SearchNode_T5(config)]
            beams[i][0].init(batches['query'][i][0])
        index = 0
        endnum = {}
        continueSet = {}
        while True:
            # print(index, batch_size)
            currCodeLen = min(index + 2, codelen)
            tmpbeam = {}
            ansV = {}

            if len(endnum) == batch_size:  # 如果已经生成结束的case数量等于batch size，就生成完了，退出beam search
                break
            if index >= codelen:  # 如果达到了最长的生成长度，不再继续生成
                break

            for p in range(beamsize):
                query = []
                validnum = []
                for i in range(batch_size):
                    # 每个batch里的每个instance只保存beam size个beam，如果超过了，则不要了
                    if p >= len(beams[i]):
                        continue
                    x = beams[i][p]
                    if x.finish:
                        ansV.setdefault(i, []).append(x)
                        continue
                    if not x.finish and x.ans[-1] == 2:
                        x.finish = True
                        ansV.setdefault(i, []).append(x)
                    else:
                        validnum.append(i)
                        query.append(x.assertids)
                if len(query) == 0:
                    continue

                newbatch = {}
                #import ipdb
                #ipdb.set_trace()
                newbatch['query'] = torch.tensor(query).to(device)
                newbatch['context'] = batches['context'][validnum].to(device)
                #newbatch['context'] = batches['context'][validnum].to(device)
                newbatch['para_mask_batch_1'] = batches['para_mask_batch_1'][validnum].to(device)
                newbatch['para_mask_batch_2'] = batches['para_mask_batch_2'][validnum].to(device)
                newbatch['para_batch_1'] = []
                newbatch['para_batch_2'] = []
                newbatch['seq_len_batch_1'] = []
                newbatch['seq_len_batch_2'] = []
                valid_index_tensor = torch.tensor(validnum).to(device)
                for x in batches:
                    if x in ['para_batch_1', 'para_batch_2']:
                        #import ipdb
                        #ipdb.set_trace()
                        #tmp_0 = batches[x][0].to(device)
                        #tmp_1 = batches[x][1].to(device)
                        #tmp_2 = batches[x][2].to(device)
                        #tmp_3 = batches[x][3].to(device)
                        #tmp_4 = batches[x][4].to(device)
                        #valid_index_tensor = valid_index_tensor.to(device)
                        newbatch[x].append((torch.index_select(batches[x][0], 1, valid_index_tensor)))
                        newbatch[x].append((torch.index_select(batches[x][1], 1, valid_index_tensor)))
                        newbatch[x].append((torch.index_select(batches[x][2], 1, valid_index_tensor)))
                        newbatch[x].append((torch.index_select(batches[x][3], 1, valid_index_tensor)))
                        newbatch[x].append((torch.index_select(batches[x][4], 1, valid_index_tensor)))
                    elif x in ['seq_len_batch_1', 'seq_len_batch_2']:
                        newbatch[x].append((torch.index_select(batches[x][0].to(device), 0, valid_index_tensor)))
                        newbatch[x].append((torch.index_select(batches[x][1].to(device), 0, valid_index_tensor)))
                        newbatch[x].append((torch.index_select(batches[x][2].to(device), 0, valid_index_tensor)))
                        newbatch[x].append((torch.index_select(batches[x][3].to(device), 0, valid_index_tensor)))
                        newbatch[x].append((torch.index_select(batches[x][4].to(device), 0, valid_index_tensor)))
                    else:
                        newbatch[x] = batches[x][validnum].to(device)
                
                #import ipdb
                #ipdb.set_trace()

                result = model(input_ids=newbatch,
                               attention_mask=newbatch['context'].ne(0).long(),
                               decoder_input_ids=newbatch['query'],
                               decoder_attention_mask=newbatch['query'].ne(
                                   0).long(),
                               output_attentions=False,
                               output_hidden_states=False,
                               return_dict=True)
                results = F.softmax(result['logits'], dim=-1)
                #import ipdb
                #ipdb.set_trace()
                currIndex = 0
                for j in range(batch_size):
                    if j not in validnum:
                        continue
                    x = beams[j][p]
                    tmpbeamsize = 0
                    # np.negative(results[currIndex, index])
                    #import ipdb
                    #ipdb.set_trace()
                    result = results[currIndex, index]
                    currIndex += 1
                    cresult = result  # np.negative(result)
                    complete_pred = torch.argsort(cresult, descending=True)
                    indexs = complete_pred
                    for i in range(len(indexs)):
                        if i > 2 * beamsize:
                            break
                        prob = x.prob + np.log(
                            cresult[indexs[i]].item())  # copynode.prob = copynode.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(j, []).append(
                            [prob, indexs[i].item(), x])  # tmpbeam.setdefault(j, []).append(copynode)
            
            #import ipdb
            #ipdb.set_trace()

            for i in range(batch_size):
                if i in ansV:
                    if len(ansV[i]) == beamsize:
                        endnum[i] = 1
            for j in range(batch_size):
                if j in tmpbeam:
                    if j in ansV:
                        for x in ansV[j]:
                            tmpbeam[j].append([x.prob, -1, x])
                    tmp = sorted(tmpbeam[j], key=lambda x: x[0], reverse=True)[:beamsize]
                    beams[j] = []
                    for x in tmp:
                        if x[1] != -1:
                            copynode = pickle.loads(pickle.dumps(x[2]))
                            copynode.update(x[1])
                            copynode.prob = x[0]
                            beams[j].append(copynode)
                        else:
                            beams[j].append(x[2])
            index += 1
        return beams


def BeamSearch(model, batches, vocabs, beamsize, assertions, config, device):
    batch_size = len(batches['context'])
    codelen = config.query_max_len
    with torch.no_grad():
        beams = {}
        for i in range(batch_size):
            beams[i] = [SearchNode(batches, vocabs, config, i)]
            beams[i][0].init(assertions[i][0])

        index = 0
        endnum = {}
        continueSet = {}
        while True:
            # print(index, batch_size)
            currCodeLen = min(index + 2, codelen)
            tmpbeam = {}
            ansV = {}
            if len(endnum) == batch_size:
                break
            if index >= codelen:
                break
            for p in range(beamsize):
                query = []
                validnum = []
                for i in range(batch_size):
                    if p >= len(beams[i]):
                        continue
                    x = beams[i][p]
                    if x.finish:
                        ansV.setdefault(i, []).append(x)
                        continue
                    if not x.finish and x.ans[-1] == '</s>':
                        x.finish = True
                        ansV.setdefault(i, []).append(x)
                    else:
                        validnum.append(i)
                        query.append(x.assertids)
                if len(query) == 0:
                    continue
                newbatch = {}
                newbatch['query'] = torch.tensor(query).to(device)
                newbatch['context'] = batches['context'][validnum].to(device)
                newbatch['vocab'] = batches['vocab'][validnum].to(device)
                result = model(newbatch)
                results = result
                currIndex = 0
                for j in range(batch_size):
                    if j not in validnum:
                        continue
                    x = beams[j][p]
                    tmpbeamsize = 0
                    # np.negative(results[currIndex, index])
                    result = results[currIndex, index]
                    currIndex += 1
                    cresult = result  # np.negative(result)
                    indexs = torch.argsort(result, descending=True)
                    for i in range(len(indexs)):
                        if i > 2 * beamsize:
                            break
                        prob = x.prob + np.log(
                            cresult[indexs[i]].item())  # copynode.prob = copynode.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(j, []).append(
                            [prob, indexs[i].item(), x])  # tmpbeam.setdefault(j, []).append(copynode)
            for i in range(batch_size):
                if i in ansV:
                    if len(ansV[i]) == beamsize:
                        endnum[i] = 1
            for j in range(batch_size):
                if j in tmpbeam:
                    if j in ansV:
                        for x in ansV[j]:
                            tmpbeam[j].append([x.prob, -1, x])
                    tmp = sorted(tmpbeam[j], key=lambda x: x[0], reverse=True)[:beamsize]
                    beams[j] = []
                    for x in tmp:
                        if x[1] != -1:
                            copynode = pickle.loads(pickle.dumps(x[2]))
                            copynode.update(x[1])
                            copynode.prob = x[0]
                            beams[j].append(copynode)
                        else:
                            beams[j].append(x[2])
            index += 1
        return beams


def BeamSearchWithType(model, batches, vocabs, beamsize, assertions, closures, config, device, debugging=False):
    batch_size = len(batches['context'])
    codelen = config.query_max_len
    with torch.no_grad():
        beams = {}
        for i in range(batch_size):
            beams[i] = [SearchNodeWithType(vocabs[i], config, closures[i])]
            # 能不能够在这里进行断言类型和（的初始化？
            # 按照和toga对比的规则，即只对第一个参数进行对比，这里可以将assertion的type和（进行初始化
            beams[i][0].init(assertions[i][0])
        index = 0
        endnum = {}
        continueSet = {}
        while True:
            # print(index, batch_size)
            currCodeLen = min(index + 2, codelen)
            tmpbeam = {}
            ansV = {}
            if len(endnum) == batch_size:
                break
            if index >= codelen:
                break
            for p in range(beamsize):
                query = []
                validnum = []
                for i in range(batch_size):
                    if p >= len(beams[i]):
                        continue
                    x = beams[i][p]
                    if x.finish:
                        ansV.setdefault(i, []).append(x)
                        continue
                    if not x.finish and x.ans[-1] == '</s>':
                        x.finish = True
                        ansV.setdefault(i, []).append(x)
                    else:
                        validnum.append(i)
                        query.append(x.assertids)
                if len(query) == 0:
                    continue
                newbatch = {}
                newbatch['query'] = torch.tensor(query).to(device)
                newbatch['context'] = batches['context'][validnum].to(device)
                newbatch['vocab'] = batches['vocab'][validnum].to(device)
                result = model(newbatch)
                results = result
                currIndex = 0
                for j in range(batch_size):
                    if j not in validnum:
                        continue
                    x = beams[j][p]
                    tmpbeamsize = 0
                    # np.negative(results[currIndex, index])
                    result = results[currIndex, index]
                    currIndex += 1
                    cresult = result  # np.negative(result)
                    indexs = torch.argsort(result, descending=True).detach().cpu().numpy()

                    for i in range(len(indexs)):
                        if i > 4 * beamsize:
                            break
                        prob = x.prob + np.log(
                            cresult[indexs[i]].item())  # copynode.prob = copynode.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(j, []).append([prob, indexs[i].item(), x])
            for i in range(batch_size):
                if i in ansV:
                    if len(ansV[i]) == beamsize:
                        endnum[i] = 1
            for j in range(batch_size):
                if j in tmpbeam:
                    if j in ansV:
                        for x in ansV[j]:
                            tmpbeam[j].append([x.prob, -1, x])
                    tmp = sorted(tmpbeam[j], key=lambda x: x[0], reverse=True)
                    beams[j] = []
                    for x in tmp:
                        if x[1] != -1:
                            copynode = pickle.loads(pickle.dumps(x[2]))
                            try:
                                flag = copynode.update_with_type(x[1])
                            except Exception as e:
                                flag = False
                            if flag == True:
                                copynode.prob = x[0]
                                beams[j].append(copynode)
                        else:
                            beams[j].append(x[2])
                        if len(beams[j]) == beamsize:
                            break

            index += 1
        return beams


def BeamSearchWithType_T5(model, batches, vocabs, beamsize, assertions, closures, config, device, debugging=False):
    batch_size = len(batches['context'])
    codelen = config.query_max_len
    with torch.no_grad():
        beams = {}
        for i in range(batch_size):
            beams[i] = [SearchNodeWithType_T5(vocabs[i], config, closures[i])]
            beams[i][0].init(tokenizer.convert_tokens_to_ids(assertions[i][0]))
        index = 0
        endnum = {}
        continueSet = {}
        while True:
            # print(index, batch_size)
            currCodeLen = min(index + 2, codelen)
            tmpbeam = {}
            ansV = {}
            if len(endnum) == batch_size:
                break
            if index >= codelen:
                break
            for p in range(beamsize):
                query = []
                validnum = []
                for i in range(batch_size):
                    if p >= len(beams[i]):
                        continue
                    x = beams[i][p]
                    if x.finish:
                        ansV.setdefault(i, []).append(x)
                        continue
                    if not x.finish and x.ans[-1] == '</s>':
                        x.finish = True
                        ansV.setdefault(i, []).append(x)
                    else:
                        validnum.append(i)
                        query.append(x.assertids)
                if len(query) == 0:
                    continue
                newbatch = {}
                newbatch['query'] = torch.tensor(query).to(device)
                newbatch['context'] = batches['context'][validnum].to(device)
                result = model(input_ids=newbatch['context'],
                               attention_mask=newbatch['context'].ne(0).long(),
                               decoder_input_ids=newbatch['query'],
                               decoder_attention_mask=newbatch['query'].ne(
                                   0).long(),
                               output_attentions=False,
                               output_hidden_states=False,
                               return_dict=True)

                results = F.softmax(result['logits'], dim=-1)
                currIndex = 0
                for j in range(batch_size):
                    if j not in validnum:
                        continue
                    x = beams[j][p]
                    tmpbeamsize = 0
                    # np.negative(results[currIndex, index])
                    result = results[currIndex, index]
                    currIndex += 1
                    cresult = result  # np.negative(result)
                    indexs = torch.argsort(result, descending=True).detach().cpu().numpy()

                    for i in range(len(indexs)):
                        if i > 4 * beamsize:
                            break
                        prob = x.prob + np.log(
                            cresult[indexs[i]].item())  # copynode.prob = copynode.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(j, []).append([prob, indexs[i].item(), x])
            for i in range(batch_size):
                if i in ansV:
                    if len(ansV[i]) == beamsize:
                        endnum[i] = 1
            for j in range(batch_size):
                if j in tmpbeam:
                    if j in ansV:
                        for x in ansV[j]:
                            tmpbeam[j].append([x.prob, -1, x])
                    tmp = sorted(tmpbeam[j], key=lambda x: x[0], reverse=True)
                    beams[j] = []
                    for x in tmp:
                        if x[1] != -1:
                            copynode = pickle.loads(pickle.dumps(x[2]))
                            try:
                                flag = copynode.update_with_type(x[1])
                            except Exception as e:
                                flag = False
                            if flag == True:
                                copynode.prob = x[0]
                                beams[j].append(copynode)
                        else:
                            beams[j].append(x[2])
                        if len(beams[j]) == beamsize:
                            break

            index += 1
        return beams


def filter_instances():
    global data, assertion, token
    from utils.Config import Configurable
    from collections import Counter
    config = Configurable('config/t5_attention-based_cpynet_no_emb.ini')
    from Modules.SumDataset import readpickle
    test = readpickle(os.path.join(config.data_dir, 'processtestdata.pkl'))
    with open(os.path.join(config.data_dir, 'testassertion.pkl'), 'rb') as f:
        assertions = pickle.load(f)
    with open(os.path.join(config.data_dir, 'testvocab.pkl'), 'rb') as f:
        vocabs = pickle.load(f)
    with open(os.path.join(config.data_dir, 'testclosure.pkl'), 'rb') as f:
        closures = pickle.load(f)
    f = open(os.path.join(config.data_dir, 'test_keys.pkl'), 'rb')
    test_keys = pickle.load(f)
    f.close()
    print(len(test))
    print(len(assertions))
    print(len(vocabs))
    print(len(closures))

    assert len(test) == len(assertions) == len(vocabs) == len(closures) == len(test_keys)
    # ff1 = open(os.path.join(config.data_dir, "filtered_processtestdata.pkl"), 'wb')
    # ff2 = open(os.path.join(config.data_dir, "filtered_testassertion.pkl"), 'wb')
    # ff3 = open(os.path.join(config.data_dir, "filtered_testvocab.pkl"), 'wb')
    # ff4 = open(os.path.join(config.data_dir, "filtered_testclosure.pkl"), 'wb')
    filtered = []
    tbar = tqdm.tqdm(len(test))
    valid_keys = []
    err = Counter()
    num_drop_constrain = 0
    for i in range(len(test)):
        data = test[i]
        assertion = assertions[i]
        vocab = vocabs[i]
        closure = closures[i]
        node = SearchNodeWithType(vocab, config, closure)
        node.init(assertion[0])
        flag = True
        for token in assertion[1:]:
            res = node.update_with_type(vocab.token2id(token))
            if res != True:
                # node.update_with_type(vocab.token2id(token))
                if res is None:
                    err['None'] += 1
                else:
                    err[res[1]] += 1
                filtered.append([assertion, vocab, closure])
                flag = False
                break
            else:
                if node.drop_constrain:
                    num_drop_constrain += 1
                    break
        if flag:
            # pickle.dump(data, ff1)
            # pickle.dump(assertion, ff2)
            # pickle.dump(vocab, ff3)
            # pickle.dump(closure, ff4)
            valid_keys.append(test_keys[i])
        tbar.update(1)
        i += 1
    # ff1.close()
    # ff2.close()
    # ff3.close()
    # ff4.close()
    # print(os.path.join(config.data_dir, 'valid_test_ids.txt'))
    with open(os.path.join(config.data_dir, 'valid_test_keys.pkl'), 'wb') as f:
        pickle.dump(valid_keys, f)
    # with open(os.path.join(config.data_dir, 'valid_test_keys.pkl'), 'rb') as f:
    #     keys = pickle.load(f)
    #     print(keys)
    #     print(type(keys))
    # f1 = open(os.path.join(config.data_dir, 'failed_testassertion.pkl'), 'wb')
    # f2 = open(os.path.join(config.data_dir, 'failed_testvocab.pkl'), 'wb')
    # f3 = open(os.path.join(config.data_dir, 'failed_testclosure.pkl'), 'wb')
    # for data in filtered:
    #     pickle.dump(data[0], f1)
    #     pickle.dump(data[1], f2)
    #     pickle.dump(data[2], f3)
    # f1.close()
    # f2.close()
    # f3.close()
    print("#failed: %d" % len(filtered))
    print("#dropped constrain: %d" % num_drop_constrain)
    print(err)


def generation_post_processing(config_file='config/t5_small.ini'):
    global data, assertion, token
    from utils.Config import Configurable
    from collections import Counter
    config = Configurable(config_file)
    from Modules.SumDataset import readpickle
    test = readpickle(os.path.join(config.data_dir, 'processtestdata.pkl'))
    with open(os.path.join(config.data_dir, 'testassertion.pkl'), 'rb') as f:
        assertions = pickle.load(f)
    vocabs = readpickle(os.path.join(config.data_dir, 'testvocab.pkl'))
    with open(os.path.join(config.data_dir, 'testclosure.pkl'), 'rb') as f:
        closures = pickle.load(f)
    with open(os.path.join(config.data_dir, 'test_keys.pkl'), 'rb') as f:
        keys = pickle.load(f)
    assert len(test) == len(assertions) == len(vocabs) == len(closures) == len(keys)
    # ff1 = open(os.path.join(config.data_dir, "filtered_processtestdata.pkl"), 'wb')
    # ff2 = open(os.path.join(config.data_dir, "filtered_testassertion.pkl"), 'wb')
    # ff3 = open(os.path.join(config.data_dir, "filtered_testvocab.pkl"), 'wb')
    # ff4 = open(os.path.join(config.data_dir, "filtered_testclosure.pkl"), 'wb')
    filtered = []
    tbar = tqdm.tqdm(len(test))
    valid_ids = []
    err = Counter()
    num_drop_constrain = 0
    for i in range(len(test)):
        data = test[i]
        assertion = assertions[i]
        vocab = vocabs[i]
        closure = closures[i]
        node = SearchNodeWithType(vocab, config, closure)
        node.init(assertion[0])
        flag = True
        for token in assertion[1:]:
            res = node.update_with_type(vocab.token2id(token))
            if res != True:
                # node.update_with_type(vocab.token2id(token))
                if res is None:
                    err['None'] += 1
                else:
                    err[res[1]] += 1
                filtered.append([assertion, vocab, closure])
                flag = False
                break
            else:
                if node.drop_constrain:
                    num_drop_constrain += 1
                    break
        if flag:
            # pickle.dump(data, ff1)
            # pickle.dump(assertion, ff2)
            # pickle.dump(vocab, ff3)
            # pickle.dump(closure, ff4)
            valid_ids.append(i)
        tbar.update(1)
        i += 1
    # ff1.close()
    # ff2.close()
    # ff3.close()
    # ff4.close()
    # print(os.path.join(config.data_dir, 'valid_test_ids.txt'))
    with open(os.path.join(config.data_dir, 'valid_test_ids.txt'), 'wb') as f:
        pickle.dump(valid_ids, f)
    # f1 = open(os.path.join(config.data_dir, 'failed_testassertion.pkl'), 'wb')
    # f2 = open(os.path.join(config.data_dir, 'failed_testvocab.pkl'), 'wb')
    # f3 = open(os.path.join(config.data_dir, 'failed_testclosure.pkl'), 'wb')
    # for data in filtered:
    #     pickle.dump(data[0], f1)
    #     pickle.dump(data[1], f2)
    #     pickle.dump(data[2], f3)
    # f1.close()
    # f2.close()
    # f3.close()
    print("#failed: %d" % len(filtered))
    print("#dropped constrain: %d" % num_drop_constrain)
    print(err)


def BeamSearch_Type_toga(model, batches, vocabs, beamsize, assertions, closures, config, device, debugging=False):
    batch_size = len(batches['context'])
    codelen = config.query_max_len
    with torch.no_grad():
        beams = {}
        for i in range(batch_size):
            beams[i] = [SearchNodeWithType(vocabs[i], config, closures[i])]
            # 能不能够在这里进行断言类型和（的初始化？
            # 按照和toga对比的规则，即只对第一个参数进行对比，这里可以将assertion的type和（进行初始化
            beams[i][0].init_toga(assertions[i])
        index = 2
        endnum = {}
        continueSet = {}
        while True:
            # print(index, batch_size)
            currCodeLen = min(index + 2, codelen)
            tmpbeam = {}
            ansV = {}
            if len(endnum) == batch_size:
                break
            if index >= codelen:
                break
            for p in range(beamsize):
                query = []
                validnum = []
                for i in range(batch_size):
                    if p >= len(beams[i]):
                        continue
                    x = beams[i][p]
                    if x.finish:
                        ansV.setdefault(i, []).append(x)
                        continue
                    if not x.finish and x.ans[-1] == '</s>':
                        x.finish = True
                        ansV.setdefault(i, []).append(x)
                    else:
                        validnum.append(i)
                        query.append(x.assertids)
                if len(query) == 0:
                    continue
                newbatch = {}
                newbatch['query'] = torch.tensor(query).to(device)
                newbatch['context'] = batches['context'][validnum].to(device)
                newbatch['vocab'] = batches['vocab'][validnum].to(device)
                result = model(newbatch)
                results = result
                currIndex = 0
                for j in range(batch_size):
                    if j not in validnum:
                        continue
                    x = beams[j][p]
                    tmpbeamsize = 0
                    # np.negative(results[currIndex, index])
                    result = results[currIndex, index]
                    currIndex += 1
                    cresult = result  # np.negative(result)
                    indexs = torch.argsort(result, descending=True).detach().cpu().numpy()

                    for i in range(len(indexs)):
                        if i > 4 * beamsize:
                            break
                        prob = x.prob + np.log(
                            cresult[indexs[i]].item())  # copynode.prob = copynode.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(j, []).append([prob, indexs[i].item(), x])
            for i in range(batch_size):
                if i in ansV:
                    if len(ansV[i]) == beamsize:
                        endnum[i] = 1
            for j in range(batch_size):
                if j in tmpbeam:
                    if j in ansV:
                        for x in ansV[j]:
                            tmpbeam[j].append([x.prob, -1, x])
                    tmp = sorted(tmpbeam[j], key=lambda x: x[0], reverse=True)
                    beams[j] = []
                    for x in tmp:
                        if x[1] != -1:
                            copynode = pickle.loads(pickle.dumps(x[2]))
                            try:
                                flag = copynode.update_with_type_toga(x[1])
                            except Exception as e:
                                flag = False
                            if flag == True:
                                copynode.prob = x[0]
                                beams[j].append(copynode)
                        else:
                            beams[j].append(x[2])
                        if len(beams[j]) == beamsize:
                            break

            index += 1
        return beams


if __name__ == '__main__':
    # f = open('datasets/our_data_t5/processtestdata.pkl', 'rb')
    # testInstances = pickle.load(f)
    # f.close()

    # filter_instances()
    generation_post_processing()
    # prefix = '<s> assertTrue ( correctStreamLength > reportedEvent . getFileLength ( ) + headerPdfVersion'.split()

    pass
