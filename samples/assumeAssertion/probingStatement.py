#!/usr/bin/python
# -*- coding: UTF-8 -*-
from re import template
import sys
import json
import torch
import random
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM, pipeline

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

sys.path.append("E:/python/CodeBERT/samples/assumeAssertion/")
# E:\python\CodeBERT\samples\assumeAssertion\raw_train.jsonl

methodNum = 0
printSize = 1  # 控制台打印信息数
index = 0
with open('E:/python/CodeBERT/samples/assumeAssertion/raw_train.jsonl',
          encoding='utf-8') as dataset_file:
    for line in dataset_file:
        methodNum = methodNum + 1
        data = json.loads(line)
        if index < printSize:
            index = index + 1
            print(index, end='\n')
            testMethodOrigin = 'void '+data[1]+'()'+data[2].replace('???;', data[6])
            # print(testMethodOrigin)

            # focal方法体及其上下文
            # focal_method = data[4]
            # focal_method_with_context = focal_method
            # for text in data[5]:
            #     focal_method_with_context = focal_method_with_context + '\n' + text
            # print(focal_method_with_context)

            # 将assertion，token序列化，并mask
            assertion_origin = data[6]
            assert_tokens = tokenizer.tokenize(assertion_origin)
            # print(assert_tokens)
            assert_tokens_len = len(assert_tokens)
            # print(assert_tokens_len)
            masked_assertion = ''
            random_index = random.randint(4, assert_tokens_len-1)
            for i in range(assert_tokens_len):
                # masked_assertion = masked_assertion + '<mask>'
                temp_token = assert_tokens[i]
                if temp_token.startswith('Ġ'):
                    temp_token = temp_token[1:]
                if i == random_index:
                    masked_assertion = masked_assertion + '<mask>'
                    # print('<mask>')
                else:
                    masked_assertion = masked_assertion + temp_token
                    # print(temp_token)
            print(masked_assertion)

            # 将测试代码中的'???;'替换成masked_assertion
            test_method_with_mask = 'void '+data[1]+'()'+data[2].replace('???;', masked_assertion)
            # print(test_method_with_mask)

            code_with_mask = testMethodOrigin+test_method_with_mask
            fill_mask = pipeline('fill-mask', model=model_mlm, tokenizer=tokenizer_mlm)
            print(code_with_mask)
            outputs = fill_mask(code_with_mask)
            print(assert_tokens)
            for entry in outputs:
                print(entry['token_str'], end=', ')
            # for entries in outputs:
            #     for entry in entries:
            #         print(entry['token_str'], end=', ')
            #     print('\n')

# print(methodNum)
