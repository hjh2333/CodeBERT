#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import json
import re
sys.path.append("E:/python/CodeBERT/samples/assumeAssertion/")
# E:\python\CodeBERT\samples\assumeAssertion\raw_train.jsonl

methodNum = 0
printSize = 2  # 控制台打印信息数
index = 0
with open('E:/python/CodeBERT/samples/assumeAssertion/raw_train.jsonl',
          encoding='utf-8') as dataset_file:
    for line in dataset_file:
        # 处理jsonl文件里的每行
        methodNum = methodNum + 1
        data = json.loads(line)
        if index < printSize:
            index = index + 1
            print(index, end='\n')
            for entry in data:
                print(entry)

print(methodNum)
