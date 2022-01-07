# import random
# list = [1, 2]
# list_len = len(list)
# rand_index = random.randint(0,list_len-1)
# print(type(rand_index), rand_index)
# list[rand_index] = 3 
# print(list)
# print('fff'[1:])
# st = 'Ġactual'
# print(st.startswith('Ġ'))
# if st.startswith('Ġ'):
#     print(st[1:])
from transformers import RobertaTokenizer, RobertaModel
# , RobertaConfig

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

# NL-PL Embeddings
nl_tokens = tokenizer.tokenize("if(!a.equals(b)) return Math.max(a,b);")
print(nl_tokens)
# ['if', '(', '!', 'a', '.', 'equ', 'als', '(', 'b', '))', 'Ġreturn', 'ĠMath', '.', 'max', '(', 'a', ',', 'b', ');']
