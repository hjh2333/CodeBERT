import torch
from transformers import RobertaTokenizer, RobertaModel
# , RobertaConfig

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

# NL-PL Embeddings
nl_tokens = tokenizer.tokenize("return maximum value")
print(nl_tokens)  # ['return', 'Ġmaximum', 'Ġvalue']
code_tokens = tokenizer.\
              tokenize("def max(a,b): if a>b: return a else return b")
print(code_tokens)
# ['def', 'Ġmax', '(', 'a', ',', 'b', '):', 'Ġif', 'Ġa', '>',
#  'b', ':', 'Ġreturn', 'Ġa', 'Ġelse', 'Ġreturn', 'Ġb']
tokens = [tokenizer.cls_token] + nl_tokens +\
         [tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
print(tokens)
# ['<s>', 'return', 'Ġmaximum', 'Ġvalue', '</s>', 'def', 'Ġmax',
#  '(', 'a', ',', 'b', '):', 'Ġif', 'Ġa', '>', 'b', ':', 'Ġreturn',
#   'Ġa', 'Ġelse', 'Ġreturn', 'Ġb', '</s>']
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
print(tokens_ids)
# [0, 30921, 4532, 923, 2, 9232, 19220, 1640, 102, 6, 428, 3256,
#  114, 10, 15698, 428, 35, 671, 10, 1493, 671, 741, 2]
context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
print(context_embeddings)
# 输出应是torch.Size([1, 23, 768])
# tensor([[[-0.1423,  0.3766,  0.0443,  ..., -0.2513, -0.3099,  0.3183],
#          [-0.5739,  0.1333,  0.2314,  ..., -0.1240, -0.1219,  0.2033],
#          [-0.1579,  0.1335,  0.0291,  ...,  0.2340, -0.8801,  0.6216],
#          ...,
#          [-0.4042,  0.2284,  0.5241,  ..., -0.2046, -0.2419,  0.7031],
#          [-0.3894,  0.4603,  0.4797,  ..., -0.3335, -0.6049,  0.4730],
#          [-0.1433,  0.3785,  0.0450,  ..., -0.2527, -0.3121,  0.3207]]],
#        grad_fn=<NativeLayerNormBackward0>)
