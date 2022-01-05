from transformers import RobertaTokenizer, RobertaForMaskedLM, pipeline
# As stated in the paper, CodeBERT is not suitable for mask prediction task,
# while CodeBERT (MLM) is suitable for mask prediction task.

# We give an example on how to use CodeBERT(MLM) for mask prediction task.
model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

CODE = "if (x is not None) <mask> (x>1)"
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

outputs = fill_mask(CODE)
print(outputs)
# [{'score': 0.7236998081207275, 'token': 8,
# 'token_str': ' and', 'sequence': 'if (x is not None) and(x>1)'},
#  {'score': 0.10633757710456848, 'token': 359,
#  'token_str': ' &', 'sequence': 'if (x is not None) &(x>1)'},
#  {'score': 0.021604055538773537, 'token': 463,
#  'token_str': 'and', 'sequence': 'if (x is not None)and(x>1)'},
#  {'score': 0.021227575838565826, 'token': 4248,
#   'token_str': ' AND', 'sequence': 'if (x is not None) AND(x>1)'},
#  {'score': 0.016991276293992996, 'token': 114,
#  'token_str': ' if', 'sequence': 'if (x is not None) if(x>1)'}]
