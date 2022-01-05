import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

#  load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)  # 加载到device(cuda或cpu)上
