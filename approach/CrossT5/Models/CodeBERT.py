from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

model = AutoModel.from_pretrained("microsoft/codebert-base")

