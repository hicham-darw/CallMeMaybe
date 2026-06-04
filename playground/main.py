from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

print(tokenizer)
print(type(tokenizer))
