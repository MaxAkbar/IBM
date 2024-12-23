import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
model_path = "ibm-granite/granite-3.0-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

# change input text as desired
chat = [
    { "role": "user", "content": "Please list one IBM Research laboratory located in the United States. You should only output its name and location." },
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt").to(device)

# generate output tokens
output_tokens = model.generate(**input_tokens, max_new_tokens=100)

# decode output tokens into text
output_text = tokenizer.batch_decode(output_tokens)

# print output
print(output_text)