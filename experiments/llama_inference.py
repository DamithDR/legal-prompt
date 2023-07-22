import sys

import torch
import transformers

from transformers import AutoTokenizer

model_name = input("Input the model you want : ")

# model = "meta-llama/Llama-2-7b-hf"
# model = "meta-llama/Llama-2-32b-hf"
model = model_name

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

while True:
    question = input("\nUser Input : ")
    if 'quit'.__eq__(question):
        sys.exit(0)
    sequences = pipeline(
        question,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
