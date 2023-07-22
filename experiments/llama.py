import sys

import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

model_name = input("Input the model you want : ")

# model = "meta-llama/Llama-2-13b-hf"
# model = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipeline = pipeline(
    "text-generation",  # task
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=10,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 1,'do_sample':True})
template = """
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

while True:
    question = input("User Input : ")
    if 'quit'.__eq__(question):
        sys.exit(0)
    response = llm_chain.run(question)
    print(response)
