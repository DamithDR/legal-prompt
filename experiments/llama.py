import pandas as pd
import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from sklearn import metrics
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

model = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation",  # task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_memory={2: "20GIB", 3: "20GIB"},
    max_length=200,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})
template = """
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

while True:
    question = input("User Input : ")
    response = llm_chain.run(question)
    print(response)
