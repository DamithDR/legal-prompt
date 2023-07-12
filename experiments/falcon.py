import pandas as pd
import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from sklearn import metrics
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

# model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct
model = "tiiuae/falcon-40b-instruct"  # tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation",  # task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})
template = """
Think you are a judge in swiss courts, and you are need to judge the following case dellimitted by ```
                        - Decide whether the case is dismissal or approval
                        - If the case is dismissal, just reply : 0
                        - If the case is approval, just reply : 1
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

test = pd.read_csv('data/de.csv', sep="\t")
final_predictions = []

for text in test['text'].to_list():
    question = f""" Is the following case a dismissal or approval? case : ```{text}```
                        """
    response = llm_chain.run(question)
    if response.split(',')[0].strip() == "1":
        final_predictions.append(1)
    else:
        final_predictions.append(0)

test['predictions'] = final_predictions
metrics.classification_report([i for i in test['label'].to_list()], final_predictions, digits=6)
