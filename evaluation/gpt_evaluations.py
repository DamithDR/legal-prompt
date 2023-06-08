import numpy as np
import pandas as pd
from sklearn import metrics

datasets = ['data/de.csv', 'data/fr.csv', 'data/it.csv']
languages = ['de', 'fr', 'it']
responses = ['data/responses/gpt4-de.txt', 'data/responses/gpt4-fr.txt', 'data/responses/gpt4-it.txt']
# responses = ['data/responses/chatgpt-de.txt', 'data/responses/chatgpt-fr.txt', 'data/responses/chatgpt-it.txt']
model = 'gpt4'
# model = 'chatgpt'

for dataset, language, file in zip(datasets, languages, responses):
    with open(file, 'r') as f:
        response_list = f.readlines()
        response_list = [res.strip() for res in response_list]
        indices=[]
        if 'context_length_exceeded' in response_list:
            array = np.array(response_list)
            indices = list(np.where(array == 'context_length_exceeded')[0])
            response_list = [v for i, v in enumerate(response_list) if i not in indices]
        predicted = [eval(i) for i in response_list]
        data = pd.read_csv(dataset,sep='\t')
        gold = data['label'].to_list()
        gold = [v for i, v in enumerate(gold) if i not in indices]

        data = pd.read_csv(dataset, sep='\t')
        with open(f'evaluation/results/{model}-results-for-{language}.txt', 'w') as r:
            r.write(metrics.classification_report(gold, predicted, digits=6))
