from Bard import Chatbot

import sys
import time

import pandas as pd

token = input('Enter your token : ')
chatbot = Chatbot(token)
datasets = ['data/de.csv', 'data/fr.csv', 'data/it.csv']
languages = ['de', 'fr', 'it']

for dataset, language in zip(datasets, languages):
    df = pd.read_csv(dataset, sep='\t')
    text_list = df['text'].to_list()

    for i in range(len(text_list)):
        print(f'processing text no : {i}')
        text = text_list[i]
        message = f"""
                    Think you are a judge in swiss courts, and you are need to judge the following case dellimitted by ```
                    - Decide whether the case is dismissal or approval
                    - If the case is dismissal, just reply : 0
                    - If the case is approval, just reply : 1
                    You do not have to provide explanations.
                    case : ```{text}```
                    """
        while True:
            try:
                # trying until it succeeds
                response = chatbot.ask(message)
                resp = response['content']
                with open(f'data/responses/bard-{language}.txt', 'a') as f:
                    f.write(str(resp).replace('\n', '##') + '\n')
                print(resp)
                sys.exit(0)
                time.sleep(20)
                break
            except:
                print(f'exception happened in processing {i}th document')
                time.sleep(20)
