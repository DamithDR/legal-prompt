import sys
import time

import openai
import pandas as pd
from openai import InvalidRequestError

openai.api_key = input("Please enter your OpenAI API key : ")
# datasets = ['data/de.csv', 'data/fr.csv', 'data/it.csv']
# languages = ['de', 'fr', 'it']
# start_points = [173, 0, 0]

datasets = ['data/it.csv']
languages = ['it']
start_points = [718]

for dataset, language, start in zip(datasets, languages, start_points):
    df = pd.read_csv(dataset, sep='\t')
    text_list = df['text'].to_list()

    for i in range(start, len(text_list)):
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
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    messages=[
                        {"role": "user", "content": message}
                    ]
                )
                resp = response['choices'][0]['message']['content']
                with open(f'data/responses/chatgpt-{language}.txt', 'a') as f:
                    f.write(str(resp).replace('\n', '##') + '\n')
                print(resp)
                time.sleep(20)
                break
            except InvalidRequestError as err:
                if err.code == 'context_length_exceeded':
                    resp = 'context_length_exceeded'
                    with open(f'data/responses/chatgpt-{language}.txt', 'a') as f:
                        f.write(str(resp).replace('\n', '##') + '\n')
                        time.sleep(20)
                        break
            except:
                print(f'exception happened in processing {i}th document')
                time.sleep(20)
