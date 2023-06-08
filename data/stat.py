import pandas as pd


datasets = ['data/de.csv', 'data/fr.csv', 'data/it.csv']
languages = ['de', 'fr', 'it']

for dataset, language in zip(datasets, languages):
    df = pd.read_csv(dataset, sep='\t')
    text_list = df['text'].to_list()
    for text in text_list:
        if len(text.split(' ')) > 3000:
            print('too many')
        else:print('no')