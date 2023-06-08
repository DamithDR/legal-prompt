import random

import numpy as np
import pandas as pd
from datasets import load_dataset

import matplotlib.pyplot as plt

languages = ['de', 'fr', 'it']
labels = ['German', 'French', 'Italian']
colors = ['lightskyblue', 'wheat', 'yellowgreen']

filtered_data = []
means = []
stds = []

length_lists = []
word_limit = 3000
for language in languages:
    sub_list = []
    data_list =[]
    dataset = load_dataset('swiss_judgment_prediction', language, split='test')
    for data in dataset:
        sub_list.append(len(data['text'].split(' ')))
        if len(data['text'].split(' ')) <= word_limit:
            data_list.append(data)
    length_lists.append(sub_list)
    means.append(np.mean(sub_list))
    stds.append(np.std(sub_list))

    if len(data_list) < 1000:
        sampled = data_list
    else:
        sampled = random.sample(data_list, 1000)  # taking random samples
    df = pd.DataFrame(sampled)

    path = f'data/{language}.csv'
    df.to_csv(path, sep='\t', index=False)

print(means)
print(stds)

# [356.84143958868896, 780.7689149560117, 458.87931034482756]
# [245.14144775766601, 521.4686560184352, 442.616136790456]

# for i in range(len(labels)):
#     counts, bins = np.histogram(length_lists[i], bins=500)
#     plt.stairs(counts, bins)
#     plt.hist(bins[:-1], bins, weights=counts, label=f'{labels[i]} Data', color=colors[i])
#
#     plt.xlabel('Documents')
#     plt.ylabel('Number of words')
#     plt.title(f'{labels[i]} Dataset')
#     plt.show()
