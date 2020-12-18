# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
import gensim
from gensim.models import Word2Vec

joined_proceedings = []
for file in os.listdir('./data/txt/en/'):
    current_file = './data/txt/en/' + file
    opened_file = open(current_file, 'r')
    lines = opened_file.readlines()
    lines = [line.strip() for line in lines if line]
    for line in lines:
        if not line.startswith('<'):
            joined_proceedings.append(line)

processed_lines = []
for element in joined_proceedings:
    processed_lines.append(gensim.utils.simple_preprocess(element.strip()))
print(len(processed_lines))

model = Word2Vec(processed_lines, min_count = 2)
print(model)
model.wv.save_word2vec_format('europarl_embeddings_W2V.bin')


