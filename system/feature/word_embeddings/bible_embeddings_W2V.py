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
import requests
import gensim
from gensim.models import Word2Vec


def pre_process(location):
    """

    Parameters
    ----------
    bible file


    Returns
    processed list of list of text file
    -------

    """

    newlines = []
    f = open(location, "r")
    for line in f:
        newlines.append(line[4:].rstrip('\n'))
    processed_lines = []
    for element in newlines[100:199717]:
        processed_lines.append(gensim.utils.simple_preprocess(element))
    return(processed_lines)


embedding_lofl = pre_process(location='data/bible.txt')


def bible_embeddings(processed_bible):
    #Parameters: processed bible file
    #Returns: writes bible embeddings to file
    model = Word2Vec(processed_bible, min_count = 3)
    print(model)
    model.wv.save_word2vec_format('bible_embeddings_W2V.bin')


bible_embeddings(embedding_lofl)


