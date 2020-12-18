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
import re
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

    original_corpus = []
    f = open(location, "r")
    for line in f:
        original_corpus.append(line)
    start="The Old Testament of the King James Version of the Bible\n"
    end="End of the Project Gutenberg EBook of The King James Bible\n"
    start_index = original_corpus.index(start)
    end_index = original_corpus.index(end)
    corpus = original_corpus[start_index:end_index]
    new_corpus = [line.strip() for line in corpus]
    new_corpus = [re.sub(r'^\d+:\d+', '', line) for line in new_corpus if line]
    processed_lines = []
    for element in new_corpus:
        processed_lines.append(gensim.utils.simple_preprocess(element.strip()))
    return(processed_lines)


embedding_lofl = pre_process(location='bible.txt')


def bible_embeddings(processed_bible):
    #Parameters: processed bible file
    #Returns: writes bible representation to file
    model = Word2Vec(processed_bible, min_count = 2)
    print(model)
    model.wv.save_word2vec_format('bible_embeddings_W2V.bin')


bible_embeddings(embedding_lofl)


