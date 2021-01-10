#!/usr/bin/env python3
import os

from gensim.models import Word2Vec, FastText
from gensim.test.utils import datapath
from gensim import utils


class CorpusGenerator:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        """Yield sentences from the training data.

        Returns
        -------

        """
        corpus_path = datapath(self.path)
        for line in open(corpus_path, encoding='utf-8'):
            # assume there's one document per line,
            # tokens separated by whitespace
            yield utils.simple_preprocess(line)
