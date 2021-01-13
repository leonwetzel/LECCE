#!/usr/bin/env python3
import os

from gensim import utils
from gensim.test.utils import datapath


class CorpusGenerator:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, files=None):
        self.files = files

    def __iter__(self):
        """Yield sentences from the training data.

        Returns
        -------

        """
        for filename in self.files:
            # = datapath(filename)

            for line in open(filename, encoding='utf-8'):
                # assume there's one document per line,
                # tokens separated by whitespace
                yield utils.simple_preprocess(line)
