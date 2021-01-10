#!/usr/bin/env python3
import io
import os
from abc import ABC, abstractmethod

from gensim.models import Word2Vec, FastText
import gensim.downloader
import numpy as np


class Embedder(ABC):
    """
    This abstract class indicates the functionality that an Embedder
    subclass should have.
    """

    def __init__(self, model_name, corpus, directory):
        """Base constructor for Embedder classes and objects.

        Parameters
        ----------
        model_name : str
            File name of the model.
        corpus : list
            Collection of sentences.
        directory : str
            Name of the directory containing the word embeddings.

        """
        self.model_name = model_name
        self.corpus = corpus
        self.directory = directory
        self.model = None

    def get_mean_vector(self, words):
        """Calculate the mean word embedding for a given
        collection of words.

        Parameters
        ----------
        model :
            Gensim model that can convert tokens to embeddings.
        words : iterable
            Collection of words that need to be transformed
            to word embeddings. Usually a sentence or list of words.

        Returns
        -------

        """
        # remove out-of-vocabulary words
        words = [word for word in words
                 if word in self.model.key_to_index]
        # generate embeddings per word
        embeddings = [self.model[word] for word in words]
        if len(words) >= 1:
            return np.mean(embeddings, axis=0)
        else:
            return []

    def is_in_vocabulary(self, token):
        """Checks if a given token is present in the
        vocabulary of the embeddings model.

        Parameters
        ----------
        token

        Returns
        -------

        """
        return token in self.model.key_to_index


class Word2VecEmbedder(Embedder):
    """
    This class contains functions for the use of word2vec embeddings
    with LECCE functionality.
    """

    def __init__(self, model_name=None,
                 corpus=None, directory="embeddings"):
        """

        Parameters
        ----------
        model_name : str, optional
            File name of the model.
        corpus : iterable, optional
            Collection of sentences.
        directory : str
            Name of the directory containing the word embeddings.

        """
        super(Word2VecEmbedder, self).__init__(model_name, corpus,
                                               directory)

        if model_name and not corpus:
            self.model = Word2Vec.load(f"{directory}/{model_name}")
        elif corpus:
            model_name = input("Please enter the filename for your new "
                               "word2vec model (including extension): ")
            self.model = gensim.models.Word2Vec.load(
                f"{directory}/GoogleNews-vectors.bin")
            # self.model.build_vocab(corpus, update=True)
            self.model.train(corpus, total_examples=len(corpus),
                             epochs=self.model.epochs)
            # self.model = Word2Vec(sentences=corpus, corpus_file=None,
            #                       vector_size=200, alpha=0.025,
            #                       window=5,
            #                       min_count=1, max_vocab_size=None,
            #                       sample=0.001, seed=1, workers=3,
            #                       min_alpha=0.0001, sg=0, hs=0,
            #                       negative=5,
            #                       ns_exponent=0.75, cbow_mean=1,
            #                       epochs=5,
            #                       null_word=0, trim_rule=None,
            #                       sorted_vocab=1,
            #                       batch_words=10000, compute_loss=False,
            #                       callbacks=(), comment=None,
            #                       max_final_vocab=None)
            self.model.save(f"{directory}/{model_name}")
        else:
            # use default GoogleNews-vectors
            self.model = gensim.downloader.load(
                "word2vec-google-news-300")

    def transform(self, token):
        """Transforms a given token into a word embedding.

        Parameters
        ----------
        token : str
            Token that needs to be transformed into a word embedding.

        Returns
        -------
        vector
            Word embedding of a given token in vector form.
        """
        return self.model.wv[token]


class FastTextEmbedder(Embedder):
    """
    This class contains functions for the use of fastText embeddings
    with LECCE functionality.
    """

    def __init__(self, model_name=None,
                 corpus=None, directory="embeddings"):
        """

        Parameters
        ----------
        model_name : str, optional
            File name of the model.
        corpus : iterable, optional
            Collection of sentences.
        directory : str
            Name of the directory containing the word embeddings.

        """
        super(FastTextEmbedder, self).__init__(model_name, corpus,
                                               directory)

        if model_name and not corpus:
            self.model = FastText.load(f"{directory}/{model_name}")
        elif corpus:
            model_name = input("Please enter the filename for your new "
                               "fastText model (including extension): ")
            self.model.build_vocab(corpus, total_words=len(corpus),
                                   epochs=self.model.epochs)
            self.model = FastText(sentences=corpus, corpus_file=None,
                                  sg=0, hs=0,
                                  vector_size=200, alpha=0.025,
                                  window=5,
                                  min_count=5, max_vocab_size=None,
                                  word_ngrams=1, sample=0.001, seed=1,
                                  workers=3, min_alpha=0.0001,
                                  negative=5,
                                  ns_exponent=0.75, cbow_mean=1,
                                  epochs=5,
                                  null_word=0, min_n=3, max_n=6,
                                  sorted_vocab=1,
                                  bucket=2000000, trim_rule=None,
                                  batch_words=10000, callbacks=(),
                                  max_final_vocab=None)
            self.model.save(f"{directory}/{model_name}")
        else:
            # use default fastText model
            self.model = gensim.downloader.load(
                'fasttext-wiki-news-subwords-300')

    def transform(self, token):
        """

        Parameters
        ----------
        token

        Returns
        -------

        """
        return self.model.wv[token]

    @staticmethod
    def load_vectors(fname):
        """Load a fastText model.

        Stolen from the fastText website
        https://fasttext.cc/docs/en/english-vectors.html

        Parameters
        ----------
        fname

        Returns
        -------

        """
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n',
                      errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return data
