#!/usr/bin/env python3
import os
from abc import ABC, abstractmethod

from gensim.models import Word2Vec, FastText, KeyedVectors

from lecce.feature.representation.google import download_google_news_vectors

GOOGLE_NEWS_VECTORS = "GoogleNews-vectors.bin"
EMBEDDINGS_DIR = "embeddings"


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

    @abstractmethod
    def transform(self, token):
        pass


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
        model_name : str
            File name of the model.
        corpus : list
            Collection of sentences.
        directory : str
            Name of the directory containing the word embeddings.

        """
        super(Word2VecEmbedder, self).__init__(model_name, corpus, directory)

        if model_name and not corpus:
            self.model = Word2Vec.load(f"{directory}/{model_name}")
        elif corpus:
            model_name = input("Please enter the filename for your new "
                               "word2vec model (including extension): ")
            self.model = Word2Vec(sentences=corpus, corpus_file=None,
                                  vector_size=300, alpha=0.025, window=5,
                                  min_count=1, max_vocab_size=None,
                                  sample=0.001, seed=1, workers=3,
                                  min_alpha=0.0001, sg=0, hs=0, negative=5,
                                  ns_exponent=0.75, cbow_mean=1, epochs=5,
                                  null_word=0, trim_rule=None, sorted_vocab=1,
                                  batch_words=10000, compute_loss=False,
                                  callbacks=(), comment=None,
                                  max_final_vocab=None)
            self.model.save(f"{directory}/{model_name}")
        else:
            if not os.path.isfile(f"{EMBEDDINGS_DIR}/{GOOGLE_NEWS_VECTORS}"):
                download_google_news_vectors(
                    destination_name=f"{directory}/{GOOGLE_NEWS_VECTORS}"
                )
            self.model = Word2Vec.load(f"{directory}/{GOOGLE_NEWS_VECTORS}")

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
    This class contains functions for the use of word2vec embeddings
    with LECCE functionality.
    """
    def __init__(self, model_name="my_first_ft_model.bin",
                 corpus=None, directory="embeddings"):
        """

        Parameters
        ----------
        model_name : str
            File name of the model.
        corpus : list
            Collection of sentences.
        directory : str
            Name of the directory containing the word embeddings.

        """
        super(FastTextEmbedder, self).__init__(model_name, corpus, directory)

        if model_name and not corpus:
            self.model = FastText.load(f"{directory}/{model_name}")
        elif corpus:
            model_name = input("Please enter the filename for your new "
                               "fasttext model (including extension): ")
            self.model = FastText(sentences=corpus, corpus_file=None, sg=0, hs=0,
                                  vector_size=300, alpha=0.025, window=5,
                                  min_count=5, max_vocab_size=None,
                                  word_ngrams=1, sample=0.001, seed=1,
                                  workers=3, min_alpha=0.0001, negative=5,
                                  ns_exponent=0.75, cbow_mean=1, epochs=5,
                                  null_word=0, min_n=3, max_n=6, sorted_vocab=1,
                                  bucket=2000000, trim_rule=None,
                                  batch_words=10000, callbacks=(),
                                  max_final_vocab=None)
            self.model.save(f"{directory}/{model_name}")
        else:
            # FIXME this part of code does not make sense,
            # as the Google News vectors are word2vec based...
            if not os.path.isfile(f"{EMBEDDINGS_DIR}/{GOOGLE_NEWS_VECTORS}"):
                download_google_news_vectors(
                    destination_name=f"{directory}/{GOOGLE_NEWS_VECTORS}"
                )
            self.model = Word2Vec.load(f"{directory}/{GOOGLE_NEWS_VECTORS}")

    def transform(self, token):
        pass
