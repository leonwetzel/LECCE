#!/usr/bin/env python3
from abc import ABC, abstractmethod

from gensim.models import Word2Vec, FastText

from lecce.feature.representation.google import download_google_news_vectors


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
    def transform(self):
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
            self.model = Word2Vec(sentences=corpus, size=300,
                                  alpha=0.025, window=5, min_count=5,
                                  max_vocab_size=None, sample=0.001, seed=1,
                                  workers=3, min_alpha=0.0001, sg=0, hs=0,
                                  negative=5, ns_exponent=0.75, cbow_mean=1,
                                  iter=5, null_word=0, trim_rule=None,
                                  sorted_vocab=1, batch_words=10000,
                                  compute_loss=False, callbacks=(),
                                  max_final_vocab=None)
            self.model.save(f"{directory}/{model_name}")
        else:
            download_google_news_vectors(
                destination_name=f"{directory}/GoogleNews-vectors.bin"
            )

    def transform(self):
        pass


class FastTextEmbedder(Embedder):
    def __init__(self, model_name="my_first_ft_model.bin",
                 corpus=None, directory="embeddings"):
        super(FastTextEmbedder, self).__init__(model_name, corpus, directory)

    def transform(self):
        pass
