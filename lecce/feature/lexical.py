#!/usr/bin/env python3
import math
import pickle

import nltk
from nltk.corpus import wordnet as wn


class Meaning:
    """
    Contains functionality related to NLTK.
    """
    ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"

    def __init__(self):
        pass

    @staticmethod
    def count_wordnet_senses(word, pos_tag=None):
        """Count the amount of WordNet senses for a given word

        Parameters
        ----------
        word : str
        pos_tag : str, optional

        Returns
        -------

        """
        if not pos_tag:
            senses = wn.synsets(word)
        else:
            senses = wn.synsets(word, pos_tag=pos_tag)
        return len(senses)

    @staticmethod
    def is_proper_name(token):
        """A simple check if a given token could be a proper name,
        by naively checking if the first character is a capital letter.

        Parameters
        ----------
        token

        Returns
        -------

        """
        return token[0].isupper()

    @staticmethod
    def get_pos_tag(token):
        """

        Parameters
        ----------
        token

        Returns
        -------

        """
        return nltk.pos_tag([token])[0][1]


class Frequencies:
    """
    Class containing word frequencies
    """
    def __init__(self):
        """

        Parameters
        ----------
        filenames
        """
        with open("bible_dict.pkl", mode='rb') as F:
            self.bible = pickle.load(F)

        with open("europarl_unfiltered_dict.pkl", mode='rb') as F:
            self.europarl = pickle.load(F)

        with open("pubmed_unfiltered_dict.pkl", mode='rb') as F:
            self.pubmed = pickle.load(F)

        with open("overall_dict.pkl", mode='rb') as F:
            self.overall = pickle.load(F)

    def get_absolute_count(self, word, corpus):
        """

        Parameters
        ----------
        word : str
            Word that should be looked up.
        corpus : str
            Name of the corpus in which the word appears.

        Returns
        -------

        """
        if corpus == "bible":
            return self._count(self.bible, word)
        elif corpus == 'pubmed':
            return self._count(self.pubmed, word)
        elif corpus == 'europarl':
            return self._count(self.europarl, word)
        return 0

    def get_logarithmic_count(self, word, corpus):
        """

        Parameters
        ----------
        word : str
            Word that should be looked up.
        corpus : str
            Name of the corpus in which the word appears.

        Returns
        -------

        """
        word = word.lower()
        if corpus == "bible":
            count = self._count(self.bible, word)
            if count == 0:
                return 0
            else:
                return math.log(self.bible[word])

        elif corpus == 'pubmed':
            count = self._count(self.pubmed, word)
            if count == 0:
                return 0
            else:
                return math.log(self.pubmed[word])

        elif corpus == 'europarl':
            count = self._count(self.europarl, word)
            if count == 0:
                return 0
            else:
                return math.log(self.europarl[word])
        return 0

    @staticmethod
    def _count(dictionary, word):
        """Obtain the frequency of a word in a given dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary that contains words as keys and frequencies as values.
        word : str
            Word of which the frequency should be looked up.

        Returns
        -------
        count : int
            Frequency of the given word.
        """
        try:
            return dictionary[word]
        except KeyError:
            return 0
