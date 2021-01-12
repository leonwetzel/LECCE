#!/usr/bin/env python3
import nltk
from nltk.corpus import wordnet as wn


class Meaning:
    """

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
