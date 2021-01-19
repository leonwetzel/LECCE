#!/usr/bin/env python3
import pickle
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from textstat import textstat

from lecce.feature.lexical import Meaning, Frequencies
from lecce.feature.representation.word_embeddings import FastTextEmbedder

ENCODER = LabelEncoder()
COUNTS = Frequencies()


def extract_features(dataframe, use_token=True,
                     use_sentence=True,
                     use_word_embeddings=True,
                     use_readability_measures=False):
    """

    Parameters
    ----------
    use_sentence
    use_readability_measures
    dataframe
    use_token
    use_word_embeddings

    Returns
    -------

    """
    dataframe["subcorpus"] = \
        ENCODER.fit_transform(dataframe["subcorpus"])

    if use_sentence:
        # dataframe['sentence_length'] = dataframe['sentence'].str.len()
        dataframe["sentence_word_count"] = dataframe[
            "sentence"].str.split().str.len()
        # dataframe["sentence_avg_word_length"] = round(
        #     dataframe["sentence_length"] / dataframe[
        #         "sentence_word_count"]).astype(int)
        # dataframe["sentence_vowel_count"] = dataframe[
        #     "sentence"].str.lower().str.count(r'[aeiou]')

    if use_readability_measures:
        # dataframe["sentence_gunning_fog"] = \
        #     dataframe.apply(lambda row:
        #                     textstat.gunning_fog(row['sentence']),
        #                     axis=1)
        dataframe["sentence_flesch_reading_ease"] = \
            dataframe.apply(lambda row:
                            textstat.flesch_reading_ease(row['sentence']),
                            axis=1)
        # dataframe["sentence_dale_chall"] = \
        #     dataframe.apply(lambda row:
        #                     textstat.dale_chall_readability_score(
        #                         row['sentence']), axis=1)
        # dataframe["sentence_syllable_count"] = \
        #     dataframe.apply(lambda row:
        #                     textstat.syllable_count(row['sentence']),
        #                     axis=1)

    if use_token:
        # dataframe["token_vowel_count"] = [
        #     textstat.syllable_count(item) for item
        #     in dataframe['sentence'].to_list()
        # ]
        dataframe['token_wordnet_senses'] = \
            dataframe.apply(lambda row:
                            Meaning.count_wordnet_senses(row['token']), axis=1)
        # dataframe['token_proper_noun'] =\
        #     dataframe.apply(lambda row: Meaning.is_proper_name(row["token"]),
        #                     axis=1)
        dataframe['token_pos_tag'] = \
            dataframe.apply(lambda row: Meaning.get_pos_tag(row["token"]),
                            axis=1)
        dataframe["token_pos_tag"] = \
            ENCODER.fit_transform(dataframe["token_pos_tag"])

        """Adds if a word is all uppercase, or only the first letter."""
        dataframe['upper'] = dataframe['token'].apply(
            lambda word: 1 if word.isupper() else 0)
        dataframe['upper_first'] = dataframe['token'].apply(
            lambda word: 1 if word[0].isupper() else 0)

        dataframe["token_vowel_count"] = dataframe[
            "token"].str.lower().str.count(r'[aeiou]')

        # dataframe["token_freq"] = [
        #    freq_overall_corpus(item) for item in dataframe['token']
        # ]
        """Adding the freq in the bible and eu corpus"""
        dataframe["token_freq_bible"] = [
            COUNTS.bible[item] for item in dataframe['token']
        ]
        dataframe["token_freq_eu_corpus"] = [
            freq_eu_corpus(item) for item in dataframe['token']
        ]

    if use_word_embeddings:
        embedder = FastTextEmbedder()

        dataframe["ft_embedding"] = \
            dataframe.apply(
                lambda row: embedder.get_mean_vector(
                    row["sentence"].lower()).tolist(),
                axis=1)

    return dataframe


def get_overall_dict():
    """
    Uses the Counter function to count the number of times a word occurs
    in a list of words from the text files
    @return: a pickle object with a dict. with words and their frequency
    @rtype: pickle dict
    """
    bag_of_words = []

    for file in ['bible.txt', 'europarl.txt', 'pubmed.txt']:
        with open(file, encoding='utf-8') as F:
            for line in F:
                for word in line.split():
                    bag_of_words.append(word.lower().strip("’,.;:''?!"))

    dict = Counter(bag_of_words)
    with open("overall_dict.pkl", "wb", encoding='utf-8') as F:
        pickle.dump(dict, F)


def get_bible_dict():
    """
    Uses the Counter function to count the number of times a word occures in a list of words from the text file
    @return: a pickle object with a dict. with words and their frequency
    @rtype: pickle dict
    """
    list_of_words = []
    with open("bible.txt", encoding="utf8") as file:
        for line in file:
            for word in line.split():
                list_of_words.append(word.lower())

    dict = Counter(list_of_words)
    pickle.dump(dict, open("bible_dict.pkl", "wb"))
    print("Freq dict. created")


def get_europarl_frequencies():
    """
    Uses the Counter function to count the number of times a word occures in a list of words from the text file
    @return: a pickle object with a dict. with words and their frequency
    @rtype: pickle dict
    """
    list_of_words = []
    with open("europarl.txt", encoding="utf8") as file:
        for line in file:
            for word in line.split():
                list_of_words.append(word.lower())

    dict = Counter(list_of_words)
    pickle.dump(dict, open("europarl_unfiltered_dict.pkl", "wb"))
    print("Freq dict. created")


def get_pubmed_frequencies():
    """
    !Takes time; large file!
    Uses the Counter function to count the number of times a word occures in a list of words from the text file
    @return: a pickle object with a dict. with words and their frequency
    @rtype: pickle dict
    """
    list_of_words = []
    with open("pubmed.txt", encoding="utf8") as file:
        for line in file:
            for word in line.split():
                list_of_words.append(word.lower())

    dict = Counter(list_of_words)
    pickle.dump(dict, open("pubmed_unfiltered_dict.pkl", "wb"))
    print("Freq dict. created")


def freq_overall_corpus(word):
    """
    @param word:
    @type word: string
    @return: number of times the word occures in the overall frequencies
    @rtype: int
    """
    frequencies = pickle.load(open("overall_dict.pkl", "rb"))
    return frequencies[word]


def freq_bible_corpus(word):
    """
    @param word:
    @type word: string
    @return: number of times the word occures in the bible frequencies
    @rtype: int
    """
    frequencies = pickle.load(open("bible_dict.pkl", "rb"))
    return frequencies[word]


def freq_eu_corpus(word):
    """
    @param word:
    @type word: string
    @return: number of times the word occures in the eu frequencies
    @rtype: int
    """
    frequencies = pickle.load(open("europarl_unfiltered_dict.pkl", "rb"))
    return frequencies[word]


def freq_pubmed_corpus(word):
    """
    @param word:
    @type word: string
    @return: number of times the word occures in the pubmed frequencies
    @rtype: int
    """
    frequencies = pickle.load(open("pubmed_unfiltered_dict.pkl", "rb"))
    return frequencies[word]
