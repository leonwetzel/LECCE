#!/usr/bin/env python3
from sklearn.preprocessing import LabelEncoder
from textstat import textstat

from lecce.feature.lexical import Meaning
from lecce.feature.representation.word_embeddings import FastTextEmbedder

ENCODER = LabelEncoder()


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
        #dataframe['sentence_length'] = dataframe['sentence'].str.len()
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
        dataframe["token_length"] = [
            len(item) for item in dataframe['token'].to_list()
        ]
        # dataframe["token_vowel_count"] = [
        #     textstat.syllable_count(item) for item
        #     in dataframe['sentence'].to_list()
        # ]
        dataframe['token_wordnet_senses'] =\
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

    if use_word_embeddings:
        embedder = FastTextEmbedder()

        dataframe["ft_embedding"] = \
            dataframe.apply(
                lambda row: embedder.get_mean_vector(
                    row["sentence"].lower()).tolist(),
                axis=1)

    return dataframe
