#!/usr/bin/env python3
from sklearn.preprocessing import LabelEncoder

ENCODER = LabelEncoder()


def extract_features(dataframe, use_token=True):
    """
    asdsada
    :param dataframe:
    :param use_token:
    :return:
    """
    dataframe['sentence_length'] = dataframe['sentence'].str.len()
    dataframe["sentence_word_count"] = dataframe[
        "sentence"].str.split().str.len()
    dataframe["sentence_avg_word_length"] = round(
        dataframe["sentence_length"] / dataframe[
            "sentence_word_count"]).astype(int)
    dataframe["sentence_vowel_count"] = dataframe[
        "sentence"].str.lower().str.count(r'[aeiou]')
    dataframe["subcorpus"] = ENCODER.fit_transform(dataframe["subcorpus"])

    if use_token:
        dataframe["token_length"] = dataframe["token"].str.len()
        dataframe["token_vowel_count"] = dataframe[
            "token"].str.lower().str.count(r'[aeiou]')

    return dataframe