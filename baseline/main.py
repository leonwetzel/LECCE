#!/usr/bin/env python3
import csv
import string
import sys
import pickle
import re

import pandas as pd
from collections import Counter


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, \
    explained_variance_score, max_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from textstat import textstat
import numpy as np

from lecce.feature.representation.word_embeddings import \
    Word2VecEmbedder, FastTextEmbedder

COLUMN_NAMES = ['id', 'subcorpus', 'sentence', 'token', 'complexity']
MULTI_TRIAL_FILE_NAME = "lcp_multi_trial.tsv"
SINGLE_TRIAL_FILE_NAME = "lcp_single_trial.tsv"
MULTI_TRAIN_FILE_NAME = "lcp_multi_train.tsv"
SINGLE_TRAIN_FILE_NAME = "lcp_single_train.tsv"

ENCODER = LabelEncoder()

pd.set_option('display.max_rows', None)


def main():
    print("Loading data...")
    try:
        if sys.argv[1] == "-S" or sys.argv[1] == "--single" or sys.argv[1] == "-s":
            token_type = "Single"
            training_data = load(
                f"../data/train/{SINGLE_TRAIN_FILE_NAME}")
            trial_data = load(f"../data/{SINGLE_TRIAL_FILE_NAME}")
        elif sys.argv[1] == "-M" or "--multi" or "-m":
            token_type = "Multi"
        elif sys.argv[1] == "-M" or sys.argv[1] == "--multi" or  sys.argv[1] == "-m":
            training_data = load(
                f"../data/train/{MULTI_TRAIN_FILE_NAME}")
            trial_data = load(f"../data/{MULTI_TRIAL_FILE_NAME}")
    except IndexError:
        exit(
            "Please specify which type of trial information you would"
            " like to use (-S for single trial, -M for multi trial"
            " information)!")

    print("Extracting features...")
    X_train, y_train = extract_features(training_data,
                                        use_sentence=True,
                                        use_word_embeddings=False,
                                        use_token=True,
                                        use_readability_measures=False), \
                       training_data[['complexity']]

    X_trial, y_trial = extract_features(trial_data,
                                        use_sentence=True,
                                        use_word_embeddings=False,
                                        use_token=True,
                                        use_readability_measures=False), \
                       trial_data[['complexity']]
    tokens = X_trial[['id', 'token', "sentence"]]
    X_train.drop(["complexity", "id", "token", "sentence"],
                 axis=1, inplace=True)
    X_trial.drop(["complexity", "id", "token", "sentence"],
                 axis=1, inplace=True)

    print(f"Features used: {list(X_train.columns)}\n")

    pipeline = Pipeline([
        ('clf', LinearRegression(n_jobs=-1))
    ])

    parameters = {
        'clf__fit_intercept': [True, False],
        'clf__normalize': [True, False]
    }

    regressor = GridSearchCV(estimator=pipeline, param_grid=parameters,
                             n_jobs=-1, cv=10, error_score=0.0,
                             return_train_score=False)

    regressor.fit(X_train, y_train)

    y_guess = regressor.predict(X_trial)

    print(f"Mean squared error: {mean_squared_error(y_trial, y_guess)}")
    print(f"R^2 score: {r2_score(y_trial, y_guess)}")
    print(f"Explained variance score:"
          f" {explained_variance_score(y_trial, y_guess)}")
    print(f"Max error: {max_error(y_trial, y_guess)}")
    print(f"Mean absolute error:"
          f" {mean_absolute_error(y_trial, y_guess)}")
    print(f"Best parameter combination: {regressor.best_params_}\n")

    results = y_trial.merge(pd.DataFrame(y_guess), left_index=True,
                            right_index=True)
    results = results.merge(tokens, left_index=True, right_index=True)
    results.columns = ["Actual", "Predicted", "Id", "Token", "Sentence", ]
    print(results[['Actual', "Predicted", "Token"]])

    fig = results.plot(kind='bar', rot=0,
                       title=f"Actual and predicted complexity scores"
                             f" by LECCE ({token_type} token_type)",
                       xlabel="Sample ID", ylabel="Complexity score",
                       grid=False, figsize=(20, 9)
                       ).get_figure()
    fig.savefig("results.png")

    results[["Id", "Predicted"]].to_csv(f"results_{token_type}.csv", index=False, header=False)

def extract_features(dataframe, use_token=True,
                     use_sentence=True,
                     use_word_embeddings=False,
                     use_readability_measures=False):
    """
    @param dataframe:
    @type dataframe: bool
    @param use_token:
    @type use_token:bool
    @param use_sentence:
    @type use_sentence:bool
    @param use_word_embeddings:
    @type use_word_embeddings:bool
    @param use_readability_measures:
    @type use_readability_measures:bool
    @return: a dataframe with all the features from the parameters
    @rtype: pandas dataframe
    """


    dataframe["subcorpus"] = \
        ENCODER.fit_transform(dataframe["subcorpus"])

    if use_sentence:
        dataframe['sentence_length'] = dataframe['sentence'].str.len()
        dataframe["sentence_word_count"] = dataframe[
            "sentence"].str.split().str.len()

    if use_readability_measures:
        dataframe["sentence_gunning_fog"] = \
            dataframe.apply(lambda row:
                            textstat.gunning_fog(row['sentence']),
                            axis=1)
        dataframe["sentence_flesch_reading_ease"] = \
            dataframe.apply(lambda row:
                            textstat.flesch_reading_ease(row['sentence']),
                            axis=1)
        dataframe["sentence_dale_chall"] = \
            dataframe.apply(lambda row:
                            textstat.dale_chall_readability_score(
                                row['sentence']), axis=1)
        dataframe["sentence_syllable_count"] = \
            dataframe.apply(lambda row:
                            textstat.syllable_count(row['sentence']),
                            axis=1)

    if use_token:
        """Add these features to the dataframe"""
        dataframe["token_length"] = [
            len(item) for item in dataframe['token']
        ]

        dataframe["token_length"] = [
            len(item) for item in dataframe['token']
        ]
        """Adds if a word is all uppercase, or only the first letter."""
        dataframe['upper'] = dataframe['token'].apply(lambda word: 1 if word.isupper() else 0)
        dataframe['upper_first'] = dataframe['token'].apply(lambda word: 1 if word[0].isupper() else 0)

        dataframe["token_vowel_count"] = dataframe[
            "token"].str.lower().str.count(r'[aeiou]')

        #dataframe["token_freq"] = [
        #    freq_overall_corpus(item) for item in dataframe['token']
        #]
        """Adding the freq in the bible and eu corpus"""
        dataframe["token_freq_bible"] = [
            freq_bible_corpus(item) for item in dataframe['token']
        ]
        dataframe["token_freq_eu_corpus"] = [
            freq_eu_corpus(item) for item in dataframe['token']
        ]


    if use_word_embeddings:
        embedder = FastTextEmbedder()

        dataframe["ft_embedding"] = \
            dataframe.apply(lambda row:
                            get_mean_vector(embedder.model,
                                            row["sentence"]),
                            axis=1)
    return dataframe


def get_overall_dict():
    """
    Uses the Counter function to count the number of times a word occures in a list of words from the text files
    @return: a pickle object with a dict. with words and their frequency
    @rtype: pickle dict
    """
    list_of_words = []
    with open("bible.txt",encoding="utf8") as file:
        for line in file:
            for word in line.split():
                list_of_words.append(word.lower())
    file.close()

    with open("europarl.txt",encoding="utf8") as file:
        for line in file:
           for word in line.split():
                list_of_words.append(word.lower().strip("’,.;:''?!"))
    file.close()

    with open("pubmed.txt",encoding="utf8") as file:
        for line in file:
           for word in line.split():
                list_of_words.append(word.lower().strip("’,.;:''?!"))
    file.close()

    dict = Counter(list_of_words)
    pickle.dump(dict, open("overall_dict", "wb"))
    print("Freq dict. created")


def get_bible_dict():
    """
    Uses the Counter function to count the number of times a word occures in a list of words from the text file
    @return: a pickle object with a dict. with words and their frequency
    @rtype: pickle dict
    """
    list_of_words = []
    with open("bible.txt",encoding="utf8") as file:
        for line in file:
            for word in line.split():
                list_of_words.append(word.lower())
    file.close()

    dict = Counter(list_of_words)
    pickle.dump(dict, open("bible_dict", "wb"))
    print("Freq dict. created")

def get_eu_dict():
    """
    Uses the Counter function to count the number of times a word occures in a list of words from the text file
    @return: a pickle object with a dict. with words and their frequency
    @rtype: pickle dict
    """
    list_of_words = []
    with open("europarl.txt",encoding="utf8") as file:
        for line in file:
            for word in line.split():
                list_of_words.append(word.lower())
    file.close()

    dict = Counter(list_of_words)
    pickle.dump(dict, open("europarl__unfiltered_dict", "wb"))
    print("Freq dict. created")

def get_pubmed_dict():
    """
    !Takes time; large file!
    Uses the Counter function to count the number of times a word occures in a list of words from the text file
    @return: a pickle object with a dict. with words and their frequency
    @rtype: pickle dict
    """
    list_of_words = []
    with open("pubmed.txt",encoding="utf8") as file:
        for line in file:
            for word in line.split():
                list_of_words.append(word.lower())
    file.close()

    dict = Counter(list_of_words)
    pickle.dump(dict, open("pubmed_unfiltered_dict", "wb"))
    print("Freq dict. created")

def freq_overall_corpus(word):
    """
    @param word:
    @type word: string
    @return: number of times the word occures in the overall dict
    @rtype: int
    """
    dict = pickle.load(open("overall_dict", "rb"))
    return(dict[word])

def freq_bible_corpus(word):
    """
    @param word:
    @type word: string
    @return: number of times the word occures in the bible dict
    @rtype: int
    """
    dict = pickle.load(open("bible_dict", "rb"))
    return(dict[word])

def freq_eu_corpus(word):
    """
    @param word:
    @type word: string
    @return: number of times the word occures in the eu dict
    @rtype: int
    """
    dict = pickle.load(open("europarl__unfiltered_dict", "rb"))
    return(dict[word])

def freq_pubmed_corpus(word):
    """
    @param word:
    @type word: string
    @return: number of times the word occures in the pubmed dict
    @rtype: int
    """
    dict = pickle.load(open("pubmed_unfiltered_dict", "rb"))
    return(dict[word])


if __name__ == '__main__':
    main()
