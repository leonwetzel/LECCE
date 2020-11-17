#!/usr/bin/env python3
import csv
import sys

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, \
    explained_variance_score, max_error, mean_absolute_error

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

COLUMN_NAMES = ['id', 'subcorpus', 'sentence', 'token', 'complexity']
MULTI_TRIAL_FILE_NAME = "lcp_multi_trial.tsv"
SINGLE_TRIAL_FILE_NAME = "lcp_single_trial.tsv"
MULTI_TRAIN_FILE_NAME = "lcp_multi_train.tsv"
SINGLE_TRAIN_FILE_NAME = "lcp_single_train.tsv"

ENCODER = LabelEncoder()

pd.set_option('display.max_rows', None)


def main():
    """

    :return:
    """
    try:
        if sys.argv[1] == "-S":
            td = load(f"../data/{SINGLE_TRAIN_FILE_NAME}")
            td_trial = load(f"../data/{SINGLE_TRIAL_FILE_NAME}")
        elif sys.argv[1] == "-M":
            td = load(f"../data/{MULTI_TRAIN_FILE_NAME}")
            td_trial = load(f"../data/{MULTI_TRIAL_FILE_NAME}")
    except IndexError:
        exit("Please specify which type of trial information you would like"
             " to use (-S for single trial, -M for multi trial information)!")

    X_train, y_train = extract_features(td), td[['complexity']]
    X_trial, y_trial = extract_features(td_trial),\
                       td_trial[['complexity']]
    X_train = X_train[['sentence_length', 'sentence_word_count',
                 'sentence_avg_word_length', 'sentence_vowel_count',
                 'token_length', 'token_vowel_count', 'subcorpus']]
    X_trial = X_trial[['sentence_length', 'sentence_word_count',
                 'sentence_avg_word_length', 'sentence_vowel_count',
                 'token_length', 'token_vowel_count', 'subcorpus']]

    print(X_train)

    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('clf', LinearRegression(n_jobs=-1))
    ])

    parameters = {
        'clf__fit_intercept': [True, False],
        'clf__normalize': [True, False]
    }

    classifier = GridSearchCV(estimator=pipeline, param_grid=parameters,
                              n_jobs=-1, cv=10, error_score=0.0,
                              return_train_score=False)

    classifier.fit(X_train, y_train)

    y_guess = classifier.predict(X_trial)

    print(f"Mean squared error: {mean_squared_error(y_trial, y_guess)}")
    print(f"R^2 score: {r2_score(y_trial, y_guess)}")
    print(f"Explained variance score:"
          f" {explained_variance_score(y_trial, y_guess)}")
    print(f"Max error: {max_error(y_trial, y_guess)}")
    print(f"Mean absolute error:"
          f" {mean_absolute_error(y_trial, y_guess)}")
    print(f"Best parameter combination: {classifier.best_params_}\n")

    print("Y_trial\tY_guess")
    print(y_trial.merge(pd.DataFrame(y_guess), left_index=True,
                        right_index=True))


def load(filename):
    """
    Load information from the .tsv files and store contents into a
    pandas DataFrame.
    :param filename:
    :return:
    """
    df = pd.read_csv(f"{filename}",  delimiter='\t', header=0,
                     names=COLUMN_NAMES, quoting=csv.QUOTE_NONE,
                     encoding='utf-8')
    return df


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


if __name__ == '__main__':
    main()
