#!/usr/bin/env python3
import csv
import sys

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, \
    explained_variance_score, max_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from textstat import textstat

from lecce.feature.representation.word_embeddings import \
    Word2VecEmbedder

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
        if sys.argv[1] == "-S" or "--single" or "-s":
            training_data = load(
                f"../data/train/{SINGLE_TRAIN_FILE_NAME}")
            trial_data = load(f"../data/{SINGLE_TRIAL_FILE_NAME}")
        elif sys.argv[1] == "-M" or "--multi" or "-m":
            training_data = load(
                f"../data/train/{MULTI_TRAIN_FILE_NAME}")
            trial_data = load(f"../data/{MULTI_TRIAL_FILE_NAME}")
    except IndexError:
        exit(
            "Please specify which type of trial information you would"
            " like to use (-S for single trial, -M for multi trial"
            " information)!")

    X_train, y_train = extract_features(training_data,
                                        use_word_embeddings=True), \
                       training_data[['complexity']]
    X_trial, y_trial = extract_features(trial_data,
                                        use_word_embeddings=True), \
                       trial_data[['complexity']]
    X_train = X_train[['sentence_length', 'sentence_word_count',
                       'sentence_avg_word_length',
                       'sentence_vowel_count',
                       'subcorpus',
                       'sentence_gunning_fog',
                       'sentence_flesch_reading_ease',
                       'sentence_dale_chall',
                       'sentence_syllable_count',
                       'token_length', 'token_vowel_count']]
    X_trial = X_trial[['sentence_length', 'sentence_word_count',
                       'sentence_avg_word_length',
                       'sentence_vowel_count',
                       'subcorpus',
                       'sentence_gunning_fog',
                       'sentence_flesch_reading_ease',
                       'sentence_dale_chall',
                       'sentence_syllable_count',
                       'token_length', 'token_vowel_count']]

    # print(X_train)

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

    # print("Y_trial\tY_guess")
    results = y_trial.merge(pd.DataFrame(y_guess), left_index=True,
                            right_index=True)
    results.columns = ["Actual", "Predicted"]
    print(results)

    fig = results.plot(kind='bar', rot=0,
                       title="Actual and predicted complexity scores"
                             " by LECCE (single token)",
                       xlabel="Sample ID", ylabel="Complexity score",
                       grid=False, figsize=(20,9)
                       ).get_figure()
    fig.savefig("results.png")


def load(filename):
    """
    Load information from the .tsv files and store contents into a
    pandas DataFrame.
    :param filename:
    :return:
    """
    df = pd.read_csv(f"{filename}", delimiter='\t', header=0,
                     names=COLUMN_NAMES, quoting=csv.QUOTE_NONE,
                     encoding='utf-8')
    return df


def extract_features(dataframe, use_token=True,
                     use_word_embeddings=True):
    """

    Parameters
    ----------
    dataframe
    use_token
    use_word_embeddings

    Returns
    -------

    """
    dataframe['sentence_length'] = dataframe['sentence'].str.len()
    dataframe["sentence_word_count"] = dataframe[
        "sentence"].str.split().str.len()
    dataframe["sentence_avg_word_length"] = round(
        dataframe["sentence_length"] / dataframe[
            "sentence_word_count"]).astype(int)
    dataframe["sentence_vowel_count"] = dataframe[
        "sentence"].str.lower().str.count(r'[aeiou]')
    dataframe["subcorpus"] = \
        ENCODER.fit_transform(dataframe["subcorpus"])

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
        dataframe["token_length"] = [
            len(item) for item in dataframe['sentence'].to_list()
        ]
        dataframe["token_vowel_count"] = [
            textstat.syllable_count(item) for item
            in dataframe['sentence'].to_list()
        ]

    if use_word_embeddings:
        embedder = Word2VecEmbedder(directory="../embeddings",
                                    model_name="GoogleNews-vectors.bin")
        dataframe["w2v_embedding"] = \
            dataframe.apply(lambda row:
                            embedder.model.wv[row['sentence']],
                            axis=1)

    return dataframe


if __name__ == '__main__':
    main()
