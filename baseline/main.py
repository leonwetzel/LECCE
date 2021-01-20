#!/usr/bin/env python3
import os
import sys

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, max_error, \
    mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from lecce.feature.extraction import extract_features
from lecce.information.retrieval import load

COLUMN_NAMES = ['id', 'subcorpus', 'sentence', 'token', 'complexity']

MULTI_TRIAL_FILE_NAME = "lcp_multi_trial.tsv"
SINGLE_TRIAL_FILE_NAME = "lcp_single_trial.tsv"

MULTI_TRAIN_FILE_NAME = "lcp_multi_train.tsv"
SINGLE_TRAIN_FILE_NAME = "lcp_single_train.tsv"

MULTI_TEST_FILE_NAME = "lcp_multi_test.tsv"
SINGLE_TEST_FILE_NAME = "lcp_single_test.tsv"

ENCODER = LabelEncoder()

pd.set_option('display.max_rows', None)


def main():
    print("Loading data from .tsv's...")
    try:
        if sys.argv[1] == "--single" or sys.argv[1].lower() == "-s":
            token_type = "Single"
            training_data = load(f"data/train/{SINGLE_TRAIN_FILE_NAME}")
            if sys.argv[2].lower() == "-t" or sys.argv[2] == "--trial":
                print("Using trial data...")
                target_type = "Trial"
                target_data = load(f"data/{SINGLE_TRIAL_FILE_NAME}")
            else:
                print("Using test data...")
                target_type = "Test"
                target_data = load(f"data/{SINGLE_TEST_FILE_NAME}")

        elif sys.argv[1] == "--multi" or sys.argv[1].lower() == "-m":
            token_type = "Multi"
            training_data = load(f"data/train/{MULTI_TRAIN_FILE_NAME}")

            if sys.argv[2].lower() == "-t" or sys.argv[2] == "--trial":
                print("Using trial data...")
                target_type = "Trial"
                target_data = load(f"data/{MULTI_TRIAL_FILE_NAME}")
            else:
                print("Using test data...")
                target_type = "Test"
                target_data = load(f"data/{MULTI_TEST_FILE_NAME}")

    except IndexError:
        exit(
            "Please specify which type of trial information you would"
            " like to use (-S for single trial, -M for multi trial"
            " information)!")

    training_data = training_data.dropna()
    target_data = target_data.dropna()

    print("Extracting features...")
    X_train, y_train = extract_features(training_data,
                                        use_sentence=False,
                                        use_word_embeddings=False,
                                        use_token=True,
                                        use_readability_measures=False), \
                       training_data[['complexity']]

    X_target, y_target = extract_features(target_data,
                                          use_sentence=False,
                                          use_word_embeddings=False,
                                          use_token=True,
                                          use_readability_measures=False), \
                         target_data[['complexity']]
    tokens = X_target[['id', 'token', "sentence"]]
    X_train.drop(["complexity", "id", "token", "sentence"],
                 axis=1, inplace=True)
    X_target.drop(["complexity", "id", "token", "sentence"],
                  axis=1, inplace=True)

    regressor = LinearRegression(n_jobs=-1,
                                 normalize=False,
                                 fit_intercept=True)

    regressor.fit(X_train, y_train)
    y_guess = regressor.predict(X_target)

    results = y_target.merge(pd.DataFrame(y_guess), left_index=True,
                             right_index=True)
    results = results.merge(tokens, left_index=True, right_index=True)
    results.columns = ["Actual", "Predicted", "Id", "Token", "Sentence", ]
    print(results[['Actual', "Predicted", "Token"]])

    print()
    print(f"Features used: {list(X_train.columns)}\n")

    print(f"Mean squared error (MSE):\t{mean_squared_error(y_target, y_guess)}")
    print(f"R^2 score (R2):\t{r2_score(y_target, y_guess)}")
    print(f"Mean absolute error (MAE):\t"
          f" {mean_absolute_error(y_target, y_guess)}")
    print(f"Explained variance score:\t"
          f" {explained_variance_score(y_target, y_guess)}")
    print(f"Max error:\t{max_error(y_target, y_guess)}")
    print()

    print("Pearson correlation")
    print(results.corr(method='pearson'), '\n')

    print("Spearman correlation (Rho)")
    print(results.corr(method='spearman'))
    print()

    if not os.path.isdir("output"):
        os.mkdir("output")

    fig = results.plot(kind='bar', rot=0,
                       title=f"Actual and predicted complexity scores"
                             f" by LECCE (token_type={token_type.lower()},"
                             f" target_type={target_type.lower()})",
                       xlabel="Sample ID", ylabel="Complexity score",
                       grid=False, figsize=(20, 9)
                       ).get_figure()
    fig.savefig("output/results.png")

    results[["Id", "Predicted"]].to_csv(
        f"output/results_{token_type.lower()}_{target_type.lower()}.csv",
        index=False, header=False)


if __name__ == '__main__':
    main()
