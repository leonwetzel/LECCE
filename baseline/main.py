#!/usr/bin/env python3
import sys

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, max_error, \
    mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from lecce.feature.extraction import extract_features
from lecce.information.retrieval import load

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
        if sys.argv[1] == "-S" or sys.argv[1] == "--single" \
                or sys.argv[1] == "-s":
            token_type = "Single"
            training_data = load(
                f"../data/train/{SINGLE_TRAIN_FILE_NAME}")
            trial_data = load(f"../data/{SINGLE_TRIAL_FILE_NAME}")
        elif sys.argv[1] == "-M" or "--multi" or "-m":
            token_type = "Multi"
        elif sys.argv[1] == "-M" or sys.argv[1] == "--multi" \
                or sys.argv[1] == "-m":
            training_data = load(
                f"../data/train/{MULTI_TRAIN_FILE_NAME}")
            trial_data = load(f"../data/{MULTI_TRIAL_FILE_NAME}")
    except IndexError:
        exit(
            "Please specify which type of trial information you would"
            " like to use (-S for single trial, -M for multi trial"
            " information)!")

    training_data = training_data.dropna()
    trial_data = trial_data.dropna()

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

    regressor = LinearRegression(n_jobs=-1)



    regressor.fit(X_train, y_train)

    y_guess = regressor.predict(X_trial)

    print(f"Mean squared error: {mean_squared_error(y_trial, y_guess)}")
    print(f"R^2 score: {r2_score(y_trial, y_guess)}")
    print(f"Explained variance score:"
          f" {explained_variance_score(y_trial, y_guess)}")
    print(f"Max error: {max_error(y_trial, y_guess)}")
    print(f"Mean absolute error:"
          f" {mean_absolute_error(y_trial, y_guess)}")

    #print(f"Best parameter combination: {regressor.best_params_}\n")

    results = y_trial.merge(pd.DataFrame(y_guess), left_index=True,
                            right_index=True)
    results = results.merge(tokens, left_index=True, right_index=True)
    results.columns = ["Actual", "Predicted", "Id", "Token", "Sentence", ]
    print(results[['Actual', "Predicted", "Token"]])
    "Print the correlation between predicted and actual"
    print(results.corr(method='pearson'))
    print(results.corr(method='spearman'))
    print()

    fig = results.plot(kind='bar', rot=0,
                       title=f"Actual and predicted complexity scores"
                             f" by LECCE ({token_type} token_type)",
                       xlabel="Sample ID", ylabel="Complexity score",
                       grid=False, figsize=(20, 9)
                       ).get_figure()
    fig.savefig("results.png")

    results[["Id", "Predicted"]].to_csv(f"results_{token_type}.csv",
                                        index=False, header=False)


if __name__ == '__main__':
    main()
