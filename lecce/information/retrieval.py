#!/usr/bin/env python3
import csv

import pandas as pd

column_names = ['id', 'subcorpus', 'sentence', 'token', 'complexity']

MULTI_TRIAL_FILE_NAME = "lcp_multi_trial.tsv"
SINGLE_TRIAL_FILE_NAME = "lcp_single_trial.tsv"

MULTI_TRAIN_FILE_NAME = "lcp_multi_train.tsv"
SINGLE_TRAIN_FILE_NAME = "lcp_single_train.tsv"

MULTI_TEST_FILE_NAME = "lcp_multi_test.tsv"
SINGLE_TEST_FILE_NAME = "lcp_single_test.tsv"


def load(filename):
    """
    Load information from the .tsv files and store contents into a
    pandas DataFrame.
    :param filename:
    :return:
    """
    if filename.endswith(MULTI_TEST_FILE_NAME) or\
            filename.endswith(SINGLE_TEST_FILE_NAME):
        column_names.remove('complexity')

    try:
        df = pd.read_csv(f"{filename}", delimiter='\t', header=0,
                         names=column_names, quoting=csv.QUOTE_NONE,
                         encoding='utf-8')
    except pd.errors.ParserError:
        # sadly occurs in MWE mode
        df = pd.read_csv(f"{filename}", delimiter='\t', header=0,
                         names=column_names, quoting=csv.QUOTE_NONE,
                         encoding='utf-8')
    return df
