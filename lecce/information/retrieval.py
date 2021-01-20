#!/usr/bin/env python3
import csv

import pandas as pd

COLUMN_NAMES = ['id', 'subcorpus', 'sentence', 'token', 'complexity']
MULTI_TRIAL_FILE_NAME = "lcp_multi_trial.tsv"
SINGLE_TRIAL_FILE_NAME = "lcp_single_trial.tsv"
MULTI_TRAIN_FILE_NAME = "lcp_multi_train.tsv"
SINGLE_TRAIN_FILE_NAME = "lcp_single_train.tsv"


def load(filename):
    """
    Load information from the .tsv files and store contents into a
    pandas DataFrame.
    :param filename:
    :return:
    """
    try:
        df = pd.read_csv(f"{filename}",  delimiter='\t', header=0,
                         names=COLUMN_NAMES, quoting=csv.QUOTE_ALL,
                         encoding='utf-8')
    except pd.errors.ParserError:
        # sadly occurs in MWE mode
        df = pd.read_csv(f"{filename}",  delimiter='\t', header=0,
                         names=COLUMN_NAMES, quoting=csv.QUOTE_NONE,
                         encoding='utf-8')
    return df
