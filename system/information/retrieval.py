#!/usr/bin/env python3
import gzip
import os
import csv
import shutil
import urllib.request as request

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
    df = pd.read_csv(f"{filename}",  delimiter='\t', header=0,
                     names=COLUMN_NAMES, quoting=csv.QUOTE_NONE,
                     encoding='utf-8')
    return df
