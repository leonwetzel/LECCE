#!/usr/bin/env python3
import os
import requests


TRAIN_DATA_URL = "https://github.com/MMU-TDMLab/CompLex/raw/master/train.7z"
MULTI_TRIAL_DATA_URL = "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master/trial/lcp_multi_trial.tsv"
SINGLE_TRIAL_DATA_URL = "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master/trial/lcp_single_trial.tsv"


def download_trial_data(urls=None, directory="data/trial"):
    """Downloads the LCP trial data from the GitHub repository and
    saves the trial data of the LCP challenge into a given directory.

    Parameters
    ----------
    directory : str
        Name of directory in which the data should be stored.
    urls : tuple
        Collection of urls containing the LCP trial data.

    Returns
    -------

    """
    if not urls:
        urls = [MULTI_TRIAL_DATA_URL, SINGLE_TRIAL_DATA_URL]

    if not os.path.isdir(directory):
        os.mkdir(directory)

    if not is_directory_empty(directory):
        print(f"Warning: directory {directory} is not empty!")
        enter = input(f"Press ENTER if you want to overwrite the"
                      f" existing data in {directory} [ENTER]")

        if enter != "":
            print("Cancelling download...")
            return None

    for url in urls:
        r = requests.get(url)
        filename = url.split('/')[-1]

        with open(rf"{directory}/{filename}", "w",
                  encoding='utf-8') as F:
            F.write(r.text)

    return None


def is_directory_empty(directory):
    """Checks if a given directory is empty or not.

    Parameters
    ----------
    directory : str
        Name of directory.

    Returns
    -------
    is_directory_empty : bool
        Indication if a directory is either empty or not.
    """
    if not os.listdir(directory):
        return True
    else:
        return False
