#!/usr/bin/env python3
import os


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
    return False
