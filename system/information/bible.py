import os
import requests


URL = "http://www.gutenberg.org/files/10/10.txt"


def download(directory, output_filename):
    """
    
    Parameters
    ----------
    directory
    output_filename

    Returns
    -------

    """
    r = requests.get(URL)

    with open(f"{directory}/{output_filename}", "w",
              encoding='utf-8') as F:
        F.write(r.text)
