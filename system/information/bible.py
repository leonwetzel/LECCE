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


def strip(location):
    """

    Parameters
    ----------
    bible file


    Returns
    stiped text file
    -------

    """

    newlines = []
    text_file = open("bibleText.txt", "w")

    print(location)
    f = open(location, "r")
    for line in f:
        newlines.append(line[4:].rstrip('\n'))
    for element in newlines[100:199717]:
        text_file.write(element)
    text_file.close()