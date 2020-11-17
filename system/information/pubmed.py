#!/usr/bin/env python3
import gzip
import os
import shutil
import urllib.request as request
import xml.etree.ElementTree as ET


# FIXME link wordt niet herkend...
def download():
    """Downloads the data from the us.gov website

    Returns
    -------

    """
    if not os.path.isdir('../data'):
        os.mkdir('../data')

    url = "ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline"

    request.urlretrieve(url, 'file')


def unzip(file, new_filename=None):
    """Unzips a given file and stores its content in a new file.

    Parameters
    ----------
    new_filename :
        Name of the unzipped file
    file : str
        Name of the zipped file.

    Returns
    -------

    """
    if not new_filename:
        new_filename = file.split('.')[:2]

    with gzip.open(file, 'rb') as f_in:
        with open(f'{".".join(new_filename)}', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def extract_abstract_texts(file):
    """Extracts the text from abstracts in a file.

    This script retrieves the abstract texts from a file and returns
    the data as a list.

    Parameters
    ----------
    file : str
        Name of the XML file.

    Returns
    -------
    texts : list
        List of abstract texts
    """
    tree = ET.parse(file)
    root = tree.getroot()
    abstracts = root.findall('.//AbstractText')

    texts = []
    for abstract in abstracts:
        texts.append(abstract.text)

    return texts
