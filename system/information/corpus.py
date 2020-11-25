import gzip
import os
import shutil
import tarfile
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

from system.information.lcp import is_directory_empty


class Corpus:
    """
    The base class for functionality for downloading
    and processing corpora.
    """
    urls = []

    def download(self, destination_dir):
        """Downloads contents from given urls.

        The progress functionality is stolen from
        https://towardsdatascience.com/how-to-download-files-using-python-part-2-19b95be4cdb5.

        Parameters
        ----------
        destination_dir : str
            Name of the directory in which data should be stored.

        Returns
        -------

        """
        if not self.urls:
            raise ValueError("Could not find URL to download data"
                             " from!")

        if not os.path.isdir(destination_dir):
            # create new dir if dir does not exist
            os.mkdir(destination_dir)

        if not is_directory_empty(destination_dir):
            # raise warning if dir is not empty
            print(f"Warning: directory {destination_dir} is"
                  f" not empty!")
            enter = input(f"Press ENTER if you want to overwrite the"
                          f" existing data in {destination_dir}"
                          f" [ENTER]")

            if enter != "":
                # quit function if no confirmation is given
                print("Cancelling download...")
                return None

        for url in self.urls:
            response = requests.get(url["url"], stream=True,
                                    allow_redirects=True)
            total_size_in_bytes = int(
                response.headers.get('content-length'))

            with open(rf"{destination_dir}/{url['name']}", 'wb') as f:
                with tqdm(total=total_size_in_bytes, unit="B",
                          unit_scale=True, desc=url["name"],
                          initial=0, ascii=True) as pbar:
                    for ch in response.iter_content(chunk_size=1024):
                        if ch:
                            f.write(ch)
                            pbar.update(len(ch))

    def to_txt(self):
        pass

    def to_list(self):
        pass

    def add_url(self, url, filename):
        """

        Parameters
        ----------
        url : str
            URL directing to data.
        filename : str
            Name of the file related to the URL.

        Returns
        -------

        """
        self.urls.append({"url": url, "name": filename})


class Bible(Corpus):
    """
    Class for Bible-like corpora.
    """
    urls = [{"url": "http://www.gutenberg.org/files/10/10.txt",
             "name": "bible.txt"}]

    def clean(self):
        pass


class Europarl(Corpus):
    """
    Class for Europarl corpora.
    """
    urls = [{"url": "https://www.statmt.org/europarl/v7/europarl.tgz",
             "name": "europarl.tgz"}]

    def clean(self, filename):
        with open(filename, "r") as F:
            file = F.readlines()

        for line in file:
            print(line)

    def extract(self, filename):
        """

        Parameters
        ----------
        filename : str
            Name of the zipped file.

        Returns
        -------

        """
        with tarfile.open(filename, "r") as tar:
            tar.extractall(path="data",
                           members=self.get_english_proceedings(tar))

    @staticmethod
    def get_english_proceedings(members):
        """

        Parameters
        ----------
        members

        Returns
        -------

        """
        for tarinfo in members:
            filename, extension = os.path.splitext(tarinfo.name)
            try:
                language = filename.split('/')[1]
            except IndexError:
                continue
            if extension == ".txt" and language == "en":
                yield tarinfo


class Pubmed(Corpus):

    # TODO invullen
    # Aangezien pubmed data via FTP ingeladen moet worden, moeten we
    # de functie overriden
    def download(self, destination_dir):
        pass

    @staticmethod
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

    @staticmethod
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


class Lcp(Corpus):
    """

    """
    urls = [
        {
            "url": "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master/trial/lcp_multi_trial.tsv",
            "name": "lcp_multi_trial.tsv"},
        {
            "url": "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master/trial/lcp_single_trial.tsv",
            "name": "lcp_single_trial.tsv"}
    ]
