import os
import tarfile
import xml.etree.ElementTree as ET
from ftplib import FTP

import requests
from tqdm import tqdm

from system.information.lcp import is_directory_empty


class Corpus:
    """
    The base class for functionality for downloading
    and processing corpora.
    """
    urls = []

    def download(self, destination_dir="data"):
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

    def remove_url(self, url, filename, remove_duplicate_urls=True):
        """

        Parameters
        ----------
        url : str
            URL to be removed.
        filename : str
            Filename related to data originating from the URL.
        remove_duplicate_urls : bool
            Indicates if all results for the URL should be
            removed from the data set or not

        Returns
        -------

        """
        if remove_duplicate_urls:
            self.urls = [pair for pair in self.urls if pair["url"] != url]
        else:
            self.urls = [pair for pair in self.urls if
                         pair["url"] == url and
                         pair["filename"] == filename]


class Extractable:
    """
    Class for data sources that are either zipped, tarred
    or whatsoever.
    """
    def extract(self):
        return 0


class Bible(Corpus):
    """
    Class for Bible-like corpora.
    """
    urls = [{"url": "http://www.gutenberg.org/files/10/10.txt",
             "name": "bible.txt"}]

    def clean(self):
        pass


class Europarl(Corpus, Extractable):
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


class Pubmed(Corpus, Extractable):

    urls = [{
        "url": "ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline",
        "name": "pubmed"
    }]

    # TODO invullen
    # Aangezien pubmed data via FTP ingeladen moet worden, moeten we
    # de functie overriden
    def download(self, destination_dir="data"):
        ftp = FTP("ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline")
        ftp.login()

    def extract(self):
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
        # if not new_filename:
        #     new_filename = file.split('.')[:2]
        #
        # with gzip.open(file, 'rb') as f_in:
        #     with open(f'{".".join(new_filename)}', 'wb') as f_out:
        #         shutil.copyfileobj(f_in, f_out)
        pass

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


class Lcp(Corpus, Extractable):
    """

    """
    urls = [
        {
            "url": "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master/trial/lcp_multi_trial.tsv",
            "name": "lcp_multi_trial.tsv"
        },
        {
            "url": "https://raw.githubusercontent.com/MMU-TDMLab/CompLex/master/trial/lcp_single_trial.tsv",
            "name": "lcp_single_trial.tsv"
        },
        {
            "url": "https://github.com/MMU-TDMLab/CompLex/raw/master/train.7z",
            "name": "lcp_train.7z"
        }
    ]

    def extract(self):
        return 0
