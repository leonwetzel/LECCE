import gzip
import os
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from ftplib import FTP

import py7zr
import requests
from tqdm import tqdm

from lecce.information.utils import is_directory_empty


class Extractable(ABC):
    """
    Class for data sources that are either zipped, tarred
    or whatsoever.
    """

    @abstractmethod
    def extract_archive(self, archive_name):
        """Extract files from a compressed archive, such as .zip,
        .tgz or .7z.

        Parameters
        ----------
        archive_name : name of the archive file

        Returns
        -------

        """
        pass


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
            raise ValueError("Could not find URL(s) to download data"
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
        """Add a URL linking to a data source.

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
        """Remove a URL linking to a data source.

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
            self.urls = [pair for pair in self.urls if
                         pair["url"] != url]
        else:
            self.urls = [pair for pair in self.urls if
                         pair["url"] == url and
                         pair["filename"] == filename]


class Bible(Corpus):
    """
    Class for Bible-like corpora.
    """
    urls = [{"url": "http://www.gutenberg.org/files/10/10.txt",
             "name": "bible.txt"}]

    @staticmethod
    def prepare(filename, output_filename,
                start="The Old Testament of the King James Version of "
                      "the Bible\n",
                end="End of the Project Gutenberg EBook of The King "
                    "James Bible\n"):
        """Prepares the bible corpus by removing dirty lines of text.

        Parameters
        ----------
        filename : str
            Name of the file that contains the bible corpus.
        output_filename : str
            Name of the output file.
        start : str
            String indicating where the bible corpus starts in a file.
        end : str
            String indicating where the bible corpus ends in a file.

        Returns
        -------

        """
        with open(filename, "r", encoding='utf-8') as F:
            original_corpus = F.readlines()
            start_index = original_corpus.index(start)
            end_index = original_corpus.index(end)
            corpus = original_corpus[start_index:end_index]

            new_corpus = [line.strip() for line in corpus]
            new_corpus = [re.sub(r'^\d+:\d+', '', line) for line
                          in new_corpus if line]

            with open(output_filename, 'w', encoding='utf-8') as f:
                for line in new_corpus:
                    f.write(f"{line.strip()}\n")


class Europarl(Corpus, Extractable):
    """
    Class for Europarl corpora.
    """
    urls = [{"url": "https://www.statmt.org/europarl/v7/europarl.tgz",
             "name": "europarl.tgz"}]

    @staticmethod
    def prepare(filename, output_filename):
        """

        Parameters
        ----------
        filename
        output_filename

        Returns
        -------

        """
        corpus = []
        with open(filename, "r") as F:
            lines = F.readlines()

        lines = [line.strip() for line in lines if line]

        for line in lines:
            if not line.startswith('<'):
                corpus.append(line)

        with open(output_filename, "w") as f:
            f.writelines(corpus)

    def extract_archive(self, archive_name):
        """Extract files from a .tgz archive. The archive file
        will be removed after extraction as well.

        Parameters
        ----------
        archive_name : str
            Name of the zipped file.

        Returns
        -------

        """
        with tarfile.open(archive_name, "r") as tar:
            tar.extractall(path="data",
                           members=self.get_english_proceedings(tar))
        os.remove(f"data/{archive_name}")

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
    """
    Class for Pubmed corpora.
    """
    urls = [{"url": "ftp.ncbi.nlm.nih.gov",
             "name": "pubmed1"}]

    def __init__(self, file_limit=500):
        """Initiates a Pubmed object

        Parameters
        ----------
        file_limit : int
            Indicates how many files should be
            downloaded from the website.
        """
        self.file_limit = file_limit

    def download(self, destination_dir="data"):
        """Downloads Pubmed corpora from the website of the
        National Center for Biotechnology Information.

        Although the FTP server can be accessed without
        credentials, we need to provide the default username
        'anonymous' and an empty password.

        Parameters
        ----------
        destination_dir : str
            Name of the directory in which the data should
            be stored.

        Returns
        -------

        """
        target_dir = f"{destination_dir}/{self.urls[0]['name']}"
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        with FTP(host=self.urls[0]['url'], user='anonymous',
                 passwd='') as ftp:
            # change to right directory
            ftp.cwd("pubmed/baseline")

            # filter files
            requested_files = [filename for filename in ftp.nlst()
                               if filename.endswith('.gz')]

            # Iterate through the filenames
            # and retrieve plus extract them one at a time
            for archive_name in requested_files[:self.file_limit]:
                with open(f"{target_dir}/{archive_name}", 'wb') as f:
                    ftp.retrbinary('RETR %s' % archive_name, f.write)

                # extract file from archive
                self.extract_archive(f"{target_dir}/{archive_name}")
                # remove archive
                os.remove(f"{target_dir}/{archive_name}")

    def extract_archive(self, archive_name):
        """Unzips a given file and stores its content in a new file.

        Parameters
        ----------
        archive_name : str
            Name of the zipped file.

        Returns
        -------

        """
        new_filename = archive_name.split('.')[:2]

        with gzip.open(archive_name, 'rb') as f_in:
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

    def corpus_to_txt(self, directory):
        """

        Parameters
        ----------
        directory

        Returns
        -------

        """
        for subdir, dirs, files in os.walk(directory):
            for filename in files:
                filepath = subdir + os.sep + filename

                texts = self.extract_abstract_texts(filepath)

                with open(f"{filename}.txt", "w",
                          encoding='utf-8') as F:
                    for row in texts:
                        F.write(f"{row}\n")

                os.remove(filepath)


class Lcp(Corpus, Extractable):
    """
    Class for LCP data.
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

    def extract_archive(self, archive_name):
        """Extracts information

        Parameters
        ----------
        archive_name

        Returns
        -------

        """
        # FIXME wachtwoord niet hardcoden
        with py7zr.SevenZipFile(archive_name, mode='r',
                                password='YellowDolphin73!') as z:
            z.extractall(path="data")
