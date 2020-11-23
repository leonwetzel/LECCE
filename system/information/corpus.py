import abc


class Corpus(metaclass=abc.ABCMeta):
    """
    The base class for functionality for downloading
    and processing corpora.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def download(self, urls, destination_directory):
        pass

    def transform(self):
        pass

    def to_txt(self):
        pass

    def to_list(self):
        pass


class Bible(Corpus):
    def download(self, urls, destination_directory):
        pass
