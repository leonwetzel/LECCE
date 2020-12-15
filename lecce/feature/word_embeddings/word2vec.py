from gensim.models import Word2Vec


class Word2VecEmbedder:
    def __init__(self, model_name, corpus, directory):
        """

        Parameters
        ----------
        model_name : str
            File name of the model.
        corpus : list
            Collection of sentences.
        directory : str
            Name of the directory containing the word embeddings.

        """
        if model_name and not corpus:
            self.model = Word2Vec.load(f"{directory}/{model_name}")
        elif corpus:
            model_name = input("Please enter the filename for your new "
                               "word2vec model (including extension): ")
            self.model = Word2Vec(sentences=corpus, size=200,
                                  alpha=0.025, window=5, min_count=5,
                                  max_vocab_size=None, sample=0.001, seed=1,
                                  workers=3, min_alpha=0.0001, sg=0, hs=0,
                                  negative=5, ns_exponent=0.75, cbow_mean=1,
                                  iter=5, null_word=0, trim_rule=None,
                                  sorted_vocab=1, batch_words=10000,
                                  compute_loss=False, callbacks=(),
                                  max_final_vocab=None)
            self.model.save(f"{directory}/{model_name}")
        else:
            raise ValueError("Cannot create empty Embedding object!"
                             " Please refer to an existing model by"
                             " providing the filename of the model OR"
                             " provide a list of sentences for the"
                             " creation of a new model.")

    def transform(self):
        pass


import os
from tqdm import tqdm
import requests
import gzip


def download_google_news_vectors(url="https://s3.amazonaws.com/dl4j-distribution/" \
                        "GoogleNews-vectors-negative300.bin.gz",
                                 file_name="GoogleNews-vectors-negative300.bin.gz"):
    """
    Downloads Google News vectors and extracts the
    information.
    :return:
    """
    print("Downloading GoogleNewsVectors...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("Download failed, something went wrong!\n")
    else:
        print("Download completed!\n")


def decompress(archive_name, destination_name):
    """
    Decompresses a given .gz file.
    :param destination_name:
    :param archive_name:
    :return:
    """
    print(f"Decompressing {archive_name}...")
    fp = open(f"{destination_name}", "wb")
    with gzip.open(archive_name, 'rb') as f:
        file_content = f.read()
    fp.write(file_content)
    fp.close()
    os.remove(archive_name)
    print(f"Decompression of {archive_name} is finished!\n")
