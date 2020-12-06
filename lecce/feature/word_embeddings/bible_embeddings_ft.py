import os
import re
import requests
import gensim
from gensim.models import FastText


def pre_process(location):
    """

    Parameters
    ----------
    bible file


    Returns
    processed list of list of text file
    -------

    """

    original_corpus = []
    f = open(location, "r")
    for line in f:
        original_corpus.append(line)
    start="The Old Testament of the King James Version of the Bible\n"
    end="End of the Project Gutenberg EBook of The King James Bible\n"
    start_index = original_corpus.index(start)
    end_index = original_corpus.index(end)
    corpus = original_corpus[start_index:end_index]
    new_corpus = [line.strip() for line in corpus]
    new_corpus = [re.sub(r'^\d+:\d+', '', line) for line in new_corpus if line]
    processed_lines = []
    for element in new_corpus:
        processed_lines.append(gensim.utils.simple_preprocess(element.strip()))
    return(processed_lines)

def bible_embeddings(processed_bible):
    #Parameters: processed bible file
    #Returns: writes bible embeddings to file
    model = FastText()
    model.build_vocab(sentences=processed_bible)
    model.train(sentences=processed_bible, total_examples=len(processed_bible), epochs=10)
    model.save("bible_ft.bin")
    
def main():
    bible = pre_process("bible.txt")
    embeddings = bible_embeddings(bible)
    
if __name__ == "__main__":
    main()
