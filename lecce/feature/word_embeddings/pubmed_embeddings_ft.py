import os
import re
import requests
import gensim
from gensim.models import FastText
  
def main():
    pubmed = []
    for f in os.listdir('./pubmed/'):
        opened_file = open('./pubmed/' + f, 'r')
        for line in opened_file:
            pubmed.append(gensim.utils.simple_preprocess(line.strip()))
    print(pubmed)
    model = FastText(min_count = 10)
    model.build_vocab(sentences=pubmed)
    model.train(sentences=pubmed, total_examples=len(pubmed), epochs=10)
    model.save("pubmed_ft.bin")
if __name__ == "__main__":
    main()
