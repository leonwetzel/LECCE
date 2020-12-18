import os
import re
import requests
import gensim
from gensim.models import FastText

    
def main():
    joined_proceedings = []
    for file in os.listdir('./en/'):
        current_file = './en/' + file
        opened_file = open(current_file, 'r')
        lines = opened_file.readlines()
        lines = [line.strip() for line in lines if line]
        for line in lines:
            if not line.startswith('<'):
                joined_proceedings.append(line)

    processed_lines = []
    for element in joined_proceedings:
        processed_lines.append(gensim.utils.simple_preprocess(element.strip()))
    model = FastText()
    model.build_vocab(sentences=processed_lines)
    model.train(sentences=processed_lines, total_examples=len(processed_lines), epochs=10)
    model.save("europarl_ft.bin")
    
    

    
if __name__ == "__main__":
    main()
