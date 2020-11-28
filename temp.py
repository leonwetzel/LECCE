#!/usr/bin/env python3
import os
import pickle

from lecce.information.corpus import Pubmed


def main():
    for subdir, dirs, files in os.walk('data/baseline'):
        for filename in files:
            filepath = subdir + os.sep + filename

            texts = Pubmed.extract_abstract_texts(filepath)

            with open(f"{filename}.txt", "w", encoding='utf-8') as F:
                for row in texts:
                    F.write(f"{row}\n")


if __name__ == '__main__':
    main()
