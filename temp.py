#!/usr/bin/env python3
import os
import pickle

from system.information.pubmed import unzip, extract_abstract_texts


def main():
    for subdir, dirs, files in os.walk('data/baseline'):
        for filename in files:
            filepath = subdir + os.sep + filename

            texts = extract_abstract_texts(filepath)

            with open(f"{filename}.txt", "w", encoding='utf-8') as F:
                for row in texts:
                    F.write(f"{row}\n")


if __name__ == '__main__':
    main()
