#!/usr/bin/env python3
import argparse
from system.information.bible import strip


def main():
    parser = argparse.ArgumentParser(
        description='Parameters for the LCP estimaror')
    # parser.add_argument('--train', help='file name for training data')
    # parser.add_argument('--test', help='file name for test data')
    parser.add_argument('-f', '--full', help="file name for complete"
                                             " data set")
    args = parser.parse_args()

    file = 'bible.txt'
    strip(file)


if __name__ == '__main__':
    main()
