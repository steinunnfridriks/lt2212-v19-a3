import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import re
from nltk import word_tokenize
from sklearn.utils import shuffle
from nltk.util import ngrams


def preprocessing(inputfile):
    vocabulary = []
    with open(inputfile, "r", encoding="utf8") as f:
        file = f.read()
        text = re.sub(r'/[^\s]+','',file)
        text = word_tokenize(text)
        for word in text:
            if word not in vocabulary:
                vocabulary.append(word)

    sorted_vocab = sorted(vocabulary)
    mapping = {word : number for number, word in enumerate(sorted_vocab)}

    return mapping


def create_vectors(inputfile):
    vocabulary_dict = preprocessing(inputfile)
    one_hot = {}
    for word, number in vocabulary_dict.items():
        vector = [0] * len(vocabulary_dict)
        vector[number] = 1
        one_hot[word] = vector
    return one_hot


def split_data(inputfile):
    with open(inputfile, "r", encoding="utf8") as f:
        file = f.read()
        text = re.sub(r'/[^\s]+','',file)
        text = word_tokenize(text)
        train_data_percentage = 0.8
        train_data, test_data = text[:int(train_data_percentage * len(text))], text[int(train_data_percentage * len(text)):]
        train_data = shuffle(train_data)
        test_data = shuffle(test_data)

    return train_data, test_data


def create_ngrams(inputfile, n):
    train_data, test_data = split_data(inputfile)
    n_grams = ngrams(train_data, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="<e>")
    print(list(n_grams))
    return list(n_grams)


def ngram_vectors():
     

# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.

parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.inputfile))
print("Starting from line {}.".format(args.startline))
if args.endline:
    print("Ending at line {}.".format(args.endline))
else:
    print("Ending at last line of file.")

print("Constructing {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.outputfile))

# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.

preprocessing(args.inputfile)
split_data(args.inputfile)
create_vectors(args.inputfile)
create_ngrams(args.inputfile, args.ngram)
