import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import re
from nltk import word_tokenize
from sklearn.utils import shuffle
from nltk.util import ngrams
from nltk import sent_tokenize


def preprocessing(inputfile):
    """Create a vocabulary and map every word to an index"""
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
    mapping["<s>"] = len(sorted_vocab)

    return mapping


def create_vectors(inputfile):
    """Take the indexes from the vocabulary dictionary and create one hot
    vectors representing each word in the vocabulary"""
    one_hot = {}
    for word, number in vocab_dict.items():
        vector = [0] * len(vocab_dict)
        vector[number] = 1
        one_hot[word] = vector
    return one_hot


def split_data(inputfile, startline, endline):
    """Split the data into test data and train data"""
    selection_list = []
    with open(inputfile, "r", encoding="utf8") as f:
        file = f.read()
        text = re.sub(r'/[^\s]+','',file)
        sentences = sent_tokenize(text)
        if endline is None:
            selection = sentences[startline:]
        else:
            selection = sentences[startline:endline]

        for sentence in selection:
            tokenized_selection = word_tokenize(sentence)
        for word in tokenized_selection:
            selection_list.append(word)

    split_perc = 0.8
    train = selection_list[:int(split_perc * len(selection_list))]
    test = selection_list[int(split_perc * len(selection_list)):]

    train_data = shuffle(train)
    test_data = shuffle(test)

    return train_data, test_data


def create_ngrams(train_data, test_data, n):
    """Create n-grams from the train data, n representing the number of words"""
    ngrams_train = ngrams(train_data, n, pad_left=True, left_pad_symbol="<s>")
    ngrams_test = ngrams(test_data, n, pad_left=True, left_pad_symbol="<s>")

    return list(ngrams_train), list(ngrams_test)


def ngrams_train(ngramstrain):
    """Take the one hot vectors representing the words, create n-grams from
    those words or vectors (a list of lists). Make a pandas dataframe from that
    array"""
    vectors = []                #CHANGE TO A DICTIONARY?
    for gram in ngramstrain:
        each_vector = []
        for word in gram[:-1]:
            each_vector += vector_dict[word]
        each_vector.append(gram[-1])
        vectors.append(each_vector)
    vector_array = np.array(vectors)
    dataframe = pd.DataFrame(vector_array)

    return dataframe


def ngrams_test(ngramstest):
    """Take the one hot vectors representing the words, create n-grams from
    those words or vectors (a list of lists). Make a pandas dataframe from that
    array"""
    vectors = []
    for gram in ngramstest:
        each_vector = []
        for word in gram[:-1]:
            each_vector += vector_dict[word]
        each_vector.append(gram[-1])
        vectors.append(each_vector)
    vector_array = np.array(vectors)
    dataframe = pd.DataFrame(vector_array)

    return dataframe


def filemaker(train_ngrams, test_ngrams, outputfile_train, outputfile_test):
    pd.DataFrame(train_ngrams).to_csv(outputfile_train)
    pd.DataFrame(test_ngrams).to_csv(outputfile_test)

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
parser.add_argument("outputfile_train", type=str,
                    help="The name of the output file for the feature table of the training data (contains 80% of the selected data)")
parser.add_argument("outputfile_test", type=str,
                    help="The name of the output file for the feature table of the testing data (contains 20% of the selected data)")

args = parser.parse_args()

print("Loading data from file {}.".format(args.inputfile))
print("Starting from line {}.".format(args.startline))


if args.endline == None:
    print("Ending at last line of file.")

elif args.endline:
    print("Ending at line {}.".format(args.endline))

    if args.endline < args.startline:
        print("Error: Endline value cannot be lower than startline value")
        exit(1)

if args.ngram < 3:
    print("Error: ngrams must be at least trigrams")
    exit(1)
else:
    print("Constructing {}-gram model.".format(args.ngram))

print("Writing train table to {}.".format(args.outputfile_train))
print("Writing test table to {}.".format(args.outputfile_test))

# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.

vocab_dict = preprocessing(args.inputfile)
vector_dict = create_vectors(args.inputfile)
train_data, test_data = split_data(args.inputfile, args.startline, args.endline)
ngramstrain, ngramstest = create_ngrams(train_data, test_data, args.ngram)
train_ngrams = ngrams_train(ngramstrain)
test_ngrams = ngrams_test(ngramstest)
filemaker(train_ngrams, test_ngrams, args.outputfile_train, args.outputfile_test)
