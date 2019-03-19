import os, sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# test.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.

def file_opener(datafile):
    """Opens the training file created by gendata as a dataframe and cleans the first column away as it contains a NaN"""
    dataframe = pd.read_csv(datafile, header=None)
    clean = dataframe.drop(dataframe.columns[0], axis=1)
    return clean


def open_pickle(modelfile):
    """Opens the pickled file from gendata, that is the trained model"""
    onions = open(modelfile, 'rb')
    model = pickle.load(onions)
    onions.close()

    return model


def get_vectors_and_labels(dataframe):
    """Gets the vectors from the dataframe which are all columns except the last one (containing the labels)
    as well as the class labels (the words from the ngrams the vectors represent)."""
    vectors = dataframe.iloc[:, :-1]
    class_labels = dataframe.iloc[:, -1]

    return vectors, class_labels

def test_model(vectors, labels, pickle_model):
    """Takes the pickled train model, the vectors and the labels from the test dataframe,
    and calculates the acuracy and perplexity over all training instances."""
    predicted_probabilities = []

    predictions = pickle_model.predict(vectors)
    accuracy = pickle_model.score(vectors, labels)
    log_probability = pickle_model.predict_log_proba(vectors)

    for index in range(len(vectors)):
        vector = vectors.iloc[index]
        probabilities_of_predictions = pickle_model.predict_proba([vector])[0]
        max_probability = max(probabilities_of_predictions)
        predicted_probabilities.append(max_probability)

    entropy = sum(predicted_probabilities) / len(predicted_probabilities)
    perplexity = 2**entropy

    print("Accuracy is ...")
    print(accuracy)

    print("Perplexity is...")
    print(perplexity)





parser = argparse.ArgumentParser(description="Test a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features in the test data.")
parser.add_argument("modelfile", type=str,
                    help="The name of the saved model file.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
print("Loading model from file {}.".format(args.modelfile))

print("Testing {}-gram model.".format(args.ngram))

dataframe = file_opener(args.datafile)
pickle_model = open_pickle(args.modelfile)
vectors, labels = get_vectors_and_labels(dataframe)
test_model(vectors, labels, pickle_model)
