import os, sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# train.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.

def file_opener(datafile):
    """Opens the training file created by gendata as a dataframe and cleans the first column away as it contains a NaN"""
    dataframe = pd.read_csv(datafile, header=None)
    clean = dataframe.drop(dataframe.columns[0], axis=1)
    return clean

def trainer(dataframe):
    """Gets the x feature values which are all columns except the last one (containing the labels) and the y target values which are just the labels.
    Trains the model based on those values using the logistic regression trainer"""
    x_feature_values = dataframe.iloc[:, :-1]
    y_target_values = dataframe.iloc[:, -1]
    trainer = LogisticRegression(solver="lbfgs", multi_class="multinomial")
    model = trainer.fit(x_feature_values, y_target_values)

    return model

def pickler(model):
    """Makes cucumbers taste like old hay"""
    pickle.dump(model, open(args.modelfile, 'wb'))



parser = argparse.ArgumentParser(description="Train a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features.")
parser.add_argument("modelfile", type=str,
                    help="The name of the file to which you write the trained model.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
print("Training {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.modelfile))

dataframe = file_opener(args.datafile)
model = trainer(dataframe)
pickler(model)

# YOU WILL HAVE TO FIGURE OUT SOME WAY TO INTERPRET THE FEATURES YOU CREATED.
# IT COULD INCLUDE CREATING AN EXTRA COMMAND-LINE ARGUMENT OR CLEVER COLUMN
# NAMES OR OTHER TRICKS. UP TO YOU.
