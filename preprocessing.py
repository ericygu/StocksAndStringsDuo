import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection as ms
from form_csv import json_to_df
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def normalization(xTrain, xTest):
    scaler = preprocessing.StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)
    return xTrain, xTest

# Unedited, can you unnormalized data
def pearson_graph(dfx, dfy):
    df = dfx.copy(deep=True)
    df['target'] = pd.Series(dfy['label'])
    correlation_matrix = df.corr(method='pearson')
    # sns.heatmap(correlation_matrix, annot=True)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlation_matrix)
    plt.show()
    return None

def process(x, y):
    # Acquire dataset
    # Don't run this again, it generates new data and replaces what we have. We will have to write something that adds to current data later.
    # I took care of reading from the current data files and converting them to dataframes/csv. -Nathan
    """
    exec(open("./load_articles.py").read())
    exec(open("./form_dictionary.py").read())
    exec(open("./form_keywords.py").read())
    exec(open("./stock_parse.py").read())
    exec(open("./value_keywords.py").read())

    # transform to csv and obtain xTrain, yTrain
    # Turning json data to csv data in either numpy format or df format
    exec(open("./form_csv.py").read())
    """

    n = 5194 # number of keywords to include in the dataset
    x, y = json_to_df(n)
    x.to_csv("x")
    y.to_csv("y")

    xTrain, xTest, yTrain, yTest = ms.train_test_split(x, y, test_size = 0.2)
    x_norm = normalization(x)
    
    xTrain.to_csv("xTrain")
    xTest.to_csv("xTest")
    yTrain.to_csv("yTrain")
    yTest.to_csv("yTest")
    
    print("Preprocessing Done")
    return None

# testing purposes
"""
def main():
    xTrain, yTrain, xTest, yTest = process()
    pearson_graph()

if __name__ == '__main__':
    main()
"""
    




